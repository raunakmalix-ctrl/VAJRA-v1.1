"""
Feature 2 — transcript-driven lip editing.

Flow:
  1. extract_transcript(video): pull audio, transcribe with segment AND word
     timestamps, return an editable transcript (one segment per line). The word
     timings are what make step 2 surgical.
  2. apply_edits(state, edited_text, ...): diff each edited line at word level
     (core/textdiff.py) so only the region covering changed words is replaced,
     choose the cleanest window of the recording as the cloning reference
     (dsp/reference.py), then for each resolved span
     re-synthesize it in the speaker's own cloned voice (XTTS)
     and run it through the dsp realism layer -- duration fit, spectral/room/
     loudness match against the surrounding real audio, room-tone injection and
     a zero-crossing crossfade -- before splicing it in. Total length is
     preserved so the track stays frame-aligned, then the video's lips are
     re-synced with LatentSync.

Unchanged lines are never regenerated: their audio passes through
sample-identical, so a four-word edit leaves the rest of the recording genuine.

WhisperX runs in the main env. Voice synthesis and lip-sync delegate to the
isolated-venv engines.
"""
import os
import subprocess
import tempfile

from core.base_engine import BaseEngine
from core.utils import timestamp_file, transcode_h264
from core.config import FFMPEG_PATH, WHISPERX_MODEL, VENV_VOICE_PY
from core.device import DEVICE
from engines.voice_engine import VoiceEngine
from engines.lipsync_engine import LipSyncEngine
from engines.separate_engine import SeparateEngine
from engines import viitor_engine

# Resolved from the voice engine rather than duplicated, so the two can never
# disagree about what the model supports.
from engines.voice_engine import SUPPORTED_LANGUAGES as _VOICE_LANGS
_XTTS_LANGS = set(_VOICE_LANGS.values())

_whisper_model = None
_whisper_device = None


def _load_whisper(device):
    global _whisper_model, _whisper_device
    if _whisper_model is not None and _whisper_device == device:
        return _whisper_model
    from faster_whisper import WhisperModel
    compute = "float16" if device == "cuda" else "int8"
    print(f"[Transcript] Loading faster-whisper ({WHISPERX_MODEL}) on {device} ...")
    _whisper_model = WhisperModel(WHISPERX_MODEL, device=device, compute_type=compute)
    _whisper_device = device
    return _whisper_model


def _reset_whisper():
    global _whisper_model, _whisper_device
    _whisper_model = None
    _whisper_device = None


# The working rate for the whole edit. 16 kHz throws away everything above
# 8 kHz before any editing happens -- including the 4-10 kHz sibilance that
# carries s, sh and t -- so an edit could never sound better than telephone
# quality no matter how well it was spliced. The synthesisers already produce
# 24 kHz, and the lip-sync models resample to their own 16 kHz internally, so
# nothing downstream needs this thrown away.
MASTER_SR = 24000


def _extract_audio(video_path):
    out = timestamp_file("extracted", "wav")
    subprocess.run(
        [FFMPEG_PATH, "-y", "-i", video_path,
         "-vn", "-ac", "1", "-ar", str(MASTER_SR), out],
        capture_output=True, text=True, check=True,
    )
    return out


# Beyond this, re-timing the picture to fit new speech stops being invisible:
# gestures and blinks visibly slow down or speed up. It is not a hard limit --
# the operator may still want the take -- but it has to be said out loud.
_RETIME_NOTICEABLE = 0.25


def _retime_video(video_path, factor, out_path=None):
    """Stretch or compress the picture by `factor` so it matches new audio.

    Used only by the whole-track retake, where the new script has its own
    natural length and the original timing no longer applies. Video is
    re-encoded without audio; the new track is muxed on afterwards.
    """
    out = out_path or timestamp_file("retimed", "mp4")
    proc = subprocess.run(
        [FFMPEG_PATH, "-y", "-i", video_path,
         "-filter:v", f"setpts={factor:.6f}*PTS",
         "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", out],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(
            f"Could not re-time the video (ffmpeg exit {proc.returncode}).\n"
            f"{proc.stderr[-600:]}")
    return out


# A shortfall smaller than this is absorbed by time-compression without being
# noticeable; beyond it, the span is re-voiced as a whole line rather than cut.
_ESCALATE_SEC = 0.12


def _widen_to_fit(need_samples, start, end, sr, span_meta, max_stretch, n):
    """Grow [start, end] into neighbouring silence so `need_samples` fits.

    Returns (start, end, info). The span may only grow as far as `slack_start`
    and `slack_end` -- the gaps either side, which stop at the neighbouring
    words. Growing past those would delete a word the user did not edit.

    Time-compression covers the rest: a span may be shortened by up to
    `max_stretch`, so the slot only has to reach need/(1+max_stretch).
    """
    span = end - start
    # The shortest slot this audio can honestly be squeezed into.
    required = int(need_samples / (1.0 + float(max_stretch)))
    info = {"widened_sec": 0.0, "short_sec": 0.0,
            "slot_sec": round(span / float(sr), 3),
            "needed_sec": round(required / float(sr), 3)}
    if required <= span:
        return start, end, info

    lo_limit = start
    hi_limit = end
    if span_meta.get("slack_start") is not None:
        lo_limit = max(0, min(start, int(round(
            float(span_meta["slack_start"]) * sr))))
    if span_meta.get("slack_end") is not None:
        hi_limit = min(n, max(end, int(round(
            float(span_meta["slack_end"]) * sr))))

    deficit = required - span
    # Take from the trailing gap first: a phrase runs into the pause after it
    # far more naturally than it starts early into the one before.
    take_end = min(deficit, hi_limit - end)
    end += take_end
    deficit -= take_end
    take_start = min(deficit, start - lo_limit)
    start -= take_start
    deficit -= take_start

    info["widened_sec"] = round((take_end + take_start) / float(sr), 3)
    info["short_sec"] = round(max(0, deficit) / float(sr), 3)
    return start, end, info


class TranscriptEngine(BaseEngine):

    def __init__(self):
        self.voice    = VoiceEngine()
        self.lipsync  = LipSyncEngine()
        self.separate = SeparateEngine()
        self.viitor   = viitor_engine.ViitorEngine()

    # ── step 1 ────────────────────────────────────────────────────────────────
    def extract_transcript(self, video_path, language=None):
        audio_path = _extract_audio(video_path)

        # Default to CPU: faster-whisper's GPU backend (ctranslate2) needs
        # cuDNN 8, but Colab ships cuDNN 9 — and the mismatch is a FATAL abort
        # that kills the whole process (not a catchable exception), so we must
        # not even try CUDA by default. Opt in with WHISPER_DEVICE=cuda only if
        # you've made cuDNN 8 available.
        # Auto-detection is weaker than it looks here: it runs on CPU with
        # int8 quantisation and decides from roughly the first 30 seconds. On
        # accented English it can settle on a related language and then
        # TRANSLITERATE -- English words written in the wrong script, which
        # looks like a transcription bug but is a detection one. So the caller
        # can state the language outright.
        want = (language or "").strip().lower() or None
        prefer = os.environ.get("WHISPER_DEVICE", "cpu")
        devices = ["cuda", "cpu"] if prefer == "cuda" else ["cpu"]
        raw, lang, last_err = None, "en", None
        for dev in devices:
            try:
                model = _load_whisper(dev)
                seg_iter, info = model.transcribe(
                    audio_path, word_timestamps=True, vad_filter=True,
                    language=want,
                )
                raw = list(seg_iter)   # materialize now to surface runtime errors
                lang = getattr(info, "language", "en") or "en"
                break
            except Exception as e:
                print(f"[Transcript] faster-whisper on {dev} failed: {e}")
                _reset_whisper()
                last_err = e
        if raw is None:
            raise RuntimeError(f"Transcription failed: {last_err}")

        segments = []
        n_words = 0
        for s in raw:
            text = (s.text or "").strip()
            if not text:
                continue
            # Keep the word timings. They were previously discarded, which forced
            # every edit to regenerate a whole line; with them, only the changed
            # words' region has to be replaced (see core/textdiff.py).
            words = []
            for w in (getattr(s, "words", None) or []):
                wt = (getattr(w, "word", "") or "").strip()
                ws, we = getattr(w, "start", None), getattr(w, "end", None)
                if wt and ws is not None and we is not None:
                    words.append({"word": wt, "start": float(ws),
                                  "end": float(we),
                                  "prob": float(getattr(w, "probability", 0.0) or 0.0)})
            n_words += len(words)
            segments.append({
                "start": float(s.start), "end": float(s.end), "text": text,
                "words": words,
            })
        if not segments:
            raise RuntimeError("No speech detected in the video.")

        if want and lang != want:
            # transcribe() honours `language`, so this should not happen; if it
            # ever does, the mismatch matters more than the guess.
            print(f"[Transcript] NOTE requested '{want}' but the model "
                  f"reported '{lang}'.")
            lang = want
        if want:
            print(f"[Transcript] Language forced to '{want}' "
                  f"(auto-detection skipped).")
        if n_words:
            print(f"[Transcript] {len(segments)} segments, {n_words} word timings "
                  f"-- word-level editing available.")
        else:
            print(f"[Transcript] {len(segments)} segments but NO word timings; "
                  f"edits will regenerate whole lines.")

        display = "\n".join(s["text"] for s in segments)
        # Keep the DETECTED language, not the one we can synthesise. Folding an
        # unsupported language to English here would hide the compromise from
        # the router, which exists precisely to report it (core/router.py).
        xtts_lang = lang if lang in _XTTS_LANGS else "en"
        # Normalize to H.264 so the lip-sync engines can decode the frames
        # (Colab can't decode AV1, which the uploaded clip may be).
        norm_video = transcode_h264(video_path)
        state = {
            "video": norm_video,
            "audio": audio_path,
            "segments": segments,
            "language": lang,
            "synthesis_language": xtts_lang,
            "has_word_timings": bool(n_words),
        }
        return display, state

    def transcribe_audio(self, audio_path):
        """Plain transcript for an audio file, with no editing state.

        The Voice Edit tab needs the original wording to align against, and
        asking an operator to type out what they can already hear is busywork
        that also invites transcription errors the aligner would then act on.
        Separate from extract_transcript because that one normalises video and
        builds the whole edit state, none of which applies to a bare clip.
        """
        if not audio_path or not os.path.exists(audio_path):
            raise ValueError("No audio supplied.")
        wav = _extract_audio(audio_path)
        prefer = os.environ.get("WHISPER_DEVICE", "cpu")
        model = _load_whisper("cuda" if prefer == "cuda" else "cpu")
        seg_iter, _info = model.transcribe(wav, vad_filter=True)
        text = " ".join((s.text or "").strip() for s in seg_iter).strip()
        if not text:
            raise RuntimeError("No speech detected in that audio.")
        return text

    def _retake_whole_track(self, state, new_lines, segments, track, sr,
                            method, inference_steps, guidance_scale, progress,
                            decision=None, allow_nc=False, prefer_tier=None):
        """Speak the whole script afresh and re-time the picture to fit it.

        The surgical path exists to keep as much of the original recording as
        possible, and everything in it follows from one invariant: the new audio
        must occupy exactly the slot it replaces, so the track stays frame-
        aligned with the video. When the entire script changes that invariant
        stops being useful -- a script three times longer cannot be squeezed
        into the original timing, and trying caps the stretch and drops words.

        So this mode drops the invariant deliberately and restores alignment the
        other way round: synthesise at natural pace, then re-time the PICTURE
        onto the new duration. Nothing of the original audio survives, which is
        the honest cost and is reported as such.
        """
        import numpy as np
        import dsp
        from core import router as edit_router
        from dsp import reference as dspref

        lines = [(new_lines[i] if i < len(new_lines) else seg["text"]).strip()
                 for i, seg in enumerate(segments)]
        if not any(lines):
            raise ValueError("The transcript is empty.")

        decision = edit_router.resolve(
            state.get("language", "en"), has_word_timings=False,
            allow_nc=allow_nc, prefer_tier=prefer_tier,
            available={"xtts": os.path.exists(VENV_VOICE_PY),
                       "viitor_nar": viitor_engine.available()})
        state["routing"] = decision
        print(f"[Transcript] Whole-track retake: {len(lines)} line(s) via "
              f"{decision['label']}")

        # The cloning reference comes from the original recording. Nothing is
        # excluded, because nothing of it is being kept.
        ref_path = state.get("audio")
        ref_clip, ref_info = dspref.pick_reference(track, sr)
        if ref_clip is not None and ref_clip.size:
            ref_path = timestamp_file("voice_ref", "wav")
            dsp.save(ref_path, ref_clip, sr)
            print(f"[Transcript] Cloning reference: {dspref.describe(ref_info)}")

        lang = (decision.get("synthesis_language")
                or state.get("synthesis_language") or "en")
        pieces = []
        for i, line in enumerate(lines):
            if progress is not None:
                progress(i / max(len(lines), 1),
                         desc=f"Re-voicing line {i + 1}/{len(lines)}")
            if not line:
                continue
            wav = self.voice.run(text=line, reference_audio_path=ref_path,
                                 language=lang)
            clip, csr = dsp.load(wav, mono=True)
            clip = dsp.resample(clip, csr, sr)
            clip, _ = dsp.trim_silence(clip, sr)
            pieces.append(clip)
            # Keep the original pause after this line, so the new take inherits
            # the speaker's pacing instead of running together.
            if i + 1 < len(segments):
                gap = float(segments[i + 1]["start"]) - float(segments[i]["end"])
                if gap > 0.02:
                    pieces.append(np.zeros(int(min(gap, 1.5) * sr),
                                           dtype=np.float32))

        if not pieces:
            raise ValueError("Nothing was synthesised.")
        new_track = np.concatenate(pieces).astype(np.float32)
        new_track = dsp.limit_peak(new_track, ceiling_db=-1.0)

        old_sec = track.shape[0] / float(sr)
        new_sec = new_track.shape[0] / float(sr)
        factor = new_sec / max(old_sec, 1e-6)

        new_audio = timestamp_file("edited_audio", "wav")
        dsp.save(new_audio, new_track, sr)
        state["edited_audio"] = new_audio
        state["retake"] = {
            "lines": len(lines), "original_sec": round(old_sec, 2),
            "new_sec": round(new_sec, 2), "retime_factor": round(factor, 3),
            "noticeable": abs(factor - 1.0) > _RETIME_NOTICEABLE,
            "note": ("The whole track was replaced, so no original audio "
                     "remains."),
        }
        print(f"[Transcript] New track {new_sec:.2f}s vs original "
              f"{old_sec:.2f}s -- re-timing the picture by {factor:.3f}x")
        if state["retake"]["noticeable"]:
            print(f"[Transcript] WARNING re-timing by {factor:.2f}x is enough "
                  f"to see: gestures and blinks will run "
                  f"{'slow' if factor > 1 else 'fast'}. Shorten or lengthen "
                  f"the script toward the original speaking time to reduce it.")

        video = state["video"]
        if abs(factor - 1.0) > 0.01:
            if progress is not None:
                progress(0.75, desc="Re-timing the video ...")
            video = _retime_video(video, factor)
            state["retimed_video"] = video

        if progress is not None:
            progress(0.85, desc="Re-syncing lips ...")
        # Whole-video sync: every frame's mouth is wrong now, so there is no
        # window to confine it to.
        state["lipsync"] = {"windowed": False,
                            "reason": "the whole track was replaced"}
        return self.lipsync.run(
            video_path=video, audio_path=new_audio, method=method,
            inference_steps=inference_steps, guidance_scale=guidance_scale)

    # ── step 2 ────────────────────────────────────────────────────────────────
    def apply_edits(self, state, edited_text, method="latentsync",
                    inference_steps=20, guidance_scale=1.5, progress=None,
                    realism=True, max_stretch=None, word_level=True,
                    auto_reference=True, separate=False,
                    windowed_lipsync=True, allow_nc=False,
                    scorecard=True, prefer_tier=None, full_retake=False):
        """Re-voice only the changed lines and splice them back in.

        realism: run the dsp match chain (duration fit, spectral/room/loudness
        match, room-tone injection, zero-crossing crossfade) on every replaced
        span. This is what makes an edit inaudible rather than merely correct --
        see dsp/ for why each stage exists. Set False only to A/B against the
        unmatched splice.

        word_level: resolve edits per word and regenerate only the affected
        region rather than the whole line. Falls back automatically when a
        segment has no word timings.

        auto_reference: choose the cleanest window of the recording as the
        cloning reference instead of handing the model the whole track. Reference
        quality dominates clone quality more than any inference setting.

        separate: "auto" (use speech/background separation when its environment
        is built), True (require it), or False (never). Editing the speech stem
        and re-laying the untouched background is the strongest single realism
        measure available, because the ambience never stops across the join.

        windowed_lipsync: re-render lips only over the edited time ranges and
        composite them into the original frames, instead of putting the whole
        clip through the model. Falls back to whole-video sync automatically when
        the edits already cover most of the footage.

        allow_nc: permit non-commercially-licensed engines in the router. Off by
        default, so a restricted licence is never a silent default.

        scorecard: measure the finished edit and record the metrics on `state`.
        """
        if not state:
            raise ValueError("Extract a transcript first.")

        import dsp
        from dsp import reference as dspref
        from core import textdiff
        from core import router as edit_router

        segments = state["segments"]
        new_lines = [l.strip() for l in edited_text.split("\n")]
        # Drop trailing empties so a stray newline doesn't break alignment.
        while new_lines and not new_lines[-1]:
            new_lines.pop()

        # Work on the whole track as one array rather than rebuilding it from
        # concatenated pieces. Every splice preserves total length, so untouched
        # audio stays sample-identical to the source and later span indices stay
        # valid -- both of which the previous concatenate-everything approach
        # could not guarantee.
        # Separate speech from background when available. The edit then happens
        # on the speech stem only, and the untouched background is re-laid over
        # the result at the end -- so the room, music and traffic continue
        # unbroken across every join, because they were never regenerated.
        background = None
        use_sep = bool(separate) and (
            self.separate.available() if separate == "auto" else True)
        if use_sep:
            try:
                speech_path, bg_path = self.separate.run(state["audio"])
                state["stem_speech"] = speech_path
                state["stem_background"] = bg_path
                track, sr = dsp.load(speech_path, mono=True)
                background, bg_sr = dsp.load(bg_path, mono=True)
                background = dsp.resample(background, bg_sr, sr)
                background = dsp.pad_or_trim(background, track.shape[0])
                print(f"[Transcript] Separated: editing the speech stem; "
                      f"background will be re-laid untouched.")
            except Exception as e:
                if separate is True:
                    raise
                # "auto" must not fail an edit over an optional enhancement.
                print(f"[Transcript] Separation unavailable ({e}); "
                      f"editing the mixed track instead.")
                background = None
                track, sr = dsp.load(state["audio"], mono=True)
        else:
            track, sr = dsp.load(state["audio"], mono=True)
        n = track.shape[0]

        def _clamp(t):
            return int(min(max(int(round(float(t) * sr)), 0), n))

        # Resolve edits at WORD level: only the region covering changed words is
        # regenerated, expanded outward to the nearest pause so the synthesiser
        # still receives a natural phrase. Lines without word timings fall back
        # to whole-line replacement, reported via each span's 'granularity'.
        edits = []
        for i, seg in enumerate(segments):
            new_text = new_lines[i] if i < len(new_lines) else seg["text"]
            spans = textdiff.resolve_edit_spans(
                seg, new_text, expand_to_pauses=word_level)
            if spans:
                print(f"[Transcript] segment {i+1}: "
                      f"{textdiff.summarise_spans(spans, seg)}")
            for sp in spans:
                edits.append((i, _clamp(sp["t_start"]), _clamp(sp["t_end"]),
                              sp["text"], sp))

        # Drop any span that collapsed to nothing after clamping.
        edits = [e for e in edits if e[2] > e[1]]

        if full_retake:
            return self._retake_whole_track(
                state, new_lines, segments, track, sr, method,
                inference_steps, guidance_scale, progress, decision=None,
                allow_nc=allow_nc, prefer_tier=prefer_tier)

        if not edits:
            raise ValueError("No changes detected in the transcript.")

        # Resolve HOW this edit gets regenerated, and record it. The decision is
        # language-dependent, so it must be stated rather than left implicit --
        # see core/router.py.
        has_words = any((e[4] or {}).get("granularity") == "word" for e in edits)
        decision = edit_router.resolve(
            state.get("language", "en"),
            has_word_timings=has_words,
            allow_nc=allow_nc,
            prefer_tier=prefer_tier,
            available={"xtts": os.path.exists(VENV_VOICE_PY),
                       "viitor_nar": viitor_engine.available()},
        )
        # The router only knows whether the edit COULD be localised, so on its
        # own it reports "no word-level timings for this segment" -- which is
        # actively wrong when the timings exist and the edit was simply too
        # extensive to localise. Replace that with what actually happened, or
        # the log contradicts the extraction step two lines above it.
        if state.get("has_word_timings") and not has_words:
            decision["warnings"] = [
                w.replace(
                    "No word-level timings for this segment, so the edit "
                    "covers the whole transcript line. Precision is reduced.",
                    "Word timings exist, but every edited line changed too "
                    "extensively to localise below the line, so whole lines "
                    "were regenerated. Change fewer words per line to keep "
                    "more of the original recording.")
                for w in decision.get("warnings", [])
            ]
        state["routing"] = decision
        print(f"[Transcript] Routing: {edit_router.describe(decision)}")

        # Tier A works differently from tiers B-D and the flow has to follow.
        # The infill model takes the AUDIO plus the original and edited text and
        # does its own alignment and masking internally -- it is not handed a
        # phrase to speak. So the unit of work becomes the segment: it gets the
        # line as context, regenerates only the words that changed inside it,
        # and returns the line with everything else conditioned on the real
        # recording. Feeding it isolated word spans instead would throw away the
        # surrounding audio that makes it a tier A engine in the first place.
        if decision["tier"] == "A":
            edits = []
            for i, seg in enumerate(segments):
                new_line = new_lines[i] if i < len(new_lines) else seg["text"]
                if textdiff.normalise(new_line.strip()) == textdiff.normalise(
                        (seg["text"] or "").strip()):
                    continue
                edits.append((i, _clamp(seg["start"]), _clamp(seg["end"]),
                              new_line,
                              {"granularity": "infill",
                               "orig_text": seg["text"],
                               "slack_start": seg["start"],
                               "slack_end": seg["end"],
                               "expanded": False}))
            edits = [e for e in edits if e[2] > e[1]]
            if not edits:
                raise ValueError("No changes detected in the transcript.")
            print(f"[Transcript] tier A: {len(edits)} line(s) sent to "
                  f"ViiTorVoice for in-place infill")
        for warn in decision.get("warnings", []):
            print(f"[Transcript] NOTE {warn}")

        saved = sum((e[2] - e[1]) for e in edits) / float(sr)
        total_changed_lines = len({e[0] for e in edits})
        line_dur = sum(float(segments[i]["end"]) - float(segments[i]["start"])
                       for i in {e[0] for e in edits})
        if line_dur > 0:
            print(f"[Transcript] regenerating {saved:.2f}s across "
                  f"{len(edits)} span(s) instead of {line_dur:.2f}s of whole "
                  f"lines ({100.0 * saved / line_dur:.0f}% as much audio) "
                  f"across {total_changed_lines} edited line(s).")

        # Pick the cloning reference deliberately instead of handing the model
        # the entire original track. Reference quality dominates clone quality,
        # and the regions being replaced are excluded so the synthesiser is never
        # conditioned on the audio it is meant to be replacing.
        ref_path = state.get("audio")
        if auto_reference:
            speech_ranges = [(float(s["start"]), float(s["end"]))
                             for s in segments]
            # Pad the excluded ranges: the splice stage snaps each boundary to a
            # nearby quiet point later on, so a span can grow by up to the snap
            # search window after the reference has already been chosen. Without
            # this margin the reference can end up abutting -- and overlapping --
            # the very audio it must not be conditioned on.
            guard = int(0.25 * sr)
            ref_clip, ref_info = dspref.pick_reference(
                track, sr,
                exclude_ranges=[(max(0, s - guard), min(n, e + guard))
                                for _, s, e, _, _ in edits],
                speech_ranges=speech_ranges)
            ref_path = timestamp_file("clone_reference", "wav")
            dsp.save(ref_path, ref_clip, sr)
            state["reference_info"] = ref_info
            print(f"[Transcript] Cloning reference: {dspref.describe(ref_info)}")

        # Harvest room tone ONCE from the original, excluding every span that is
        # about to be replaced -- a span's own audio must not contribute the tone
        # used to disguise its replacement.
        room_tone = None
        tone_info = {"found_sec": 0.0, "reason": "realism disabled"}
        if realism:
            room_tone, tone_info = dsp.extract_room_tone(
                track, sr, exclude_ranges=[(s, e) for _, s, e, _, _ in edits])
            if room_tone.size == 0:
                room_tone = None
                print("[Transcript] No usable room tone found in the source; "
                      "noise-floor matching will be skipped.")
            else:
                print(f"[Transcript] Room tone: {tone_info['found_sec']}s "
                      f"at {tone_info.get('floor_db')} dBFS")

        stretch = (dsp.DEFAULT_MAX_STRETCH if max_stretch is None
                   else float(max_stretch))
        reports = []
        # Lines re-voiced whole because a span would not fit. The whole-line
        # take already contains every edit on that line, so any remaining span
        # for it would splice the same words in a second time.
        escalated = set()

        for k, (i, start, end, new_text, span_meta) in enumerate(edits):
            if i in escalated:
                print(f"[Transcript] edit {k+1}: covered by the whole-line "
                      f"re-voice of segment {i+1} — skipping")
                continue
            if progress is not None:
                progress(k / max(len(edits), 1),
                         desc=f"Re-voicing edit {k+1}/{len(edits)}")

            synth_lang = (decision.get("synthesis_language")
                          or state.get("synthesis_language") or "en")
            if decision["tier"] == "A":
                # Hand the model the real audio of this line. It masks and
                # regenerates only the changed words itself, conditioned on
                # everything around them.
                line_wav = timestamp_file(f"line_{i+1}", "wav")
                dsp.save(line_wav, track[start:end], sr)
                clip_wav = self.viitor.local_edit(
                    source_audio_path=line_wav,
                    original_text=span_meta.get("orig_text") or "",
                    edited_text=new_text,
                    language=synth_lang,
                    progress=progress,
                )
            else:
                clip_wav = self.voice.run(
                    text=new_text,
                    reference_audio_path=ref_path,
                    # The router already decided what we can actually synthesise
                    # in, and warned if that differs from the detected language.
                    language=synth_lang,
                )
            span, span_sr = dsp.load(clip_wav, mono=True)
            span = dsp.resample(span, span_sr, sr)
            # Synthesisers pad their output with silence. That padding is not
            # speech but still counts toward the clip's length, so leaving it on
            # would make a span look too long for its slot and trigger a widen
            # or a whole-line re-voice that the actual speech never needed.
            span, trim_info = dsp.trim_silence(span, sr)

            # A word-level slot is only as long as the words it replaces, so a
            # replacement with more syllables does not fit. Truncating it is the
            # worst option available: the listener hears the start of the new
            # wording and then the ORIGINAL words resuming. Grow the slot into
            # the neighbouring silence first -- the word timings say exactly how
            # much there is, and it is what a human editor would use.
            start, end, widen = _widen_to_fit(
                span.shape[0], start, end, sr, span_meta, stretch, n)
            if widen["widened_sec"]:
                print(f"[Transcript] edit {k+1}: absorbed "
                      f"{widen['widened_sec']:.2f}s of neighbouring silence so "
                      f"the new wording fits without being cut")

            # Still too long even with the silence: re-speak the WHOLE line
            # instead. A fully synthetic line is a real cost, but it is coherent,
            # whereas a truncated span is audibly broken. Reported either way.
            # Tier A already works line-at-a-time, so there is no smaller unit
            # to escalate from -- only the phrase path can escalate.
            if widen["short_sec"] > _ESCALATE_SEC and span_meta.get(
                    "granularity") == "word":
                seg = segments[i]
                whole_text = new_lines[i] if i < len(new_lines) else seg["text"]
                print(f"[Transcript] edit {k+1}: {widen['short_sec']:.2f}s too "
                      f"long even after absorbing silence — re-voicing the "
                      f"whole line so nothing is cut mid-phrase")
                clip_wav = self.voice.run(
                    text=whole_text, reference_audio_path=ref_path,
                    language=decision.get("synthesis_language")
                             or state.get("synthesis_language") or "en",
                )
                span, span_sr = dsp.load(clip_wav, mono=True)
                span = dsp.resample(span, span_sr, sr)
                start, end = _clamp(seg["start"]), _clamp(seg["end"])
                new_text = whole_text
                span_meta = dict(span_meta, granularity="segment",
                                 escalated=True)
                escalated.add(i)

            # The disguise chain exists because a sentence synthesiser cannot
            # hear the recording it is editing, so its output has to be pushed
            # into place: spectrum corrected, room convolved on, a noise bed laid
            # under it. A tier A infill model already heard that audio, so its
            # output is ALREADY in place -- running the chain over it stacks
            # correction on something that needs none and audibly dirties the
            # span. Keep only the two stages that are about fitting a timeline,
            # not about disguise.
            if realism and decision["tier"] == "A":
                track, rep = dsp.match_and_splice(
                    track, span, start, end, sr,
                    room_tone=None, max_stretch=stretch,
                    spectral_strength=0.0, room_strength=0.0)
                rep["disguise"] = "skipped — tier A output is already in place"
            elif realism:
                track, rep = dsp.match_and_splice(
                    track, span, start, end, sr,
                    room_tone=room_tone, max_stretch=stretch)
            else:
                span = dsp.fit_duration(span, sr, (end - start) / sr,
                                        max_stretch=stretch)[0]
                track, rep = dsp.crossfade_splice(
                    track, span, start, end, sr, fade_ms=8.0)
                rep = {"splice": rep}

            rep["segment"] = i
            rep["text"] = new_text
            rep["granularity"] = span_meta.get("granularity")
            rep["orig_text"] = span_meta.get("orig_text")
            rep["span_expanded"] = span_meta.get("expanded")
            rep["escalated"] = bool(span_meta.get("escalated"))
            rep["widened_sec"] = widen.get("widened_sec", 0.0)
            # Tone was harvested once up front, so match_and_splice never saw a
            # None to fill in -- record it here to keep each report complete.
            rep.setdefault("room_tone", tone_info)
            reports.append(rep)
            print(f"[Transcript] edit {k+1}/{len(edits)} "
                  f"(segment {i+1}): {dsp.summarise(rep)}")

            d = rep.get("duration") or {}
            if d.get("status") == "exceeded":
                # Surfaced rather than silently applied: past ~15% the artefact
                # is worse than the timing error it fixes.
                print(f"[Transcript] WARNING segment {i+1}: needed "
                      f"{d.get('required_rate')}x time-scaling, capped at "
                      f"{d.get('applied_rate')}x -- still "
                      f"{d.get('shortfall_sec')}s off. {d.get('advice')}")

        state["match_reports"] = reports

        if background is not None:
            # Re-lay the original background over the edited speech. Nothing in
            # this bed was regenerated, so its continuity is genuine.
            track = (dsp.as_float32(track)
                     + dsp.pad_or_trim(background, track.shape[0]))
            track = dsp.limit_peak(track, ceiling_db=-0.5)
            print("[Transcript] Re-laid the untouched background stem.")

        if scorecard:
            try:
                import evaluation
                card = evaluation.score_all(
                    track, sr,
                    spans=[(r["splice"]["start"], r["splice"]["end"])
                           for r in reports if r.get("splice")],
                    reports=reports)
                state["scorecard"] = card
                for line in evaluation.summarise(card).splitlines():
                    print(f"[Transcript] {line}")
            except Exception as e:
                # Measurement must never cost the operator their edit.
                print(f"[Transcript] Scorecard unavailable ({e}).")

        new_audio = timestamp_file("edited_audio", "wav")
        dsp.save(new_audio, track, sr)
        # Surfaced so the operator can A/B the audio directly, without waiting
        # for the lip-sync pass or hunting for the file.
        state["edited_audio"] = new_audio

        if progress is not None:
            progress(0.7, desc="Re-syncing lips ...")

        # The edit spans are already known in seconds, so only those ranges need
        # re-rendering. Everything else keeps its original pixels, which both
        # saves model time proportional to how little changed and keeps the
        # model from touching footage that had no reason to change.
        edit_windows = [(s0 / float(sr), e0 / float(sr))
                        for _, s0, e0, _, _ in edits]
        state["lipsync_windows"] = edit_windows

        if windowed_lipsync and hasattr(self.lipsync, "run_windowed"):
            ls_report = {}
            out = self.lipsync.run_windowed(
                video_path=state["video"], audio_path=new_audio,
                windows=edit_windows,
                method=method, inference_steps=inference_steps,
                guidance_scale=guidance_scale, progress=progress,
                report=ls_report,
            )
            state["lipsync"] = ls_report
        else:
            out = self.lipsync.run(
                video_path=state["video"], audio_path=new_audio,
                method=method, inference_steps=inference_steps,
                guidance_scale=guidance_scale,
            )
        return out
