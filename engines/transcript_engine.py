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
from core.config import FFMPEG_PATH, WHISPERX_MODEL
from core.device import DEVICE
from engines.voice_engine import VoiceEngine
from engines.lipsync_engine import LipSyncEngine
from engines.separate_engine import SeparateEngine

# XTTS supports en/hi; anything else falls back to English.
_XTTS_LANGS = {"en", "hi"}

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


def _extract_audio(video_path):
    out = timestamp_file("extracted", "wav")
    subprocess.run(
        [FFMPEG_PATH, "-y", "-i", video_path,
         "-vn", "-ac", "1", "-ar", "16000", out],
        capture_output=True, text=True, check=True,
    )
    return out


class TranscriptEngine(BaseEngine):

    def __init__(self):
        self.voice    = VoiceEngine()
        self.lipsync  = LipSyncEngine()
        self.separate = SeparateEngine()

    # ── step 1 ────────────────────────────────────────────────────────────────
    def extract_transcript(self, video_path):
        audio_path = _extract_audio(video_path)

        # Default to CPU: faster-whisper's GPU backend (ctranslate2) needs
        # cuDNN 8, but Colab ships cuDNN 9 — and the mismatch is a FATAL abort
        # that kills the whole process (not a catchable exception), so we must
        # not even try CUDA by default. Opt in with WHISPER_DEVICE=cuda only if
        # you've made cuDNN 8 available.
        prefer = os.environ.get("WHISPER_DEVICE", "cpu")
        devices = ["cuda", "cpu"] if prefer == "cuda" else ["cpu"]
        raw, lang, last_err = None, "en", None
        for dev in devices:
            try:
                model = _load_whisper(dev)
                seg_iter, info = model.transcribe(
                    audio_path, word_timestamps=True, vad_filter=True
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

        if n_words:
            print(f"[Transcript] {len(segments)} segments, {n_words} word timings "
                  f"-- word-level editing available.")
        else:
            print(f"[Transcript] {len(segments)} segments but NO word timings; "
                  f"edits will regenerate whole lines.")

        display = "\n".join(s["text"] for s in segments)
        xtts_lang = lang if lang in _XTTS_LANGS else "en"
        # Normalize to H.264 so the lip-sync engines can decode the frames
        # (Colab can't decode AV1, which the uploaded clip may be).
        norm_video = transcode_h264(video_path)
        state = {
            "video": norm_video,
            "audio": audio_path,
            "segments": segments,
            "language": xtts_lang,
        }
        return display, state

    # ── step 2 ────────────────────────────────────────────────────────────────
    def apply_edits(self, state, edited_text, method="latentsync",
                    inference_steps=20, guidance_scale=1.5, progress=None,
                    realism=True, max_stretch=None, word_level=True,
                    auto_reference=True, separate="auto"):
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
        """
        if not state:
            raise ValueError("Extract a transcript first.")

        import dsp
        from dsp import reference as dspref
        from core import textdiff

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

        if not edits:
            raise ValueError("No changes detected in the transcript.")

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

        for k, (i, start, end, new_text, span_meta) in enumerate(edits):
            if progress is not None:
                progress(k / max(len(edits), 1),
                         desc=f"Re-voicing edit {k+1}/{len(edits)}")

            clip_wav = self.voice.run(
                text=new_text,
                reference_audio_path=ref_path,
                language=state["language"],
            )
            span, span_sr = dsp.load(clip_wav, mono=True)
            span = dsp.resample(span, span_sr, sr)

            if realism:
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

        new_audio = timestamp_file("edited_audio", "wav")
        dsp.save(new_audio, track, sr)

        if progress is not None:
            progress(0.7, desc="Re-syncing lips ...")
        out = self.lipsync.run(
            video_path=state["video"], audio_path=new_audio,
            method=method, inference_steps=inference_steps,
            guidance_scale=guidance_scale,
        )
        return out
