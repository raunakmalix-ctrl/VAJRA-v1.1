"""
Feature 2 — transcript-driven lip editing.

Flow:
  1. extract_transcript(video): pull audio, run WhisperX for segment + word
     timestamps, return an editable transcript (one segment per line).
  2. apply_edits(state, edited_text, ...): for each line the user changed,
     re-synthesize just that segment in the speaker's own cloned voice (XTTS),
     then run it through the dsp realism layer -- duration fit, spectral/room/
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
        self.voice   = VoiceEngine()
        self.lipsync = LipSyncEngine()

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
        for s in raw:
            text = (s.text or "").strip()
            if text:
                segments.append({
                    "start": float(s.start), "end": float(s.end), "text": text,
                })
        if not segments:
            raise RuntimeError("No speech detected in the video.")

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
                    realism=True, max_stretch=None):
        """Re-voice only the changed lines and splice them back in.

        realism: run the dsp match chain (duration fit, spectral/room/loudness
        match, room-tone injection, zero-crossing crossfade) on every replaced
        span. This is what makes an edit inaudible rather than merely correct --
        see dsp/ for why each stage exists. Set False only to A/B against the
        unmatched splice.
        """
        if not state:
            raise ValueError("Extract a transcript first.")

        import dsp

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
        track, sr = dsp.load(state["audio"], mono=True)
        n = track.shape[0]

        def _clamp(t):
            return int(min(max(int(round(float(t) * sr)), 0), n))

        # Resolve which lines actually changed before touching any audio.
        edits = []
        for i, seg in enumerate(segments):
            new_text = new_lines[i] if i < len(new_lines) else seg["text"]
            if new_text and new_text != seg["text"]:
                edits.append((i, _clamp(seg["start"]), _clamp(seg["end"]),
                              new_text))

        if not edits:
            raise ValueError("No changes detected in the transcript.")

        # Harvest room tone ONCE from the original, excluding every span that is
        # about to be replaced -- a span's own audio must not contribute the tone
        # used to disguise its replacement.
        room_tone = None
        tone_info = {"found_sec": 0.0, "reason": "realism disabled"}
        if realism:
            room_tone, tone_info = dsp.extract_room_tone(
                track, sr, exclude_ranges=[(s, e) for _, s, e, _ in edits])
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

        for k, (i, start, end, new_text) in enumerate(edits):
            if progress is not None:
                progress(k / max(len(edits), 1),
                         desc=f"Re-voicing edit {k+1}/{len(edits)}")

            clip_wav = self.voice.run(
                text=new_text,
                reference_audio_path=state["audio"],
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
