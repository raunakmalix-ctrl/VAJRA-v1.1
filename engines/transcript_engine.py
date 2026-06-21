"""
Feature 2 — transcript-driven lip editing.

Flow:
  1. extract_transcript(video): pull audio, run WhisperX for segment + word
     timestamps, return an editable transcript (one segment per line).
  2. apply_edits(state, edited_text, ...): for each line the user changed,
     re-synthesize just that segment in the speaker's own cloned voice (XTTS),
     time-fit it to the original segment's duration, splice it into the audio
     track (total length preserved), then re-sync the video's lips with
     LatentSync.

WhisperX runs in the main env. Voice synthesis and lip-sync delegate to the
isolated-venv engines.
"""
import os
import subprocess
import tempfile

from core.base_engine import BaseEngine
from core.utils import timestamp_file
from core.config import FFMPEG_PATH, FFPROBE_PATH, WHISPERX_MODEL
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


def _seg_seconds(audio_seg):
    return len(audio_seg) / 1000.0


def _fit_duration(clip_wav, target_sec):
    """Time-stretch a wav to exactly target_sec using ffmpeg atempo, return an
    AudioSegment trimmed/padded to the exact target length."""
    from pydub import AudioSegment
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffprobe   = FFPROBE_PATH

    clip = AudioSegment.from_file(clip_wav)
    src_sec = _seg_seconds(clip)
    target_ms = int(target_sec * 1000)
    if src_sec <= 0 or target_sec <= 0:
        return clip[:target_ms] if target_ms else clip

    rate = src_sec / target_sec   # >1 => speed up, <1 => slow down
    # atempo only accepts 0.5–2.0; chain filters to cover the full range.
    factors = []
    r = rate
    while r > 2.0:
        factors.append(2.0); r /= 2.0
    while r < 0.5:
        factors.append(0.5); r /= 0.5
    factors.append(r)
    afilter = ",".join(f"atempo={f:.4f}" for f in factors)

    stretched = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        [FFMPEG_PATH, "-y", "-i", clip_wav, "-filter:a", afilter, stretched],
        capture_output=True, text=True, check=True,
    )
    out = AudioSegment.from_file(stretched)
    try:
        os.unlink(stretched)
    except Exception:
        pass

    # Snap to exact target length.
    if len(out) > target_ms:
        out = out[:target_ms]
    elif len(out) < target_ms:
        out = out + AudioSegment.silent(duration=target_ms - len(out))
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
        state = {
            "video": video_path,
            "audio": audio_path,
            "segments": segments,
            "language": xtts_lang,
        }
        return display, state

    # ── step 2 ────────────────────────────────────────────────────────────────
    def apply_edits(self, state, edited_text, method="latentsync",
                    inference_steps=20, guidance_scale=1.5, progress=None):
        if not state:
            raise ValueError("Extract a transcript first.")

        from pydub import AudioSegment
        AudioSegment.converter = FFMPEG_PATH
        AudioSegment.ffprobe   = FFPROBE_PATH

        segments = state["segments"]
        new_lines = [l.strip() for l in edited_text.split("\n")]
        # Drop trailing empties so a stray newline doesn't break alignment.
        while new_lines and not new_lines[-1]:
            new_lines.pop()

        # Align one segment per line. If the user removed/added line breaks we
        # don't crash: segments without a matching edited line keep their
        # original audio. (For best results, edit words in place per line.)
        original = AudioSegment.from_file(state["audio"])
        result = AudioSegment.empty()
        cursor_ms = 0
        changed_count = 0

        for i, seg in enumerate(segments):
            start_ms = int(seg["start"] * 1000)
            end_ms   = int(seg["end"] * 1000)
            if start_ms > cursor_ms:
                result += original[cursor_ms:start_ms]   # keep gaps/silence

            new_text = new_lines[i] if i < len(new_lines) else seg["text"]

            if new_text and new_text != seg["text"]:
                changed_count += 1
                if progress is not None:
                    progress(i / len(segments),
                             desc=f"Re-voicing segment {i+1}/{len(segments)}")
                clip_wav = self.voice.run(
                    text=new_text,
                    reference_audio_path=state["audio"],
                    language=state["language"],
                )
                seg_audio = _fit_duration(clip_wav, (end_ms - start_ms) / 1000.0)
            else:
                seg_audio = original[start_ms:end_ms]

            result += seg_audio
            cursor_ms = end_ms

        result += original[cursor_ms:]   # tail after last segment

        if changed_count == 0:
            raise ValueError("No changes detected in the transcript.")

        new_audio = timestamp_file("edited_audio", "wav")
        result.export(new_audio, format="wav")

        if progress is not None:
            progress(0.7, desc="Re-syncing lips ...")
        out = self.lipsync.run(
            video_path=state["video"], audio_path=new_audio,
            method=method, inference_steps=inference_steps,
            guidance_scale=guidance_scale,
        )
        return out
