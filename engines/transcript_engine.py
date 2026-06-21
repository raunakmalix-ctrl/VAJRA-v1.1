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
_align_cache   = {}


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisperx
        compute = "float16" if DEVICE == "cuda" else "int8"
        print(f"[Transcript] Loading WhisperX ({WHISPERX_MODEL}) ...")
        _whisper_model = whisperx.load_model(
            WHISPERX_MODEL, DEVICE, compute_type=compute
        )
    return _whisper_model


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
        import whisperx
        audio_path = _extract_audio(video_path)

        model = _load_whisper()
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        lang = result.get("language", "en")

        # Word-level alignment (best-effort; some langs lack an align model).
        try:
            if lang not in _align_cache:
                _align_cache[lang] = whisperx.load_align_model(
                    language_code=lang, device=DEVICE
                )
            amodel, meta = _align_cache[lang]
            result = whisperx.align(
                result["segments"], amodel, meta, audio, DEVICE
            )
        except Exception as e:
            print(f"[Transcript] Alignment skipped ({e}).")

        segments = [
            {"start": float(s["start"]), "end": float(s["end"]),
             "text": s["text"].strip()}
            for s in result["segments"] if s.get("text", "").strip()
        ]
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

        if len(new_lines) != len(segments):
            raise ValueError(
                f"Edited transcript has {len(new_lines)} lines but the original "
                f"has {len(segments)} segments. Keep one segment per line "
                f"(edit words, don't add/remove line breaks)."
            )

        original = AudioSegment.from_file(state["audio"])
        result = AudioSegment.empty()
        cursor_ms = 0
        changed_count = 0

        for i, seg in enumerate(segments):
            start_ms = int(seg["start"] * 1000)
            end_ms   = int(seg["end"] * 1000)
            if start_ms > cursor_ms:
                result += original[cursor_ms:start_ms]   # keep gaps/silence

            if new_lines[i] != seg["text"]:
                changed_count += 1
                if progress:
                    progress(i / len(segments),
                             desc=f"Re-voicing segment {i+1}/{len(segments)}")
                clip_wav = self.voice.run(
                    text=new_lines[i],
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

        if progress:
            progress(0.7, desc="Re-syncing lips ...")
        out = self.lipsync.run(
            video_path=state["video"], audio_path=new_audio,
            method=method, inference_steps=inference_steps,
            guidance_scale=guidance_scale,
        )
        return out
