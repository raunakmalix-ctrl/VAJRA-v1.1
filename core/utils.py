import os
import time
import tempfile
import subprocess

from core.config import OUTPUTS_DIR, FFMPEG_PATH, FFPROBE_PATH


# Every artefact this app produces carries one prefix, so a downloaded file is
# identifiable as VAJRA output without needing the folder it came from. Applied
# here rather than at each call site so no engine can forget it.
OUTPUT_PREFIX = "vajra"


def timestamp_file(prefix, ext):
    """Return a timestamped path inside OUTPUTS_DIR."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S_") + str(int(time.time() * 1000) % 1000)
    stem = str(prefix or "output").strip("_")
    if not stem.startswith(OUTPUT_PREFIX):
        stem = f"{OUTPUT_PREFIX}_{stem}"
    return os.path.join(OUTPUTS_DIR, f"{stem}_{ts}.{ext}")


def to_wav(audio_path, sr=16000, channels=1):
    """
    Convert any audio file to a temp WAV at the given sample rate / channels.
    Returns (wav_path, is_tmp). Caller deletes wav_path if is_tmp is True.
    """
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        return audio_path, False

    from pydub import AudioSegment
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffprobe   = FFPROBE_PATH

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(sr).set_channels(channels)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    audio.export(tmp.name, format="wav")
    return tmp.name, True


def transcode_h264(video_path):
    """
    Re-encode any video to H.264 / yuv420p so OpenCV/ffmpeg can decode it
    everywhere (Colab can't decode AV1, which many phone/screen recordings use).
    Returns a new mp4 path.
    """
    out = timestamp_file("norm", "mp4")
    subprocess.run(
        [FFMPEG_PATH, "-y", "-i", video_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-c:a", "aac", out],
        check=True, capture_output=True, text=True,
    )
    return out


def mux_audio(video_path, audio_path, out_path=None):
    """Replace video_path's audio track with audio_path's (re-encoded to
    AAC); the video stream is copied untouched. The shorter of the two
    streams determines the output length. Returns the new video path.

    This is audio PAIRING, not conditioning -- the video's visual content is
    already generated and unaffected by audio_path's actual content."""
    out_path = out_path or timestamp_file("muxed", "mp4")
    subprocess.run(
        [FFMPEG_PATH, "-y", "-i", video_path, "-i", audio_path,
         "-c:v", "copy", "-c:a", "aac",
         "-map", "0:v:0", "-map", "1:a:0",
         "-shortest", out_path],
        check=True, capture_output=True, text=True,
    )
    return out_path


def audio_duration(audio_path):
    """Duration in seconds, or None on failure."""
    try:
        from pydub import AudioSegment
        AudioSegment.converter = FFMPEG_PATH
        AudioSegment.ffprobe   = FFPROBE_PATH
        return len(AudioSegment.from_file(audio_path)) / 1000.0
    except Exception:
        return None
