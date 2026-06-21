"""
Re-sync the lips of a real video to a new audio track.

Primary:  LatentSync (current SOTA diffusion lip-sync) via venv_latentsync.
Fallback: Wav2Lip GAN (same venv) if LatentSync fails.

Both tools ship their own CLIs; we invoke them with the isolated venv's python
and return the muxed output path.
"""
import os
import subprocess

from core.base_engine import BaseEngine
from core.utils import timestamp_file, to_wav
from core.config import (
    VENV_LATENTSYNC_PY, LATENTSYNC_DIR, LATENTSYNC_CKPT, LATENTSYNC_CONFIG,
    WAV2LIP_DIR, WAV2LIP_CKPT, FFMPEG_PATH, FFPROBE_PATH,
)


class LipSyncEngine(BaseEngine):

    def run(self, video_path, audio_path, method="latentsync",
            inference_steps=20, guidance_scale=1.5):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        if not os.path.exists(VENV_LATENTSYNC_PY):
            raise RuntimeError("venv_latentsync missing — run setup/make_venvs.sh")

        wav_path, is_tmp = to_wav(audio_path)
        tmp = [wav_path] if is_tmp else []
        try:
            if method == "latentsync":
                try:
                    return self._run_latentsync(video_path, wav_path,
                                                inference_steps, guidance_scale)
                except Exception as e:
                    print(f"[LipSync] LatentSync failed ({e}); "
                          f"falling back to Wav2Lip.")
            return self._run_wav2lip(video_path, wav_path)
        finally:
            for f in tmp:
                try:
                    os.unlink(f)
                except Exception:
                    pass

    # ── LatentSync ──────────────────────────────────────────────────────────
    def _run_latentsync(self, video_path, wav_path, steps, guidance):
        out_path = timestamp_file("relip", "mp4")
        cmd = [
            VENV_LATENTSYNC_PY,
            os.path.join(LATENTSYNC_DIR, "scripts", "inference.py"),
            "--unet_config_path", LATENTSYNC_CONFIG,
            "--inference_ckpt_path", LATENTSYNC_CKPT,
            "--video_path", video_path,
            "--audio_path", wav_path,
            "--video_out_path", out_path,
            "--inference_steps", str(steps),
            "--guidance_scale", str(guidance),
        ]
        print("[LipSync] Running LatentSync ...")
        proc = subprocess.run(cmd, cwd=LATENTSYNC_DIR,
                              capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(
                f"LatentSync exit {proc.returncode}\n"
                f"{proc.stdout[-800:]}\n{proc.stderr[-1500:]}"
            )
        print(f"[LipSync] Output: {out_path}")
        return out_path

    # ── Wav2Lip fallback ──────────────────────────────────────────────────────
    def _run_wav2lip(self, video_path, wav_path):
        out_path = timestamp_file("relip", "mp4")
        cmd = [
            VENV_LATENTSYNC_PY,
            os.path.join(WAV2LIP_DIR, "inference.py"),
            "--checkpoint_path", WAV2LIP_CKPT,
            "--face", video_path,
            "--audio", wav_path,
            "--outfile", out_path,
            "--nosmooth",
            "--pads", "0", "20", "0", "0",
        ]
        print("[LipSync] Running Wav2Lip (fallback) ...")
        proc = subprocess.run(cmd, cwd=WAV2LIP_DIR,
                              capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(
                f"Wav2Lip exit {proc.returncode}\n"
                f"{proc.stdout[-800:]}\n{proc.stderr[-1500:]}"
            )

        # Ensure the output carries an audio track.
        probe = subprocess.run(
            [FFPROBE_PATH, "-v", "quiet", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", out_path],
            capture_output=True, text=True,
        )
        if "audio" not in probe.stdout:
            muxed = out_path.replace(".mp4", "_a.mp4")
            r = subprocess.run(
                [FFMPEG_PATH, "-y", "-i", out_path, "-i", wav_path,
                 "-c:v", "copy", "-c:a", "aac", "-shortest", muxed],
                capture_output=True, text=True,
            )
            if r.returncode == 0 and os.path.exists(muxed):
                os.replace(muxed, out_path)
        print(f"[LipSync] Output: {out_path}")
        return out_path
