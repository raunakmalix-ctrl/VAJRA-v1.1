"""
Talking head from a single portrait + audio, via SadTalker.

SadTalker has heavy, pinned dependencies, so it runs from `venv_sadtalker`:
we invoke its own inference.py CLI with that venv's python. Default is the
512px model + GFPGAN enhancement (best quality / reliability tradeoff).
"""
import os
import subprocess

from core.base_engine import BaseEngine
from core.utils import timestamp_file, to_wav
from core.subprocess_runner import clean_env
from core.config import (
    VENV_SADTALKER_PY, SADTALKER_DIR, SADTALKER_CKPT_DIR, OUTPUTS_DIR,
)


class TalkingFaceEngine(BaseEngine):

    def run(self, portrait_path, audio_path,
            size=512, enhance_face=True, still_mode=False,
            expression_scale=1.0):
        if not os.path.exists(portrait_path):
            raise FileNotFoundError(f"Portrait not found: {portrait_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        if not os.path.exists(VENV_SADTALKER_PY):
            raise RuntimeError("venv_sadtalker missing — run setup/make_venvs.sh")

        from PIL import Image
        w, h = Image.open(portrait_path).size
        if min(w, h) < 100:
            raise ValueError(f"Portrait too small ({w}x{h}). Need >=100px.")

        wav_path, is_tmp = to_wav(audio_path)
        tmp_files = [wav_path] if is_tmp else []

        try:
            result_dir = os.path.join(OUTPUTS_DIR, "sadtalker_tmp")
            os.makedirs(result_dir, exist_ok=True)

            cmd = [
                VENV_SADTALKER_PY,
                os.path.join(SADTALKER_DIR, "inference.py"),
                "--driven_audio",   wav_path,
                "--source_image",   portrait_path,
                "--checkpoint_dir", SADTALKER_CKPT_DIR,
                "--result_dir",     result_dir,
                "--size",           str(size),
                "--expression_scale", str(expression_scale),
            ]
            if enhance_face:
                cmd += ["--enhancer", "gfpgan"]
            if still_mode:
                cmd += ["--still"]

            print(f"[TalkingFace] Running SadTalker (size={size}, "
                  f"enhance={enhance_face}, still={still_mode}) ...")
            proc = subprocess.run(
                cmd, cwd=SADTALKER_DIR, capture_output=True, text=True,
                env=clean_env(),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"SadTalker failed (exit {proc.returncode}).\n"
                    f"{proc.stdout[-1000:]}\n{proc.stderr[-1500:]}"
                )

            mp4s = []
            for root, _, files in os.walk(result_dir):
                for f in files:
                    if f.endswith(".mp4"):
                        mp4s.append(os.path.join(root, f))
            if not mp4s:
                raise RuntimeError("SadTalker ran but produced no .mp4.")

            latest   = max(mp4s, key=os.path.getmtime)
            out_path = timestamp_file("talkingface", "mp4")
            os.replace(latest, out_path)
            print(f"[TalkingFace] Output: {out_path}")
            return out_path

        finally:
            for f in tmp_files:
                try:
                    os.unlink(f)
                except Exception:
                    pass
