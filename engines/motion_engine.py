"""
Wan2.2-I2V motion video: a reference photo + text prompt -> identity-
preserving motion video, run in venv_wan via a worker subprocess.

Replaces classic audio-driven "talking head" animation (SadTalker/MuseTalk)
for the Text -> Video tab's optional-image path: the uploaded photo becomes
the first frame and the model generates motion following the prompt, with no
audio/lip-sync involved. Because it's whole-image conditioned rather than
per-face like SadTalker/MuseTalk, multi-subject photos are handled directly.
"""
import os

from core.base_engine import BaseEngine
from core.utils import timestamp_file
from core.config import VENV_WAN_PY, WAN_I2V_REPO, PROJECT_ROOT
from core.subprocess_runner import run_worker

WORKER = os.path.join(PROJECT_ROOT, "workers", "wan_i2v_worker.py")


class MotionVideoEngine(BaseEngine):

    def run(self, image_path, prompt, negative_prompt="",
            width=832, height=480, num_frames=81,
            guidance=5.0, fps=16, seed=-1):
        if not image_path:
            raise ValueError("A reference image is required.")
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if not os.path.exists(VENV_WAN_PY):
            raise RuntimeError(
                "venv_wan missing — run setup/make_wan_venv.sh first "
                "(Wan2.2-I2V is an optional, heavy add-on)."
            )

        out_path = timestamp_file("motion", "mp4")
        return run_worker(
            VENV_WAN_PY, WORKER,
            {
                "repo": WAN_I2V_REPO,
                "image": image_path,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width, "height": height,
                "num_frames": num_frames,
                "guidance": guidance, "fps": fps, "seed": seed,
                "out_path": out_path,
            },
            cwd=PROJECT_ROOT,
            timeout=3600,
        )
