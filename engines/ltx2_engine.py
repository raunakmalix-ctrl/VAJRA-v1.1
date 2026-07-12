"""
LTX-2.3 (Lightricks) text -> video, or image+prompt -> motion video, with
synchronized audio, run in its own venv_ltx2 via a worker subprocess.

Covers both of Text -> Video's paths: prompt-only (previously
LTX-Video-0.9.7-distilled, removed -- LTX-2.3 supersedes it, using
diffusers' LTX2Pipeline) and reference-photo motion video (an alternative
to Wan2.2-I2V, using LTX2ImageToVideoPipeline: faster/lighter and
additionally generates synchronized audio, since Wan2.2-I2V is video-only,
trading some multi-subject identity fidelity per early comparisons).
workers/ltx2_worker.py picks the pipeline class based on whether an image
path is given.
"""
import os

from core.base_engine import BaseEngine
from core.utils import timestamp_file
from core.config import VENV_LTX2_PY, LTX2_REPO, PROJECT_ROOT
from core.subprocess_runner import run_worker

WORKER = os.path.join(PROJECT_ROOT, "workers", "ltx2_worker.py")


class LTX2Engine(BaseEngine):

    def run(self, prompt, image_path=None, negative_prompt="",
            width=768, height=512, num_frames=121, steps=40, fps=24, seed=-1):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if not os.path.exists(VENV_LTX2_PY):
            raise RuntimeError(
                "venv_ltx2 missing — run setup/make_ltx2_venv.sh first "
                "(LTX 2.3 is an optional, heavy add-on)."
            )

        out_path = timestamp_file("ltx2", "mp4")
        return run_worker(
            VENV_LTX2_PY, WORKER,
            {
                "repo": LTX2_REPO,
                "image": image_path,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width, "height": height,
                "num_frames": num_frames, "steps": steps,
                "fps": fps, "seed": seed,
                "out_path": out_path,
            },
            cwd=PROJECT_ROOT,
            timeout=3600,
        )
