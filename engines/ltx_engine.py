"""
LTX-2 text/image -> video (Lightricks), run in venv_ltx via a worker subprocess.
"""
import os

from core.base_engine import BaseEngine
from core.utils import timestamp_file
from core.config import VENV_LTX_PY, LTX_REPO, PROJECT_ROOT
from core.subprocess_runner import run_worker

WORKER = os.path.join(PROJECT_ROOT, "workers", "ltx_worker.py")


class LTXEngine(BaseEngine):

    def run(self, prompt, negative_prompt="", image_path=None,
            width=768, height=512, num_frames=121,
            steps=40, guidance=4.0, fps=24):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if not os.path.exists(VENV_LTX_PY):
            raise RuntimeError(
                "venv_ltx missing — run setup/make_ltx_venv.sh first "
                "(LTX-2 is an optional, heavy add-on)."
            )

        out_path = timestamp_file("ltx", "mp4")
        return run_worker(
            VENV_LTX_PY, WORKER,
            {
                "repo": LTX_REPO,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image_path,
                "width": width, "height": height,
                "num_frames": num_frames,
                "steps": steps, "guidance": guidance, "fps": fps,
                "out_path": out_path,
            },
            cwd=PROJECT_ROOT,
            timeout=3600,
        )
