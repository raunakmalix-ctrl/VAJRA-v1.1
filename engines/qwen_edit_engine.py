"""
Qwen-Image-Edit-2509 instruction-based image editing, run in venv_qwen via a
worker subprocess. Replaces FLUX.1-Kontext-dev: Apache-2.0/fully open (no
HF_TOKEN or license click-through), and takes 1-3 reference images per edit
(e.g. "person + product", "person + scene") instead of Kontext's single-
image-only input.
"""
import os

from core.base_engine import BaseEngine
from core.utils import timestamp_file
from core.config import VENV_QWEN_PY, QWEN_EDIT_REPO, PROJECT_ROOT
from core.subprocess_runner import run_worker

WORKER = os.path.join(PROJECT_ROOT, "workers", "qwen_edit_worker.py")


class QwenEditEngine(BaseEngine):

    def run(self, image_paths, prompt, negative_prompt="",
            steps=40, guidance=4.0, seed=-1):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        image_paths = [p for p in (image_paths or []) if p]
        if not image_paths:
            raise ValueError("Upload at least one image to edit.")
        if len(image_paths) > 3:
            raise ValueError("Qwen-Image-Edit works best with 1-3 reference images.")
        if not prompt or not prompt.strip():
            raise ValueError("Describe the edit.")
        if not os.path.exists(VENV_QWEN_PY):
            raise RuntimeError(
                "venv_qwen missing — run setup/make_qwen_venv.sh first "
                "(Qwen-Image-Edit is an optional, heavy add-on)."
            )

        out_path = timestamp_file("qwen_edit", "png")
        return run_worker(
            VENV_QWEN_PY, WORKER,
            {
                "repo": QWEN_EDIT_REPO,
                "images": image_paths,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps, "guidance": guidance, "seed": seed,
                "out_path": out_path,
            },
            cwd=PROJECT_ROOT,
            timeout=1800,
        )
