"""
Speech/background separation (Demucs v4), run in venv_demucs via a worker.

Used by Edit & Relip: the speech stem is what gets edited, and the untouched
background is re-laid over the finished result so the room, music and ambience
play continuously across every join. See workers/demucs_worker.py for why that
matters more than any single matching stage.

Separation is optional. When its environment is not built, callers fall back to
editing the mixed track -- which still works, just with more for the realism
layer to disguise.
"""
import os

from core.base_engine import BaseEngine
from core.utils import timestamp_file
from core.config import VENV_DEMUCS_PY, DEMUCS_MODEL, PROJECT_ROOT
from core.subprocess_runner import run_worker

WORKER = os.path.join(PROJECT_ROOT, "workers", "demucs_worker.py")


class SeparateEngine(BaseEngine):

    def available(self):
        """Whether separation can run, without raising.

        Lets callers offer separation when it is present and quietly proceed
        without it when it is not, rather than failing a whole edit over an
        optional enhancement.
        """
        return os.path.exists(VENV_DEMUCS_PY)

    def run(self, audio_path, model=None):
        """Separate `audio_path`. Returns (speech_path, background_path)."""
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        speech_out = timestamp_file("stem_speech", "wav")
        bg_out = timestamp_file("stem_background", "wav")

        # Both output paths are passed in explicitly and the worker writes
        # exactly those, so the single-RESULT-path contract still holds and the
        # caller never has to infer a filename by convention.
        returned = run_worker(
            VENV_DEMUCS_PY, WORKER,
            {
                "audio": audio_path,
                "speech_out": speech_out,
                "background_out": bg_out,
                "model": model or DEMUCS_MODEL,
            },
            cwd=PROJECT_ROOT,
            timeout=3600,
        )

        if not os.path.exists(bg_out):
            raise RuntimeError(
                "Separation produced a speech stem but no background stem "
                f"({bg_out} missing)."
            )
        return returned, bg_out
