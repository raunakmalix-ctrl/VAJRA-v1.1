"""
Voice cloning + multilingual TTS (English / Hindi) with XTTS-v2.

XTTS pins old transformers/torch that clash with the main FLUX env, so it runs
inside `venv_voice` via a worker subprocess. This orchestrator just marshals
arguments and returns the generated .wav path.
"""
import os

from core.base_engine import BaseEngine
from core.utils import timestamp_file
from core.config import VENV_VOICE_PY, XTTS_DIR, PROJECT_ROOT
from core.subprocess_runner import run_worker

WORKER = os.path.join(PROJECT_ROOT, "workers", "voice_worker.py")

SUPPORTED_LANGUAGES = {"English": "en", "Hindi": "hi"}


class VoiceEngine(BaseEngine):

    def run(self, text, reference_audio_path, language="en"):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

        out_path = timestamp_file("voice", "wav")
        return run_worker(
            VENV_VOICE_PY, WORKER,
            {
                "text": text,
                "reference_audio": reference_audio_path,
                "language": language,
                "xtts_dir": XTTS_DIR,
                "out_path": out_path,
            },
            cwd=PROJECT_ROOT,
            timeout=1800,
        )
