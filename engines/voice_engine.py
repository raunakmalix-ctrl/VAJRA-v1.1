"""
Voice cloning + multilingual TTS with XTTS-v2 (17 languages).

XTTS pins old transformers/torch that clash with the main SDXL env, so it runs
inside `venv_voice` via a worker subprocess. This orchestrator just marshals
arguments and returns the generated .wav path.
"""
import os

from core.base_engine import BaseEngine
from core.utils import timestamp_file
from core.config import VENV_VOICE_PY, XTTS_DIR, PROJECT_ROOT
from core.subprocess_runner import run_worker

WORKER = os.path.join(PROJECT_ROOT, "workers", "voice_worker.py")

# XTTS-v2's full documented language set. Only English and Hindi were exposed
# before, so an edit to Spanish or French audio was silently synthesised with
# English pronunciation -- the model could always do better than the UI allowed.
SUPPORTED_LANGUAGES = {
    "English": "en",      "Spanish": "es",     "French": "fr",
    "German": "de",       "Italian": "it",     "Portuguese": "pt",
    "Polish": "pl",       "Turkish": "tr",     "Russian": "ru",
    "Dutch": "nl",        "Czech": "cs",       "Arabic": "ar",
    "Chinese": "zh-cn",   "Japanese": "ja",    "Hungarian": "hu",
    "Korean": "ko",       "Hindi": "hi",
}


# A few built-in XTTS-v2 preset voices (male / female).
PRESET_VOICES = {
    "Female · Claribel": "Claribel Dervla",
    "Female · Daisy":    "Daisy Studious",
    "Female · Gracie":   "Gracie Wise",
    "Male · Andrew":     "Andrew Chipper",
    "Male · Damien":     "Damien Black",
    "Male · Viktor":     "Viktor Eka",
}



class VoiceEngine(BaseEngine):

    def run(self, text, reference_audio_path=None, language="en", speaker=None):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")
        if not speaker:
            if not reference_audio_path or not os.path.exists(reference_audio_path):
                raise FileNotFoundError("Provide a reference audio to clone, or pick a preset voice.")

        out_path = timestamp_file("voice", "wav")
        # Digits are expanded inside the worker, not here: the expansion wants
        # num2words, which lives in venv_voice rather than the principal
        # runtime. Doing it here would silently fall back to reading numbers
        # digit by digit even for languages num2words handles properly.
        return run_worker(
            VENV_VOICE_PY, WORKER,
            {
                "text": text,
                "reference_audio": reference_audio_path,
                "speaker": speaker,
                "language": language,
                "xtts_dir": XTTS_DIR,
                "out_path": out_path,
            },
            cwd=PROJECT_ROOT,
            timeout=1800,
        )
