"""
Central path configuration for Image-Talk v1.1 (Colab).

Everything derives from PROJECT_ROOT. In Colab the repo is cloned to
/content/Image-Talkv1.1 and this resolves automatically. Override with the
IMAGE_TALK_ROOT env var if you clone elsewhere.

Model weights live under MODEL_ROOT. To persist them across Colab sessions,
set MODEL_ROOT to a Google Drive path (e.g. /content/drive/MyDrive/image_talk_models)
via the IMAGE_TALK_MODELS env var before launching.
"""
import os
import shutil

# ── Roots ───────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("IMAGE_TALK_ROOT", os.path.dirname(_HERE))
MODEL_ROOT   = os.environ.get("IMAGE_TALK_MODELS", os.path.join(PROJECT_ROOT, "models"))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
UPLOADS_DIR  = os.path.join(PROJECT_ROOT, "uploads")
THIRD_PARTY  = os.path.join(PROJECT_ROOT, "third_party")   # cloned model repos
VENV_ROOT    = os.environ.get("IMAGE_TALK_VENVS", os.path.join(PROJECT_ROOT, "venvs"))

for _d in (MODEL_ROOT, OUTPUTS_DIR, UPLOADS_DIR, THIRD_PARTY, VENV_ROOT):
    os.makedirs(_d, exist_ok=True)

# ── Cloned model repositories (third_party/) ────────────────────────────────
SADTALKER_DIR  = os.path.join(THIRD_PARTY, "SadTalker")
WAV2LIP_DIR    = os.path.join(THIRD_PARTY, "Wav2Lip")
LATENTSYNC_DIR = os.path.join(THIRD_PARTY, "LatentSync")

# ── Model weight paths ──────────────────────────────────────────────────────
# Text → image
FLUX_DEV_REPO     = "black-forest-labs/FLUX.1-dev"
FLUX_SCHNELL_REPO = "black-forest-labs/FLUX.1-schnell"

# Face swap
INSIGHTFACE_ROOT = os.path.join(MODEL_ROOT, "insightface")
INSWAPPER_PATH   = os.path.join(INSIGHTFACE_ROOT, "models", "inswapper_128.onnx")
GFPGAN_PATH      = os.path.join(MODEL_ROOT, "gfpgan", "GFPGANv1.4.pth")
CODEFORMER_PATH  = os.path.join(MODEL_ROOT, "codeformer", "codeformer.pth")

# Voice clone
XTTS_DIR = os.path.join(MODEL_ROOT, "xtts")

# Talking head — SadTalker reads from its own repo checkpoints dir (populated
# by its bundled downloader, which is the reliable path).
SADTALKER_CKPT_DIR = os.path.join(SADTALKER_DIR, "checkpoints")

# Lip sync — LatentSync likewise reads from its repo checkpoints dir.
WAV2LIP_CKPT      = os.path.join(MODEL_ROOT, "wav2lip", "wav2lip_gan.pth")
LATENTSYNC_CKPT   = os.path.join(LATENTSYNC_DIR, "checkpoints", "latentsync_unet.pt")
LATENTSYNC_CONFIG = os.path.join(LATENTSYNC_DIR, "configs", "unet", "stage2.yaml")

# Transcript
WHISPERX_MODEL = os.environ.get("WHISPERX_MODEL", "large-v3")

# ── Isolated venv interpreters (built by setup/make_venvs.sh) ───────────────
def _venv_python(name):
    return os.path.join(VENV_ROOT, name, "bin", "python")

VENV_VOICE_PY      = _venv_python("venv_voice")
VENV_SADTALKER_PY  = _venv_python("venv_sadtalker")
VENV_LATENTSYNC_PY = _venv_python("venv_latentsync")
VENV_LTX_PY        = _venv_python("venv_ltx")

# LTX-2 (Lightricks) text/image -> video. Needs latest diffusers + torch ~2.7
# + Python 3.12, so it lives in its own venv (built by setup/make_ltx_venv.sh).
LTX_REPO = "Lightricks/LTX-2"

# ── FFmpeg (Colab: apt-installed, on PATH) ──────────────────────────────────
FFMPEG_PATH  = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_PATH = shutil.which("ffprobe") or "ffprobe"
