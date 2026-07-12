"""
Central path configuration for VAJRA v1.1 (Colab).

Everything derives from PROJECT_ROOT. In Colab the repo is cloned to
/content/VAJRA-v1.1 and this resolves automatically. Override with the
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
WAV2LIP_DIR    = os.path.join(THIRD_PARTY, "Wav2Lip")
LATENTSYNC_DIR = os.path.join(THIRD_PARTY, "LatentSync")
MUSETALK_DIR   = os.path.join(THIRD_PARTY, "MuseTalk")
CODEFORMER_DIR = os.path.join(THIRD_PARTY, "CodeFormer")

# MuseTalk v1.5 weights live under its repo models/ (download_weights.sh).
MUSETALK_UNET        = os.path.join(MUSETALK_DIR, "models", "musetalkV15", "unet.pth")
MUSETALK_UNET_CONFIG = os.path.join(MUSETALK_DIR, "models", "musetalkV15", "musetalk.json")

# ── Model weight paths ──────────────────────────────────────────────────────
# Text → image
FLUX_DEV_REPO     = "black-forest-labs/FLUX.1-dev"
FLUX_SCHNELL_REPO = "black-forest-labs/FLUX.1-schnell"
# Instruction-based image editing (gated, needs HF_TOKEN + license)
FLUX_KONTEXT_REPO = "black-forest-labs/FLUX.1-Kontext-dev"

# Face swap
INSIGHTFACE_ROOT = os.path.join(MODEL_ROOT, "insightface")
INSWAPPER_PATH   = os.path.join(INSIGHTFACE_ROOT, "models", "inswapper_128.onnx")
GFPGAN_PATH      = os.path.join(MODEL_ROOT, "gfpgan", "GFPGANv1.4.pth")
CODEFORMER_PATH  = os.path.join(MODEL_ROOT, "codeformer", "codeformer.pth")

# Voice clone
XTTS_DIR = os.path.join(MODEL_ROOT, "xtts")

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
VENV_LATENTSYNC_PY = _venv_python("venv_latentsync")
VENV_LTX_PY        = _venv_python("venv_ltx")
VENV_MUSETALK_PY   = _venv_python("venv_musetalk")
VENV_WAN_PY        = _venv_python("venv_wan")

# LTX-Video 0.9.7-distilled (Lightricks) text -> video. Open (no token), fast
# few-step. Runs in its own venv (built by setup/make_ltx_venv.sh).
LTX_REPO = "Lightricks/LTX-Video-0.9.7-distilled"

# Wan2.2-I2V (Alibaba/Tongyi Wanxiang) image+prompt -> motion video, identity
# preserving, handles multi-subject images (not per-face like SadTalker/
# MuseTalk -- the uploaded photo is the first frame, diffusion generates the
# rest following the prompt). Apache-2.0, open. Runs in its own venv (built by
# setup/make_wan_venv.sh) -- needs transformers 4.49-4.51.3, incompatible with
# venv_ltx's <4.50 pin (see requirements/ltx.txt), hence a separate venv.
WAN_I2V_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# ── FFmpeg (Colab: apt-installed, on PATH) ──────────────────────────────────
FFMPEG_PATH  = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_PATH = shutil.which("ffprobe") or "ffprobe"
