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
HF_CACHE_DIR = os.path.join(MODEL_ROOT, "hf_cache")

for _d in (MODEL_ROOT, OUTPUTS_DIR, UPLOADS_DIR, THIRD_PARTY, VENV_ROOT, HF_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# Every from_pretrained()/snapshot_download() call in this project (SDXL
# here in the main process, LTX/Qwen-Edit/Wan2.2 in their subprocess workers)
# otherwise falls back to Hugging Face's default cache
# (~/.cache/huggingface/hub) -- local to the Colab VM disk, wiped every
# session, and NOT covered by the MODEL_ROOT Drive redirect above despite
# USE_DRIVE claiming to cache "model weights". Route it through MODEL_ROOT
# too, so USE_DRIVE actually covers everything. Derived fresh from MODEL_ROOT
# on every import (rather than a notebook cell setting it once) so it's
# correct in every process: the main app imports core.config directly;
# subprocess workers inherit it via core/subprocess_runner.py's clean_env(),
# which copies the parent's environment; the download scripts import
# core.config directly too.
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE_DIR)

# ── Cloned model repositories (third_party/) ────────────────────────────────
WAV2LIP_DIR    = os.path.join(THIRD_PARTY, "Wav2Lip")
LATENTSYNC_DIR = os.path.join(THIRD_PARTY, "LatentSync")
CODEFORMER_DIR = os.path.join(THIRD_PARTY, "CodeFormer")

# ── Model weight paths ──────────────────────────────────────────────────────
# Face swap
INSIGHTFACE_ROOT = os.path.join(MODEL_ROOT, "insightface")
INSWAPPER_PATH   = os.path.join(INSIGHTFACE_ROOT, "models", "inswapper_128.onnx")
GFPGAN_PATH      = os.path.join(MODEL_ROOT, "gfpgan", "GFPGANv1.4.pth")
CODEFORMER_PATH  = os.path.join(MODEL_ROOT, "codeformer", "codeformer.pth")

# Voice clone
XTTS_DIR = os.path.join(MODEL_ROOT, "xtts")

# Lip sync — LatentSync's own inference script expects its checkpoint at a
# fixed path inside the cloned repo (LATENTSYNC_CKPT). The real weight files
# live under MODEL_ROOT (Drive-persisted); download_models.py symlinks them
# into place -- previously this path pointed straight into third_party/,
# which is never Drive-cached and was silently rebuilt every session.
LATENTSYNC_WEIGHTS_DIR = os.path.join(MODEL_ROOT, "latentsync")
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
VENV_LTX2_PY       = _venv_python("venv_ltx2")
VENV_WAN_PY        = _venv_python("venv_wan")
VENV_QWEN_PY       = _venv_python("venv_qwen")

# LTX-2.3 (Lightricks) text -> video, or image+prompt -> motion video, WITH
# synchronized audio (a single DiT-based audio-video model; diffusers exposes
# separate pipeline classes -- LTX2Pipeline vs LTX2ImageToVideoPipeline -- for
# the two modes, selected by workers/ltx2_worker.py based on whether a
# reference image is given). Covers the Text -> Video tab's prompt-only path
# (previously LTX-Video-0.9.7-distilled, removed) as well as being an
# alternative to Wan2.2-I2V for the reference-photo path: faster/lighter,
# generates audio too, trades some multi-subject identity fidelity per early
# comparisons. Confirmed diffusers-compatible checkpoint per
# https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2 -- runs in
# its own venv (built by setup/make_ltx2_venv.sh): its pipeline
# unconditionally imports Gemma3ForConditionalGeneration (its default text
# encoder) at module load time, needing transformers>=~4.50 (confirmed via a
# real ImportError in Colab when this was still sharing a venv with the now-
# removed LTX-Video-0.9.7-distilled, which needed transformers<4.50).
LTX2_REPO = "diffusers/LTX-2.3-Diffusers"

# Wan2.2-I2V (Alibaba/Tongyi Wanxiang) image+prompt -> motion video, identity
# preserving, handles multi-subject images (not per-face like classic
# talking-head methods -- the uploaded photo is the first frame, diffusion
# generates the rest following the prompt). Apache-2.0, open. Runs in its own
# venv (built by setup/make_wan_venv.sh) -- needs transformers 4.49-4.51.3,
# incompatible with venv_qwen/venv_ltx2's git-installed transformers and with
# every other venv's pins, hence a separate venv.
WAN_I2V_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# Qwen-Image-Edit-2509 ("Plus") — instruction-based image editing, replaces
# FLUX.1-Kontext-dev for the Image Edit tab. Apache-2.0, fully open (no
# token/license click-through, unlike Kontext), and supports 1-3 reference
# images per edit (e.g. "person + product") -- Kontext only took one.
# Confirmed official diffusers support (QwenImageEditPlusPipeline). Its
# Qwen2.5-VL-7B-Instruct text encoder needs transformers>=4.49, and the
# maintainers recommend installing transformers from git to guarantee the
# qwen2_5_vl model type is registered -- runs in its own venv (built by
# setup/make_qwen_venv.sh).
QWEN_EDIT_REPO = "Qwen/Qwen-Image-Edit-2509"

# ── FFmpeg (Colab: apt-installed, on PATH) ──────────────────────────────────
FFMPEG_PATH  = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_PATH = shutil.which("ffprobe") or "ffprobe"
