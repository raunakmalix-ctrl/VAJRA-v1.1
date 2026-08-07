"""
Central path configuration for VAJRA v1.1 (Colab).

Everything derives from PROJECT_ROOT. In Colab the repo is cloned to
/content/VAJRA-v1.1 and this resolves automatically. Override with the
VAJRA_ROOT env var if you clone elsewhere.

Model weights live under MODEL_ROOT. To persist them across Colab sessions,
set MODEL_ROOT to a Google Drive path (e.g. /content/drive/MyDrive/vajra/models)
via the VAJRA_MODELS env var before launching.
"""
import os
import shutil

# ── Roots ───────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("VAJRA_ROOT", os.path.dirname(_HERE))
MODEL_ROOT   = os.environ.get("VAJRA_MODELS", os.path.join(PROJECT_ROOT, "models"))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
UPLOADS_DIR  = os.path.join(PROJECT_ROOT, "uploads")
THIRD_PARTY  = os.path.join(PROJECT_ROOT, "third_party")   # cloned model repos
VENV_ROOT    = os.environ.get("VAJRA_VENVS", os.path.join(PROJECT_ROOT, "venvs"))
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
VIITOR_DIR     = os.path.join(THIRD_PARTY, "viitor-voice-nar")

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
# The weights and the UNet config are a matched pair, not two settings: 1.6 was
# trained at 512x512 and its checkpoint does not fit the 256px stage2 config.
# Changing one without the other loads a mismatched model, so they live
# together here and tests/ checks they stay in step.
#
# 1.6 over 1.5 for one reason: 1.5 generates a 256px face crop that is upscaled
# back into the frame, leaving a soft mouth inside a sharp face -- the most
# reliable visual tell in an edited clip. 1.6 doubles that to 512.
LATENTSYNC_HF_REPO = os.environ.get("LATENTSYNC_HF_REPO",
                                    "ByteDance/LatentSync-1.6")
LATENTSYNC_CONFIG = os.path.join(LATENTSYNC_DIR, "configs", "unet",
                                 "stage2_512.yaml")

# Transcript
WHISPERX_MODEL = os.environ.get("WHISPERX_MODEL", "large-v3")

# ── Isolated venv interpreters (built by setup/make_venvs.sh) ───────────────
def _venv_python(name):
    return os.path.join(VENV_ROOT, name, "bin", "python")

VENV_VOICE_PY      = _venv_python("venv_voice")
VENV_DEMUCS_PY     = _venv_python("venv_demucs")
VENV_VIITOR_PY     = _venv_python("venv_viitor")
VENV_LATENTSYNC_PY = _venv_python("venv_latentsync")
VENV_LTX2_PY       = _venv_python("venv_ltx2")
VENV_WAN_PY        = _venv_python("venv_wan")
VENV_QWEN_PY       = _venv_python("venv_qwen")

# Demucs v4 (htdemucs, MIT) speech/background separation for Video Edit.
# Editing the speech stem alone and re-laying the UNTOUCHED background over the
# result is the single strongest realism measure available: the room ambience,
# music and traffic play continuously across the edit because they were never
# regenerated. Runs in its own venv (built by setup/make_demucs_venv.sh) -- it
# pins its own torch/torchaudio pair, which conflicts with the principal
# runtime's diffusion stack.
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")

# ── ViiTorVoice-NAR (tier A local infill) ───────────────────────────────────
# Ships as five gRPC services behind an HTTP gateway rather than as a library,
# so it is supervised as a process group and spoken to over HTTP on localhost.
VIITOR_REPO      = "https://github.com/viitor-ai/viitor-voice-nar.git"
VIITOR_HF_REPO   = "ZzWater/ViiTorVoice-NAR"
# Weights go under MODEL_ROOT so USE_DRIVE persists them like every other model,
# rather than re-downloading a multi-GB stack every session.
VIITOR_MODELS    = os.path.join(MODEL_ROOT, "viitor")
VIITOR_HTTP_PORT = int(os.environ.get("VIITOR_HTTP_PORT", "7861"))
VIITOR_HOST      = os.environ.get("VIITOR_HOST", "127.0.0.1")
VIITOR_BASE_URL  = f"http://{VIITOR_HOST}:{VIITOR_HTTP_PORT}"
# The gRPC services behind the gateway. The gateway answers /health as soon as
# IT is up, which is not the same as the group being usable -- a request then
# fails with "failed to connect to 127.0.0.1:51051". Readiness has to mean every
# backing service is accepting connections, so their ports are listed here.
VIITOR_SERVICE_PORTS = {
    "encoder": int(os.environ.get("VIITOR_ENCODER_PORT", "51051")),
    "llm": int(os.environ.get("VIITOR_LLM_PORT", "51052")),
    "decoder": int(os.environ.get("VIITOR_DECODER_PORT", "51053")),
    "orchestrator": int(os.environ.get("VIITOR_ORCH_PORT", "50051")),
}
# English only, by deliberate choice. The upstream aligner defaults to Chinese,
# so this has to be stated explicitly or an English edit is aligned with the
# wrong model.
VIITOR_LANGUAGE  = os.environ.get("VIITOR_LANGUAGE", "en")
# Cold start loads five models; the gateway answers /health only once the whole
# group is up, so the first call has to wait rather than assume.
VIITOR_START_TIMEOUT_SEC = int(os.environ.get("VIITOR_START_TIMEOUT_SEC", "600"))
VIITOR_REQUEST_TIMEOUT_SEC = int(
    os.environ.get("VIITOR_REQUEST_TIMEOUT_SEC", "900"))

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
