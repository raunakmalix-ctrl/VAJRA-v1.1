"""
Download model weights. Best-effort and idempotent: each model is independent,
so one failure won't abort the rest. Re-run safely.

Notes:
  - FLUX.1-dev is GATED. Accept the license at
    https://huggingface.co/black-forest-labs/FLUX.1-dev and set HF_TOKEN
    (or run `huggingface-cli login`) before launching. FLUX.1-schnell is open.
  - InsightFace 'buffalo_l' detector downloads itself on first face-swap run.
"""
import os
import subprocess
import urllib.request

from core.config import (
    MODEL_ROOT, XTTS_DIR, INSIGHTFACE_ROOT, INSWAPPER_PATH, GFPGAN_PATH,
    WAV2LIP_CKPT, SADTALKER_DIR, LATENTSYNC_DIR,
)

HF_TOKEN = os.environ.get("HF_TOKEN")


def _hf_snapshot(repo_id, local_dir, **kw):
    from huggingface_hub import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=local_dir,
                      token=HF_TOKEN, **kw)


def _hf_file(repo_id, filename, local_path):
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        print(f"  exists: {local_path}"); return
    p = hf_hub_download(repo_id=repo_id, filename=filename, token=HF_TOKEN)
    import shutil; shutil.copy(p, local_path)
    print(f"  -> {local_path}")


def _wget(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        print(f"  exists: {local_path}"); return
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, local_path)
    print(f"  -> {local_path}")


def step(name, fn):
    print(f"==> {name}")
    try:
        fn()
    except Exception as e:
        print(f"  !! {name} failed: {e}")


def xtts():
    if os.path.exists(os.path.join(XTTS_DIR, "model.pth")):
        print("  exists"); return
    _hf_snapshot("coqui/XTTS-v2", XTTS_DIR)


def inswapper():
    _hf_file("ezioruan/inswapper_128.onnx", "inswapper_128.onnx", INSWAPPER_PATH)


def gfpgan():
    _wget("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/"
          "GFPGANv1.4.pth", GFPGAN_PATH)


def wav2lip():
    # Community mirror of the original Wav2Lip GAN checkpoint.
    _hf_file("numz/wav2lip_studio", "Wav2Lip/wav2lip_gan.pth", WAV2LIP_CKPT)


def sadtalker():
    # Use SadTalker's own downloader -> third_party/SadTalker/checkpoints
    script = os.path.join(SADTALKER_DIR, "scripts", "download_models.sh")
    if not os.path.exists(script):
        print("  SadTalker repo not cloned; run install_main.sh first"); return
    subprocess.run(["bash", script], cwd=SADTALKER_DIR, check=True)


def latentsync():
    # LatentSync 1.5 weights -> third_party/LatentSync/checkpoints
    ckpt_dir = os.path.join(LATENTSYNC_DIR, "checkpoints")
    if os.path.exists(os.path.join(ckpt_dir, "latentsync_unet.pt")):
        print("  exists"); return
    _hf_snapshot("ByteDance/LatentSync-1.5", ckpt_dir)


def main():
    step("XTTS-v2 (voice clone)", xtts)
    step("inswapper_128 (face swap)", inswapper)
    step("GFPGAN v1.4 (enhance)", gfpgan)
    step("Wav2Lip GAN (lip-sync fallback)", wav2lip)
    step("SadTalker (talking head)", sadtalker)
    step("LatentSync 1.5 (lip re-sync)", latentsync)
    print("\nAll downloads attempted. MODEL_ROOT =", MODEL_ROOT)
    if not HF_TOKEN:
        print("NOTE: HF_TOKEN not set — FLUX.1-dev (gated) will fail to load. "
              "Either set HF_TOKEN after accepting its license, or use "
              "FLUX.1-schnell (open) in the Text -> Image tab.")


if __name__ == "__main__":
    main()
