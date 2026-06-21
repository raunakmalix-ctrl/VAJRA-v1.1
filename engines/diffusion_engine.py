"""
Text -> image with FLUX.1 (best open-weight quality).

  - "dev"     : FLUX.1-dev, ~28 steps, guidance ~3.5  (highest quality)
  - "schnell" : FLUX.1-schnell, 4 steps, guidance 0    (fast previews)

Runs in the main Colab env. Lazy-loaded and unloaded by the model_manager so
it shares the GPU with the face-swap engine.
"""
import torch

from core.base_engine import BaseEngine
from core.model_manager import load_model, unload_model
from core.utils import timestamp_file
from core.device import DEVICE
from core.config import FLUX_DEV_REPO, FLUX_SCHNELL_REPO

_pipes = {}   # variant -> pipeline


def _load_flux(variant):
    from diffusers import FluxPipeline

    repo  = FLUX_DEV_REPO if variant == "dev" else FLUX_SCHNELL_REPO
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    print(f"[Diffusion] Loading FLUX.1-{variant} ({repo}) ...")

    pipe = FluxPipeline.from_pretrained(repo, torch_dtype=dtype)

    if DEVICE == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_gb >= 38:
            # A100/L4-40G+: keep it all on GPU for speed.
            pipe.to("cuda")
        else:
            # Smaller GPUs (16-24 GB): offload to fit.
            pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

    print(f"[Diffusion] FLUX.1-{variant} ready on {DEVICE.upper()}.")
    return pipe


class DiffusionEngine(BaseEngine):

    def load(self, variant="dev"):
        key = f"flux_{variant}"
        self.pipe = load_model(key, lambda: _load_flux(variant))
        return self.pipe

    def unload(self):
        for variant in ("dev", "schnell"):
            unload_model(f"flux_{variant}")

    def run(self, prompt, variant="dev",
            width=1024, height=1024,
            steps=None, guidance=None, seed=-1):

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if steps is None:
            steps = 28 if variant == "dev" else 4
        if guidance is None:
            guidance = 3.5 if variant == "dev" else 0.0

        pipe = self.load(variant)

        generator = None
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        # FLUX requires width/height divisible by 16.
        width  = (int(width)  // 16) * 16
        height = (int(height) // 16) * 16

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=generator,
            ).images[0]

        out_path = timestamp_file("flux", "png")
        image.save(out_path)
        print(f"[Diffusion] Saved: {out_path}")
        return out_path
