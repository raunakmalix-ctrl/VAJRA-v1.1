"""
Text -> image.

  - "sdxl_real" : RealVisXL V5.0 (photoreal SDXL) — OPEN, no token. Default.
  - "sdxl"      : Stable Diffusion XL base 1.0   — OPEN, no token.
  - "schnell"   : FLUX.1-schnell, 4 steps        — gated, needs HF_TOKEN + license.
  - "dev"       : FLUX.1-dev, ~28 steps          — gated, needs HF_TOKEN + license.

Runs in the main Colab env. Lazy-loaded and unloaded by the model_manager so
it shares the GPU with the face-swap engine.
"""
import torch

from core.base_engine import BaseEngine
from core.model_manager import load_model, unload_model
from core.utils import timestamp_file
from core.device import DEVICE
from core.config import FLUX_DEV_REPO, FLUX_SCHNELL_REPO

SDXL_REPO      = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REAL_REPO = "SG161222/RealVisXL_V5.0"   # photoreal SDXL, diffusers format

# Quality negative prompt for SDXL variants (FLUX ignores negatives).
SDXL_NEGATIVE = (
    "worst quality, low quality, blurry, jpeg artifacts, deformed, disfigured, "
    "bad anatomy, extra limbs, extra fingers, mutated hands, watermark, text, "
    "signature, cartoon, 3d render, plastic skin, overexposed"
)


def _load_pipe(variant):
    from diffusers import FluxPipeline, StableDiffusionXLPipeline

    if variant in ("sdxl", "sdxl_real"):
        repo = SDXL_REAL_REPO if variant == "sdxl_real" else SDXL_REPO
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        print(f"[Diffusion] Loading SDXL ({repo}) ...")
        kw = {"torch_dtype": dtype, "use_safetensors": True}
        if DEVICE == "cuda" and variant == "sdxl":
            kw["variant"] = "fp16"   # base SDXL ships an fp16 variant; RealVis may not
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(repo, **kw)
        except Exception:
            kw.pop("variant", None)
            pipe = StableDiffusionXLPipeline.from_pretrained(repo, **kw)
    else:
        repo  = FLUX_DEV_REPO if variant == "dev" else FLUX_SCHNELL_REPO
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        print(f"[Diffusion] Loading FLUX.1-{variant} ({repo}) ...")
        pipe = FluxPipeline.from_pretrained(repo, torch_dtype=dtype)

    if DEVICE == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_gb >= 38:
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

    print(f"[Diffusion] {variant} ready on {DEVICE.upper()}.")
    return pipe


# Per-variant defaults: (steps, guidance)
_DEFAULTS = {"sdxl_real": (30, 6.0), "sdxl": (30, 6.0),
             "dev": (28, 3.5), "schnell": (4, 0.0)}


class DiffusionEngine(BaseEngine):

    def load(self, variant="sdxl_real"):
        self.pipe = load_model(f"img_{variant}", lambda: _load_pipe(variant))
        return self.pipe

    def unload(self):
        for variant in ("sdxl_real", "sdxl", "dev", "schnell"):
            unload_model(f"img_{variant}")

    def run(self, prompt, variant="sdxl_real",
            width=1024, height=1024,
            steps=None, guidance=None, seed=-1, negative_prompt=None):

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        d_steps, d_guid = _DEFAULTS.get(variant, _DEFAULTS["sdxl_real"])
        if steps is None:
            steps = d_steps
        if guidance is None:
            guidance = d_guid

        pipe = self.load(variant)

        generator = None
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        # Both FLUX and SDXL are happy with dimensions divisible by 16.
        width  = (int(width)  // 16) * 16
        height = (int(height) // 16) * 16

        kwargs = dict(
            prompt=prompt, width=width, height=height,
            num_inference_steps=int(steps), guidance_scale=float(guidance),
            generator=generator,
        )
        if variant in ("sdxl", "sdxl_real"):   # FLUX has no negative prompt
            kwargs["negative_prompt"] = negative_prompt or SDXL_NEGATIVE

        with torch.inference_mode():
            image = pipe(**kwargs).images[0]

        out_path = timestamp_file(variant, "png")
        image.save(out_path)
        print(f"[Diffusion] Saved: {out_path}")
        return out_path
