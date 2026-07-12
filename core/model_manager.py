"""
In-process model cache with single-occupant semantics: loading a new model
unloads the previous one so only one heavy model sits on the GPU at a time.
Used by the in-process engines (SDXL diffusion, face swap). The subprocess
engines (XTTS, LatentSync, LTX, Wan2.2, Qwen-Image-Edit) free their VRAM
automatically on exit.
"""
import gc
import torch

from core.device import empty_cache, mem_status

MODELS = {}
CURRENT_MODEL = None


def load_model(name, loader):
    global CURRENT_MODEL

    if CURRENT_MODEL and CURRENT_MODEL != name:
        unload_model(CURRENT_MODEL)

    if name not in MODELS:
        print(f"[ModelManager] Loading: {name}")
        MODELS[name] = loader()

    CURRENT_MODEL = name
    return MODELS[name]


def unload_model(name):
    global CURRENT_MODEL

    if name not in MODELS:
        return

    print(f"[ModelManager] Unloading: {name}")
    model = MODELS.pop(name)

    # Diffusers fp16/bf16 pipelines: delete GPU components directly rather than
    # moving to CPU (the move is slow and doesn't reliably free VRAM).
    if hasattr(model, "components"):
        for attr in ["transformer", "unet", "vae", "text_encoder",
                     "text_encoder_2", "image_encoder", "safety_checker"]:
            comp = getattr(model, attr, None)
            if comp is not None:
                try:
                    del comp
                except Exception:
                    pass
    elif hasattr(model, "to"):
        try:
            model.to("cpu")
        except Exception:
            pass

    del model
    gc.collect()
    empty_cache()

    if CURRENT_MODEL == name:
        CURRENT_MODEL = None

    print(f"[ModelManager] Unloaded {name}. {mem_status()}")


def current():
    return CURRENT_MODEL


def vram_status():
    return mem_status()
