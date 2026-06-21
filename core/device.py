"""
Device abstraction. On Colab this is always CUDA, but the helper keeps
the code portable (CPU fallback for local smoke tests).
Import `DEVICE` and `empty_cache` everywhere instead of hardcoding torch.cuda.
"""
import torch


def _detect():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = _detect()


def empty_cache():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def mem_status():
    try:
        if DEVICE == "cuda":
            free, total = torch.cuda.mem_get_info()
            used = total - free
            name = torch.cuda.get_device_name(0)
            return (f"{name} | "
                    f"{used/1024**3:.2f} GB used / {total/1024**3:.2f} GB total")
        return "CPU (no dedicated VRAM tracking)"
    except Exception:
        return f"Device: {DEVICE.upper()} | VRAM info unavailable"
