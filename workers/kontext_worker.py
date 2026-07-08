"""
Runs inside venv_ltx (diffusers from git, which has FluxKontextPipeline).
Instruction-based image editing with FLUX.1-Kontext-dev (gated — reads HF_TOKEN
from the environment). Invoked by core.subprocess_runner.run_worker.
"""
import os
import sys

os.environ["MPLBACKEND"] = "Agg"
# HF's Xet transfer backend has a background-writer bug on Colab ("Internal
# Writer Error: Background writer channel closed"); force the classic HTTP
# downloader instead. Must be set before huggingface_hub/diffusers import.
os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.subprocess_runner import read_args, emit_result   # noqa: E402


def main():
    args = read_args()

    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image

    print("[kontext_worker] Loading FLUX.1-Kontext-dev ...", flush=True)
    pipe = FluxKontextPipeline.from_pretrained(args["repo"], torch_dtype=torch.bfloat16)
    # FLUX.1-Kontext-dev's own weights (12B transformer + T5-XXL encoder, bf16)
    # are ~33GB by themselves -- within a hair of a 40GB A100's full capacity
    # even with nothing else on the GPU (observed OOM: "Tried to allocate
    # 108.00 MiB" with the device already full). A full pipe.to("cuda") has no
    # real safety margin at this model size; always offload instead.
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()

    generator = None
    if int(args.get("seed", -1)) >= 0:
        generator = torch.Generator(device="cuda").manual_seed(int(args["seed"]))

    print("[kontext_worker] Editing ...", flush=True)
    image = pipe(
        image=load_image(args["image"]),
        prompt=args["prompt"],
        num_inference_steps=int(args["steps"]),
        guidance_scale=float(args["guidance"]),
        generator=generator,
    ).images[0]

    os.makedirs(os.path.dirname(args["out_path"]), exist_ok=True)
    image.save(args["out_path"])
    emit_result(args["out_path"])


if __name__ == "__main__":
    main()
