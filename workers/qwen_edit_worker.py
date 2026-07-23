"""
Runs inside venv_qwen (diffusers, QwenImageEditPlusPipeline). Instruction-
based image editing with Qwen-Image-Edit-2509 -- replaces FLUX.1-Kontext-dev.
Apache-2.0, open (no HF_TOKEN needed), and supports 1-3 reference images per
edit (e.g. "person + product"), unlike Kontext's single-image-only input.

Invoked by core.subprocess_runner.run_worker.
"""
import os
import sys

os.environ["MPLBACKEND"] = "Agg"
# HF's Xet transfer backend has a background-writer bug on Colab ("Internal
# Writer Error: Background writer channel closed"); force the classic HTTP
# downloader instead. Must be set before huggingface_hub/diffusers import.
os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.subprocess_runner import read_args, emit_result, warn_low_disk  # noqa: E402


def main():
    args = read_args()
    warn_low_disk(45, "Qwen-Image-Edit-2509 (20B transformer + 7B text encoder)")

    import torch
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.utils import load_image

    print("[qwen_edit_worker] Loading Qwen-Image-Edit-2509 ...", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(args["repo"], torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        # No official VRAM figure is published for this model (a 20B
        # transformer plus a 7B Qwen2.5-VL text encoder) -- offload
        # unconditionally, the same conservative default used for every
        # other large model in this project without a confirmed "fits fully
        # unoffloaded" number.
        pipe.enable_model_cpu_offload()

    images = [load_image(p) for p in args["images"]]

    generator = None
    if int(args.get("seed", -1)) >= 0:
        generator = torch.Generator(device="cuda").manual_seed(int(args["seed"]))

    print(f"[qwen_edit_worker] Editing ({len(images)} reference image(s)) ...", flush=True)
    result = pipe(
        image=images if len(images) > 1 else images[0],
        prompt=args["prompt"],
        # true_cfg_scale only takes effect with a negative_prompt supplied
        # (even an empty-ish one) -- see the diffusers QwenImageEdit docs.
        negative_prompt=args.get("negative_prompt") or " ",
        true_cfg_scale=float(args.get("guidance", 4.0)),
        num_inference_steps=int(args.get("steps", 40)),
        generator=generator,
    ).images[0]

    os.makedirs(os.path.dirname(args["out_path"]), exist_ok=True)
    result.save(args["out_path"])
    emit_result(args["out_path"])


if __name__ == "__main__":
    main()
