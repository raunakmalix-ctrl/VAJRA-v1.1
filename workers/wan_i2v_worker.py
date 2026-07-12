"""
Runs inside venv_wan (diffusers, WanImageToVideoPipeline). Identity-preserving
motion video from a reference photo + a text prompt describing the action
(Wan2.2-I2V-A14B) -- the photo is the first frame, diffusion generates the
rest following the prompt. Not audio/lip-sync driven, and not per-face like
classic talking-head methods, so multi-subject photos (e.g. two people) are
handled naturally as part of the whole-image conditioning.

Invoked by core.subprocess_runner.run_worker.

NOTE: WanImageToVideoPipeline is a very recent diffusers addition (as of this
engine's introduction) -- the exact call signature below follows the
documented pattern as of writing but hasn't been exercised against a real
Colab GPU yet. If pipe(...) kwargs have shifted in the installed diffusers
version, check `diffusers.WanImageToVideoPipeline.__call__`'s signature first.
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
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import load_image, export_to_video

    print("[wan_i2v_worker] Loading Wan2.2-I2V-A14B ...", flush=True)
    pipe = WanImageToVideoPipeline.from_pretrained(args["repo"], torch_dtype=torch.bfloat16)
    # Wan2.2's MoE architecture (~27B total, two ~14B experts) needs ~80GB
    # resident in bf16 -- infeasible on a 40GB A100. Always offload
    # unconditionally, the same pattern used by every other large model in
    # this project without a confirmed "fits fully unoffloaded" number --
    # there is no "big enough GPU, skip offload" fast path here.
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

    generator = None
    if int(args.get("seed", -1)) >= 0:
        generator = torch.Generator(device="cuda").manual_seed(int(args["seed"]))

    print("[wan_i2v_worker] Generating motion video ...", flush=True)
    frames = pipe(
        image=load_image(args["image"]),
        prompt=args["prompt"],
        negative_prompt=args.get("negative_prompt") or None,
        height=int(args["height"]),
        width=int(args["width"]),
        num_frames=int(args["num_frames"]),
        guidance_scale=float(args.get("guidance", 5.0)),
        generator=generator,
    ).frames[0]

    os.makedirs(os.path.dirname(args["out_path"]), exist_ok=True)
    export_to_video(frames, args["out_path"], fps=int(args.get("fps", 16)))
    emit_result(args["out_path"])


if __name__ == "__main__":
    main()
