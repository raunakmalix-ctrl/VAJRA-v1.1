"""
Runs inside venv_ltx (Python 3.12, torch ~2.7, diffusers git). Generates a
video from a text prompt (optionally conditioned on an image) with LTX-2,
exports it to mp4, and reports the path.

Invoked by core.subprocess_runner.run_worker.
"""
import os
import sys

os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.subprocess_runner import read_args, emit_result   # noqa: E402


def main():
    args = read_args()

    import torch
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video, load_image

    print("[ltx_worker] Loading LTX-2 ...", flush=True)
    pipe = LTX2Pipeline.from_pretrained(args["repo"], torch_dtype=torch.bfloat16)
    # Sequential CPU offload keeps it within ~40 GB VRAM (slower but fits).
    pipe.enable_sequential_cpu_offload()

    kwargs = dict(
        prompt=args["prompt"],
        negative_prompt=args.get("negative_prompt") or None,
        width=int(args["width"]),
        height=int(args["height"]),
        num_frames=int(args["num_frames"]),
        num_inference_steps=int(args["steps"]),
        guidance_scale=float(args["guidance"]),
    )
    if args.get("image"):
        kwargs["image"] = load_image(args["image"])

    print("[ltx_worker] Generating ...", flush=True)
    result = pipe(**kwargs)

    # LTX-2's return shape is still settling across diffusers versions; handle
    # the common variants (object with .frames, or a (video, audio) tuple).
    if hasattr(result, "frames"):
        frames = result.frames[0]
    elif isinstance(result, (tuple, list)):
        frames = result[0]
        if hasattr(frames, "frames"):
            frames = frames.frames[0]
    else:
        frames = result

    os.makedirs(os.path.dirname(args["out_path"]), exist_ok=True)
    export_to_video(frames, args["out_path"], fps=int(args["fps"]))
    emit_result(args["out_path"])


if __name__ == "__main__":
    main()
