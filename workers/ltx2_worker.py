"""
Runs inside venv_ltx2 (diffusers from git, includes LTX2Pipeline and
LTX2ImageToVideoPipeline). Text prompt alone -> video, or reference photo +
text prompt -> motion video, WITH synchronized audio either way (LTX-2.3,
diffusers/LTX-2.3-Diffusers) -- picks the pipeline class based on whether
args["image"] is given. Single-stage generation (no Stage-2 upsample/refine
pass) -- see the "Prompt Enhancement" / "Multimodal Guidance" examples at
https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2 for the
two-stage production-quality pipeline if higher fidelity is wanted later.
The guidance kwargs below (guidance_scale=3.0, audio_guidance_scale=7.0,
spatio_temporal_guidance_blocks=[28], use_cross_timestep=True) are that
page's documented LTX-2.3-recommended values, not arbitrary defaults.

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
    warn_low_disk(40, "LTX-2.3 (DiT audio-video model)")

    import torch
    from diffusers import LTX2Pipeline, LTX2ImageToVideoPipeline
    from diffusers.utils import load_image, encode_video

    has_image = bool(args.get("image"))
    pipeline_cls = LTX2ImageToVideoPipeline if has_image else LTX2Pipeline

    print(f"[ltx2_worker] Loading LTX-2.3 ({pipeline_cls.__name__}) ...", flush=True)
    pipe = pipeline_cls.from_pretrained(args["repo"], torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        # The diffusers docs' own LTX-2.3 examples use sequential (submodule-
        # level) offload rather than the coarser enable_model_cpu_offload()
        # used by the other engines here -- following that guidance since
        # LTX-2.3's transformer is large (docs' examples reference a ~19B
        # distilled variant).
        pipe.enable_sequential_cpu_offload(device="cuda")
        pipe.vae.enable_tiling()

    generator = None
    if int(args.get("seed", -1)) >= 0:
        generator = torch.Generator(device="cuda").manual_seed(int(args["seed"]))

    print("[ltx2_worker] Generating video (with audio) ...", flush=True)
    frame_rate = float(args.get("fps", 24))
    call_kwargs = dict(
        prompt=args["prompt"],
        negative_prompt=args.get("negative_prompt") or None,
        width=int(args["width"]),
        height=int(args["height"]),
        num_frames=int(args["num_frames"]),
        frame_rate=frame_rate,
        num_inference_steps=int(args.get("steps") or 40),
        guidance_scale=3.0,
        stg_scale=1.0,
        modality_scale=3.0,
        guidance_rescale=0.7,
        audio_guidance_scale=7.0,
        audio_stg_scale=1.0,
        audio_modality_scale=3.0,
        audio_guidance_rescale=0.7,
        spatio_temporal_guidance_blocks=[28],
        use_cross_timestep=True,
        generator=generator,
        output_type="np",
        return_dict=False,
    )
    if has_image:
        call_kwargs["image"] = load_image(args["image"])

    video, audio = pipe(**call_kwargs)

    os.makedirs(os.path.dirname(args["out_path"]), exist_ok=True)
    encode_video(
        video[0],
        fps=frame_rate,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=args["out_path"],
    )
    emit_result(args["out_path"])


if __name__ == "__main__":
    main()
