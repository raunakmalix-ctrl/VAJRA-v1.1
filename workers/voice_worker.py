"""
Runs inside venv_voice (Coqui TTS). Loads XTTS-v2, clones the reference
speaker and synthesizes the text, then reports the output wav path.

Invoked by core.subprocess_runner.run_worker — do not import the main env here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.subprocess_runner import read_args, emit_result   # noqa: E402


def main():
    args = read_args()
    text     = args["text"]
    ref      = args["reference_audio"]
    language = args.get("language", "en")
    xtts_dir = args["xtts_dir"]
    out_path = args["out_path"]

    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[voice_worker] Loading XTTS-v2 ...", flush=True)
    config = XttsConfig()
    config.load_json(os.path.join(xtts_dir, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=xtts_dir, eval=True)
    if device != "cpu":
        model.to(device)

    print("[voice_worker] Computing speaker latents ...", flush=True)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[ref], gpt_cond_len=30, max_ref_length=60, sound_norm_refs=False
    )

    print(f"[voice_worker] Synthesizing ({language}) ...", flush=True)
    result = model.inference(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
    )

    import soundfile as sf
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, result["wav"], 24000)
    emit_result(out_path)


if __name__ == "__main__":
    main()
