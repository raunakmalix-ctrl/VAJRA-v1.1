"""
Runs inside venv_demucs. Splits a track into a SPEECH stem and a BACKGROUND
stem using Demucs v4 (htdemucs, MIT).

Why this matters more than it sounds. Editing the mixed track means the
regenerated span must reproduce not just the voice but the room, the music and
the traffic underneath it -- and any mismatch in that bed is audible across the
whole edit. Editing the speech stem alone and then re-laying the UNTOUCHED
background over the finished result sidesteps the problem entirely: the ambience
plays continuously across the join because it was never regenerated. Background
continuity alone defeats most casual detection.

Demucs separates into four stems (drums, bass, other, vocals). 'vocals' is the
speech stem; the remaining three are summed into the background. That sum is
lossless with respect to the model's own decomposition: speech + background
reconstructs the input to within the model's residual, so nothing is silently
discarded.

Invoked by core.subprocess_runner.run_worker.
"""
import os
import sys

os.environ["MPLBACKEND"] = "Agg"
os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.subprocess_runner import read_args, emit_result   # noqa: E402


def main():
    args = read_args()

    import numpy as np
    import torch
    import soundfile as sf
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    in_path = args["audio"]
    speech_out = args["speech_out"]
    bg_out = args["background_out"]
    model_name = args.get("model", "htdemucs")

    data, sr = sf.read(in_path, dtype="float32", always_2d=True)   # (n, ch)
    n_in, ch_in = data.shape

    print(f"[demucs_worker] Loading {model_name} ...", flush=True)
    model = get_model(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model_sr = int(getattr(model, "samplerate", 44100))
    model_ch = int(getattr(model, "audio_channels", 2))

    wav = torch.from_numpy(data.T).float()          # (ch, n)

    # Match the model's expected channel count. Demucs is trained on stereo;
    # feeding it mono by duplication is the documented approach.
    if wav.shape[0] < model_ch:
        wav = wav.repeat(model_ch, 1)[:model_ch]
    elif wav.shape[0] > model_ch:
        wav = wav[:model_ch]

    if sr != model_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, model_sr)

    # Demucs expects the mixture standardised by its own statistics, and the
    # separated stems rescaled back afterwards. Skipping this measurably
    # degrades separation.
    ref = wav.mean(0)
    ref_mean, ref_std = ref.mean(), ref.std().clamp_min(1e-8)
    wav = (wav - ref_mean) / ref_std

    print(f"[demucs_worker] Separating on {device.upper()} "
          f"({n_in / sr:.1f}s @ {sr} Hz) ...", flush=True)
    with torch.no_grad():
        # split=True processes in overlapping chunks so long tracks fit in VRAM.
        est = apply_model(model, wav[None].to(device), split=True,
                          overlap=0.25, progress=False, device=device)[0]
    est = est * ref_std + ref_mean
    est = est.cpu()

    sources = list(getattr(model, "sources", []))
    if "vocals" not in sources:
        raise RuntimeError(f"model '{model_name}' has no 'vocals' stem: {sources}")
    vi = sources.index("vocals")

    speech = est[vi]
    background = torch.zeros_like(speech)
    for i, name in enumerate(sources):
        if i != vi:
            background += est[i]

    if sr != model_sr:
        import torchaudio
        speech = torchaudio.functional.resample(speech, model_sr, sr)
        background = torchaudio.functional.resample(background, model_sr, sr)

    def _finish(t):
        a = t.numpy().T                        # (n, ch)
        # Return the channel layout we were given, so downstream stages don't
        # have to care that separation happened.
        if ch_in == 1 and a.shape[1] > 1:
            a = a.mean(axis=1, keepdims=True)
        elif ch_in > 1 and a.shape[1] == 1:
            a = np.repeat(a, ch_in, axis=1)
        a = a[:n_in] if a.shape[0] >= n_in else np.pad(
            a, ((0, n_in - a.shape[0]), (0, 0)))
        return np.squeeze(a, axis=1) if ch_in == 1 else a

    speech_np = _finish(speech)
    bg_np = _finish(background)

    for p in (speech_out, bg_out):
        d = os.path.dirname(os.path.abspath(p))
        if d:
            os.makedirs(d, exist_ok=True)
    # Float output: these are intermediates that later stages re-read, and
    # quantising them would apply an error to audio the edit never touches.
    sf.write(speech_out, speech_np, sr, subtype="FLOAT")
    sf.write(bg_out, bg_np, sr, subtype="FLOAT")

    print(f"[demucs_worker] speech -> {speech_out}", flush=True)
    print(f"[demucs_worker] background -> {bg_out}", flush=True)
    emit_result(speech_out)


if __name__ == "__main__":
    main()
