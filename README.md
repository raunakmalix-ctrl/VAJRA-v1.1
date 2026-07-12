# VAJRA v1.1 — AI Media Studio

A single Gradio app (built for Google Colab Pro) bundling four AI media tools:

1. **Talking Video** — a portrait image + a reference voice clip + text → a realistic
   video of that person speaking the text (English/Hindi) in the **cloned** voice.
2. **Edit & Relip** — upload a talking-head video, get its transcript, change words,
   and the changed segments are re-voiced (cloned) and the lips re-synced.
3. **Text → Image** — SDXL (RealVisXL) generation.
4. **Face Swap** — source face onto a target **image or video**.

## Model stack

| Feature | Models |
|---|---|
| Voice clone (en/hi) | XTTS-v2 |
| Talking head | SadTalker (512) + GFPGAN |
| Transcript | WhisperX (word timestamps) |
| Lip re-sync | LatentSync (primary) · Wav2Lip (fallback) |
| Text → image | RealVisXL V5.0 (SDXL) |
| Face swap | InsightFace `inswapper_128` + GFPGAN |

## Run in Colab

Open **`VAJRA_v1.1_Colab.ipynb`**, set the runtime to a GPU (A100 recommended),
and run the cells top to bottom. The last cell prints a public `*.gradio.live`
link to the studio.

The defaults (SDXL Realistic, Qwen-Image-Edit) need no `HF_TOKEN`. Only set one
if Wan2.2-I2V or LTX 2.3 (Text → Video's optional photo-animation engines)
turn out to need it — check their HF model pages.

To persist weights/venvs across sessions, set `USE_DRIVE = True` in cell 2.

## Architecture

```
app.py / app_theme.py     # Gradio UI (4 tabs, themed, share link)
core/                     # config, device, model_manager, subprocess_runner
engines/                  # one module per feature
workers/                  # scripts run inside isolated venvs (XTTS)
setup/                    # install_main.sh · make_venvs.sh · download_models.py
requirements/             # one pinned file per venv
third_party/              # cloned at setup: SadTalker, Wav2Lip, LatentSync
```

**Why isolated venvs?** XTTS, LatentSync, LTX, Wan2.2 and Qwen-Image-Edit each pin
mutually incompatible `torch`/`transformers`/`diffusers` versions. Each runs in
its own venv, invoked via `core/subprocess_runner.py`; the main env keeps only
Gradio + SDXL/diffusers + InsightFace + faster-whisper. A side benefit: subprocess
engines release their VRAM on exit, so heavy models don't pile up on the GPU.
