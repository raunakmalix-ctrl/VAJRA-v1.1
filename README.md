# Image-Talk v1.1 — AI Media Studio

A single Gradio app (built for Google Colab Pro) bundling four AI media tools:

1. **Talking Video** — a portrait image + a reference voice clip + text → a realistic
   video of that person speaking the text (English/Hindi) in the **cloned** voice.
2. **Edit & Relip** — upload a talking-head video, get its transcript, change words,
   and the changed segments are re-voiced (cloned) and the lips re-synced.
3. **Text → Image** — FLUX.1 generation.
4. **Face Swap** — source face onto a target **image or video**.

## Model stack

| Feature | Models |
|---|---|
| Voice clone (en/hi) | XTTS-v2 |
| Talking head | SadTalker (512) + GFPGAN |
| Transcript | WhisperX (word timestamps) |
| Lip re-sync | LatentSync (primary) · Wav2Lip (fallback) |
| Text → image | FLUX.1-dev (quality) / FLUX.1-schnell (fast) |
| Face swap | InsightFace `inswapper_128` + GFPGAN |

## Run in Colab

Open **`Image_Talk_Colab.ipynb`**, set the runtime to a GPU (A100 recommended),
and run the cells top to bottom. The last cell prints a public `*.gradio.live`
link to the studio.

For **FLUX.1-dev** (gated): accept the license at
<https://huggingface.co/black-forest-labs/FLUX.1-dev> and set `HF_TOKEN` in the
notebook. Otherwise pick **FLUX.1-schnell** (open) in the Text → Image tab.

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

**Why isolated venvs?** XTTS, SadTalker and LatentSync pin mutually incompatible
`torch`/`transformers`/`numpy` versions that also clash with FLUX/diffusers. Each
runs in its own venv, invoked via `core/subprocess_runner.py`; the main env keeps
only Gradio + FLUX + InsightFace + WhisperX. A side benefit: subprocess engines
release their VRAM on exit, so heavy models don't pile up on the GPU.
