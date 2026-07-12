# VAJRA v1.1 — AI Media Studio

A single Gradio app (built for Google Colab Pro, runs 100% local/offline once
models are cached) bundling six AI media tools:

1. **Edit & Relip** — upload a talking-head video, extract its transcript, edit
   the words, and only the changed segments are re-voiced (in the speaker's own
   cloned voice) and the lips re-synced to match.
2. **Text → Image** — photorealistic image generation from a prompt.
3. **Face Swap** — source face onto a target **image or video**.
4. **Text → Video** — prompt-only video generation with synchronized audio, or
   supply a reference photo for identity-preserving **motion video** (pick an
   engine) with an optional audio track paired onto the output.
5. **Image Edit** — edit 1-3 images by instruction (e.g. "put the product from
   image 2 into image 1's scene").
6. **Media Studio** — trim, convert, resize, denoise, caption, remove
   backgrounds, merge clips — all CPU/ffmpeg, no GPU cost, with "→ Send to"
   wiring into the AI tabs.

## Model stack

| Feature | Models |
|---|---|
| Transcript | faster-whisper (word-level timestamps) |
| Voice clone (en/hi) | XTTS-v2 |
| Lip re-sync | LatentSync (primary) · Wav2Lip (fallback) |
| Text → image | RealVisXL V5.0 (default) / SDXL base |
| Face swap | InsightFace `inswapper_128` + GFPGAN (default) / CodeFormer |
| Text → video (prompt only, with audio) | LTX-2.3 |
| Motion video (photo + prompt) | Wan2.2-I2V-A14B (default) / LTX-2.3 |
| Image editing | Qwen-Image-Edit-2509 (1-3 reference images) |
| Media utilities | ffmpeg · rembg (background removal) |

Every model here is open — none of the defaults need a Hugging Face token.
Wan2.2-I2V and LTX-2.3 (Text → Video's optional motion-video engines) are the
only ones whose gating status wasn't confirmed at integration time.

## Run in Colab

Open **`VAJRA_v1.1_Colab.ipynb`**, set the runtime to a GPU (A100 recommended),
and run the cells top to bottom. The last cell prints a public `*.gradio.live`
link to the studio.

Cells **7b / 7c / 7d** build the optional, heavy venvs (Wan2.2-I2V,
Qwen-Image-Edit, LTX-2.3) — skip whichever tabs/engines you don't need.
LTX-2.3 (cell 7d) powers all of Text → Video (prompt-only and, optionally,
the reference-photo motion-video path) and needs its own venv: its pipeline
needs a newer `transformers` (for its Gemma 3 text encoder) than any other
venv here pins.

To persist **model weights** across sessions, set `USE_DRIVE = True` in cell 2
(covers everything fetched at setup *and* anything downloaded on first use of
a tab). Isolated venvs are always rebuilt locally each session — Google Drive
can't execute a venv's python.

## Architecture

```
app.py / app_theme.py     # Gradio UI (6 tabs, themed, share link)
core/                     # config, device, model_manager, subprocess_runner
engines/                  # one module per feature
workers/                  # scripts run inside isolated venvs (voice, LTX-2.3, Wan2.2-I2V, Qwen-Image-Edit)
setup/                    # install_main.sh · make_venvs.sh · make_ltx2_venv.sh · make_wan_venv.sh · make_qwen_venv.sh · download_models.py
requirements/             # one pinned file per venv
third_party/              # cloned at setup: Wav2Lip, LatentSync, CodeFormer
```

**Why isolated venvs?** XTTS, LatentSync, LTX-2.3, Wan2.2 and Qwen-Image-Edit
each pin mutually incompatible `torch`/`transformers`/`diffusers` versions,
so each runs in its own venv, invoked via `core/subprocess_runner.py`;
the main env
keeps only Gradio + SDXL/diffusers + InsightFace + faster-whisper. A side
benefit: subprocess engines release their VRAM on exit, so heavy models don't
pile up on the GPU. In-process models (SDXL, face swap) share a single-
occupant GPU cache (`core/model_manager.py`), and a process-wide lock
(`GPU_LOCK` in `app.py`) serializes every GPU-heavy tab so two heavy models
are never mid-load on the GPU at the same time.
