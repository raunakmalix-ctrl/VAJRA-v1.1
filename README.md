# VAJRA 2.0 — AI Media Studio

A single Gradio app (built for Google Colab Pro, runs 100% local/offline once
models are cached) bundling seven AI media tools:

1. **Video Edit** — upload a talking-head video, extract its transcript, edit
   the words, and only the changed *words* are re-voiced (in the speaker's own
   cloned voice) and only the affected video windows re-synced. Replaced audio is
   matched to its surroundings — duration, loudness, tone, room and noise floor —
   and the result is scored against the recording's own natural word boundaries,
   so the edit's detectability is measured rather than assumed. Everything
   outside an edit passes through bit-identical.
2. **Voice Generation** — regenerate only the words you changed in a
   recording, conditioned on the real audio either side, or speak new text in a
   cloned voice. English only.
3. **Image Generation** — photorealistic image generation from a prompt.
4. **Face Swap** — source face onto a target **image or video**.
5. **Text → Video** — prompt-only video generation with synchronized audio, or
   supply a reference photo for identity-preserving **motion video** (pick an
   engine) with an optional audio track paired onto the output.
6. **Image Edit** — edit 1-3 images by instruction (e.g. "put the product from
   image 2 into image 1's scene").
7. **Media Studio** — trim, convert, resize, denoise, caption, remove
   backgrounds, merge clips — all CPU/ffmpeg, no GPU cost, with "→ Send to"
   wiring into the AI tabs.

## Model stack

| Feature | Models |
|---|---|
| Transcript | faster-whisper (word-level timestamps) |
| Local speech infill (tier A, English) | ViiTorVoice-NAR |
| Voice clone (17 languages) | XTTS-v2 |
| Speech / background separation | Demucs v4 (htdemucs) |
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

Open **`VAJRA_2.0_Colab.ipynb`**, set the runtime to a GPU (A100 recommended),
and run the cells top to bottom. The last cell prints a public `*.gradio.live`
link to the studio.

**Steps 6 and 7 are selective.** Each module's dependency stack is mutually
incompatible with the others, so each lives in its own isolated environment —
and building all of them takes many minutes you don't need to spend. Set the
flags in Step 6 for the modules you actually intend to use; unselected ones
cost nothing, and re-running later adds a module without rebuilding what you
already have.

| Module | Step 6 environments | Step 7 weight groups |
|---|---|---|
| Video Edit | `VOICE` + `LIPSYNC` (+ `SEPARATE`, recommended) | `voice`, `lipsync` |
| Voice Generation | `VIITOR` | `viitor` |
| Image Generation | *none* | *none* |
| Face Swap | *none* | `faceswap` |
| Text → Video | `LTX2` and/or `WAN` | *none* |
| Image Edit | `QWEN` | *none* |
| Media Studio | *none* | *none* |

Image Generation, Face Swap and Media Studio run in the principal runtime, so they
need no isolated environment at all. LTX-2.3 needs its own because its pipeline
requires a newer `transformers` (for its Gemma 3 text encoder) than any other
environment here pins.

To persist **model weights** across sessions, set `USE_DRIVE = True` in Step 2
(covers everything fetched at setup *and* anything downloaded on first use of
a tab). Isolated venvs are always rebuilt locally each session — Google Drive
can't execute a venv's python.

## Architecture

```
app.py / app_theme.py     # Gradio UI (7 tabs, themed, share link)
core/                     # config, device, model_manager, subprocess_runner, router, textdiff
engines/                  # one module per feature
workers/                  # scripts run inside isolated venvs (voice, Demucs, LTX-2.3, Wan2.2-I2V, Qwen-Image-Edit)
dsp/                      # audio realism: loudness, spectral, room, noise floor, splice, time-fit
vision/                   # frame windowing, feathered masks, compositing
evaluation/               # objective detectability scorecard
setup/                    # install_main.sh · make_venvs.sh (selective dispatcher) · one builder per environment · download_models.py
requirements/             # one pinned file per venv
third_party/              # cloned at setup: Wav2Lip, LatentSync, CodeFormer
tests/                    # offline CPU checks — see tests/README.md
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
