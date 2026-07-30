# Tests

Offline, CPU-only checks for the parts of the pipeline that don't need a GPU or
model weights. They run in seconds and are the fastest way to catch a
regression in the realism layer.

```bash
python tests/test_dsp.py                 # 80 numerical assertions on dsp/
python tests/test_textdiff.py            # 50 assertions on word-level diffing
python tests/test_vision.py              # 93 assertions on masks and windowing
python tests/test_relip_integration.py   # 49 assertions on the relip flow
python tests/test_router_scorecard.py    # 74 assertions on routing + metrics
python tests/test_app_ui.py              # 15 assertions on the UI report panels
python tests/test_notebook.py            # 23 assertions on the Colab notebook
python tests/test_setup_scripts.py       # 45 assertions on the env builders
python tests/test_fit_regression.py      # 21 assertions on never truncating
python tests/test_viitor.py              # 46 assertions on tier A routing + HTTP
```

What each one is actually for:

- **`test_dsp.py`** — the numerical core: loudness matching by a single method,
  spectral correction bounded to a sane range, WSOLA time-scaling, zero-crossing
  splices, room-tone estimation, and the reference-window scorer.
- **`test_textdiff.py`** — that a small wording change resolves to a small span.
  Includes the collapse-to-segment rule for near-total rewrites and
  pause-boundary expansion.
- **`test_vision.py`** — feathered masks, window merging, coverage, and
  changed-region detection, including the cases where detection must decline
  (global change, negligible change).
- **`test_relip_integration.py`** — stubs the voice, lip-sync and separation
  engines, so it exercises the real `TranscriptEngine.apply_edits` code path —
  span resolution, room-tone harvesting, the match chain, length preservation
  and untouched-audio integrity — without XTTS, ffmpeg or CUDA.
- **`test_router_scorecard.py`** — that the router reports the tier it chose
  *and* the tier the language could reach, and that the scorecard ranks an edit
  against the natural seams of the same recording rather than an absolute
  threshold. Also pins the seam percentile's contract: it is a rank, so it must
  never invert the dB ordering, but two seams inside one gap of the distribution
  may legitimately tie.
- **`test_app_ui.py`** — builds the whole Gradio graph against stubs and renders
  the Edit & Relip report panels from real `core.router` / `evaluation.scorecard`
  payloads. A renamed dict key would otherwise pass every other check here and
  fail only on a GPU.
- **`test_fit_regression.py`** — that a replacement is never cut off
  mid-phrase. It grows into neighbouring silence, or the whole line is re-voiced;
  truncation, which made an edit sound like the new wording followed by the old
  words, is never an outcome.
- **`test_viitor.py`** — that tier A is only promised when the environment, the
  repo and the weights are all present; that a tier the operator requests is
  honoured or explained; and that the multipart body matches the documented API
  field names. No service runs here.
- **`test_setup_scripts.py`** — that no environment builder touches system pip
  or apt outside the shared lock. The race it guards needs three parallel builds
  on a clean machine to reproduce, so it is checked statically instead.
- **`test_notebook.py`** — parses every code cell in the Colab notebook and
  checks the Step 6 flags against the build dispatcher's target list. A string
  literal split across two cell lines is invisible in a diff and fails minutes
  into a GPU session; that has happened twice.

Requires only `numpy`, `scipy`, `soundfile` and (optionally) `pyloudnorm`; all
are in `requirements/main.txt`. Without `pyloudnorm` the loudness stage falls
back to RMS matching and the tests still pass.

## What these tests do NOT cover

Every path that needs a GPU or real weights: Demucs' `apply_model`, the ffmpeg
cut/composite path in windowed lip-sync, real LatentSync/XTTS calls, and — the
largest gap — whether ViiTorVoice's five-service group actually starts and
returns good audio. Those are stubbed here and remain unvalidated until a GPU
session runs them.
