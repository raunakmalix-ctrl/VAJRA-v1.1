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

Requires only `numpy`, `scipy`, `soundfile` and (optionally) `pyloudnorm`; all
are in `requirements/main.txt`. Without `pyloudnorm` the loudness stage falls
back to RMS matching and the tests still pass.

## What these tests do NOT cover

Every path that needs a GPU or real weights: Demucs' `apply_model`, the ffmpeg
cut/composite path in windowed lip-sync, and real LatentSync/XTTS calls. Those
are stubbed here and remain unvalidated until a GPU session runs them.
