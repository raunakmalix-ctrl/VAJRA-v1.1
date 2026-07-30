# Tests

Offline, CPU-only checks for the parts of the pipeline that don't need a GPU or
model weights. They run in seconds and are the fastest way to catch a
regression in the realism layer.

```bash
python tests/test_dsp.py                 # 80 numerical assertions on dsp/
python tests/test_relip_integration.py   # 29 assertions on the relip flow
```

`test_relip_integration.py` stubs the voice and lip-sync engines, so it
exercises the real `TranscriptEngine.apply_edits` code path -- span resolution,
room-tone harvesting, the match chain, length preservation and untouched-audio
integrity -- without XTTS, ffmpeg or CUDA.

Requires only `numpy`, `scipy`, `soundfile` and (optionally) `pyloudnorm`; all
are in `requirements/main.txt`. Without `pyloudnorm` the loudness stage falls
back to RMS matching and the tests still pass.
