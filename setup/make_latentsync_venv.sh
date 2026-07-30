#!/usr/bin/env bash
# Build ONLY the lip-sync environment (LatentSync + Wav2Lip fallback).
# Required by: Edit & Relip.
set -e
source "$(cd "$(dirname "$0")" && pwd)/_venv_common.sh"

echo "==> venv_latentsync (LatentSync + Wav2Lip)"
ensure_py310

if make_venv venv_latentsync "$PY310"; then
  strip_torch_pins "$TP/LatentSync/requirements.txt"
  "$VENVS/venv_latentsync/bin/pip" install -q \
    -r "$ROOT/requirements/latentsync.txt" || true
  [ -f "$TP/LatentSync/requirements.txt" ] && \
    "$VENVS/venv_latentsync/bin/pip" install -q \
      -r "$TP/LatentSync/requirements.txt" || true
  # Match LatentSync 1.5's torch (its requirements pull 2.5.1); align the trio.
  "$VENVS/venv_latentsync/bin/pip" install -q \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "$CU121"
fi

echo "==> venv_latentsync ready."
