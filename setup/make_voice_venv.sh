#!/usr/bin/env bash
# Build ONLY the voice-cloning environment (XTTS-v2).
# Required by: Edit & Relip.
set -e
source "$(cd "$(dirname "$0")" && pwd)/_venv_common.sh"

echo "==> venv_voice (XTTS-v2 voice cloning)"
ensure_py310

if make_venv venv_voice "$PY310"; then
  # coqui-tts >=0.25 uses a forked coqpit ("coqpit-config"). Remove the ORIGINAL
  # coqpit FIRST so its leftover directory can't shadow coqpit-config (else:
  # "cannot import name 'Coqpit' from 'coqpit' (unknown location)").
  "$VENVS/venv_voice/bin/pip" uninstall -y -q coqpit coqpit-config 2>/dev/null || true
  # Pin torch/torchaudio FIRST: coqui-tts declares only a loose lower bound, so
  # installing our exact pin first means pip sees it satisfied and never
  # downloads a second copy.
  "$VENVS/venv_voice/bin/pip" install -q \
    torch==2.5.1 torchaudio==2.5.1 --index-url "$CU121"
  "$VENVS/venv_voice/bin/pip" install -q -r "$ROOT/requirements/voice.txt"
fi

echo "==> venv_voice ready."
