#!/usr/bin/env bash
# Build ONLY the speech/background separation environment (Demucs v4).
# Optional. Required by: Edit & Relip when separation is enabled.
set -e
source "$(cd "$(dirname "$0")" && pwd)/_venv_common.sh"

echo "==> venv_demucs (Demucs v4 source separation)"
ensure_py312

if make_venv venv_demucs "$PY312"; then
  # Install demucs FIRST, torch trio LAST -- same ABI-mismatch fix used for the
  # other Python 3.12 environments ("undefined symbol: torch_library_impl" when
  # an unpinned dependency pulls a mismatched torchaudio).
  echo "==> demucs + deps"
  "$VENVS/venv_demucs/bin/pip" install -q -r "$ROOT/requirements/demucs.txt"

  echo "==> torch 2.7 (cu126) — installed last to pin a matched trio"
  "$VENVS/venv_demucs/bin/pip" install -q \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "$CU126"
fi

echo "==> venv_demucs ready."
