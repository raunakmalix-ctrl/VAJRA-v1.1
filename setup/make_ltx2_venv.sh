#!/usr/bin/env bash
# Build the isolated venv for LTX-2.3 (text -> video, or image+prompt ->
# motion video, w/ audio). Its pipeline needs transformers with
# Gemma3ForConditionalGeneration (>=~4.50), incompatible with every other
# venv's pins, hence its own venv. Optional/heavy: run only if you want the
# Text -> Video tab.
set -e

ROOT="${IMAGE_TALK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENVS="${IMAGE_TALK_VENVS:-$ROOT/venvs}"
mkdir -p "$VENVS"

# Matches venv_wan/venv_qwen's CUDA 12.6 torch wheels.
CU="https://download.pytorch.org/whl/cu126"
PY312="$(command -v python3.12 || command -v python3)"
echo "==> building venv_ltx2 with $PY312"

pip install -q virtualenv
if [ ! -x "$VENVS/venv_ltx2/bin/python" ]; then
  python -m virtualenv -p "$PY312" "$VENVS/venv_ltx2"
  "$VENVS/venv_ltx2/bin/pip" install -q --upgrade pip wheel
fi

# Install diffusers/transformers FIRST, torch trio LAST -- same ABI-mismatch
# fix used for venv_wan/venv_qwen ("undefined symbol: torch_library_impl"
# when an unpinned dependency pulls a mismatched torchaudio).
echo "==> diffusers (git) + transformers (git) + deps"
"$VENVS/venv_ltx2/bin/pip" install -q -r "$ROOT/requirements/ltx2.txt"

echo "==> torch 2.7 (cu126) — installed last to pin a matched trio"
"$VENVS/venv_ltx2/bin/pip" install -q \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "$CU"

echo "==> venv_ltx2 ready."
