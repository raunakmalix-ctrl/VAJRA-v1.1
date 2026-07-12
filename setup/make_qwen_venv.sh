#!/usr/bin/env bash
# Build the isolated venv for Qwen-Image-Edit-2509 (instruction-based image
# editing, replaces FLUX.1-Kontext-dev). Separate from every other venv
# because its Qwen2.5-VL-7B text encoder needs transformers installed from
# git (the maintainers' own guidance -- a pinned release can hit
# "KeyError: 'qwen2_5_vl'"), which we don't want to force on any other venv.
# Optional/heavy: run only if you want the Image Edit tab.
set -e

ROOT="${IMAGE_TALK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENVS="${IMAGE_TALK_VENVS:-$ROOT/venvs}"
mkdir -p "$VENVS"

# Matches venv_wan/venv_ltx2's CUDA 12.6 torch wheels.
CU="https://download.pytorch.org/whl/cu126"
PY312="$(command -v python3.12 || command -v python3)"
echo "==> building venv_qwen with $PY312"

pip install -q virtualenv
if [ ! -x "$VENVS/venv_qwen/bin/python" ]; then
  python -m virtualenv -p "$PY312" "$VENVS/venv_qwen"
  "$VENVS/venv_qwen/bin/pip" install -q --upgrade pip wheel
fi

# Install diffusers/transformers FIRST, torch trio LAST -- same ABI-mismatch
# fix used for venv_wan/venv_ltx2 ("undefined symbol: torch_library_impl" when
# an unpinned dependency pulls a mismatched torchaudio).
echo "==> diffusers (git) + transformers (git) + deps"
"$VENVS/venv_qwen/bin/pip" install -q -r "$ROOT/requirements/qwen.txt"

echo "==> torch 2.7 (cu126) — installed last to pin a matched trio"
"$VENVS/venv_qwen/bin/pip" install -q \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "$CU"

echo "==> venv_qwen ready."
