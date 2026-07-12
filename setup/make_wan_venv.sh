#!/usr/bin/env bash
# Build the isolated venv for Wan2.2-I2V (image + prompt -> motion video).
# Separate from make_venvs.sh/make_ltx_venv.sh because Wan2.2's diffusers
# integration needs transformers 4.49-4.51.3, which conflicts with venv_ltx's
# <4.50 pin (too narrow an overlap to share safely -- see core/config.py's
# WAN_I2V_REPO comment). Optional/heavy: run only if you want the Text->Video
# tab's reference-image (motion video) mode.
set -e

ROOT="${IMAGE_TALK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENVS="${IMAGE_TALK_VENVS:-$ROOT/venvs}"
mkdir -p "$VENVS"

# Wan2.2 wants CUDA > 12.7, matching venv_ltx's cu126 torch wheels.
CU="https://download.pytorch.org/whl/cu126"
PY312="$(command -v python3.12 || command -v python3)"
echo "==> building venv_wan with $PY312"

pip install -q virtualenv
if [ ! -x "$VENVS/venv_wan/bin/python" ]; then
  python -m virtualenv -p "$PY312" "$VENVS/venv_wan"
  "$VENVS/venv_wan/bin/pip" install -q --upgrade pip wheel
fi

# Install diffusers/transformers FIRST, torch trio LAST -- same ABI-mismatch
# fix used for venv_ltx/venv_latentsync ("undefined symbol: torch_library_impl"
# when an unpinned dependency pulls a mismatched torchaudio).
echo "==> diffusers (git) + deps"
"$VENVS/venv_wan/bin/pip" install -q -r "$ROOT/requirements/wan.txt"

echo "==> torch 2.7 (cu126) — installed last to pin a matched trio"
"$VENVS/venv_wan/bin/pip" install -q \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "$CU"

echo "==> venv_wan ready."
