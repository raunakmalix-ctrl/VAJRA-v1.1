#!/usr/bin/env bash
# Build the isolated venv for LTX-2 (text/image -> video).
# Separate from make_venvs.sh because LTX-2 needs Python 3.12 + torch ~2.7 +
# bleeding-edge diffusers, and is optional/heavy. Run only if you want video gen.
set -e

ROOT="${IMAGE_TALK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENVS="${IMAGE_TALK_VENVS:-$ROOT/venvs}"
mkdir -p "$VENVS"

# LTX-2 wants CUDA > 12.7; use the cu126 torch wheels.
CU="https://download.pytorch.org/whl/cu126"
PY312="$(command -v python3.12 || command -v python3)"
echo "==> building venv_ltx with $PY312"

pip install -q virtualenv
if [ ! -x "$VENVS/venv_ltx/bin/python" ]; then
  python -m virtualenv -p "$PY312" "$VENVS/venv_ltx"
  "$VENVS/venv_ltx/bin/pip" install -q --upgrade pip wheel
fi

echo "==> torch 2.7 (cu126)"
"$VENVS/venv_ltx/bin/pip" install -q \
  torch==2.7.1 torchvision torchaudio --index-url "$CU"

echo "==> diffusers (git) + deps"
"$VENVS/venv_ltx/bin/pip" install -q -r "$ROOT/requirements/ltx.txt"

echo "==> venv_ltx ready."
