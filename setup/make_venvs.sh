#!/usr/bin/env bash
# Build the three isolated venvs for the subprocess engines.
# These hold mutually-incompatible torch/transformers stacks, kept out of the
# main FLUX env. Slow on first run; cache VENV_ROOT on Drive to reuse.
set -e

ROOT="${IMAGE_TALK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENVS="${IMAGE_TALK_VENVS:-$ROOT/venvs}"
TP="$ROOT/third_party"
mkdir -p "$VENVS"

# CUDA 12.1 wheel index (matches Colab's CUDA runtime).
CU="https://download.pytorch.org/whl/cu121"

make_venv() {  # make_venv <name>
  local name="$1"
  if [ -x "$VENVS/$name/bin/python" ]; then echo "  $name exists"; return; fi
  python -m venv "$VENVS/$name"
  "$VENVS/$name/bin/pip" install -q --upgrade pip wheel
}

echo "==> venv_voice (XTTS-v2)"
make_venv venv_voice
"$VENVS/venv_voice/bin/pip" install -q \
  torch==2.1.2 torchaudio==2.1.2 --index-url "$CU"
"$VENVS/venv_voice/bin/pip" install -q -r "$ROOT/requirements/voice.txt" \
  --extra-index-url "$CU"

echo "==> venv_sadtalker (SadTalker)"
make_venv venv_sadtalker
"$VENVS/venv_sadtalker/bin/pip" install -q \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url "$CU"
"$VENVS/venv_sadtalker/bin/pip" install -q -r "$ROOT/requirements/sadtalker.txt" \
  --extra-index-url "$CU" || true
[ -f "$TP/SadTalker/requirements.txt" ] && \
  "$VENVS/venv_sadtalker/bin/pip" install -q -r "$TP/SadTalker/requirements.txt" || true

echo "==> venv_latentsync (LatentSync + Wav2Lip)"
make_venv venv_latentsync
"$VENVS/venv_latentsync/bin/pip" install -q \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url "$CU"
"$VENVS/venv_latentsync/bin/pip" install -q -r "$ROOT/requirements/latentsync.txt" \
  --extra-index-url "$CU" || true
[ -f "$TP/LatentSync/requirements.txt" ] && \
  "$VENVS/venv_latentsync/bin/pip" install -q -r "$TP/LatentSync/requirements.txt" || true

echo "==> venvs ready."
