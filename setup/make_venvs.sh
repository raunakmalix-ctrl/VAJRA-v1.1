#!/usr/bin/env bash
# Build the three isolated venvs for the subprocess engines.
#
# These hold legacy torch/transformers stacks that DON'T have Python 3.12 wheels,
# while Colab now runs Python 3.12. So we install Python 3.10 via apt and build
# the venvs from it with `virtualenv` (which bundles pip and avoids Colab's
# broken `python -m venv` ensurepip step). The main FLUX env stays on 3.12.
set -e

ROOT="${IMAGE_TALK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENVS="${IMAGE_TALK_VENVS:-$ROOT/venvs}"
TP="$ROOT/third_party"
mkdir -p "$VENVS"

# CUDA 12.1 wheel index (matches Colab's CUDA runtime; cp310 wheels exist here).
CU="https://download.pytorch.org/whl/cu121"
PY310="$(command -v python3.10 || true)"

echo "==> ensuring Python 3.10 + virtualenv"
if [ -z "$PY310" ]; then
  apt-get -qq update >/dev/null
  apt-get -qq install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils >/dev/null
  PY310="$(command -v python3.10)"
fi
pip install -q virtualenv
echo "    using $PY310"

make_venv() {  # make_venv <name>
  local name="$1"
  if [ -x "$VENVS/$name/bin/python" ]; then echo "  $name exists"; return; fi
  python -m virtualenv -p "$PY310" "$VENVS/$name"
  "$VENVS/$name/bin/pip" install -q --upgrade pip wheel
}

# torch trio is installed LAST in each venv so it overrides whatever a package's
# own requirements pulled in — keeping torch/torchvision/torchaudio consistent.
echo "==> venv_voice (XTTS-v2)"
make_venv venv_voice
"$VENVS/venv_voice/bin/pip" install -q -r "$ROOT/requirements/voice.txt"
# coqui-tts >=0.25 uses a forked coqpit ("coqpit-config"); remove the original
# coqpit if an earlier build left it behind, or the import conflicts.
"$VENVS/venv_voice/bin/pip" uninstall -y -q coqpit 2>/dev/null || true
"$VENVS/venv_voice/bin/pip" install -q coqpit-config
# transformers (pulled by coqui-tts >=0.25) needs torch >= 2.4.
"$VENVS/venv_voice/bin/pip" install -q \
  torch==2.5.1 torchaudio==2.5.1 --index-url "$CU"

echo "==> venv_sadtalker (SadTalker)"
make_venv venv_sadtalker
"$VENVS/venv_sadtalker/bin/pip" install -q -r "$ROOT/requirements/sadtalker.txt" || true
[ -f "$TP/SadTalker/requirements.txt" ] && \
  "$VENVS/venv_sadtalker/bin/pip" install -q -r "$TP/SadTalker/requirements.txt" || true
"$VENVS/venv_sadtalker/bin/pip" install -q \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url "$CU"

echo "==> venv_latentsync (LatentSync + Wav2Lip)"
make_venv venv_latentsync
"$VENVS/venv_latentsync/bin/pip" install -q -r "$ROOT/requirements/latentsync.txt" || true
[ -f "$TP/LatentSync/requirements.txt" ] && \
  "$VENVS/venv_latentsync/bin/pip" install -q -r "$TP/LatentSync/requirements.txt" || true
# Match LatentSync 1.5's torch (its requirements pull 2.5.1); align the trio.
"$VENVS/venv_latentsync/bin/pip" install -q \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "$CU"

echo "==> venvs ready."
