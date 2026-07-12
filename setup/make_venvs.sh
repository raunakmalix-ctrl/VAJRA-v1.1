#!/usr/bin/env bash
# Build the two isolated venvs for the subprocess engines (voice, lip-sync).
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

# Third-party repos vendor their own requirements.txt, which (like virtually
# every ML repo) pins its own torch/torchvision/torchaudio. We still install
# the rest of that file's pins, but installing THEIR torch just to overwrite
# it a few lines later wastes a multi-GB CUDA wheel download every session --
# strip those lines first so pip never installs a throwaway torch at all.
# (No-op, safely, if a file doesn't mention torch.)
strip_torch_pins() {  # strip_torch_pins <requirements-file>
  [ -f "$1" ] || return 0
  sed -i -E '/^(torch|torchvision|torchaudio)([=<>! ].*)?$/Id' "$1"
}

build_voice() {
  echo "==> venv_voice (XTTS-v2)"
  make_venv venv_voice
  # coqui-tts >=0.25 uses a forked coqpit ("coqpit-config"). Remove the ORIGINAL
  # coqpit FIRST so its leftover dir can't shadow coqpit-config (else you get
  # "cannot import name 'Coqpit' from 'coqpit' (unknown location)").
  "$VENVS/venv_voice/bin/pip" uninstall -y -q coqpit coqpit-config 2>/dev/null || true
  # Pin torch/torchaudio FIRST: coqui-tts only declares a loose lower bound on
  # torch, so installing our exact pin first means pip sees it already
  # satisfied and never downloads a second version for coqui-tts.
  "$VENVS/venv_voice/bin/pip" install -q \
    torch==2.5.1 torchaudio==2.5.1 --index-url "$CU"
  "$VENVS/venv_voice/bin/pip" install -q -r "$ROOT/requirements/voice.txt"
}

build_latentsync() {
  echo "==> venv_latentsync (LatentSync + Wav2Lip)"
  make_venv venv_latentsync
  strip_torch_pins "$TP/LatentSync/requirements.txt"
  "$VENVS/venv_latentsync/bin/pip" install -q -r "$ROOT/requirements/latentsync.txt" || true
  [ -f "$TP/LatentSync/requirements.txt" ] && \
    "$VENVS/venv_latentsync/bin/pip" install -q -r "$TP/LatentSync/requirements.txt" || true
  # Match LatentSync 1.5's torch (its requirements pull 2.5.1); align the trio.
  "$VENVS/venv_latentsync/bin/pip" install -q \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "$CU"
}

# The two venvs are fully independent (separate dirs, separate pip binaries)
# -- build them concurrently instead of one after another. Each logs to its
# own file so parallel output doesn't interleave into a garbled cell output;
# failures still fail the whole step and print their log tail for debugging.
echo "==> building venv_voice / venv_latentsync in parallel"
LOGS="$VENVS/.build-logs"; mkdir -p "$LOGS"
build_voice      > "$LOGS/voice.log"      2>&1 &  PID_VOICE=$!
build_latentsync > "$LOGS/latentsync.log" 2>&1 &  PID_LATENTSYNC=$!

fail=0
for spec in "voice:$PID_VOICE" "latentsync:$PID_LATENTSYNC"; do
  name="${spec%%:*}"; pid="${spec##*:}"
  if wait "$pid"; then
    echo "==> venv_$name OK"
  else
    echo "==> venv_$name FAILED -- last 40 lines of $LOGS/$name.log:"
    tail -n 40 "$LOGS/$name.log"
    fail=1
  fi
done
[ "$fail" -eq 0 ] || { echo "==> one or more venv builds failed (full logs in $LOGS/)"; exit 1; }

echo "==> venvs ready."
