#!/usr/bin/env bash
# Shared helpers for the per-module venv builders. SOURCE this, don't run it.
#
#   source "$(dirname "$0")/_venv_common.sh"
#
# Why per-module builders: building every isolated environment up front costs
# many minutes, and most sessions only exercise one or two modules. Each
# module's environment is independent, so each gets its own entry point and
# only what you actually intend to run has to be built.

ROOT="${IMAGE_TALK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENVS="${IMAGE_TALK_VENVS:-$ROOT/venvs}"
TP="$ROOT/third_party"
mkdir -p "$VENVS"

# CUDA 12.1 wheels — cp310 wheels exist here (needed by the legacy voice /
# lip-sync stacks). The Python 3.12 environments use cu126 instead.
CU121="https://download.pytorch.org/whl/cu121"
CU126="https://download.pytorch.org/whl/cu126"

PY310=""
PY312=""

# Python 3.10 is only needed by the legacy stacks (voice, lip-sync) whose
# dependencies publish no 3.12 wheels. Installing it costs time, so only the
# builders that need it call this.
ensure_py310() {
  PY310="$(command -v python3.10 || true)"
  if [ -z "$PY310" ]; then
    echo "==> installing Python 3.10 (legacy stacks have no 3.12 wheels)"
    apt-get -qq update >/dev/null
    apt-get -qq install -y python3.10 python3.10-venv python3.10-dev \
      python3.10-distutils >/dev/null
    PY310="$(command -v python3.10)"
  fi
  pip install -q virtualenv
  echo "    using $PY310"
}

ensure_py312() {
  PY312="$(command -v python3.12 || command -v python3)"
  pip install -q virtualenv
  echo "    using $PY312"
}

# make_venv <name> <interpreter>
make_venv() {
  local name="$1" interp="$2"
  if [ -x "$VENVS/$name/bin/python" ]; then
    echo "  $name already built — skipping (delete venvs/$name to rebuild)"
    return 1
  fi
  python -m virtualenv -p "$interp" "$VENVS/$name"
  "$VENVS/$name/bin/pip" install -q --upgrade pip wheel
  return 0
}

# Third-party repos vendor a requirements.txt that pins its own
# torch/torchvision/torchaudio. We install the rest of those pins, but
# installing THEIR torch just to overwrite it moments later wastes a multi-GB
# CUDA wheel download every session — strip those lines first.
strip_torch_pins() {
  [ -f "$1" ] || return 0
  sed -i -E '/^(torch|torchvision|torchaudio)([=<>! ].*)?$/Id' "$1"
}

# Report what exists, so the notebook/operator can see the current state.
venv_status() {
  echo "==> environment status ($VENVS)"
  for v in venv_voice venv_latentsync venv_wan venv_qwen venv_ltx2; do
    if [ -x "$VENVS/$v/bin/python" ]; then
      echo "    [built]   $v"
    else
      echo "    [missing] $v"
    fi
  done
}
