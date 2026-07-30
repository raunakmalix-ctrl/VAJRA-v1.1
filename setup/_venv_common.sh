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

# ── shared, system-level prerequisites ─────────────────────────────────────
# The per-module builders run CONCURRENTLY, but these two steps write to places
# there is only one of: the system site-packages (virtualenv) and dpkg (apt).
# Two pip processes installing into the same site-packages at once will delete
# and rewrite each other's files mid-read, which surfaces as a baffling
# "No such file or directory: .../python_discovery/_cache.py" from a package
# nobody asked for. So both are guarded by one lock and skipped when already
# satisfied -- the fast path costs a version check, not a download.
_LOCKFILE="$VENVS/.bootstrap.lock"

_with_bootstrap_lock() {
  # flock is part of util-linux and present on Colab; degrade gracefully if not,
  # since a missing lock must not make the build impossible.
  if command -v flock >/dev/null 2>&1; then
    ( flock 9; "$@" ) 9>"$_LOCKFILE"
  else
    "$@"
  fi
}

_install_virtualenv() {
  python -m virtualenv --version >/dev/null 2>&1 && return 0
  echo "==> installing virtualenv (shared)"
  pip install -q virtualenv
}

ensure_virtualenv() {
  # Check outside the lock first so the common case does not serialise.
  python -m virtualenv --version >/dev/null 2>&1 && return 0
  _with_bootstrap_lock _install_virtualenv
}

_install_py310() {
  command -v python3.10 >/dev/null 2>&1 && return 0
  echo "==> installing Python 3.10 (legacy stacks have no 3.12 wheels)"
  apt-get -qq update >/dev/null
  apt-get -qq install -y python3.10 python3.10-venv python3.10-dev \
    python3.10-distutils >/dev/null
}

# Python 3.10 is only needed by the legacy stacks (voice, lip-sync) whose
# dependencies publish no 3.12 wheels. Installing it costs time, so only the
# builders that need it call this.
ensure_py310() {
  if ! command -v python3.10 >/dev/null 2>&1; then
    _with_bootstrap_lock _install_py310
  fi
  PY310="$(command -v python3.10)"
  [ -n "$PY310" ] || { echo "!! python3.10 could not be installed"; return 1; }
  ensure_virtualenv
  echo "    using $PY310"
}

ensure_py312() {
  PY312="$(command -v python3.12 || command -v python3)"
  ensure_virtualenv
  echo "    using $PY312"
}

# require_disk_gb <gb> <what>
# Several of these environments are multi-gigabyte. Running out of disk halfway
# through leaves a venv that LOOKS built -- make_venv skips anything with a
# bin/python -- so the next run silently uses a broken environment. Checking
# first turns that into one clear message before anything is downloaded.
require_disk_gb() {
  local need="$1" what="${2:-this environment}"
  local avail
  avail="$(df -Pk "$VENVS" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}')"
  [ -n "$avail" ] || return 0          # cannot measure: do not block the build
  if [ "$avail" -lt "$need" ]; then
    echo "!! Not enough disk for $what: ${avail}GB free, ~${need}GB needed." >&2
    echo "   Free space, or set USE_DRIVE=True to move model weights off this" >&2
    echo "   disk, or build fewer environments in one session." >&2
    return 1
  fi
  echo "    disk: ${avail}GB free (~${need}GB needed)"
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
  for v in venv_voice venv_latentsync venv_demucs venv_viitor \
           venv_wan venv_qwen venv_ltx2; do
    if [ -x "$VENVS/$v/bin/python" ]; then
      echo "    [built]   $v"
    else
      echo "    [missing] $v"
    fi
  done
}
