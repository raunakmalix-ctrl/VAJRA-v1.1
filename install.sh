#!/usr/bin/env bash
# VAJRA — one-command install on a Linux GPU host.
#
#   bash install.sh                 # core modules (Video Edit, Image Gen, Face Swap, Media Studio)
#   bash install.sh voice lipsync   # only these environments
#   bash install.sh all             # everything, including the very large video models
#
# This is the non-notebook entry point. The Colab notebook drives the same
# scripts step by step; on a plain host they are run in order here instead.
#
# What it does, and why in this order:
#   1. preflight   — fail on a missing prerequisite before downloading anything
#   2. venv_main   — the principal runtime, in its OWN virtual environment
#   3. install_main.sh — ffmpeg, third-party repos, main requirements
#   4. make_venvs.sh   — the isolated per-module environments
#   5. download_models.py — weights for the selected modules
#
# Step 2 is the significant difference from the notebook. Colab lets you pip
# install into the ambient interpreter; on a shared machine that either needs
# root or quietly breaks something else. Everything here goes into venv_main.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
export VAJRA_ROOT="$ROOT"
cd "$ROOT"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
  TARGETS=(voice lipsync)
  WEIGHTS=(voice lipsync faceswap)
  echo "==> no targets given; installing the core modules (voice lipsync)"
  echo "    pass targets to add more, e.g.  bash install.sh voice lipsync viitor"
elif [ "${TARGETS[0]}" = "all" ]; then
  TARGETS=(voice lipsync demucs viitor wan qwen ltx2)
  WEIGHTS=(all)
  echo "!! 'all' includes the video-generation models, which exceed 250 GB."
else
  WEIGHTS=()
  for t in "${TARGETS[@]}"; do
    case "$t" in
      voice)   WEIGHTS+=(voice) ;;
      lipsync) WEIGHTS+=(lipsync) ;;
      viitor)  WEIGHTS+=(viitor) ;;
    esac
  done
  WEIGHTS+=(faceswap)
fi

echo
echo "=== 1/5  preflight ==============================================="
PY_BOOT="$(command -v python3.12 || command -v python3)"
"$PY_BOOT" setup/preflight.py

echo
echo "=== 2/5  principal runtime ======================================="
VENV_MAIN="$ROOT/venv_main"
if [ ! -x "$VENV_MAIN/bin/python" ]; then
  echo "==> creating $VENV_MAIN"
  "$PY_BOOT" -m venv "$VENV_MAIN" 2>/dev/null || {
    "$PY_BOOT" -m pip install --user -q virtualenv
    "$PY_BOOT" -m virtualenv "$VENV_MAIN"
  }
fi
"$VENV_MAIN/bin/pip" install -q --upgrade pip wheel setuptools
echo "    using $("$VENV_MAIN/bin/python" -V)"

echo
echo "=== 3/5  main environment ========================================"
# install_main.sh installs with $PIP, which defaults to the ambient pip for the
# notebook's benefit. Point it at the venv instead.
PIP="$VENV_MAIN/bin/pip" bash setup/install_main.sh

echo
echo "=== 4/5  isolated module environments ============================"
echo "    targets: ${TARGETS[*]}"
bash setup/make_venvs.sh "${TARGETS[@]}"

echo
echo "=== 5/5  model weights ==========================================="
if [ -n "${VAJRA_MODELS:-}" ]; then
  echo "    into VAJRA_MODELS=$VAJRA_MODELS"
fi
echo "    groups: ${WEIGHTS[*]}"
"$VENV_MAIN/bin/python" setup/download_models.py "${WEIGHTS[@]}"

echo
echo "=================================================================="
echo "Installed. Start it with:   bash run.sh"
echo
echo "Weights live under \${VAJRA_MODELS:-$ROOT/models} — set VAJRA_MODELS to a"
echo "shared volume if several people use this host, so they are fetched once."
echo "=================================================================="
