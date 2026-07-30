#!/usr/bin/env bash
# Build the isolated environment for ViiTorVoice-NAR (tier A local infill).
# Required by: Edit & Relip (tier A), Voice Editing & Cloning.
#
# Unlike every other engine here, ViiTorVoice ships as five gRPC services behind
# an HTTP gateway rather than as a library, so this step clones the repo and
# builds the environment its launcher expects. The launcher resolves its
# interpreter from VIITORVOICE_V2_VENV_DIR, and every value in its deploy.env is
# written as ${VAR:-default}, so we point it at our venv through the environment
# instead of patching upstream files -- which would conflict on every `git pull`.
#
# Heavy: torch 2.8.0+cu128 plus ONNX Runtime GPU and TensorRT. Its pins collide
# with every other environment here (venv_voice is on torch 2.5.1/cu121), which
# is exactly why it is isolated.
set -e
source "$(cd "$(dirname "$0")" && pwd)/_venv_common.sh"

echo "==> venv_viitor (ViiTorVoice-NAR local speech infill)"
# torch 2.8+cu128 alone brings ~8 nvidia CUDA libraries, and ONNX Runtime GPU
# and TensorRT sit on top of that. The weights are a further ~12GB, but those go
# to MODEL_ROOT and can live on Drive; this figure is the environment only.
require_disk_gb 14 "venv_viitor (torch cu128 + ONNX Runtime + TensorRT)"
ensure_py312

REPO_DIR="$TP/viitor-voice-nar"
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "    cloning ViiTorVoice-NAR"
  git clone --depth 1 https://github.com/viitor-ai/viitor-voice-nar.git "$REPO_DIR"
else
  echo "    repo already present"
fi

if make_venv venv_viitor "$PY312"; then
  PIP="$VENVS/venv_viitor/bin/pip"
  # Their requirements carry their own --extra-index-url for the cu128 wheels,
  # so install them as written rather than re-deriving the pins here.
  if [ -f "$REPO_DIR/requirements-grpc.txt" ]; then
    "$PIP" install -q -r "$REPO_DIR/requirements-grpc.txt"
  else
    echo "!! requirements-grpc.txt missing — upstream layout changed" >&2
    exit 1
  fi
  [ -f "$REPO_DIR/requirements-alone.txt" ] && \
    "$PIP" install -q -r "$REPO_DIR/requirements-alone.txt"
  # Pinned by upstream's init_env.sh after the requirements, because the
  # requirements resolve an older protobuf that the gRPC stubs reject.
  "$PIP" install -q protobuf==4.25.3
fi

echo "==> venv_viitor ready."
echo "    weights are fetched by Step 7 (target: viitor), not here, so they can"
echo "    live under MODEL_ROOT and persist across sessions."
