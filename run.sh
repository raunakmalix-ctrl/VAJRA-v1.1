#!/usr/bin/env bash
# VAJRA — start the studio.
#
#   bash run.sh                 # http://localhost:7860
#   VAJRA_SHARE=1 bash run.sh   # also print a public link (needs internet)
#   VAJRA_PORT=8080 bash run.sh # different port
#
# Sharing is OFF by default here, unlike the notebook: a public tunnel makes
# sense for a throwaway Colab session and is rarely what an institute wants on
# a machine that stays up.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
export VAJRA_ROOT="$ROOT"
cd "$ROOT"

VENV_MAIN="$ROOT/venv_main"
if [ ! -x "$VENV_MAIN/bin/python" ]; then
  echo "!! Not installed yet: $VENV_MAIN is missing." >&2
  echo "   Run:  bash install.sh" >&2
  exit 1
fi

export VAJRA_SHARE="${VAJRA_SHARE:-0}"
export VAJRA_PORT="${VAJRA_PORT:-7860}"

echo "==> VAJRA on http://localhost:${VAJRA_PORT}"
[ "$VAJRA_SHARE" = "1" ] && echo "==> a public link will also be printed below"
exec "$VENV_MAIN/bin/python" app.py
