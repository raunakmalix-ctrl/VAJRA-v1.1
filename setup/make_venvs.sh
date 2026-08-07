#!/usr/bin/env bash
# Selective environment builder.
#
#   bash setup/make_venvs.sh                 # status only — builds nothing
#   bash setup/make_venvs.sh voice lipsync   # Video Edit
#   bash setup/make_venvs.sh qwen            # Image Edit
#   bash setup/make_venvs.sh all             # everything (slow)
#
# Independent targets are built CONCURRENTLY, each logging to its own file so
# parallel output doesn't interleave into a garbled cell. A failure fails the
# whole step and prints that target's log tail.
#
# Module -> required targets
#   Video Edit    voice lipsync  (+ demucs for separation,
#                                   + viitor for tier A infill)
#   Voice Edit      viitor
#   Image Generation   (none — runs in the principal runtime)
#   Face Swap       (none)
#   Media Studio    (none)
#   Text -> Video   ltx2  and/or  wan
#   Image Edit      qwen
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_venv_common.sh"

ALL_TARGETS="voice lipsync demucs viitor wan qwen ltx2"

declare -A SCRIPT=(
  [voice]="make_voice_venv.sh"
  [lipsync]="make_latentsync_venv.sh"
  [demucs]="make_demucs_venv.sh"
  [viitor]="make_viitor_venv.sh"
  [wan]="make_wan_venv.sh"
  [qwen]="make_qwen_venv.sh"
  [ltx2]="make_ltx2_venv.sh"
)

if [ "$#" -eq 0 ]; then
  venv_status
  echo
  echo "Nothing requested. Pass one or more targets: $ALL_TARGETS"
  echo "  e.g.  bash setup/make_venvs.sh voice lipsync     # Video Edit"
  echo "        bash setup/make_venvs.sh all               # everything (slow)"
  exit 0
fi

REQUESTED=()
for arg in "$@"; do
  case "$arg" in
    all) for t in $ALL_TARGETS; do REQUESTED+=("$t"); done ;;
    voice|lipsync|demucs|viitor|wan|qwen|ltx2) REQUESTED+=("$arg") ;;
    *) echo "!! unknown target '$arg' (valid: $ALL_TARGETS all)"; exit 2 ;;
  esac
done

# De-duplicate while preserving order.
UNIQ=()
for t in "${REQUESTED[@]}"; do
  skip=""
  for u in "${UNIQ[@]:-}"; do [ "$u" = "$t" ] && skip=1 && break; done
  [ -z "$skip" ] && UNIQ+=("$t")
done

echo "==> building: ${UNIQ[*]}"
LOGS="$VENVS/.build-logs"; mkdir -p "$LOGS"

PIDS=(); NAMES=()
for t in "${UNIQ[@]}"; do
  bash "$HERE/${SCRIPT[$t]}" > "$LOGS/$t.log" 2>&1 &
  PIDS+=("$!"); NAMES+=("$t")
done

fail=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "==> ${NAMES[$i]} OK"
  else
    echo "==> ${NAMES[$i]} FAILED — last 40 lines of $LOGS/${NAMES[$i]}.log:"
    tail -n 40 "$LOGS/${NAMES[$i]}.log"
    fail=1
  fi
done

echo
venv_status
[ "$fail" -eq 0 ] || { echo "!! one or more builds failed (full logs in $LOGS/)"; exit 1; }
echo "==> done."
