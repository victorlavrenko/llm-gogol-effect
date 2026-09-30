#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
EXP="$ROOT/reproduction/self_monitoring_control"
PY="${PYTHON:-python}"
RUNNER="$EXP/frozen_control_v3.py"

MODE="${1:-smoke}"

case "$MODE" in
  dry)
    "$PY" "$RUNNER" \
      --mode smoke \
      --out "$EXP/run_smoke_v3" \
      --dry-run
    ;;
  smoke)
    "$PY" "$RUNNER" \
      --mode smoke \
      --out "$EXP/run_smoke_v3"
    "$PY" "$EXP/analyze_self_monitoring_control.py" \
      --run "$EXP/run_smoke_v3"
    ;;
  full)
    "$PY" "$RUNNER" \
      --mode full \
      --out "$EXP/run_full_v3"
    "$PY" "$EXP/analyze_self_monitoring_control.py" \
      --run "$EXP/run_full_v3"
    ;;
  analyze)
    "$PY" "$EXP/analyze_self_monitoring_control.py" \
      --run "$EXP/run_full_v3"
    ;;
  *)
    echo "usage: sh run_self_monitoring_control.sh [dry|smoke|full|analyze]" >&2
    exit 2
    ;;
esac
