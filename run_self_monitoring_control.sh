#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
EXP="$ROOT/reproduction/self_monitoring_control"
PY="${PYTHON:-python}"
RUNNER="$EXP/frozen_control.py"

MODE="${1:-smoke}"

case "$MODE" in
  dry)
    "$PY" "$RUNNER" \
      --mode smoke \
      --out "$EXP/run_smoke" \
      --dry-run
    ;;
  smoke)
    "$PY" "$RUNNER" \
      --mode smoke \
      --out "$EXP/run_smoke"
    "$PY" "$EXP/analyze_self_monitoring_control.py" \
      --run "$EXP/run_smoke"
    ;;
  full)
    "$PY" "$RUNNER" \
      --mode full \
      --out "$EXP/run_full"
    "$PY" "$EXP/analyze_self_monitoring_control.py" \
      --run "$EXP/run_full"
    ;;
  analyze)
    "$PY" "$EXP/analyze_self_monitoring_control.py" \
      --run "$EXP/run_full"
    ;;
  *)
    echo "usage: sh run_self_monitoring_control.sh [dry|smoke|full|analyze]" >&2
    exit 2
    ;;
esac
