#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
EXP="$ROOT/reproduction/self_monitoring_control"
PY="${PYTHON:-python}"

MODE="${1:-smoke}"

case "$MODE" in
  dry)
    "$PY" "$EXP/self_monitoring_control.py" \
      --mode smoke \
      --out "$EXP/run_smoke" \
      --dry-run
    ;;
  smoke)
    "$PY" "$EXP/self_monitoring_control.py" \
      --mode smoke \
      --out "$EXP/run_smoke"
    "$PY" "$EXP/analyze_self_monitoring_control.py" \
      --run "$EXP/run_smoke"
    ;;
  full)
    "$PY" "$EXP/self_monitoring_control.py" \
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
