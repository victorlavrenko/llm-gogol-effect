#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" reproduction/reproduce_all.py
"$PYTHON_BIN" gogol_extension/analyze_six_model_effects.py \
  --db reproduction/main_v57/run/experiment.sqlite3 \
  --adaptive-results gogol_extension/gogol_step4_results.json \
  --out results/six_model_effects.csv

printf '\nReproduction complete. Headline table: results/six_model_effects.csv\n'
