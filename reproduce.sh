#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" reproduction/reproduce_all.py
"$PYTHON_BIN" gogol_extension/analyze_six_model_effects.py \
  --db reproduction/main_v57/run/experiment.sqlite3 \
  --adaptive-results gogol_extension/gogol_step4_results.json \
  --out results/six_model_effects.csv

echo
echo "Reproduction complete. Headline table: results/six_model_effects.csv"
