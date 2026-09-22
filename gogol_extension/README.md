# Gogol-effect extension

This directory contains the cross-model Replay extension used to broaden the Live-Replay-Post decomposition beyond GPT-5.5 and Gemini 3.1 Pro.

Key files:

- `FROZEN_ADAPTIVE_PROTOCOL.md` - protocol frozen before the extension.
- `adaptive_gogol_replay_step4.py` - model-wise adaptive runner using total N = 4, 8, 12, 16, 20, 24.
- `gogol_step4_protocol.json` - machine-readable frozen prompt order and stopping configuration.
- `gogol_step4_results.json` - every interim look and final stopped/capped state.
- `analyze_six_model_effects.py` - reconstructs the exact six-model headline table used in the paper.

Rebuild the headline table from repository root:

```bash
python gogol_extension/analyze_six_model_effects.py \
  --db reproduction/main_v57/run/experiment.sqlite3 \
  --adaptive-results gogol_extension/gogol_step4_results.json \
  --out results/six_model_effects.csv
```
