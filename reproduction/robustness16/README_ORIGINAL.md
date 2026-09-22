# AI-slop elicitation robustness V1

## Git Bash
```bash
cd ~/Downloads
rm -rf ai_slop_robustness_v1
unzip ai_slop_robustness_v1.zip

cd ~/Downloads/ai_slop_robustness_v1

python ai_slop_robustness_v1.py --self-test

python ai_slop_robustness_v1.py   --out ~/Downloads/ai_slop_robustness_run   --dry-run
```

If the dry-run is correct and `OPENROUTER_API_KEY` is exported (or in `.env`):

```bash
python ai_slop_robustness_v1.py   --out ~/Downloads/ai_slop_robustness_run
```

The run is resumable. To regenerate analysis without API calls:

```bash
python ai_slop_robustness_v1.py   --out ~/Downloads/ai_slop_robustness_run   --analyze-only
```

Key outputs: `experiment.sqlite3`, `manifest.json`, `scores_long.csv`,
`question_metrics.csv`, `robustness_summary.csv`, `model_contrasts.csv`,
and `analysis_summary.json`.
