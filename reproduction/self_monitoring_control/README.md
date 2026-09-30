# Self-monitoring intervention control

This directory adds a prospective control to the Gogol-effect experiment.

It compares two generation arms on the same frozen prompts:

1. `scored`: exact existing V5.7 LIVE prompt with online sentence scores.
2. `plain`: same writing task with the self-scoring instruction removed.

Both artifacts are then scored by the same model using:

- clean `PREFIX` evaluation with no previous score tokens; and
- completed-output `POST` evaluation.

The primary question is whether `POST - PREFIX` remains positive in the
`plain` generation arm.

## Protocol version

The current runner is **v2**. The first disjoint smoke test exposed a formatting-only
Gemini parser problem in the inherited `<AI SCORE: n>` evaluator output syntax.
Before any full-cohort calls were made, PREFIX/POST output was changed to bare
numeric lines. See `FROZEN_PROTOCOL.md` for the amendment and rationale.

The original v1 smoke workspace, if present, is retained and never pooled with v2.

## Windows Git Bash

From the repository root:

```bash
git pull
python -m pip install -r reproduction/requirements.txt
```

First inspect prompts without spending API credits:

```bash
sh run_self_monitoring_control.sh dry
```

Then run the corrected disjoint smoke cohort:

```bash
sh run_self_monitoring_control.sh smoke
```

Inspect:

```text
reproduction/self_monitoring_control/run_smoke_v2/generation_qc.csv
reproduction/self_monitoring_control/run_smoke_v2/model_summary.csv
```

Smoke results are not used in the full analysis.

Then run the frozen 40-prompt experiment:

```bash
sh run_self_monitoring_control.sh full
```

The script runs collection and then analysis. It is resumable: rerunning the
same command skips stored complete/error cells rather than resampling until
compliance.

## Return results

After the full run, package the complete v2 workspace:

```bash
tar -czf self-monitoring-control-results-v2.tgz \
  reproduction/self_monitoring_control/run_full_v2
```

Send `self-monitoring-control-results-v2.tgz` back for interpretation and paper
integration.

## Main outputs

- `protocol.json`: exact frozen run configuration.
- `experiment.sqlite3`: raw/resumable store including API responses.
- `generation_qc.csv`: generation/segmentation/provider status.
- `question_metrics.csv`: prompt-level PREFIX and POST means.
- `paired_prompt_contrasts.csv`: paired scored-vs-plain prompt contrasts.
- `model_summary.csv`: model-level estimands, CIs, and tests.
- `analysis_summary.json`: machine-readable interpretation guide.

See `FROZEN_PROTOCOL.md` for the estimands and inferential plan.
