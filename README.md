# The Gogol Effect in LLMs: Post-Completion Self-Devaluation

**Victor Lavrenko**  
PeaceTech VC, Israel  
victor@peacetech.vc

This repository contains the paper, frozen experimental artifacts, analysis code, and the adaptive Replay extension for **The Gogol Effect in LLMs: Post-Completion Self-Devaluation**.

## Headline result

We distinguish two operational contrasts:

- **IKEA-like component:** `Replay - Live > 0` -- the model is more favorable to its text while on the original writing trajectory than when the same visible prefix is replayed in a fresh context.
- **Gogol effect:** `Post - Replay > 0` -- the model becomes more critical of its earlier text after the completed output is visible.

Across the six models with sufficiently developed Live-Replay-Post data, only **1/6** has a positive mean IKEA-like shift, while **4/6** show positive evidence of post-completion self-devaluation. The positive Gogol shifts range from about **+3.2 to +20.1 points**. The remaining two models are near zero or negative.

| Model | N prompts | Replay-Live | Post-Replay | Interpretation |
|---|---:|---:|---:|---|
| GPT-5.5 | 40 | -3.10 | +6.91 | Gogol |
| Gemini 3.1 Pro | 40 | +2.77 | +20.06 | IKEA-like + Gogol |
| GPT-OSS-120B | 8 | -6.48 | +10.00 | Gogol |
| DeepSeek V4 Flash | 12 | -8.40 | +3.18 | Gogol |
| Llama 3.3 70B | 24 | -3.81 | +0.24 | no positive Gogol evidence |
| Claude Opus 5 | 24 | -3.01 | -2.05 | no positive Gogol evidence |

The two adaptive breadth models with positive effects also pass the prespecified fresh-extension sequential correction (GPT-OSS-120B adjusted `p=.021`; DeepSeek V4 Flash adjusted `p=.040`). The cumulative stopping p-values are retained as descriptive/exploratory because the original four-prompt pilot had already been inspected before the adaptive extension was designed.

## Repository layout

- `paper/` - current arXiv manuscript source, figures, and compiled PDF.
- `reproduction/` - frozen original experiment package, including SQLite workspaces, prompt banks, robustness studies, temperature controls, and deterministic analysis code.
- `gogol_extension/` - adaptive cross-model Replay protocol, runner, results, and six-model headline analysis.
- `results/` - convenient exported result tables used for checking the manuscript.
- `reproduce.sh` - one-command reproduction of stored-output analyses (no model API calls).
- `publish_github.sh` - rerunnable Git-Bash script that creates/updates this public GitHub repository.

## Reproduce analyses from frozen outputs

Python 3.10+ is recommended.

```bash
python -m pip install -r reproduction/requirements.txt
./reproduce.sh
```

`reproduce.sh` first runs the original deterministic analysis package and then reconstructs `results/six_model_effects.csv` from the frozen main-run database plus the adaptive Replay result log.

No API key is needed to reproduce the stored-output analyses.

## Re-run model calls (optional)

Exact regeneration of hosted model outputs cannot be guaranteed because model versions, provider routing, and serving infrastructure can change. If you intentionally want to issue new Replay calls, inspect:

- `gogol_extension/FROZEN_ADAPTIVE_PROTOCOL.md`
- `gogol_extension/adaptive_gogol_replay_step4.py`

The runner reads `OPENROUTER_API_KEY` from the environment; no credential is stored in this repository.

## Paper

Compile locally with:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

The compiled manuscript is also included as `paper/paper.pdf`.

## Reproducibility note

The adaptive extension began from an already-inspected four-prompt Replay pilot. The repository therefore preserves both the exploratory cumulative stopping statistics and the separately computed sequentially adjusted tests on fresh extension prompts. This distinction is also stated in the manuscript.

## Repository URL

https://github.com/victorlavrenko/llm-gogol-effect
