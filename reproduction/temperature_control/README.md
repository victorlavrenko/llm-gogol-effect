# Held-out temperature control

This directory contains the narrow temperature-control experiment added after automated venue feedback raised the possibility that stochastic numeric-score sampling at temperature 0.7 could contribute to the observed state instability.

Design:
- 10 held-out English prompts, exactly one from each original domain;
- GPT-5.5 and Gemini 3.1 Pro Preview;
- exact V5.7 Live--Replay--Post protocol;
- full Replay for every prompt;
- matched arms at temperature 0.0 and 0.7;
- no correction intervention and no cross-language validation in this control.

The prompt bank and protocol were frozen before these outputs were generated. `run_T0/` and `run_T07/` contain the complete frozen workspaces, including SQLite databases and manifests. `analysis_expected/` contains the deterministic prompt-bootstrap analysis used by the manuscript. The analysis uses 20,000 bootstrap replicates with SHA-256-derived fixed seeds so it is reproducible across Python processes.

This control is interpreted narrowly: it tests whether the qualitative Live--Replay--Post pattern requires non-zero decoding temperature. It is not an equivalence test, and changing temperature also changes the generated prose trajectory, so between-temperature differences are not treated as a clean causal estimate of temperature on score tokens alone.
