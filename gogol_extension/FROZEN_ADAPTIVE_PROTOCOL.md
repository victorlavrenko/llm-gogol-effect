# Frozen adaptive extension protocol: step size 4

## Objective

Extend the Live–Replay–Post decomposition beyond GPT-5.5 and Gemini 3.1 Pro while minimizing additional Replay API calls.

The existing four-prompt Replay pilot is **retained** in cumulative effect estimates. Additional Replay is added independently per unresolved model in increments of four prompts.

## Models extended

- Claude Opus 5
- DeepSeek V4 Flash
- GPT-OSS-120B
- Llama 3.3 70B Instruct

GPT-5.5 and Gemini 3.1 Pro already have full English Replay decompositions and receive no additional calls.

## Operational effects

For model `m` and prompt `q`, after averaging sentence scores within each condition:

- **Gogol effect:** `G_mq = Post_mq - Replay_mq`
- **IKEA-like component:** `I_mq = Replay_mq - Live_mq`

Positive `G` means post-completion self-devaluation.

## Adaptive schedule

Existing pilot: **N=4**.

Future cumulative looks:

- N=8
- N=12
- N=16
- N=20
- N=24

At every look, each model is evaluated independently. Once a model meets the selected stopping criterion, it is frozen and gets no further Replay calls. Other models continue.

## Two statistical views

### A. Cumulative descriptive/adaptive p-value

At each look, compute the one-sided prompt-level t-test using **all available prompts**, including the original four-prompt pilot.

This is the intuitive p-value for the growing sample and is useful for effect estimation and monitoring. Because the decision to extend the experiment was made after inspecting the pilot, it should be described as **adaptive/exploratory**, not as a fresh fixed-sample confirmatory p-value.

### B. Sequentially adjusted extension p-value

For a cleaner confirmatory sensitivity analysis, test only the newly added prompts at new-sample sizes 4, 8, 12, 16, and 20. The raw one-sided p-value is multiplied by 5 (Bonferroni over the five prespecified future looks).

The runner's default `--stop-rule confirmatory` freezes a model when this adjusted p-value is at most 0.05. `--stop-rule cumulative` implements the more aggressive cost-saving rule requested by the investigator and freezes once the cumulative raw p-value is at most 0.05.

Both statistics are saved and should be reported transparently.

## Prompt selection

Only English prompts with complete frozen Live and Post data for the model are eligible. The four pilot IDs (`p001`, `p042`, `p044`, `p070`) are removed from the new-prompt pool. A deterministic model-specific order is frozen with seed `20260922` before new Replay calls.

Prompts are never selected or reordered based on observed Replay outcomes.

## Unit of inference

The prompt is the inferential unit. Sentence-level scores are averaged within `(model, prompt, condition)` before contrasts are computed. Sentences are not treated as independent observations.

## Reporting

For every model report:

- starting pilot N;
- total N at each look;
- mean/median `Post-Replay`;
- 95% CI;
- cumulative raw one-sided p-value;
- new-only sequentially adjusted p-value;
- secondary Wilcoxon p-value;
- mean `Replay-Live` IKEA-like contrast;
- stopping N and stopping rule.

A model that does not meet a criterion by N=24 should be reported as not established by this extension, not as evidence of absence.
