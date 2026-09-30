# Frozen protocol: self-monitoring intervention control (v2)

## Purpose

The main LIVE-REPLAY-POST study shows when a model's self-score changes, but the
LIVE generation prompt itself asks the model to evaluate every sentence while
writing. This control asks whether that online self-monitoring instruction
changes the generated artifact or is necessary for the post-completion
self-devaluation pattern.

This is a prospective follow-up control. The substantive design was specified
before any calls from the control were inspected.

## Protocol-v2 amendment after disjoint smoke test

The four-prompt smoke cohort (balanced-order positions 41-44) exposed a purely
syntactic failure in the initial evaluator interface: Gemini frequently returned
an interpretable numeric score using a slightly noncanonical `<AI SCORE: ...>`
wrapper, causing the inherited V5.7 tag parser to mark otherwise usable PREFIX
calls missing. No full-cohort calls had been made.

Before running the full cohort, evaluator output syntax was therefore simplified
without changing the scoring criterion or any scientific estimand:

- PREFIX now requests exactly one bare integer from 0 to 100.
- POST now sees clean completed prose with no score placeholders and requests
  exactly N bare integers, one per line.
- Parsing accepts only unambiguous numeric score lines in 0..100, with narrow
  tolerance for harmless wrappers such as a missing closing angle bracket or `%`.
- The original v1 smoke workspace is retained and excluded. V2 uses fresh
  `run_smoke_v2` and `run_full_v2` workspaces.

This amendment was made solely in response to output-format compliance observed
in the disjoint smoke cohort; no full-cohort outcome was observed.

## Models

Primary models:

- `openai/gpt-5.5`
- `google/gemini-3.1-pro-preview`

The exact route strings are inherited from the frozen V5.7 codebase.

## Prompts and cohorts

The experiment reuses the frozen `prompts-v3` bank and the same deterministic,
domain-balanced ordering (seed `20260907`) used by the main study.

- **Full cohort:** balanced-order positions 1-40.
- **Smoke cohort:** balanced-order positions 41-44.

Smoke results are for implementation checking only and must never be pooled
with the full cohort.

## Generation arms

For every model x prompt cell, generate two independent artifacts. Arm order is
deterministically randomized within each model x prompt cell.

### SCORED

Use the exact V5.7 LIVE generation prompt. The model must append
`<AI SCORE: n>` after every sentence while generating.

### PLAIN

Use the same system prompt and exact original writing task, but remove the
self-scoring instruction entirely. The only additional line is:

> Produce the requested text now.

No mention of AI-likeness, scoring, self-evaluation, or later evaluation is
present in the PLAIN generation prompt.

Both generation arms use temperature 0.7 by default.

## Sentence alignment

SCORED sentence boundaries are defined by the existing score-tag parser.

PLAIN is segmented deterministically without an LLM. Natural one-sentence-per-line
output is used when it yields 5-8 sentence-final lines; otherwise punctuation
boundaries with a small abbreviation guard are used. Cells not yielding 5-8
sentences are marked failed rather than silently repaired or regenerated for
content reasons.

Before evaluation, only inter-sentence whitespace is canonicalized. Lexical
content and punctuation are retained.

## Common evaluators

The two generated artifacts are evaluated using identical evaluators and the
same model route that generated them. Evaluation arm order is deterministically
randomized.

### PREFIX

For sentence i, a fresh call receives the original writing task and prose only
through sentence i. It sees no earlier scores. The last visible sentence is
rated 0-100 for how likely an average human reader would think it AI-generated.
The response format is exactly one bare integer.

This is intentionally cleaner than the original REPLAY condition because no
prior score tokens are inserted into the prefix.

### POST

A fresh call sees the clean completed generated prose, with no score placeholders,
and scores every sentence under the same criterion. If the frozen segmentation
has N sentences, the response format is exactly N bare integers, one per line
and in sentence order.

Evaluation temperature is 0.7 by default to match the main study.

## Prompt-level estimands

Sentence scores are averaged within prompt before inferential analysis, so
outputs with more sentences do not receive more weight.

For each generation arm:

`C_arm = mean_sentence(POST - PREFIX)`

Primary quantity:

`C_plain`

If `C_plain > 0`, post-completion self-devaluation is present even when the
generation trajectory never contained an online self-scoring instruction.

Main interaction:

`I = C_scored - C_plain`

Interpretation:

- `C_plain > 0`, `I ~ 0`: completion effect persists without online
  self-monitoring; the scoring instruction is not needed.
- `C_plain > 0`, `I > 0`: completion effect persists without scoring, but
  online self-monitoring may amplify it.
- `C_plain ~ 0`, `C_scored > 0`: the measurement intervention itself may be
  important in creating the observed shift.

Secondary artifact estimand:

`POST_scored - POST_plain`

This asks whether requiring online self-monitoring changes the kind of text the
model later judges as AI-like.

## Inference

For each model separately, the pre-specified H1 test is one-sided:

`H1: mean(C_plain) > 0`

The two model-wise H1 p-values are Bonferroni-adjusted for two tests. Report
prompt-bootstrap 95% intervals (20,000 deterministic resamples) for all main
means.

The interaction and artifact contrasts are reported with two-sided tests and
95% prompt-bootstrap intervals. They are secondary/exploratory.

No adaptive stopping is used in the full cohort.

## Missingness and retries

Transport/429/5xx failures may be retried using the inherited OpenRouter
transport retry policy. A successfully completed generation is never regenerated
because of its observed content or score. Parse/format failures are reported as
missing cells rather than imputed. Stored errors are terminal within a frozen
workspace, so rerunning does not resample until compliance.

## Separation from the main paper

This follow-up does not reinterpret PREFIX or POST as human ground truth. Both
are state-indexed same-model judgments. The control isolates whether the
online self-scoring *instruction during generation* is necessary for the
completion-associated score shift.
