# Frozen independent 24-prompt extension

This extension was designed after observing the completed 16-prompt elicitation-robustness run and before observing any output from these 24 prompts.

## Purpose
Test whether the central Live→Post state-shift result independently reproduces on a fresh prompt set under the same four elicitation schemes.

## Frozen design
- 24 entirely new writing prompts.
- GPT-5.5 and Gemini 3.1 Pro Preview.
- Four elicitation variants unchanged from the initial robustness study:
  1. original AI-likeness wording / `<AI SCORE: n>`
  2. semantic paraphrase / `<AI SCORE: n>`
  3. reverse-coded human-likeness / `<HUMAN SCORE: n>`, normalized as `100 - raw`
  4. minimal AI-likeness score / `[n]`
- Live and Post only.
- No Replay in this extension.
- Temperature 0.7.
- Prompt-bank SHA-256: `00f76758a881710038e5a80188668fc8df9fc02737482d6c1fe843189b0aa22a`.

## Primary confirmatory checks
1. Mean normalized Post−Live is positive for both models under each of the four elicitation variants.
2. Averaged across the four variants within each matched prompt, Gemini has a larger Post−Live shift than GPT-5.5.

## Secondary/descriptive
Within-output Live→Post rank correlations and pairwise ordering are reported, but the extension is not enlarged or interpreted as successful/failed based on those quantities.

## Reporting
The original 16-prompt cohort, this fresh 24-prompt cohort, and the pooled 40-prompt dataset must all be reported separately. The 24-prompt extension is not described as preregistered before the original study; it is an independently frozen follow-up extension.
