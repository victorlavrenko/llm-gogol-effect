# Preregistered robustness protocol — AI-slop self-assessment V1

Purpose: test whether the main Live/Post/Replay findings depend on the exact wording or representation of the same AI-likeness score.

Frozen design:
- 16 held-out writing prompts.
- GPT-5.5 and Gemini 3.1 Pro Preview.
- Four variants: original, semantic paraphrase, reverse-coded human-likeness, and minimal [n] score format.
- LIVE and POST on all 128 model×prompt×variant cells.
- Full sequential REPLAY on a fixed 8 prompts across both models and all variants.
- All analysis normalizes to 0..100 AI-likeness risk.
- No post-hoc variant selection.

Primary robustness claims:
1. GPT-5.5 shows greater within-output Live→Post rank stability than Gemini in the same direction across variants.
2. Gemini shows a larger normalized Post−Live shift than GPT-5.5 in the same direction across variants.
3. On the fixed Replay subset, Gemini's Post−Replay component remains larger than Replay−Live across variants.

Interpretation emphasizes cross-variant direction and matched-question effect sizes rather than requiring every small-cell test to be individually significant.
