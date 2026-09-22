# Temperature control V1 — frozen before execution

## Motivation
The venue feedback noted that the original protocol used temperature 0.7 for generated score tokens, so sampling variance could contribute to observed Live/Post/Replay instability.

## Design
- 10 held-out English prompts from the original frozen prompt bank that were not part of the 40-prompt main confirmatory sample.
- Exactly one prompt from each of the 10 original domains.
- Prompt IDs: p085, p043, p014, p093, p051, p006, p026, p076, p031, p069
- Models: GPT-5.5 and Gemini 3.1 Pro Preview.
- Exact V5.7 Live/Post/Replay prompts and parsing code.
- Full Replay for every prompt.
- No correction experiment.
- No native-language validation.
- Two matched arms on the same prompt IDs:
  - temperature = 0.0
  - temperature = 0.7
- Prompt-bank SHA-256: `3efdd073e3faf97d821de1f35ad52cc1f1acb04890a22bf5ea98e919754426aa`
- Runner SHA-256: `55b02beb89980768711dc06d5ee7e0ff619ce6d51136a2bba92fd2d30c93c6fb`

## Purpose
This is a narrow control for sampling-temperature confounding. It is not a new primary experiment and is not intended to establish generalization to reasoning or code.

## Interpretation fixed in advance
The important question is whether the qualitative state-dependent pattern remains at T=0.0:
1. Post−Live remains positive, especially for Gemini.
2. The Live–Replay–Post decomposition remains nontrivial at T=0.0.
3. We report effect sizes and prompt-level bootstrap intervals; with N=10 we do not require significance in every cell.
4. We also report paired T=0.0 minus T=0.7 differences on the same prompt IDs, but the control is not powered to demonstrate equivalence between temperatures.
