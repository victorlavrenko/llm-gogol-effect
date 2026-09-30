# Protocol revision v3: evaluator response contract

This revision was made after the disjoint four-prompt smoke cohort and before any
full-cohort collection.

The v2 smoke run showed that Gemini 3.1 Pro sometimes ignored the instruction to
return only a bare integer and instead returned explanatory prose. This caused
format-driven missingness unrelated to the scientific quantity being measured.

V3 changes only the evaluator response syntax:

- PREFIX must begin with `SCORE=<integer>` on the first nonempty line.
- POST must begin with `SCORES=n1,n2,...,nN` on the first nonempty line.
- Only the required first line is parsed; any later explanation is ignored.
- If the first line does not satisfy the contract, the cell remains missing and
  is not resampled until compliance.

The following remain unchanged from the frozen scientific protocol:

- models;
- frozen prompt bank and prompt cohorts;
- SCORED vs PLAIN generation arms;
- generation and evaluation temperatures;
- sentence segmentation;
- AI-likeness criterion;
- PREFIX and POST information sets;
- prompt-level estimands and inferential plan.

V1 and v2 smoke outputs are implementation diagnostics only and must never be
pooled with the v3 full cohort.
