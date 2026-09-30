#!/usr/bin/env python3
"""Frozen execution wrapper for the self-monitoring control, protocol v3.

V3 was frozen after the disjoint smoke cohort showed that Gemini sometimes
ignored a bare-integer-only instruction and emitted prose instead. No full-cohort
results had been collected.

Changes from v2 are output-contract only:
- PREFIX response must START with: SCORE=<integer>
- POST response must START with: SCORES=n1,n2,...,nN
- the parser reads only that first line and ignores any trailing prose
- if the first line does not contain the required score contract, the cell is
  marked missing; no resampling-until-compliance is performed

Generation arms, models, frozen prompt cohorts, temperatures, evaluation
criterion, segmentation, and estimands are unchanged.
"""
from __future__ import annotations

import re

import frozen_control as v2

v2.impl.VERSION = "self-monitoring-control-v3"

_PREFIX_FIRST_LINE_RE = re.compile(
    r"^\s*SCORE\s*=\s*(\d{1,3})\s*$",
    re.IGNORECASE,
)
_POST_FIRST_LINE_RE = re.compile(
    r"^\s*SCORES\s*=\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*$",
    re.IGNORECASE,
)


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    raise ValueError("empty evaluator response")


def _parse_score_contract(text: str, expected_n: int) -> tuple[int, ...]:
    first = _first_nonempty_line(text)
    if expected_n == 1:
        m = _PREFIX_FIRST_LINE_RE.fullmatch(first)
        if not m:
            raise ValueError(
                f"expected first line SCORE=<0..100>, got: {first!r}"
            )
        scores = (int(m.group(1)),)
    else:
        m = _POST_FIRST_LINE_RE.fullmatch(first)
        if not m:
            raise ValueError(
                f"expected first line SCORES=n1,...,n{expected_n}, got: {first!r}"
            )
        scores = tuple(int(x.strip()) for x in m.group(1).split(","))
        if len(scores) != expected_n:
            raise ValueError(
                f"POST first line contains {len(scores)} scores; expected {expected_n}"
            )
    if any(not 0 <= s <= 100 for s in scores):
        raise ValueError(f"score outside 0..100: {scores}")
    return scores


def _prefix_messages(task: str, units, target_index: int):
    text_so_far = v2.impl.canonical_text(units[: target_index + 1])
    return [
        {"role": "system", "content": v2.impl.base.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "ORIGINAL WRITING TASK\n"
                "---------------------\n"
                f"{task}\n\n"
                "EVALUATION CRITERION\n"
                "--------------------\n"
                f"{v2.impl.EVAL_CRITERION}\n\n"
                "TEXT SO FAR\n"
                "-----------\n"
                f"{text_so_far}\n\n"
                "Score only the LAST sentence in TEXT SO FAR. Later sentences "
                "do not exist in this evaluation context. Your response MUST "
                "begin with exactly SCORE=<integer>, for example SCORE=42. "
                "The integer must be 0 through 100. Put nothing before that "
                "first line. Any explanation after the first line will be ignored."
            ),
        },
    ]


def _post_messages(task: str, units):
    text = v2.impl.canonical_text(units)
    n = len(units)
    example = ",".join(str(10 * (i + 1)) for i in range(n))
    return [
        {"role": "system", "content": v2.impl.base.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "ORIGINAL WRITING TASK\n"
                "---------------------\n"
                f"{task}\n\n"
                "EVALUATION CRITERION\n"
                "--------------------\n"
                f"{v2.impl.EVAL_CRITERION}\n\n"
                f"The complete generated text below contains exactly {n} sentences "
                "under the experiment's frozen segmentation. Assess each sentence "
                "in the context of the COMPLETE text. Your response MUST begin "
                f"with exactly one line of the form SCORES=n1,n2,...,n{n}. "
                f"For format illustration only: SCORES={example}. Use exactly {n} "
                "integers from 0 through 100 in sentence order. Put nothing before "
                "that first line. Any explanation after the first line will be ignored.\n\n"
                "COMPLETE TEXT\n"
                "-------------\n"
                f"{text}"
            ),
        },
    ]


# v2.frozen_evaluations resolves these module globals at runtime.
v2._parse_numeric_lines = _parse_score_contract
v2._prefix_messages = _prefix_messages
v2._post_messages = _post_messages

if __name__ == "__main__":
    raise SystemExit(v2.impl.main())
