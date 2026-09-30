#!/usr/bin/env python3
"""Frozen execution wrapper for self_monitoring_control.py.

This wrapper makes stored generation/evaluation errors terminal across reruns,
preventing format-failure survivorship bias. Transport retries still occur
inside the original OpenRouter call according to the frozen retry count.
"""
from __future__ import annotations

import self_monitoring_control as impl

impl.EVAL_CRITERION = (
    "Assess how likely an average human reader would be to think the sentence "
    "being evaluated was AI-generated. Use an integer from 0 to 100, where 0 "
    "means very human-like and 100 means obviously AI-generated."
)

_original_generation = impl.ensure_generation
_original_evaluations = impl.ensure_evaluations


def frozen_generation(store, client, *, model_id, route, question, arm, max_tokens):
    # Any stored complete/error generation is terminal for this frozen run.
    if store.generation(model_id, question.question_id, arm) is not None:
        return
    return _original_generation(
        store, client, model_id=model_id, route=route, question=question,
        arm=arm, max_tokens=max_tokens,
    )


def frozen_evaluations(
    store, client, *, model_id, route, question, arm,
    prefix_max_tokens, post_max_tokens,
):
    # Once an evaluation parser/API error has been stored, keep the whole
    # prompt-arm missing rather than resampling until compliance.
    with store.connect() as con:
        bad = con.execute(
            """
            SELECT 1 FROM evaluations
            WHERE model_id=? AND question_id=? AND arm=? AND status='error'
            LIMIT 1
            """,
            (model_id, question.question_id, arm),
        ).fetchone()
    if bad is not None:
        return
    return _original_evaluations(
        store, client, model_id=model_id, route=route, question=question,
        arm=arm, prefix_max_tokens=prefix_max_tokens,
        post_max_tokens=post_max_tokens,
    )


impl.ensure_generation = frozen_generation
impl.ensure_evaluations = frozen_evaluations

if __name__ == "__main__":
    raise SystemExit(impl.main())
