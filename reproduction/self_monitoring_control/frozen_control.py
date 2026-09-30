#!/usr/bin/env python3
"""Frozen execution wrapper for the self-monitoring control, protocol v2.

V2 was frozen after the disjoint smoke cohort exposed a formatting-only parser
failure in Gemini PREFIX responses. No full-cohort results had been collected.

Changes from v1:
- PREFIX asks for exactly one bare integer (0..100), not an <AI SCORE: n> tag.
- POST shows clean completed prose with no score placeholders and asks for exactly
  N bare integers, one per line.
- Parsing accepts only unambiguous numeric score lines (0..100), with a narrow
  tolerance for harmless wrappers such as a missing closing '>' or '%'.
- Stored generation/evaluation errors remain terminal across reruns.

These changes affect output syntax only; the evaluation criterion, generation
arms, models, prompt cohorts, temperatures, and estimands are unchanged.
"""
from __future__ import annotations

import json
import re

import self_monitoring_control as impl

impl.VERSION = "self-monitoring-control-v2"
impl.EVAL_CRITERION = (
    "Assess how likely an average human reader would be to think the sentence "
    "being evaluated was AI-generated. Use an integer from 0 to 100, where 0 "
    "means very human-like and 100 means obviously AI-generated."
)

# Conservative one-line parser. The intended protocol is a bare integer. We also
# accept narrow, semantically unambiguous presentation variants so a response
# like '<AI SCORE: 25' is not converted into missing data merely because the
# closing angle bracket was omitted.
_SCORE_LINE_RE = re.compile(
    r"^\s*`{0,3}\s*"
    r"(?:<\s*AI\s+SCORE\s*:\s*)?"
    r"\(?\s*(\d{1,3})\s*\)?"
    r"\s*(?:(?:/\s*100)|(?:%))?"
    r"\s*>?\s*[.!]?\s*`{0,3}\s*$",
    re.IGNORECASE,
)


def _parse_numeric_lines(text: str, expected_n: int) -> tuple[int, ...]:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and line.strip().lower() not in {"```", "```text"}
    ]
    if len(lines) != expected_n:
        raise ValueError(
            f"expected exactly {expected_n} numeric score line(s), got {len(lines)}: {lines!r}"
        )
    scores: list[int] = []
    for line in lines:
        m = _SCORE_LINE_RE.fullmatch(line)
        if not m:
            raise ValueError(f"not an unambiguous numeric score line: {line!r}")
        score = int(m.group(1))
        if not 0 <= score <= 100:
            raise ValueError(f"score outside 0..100: {score}")
        scores.append(score)
    return tuple(scores)


def _prefix_messages(task: str, units, target_index: int):
    text_so_far = impl.canonical_text(units[: target_index + 1])
    return [
        {"role": "system", "content": impl.base.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "ORIGINAL WRITING TASK\n"
                "---------------------\n"
                f"{task}\n\n"
                "EVALUATION CRITERION\n"
                "--------------------\n"
                f"{impl.EVAL_CRITERION}\n\n"
                "TEXT SO FAR\n"
                "-----------\n"
                f"{text_so_far}\n\n"
                "Score only the LAST sentence in TEXT SO FAR. Later sentences "
                "do not exist in this evaluation context. Return exactly one "
                "bare integer from 0 to 100. Do not output words, punctuation, "
                "labels, angle-bracket tags, markdown, or explanation."
            ),
        },
    ]


def _post_messages(task: str, units):
    text = impl.canonical_text(units)
    n = len(units)
    return [
        {"role": "system", "content": impl.base.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "ORIGINAL WRITING TASK\n"
                "---------------------\n"
                f"{task}\n\n"
                "EVALUATION CRITERION\n"
                "--------------------\n"
                f"{impl.EVAL_CRITERION}\n\n"
                f"The complete generated text below contains exactly {n} sentences "
                "under the experiment's frozen segmentation. Assess each sentence "
                "in the context of the COMPLETE text. Return exactly "
                f"{n} bare integers from 0 to 100, one per line and in sentence "
                "order. Do not output words, punctuation, labels, angle-bracket "
                "tags, markdown, or explanation.\n\n"
                "COMPLETE TEXT\n"
                "-------------\n"
                f"{text}"
            ),
        },
    ]


_original_generation = impl.ensure_generation


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

    row = store.generation(model_id, question.question_id, arm)
    if row is None or row["status"] != "complete":
        return
    units = json.loads(row["units_json"])
    n = len(units)

    existing_prefix = store.eval_scores(model_id, question.question_id, arm, "prefix")
    for idx in range(n):
        if idx in existing_prefix:
            continue
        messages = _prefix_messages(question.task, units, idx)
        prompt_hash = impl.sha256_text(
            json.dumps(messages, ensure_ascii=False, sort_keys=True)
        )
        try:
            result, prompt_hash = impl.call_and_log(
                store, client, model_id=model_id, route=route,
                qid=question.question_id, arm=arm, phase="prefix_v2",
                sentence_index=idx, messages=messages, max_tokens=prefix_max_tokens,
            )
            score = _parse_numeric_lines(result.text, 1)[0]
            store.save_eval_score(
                model_id=model_id, qid=question.question_id, arm=arm,
                kind="prefix", idx=idx, score=score,
                prompt_hash=prompt_hash, error=None,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            store.save_eval_score(
                model_id=model_id, qid=question.question_id, arm=arm,
                kind="prefix", idx=idx, score=None,
                prompt_hash=prompt_hash, error=error,
            )
            print(
                f"[ERROR] {model_id} {question.question_id} {arm} "
                f"prefix[{idx}]: {error}", flush=True,
            )
            return

    existing_post = store.eval_scores(model_id, question.question_id, arm, "post")
    if len(existing_post) == n and all(i in existing_post for i in range(n)):
        return

    messages = _post_messages(question.task, units)
    prompt_hash = impl.sha256_text(
        json.dumps(messages, ensure_ascii=False, sort_keys=True)
    )
    try:
        result, prompt_hash = impl.call_and_log(
            store, client, model_id=model_id, route=route,
            qid=question.question_id, arm=arm, phase="post_v2",
            sentence_index=None, messages=messages, max_tokens=post_max_tokens,
        )
        scores = _parse_numeric_lines(result.text, n)
        store.save_post_scores(
            model_id=model_id, qid=question.question_id, arm=arm,
            scores=scores, prompt_hash=prompt_hash,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        store.save_eval_score(
            model_id=model_id, qid=question.question_id, arm=arm,
            kind="post", idx=-1, score=None, prompt_hash=prompt_hash, error=error,
        )
        print(
            f"[ERROR] {model_id} {question.question_id} {arm} post: {error}",
            flush=True,
        )


impl.ensure_generation = frozen_generation
impl.ensure_evaluations = frozen_evaluations

if __name__ == "__main__":
    raise SystemExit(impl.main())
