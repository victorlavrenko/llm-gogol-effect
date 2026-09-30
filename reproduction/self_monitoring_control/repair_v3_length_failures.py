#!/usr/bin/env python3
"""Repair only v3 evaluator calls that were truncated by max_tokens.

This is a post-run technical repair, not outcome-driven resampling.
Eligibility is determined solely from the frozen run metadata:
  * model == gemini-3.1-pro-preview
  * PREFIX evaluation status == error
  * the corresponding original API call has finish_reason == "length"

For each eligible prompt-arm, the already-generated prose is frozen. The script:
  1) copies the original run directory to a separate repaired directory;
  2) retries the length-truncated PREFIX cell with a larger output budget;
  3) evaluates any later PREFIX sentences that were never reached because the
     original runner stopped after the truncation;
  4) runs POST for that arm if it was never reached;
  5) never regenerates SCORED or PLAIN prose.

All repair calls are logged in the copied SQLite database. The original run
remains untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import frozen_control_v3 as v3

impl = v3.v2.impl
TARGET_MODEL = "gemini-3.1-pro-preview"
REPAIR_VERSION = "v3-length-repair-v1"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Repair only length-truncated v3 evaluator calls")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prefix-max-tokens", type=int, default=4096)
    p.add_argument("--post-max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    return p


def finish_reason(raw_json: str | None) -> str | None:
    if not raw_json:
        return None
    try:
        obj = json.loads(raw_json)
        choices = obj.get("choices") or []
        if not choices:
            return None
        return choices[0].get("finish_reason")
    except Exception:
        return None


def load_protocol(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))


def eligible_arms(db_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT e.model_id,e.question_id,e.arm,e.sentence_index,e.error,
               c.call_id,c.phase,c.raw_json,c.completion_tokens,c.response_text
        FROM evaluations e
        JOIN calls c
          ON c.model_id=e.model_id
         AND c.question_id=e.question_id
         AND c.arm=e.arm
         AND c.sentence_index=e.sentence_index
        WHERE e.model_id=? AND e.kind='prefix' AND e.status='error'
          AND c.phase LIKE 'prefix%'
        ORDER BY e.question_id,e.arm,c.call_id DESC
        """,
        (TARGET_MODEL,),
    ).fetchall()
    con.close()

    # One original failure per prompt-arm. Keep the latest matching call if a
    # copied repair directory is resumed and contains additional call history.
    seen: set[tuple[str, str]] = set()
    eligible: list[dict[str, Any]] = []
    noneligible: list[dict[str, Any]] = []
    for r in rows:
        key = (str(r["question_id"]), str(r["arm"]))
        if key in seen:
            continue
        seen.add(key)
        item = {
            "model_id": str(r["model_id"]),
            "question_id": str(r["question_id"]),
            "arm": str(r["arm"]),
            "failed_sentence_index": int(r["sentence_index"]),
            "original_error": r["error"],
            "original_call_id": int(r["call_id"]),
            "original_phase": str(r["phase"]),
            "original_finish_reason": finish_reason(r["raw_json"]),
            "original_completion_tokens": r["completion_tokens"],
        }
        if item["original_finish_reason"] == "length":
            eligible.append(item)
        else:
            noneligible.append(item)
    return eligible, noneligible


def questions_for_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    bank_path = Path(protocol["prompt_bank"])
    rows, _ = impl.base.load_fixed_prompt_bank(
        bank_path, protocol["prompt_bank_sha256"]
    )
    questions = impl.choose_questions(rows, str(protocol["mode"]), int(protocol["seed"]))
    return {q.question_id: q for q in questions}


def all_prefix_complete(store: Any, model_id: str, qid: str, arm: str, n: int) -> bool:
    scores = store.eval_scores(model_id, qid, arm, "prefix")
    return len(scores) == n and all(i in scores for i in range(n))


def repair_arm(
    *,
    store: Any,
    client: Any,
    route: str,
    question: Any,
    arm: str,
    prefix_max_tokens: int,
    post_max_tokens: int,
) -> dict[str, Any]:
    qid = question.question_id
    model_id = TARGET_MODEL
    gen = store.generation(model_id, qid, arm)
    if gen is None or gen["status"] != "complete":
        return {"question_id": qid, "arm": arm, "status": "generation_not_complete"}
    units = json.loads(gen["units_json"])
    n = len(units)

    # Retry the failed PREFIX and fill any later PREFIX calls that were never
    # attempted because the original runner returned immediately on failure.
    for idx in range(n):
        existing = store.eval_scores(model_id, qid, arm, "prefix")
        if idx in existing:
            continue
        messages = v3._prefix_messages(question.task, units, idx)
        prompt_hash = impl.sha256_text(
            json.dumps(messages, ensure_ascii=False, sort_keys=True)
        )
        result = None
        try:
            result, prompt_hash = impl.call_and_log(
                store, client, model_id=model_id, route=route,
                qid=qid, arm=arm, phase="prefix_v3_length_repair",
                sentence_index=idx, messages=messages,
                max_tokens=prefix_max_tokens,
            )
            score = v3._parse_score_contract(result.text, 1)[0]
            store.save_eval_score(
                model_id=model_id, qid=qid, arm=arm, kind="prefix",
                idx=idx, score=score, prompt_hash=prompt_hash, error=None,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            store.save_eval_score(
                model_id=model_id, qid=qid, arm=arm, kind="prefix",
                idx=idx, score=None, prompt_hash=prompt_hash, error=error,
            )
            return {
                "question_id": qid, "arm": arm, "status": "prefix_repair_error",
                "sentence_index": idx, "error": error,
            }

    if not all_prefix_complete(store, model_id, qid, arm, n):
        return {"question_id": qid, "arm": arm, "status": "prefix_incomplete"}

    post = store.eval_scores(model_id, qid, arm, "post")
    if not (len(post) == n and all(i in post for i in range(n))):
        messages = v3._post_messages(question.task, units)
        prompt_hash = impl.sha256_text(
            json.dumps(messages, ensure_ascii=False, sort_keys=True)
        )
        try:
            result, prompt_hash = impl.call_and_log(
                store, client, model_id=model_id, route=route,
                qid=qid, arm=arm, phase="post_v3_after_length_repair",
                sentence_index=None, messages=messages,
                max_tokens=post_max_tokens,
            )
            scores = v3._parse_score_contract(result.text, n)
            store.save_post_scores(
                model_id=model_id, qid=qid, arm=arm,
                scores=scores, prompt_hash=prompt_hash,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            store.save_eval_score(
                model_id=model_id, qid=qid, arm=arm, kind="post",
                idx=-1, score=None, prompt_hash=prompt_hash, error=error,
            )
            return {"question_id": qid, "arm": arm, "status": "post_repair_error", "error": error}

    return {"question_id": qid, "arm": arm, "status": "complete"}


def main() -> int:
    args = parser().parse_args()
    source = args.source.resolve()
    out = args.out.resolve()
    if not (source / "experiment.sqlite3").exists():
        raise SystemExit(f"source run has no experiment.sqlite3: {source}")

    if out.exists():
        if not args.resume:
            raise SystemExit(
                f"repair output already exists: {out}\n"
                "Use --resume only to continue a previously started repair."
            )
    else:
        shutil.copytree(source, out)

    protocol = load_protocol(out)
    if protocol.get("version") != "self-monitoring-control-v3":
        raise SystemExit(f"expected v3 run, got {protocol.get('version')!r}")

    db_path = out / "experiment.sqlite3"
    eligible, noneligible = eligible_arms(db_path)
    audit = {
        "repair_version": REPAIR_VERSION,
        "source_run": str(source),
        "repaired_run": str(out),
        "selection_rule": "Gemini PREFIX error whose original API finish_reason is exactly 'length'",
        "prefix_max_tokens_repair": args.prefix_max_tokens,
        "post_max_tokens_repair": args.post_max_tokens,
        "eligible_before_repair": eligible,
        "noneligible_prefix_errors_before_repair": noneligible,
    }
    (out / "repair_eligibility.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Eligible length-truncated prompt-arms: {len(eligible)}")
    if noneligible:
        print(f"Non-length PREFIX errors left untouched: {len(noneligible)}")

    impl.base.load_simple_dotenv(Path.cwd() / ".env")
    impl.base.load_simple_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    qmap = questions_for_protocol(protocol)
    route = protocol["models"][TARGET_MODEL]
    client = impl.base.OpenRouterClient(
        api_key=api_key,
        temperature=float(protocol["evaluation_temperature"]),
        timeout_s=args.timeout,
        retries=args.retries,
    )
    store = impl.Store(db_path)

    results: list[dict[str, Any]] = []
    for item in eligible:
        qid = item["question_id"]
        arm = item["arm"]
        print(f"[REPAIR] {TARGET_MODEL} {qid} {arm}", flush=True)
        result = repair_arm(
            store=store, client=client, route=route, question=qmap[qid], arm=arm,
            prefix_max_tokens=args.prefix_max_tokens,
            post_max_tokens=args.post_max_tokens,
        )
        print(f"  -> {result['status']}", flush=True)
        results.append(result)

    audit["repair_results"] = results
    audit["n_complete_repairs"] = sum(r.get("status") == "complete" for r in results)
    (out / "repair_eligibility.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    impl.export_qc(store, out)
    print(f"Repair finished. Repaired copy: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
