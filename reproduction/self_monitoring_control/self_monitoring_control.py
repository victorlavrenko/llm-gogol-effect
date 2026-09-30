#!/usr/bin/env python3
"""
Prospective self-monitoring control for the Gogol-effect study.

Two generation arms for the same frozen prompts and same model route:
  scored: exact V5.7 LIVE prompt, with sentence-by-sentence self-scoring.
  plain:  same writing task, with the scoring instruction removed entirely.

Both frozen artifacts are then evaluated by the SAME clean evaluators:
  prefix: score sentence i with only prose through sentence i visible; no prior score tokens.
  post:   score every sentence with the completed prose visible.

Primary estimands are computed at prompt level:
  C_plain  = mean(POST_plain - PREFIX_plain)
  C_scored = mean(POST_scored - PREFIX_scored)
  interaction = C_scored - C_plain

The runner is resumable. Successful API calls are never regenerated.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import random
import re
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
BASE_PATH = REPRO / "main_v57" / "self_confidence_v5_7_promptbank.py"

spec = importlib.util.spec_from_file_location("gogol_v57_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import base runner from {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

VERSION = "self-monitoring-control-v1"
DEFAULT_MODELS = ("gpt-5.5", "gemini-3.1-pro-preview")
DEFAULT_SEED = 20260907
DEFAULT_PROMPT_BANK = REPRO / "main_v57" / "run" / "prompt_bank_v3_frozen.jsonl"
SLOT = base.SLOT

EVAL_CRITERION = (
    "Assess how likely an average human reader would be to think the target "
    "sentence was AI-generated. Use an integer from 0 to 100, where 0 means "
    "very human-like and 100 means obviously AI-generated."
)

ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "u.s.", "u.k.", "no.", "fig.", "approx.", "dept.", "inc.",
}
BOUNDARY_RE = re.compile(r'[.!?…]+(?:["”’\)\]]+)?(?=\s+|$)', re.UNICODE)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def deterministic_rng(*parts: object) -> random.Random:
    material = "|".join(str(p) for p in parts)
    seed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def plain_generation_messages(task: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": base.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "ORIGINAL WRITING TASK\n"
                "---------------------\n"
                f"{task}\n\n"
                "Produce the requested text now."
            ),
        },
    ]


def scored_generation_messages(task: str) -> list[dict[str, str]]:
    return base.live_messages(task)


def _looks_like_abbreviation(sentence_so_far: str) -> bool:
    stripped = sentence_so_far.rstrip(' \t\r\n"”’)]')
    if not stripped.endswith("."):
        return False
    token = stripped.split()[-1].casefold() if stripped.split() else ""
    if token in ABBREVIATIONS:
        return True
    if re.search(r"(?:\b[A-Za-z]\.){2,}$", stripped):
        return True
    if re.search(r"\b[A-Z]\.$", stripped):
        return True
    if re.search(r"\b\d+\.\d+$", stripped):
        return True
    return False


def split_plain_sentences(text: str) -> tuple[list[str], str]:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines0 = cleaned.splitlines()
        if len(lines0) >= 3:
            cleaned = "\n".join(lines0[1:-1]).strip()

    if base._SCORE_LIKE_RE.search(cleaned) or base._BARE_SCORE_RE.search(cleaned):
        raise ValueError("plain generation unexpectedly contains AI SCORE markup")

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if 5 <= len(lines) <= 8 and all(re.search(r'[.!?…]["”’\)\]]*$', ln) for ln in lines):
        return lines, "lines"

    spans: list[str] = []
    start = 0
    for m in BOUNDARY_RE.finditer(cleaned):
        end = m.end()
        candidate = cleaned[start:end].strip()
        if not candidate:
            start = end
            continue
        if _looks_like_abbreviation(candidate):
            continue
        spans.append(candidate)
        start = end
    tail = cleaned[start:].strip()
    if tail:
        if spans and not re.search(r"\w", tail):
            spans[-1] = spans[-1] + tail
        else:
            spans.append(tail)

    if not (5 <= len(spans) <= 8):
        raise ValueError(
            f"plain generation segmented into {len(spans)} sentences; expected 5-8"
        )
    return spans, "punctuation"


def canonical_text(units: Sequence[str]) -> str:
    return " ".join(u.strip() for u in units if u.strip())


def post_template(units: Sequence[str]) -> str:
    return " ".join(f"{u.strip()} {SLOT}" for u in units if u.strip())


def prefix_messages(task: str, units: Sequence[str], target_index: int) -> list[dict[str, str]]:
    text_so_far = canonical_text(units[: target_index + 1])
    return [
        {"role": "system", "content": base.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "ORIGINAL WRITING TASK\n"
                "---------------------\n"
                f"{task}\n\n"
                "EVALUATION CRITERION\n"
                "--------------------\n"
                f"{EVAL_CRITERION}\n\n"
                "TEXT SO FAR\n"
                "-----------\n"
                f"{text_so_far}\n\n"
                "Score only the LAST sentence in TEXT SO FAR. Later sentences "
                "do not exist in this evaluation context. Return only "
                "<AI SCORE: n>."
            ),
        },
    ]


def post_messages_clean(task: str, units: Sequence[str]) -> list[dict[str, str]]:
    template = post_template(units)
    n = len(units)
    return [
        {"role": "system", "content": base.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "ORIGINAL WRITING TASK\n"
                "---------------------\n"
                f"{task}\n\n"
                "EVALUATION CRITERION\n"
                "--------------------\n"
                f"{EVAL_CRITERION}\n\n"
                "The complete generated text is shown below with one "
                "<AI SCORE: ?> placeholder after each sentence. Assess each "
                "sentence in the context of the COMPLETE text. Return exactly "
                f"{n} <AI SCORE: n> tags, one per line and in sentence order. "
                "Do not repeat or rewrite the text.\n\n"
                "COMPLETE TEXT\n"
                "-------------\n"
                f"{template}"
            ),
        },
    ]


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta(
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS generations(
                    model_id TEXT NOT NULL,
                    model_route TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    arm TEXT NOT NULL CHECK(arm IN ('scored','plain')),
                    status TEXT NOT NULL,
                    raw_text TEXT,
                    clean_text TEXT,
                    units_json TEXT,
                    live_scores_json TEXT,
                    segmentation_method TEXT,
                    provider TEXT,
                    prompt_sha256 TEXT,
                    error TEXT,
                    PRIMARY KEY(model_id, question_id, arm)
                );
                CREATE TABLE IF NOT EXISTS evaluations(
                    model_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    arm TEXT NOT NULL CHECK(arm IN ('scored','plain')),
                    kind TEXT NOT NULL CHECK(kind IN ('prefix','post')),
                    sentence_index INTEGER NOT NULL,
                    score INTEGER,
                    status TEXT NOT NULL,
                    prompt_sha256 TEXT,
                    error TEXT,
                    PRIMARY KEY(model_id, question_id, arm, kind, sentence_index)
                );
                CREATE TABLE IF NOT EXISTS calls(
                    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    model_id TEXT NOT NULL,
                    model_route TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    arm TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    sentence_index INTEGER,
                    provider TEXT,
                    prompt_sha256 TEXT NOT NULL,
                    response_text TEXT,
                    raw_json TEXT,
                    error TEXT,
                    latency_s REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER
                );
                """
            )

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=60)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=60000")
        return con

    def meta(self, key: str, value: Any) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def generation(self, model_id: str, qid: str, arm: str) -> sqlite3.Row | None:
        with self.connect() as con:
            return con.execute(
                "SELECT * FROM generations WHERE model_id=? AND question_id=? AND arm=?",
                (model_id, qid, arm),
            ).fetchone()

    def save_generation(
        self,
        *,
        model_id: str,
        route: str,
        qid: str,
        arm: str,
        raw_text: str | None,
        units: Sequence[str] | None,
        live_scores: Sequence[int] | None,
        segmentation_method: str | None,
        provider: str | None,
        prompt_hash: str,
        error: str | None,
    ) -> None:
        status = "complete" if error is None else "error"
        clean = canonical_text(units or ()) if units else None
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO generations(
                    model_id,model_route,question_id,arm,status,raw_text,clean_text,
                    units_json,live_scores_json,segmentation_method,provider,
                    prompt_sha256,error
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(model_id,question_id,arm) DO UPDATE SET
                    model_route=excluded.model_route,status=excluded.status,
                    raw_text=excluded.raw_text,clean_text=excluded.clean_text,
                    units_json=excluded.units_json,
                    live_scores_json=excluded.live_scores_json,
                    segmentation_method=excluded.segmentation_method,
                    provider=excluded.provider,prompt_sha256=excluded.prompt_sha256,
                    error=excluded.error
                """,
                (
                    model_id, route, qid, arm, status, raw_text, clean,
                    json.dumps(list(units), ensure_ascii=False) if units else None,
                    json.dumps(list(live_scores)) if live_scores is not None else None,
                    segmentation_method, provider, prompt_hash, error,
                ),
            )

    def eval_scores(self, model_id: str, qid: str, arm: str, kind: str) -> dict[int, int]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT sentence_index,score FROM evaluations
                WHERE model_id=? AND question_id=? AND arm=? AND kind=? AND status='complete'
                ORDER BY sentence_index
                """,
                (model_id, qid, arm, kind),
            ).fetchall()
        return {int(r["sentence_index"]): int(r["score"]) for r in rows}

    def save_eval_score(
        self, *, model_id: str, qid: str, arm: str, kind: str,
        idx: int, score: int | None, prompt_hash: str, error: str | None
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO evaluations(
                    model_id,question_id,arm,kind,sentence_index,score,status,prompt_sha256,error
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(model_id,question_id,arm,kind,sentence_index) DO UPDATE SET
                    score=excluded.score,status=excluded.status,
                    prompt_sha256=excluded.prompt_sha256,error=excluded.error
                """,
                (
                    model_id, qid, arm, kind, idx, score,
                    "complete" if error is None else "error", prompt_hash, error,
                ),
            )

    def save_post_scores(
        self, *, model_id: str, qid: str, arm: str,
        scores: Sequence[int], prompt_hash: str
    ) -> None:
        with self.connect() as con:
            for idx, score in enumerate(scores):
                con.execute(
                    """
                    INSERT INTO evaluations(
                        model_id,question_id,arm,kind,sentence_index,score,status,prompt_sha256,error
                    ) VALUES(?,?,?,'post',?,?,'complete',?,NULL)
                    ON CONFLICT(model_id,question_id,arm,kind,sentence_index) DO UPDATE SET
                        score=excluded.score,status='complete',
                        prompt_sha256=excluded.prompt_sha256,error=NULL
                    """,
                    (model_id, qid, arm, idx, int(score), prompt_hash),
                )

    def record_call(
        self, *, model_id: str, route: str, qid: str, arm: str, phase: str,
        sentence_index: int | None, prompt_hash: str, result: Any | None,
        error: str | None
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO calls(
                    model_id,model_route,question_id,arm,phase,sentence_index,
                    provider,prompt_sha256,response_text,raw_json,error,latency_s,
                    prompt_tokens,completion_tokens
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    model_id, route, qid, arm, phase, sentence_index,
                    getattr(result, "provider", None) if result else None,
                    prompt_hash,
                    getattr(result, "text", None) if result else None,
                    json.dumps(getattr(result, "raw_json", None), ensure_ascii=False)
                    if result else None,
                    error,
                    getattr(result, "latency_s", None) if result else None,
                    getattr(result, "prompt_tokens", None) if result else None,
                    getattr(result, "completion_tokens", None) if result else None,
                ),
            )


def call_and_log(
    store: Store, client: Any, *, model_id: str, route: str, qid: str,
    arm: str, phase: str, sentence_index: int | None,
    messages: Sequence[Mapping[str, str]], max_tokens: int
) -> Any:
    prompt_hash = sha256_text(json.dumps(list(messages), ensure_ascii=False, sort_keys=True))
    result = None
    try:
        result = client.call(
            model_id=model_id,
            model_route=route,
            messages=messages,
            max_tokens=max_tokens,
        )
        store.record_call(
            model_id=model_id, route=route, qid=qid, arm=arm, phase=phase,
            sentence_index=sentence_index, prompt_hash=prompt_hash,
            result=result, error=None,
        )
        return result, prompt_hash
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        store.record_call(
            model_id=model_id, route=route, qid=qid, arm=arm, phase=phase,
            sentence_index=sentence_index, prompt_hash=prompt_hash,
            result=result, error=error,
        )
        raise


def ensure_generation(
    store: Store, client: Any, *, model_id: str, route: str,
    question: Any, arm: str, max_tokens: int
) -> None:
    existing = store.generation(model_id, question.question_id, arm)
    if existing is not None and existing["status"] == "complete":
        return

    messages = (
        scored_generation_messages(question.task)
        if arm == "scored"
        else plain_generation_messages(question.task)
    )
    prompt_hash = sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    result = None
    try:
        result, prompt_hash = call_and_log(
            store, client, model_id=model_id, route=route,
            qid=question.question_id, arm=arm, phase="generate",
            sentence_index=None, messages=messages, max_tokens=max_tokens,
        )
        if arm == "scored":
            parsed = base.parse_live(
                result.text,
                expected_min_scores=base.CASE_SENTENCE_MIN,
                expected_max_scores=base.CASE_SENTENCE_MAX,
            )
            units = [u.strip() for u in parsed.units]
            live_scores = list(parsed.scores)
            method = "score_tags"
        else:
            units, method = split_plain_sentences(result.text)
            live_scores = None

        store.save_generation(
            model_id=model_id, route=route, qid=question.question_id, arm=arm,
            raw_text=result.text, units=units, live_scores=live_scores,
            segmentation_method=method, provider=result.provider,
            prompt_hash=prompt_hash, error=None,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        store.save_generation(
            model_id=model_id, route=route, qid=question.question_id, arm=arm,
            raw_text=getattr(result, "text", None) if result else None,
            units=None, live_scores=None, segmentation_method=None,
            provider=getattr(result, "provider", None) if result else None,
            prompt_hash=prompt_hash, error=error,
        )
        print(f"[ERROR] {model_id} {question.question_id} {arm} generation: {error}", flush=True)


def ensure_evaluations(
    store: Store, client: Any, *, model_id: str, route: str,
    question: Any, arm: str, prefix_max_tokens: int, post_max_tokens: int
) -> None:
    row = store.generation(model_id, question.question_id, arm)
    if row is None or row["status"] != "complete":
        return
    units = json.loads(row["units_json"])
    n = len(units)

    existing_prefix = store.eval_scores(model_id, question.question_id, arm, "prefix")
    for idx in range(n):
        if idx in existing_prefix:
            continue
        messages = prefix_messages(question.task, units, idx)
        prompt_hash = sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        try:
            result, prompt_hash = call_and_log(
                store, client, model_id=model_id, route=route,
                qid=question.question_id, arm=arm, phase="prefix",
                sentence_index=idx, messages=messages, max_tokens=prefix_max_tokens,
            )
            score, _mode = base.parse_single_replay_score(result.text, [])
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
            print(f"[ERROR] {model_id} {question.question_id} {arm} prefix[{idx}]: {error}", flush=True)
            return

    existing_post = store.eval_scores(model_id, question.question_id, arm, "post")
    if len(existing_post) == n and all(i in existing_post for i in range(n)):
        return

    messages = post_messages_clean(question.task, units)
    prompt_hash = sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    try:
        result, prompt_hash = call_and_log(
            store, client, model_id=model_id, route=route,
            qid=question.question_id, arm=arm, phase="post",
            sentence_index=None, messages=messages, max_tokens=post_max_tokens,
        )
        scores = base.parse_post(result.text, post_template(units))
        if len(scores) != n:
            raise ValueError(f"POST returned {len(scores)} scores for {n} sentences")
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
        print(f"[ERROR] {model_id} {question.question_id} {arm} post: {error}", flush=True)


def choose_questions(bank_rows: Sequence[Mapping[str, Any]], mode: str, seed: int) -> list[Any]:
    ordered = base.balanced_primary_questions(bank_rows, seed=seed)
    if mode == "full":
        if len(ordered) < 40:
            raise ValueError("need at least 40 primary prompts")
        return ordered[:40]
    if mode == "smoke":
        if len(ordered) < 44:
            raise ValueError("need at least 44 primary prompts for disjoint smoke cohort")
        return ordered[40:44]
    raise ValueError(mode)


def export_qc(store: Store, out_dir: Path) -> None:
    import csv
    out = out_dir / "generation_qc.csv"
    with store.connect() as con:
        rows = con.execute(
            """
            SELECT model_id,question_id,arm,status,segmentation_method,
                   units_json,provider,error
            FROM generations ORDER BY model_id,question_id,arm
            """
        ).fetchall()
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_id","question_id","arm","status","segmentation_method","n_sentences","provider","error"])
        for r in rows:
            n_sentences = len(json.loads(r["units_json"])) if r["units_json"] else None
            w.writerow([
                r["model_id"], r["question_id"], r["arm"], r["status"],
                r["segmentation_method"], n_sentences, r["provider"], r["error"],
            ])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self-monitoring intervention control")
    p.add_argument("--mode", choices=("smoke","full"), default="smoke")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--generation-temperature", type=float, default=0.7)
    p.add_argument("--evaluation-temperature", type=float, default=0.7)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--generation-max-tokens", type=int, default=4096)
    p.add_argument("--prefix-max-tokens", type=int, default=256)
    p.add_argument("--post-max-tokens", type=int, default=1024)
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    out_dir = args.out or (HERE / f"run_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, bank_sha = base.load_fixed_prompt_bank(
        args.prompt_bank, base.PROMPT_BANK_EXPECTED_SHA256
    )
    questions = choose_questions(rows, args.mode, args.seed)

    model_ids = [x.strip() for x in args.models.split(",") if x.strip()]
    unknown = [m for m in model_ids if m not in base.MODEL_SPECS]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}")
    models = {m: base.MODEL_SPECS[m] for m in model_ids}

    protocol = {
        "version": VERSION,
        "mode": args.mode,
        "seed": args.seed,
        "prompt_bank": str(args.prompt_bank),
        "prompt_bank_sha256": bank_sha,
        "question_ids": [q.question_id for q in questions],
        "models": models,
        "generation_temperature": args.generation_temperature,
        "evaluation_temperature": args.evaluation_temperature,
        "generation_arms": {
            "scored": "exact V5.7 LIVE prompt with online AI-score tags",
            "plain": "same original writing task with scoring instruction entirely absent",
        },
        "evaluators": {
            "prefix": "same-model clean prefix judge; no earlier score tokens; target is last visible sentence",
            "post": "same-model complete-text judge with one score per sentence",
        },
        "primary_estimands": [
            "plain completion shift = POST_plain - PREFIX_plain",
            "scored completion shift = POST_scored - PREFIX_scored",
            "interaction = scored completion shift - plain completion shift",
        ],
        "smoke_exclusion": "smoke cohort uses balanced-order positions 41-44 and is never pooled with full positions 1-40",
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(protocol, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("DRY RUN: no API calls.", flush=True)
        q = questions[0]
        print("\n--- SCORED GENERATION PROMPT ---")
        print(json.dumps(scored_generation_messages(q.task), indent=2, ensure_ascii=False))
        print("\n--- PLAIN GENERATION PROMPT ---")
        print(json.dumps(plain_generation_messages(q.task), indent=2, ensure_ascii=False))
        return 0

    base.load_simple_dotenv(Path.cwd() / ".env")
    base.load_simple_dotenv(HERE / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (put it in repo-root .env or environment)")

    store = Store(out_dir / "experiment.sqlite3")
    for k, v in protocol.items():
        store.meta(k, v)

    gen_client = base.OpenRouterClient(
        api_key=api_key, temperature=args.generation_temperature,
        timeout_s=args.timeout, retries=args.retries,
    )
    eval_client = base.OpenRouterClient(
        api_key=api_key, temperature=args.evaluation_temperature,
        timeout_s=args.timeout, retries=args.retries,
    )

    def model_job(model_id: str, route: str) -> None:
        for q in questions:
            arm_order = ["scored", "plain"]
            deterministic_rng(VERSION, args.seed, model_id, q.question_id, "generation-order").shuffle(arm_order)
            for arm in arm_order:
                ensure_generation(
                    store, gen_client, model_id=model_id, route=route,
                    question=q, arm=arm, max_tokens=args.generation_max_tokens,
                )
            eval_order = ["scored", "plain"]
            deterministic_rng(VERSION, args.seed, model_id, q.question_id, "evaluation-order").shuffle(eval_order)
            for arm in eval_order:
                ensure_evaluations(
                    store, eval_client, model_id=model_id, route=route, question=q, arm=arm,
                    prefix_max_tokens=args.prefix_max_tokens,
                    post_max_tokens=args.post_max_tokens,
                )
            print(f"[OK] {model_id} {q.question_id}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(models))) as ex:
        futures = [ex.submit(model_job, m, r) for m, r in models.items()]
        for fut in futures:
            fut.result()

    export_qc(store, out_dir)
    print(f"\nCollection complete/resumed: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
