#!/usr/bin/env python3
"""
Adaptive Replay extension for the Gogol-effect analysis.

This extension RETAINS the existing 4-prompt Replay pilot and grows each
unresolved model in increments of four prompts:

    total N = 4 -> 8 -> 12 -> 16 -> 20 -> 24

Models are handled independently. A model already established by the existing
full Replay data is never rerun. For the four additional models in the breadth
extension, Replay is added only until the selected stopping rule is met or N=24.

Two p-values are reported at every look:

1. cumulative_raw_p
   One-sided prompt-level t-test using the existing pilot plus all added prompts.
   This is the intuitive cumulative p-value requested for monitoring, but because
   the extension was chosen after looking at the initial pilot it is EXPLORATORY.

2. extension_seq_adjusted_p
   One-sided t-test using NEW prompts only, Bonferroni-adjusted over the five
   prespecified future looks (new N = 4, 8, 12, 16, 20). This is the cleaner
   confirmatory statistic for the adaptive extension.

By default, the runner stops a model only when extension_seq_adjusted_p <= .05.
Use --stop-rule cumulative to stop as soon as cumulative_raw_p <= .05; this
saves more API cost but must be described as adaptive/exploratory in a paper.

The original Live/Post outputs are never regenerated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

EXTENSION_MODELS = (
    "claude-opus-5",
    "deepseek-v4-flash",
    "gpt-oss-120b",
    "llama-3.3-70b-instruct",
)

ALREADY_FULL_REPLAY = (
    "gpt-5.5",
    "gemini-3.1-pro-preview",
)

PILOT_IDS = ("p001", "p042", "p044", "p070")
TOTAL_LOOKS = (8, 12, 16, 20, 24)
NEW_LOOKS = (4, 8, 12, 16, 20)
N_FUTURE_LOOKS = len(NEW_LOOKS)
ALPHA = 0.05
SEED = 20260922


def connect(db: Path):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("v57base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import original runner: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v57base"] = mod
    spec.loader.exec_module(mod)
    return mod


def q_language(metadata_json: str) -> str | None:
    try:
        return json.loads(metadata_json).get("language")
    except Exception:
        return None


def live_or_post_mean(con, model: str, qid: str, condition: str):
    row = con.execute(
        """
        SELECT AVG(score) AS m, COUNT(*) AS n
        FROM scores
        WHERE model_id=? AND question_id=? AND condition=?
        """,
        (model, qid, condition),
    ).fetchone()
    if row is None or int(row["n"] or 0) == 0:
        return None, 0
    return float(row["m"]), int(row["n"])


def replay_mean(con, model: str, qid: str):
    """Prefer expanded Replay, otherwise baseline/adaptive replay_steps."""
    try:
        row = con.execute(
            """
            SELECT AVG(score) AS m, COUNT(*) AS n
            FROM replay_extension_steps
            WHERE model_id=? AND question_id=? AND status='complete'
            """,
            (model, qid),
        ).fetchone()
        if row is not None and int(row["n"] or 0) > 0:
            return float(row["m"]), int(row["n"])
    except sqlite3.OperationalError:
        pass

    row = con.execute(
        """
        SELECT AVG(score) AS m, COUNT(*) AS n
        FROM replay_steps
        WHERE model_id=? AND question_id=? AND status='complete'
        """,
        (model, qid),
    ).fetchone()
    if row is None or int(row["n"] or 0) == 0:
        return None, 0
    return float(row["m"]), int(row["n"])


def replay_complete(con, model: str, qid: str) -> bool:
    live, nl = live_or_post_mean(con, model, qid, "live")
    rep, nr = replay_mean(con, model, qid)
    return live is not None and rep is not None and nl > 0 and nr == nl


def prompt_effect(con, model: str, qid: str):
    live, _ = live_or_post_mean(con, model, qid, "live")
    post, _ = live_or_post_mean(con, model, qid, "post")
    rep, _ = replay_mean(con, model, qid)
    if live is None or post is None or rep is None:
        return None
    return {
        "question_id": qid,
        "gogol": post - rep,
        "ikea": rep - live,
        "live": live,
        "replay": rep,
        "post": post,
    }


def one_sided_t(x):
    a = np.asarray(x, dtype=float)
    if len(a) < 2:
        return math.nan, math.nan
    r = stats.ttest_1samp(a, 0.0, alternative="greater")
    return float(r.statistic), float(r.pvalue)


def wilcoxon_greater(x):
    a = np.asarray(x, dtype=float)
    if len(a) == 0 or np.allclose(a, 0):
        return math.nan
    try:
        return float(stats.wilcoxon(a, alternative="greater", method="auto").pvalue)
    except Exception:
        return math.nan


def ci95(x):
    a = np.asarray(x, dtype=float)
    if len(a) < 2:
        return [math.nan, math.nan]
    m = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(len(a)))
    crit = float(stats.t.ppf(0.975, len(a) - 1))
    return [m - crit * se, m + crit * se]


def english_ids(con) -> list[str]:
    rows = con.execute("SELECT question_id, metadata_json FROM questions").fetchall()
    return sorted(
        str(r["question_id"])
        for r in rows
        if q_language(str(r["metadata_json"])) == "en"
    )


def eligible_new_order(con, model: str) -> list[str]:
    eligible = set()
    for qid in english_ids(con):
        lr, nl = live_or_post_mean(con, model, qid, "live")
        pr, np_ = live_or_post_mean(con, model, qid, "post")
        if lr is not None and pr is not None and nl > 0 and np_ > 0:
            eligible.add(qid)

    eligible -= set(PILOT_IDS)
    order = sorted(eligible)
    rng = random.Random(SEED)
    rng.shuffle(order)
    return order


def existing_pilot_rows(con, model: str):
    rows = []
    for qid in PILOT_IDS:
        if replay_complete(con, model, qid):
            r = prompt_effect(con, model, qid)
            if r is not None:
                rows.append(r)
    return rows


def complete_new_rows(con, model: str, order: list[str]):
    rows = []
    for qid in order:
        if replay_complete(con, model, qid):
            r = prompt_effect(con, model, qid)
            if r is not None:
                rows.append(r)
    return rows


def summarize(model: str, pilot_rows, new_rows):
    cum_rows = list(pilot_rows) + list(new_rows)
    g_cum = np.asarray([r["gogol"] for r in cum_rows], dtype=float)
    i_cum = np.asarray([r["ikea"] for r in cum_rows], dtype=float)
    g_new = np.asarray([r["gogol"] for r in new_rows], dtype=float)

    _, p_cum = one_sided_t(g_cum)
    _, p_new = one_sided_t(g_new)
    p_new_adj = min(1.0, p_new * N_FUTURE_LOOKS) if np.isfinite(p_new) else math.nan

    return {
        "model_id": model,
        "n_pilot": len(pilot_rows),
        "n_new": len(new_rows),
        "n_total": len(cum_rows),
        "mean_gogol_total": float(np.mean(g_cum)) if len(g_cum) else math.nan,
        "median_gogol_total": float(np.median(g_cum)) if len(g_cum) else math.nan,
        "ci95_gogol_total": ci95(g_cum),
        "cumulative_raw_p": p_cum,
        "cumulative_wilcoxon_p": wilcoxon_greater(g_cum),
        "mean_ikea_total": float(np.mean(i_cum)) if len(i_cum) else math.nan,
        "extension_raw_p_new_only": p_new,
        "extension_seq_adjusted_p": p_new_adj,
        "pilot_question_ids": [r["question_id"] for r in pilot_rows],
        "new_question_ids": [r["question_id"] for r in new_rows],
    }


def question_objects(base, con, qids):
    if not qids:
        return []
    marks = ",".join("?" for _ in qids)
    rows = con.execute(
        f"SELECT question_id, family, task, metadata_json FROM questions WHERE question_id IN ({marks})",
        tuple(qids),
    ).fetchall()
    by = {}
    for r in rows:
        by[str(r["question_id"])] = base.Question(
            question_id=str(r["question_id"]),
            family=str(r["family"]),
            task=str(r["task"]),
            metadata=json.loads(str(r["metadata_json"])),
        )
    return [by[q] for q in qids]


def route_for(con, model: str):
    row = con.execute(
        "SELECT model_route FROM cells WHERE model_id=? LIMIT 1", (model,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Model route missing: {model}")
    return str(row["model_route"])


def load_key(base, runner_path: Path):
    base.load_simple_dotenv(Path.cwd() / ".env")
    base.load_simple_dotenv(runner_path.parent / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return key


def run_replay(base, db: Path, runner_path: Path, model: str, qids, args):
    if not qids:
        return
    with connect(db) as con:
        qs = question_objects(base, con, qids)
        route = route_for(con, model)

    store = base.Store(db)
    client = base.FastFailOpenRouterClient(
        api_key=load_key(base, runner_path),
        temperature=args.temperature,
        timeout_s=args.timeout,
        retries=args.retries,
        max_retry_delay_s=args.max_retry_delay,
    )
    runner = base.AdaptiveRunner(
        store=store,
        client=client,
        questions=qs,
        models={model: route},
        workers=args.workers,
        per_model_workers=args.per_model_workers,
        live_max_tokens=4096,
        post_max_tokens=4096,
        replay_max_tokens=args.replay_max_tokens,
    )
    runner.run_replay_selected(set(qids))


def ensure_backup(db: Path):
    bak = db.with_name("experiment_pre_gogol_step4.sqlite3")
    if not bak.exists():
        shutil.copy2(db, bak)
        print(f"Backup created: {bak}")


def dump(path: Path, obj: Any):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def existing_full_replay_summary(con, model: str):
    rows = []
    for qid in english_ids(con):
        if replay_complete(con, model, qid):
            r = prompt_effect(con, model, qid)
            if r is not None:
                rows.append(r)
    return summarize(model, [], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("run"))
    ap.add_argument("--runner", type=Path, default=Path("self_confidence_v5_7_promptbank.py"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument(
        "--stop-rule",
        choices=("confirmatory", "cumulative"),
        default="confirmatory",
        help=(
            "confirmatory: stop at extension sequential-adjusted p<=.05; "
            "cumulative: stop at cumulative raw p<=.05 (exploratory/adaptive)"
        ),
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-model-workers", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-retry-delay", type=float, default=20.0)
    ap.add_argument("--replay-max-tokens", type=int, default=1024)
    args = ap.parse_args()

    db = args.run_dir / "experiment.sqlite3"
    if not db.exists():
        raise SystemExit(f"Missing database: {db}")
    if not args.runner.exists():
        alt = args.run_dir.parent / args.runner.name
        if alt.exists():
            args.runner = alt
        else:
            raise SystemExit(f"Missing original runner: {args.runner}")

    base = load_base(args.runner)

    with connect(db) as con:
        orders = {m: eligible_new_order(con, m) for m in EXTENSION_MODELS}
        initial = {m: existing_pilot_rows(con, m) for m in EXTENSION_MODELS}
        full = {m: existing_full_replay_summary(con, m) for m in ALREADY_FULL_REPLAY}

    for m in EXTENSION_MODELS:
        if len(initial[m]) != 4:
            print(f"WARNING: {m} has {len(initial[m])}/4 complete pilot prompts")
        if len(orders[m]) < 20:
            raise SystemExit(f"{m} has only {len(orders[m])} eligible new prompts; need 20")

    manifest = {
        "design": "retain existing 4-prompt pilot, then +4 per unresolved model",
        "extension_models": list(EXTENSION_MODELS),
        "already_full_replay_models": list(ALREADY_FULL_REPLAY),
        "pilot_ids": list(PILOT_IDS),
        "total_looks": list(TOTAL_LOOKS),
        "new_sample_looks": list(NEW_LOOKS),
        "seed": SEED,
        "model_specific_new_prompt_orders": {m: orders[m][:20] for m in EXTENSION_MODELS},
        "default_stop_rule": args.stop_rule,
        "notes": {
            "cumulative_raw_p": "uses pilot+new; exploratory because pilot was inspected before extension",
            "extension_seq_adjusted_p": "new-only one-sided t p multiplied by 5 future looks",
        },
    }
    dump(args.run_dir / "gogol_step4_protocol.json", manifest)

    print("=== EXISTING FULL REPLAY (NO NEW CALLS) ===")
    for m, s in full.items():
        print(
            f"{m}: N={s['n_total']}, mean Gogol={s['mean_gogol_total']:.3f}, "
            f"p={s['cumulative_raw_p']:.3g}"
        )

    print("\n=== EXISTING 4-PROMPT PILOT FOR EXTENSION MODELS ===")
    initial_summaries = {}
    for m in EXTENSION_MODELS:
        s = summarize(m, initial[m], [])
        initial_summaries[m] = s
        print(
            f"{m}: N={s['n_total']}, mean Gogol={s['mean_gogol_total']:.3f}, "
            f"cumulative raw p={s['cumulative_raw_p']:.4g}"
        )
        print("  next frozen prompts:", ", ".join(orders[m][:20]))

    if args.dry_run or not args.auto:
        return

    ensure_backup(db)
    status = {
        m: {
            "stopped": False,
            "stopped_at_total_n": None,
            "reason": None,
            "history": [initial_summaries[m]],
        }
        for m in EXTENSION_MODELS
    }

    # If an extension model was already significant on the existing pilot,
    # honor the user's requested economy rule and do not spend more on it.
    for m in EXTENSION_MODELS:
        s = initial_summaries[m]
        if np.isfinite(s["cumulative_raw_p"]) and s["cumulative_raw_p"] <= 0.05 and s["mean_gogol_total"] > 0:
            status[m]["stopped"] = True
            status[m]["stopped_at_total_n"] = s["n_total"]
            status[m]["reason"] = "already cumulative p<=.05 before extension"
            print(f"{m}: already p<=.05; FROZEN before new calls")

    for total_target, new_target in zip(TOTAL_LOOKS, NEW_LOOKS):
        print(f"\n========== LOOK total N={total_target} (new N={new_target}) ==========")
        for m in EXTENSION_MODELS:
            if status[m]["stopped"]:
                print(f"{m}: frozen at N={status[m]['stopped_at_total_n']}; skip")
                continue

            order = orders[m]
            while True:
                with connect(db) as con:
                    new_rows = complete_new_rows(con, m, order)
                if len(new_rows) >= new_target:
                    break

                need = new_target - len(new_rows)
                done_ids = {r["question_id"] for r in new_rows}
                batch = [q for q in order if q not in done_ids][:need]
                if not batch:
                    break
                print(f"{m}: Replay +{len(batch)} -> {', '.join(batch)}")
                before = len(new_rows)
                run_replay(base, db, args.runner, m, batch, args)
                with connect(db) as con:
                    after = len(complete_new_rows(con, m, order))
                if after <= before:
                    print(f"{m}: no progress; inspect technical failure before resuming")
                    break

            with connect(db) as con:
                pilot_rows = existing_pilot_rows(con, m)
                new_rows = complete_new_rows(con, m, order)[:new_target]
            s = summarize(m, pilot_rows, new_rows)
            status[m]["history"].append(s)

            print(
                f"{m}: total N={s['n_total']} | mean G={s['mean_gogol_total']:.3f} | "
                f"cumulative p={s['cumulative_raw_p']:.5g} | "
                f"new-only seq-adj p={s['extension_seq_adjusted_p']:.5g}"
            )

            stop = False
            reason = None
            if args.stop_rule == "cumulative":
                if np.isfinite(s["cumulative_raw_p"]) and s["cumulative_raw_p"] <= 0.05 and s["mean_gogol_total"] > 0:
                    stop = True
                    reason = "cumulative raw p<=.05 (adaptive/exploratory)"
            else:
                if np.isfinite(s["extension_seq_adjusted_p"]) and s["extension_seq_adjusted_p"] <= 0.05 and s["mean_gogol_total"] > 0:
                    stop = True
                    reason = "new-only sequential-adjusted p<=.05"

            if stop:
                status[m]["stopped"] = True
                status[m]["stopped_at_total_n"] = s["n_total"]
                status[m]["reason"] = reason
                print(f"  -> SIGNIFICANT under selected rule; freeze {m}")

            dump(args.run_dir / "gogol_step4_results.json", status)

        if all(status[m]["stopped"] for m in EXTENSION_MODELS):
            print("\nAll extension models frozen. STOP.")
            break

    print("\n=== FINAL STATUS ===")
    for m in EXTENSION_MODELS:
        s = status[m]
        print(f"{m}: stopped={s['stopped']} N={s['stopped_at_total_n']} reason={s['reason']}")


if __name__ == "__main__":
    main()
