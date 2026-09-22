#!/usr/bin/env python3
"""Reconstruct the six-model IKEA/Gogol headline table used in the arXiv paper.

For GPT-5.5 and Gemini 3.1 Pro, effects are recomputed from the fixed 40-prompt
English Live/Replay/Post data in experiment.sqlite3.  For the four adaptive
breadth models, the final analysis state is read from gogol_step4_results.json,
which records every interim look and the stopped/capped sample sizes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
from scipy import stats

FIXED = ("gpt-5.5", "gemini-3.1-pro-preview")
ADAPTIVE = (
    "gpt-oss-120b",
    "deepseek-v4-flash",
    "llama-3.3-70b-instruct",
    "claude-opus-5",
)
DISPLAY = {
    "gpt-5.5": "GPT-5.5",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gpt-oss-120b": "GPT-OSS-120B",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "claude-opus-5": "Claude Opus 5",
}


def db_connect(path: Path):
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    return c


def cond_mean(c, model: str, qid: str, condition: str):
    r = c.execute(
        "SELECT AVG(score) AS m, COUNT(*) AS n FROM scores "
        "WHERE model_id=? AND question_id=? AND condition=?",
        (model, qid, condition),
    ).fetchone()
    if not r or not r["n"]:
        return None, 0
    return float(r["m"]), int(r["n"])


def replay_mean(c, model: str, qid: str):
    # Expanded/final replay may live in replay_extension_steps for adaptive runs.
    for table in ("replay_extension_steps", "replay_steps"):
        try:
            r = c.execute(
                f"SELECT AVG(score) AS m, COUNT(*) AS n FROM {table} "
                "WHERE model_id=? AND question_id=? AND status='complete'",
                (model, qid),
            ).fetchone()
        except sqlite3.OperationalError:
            continue
        if r and r["n"]:
            return float(r["m"]), int(r["n"])
    return None, 0


def english_qids(c):
    out = []
    for r in c.execute("SELECT question_id, metadata_json FROM questions ORDER BY question_id"):
        try:
            lang = json.loads(r["metadata_json"]).get("language")
        except Exception:
            lang = None
        if lang == "en":
            out.append(str(r["question_id"]))
    return out


def fixed_effects(c, model: str):
    vals = []
    for q in english_qids(c):
        live, nl = cond_mean(c, model, q, "live")
        post, _ = cond_mean(c, model, q, "post")
        replay, nr = replay_mean(c, model, q)
        if live is None or post is None or replay is None or nl != nr:
            continue
        vals.append((post - replay, replay - live))
    g = np.asarray([x[0] for x in vals], dtype=float)
    i = np.asarray([x[1] for x in vals], dtype=float)
    if len(g) < 2:
        raise RuntimeError(f"Not enough complete Replay prompts for {model}: {len(g)}")
    pg = float(stats.ttest_1samp(g, 0, alternative="greater").pvalue)
    pi = float(stats.ttest_1samp(i, 0, alternative="greater").pvalue)
    return {
        "n_prompts": int(len(g)),
        "mean_replay_minus_live": float(i.mean()),
        "mean_post_minus_replay": float(g.mean()),
        "p_gogol_one_sided": pg,
        "p_ikea_one_sided": pi,
        "fresh_extension_seq_adjusted_p": math.nan,
        "design": "fixed 40-prompt English decomposition",
    }


def adaptive_effects(results_json: dict, model: str):
    rec = results_json[model]
    hist = rec.get("history") or []
    if not hist:
        raise RuntimeError(f"No adaptive history for {model}")
    final = hist[-1]
    # If a model stopped earlier, history[-1] is the stopped look; otherwise cap.
    return {
        "n_prompts": int(final["n_total"]),
        "mean_replay_minus_live": float(final["mean_ikea_total"]),
        "mean_post_minus_replay": float(final["mean_gogol_total"]),
        "p_gogol_one_sided": float(final["cumulative_raw_p"]),
        "p_ikea_one_sided": math.nan,
        "fresh_extension_seq_adjusted_p": (
            float(final["extension_seq_adjusted_p"])
            if final.get("extension_seq_adjusted_p") is not None
            and math.isfinite(float(final["extension_seq_adjusted_p"]))
            else math.nan
        ),
        "design": (
            "adaptive breadth extension; cumulative p is exploratory/descriptive; "
            "fresh-extension sequential p is confirmatory support"
        ),
    }


def classify(row):
    i = row["mean_replay_minus_live"]
    g = row["mean_post_minus_replay"]
    # Descriptive operational classification used in the manuscript narrative.
    if i > 0 and g > 0:
        return "IKEA-like + Gogol"
    if g > 0 and row["model_id"] in {
        "gpt-5.5", "gpt-oss-120b", "deepseek-v4-flash"
    }:
        return "Gogol"
    if g > 0:
        return "no positive Gogol evidence at available sample"
    return "no positive Gogol evidence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--adaptive-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("six_model_effects.csv"))
    args = ap.parse_args()

    c = db_connect(args.db)
    adaptive = json.loads(args.adaptive_results.read_text(encoding="utf-8"))
    rows = []
    for model in FIXED:
        r = fixed_effects(c, model)
        r["model_id"] = model
        rows.append(r)
    for model in ADAPTIVE:
        r = adaptive_effects(adaptive, model)
        r["model_id"] = model
        rows.append(r)

    # Match the paper's display order.
    order = [
        "gpt-5.5", "gemini-3.1-pro-preview", "gpt-oss-120b",
        "deepseek-v4-flash", "llama-3.3-70b-instruct", "claude-opus-5",
    ]
    rows = sorted(rows, key=lambda x: order.index(x["model_id"]))
    for r in rows:
        r["model"] = DISPLAY[r["model_id"]]
        r["classification"] = classify(r)

    fields = [
        "model", "model_id", "n_prompts", "mean_replay_minus_live",
        "mean_post_minus_replay", "p_ikea_one_sided", "p_gogol_one_sided",
        "fresh_extension_seq_adjusted_p", "classification", "design",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.out}")
    for r in rows:
        print(
            f"{r['model']:<22} N={r['n_prompts']:>2}  "
            f"IKEA={r['mean_replay_minus_live']:+6.2f}  "
            f"Gogol={r['mean_post_minus_replay']:+6.2f}  "
            f"p_G={r['p_gogol_one_sided']:.4g}"
        )


if __name__ == "__main__":
    main()
