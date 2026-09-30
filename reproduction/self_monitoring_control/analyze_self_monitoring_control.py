#!/usr/bin/env python3
"""Analyze the prospective self-monitoring control experiment."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent


def bootstrap_ci(values, *, seed: int, n_boot: int = 20000):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def one_sided_greater(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        if len(x) and x[0] > 0:
            return 0.0
        return 1.0
    return float(stats.ttest_1samp(x, 0.0, alternative="greater").pvalue)


def two_sided(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0 if len(x) and not np.isclose(x[0], 0) else 1.0
    return float(stats.ttest_1samp(x, 0.0).pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "run_full")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()

    db = args.run / "experiment.sqlite3"
    if not db.exists():
        raise SystemExit(f"missing {db}")

    con = sqlite3.connect(db)
    gen = pd.read_sql_query(
        """
        SELECT model_id,question_id,arm,status,units_json,live_scores_json,
               segmentation_method,provider,error
        FROM generations
        """, con
    )
    ev = pd.read_sql_query(
        """
        SELECT model_id,question_id,arm,kind,sentence_index,score,status,error
        FROM evaluations
        WHERE sentence_index >= 0
        """, con
    )
    con.close()

    rows = []
    for _, g in gen[gen.status == "complete"].iterrows():
        units = json.loads(g.units_json)
        n = len(units)
        sub = ev[
            (ev.model_id == g.model_id)
            & (ev.question_id == g.question_id)
            & (ev.arm == g.arm)
            & (ev.status == "complete")
        ]
        pre = sub[sub.kind == "prefix"].sort_values("sentence_index")
        post = sub[sub.kind == "post"].sort_values("sentence_index")
        if len(pre) != n or len(post) != n:
            continue
        if list(pre.sentence_index) != list(range(n)) or list(post.sentence_index) != list(range(n)):
            continue
        live = json.loads(g.live_scores_json) if isinstance(g.live_scores_json, str) else None
        rows.append({
            "model_id": g.model_id,
            "question_id": g.question_id,
            "arm": g.arm,
            "n_sentences": n,
            "prefix_mean": float(pre.score.mean()),
            "post_mean": float(post.score.mean()),
            "completion_shift": float(post.score.mean() - pre.score.mean()),
            "live_mean": float(np.mean(live)) if live else np.nan,
            "provider": g.provider,
            "segmentation_method": g.segmentation_method,
        })

    qm = pd.DataFrame(rows)
    if qm.empty:
        raise SystemExit("no complete prompt-level cells")
    qm.to_csv(args.run / "question_metrics.csv", index=False)

    summary_rows = []
    model_ids = sorted(qm.model_id.unique())
    for mi, model_id in enumerate(model_ids):
        for arm in ("plain", "scored"):
            x = qm[(qm.model_id == model_id) & (qm.arm == arm)].completion_shift.to_numpy()
            lo, hi = bootstrap_ci(x, seed=args.seed + mi * 10 + (arm == "scored"), n_boot=args.bootstrap)
            p = one_sided_greater(x)
            summary_rows.append({
                "model_id": model_id,
                "estimand": f"completion_shift_{arm}",
                "n_prompts": len(x),
                "mean": float(np.mean(x)) if len(x) else np.nan,
                "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "ci95_low": lo,
                "ci95_high": hi,
                "p_one_sided_gt0": p,
                "p_two_sided": np.nan,
            })

    wide = qm.pivot_table(
        index=["model_id","question_id"],
        columns="arm",
        values=["completion_shift","post_mean","prefix_mean"],
        aggfunc="first",
    ).dropna()
    contrast_rows = []
    paired_rows = []
    for mi, model_id in enumerate(model_ids):
        if model_id not in wide.index.get_level_values("model_id"):
            continue
        w = wide.xs(model_id, level="model_id")
        interaction = (
            w[("completion_shift","scored")] - w[("completion_shift","plain")]
        )
        post_artifact = w[("post_mean","scored")] - w[("post_mean","plain")]
        prefix_artifact = w[("prefix_mean","scored")] - w[("prefix_mean","plain")]
        for qid in w.index:
            paired_rows.append({
                "model_id": model_id,
                "question_id": qid,
                "completion_plain": w.loc[qid, ("completion_shift","plain")],
                "completion_scored": w.loc[qid, ("completion_shift","scored")],
                "interaction_scored_minus_plain": interaction.loc[qid],
                "post_scored_minus_plain": post_artifact.loc[qid],
                "prefix_scored_minus_plain": prefix_artifact.loc[qid],
            })
        for j, (name, x) in enumerate([
            ("interaction_scored_minus_plain", interaction.to_numpy()),
            ("post_artifact_scored_minus_plain", post_artifact.to_numpy()),
            ("prefix_artifact_scored_minus_plain", prefix_artifact.to_numpy()),
        ]):
            lo, hi = bootstrap_ci(x, seed=args.seed + 100 + mi * 10 + j, n_boot=args.bootstrap)
            contrast_rows.append({
                "model_id": model_id,
                "estimand": name,
                "n_prompts": len(x),
                "mean": float(np.mean(x)),
                "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "ci95_low": lo,
                "ci95_high": hi,
                "p_one_sided_gt0": np.nan,
                "p_two_sided": two_sided(x),
            })

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.run / "paired_prompt_contrasts.csv", index=False)
    summary = pd.DataFrame(summary_rows + contrast_rows)

    summary["p_h1_bonferroni_2models"] = np.nan
    mask = summary.estimand == "completion_shift_plain"
    summary.loc[mask, "p_h1_bonferroni_2models"] = np.minimum(
        1.0, summary.loc[mask, "p_one_sided_gt0"].astype(float) * max(1, mask.sum())
    )
    summary.to_csv(args.run / "model_summary.csv", index=False)

    machine = {
        "analysis_version": "self-monitoring-control-analysis-v1",
        "primary_interpretation": {
            "plain_completion_positive": (
                "POST > PREFIX even when generation never saw a self-scoring instruction; "
                "supports a completion-associated effect not requiring online self-monitoring."
            ),
            "interaction_positive": (
                "completion shift is larger after scored generation; online self-monitoring "
                "may amplify the completion-associated shift."
            ),
            "interaction_near_zero": (
                "completion shift is similar across generation arms; online self-monitoring "
                "does not appear necessary to produce the shift."
            ),
        },
        "n_complete_prompt_arm_cells": int(len(qm)),
        "models": model_ids,
    }
    (args.run / "analysis_summary.json").write_text(
        json.dumps(machine, indent=2), encoding="utf-8"
    )

    print("\nMODEL SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nWrote {args.run / 'question_metrics.csv'}")
    print(f"Wrote {args.run / 'paired_prompt_contrasts.csv'}")
    print(f"Wrote {args.run / 'model_summary.csv'}")


if __name__ == "__main__":
    main()
