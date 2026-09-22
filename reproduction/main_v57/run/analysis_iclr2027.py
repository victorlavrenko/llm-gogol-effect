#!/usr/bin/env python3
"""Recompute the secondary ICLR-2027 analyses from frozen exported CSVs.

The confirmatory adaptive tests are read from adaptive_looks.csv exactly as
recorded by the experiment. All reliability/calibration analyses here are
secondary analyses on frozen model outputs; no generations are regenerated.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon, binomtest
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "derived"
OUT.mkdir(exist_ok=True)


def within_post_metrics(g: pd.DataFrame) -> dict[str, float]:
    rhos: list[float] = []
    concordant = 0
    comparable = 0
    top1 = 0
    top2 = 0
    nq = 0
    for _, q in g.groupby("question_id"):
        q = q[["live_score", "post_score"]].dropna()
        if len(q) < 2:
            continue
        x = q.live_score.to_numpy(float)
        y = q.post_score.to_numpy(float)
        nq += 1
        if len(q) >= 3 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            rhos.append(float(spearmanr(x, y).statistic))
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                dx = np.sign(x[i] - x[j])
                dy = np.sign(y[i] - y[j])
                if dx == 0 or dy == 0:
                    continue
                comparable += 1
                concordant += int(dx == dy)
        live_max = set(np.flatnonzero(x == np.max(x)).tolist())
        post_max = set(np.flatnonzero(y == np.max(y)).tolist())
        top1 += int(bool(live_max & post_max))
        k = min(2, len(y))
        thresh = np.partition(y, len(y) - k)[len(y) - k]
        post_top2 = set(np.flatnonzero(y >= thresh).tolist())
        top2 += int(bool(live_max & post_top2))
    pooled = g[["live_score", "post_score"]].dropna()
    return {
        "n_questions": nq,
        "n_sentences": len(pooled),
        "global_spearman": float(spearmanr(pooled.live_score, pooled.post_score).statistic),
        "mean_within_post_spearman": float(np.mean(rhos)),
        "pairwise_order_agreement": concordant / comparable,
        "live_max_is_post_max": top1 / nq,
        "live_max_in_post_top2": top2 / nq,
        "bias_live_minus_post": float((pooled.live_score - pooled.post_score).mean()),
        "mae_live_post": float((pooled.live_score - pooled.post_score).abs().mean()),
    }


def grouped_calibration(g: pd.DataFrame) -> dict[str, float]:
    d = g[["question_id", "live_score", "post_score"]].dropna().copy()
    X = d[["live_score"]].to_numpy(float)
    y = d.post_score.to_numpy(float)
    groups = d.question_id.to_numpy()
    k = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=k)
    pred_offset = np.empty(len(d), dtype=float)
    pred_affine = np.empty(len(d), dtype=float)
    for tr, te in splitter.split(X, y, groups):
        offset = float(np.mean(y[tr] - X[tr, 0]))
        pred_offset[te] = X[te, 0] + offset
        reg = LinearRegression().fit(X[tr], y[tr])
        pred_affine[te] = reg.predict(X[te])
    return {
        "raw_mae": float(np.mean(np.abs(X[:, 0] - y))),
        "groupcv_offset_mae": float(np.mean(np.abs(pred_offset - y))),
        "groupcv_affine_mae": float(np.mean(np.abs(pred_affine - y))),
    }


def rank_bootstrap(g: pd.DataFrame, seed: int = 20260908, B: int = 10000):
    qvals = []
    for qid, q in g.groupby("question_id"):
        q = q[["live_score", "post_score"]].dropna()
        if len(q) >= 3 and q.live_score.nunique() > 1 and q.post_score.nunique() > 1:
            qvals.append(float(spearmanr(q.live_score, q.post_score).statistic))
    vals = np.asarray(qvals)
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(vals, (B, len(vals)), replace=True), axis=1)
    return float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


paired = pd.read_csv(ROOT / "paired_scores.csv")
english = paired[~paired.question_id.str.startswith("ru")].copy()
metrics = []
calib = []
for model, g in english.groupby("model_id"):
    r = {"model_id": model, **within_post_metrics(g)}
    lo, hi = rank_bootstrap(g)
    r["within_post_spearman_ci_low"] = lo
    r["within_post_spearman_ci_high"] = hi
    metrics.append(r)
    calib.append({"model_id": model, **grouped_calibration(g)})
pd.DataFrame(metrics).sort_values("mean_within_post_spearman", ascending=False).to_csv(
    OUT / "model_reliability.csv", index=False
)
pd.DataFrame(calib).sort_values("raw_mae").to_csv(OUT / "grouped_calibration.csv", index=False)

# Expanded Replay: merge the replay extension with frozen live/post sentence scores.
rep = pd.read_csv(ROOT / "expanded_replay_scores.csv")[["model_id", "question_id", "sentence_index", "score"]]
rep = rep.rename(columns={"score": "expanded_replay"})
merged = paired.merge(rep, on=["model_id", "question_id", "sentence_index"], how="inner")
merged["language"] = np.where(merged.question_id.str.startswith("ru"), "ru", "en")
qmeans = merged.groupby(["language", "model_id", "question_id"], as_index=False)[
    ["live_score", "expanded_replay", "post_score"]
].mean()
qmeans["writer_critic"] = qmeans.expanded_replay - qmeans.live_score
qmeans["future_context"] = qmeans.post_score - qmeans.expanded_replay
qmeans["total_retrospective"] = qmeans.post_score - qmeans.live_score
qmeans.to_csv(OUT / "replay_question_means.csv", index=False)

replay_summary = qmeans.groupby(["language", "model_id"])[
    ["live_score", "expanded_replay", "post_score", "writer_critic", "future_context", "total_retrospective"]
].agg(["mean", "std", "count"])
replay_summary.to_csv(OUT / "replay_decomposition_summary.csv")

# English paired cross-model writer--critic contrast.
en = qmeans[qmeans.language == "en"]
wide = en.pivot(index="question_id", columns="model_id", values="writer_critic").dropna()
diff = wide["gemini-3.1-pro-preview"] - wide["gpt-5.5"]
se = diff.std(ddof=1) / math.sqrt(len(diff))
from scipy.stats import t as student_t
crit = student_t.ppf(.975, len(diff)-1)
contrast = pd.DataFrame([{
    "n": len(diff),
    "mean_difference_gemini_minus_gpt": diff.mean(),
    "ci95_low": diff.mean() - crit*se,
    "ci95_high": diff.mean() + crit*se,
    "paired_t_p_two_sided": ttest_rel(wide["gemini-3.1-pro-preview"], wide["gpt-5.5"]).pvalue,
}])
contrast.to_csv(OUT / "replay_writer_critic_crossmodel.csv", index=False)

# Correction intervention, English primary set only.
corr = pd.read_csv(ROOT / "correction_pairwise.csv")
corr = corr[~corr.question_id.str.startswith("ru")].copy()
# Positive advantage means guided reduces Post score more than random.
# Frozen export column names are retained; inspect and derive defensively.
rows = []
for model, g in corr.groupby("model_id"):
    # The export already includes the paired advantage as random minus guided corrected score change.
    if "guided_advantage_random_minus_guided" in g.columns:
        adv = g["guided_advantage_random_minus_guided"].dropna().to_numpy(float)
    elif "advantage_random_minus_guided" in g.columns:
        adv = g["advantage_random_minus_guided"].dropna().to_numpy(float)
    else:
        # Fall back to fields used by v5.7.
        cand = [c for c in g.columns if "advantage" in c.lower()]
        if not cand:
            raise RuntimeError(f"Cannot locate correction advantage column; columns={list(g.columns)}")
        adv = g[cand[0]].dropna().to_numpy(float)
    nonzero = adv[adv != 0]
    wins = int(np.sum(nonzero > 0))
    p_binom = binomtest(wins, len(nonzero), .5, alternative="greater").pvalue if len(nonzero) else np.nan
    p_w = wilcoxon(adv, alternative="greater", zero_method="wilcox").pvalue if np.any(adv != 0) else np.nan
    rows.append({
        "model_id": model,
        "n": len(adv),
        "mean_guided_advantage": float(np.mean(adv)),
        "median_guided_advantage": float(np.median(adv)),
        "non_tie_wins": wins,
        "non_tie_n": len(nonzero),
        "one_sided_binomial_p": p_binom,
        "one_sided_wilcoxon_p": p_w,
    })
pd.DataFrame(rows).to_csv(OUT / "correction_english.csv", index=False)

print("Wrote", OUT)
print(pd.read_csv(OUT / "model_reliability.csv").to_string(index=False))
