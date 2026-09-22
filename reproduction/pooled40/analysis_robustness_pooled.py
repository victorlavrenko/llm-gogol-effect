#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, statistics
from pathlib import Path

MODELS=("gpt-5.5","gemini-3.1-pro-preview")
VARIANTS=("original","paraphrase","reverse","minimal")
BASE_SEED=20260910
B=10000

def stable_seed(*parts:str)->int:
    h=hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return (BASE_SEED + int(h[:8],16)) % (2**31-1)

def f(x):
    if x in (None,"","None","nan","NaN"): return None
    return float(x)

def read_metrics(path:Path, cohort:str):
    rows=[]
    with path.open(encoding="utf-8",newline="") as fh:
        for r in csv.DictReader(fh):
            d=dict(r); d["cohort"]=cohort
            for k in ("live_mean","post_mean","post_minus_live","live_post_spearman","live_post_pairwise_order"):
                if k in d: d[k]=f(d[k])
            rows.append(d)
    return rows

def mean(vals):
    vals=[float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return statistics.fmean(vals) if vals else None

def boot(vals, seed, B=B):
    vals=[float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if len(vals)<2: return None,None
    rng=random.Random(seed); n=len(vals); out=[]
    for _ in range(B):
        out.append(statistics.fmean(vals[rng.randrange(n)] for _ in range(n)))
    out.sort()
    return out[int(.025*B)],out[min(B-1,int(.975*B))]

def cohort_rows(rows, cohort):
    return rows if cohort=="pooled40" else [r for r in rows if r["cohort"]==cohort]

def variant_summary(rows, cohort):
    use=cohort_rows(rows,cohort); out=[]
    for m in MODELS:
        for v in VARIANTS:
            s=[r for r in use if r["model_id"]==m and r["variant_id"]==v]
            shifts=[r["post_minus_live"] for r in s]
            rhos=[r["live_post_spearman"] for r in s if r["live_post_spearman"] is not None]
            lo,hi=boot(shifts,stable_seed("variant",cohort,m,v))
            out.append({"cohort":cohort,"model_id":m,"variant_id":v,"n_questions":len(s),
                        "mean_post_minus_live":mean(shifts),"shift_ci_lo":lo,"shift_ci_hi":hi,
                        "mean_live_post_spearman":mean(rhos)})
    return out

def model_contrast(rows, cohort):
    use=cohort_rows(rows,cohort); by={}
    for r in use:
        key=(r["cohort"],r["question_id"],r["model_id"])
        by.setdefault(key,[]).append(r["post_minus_live"])
    diffs=[]
    for c,q in sorted({(r["cohort"],r["question_id"]) for r in use}):
        g=by.get((c,q,"gpt-5.5"),[]); m=by.get((c,q,"gemini-3.1-pro-preview"),[])
        if len(g)==4 and len(m)==4: diffs.append(statistics.fmean(m)-statistics.fmean(g))
    lo,hi=boot(diffs,BASE_SEED+777)
    return {"cohort":cohort,"n_matched_prompts":len(diffs),
            "mean_gemini_minus_gpt_post_live_shift":mean(diffs),"ci_lo":lo,"ci_hi":hi}

def per_variant_contrast(rows, cohort="pooled40"):
    use=cohort_rows(rows,cohort); out=[]
    for v in VARIANTS:
        diffs=[]
        for c,q in sorted({(r["cohort"],r["question_id"]) for r in use if r["variant_id"]==v}):
            vals={r["model_id"]:r["post_minus_live"] for r in use if r["cohort"]==c and r["question_id"]==q and r["variant_id"]==v}
            if all(m in vals for m in MODELS): diffs.append(vals["gemini-3.1-pro-preview"]-vals["gpt-5.5"])
        lo,hi=boot(diffs,stable_seed("contrast",cohort,v))
        out.append({"cohort":cohort,"variant_id":v,"n_matched_prompts":len(diffs),
                    "mean_gemini_minus_gpt_post_live_shift":mean(diffs),"ci_lo":lo,"ci_hi":hi})
    return out

def write_csv(path, rows):
    with path.open("w",encoding="utf-8",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--initial-run",type=Path,required=True)
    p.add_argument("--extension-run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    rows=read_metrics(a.initial_run/"question_metrics.csv","initial16")+read_metrics(a.extension_run/"question_metrics.csv","extension24")
    summaries=[]; contrasts=[]
    for cohort in ("initial16","extension24","pooled40"):
        summaries.extend(variant_summary(rows,cohort)); contrasts.append(model_contrast(rows,cohort))
    pv=per_variant_contrast(rows,"pooled40")
    write_csv(a.out/"cohort_variant_summary.csv",summaries)
    write_csv(a.out/"cohort_model_contrast.csv",contrasts)
    write_csv(a.out/"pooled_variant_model_contrasts.csv",pv)
    payload={"bootstrap_replicates":B,"bootstrap_unit":"prompt","variant_summaries":summaries,"matched_model_contrasts":contrasts,"pooled_variant_model_contrasts":pv,
             "reporting_note":"The 24-prompt extension was frozen after observing the initial 16-prompt robustness cohort; initial16, extension24, and pooled40 are therefore reported separately."}
    (a.out/"robustness_pooled40_summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
