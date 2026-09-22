#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sqlite3,statistics
from pathlib import Path
MODELS=("gpt-5.5","gemini-3.1-pro-preview")
TEMPS=("0.0","0.7")
BASE_SEED=20260917

def stable_seed(*parts):
    s='|'.join(map(str,parts)).encode()
    return BASE_SEED + int(hashlib.sha256(s).hexdigest()[:8],16)

def mean(x): return statistics.fmean(x) if x else None

def ranks(vals):
    order=sorted(enumerate(vals), key=lambda z:z[1]); out=[0.0]*len(vals); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and order[j][1]==order[i][1]: j+=1
        r=(i+1+j)/2.0
        for k in range(i,j): out[order[k][0]]=r
        i=j
    return out

def pearson(x,y):
    if len(x)!=len(y) or len(x)<2:return None
    mx=mean(x); my=mean(y); dx=[a-mx for a in x]; dy=[b-my for b in y]
    den=math.sqrt(sum(a*a for a in dx)*sum(b*b for b in dy))
    return None if den==0 else sum(a*b for a,b in zip(dx,dy))/den

def spearman(x,y): return pearson(ranks(x),ranks(y)) if len(x)>=2 else None

def boot(vals,seed,B=20000):
    vals=[float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if len(vals)<2:return None,None
    rng=random.Random(seed); out=[]; n=len(vals)
    for _ in range(B): out.append(mean([vals[rng.randrange(n)] for _ in range(n)]))
    out.sort(); return out[int(.025*B)],out[min(B-1,int(.975*B))]

def load_db(path,temp):
    con=sqlite3.connect(path); con.row_factory=sqlite3.Row
    rows=con.execute("""SELECT model_id,question_id,condition,sentence_index,score FROM scores
        WHERE model_id IN ('gpt-5.5','gemini-3.1-pro-preview') AND condition IN ('live','post','replay')
        ORDER BY model_id,question_id,condition,sentence_index""").fetchall(); con.close()
    by={}
    for r in rows:
        by.setdefault((r['model_id'],r['question_id']),{}).setdefault(r['condition'],[]).append((int(r['sentence_index']),float(r['score'])))
    out=[]
    for (m,q),c in sorted(by.items()):
        if not all(k in c for k in ('live','post','replay')): continue
        lv=[v for _,v in sorted(c['live'])]; po=[v for _,v in sorted(c['post'])]; rp=[v for _,v in sorted(c['replay'])]
        if not (len(lv)==len(po)==len(rp)): continue
        out.append(dict(temperature=temp,model_id=m,question_id=q,n_sentences=len(lv),live_mean=mean(lv),replay_mean=mean(rp),post_mean=mean(po),
                        replay_minus_live=mean(rp)-mean(lv),post_minus_replay=mean(po)-mean(rp),post_minus_live=mean(po)-mean(lv),live_post_spearman=spearman(lv,po)))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--temp0',type=Path,required=True); ap.add_argument('--temp07',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    rows=load_db(a.temp0/'experiment.sqlite3','0.0')+load_db(a.temp07/'experiment.sqlite3','0.7')
    fields=list(rows[0]);
    with (a.out/'question_metrics.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary=[]
    for t in TEMPS:
      for m in MODELS:
        s=[r for r in rows if r['temperature']==t and r['model_id']==m]
        for metric in ('post_minus_live','replay_minus_live','post_minus_replay','live_post_spearman'):
            vals=[r[metric] for r in s if r[metric] is not None]; lo,hi=boot(vals,stable_seed('summary',t,m,metric))
            summary.append(dict(temperature=t,model_id=m,metric=metric,n_prompts=len(vals),mean=mean(vals),bootstrap_lo=lo,bootstrap_hi=hi))
    with (a.out/'summary.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    paired=[]
    for m in MODELS:
      d0={r['question_id']:r for r in rows if r['temperature']=='0.0' and r['model_id']==m}; d7={r['question_id']:r for r in rows if r['temperature']=='0.7' and r['model_id']==m}; qs=sorted(d0.keys()&d7.keys())
      for metric in ('post_minus_live','replay_minus_live','post_minus_replay','live_post_spearman'):
        vals=[d0[q][metric]-d7[q][metric] for q in qs if d0[q][metric] is not None and d7[q][metric] is not None]; lo,hi=boot(vals,stable_seed('paired-temp',m,metric))
        paired.append(dict(model_id=m,metric=metric,n_paired=len(vals),mean_temp0_minus_temp07=mean(vals),bootstrap_lo=lo,bootstrap_hi=hi))
    with (a.out/'paired_temperature_difference.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    contrasts=[]
    for t in TEMPS:
        g={r['question_id']:r for r in rows if r['temperature']==t and r['model_id']=='gpt-5.5'}; m={r['question_id']:r for r in rows if r['temperature']==t and r['model_id']=='gemini-3.1-pro-preview'}; qs=sorted(g.keys()&m.keys())
        for metric in ('post_minus_live','replay_minus_live','post_minus_replay','live_post_spearman'):
            vals=[m[q][metric]-g[q][metric] for q in qs if g[q][metric] is not None and m[q][metric] is not None]; lo,hi=boot(vals,stable_seed('model-contrast',t,metric))
            contrasts.append(dict(temperature=t,metric=metric,n_paired=len(vals),mean_gemini_minus_gpt=mean(vals),bootstrap_lo=lo,bootstrap_hi=hi))
    with (a.out/'model_contrasts.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(contrasts[0])); w.writeheader(); w.writerows(contrasts)
    payload={'summary':summary,'paired_temperature_difference':paired,'model_contrasts':contrasts,'bootstrap':'prompt-level percentile bootstrap, B=20000, deterministic SHA-256-derived seeds'}
    (a.out/'summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
