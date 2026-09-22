#!/usr/bin/env python3
"""Summarize existing Live-Replay-Post evidence from the frozen V5.7 DB."""
from __future__ import annotations
import argparse, csv, json, math, sqlite3
from pathlib import Path
import numpy as np
from scipy import stats

PILOT_IDS=("p001","p042","p044","p070")
MODELS=(
 "claude-opus-5","deepseek-v4-flash","gpt-oss-120b","llama-3.3-70b-instruct",
 "gpt-5.5","gemini-3.1-pro-preview"
)

def con(db):
 c=sqlite3.connect(db); c.row_factory=sqlite3.Row; return c

def mean(c,m,q,cond):
 r=c.execute("select avg(score) a,count(*) n from scores where model_id=? and question_id=? and condition=?",(m,q,cond)).fetchone();
 return (None,0) if not r or not r['n'] else (float(r['a']),int(r['n']))

def rep(c,m,q):
 try:
  r=c.execute("select avg(score) a,count(*) n from replay_extension_steps where model_id=? and question_id=? and status='complete'",(m,q)).fetchone()
  if r and r['n']: return float(r['a']),int(r['n'])
 except sqlite3.OperationalError: pass
 r=c.execute("select avg(score) a,count(*) n from replay_steps where model_id=? and question_id=? and status='complete'",(m,q)).fetchone()
 return (None,0) if not r or not r['n'] else (float(r['a']),int(r['n']))

def lang(c,q):
 r=c.execute('select metadata_json from questions where question_id=?',(q,)).fetchone(); return json.loads(r['metadata_json']).get('language')

def effects(c,m,qids):
 out=[]
 for q in qids:
  lv,nl=mean(c,m,q,'live'); po,np_=mean(c,m,q,'post'); rp,nr=rep(c,m,q)
  if lv is not None and po is not None and rp is not None and nl==nr:
   out.append((q,po-rp,rp-lv))
 return out

def p1(x):
 if len(x)<2:return math.nan
 return float(stats.ttest_1samp(np.array(x),0,alternative='greater').pvalue)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--db',type=Path,required=True); ap.add_argument('--out',type=Path,default=Path('gogol_existing_summary.csv')); a=ap.parse_args()
 c=con(a.db)
 eng=[r['question_id'] for r in c.execute('select question_id from questions order by question_id') if lang(c,r['question_id'])=='en']
 rows=[]
 for m in MODELS:
  qids=eng if m in ('gpt-5.5','gemini-3.1-pro-preview') else PILOT_IDS
  es=effects(c,m,qids); g=[x[1] for x in es]; i=[x[2] for x in es]
  rows.append({'model_id':m,'n_prompts':len(g),'mean_post_minus_replay':np.mean(g) if g else math.nan,'p_one_sided_gogol':p1(g),'mean_replay_minus_live':np.mean(i) if i else math.nan})
 with a.out.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 print(a.out)
 for r in rows: print(r)
if __name__=='__main__':main()
