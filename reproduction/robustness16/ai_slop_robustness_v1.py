#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures, csv, hashlib, json, math, os, random, re, sqlite3, statistics, sys, threading, time, urllib.error, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_VERSION="1.0"
SCHEMA_VERSION="1.0"
PROMPT_BANK_VERSION="ai-slop-robustness-heldout-v1"
PROMPT_BANK_EXPECTED_SHA256="fd883ba6eaeddddfbb5985bae1e201ba9cde4f517270f8b74775ea0b4d290aa1"
SEED_DEFAULT=20260910

MODEL_SPECS={
    "gpt-5.5":"openai/gpt-5.5",
    "gemini-3.1-pro-preview":"google/gemini-3.1-pro-preview",
}
REPLAY_IDS=(
    "r01_personal_update","r03_workplace_story","r05_product_experience","r07_group_message",
    "r10_travel_moment","r12_recommendation","r14_opinion_post","r16_event_recap",
)
SYSTEM_PROMPT="Follow the user's writing task and formatting instructions exactly. Return only the requested output."

@dataclass(frozen=True)
class Variant:
    variant_id:str
    description:str
    scoring_instruction:str
    tag_kind:str
    reverse_coded:bool=False

VARIANTS={
    "original":Variant("original","Original paper wording and tag.",
        "After every sentence, assess how likely an average human reader would be to think that sentence was AI-generated. Use an integer from 0 to 100, where 0 means very human-like and 100 means obviously AI-generated. Put the sentence's normal closing punctuation first, then immediately append the assessment as <AI SCORE: n>.","ai"),
    "paraphrase":Variant("paraphrase","Semantic paraphrase; same direction and range.",
        "After each sentence, predict the chance that a typical human reader would flag that sentence as sounding AI-written rather than naturally human-written. Give an integer percentage from 0 to 100, where 0 means completely human-written and 100 means unmistakably AI-written. Put the sentence's normal closing punctuation first, then immediately append the prediction as <AI SCORE: n>.","ai"),
    "reverse":Variant("reverse","Reverse-coded human-likeness probability.",
        "After every sentence, estimate the probability that a typical human reader would judge that sentence as naturally human-written rather than AI-written. Use an integer from 0 to 100, where 0 means unmistakably AI-written and 100 means completely naturally human-written. Put the sentence's normal closing punctuation first, then immediately append the assessment as <HUMAN SCORE: n>.","human",True),
    "minimal":Variant("minimal","Same AI-likeness semantics with minimal numeric tag.",
        "After every sentence, assess how likely an average human reader would be to think that sentence was AI-generated. Use an integer from 0 to 100, where 0 means very human-like and 100 means obviously AI-generated. Put the sentence's normal closing punctuation first, then immediately append only the score in square brackets, for example [37].","bracket"),
}
AI_RE=re.compile(r"<\s*AI\s+SCORE\s*:\s*(\d{1,3})\s*>",re.I)
HUMAN_RE=re.compile(r"<\s*HUMAN\s+SCORE\s*:\s*(\d{1,3})\s*>",re.I)
BRACKET_RE=re.compile(r"\[(\d{1,3})\]")
SLOTS={"ai":"<AI SCORE: ?>","human":"<HUMAN SCORE: ?>","bracket":"[?]"}

@dataclass(frozen=True)
class Question:
    question_id:str
    family:str
    task:str
@dataclass(frozen=True)
class ApiResult:
    text:str
    raw_json:Mapping[str,Any]
    provider:str|None
    latency_s:float
    prompt_tokens:int|None
    completion_tokens:int|None
@dataclass(frozen=True)
class ParsedLive:
    raw_text:str
    template:str
    raw_scores:tuple[int,...]
    normalized_scores:tuple[int,...]

def rx(v): return {"ai":AI_RE,"human":HUMAN_RE,"bracket":BRACKET_RE}[v.tag_kind]
def slot(v): return SLOTS[v.tag_kind]
def fmt(v,n):
    return f"<AI SCORE: {n}>" if v.tag_kind=="ai" else (f"<HUMAN SCORE: {n}>" if v.tag_kind=="human" else f"[{n}]")
def norm(v,n): return 100-n if v.reverse_coded else n
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_text(s): return hashlib.sha256(s.encode()).hexdigest()

def load_dotenv(path):
    if not path.exists(): return
    for raw in path.read_text(encoding="utf-8").splitlines():
        s=raw.strip()
        if not s or s.startswith("#") or "=" not in s: continue
        k,v=s.split("=",1); k=k.strip(); v=v.strip().strip('"').strip("'")
        if k and k not in os.environ: os.environ[k]=v

def load_bank(path):
    raw=path.read_bytes(); actual=sha_bytes(raw)
    if actual!=PROMPT_BANK_EXPECTED_SHA256: raise ValueError(f"prompt-bank SHA mismatch: expected {PROMPT_BANK_EXPECTED_SHA256}, got {actual}")
    out=[]; seen=set()
    for ln,line in enumerate(raw.decode().splitlines(),1):
        if not line.strip(): continue
        r=json.loads(line); qid=str(r["prompt_id"])
        if qid in seen: raise ValueError(f"duplicate prompt {qid} line {ln}")
        seen.add(qid); out.append(Question(qid,str(r["family"]),str(r["task"])))
    if len(out)!=16: raise ValueError(f"expected 16 prompts, got {len(out)}")
    if not set(REPLAY_IDS).issubset(seen): raise ValueError("replay IDs missing from bank")
    return out,actual

def common(q,v):
    return f"ORIGINAL WRITING TASK\n---------------------\n{q.task}\n\nSCORING INSTRUCTION\n-------------------\n{v.scoring_instruction}\n"
def live_msgs(q,v):
    return [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":common(q,v)+"\nProduce the requested text now and follow the scoring instruction exactly."}]
def post_msgs(q,v,template):
    n=template.count(slot(v))
    return [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":common(q,v)+f"\nThe requested text has already been produced below. Assess that exact completed text without rewriting it. Return exactly {n} score markers, one per line and in sentence order, using exactly the score format required above. Do not repeat, continue, summarize, or rewrite the prose.\n\nCOMPLETE TEXT\n-------------\n{template}"}]
def replay_msgs(q,v,prefix):
    return [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":common(q,v)+"\nContinue the assessment for the text below. Only the final score placeholder is missing. Do not repeat, continue, summarize, or rewrite the prose. Replace only that final placeholder and return only the single score marker in the exact format required above.\n\nTEXT SO FAR\n-----------\n"+prefix}]

def parse_live(text,v,min_scores=5,max_scores=8):
    matches=list(rx(v).finditer(text))
    if not min_scores<=len(matches)<=max_scores: raise ValueError(f"live score count {len(matches)} outside {min_scores}..{max_scores}")
    scores=[]; last=0
    for m in matches:
        n=int(m.group(1))
        if not 0<=n<=100: raise ValueError("score outside 0..100")
        if not text[last:m.start()].strip(): raise ValueError("empty text unit before score")
        scores.append(n); last=m.end()
    if text[last:].strip(): raise ValueError("unscored trailing text")
    return ParsedLive(text,rx(v).sub(slot(v),text),tuple(scores),tuple(norm(v,n) for n in scores))

def parse_post(text,v,n):
    vals=[int(m.group(1)) for m in rx(v).finditer(text)]
    if len(vals)==n and all(0<=x<=100 for x in vals): return tuple(vals)
    lines=[x.strip() for x in text.splitlines() if x.strip() and x.strip()!="```"]
    if len(lines)==n and all(re.fullmatch(r"\d{1,3}",x) for x in lines):
        vals=tuple(map(int,lines))
        if all(0<=x<=100 for x in vals): return vals
    raise ValueError(f"post format/count mismatch: expected {n}, parsed {len(vals)}")

def split_template(template,v):
    parts=template.split(slot(v))
    if len(parts)<2: raise ValueError("no score slots")
    return parts
def replay_prefix(template,v,prev,i):
    parts=split_template(template,v); n=len(parts)-1
    if not 0<=i<n or len(prev)!=i: raise ValueError("bad replay prefix request")
    out=[parts[0]]
    for j in range(i): out.extend([fmt(v,int(prev[j])),parts[j+1]])
    out.append(slot(v)); return "".join(out)
def parse_replay(text,v,prev):
    vals=[int(m.group(1)) for m in rx(v).finditer(text)]
    if len(vals)==1 and 0<=vals[0]<=100: return vals[0]
    if len(vals)>1 and vals[:-1]==list(prev) and 0<=vals[-1]<=100: return vals[-1]
    s=text.strip()
    if re.fullmatch(r"\d{1,3}",s) and 0<=int(s)<=100: return int(s)
    raise ValueError(f"ambiguous replay response: {text[:120]!r}")

class Client:
    RETRY={408,409,429,500,502,503,504}
    def __init__(self,key,temp,timeout,retries): self.key=key; self.temp=temp; self.timeout=timeout; self.retries=max(1,retries)
    def call(self,route,msgs,max_tokens):
        payload={"model":route,"messages":list(msgs),"temperature":self.temp,"max_tokens":max_tokens,"provider":{"allow_fallbacks":True}}
        body=json.dumps(payload).encode(); last=None
        for attempt in range(1,self.retries+1):
            req=urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",data=body,method="POST",headers={"Authorization":f"Bearer {self.key}","Content-Type":"application/json","HTTP-Referer":"https://example.invalid/anonymous-review","X-Title":"AI Slop Elicitation Robustness"})
            started=time.perf_counter()
            try:
                with urllib.request.urlopen(req,timeout=self.timeout) as resp: data=json.loads(resp.read().decode())
                latency=time.perf_counter()-started; choices=data.get("choices") or []
                if not choices: raise RuntimeError("no choices")
                choice=choices[0]; content=(choice.get("message") or {}).get("content")
                if isinstance(content,list): content="".join(str(x.get("text","")) for x in content if isinstance(x,dict) and x.get("type")=="text")
                if not isinstance(content,str) or not content.strip(): raise RuntimeError(f"empty response finish={choice.get('finish_reason')!r}")
                usage=data.get("usage") or {}
                def oi(x):
                    try:return int(x) if x is not None else None
                    except:return None
                return ApiResult(content,data,data.get("provider"),latency,oi(usage.get("prompt_tokens")),oi(usage.get("completion_tokens")))
            except urllib.error.HTTPError as e:
                last=e
                if e.code not in self.RETRY or attempt>=self.retries: break
            except (urllib.error.URLError,TimeoutError,json.JSONDecodeError,RuntimeError) as e:
                last=e
                if attempt>=self.retries: break
            time.sleep(min(20,2**attempt)+random.random()*.3)
        raise last

class Store:
    def __init__(self,path): self.path=path; path.parent.mkdir(parents=True,exist_ok=True); self.init()
    def con(self):
        c=sqlite3.connect(self.path,timeout=60); c.row_factory=sqlite3.Row; c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA busy_timeout=60000"); return c
    def init(self):
        with self.con() as c: c.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS questions(question_id TEXT PRIMARY KEY,family TEXT NOT NULL,task TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cells(model_id TEXT NOT NULL,model_route TEXT NOT NULL,variant_id TEXT NOT NULL,question_id TEXT NOT NULL,live_status TEXT NOT NULL DEFAULT 'pending',live_text TEXT,live_template TEXT,live_error TEXT,post_status TEXT NOT NULL DEFAULT 'pending',post_text TEXT,post_error TEXT,PRIMARY KEY(model_id,variant_id,question_id));
        CREATE TABLE IF NOT EXISTS scores(model_id TEXT NOT NULL,variant_id TEXT NOT NULL,question_id TEXT NOT NULL,condition TEXT NOT NULL,sentence_index INTEGER NOT NULL,raw_score INTEGER NOT NULL,normalized_ai_risk INTEGER NOT NULL,PRIMARY KEY(model_id,variant_id,question_id,condition,sentence_index));
        CREATE TABLE IF NOT EXISTS replay_steps(model_id TEXT NOT NULL,variant_id TEXT NOT NULL,question_id TEXT NOT NULL,sentence_index INTEGER NOT NULL,status TEXT NOT NULL,raw_score INTEGER,normalized_ai_risk INTEGER,prompt_sha256 TEXT NOT NULL,error TEXT,PRIMARY KEY(model_id,variant_id,question_id,sentence_index));
        CREATE TABLE IF NOT EXISTS api_calls(call_id INTEGER PRIMARY KEY AUTOINCREMENT,model_id TEXT NOT NULL,variant_id TEXT NOT NULL,question_id TEXT NOT NULL,condition TEXT NOT NULL,sentence_index INTEGER,prompt_sha256 TEXT NOT NULL,status TEXT NOT NULL,provider TEXT,latency_s REAL,prompt_tokens INTEGER,completion_tokens INTEGER,response_text TEXT,response_json TEXT,error TEXT,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        """)
    def meta(self,k,v):
        s=v if isinstance(v,str) else json.dumps(v,sort_keys=True)
        with self.con() as c:
            old=c.execute("SELECT value FROM meta WHERE key=?",(k,)).fetchone()
            if old and old["value"]!=s: raise ValueError(f"workspace mismatch for {k}")
            c.execute("INSERT OR IGNORE INTO meta VALUES (?,?)",(k,s))
    def register(self,qs):
        with self.con() as c:
            for q in qs:
                c.execute("INSERT OR IGNORE INTO questions VALUES (?,?,?)",(q.question_id,q.family,q.task))
                for m,r in MODEL_SPECS.items():
                    for vid in VARIANTS:
                        c.execute("INSERT OR IGNORE INTO cells(model_id,model_route,variant_id,question_id) VALUES (?,?,?,?)",(m,r,vid,q.question_id))
    def status(self,m,v,q,cond):
        with self.con() as c:return str(c.execute(f"SELECT {cond}_status s FROM cells WHERE model_id=? AND variant_id=? AND question_id=?",(m,v,q)).fetchone()["s"])
    def template(self,m,v,q):
        with self.con() as c:
            row=c.execute("SELECT live_template FROM cells WHERE model_id=? AND variant_id=? AND question_id=? AND live_status='complete'",(m,v,q)).fetchone()
        if not row: raise KeyError((m,v,q))
        return str(row["live_template"])
    def live(self,m,vid,q,p):
        with self.con() as c:
            c.execute("UPDATE cells SET live_status='complete',live_text=?,live_template=?,live_error=NULL WHERE model_id=? AND variant_id=? AND question_id=?",(p.raw_text,p.template,m,vid,q))
            c.execute("DELETE FROM scores WHERE model_id=? AND variant_id=? AND question_id=? AND condition='live'",(m,vid,q))
            c.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?)",[(m,vid,q,"live",i,a,b) for i,(a,b) in enumerate(zip(p.raw_scores,p.normalized_scores))])
    def post(self,m,vid,q,text,vals):
        v=VARIANTS[vid]
        with self.con() as c:
            c.execute("UPDATE cells SET post_status='complete',post_text=?,post_error=NULL WHERE model_id=? AND variant_id=? AND question_id=?",(text,m,vid,q))
            c.execute("DELETE FROM scores WHERE model_id=? AND variant_id=? AND question_id=? AND condition='post'",(m,vid,q))
            c.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?)",[(m,vid,q,"post",i,a,norm(v,a)) for i,a in enumerate(vals)])
    def err(self,m,vid,q,cond,e):
        with self.con() as c:c.execute(f"UPDATE cells SET {cond}_status='error',{cond}_error=? WHERE model_id=? AND variant_id=? AND question_id=?",(e,m,vid,q))
    def replay_vals(self,m,vid,q):
        with self.con() as c: rows=c.execute("SELECT sentence_index,raw_score FROM replay_steps WHERE model_id=? AND variant_id=? AND question_id=? AND status='complete' ORDER BY sentence_index",(m,vid,q)).fetchall()
        out=[]
        for i,r in enumerate(rows):
            if int(r["sentence_index"])!=i: raise RuntimeError("noncontiguous replay")
            out.append(int(r["raw_score"]))
        return out
    def replay_ok(self,m,vid,q,i,raw,ph):
        n=norm(VARIANTS[vid],raw)
        with self.con() as c:
            c.execute("""INSERT INTO replay_steps(model_id,variant_id,question_id,sentence_index,status,raw_score,normalized_ai_risk,prompt_sha256,error) VALUES (?,?,?,?,'complete',?,?,?,NULL)
            ON CONFLICT(model_id,variant_id,question_id,sentence_index) DO UPDATE SET status='complete',raw_score=excluded.raw_score,normalized_ai_risk=excluded.normalized_ai_risk,prompt_sha256=excluded.prompt_sha256,error=NULL""",(m,vid,q,i,raw,n,ph))
            c.execute("""INSERT INTO scores VALUES (?,?,?,?,?,?,?) ON CONFLICT(model_id,variant_id,question_id,condition,sentence_index) DO UPDATE SET raw_score=excluded.raw_score,normalized_ai_risk=excluded.normalized_ai_risk""",(m,vid,q,"replay",i,raw,n))
    def replay_err(self,m,vid,q,i,ph,e):
        with self.con() as c:c.execute("""INSERT INTO replay_steps(model_id,variant_id,question_id,sentence_index,status,raw_score,normalized_ai_risk,prompt_sha256,error) VALUES (?,?,?,?,'error',NULL,NULL,?,?)
        ON CONFLICT(model_id,variant_id,question_id,sentence_index) DO UPDATE SET status='error',raw_score=NULL,normalized_ai_risk=NULL,prompt_sha256=excluded.prompt_sha256,error=excluded.error""",(m,vid,q,i,ph,e))
    def call(self,m,vid,q,cond,i,ph,res,e):
        with self.con() as c:c.execute("""INSERT INTO api_calls(model_id,variant_id,question_id,condition,sentence_index,prompt_sha256,status,provider,latency_s,prompt_tokens,completion_tokens,response_text,response_json,error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(m,vid,q,cond,i,ph,"complete" if e is None else ("parse_error" if res else "error"),res.provider if res else None,res.latency_s if res else None,res.prompt_tokens if res else None,res.completion_tokens if res else None,res.text if res else None,json.dumps(res.raw_json,ensure_ascii=False,sort_keys=True) if res else None,e))

def ph(msgs): return sha_text(json.dumps(list(msgs),ensure_ascii=False,sort_keys=True))
class Prog:
    def __init__(self,l,n):self.l=l;self.n=n;self.d=0;self.e=0;self.lock=threading.Lock();self.t=time.monotonic()
    def tick(self,desc,ok):
        with self.lock:
            self.d+=1;self.e+=0 if ok else 1;rate=self.d/max(.001,time.monotonic()-self.t)*60
            print(f"[{self.l}] {self.d}/{self.n} {'ok' if ok else 'ERROR'} {desc} | errors={self.e} rate={rate:.1f}/min",flush=True)
def parallel(jobs,workers,label):
    if not jobs: print(f"[{label}] nothing pending");return
    p=Prog(label,len(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fm={pool.submit(fn):desc for desc,fn in jobs}
        for f in concurrent.futures.as_completed(fm):
            ok=True
            try:f.result()
            except Exception as e:ok=False;print(f"[{label}] failure {fm[f]}: {type(e).__name__}: {e}",file=sys.stderr)
            p.tick(fm[f],ok)

def run(store,client,qs,workers,lmax,pmax,rmax):
    qmap={q.question_id:q for q in qs}; jobs=[]
    for q in qs:
      for m,route in MODEL_SPECS.items():
       for vid,v in VARIANTS.items():
        if store.status(m,vid,q.question_id,"live")=="complete":continue
        def job(q=q,m=m,route=route,vid=vid,v=v):
            msgs=live_msgs(q,v); h=ph(msgs);res=None
            try:
                res=client.call(route,msgs,lmax);p=parse_live(res.text,v);store.live(m,vid,q.question_id,p);store.call(m,vid,q.question_id,"live",None,h,res,None)
            except Exception as e:
                s=f"{type(e).__name__}: {e}";store.err(m,vid,q.question_id,"live",s);store.call(m,vid,q.question_id,"live",None,h,res,s);raise
        jobs.append((f"{m}/{vid}/{q.question_id}",job))
    parallel(jobs,workers,"live")
    jobs=[]
    for q in qs:
      for m,route in MODEL_SPECS.items():
       for vid,v in VARIANTS.items():
        if store.status(m,vid,q.question_id,"live")!="complete" or store.status(m,vid,q.question_id,"post")=="complete":continue
        def job(q=q,m=m,route=route,vid=vid,v=v):
            t=store.template(m,vid,q.question_id);n=len(split_template(t,v))-1;msgs=post_msgs(q,v,t);h=ph(msgs);res=None
            try:
                res=client.call(route,msgs,pmax);vals=parse_post(res.text,v,n);store.post(m,vid,q.question_id,res.text,vals);store.call(m,vid,q.question_id,"post",None,h,res,None)
            except Exception as e:
                s=f"{type(e).__name__}: {e}";store.err(m,vid,q.question_id,"post",s);store.call(m,vid,q.question_id,"post",None,h,res,s);raise
        jobs.append((f"{m}/{vid}/{q.question_id}",job))
    parallel(jobs,workers,"post")
    jobs=[]
    for qid in REPLAY_IDS:
      q=qmap[qid]
      for m,route in MODEL_SPECS.items():
       for vid,v in VARIANTS.items():
        if store.status(m,vid,qid,"live")!="complete" or store.status(m,vid,qid,"post")!="complete":continue
        def job(q=q,m=m,route=route,vid=vid,v=v):
            t=store.template(m,vid,q.question_id);n=len(split_template(t,v))-1;prev=store.replay_vals(m,vid,q.question_id)
            for i in range(len(prev),n):
                pref=replay_prefix(t,v,prev,i);msgs=replay_msgs(q,v,pref);h=ph(msgs);res=None
                try:
                    res=client.call(route,msgs,rmax);raw=parse_replay(res.text,v,prev);store.replay_ok(m,vid,q.question_id,i,raw,h);store.call(m,vid,q.question_id,"replay",i,h,res,None);prev.append(raw)
                except Exception as e:
                    s=f"{type(e).__name__}: {e}";store.replay_err(m,vid,q.question_id,i,h,s);store.call(m,vid,q.question_id,"replay",i,h,res,s);raise
        jobs.append((f"{m}/{vid}/{qid}",job))
    parallel(jobs,workers,"replay")

def ranks(vals):
    o=sorted(enumerate(vals),key=lambda x:x[1]);r=[0.0]*len(vals);i=0
    while i<len(o):
        j=i+1
        while j<len(o) and o[j][1]==o[i][1]:j+=1
        a=(i+1+j)/2
        for k in range(i,j):r[o[k][0]]=a
        i=j
    return r
def pearson(x,y):
    if len(x)!=len(y) or len(x)<2:return None
    mx=statistics.fmean(x);my=statistics.fmean(y);dx=[a-mx for a in x];dy=[b-my for b in y];d=math.sqrt(sum(a*a for a in dx)*sum(b*b for b in dy))
    return None if d==0 else sum(a*b for a,b in zip(dx,dy))/d
def spearman(x,y):return pearson(ranks(x),ranks(y)) if len(x)>=2 else None
def orderacc(x,y):
    c=t=0
    for i in range(len(x)):
      for j in range(i+1,len(x)):
        a=x[i]-x[j];b=y[i]-y[j]
        if a==0 or b==0:continue
        t+=1;c+=int((a>0)==(b>0))
    return c/t if t else None
def boot(vals,seed,B=5000):
    vals=[float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if len(vals)<2:return (None,None)
    rng=random.Random(seed);ms=[]
    for _ in range(B):ms.append(statistics.fmean(vals[rng.randrange(len(vals))] for _ in vals))
    ms.sort();return ms[int(.025*B)],ms[min(B-1,int(.975*B))]
def mean(vals):return statistics.fmean(vals) if vals else None

def analyze(out,store):
    with store.con() as c:
        rows=c.execute("""SELECT s.model_id,s.variant_id,s.question_id,q.family,s.condition,s.sentence_index,s.raw_score,s.normalized_ai_risk FROM scores s JOIN questions q USING(question_id) ORDER BY s.model_id,s.variant_id,s.question_id,s.condition,s.sentence_index""").fetchall()
        cells=c.execute("SELECT model_id,variant_id,question_id,live_status,post_status,live_error,post_error FROM cells ORDER BY model_id,variant_id,question_id").fetchall()
    with (out/"scores_long.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["model_id","variant_id","question_id","family","condition","sentence_index","raw_score","normalized_ai_risk"])
        for r in rows:w.writerow([r[k] for k in ["model_id","variant_id","question_id","family","condition","sentence_index","raw_score","normalized_ai_risk"]])
    with (out/"cell_status.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["model_id","variant_id","question_id","live_status","post_status","live_error","post_error"])
        for r in cells:w.writerow([r[k] for k in ["model_id","variant_id","question_id","live_status","post_status","live_error","post_error"]])
    by={}
    for r in rows:by.setdefault((r["model_id"],r["variant_id"],r["question_id"]),{}).setdefault(r["condition"],[]).append((r["sentence_index"],float(r["normalized_ai_risk"])))
    qm=[]
    for (m,v,q),conds in sorted(by.items()):
        if "live" not in conds or "post" not in conds:continue
        lv=[x for _,x in sorted(conds["live"])];po=[x for _,x in sorted(conds["post"])]
        if len(lv)!=len(po):continue
        rp=[x for _,x in sorted(conds.get("replay",[]))]
        qm.append({"model_id":m,"variant_id":v,"question_id":q,"n_sentences":len(lv),"live_mean":mean(lv),"post_mean":mean(po),"post_minus_live":mean(po)-mean(lv),"live_post_spearman":spearman(lv,po),"live_post_pairwise_order":orderacc(lv,po),"replay_mean":mean(rp) if len(rp)==len(lv) else None,"replay_minus_live":mean(rp)-mean(lv) if len(rp)==len(lv) else None,"post_minus_replay":mean(po)-mean(rp) if len(rp)==len(lv) else None})
    fields=list(qm[0]) if qm else []
    with (out/"question_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if fields:w.writeheader();w.writerows(qm)
    summ=[]
    for m in MODEL_SPECS:
      for v in VARIANTS:
        s=[r for r in qm if r["model_id"]==m and r["variant_id"]==v];rp=[r for r in s if r["replay_mean"] is not None]
        rho=[r["live_post_spearman"] for r in s if r["live_post_spearman"] is not None];oa=[r["live_post_pairwise_order"] for r in s if r["live_post_pairwise_order"] is not None];sh=[r["post_minus_live"] for r in s];rl=[r["replay_minus_live"] for r in rp];pr=[r["post_minus_replay"] for r in rp]
        rlo,rhi=boot(rho,SEED_DEFAULT+101);slo,shi=boot(sh,SEED_DEFAULT+202)
        summ.append({"model_id":m,"variant_id":v,"n_questions":len(s),"mean_within_output_live_post_spearman":mean(rho),"rho_bootstrap_lo":rlo,"rho_bootstrap_hi":rhi,"mean_pairwise_order_accuracy":mean(oa),"mean_post_minus_live":mean(sh),"shift_bootstrap_lo":slo,"shift_bootstrap_hi":shi,"n_replay_questions":len(rp),"mean_replay_minus_live":mean(rl),"mean_post_minus_replay":mean(pr)})
    with (out/"robustness_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summ[0]));w.writeheader();w.writerows(summ)
    contrasts=[]
    for v in VARIANTS:
        g={r["question_id"]:r for r in qm if r["model_id"]=="gpt-5.5" and r["variant_id"]==v};m={r["question_id"]:r for r in qm if r["model_id"]=="gemini-3.1-pro-preview" and r["variant_id"]==v};common=sorted(g.keys()&m.keys())
        rd=[];sd=[]
        for q in common:
            if g[q]["live_post_spearman"] is not None and m[q]["live_post_spearman"] is not None:rd.append(g[q]["live_post_spearman"]-m[q]["live_post_spearman"])
            sd.append(m[q]["post_minus_live"]-g[q]["post_minus_live"])
        rlo,rhi=boot(rd,SEED_DEFAULT+303);slo,shi=boot(sd,SEED_DEFAULT+404)
        contrasts.append({"variant_id":v,"n_matched_questions":len(common),"mean_rank_reliability_gpt_minus_gemini":mean(rd),"rank_contrast_bootstrap_lo":rlo,"rank_contrast_bootstrap_hi":rhi,"mean_state_shift_gemini_minus_gpt":mean(sd),"shift_contrast_bootstrap_lo":slo,"shift_contrast_bootstrap_hi":shi})
    with (out/"model_contrasts.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(contrasts[0]));w.writeheader();w.writerows(contrasts)
    payload={"summary":summ,"contrasts":contrasts};(out/"analysis_summary.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8");return payload

def manifest(out,args,qs,banksha,scriptsha):
    p={"code_version":CODE_VERSION,"schema_version":SCHEMA_VERSION,"script_filename":Path(__file__).name,"script_sha256":scriptsha,"prompt_bank_version":PROMPT_BANK_VERSION,"prompt_bank_sha256":banksha,"seed":args.seed,"models":MODEL_SPECS,"variants":{k:{"description":v.description,"scoring_instruction":v.scoring_instruction,"reverse_coded":v.reverse_coded} for k,v in VARIANTS.items()},"replay_question_ids":list(REPLAY_IDS),"n_prompts":len(qs),"temperature":args.temperature,"primary_robustness_claims":["GPT-5.5 has greater within-output Live-to-Post rank stability than Gemini in the same direction across elicitation variants.","Gemini has a larger normalized Post-minus-Live shift than GPT-5.5 in the same direction across elicitation variants.","On the fixed Replay subset, Gemini's Post-minus-Replay component remains larger than its Replay-minus-Live component across elicitation variants."],"questions":[{"question_id":q.question_id,"family":q.family,"task":q.task} for q in qs]}
    path=out/"manifest.json";path.write_text(json.dumps(p,indent=2,ensure_ascii=False),encoding="utf-8");return path

def selftest(bank):
    qs,sha=load_bank(bank);q=qs[0]
    tests={"original":"One. <AI SCORE: 20>\nTwo. <AI SCORE: 70>","paraphrase":"One. <AI SCORE: 20>\nTwo. <AI SCORE: 70>","reverse":"One. <HUMAN SCORE: 80>\nTwo. <HUMAN SCORE: 30>","minimal":"One. [20]\nTwo. [70]"}
    for vid,text in tests.items():
        v=VARIANTS[vid];p=parse_live(text,v,2,2);assert p.normalized_scores==(20,70);pref=replay_prefix(p.template,v,[],0);assert slot(v) in pref;assert parse_post("\n".join(fmt(v,x) for x in p.raw_scores),v,2)==p.raw_scores;assert parse_replay(fmt(v,p.raw_scores[0]),v,[])==p.raw_scores[0]
    print(f"self-test: all checks passed | prompt_bank_sha256={sha}")

def getargs():
    p=argparse.ArgumentParser();p.add_argument("--prompt-bank",type=Path,default=Path("robustness_prompt_bank_v1.jsonl"));p.add_argument("--out",type=Path,default=Path("ai_slop_robustness_run"));p.add_argument("--seed",type=int,default=SEED_DEFAULT);p.add_argument("--temperature",type=float,default=.7);p.add_argument("--workers",type=int,default=12);p.add_argument("--timeout",type=float,default=240);p.add_argument("--retries",type=int,default=3);p.add_argument("--live-max-tokens",type=int,default=8192);p.add_argument("--post-max-tokens",type=int,default=4096);p.add_argument("--replay-max-tokens",type=int,default=2048);p.add_argument("--dry-run",action="store_true");p.add_argument("--self-test",action="store_true");p.add_argument("--analyze-only",action="store_true");return p.parse_args()
def main():
    args=getargs();qs,bsha=load_bank(args.prompt_bank);sp=Path(__file__).resolve();ssha=sha_bytes(sp.read_bytes())
    if args.self_test:selftest(args.prompt_bank);return 0
    args.out.mkdir(parents=True,exist_ok=True);db=args.out/"experiment.sqlite3"
    if args.analyze_only:
        if not db.exists():raise SystemExit(f"missing {db}")
        analyze(args.out,Store(db));print("analysis complete");return 0
    frozen=args.out/"robustness_prompt_bank_v1_frozen.jsonl";raw=args.prompt_bank.read_bytes()
    if frozen.exists() and frozen.read_bytes()!=raw:raise SystemExit("frozen prompt bank mismatch")
    frozen.write_bytes(raw);mp=manifest(args.out,args,qs,bsha,ssha)
    print(f"Runner: {sp.name} | SHA256={ssha}\nPrompt bank SHA256={bsha}\nManifest: {mp}\nPrompts=16 Models=2 Variants=4 Live cells=128 Post cells=128 Replay output-cells=64")
    if args.dry_run:print("DRY RUN: no API calls made.");return 0
    load_dotenv(Path.cwd()/".env");load_dotenv(sp.parent/".env");key=os.getenv("OPENROUTER_API_KEY")
    if not key:raise SystemExit("OPENROUTER_API_KEY is not set")
    st=Store(db);st.meta("code_version",CODE_VERSION);st.meta("script_sha256",ssha);st.meta("prompt_bank_sha256",bsha);st.meta("seed",args.seed);st.register(qs)
    run(st,Client(key,args.temperature,args.timeout,args.retries),qs,args.workers,args.live_max_tokens,args.post_max_tokens,args.replay_max_tokens);analyze(args.out,st)
    print(f"Experiment finished.\nSQLite: {db}\nSummary: {args.out/'robustness_summary.csv'}\nContrasts: {args.out/'model_contrasts.csv'}");return 0
if __name__=="__main__":raise SystemExit(main())
