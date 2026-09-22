#!/usr/bin/env python3
"""Recompute every reported analysis from the frozen reviewer artifacts.

No model API is called. Generation runners were anonymized only by replacing a
non-semantic HTTP-Referer URL where needed; ANONYMIZATION_PATCHES.json records
those reviewer-copy hashes.
"""
from __future__ import annotations
import hashlib, json, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def run(cmd,cwd=None):
    print('+',' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),cwd=cwd,check=True)

def verify_anonymized_runners():
    data=json.loads((ROOT/'ANONYMIZATION_PATCHES.json').read_text())
    for rec in data['patches']:
        runner=ROOT/rec['runner']; manifest=ROOT/rec['run_manifest']
        raw=json.loads(manifest.read_text())['script_sha256']
        if raw != rec['frozen_manifest_runner_sha256']:
            raise SystemExit(f"frozen manifest SHA changed: {manifest}")
        actual=sha256(runner)
        if actual != rec['reviewer_runner_sha256']:
            raise SystemExit(f"reviewer runner SHA mismatch: {runner}\nexpected {rec['reviewer_runner_sha256']}\nactual   {actual}")
        print(f"OK anonymized runner: {runner.name} {actual}")

def compare_files(expected:Path, generated:Path, names:list[str]):
    for name in names:
        e=expected/name; g=generated/name
        if sha256(e)!=sha256(g):
            raise SystemExit(f"reproduction mismatch: {name}\nexpected {sha256(e)}\ngenerated {sha256(g)}")
        print(f"OK reproduced: {name} {sha256(g)}")

def main():
    verify_anonymized_runners()
    run([sys.executable,'analysis_iclr2027.py'],cwd=ROOT/'main_v57'/'run')
    run([sys.executable,ROOT/'robustness16'/'ai_slop_robustness_v1.py',
         '--prompt-bank',ROOT/'robustness16'/'robustness_prompt_bank_v1.jsonl',
         '--out',ROOT/'robustness16'/'run','--analyze-only'])
    run([sys.executable,ROOT/'robustness24'/'ai_slop_robustness_extension_v1.py',
         '--prompt-bank',ROOT/'robustness24'/'robustness_extension_prompt_bank_v1.jsonl',
         '--out',ROOT/'robustness24'/'run','--analyze-only'])
    pooled=ROOT/'pooled40'/'generated'
    if pooled.exists(): shutil.rmtree(pooled)
    pooled.mkdir(exist_ok=True)
    run([sys.executable,ROOT/'pooled40'/'analysis_robustness_pooled.py',
         '--initial-run',ROOT/'robustness16'/'run',
         '--extension-run',ROOT/'robustness24'/'run','--out',pooled])
    tc=ROOT/'temperature_control'; tcgen=tc/'analysis_generated'
    if tcgen.exists(): shutil.rmtree(tcgen)
    tcgen.mkdir(exist_ok=True)
    run([sys.executable,tc/'analyze_temperature_control.py',
         '--temp0',tc/'run_T0','--temp07',tc/'run_T07','--out',tcgen])
    compare_files(tc/'analysis_expected',tcgen,[
        'question_metrics.csv','summary.csv','paired_temperature_difference.csv',
        'model_contrasts.csv','summary.json'])
    # Verify the temperature-run manifests match the included generation runner.
    runner_sha=sha256(tc/'self_confidence_v5_7_promptbank.py')
    for d in ('run_T0','run_T07'):
        manifest=json.loads((tc/d/'manifest.json').read_text())
        if manifest['script_sha256'] != runner_sha:
            raise SystemExit(f"temperature runner SHA mismatch in {d}")
    print('\nAll reported analyses reproduced from frozen outputs.')
    return 0
if __name__=='__main__': raise SystemExit(main())
