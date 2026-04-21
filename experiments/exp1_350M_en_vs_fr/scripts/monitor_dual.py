#!/usr/bin/env python3
"""Monitor 350M EN/FR training progress (single-GPU mode)"""
import os
import csv
import subprocess
from pathlib import Path
from datetime import datetime

DATA_PATH = Path(os.environ.get("DATA_PATH", "/workspace/data"))
LOG_DIR = DATA_PATH / "logs"
MODEL_SIZE = "350M"

def read_training_log(lang, n=10):
    log_file = LOG_DIR / f"training_{lang}_{MODEL_SIZE}.csv"
    if not log_file.exists():
        return []
    rows = []
    with open(log_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows[-n:]

def get_gpu_usage():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', 
             '--format=csv,noheader,nounits'], 
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            idx, util, used, total = parts[0], parts[1], float(parts[2])/1024, float(parts[3])/1024
            gpu_info.append(f"GPU{idx}: {util}% | {used:.1f}/{total:.1f}GB")
        return "  ".join(gpu_info)
    except Exception as e:
        return f"GPU: Error ({e})"

def main():
    os.system('clear')
    
    print("=" * 80)
    print(f"          350M EN/FR TRAINING MONITOR          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Get latest from each
    en_rows = read_training_log("en", 20)
    fr_rows = read_training_log("fr", 20)
    
    # Latest stats
    en_latest = en_rows[-1] if en_rows else {}
    fr_latest = fr_rows[-1] if fr_rows else {}
    
    en_step = en_latest.get('step', '?')
    fr_step = fr_latest.get('step', '?')
    en_ppl = float(en_latest.get('ppl', 0))
    fr_ppl = float(fr_latest.get('ppl', 0))
    en_loss = float(en_latest.get('loss', 0))
    fr_loss = float(fr_latest.get('loss', 0))
    en_acc = en_latest.get('grammar_acc', '')
    fr_acc = fr_latest.get('grammar_acc', '')
    
    print(f"\n{'':^38} | {'ENGLISH':^18} | {'FRENCH':^18}")
    print("-" * 80)
    print(f"{'Step':^38} | {en_step:^18} | {fr_step:^18}")
    print(f"{'Loss':^38} | {en_loss:^18.4f} | {fr_loss:^18.4f}")
    print(f"{'Perplexity':^38} | {en_ppl:^18.1f} | {fr_ppl:^18.1f}")
    if en_acc or fr_acc:
        en_acc_str = f"{float(en_acc)*100:.1f}%" if en_acc else "--"
        fr_acc_str = f"{float(fr_acc)*100:.1f}%" if fr_acc else "--"
        print(f"{'Grammar Accuracy':^38} | {en_acc_str:^18} | {fr_acc_str:^18}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    if en_ppl > 0 and fr_ppl > 0:
        ppl_ratio = en_ppl / fr_ppl if fr_ppl > 0 else 0
        print(f"PPL Ratio (EN/FR): {ppl_ratio:.2f}x")
        print(f"Loss Diff (EN-FR): {en_loss - fr_loss:+.4f}")
    
    # Recent trajectory
    print("\n" + "=" * 80)
    print("RECENT TRAINING (last 10 checkpoints with grammar probes)")
    print("=" * 80)
    print(f"{'STEP':>8} | {'EN_PPL':>8} {'EN_LOSS':>8} {'EN_GRAM':>8} | {'FR_PPL':>8} {'FR_LOSS':>8} {'FR_GRAM':>8}")
    print("-" * 80)
    
    # Get rows with grammar data
    en_with_gram = [r for r in en_rows if r.get('grammar_acc')]
    fr_with_gram = [r for r in fr_rows if r.get('grammar_acc')]
    
    # Match by step
    fr_by_step = {r['step']: r for r in fr_with_gram}
    for en_row in en_with_gram[-10:]:
        step = en_row['step']
        en_p = float(en_row.get('ppl', 0))
        en_l = float(en_row.get('loss', 0))
        en_g = float(en_row.get('grammar_acc', 0)) * 100
        
        fr_row = fr_by_step.get(step, {})
        fr_p = float(fr_row.get('ppl', 0)) if fr_row else 0
        fr_l = float(fr_row.get('loss', 0)) if fr_row else 0
        fr_g = float(fr_row.get('grammar_acc', 0)) * 100 if fr_row else 0
        
        print(f"{step:>8} | {en_p:>8.1f} {en_l:>8.4f} {en_g:>7.1f}% | {fr_p:>8.1f} {fr_l:>8.4f} {fr_g:>7.1f}%")
    
    print(f"\n{get_gpu_usage()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
