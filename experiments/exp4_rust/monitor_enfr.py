#!/usr/bin/env python3
"""Monitor for interleaved EN+FR training experiment."""
import csv
from pathlib import Path
from datetime import datetime

BASE_PATH = Path("/workspace/en_training")
LOG_DIR = BASE_PATH / "logs"

def read_last_n_lines(filepath, n=15):
    if not filepath.exists():
        return []
    with open(filepath, 'r') as f:
        reader = list(csv.DictReader(f))
        return reader[-n:] if reader else []

def main():
    print("=" * 60)
    print("         INTERLEAVED EN+FR 125M TRAINING MONITOR")
    print("=" * 60)
    print()

    # Training progress
    training_log = LOG_DIR / "training_enfr.csv"
    training_data = read_last_n_lines(training_log, 15)
    
    if training_data:
        latest = training_data[-1]
        step = int(latest['step'])
        loss = float(latest['loss'])
        ppl = float(latest['perplexity'])
        
        print(f"CURRENT: Step {step:,} | Loss: {loss:.4f} | PPL: {ppl:.1f}")
        
        # ETA calculation
        target = 200000
        remaining = target - step
        if len(training_data) >= 2:
            first = training_data[0]
            last = training_data[-1]
            steps_done = int(last['step']) - int(first['step'])
            t1 = datetime.fromisoformat(first['timestamp'])
            t2 = datetime.fromisoformat(last['timestamp'])
            elapsed = (t2 - t1).total_seconds()
            if elapsed > 0 and steps_done > 0:
                steps_per_sec = steps_done / elapsed
                eta_hours = remaining / steps_per_sec / 3600
                print(f"SPEED:   {steps_per_sec:.2f} steps/sec")
                print(f"ETA:     {eta_hours:.1f} hours to {target:,} steps")
        print()

        print("-" * 60)
        print("TRAINING (last 10 steps)")
        print("-" * 60)
        print(f"{'STEP':>8} {'LOSS':>10} {'PPL':>12}")
        print(f"{'-'*8} {'-'*10} {'-'*12}")
        for row in training_data[-10:]:
            print(f"{int(row['step']):>8} {float(row['loss']):>10.4f} {float(row['perplexity']):>12.1f}")
    else:
        print("No training data yet.")
    
    print()
    
    # Grammar probes
    print("-" * 60)
    print("GRAMMAR PROBES")
    print("-" * 60)
    
    for lang in ["en", "fr"]:
        probe_log = LOG_DIR / f"grammar_probes_enfr_{lang}.csv"
        probe_data = read_last_n_lines(probe_log, 5)
        if probe_data:
            print(f"\n{lang.upper()} Grammar (last 5):" )
            print(f"{'STEP':>8} {'ACC':>8} {'LOG_RATIO':>12}")
            for row in probe_data:
                print(f"{int(row['step']):>8} {float(row['accuracy']):>7.1f}% {float(row['mean_log_ratio']):>12.4f}")
        else:
            print(f"\n{lang.upper()} Grammar: No data yet")
    
    # Checkpoints
    print()
    print("-" * 60)
    print("CHECKPOINTS")
    print("-" * 60)
    ckpt_dir = BASE_PATH / "checkpoints" / "enfr" / "125M"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("checkpoint_*.json"))
        if ckpts:
            latest_ckpt = ckpts[-1]
            step = int(latest_ckpt.stem.split('_')[1])
            print(f"Latest: step {step:,}")
            print(f"Total checkpoints: {len(ckpts)}")
        else:
            print("No checkpoints yet")
    else:
        print("Checkpoint directory not created yet")

if __name__ == "__main__":
    main()
