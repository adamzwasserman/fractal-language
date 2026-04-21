#!/usr/bin/env python3
"""Regenerate 350M perplexity from checkpoints."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

# 350M checkpoint locations
EN_CKPTS = Path("/Volumes/fractal/exp1_350M_en_vs_fr/checkpoints_en")
FR_CKPTS = Path("/Volumes/fractal/exp1_350M_en_vs_fr/checkpoints_fr")
OUTPUT_DIR = Path("/Volumes/fractal/exp1_350M_en_vs_fr/logs")

def get_steps_and_ppl(ckpt_dir):
    results = []
    for f in ckpt_dir.glob("checkpoint_*.json"):
        step = int(f.stem.split("_")[1])
        with open(f) as fp:
            data = json.load(fp)
            loss = data.get('loss')
            if loss:
                ppl = np.exp(loss)
                results.append({'step': step, 'loss': loss, 'perplexity': ppl})
    return sorted(results, key=lambda x: x['step'])

print("Processing EN 350M...")
en_results = get_steps_and_ppl(EN_CKPTS)
en_df = pd.DataFrame(en_results)
print(f"  Found {len(en_df)} checkpoints, steps {en_df['step'].min()}-{en_df['step'].max()}")
print(f"  PPL range: {en_df['perplexity'].min():.1f} - {en_df['perplexity'].max():.1f}")

print("Processing FR 350M...")
fr_results = get_steps_and_ppl(FR_CKPTS)
fr_df = pd.DataFrame(fr_results)
print(f"  Found {len(fr_df)} checkpoints, steps {fr_df['step'].min()}-{fr_df['step'].max()}")
print(f"  PPL range: {fr_df['perplexity'].min():.1f} - {fr_df['perplexity'].max():.1f}")

# Save individual files
en_df.to_csv(OUTPUT_DIR / "regenerated_en_350M_training.csv", index=False)
fr_df.to_csv(OUTPUT_DIR / "regenerated_fr_350M_training.csv", index=False)

# Create dual format
merged = pd.merge(
    en_df[['step', 'loss', 'perplexity']].rename(columns={'loss': 'en_loss', 'perplexity': 'en_ppl'}),
    fr_df[['step', 'loss', 'perplexity']].rename(columns={'loss': 'fr_loss', 'perplexity': 'fr_ppl'}),
    on='step', how='outer'
).sort_values('step')
merged.to_csv(OUTPUT_DIR / "regenerated_training_dual_350M.csv", index=False)

print("\nSaved to:")
print(f"  {OUTPUT_DIR}/regenerated_en_350M_training.csv")
print(f"  {OUTPUT_DIR}/regenerated_fr_350M_training.csv")
print(f"  {OUTPUT_DIR}/regenerated_training_dual_350M.csv")

print("\nEN tail:")
print(en_df.tail())
print("\nFR tail:")
print(fr_df.tail())
