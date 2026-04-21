#!/usr/bin/env python3
"""
Regenerate training metrics (perplexity + grammar probes) from checkpoints.
"""
import sys
sys.path.insert(0, '/Users/adam/dev/fractal-language')

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gzip
import json

# Try MLX first, fall back to numpy-only
try:
    import mlx.core as mx
    import mlx.nn as nn
    USE_MLX = True
except ImportError:
    USE_MLX = False
    print("MLX not available, using numpy-only mode")

# Paths
CHECKPOINT_ROOT = Path("/Volumes/Misc Backup/fractal/exp1_125M_en_vs_fr/checkpoints")
TOKENIZER_PATH = Path("/Volumes/Misc Backup/fractal/exp1_125M_en_vs_fr/joint_tokenizer.json.gz")
VALIDATION_DIR = Path("/Volumes/Misc Backup/fractal/exp1_125M_en_vs_fr/validation")
OUTPUT_DIR = Path("/Volumes/fractal/exp1_125M_en_vs_fr/logs")

# Model config for 125M
CONFIG = {
    'vocab_size': 50000,
    'n_layers': 12,
    'd_model': 768,
    'n_heads': 12,
    'd_ff': 3072,
    'max_seq_len': 512,
}


def load_tokenizer():
    """Load the joint tokenizer."""
    from tokenizers import Tokenizer
    with gzip.open(TOKENIZER_PATH, 'rt') as f:
        return Tokenizer.from_str(f.read())


def get_checkpoint_steps(lang: str, size: str = "125M") -> list:
    """Get list of available checkpoint steps."""
    ckpt_dir = CHECKPOINT_ROOT / lang / size
    steps = []
    for f in ckpt_dir.glob("model_*.safetensors"):
        step = int(f.stem.split("_")[1])
        steps.append(step)
    return sorted(steps)


def compute_perplexity_from_checkpoint(ckpt_path: Path, val_data: np.ndarray, sample_size: int = 10000) -> float:
    """Compute perplexity using checkpoint weights (simplified numpy version)."""
    # For now, read loss from checkpoint.json if available
    json_path = ckpt_path.parent / f"checkpoint_{ckpt_path.stem.split('_')[1]}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
            if 'loss' in data:
                return np.exp(data['loss'])
    return None


def run_grammar_probes_from_checkpoint(ckpt_path: Path, tokenizer, lang: str) -> dict:
    """Run grammar probes on a checkpoint."""
    # Define probes based on language
    if lang == 'fr':
        probes = [
            ("Le chat", "dort", "dorment"),
            ("Les chats", "dorment", "dort"),
            ("La fille", "mange", "mangent"),
            ("Les filles", "mangent", "mange"),
            ("Il", "est", "sont"),
            ("Ils", "sont", "est"),
            ("Elle", "parle", "parlent"),
            ("Elles", "parlent", "parle"),
            ("Le garçon", "court", "courent"),
            ("Les garçons", "courent", "court"),
        ]
    else:  # en
        probes = [
            ("The cat", "sleeps", "sleep"),
            ("The cats", "sleep", "sleeps"),
            ("He", "runs", "run"),
            ("They", "run", "runs"),
            ("She", "walks", "walk"),
            ("We", "walk", "walks"),
            ("The dog", "eats", "eat"),
            ("The dogs", "eat", "eats"),
            ("It", "works", "work"),
            ("You", "work", "works"),
        ]

    # For now, return placeholder - actual probe requires model loading
    # This would need MLX or PyTorch to actually run
    return {'accuracy': None, 'correct': 0, 'total': len(probes)}


def regenerate_metrics(lang: str, size: str = "125M"):
    """Regenerate all metrics for a language."""
    print(f"\n{'='*60}")
    print(f"Regenerating metrics for {lang.upper()} {size}")
    print(f"{'='*60}")

    steps = get_checkpoint_steps(lang, size)
    print(f"Found {len(steps)} checkpoints")

    tokenizer = load_tokenizer()

    results = []
    for step in tqdm(steps, desc=f"Processing {lang}"):
        ckpt_path = CHECKPOINT_ROOT / lang / size / f"model_{step}.safetensors"
        json_path = CHECKPOINT_ROOT / lang / size / f"checkpoint_{step}.json"

        # Get loss/perplexity from checkpoint metadata
        ppl = None
        loss = None
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                loss = data.get('loss')
                if loss:
                    ppl = np.exp(loss)

        # Grammar probes would need actual model inference
        # For now, just record what we have
        results.append({
            'step': step,
            'loss': loss,
            'perplexity': ppl,
        })

    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values('step')

    output_path = OUTPUT_DIR / f"regenerated_{lang}_{size}_training.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Regenerate for both languages
    en_df = regenerate_metrics('en', '125M')
    fr_df = regenerate_metrics('fr', '125M')

    # Also create combined dual format
    merged = pd.merge(
        en_df[['step', 'loss', 'perplexity']].rename(columns={'loss': 'en_loss', 'perplexity': 'en_ppl'}),
        fr_df[['step', 'loss', 'perplexity']].rename(columns={'loss': 'fr_loss', 'perplexity': 'fr_ppl'}),
        on='step',
        how='outer'
    ).sort_values('step')

    dual_path = OUTPUT_DIR / "regenerated_training_dual.csv"
    merged.to_csv(dual_path, index=False)
    print(f"\nSaved dual format: {dual_path}")

    # Show summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"EN: {len(en_df)} checkpoints, step range {en_df['step'].min()}-{en_df['step'].max()}")
    print(f"FR: {len(fr_df)} checkpoints, step range {fr_df['step'].min()}-{fr_df['step'].max()}")

    # Show sample of data
    print("\nEN tail:")
    print(en_df.tail())
    print("\nFR tail:")
    print(fr_df.tail())


if __name__ == "__main__":
    main()
