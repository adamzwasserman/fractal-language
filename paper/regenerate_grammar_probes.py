#!/usr/bin/env python3
"""
Regenerate grammar probes from checkpoints using MLX.
"""
import sys
sys.path.insert(0, '/Users/adam/dev/fractal-language')

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gzip
import json

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

# Paths
CHECKPOINT_ROOT = Path("/Volumes/Misc Backup/fractal/exp1_125M_en_vs_fr/checkpoints")
TOKENIZER_PATH = Path("/Volumes/Misc Backup/fractal/exp1_125M_en_vs_fr/joint_tokenizer.json.gz")
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


class GPT2Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def __call__(self, x, mask=None):
        h = self.ln1(x)
        h = self.attn(h, h, h, mask=mask)
        x = x + h
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_embed = nn.Embedding(config['max_seq_len'], config['d_model'])
        self.blocks = [GPT2Block(config['d_model'], config['n_heads'], config['d_ff'])
                       for _ in range(config['n_layers'])]
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

    def __call__(self, x):
        B, T = x.shape
        pos = mx.arange(T)
        x = self.embed(x) + self.pos_embed(pos)

        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        return self.head(x)


def load_tokenizer():
    """Load the joint tokenizer."""
    from tokenizers import Tokenizer
    with gzip.open(TOKENIZER_PATH, 'rt') as f:
        return Tokenizer.from_str(f.read())


def load_model_from_checkpoint(ckpt_path: Path) -> GPT2:
    """Load model from safetensors checkpoint."""
    model = GPT2(CONFIG)

    # Load weights from safetensors
    weights = {}
    with safe_open(ckpt_path, framework="numpy") as f:
        for key in f.keys():
            weights[key] = mx.array(f.get_tensor(key))

    # Unflatten weights back to nested structure
    nested = {}
    for key, value in weights.items():
        parts = key.split('.')
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    model.load_weights(list(nested.items()))
    return model


def run_grammar_probes(model, tokenizer, lang: str) -> dict:
    """Run grammar probes on a model."""
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
    else:
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

    correct = 0
    total = 0

    for context, good, bad in probes:
        tokens = tokenizer.encode(context).ids
        if not tokens:
            continue

        good_id = tokenizer.encode(" " + good).ids[0] if tokenizer.encode(" " + good).ids else tokenizer.encode(good).ids[0]
        bad_id = tokenizer.encode(" " + bad).ids[0] if tokenizer.encode(" " + bad).ids else tokenizer.encode(bad).ids[0]

        x = mx.array([tokens])
        logits = model(x)
        next_logits = logits[0, -1]

        if next_logits[good_id].item() > next_logits[bad_id].item():
            correct += 1
        total += 1

    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total
    }


def get_checkpoint_steps(lang: str, size: str = "125M") -> list:
    """Get list of available checkpoint steps."""
    ckpt_dir = CHECKPOINT_ROOT / lang / size
    steps = []
    for f in ckpt_dir.glob("model_*.safetensors"):
        step = int(f.stem.split("_")[1])
        steps.append(step)
    return sorted(steps)


def regenerate_probes(lang: str, size: str = "125M", sample_every: int = 5000):
    """Regenerate grammar probes for a language."""
    print(f"\n{'='*60}")
    print(f"Regenerating grammar probes for {lang.upper()} {size}")
    print(f"{'='*60}")

    steps = get_checkpoint_steps(lang, size)
    # Sample every N steps to speed up
    steps = [s for s in steps if s % sample_every == 0 or s == steps[-1]]
    print(f"Processing {len(steps)} checkpoints (every {sample_every} steps)")

    tokenizer = load_tokenizer()

    results = []
    for step in tqdm(steps, desc=f"Probing {lang}"):
        ckpt_path = CHECKPOINT_ROOT / lang / size / f"model_{step}.safetensors"

        try:
            model = load_model_from_checkpoint(ckpt_path)
            probe_result = run_grammar_probes(model, tokenizer, lang)

            results.append({
                'step': step,
                'accuracy': probe_result['accuracy'] * 100,
                'correct': probe_result['correct'],
                'total': probe_result['total'],
            })
        except Exception as e:
            print(f"  Error at step {step}: {e}")
            continue

    df = pd.DataFrame(results)
    df = df.sort_values('step')

    output_path = OUTPUT_DIR / f"regenerated_grammar_probes_{lang}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Regenerate probes (sample every 5000 steps for speed)
    en_df = regenerate_probes('en', '125M', sample_every=10000)
    fr_df = regenerate_probes('fr', '125M', sample_every=10000)

    print("\n" + "="*60)
    print("Summary:")
    print(f"\nEN grammar probes:")
    print(en_df)
    print(f"\nFR grammar probes:")
    print(fr_df)


if __name__ == "__main__":
    main()
