#!/usr/bin/env python3
"""
Comprehensive Hurst exponent analysis for exp8 multilingual experiment.

Computes H at three levels:
1. Corpus level: H(char), H(word_freq) from raw text
2. Tokenized level: H(token_freq) from .npy chunks
3. Model level: H(loss), H(entropy), H(top_k) at each checkpoint

Usage:
    python compute_hurst_comprehensive.py --corpus      # Corpus-level H
    python compute_hurst_comprehensive.py --tokenized   # Tokenized H
    python compute_hurst_comprehensive.py --model LANG STEP  # Model H at checkpoint
    python compute_hurst_comprehensive.py --all         # Everything
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path("/Volumes/fractal/exp8b_balanced_multilingual")
CHUNKS_PATH = BASE_PATH / "checkpoints"  # Will need to adjust
REMOTE_CHUNKS = "/workspace/exp8/chunks"
REMOTE_TEXT = "/workspace/exp8/raw_text_extended"

LANGUAGES = ['en', 'fr', 'es', 'fi', 'ru', 'id', 'vi', 'zh',
             'synth_a', 'synth_b', 'synth_c', 'synth_d']


def compute_hurst_dfa(series, min_window=10, max_window=None):
    """Compute Hurst exponent using Detrended Fluctuation Analysis (DFA)."""
    series = np.array(series, dtype=np.float64)
    n = len(series)

    if max_window is None:
        max_window = n // 4

    # Cumulative sum (profile)
    profile = np.cumsum(series - np.mean(series))

    # Window sizes (log-spaced)
    window_sizes = np.unique(np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        num=20
    ).astype(int))

    fluctuations = []
    valid_windows = []

    for w in window_sizes:
        if w < 4 or w > n // 4:
            continue

        # Number of segments
        n_segments = n // w
        if n_segments < 2:
            continue

        f2_sum = 0
        count = 0

        for i in range(n_segments):
            segment = profile[i*w:(i+1)*w]

            # Linear detrend
            x = np.arange(w)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend

            f2_sum += np.mean(detrended ** 2)
            count += 1

        if count > 0:
            fluctuations.append(np.sqrt(f2_sum / count))
            valid_windows.append(w)

    if len(valid_windows) < 4:
        return np.nan

    # Log-log fit
    log_w = np.log(valid_windows)
    log_f = np.log(fluctuations)

    slope, _ = np.polyfit(log_w, log_f, 1)
    return slope


def compute_corpus_hurst(text_path, max_chars=5_000_000):
    """Compute H from raw corpus text."""
    results = {}

    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read(max_chars)

    # Character-level H
    char_series = [ord(c) for c in text[:max_chars]]
    results['H_char'] = compute_hurst_dfa(char_series)

    # Word frequency series
    words = text.split()[:500000]
    word_counts = Counter(words)
    word_freq_series = [word_counts[w] for w in words]
    results['H_word_freq'] = compute_hurst_dfa(word_freq_series)

    # Word ID series (order of appearance)
    word_to_id = {}
    word_id_series = []
    for w in words:
        if w not in word_to_id:
            word_to_id[w] = len(word_to_id)
        word_id_series.append(word_to_id[w])
    results['H_word_id'] = compute_hurst_dfa(word_id_series)

    results['n_chars'] = len(text)
    results['n_words'] = len(words)
    results['vocab_size'] = len(word_counts)

    return results


def compute_tokenized_hurst(chunks_dir, lang, max_tokens=1_000_000):
    """Compute H from tokenized chunks."""
    results = {}

    # Load chunks
    chunk_files = sorted(Path(chunks_dir).glob(f"{lang}_chunk_*.npy"))
    if not chunk_files:
        return {'error': f'No chunks found for {lang}'}

    all_tokens = []
    for cf in chunk_files:
        tokens = np.load(cf)
        all_tokens.extend(tokens.tolist())
        if len(all_tokens) >= max_tokens:
            break

    all_tokens = all_tokens[:max_tokens]

    # Token frequency series
    token_counts = Counter(all_tokens)
    token_freq_series = [token_counts[t] for t in all_tokens]
    results['H_token_freq'] = compute_hurst_dfa(token_freq_series)

    # Token ID series
    results['H_token_id'] = compute_hurst_dfa(all_tokens)

    results['n_tokens'] = len(all_tokens)
    results['vocab_used'] = len(token_counts)

    return results


def compute_model_hurst(model_path, tokenizer_path, val_data_path, device='cpu'):
    """Compute H from model outputs at a checkpoint."""
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    # This requires the model architecture - import from training script
    # For now, return placeholder
    results = {
        'H_loss': None,
        'H_entropy': None,
        'H_top_k': None,
        'note': 'Requires model architecture import'
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', action='store_true', help='Compute corpus-level H')
    parser.add_argument('--tokenized', action='store_true', help='Compute tokenized H')
    parser.add_argument('--model', nargs=2, metavar=('LANG', 'STEP'), help='Compute model H')
    parser.add_argument('--all', action='store_true', help='Compute all H measures')
    parser.add_argument('--output', default='hurst_comprehensive.json', help='Output file')
    args = parser.parse_args()

    results = {}

    if args.corpus or args.all:
        print("=== Computing Corpus-level H ===")
        results['corpus'] = {}
        for lang in LANGUAGES:
            text_path = Path(REMOTE_TEXT) / f"{lang}.txt"
            if text_path.exists():
                print(f"  {lang}...", end=' ', flush=True)
                results['corpus'][lang] = compute_corpus_hurst(text_path)
                print(f"H_word_freq={results['corpus'][lang].get('H_word_freq', 'N/A'):.3f}")
            else:
                print(f"  {lang}: text not found at {text_path}")

    if args.tokenized or args.all:
        print("\n=== Computing Tokenized H ===")
        results['tokenized'] = {}
        chunks_dir = Path(REMOTE_CHUNKS)
        for lang in LANGUAGES:
            print(f"  {lang}...", end=' ', flush=True)
            results['tokenized'][lang] = compute_tokenized_hurst(chunks_dir, lang)
            h = results['tokenized'][lang].get('H_token_freq', 'N/A')
            if isinstance(h, float):
                print(f"H_token_freq={h:.3f}")
            else:
                print(h)

    if args.model:
        lang, step = args.model
        print(f"\n=== Computing Model H for {lang} at step {step} ===")
        # Would need checkpoint path and model loading
        print("Model H computation requires running on GPU with model loaded")

    # Save results
    output_path = BASE_PATH / args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
