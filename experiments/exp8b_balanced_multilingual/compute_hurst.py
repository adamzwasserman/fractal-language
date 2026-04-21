#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from collections import Counter
import json

LANGUAGES = ["en", "fr", "es", "fi", "ru", "id", "vi", "zh",
             "synth_a", "synth_b", "synth_c", "synth_d"]

def compute_hurst_dfa(series, min_window=10, max_window=None):
    series = np.array(series, dtype=np.float64)
    n = len(series)
    if n < 100:
        return float("nan")
    if max_window is None:
        max_window = n // 4
    profile = np.cumsum(series - np.mean(series))
    window_sizes = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), num=20).astype(int))
    fluctuations, valid_windows = [], []
    for w in window_sizes:
        if w < 4 or w > n // 4:
            continue
        n_segments = n // w
        if n_segments < 2:
            continue
        f2_sum, count = 0, 0
        for i in range(n_segments):
            segment = profile[i*w:(i+1)*w]
            x = np.arange(w)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            f2_sum += np.mean((segment - trend) ** 2)
            count += 1
        if count > 0:
            fluctuations.append(np.sqrt(f2_sum / count))
            valid_windows.append(w)
    if len(valid_windows) < 4:
        return float("nan")
    slope, _ = np.polyfit(np.log(valid_windows), np.log(fluctuations), 1)
    return slope

def compute_corpus_h(lang, max_chars=5000000):
    text_path = Path(f"/workspace/exp8/raw_text_extended/{lang}.txt")
    if not text_path.exists():
        return None
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read(max_chars)
    words = text.split()[:500000]
    word_counts = Counter(words)
    word_freq_series = [word_counts[w] for w in words]
    return {"H_word_freq": compute_hurst_dfa(word_freq_series), "n_words": len(words), "vocab_size": len(word_counts)}

def compute_tokenized_h(lang, max_tokens=1000000):
    chunks = sorted(Path("/workspace/exp8/chunks").glob(f"{lang}_chunk_*.npy"))
    if not chunks:
        return None
    all_tokens = []
    for cf in chunks:
        tokens = np.load(cf)
        all_tokens.extend(tokens.tolist())
        if len(all_tokens) >= max_tokens:
            break
    all_tokens = all_tokens[:max_tokens]
    token_counts = Counter(all_tokens)
    token_freq_series = [token_counts[t] for t in all_tokens]
    return {"H_token_freq": compute_hurst_dfa(token_freq_series), "n_tokens": len(all_tokens), "vocab_used": len(token_counts)}

if __name__ == "__main__":
    results = {"corpus": {}, "tokenized": {}}
    print("=== Corpus H ===")
    for lang in LANGUAGES:
        print(f"  {lang}...", end=" ", flush=True)
        r = compute_corpus_h(lang)
        if r:
            results["corpus"][lang] = r
            print(f"H_word_freq={r[H_word_freq]:.3f}")
        else:
            print("no data")
    print("\n=== Tokenized H ===")
    for lang in LANGUAGES:
        print(f"  {lang}...", end=" ", flush=True)
        r = compute_tokenized_h(lang)
        if r:
            results["tokenized"][lang] = r
            print(f"H_token_freq={r[H_token_freq]:.3f}")
        else:
            print("no data")
    with open("/workspace/exp8/hurst_corpus_tokenized.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /workspace/exp8/hurst_corpus_tokenized.json")
