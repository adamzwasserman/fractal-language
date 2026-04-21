#!/usr/bin/env python3
"""Compute model-level Hurst exponents: H(loss), H(entropy), H(top_k) for all languages."""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from tokenizers import Tokenizer
import json
import math

# Config
LANGUAGES = ["en", "fr", "es", "fi", "ru", "id", "vi", "zh", "synth_a", "synth_b", "synth_c", "synth_d"]
BASE_PATH = Path("/workspace/exp8")
CHECKPOINT_PATH = BASE_PATH / "checkpoints"
TOKENIZER_PATH = BASE_PATH / "joint_tokenizer.json"
CHUNKS_PATH = BASE_PATH / "chunks"
SEQ_LEN = 512
N_SEQUENCES = 200  # Number of sequences to process
TOP_K = 10

# Model config (125M)
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 12
D_FF = 3072
VOCAB_SIZE = 50000

def compute_hurst_dfa(series, min_window=10, max_window=None):
    """DFA-based Hurst exponent."""
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


class SimpleTransformer(torch.nn.Module):
    """Minimal transformer for inference."""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(max_seq_len, d_model)

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=0.0, activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        h = self.token_embedding(x) + self.position_embedding(pos)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)

        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits


def load_model(lang, device):
    """Load the latest checkpoint for a language."""
    ckpt_dir = CHECKPOINT_PATH / lang / "125M"
    if not ckpt_dir.exists():
        return None

    # Find latest checkpoint
    model_files = sorted(ckpt_dir.glob("model_*.safetensors"))
    if not model_files:
        return None

    latest = model_files[-1]
    step = int(latest.stem.split("_")[1])

    # Create model
    model = SimpleTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF)

    # Load weights
    state_dict = load_file(latest)

    # Map flat keys to nested structure
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("layers."):
            # Parse layer index and param name
            parts = k.split(".")
            layer_idx = int(parts[1])
            param_name = ".".join(parts[2:])
            new_key = f"layers.{layer_idx}.{param_name}"
            new_state[new_key] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model = model.to(device)
    model.eval()

    return model, step


def load_validation_data(lang, n_sequences, seq_len):
    """Load validation sequences from chunks."""
    chunks = sorted(CHUNKS_PATH.glob(f"{lang}_chunk_*.npy"))
    if not chunks:
        return None

    # Use second-to-last chunk for validation
    if len(chunks) >= 2:
        chunk = np.load(chunks[-2])
    else:
        chunk = np.load(chunks[0])

    # Extract sequences
    sequences = []
    for i in range(0, len(chunk) - seq_len, seq_len):
        seq = chunk[i:i+seq_len]
        if not np.any(seq == 0):  # Skip padding
            sequences.append(seq)
        if len(sequences) >= n_sequences:
            break

    return np.array(sequences)


def compute_model_h(model, data, device):
    """Compute H(loss), H(entropy), H(top_k) from model predictions."""
    losses = []
    entropies = []
    top_k_probs = []

    model.eval()
    with torch.no_grad():
        for seq in data:
            x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.tensor(seq[1:], dtype=torch.long, device=device)

            logits = model(x)[0]  # [seq_len-1, vocab]

            # Per-token loss
            loss_per_token = F.cross_entropy(logits, targets, reduction='none')
            losses.extend(loss_per_token.cpu().numpy().tolist())

            # Per-token entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropies.extend(entropy.cpu().numpy().tolist())

            # Top-k probability mass
            top_k_prob = torch.topk(probs, TOP_K, dim=-1).values.sum(dim=-1)
            top_k_probs.extend(top_k_prob.cpu().numpy().tolist())

    # Compute Hurst exponents
    results = {
        "H_loss": compute_hurst_dfa(losses),
        "H_entropy": compute_hurst_dfa(entropies),
        "H_top_k": compute_hurst_dfa(top_k_probs),
        "n_tokens": len(losses),
        "mean_loss": float(np.mean(losses)),
        "mean_entropy": float(np.mean(entropies)),
        "mean_top_k": float(np.mean(top_k_probs))
    }

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}

    for lang in LANGUAGES:
        print(f"\n{'='*60}")
        print(f"Processing {lang}...")
        print(f"{'='*60}")

        # Load model
        print(f"  Loading model...", end=" ", flush=True)
        model_result = load_model(lang, device)
        if model_result is None:
            print("FAILED - no checkpoint")
            continue
        model, step = model_result
        print(f"step {step}")

        # Load validation data
        print(f"  Loading data...", end=" ", flush=True)
        data = load_validation_data(lang, N_SEQUENCES, SEQ_LEN)
        if data is None or len(data) == 0:
            print("FAILED - no data")
            continue
        print(f"{len(data)} sequences")

        # Compute H values
        print(f"  Computing H values...", end=" ", flush=True)
        try:
            h_results = compute_model_h(model, data, device)
            h_results["step"] = step
            results[lang] = h_results
            print(f"done")
            print(f"    H_loss={h_results['H_loss']:.3f}  H_entropy={h_results['H_entropy']:.3f}  H_top_k={h_results['H_top_k']:.3f}")
        except Exception as e:
            print(f"FAILED - {e}")
            continue

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    # Save results
    output_path = BASE_PATH / "hurst_model_level.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Lang':<10} {'Step':<8} {'H_loss':<10} {'H_entropy':<10} {'H_top_k':<10}")
    print("-"*50)
    for lang in LANGUAGES:
        if lang in results:
            r = results[lang]
            print(f"{lang:<10} {r['step']:<8} {r['H_loss']:<10.3f} {r['H_entropy']:<10.3f} {r['H_top_k']:<10.3f}")


if __name__ == "__main__":
    main()
