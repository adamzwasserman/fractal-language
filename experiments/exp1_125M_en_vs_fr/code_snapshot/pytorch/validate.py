#!/usr/bin/env python
"""
Validation script - compute perplexity on held-out validation set.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path
from safetensors.torch import load_file

# ==================== DEVICE ====================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ==================== CONFIG ====================
CONFIGS = {
    "125M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072, seq=512),
    "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096, seq=512),
}

# ==================== MODEL ====================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        ln_x = self.ln1(x)
        T = ln_x.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=ln_x.device), diagonal=1).bool()
        attn_out, _ = self.attn(ln_x, ln_x, ln_x, attn_mask=causal_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.drop(self.embed(x) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def load_model(checkpoint_path, vocab_size, cfg):
    """Load model from checkpoint."""
    model = Transformer(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"]
    ).to(DEVICE)

    state_dict = load_file(str(checkpoint_path))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_validation_perplexity(model, sequences, batch_size=16):
    """Compute perplexity on validation sequences."""
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            x = torch.tensor(batch, dtype=torch.long, device=DEVICE)

            # Input is all but last token, target is all but first token
            inputs = x[:, :-1]
            targets = x[:, 1:]

            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='sum')

            total_loss += loss.item()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


def main():
    if len(sys.argv) < 3:
        print("Usage: python validate.py <step> <model_size>")
        print("Example: python validate.py 13000 125M")
        sys.exit(1)

    step = int(sys.argv[1])
    model_size = sys.argv[2]
    cfg = CONFIGS[model_size]

    base_path = Path("/workspace/data")
    validation_path = base_path / "validation"

    # Load tokenizer to get vocab size
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(base_path / "joint_tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    results = {}

    for lang in ["en", "fr"]:
        print(f"\n{'='*50}")
        print(f"Evaluating {lang.upper()} model at step {step}")
        print('='*50)

        # Load model
        model_path = base_path / "checkpoints" / lang / model_size / f"model_{step}.safetensors"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        model = load_model(model_path, vocab_size, cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded model: {n_params/1e6:.1f}M parameters")

        # Load validation data
        val_file = validation_path / f"{lang}_validation.json"
        if not val_file.exists():
            print(f"Validation file not found: {val_file}")
            continue

        with open(val_file) as f:
            val_data = json.load(f)

        sequences = val_data["sequences"]
        print(f"Validation sequences: {len(sequences)}")

        # Compute perplexity
        loss, ppl = compute_validation_perplexity(model, sequences)

        results[lang] = {"loss": loss, "perplexity": ppl}
        print(f"\n{lang.upper()} Validation Results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")

    # Summary comparison
    if "en" in results and "fr" in results:
        print(f"\n{'='*50}")
        print("COMPARISON")
        print('='*50)
        print(f"{'Metric':<15} {'EN':>12} {'FR':>12} {'Ratio':>12}")
        print("-"*50)
        print(f"{'Loss':<15} {results['en']['loss']:>12.4f} {results['fr']['loss']:>12.4f} {results['en']['loss']/results['fr']['loss']:>12.2f}x")
        print(f"{'Perplexity':<15} {results['en']['perplexity']:>12.2f} {results['fr']['perplexity']:>12.2f} {results['en']['perplexity']/results['fr']['perplexity']:>12.2f}x")


if __name__ == "__main__":
    main()
