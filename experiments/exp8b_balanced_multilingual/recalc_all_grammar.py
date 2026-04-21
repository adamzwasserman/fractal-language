#!/usr/bin/env python3
"""
Recalculate grammar probes for ALL languages from existing checkpoints.
Updates the JSONL log files with corrected grammar_acc values.
Uses v2 probes: 10 probes per testable dimension per language.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from safetensors.torch import load_file

# Config
DATA_PATH = Path(os.environ.get("DATA_PATH", "/workspace/exp8"))
LANGUAGES = ["en", "fr", "es", "fi", "ru", "id", "vi", "zh", "synth_a", "synth_b", "synth_c", "synth_d"]
MODEL_SIZE = "125M"
N_LAYERS = 12
D_MODEL = 768
N_HEADS = 12
D_FF = 3072
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPERATURE = 0.1  # exp1 default

# Import probes from grammar_probes_v2
from grammar_probes_v2 import GRAMMAR_PROBES


# Model definition
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
            nn.Dropout(dropout)
        )
        self.register_buffer("mask", None)

    def get_mask(self, seq_len, device):
        if self.mask is None or self.mask.size(0) < seq_len:
            self.mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return self.mask[:seq_len, :seq_len]

    def forward(self, x):
        mask = self.get_mask(x.size(1), x.device)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def get_token_probability(model, tokenizer, prompt, target, device):
    """
    Get probability of target sequence given prompt.

    BUG (FIXED): XLM-R tokenizer splits inflected forms differently than custom BPE.
    For example:
        "sits" -> ['▁sit', 's']  (2 tokens)
        "sit"  -> ['▁sit']       (1 token)

    The old code only checked P(first_token). This meant the probe was comparing:
        P(▁sit | "The cat ") vs P(▁sit | "The cats ")

    Since both "sits" and "sit" share the same first token (▁sit), the probe
    was NOT actually testing the inflection at all for these verbs!

    This explains why EN grammar was artificially high in exp8 - the probes
    were broken for XLM-R tokenization when verbs share stems.

    FIX: Compute joint probability of ALL tokens in the target sequence.
    P("sits" | prompt) = P(▁sit | prompt) * P(s | prompt, ▁sit)
    P("sit" | prompt)  = P(▁sit | prompt)

    These are now different, correctly testing whether the model prefers
    the inflected vs uninflected form.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Add space prefix for proper tokenization
    target_ids = tokenizer.encode(" " + target, add_special_tokens=False)
    if not target_ids:
        target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        return 0.0

    # For Chinese/CJK: token 6 is ▁ (space marker), skip it
    if target_ids[0] == 6 and len(target_ids) > 1:
        target_ids = target_ids[1:]

    # Concatenate prompt + target for autoregressive scoring
    full_ids = prompt_ids + target_ids
    x = torch.tensor([full_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(x)
        # Apply temperature for stable measurements
        logits = logits / TEMPERATURE

        # Compute log probability of each target token at its position
        # Position i in logits predicts token i+1
        # So logits[prompt_len-1] predicts target[0], logits[prompt_len] predicts target[1], etc.
        log_prob_sum = 0.0
        for i, target_token in enumerate(target_ids):
            pos = len(prompt_ids) - 1 + i  # Position in logits that predicts this token
            log_probs = F.log_softmax(logits[0, pos, :], dim=-1)
            log_prob_sum += log_probs[target_token].item()

        # Return joint probability (exp of sum of log probs)
        # Normalize by sequence length to avoid penalizing longer targets
        return np.exp(log_prob_sum / len(target_ids))


def run_grammar_probes(model, tokenizer, lang, device):
    """Evaluate grammar with exp1-style probes."""
    model.eval()
    probes = GRAMMAR_PROBES.get(lang, GRAMMAR_PROBES.get("en", []))

    if not probes:
        return 0.5, 0.0, {}

    correct = 0
    total = 0
    log_ratios = []
    by_category = {}

    for probe in probes:
        prompt = probe["prompt"]
        good_words = probe["good"]
        bad_words = probe["bad"]

        good_probs = [get_token_probability(model, tokenizer, prompt, w, device) for w in good_words]
        bad_probs = [get_token_probability(model, tokenizer, prompt, w, device) for w in bad_words]

        avg_good = np.mean(good_probs)
        avg_bad = np.mean(bad_probs)

        log_ratio = np.log(avg_good + 1e-10) - np.log(avg_bad + 1e-10)

        is_correct = avg_good > avg_bad
        if is_correct:
            correct += 1
        total += 1
        log_ratios.append(log_ratio)

        category = probe.get("category", "other")
        if category not in by_category:
            by_category[category] = {"correct": 0, "total": 0}
        by_category[category]["total"] += 1
        if is_correct:
            by_category[category]["correct"] += 1

    for cat in by_category:
        t = by_category[cat]["total"]
        by_category[cat]["accuracy"] = by_category[cat]["correct"] / t if t > 0 else 0.0

    accuracy = correct / total if total > 0 else 0.5
    mean_lr = np.mean(log_ratios) if log_ratios else 0
    return accuracy, mean_lr, by_category


def load_checkpoint(model, checkpoint_path):
    """Load model from safetensors checkpoint."""
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    model.head.weight = model.tok_emb.weight
    return model


def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    print(f"Device: {DEVICE}")

    # Create model
    model = Transformer(vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF).to(DEVICE)

    # Process each language
    for lang in LANGUAGES:
        print(f"\n{'='*60}")
        print(f"Processing {lang}")
        print(f"{'='*60}")

        checkpoint_dir = DATA_PATH / "checkpoints" / lang / MODEL_SIZE
        log_file = DATA_PATH / "logs" / f"{lang}_{MODEL_SIZE}_probes.jsonl"

        if not checkpoint_dir.exists():
            print(f"  No checkpoints found for {lang}")
            continue

        # Find all checkpoints
        checkpoints = sorted(checkpoint_dir.glob("model_*.safetensors"),
                           key=lambda x: int(x.stem.split('_')[1]))

        if not checkpoints:
            print(f"  No model checkpoints found for {lang}")
            continue

        print(f"  Found {len(checkpoints)} checkpoints")

        # Load existing log data
        existing_data = {}
        if log_file.exists():
            with open(log_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            existing_data[entry.get('step', 0)] = entry
                        except:
                            pass

        print(f"  Existing log entries: {len(existing_data)}")

        # Process each checkpoint
        updated_count = 0
        for ckpt_path in checkpoints:
            step = int(ckpt_path.stem.split('_')[1])

            if step not in existing_data:
                print(f"  Step {step}: no log entry, skipping")
                continue

            # Load model
            try:
                model = load_checkpoint(model, ckpt_path)
            except Exception as e:
                print(f"  Step {step}: failed to load checkpoint: {e}")
                continue

            # Run grammar probes
            grammar_acc, grammar_log_ratio, grammar_by_category = run_grammar_probes(
                model, tokenizer, lang, DEVICE
            )

            # Update log entry
            old_acc = existing_data[step].get('grammar_acc', 0)
            existing_data[step]['grammar_acc'] = grammar_acc
            existing_data[step]['grammar_log_ratio'] = grammar_log_ratio
            existing_data[step]['grammar_by_category'] = grammar_by_category

            print(f"  Step {step}: {old_acc*100:.0f}% -> {grammar_acc*100:.0f}%")
            updated_count += 1

        # Write updated log file
        with open(log_file, 'w') as f:
            for step in sorted(existing_data.keys()):
                f.write(json.dumps(existing_data[step]) + '\n')

        print(f"  Updated {updated_count} entries, wrote to {log_file}")

    print("\n" + "="*60)
    print("Done! All grammar probes recalculated.")
    print("="*60)


if __name__ == "__main__":
    main()
