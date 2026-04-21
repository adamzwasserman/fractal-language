#!/usr/bin/env python
"""
Grammar probes for PyTorch models.
Tests basic grammatical capabilities at 125M scale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from safetensors.torch import load_file
import json
import sys
import os

# ==================== CONFIG ====================
BASE_PATH = Path(os.environ.get("DATA_PATH", "/workspace/data"))
CHECKPOINTS_DIR = BASE_PATH / "checkpoints"
TOKENIZER_PATH = BASE_PATH / "joint_tokenizer.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model config (must match training)
cfg = {"125M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072)}

# ==================== MODEL (same as train.py) ====================
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

    def forward(self, x, mask=None):
        ln_x = self.ln1(x)
        attn_out, _ = self.attn(ln_x, ln_x, ln_x, attn_mask=mask, is_causal=True)
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


# ==================== GRAMMAR PROBES ====================
GRAMMAR_PROBES = {
    "en": [
        {"prompt": "The cat ", "good": ["is", "was", "sits", "runs"], "bad": ["are", "were", "sit", "run"], "category": "sv_agreement"},
        {"prompt": "The dog ", "good": ["is", "was", "barks", "runs"], "bad": ["are", "were", "bark", "run"], "category": "sv_agreement"},
        {"prompt": "A bird ", "good": ["is", "was", "flies", "sings"], "bad": ["are", "were", "fly", "sing"], "category": "sv_agreement"},
        {"prompt": "The man ", "good": ["is", "was", "walks", "runs"], "bad": ["are", "were", "walk", "run"], "category": "sv_agreement"},
        {"prompt": "She ", "good": ["is", "was", "has", "does"], "bad": ["are", "were", "have", "do"], "category": "sv_agreement"},
        {"prompt": "The cats ", "good": ["are", "were", "sit", "run"], "bad": ["is", "was", "sits", "runs"], "category": "sv_agreement"},
        {"prompt": "The dogs ", "good": ["are", "were", "bark", "run"], "bad": ["is", "was", "barks", "runs"], "category": "sv_agreement"},
        {"prompt": "The birds ", "good": ["are", "were", "fly", "sing"], "bad": ["is", "was", "flies", "sings"], "category": "sv_agreement"},
        {"prompt": "They ", "good": ["are", "were", "have", "do"], "bad": ["is", "was", "has", "does"], "category": "sv_agreement"},
        {"prompt": "We ", "good": ["are", "were", "have", "do"], "bad": ["is", "was", "has", "does"], "category": "sv_agreement"},
        {"prompt": "I saw a ", "good": ["cat", "dog", "bird", "man", "book"], "bad": ["apple", "elephant", "orange", "hour"], "category": "article"},
        {"prompt": "I saw an ", "good": ["apple", "elephant", "orange", "animal"], "bad": ["cat", "dog", "bird", "man"], "category": "article"},
        {"prompt": "She is a ", "good": ["woman", "teacher", "doctor", "student"], "bad": ["engineer", "artist", "actor"], "category": "article"},
        {"prompt": "He is an ", "good": ["engineer", "artist", "actor", "adult"], "bad": ["woman", "teacher", "doctor"], "category": "article"},
        {"prompt": "black and ", "good": ["white"], "bad": ["red", "green", "blue"], "category": "collocation"},
        {"prompt": "up and ", "good": ["down"], "bad": ["left", "right", "over"], "category": "collocation"},
        {"prompt": "day and ", "good": ["night"], "bad": ["morning", "evening", "time"], "category": "collocation"},
        {"prompt": "hot and ", "good": ["cold"], "bad": ["warm", "cool", "wet"], "category": "collocation"},
        {"prompt": "The sun rises in the ", "good": ["morning", "east"], "bad": ["night", "west", "afternoon"], "category": "completion"},
        {"prompt": "Water is ", "good": ["wet", "cold", "clear", "blue"], "bad": ["dry", "hard", "solid"], "category": "completion"},
    ],
    "fr": [
        {"prompt": "Le chat ", "good": ["est", "était", "noir", "petit"], "bad": ["sont", "étaient", "noire", "petite"], "category": "gender"},
        {"prompt": "Le chien ", "good": ["est", "était", "noir", "grand"], "bad": ["sont", "étaient", "noire", "grande"], "category": "gender"},
        {"prompt": "Un homme ", "good": ["est", "était", "grand", "petit"], "bad": ["sont", "étaient", "grande", "petite"], "category": "gender"},
        {"prompt": "Le livre ", "good": ["est", "était", "petit", "nouveau"], "bad": ["sont", "étaient", "petite", "nouvelle"], "category": "gender"},
        {"prompt": "Il ", "good": ["est", "était", "a", "fait"], "bad": ["sont", "étaient", "ont", "font"], "category": "gender"},
        {"prompt": "La maison ", "good": ["est", "était", "grande", "belle"], "bad": ["sont", "étaient", "grand", "beau"], "category": "gender"},
        {"prompt": "La femme ", "good": ["est", "était", "grande", "belle"], "bad": ["sont", "étaient", "grand", "beau"], "category": "gender"},
        {"prompt": "Une fille ", "good": ["est", "était", "petite", "belle"], "bad": ["sont", "étaient", "petit", "beau"], "category": "gender"},
        {"prompt": "La table ", "good": ["est", "était", "grande", "petite"], "bad": ["sont", "étaient", "grand", "petit"], "category": "gender"},
        {"prompt": "Elle ", "good": ["est", "était", "a", "fait"], "bad": ["sont", "étaient", "ont", "font"], "category": "gender"},
        {"prompt": "Les chats ", "good": ["sont", "étaient", "noirs", "petits"], "bad": ["est", "était", "noir", "petit"], "category": "number"},
        {"prompt": "Les maisons ", "good": ["sont", "étaient", "grandes", "belles"], "bad": ["est", "était", "grande", "belle"], "category": "number"},
        {"prompt": "Ils ", "good": ["sont", "étaient", "ont", "font"], "bad": ["est", "était", "a", "fait"], "category": "number"},
        {"prompt": "Elles ", "good": ["sont", "étaient", "ont", "font"], "bad": ["est", "était", "a", "fait"], "category": "number"},
        {"prompt": "Je vois le ", "good": ["chat", "chien", "livre", "garçon"], "bad": ["maison", "femme", "fille", "table"], "category": "article_gender"},
        {"prompt": "Je vois la ", "good": ["maison", "femme", "fille", "table"], "bad": ["chat", "chien", "livre", "garçon"], "category": "article_gender"},
        {"prompt": "noir et ", "good": ["blanc"], "bad": ["rouge", "vert", "bleu"], "category": "collocation"},
        {"prompt": "jour et ", "good": ["nuit"], "bad": ["matin", "soir", "temps"], "category": "collocation"},
        {"prompt": "chaud et ", "good": ["froid"], "bad": ["tiède", "frais", "sec"], "category": "collocation"},
        {"prompt": "Le soleil se lève le ", "good": ["matin"], "bad": ["soir", "nuit", "jour"], "category": "completion"},
    ],
}


def load_model(step: int, lang: str):
    """Load model checkpoint."""
    model_path = CHECKPOINTS_DIR / lang / "125M" / f"model_{step}.safetensors"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()

    c = cfg["125M"]
    model = Transformer(vocab_size, c["d_model"], c["n_layers"], c["n_heads"], c["d_ff"])

    state_dict = load_file(str(model_path))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, tokenizer


def get_token_probability(model, tokenizer, prompt: str, target: str) -> float:
    """Get probability of target token given prompt."""
    prompt_ids = tokenizer.encode(prompt).ids
    target_ids = tokenizer.encode(" " + target).ids
    if not target_ids:
        target_ids = tokenizer.encode(target).ids
    if not target_ids:
        return 0.0
    target_id = target_ids[0]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        return probs[target_id].item()


def evaluate_probe(model, tokenizer, probe: dict) -> dict:
    """Evaluate a single probe."""
    prompt = probe["prompt"]
    good_probs = [get_token_probability(model, tokenizer, prompt, w) for w in probe["good"]]
    bad_probs = [get_token_probability(model, tokenizer, prompt, w) for w in probe["bad"]]

    avg_good = np.mean(good_probs)
    avg_bad = np.mean(bad_probs)
    ratio = avg_good / (avg_bad + 1e-10)
    correct = max(good_probs) > max(bad_probs)

    return {
        "prompt": prompt,
        "category": probe["category"],
        "avg_good_prob": avg_good,
        "avg_bad_prob": avg_bad,
        "ratio": ratio,
        "correct": correct,
    }


def run_probes(step: int, lang: str):
    """Run all grammar probes."""
    print(f"Loading model: {lang} step {step}")
    result = load_model(step, lang)
    if result is None:
        return None
    model, tokenizer = result

    probes = GRAMMAR_PROBES[lang]
    results = []

    print(f"\nRunning {len(probes)} grammar probes for {lang.upper()}...\n")

    for probe in probes:
        r = evaluate_probe(model, tokenizer, probe)
        results.append(r)
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} [{r['category']:15}] \"{probe['prompt']}\" ratio={r['ratio']:.2f}")

    # Summary
    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    overall_pct = 100 * total_correct / total
    mean_ratio = np.mean([r["ratio"] for r in results])

    print(f"\n{'='*50}")
    print(f"SUMMARY: {lang.upper()} step {step}")
    print(f"{'='*50}")
    print(f"  OVERALL: {total_correct}/{total} ({overall_pct:.1f}%)")
    print(f"  Mean ratio (good/bad): {mean_ratio:.2f}")

    return {
        "step": step,
        "lang": lang,
        "accuracy": overall_pct,
        "mean_ratio": mean_ratio,
        "details": results,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python grammar_probes.py <step> <lang>")
        sys.exit(1)

    step = int(sys.argv[1])
    lang = sys.argv[2]

    results = run_probes(step, lang)

    if results:
        output_dir = BASE_PATH / "results" / "grammar_probes"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"grammar_{lang}_{step}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
