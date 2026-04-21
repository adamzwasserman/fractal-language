#!/usr/bin/env python
"""
Simple grammar probes for 125M models.
Tests capabilities that should emerge early:
- Subject-verb agreement
- Article selection
- Gender/number agreement
- Basic completion

Usage:
    uv run python scripts/grammar_probes.py <step> <lang>
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import json
import sys

mx.random.seed(42)
np.random.seed(42)

BASE_DIR = Path("/Volumes/Misc Backup/fractal")
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TOKENIZER_PATH = BASE_DIR / "joint_tokenizer.json"

# Simple grammar probes - testing what 125M models CAN do
GRAMMAR_PROBES = {
    "en": [
        # Subject-verb agreement (singular)
        {"prompt": "The cat ", "good": ["is", "was", "sits", "runs"], "bad": ["are", "were", "sit", "run"], "category": "sv_agreement"},
        {"prompt": "The dog ", "good": ["is", "was", "barks", "runs"], "bad": ["are", "were", "bark", "run"], "category": "sv_agreement"},
        {"prompt": "A bird ", "good": ["is", "was", "flies", "sings"], "bad": ["are", "were", "fly", "sing"], "category": "sv_agreement"},
        {"prompt": "The man ", "good": ["is", "was", "walks", "runs"], "bad": ["are", "were", "walk", "run"], "category": "sv_agreement"},
        {"prompt": "She ", "good": ["is", "was", "has", "does"], "bad": ["are", "were", "have", "do"], "category": "sv_agreement"},

        # Subject-verb agreement (plural)
        {"prompt": "The cats ", "good": ["are", "were", "sit", "run"], "bad": ["is", "was", "sits", "runs"], "category": "sv_agreement"},
        {"prompt": "The dogs ", "good": ["are", "were", "bark", "run"], "bad": ["is", "was", "barks", "runs"], "category": "sv_agreement"},
        {"prompt": "The birds ", "good": ["are", "were", "fly", "sing"], "bad": ["is", "was", "flies", "sings"], "category": "sv_agreement"},
        {"prompt": "They ", "good": ["are", "were", "have", "do"], "bad": ["is", "was", "has", "does"], "category": "sv_agreement"},
        {"prompt": "We ", "good": ["are", "were", "have", "do"], "bad": ["is", "was", "has", "does"], "category": "sv_agreement"},

        # Article selection (a/an)
        {"prompt": "I saw a ", "good": ["cat", "dog", "bird", "man", "book"], "bad": ["apple", "elephant", "orange", "hour"], "category": "article"},
        {"prompt": "I saw an ", "good": ["apple", "elephant", "orange", "animal"], "bad": ["cat", "dog", "bird", "man"], "category": "article"},
        {"prompt": "She is a ", "good": ["woman", "teacher", "doctor", "student"], "bad": ["engineer", "artist", "actor"], "category": "article"},
        {"prompt": "He is an ", "good": ["engineer", "artist", "actor", "adult"], "bad": ["woman", "teacher", "doctor"], "category": "article"},

        # Common collocations
        {"prompt": "black and ", "good": ["white"], "bad": ["red", "green", "blue"], "category": "collocation"},
        {"prompt": "up and ", "good": ["down"], "bad": ["left", "right", "over"], "category": "collocation"},
        {"prompt": "day and ", "good": ["night"], "bad": ["morning", "evening", "time"], "category": "collocation"},
        {"prompt": "hot and ", "good": ["cold"], "bad": ["warm", "cool", "wet"], "category": "collocation"},

        # Basic completion
        {"prompt": "The sun rises in the ", "good": ["morning", "east"], "bad": ["night", "west", "afternoon"], "category": "completion"},
        {"prompt": "Water is ", "good": ["wet", "cold", "clear", "blue"], "bad": ["dry", "hard", "solid"], "category": "completion"},
    ],

    "fr": [
        # Gender agreement (masculine)
        {"prompt": "Le chat ", "good": ["est", "était", "noir", "petit"], "bad": ["sont", "étaient", "noire", "petite"], "category": "gender"},
        {"prompt": "Le chien ", "good": ["est", "était", "noir", "grand"], "bad": ["sont", "étaient", "noire", "grande"], "category": "gender"},
        {"prompt": "Un homme ", "good": ["est", "était", "grand", "petit"], "bad": ["sont", "étaient", "grande", "petite"], "category": "gender"},
        {"prompt": "Le livre ", "good": ["est", "était", "petit", "nouveau"], "bad": ["sont", "étaient", "petite", "nouvelle"], "category": "gender"},
        {"prompt": "Il ", "good": ["est", "était", "a", "fait"], "bad": ["sont", "étaient", "ont", "font"], "category": "gender"},

        # Gender agreement (feminine)
        {"prompt": "La maison ", "good": ["est", "était", "grande", "belle"], "bad": ["sont", "étaient", "grand", "beau"], "category": "gender"},
        {"prompt": "La femme ", "good": ["est", "était", "grande", "belle"], "bad": ["sont", "étaient", "grand", "beau"], "category": "gender"},
        {"prompt": "Une fille ", "good": ["est", "était", "petite", "belle"], "bad": ["sont", "étaient", "petit", "beau"], "category": "gender"},
        {"prompt": "La table ", "good": ["est", "était", "grande", "petite"], "bad": ["sont", "étaient", "grand", "petit"], "category": "gender"},
        {"prompt": "Elle ", "good": ["est", "était", "a", "fait"], "bad": ["sont", "étaient", "ont", "font"], "category": "gender"},

        # Number agreement (plural)
        {"prompt": "Les chats ", "good": ["sont", "étaient", "noirs", "petits"], "bad": ["est", "était", "noir", "petit"], "category": "number"},
        {"prompt": "Les maisons ", "good": ["sont", "étaient", "grandes", "belles"], "bad": ["est", "était", "grande", "belle"], "category": "number"},
        {"prompt": "Ils ", "good": ["sont", "étaient", "ont", "font"], "bad": ["est", "était", "a", "fait"], "category": "number"},
        {"prompt": "Elles ", "good": ["sont", "étaient", "ont", "font"], "bad": ["est", "était", "a", "fait"], "category": "number"},

        # Article-noun gender
        {"prompt": "Je vois le ", "good": ["chat", "chien", "livre", "garçon"], "bad": ["maison", "femme", "fille", "table"], "category": "article_gender"},
        {"prompt": "Je vois la ", "good": ["maison", "femme", "fille", "table"], "bad": ["chat", "chien", "livre", "garçon"], "category": "article_gender"},

        # Common collocations
        {"prompt": "noir et ", "good": ["blanc"], "bad": ["rouge", "vert", "bleu"], "category": "collocation"},
        {"prompt": "jour et ", "good": ["nuit"], "bad": ["matin", "soir", "temps"], "category": "collocation"},
        {"prompt": "chaud et ", "good": ["froid"], "bad": ["tiède", "frais", "sec"], "category": "collocation"},

        # Basic completion
        {"prompt": "Le soleil se lève le ", "good": ["matin"], "bad": ["soir", "nuit", "jour"], "category": "completion"},
    ],
}


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiHeadAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def __call__(self, x, mask=None):
        x = x + self.attention(self.ln1(x), self.ln1(x), self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size=50257):
        super().__init__()
        cfg = dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072)
        self.embed = nn.Embedding(vocab_size, cfg["d_model"])
        self.blocks = [TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["d_ff"])
                       for _ in range(cfg["n_layers"])]
        self.ln_f = nn.LayerNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], vocab_size, bias=False)

    def __call__(self, x):
        B, T = x.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)


def load_model(step: int, lang: str):
    """Load model checkpoint."""
    ckpt_dir = CHECKPOINTS_DIR / lang / "125M"
    model_path = ckpt_dir / f"model_{step}.safetensors"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    # Create model and load weights
    model = Transformer()
    weights = mx.load(str(model_path))
    model.update(weights)
    mx.eval(model.parameters())

    return model


def get_token_probability(model, tokenizer, prompt: str, target: str) -> float:
    """Get probability of target token given prompt."""
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt).ids

    # Tokenize target (just first token if multi-token)
    target_ids = tokenizer.encode(" " + target).ids  # space prefix for proper tokenization
    if not target_ids:
        target_ids = tokenizer.encode(target).ids
    if not target_ids:
        return 0.0
    target_id = target_ids[0]

    # Get model output
    x = mx.array([prompt_ids])
    logits = model(x)

    # Get probability of target token at last position
    last_logits = logits[0, -1, :]
    probs = mx.softmax(last_logits)

    target_prob = float(probs[target_id])
    return target_prob


def evaluate_probe(model, tokenizer, probe: dict) -> dict:
    """Evaluate a single probe - does model prefer good over bad completions?"""
    prompt = probe["prompt"]
    good_words = probe["good"]
    bad_words = probe["bad"]

    # Get probabilities for good and bad completions
    good_probs = [get_token_probability(model, tokenizer, prompt, w) for w in good_words]
    bad_probs = [get_token_probability(model, tokenizer, prompt, w) for w in bad_words]

    avg_good = np.mean(good_probs)
    avg_bad = np.mean(bad_probs)

    # Score: how much does model prefer good over bad?
    # Ratio > 1 means model prefers grammatical options
    ratio = avg_good / (avg_bad + 1e-10)

    # Binary: does best good beat best bad?
    best_good = max(good_probs)
    best_bad = max(bad_probs)
    correct = best_good > best_bad

    return {
        "prompt": prompt,
        "category": probe["category"],
        "avg_good_prob": avg_good,
        "avg_bad_prob": avg_bad,
        "ratio": ratio,
        "correct": correct,
        "best_good": best_good,
        "best_bad": best_bad,
    }


def run_probes(step: int, lang: str):
    """Run all grammar probes for a checkpoint."""
    print(f"Loading model: {lang} step {step}")
    model = load_model(step, lang)
    if model is None:
        return None

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    probes = GRAMMAR_PROBES[lang]
    results = []

    print(f"\nRunning {len(probes)} grammar probes for {lang.upper()}...\n")

    for probe in probes:
        result = evaluate_probe(model, tokenizer, probe)
        results.append(result)

        status = "✓" if result["correct"] else "✗"
        print(f"  {status} [{result['category']:15}] \"{probe['prompt']}\" ratio={result['ratio']:.2f}")

    # Summary by category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0, "ratios": []}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1
        categories[cat]["ratios"].append(r["ratio"])

    print(f"\n{'='*50}")
    print(f"SUMMARY: {lang.upper()} step {step}")
    print(f"{'='*50}")

    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    overall_pct = 100 * total_correct / total

    for cat, data in sorted(categories.items()):
        pct = 100 * data["correct"] / data["total"]
        avg_ratio = np.mean(data["ratios"])
        print(f"  {cat:20}: {data['correct']}/{data['total']} ({pct:5.1f}%) avg_ratio={avg_ratio:.2f}")

    print(f"  {'OVERALL':20}: {total_correct}/{total} ({overall_pct:5.1f}%)")
    print(f"  Mean ratio (good/bad): {np.mean([r['ratio'] for r in results]):.2f}")

    return {
        "step": step,
        "lang": lang,
        "total_correct": total_correct,
        "total": total,
        "accuracy": overall_pct,
        "mean_ratio": float(np.mean([r["ratio"] for r in results])),
        "by_category": {cat: {
            "accuracy": 100 * data["correct"] / data["total"],
            "mean_ratio": float(np.mean(data["ratios"]))
        } for cat, data in categories.items()},
        "details": results,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run python scripts/grammar_probes.py <step> <lang>")
        sys.exit(1)

    step = int(sys.argv[1])
    lang = sys.argv[2]

    if lang not in GRAMMAR_PROBES:
        print(f"Unknown language: {lang}")
        sys.exit(1)

    results = run_probes(step, lang)

    if results:
        # Save results
        output_dir = BASE_DIR / "results" / "grammar_probes"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"grammar_{lang}_{step}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
