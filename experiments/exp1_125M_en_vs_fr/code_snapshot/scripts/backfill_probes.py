#!/usr/bin/env python
"""
Backfill probes on existing checkpoints.
Runs grammar/capability probes on all checkpoints that don't have eval results yet.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import json
import re
import sys
from pathlib import Path
from tokenizers import Tokenizer

# ==================== CONFIG ====================
BASE_PATH = Path("/Volumes/Misc Backup/fractal")
TOKENIZER = Tokenizer.from_file(str(BASE_PATH / "joint_tokenizer.json"))
VOCAB = TOKENIZER.get_vocab_size()

cfg = {
    "125M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072, batch=2, seq=512),
    "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096, batch=1, seq=512),
}

# ==================== MODEL ====================
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

    def __call__(self, x):
        attn_out = self.attention(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        return x

class Transformer(nn.Module):
    def __init__(self, model_size="125M"):
        super().__init__()
        c = cfg[model_size]
        self.embed = nn.Embedding(VOCAB, c["d_model"])
        self.blocks = [TransformerBlock(c["d_model"], c["n_heads"], c["d_ff"])
                       for _ in range(c["n_layers"])]
        self.ln_f = nn.LayerNorm(c["d_model"])
        self.head = nn.Linear(c["d_model"], VOCAB, bias=False)
        self.head.weight = self.embed.weight

    def __call__(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

# ==================== PROBES ====================
PROBES = [
    {
        "id": "arithmetic_basic",
        "prompt_en": "Calculate: 47 + 83 = ",
        "prompt_fr": "Calculer : 47 + 83 = ",
        "expected_pattern": r"130",
        "category": "arithmetic"
    },
    {
        "id": "sequence_completion",
        "prompt_en": "Complete the sequence: 2, 4, 8, 16, ",
        "prompt_fr": "Completer la sequence : 2, 4, 8, 16, ",
        "expected_pattern": r"32",
        "category": "pattern"
    },
    {
        "id": "logical_reasoning",
        "prompt_en": "If A is bigger than B, and B is bigger than C, then A is bigger than",
        "prompt_fr": "Si A est plus grand que B, et B est plus grand que C, alors A est plus grand que",
        "expected_pattern": r"C",
        "category": "logic"
    },
    {
        "id": "common_sense",
        "prompt_en": "In winter, water becomes",
        "prompt_fr": "En hiver, l'eau devient",
        "expected_pattern": r"ice|frozen|glace|gel",
        "category": "knowledge"
    },
    {
        "id": "world_knowledge",
        "prompt_en": "The capital of France is",
        "prompt_fr": "La capitale de la France est",
        "expected_pattern": r"Paris",
        "category": "knowledge"
    },
    {
        "id": "negation",
        "prompt_en": "The opposite of hot is",
        "prompt_fr": "Le contraire de chaud est",
        "expected_pattern": r"cold|froid",
        "category": "logic"
    },
    {
        "id": "counting",
        "prompt_en": "Count: one, two, three, four,",
        "prompt_fr": "Compter: un, deux, trois, quatre,",
        "expected_pattern": r"five|cinq|5",
        "category": "arithmetic"
    },
    {
        "id": "gender_agreement_fr",
        "prompt_en": "N/A",
        "prompt_fr": "La grande maison est",
        "expected_pattern": r"belle|blanche|vieille|grande",
        "category": "morphology"
    },
]

# Grammar probes (more detailed)
GRAMMAR_PROBES = {
    "en": [
        {"prompt": "The cat ", "good": ["is", "was", "runs"], "bad": ["are", "were", "run"], "category": "sv_agreement"},
        {"prompt": "The cats ", "good": ["are", "were", "run"], "bad": ["is", "was", "runs"], "category": "sv_agreement"},
        {"prompt": "She ", "good": ["is", "was", "has"], "bad": ["are", "were", "have"], "category": "sv_agreement"},
        {"prompt": "They ", "good": ["are", "were", "have"], "bad": ["is", "was", "has"], "category": "sv_agreement"},
        {"prompt": "I saw a ", "good": ["cat", "dog", "bird"], "bad": ["apple", "elephant", "hour"], "category": "article"},
        {"prompt": "I saw an ", "good": ["apple", "elephant", "hour"], "bad": ["cat", "dog", "bird"], "category": "article"},
        {"prompt": "black and ", "good": ["white"], "bad": ["red", "green", "blue"], "category": "collocation"},
        {"prompt": "up and ", "good": ["down"], "bad": ["left", "right", "over"], "category": "collocation"},
        {"prompt": "day and ", "good": ["night"], "bad": ["morning", "evening"], "category": "collocation"},
        {"prompt": "The sun rises in the ", "good": ["morning", "east"], "bad": ["night", "west"], "category": "world_knowledge"},
    ],
    "fr": [
        {"prompt": "Le chat ", "good": ["est", "était", "noir"], "bad": ["sont", "étaient", "noire"], "category": "gender"},
        {"prompt": "La maison ", "good": ["est", "était", "grande"], "bad": ["sont", "étaient", "grand"], "category": "gender"},
        {"prompt": "Les chats ", "good": ["sont", "étaient", "noirs"], "bad": ["est", "était", "noir"], "category": "number"},
        {"prompt": "Les maisons ", "good": ["sont", "étaient", "grandes"], "bad": ["est", "était", "grande"], "category": "number"},
        {"prompt": "Il ", "good": ["est", "était", "a"], "bad": ["sont", "étaient", "ont"], "category": "gender"},
        {"prompt": "Elle ", "good": ["est", "était", "a"], "bad": ["sont", "étaient", "ont"], "category": "gender"},
        {"prompt": "Je vois le ", "good": ["chat", "chien", "livre"], "bad": ["maison", "table", "femme"], "category": "article_gender"},
        {"prompt": "Je vois la ", "good": ["maison", "table", "femme"], "bad": ["chat", "chien", "livre"], "category": "article_gender"},
        {"prompt": "noir et ", "good": ["blanc"], "bad": ["rouge", "vert"], "category": "collocation"},
        {"prompt": "jour et ", "good": ["nuit"], "bad": ["matin", "soir"], "category": "collocation"},
    ],
}

def generate_text(model, prompt, max_tokens=20):
    """Generate text from a prompt."""
    tokens = TOKENIZER.encode(prompt).ids
    x = mx.array(tokens)[None, :]

    generated = []
    for _ in range(max_tokens):
        logits = model(x)
        next_token = mx.argmax(logits[0, -1]).item()
        generated.append(next_token)
        x = mx.concatenate([x, mx.array([[next_token]])], axis=1)
        if next_token == TOKENIZER.token_to_id("<eos>"):
            break

    return TOKENIZER.decode(generated)

def get_token_prob(model, prompt, target):
    """Get probability of target token given prompt."""
    prompt_ids = TOKENIZER.encode(prompt).ids
    target_ids = TOKENIZER.encode(" " + target).ids
    if not target_ids:
        target_ids = TOKENIZER.encode(target).ids
    if not target_ids:
        return 0.0
    target_id = target_ids[0]

    x = mx.array([prompt_ids])
    logits = model(x)
    probs = mx.softmax(logits[0, -1] / 0.1)  # temperature=0.1 for stability
    return probs[target_id].item()

def run_grammar_probes(model, lang):
    """Run grammar probes."""
    probes = GRAMMAR_PROBES[lang]
    results = []

    for probe in probes:
        good_probs = [get_token_prob(model, probe["prompt"], w) for w in probe["good"]]
        bad_probs = [get_token_prob(model, probe["prompt"], w) for w in probe["bad"]]

        avg_good = np.mean(good_probs)
        avg_bad = np.mean(bad_probs)
        ratio = avg_good / (avg_bad + 1e-10)
        correct = max(good_probs) > max(bad_probs)

        results.append({
            "prompt": probe["prompt"],
            "category": probe["category"],
            "avg_good_prob": float(avg_good),
            "avg_bad_prob": float(avg_bad),
            "ratio": float(ratio),
            "correct": correct,
        })

    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    mean_ratio = np.mean([r["ratio"] for r in results])

    return {
        "accuracy": accuracy,
        "mean_ratio": float(mean_ratio),
        "details": results,
    }

def run_capability_probes(model, lang):
    """Run capability probes."""
    results = {}
    total_correct = 0
    total = 0

    for probe in PROBES:
        prompt = probe[f"prompt_{lang}"]
        if prompt == "N/A":
            continue

        try:
            output = generate_text(model, prompt, max_tokens=15)
            match = bool(re.search(probe["expected_pattern"], output, re.IGNORECASE))
            results[probe["id"]] = {
                "prompt": prompt,
                "output": output,
                "expected": probe["expected_pattern"],
                "correct": match,
                "category": probe["category"]
            }
            if match:
                total_correct += 1
            total += 1
        except Exception as e:
            results[probe["id"]] = {"error": str(e), "correct": False}
            total += 1

    return {
        "accuracy": total_correct / total if total > 0 else 0,
        "details": results,
    }

def load_model(model_path, model_size="125M"):
    """Load model from checkpoint."""
    model = Transformer(model_size)
    flat_params = mx.load(str(model_path))
    model.update(flat_params)
    mx.eval(model.parameters())
    return model

def backfill_checkpoint(lang, model_size, step):
    """Run probes on a single checkpoint."""
    checkpoints_dir = BASE_PATH / "checkpoints" / lang / model_size
    eval_dir = BASE_PATH / "eval_results" / lang / model_size
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoints_dir / f"model_{step}.safetensors"
    eval_path = eval_dir / f"eval_{step}.json"

    if not model_path.exists():
        print(f"  Checkpoint not found: {model_path}")
        return None

    if eval_path.exists():
        print(f"  Already exists: {eval_path}")
        return None

    print(f"  Loading model from step {step}...")
    model = load_model(model_path, model_size)

    print(f"  Running grammar probes...")
    grammar_results = run_grammar_probes(model, lang)

    print(f"  Running capability probes...")
    capability_results = run_capability_probes(model, lang)

    result = {
        "step": step,
        "lang": lang,
        "model_size": model_size,
        "grammar": grammar_results,
        "capability": capability_results,
    }

    with open(eval_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Grammar: {grammar_results['accuracy']:.1%}, Capability: {capability_results['accuracy']:.1%}")
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python backfill_probes.py <lang> [model_size] [start_step]")
        print("  lang: en or fr")
        print("  model_size: 125M (default) or 350M")
        print("  start_step: optional, start from this step (default: 0)")
        sys.exit(1)

    lang = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "125M"
    start_step = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    checkpoints_dir = BASE_PATH / "checkpoints" / lang / model_size

    # Find all checkpoints
    checkpoint_files = list(checkpoints_dir.glob("model_*.safetensors"))
    steps = sorted([int(f.stem.split('_')[1]) for f in checkpoint_files])
    steps = [s for s in steps if s >= start_step]

    print(f"=== Backfilling probes for {lang.upper()} {model_size} ===")
    print(f"Found {len(steps)} checkpoints to process")
    print()

    for i, step in enumerate(steps):
        print(f"[{i+1}/{len(steps)}] Step {step}:")
        try:
            backfill_checkpoint(lang, model_size, step)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

if __name__ == "__main__":
    main()
