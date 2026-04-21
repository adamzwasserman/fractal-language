#!/usr/bin/env python
"""
Rust Structural Probes for Experiment 4
========================================

Analogous to grammar probes for French, these test whether the model
has learned Rust's structural patterns:

1. Lifetime Agreement - do lifetime annotations match across signatures?
2. Ownership Patterns - correct use of &, &mut, ownership transfer
3. Type Consistency - proper generic instantiation
4. Borrow Checker Patterns - patterns that would pass/fail the borrow checker

Each probe is a minimal pair: one correct, one incorrect.
The model should assign higher probability to the correct version.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import json
import sys

# Probe categories with minimal pairs
# Format: (correct, incorrect, category, description)
RUST_PROBES = [
    # === LIFETIME AGREEMENT ===
    # Lifetime must match between parameter and return
    ("fn get<'a>(s: &'a str) -> &'a str { s }",
     "fn get<'a>(s: &'a str) -> &'b str { s }",
     "lifetime_agreement", "lifetime in return matches parameter"),

    ("fn first<'a>(x: &'a [i32]) -> &'a i32 { &x[0] }",
     "fn first<'a>(x: &'a [i32]) -> &'b i32 { &x[0] }",
     "lifetime_agreement", "lifetime in slice return"),

    ("struct Ref<'a> { data: &'a str }",
     "struct Ref<'a> { data: &'b str }",
     "lifetime_agreement", "lifetime in struct field"),

    ("impl<'a> Ref<'a> { fn get(&self) -> &'a str { self.data } }",
     "impl<'a> Ref<'a> { fn get(&self) -> &'b str { self.data } }",
     "lifetime_agreement", "lifetime in impl block"),

    # === OWNERSHIP PATTERNS ===
    # Immutable reference vs mutable reference
    ("fn read(v: &Vec<i32>) { println!(\"{:?}\", v); }",
     "fn read(v: &mut Vec<i32>) { println!(\"{:?}\", v); }",
     "ownership", "immutable ref for read-only"),

    ("fn modify(v: &mut Vec<i32>) { v.push(1); }",
     "fn modify(v: &Vec<i32>) { v.push(1); }",
     "ownership", "mutable ref for modification"),

    # Move vs borrow
    ("fn use_string(s: &String) { println!(\"{}\", s); }",
     "fn use_string(s: String) { println!(\"{}\", s); }",
     "ownership", "borrow when ownership not needed"),

    ("fn take_ownership(s: String) -> String { s }",
     "fn take_ownership(s: &String) -> String { s }",
     "ownership", "take ownership to return"),

    # === TYPE CONSISTENCY ===
    # Generic types must match
    ("fn identity<T>(x: T) -> T { x }",
     "fn identity<T>(x: T) -> U { x }",
     "type_consistency", "generic return matches param"),

    ("let v: Vec<i32> = vec![1, 2, 3];",
     "let v: Vec<i32> = vec![\"a\", \"b\"];",
     "type_consistency", "vec type matches contents"),

    ("fn pair<T>(a: T, b: T) -> (T, T) { (a, b) }",
     "fn pair<T>(a: T, b: U) -> (T, T) { (a, b) }",
     "type_consistency", "pair params same type"),

    ("Option::<i32>::Some(42)",
     "Option::<i32>::Some(\"42\")",
     "type_consistency", "option type matches value"),

    # === BORROW CHECKER PATTERNS ===
    # Cannot have mutable and immutable refs simultaneously
    ("let x = 5; let r1 = &x; let r2 = &x; println!(\"{}{}\", r1, r2);",
     "let mut x = 5; let r1 = &x; let r2 = &mut x; println!(\"{}{}\", r1, r2);",
     "borrow_checker", "multiple immutable refs ok"),

    # Return reference to local (dangling reference)
    ("fn get_ref(s: &str) -> &str { s }",
     "fn get_ref() -> &str { let s = String::new(); &s }",
     "borrow_checker", "no dangling reference"),

    # Move after borrow
    ("let s = String::from(\"hi\"); let r = &s; println!(\"{}\", r);",
     "let s = String::from(\"hi\"); let r = &s; drop(s); println!(\"{}\", r);",
     "borrow_checker", "no use after move"),

    # === TRAIT BOUNDS ===
    ("fn print_debug<T: std::fmt::Debug>(x: T) { println!(\"{:?}\", x); }",
     "fn print_debug<T>(x: T) { println!(\"{:?}\", x); }",
     "trait_bounds", "debug trait required for {:?}"),

    ("fn add<T: std::ops::Add<Output=T>>(a: T, b: T) -> T { a + b }",
     "fn add<T>(a: T, b: T) -> T { a + b }",
     "trait_bounds", "add trait required for +"),

    # === MUT KEYWORD ===
    ("let mut x = 5; x = 10;",
     "let x = 5; x = 10;",
     "mutability", "mut required for reassignment"),

    ("fn inc(x: &mut i32) { *x += 1; }",
     "fn inc(x: &i32) { *x += 1; }",
     "mutability", "mut ref required for modification"),

    # === SEMICOLON / EXPRESSION ===
    ("fn square(x: i32) -> i32 { x * x }",
     "fn square(x: i32) -> i32 { x * x; }",
     "expression", "no semicolon for return value"),

    ("fn nothing() { let x = 5; }",
     "fn nothing() { let x = 5 }",
     "expression", "semicolon for statement"),
]


def load_model_and_tokenizer(mode: str, model_size: str, step: int):
    """Load trained model and tokenizer"""
    base_dir = Path("/Volumes/Misc Backup/fractal")
    checkpoint_dir = base_dir / "checkpoints" / mode / model_size

    # Tokenizer
    if mode == "rust":
        tokenizer_path = base_dir / "rust_tokenizer.json"
    else:
        tokenizer_path = base_dir / "rust_en_tokenizer.json"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    # Model config
    cfg = {
        "125M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072),
        "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096),
    }[model_size]

    # Build model
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
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, cfg["d_model"])
            self.blocks = [TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["d_ff"])
                           for _ in range(cfg["n_layers"])]
            self.ln_f = nn.LayerNorm(cfg["d_model"])
            self.head = nn.Linear(cfg["d_model"], vocab_size, bias=False)
            self.head.weight = self.embed.weight

        def __call__(self, x):
            x = self.embed(x)
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln_f(x))

    model = Transformer()

    # Load weights
    model_path = checkpoint_dir / f"model_{step}.safetensors"
    if model_path.exists():
        params = mx.load(str(model_path))
        model.update(params)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}")

    return model, tokenizer


def compute_sequence_probability(model, tokenizer, text: str) -> float:
    """Compute log probability of a sequence"""
    tokens = tokenizer.encode(text).ids
    if len(tokens) < 2:
        return float('-inf')

    x = mx.array(tokens[:-1])[None, :]  # Input
    y = mx.array(tokens[1:])  # Target

    logits = model(x)[0]  # Remove batch dim
    log_probs = mx.log_softmax(logits, axis=-1)

    # Gather log probs for actual tokens
    token_log_probs = log_probs[mx.arange(len(y)), y]

    return float(mx.sum(token_log_probs).item())


def run_probes(mode: str, model_size: str, step: int):
    """Run all probes and return results"""
    model, tokenizer = load_model_and_tokenizer(mode, model_size, step)
    mx.eval(model.parameters())

    results = {
        "mode": mode,
        "model_size": model_size,
        "step": step,
        "probes": [],
        "summary": {}
    }

    category_correct = {}
    category_total = {}

    for correct, incorrect, category, description in RUST_PROBES:
        prob_correct = compute_sequence_probability(model, tokenizer, correct)
        prob_incorrect = compute_sequence_probability(model, tokenizer, incorrect)

        is_correct = prob_correct > prob_incorrect

        probe_result = {
            "category": category,
            "description": description,
            "correct_text": correct,
            "incorrect_text": incorrect,
            "prob_correct": prob_correct,
            "prob_incorrect": prob_incorrect,
            "model_correct": is_correct,
            "margin": prob_correct - prob_incorrect
        }
        results["probes"].append(probe_result)

        # Track by category
        if category not in category_correct:
            category_correct[category] = 0
            category_total[category] = 0
        category_total[category] += 1
        if is_correct:
            category_correct[category] += 1

    # Summary
    total_correct = sum(category_correct.values())
    total_probes = sum(category_total.values())

    results["summary"] = {
        "total_correct": total_correct,
        "total_probes": total_probes,
        "accuracy": total_correct / total_probes if total_probes > 0 else 0,
        "by_category": {
            cat: {
                "correct": category_correct[cat],
                "total": category_total[cat],
                "accuracy": category_correct[cat] / category_total[cat]
            }
            for cat in category_total
        }
    }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Rust structural probes")
    parser.add_argument("step", type=int, help="Checkpoint step to evaluate")
    parser.add_argument("mode", choices=["rust", "rust_en"], help="Training mode")
    parser.add_argument("--model-size", default="125M", help="Model size")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    print(f"Running Rust probes: {args.mode} {args.model_size} step {args.step}")

    results = run_probes(args.mode, args.model_size, args.step)

    # Print summary
    print(f"\n=== Results ===")
    print(f"Overall: {results['summary']['total_correct']}/{results['summary']['total_probes']} "
          f"({results['summary']['accuracy']*100:.1f}%)")
    print(f"\nBy category:")
    for cat, stats in results["summary"]["by_category"].items():
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({stats['accuracy']*100:.1f}%)")

    # Save if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        # Default output location
        output_dir = Path("/Volumes/Misc Backup/fractal/results")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"rust_probes_{args.mode}_{args.step}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
