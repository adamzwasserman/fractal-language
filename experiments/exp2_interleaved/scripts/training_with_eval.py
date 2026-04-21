#!/usr/bin/env python
"""
Training with integrated evaluation for emergence detection.
Runs smoke tests at each checkpoint, full evaluation after emergence,
and retains checkpoints with interesting results.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np
import os
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
import sys
import gc
import psutil
import json
import time
import signal
import traceback
import re
from datetime import datetime

# ==================== EVALUATION CONFIG ====================
SMOKE_TEST_INTERVAL = 1000      # Run smoke test every N steps
FULL_EVAL_INTERVAL = 1000       # Run full eval every N steps (after emergence)
PERPLEXITY_THRESHOLD = 500      # Smoke test: model is learning
EMERGENCE_THRESHOLD = 150       # Full eval trigger: model is coherent
PROBE_IMPROVEMENT_THRESHOLD = 0.10  # 10% improvement = interesting

# ==================== SEED CONTROL ====================
RANDOM_SEED = 42
mx.random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== CONFIG ====================
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
MODEL_SIZE = sys.argv[2] if len(sys.argv) > 2 else "125M"
DAYTIME_MODE = sys.argv[3] if len(sys.argv) > 3 else "auto"
TARGET_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000

# Memory limits (4GB each for concurrent EN + FR training)
DAYTIME_MEMORY_LIMIT = 4.0
NIGHTTIME_MEMORY_LIMIT = 4.0

def is_daytime():
    hour = datetime.now().hour
    return 9 <= hour <= 18

if "FRACTAL_MEMORY_LIMIT" in os.environ:
    MEMORY_LIMIT = float(os.environ["FRACTAL_MEMORY_LIMIT"])
elif DAYTIME_MODE == "daytime":
    MEMORY_LIMIT = DAYTIME_MEMORY_LIMIT
elif DAYTIME_MODE == "nighttime":
    MEMORY_LIMIT = NIGHTTIME_MEMORY_LIMIT
else:
    MEMORY_LIMIT = DAYTIME_MEMORY_LIMIT if is_daytime() else NIGHTTIME_MEMORY_LIMIT

MEMORY_WARNING = MEMORY_LIMIT * 0.80  # gc.collect at 80%
MEMORY_HARD_LIMIT = MEMORY_LIMIT * 0.95  # exit at 95% to leave headroom
print(f"Memory limit: {MEMORY_LIMIT}GB (warn={MEMORY_WARNING:.1f}GB, hard={MEMORY_HARD_LIMIT:.1f}GB)")

# Model config
cfg = {
    "125M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072, batch=2, seq=512),
    "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096, batch=1, seq=512),
}[MODEL_SIZE]

BATCH = cfg["batch"]
SEQ = cfg["seq"]
BASE_PATH = Path("/Volumes/Misc Backup/fractal")
CHECKPOINTS = BASE_PATH / "checkpoints" / LANG / MODEL_SIZE
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
CHUNK_DIR = BASE_PATH / "chunks"
TOKENIZER = Tokenizer.from_file(str(BASE_PATH / "joint_tokenizer.json"))
VOCAB = TOKENIZER.get_vocab_size()

# Evaluation results storage
EVAL_RESULTS_DIR = BASE_PATH / "eval_results" / LANG / MODEL_SIZE
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== MEMORY MONITORING ====================
def get_memory_usage_gb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def check_memory_and_manage():
    """Check memory and take action. Returns True if should exit."""
    usage = get_memory_usage_gb()

    if usage > MEMORY_HARD_LIMIT:
        print(f"HARD LIMIT: {usage:.2f}GB > {MEMORY_HARD_LIMIT:.1f}GB - exiting")
        return True

    if usage > MEMORY_WARNING:
        # Aggressive cleanup
        gc.collect()
        mx.metal.clear_cache() if hasattr(mx, 'metal') else None
        new_usage = get_memory_usage_gb()
        print(f"WARNING: {usage:.2f}GB > {MEMORY_WARNING:.1f}GB - gc freed {usage-new_usage:.2f}GB")
        if new_usage > MEMORY_HARD_LIMIT:
            print(f"Still over hard limit after gc - exiting")
            return True

    return False

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
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, cfg["d_model"])
        self.blocks = [TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["d_ff"])
                       for _ in range(cfg["n_layers"])]
        self.ln_f = nn.LayerNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], VOCAB, bias=False)
        self.head.weight = self.embed.weight

    def __call__(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

# ==================== EVALUATION FUNCTIONS ====================

def load_validation_sample(num_sequences=100):
    """Load a small validation sample for smoke tests."""
    val_path = BASE_PATH / "validation" / f"{LANG}_validation.json"
    if not val_path.exists():
        # Create validation set if needed
        chunks = sorted(CHUNK_DIR.glob(f"{LANG}_chunk_*.npy"))[-3:]
        sequences = []
        for chunk_path in chunks:
            tokens = np.load(chunk_path, mmap_mode='r')
            for i in range(0, len(tokens) - SEQ, SEQ):
                if len(sequences) >= num_sequences:
                    break
                sequences.append(tokens[i:i+SEQ].tolist())
            if len(sequences) >= num_sequences:
                break
        return sequences

    with open(val_path) as f:
        data = json.load(f)
    return data["sequences"][:num_sequences]

def calculate_perplexity_fast(model, sequences):
    """Fast perplexity calculation on a small sample."""
    total_loss = 0.0

    for seq in sequences[:50]:  # Use only 50 sequences for speed
        x = mx.array(seq[:-1])[None, :]
        y = mx.array(seq[1:])[None, :]
        logits = model(x)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        total_loss += loss.item()

    avg_loss = total_loss / min(50, len(sequences))
    perplexity = float(np.exp(avg_loss))
    return perplexity

# Capability probes (subset for speed)
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

        # Stop at EOS or newline
        if next_token == TOKENIZER.token_to_id("<eos>"):
            break

    return TOKENIZER.decode(generated)

def run_probes(model):
    """Run capability probes and return results."""
    results = {}
    total_correct = 0

    for probe in PROBES:
        prompt = probe[f"prompt_{LANG}"]
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
        except Exception as e:
            results[probe["id"]] = {"error": str(e), "correct": False}

    accuracy = total_correct / len([p for p in PROBES if p[f"prompt_{LANG}"] != "N/A"])
    return results, accuracy

def smoke_test(model, validation_sequences):
    """Fast smoke test: just perplexity."""
    ppl = calculate_perplexity_fast(model, validation_sequences)
    return {
        "perplexity": ppl,
        "passed": ppl < PERPLEXITY_THRESHOLD
    }

def full_evaluation(model, validation_sequences, step):
    """Full evaluation: perplexity + probes."""
    ppl = calculate_perplexity_fast(model, validation_sequences)
    probe_results, accuracy = run_probes(model)

    result = {
        "step": step,
        "perplexity": ppl,
        "probe_accuracy": accuracy,
        "probe_results": probe_results,
        "timestamp": datetime.now().isoformat()
    }

    # Save results
    result_path = EVAL_RESULTS_DIR / f"eval_{step}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result

def is_interesting(current_result, previous_results):
    """Determine if checkpoint is interesting enough to keep."""
    if not previous_results:
        # First result after emergence is always interesting
        return True

    prev = previous_results[-1]

    # Significant accuracy improvement
    if current_result["probe_accuracy"] - prev.get("probe_accuracy", 0) >= PROBE_IMPROVEMENT_THRESHOLD:
        return True

    # Significant perplexity drop (more than 20%)
    if prev.get("perplexity", float('inf')) > 0:
        ppl_improvement = (prev["perplexity"] - current_result["perplexity"]) / prev["perplexity"]
        if ppl_improvement >= 0.20:
            return True

    # New probe passing that wasn't passing before
    for probe_id, result in current_result["probe_results"].items():
        if result.get("correct") and not prev.get("probe_results", {}).get(probe_id, {}).get("correct"):
            return True

    return False

# ==================== CHECKPOINT MANAGEMENT ====================
def flatten_parameters(params, prefix=""):
    flat = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_parameters(value, full_key))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flat.update(flatten_parameters(item, f"{full_key}.{i}"))
                else:
                    flat[f"{full_key}.{i}"] = item
        else:
            flat[full_key] = value
    return flat

def save_checkpoint(step, model, optimizer, loss, keep=True):
    """Save checkpoint."""
    checkpoint_path = CHECKPOINTS / f"checkpoint_{step}.json"
    model_path = CHECKPOINTS / f"model_{step}.safetensors"
    optimizer_path = CHECKPOINTS / f"optimizer_{step}.safetensors"

    checkpoint_meta = {
        "step": step,
        "loss": float(loss.item()) if hasattr(loss, 'item') else float(loss),
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
        "lang": LANG,
        "model_size": MODEL_SIZE,
        "random_seed": RANDOM_SEED,
        "keep": keep  # Flag for pruning
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_meta, f, indent=2)

    mx.eval(model.parameters())
    raw_params = dict(model.parameters())
    flat_params = flatten_parameters(raw_params)
    mx.save_safetensors(str(model_path), flat_params)

    # Save optimizer state (critical for stable restarts)
    if hasattr(optimizer, 'state') and optimizer.state:
        flat_opt = flatten_parameters(dict(optimizer.state))
        mx.save_safetensors(str(optimizer_path), flat_opt)

    print(f"Checkpoint saved: step {step}, loss {checkpoint_meta['loss']:.4f}, keep={keep}")
    return checkpoint_path


def load_checkpoint(model, optimizer):
    """Load latest checkpoint if available."""
    checkpoints = list(CHECKPOINTS.glob("checkpoint_*.json"))
    if not checkpoints:
        return 0, False, []

    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest.stem.split('_')[1])

    model_path = CHECKPOINTS / f"model_{step}.safetensors"
    if model_path.exists():
        flat_params = mx.load(str(model_path))
        model.update(flat_params)
        print(f"Loaded checkpoint: step {step}")

    # Restore optimizer state (critical for stable restarts)
    optimizer_path = CHECKPOINTS / f"optimizer_{step}.safetensors"
    if optimizer_path.exists():
        flat_opt = mx.load(str(optimizer_path))
        optimizer.state.update(flat_opt)
        print(f"Restored optimizer state from step {step}")

    # Restore emergence state from eval results
    eval_files = sorted(EVAL_RESULTS_DIR.glob("eval_*.json"),
                       key=lambda x: int(x.stem.split('_')[1]))
    eval_history = []
    emergence_detected = False

    for ef in eval_files:
        with open(ef) as f:
            result = json.load(f)
            eval_history.append(result)
            if result.get("perplexity", float('inf')) < EMERGENCE_THRESHOLD:
                emergence_detected = True

    if emergence_detected:
        print(f"Restored emergence state: detected=True, {len(eval_history)} eval results")

    return step, emergence_detected, eval_history

# ==================== TRAINING STATE PERSISTENCE ====================
def save_training_state(step, rng_state):
    """Save training state for idempotent restarts."""
    state_path = CHECKPOINTS / "training_state.json"
    state = {
        "step": step,
        "numpy_rng_state": rng_state.tolist() if hasattr(rng_state, 'tolist') else list(rng_state),
        "timestamp": datetime.now().isoformat()
    }
    with open(state_path, 'w') as f:
        json.dump(state, f)

def load_training_state():
    """Load training state if available."""
    state_path = CHECKPOINTS / "training_state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        # Restore numpy RNG state
        if "numpy_rng_state" in state:
            try:
                rng_state = tuple(np.array(x) if isinstance(x, list) else x
                                 for x in state["numpy_rng_state"])
                np.random.set_state(('MT19937', np.array(state["numpy_rng_state"][1]),
                                    state["numpy_rng_state"][2],
                                    state["numpy_rng_state"][3],
                                    state["numpy_rng_state"][4]))
                print(f"Restored RNG state from step {state['step']}")
            except Exception as e:
                print(f"Warning: Could not restore RNG state: {e}")
        return state.get("step", 0)
    return 0

# ==================== DATA STREAM ====================
def data_stream(start_step=0):
    """Deterministic data stream that can resume from a step."""
    chunks = sorted(CHUNK_DIR.glob(f"{LANG}_chunk_*.npy"))
    print(f"Found {len(chunks)} chunks for {LANG}")

    # Calculate how many sequences to skip for resumption
    seqs_per_chunk_approx = 1000000 // (SEQ // 2)  # Approximate

    while True:
        # Use consistent shuffling based on epoch
        epoch_seed = RANDOM_SEED
        rng = np.random.RandomState(epoch_seed)
        chunk_order = list(range(len(chunks)))
        rng.shuffle(chunk_order)

        for chunk_idx in chunk_order:
            chunk_path = chunks[chunk_idx]
            tokens = np.load(chunk_path, mmap_mode='r')
            for i in range(0, len(tokens) - SEQ - 1, SEQ//2):
                yield tokens[i:i+SEQ+1]

# ==================== MAIN TRAINING LOOP ====================
def main():
    global MEMORY_LIMIT

    print(f"=== Training {MODEL_SIZE} {LANG.upper()} with Evaluation ===")

    # Initialize model
    model = Transformer()
    mx.eval(model.parameters())
    optimizer = opt.Adam(learning_rate=6e-4)

    # Load checkpoint and restore state
    start_step, emergence_detected, eval_results_history = load_checkpoint(model, optimizer)
    start_step += 1
    total_steps = TARGET_STEPS

    # Already completed?
    if start_step > total_steps:
        print(f"Training already complete (step {start_step - 1} >= {total_steps})")
        sys.exit(0)

    # Load validation data
    print("Loading validation data...")
    validation_sequences = load_validation_sample(100)

    # Training state
    stream = data_stream()

    # Signal handling
    shutdown_requested = False
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Loss function
    def loss_fn(x, y):
        return mx.mean(nn.losses.cross_entropy(model(x), y))
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def step_fn(x, y):
        loss, grads = loss_and_grad_fn(x, y)
        optimizer.update(model, grads)
        return loss

    print(f"Starting from step {start_step}")
    pbar = tqdm(range(start_step, total_steps + 1), desc="steps", initial=start_step-1, total=total_steps)

    session_end_reason = "completed"

    for step_idx in pbar:
        if shutdown_requested:
            session_end_reason = "interrupted"
            break

        # Check memory every 10 steps (strict monitoring)
        if step_idx % 10 == 0:
            if check_memory_and_manage():
                session_end_reason = "memory_limit"
                break

        # Training step
        batch_x, batch_y = [], []
        for _ in range(BATCH):
            seq = next(stream)
            batch_x.append(mx.array(seq[:-1]))
            batch_y.append(mx.array(seq[1:]))

        x = mx.stack(batch_x)
        y = mx.stack(batch_y)
        loss = step_fn(x, y)

        # Progress update
        if step_idx % 50 == 0:
            mx.eval(loss)
            mem = get_memory_usage_gb()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mem': f'{mem:.1f}GB'})

        # Checkpoint + evaluation
        if step_idx % SMOKE_TEST_INTERVAL == 0:
            mx.eval(loss)

            # Save training state for resumability
            save_training_state(step_idx, np.random.get_state()[1])

            # Always save checkpoint (mark keep=False initially)
            save_checkpoint(step_idx, model, optimizer, loss, keep=False)

            # Smoke test
            smoke_result = smoke_test(model, validation_sequences)
            print(f"  Smoke test: ppl={smoke_result['perplexity']:.1f}, passed={smoke_result['passed']}")

            if smoke_result['perplexity'] < EMERGENCE_THRESHOLD:
                emergence_detected = True

            # Always run full evaluation (probes are valuable pre-emergence too)
            print(f"  Running full evaluation...")
            eval_result = full_evaluation(model, validation_sequences, step_idx)
            print(f"  Full eval: ppl={eval_result['perplexity']:.1f}, probe_acc={eval_result['probe_accuracy']:.1%}")

            # Decide if interesting (only after emergence)
            if emergence_detected and is_interesting(eval_result, eval_results_history):
                # Mark checkpoint to keep
                ckpt_path = CHECKPOINTS / f"checkpoint_{step_idx}.json"
                with open(ckpt_path) as f:
                    meta = json.load(f)
                meta["keep"] = True
                meta["eval_result"] = {
                    "perplexity": eval_result["perplexity"],
                    "probe_accuracy": eval_result["probe_accuracy"]
                }
                with open(ckpt_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                print(f"  ** Checkpoint {step_idx} marked INTERESTING **")

            eval_results_history.append(eval_result)

    # Final checkpoint (always keep)
    save_checkpoint(step_idx, model, optimizer, loss, keep=True)

    # Exit codes
    if step_idx >= total_steps:
        print("Training complete!")
        sys.exit(0)
    elif session_end_reason == "memory_limit":
        print("Memory limit reached")
        sys.exit(42)
    else:
        print("Training interrupted (checkpoints preserved)")
        sys.exit(43)

if __name__ == "__main__":
    main()
