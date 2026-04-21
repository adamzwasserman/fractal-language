#!/usr/bin/env python
# fractal-language — Interleaved EN/FR training for transfer experiment
# Based on train_resumable.py - trains ONE model on alternating EN/FR chunks
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
import csv
from datetime import datetime

# ==================== SEED CONTROL ====================
# Fixed seed for reproducible experiments
RANDOM_SEED = 42
mx.random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"Random seed set to: {RANDOM_SEED}")

# ==================== CONFIG ====================
LANG = "enfr"  # Interleaved English/French
MODEL_SIZE = sys.argv[1] if len(sys.argv) > 1 else "125M"
DAYTIME_MODE = sys.argv[2] if len(sys.argv) > 2 else "auto"
TARGET_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000

# Memory limits (GB)
DAYTIME_MEMORY_LIMIT = 8.0
NIGHTTIME_MEMORY_LIMIT = 10.0

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

print(f"Memory limit: {MEMORY_LIMIT}GB")

# FIXED PARAMETERS FOR CONTROLLED EXPERIMENT
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

# Logging paths
LOG_DIR = BASE_PATH / "logs"
LOG_DIR.mkdir(exist_ok=True)
TRAINING_LOG = LOG_DIR / f"{LANG}_{MODEL_SIZE}_training.csv"
GRAMMAR_LOG_EN = LOG_DIR / f"grammar_probes_{LANG}_en.csv"
GRAMMAR_LOG_FR = LOG_DIR / f"grammar_probes_{LANG}_fr.csv"

# ==================== GRAMMAR PROBES ====================
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
    ],
}

def get_token_probability(model, prompt: str, target: str) -> float:
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
    last_logits = logits[0, -1, :]
    probs = mx.softmax(last_logits)
    mx.eval(probs)

    return float(probs[target_id])

def evaluate_probe(model, probe: dict) -> dict:
    """Evaluate a single probe."""
    prompt = probe["prompt"]
    good_probs = [get_token_probability(model, prompt, w) for w in probe["good"]]
    bad_probs = [get_token_probability(model, prompt, w) for w in probe["bad"]]

    avg_good = np.mean(good_probs)
    avg_bad = np.mean(bad_probs)
    ratio = avg_good / (avg_bad + 1e-10)
    best_good = max(good_probs)
    best_bad = max(bad_probs)
    correct = best_good > best_bad

    return {
        "prompt": prompt,
        "category": probe["category"],
        "correct": correct,
        "ratio": ratio,
    }

def run_grammar_probes(model, step):
    """Run grammar probes for both EN and FR, log results."""
    results = {"en": [], "fr": []}

    for lang in ["en", "fr"]:
        for probe in GRAMMAR_PROBES[lang]:
            result = evaluate_probe(model, probe)
            results[lang].append(result)

    # Calculate accuracies
    en_correct = sum(1 for r in results["en"] if r["correct"])
    en_total = len(results["en"])
    en_accuracy = 100 * en_correct / en_total
    en_mean_ratio = np.mean([r["ratio"] for r in results["en"]])

    fr_correct = sum(1 for r in results["fr"] if r["correct"])
    fr_total = len(results["fr"])
    fr_accuracy = 100 * fr_correct / fr_total
    fr_mean_ratio = np.mean([r["ratio"] for r in results["fr"]])

    timestamp = datetime.now().isoformat()

    # Log EN results
    en_exists = GRAMMAR_LOG_EN.exists()
    with open(GRAMMAR_LOG_EN, 'a', newline='') as f:
        writer = csv.writer(f)
        if not en_exists:
            writer.writerow(["step", "accuracy", "mean_log_ratio", "correct", "total", "timestamp"])
        writer.writerow([step, f"{en_accuracy:.2f}", f"{en_mean_ratio:.4f}", en_correct, en_total, timestamp])

    # Log FR results
    fr_exists = GRAMMAR_LOG_FR.exists()
    with open(GRAMMAR_LOG_FR, 'a', newline='') as f:
        writer = csv.writer(f)
        if not fr_exists:
            writer.writerow(["step", "accuracy", "mean_log_ratio", "correct", "total", "timestamp"])
        writer.writerow([step, f"{fr_accuracy:.2f}", f"{fr_mean_ratio:.4f}", fr_correct, fr_total, timestamp])

    print(f"  Grammar probes: EN={en_accuracy:.1f}% ({en_correct}/{en_total}) | FR={fr_accuracy:.1f}% ({fr_correct}/{fr_total})")

    return {"en_accuracy": en_accuracy, "fr_accuracy": fr_accuracy}

# ==================== MEMORY MONITORING ====================
def get_memory_usage_gb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def check_memory_limit():
    usage = get_memory_usage_gb()
    if usage > MEMORY_LIMIT:
        print(f"Stopping: Process={usage:.1f}GB/{MEMORY_LIMIT}GB")
        return True
    return False

# ==================== CHECKPOINT MANAGEMENT ====================
MAX_CHECKPOINTS = 5

def prune_old_checkpoints():
    checkpoints = sorted(CHECKPOINTS.glob("checkpoint_*.json"),
                        key=lambda x: int(x.stem.split('_')[1]))
    if len(checkpoints) <= MAX_CHECKPOINTS:
        return
    to_delete = checkpoints[:-MAX_CHECKPOINTS]
    for ckpt_json in to_delete:
        step = int(ckpt_json.stem.split('_')[1])
        model_file = CHECKPOINTS / f"model_{step}.safetensors"
        try:
            ckpt_json.unlink()
            if model_file.exists():
                model_file.unlink()
            print(f"Pruned old checkpoint: step {step}")
        except Exception as e:
            print(f"Warning: Failed to prune checkpoint {step}: {e}")

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

def save_checkpoint(step, model, optimizer, loss):
    try:
        checkpoint_path = CHECKPOINTS / f"checkpoint_{step}.json"
        model_path = CHECKPOINTS / f"model_{step}.safetensors"

        checkpoint_meta = {
            "step": step,
            "loss": float(loss.item()) if hasattr(loss, 'item') else float(loss),
            "timestamp": datetime.now().isoformat(),
            "config": cfg,
            "lang": LANG,
            "model_size": MODEL_SIZE,
            "random_seed": RANDOM_SEED,
            "experiment": "interleaved_transfer",
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_meta, f, indent=2)

        mx.eval(model.parameters())
        raw_params = dict(model.parameters())
        flat_params = flatten_parameters(raw_params)
        mx.save_safetensors(str(model_path), flat_params)
        print(f"Checkpoint saved: step {step}, loss {checkpoint_meta['loss']:.4f}")
        return checkpoint_path
    except Exception as e:
        print(f"Error saving checkpoint at step {step}: {e}")
        return None

def load_checkpoint(model, optimizer):
    checkpoints = list(CHECKPOINTS.glob("checkpoint_*.json"))
    if not checkpoints:
        print("No checkpoints found, starting from scratch")
        return 0

    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest.stem.split('_')[1])

    model_path = CHECKPOINTS / f"model_{step}.safetensors"
    if model_path.exists():
        flat_params = mx.load(str(model_path))
        model.update(flat_params)
        print(f"Loaded checkpoint from step {step}")

    with open(latest, 'r') as f:
        checkpoint = json.load(f)
    print(f"Resuming from step {step}, previous loss: {checkpoint.get('loss', 'unknown')}")
    return step

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

model = Transformer()
mx.eval(model.parameters())

total_params = sum(mx.prod(mx.array(p.shape)).item() for p in model.parameters().values() if hasattr(p, 'shape'))
print(f"→ {MODEL_SIZE} ENFR (interleaved) | {total_params/1e6:.1f}M params")

# ==================== INTERLEAVED DATA STREAM ====================
def data_stream():
    """Stream interleaved EN/FR chunks: EN0, FR0, EN1, FR1, ..."""
    en_chunks = sorted(CHUNK_DIR.glob("en_chunk_*.npy"))
    fr_chunks = sorted(CHUNK_DIR.glob("fr_chunk_*.npy"))
    print(f"Found {len(en_chunks)} EN + {len(fr_chunks)} FR chunks")

    # Interleave: EN chunk 0, FR chunk 0, EN chunk 1, FR chunk 1, ...
    interleaved = []
    for i in range(max(len(en_chunks), len(fr_chunks))):
        if i < len(en_chunks):
            interleaved.append(en_chunks[i])
        if i < len(fr_chunks):
            interleaved.append(fr_chunks[i])

    print(f"Interleaved to {len(interleaved)} total chunks")

    while True:
        # Use fixed seed for reproducible shuffling
        rng = np.random.RandomState(RANDOM_SEED)
        chunk_order = list(range(len(interleaved)))
        rng.shuffle(chunk_order)

        for chunk_idx in chunk_order:
            chunk_path = interleaved[chunk_idx]
            tokens = np.load(chunk_path, mmap_mode='r')
            for i in range(0, len(tokens) - SEQ - 1, SEQ//2):
                yield tokens[i:i+SEQ+1]

stream = data_stream()

# ==================== TRAINING LOOP ====================
optimizer = opt.Adam(learning_rate=6e-4)

def loss_fn(x, y):
    return mx.mean(nn.losses.cross_entropy(model(x), y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

def step_fn(x, y):
    loss, grads = loss_and_grad_fn(x, y)
    optimizer.update(model, grads)
    return loss

start_step = load_checkpoint(model, optimizer) + 1
total_steps = TARGET_STEPS

shutdown_requested = False
def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\nShutdown requested (signal {signum})")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print(f"Training {MODEL_SIZE} ENFR (interleaved) — BATCH={BATCH} SEQ={SEQ}")
print(f"Starting from step {start_step}, target {total_steps}")

# Initialize training log
if not TRAINING_LOG.exists():
    with open(TRAINING_LOG, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "memory_gb", "timestamp"])

pbar = tqdm(range(start_step, total_steps + 1), desc="steps", initial=start_step-1, total=total_steps)

session_end_reason = "completed"
for step_idx in pbar:
    if shutdown_requested:
        session_end_reason = "interrupted"
        break
    elif check_memory_limit():
        session_end_reason = "memory_limit"
        gc.collect()
        time.sleep(2)
        break

    # Prepare batch
    batch_x = []
    batch_y = []
    for _ in range(BATCH):
        seq = next(stream)
        batch_x.append(mx.array(seq[:-1]))
        batch_y.append(mx.array(seq[1:]))

    x = mx.stack(batch_x)
    y = mx.stack(batch_y)

    # Training step
    loss = step_fn(x, y)

    # Progress reporting
    if step_idx % 50 == 0:
        mx.eval(loss)
        memory_gb = get_memory_usage_gb()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mem': f'{memory_gb:.1f}GB'})

    # Log training progress
    if step_idx % 500 == 0:
        mx.eval(loss)
        with open(TRAINING_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step_idx, f"{loss.item():.4f}", f"{get_memory_usage_gb():.2f}", datetime.now().isoformat()])

    # Checkpoint + grammar probes
    if step_idx % 1000 == 0:
        mx.eval(loss)
        save_checkpoint(step_idx, model, optimizer, loss)
        prune_old_checkpoints()
        run_grammar_probes(model, step_idx)

    # Memory management
    if step_idx % 100 == 0:
        gc.collect()
        time.sleep(0.01)

# Final checkpoint
exit_code = 0
try:
    final_loss = loss if 'loss' in locals() else 0.0
    save_checkpoint(step_idx, model, optimizer, final_loss)
    run_grammar_probes(model, step_idx)

    if step_idx >= total_steps:
        print("Training complete!")
        exit_code = 0
    elif session_end_reason == "memory_limit":
        print("Session ended due to memory limit")
        exit_code = 42
    else:
        print("Training interrupted")
        exit_code = 43

except Exception as e:
    print(f"Error saving final checkpoint: {e}")
    traceback.print_exc()
    exit_code = 1

sys.exit(exit_code)
