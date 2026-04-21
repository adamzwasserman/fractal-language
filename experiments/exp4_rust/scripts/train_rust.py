#!/usr/bin/env python
"""
Experiment 4: Rust as Synthetic Morphology
==========================================
Test if English "pollutes" code learning like it polluted French.

Hypothesis: Rust provides explicit structural markers (lifetimes, ownership, types)
analogous to French morphology. English may interfere with learning these patterns.

Conditions:
- rust: Rust-only baseline
- rust_en: Rust + English interleaved (test pollution)

Prediction: rust-only will learn structural patterns faster than rust_en.
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
from datetime import datetime

# Import experiment logger
from experiment_logger import ExperimentLogger

# ==================== SEED CONTROL ====================
RANDOM_SEED = 42
mx.random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"Random seed set to: {RANDOM_SEED}")

# ==================== CONFIG ====================
MODE = sys.argv[1] if len(sys.argv) > 1 else "rust"  # "rust" or "rust_en"
MODEL_SIZE = sys.argv[2] if len(sys.argv) > 2 else "125M"
DAYTIME_MODE = sys.argv[3] if len(sys.argv) > 3 else "auto"

# Validate mode
if MODE not in ["rust", "rust_en"]:
    print(f"Error: MODE must be 'rust' or 'rust_en', got '{MODE}'")
    sys.exit(1)

# Memory limits
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

# Model configs (same as EN/FR experiments for comparability)
cfg = {
    "125M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072, batch=2, seq=512),
    "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096, batch=1, seq=512),
}[MODEL_SIZE]

BATCH = cfg["batch"]
SEQ = cfg["seq"]

# Paths
BASE_DIR = Path("/Volumes/Misc Backup/fractal")
CHECKPOINTS = BASE_DIR / "checkpoints" / MODE / MODEL_SIZE
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
CHUNK_DIR = BASE_DIR / "chunks"

# Tokenizer (use appropriate one for mode)
if MODE == "rust":
    TOKENIZER_PATH = BASE_DIR / "rust_tokenizer.json"
else:
    TOKENIZER_PATH = BASE_DIR / "rust_en_tokenizer.json"

TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB = TOKENIZER.get_vocab_size()
EOS = TOKENIZER.token_to_id("<eos>")

print(f"Mode: {MODE} | Tokenizer: {TOKENIZER_PATH.name} | Vocab: {VOCAB}")

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
MAX_CHECKPOINTS = 9999  # DISABLED - never delete

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
            "mode": MODE,
            "model_size": MODEL_SIZE,
            "random_seed": RANDOM_SEED,
            "tokenizer_vocab_size": VOCAB,
            "experiment": "rust_synthetic_morphology_v1"
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_meta, f, indent=2)

        try:
            mx.eval(model.parameters())
            raw_params = dict(model.parameters())
            flat_params = flatten_parameters(raw_params)
            mx.save_safetensors(str(model_path), flat_params)
            print(f"Checkpoint saved: step {step}, loss {checkpoint_meta['loss']:.4f}")
        except Exception as e:
            print(f"Warning: Failed to save model parameters: {e}")

        return checkpoint_path
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return None

def load_checkpoint(model, optimizer):
    checkpoints = list(CHECKPOINTS.glob("checkpoint_*.json"))
    if not checkpoints:
        print("No checkpoints found, starting from scratch")
        return 0

    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest.stem.split('_')[1])

    print(f"Loading checkpoint from step {step}...")

    model_path = CHECKPOINTS / f"model_{step}.safetensors"
    if model_path.exists():
        flat_params = mx.load(str(model_path))
        try:
            model.update(flat_params)
            print(f"Model parameters loaded")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")

    with open(latest, 'r') as f:
        checkpoint = json.load(f)

    print(f"Resuming from step {step}, loss: {checkpoint.get('loss', 'unknown')}")
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
        self.head.weight = self.embed.weight  # tied

    def __call__(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

model = Transformer()
mx.eval(model.parameters())

total_params = sum(mx.prod(mx.array(p.shape)).item() for p in model.parameters().values() if hasattr(p, 'shape'))
print(f"-> {MODEL_SIZE} {MODE.upper()} | {total_params/1e6:.1f}M params")

# ==================== EXPERIMENT LOGGING ====================
exp_logger = ExperimentLogger(
    experiment_name=f"rust_synthetic_morphology_{MODE}_{MODEL_SIZE}",
    architecture="gpt2",
    lang=MODE,
    model_size=MODEL_SIZE
)
exp_logger.log_config({
    "random_seed": RANDOM_SEED,
    "mode": MODE,
    "hypothesis": "Rust provides synthetic morphology; EN may pollute learning",
    **cfg,
    "vocab_size": VOCAB,
    "total_params": total_params,
    "learning_rate": 6e-4,
})

# ==================== DATA STREAM ====================
def data_stream():
    """Stream training data from .npy chunks"""
    # Use mode-specific chunks (rust_chunk_* or rust_en_chunk_*)
    chunk_pattern = f"{MODE}_chunk_*.npy"
    chunks = sorted(CHUNK_DIR.glob(chunk_pattern))
    print(f"Found {len(chunks)} chunks for {MODE}")

    if len(chunks) == 0:
        print(f"ERROR: No chunks found matching {chunk_pattern}")
        print(f"Run: uv run python scripts/create_rust_chunks.py {MODE}")
        sys.exit(1)

    while True:
        np.random.shuffle(chunks)
        for chunk_path in chunks:
            tokens = np.load(chunk_path, mmap_mode='r')
            for i in range(0, len(tokens) - SEQ - 1, SEQ//2):
                yield tokens[i:i+SEQ+1]

stream = data_stream()

# ==================== TRAINING LOOP ====================
optimizer = opt.Adam(learning_rate=6e-4)

def loss_fn(x, y):
    return mx.mean(nn.losses.cross_entropy(model(x), y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

def step(x, y):
    loss, grads = loss_and_grad_fn(x, y)
    optimizer.update(model, grads)
    return loss

start_step = load_checkpoint(model, optimizer) + 1
total_steps = 200_000

shutdown_requested = False
def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\nShutdown requested (signal {signum})")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print(f"Training {MODEL_SIZE} on {MODE.upper()} - BATCH={BATCH} SEQ={SEQ}")
print(f"Starting from step {start_step}")

exp_logger.log_session_start(1, start_step)

pbar = tqdm(range(start_step, total_steps + 1), desc="steps", initial=start_step-1, total=total_steps)
tokens_processed = start_step * BATCH * SEQ
step_start_time = time.time()

session_end_reason = "completed"
for step_idx in pbar:
    if shutdown_requested:
        session_end_reason = "interrupted"
        break
    elif check_memory_limit():
        exp_logger.log_memory_warning(step_idx, get_memory_usage_gb(), MEMORY_LIMIT)
        session_end_reason = "memory_limit"
        gc.collect()
        time.sleep(2)
        break

    batch_x = []
    batch_y = []
    for _ in range(BATCH):
        seq = next(stream)
        batch_x.append(mx.array(seq[:-1]))
        batch_y.append(mx.array(seq[1:]))

    x = mx.stack(batch_x)
    y = mx.stack(batch_y)

    loss = step(x, y)
    tokens_processed += BATCH * SEQ

    if step_idx % 50 == 0:
        mx.eval(loss)
        memory_gb = get_memory_usage_gb()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mem': f'{memory_gb:.1f}GB',
        })

    if step_idx % 500 == 0:
        mx.eval(loss)
        batch_time_ms = (time.time() - step_start_time) * 1000 / 500
        exp_logger.log_training_step(
            step=step_idx,
            loss=loss.item(),
            learning_rate=6e-4,
            tokens_processed=tokens_processed,
            batch_time_ms=batch_time_ms
        )
        step_start_time = time.time()

    if step_idx % 1000 == 0:
        checkpoint_start = time.time()
        save_checkpoint(step_idx, model, optimizer, loss)
        prune_old_checkpoints()
        checkpoint_time = time.time() - checkpoint_start
        exp_logger.log_checkpoint(
            step=step_idx,
            loss=loss.item(),
            checkpoint_path=str(CHECKPOINTS / f"model_{step_idx}.safetensors"),
            save_time_seconds=checkpoint_time
        )

    if step_idx % 100 == 0:
        gc.collect()
        time.sleep(0.01)

# Final checkpoint
exit_code = 0
try:
    final_loss = loss if 'loss' in locals() else 0.0
    final_loss_val = final_loss.item() if hasattr(final_loss, 'item') else float(final_loss)

    save_checkpoint(step_idx, model, optimizer, final_loss)
    exp_logger.log_session_end(1, step_idx, session_end_reason)

    if step_idx >= total_steps:
        exp_logger.log_experiment_end("completed", step_idx, final_loss_val)
        print("Training complete!")
        exit_code = 0
    elif session_end_reason == "memory_limit":
        print("Session ended due to memory limit")
        exit_code = 42
    else:
        print("Training interrupted")
        exit_code = 43

except Exception as e:
    exp_logger.log_error(step_idx, "checkpoint_error", str(e), traceback.format_exc())
    print(f"Error: {e}")
    exit_code = 1

sys.exit(exit_code)
