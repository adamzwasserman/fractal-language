#!/usr/bin/env python
# fractal-language — Memory-aware resumable training with .npy chunks
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
# Fixed seed for reproducible experiments
RANDOM_SEED = 42
mx.random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"Random seed set to: {RANDOM_SEED}")

# ==================== CONFIG ====================
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"          # "en" or "fr"
MODEL_SIZE = sys.argv[2] if len(sys.argv) > 2 else "125M"  # "125M" or "350M"
DAYTIME_MODE = sys.argv[3] if len(sys.argv) > 3 else "auto"  # "daytime", "nighttime", or "auto"

# Memory limits (GB) - very conservative to prevent crashes
DAYTIME_MEMORY_LIMIT = 8.0
NIGHTTIME_MEMORY_LIMIT = 10.0

# Determine if it's daytime (9 AM - 6 PM)
def is_daytime():
    hour = datetime.now().hour
    return 9 <= hour <= 18

# Set memory limit based on mode (env override for dual training)
if "FRACTAL_MEMORY_LIMIT" in os.environ:
    MEMORY_LIMIT = float(os.environ["FRACTAL_MEMORY_LIMIT"])
elif DAYTIME_MODE == "daytime":
    MEMORY_LIMIT = DAYTIME_MEMORY_LIMIT
elif DAYTIME_MODE == "nighttime":
    MEMORY_LIMIT = NIGHTTIME_MEMORY_LIMIT
else:  # auto
    MEMORY_LIMIT = DAYTIME_MEMORY_LIMIT if is_daytime() else NIGHTTIME_MEMORY_LIMIT

print(f"Memory limit: {MEMORY_LIMIT}GB ({'daytime' if MEMORY_LIMIT == DAYTIME_MEMORY_LIMIT else 'nighttime'} mode)")

# FIXED PARAMETERS FOR CONTROLLED EXPERIMENT - NO VARIATION ALLOWED
cfg = {
    "125M": dict(n_layers=12, d_model=768,  n_heads=12, d_ff=3072,  batch=2,  seq=512),
    "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096, batch=1,   seq=512),
}[MODEL_SIZE]

# NO batch/sequence adjustments - must be identical for both languages
print(f"FIXED EXPERIMENT PARAMS: batch={cfg['batch']}, seq={cfg['seq']} (no memory-based adjustments)")

BATCH = cfg["batch"]
SEQ   = cfg["seq"]
CHECKPOINTS = Path("/Volumes/Misc Backup/fractal/checkpoints") / LANG / MODEL_SIZE
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
CHUNK_DIR = Path("/Volumes/Misc Backup/fractal/chunks")
TOKENIZER = Tokenizer.from_file(str(Path("/Volumes/Misc Backup/fractal") / "joint_tokenizer.json"))
VOCAB = TOKENIZER.get_vocab_size()
EOS = TOKENIZER.token_to_id("<eos>")

# ==================== MEMORY MONITORING ====================
def get_memory_usage_gb():
    """Get current process memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def check_memory_limit():
    """Check if memory usage exceeds limit"""
    usage = get_memory_usage_gb()
    system_memory = psutil.virtual_memory()
    system_usage_pct = system_memory.percent
    
    # Only check process memory - system memory check was too aggressive
    if usage > MEMORY_LIMIT:
        print(f"Stopping: Process={usage:.1f}GB/{MEMORY_LIMIT}GB")
        return True
    return False

# ==================== CHECKPOINT MANAGEMENT ====================
# Keep only the N most recent checkpoints to prevent disk exhaustion
MAX_CHECKPOINTS = 5

def prune_old_checkpoints():
    """Remove old checkpoints, keeping only the most recent MAX_CHECKPOINTS"""
    checkpoints = sorted(CHECKPOINTS.glob("checkpoint_*.json"),
                        key=lambda x: int(x.stem.split('_')[1]))

    if len(checkpoints) <= MAX_CHECKPOINTS:
        return

    # Keep the most recent ones
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
    """Recursively flatten nested parameter dictionaries and lists"""
    flat = {}
    
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            # Nested dictionary (like attention, mlp)
            flat.update(flatten_parameters(value, full_key))
        elif isinstance(value, (list, tuple)):
            # List of modules (like transformer blocks)
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flat.update(flatten_parameters(item, f"{full_key}.{i}"))
                else:
                    flat[f"{full_key}.{i}"] = item
        else:
            # Actual parameter array
            flat[full_key] = value
            
    return flat

def save_checkpoint(step, model, optimizer, loss):
    """Save training checkpoint with proper parameter flattening"""
    try:
        checkpoint_path = CHECKPOINTS / f"checkpoint_{step}.json"
        model_path = CHECKPOINTS / f"model_{step}.safetensors"
        
        # Save checkpoint metadata first (safer)
        checkpoint_meta = {
            "step": step,
            "loss": float(loss.item()) if hasattr(loss, 'item') else float(loss),
            "timestamp": datetime.now().isoformat(),
            "config": cfg,
            "lang": LANG,
            "model_size": MODEL_SIZE,
            "random_seed": RANDOM_SEED,
            "tokenizer_vocab_size": VOCAB,
            "experiment_version": "controlled_comparison_v1"
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_meta, f, indent=2)
        
        # Save model parameters with proper flattening
        try:
            # Ensure parameters are evaluated before saving
            mx.eval(model.parameters())
            # Flatten nested parameters structure
            raw_params = dict(model.parameters())
            flat_params = flatten_parameters(raw_params)
            mx.save_safetensors(str(model_path), flat_params)
            print(f"Checkpoint saved: step {step}, loss {checkpoint_meta['loss']:.4f} ({len(flat_params)} params) → {checkpoint_path.name}")
        except Exception as e:
            print(f"Warning: Failed to save model parameters at step {step}: {e}")
            print(f"Metadata checkpoint still saved: {checkpoint_path.name}")
        
        return checkpoint_path
        
    except Exception as e:
        print(f"Error saving checkpoint at step {step}: {e}")
        return None

def unflatten_parameters(flat_params, target_structure):
    """Reconstruct nested parameter structure from flattened parameters"""
    result = {}
    
    for key, value in target_structure.items():
        if isinstance(value, dict):
            # Nested dictionary - recurse
            result[key] = unflatten_parameters(flat_params, value)
        elif isinstance(value, (list, tuple)):
            # List of modules
            result[key] = []
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    # Get all parameters for this block
                    block_params = {}
                    prefix = f"{key}.{i}."
                    for flat_key, flat_val in flat_params.items():
                        if flat_key.startswith(prefix):
                            nested_key = flat_key[len(prefix):]
                            block_params[nested_key] = flat_val
                    result[key].append(unflatten_parameters(block_params, item))
                else:
                    # Direct parameter
                    param_key = f"{key}.{i}"
                    if param_key in flat_params:
                        if not result[key]:
                            result[key] = []
                        result[key].append(flat_params[param_key])
        else:
            # Direct parameter
            if key in flat_params:
                result[key] = flat_params[key]
    
    return result

def load_checkpoint(model, optimizer):
    """Load latest checkpoint if available"""
    checkpoints = list(CHECKPOINTS.glob("checkpoint_*.json"))
    if not checkpoints:
        print("No checkpoints found, starting from scratch")
        return 0
    
    # Find latest checkpoint
    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest.stem.split('_')[1])
    
    print(f"Loading checkpoint from step {step}...")
    
    # Load model parameters
    model_path = CHECKPOINTS / f"model_{step}.safetensors"
    if model_path.exists():
        flat_params = mx.load(str(model_path))
        print(f"Loaded {len(flat_params)} flattened parameters")
        
        # Get target structure from current model
        target_structure = dict(model.parameters())
        
        # Reconstruct nested structure and update model
        try:
            # For now, just update directly with flattened params
            # The model should be able to handle this
            model.update(flat_params)
            print(f"Model parameters loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model parameters: {e}")
            print("Starting with fresh parameters")
    else:
        print(f"Model file {model_path} not found, starting with fresh parameters")
    
    # Load checkpoint metadata
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
        # Self-attention with residual
        attn_out = self.attention(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        # MLP with residual
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

# Calculate parameter count
total_params = sum(mx.prod(mx.array(p.shape)).item() for p in model.parameters().values() if hasattr(p, 'shape'))
print(f"→ {MODEL_SIZE} {LANG.upper()} | {total_params/1e6:.1f}M params")

# ==================== EXPERIMENT LOGGING ====================
exp_logger = ExperimentLogger(
    experiment_name=f"language_only_hypothesis_{LANG}_{MODEL_SIZE}_gpt2",
    architecture="gpt2",
    lang=LANG,
    model_size=MODEL_SIZE
)
exp_logger.log_config({
    "random_seed": RANDOM_SEED,
    "n_layers": cfg["n_layers"],
    "d_model": cfg["d_model"],
    "n_heads": cfg["n_heads"],
    "d_ff": cfg["d_ff"],
    "batch_size": BATCH,
    "seq_len": SEQ,
    "vocab_size": VOCAB,
    "total_params": total_params,
    "learning_rate": 6e-4,
    "memory_limit_gb": MEMORY_LIMIT,
    "daytime_mode": DAYTIME_MODE,
})

# ==================== DATA STREAM (memory-safe with .npy chunks) ====================
def data_stream():
    """Stream training data from .npy chunks with zero-copy mmap"""
    chunks = sorted(CHUNK_DIR.glob(f"{LANG}_chunk_*.npy"))
    print(f"Found {len(chunks)} chunks for {LANG}")
    
    while True:
        np.random.shuffle(chunks)
        for chunk_path in chunks:
            # Zero-copy load with mmap - never loads full chunk into RAM
            tokens = np.load(chunk_path, mmap_mode='r')
            
            # Generate sequences with 50% overlap for better training
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

# Load checkpoint if available
start_step = load_checkpoint(model, optimizer) + 1
total_steps = 200_000

# Graceful shutdown handler
shutdown_requested = False
def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\nShutdown requested (signal {signum})")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print(f"Training {MODEL_SIZE} on {LANG.upper()} — BATCH={BATCH} SEQ={SEQ} — Memory limit: {MEMORY_LIMIT}GB")
print(f"Starting from step {start_step}")

# Log session start
session_num = 1  # Will be incremented by auto-restart wrapper
exp_logger.log_session_start(session_num, start_step)

pbar = tqdm(range(start_step, total_steps + 1), desc="steps", initial=start_step-1, total=total_steps)
tokens_processed = start_step * BATCH * SEQ
step_start_time = time.time()

session_end_reason = "completed"
for step_idx in pbar:
    if shutdown_requested:
        print(f"Shutdown requested at step {step_idx}")
        session_end_reason = "interrupted"
        break
    elif check_memory_limit():
        print(f"Memory limit reached at step {step_idx}")
        exp_logger.log_memory_warning(step_idx, get_memory_usage_gb(), MEMORY_LIMIT)
        session_end_reason = "memory_limit"
        print("Performing cleanup before exit...")
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
    loss = step(x, y)
    tokens_processed += BATCH * SEQ

    # Progress reporting and memory check
    if step_idx % 50 == 0:
        mx.eval(loss)
        memory_gb = get_memory_usage_gb()
        system_mem = psutil.virtual_memory().percent
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mem': f'{memory_gb:.1f}GB',
            'sys': f'{system_mem:.0f}%'
        })

    # Log training progress every 500 steps
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

    # More frequent checkpointing for auto-restart
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

    # Aggressive memory management
    if step_idx % 100 == 0:
        gc.collect()
        time.sleep(0.01)

    # Check if we need to switch memory mode (daytime/nighttime)
    if step_idx % 10000 == 0 and DAYTIME_MODE == "auto":
        new_limit = DAYTIME_MEMORY_LIMIT if is_daytime() else NIGHTTIME_MEMORY_LIMIT
        if new_limit != MEMORY_LIMIT:
            print(f"Switching memory limit: {MEMORY_LIMIT}GB → {new_limit}GB")
            MEMORY_LIMIT = new_limit

# Final checkpoint and logging
exit_code = 0
try:
    final_loss = loss if 'loss' in locals() else 0.0
    final_loss_val = final_loss.item() if hasattr(final_loss, 'item') else float(final_loss)

    save_checkpoint(step_idx, model, optimizer, final_loss)

    # Log session end
    exp_logger.log_session_end(session_num, step_idx, session_end_reason)

    if step_idx >= total_steps:
        exp_logger.log_experiment_end("completed", step_idx, final_loss_val)
        print("Training complete!")
        exit_code = 0  # True completion
    elif session_end_reason == "memory_limit":
        print("Session ended due to memory limit - checkpoint saved for resuming")
        exit_code = 42  # Memory limit - needs restart
    else:
        print("Training interrupted - checkpoint saved for resuming")
        exit_code = 43  # Interrupted - needs restart

except Exception as e:
    exp_logger.log_error(step_idx, "checkpoint_error", str(e), traceback.format_exc())
    print(f"Error saving final checkpoint: {e}")
    exit_code = 1  # Error

sys.exit(exit_code)
