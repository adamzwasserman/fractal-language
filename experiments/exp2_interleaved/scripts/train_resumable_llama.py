#!/usr/bin/env python
# fractal-language — LLaMA-style architecture variant
# Uses RoPE, SwiGLU, RMSNorm instead of learned positions, GELU, LayerNorm
# For testing "instrument replaceability" hypothesis

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
import math

# Import experiment logger
from experiment_logger import ExperimentLogger

# ==================== SEED CONTROL ====================
RANDOM_SEED = 42
mx.random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"Random seed set to: {RANDOM_SEED}")

# ==================== CONFIG ====================
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
MODEL_SIZE = sys.argv[2] if len(sys.argv) > 2 else "125M"
DAYTIME_MODE = sys.argv[3] if len(sys.argv) > 3 else "auto"

# Memory limits (GB) - strict to prevent crashes
DAYTIME_MEMORY_LIMIT = 8.0
NIGHTTIME_MEMORY_LIMIT = 10.0

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
else:
    MEMORY_LIMIT = DAYTIME_MEMORY_LIMIT if is_daytime() else NIGHTTIME_MEMORY_LIMIT

print(f"Memory limit: {MEMORY_LIMIT}GB ({'daytime' if MEMORY_LIMIT == DAYTIME_MEMORY_LIMIT else 'nighttime'} mode)")

# Architecture configs - matched parameter counts to GPT-2 style
# SwiGLU uses 2/3 * 4d for hidden dim to match parameter count
cfg = {
    "125M": dict(n_layers=12, d_model=768,  n_heads=12, d_ff=2048,  batch=2, seq=512),  # d_ff reduced for SwiGLU
    "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=2730,  batch=1, seq=512),  # d_ff ~= 4*1024 * 2/3
}[MODEL_SIZE]

print(f"LLAMA-STYLE ARCHITECTURE: batch={cfg['batch']}, seq={cfg['seq']}")
print(f"  RoPE positional encoding, SwiGLU activation, RMSNorm")

BATCH = cfg["batch"]
SEQ = cfg["seq"]
# Separate checkpoint directory for LLaMA architecture
CHECKPOINTS = Path("/Volumes/Misc Backup/fractal/checkpoints") / f"{LANG}_llama" / MODEL_SIZE
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
CHUNK_DIR = Path("/Volumes/Misc Backup/fractal/chunks")
TOKENIZER = Tokenizer.from_file(str(Path("/Volumes/Misc Backup/fractal") / "joint_tokenizer.json"))
VOCAB = TOKENIZER.get_vocab_size()
EOS = TOKENIZER.token_to_id("<eos>")

# ==================== MEMORY MONITORING ====================
def get_memory_usage_gb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def check_memory_limit():
    usage = get_memory_usage_gb()
    system_memory = psutil.virtual_memory()
    system_usage_pct = system_memory.percent

    # Only check process memory - system memory check was too aggressive
    if usage > MEMORY_LIMIT:
        print(f"Stopping: Process={usage:.1f}GB/{MEMORY_LIMIT}GB")
        return True
    return False


# ==================== LLAMA-STYLE COMPONENTS ====================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean centering)"""
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x):
        # RMS = sqrt(mean(x^2))
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.weight * (x / rms)


class RotaryPositionEmbedding:
    """Rotary Position Embedding (RoPE) - applies rotation to q, k based on position"""
    def __init__(self, dims: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.dims = dims
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2).astype(mx.float32) / dims))
        positions = mx.arange(max_seq_len).astype(mx.float32)

        # Shape: (max_seq_len, dims/2)
        freqs = mx.outer(positions, inv_freq)

        # Precompute cos and sin
        self._cos = mx.cos(freqs)
        self._sin = mx.sin(freqs)

    def __call__(self, x, offset: int = 0):
        """Apply rotary embedding to input tensor

        Args:
            x: tensor of shape (batch, seq_len, n_heads, head_dim) or (batch, n_heads, seq_len, head_dim)
            offset: position offset for cached inference
        """
        seq_len = x.shape[1] if x.ndim == 4 else x.shape[2]

        cos = self._cos[offset:offset + seq_len]
        sin = self._sin[offset:offset + seq_len]

        return self._apply_rotary(x, cos, sin)

    def _apply_rotary(self, x, cos, sin):
        """Apply rotation using the rotate-half method"""
        # Split into two halves
        x1 = x[..., :self.dims // 2]
        x2 = x[..., self.dims // 2:]

        # Reshape cos/sin for broadcasting
        # x shape: (batch, seq, heads, head_dim)
        cos = cos[:, None, :]  # (seq, 1, dim/2)
        sin = sin[:, None, :]

        # Apply rotation
        rotated = mx.concatenate([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], axis=-1)

        return rotated


class SwiGLU(nn.Module):
    """SwiGLU activation: Swish(x @ W_gate) * (x @ W_up), then project down

    More expressive than GELU, commonly used in LLaMA/Mistral.
    Uses 3 weight matrices but with reduced hidden dim to match param count.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x):
        # SwiGLU: swish(gate) * up, then project down
        gate = nn.silu(self.w_gate(x))  # silu = swish = x * sigmoid(x)
        up = self.w_up(x)
        return self.w_down(gate * up)


class LlamaAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings"""
    def __init__(self, d_model: int, n_heads: int, rope: RotaryPositionEmbedding):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, mask=None):
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape for multi-head: (batch, seq, n_heads, head_dim)
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to queries and keys
        q = self.rope(q)
        k = self.rope(k)

        # Transpose for attention: (batch, n_heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        # Causal mask
        if mask is None:
            mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
        scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = attn @ v

        # Reshape back: (batch, seq, d_model)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.wo(out)


class LlamaBlock(nn.Module):
    """LLaMA-style transformer block with pre-norm, RoPE attention, SwiGLU"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rope: RotaryPositionEmbedding):
        super().__init__()
        self.attention = LlamaAttention(d_model, n_heads, rope)
        self.feed_forward = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def __call__(self, x):
        # Pre-norm architecture (like LLaMA)
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class LlamaTransformer(nn.Module):
    """LLaMA-style decoder-only transformer"""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, max_seq_len: int = 2048):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)

        # Shared RoPE for all layers
        head_dim = d_model // n_heads
        self.rope = RotaryPositionEmbedding(head_dim, max_seq_len)

        self.blocks = [LlamaBlock(d_model, n_heads, d_ff, self.rope)
                       for _ in range(n_layers)]
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embeddings
        self.head.weight = self.embed.weight

    def __call__(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


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
    """Save training checkpoint"""
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
            "architecture": "llama",  # Mark as LLaMA-style
            "random_seed": RANDOM_SEED,
            "tokenizer_vocab_size": VOCAB,
            "experiment_version": "controlled_comparison_v1"
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_meta, f, indent=2)

        try:
            mx.eval(model.parameters())
            raw_params = dict(model.parameters())
            flat_params = flatten_parameters(raw_params)
            mx.save_safetensors(str(model_path), flat_params)
            print(f"Checkpoint saved: step {step}, loss {checkpoint_meta['loss']:.4f} ({len(flat_params)} params) -> {checkpoint_path.name}")
        except Exception as e:
            print(f"Warning: Failed to save model parameters at step {step}: {e}")

        return checkpoint_path

    except Exception as e:
        print(f"Error saving checkpoint at step {step}: {e}")
        return None

def load_checkpoint(model, optimizer):
    """Load latest checkpoint if available"""
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
        print(f"Loaded {len(flat_params)} flattened parameters")

        try:
            model.update(flat_params)
            print(f"Model parameters loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model parameters: {e}")
            print("Starting with fresh parameters")
    else:
        print(f"Model file {model_path} not found, starting with fresh parameters")

    with open(latest, 'r') as f:
        checkpoint = json.load(f)

    print(f"Resuming from step {step}, previous loss: {checkpoint.get('loss', 'unknown')}")
    return step


# ==================== MODEL INSTANTIATION ====================
model = LlamaTransformer(
    vocab_size=VOCAB,
    d_model=cfg["d_model"],
    n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"],
    d_ff=cfg["d_ff"],
    max_seq_len=cfg["seq"] + 128  # Small buffer
)
mx.eval(model.parameters())

# Calculate parameter count
total_params = sum(mx.prod(mx.array(p.shape)).item() for p in model.parameters().values() if hasattr(p, 'shape'))
print(f"-> {MODEL_SIZE} {LANG.upper()} LLAMA | {total_params/1e6:.1f}M params")

# ==================== EXPERIMENT LOGGING ====================
exp_logger = ExperimentLogger(
    experiment_name=f"language_only_hypothesis_{LANG}_{MODEL_SIZE}_llama",
    architecture="llama",
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
    "architecture_details": {
        "normalization": "RMSNorm",
        "activation": "SwiGLU",
        "position_encoding": "RoPE",
    }
})


# ==================== DATA STREAM ====================
def data_stream():
    """Stream training data from .npy chunks with zero-copy mmap"""
    chunks = sorted(CHUNK_DIR.glob(f"{LANG}_chunk_*.npy"))
    print(f"Found {len(chunks)} chunks for {LANG}")

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

print(f"Training {MODEL_SIZE} LLAMA on {LANG.upper()} - BATCH={BATCH} SEQ={SEQ} - Memory limit: {MEMORY_LIMIT}GB")
print(f"Starting from step {start_step}")

# Log session start
session_num = 1
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

    # Progress reporting
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

    # Checkpointing
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

    # Memory management
    if step_idx % 100 == 0:
        gc.collect()
        time.sleep(0.01)

    # Check for daytime/nighttime switch
    if step_idx % 10000 == 0 and DAYTIME_MODE == "auto":
        new_limit = DAYTIME_MEMORY_LIMIT if is_daytime() else NIGHTTIME_MEMORY_LIMIT
        if new_limit != MEMORY_LIMIT:
            print(f"Switching memory limit: {MEMORY_LIMIT}GB -> {new_limit}GB")
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
