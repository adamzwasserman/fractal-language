#!/usr/bin/env python3
"""
Experiment 4: Rust as Synthetic Morphology (PyTorch)
=====================================================
Memory-aware resumable training with proper exit codes for auto-restart.

Exit codes:
  0  = Training complete (reached 200k steps)
  42 = Memory limit reached (restart needed)
  43 = Interrupted (restart needed)
  1  = Error

Run on separate GPUs:
  CUDA_VISIBLE_DEVICES=0 python3 train_rust_ddp.py rust
  CUDA_VISIBLE_DEVICES=1 python3 train_rust_ddp.py rust_en
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
import sys
import json
import time
import signal
import gc
import psutil
import traceback
from datetime import datetime
import os

# ==================== SEED CONTROL ====================
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
print(f"Random seed set to: {RANDOM_SEED}")

# ==================== CONFIG ====================
MODE = sys.argv[1] if len(sys.argv) > 1 else "rust"  # "rust" or "rust_en"
MODEL_SIZE = sys.argv[2] if len(sys.argv) > 2 else "125M"

if MODE not in ["rust", "rust_en"]:
    print(f"Error: MODE must be 'rust' or 'rust_en', got '{MODE}'")
    sys.exit(1)

# Memory limit (GPU memory in GB)
GPU_MEMORY_LIMIT = float(os.environ.get("GPU_MEMORY_LIMIT", "11.0"))
print(f"GPU memory limit: {GPU_MEMORY_LIMIT}GB")

# Model configs (same as EN/FR experiments for comparability)
cfg = {
    "125M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072, batch=8, seq=512),
    "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096, batch=4, seq=512),
}[MODEL_SIZE]

BATCH = cfg["batch"]
SEQ = cfg["seq"]

# Paths
BASE_DIR = Path("/workspace/data")
CHECKPOINTS = BASE_DIR / "checkpoints" / MODE / MODEL_SIZE
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
CHUNK_DIR = BASE_DIR / "chunks"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Tokenizer
if MODE == "rust":
    TOKENIZER_PATH = BASE_DIR / "rust_tokenizer.json"
else:
    TOKENIZER_PATH = BASE_DIR / "rust_en_tokenizer.json"

TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB = TOKENIZER.get_vocab_size()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Mode: {MODE} | Device: {device} | Vocab: {VOCAB}")

# ==================== MEMORY MONITORING ====================
def get_gpu_memory_gb():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0

def check_memory_limit():
    """Check if GPU memory exceeds limit"""
    usage = get_gpu_memory_gb()
    if usage > GPU_MEMORY_LIMIT:
        print(f"GPU memory limit reached: {usage:.1f}GB / {GPU_MEMORY_LIMIT}GB")
        return True
    return False

# ==================== RUST STRUCTURAL PROBES ====================
RUST_PROBES = [
    # Lifetime agreement (3 probes)
    ("fn get<'a>(s: &'a str) -> &'a str { s }",
     "fn get<'a>(s: &'a str) -> &'b str { s }",
     "lifetime", "return lifetime matches param"),
    ("struct Ref<'a> { data: &'a str }",
     "struct Ref<'a> { data: &'b str }",
     "lifetime", "struct field lifetime"),
    ("fn first<'a>(x: &'a [i32]) -> &'a i32 { &x[0] }",
     "fn first<'a>(x: &'a [i32]) -> &'b i32 { &x[0] }",
     "lifetime", "slice return lifetime"),

    # Ownership patterns (3 probes)
    ("fn read(v: &Vec<i32>) { println!(\"{:?}\", v); }",
     "fn read(v: &mut Vec<i32>) { println!(\"{:?}\", v); }",
     "ownership", "immutable ref for read"),
    ("fn modify(v: &mut Vec<i32>) { v.push(1); }",
     "fn modify(v: &Vec<i32>) { v.push(1); }",
     "ownership", "mutable ref for modify"),
    ("fn take(s: String) -> String { s }",
     "fn take(s: &String) -> String { s }",
     "ownership", "ownership transfer"),

    # Type consistency (3 probes)
    ("fn identity<T>(x: T) -> T { x }",
     "fn identity<T>(x: T) -> U { x }",
     "type", "generic return matches"),
    ("let v: Vec<i32> = vec![1, 2, 3];",
     "let v: Vec<i32> = vec![\"a\", \"b\"];",
     "type", "vec type matches contents"),
    ("Option::<i32>::Some(42)",
     "Option::<i32>::Some(\"42\")",
     "type", "option type matches"),

    # Borrow checker (2 probes)
    ("let x = 5; let r1 = &x; let r2 = &x;",
     "let mut x = 5; let r1 = &x; let r2 = &mut x;",
     "borrow", "multiple immutable ok"),
    ("fn ret(s: &str) -> &str { s }",
     "fn ret() -> &str { let s = String::new(); &s }",
     "borrow", "no dangling ref"),

    # Mutability (2 probes)
    ("let mut x = 5; x = 10;",
     "let x = 5; x = 10;",
     "mut", "mut for reassign"),
    ("fn inc(x: &mut i32) { *x += 1; }",
     "fn inc(x: &i32) { *x += 1; }",
     "mut", "mut ref for modify"),

    # Expression syntax (2 probes)
    ("fn square(x: i32) -> i32 { x * x }",
     "fn square(x: i32) -> i32 { x * x; }",
     "expr", "no semicolon return"),
    ("fn nothing() { let x = 5; }",
     "fn nothing() { let x = 5 }",
     "expr", "semicolon for stmt"),
]


def compute_sequence_logprob(model, tokenizer, text, device):
    """Compute log probability of a sequence"""
    tokens = tokenizer.encode(text).ids
    if len(tokens) < 2:
        return float('-inf')

    x = torch.tensor(tokens[:-1], device=device).unsqueeze(0)
    y = torch.tensor(tokens[1:], device=device)

    with torch.no_grad():
        logits = model(x)[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[torch.arange(len(y)), y]

    return token_log_probs.sum().item()


def run_rust_probes(model, tokenizer, device):
    """Run all probes and return accuracy by category"""
    model.eval()

    category_correct = {}
    category_total = {}

    for correct, incorrect, category, desc in RUST_PROBES:
        prob_correct = compute_sequence_logprob(model, tokenizer, correct, device)
        prob_incorrect = compute_sequence_logprob(model, tokenizer, incorrect, device)

        is_correct = prob_correct > prob_incorrect

        if category not in category_correct:
            category_correct[category] = 0
            category_total[category] = 0
        category_total[category] += 1
        if is_correct:
            category_correct[category] += 1

    total_correct = sum(category_correct.values())
    total_probes = sum(category_total.values())
    accuracy = total_correct / total_probes if total_probes > 0 else 0

    by_category = {
        cat: category_correct[cat] / category_total[cat]
        for cat in category_total
    }

    model.train()
    return accuracy, by_category


# ==================== DATASET ====================
class ChunkDataset(Dataset):
    def __init__(self, chunk_dir, mode, seq_len):
        self.seq_len = seq_len
        pattern = f"{mode}_chunk_*.npy"
        self.chunks = sorted(Path(chunk_dir).glob(pattern))
        print(f"Found {len(self.chunks)} chunks for {mode}")

        if len(self.chunks) == 0:
            raise ValueError(f"No chunks found matching {pattern}")

        # Load chunks into memory
        self.all_tokens = []
        for chunk_path in tqdm(self.chunks[:100], desc="Loading chunks"):
            tokens = np.load(chunk_path)
            self.all_tokens.append(tokens)
        self.all_tokens = np.concatenate(self.all_tokens)
        print(f"Loaded {len(self.all_tokens):,} tokens")

    def __len__(self):
        return (len(self.all_tokens) - self.seq_len - 1) // (self.seq_len // 2)

    def __getitem__(self, idx):
        start = idx * (self.seq_len // 2)
        end = start + self.seq_len + 1
        tokens = self.all_tokens[start:end]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# ==================== MODEL ====================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None):
        # Pre-norm attention
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out
        # Pre-norm MLP
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # Tie weights

        # Register causal mask buffer
        self.register_buffer("causal_mask", None)

    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.embed(x) + self.pos_embed(pos)

        mask = self._get_causal_mask(T, x.device)
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.head(x)


# ==================== CHECKPOINT MANAGEMENT ====================
MAX_CHECKPOINTS = 9999  # DISABLED - never delete

def prune_old_checkpoints():
    """Remove old checkpoints, keeping only the most recent"""
    checkpoints = sorted(CHECKPOINTS.glob("checkpoint_*.pt"),
                        key=lambda x: int(x.stem.split('_')[1]))
    if len(checkpoints) <= MAX_CHECKPOINTS:
        return

    for old_ckpt in checkpoints[:-MAX_CHECKPOINTS]:
        step = int(old_ckpt.stem.split('_')[1])
        try:
            old_ckpt.unlink()
            meta = CHECKPOINTS / f"checkpoint_{step}.json"
            if meta.exists():
                meta.unlink()
            print(f"Pruned old checkpoint: step {step}")
        except Exception as e:
            print(f"Warning: Failed to prune checkpoint {step}: {e}")


def save_checkpoint(step, model, optimizer, loss, probe_accuracy=None, probe_by_category=None):
    """Save training checkpoint"""
    checkpoint_path = CHECKPOINTS / f"checkpoint_{step}.pt"
    meta_path = CHECKPOINTS / f"checkpoint_{step}.json"

    # Save model state (exclude causal_mask buffer)
    state_dict = {k: v for k, v in model.state_dict().items() if 'causal_mask' not in k}

    torch.save({
        "step": step,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)

    meta = {
        "step": step,
        "loss": float(loss),
        "timestamp": datetime.now().isoformat(),
        "mode": MODE,
        "model_size": MODEL_SIZE,
        "random_seed": RANDOM_SEED,
        "probe_accuracy": probe_accuracy,
        "probe_by_category": probe_by_category,
        "experiment": "rust_synthetic_morphology_v1",
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Checkpoint saved: step {step}, loss {loss:.4f}")


def load_checkpoint(model, optimizer):
    """Load latest checkpoint if available"""
    checkpoints = list(CHECKPOINTS.glob("checkpoint_*.pt"))
    if not checkpoints:
        print("No checkpoints found, starting from scratch")
        return 0

    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest.stem.split('_')[1])

    print(f"Loading checkpoint from step {step}...")
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Load metadata
    meta_path = CHECKPOINTS / f"checkpoint_{step}.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Resuming from step {step}, loss: {meta.get('loss', 'unknown')}")

    return step


# ==================== TRAINING ====================
def main():
    print(f"=== Rust Experiment: {MODE} on {device} ===")

    # Dataset
    dataset = ChunkDataset(CHUNK_DIR, MODE, SEQ)
    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=2)

    # Model
    model = Transformer(
        vocab_size=VOCAB,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        seq_len=SEQ
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params/1e6:.1f}M params")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=6e-4)
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint
    start_step = load_checkpoint(model, optimizer)
    total_steps = 200_000

    # CSV log
    log_file = LOG_DIR / f"{MODE}_{MODEL_SIZE}_training.csv"
    if not log_file.exists() or start_step == 0:
        with open(log_file, 'w') as f:
            f.write("step,loss,probe_accuracy,timestamp\n")

    # Signal handler for graceful shutdown
    shutdown_requested = False
    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        print(f"\nShutdown requested (signal {sig})")
        shutdown_requested = True
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"Training {MODEL_SIZE} on {MODE.upper()} - BATCH={BATCH} SEQ={SEQ}")
    print(f"Starting from step {start_step + 1}")

    # Training loop
    model.train()
    data_iter = iter(dataloader)
    step = start_step
    running_loss = 0
    session_end_reason = "completed"

    pbar = tqdm(range(start_step + 1, total_steps + 1), desc=f"{MODE}", initial=start_step, total=total_steps)

    for step in pbar:
        # Check for shutdown or memory limit
        if shutdown_requested:
            print(f"Shutdown at step {step}")
            session_end_reason = "interrupted"
            break

        if step % 100 == 0 and check_memory_limit():
            print(f"Memory limit at step {step}")
            session_end_reason = "memory_limit"
            break

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, VOCAB), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss = 0.9 * running_loss + 0.1 * loss.item() if running_loss else loss.item()

        if step % 50 == 0:
            gpu_mem = get_gpu_memory_gb()
            pbar.set_postfix({"loss": f"{running_loss:.4f}", "gpu": f"{gpu_mem:.1f}GB"})

        # Log every 500 steps
        if step % 500 == 0:
            with open(log_file, 'a') as f:
                f.write(f"{step},{running_loss:.6f},,{datetime.now().isoformat()}\n")

        # Checkpoint + probes every 1000 steps
        if step % 1000 == 0:
            accuracy, by_cat = run_rust_probes(model, TOKENIZER, device)
            cat_str = " ".join([f"{k}:{v*100:.0f}%" for k, v in by_cat.items()])
            print(f"\n[Step {step}] Probe accuracy: {accuracy*100:.1f}% | {cat_str}")

            with open(log_file, 'a') as f:
                f.write(f"{step},{running_loss:.6f},{accuracy:.4f},{datetime.now().isoformat()}\n")

            save_checkpoint(step, model, optimizer, running_loss, accuracy, by_cat)
            prune_old_checkpoints()

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

    # Determine exit code
    exit_code = 0
    try:
        # Final checkpoint
        if step > start_step:
            accuracy, by_cat = run_rust_probes(model, TOKENIZER, device)
            save_checkpoint(step, model, optimizer, running_loss, accuracy, by_cat)

        if step >= total_steps:
            print(f"\n=== Training COMPLETE at step {step} ===")
            exit_code = 0
        elif session_end_reason == "memory_limit":
            print(f"\n=== Memory limit reached at step {step} - checkpoint saved ===")
            exit_code = 42
        else:
            print(f"\n=== Training interrupted at step {step} - checkpoint saved ===")
            exit_code = 43

    except Exception as e:
        print(f"Error saving final checkpoint: {e}")
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
