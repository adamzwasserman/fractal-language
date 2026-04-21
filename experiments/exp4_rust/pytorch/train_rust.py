#\!/usr/bin/env python
"""
PyTorch training script for Rust Experiment (Experiment 4).
Tests if English "pollutes" Rust learning like it polluted French.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
import sys
import json
import signal
from datetime import datetime
from safetensors.torch import save_file, load_file

# ==================== SEED ====================
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==================== DEVICE ====================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ==================== CONFIG ====================
MODE = sys.argv[1] if len(sys.argv) > 1 else "rust"  # "rust" or "rust_en"
MODEL_SIZE = sys.argv[2] if len(sys.argv) > 2 else "125M"
TARGET_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000

CONFIGS = {
    "125M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072, batch=16, seq=512),
    "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096, batch=4, seq=512),
}
cfg = CONFIGS[MODEL_SIZE]
BATCH = cfg["batch"]
SEQ = cfg["seq"]

BASE_PATH = Path(os.environ.get("DATA_PATH", "/workspace/data"))
CHECKPOINTS = BASE_PATH / "checkpoints" / MODE / MODEL_SIZE
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
CHUNK_DIR = BASE_PATH / "chunks"

# Tokenizer depends on mode
if MODE == "rust":
    TOKENIZER_PATH = BASE_PATH / "rust_tokenizer.json"
else:
    TOKENIZER_PATH = BASE_PATH / "rust_en_tokenizer.json"

TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB = TOKENIZER.get_vocab_size()

print(f"Mode: {MODE} | Size: {MODEL_SIZE} | Batch: {BATCH} | Vocab: {VOCAB}")

# ==================== MODEL ====================
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

    def forward(self, x):
        ln_x = self.ln1(x)
        T = ln_x.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=ln_x.device), diagonal=1).bool()
        attn_out, _ = self.attn(ln_x, ln_x, ln_x, attn_mask=causal_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq=2048, dropout=0.1):
        super().__init__()
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
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.drop(self.embed(x) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ==================== DATA ====================
def data_stream():
    chunk_pattern = f"{MODE}_chunk_*.npy"
    chunks = sorted(CHUNK_DIR.glob(chunk_pattern))
    print(f"Found {len(chunks)} chunks for {MODE}")
    if len(chunks) == 0:
        raise ValueError(f"No chunks found matching {chunk_pattern} in {CHUNK_DIR}")
    while True:
        rng = np.random.RandomState(RANDOM_SEED)
        chunk_order = list(range(len(chunks)))
        rng.shuffle(chunk_order)
        for chunk_idx in chunk_order:
            tokens = np.load(chunks[chunk_idx], mmap_mode='r')
            for i in range(0, len(tokens) - SEQ - 1, SEQ // 2):
                yield tokens[i:i + SEQ + 1]


# ==================== CHECKPOINTING ====================
def save_checkpoint(step, model, optimizer, loss):
    checkpoint_path = CHECKPOINTS / f"checkpoint_{step}.json"
    model_path = CHECKPOINTS / f"model_{step}.safetensors"
    optimizer_path = CHECKPOINTS / f"optimizer_{step}.pt"

    meta = {"step": step, "loss": float(loss), "timestamp": datetime.now().isoformat(), 
            "config": cfg, "mode": MODE, "model_size": MODEL_SIZE}
    with open(checkpoint_path, 'w') as f:
        json.dump(meta, f, indent=2)

    state = {k: v.clone() for k, v in model.state_dict().items()}
    save_file(state, str(model_path))
    torch.save(optimizer.state_dict(), str(optimizer_path))
    print(f"Checkpoint saved: step {step}, loss {loss:.4f}")


def load_checkpoint(model, optimizer):
    checkpoints = list(CHECKPOINTS.glob("checkpoint_*.json"))
    if not checkpoints:
        return 0
    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest.stem.split('_')[1])
    model_path = CHECKPOINTS / f"model_{step}.safetensors"
    optimizer_path = CHECKPOINTS / f"optimizer_{step}.pt"
    if model_path.exists():
        state_dict = load_file(str(model_path))
        model.load_state_dict(state_dict)
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(str(optimizer_path), weights_only=True))
    print(f"Resumed from step {step}")
    return step


def log_training(step, loss, ppl, mode, base_path):
    log_dir = base_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{mode}.csv"
    file_exists = log_file.exists()
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "loss", "perplexity", "timestamp"])
        writer.writerow([step, f"{loss:.4f}", f"{ppl:.2f}", datetime.now().isoformat()])


# ==================== TRAINING ====================
def main():
    print(f"=== Rust Experiment: {MODE} {MODEL_SIZE} ===")

    model = Transformer(
        vocab_size=VOCAB, d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"]
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
    start_step = load_checkpoint(model, optimizer) + 1

    if start_step > TARGET_STEPS:
        print(f"Already complete (step {start_step-1} >= {TARGET_STEPS})")
        return

    stream = data_stream()

    shutdown = False
    def handler(sig, frame):
        nonlocal shutdown
        shutdown = True
        print("\nShutdown requested...")
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    model.train()
    pbar = tqdm(range(start_step, TARGET_STEPS + 1), initial=start_step-1, total=TARGET_STEPS)
    running_loss = 0.0

    for step in pbar:
        if shutdown:
            save_checkpoint(step-1, model, optimizer, running_loss/50 if running_loss else 0)
            break

        batch_x, batch_y = [], []
        for _ in range(BATCH):
            seq = next(stream)
            batch_x.append(seq[:-1])
            batch_y.append(seq[1:])

        x = torch.tensor(np.array(batch_x), dtype=torch.long, device=DEVICE)
        y = torch.tensor(np.array(batch_y), dtype=torch.long, device=DEVICE)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

        if step % 50 == 0:
            avg_loss = running_loss / 50
            ppl = np.exp(avg_loss)
            log_training(step, avg_loss, ppl, MODE, BASE_PATH)
            mem = torch.cuda.memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0
            pbar.set_postfix({'loss': f'{avg_loss:.3f}', 'ppl': f'{ppl:.1f}', 'mem': f'{mem:.1f}GB'})
            running_loss = 0.0

        if step % 1000 == 0:
            save_checkpoint(step, model, optimizer, loss.item())

    print("Done\!")


if __name__ == "__main__":
    main()
