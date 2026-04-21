#!/usr/bin/env python3
"""
Single-language PyTorch training for FR/EN experiment.
Run one language per GPU for better memory utilization.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 train_single.py en 350M 12
  CUDA_VISIBLE_DEVICES=1 python3 train_single.py fr 350M 12
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==================== CONFIG ====================
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
MODEL_SIZE = sys.argv[2] if len(sys.argv) > 2 else "350M"
BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 8
TARGET_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000

CONFIGS = {
    "125M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072),
    "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096),
}

cfg = CONFIGS[MODEL_SIZE]
SEQ_LEN = 512
DATA_PATH = Path(os.environ.get("DATA_PATH", "/workspace/data"))
CHECKPOINT_DIR = DATA_PATH / "checkpoints" / LANG / MODEL_SIZE
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = DATA_PATH / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Tokenizer
TOKENIZER = Tokenizer.from_file(str(DATA_PATH / "joint_tokenizer.json"))
VOCAB_SIZE = TOKENIZER.get_vocab_size()

print(f"Config: {MODEL_SIZE} {LANG.upper()}, batch={BATCH_SIZE}, seq={SEQ_LEN}, vocab={VOCAB_SIZE}")

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
            nn.Dropout(dropout)
        )
        self.register_buffer("mask", None)

    def get_mask(self, seq_len, device):
        if self.mask is None or self.mask.size(0) < seq_len:
            self.mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return self.mask[:seq_len, :seq_len]

    def forward(self, x):
        mask = self.get_mask(x.size(1), x.device)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ==================== DATA ====================
class ChunkDataset:
    def __init__(self, chunk_dir, lang, seq_len):
        self.seq_len = seq_len
        self.chunks = sorted(Path(chunk_dir).glob(f"{lang}_chunk_*.npy"))
        print(f"Found {len(self.chunks)} chunks for {lang}")
        self.current_chunk = None
        self.current_idx = 0
        self.chunk_idx = 0
        np.random.shuffle(self.chunks)

    def get_batch(self, batch_size):
        sequences = []
        while len(sequences) < batch_size:
            if self.current_chunk is None or self.current_idx >= len(self.current_chunk) - self.seq_len - 1:
                self.chunk_idx = (self.chunk_idx + 1) % len(self.chunks)
                self.current_chunk = np.load(self.chunks[self.chunk_idx])
                self.current_idx = 0
                if self.chunk_idx == 0:
                    np.random.shuffle(self.chunks)

            seq = self.current_chunk[self.current_idx:self.current_idx + self.seq_len + 1]
            sequences.append(seq)
            self.current_idx += self.seq_len // 2

        batch = np.stack(sequences)
        x = torch.tensor(batch[:, :-1], dtype=torch.long)
        y = torch.tensor(batch[:, 1:], dtype=torch.long)
        return x, y


# ==================== GRAMMAR PROBES ====================
GRAMMAR_PROBES_EN = [
    ("The cat sits", "The cat sit"),
    ("The cats sit", "The cats sits"),
    ("She walks", "She walk"),
    ("They walk", "They walks"),
    ("He is running", "He are running"),
]

GRAMMAR_PROBES_FR = [
    ("Le chat noir", "Le chat noire"),
    ("La maison blanche", "La maison blanc"),
    ("Les chats noirs", "Les chats noir"),
    ("Une grande maison", "Une grand maison"),
    ("Il est content", "Il est contente"),
]


def run_grammar_probes(model, tokenizer, lang, device):
    model.eval()
    probes = GRAMMAR_PROBES_FR if lang == "fr" else GRAMMAR_PROBES_EN
    correct = 0
    total = 0
    log_ratios = []

    with torch.no_grad():
        for correct_sent, incorrect_sent in probes:
            correct_ids = tokenizer.encode(correct_sent).ids
            incorrect_ids = tokenizer.encode(incorrect_sent).ids

            if len(correct_ids) < 2 or len(incorrect_ids) < 2:
                continue

            # Compute log prob for correct
            x_c = torch.tensor(correct_ids[:-1], device=device).unsqueeze(0)
            y_c = torch.tensor(correct_ids[1:], device=device)
            logits_c = model(x_c)[0]
            log_probs_c = F.log_softmax(logits_c, dim=-1)
            score_c = log_probs_c[range(len(y_c)), y_c].sum().item()

            # Compute log prob for incorrect
            x_i = torch.tensor(incorrect_ids[:-1], device=device).unsqueeze(0)
            y_i = torch.tensor(incorrect_ids[1:], device=device)
            logits_i = model(x_i)[0]
            log_probs_i = F.log_softmax(logits_i, dim=-1)
            score_i = log_probs_i[range(len(y_i)), y_i].sum().item()

            if score_c > score_i:
                correct += 1
            total += 1
            log_ratios.append(score_c - score_i)

    model.train()
    accuracy = correct / total if total > 0 else 0
    mean_lr = np.mean(log_ratios) if log_ratios else 0
    return accuracy, mean_lr


# ==================== CHECKPOINT ====================
def save_checkpoint(model, optimizer, step, loss, lang):
    model_path = CHECKPOINT_DIR / f"model_{step}.safetensors"
    opt_path = CHECKPOINT_DIR / f"optimizer_{step}.pt"
    meta_path = CHECKPOINT_DIR / f"checkpoint_{step}.json"

    # Save model (exclude mask buffers and head.weight which shares memory with tok_emb.weight)
    state_dict = {k: v for k, v in model.state_dict().items()
                  if 'mask' not in k and k != 'head.weight'}
    save_file(state_dict, str(model_path))

    # Save optimizer
    torch.save(optimizer.state_dict(), opt_path)

    # Save metadata
    meta = {
        "step": step,
        "loss": float(loss),
        "lang": lang,
        "model_size": MODEL_SIZE,
        "timestamp": datetime.now().isoformat()
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Checkpoint saved: step {step}")

    # Prune old checkpoints (keep last 5)
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*.json"),
                        key=lambda x: int(x.stem.split('_')[1]))
    for old in checkpoints[:-5]:
        step_num = int(old.stem.split('_')[1])
        old.unlink()
        (CHECKPOINT_DIR / f"model_{step_num}.safetensors").unlink(missing_ok=True)
        (CHECKPOINT_DIR / f"optimizer_{step_num}.pt").unlink(missing_ok=True)


def load_checkpoint(model, optimizer):
    checkpoints = list(CHECKPOINT_DIR.glob("checkpoint_*.json"))
    if not checkpoints:
        return 0

    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest.stem.split('_')[1])

    model_path = CHECKPOINT_DIR / f"model_{step}.safetensors"
    opt_path = CHECKPOINT_DIR / f"optimizer_{step}.pt"

    if model_path.exists():
        state_dict = load_file(str(model_path))
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from step {step}")

    # Skip optimizer loading - incompatible with DataParallel checkpoints
    # Model weights are preserved, optimizer momentum will rebuild
    print(f"Skipping optimizer load (DataParallel incompatible), starting fresh")

    return step


# ==================== TRAINING ====================
def main():
    print(f"=== Training {MODEL_SIZE} {LANG.upper()} ===")

    # Model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=SEQ_LEN
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params/1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)

    # Resume
    start_step = load_checkpoint(model, optimizer)

    # Data
    dataset = ChunkDataset(DATA_PATH / "chunks", LANG, SEQ_LEN)

    # Log file
    log_file = LOG_DIR / f"training_{LANG}_{MODEL_SIZE}.csv"
    if not log_file.exists() or start_step == 0:
        with open(log_file, 'w') as f:
            f.write("step,loss,ppl,grammar_acc,grammar_lr,timestamp\n")

    # Signal handler
    shutdown = False
    def handler(sig, frame):
        nonlocal shutdown
        print("\nShutdown requested...")
        shutdown = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # Training loop
    model.train()
    running_loss = 0
    pbar = tqdm(range(start_step + 1, TARGET_STEPS + 1), desc=LANG.upper(), initial=start_step, total=TARGET_STEPS)

    for step in pbar:
        if shutdown:
            break

        x, y = dataset.get_batch(BATCH_SIZE)
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss = 0.9 * running_loss + 0.1 * loss.item() if running_loss else loss.item()

        if step % 50 == 0:
            ppl = np.exp(running_loss)
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({"loss": f"{running_loss:.3f}", "ppl": f"{ppl:.1f}", "mem": f"{mem:.1f}GB"})

        if step % 500 == 0:
            ppl = np.exp(running_loss)
            with open(log_file, 'a') as f:
                f.write(f"{step},{running_loss:.6f},{ppl:.2f},,,{datetime.now().isoformat()}\n")

        if step % 1000 == 0:
            # Grammar probes
            acc, lr = run_grammar_probes(model, TOKENIZER, LANG, DEVICE)
            ppl = np.exp(running_loss)
            print(f"\n[{step}] Loss: {running_loss:.4f} PPL: {ppl:.1f} Grammar: {acc*100:.1f}%")

            with open(log_file, 'a') as f:
                f.write(f"{step},{running_loss:.6f},{ppl:.2f},{acc:.4f},{lr:.4f},{datetime.now().isoformat()}\n")

            # Checkpoint
            save_checkpoint(model, optimizer, step, running_loss, LANG)

    print(f"\nTraining {'completed' if step >= TARGET_STEPS else 'interrupted'} at step {step}")


if __name__ == "__main__":
    main()
