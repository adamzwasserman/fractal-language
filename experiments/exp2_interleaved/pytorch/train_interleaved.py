#!/usr/bin/env python
"""
Interleaved EN/FR training for transfer experiment.
Trains ONE model on alternating EN/FR chunks (EN0, FR0, EN1, FR1, ...).
Tests if French morphology transfers grammatical signal to English.
Pre-registered: OSF 10.17605/OSF.IO/SJ48B
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
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# ==================== CONFIG ====================
MODEL_SIZE = sys.argv[1] if len(sys.argv) > 1 else "125M"
TARGET_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 2

CONFIGS = {
    "125M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072, seq=512),
    "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096, seq=512),
}
cfg = CONFIGS[MODEL_SIZE]
SEQ = cfg["seq"]

BASE_PATH = Path(os.environ.get("DATA_PATH", "/workspace/data"))
CHUNK_DIR = BASE_PATH / "chunks"
TOKENIZER_PATH = BASE_PATH / "joint_tokenizer.json"

TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB = TOKENIZER.get_vocab_size()

print(f"Config: {MODEL_SIZE} INTERLEAVED (EN+FR), batch={BATCH_SIZE}, seq={SEQ}, vocab={VOCAB}")
print(f"Experiment: Transfer hypothesis - interleaved chunks")

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
        self.d_model = d_model
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
        x = self.ln_f(x)
        return self.head(x)


# ==================== PROBES ====================
TEMPERATURE = 0.1

GRAMMAR_PROBES = {
    "en": [
        {"prompt": "The cat ", "good": ["is", "was", "sits", "runs"], "bad": ["are", "were", "sit", "run"], "category": "sv_agreement"},
        {"prompt": "The dog ", "good": ["is", "was", "barks", "runs"], "bad": ["are", "were", "bark", "run"], "category": "sv_agreement"},
        {"prompt": "She ", "good": ["is", "was", "has", "does"], "bad": ["are", "were", "have", "do"], "category": "sv_agreement"},
        {"prompt": "The cats ", "good": ["are", "were", "sit", "run"], "bad": ["is", "was", "sits", "runs"], "category": "sv_agreement"},
        {"prompt": "They ", "good": ["are", "were", "have", "do"], "bad": ["is", "was", "has", "does"], "category": "sv_agreement"},
        {"prompt": "I saw a ", "good": ["cat", "dog", "bird", "man"], "bad": ["apple", "elephant", "orange"], "category": "article"},
        {"prompt": "I saw an ", "good": ["apple", "elephant", "orange"], "bad": ["cat", "dog", "bird"], "category": "article"},
        {"prompt": "black and ", "good": ["white"], "bad": ["red", "green", "blue"], "category": "collocation"},
        {"prompt": "up and ", "good": ["down"], "bad": ["left", "right", "over"], "category": "collocation"},
        {"prompt": "day and ", "good": ["night"], "bad": ["morning", "evening", "time"], "category": "collocation"},
    ],
    "fr": [
        {"prompt": "Le chat ", "good": ["est", "était", "noir", "petit"], "bad": ["sont", "étaient", "noire", "petite"], "category": "gender"},
        {"prompt": "La maison ", "good": ["est", "était", "grande", "belle"], "bad": ["sont", "étaient", "grand", "beau"], "category": "gender"},
        {"prompt": "Il ", "good": ["est", "était", "a", "fait"], "bad": ["sont", "étaient", "ont", "font"], "category": "gender"},
        {"prompt": "Elle ", "good": ["est", "était", "a", "fait"], "bad": ["sont", "étaient", "ont", "font"], "category": "gender"},
        {"prompt": "Les chats ", "good": ["sont", "étaient", "noirs"], "bad": ["est", "était", "noir"], "category": "number"},
        {"prompt": "Les maisons ", "good": ["sont", "étaient", "grandes"], "bad": ["est", "était", "grande"], "category": "number"},
        {"prompt": "Je vois le ", "good": ["chat", "chien", "livre"], "bad": ["maison", "femme", "table"], "category": "article_gender"},
        {"prompt": "Je vois la ", "good": ["maison", "femme", "table"], "bad": ["chat", "chien", "livre"], "category": "article_gender"},
        {"prompt": "noir et ", "good": ["blanc"], "bad": ["rouge", "vert", "bleu"], "category": "collocation"},
        {"prompt": "jour et ", "good": ["nuit"], "bad": ["matin", "soir", "temps"], "category": "collocation"},
    ],
}


def get_token_probability(model, tokenizer, prompt, target, device):
    prompt_ids = tokenizer.encode(prompt).ids
    target_ids = tokenizer.encode(" " + target).ids
    if not target_ids:
        target_ids = tokenizer.encode(target).ids
    if not target_ids:
        return 0.0
    target_id = target_ids[0]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits[0, -1, :] / TEMPERATURE, dim=-1)
        return probs[target_id].item()


def evaluate_probe(model, tokenizer, probe, device):
    prompt = probe["prompt"]
    good_probs = [get_token_probability(model, tokenizer, prompt, w, device) for w in probe["good"]]
    bad_probs = [get_token_probability(model, tokenizer, prompt, w, device) for w in probe["bad"]]
    avg_good = np.mean(good_probs)
    avg_bad = np.mean(bad_probs)
    log_ratio = np.log(avg_good + 1e-10) - np.log(avg_bad + 1e-10)
    correct = max(good_probs) > max(bad_probs)
    return {
        "prompt": prompt,
        "category": probe["category"],
        "avg_good_prob": avg_good,
        "avg_bad_prob": avg_bad,
        "log_ratio": log_ratio,
        "correct": correct,
    }


def run_grammar_probes(model, tokenizer, lang, step, device):
    model.eval()
    results = [evaluate_probe(model, tokenizer, p, device) for p in GRAMMAR_PROBES[lang]]
    total_correct = sum(1 for r in results if r["correct"])
    accuracy = 100 * total_correct / len(results)
    mean_log_ratio = np.mean([r["log_ratio"] for r in results])
    model.train()
    return {
        "step": step,
        "accuracy": accuracy,
        "mean_log_ratio": mean_log_ratio,
        "correct": total_correct,
        "total": len(results),
        "detailed_results": results,
    }


def log_grammar_results(results, lang, base_path):
    log_dir = base_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"grammar_probes_enfr_{lang}.csv"
    file_exists = log_file.exists()
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "accuracy", "mean_log_ratio", "correct", "total", "timestamp"])
        writer.writerow([
            results["step"],
            f"{results['accuracy']:.2f}",
            f"{results['mean_log_ratio']:.4f}",
            results["correct"],
            results["total"],
            datetime.now().isoformat()
        ])


def log_training_step(step, loss, ppl, base_path):
    log_dir = base_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training_enfr.csv"
    file_exists = log_file.exists()
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "loss", "perplexity", "timestamp"])
        writer.writerow([step, f"{loss:.4f}", f"{ppl:.2f}", datetime.now().isoformat()])


# ==================== INTERLEAVED DATA STREAM ====================
def create_interleaved_stream(chunk_dir, seq_len, seed):
    """Create interleaved EN/FR data stream: EN0, FR0, EN1, FR1, ..."""
    en_chunks = sorted(chunk_dir.glob("en_chunk_*.npy"))
    fr_chunks = sorted(chunk_dir.glob("fr_chunk_*.npy"))
    print(f"Found {len(en_chunks)} EN + {len(fr_chunks)} FR chunks")

    # Interleave: EN0, FR0, EN1, FR1, ...
    interleaved = []
    for i in range(max(len(en_chunks), len(fr_chunks))):
        if i < len(en_chunks):
            interleaved.append(en_chunks[i])
        if i < len(fr_chunks):
            interleaved.append(fr_chunks[i])
    print(f"Interleaved to {len(interleaved)} total chunks")

    def stream():
        while True:
            rng = np.random.RandomState(seed)
            chunk_order = list(range(len(interleaved)))
            rng.shuffle(chunk_order)
            for chunk_idx in chunk_order:
                tokens = np.load(interleaved[chunk_idx], mmap_mode='r')
                for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
                    yield tokens[i:i + seq_len + 1]

    return stream()


# ==================== CHECKPOINTING ====================
def get_checkpoint_dir(base_path, model_size):
    checkpoint_dir = base_path / "checkpoints" / "enfr" / model_size
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(step, model, optimizer, loss, base_path, model_size):
    checkpoint_dir = get_checkpoint_dir(base_path, model_size)
    checkpoint_path = checkpoint_dir / f"checkpoint_{step}.json"
    model_path = checkpoint_dir / f"model_{step}.safetensors"
    optimizer_path = checkpoint_dir / f"optimizer_{step}.pt"

    meta = {
        "step": step,
        "loss": float(loss),
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
        "lang": "enfr",
        "model_size": model_size,
        "batch_size": BATCH_SIZE,
        "experiment": "interleaved_transfer",
        "random_seed": RANDOM_SEED,
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(meta, f, indent=2)

    state = {k: v.clone() for k, v in model.state_dict().items()}
    save_file(state, str(model_path))
    torch.save(optimizer.state_dict(), str(optimizer_path))
    print(f"Checkpoint saved: step {step}")


def load_checkpoint(model, optimizer, base_path, model_size):
    checkpoint_dir = get_checkpoint_dir(base_path, model_size)
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
    if not checkpoints:
        return 0
    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest.stem.split('_')[1])
    model_path = checkpoint_dir / f"model_{step}.safetensors"
    optimizer_path = checkpoint_dir / f"optimizer_{step}.pt"
    if model_path.exists():
        state_dict = load_file(str(model_path))
        model.load_state_dict(state_dict)
        print(f"Loaded model from step {step}")
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(str(optimizer_path), weights_only=True))
        print(f"Loaded optimizer from step {step}")
    return step


# ==================== TRAINING ====================
def main():
    print(f"=== Interleaved Training {MODEL_SIZE} EN+FR ===")
    print(f"Target: {TARGET_STEPS:,} steps, batch={BATCH_SIZE}")
    print(f"Seed: {RANDOM_SEED}")

    # Create ONE model
    print("\nInitializing model...")
    model = Transformer(
        vocab_size=VOCAB, d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"]
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)

    # Load checkpoint if exists
    start_step = load_checkpoint(model, optimizer, BASE_PATH, MODEL_SIZE) + 1

    if start_step > TARGET_STEPS:
        print(f"Training already complete (step {start_step - 1} >= {TARGET_STEPS})")
        return

    # Create interleaved data stream
    stream = create_interleaved_stream(CHUNK_DIR, SEQ, RANDOM_SEED)

    # Signal handling
    shutdown_requested = False
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        print("\nShutdown requested, finishing current step...")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    model.train()
    pbar = tqdm(range(start_step, TARGET_STEPS + 1), desc="Interleaved", initial=start_step - 1, total=TARGET_STEPS)

    running_loss = 0.0
    log_interval = 50
    checkpoint_interval = 1000

    for step in pbar:
        if shutdown_requested:
            avg_loss = running_loss / log_interval if running_loss > 0 else 0
            save_checkpoint(step - 1, model, optimizer, avg_loss, BASE_PATH, MODEL_SIZE)
            break

        # Prepare batch
        batch_x, batch_y = [], []
        for _ in range(BATCH_SIZE):
            seq = next(stream)
            batch_x.append(seq[:-1])
            batch_y.append(seq[1:])

        x = torch.tensor(np.array(batch_x), dtype=torch.long, device=DEVICE)
        y = torch.tensor(np.array(batch_y), dtype=torch.long, device=DEVICE)

        # Forward + backward
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            ppl = np.exp(avg_loss)
            log_training_step(step, avg_loss, ppl, BASE_PATH)

            if DEVICE.type == "cuda":
                mem = torch.cuda.memory_allocated() / 1e9
                pbar.set_postfix({'loss': f'{avg_loss:.3f}', 'ppl': f'{ppl:.1f}', 'mem': f'{mem:.1f}GB'})
            else:
                pbar.set_postfix({'loss': f'{avg_loss:.3f}', 'ppl': f'{ppl:.1f}'})
            running_loss = 0.0

        if step % checkpoint_interval == 0:
            save_checkpoint(step, model, optimizer, loss.item(), BASE_PATH, MODEL_SIZE)

            # Run grammar probes for BOTH languages
            grammar_en = run_grammar_probes(model, TOKENIZER, "en", step, DEVICE)
            grammar_fr = run_grammar_probes(model, TOKENIZER, "fr", step, DEVICE)
            log_grammar_results(grammar_en, "en", BASE_PATH)
            log_grammar_results(grammar_fr, "fr", BASE_PATH)
            print(f"  EN Grammar: {grammar_en['accuracy']:.1f}% | FR Grammar: {grammar_fr['accuracy']:.1f}%")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
