#!/usr/bin/env python
"""
Dual-language PyTorch training script for Fractal Language experiment.
STREAMING VERSION - streams C4 data directly from HuggingFace.
Trains EN and FR models simultaneously with equal progress.
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
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

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
TARGET_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
BATCH_PER_LANG = int(sys.argv[3]) if len(sys.argv) > 3 else 4

CONFIGS = {
    "125M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072, seq=512),
    "350M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096, seq=512),
}
cfg = CONFIGS[MODEL_SIZE]
SEQ = cfg["seq"]

BASE_PATH = Path(os.environ.get("DATA_PATH", "/workspace/data"))
TOKENIZER_PATH = BASE_PATH / "joint_tokenizer.json"

TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB = TOKENIZER.get_vocab_size()

print(f"Config: {MODEL_SIZE} DUAL STREAMING (EN+FR), batch={BATCH_PER_LANG}/lang, seq={SEQ}, vocab={VOCAB}")

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
        self.head.weight = self.embed.weight  # Weight tying
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


# ==================== STREAMING DATA ====================
class StreamingC4Dataset(IterableDataset):
    """Stream C4 data from HuggingFace for a specific language."""
    def __init__(self, lang, tokenizer, seq_len=512, seed=42):
        self.lang = lang
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        # C4 language codes
        c4_lang = "en" if self.lang == "en" else "fr"

        # Stream C4 with shuffling
        dataset = load_dataset(
            "allenai/c4",
            c4_lang,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)

        token_buffer = []
        for example in dataset:
            # Tokenize
            encoded = self.tokenizer.encode(example["text"])
            token_buffer.extend(encoded.ids)

            # Yield sequences when we have enough
            while len(token_buffer) >= self.seq_len + 1:
                seq = token_buffer[:self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len // 2:]  # Overlap for context
                yield np.array(seq, dtype=np.int64)


def create_streaming_loader(lang, tokenizer, seq_len, batch_size, seed):
    """Create a DataLoader for streaming C4 data."""
    dataset = StreamingC4Dataset(lang, tokenizer, seq_len, seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4
    )


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

EMERGENCE_PROBES = {
    "en": [
        {"prompt": "2 + 2 =", "good": ["4"], "bad": ["3", "5", "6"], "category": "arithmetic"},
        {"prompt": "5 + 3 =", "good": ["8"], "bad": ["7", "9", "6"], "category": "arithmetic"},
        {"prompt": "10 - 4 =", "good": ["6"], "bad": ["5", "7", "14"], "category": "arithmetic"},
        {"prompt": "3 x 3 =", "good": ["9"], "bad": ["6", "8", "12"], "category": "arithmetic"},
        {"prompt": "The capital of France is", "good": ["Paris"], "bad": ["London", "Berlin", "Rome"], "category": "knowledge"},
        {"prompt": "The sun rises in the", "good": ["east"], "bad": ["west", "north", "south"], "category": "knowledge"},
        {"prompt": "Water freezes at", "good": ["zero", "0"], "bad": ["100", "50", "ten"], "category": "knowledge"},
        {"prompt": "The opposite of hot is", "good": ["cold"], "bad": ["warm", "heat", "fire"], "category": "semantic"},
        {"prompt": "The opposite of big is", "good": ["small"], "bad": ["large", "huge", "giant"], "category": "semantic"},
        {"prompt": "Dog is to puppy as cat is to", "good": ["kitten"], "bad": ["dog", "cat", "pet"], "category": "analogy"},
        {"prompt": "King is to queen as man is to", "good": ["woman"], "bad": ["king", "man", "boy"], "category": "analogy"},
    ],
    "fr": [
        {"prompt": "2 + 2 =", "good": ["4"], "bad": ["3", "5", "6"], "category": "arithmetic"},
        {"prompt": "5 + 3 =", "good": ["8"], "bad": ["7", "9", "6"], "category": "arithmetic"},
        {"prompt": "La capitale de la France est", "good": ["Paris"], "bad": ["Londres", "Berlin", "Rome"], "category": "knowledge"},
        {"prompt": "Le soleil se lève à l'", "good": ["est"], "bad": ["ouest", "nord", "sud"], "category": "knowledge"},
        {"prompt": "Le contraire de chaud est", "good": ["froid"], "bad": ["tiède", "chaleur", "feu"], "category": "semantic"},
        {"prompt": "Le contraire de grand est", "good": ["petit"], "bad": ["large", "énorme", "gros"], "category": "semantic"},
        {"prompt": "Chien est à chiot comme chat est à", "good": ["chaton"], "bad": ["chien", "chat", "animal"], "category": "analogy"},
        {"prompt": "Roi est à reine comme homme est à", "good": ["femme"], "bad": ["roi", "homme", "garçon"], "category": "analogy"},
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
        "good_probs": good_probs,
        "bad_probs": bad_probs,
    }


def run_grammar_probes(model, tokenizer, lang, step, device):
    model.eval()
    results = [evaluate_probe(model, tokenizer, p, device) for p in GRAMMAR_PROBES[lang]]
    total_correct = sum(1 for r in results if r["correct"])
    accuracy = 100 * total_correct / len(results)
    mean_log_ratio = np.mean([r["log_ratio"] for r in results])

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0, "log_ratios": []}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1
        categories[cat]["log_ratios"].append(r["log_ratio"])

    model.train()
    return {
        "step": step,
        "accuracy": accuracy,
        "mean_log_ratio": mean_log_ratio,
        "correct": total_correct,
        "total": len(results),
        "categories": categories,
        "detailed_results": results,
    }


def run_emergence_probes(model, tokenizer, lang, step, device):
    model.eval()
    results = [evaluate_probe(model, tokenizer, p, device) for p in EMERGENCE_PROBES[lang]]
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0, "log_ratios": []}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1
        categories[cat]["log_ratios"].append(r["log_ratio"])
    model.train()
    return {"step": step, "categories": categories, "detailed_results": results}


def log_grammar_results(results, lang, base_path):
    log_dir = base_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"grammar_probes_{lang}.csv"
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

    cat_file = log_dir / f"grammar_by_category_{lang}.csv"
    cat_exists = cat_file.exists()
    with open(cat_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not cat_exists:
            writer.writerow(["step", "category", "accuracy", "mean_log_ratio", "correct", "total", "timestamp"])
        for cat, data in results["categories"].items():
            acc = 100 * data["correct"] / data["total"]
            mean_lr = np.mean(data["log_ratios"])
            writer.writerow([
                results["step"], cat, f"{acc:.2f}", f"{mean_lr:.4f}",
                data["correct"], data["total"], datetime.now().isoformat()
            ])

    detail_file = log_dir / f"grammar_detailed_{lang}.csv"
    detail_exists = detail_file.exists()
    with open(detail_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not detail_exists:
            writer.writerow(["step", "prompt", "category", "avg_good_prob", "avg_bad_prob", "log_ratio", "correct", "timestamp"])
        for r in results["detailed_results"]:
            writer.writerow([
                results["step"], r["prompt"].strip(), r["category"],
                f"{r['avg_good_prob']:.6f}", f"{r['avg_bad_prob']:.6f}",
                f"{r['log_ratio']:.4f}", r["correct"], datetime.now().isoformat()
            ])


def log_emergence_results(results, lang, base_path):
    log_dir = base_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"emergence_probes_{lang}.csv"
    file_exists = log_file.exists()
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "category", "accuracy", "mean_log_ratio", "correct", "total", "timestamp"])
        for cat, data in results["categories"].items():
            acc = 100 * data["correct"] / data["total"]
            log_ratio = np.mean(data["log_ratios"])
            writer.writerow([
                results["step"], cat, f"{acc:.2f}", f"{log_ratio:.4f}",
                data["correct"], data["total"], datetime.now().isoformat()
            ])

    detail_file = log_dir / f"emergence_detailed_{lang}.csv"
    detail_exists = detail_file.exists()
    with open(detail_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not detail_exists:
            writer.writerow(["step", "prompt", "category", "avg_good_prob", "avg_bad_prob", "log_ratio", "correct", "timestamp"])
        for r in results["detailed_results"]:
            writer.writerow([
                results["step"], r["prompt"].strip(), r["category"],
                f"{r['avg_good_prob']:.6f}", f"{r['avg_bad_prob']:.6f}",
                f"{r['log_ratio']:.4f}", r["correct"], datetime.now().isoformat()
            ])


def log_training_step(step, en_loss, en_ppl, fr_loss, fr_ppl, base_path):
    log_dir = base_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "training_dual.csv"
    file_exists = log_file.exists()
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "en_loss", "en_ppl", "fr_loss", "fr_ppl", "loss_diff", "timestamp"])
        writer.writerow([
            step, f"{en_loss:.4f}", f"{en_ppl:.2f}", f"{fr_loss:.4f}", f"{fr_ppl:.2f}",
            f"{en_loss - fr_loss:.4f}", datetime.now().isoformat()
        ])

    for lang, loss, ppl in [("en", en_loss, en_ppl), ("fr", fr_loss, fr_ppl)]:
        lang_file = log_dir / f"training_{lang}.csv"
        lang_exists = lang_file.exists()
        with open(lang_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not lang_exists:
                writer.writerow(["step", "loss", "perplexity", "timestamp"])
            writer.writerow([step, f"{loss:.4f}", f"{ppl:.2f}", datetime.now().isoformat()])


# ==================== CHECKPOINTING ====================
def get_checkpoint_dir(lang, base_path, model_size):
    checkpoint_dir = base_path / "checkpoints" / lang / model_size
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(step, model, optimizer, loss, lang, base_path, model_size):
    checkpoint_dir = get_checkpoint_dir(lang, base_path, model_size)
    checkpoint_path = checkpoint_dir / f"checkpoint_{step}.json"
    model_path = checkpoint_dir / f"model_{step}.safetensors"
    optimizer_path = checkpoint_dir / f"optimizer_{step}.pt"

    meta = {
        "step": step,
        "loss": float(loss),
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
        "lang": lang,
        "model_size": model_size,
        "batch_per_lang": BATCH_PER_LANG,
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(meta, f, indent=2)

    state = {k: v.clone() for k, v in model.state_dict().items()}
    save_file(state, str(model_path))
    torch.save(optimizer.state_dict(), str(optimizer_path))


def load_checkpoint(model, optimizer, lang, base_path, model_size):
    checkpoint_dir = get_checkpoint_dir(lang, base_path, model_size)
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
        print(f"Loaded {lang.upper()} model from step {step}")
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(str(optimizer_path), weights_only=True))
        print(f"Loaded {lang.upper()} optimizer from step {step}")
    return step


# ==================== TRAINING ====================
def main():
    print(f"=== Dual Streaming Training {MODEL_SIZE} EN+FR ===")
    print(f"Target: {TARGET_STEPS:,} steps, batch={BATCH_PER_LANG}/lang")
    print("Streaming directly from HuggingFace C4 dataset")

    # Create both models
    print("\nInitializing EN model...")
    model_en = Transformer(
        vocab_size=VOCAB, d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"]
    ).to(DEVICE)

    print("Initializing FR model...")
    model_fr = Transformer(
        vocab_size=VOCAB, d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"]
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model_en.parameters())
    print(f"Parameters per model: {n_params / 1e6:.1f}M")
    print(f"Total parameters: {2 * n_params / 1e6:.1f}M")

    # Create optimizers
    opt_en = torch.optim.AdamW(model_en.parameters(), lr=6e-4, weight_decay=0.01)
    opt_fr = torch.optim.AdamW(model_fr.parameters(), lr=6e-4, weight_decay=0.01)

    # Load checkpoints
    step_en = load_checkpoint(model_en, opt_en, "en", BASE_PATH, MODEL_SIZE)
    step_fr = load_checkpoint(model_fr, opt_fr, "fr", BASE_PATH, MODEL_SIZE)

    start_step = min(step_en, step_fr) + 1
    if step_en != step_fr:
        print(f"WARNING: EN at step {step_en}, FR at step {step_fr}. Starting from {start_step - 1}")

    if start_step > TARGET_STEPS:
        print(f"Training already complete (step {start_step - 1} >= {TARGET_STEPS})")
        return

    # Create streaming data loaders
    print("\nInitializing C4 streaming datasets...")
    loader_en = create_streaming_loader("en", TOKENIZER, SEQ, BATCH_PER_LANG, RANDOM_SEED)
    loader_fr = create_streaming_loader("fr", TOKENIZER, SEQ, BATCH_PER_LANG, RANDOM_SEED)

    iter_en = iter(loader_en)
    iter_fr = iter(loader_fr)
    print("Streaming initialized!")

    # Signal handling
    shutdown_requested = False
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        print("\nShutdown requested, finishing current step...")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    model_en.train()
    model_fr.train()

    pbar = tqdm(range(start_step, TARGET_STEPS + 1), desc="Dual Streaming", initial=start_step - 1, total=TARGET_STEPS)

    running_loss_en = 0.0
    running_loss_fr = 0.0
    log_interval = 50
    checkpoint_interval = 1000

    for step in pbar:
        if shutdown_requested:
            avg_en = running_loss_en / log_interval if running_loss_en > 0 else 0
            avg_fr = running_loss_fr / log_interval if running_loss_fr > 0 else 0
            save_checkpoint(step - 1, model_en, opt_en, avg_en, "en", BASE_PATH, MODEL_SIZE)
            save_checkpoint(step - 1, model_fr, opt_fr, avg_fr, "fr", BASE_PATH, MODEL_SIZE)
            print(f"Saved checkpoints at step {step - 1}")
            break

        # ===== Train EN =====
        try:
            batch_en = next(iter_en)
        except StopIteration:
            iter_en = iter(loader_en)
            batch_en = next(iter_en)

        x_en = batch_en[:, :-1].to(DEVICE)
        y_en = batch_en[:, 1:].to(DEVICE)

        logits_en = model_en(x_en)
        loss_en = F.cross_entropy(logits_en.view(-1, VOCAB), y_en.view(-1))

        opt_en.zero_grad()
        loss_en.backward()
        torch.nn.utils.clip_grad_norm_(model_en.parameters(), 1.0)
        opt_en.step()

        # ===== Train FR =====
        try:
            batch_fr = next(iter_fr)
        except StopIteration:
            iter_fr = iter(loader_fr)
            batch_fr = next(iter_fr)

        x_fr = batch_fr[:, :-1].to(DEVICE)
        y_fr = batch_fr[:, 1:].to(DEVICE)

        logits_fr = model_fr(x_fr)
        loss_fr = F.cross_entropy(logits_fr.view(-1, VOCAB), y_fr.view(-1))

        opt_fr.zero_grad()
        loss_fr.backward()
        torch.nn.utils.clip_grad_norm_(model_fr.parameters(), 1.0)
        opt_fr.step()

        running_loss_en += loss_en.item()
        running_loss_fr += loss_fr.item()

        if step % log_interval == 0:
            avg_loss_en = running_loss_en / log_interval
            avg_loss_fr = running_loss_fr / log_interval
            ppl_en = np.exp(avg_loss_en)
            ppl_fr = np.exp(avg_loss_fr)

            log_training_step(step, avg_loss_en, ppl_en, avg_loss_fr, ppl_fr, BASE_PATH)

            if DEVICE.type == "cuda":
                mem = torch.cuda.memory_allocated() / 1e9
                pbar.set_postfix({
                    'EN': f'{avg_loss_en:.2f}',
                    'FR': f'{avg_loss_fr:.2f}',
                    'diff': f'{avg_loss_en - avg_loss_fr:+.2f}',
                    'mem': f'{mem:.1f}GB'
                })
            else:
                pbar.set_postfix({
                    'EN': f'{avg_loss_en:.2f}',
                    'FR': f'{avg_loss_fr:.2f}',
                    'diff': f'{avg_loss_en - avg_loss_fr:+.2f}'
                })
            running_loss_en = 0.0
            running_loss_fr = 0.0

        if step % checkpoint_interval == 0:
            save_checkpoint(step, model_en, opt_en, loss_en.item(), "en", BASE_PATH, MODEL_SIZE)
            save_checkpoint(step, model_fr, opt_fr, loss_fr.item(), "fr", BASE_PATH, MODEL_SIZE)
            print(f"\nCheckpoint saved: step {step}")

            grammar_en = run_grammar_probes(model_en, TOKENIZER, "en", step, DEVICE)
            grammar_fr = run_grammar_probes(model_fr, TOKENIZER, "fr", step, DEVICE)
            log_grammar_results(grammar_en, "en", BASE_PATH)
            log_grammar_results(grammar_fr, "fr", BASE_PATH)
            print(f"  EN Grammar: {grammar_en['accuracy']:.1f}% acc, log_ratio={grammar_en['mean_log_ratio']:.2f}")
            print(f"  FR Grammar: {grammar_fr['accuracy']:.1f}% acc, log_ratio={grammar_fr['mean_log_ratio']:.2f}")

            emergence_en = run_emergence_probes(model_en, TOKENIZER, "en", step, DEVICE)
            emergence_fr = run_emergence_probes(model_fr, TOKENIZER, "fr", step, DEVICE)
            log_emergence_results(emergence_en, "en", BASE_PATH)
            log_emergence_results(emergence_fr, "fr", BASE_PATH)
            print(f"  Emergence probes logged for both languages")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
