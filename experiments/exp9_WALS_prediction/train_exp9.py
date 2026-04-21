#!/usr/bin/env python3
"""
Exp8 DDP Training: 12 languages on 12x RTX 4090.

Memory-aware batch sizing for 24GB VRAM with 250k vocab.
Idempotent and resumable with probes every 1000 steps.

Usage:
  # Train single language on single GPU
  CUDA_VISIBLE_DEVICES=0 python train_exp8_ddp.py en

  # Train all 12 languages in parallel (one per GPU)
  ./start_all_training.sh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import gzip
import signal
import re
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
from datetime import datetime
from safetensors.torch import save_file, load_file

# ==================== CONFIG ====================
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
TARGET_TOKENS = 3_000_000_000  # 3B tokens per language

# Memory-aware config for RTX 4090 (24GB VRAM)
# Model with 250k vocab uses ~277M params = 1.1GB
# With gradients and optimizer: ~4.4GB base
# Using mixed precision (bfloat16) + gradient checkpointing:
# - Model: 0.55GB (bf16)
# - Gradients: 0.55GB (bf16)
# - Optimizer: 2.2GB (fp32 states)
# - Activations: reduced via checkpointing
# Target: Use ~20GB of 24GB VRAM
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))  # Reduced for 250k vocab OOM
SEQ_LEN = 512
USE_AMP = True  # Automatic mixed precision (bfloat16)
USE_GRADIENT_CHECKPOINTING = True

# 125M architecture (params inflated by large vocab)
MODEL_SIZE = "125M"
N_LAYERS = 12
D_MODEL = 768
N_HEADS = 12
D_FF = 3072

# Calculate steps from tokens
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN
TARGET_STEPS = TARGET_TOKENS // TOKENS_PER_STEP

# Important checkpoints
TOKENS_500M = 500_000_000 // TOKENS_PER_STEP
TOKENS_1B = 1_000_000_000 // TOKENS_PER_STEP
TOKENS_2B = 2_000_000_000 // TOKENS_PER_STEP

# Paths
DATA_PATH = Path(os.environ.get("DATA_PATH", "/workspace/exp8"))
CHECKPOINT_DIR = DATA_PATH / "checkpoints" / LANG / MODEL_SIZE
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = DATA_PATH / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Checkpoint and probe frequency
CHECKPOINT_INTERVAL = 1000  # Save and run probes every 1000 steps
BACKUP_INTERVAL = 5000  # Extra safety checkpoint every 5000 steps

# ==================== SEED ====================
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== DEVICE ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Enable TF32 for faster training on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ==================== TOKENIZER ====================
TOKENIZER_PATH = DATA_PATH / "joint_tokenizer.json"
TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB_SIZE = TOKENIZER.get_vocab_size()  # 50000

print(f"\n=== Exp8: {LANG.upper()} on {DEVICE} ===")
print(f"Target: {TARGET_STEPS:,} steps ({TARGET_TOKENS/1e9:.1f}B tokens)")
print(f"Batch: {BATCH_SIZE}, Seq: {SEQ_LEN}, Vocab: {VOCAB_SIZE:,}")
print(f"Memory budget: batch={BATCH_SIZE} × seq={SEQ_LEN} × {D_MODEL}d")

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
        self.use_checkpoint = USE_GRADIENT_CHECKPOINTING

    def get_mask(self, seq_len, device):
        if self.mask is None or self.mask.size(0) < seq_len:
            self.mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return self.mask[:seq_len, :seq_len]

    def _forward(self, x):
        mask = self.get_mask(x.size(1), x.device)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


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
        self.head.weight = self.tok_emb.weight  # Weight tying

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
        self.chunk_dir = Path(chunk_dir)
        self.lang = lang
        self.chunks = self._scan_chunks()
        if not self.chunks:
            raise FileNotFoundError(f"No chunks found for {lang} in {chunk_dir}")
        print(f"Found {len(self.chunks)} chunks for {lang}")
        self.current_chunk = None
        self.current_idx = 0
        self.chunk_idx = 0
        self.epoch = 0
        np.random.shuffle(self.chunks)

    def _scan_chunks(self):
        """Scan for available chunks (handles growing dataset)."""
        return sorted(self.chunk_dir.glob(f"{self.lang}_chunk_*.npy"))

    def rescan_chunks(self):
        """Rescan for new chunks (for synthetic languages still being created)."""
        new_chunks = self._scan_chunks()
        if len(new_chunks) > len(self.chunks):
            print(f"  Found {len(new_chunks) - len(self.chunks)} new chunks for {self.lang}")
            self.chunks = new_chunks

    def get_batch(self, batch_size):
        sequences = []
        while len(sequences) < batch_size:
            if self.current_chunk is None or self.current_idx >= len(self.current_chunk) - self.seq_len - 1:
                self.chunk_idx = (self.chunk_idx + 1) % len(self.chunks)
                try:
                    self.current_chunk = np.load(self.chunks[self.chunk_idx])
                except Exception as e:
                    print(f"  Warning: Failed to load {self.chunks[self.chunk_idx]}: {e}")
                    self.chunks.pop(self.chunk_idx)
                    if not self.chunks:
                        raise RuntimeError(f"No valid chunks remaining for {self.lang}")
                    continue
                self.current_idx = 0
                if self.chunk_idx == 0:
                    self.epoch += 1
                    np.random.shuffle(self.chunks)
                    # Rescan for new chunks at epoch boundary
                    self.rescan_chunks()

            seq = self.current_chunk[self.current_idx:self.current_idx + self.seq_len + 1]
            sequences.append(seq)
            self.current_idx += self.seq_len // 2  # 50% overlap

        batch = np.stack(sequences)
        x = torch.tensor(batch[:, :-1], dtype=torch.long)
        y = torch.tensor(batch[:, 1:], dtype=torch.long)
        return x, y


# ==================== REASONING PROBES ====================
# Language-agnostic probes that work across all 12 languages
REASONING_PROBES = [
    # Simple arithmetic (works in any language)
    {
        "id": "simple_add",
        "prompt": "2 + 2 =",
        "expected_pattern": r"4|four|quatre|cuatro|neljä|четыре|empat|bốn|四",
        "category": "arithmetic"
    },
    {
        "id": "simple_add_2",
        "prompt": "5 + 3 =",
        "expected_pattern": r"8|eight|huit|ocho|kahdeksan|восемь|delapan|tám|八",
        "category": "arithmetic"
    },
    {
        "id": "simple_sequence",
        "prompt": "1, 2, 3, 4,",
        "expected_pattern": r"5",
        "category": "sequence"
    },
    {
        "id": "doubling_sequence",
        "prompt": "2, 4, 8, 16,",
        "expected_pattern": r"32",
        "category": "sequence"
    },
]

# Language-specific reasoning prompts
REASONING_PROMPTS = {
    "en": [
        {"id": "common_sense", "prompt": "In winter, water becomes", "expected": r"ice|frozen"},
        {"id": "logic", "prompt": "If A > B and B > C, then A", "expected": r">|greater|more"},
        {"id": "completion", "prompt": "The cat sat on the", "expected": r"mat|floor|chair|bed|couch|sofa"},
    ],
    "fr": [
        {"id": "common_sense", "prompt": "En hiver, l'eau devient", "expected": r"glace|gelée|froide"},
        {"id": "logic", "prompt": "Si A > B et B > C, alors A", "expected": r">|plus grand|supérieur"},
        {"id": "completion", "prompt": "Le chat était assis sur le", "expected": r"tapis|sol|canapé|lit"},
    ],
    "es": [
        {"id": "common_sense", "prompt": "En invierno, el agua se convierte en", "expected": r"hielo|congelada"},
        {"id": "logic", "prompt": "Si A > B y B > C, entonces A", "expected": r">|mayor|más grande"},
        {"id": "completion", "prompt": "El gato estaba sentado en el", "expected": r"suelo|sofá|sillón"},
    ],
    "fi": [
        {"id": "common_sense", "prompt": "Talvella vesi muuttuu", "expected": r"jääksi|jää"},
        {"id": "completion", "prompt": "Kissa istui", "expected": r"lattialla|sohvalla|tuolilla"},
    ],
    "ru": [
        {"id": "common_sense", "prompt": "Зимой вода превращается в", "expected": r"лёд|лед|замерзает"},
        {"id": "completion", "prompt": "Кошка сидела на", "expected": r"полу|диване|стуле"},
    ],
    "id": [
        {"id": "common_sense", "prompt": "Di musim dingin, air menjadi", "expected": r"es|beku"},
        {"id": "completion", "prompt": "Kucing duduk di", "expected": r"lantai|sofa|kursi"},
    ],
    "vi": [
        {"id": "common_sense", "prompt": "Vào mùa đông, nước trở thành", "expected": r"băng|đá"},
        {"id": "completion", "prompt": "Con mèo ngồi trên", "expected": r"sàn|ghế|giường"},
    ],
    "zh": [
        {"id": "common_sense", "prompt": "在冬天，水变成", "expected": r"冰|冻"},
        {"id": "completion", "prompt": "猫坐在", "expected": r"地上|椅子|沙发|床"},
    ],
    # Synthetic languages use English probes
    "synth_a": [
        {"id": "common_sense", "prompt": "In winter, water becomes", "expected": r"ice|frozen"},
        {"id": "completion", "prompt": "The cat sat on the", "expected": r"mat|floor|chair"},
    ],
    "synth_b": [
        {"id": "common_sense", "prompt": "In winter, water becomes", "expected": r"ice|frozen"},
        {"id": "completion", "prompt": "The cat sat on the", "expected": r"mat|floor|chair"},
    ],
    "synth_c": [
        {"id": "common_sense", "prompt": "In winter, water becomes", "expected": r"ice|frozen"},
        {"id": "completion", "prompt": "The cat sat on the", "expected": r"mat|floor|chair"},
    ],
    "synth_d": [
        {"id": "common_sense", "prompt": "In winter, water becomes", "expected": r"ice|frozen"},
        {"id": "completion", "prompt": "The cat sat on the", "expected": r"mat|floor|chair"},
    ],
}


def generate_text(model, tokenizer, prompt: str, max_tokens: int = 20, device=None, temperature: float = 0.0):
    """Generate text continuation from prompt."""
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt).ids
    tokens = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([tokens[-512:]], dtype=torch.long, device=device)  # Keep last 512 tokens
            logits = model(x)[0, -1, :]  # Last position logits

            if temperature <= 0:
                # Greedy decoding
                next_token = logits.argmax().item()
            else:
                # Temperature sampling
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            tokens.append(next_token)

            # Stop on EOS or newline
            if next_token == tokenizer.token_to_id("<eos>"):
                break

    # Decode only the generated part
    generated = tokenizer.decode(tokens[len(input_ids):])
    model.train()
    return generated.strip()


def run_reasoning_probes(model, tokenizer, lang: str, device) -> dict:
    """Run reasoning probes and return results."""
    results = {
        "total": 0,
        "correct": 0,
        "by_category": {},
        "samples": []
    }

    # Run language-agnostic probes
    for probe in REASONING_PROBES:
        generated = generate_text(model, tokenizer, probe["prompt"], max_tokens=10, device=device)
        matched = bool(re.search(probe["expected_pattern"], generated[:50], re.IGNORECASE))

        category = probe["category"]
        if category not in results["by_category"]:
            results["by_category"][category] = {"correct": 0, "total": 0}
        results["by_category"][category]["total"] += 1
        results["total"] += 1

        if matched:
            results["correct"] += 1
            results["by_category"][category]["correct"] += 1

        results["samples"].append({
            "id": probe["id"],
            "prompt": probe["prompt"],
            "generated": generated[:50],
            "matched": matched
        })

    # Run language-specific probes
    lang_probes = REASONING_PROMPTS.get(lang, REASONING_PROMPTS.get("en", []))
    for probe in lang_probes:
        generated = generate_text(model, tokenizer, probe["prompt"], max_tokens=15, device=device)
        matched = bool(re.search(probe["expected"], generated[:50], re.IGNORECASE))

        category = probe["id"]
        if category not in results["by_category"]:
            results["by_category"][category] = {"correct": 0, "total": 0}
        results["by_category"][category]["total"] += 1
        results["total"] += 1

        if matched:
            results["correct"] += 1
            results["by_category"][category]["correct"] += 1

        results["samples"].append({
            "id": probe["id"],
            "prompt": probe["prompt"],
            "generated": generated[:50],
            "matched": matched
        })

    # Compute accuracy
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0

    return results


# ==================== GRAMMAR PROBES ====================
# Exp1-style probes: next-token prediction with multiple good/bad options
# All languages have 10 probes for fair comparison (matches exp1)
GRAMMAR_PROBE_TEMPERATURE = 0.1  # Low temperature for stable measurements (exp1 default)

GRAMMAR_PROBES = {
    # === Grammar Probes v2: 10 probes per testable dimension per language ===
    # EN: sv_num, article, colloc (3 dims = 30 probes)
    # FR/ES/RU: sv_num, sv_pers, gender, num_adj, art_gen, colloc (6 dims = 60 probes)
    # FI: sv_num, sv_pers, num_adj, case, colloc (5 dims = 50 probes)
    # ID/VI/ZH: classifier, colloc (2 dims = 20 probes)
    "en": [
        # === SV Agreement (number) - 10 probes ===
        {"prompt": "The cat ", "good": ["is", "sits", "sleeps"], "bad": ["are", "sit", "sleep"], "category": "sv_num"},
        {"prompt": "The dog ", "good": ["is", "runs", "barks"], "bad": ["are", "run", "bark"], "category": "sv_num"},
        {"prompt": "The bird ", "good": ["is", "flies", "sings"], "bad": ["are", "fly", "sing"], "category": "sv_num"},
        {"prompt": "The man ", "good": ["is", "walks", "talks"], "bad": ["are", "walk", "talk"], "category": "sv_num"},
        {"prompt": "The woman ", "good": ["is", "walks", "talks"], "bad": ["are", "walk", "talk"], "category": "sv_num"},
        {"prompt": "The cats ", "good": ["are", "sit", "sleep"], "bad": ["is", "sits", "sleeps"], "category": "sv_num"},
        {"prompt": "The dogs ", "good": ["are", "run", "bark"], "bad": ["is", "runs", "barks"], "category": "sv_num"},
        {"prompt": "The birds ", "good": ["are", "fly", "sing"], "bad": ["is", "flies", "sings"], "category": "sv_num"},
        {"prompt": "The men ", "good": ["are", "walk", "talk"], "bad": ["is", "walks", "talks"], "category": "sv_num"},
        {"prompt": "The women ", "good": ["are", "walk", "talk"], "bad": ["is", "walks", "talks"], "category": "sv_num"},
        # === Article (a/an) - 10 probes ===
        {"prompt": "I saw a ", "good": ["cat", "dog", "bird"], "bad": ["apple", "elephant", "owl"], "category": "article"},
        {"prompt": "I saw a ", "good": ["book", "tree", "house"], "bad": ["egg", "orange", "umbrella"], "category": "article"},
        {"prompt": "I saw a ", "good": ["car", "man", "girl"], "bad": ["ant", "eagle", "inch"], "category": "article"},
        {"prompt": "I saw a ", "good": ["table", "chair", "desk"], "bad": ["apple", "ear", "eye"], "category": "article"},
        {"prompt": "I saw a ", "good": ["pen", "cup", "box"], "bad": ["arm", "uncle", "ocean"], "category": "article"},
        {"prompt": "I saw an ", "good": ["apple", "elephant", "owl"], "bad": ["cat", "dog", "bird"], "category": "article"},
        {"prompt": "I saw an ", "good": ["egg", "orange", "umbrella"], "bad": ["book", "tree", "house"], "category": "article"},
        {"prompt": "I saw an ", "good": ["ant", "eagle", "inch"], "bad": ["car", "man", "girl"], "category": "article"},
        {"prompt": "I saw an ", "good": ["arm", "uncle", "ocean"], "bad": ["table", "chair", "desk"], "category": "article"},
        {"prompt": "I saw an ", "good": ["ice", "onion", "hour"], "bad": ["pen", "cup", "box"], "category": "article"},
        # === Collocations - 10 probes ===
        {"prompt": "black and ", "good": ["white"], "bad": ["red", "green", "blue"], "category": "colloc"},
        {"prompt": "day and ", "good": ["night"], "bad": ["morning", "evening", "sun"], "category": "colloc"},
        {"prompt": "up and ", "good": ["down"], "bad": ["left", "right", "over"], "category": "colloc"},
        {"prompt": "hot and ", "good": ["cold"], "bad": ["warm", "cool", "wet"], "category": "colloc"},
        {"prompt": "left and ", "good": ["right"], "bad": ["up", "down", "over"], "category": "colloc"},
        {"prompt": "back and ", "good": ["forth"], "bad": ["front", "side", "over"], "category": "colloc"},
        {"prompt": "bread and ", "good": ["butter"], "bad": ["cheese", "milk", "water"], "category": "colloc"},
        {"prompt": "salt and ", "good": ["pepper"], "bad": ["sugar", "spice", "water"], "category": "colloc"},
        {"prompt": "husband and ", "good": ["wife"], "bad": ["man", "woman", "child"], "category": "colloc"},
        {"prompt": "king and ", "good": ["queen"], "bad": ["prince", "lord", "knight"], "category": "colloc"},
    ],

    "fr": [
        # === SV Agreement (number) - 10 probes ===
        {"prompt": "Le chat ", "good": ["est", "dort", "mange"], "bad": ["sont", "dorment", "mangent"], "category": "sv_num"},
        {"prompt": "Le chien ", "good": ["est", "court", "aboie"], "bad": ["sont", "courent", "aboient"], "category": "sv_num"},
        {"prompt": "L'oiseau ", "good": ["est", "vole", "chante"], "bad": ["sont", "volent", "chantent"], "category": "sv_num"},
        {"prompt": "L'homme ", "good": ["est", "marche", "parle"], "bad": ["sont", "marchent", "parlent"], "category": "sv_num"},
        {"prompt": "La femme ", "good": ["est", "marche", "parle"], "bad": ["sont", "marchent", "parlent"], "category": "sv_num"},
        {"prompt": "Les chats ", "good": ["sont", "dorment", "mangent"], "bad": ["est", "dort", "mange"], "category": "sv_num"},
        {"prompt": "Les chiens ", "good": ["sont", "courent", "aboient"], "bad": ["est", "court", "aboie"], "category": "sv_num"},
        {"prompt": "Les oiseaux ", "good": ["sont", "volent", "chantent"], "bad": ["est", "vole", "chante"], "category": "sv_num"},
        {"prompt": "Les hommes ", "good": ["sont", "marchent", "parlent"], "bad": ["est", "marche", "parle"], "category": "sv_num"},
        {"prompt": "Les femmes ", "good": ["sont", "marchent", "parlent"], "bad": ["est", "marche", "parle"], "category": "sv_num"},
        # === SV Agreement (person) - 10 probes ===
        {"prompt": "Je ", "good": ["suis", "mange", "vais"], "bad": ["es", "manges", "vas"], "category": "sv_pers"},
        {"prompt": "Je ", "good": ["parle", "dors", "lis"], "bad": ["parles", "dors", "lis"], "category": "sv_pers"},
        {"prompt": "Tu ", "good": ["es", "manges", "vas"], "bad": ["suis", "mange", "vais"], "category": "sv_pers"},
        {"prompt": "Tu ", "good": ["parles", "dors", "lis"], "bad": ["parle", "dort", "lit"], "category": "sv_pers"},
        {"prompt": "Il ", "good": ["est", "mange", "va"], "bad": ["suis", "manges", "vais"], "category": "sv_pers"},
        {"prompt": "Elle ", "good": ["est", "mange", "va"], "bad": ["es", "manges", "vas"], "category": "sv_pers"},
        {"prompt": "Nous ", "good": ["sommes", "mangeons", "allons"], "bad": ["sont", "mangent", "vont"], "category": "sv_pers"},
        {"prompt": "Vous ", "good": ["êtes", "mangez", "allez"], "bad": ["sommes", "mangeons", "allons"], "category": "sv_pers"},
        {"prompt": "Ils ", "good": ["sont", "mangent", "vont"], "bad": ["est", "mange", "va"], "category": "sv_pers"},
        {"prompt": "Elles ", "good": ["sont", "mangent", "vont"], "bad": ["est", "mange", "va"], "category": "sv_pers"},
        # === Gender Agreement - 10 probes ===
        {"prompt": "Le chat ", "good": ["noir", "petit", "gros"], "bad": ["noire", "petite", "grosse"], "category": "gender"},
        {"prompt": "Le chien ", "good": ["blanc", "grand", "beau"], "bad": ["blanche", "grande", "belle"], "category": "gender"},
        {"prompt": "Le livre ", "good": ["rouge", "vieux", "nouveau"], "bad": ["vieille", "nouvelle"], "category": "gender"},
        {"prompt": "L'homme ", "good": ["grand", "petit", "vieux"], "bad": ["grande", "petite", "vieille"], "category": "gender"},
        {"prompt": "Le garçon ", "good": ["blond", "content", "fatigué"], "bad": ["blonde", "contente", "fatiguée"], "category": "gender"},
        {"prompt": "La maison ", "good": ["blanche", "grande", "belle"], "bad": ["blanc", "grand", "beau"], "category": "gender"},
        {"prompt": "La voiture ", "good": ["rouge", "petite", "vieille"], "bad": ["petit", "vieux"], "category": "gender"},
        {"prompt": "La femme ", "good": ["grande", "petite", "belle"], "bad": ["grand", "petit", "beau"], "category": "gender"},
        {"prompt": "La fille ", "good": ["blonde", "contente", "fatiguée"], "bad": ["blond", "content", "fatigué"], "category": "gender"},
        {"prompt": "La table ", "good": ["ronde", "carrée", "haute"], "bad": ["rond", "carré", "haut"], "category": "gender"},
        # === Number Agreement (adj) - 10 probes ===
        {"prompt": "Les chats ", "good": ["noirs", "petits", "gros"], "bad": ["noir", "petit"], "category": "num_adj"},
        {"prompt": "Les chiens ", "good": ["blancs", "grands", "beaux"], "bad": ["blanc", "grand", "beau"], "category": "num_adj"},
        {"prompt": "Les livres ", "good": ["rouges", "vieux", "nouveaux"], "bad": ["rouge", "vieux", "nouveau"], "category": "num_adj"},
        {"prompt": "Les hommes ", "good": ["grands", "petits", "vieux"], "bad": ["grand", "petit", "vieux"], "category": "num_adj"},
        {"prompt": "Les garçons ", "good": ["blonds", "contents", "fatigués"], "bad": ["blond", "content", "fatigué"], "category": "num_adj"},
        {"prompt": "Les maisons ", "good": ["blanches", "grandes", "belles"], "bad": ["blanche", "grande", "belle"], "category": "num_adj"},
        {"prompt": "Les voitures ", "good": ["rouges", "petites", "vieilles"], "bad": ["rouge", "petite", "vieille"], "category": "num_adj"},
        {"prompt": "Les femmes ", "good": ["grandes", "petites", "belles"], "bad": ["grande", "petite", "belle"], "category": "num_adj"},
        {"prompt": "Les filles ", "good": ["blondes", "contentes", "fatiguées"], "bad": ["blonde", "contente", "fatiguée"], "category": "num_adj"},
        {"prompt": "Les tables ", "good": ["rondes", "carrées", "hautes"], "bad": ["ronde", "carrée", "haute"], "category": "num_adj"},
        # === Article (gender) - 10 probes ===
        {"prompt": "Je vois le ", "good": ["chat", "chien", "livre"], "bad": ["maison", "voiture", "table"], "category": "art_gen"},
        {"prompt": "Je vois le ", "good": ["garçon", "homme", "père"], "bad": ["fille", "femme", "mère"], "category": "art_gen"},
        {"prompt": "Je vois le ", "good": ["soleil", "ciel", "jour"], "bad": ["lune", "nuit", "étoile"], "category": "art_gen"},
        {"prompt": "Je vois le ", "good": ["pain", "fromage", "vin"], "bad": ["viande", "salade", "bière"], "category": "art_gen"},
        {"prompt": "Je vois le ", "good": ["jardin", "arbre", "mur"], "bad": ["fleur", "plante", "porte"], "category": "art_gen"},
        {"prompt": "Je vois la ", "good": ["maison", "voiture", "table"], "bad": ["chat", "chien", "livre"], "category": "art_gen"},
        {"prompt": "Je vois la ", "good": ["fille", "femme", "mère"], "bad": ["garçon", "homme", "père"], "category": "art_gen"},
        {"prompt": "Je vois la ", "good": ["lune", "nuit", "étoile"], "bad": ["soleil", "ciel", "jour"], "category": "art_gen"},
        {"prompt": "Je vois la ", "good": ["viande", "salade", "bière"], "bad": ["pain", "fromage", "vin"], "category": "art_gen"},
        {"prompt": "Je vois la ", "good": ["fleur", "plante", "porte"], "bad": ["jardin", "arbre", "mur"], "category": "art_gen"},
        # === Collocations - 10 probes ===
        {"prompt": "noir et ", "good": ["blanc"], "bad": ["rouge", "vert", "bleu"], "category": "colloc"},
        {"prompt": "jour et ", "good": ["nuit"], "bad": ["matin", "soir", "soleil"], "category": "colloc"},
        {"prompt": "haut et ", "good": ["bas"], "bad": ["grand", "petit", "large"], "category": "colloc"},
        {"prompt": "chaud et ", "good": ["froid"], "bad": ["tiède", "frais", "sec"], "category": "colloc"},
        {"prompt": "gauche et ", "good": ["droite"], "bad": ["haut", "bas", "avant"], "category": "colloc"},
        {"prompt": "sel et ", "good": ["poivre"], "bad": ["sucre", "épice", "eau"], "category": "colloc"},
        {"prompt": "pain et ", "good": ["beurre"], "bad": ["fromage", "lait", "eau"], "category": "colloc"},
        {"prompt": "mari et ", "good": ["femme"], "bad": ["homme", "fille", "enfant"], "category": "colloc"},
        {"prompt": "roi et ", "good": ["reine"], "bad": ["prince", "duc", "chevalier"], "category": "colloc"},
        {"prompt": "frère et ", "good": ["sœur"], "bad": ["père", "mère", "fils"], "category": "colloc"},
    ],

    "es": [
        # === SV Agreement (number) - 10 probes ===
        {"prompt": "El gato ", "good": ["es", "duerme", "come"], "bad": ["son", "duermen", "comen"], "category": "sv_num"},
        {"prompt": "El perro ", "good": ["es", "corre", "ladra"], "bad": ["son", "corren", "ladran"], "category": "sv_num"},
        {"prompt": "El pájaro ", "good": ["es", "vuela", "canta"], "bad": ["son", "vuelan", "cantan"], "category": "sv_num"},
        {"prompt": "El hombre ", "good": ["es", "camina", "habla"], "bad": ["son", "caminan", "hablan"], "category": "sv_num"},
        {"prompt": "La mujer ", "good": ["es", "camina", "habla"], "bad": ["son", "caminan", "hablan"], "category": "sv_num"},
        {"prompt": "Los gatos ", "good": ["son", "duermen", "comen"], "bad": ["es", "duerme", "come"], "category": "sv_num"},
        {"prompt": "Los perros ", "good": ["son", "corren", "ladran"], "bad": ["es", "corre", "ladra"], "category": "sv_num"},
        {"prompt": "Los pájaros ", "good": ["son", "vuelan", "cantan"], "bad": ["es", "vuela", "canta"], "category": "sv_num"},
        {"prompt": "Los hombres ", "good": ["son", "caminan", "hablan"], "bad": ["es", "camina", "habla"], "category": "sv_num"},
        {"prompt": "Las mujeres ", "good": ["son", "caminan", "hablan"], "bad": ["es", "camina", "habla"], "category": "sv_num"},
        # === SV Agreement (person) - 10 probes ===
        {"prompt": "Yo ", "good": ["soy", "como", "voy"], "bad": ["eres", "comes", "vas"], "category": "sv_pers"},
        {"prompt": "Yo ", "good": ["hablo", "duermo", "leo"], "bad": ["hablas", "duermes", "lees"], "category": "sv_pers"},
        {"prompt": "Tú ", "good": ["eres", "comes", "vas"], "bad": ["soy", "como", "voy"], "category": "sv_pers"},
        {"prompt": "Tú ", "good": ["hablas", "duermes", "lees"], "bad": ["habla", "duerme", "lee"], "category": "sv_pers"},
        {"prompt": "Él ", "good": ["es", "come", "va"], "bad": ["soy", "comes", "voy"], "category": "sv_pers"},
        {"prompt": "Ella ", "good": ["es", "come", "va"], "bad": ["eres", "comes", "vas"], "category": "sv_pers"},
        {"prompt": "Nosotros ", "good": ["somos", "comemos", "vamos"], "bad": ["son", "comen", "van"], "category": "sv_pers"},
        {"prompt": "Vosotros ", "good": ["sois", "coméis", "vais"], "bad": ["somos", "comemos", "vamos"], "category": "sv_pers"},
        {"prompt": "Ellos ", "good": ["son", "comen", "van"], "bad": ["es", "come", "va"], "category": "sv_pers"},
        {"prompt": "Ellas ", "good": ["son", "comen", "van"], "bad": ["es", "come", "va"], "category": "sv_pers"},
        # === Gender Agreement - 10 probes ===
        {"prompt": "El gato ", "good": ["negro", "pequeño", "gordo"], "bad": ["negra", "pequeña", "gorda"], "category": "gender"},
        {"prompt": "El perro ", "good": ["blanco", "grande", "bonito"], "bad": ["blanca", "grande", "bonita"], "category": "gender"},
        {"prompt": "El libro ", "good": ["rojo", "viejo", "nuevo"], "bad": ["roja", "vieja", "nueva"], "category": "gender"},
        {"prompt": "El hombre ", "good": ["alto", "bajo", "viejo"], "bad": ["alta", "baja", "vieja"], "category": "gender"},
        {"prompt": "El niño ", "good": ["rubio", "contento", "cansado"], "bad": ["rubia", "contenta", "cansada"], "category": "gender"},
        {"prompt": "La casa ", "good": ["blanca", "grande", "bonita"], "bad": ["blanco", "grande", "bonito"], "category": "gender"},
        {"prompt": "La mesa ", "good": ["roja", "pequeña", "vieja"], "bad": ["rojo", "pequeño", "viejo"], "category": "gender"},
        {"prompt": "La mujer ", "good": ["alta", "baja", "vieja"], "bad": ["alto", "bajo", "viejo"], "category": "gender"},
        {"prompt": "La niña ", "good": ["rubia", "contenta", "cansada"], "bad": ["rubio", "contento", "cansado"], "category": "gender"},
        {"prompt": "La silla ", "good": ["redonda", "cuadrada", "alta"], "bad": ["redondo", "cuadrado", "alto"], "category": "gender"},
        # === Number Agreement (adj) - 10 probes ===
        {"prompt": "Los gatos ", "good": ["negros", "pequeños", "gordos"], "bad": ["negro", "pequeño", "gordo"], "category": "num_adj"},
        {"prompt": "Los perros ", "good": ["blancos", "grandes", "bonitos"], "bad": ["blanco", "grande", "bonito"], "category": "num_adj"},
        {"prompt": "Los libros ", "good": ["rojos", "viejos", "nuevos"], "bad": ["rojo", "viejo", "nuevo"], "category": "num_adj"},
        {"prompt": "Los hombres ", "good": ["altos", "bajos", "viejos"], "bad": ["alto", "bajo", "viejo"], "category": "num_adj"},
        {"prompt": "Los niños ", "good": ["rubios", "contentos", "cansados"], "bad": ["rubio", "contento", "cansado"], "category": "num_adj"},
        {"prompt": "Las casas ", "good": ["blancas", "grandes", "bonitas"], "bad": ["blanca", "grande", "bonita"], "category": "num_adj"},
        {"prompt": "Las mesas ", "good": ["rojas", "pequeñas", "viejas"], "bad": ["roja", "pequeña", "vieja"], "category": "num_adj"},
        {"prompt": "Las mujeres ", "good": ["altas", "bajas", "viejas"], "bad": ["alta", "baja", "vieja"], "category": "num_adj"},
        {"prompt": "Las niñas ", "good": ["rubias", "contentas", "cansadas"], "bad": ["rubia", "contenta", "cansada"], "category": "num_adj"},
        {"prompt": "Las sillas ", "good": ["redondas", "cuadradas", "altas"], "bad": ["redonda", "cuadrada", "alta"], "category": "num_adj"},
        # === Article (gender) - 10 probes ===
        {"prompt": "Yo veo el ", "good": ["gato", "perro", "libro"], "bad": ["casa", "mesa", "silla"], "category": "art_gen"},
        {"prompt": "Yo veo el ", "good": ["niño", "hombre", "padre"], "bad": ["niña", "mujer", "madre"], "category": "art_gen"},
        {"prompt": "Yo veo el ", "good": ["sol", "cielo", "día"], "bad": ["luna", "noche", "estrella"], "category": "art_gen"},
        {"prompt": "Yo veo el ", "good": ["pan", "queso", "vino"], "bad": ["carne", "leche", "cerveza"], "category": "art_gen"},
        {"prompt": "Yo veo el ", "good": ["jardín", "árbol", "muro"], "bad": ["flor", "planta", "puerta"], "category": "art_gen"},
        {"prompt": "Yo veo la ", "good": ["casa", "mesa", "silla"], "bad": ["gato", "perro", "libro"], "category": "art_gen"},
        {"prompt": "Yo veo la ", "good": ["niña", "mujer", "madre"], "bad": ["niño", "hombre", "padre"], "category": "art_gen"},
        {"prompt": "Yo veo la ", "good": ["luna", "noche", "estrella"], "bad": ["sol", "cielo", "día"], "category": "art_gen"},
        {"prompt": "Yo veo la ", "good": ["carne", "leche", "cerveza"], "bad": ["pan", "queso", "vino"], "category": "art_gen"},
        {"prompt": "Yo veo la ", "good": ["flor", "planta", "puerta"], "bad": ["jardín", "árbol", "muro"], "category": "art_gen"},
        # === Collocations - 10 probes ===
        {"prompt": "blanco y ", "good": ["negro"], "bad": ["rojo", "verde", "azul"], "category": "colloc"},
        {"prompt": "día y ", "good": ["noche"], "bad": ["mañana", "tarde", "sol"], "category": "colloc"},
        {"prompt": "arriba y ", "good": ["abajo"], "bad": ["grande", "pequeño", "largo"], "category": "colloc"},
        {"prompt": "caliente y ", "good": ["frío"], "bad": ["tibio", "fresco", "seco"], "category": "colloc"},
        {"prompt": "izquierda y ", "good": ["derecha"], "bad": ["arriba", "abajo", "delante"], "category": "colloc"},
        {"prompt": "sal y ", "good": ["pimienta"], "bad": ["azúcar", "especia", "agua"], "category": "colloc"},
        {"prompt": "pan y ", "good": ["mantequilla"], "bad": ["queso", "leche", "agua"], "category": "colloc"},
        {"prompt": "marido y ", "good": ["mujer"], "bad": ["hombre", "hija", "niño"], "category": "colloc"},
        {"prompt": "rey y ", "good": ["reina"], "bad": ["príncipe", "duque", "caballero"], "category": "colloc"},
        {"prompt": "hermano y ", "good": ["hermana"], "bad": ["padre", "madre", "hijo"], "category": "colloc"},
    ],

    "fi": [
        # === SV Agreement (number) - 10 probes ===
        {"prompt": "Kissa ", "good": ["on", "nukkuu", "syö"], "bad": ["ovat", "nukkuvat", "syövät"], "category": "sv_num"},
        {"prompt": "Koira ", "good": ["on", "juoksee", "haukkuu"], "bad": ["ovat", "juoksevat", "haukkuvat"], "category": "sv_num"},
        {"prompt": "Lintu ", "good": ["on", "lentää", "laulaa"], "bad": ["ovat", "lentävät", "laulavat"], "category": "sv_num"},
        {"prompt": "Mies ", "good": ["on", "kävelee", "puhuu"], "bad": ["ovat", "kävelevät", "puhuvat"], "category": "sv_num"},
        {"prompt": "Nainen ", "good": ["on", "kävelee", "puhuu"], "bad": ["ovat", "kävelevät", "puhuvat"], "category": "sv_num"},
        {"prompt": "Kissat ", "good": ["ovat", "nukkuvat", "syövät"], "bad": ["on", "nukkuu", "syö"], "category": "sv_num"},
        {"prompt": "Koirat ", "good": ["ovat", "juoksevat", "haukkuvat"], "bad": ["on", "juoksee", "haukkuu"], "category": "sv_num"},
        {"prompt": "Linnut ", "good": ["ovat", "lentävät", "laulavat"], "bad": ["on", "lentää", "laulaa"], "category": "sv_num"},
        {"prompt": "Miehet ", "good": ["ovat", "kävelevät", "puhuvat"], "bad": ["on", "kävelee", "puhuu"], "category": "sv_num"},
        {"prompt": "Naiset ", "good": ["ovat", "kävelevät", "puhuvat"], "bad": ["on", "kävelee", "puhuu"], "category": "sv_num"},
        # === SV Agreement (person) - 10 probes ===
        {"prompt": "Minä ", "good": ["olen", "syön", "menen"], "bad": ["olet", "syöt", "menet"], "category": "sv_pers"},
        {"prompt": "Minä ", "good": ["puhun", "nukun", "luen"], "bad": ["puhut", "nukut", "luet"], "category": "sv_pers"},
        {"prompt": "Sinä ", "good": ["olet", "syöt", "menet"], "bad": ["olen", "syön", "menen"], "category": "sv_pers"},
        {"prompt": "Sinä ", "good": ["puhut", "nukut", "luet"], "bad": ["puhuu", "nukkuu", "lukee"], "category": "sv_pers"},
        {"prompt": "Hän ", "good": ["on", "syö", "menee"], "bad": ["olen", "syöt", "menen"], "category": "sv_pers"},
        {"prompt": "Hän ", "good": ["puhuu", "nukkuu", "lukee"], "bad": ["puhut", "nukut", "luet"], "category": "sv_pers"},
        {"prompt": "Me ", "good": ["olemme", "syömme", "menemme"], "bad": ["ovat", "syövät", "menevät"], "category": "sv_pers"},
        {"prompt": "Te ", "good": ["olette", "syötte", "menette"], "bad": ["olemme", "syömme", "menemme"], "category": "sv_pers"},
        {"prompt": "He ", "good": ["ovat", "syövät", "menevät"], "bad": ["on", "syö", "menee"], "category": "sv_pers"},
        {"prompt": "He ", "good": ["puhuvat", "nukkuvat", "lukevat"], "bad": ["puhuu", "nukkuu", "lukee"], "category": "sv_pers"},
        # === Number Agreement (adj) - 10 probes ===
        {"prompt": "Iso ", "good": ["talo", "koira", "kissa"], "bad": ["talot", "koirat", "kissat"], "category": "num_adj"},
        {"prompt": "Pieni ", "good": ["auto", "lintu", "lapsi"], "bad": ["autot", "linnut", "lapset"], "category": "num_adj"},
        {"prompt": "Vanha ", "good": ["mies", "nainen", "puu"], "bad": ["miehet", "naiset", "puut"], "category": "num_adj"},
        {"prompt": "Uusi ", "good": ["kirja", "koti", "työ"], "bad": ["kirjat", "kodit", "työt"], "category": "num_adj"},
        {"prompt": "Kaunis ", "good": ["kukka", "päivä", "tyttö"], "bad": ["kukat", "päivät", "tytöt"], "category": "num_adj"},
        {"prompt": "Isot ", "good": ["talot", "koirat", "kissat"], "bad": ["talo", "koira", "kissa"], "category": "num_adj"},
        {"prompt": "Pienet ", "good": ["autot", "linnut", "lapset"], "bad": ["auto", "lintu", "lapsi"], "category": "num_adj"},
        {"prompt": "Vanhat ", "good": ["miehet", "naiset", "puut"], "bad": ["mies", "nainen", "puu"], "category": "num_adj"},
        {"prompt": "Uudet ", "good": ["kirjat", "kodit", "työt"], "bad": ["kirja", "koti", "työ"], "category": "num_adj"},
        {"prompt": "Kauniit ", "good": ["kukat", "päivät", "tytöt"], "bad": ["kukka", "päivä", "tyttö"], "category": "num_adj"},
        # === Case Agreement - 10 probes ===
        {"prompt": "Näen ison ", "good": ["talon", "koiran", "kissan"], "bad": ["talo", "koira", "kissa"], "category": "case"},
        {"prompt": "Näen pienen ", "good": ["auton", "linnun", "lapsen"], "bad": ["auto", "lintu", "lapsi"], "category": "case"},
        {"prompt": "Isossa ", "good": ["talossa", "autossa", "huoneessa"], "bad": ["talo", "auto", "huone"], "category": "case"},
        {"prompt": "Pienessä ", "good": ["kodissa", "kaupassa", "koulussa"], "bad": ["koti", "kauppa", "koulu"], "category": "case"},
        {"prompt": "Isoon ", "good": ["taloon", "autoon", "huoneeseen"], "bad": ["talo", "auto", "huone"], "category": "case"},
        {"prompt": "Pieneen ", "good": ["kotiin", "kauppaan", "kouluun"], "bad": ["koti", "kauppa", "koulu"], "category": "case"},
        {"prompt": "Isosta ", "good": ["talosta", "autosta", "huoneesta"], "bad": ["talo", "auto", "huone"], "category": "case"},
        {"prompt": "Pienestä ", "good": ["kodista", "kaupasta", "koulusta"], "bad": ["koti", "kauppa", "koulu"], "category": "case"},
        {"prompt": "Ison talon ", "good": ["edessä", "takana", "vieressä"], "bad": ["edessä", "takana", "vieressä"], "category": "case"},
        {"prompt": "Vanhan miehen ", "good": ["kanssa", "luona", "takia"], "bad": ["kanssa", "luona", "takia"], "category": "case"},
        # === Collocations - 10 probes ===
        {"prompt": "musta ja ", "good": ["valkoinen"], "bad": ["punainen", "sininen", "vihreä"], "category": "colloc"},
        {"prompt": "yö ja ", "good": ["päivä"], "bad": ["aamu", "ilta", "aurinko"], "category": "colloc"},
        {"prompt": "ylös ja ", "good": ["alas"], "bad": ["iso", "pieni", "pitkä"], "category": "colloc"},
        {"prompt": "kuuma ja ", "good": ["kylmä"], "bad": ["lämmin", "viileä", "kuiva"], "category": "colloc"},
        {"prompt": "vasen ja ", "good": ["oikea"], "bad": ["ylös", "alas", "eteen"], "category": "colloc"},
        {"prompt": "suola ja ", "good": ["pippuri"], "bad": ["sokeri", "mauste", "vesi"], "category": "colloc"},
        {"prompt": "leipä ja ", "good": ["voi"], "bad": ["juusto", "maito", "vesi"], "category": "colloc"},
        {"prompt": "mies ja ", "good": ["vaimo"], "bad": ["nainen", "tyttö", "lapsi"], "category": "colloc"},
        {"prompt": "kuningas ja ", "good": ["kuningatar"], "bad": ["prinssi", "herttua", "ritari"], "category": "colloc"},
        {"prompt": "veli ja ", "good": ["sisko"], "bad": ["isä", "äiti", "poika"], "category": "colloc"},
    ],

    "ru": [
        # === SV Agreement (number) - 10 probes ===
        {"prompt": "Кот ", "good": ["спит", "ест", "бежит"], "bad": ["спят", "едят", "бегут"], "category": "sv_num"},
        {"prompt": "Собака ", "good": ["спит", "ест", "бежит"], "bad": ["спят", "едят", "бегут"], "category": "sv_num"},
        {"prompt": "Птица ", "good": ["летит", "поёт", "сидит"], "bad": ["летят", "поют", "сидят"], "category": "sv_num"},
        {"prompt": "Мужчина ", "good": ["идёт", "говорит", "работает"], "bad": ["идут", "говорят", "работают"], "category": "sv_num"},
        {"prompt": "Женщина ", "good": ["идёт", "говорит", "работает"], "bad": ["идут", "говорят", "работают"], "category": "sv_num"},
        {"prompt": "Коты ", "good": ["спят", "едят", "бегут"], "bad": ["спит", "ест", "бежит"], "category": "sv_num"},
        {"prompt": "Собаки ", "good": ["спят", "едят", "бегут"], "bad": ["спит", "ест", "бежит"], "category": "sv_num"},
        {"prompt": "Птицы ", "good": ["летят", "поют", "сидят"], "bad": ["летит", "поёт", "сидит"], "category": "sv_num"},
        {"prompt": "Мужчины ", "good": ["идут", "говорят", "работают"], "bad": ["идёт", "говорит", "работает"], "category": "sv_num"},
        {"prompt": "Женщины ", "good": ["идут", "говорят", "работают"], "bad": ["идёт", "говорит", "работает"], "category": "sv_num"},
        # === SV Agreement (person) - 10 probes ===
        {"prompt": "Я ", "good": ["иду", "ем", "сплю"], "bad": ["идёшь", "ешь", "спишь"], "category": "sv_pers"},
        {"prompt": "Я ", "good": ["говорю", "читаю", "пишу"], "bad": ["говоришь", "читаешь", "пишешь"], "category": "sv_pers"},
        {"prompt": "Ты ", "good": ["идёшь", "ешь", "спишь"], "bad": ["иду", "ем", "сплю"], "category": "sv_pers"},
        {"prompt": "Ты ", "good": ["говоришь", "читаешь", "пишешь"], "bad": ["говорит", "читает", "пишет"], "category": "sv_pers"},
        {"prompt": "Он ", "good": ["идёт", "ест", "спит"], "bad": ["иду", "ешь", "сплю"], "category": "sv_pers"},
        {"prompt": "Она ", "good": ["идёт", "ест", "спит"], "bad": ["идёшь", "ешь", "спишь"], "category": "sv_pers"},
        {"prompt": "Мы ", "good": ["идём", "едим", "спим"], "bad": ["идут", "едят", "спят"], "category": "sv_pers"},
        {"prompt": "Вы ", "good": ["идёте", "едите", "спите"], "bad": ["идём", "едим", "спим"], "category": "sv_pers"},
        {"prompt": "Они ", "good": ["идут", "едят", "спят"], "bad": ["идёт", "ест", "спит"], "category": "sv_pers"},
        {"prompt": "Они ", "good": ["говорят", "читают", "пишут"], "bad": ["говорит", "читает", "пишет"], "category": "sv_pers"},
        # === Gender Agreement - 10 probes ===
        {"prompt": "Большой ", "good": ["дом", "стол", "кот"], "bad": ["машина", "книга", "кошка"], "category": "gender"},
        {"prompt": "Новый ", "good": ["друг", "город", "год"], "bad": ["подруга", "страна", "жизнь"], "category": "gender"},
        {"prompt": "Старый ", "good": ["человек", "мир", "лес"], "bad": ["женщина", "земля", "река"], "category": "gender"},
        {"prompt": "Красивый ", "good": ["мальчик", "цветок", "день"], "bad": ["девочка", "роза", "ночь"], "category": "gender"},
        {"prompt": "Хороший ", "good": ["отец", "брат", "сын"], "bad": ["мать", "сестра", "дочь"], "category": "gender"},
        {"prompt": "Большая ", "good": ["машина", "книга", "кошка"], "bad": ["дом", "стол", "кот"], "category": "gender"},
        {"prompt": "Новая ", "good": ["подруга", "страна", "жизнь"], "bad": ["друг", "город", "год"], "category": "gender"},
        {"prompt": "Старая ", "good": ["женщина", "земля", "река"], "bad": ["человек", "мир", "лес"], "category": "gender"},
        {"prompt": "Красивая ", "good": ["девочка", "роза", "ночь"], "bad": ["мальчик", "цветок", "день"], "category": "gender"},
        {"prompt": "Хорошая ", "good": ["мать", "сестра", "дочь"], "bad": ["отец", "брат", "сын"], "category": "gender"},
        # === Number Agreement (adj) - 10 probes ===
        {"prompt": "Большие ", "good": ["дома", "столы", "коты"], "bad": ["дом", "стол", "кот"], "category": "num_adj"},
        {"prompt": "Новые ", "good": ["друзья", "города", "годы"], "bad": ["друг", "город", "год"], "category": "num_adj"},
        {"prompt": "Старые ", "good": ["люди", "леса", "реки"], "bad": ["человек", "лес", "река"], "category": "num_adj"},
        {"prompt": "Красивые ", "good": ["цветы", "дни", "ночи"], "bad": ["цветок", "день", "ночь"], "category": "num_adj"},
        {"prompt": "Хорошие ", "good": ["отцы", "братья", "сыновья"], "bad": ["отец", "брат", "сын"], "category": "num_adj"},
        {"prompt": "Маленькие ", "good": ["дети", "птицы", "кошки"], "bad": ["ребёнок", "птица", "кошка"], "category": "num_adj"},
        {"prompt": "Белые ", "good": ["облака", "стены", "цветы"], "bad": ["облако", "стена", "цветок"], "category": "num_adj"},
        {"prompt": "Чёрные ", "good": ["коты", "собаки", "птицы"], "bad": ["кот", "собака", "птица"], "category": "num_adj"},
        {"prompt": "Высокие ", "good": ["деревья", "горы", "здания"], "bad": ["дерево", "гора", "здание"], "category": "num_adj"},
        {"prompt": "Длинные ", "good": ["дороги", "реки", "ночи"], "bad": ["дорога", "река", "ночь"], "category": "num_adj"},
        # === Case Agreement - 10 probes ===
        {"prompt": "Вижу большого ", "good": ["кота", "пса", "мальчика"], "bad": ["кот", "пёс", "мальчик"], "category": "case"},
        {"prompt": "Вижу красивую ", "good": ["девушку", "машину", "книгу"], "bad": ["девушка", "машина", "книга"], "category": "case"},
        {"prompt": "В большом ", "good": ["доме", "городе", "лесу"], "bad": ["дом", "город", "лес"], "category": "case"},
        {"prompt": "На высокой ", "good": ["горе", "башне", "крыше"], "bad": ["гора", "башня", "крыша"], "category": "case"},
        {"prompt": "С хорошим ", "good": ["другом", "человеком", "отцом"], "bad": ["друг", "человек", "отец"], "category": "case"},
        {"prompt": "Для новой ", "good": ["работы", "жизни", "книги"], "bad": ["работа", "жизнь", "книга"], "category": "case"},
        {"prompt": "О старом ", "good": ["друге", "городе", "времени"], "bad": ["друг", "город", "время"], "category": "case"},
        {"prompt": "К большому ", "good": ["дому", "озеру", "морю"], "bad": ["дом", "озеро", "море"], "category": "case"},
        {"prompt": "Из маленького ", "good": ["города", "села", "дома"], "bad": ["город", "село", "дом"], "category": "case"},
        {"prompt": "За высоким ", "good": ["забором", "деревом", "домом"], "bad": ["забор", "дерево", "дом"], "category": "case"},
        # === Collocations - 10 probes ===
        {"prompt": "чёрный и ", "good": ["белый"], "bad": ["красный", "синий", "зелёный"], "category": "colloc"},
        {"prompt": "день и ", "good": ["ночь"], "bad": ["утро", "вечер", "солнце"], "category": "colloc"},
        {"prompt": "вверх и ", "good": ["вниз"], "bad": ["большой", "маленький", "длинный"], "category": "colloc"},
        {"prompt": "горячий и ", "good": ["холодный"], "bad": ["тёплый", "прохладный", "сухой"], "category": "colloc"},
        {"prompt": "левый и ", "good": ["правый"], "bad": ["верхний", "нижний", "передний"], "category": "colloc"},
        {"prompt": "соль и ", "good": ["перец"], "bad": ["сахар", "специя", "вода"], "category": "colloc"},
        {"prompt": "хлеб и ", "good": ["масло"], "bad": ["сыр", "молоко", "вода"], "category": "colloc"},
        {"prompt": "муж и ", "good": ["жена"], "bad": ["мужчина", "дочь", "ребёнок"], "category": "colloc"},
        {"prompt": "царь и ", "good": ["царица"], "bad": ["принц", "герцог", "рыцарь"], "category": "colloc"},
        {"prompt": "брат и ", "good": ["сестра"], "bad": ["отец", "мать", "сын"], "category": "colloc"},
    ],

    "id": [
        # === Classifiers - 10 probes ===
        {"prompt": "seekor ", "good": ["kucing", "anjing", "burung"], "bad": ["buku", "meja", "rumah"], "category": "classifier"},
        {"prompt": "seekor ", "good": ["ikan", "kuda", "sapi"], "bad": ["kursi", "pintu", "jendela"], "category": "classifier"},
        {"prompt": "seekor ", "good": ["ayam", "bebek", "kambing"], "bad": ["pensil", "tas", "sepatu"], "category": "classifier"},
        {"prompt": "seekor ", "good": ["gajah", "harimau", "singa"], "bad": ["mobil", "sepeda", "kereta"], "category": "classifier"},
        {"prompt": "seekor ", "good": ["kelinci", "tikus", "ular"], "bad": ["televisi", "komputer", "telepon"], "category": "classifier"},
        {"prompt": "sebuah ", "good": ["buku", "meja", "rumah"], "bad": ["kucing", "anjing", "burung"], "category": "classifier"},
        {"prompt": "sebuah ", "good": ["kursi", "pintu", "jendela"], "bad": ["ikan", "kuda", "sapi"], "category": "classifier"},
        {"prompt": "sebuah ", "good": ["pensil", "tas", "sepatu"], "bad": ["ayam", "bebek", "kambing"], "category": "classifier"},
        {"prompt": "sebuah ", "good": ["mobil", "sepeda", "kereta"], "bad": ["gajah", "harimau", "singa"], "category": "classifier"},
        {"prompt": "sebuah ", "good": ["televisi", "komputer", "telepon"], "bad": ["kelinci", "tikus", "ular"], "category": "classifier"},
        # === Collocations - 10 probes ===
        {"prompt": "hitam dan ", "good": ["putih"], "bad": ["merah", "biru", "hijau"], "category": "colloc"},
        {"prompt": "siang dan ", "good": ["malam"], "bad": ["pagi", "sore", "matahari"], "category": "colloc"},
        {"prompt": "atas dan ", "good": ["bawah"], "bad": ["besar", "kecil", "panjang"], "category": "colloc"},
        {"prompt": "panas dan ", "good": ["dingin"], "bad": ["hangat", "sejuk", "kering"], "category": "colloc"},
        {"prompt": "kiri dan ", "good": ["kanan"], "bad": ["atas", "bawah", "depan"], "category": "colloc"},
        {"prompt": "garam dan ", "good": ["merica"], "bad": ["gula", "bumbu", "air"], "category": "colloc"},
        {"prompt": "roti dan ", "good": ["mentega"], "bad": ["keju", "susu", "air"], "category": "colloc"},
        {"prompt": "suami dan ", "good": ["istri"], "bad": ["pria", "anak", "wanita"], "category": "colloc"},
        {"prompt": "raja dan ", "good": ["ratu"], "bad": ["pangeran", "duke", "ksatria"], "category": "colloc"},
        {"prompt": "kakak dan ", "good": ["adik"], "bad": ["ayah", "ibu", "anak"], "category": "colloc"},
    ],

    "vi": [
        # === Classifiers - 10 probes ===
        {"prompt": "một con ", "good": ["mèo", "chó", "chim"], "bad": ["sách", "bàn", "nhà"], "category": "classifier"},
        {"prompt": "một con ", "good": ["cá", "ngựa", "bò"], "bad": ["ghế", "cửa", "cửa sổ"], "category": "classifier"},
        {"prompt": "một con ", "good": ["gà", "vịt", "dê"], "bad": ["bút", "túi", "giày"], "category": "classifier"},
        {"prompt": "một con ", "good": ["voi", "hổ", "sư tử"], "bad": ["xe", "xe đạp", "tàu"], "category": "classifier"},
        {"prompt": "một con ", "good": ["thỏ", "chuột", "rắn"], "bad": ["tivi", "máy tính", "điện thoại"], "category": "classifier"},
        {"prompt": "một quyển ", "good": ["sách", "vở", "tạp chí"], "bad": ["mèo", "chó", "chim"], "category": "classifier"},
        {"prompt": "một cái ", "good": ["bàn", "ghế", "cửa"], "bad": ["cá", "ngựa", "bò"], "category": "classifier"},
        {"prompt": "một cái ", "good": ["bút", "túi", "giày"], "bad": ["gà", "vịt", "dê"], "category": "classifier"},
        {"prompt": "một chiếc ", "good": ["xe", "xe đạp", "thuyền"], "bad": ["voi", "hổ", "sư tử"], "category": "classifier"},
        {"prompt": "một ngôi ", "good": ["nhà", "chùa", "đền"], "bad": ["thỏ", "chuột", "rắn"], "category": "classifier"},
        # === Collocations - 10 probes ===
        {"prompt": "đen và ", "good": ["trắng"], "bad": ["đỏ", "xanh", "vàng"], "category": "colloc"},
        {"prompt": "ngày và ", "good": ["đêm"], "bad": ["sáng", "chiều", "mặt trời"], "category": "colloc"},
        {"prompt": "trên và ", "good": ["dưới"], "bad": ["to", "nhỏ", "dài"], "category": "colloc"},
        {"prompt": "nóng và ", "good": ["lạnh"], "bad": ["ấm", "mát", "khô"], "category": "colloc"},
        {"prompt": "trái và ", "good": ["phải"], "bad": ["trên", "dưới", "trước"], "category": "colloc"},
        {"prompt": "muối và ", "good": ["tiêu"], "bad": ["đường", "gia vị", "nước"], "category": "colloc"},
        {"prompt": "bánh mì và ", "good": ["bơ"], "bad": ["phô mai", "sữa", "nước"], "category": "colloc"},
        {"prompt": "chồng và ", "good": ["vợ"], "bad": ["đàn ông", "con", "phụ nữ"], "category": "colloc"},
        {"prompt": "vua và ", "good": ["hoàng hậu"], "bad": ["hoàng tử", "công tước", "hiệp sĩ"], "category": "colloc"},
        {"prompt": "anh và ", "good": ["em"], "bad": ["bố", "mẹ", "con"], "category": "colloc"},
    ],

    "zh": [
        # === Classifiers - 10 probes ===
        {"prompt": "一只 ", "good": ["猫", "狗", "鸟"], "bad": ["书", "桌", "房"], "category": "classifier"},
        {"prompt": "一只 ", "good": ["鱼", "鸡", "鸭"], "bad": ["椅", "门", "窗"], "category": "classifier"},
        {"prompt": "一只 ", "good": ["兔", "虎", "象"], "bad": ["笔", "包", "鞋"], "category": "classifier"},
        {"prompt": "一只 ", "good": ["羊", "牛", "马"], "bad": ["车", "船", "机"], "category": "classifier"},
        {"prompt": "一只 ", "good": ["蛇", "鼠", "蛙"], "bad": ["电视", "电脑", "手机"], "category": "classifier"},
        {"prompt": "一本 ", "good": ["书", "杂志", "词典"], "bad": ["猫", "狗", "鸟"], "category": "classifier"},
        {"prompt": "一张 ", "good": ["桌", "椅", "床"], "bad": ["鱼", "鸡", "鸭"], "category": "classifier"},
        {"prompt": "一支 ", "good": ["笔", "枪", "箭"], "bad": ["兔", "虎", "象"], "category": "classifier"},
        {"prompt": "一辆 ", "good": ["车", "自行车", "摩托车"], "bad": ["羊", "牛", "马"], "category": "classifier"},
        {"prompt": "一间 ", "good": ["房", "屋", "店"], "bad": ["蛇", "鼠", "蛙"], "category": "classifier"},
        # === Collocations - 10 probes ===
        {"prompt": "黑与 ", "good": ["白"], "bad": ["红", "蓝", "绿"], "category": "colloc"},
        {"prompt": "日与 ", "good": ["夜"], "bad": ["晨", "午", "阳"], "category": "colloc"},
        {"prompt": "上与 ", "good": ["下"], "bad": ["大", "小", "长"], "category": "colloc"},
        {"prompt": "冷与 ", "good": ["热"], "bad": ["温", "凉", "干"], "category": "colloc"},
        {"prompt": "左与 ", "good": ["右"], "bad": ["上", "下", "前"], "category": "colloc"},
        {"prompt": "盐与 ", "good": ["胡椒"], "bad": ["糖", "料", "水"], "category": "colloc"},
        {"prompt": "面包与 ", "good": ["黄油"], "bad": ["奶酪", "牛奶", "水"], "category": "colloc"},
        {"prompt": "丈夫与 ", "good": ["妻子"], "bad": ["男人", "孩子", "女人"], "category": "colloc"},
        {"prompt": "国王与 ", "good": ["王后"], "bad": ["王子", "公爵", "骑士"], "category": "colloc"},
        {"prompt": "兄与 ", "good": ["弟"], "bad": ["父", "母", "子"], "category": "colloc"},
    ],
}

# Synthetic languages use EN probes (they're English-based)
GRAMMAR_PROBES["synth_a"] = GRAMMAR_PROBES["en"].copy()
GRAMMAR_PROBES["synth_b"] = GRAMMAR_PROBES["en"].copy()
GRAMMAR_PROBES["synth_c"] = GRAMMAR_PROBES["en"].copy()
GRAMMAR_PROBES["synth_d"] = GRAMMAR_PROBES["en"].copy()


def get_token_probability(model, tokenizer, prompt: str, target: str, device) -> float:
    """
    Get probability of target sequence given prompt.

    BUG (FIXED): XLM-R tokenizer splits inflected forms differently than custom BPE.
    For example:
        "sits" -> ['▁sit', 's']  (2 tokens)
        "sit"  -> ['▁sit']       (1 token)

    The old code only checked P(first_token). This meant the probe was comparing:
        P(▁sit | "The cat ") vs P(▁sit | "The cats ")

    Since both "sits" and "sit" share the same first token (▁sit), the probe
    was NOT actually testing the inflection at all for these verbs!

    This explains why EN grammar was artificially high in exp8 - the probes
    were broken for XLM-R tokenization when verbs share stems.

    FIX: Compute joint probability of ALL tokens in the target sequence.
    P("sits" | prompt) = P(▁sit | prompt) * P(s | prompt, ▁sit)
    P("sit" | prompt)  = P(▁sit | prompt)

    These are now different, correctly testing whether the model prefers
    the inflected vs uninflected form.
    """
    prompt_ids = tokenizer.encode(prompt).ids

    # Add space prefix for proper tokenization
    target_ids = tokenizer.encode(" " + target).ids
    if not target_ids:
        target_ids = tokenizer.encode(target).ids
    if not target_ids:
        return 0.0

    # For Chinese/CJK: token 6 is ▁ (space marker), skip it
    if target_ids[0] == 6 and len(target_ids) > 1:
        target_ids = target_ids[1:]

    # Concatenate prompt + target for autoregressive scoring
    full_ids = prompt_ids + target_ids
    x = torch.tensor([full_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(x)
        # Apply temperature for stable measurements
        logits = logits / GRAMMAR_PROBE_TEMPERATURE

        # Compute log probability of each target token at its position
        # Position i in logits predicts token i+1
        # So logits[prompt_len-1] predicts target[0], logits[prompt_len] predicts target[1], etc.
        log_prob_sum = 0.0
        for i, target_token in enumerate(target_ids):
            pos = len(prompt_ids) - 1 + i  # Position in logits that predicts this token
            log_probs = F.log_softmax(logits[0, pos, :], dim=-1)
            log_prob_sum += log_probs[target_token].item()

        # Return joint probability (exp of sum of log probs)
        # Normalize by sequence length to avoid penalizing longer targets
        return np.exp(log_prob_sum / len(target_ids))


def run_grammar_probes(model, tokenizer, lang, device):
    """Evaluate grammar with next-token prediction (exp1 method)."""
    model.eval()
    probes = GRAMMAR_PROBES.get(lang, GRAMMAR_PROBES.get("en", []))

    if not probes:
        return 0.5, 0.0, {}

    correct = 0
    total = 0
    log_ratios = []
    by_category = {}

    for probe in probes:
        prompt = probe["prompt"]
        good_words = probe["good"]
        bad_words = probe["bad"]

        # Get probabilities for good and bad completions
        good_probs = [get_token_probability(model, tokenizer, prompt, w, device) for w in good_words]
        bad_probs = [get_token_probability(model, tokenizer, prompt, w, device) for w in bad_words]

        avg_good = np.mean(good_probs)
        avg_bad = np.mean(bad_probs)

        # Log ratio for numeric stability (exp1 method)
        log_ratio = np.log(avg_good + 1e-10) - np.log(avg_bad + 1e-10)

        # Correct if average good prob > average bad prob
        is_correct = avg_good > avg_bad
        if is_correct:
            correct += 1
        total += 1
        log_ratios.append(log_ratio)

        category = probe.get("category", "other")
        if category not in by_category:
            by_category[category] = {"correct": 0, "total": 0}
        by_category[category]["total"] += 1
        if is_correct:
            by_category[category]["correct"] += 1

    for cat in by_category:
        t = by_category[cat]["total"]
        by_category[cat]["accuracy"] = by_category[cat]["correct"] / t if t > 0 else 0.0

    model.train()
    accuracy = correct / total if total > 0 else 0.5
    mean_lr = np.mean(log_ratios) if log_ratios else 0
    return accuracy, mean_lr, by_category


# ==================== VALIDATION ====================
def load_validation_tokens(data_path: Path, lang: str, max_tokens: int = 50000):
    """Load validation tokens."""
    val_dir = data_path / "validation"
    val_path = val_dir / f"{lang}_val_chunk_000000.npy"

    if val_path.exists():
        return np.load(val_path)[:max_tokens]

    # Try gzipped
    val_path_gz = val_dir / f"{lang}_val_chunk_000000.npy.gz"
    if val_path_gz.exists():
        with gzip.open(val_path_gz, 'rb') as f:
            return np.load(f)[:max_tokens]

    # Fall back to last training chunk
    chunks = sorted((data_path / "chunks").glob(f"{lang}_chunk_*.npy"))
    if chunks:
        return np.load(chunks[-1])[:max_tokens]

    return None


def compute_validation_ppl(model, val_tokens, device, seq_len=512):
    """Compute validation perplexity."""
    if val_tokens is None or len(val_tokens) < seq_len + 1:
        return {"ppl": float("nan"), "loss": float("nan")}

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_size = 4
    n_seqs = min(len(val_tokens) // (seq_len + 1), 50)

    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            batch_seqs = []
            for j in range(i, min(i + batch_size, n_seqs)):
                start = j * seq_len
                seq = val_tokens[start:start + seq_len + 1]
                if len(seq) == seq_len + 1:
                    batch_seqs.append(seq)

            if not batch_seqs:
                continue

            batch = np.stack(batch_seqs)
            x = torch.tensor(batch[:, :-1], dtype=torch.long, device=device)
            y = torch.tensor(batch[:, 1:], dtype=torch.long, device=device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    ppl = float(np.exp(min(avg_loss, 20)))
    return {"ppl": ppl, "loss": avg_loss}


# ==================== HURST EXPONENT ====================
def hurst_exponent(series: np.ndarray, max_lag: int = None) -> float:
    """Compute Hurst exponent via R/S analysis."""
    n = len(series)
    if n < 100:
        return float("nan")

    if max_lag is None:
        max_lag = min(n // 4, 5000)

    lags = []
    rs_values = []

    for lag in range(10, max_lag, max(1, max_lag // 50)):
        rs_list = []
        for start in range(0, n - lag, lag):
            subseries = series[start:start + lag]
            mean = np.mean(subseries)
            cumdev = np.cumsum(subseries - mean)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(subseries, ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)

        if rs_list:
            lags.append(lag)
            rs_values.append(np.mean(rs_list))

    if len(lags) < 2:
        return float("nan")

    log_lags = np.log(lags)
    log_rs = np.log(rs_values)
    slope, _ = np.polyfit(log_lags, log_rs, 1)
    return float(slope)


def compute_model_hurst(model, tokens: np.ndarray, device, max_seq_len: int = 512) -> dict:
    """Compute Hurst exponents from model internals.

    Process tokens in chunks to respect model's max position embeddings.
    """
    model.eval()

    all_losses = []
    all_entropies = []
    all_top_k = []

    # Process in chunks of max_seq_len
    with torch.no_grad():
        for start in range(0, len(tokens) - 1, max_seq_len):
            end = min(start + max_seq_len, len(tokens) - 1)
            chunk_tokens = tokens[start:end + 1]

            x = torch.tensor(chunk_tokens[:-1].reshape(1, -1), dtype=torch.long, device=device)
            targets = chunk_tokens[1:]

            B, T = x.shape
            pos = torch.arange(T, device=device)
            h = model.tok_emb(x) + model.pos_emb(pos)

            for block in model.blocks:
                mask = block.get_mask(T, device)
                h_ln = block.ln1(h)
                attn_out, _ = block.attn(h_ln, h_ln, h_ln, attn_mask=mask)
                h = h + attn_out
                h = h + block.mlp(block.ln2(h))

            h = model.ln_f(h)
            logits = F.linear(h, model.tok_emb.weight)[0].cpu().numpy()

            log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
            losses = -log_probs[np.arange(len(targets)), targets]
            all_losses.extend(losses.tolist())

            probs = np.exp(log_probs)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
            all_entropies.extend(entropy.tolist())

            top_k = np.sort(probs, axis=-1)[:, -10:].sum(axis=-1)
            all_top_k.extend(top_k.tolist())

    H_loss = hurst_exponent(np.array(all_losses))
    H_logit_entropy = hurst_exponent(np.array(all_entropies))
    H_top_k = hurst_exponent(np.array(all_top_k))

    model.train()
    return {
        "H_loss": H_loss,
        "H_logit_entropy": H_logit_entropy,
        "H_top_k": H_top_k,
    }


# ==================== FULL PROBE SUITE ====================
def run_full_probe(model, tokenizer, lang: str, step: int, device, data_path: Path) -> dict:
    """Run complete probe suite."""
    tokens_consumed = step * BATCH_SIZE * SEQ_LEN

    results = {
        "language": lang,
        "step": step,
        "tokens_consumed": tokens_consumed,
        "tokens_M": tokens_consumed / 1e6,
        "tokens_B": tokens_consumed / 1e9,
        "timestamp": datetime.now().isoformat(),
    }

    # Grammar probes
    grammar_acc, grammar_lr, grammar_by_category = run_grammar_probes(model, tokenizer, lang, device)
    results["grammar_acc"] = grammar_acc
    results["grammar_log_ratio"] = grammar_lr
    results["grammar_by_category"] = grammar_by_category

    # Reasoning probes
    reasoning_results = run_reasoning_probes(model, tokenizer, lang, device)
    results["reasoning_acc"] = reasoning_results["accuracy"]
    results["reasoning_correct"] = reasoning_results["correct"]
    results["reasoning_total"] = reasoning_results["total"]
    results["reasoning_by_category"] = reasoning_results["by_category"]
    results["reasoning_samples"] = reasoning_results["samples"]

    # Validation perplexity
    val_tokens = load_validation_tokens(data_path, lang)
    val_results = compute_validation_ppl(model, val_tokens, device)
    results["val_ppl"] = val_results["ppl"]
    results["val_loss"] = val_results["loss"]

    # Model Hurst exponents
    if val_tokens is not None and len(val_tokens) >= 3000:
        hurst_results = compute_model_hurst(model, val_tokens[:3000], device)
        results["H_loss"] = hurst_results["H_loss"]
        results["H_logit_entropy"] = hurst_results["H_logit_entropy"]
        results["H_top_k"] = hurst_results["H_top_k"]
    else:
        results["H_loss"] = float("nan")
        results["H_logit_entropy"] = float("nan")
        results["H_top_k"] = float("nan")

    return results


# ==================== CHECKPOINT (IDEMPOTENT) ====================
def save_checkpoint(model, optimizer, step, loss, lang, probe_results=None):
    """Save checkpoint with atomic write for safety."""
    model_path = CHECKPOINT_DIR / f"model_{step}.safetensors"
    meta_path = CHECKPOINT_DIR / f"checkpoint_{step}.json"
    temp_model = CHECKPOINT_DIR / f".model_{step}.tmp"
    temp_meta = CHECKPOINT_DIR / f".checkpoint_{step}.tmp"

    # Save model (exclude mask buffers)
    state_dict = {k: v for k, v in model.state_dict().items()
                  if 'mask' not in k and k != 'head.weight'}
    save_file(state_dict, str(temp_model))

    # Save metadata
    meta = {
        "step": step,
        "loss": float(loss),
        "lang": lang,
        "model_size": MODEL_SIZE,
        "tokens": step * BATCH_SIZE * SEQ_LEN,
        "timestamp": datetime.now().isoformat(),
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
    }
    if probe_results:
        meta.update(probe_results)

    with open(temp_meta, 'w') as f:
        json.dump(meta, f, indent=2)

    # Atomic rename
    os.rename(temp_model, model_path)
    os.rename(temp_meta, meta_path)

    print(f"  Checkpoint: step {step} ({meta['tokens']/1e6:.0f}M tokens)")

    # Cleanup old checkpoints (keep important + last 3)
    important_steps = {TOKENS_500M, TOKENS_1B, TOKENS_2B}
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*.json"),
                        key=lambda x: int(x.stem.split('_')[1]))

    for old in checkpoints[:-3]:
        old_step = int(old.stem.split('_')[1])
        if old_step not in important_steps and old_step % BACKUP_INTERVAL != 0:
            old.unlink()
            (CHECKPOINT_DIR / f"model_{old_step}.safetensors").unlink(missing_ok=True)


def load_checkpoint(model, optimizer=None):
    """Load latest checkpoint (idempotent resume)."""
    checkpoints = list(CHECKPOINT_DIR.glob("checkpoint_*.json"))
    if not checkpoints:
        return 0

    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest.stem.split('_')[1])

    model_path = CHECKPOINT_DIR / f"model_{step}.safetensors"
    if model_path.exists():
        state_dict = load_file(str(model_path))
        model.load_state_dict(state_dict, strict=False)
        print(f"Resumed from step {step} ({step * BATCH_SIZE * SEQ_LEN / 1e9:.2f}B tokens)")

    return step


# ==================== TRAINING ====================
def main():
    # Model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params/1e6:.1f}M (inflated by {VOCAB_SIZE:,} vocab)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)

    # Resume from checkpoint (idempotent)
    start_step = load_checkpoint(model, optimizer)

    if start_step >= TARGET_STEPS:
        print(f"Already completed {start_step} steps. Nothing to do.")
        return

    # Data
    try:
        dataset = ChunkDataset(DATA_PATH / "chunks", LANG, SEQ_LEN)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"Skipping {LANG} - no data available yet")
        return

    # Log files
    log_file = LOG_DIR / f"{LANG}_{MODEL_SIZE}.csv"
    probe_log = LOG_DIR / f"{LANG}_{MODEL_SIZE}_probes.jsonl"

    if not log_file.exists() or start_step == 0:
        with open(log_file, 'w') as f:
            f.write("step,loss,ppl,tokens_M,grammar_acc,reasoning_acc,val_ppl,H_logit_entropy,H_top_k,timestamp\n")

    # Signal handler for graceful shutdown
    shutdown = False
    def handler(sig, frame):
        nonlocal shutdown
        print(f"\n[{LANG}] Shutdown requested, saving checkpoint...")
        shutdown = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # AMP setup for memory efficiency and speed
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"AMP enabled: {USE_AMP}, dtype: {autocast_dtype}")

    # Training loop
    model.train()
    running_loss = 0
    pbar = tqdm(range(start_step + 1, TARGET_STEPS + 1),
                desc=f"{LANG}", initial=start_step, total=TARGET_STEPS)

    for step in pbar:
        if shutdown:
            save_checkpoint(model, optimizer, step - 1, running_loss, LANG)
            break

        x, y = dataset.get_batch(BATCH_SIZE)
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass with automatic mixed precision
        with torch.amp.autocast('cuda', dtype=autocast_dtype, enabled=USE_AMP):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss = 0.9 * running_loss + 0.1 * loss.item() if running_loss else loss.item()

        if step % 50 == 0:
            ppl = np.exp(min(running_loss, 10))
            tokens_m = step * BATCH_SIZE * SEQ_LEN / 1e6
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({"loss": f"{running_loss:.3f}", "ppl": f"{ppl:.1f}",
                            "tok": f"{tokens_m:.0f}M", "mem": f"{mem:.1f}G"})

        if step % 500 == 0:
            tokens_m = step * BATCH_SIZE * SEQ_LEN / 1e6
            ppl = np.exp(min(running_loss, 20))
            with open(log_file, 'a') as f:
                f.write(f"{step},{running_loss:.6f},{ppl:.2f},{tokens_m:.1f},,,,,{datetime.now().isoformat()}\n")

        # Rescan for new chunks every 500 steps (in case uploads are ongoing)
        if step % 500 == 0:
            dataset.rescan_chunks()

        if step % CHECKPOINT_INTERVAL == 0:
            print(f"\n[{LANG}] Step {step}: Running probes...")
            probe_results = run_full_probe(model, TOKENIZER, LANG, step, DEVICE, DATA_PATH)

            tokens_m = step * BATCH_SIZE * SEQ_LEN / 1e6
            ppl = np.exp(min(running_loss, 20))

            print(f"  Loss: {running_loss:.4f} | PPL: {ppl:.1f} | ValPPL: {probe_results['val_ppl']:.1f}")
            print(f"  Grammar: {probe_results['grammar_acc']*100:.0f}% | Reasoning: {probe_results['reasoning_acc']*100:.0f}%")
            print(f"  H_logit: {probe_results['H_logit_entropy']:.3f} | H_top_k: {probe_results['H_top_k']:.3f}")

            # CSV log
            with open(log_file, 'a') as f:
                f.write(f"{step},{running_loss:.6f},{ppl:.2f},{tokens_m:.1f},"
                        f"{probe_results['grammar_acc']:.4f},{probe_results['reasoning_acc']:.4f},"
                        f"{probe_results['val_ppl']:.2f},"
                        f"{probe_results['H_logit_entropy']:.4f},{probe_results['H_top_k']:.4f},"
                        f"{datetime.now().isoformat()}\n")

            # JSONL log
            with open(probe_log, 'a') as f:
                f.write(json.dumps(probe_results) + "\n")

            save_checkpoint(model, optimizer, step, running_loss, LANG, probe_results)

    final_step = step if 'step' in dir() else start_step
    print(f"\n[{LANG}] Training {'completed' if final_step >= TARGET_STEPS else 'interrupted'}")
    print(f"  Final step: {final_step:,}")
    print(f"  Total tokens: {final_step * BATCH_SIZE * SEQ_LEN / 1e9:.2f}B")


if __name__ == "__main__":
    main()
