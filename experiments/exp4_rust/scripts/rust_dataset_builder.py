#!/usr/bin/env python
"""Download and prepare Rust corpus for Experiment 4"""
import hashlib
import gc
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tqdm import tqdm
import json

OUTPUT_DIR = Path("/workspace/data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MIN_FILE_SIZE = 100
MAX_FILE_SIZE = 100_000
MIN_ALPHANUM = 0.25
TOKENIZER_SAMPLE_SIZE = 500_000

def stream_rust_files():
    print("Loading Rust dataset from HuggingFace...")
    ds = load_dataset("ammarnasr/the-stack-rust-clean", split="train", streaming=True)
    seen = set()
    for example in ds:
        content = example["content"]
        size = example.get("size", len(content))
        alphanum = example.get("alphanum_fraction", 0.5)
        if size < MIN_FILE_SIZE or size > MAX_FILE_SIZE:
            continue
        if alphanum < MIN_ALPHANUM:
            continue
        h = hashlib.md5(content.encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        yield content

def main():
    print("Downloading Rust corpus...")
    filename = OUTPUT_DIR / "rust_texts.txt"
    tokenizer_sample = []
    total = 0
    
    with open(filename, 'w', encoding='utf-8') as f:
        for content in tqdm(stream_rust_files(), desc="Rust files"):
            escaped = content.replace('\\n', '\\\\n').replace('\\r', '\\\\r')
            f.write(escaped + '\\n')
            total += 1
            if len(tokenizer_sample) < TOKENIZER_SAMPLE_SIZE:
                tokenizer_sample.append(content)
    
    print(f"Downloaded {total:,} Rust files")
    
    # Train tokenizer
    print(f"Training tokenizer on {len(tokenizer_sample):,} samples...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=50_000, special_tokens=["<pad>", "<unk>", "<eos>"])
    tokenizer.train_from_iterator(tokenizer_sample, trainer=trainer)
    tokenizer.save(str(OUTPUT_DIR / "rust_tokenizer.json"))
    print(f"Tokenizer saved to {OUTPUT_DIR / 'rust_tokenizer.json'}")
    
    print("Done!")

if __name__ == "__main__":
    main()
