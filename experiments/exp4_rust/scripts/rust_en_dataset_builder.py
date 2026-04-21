#!/usr/bin/env python
"""Create rust+en interleaved dataset and joint tokenizer"""
import hashlib
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tqdm import tqdm

OUTPUT_DIR = Path("/workspace/data")

def stream_english():
    print("Loading English C4...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    seen = set()
    count = 0
    target = 900_000  # Match rust file count roughly
    for ex in ds:
        text = ex["text"].strip()
        if len(text) < 128:
            continue
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        yield text
        count += 1
        if count >= target:
            break

def main():
    rust_file = OUTPUT_DIR / "rust_texts.txt"
    en_file = OUTPUT_DIR / "en_texts.txt"
    
    # Download English if not exists
    if not en_file.exists():
        print("Downloading English corpus...")
        with open(en_file, "w", encoding="utf-8") as f:
            for text in tqdm(stream_english(), desc="English", total=900_000):
                escaped = text.replace("\\n", "\\\\n").replace("\\r", "\\\\r")
                f.write(escaped + "\\n")
        print(f"English saved to {en_file}")
    else:
        print(f"English already exists: {en_file}")
    
    # Create interleaved file
    print("Creating interleaved rust+en file...")
    out_file = OUTPUT_DIR / "rust_en_texts.txt"
    
    with open(rust_file, "r") as rf, open(en_file, "r") as ef, open(out_file, "w") as of:
        rust_lines = rf.readlines()
        en_lines = ef.readlines()
        
        # Interleave
        total = 0
        for r, e in tqdm(zip(rust_lines, en_lines), desc="Interleaving", total=min(len(rust_lines), len(en_lines))):
            of.write(r)
            of.write(e)
            total += 2
        
        # Write remaining
        if len(rust_lines) > len(en_lines):
            for r in rust_lines[len(en_lines):]:
                of.write(r)
                total += 1
    
    print(f"Interleaved file: {total} documents")
    
    # Train joint tokenizer
    print("Training joint tokenizer...")
    rust_sample = []
    en_sample = []
    
    with open(rust_file, "r") as f:
        for i, line in enumerate(f):
            if i >= 250_000: break
            rust_sample.append(line.strip().replace("\\\\n", "\\n"))
    
    with open(en_file, "r") as f:
        for i, line in enumerate(f):
            if i >= 250_000: break
            en_sample.append(line.strip().replace("\\\\n", "\\n"))
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=50_000, special_tokens=["<pad>", "<unk>", "<eos>"])
    tokenizer.train_from_iterator(rust_sample + en_sample, trainer=trainer)
    tokenizer.save(str(OUTPUT_DIR / "rust_en_tokenizer.json"))
    
    print("Done! Now run: python scripts/create_rust_chunks.py rust_en")

if __name__ == "__main__":
    main()
