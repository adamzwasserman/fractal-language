#!/usr/bin/env python
import hashlib
import gc
import sys
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
from tqdm import tqdm
import psutil
import pickle
import json

TARGET_TOKENS = 10_000_000_000
OUTPUT_DIR = Path("/Volumes/Misc Backup/fractal")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Memory management settings
MAX_MEMORY_GB = 15
CHUNK_SIZE = 100_000  # Process texts in chunks of 100k documents
TOKENIZER_SAMPLE_SIZE = 1_000_000  # Use 1M texts to train tokenizer

def get_memory_usage_gb():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def stream_deduped_texts(lang: str):
    """Stream deduplicated texts from C4 dataset"""
    if lang == "en":
        name = "en"
    elif lang == "fr":
        name = "fr"
    else:
        raise ValueError(f"Unsupported language: {lang}")
    
    ds = load_dataset("allenai/c4", name=name, split="train", streaming=True)
    seen = set()
    rough_tokens = 0
    min_len = 128

    for example in ds:
        text = example["text"].strip()
        if len(text) < min_len:
            continue
        h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rough_tokens += int(len(text.split()) * 1.3)
        yield text
        if rough_tokens >= TARGET_TOKENS * 1.1:
            break

def process_language_streaming(lang: str):
    """Process a language in streaming mode, appending to single file"""
    print(f"Processing {lang.upper()} documents...")
    
    filename = OUTPUT_DIR / f"{lang}_texts.txt"
    texts_buffer = []
    total_docs = 0
    tokenizer_sample = []  # For training tokenizer
    
    stream = stream_deduped_texts(lang)
    pbar = tqdm(desc=f"{lang.upper()} docs", unit="doc")
    
    # Open file in append mode
    with open(filename, 'w', encoding='utf-8') as f:
        for text in stream:
            texts_buffer.append(text)
            total_docs += 1
            pbar.update(1)
            
            # Collect sample for tokenizer training
            if len(tokenizer_sample) < TOKENIZER_SAMPLE_SIZE:
                tokenizer_sample.append(text)
            
            # Check memory usage and flush if needed
            memory_gb = get_memory_usage_gb()
            if len(texts_buffer) >= CHUNK_SIZE or memory_gb > MAX_MEMORY_GB * 0.8:
                # Append to file
                for text in texts_buffer:
                    # Escape newlines to store one document per line
                    escaped_text = text.replace('\n', '\\n').replace('\r', '\\r')
                    f.write(escaped_text + '\n')
                f.flush()  # Ensure data is written to disk
                
                print(f"\nFlushed {len(texts_buffer):,} docs to {filename}")
                print(f"Memory usage: {memory_gb:.1f}GB, Total docs: {total_docs:,}")
                
                # Clear buffer and force garbage collection
                texts_buffer.clear()
                gc.collect()
        
        # Write any remaining texts
        if texts_buffer:
            for text in texts_buffer:
                escaped_text = text.replace('\n', '\\n').replace('\r', '\\r')
                f.write(escaped_text + '\n')
            f.flush()
            print(f"\nWrote final {len(texts_buffer):,} docs to {filename}")
            texts_buffer.clear()
            gc.collect()
    
    pbar.close()
    print(f"{lang.upper()} complete: {total_docs:,} documents saved to {filename}")
    
    return filename, tokenizer_sample

# Process languages in streaming mode
en_file, en_sample = process_language_streaming("en")
fr_file, fr_sample = process_language_streaming("fr")

# Train tokenizer on samples to avoid memory issues
print(f"\nTraining joint 50k BPE tokenizer on {len(en_sample):,} EN + {len(fr_sample):,} FR samples...")
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=50_000, special_tokens=["<pad>", "<unk>", "<eos>"])

# Train on combined samples
tokenizer.train_from_iterator(en_sample + fr_sample, trainer=trainer)
tokenizer.save(str(OUTPUT_DIR / "joint_tokenizer.json"))
print(f"Tokenizer saved to {OUTPUT_DIR / 'joint_tokenizer.json'}")

# Count total documents in files
def count_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

# Save metadata
metadata = {
    "en_file": str(en_file),
    "fr_file": str(fr_file),
    "total_en_docs": count_lines(en_file),
    "total_fr_docs": count_lines(fr_file),
    "tokenizer_file": str(OUTPUT_DIR / "joint_tokenizer.json"),
    "max_memory_gb": MAX_MEMORY_GB,
    "chunk_size": CHUNK_SIZE,
}

with open(OUTPUT_DIR / "dataset_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nDataset processing complete!")
print(f"English file: {en_file} ({metadata['total_en_docs']:,} docs)")
print(f"French file: {fr_file} ({metadata['total_fr_docs']:,} docs)")
print(f"All files saved to: {OUTPUT_DIR}")
print(f"Final memory usage: {get_memory_usage_gb():.1f}GB")
