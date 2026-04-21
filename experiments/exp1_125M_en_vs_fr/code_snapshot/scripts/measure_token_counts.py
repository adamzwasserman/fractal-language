#!/usr/bin/env python
"""
Measure exact token counts for controlled French vs English experiment.
Addresses concern that French C4 is ~20-25% smaller after cleaning.
"""
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import json
import time

def measure_token_counts():
    base_path = Path("/Volumes/Misc Backup/fractal")
    chunk_dir = base_path / "chunks"
    tokenizer_path = base_path / "joint_tokenizer.json"
    results_path = base_path / "results" / "token_count_analysis.json"
    
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        return None
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    
    print(f"📊 CONTROLLED EXPERIMENT TOKEN ANALYSIS")
    print(f"Using joint tokenizer: {vocab_size:,} tokens vocabulary")
    print(f"Sequence length: 512 tokens (fixed)")
    print(f"Batch size: 2 (fixed)\n")
    
    results = {
        "experiment_config": {
            "vocab_size": vocab_size,
            "sequence_length": 512,
            "batch_size": 2,
            "random_seed": 42,
            "tokenizer_type": "joint_multilingual_bpe"
        },
        "languages": {}
    }
    
    for lang in ['en', 'fr']:
        print(f"🔍 Analyzing {lang.upper()}...")
        chunks = sorted(chunk_dir.glob(f"{lang}_chunk_*.npy"))
        
        if not chunks:
            print(f"❌ No chunks found for {lang}")
            continue
            
        lang_stats = {
            "num_chunks": len(chunks),
            "total_tokens": 0,
            "unique_vocab_sample": set(),
            "chunk_sizes": [],
            "effective_training_sequences": 0
        }
        
        start_time = time.time()
        
        for i, chunk_path in enumerate(chunks):
            tokens = np.load(chunk_path, mmap_mode='r')
            chunk_size = len(tokens)
            lang_stats["total_tokens"] += chunk_size
            lang_stats["chunk_sizes"].append(chunk_size)
            
            # Sample vocabulary usage (avoid memory issues)
            sample_size = min(1000, chunk_size)
            sample = tokens[:sample_size]
            lang_stats["unique_vocab_sample"].update(sample.tolist())
            
            # Calculate effective training sequences for this chunk
            seq_len = 512
            overlap = seq_len // 2  # 50% overlap
            chunk_sequences = max(0, (chunk_size - seq_len) // overlap)
            lang_stats["effective_training_sequences"] += chunk_sequences
            
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {i+1:,}/{len(chunks):,} chunks in {elapsed:.1f}s")
        
        # Convert set to count for JSON serialization
        lang_stats["unique_vocab_sampled"] = len(lang_stats["unique_vocab_sample"])
        del lang_stats["unique_vocab_sample"]
        
        # Calculate training statistics
        lang_stats["avg_tokens_per_chunk"] = lang_stats["total_tokens"] / lang_stats["num_chunks"]
        lang_stats["total_training_steps_possible"] = lang_stats["effective_training_sequences"] // 2  # batch_size=2
        
        results["languages"][lang] = lang_stats
        
        elapsed = time.time() - start_time
        print(f"✅ {lang.upper()} complete in {elapsed:.1f}s")
        print(f"   Total tokens: {lang_stats['total_tokens']:,}")
        print(f"   Training sequences: {lang_stats['effective_training_sequences']:,}")
        print(f"   Training steps possible: {lang_stats['total_training_steps_possible']:,}")
        print(f"   Vocabulary coverage sample: {lang_stats['unique_vocab_sampled']:,} unique tokens\n")
    
    # Cross-language comparison
    if 'en' in results["languages"] and 'fr' in results["languages"]:
        en_tokens = results["languages"]["en"]["total_tokens"]
        fr_tokens = results["languages"]["fr"]["total_tokens"]
        
        if en_tokens > 0:
            ratio = fr_tokens / en_tokens
            deficit = en_tokens - fr_tokens
            
            en_steps = results["languages"]["en"]["total_training_steps_possible"]
            fr_steps = results["languages"]["fr"]["total_training_steps_possible"]
            step_ratio = fr_steps / en_steps if en_steps > 0 else 0
            
            comparison = {
                "fr_to_en_token_ratio": ratio,
                "fr_token_deficit": deficit,
                "fr_to_en_step_ratio": step_ratio,
                "recommendation": {}
            }
            
            print(f"🎯 CONTROLLED EXPERIMENT ANALYSIS:")
            print(f"French/English token ratio: {ratio:.3f} ({ratio*100:.1f}%)")
            print(f"French token deficit: {deficit:,} tokens")
            print(f"Training step ratio: {step_ratio:.3f}")
            
            if ratio < 0.75:
                oversample = 1 / ratio
                comparison["recommendation"] = {
                    "action": "oversample_french",
                    "factor": oversample,
                    "reason": f"French has <75% of English tokens ({ratio*100:.1f}%)"
                }
                print(f"⚠️  RECOMMENDATION: Oversample French by {oversample:.2f}x to balance")
            elif ratio < 0.85:
                comparison["recommendation"] = {
                    "action": "monitor_closely", 
                    "reason": f"French has {ratio*100:.1f}% of English tokens - borderline imbalance"
                }
                print(f"⚠️  Monitor closely: French has {ratio*100:.1f}% of English tokens")
            else:
                comparison["recommendation"] = {
                    "action": "proceed_as_is",
                    "reason": f"Token balance acceptable ({ratio*100:.1f}%)"
                }
                print(f"✅ Token balance acceptable: {ratio*100:.1f}%")
            
            results["comparison"] = comparison
    
    # Save results
    results["analysis_timestamp"] = time.time()
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Analysis saved to: {results_path}")
    return results

if __name__ == "__main__":
    measure_token_counts()