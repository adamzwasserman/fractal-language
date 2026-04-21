#!/usr/bin/env python
"""
Create held-out validation sets and measure perplexity for controlled experiment.
Critical for fair comparison between French and English models.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import json
from datetime import datetime
import sys

def create_validation_sets(num_sequences=1000):
    """Create held-out validation sets from chunk data."""
    base_path = Path("/Volumes/Misc Backup/fractal")
    chunk_dir = base_path / "chunks" 
    validation_dir = base_path / "validation"
    
    validation_data = {}
    
    for lang in ['en', 'fr']:
        print(f"🔍 Creating {lang.upper()} validation set...")
        chunks = sorted(chunk_dir.glob(f"{lang}_chunk_*.npy"))
        
        if not chunks:
            print(f"❌ No chunks found for {lang}")
            continue
        
        # Use last few chunks as validation (untouched by training)
        val_chunks = chunks[-5:]  # Last 5 chunks as validation
        sequences = []
        
        for chunk_path in val_chunks:
            tokens = np.load(chunk_path, mmap_mode='r')
            
            # Extract non-overlapping sequences
            seq_len = 512
            for i in range(0, len(tokens) - seq_len, seq_len):  # No overlap for validation
                if len(sequences) >= num_sequences:
                    break
                sequences.append(tokens[i:i+seq_len].tolist())
            
            if len(sequences) >= num_sequences:
                break
        
        # Save validation set
        val_file = validation_dir / f"{lang}_validation.json"
        val_data = {
            "language": lang,
            "num_sequences": len(sequences),
            "sequence_length": 512,
            "source_chunks": [str(p.name) for p in val_chunks],
            "created": datetime.now().isoformat(),
            "sequences": sequences
        }
        
        with open(val_file, 'w') as f:
            json.dump(val_data, f)
        
        validation_data[lang] = {
            "file": str(val_file),
            "num_sequences": len(sequences),
            "avg_tokens_per_seq": np.mean([len(seq) for seq in sequences])
        }
        
        print(f"✅ {lang.upper()}: {len(sequences)} validation sequences → {val_file.name}")
    
    return validation_data

def calculate_perplexity(model_path, validation_data_path, tokenizer_path):
    """Calculate perplexity on held-out validation set."""
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    
    # Recreate model architecture (must match training)
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff):
            super().__init__()
            self.attention = nn.MultiHeadAttention(d_model, n_heads)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
        
        def __call__(self, x):
            attn_out = self.attention(self.ln1(x), self.ln1(x), self.ln1(x))
            x = x + attn_out
            mlp_out = self.mlp(self.ln2(x))
            x = x + mlp_out
            return x

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            # 125M architecture
            cfg = dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072)
            
            self.embed = nn.Embedding(vocab_size, cfg["d_model"])
            self.blocks = [TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["d_ff"])
                          for _ in range(cfg["n_layers"])]
            self.ln_f = nn.LayerNorm(cfg["d_model"])
            self.head = nn.Linear(cfg["d_model"], vocab_size, bias=False)
            self.head.weight = self.embed.weight  # tied

        def __call__(self, x):
            x = self.embed(x)
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln_f(x))
    
    # Load model
    model = Transformer()
    try:
        model.load_weights(str(model_path))
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Load validation data
    with open(validation_data_path) as f:
        val_data = json.load(f)
    
    sequences = val_data["sequences"]
    print(f"📊 Calculating perplexity on {len(sequences)} sequences...")
    
    total_loss = 0.0
    total_tokens = 0
    
    # Calculate loss in batches
    batch_size = 16
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        
        # Prepare batch
        batch_x = [mx.array(seq[:-1]) for seq in batch_seqs]
        batch_y = [mx.array(seq[1:]) for seq in batch_seqs]
        
        x = mx.stack(batch_x)
        y = mx.stack(batch_y)
        
        # Forward pass
        logits = model(x)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        
        total_loss += loss.item() * len(batch_seqs)
        total_tokens += len(batch_seqs) * 511  # seq_len - 1
    
    avg_loss = total_loss / len(sequences)
    perplexity = mx.exp(mx.array(avg_loss)).item()
    
    result = {
        "model_path": str(model_path),
        "validation_sequences": len(sequences),
        "average_loss": avg_loss,
        "perplexity": perplexity,
        "language": val_data["language"],
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"📈 PERPLEXITY RESULTS:")
    print(f"   Language: {val_data['language'].upper()}")
    print(f"   Average loss: {avg_loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    
    return result

def run_validation_analysis(model_checkpoint_step, lang):
    """Run complete validation analysis for a trained model."""
    base_path = Path("/Volumes/Misc Backup/fractal")
    
    # Paths
    model_path = base_path / "checkpoints" / lang / "125M" / f"model_{model_checkpoint_step}.safetensors"
    validation_path = base_path / "validation" / f"{lang}_validation.json"
    tokenizer_path = base_path / "joint_tokenizer.json"
    results_path = base_path / "results" / f"perplexity_analysis_{lang}_{model_checkpoint_step}.json"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    if not validation_path.exists():
        print(f"❌ Creating validation set first...")
        create_validation_sets()
    
    # Calculate perplexity
    result = calculate_perplexity(model_path, validation_path, tokenizer_path)
    
    if result:
        # Save results
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Results saved: {results_path}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python validation_perplexity.py <checkpoint_step> <language>")
        print("   or: python validation_perplexity.py create_validation_sets")
        sys.exit(1)
    
    if sys.argv[1] == "create_validation_sets":
        create_validation_sets()
    else:
        checkpoint_step = int(sys.argv[1])
        language = sys.argv[2]
        run_validation_analysis(checkpoint_step, language)