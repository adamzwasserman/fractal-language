# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the experimental codebase for **The Language-Only Hypothesis** paper (`paper/The Language-Only Hypothesis.pdf`).

**Core claim**: Emergent capabilities in LLMs are properties of natural language structure, not of neural networks or scale. The technology functions as *instrumentation* — it reveals structure already present in language, like a microscope reveals cells it doesn't create.

**Experiment**: Train identical transformers on English vs French C4 corpora, holding all variables constant. If emergence thresholds vary by language, this falsifies scale as the primary driver of emergence.

**Prediction**: French's richer morphological marking (redundant agreement signals) will lower emergence thresholds — 125M French should outperform 350M English on multiple capability clusters.

## Commands

### Training

Two architecture variants to test "instrument replaceability":

```bash
# GPT-2 style (LayerNorm, GELU, learned positions)
uv run python scripts/train_resumable.py <lang> <model_size> <mode>
uv run python scripts/train_auto_restart.py en 125M auto

# LLaMA style (RMSNorm, SwiGLU, RoPE)
uv run python scripts/train_resumable_llama.py <lang> <model_size> <mode>
uv run python scripts/train_auto_restart_llama.py fr 125M auto

# Arguments:
#   lang: en | fr
#   model_size: 125M | 350M
#   mode: auto | daytime | nighttime
```

### Data Preparation

```bash
# Download and preprocess C4 corpus (creates text files)
uv run python scripts/dataset_builder.py

# Convert text to tokenized .npy chunks
uv run python scripts/create_chunks.py
```

### Evaluation

```bash
# Run capability probes on a checkpoint
uv run python scripts/enhanced_capability_probes.py <checkpoint_step> <lang>

# Calculate perplexity on validation set
uv run python scripts/validation_perplexity.py <checkpoint_step> <lang>
uv run python scripts/validation_perplexity.py create_validation_sets
```

## Architecture

### Two Architecture Variants

Testing "instrument replaceability" — if the French advantage holds across both architectures, it's the language, not the architecture.

| Component | GPT-2 Style | LLaMA Style |
|-----------|-------------|-------------|
| Normalization | LayerNorm | RMSNorm |
| Activation | GELU | SwiGLU |
| Position encoding | Learned/implicit | Rotary (RoPE) |
| Checkpoints | `checkpoints/{lang}/` | `checkpoints/{lang}_llama/` |

### Model Configurations

- **125M**: 12 layers, d_model=768, 12 heads, batch=2, seq=512
  - GPT-2: d_ff=3072
  - LLaMA: d_ff=2048 (SwiGLU uses 3 matrices, reduced for param parity)
- **350M**: 24 layers, d_model=1024, 16 heads, batch=1, seq=512
  - GPT-2: d_ff=4096
  - LLaMA: d_ff=2730

### Data Pipeline

1. `dataset_builder.py` → Streams C4, deduplicates via MD5, trains joint 50k BPE tokenizer
2. `create_chunks.py` → Tokenizes text into 1M-token .npy chunks (uint32, ~4MB each)
3. Training uses mmap-mode chunk loading for memory efficiency

### External Data Paths

All heavy data lives on external drive at `/Volumes/Misc Backup/fractal/`:
- `joint_tokenizer.json` - Shared tokenizer for both languages
- `chunks/` - Tokenized training data (`{lang}_chunk_*.npy`)
- `checkpoints/{lang}/{model_size}/` - GPT-2 style checkpoints
- `checkpoints/{lang}_llama/{model_size}/` - LLaMA style checkpoints
- `validation/` - Held-out validation sequences
- `results/` - Evaluation outputs

### Checkpoint Format

Checkpoints consist of:
- `model_{step}.safetensors` - Flattened model parameters
- `checkpoint_{step}.json` - Metadata (step, loss, config, timestamp)

The checkpoint system handles nested MLX parameter structures by flattening during save and reconstructing on load.

### Memory Management

Training uses aggressive memory limits (8GB daytime / 10GB nighttime) and checkpoints every 1000 steps to enable auto-restart when hitting limits. The `train_auto_restart.py` wrapper relaunches training sessions automatically.

## Experiment Design

Fixed random seed (42) ensures reproducibility. Both languages use identical hyperparameters - no memory-based adjustments allowed to maintain controlled comparison.
