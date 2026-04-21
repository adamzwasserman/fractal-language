# English Considered Harmful: How Morphological Poverty Pollutes Language Model Training

I've been running controlled experiments comparing language model training on French vs English, and the results are strange enough that I want to share them for feedback before formal publication.

**TL;DR**: When we train identical transformers on French vs English (same architecture, same hyperparameters, same data source), French achieves 100% grammar accuracy in 197M tokens while English stays at chance after 3B tokens. More surprisingly, when we mix English with French in the same model, French grammar *degrades* from 100% to 50-60%. English appears to actively interfere with learning structured patterns.

## The Experiment

We trained 125M parameter GPT-2 style transformers on C4 corpus, comparing:
- French-only
- English-only
- Interleaved French+English
- Rust-only (code)
- Rust+English

Everything held constant: architecture, learning rate (6e-4), batch size, sequence length (512), random seed (42), tokenizer (joint 50K BPE trained on both languages). Pre-registered on OSF before data collection.

## Results That Made Us Do a Double-Take

**French vs English (controlled comparison):**

| Metric | English | French | Ratio |
|--------|---------|--------|-------|
| Perplexity (step 181k) | ~1340 | ~27 | 50x |
| Grammar probe accuracy | 40% (chance) | 100% | — |
| Tokens to grammar saturation | >3B (never) | ~197M | >15x |

French hits 100% on grammar probes (minimal pairs testing agreement) at step 12,000 and stays there with zero fluctuation. English bounces around 40% forever.

**The pollution experiment (this is the weird part):**

We trained a single model on interleaved French and English chunks. If English merely provided "less signal," French should still learn, just slower.

Instead:
- French grammar dropped from 100% → 50-60%
- English grammar stayed at chance (40%)

English didn't just fail to help. It actively corrupted the French grammatical signal.

**Replication in code:**

We ran the same experiment with Rust:
- Rust-only: 3.7 perplexity, 87% structural probe accuracy
- Rust+English: 41.8 perplexity, 87% accuracy

Same accuracy, 11x worse perplexity. The model can still discriminate correct from incorrect code, but its probability distributions are dramatically more diffuse.

## Why This Might Happen

English is morphologically impoverished. Consider:

**French**: "Les grandes portes sont ouvertes"
- Les (plural), grandes (fem, plural), portes (fem, plural), sont (plural), ouvertes (fem, plural)
- Five words independently confirm gender and number

**English**: "The big doors are open"
- No gender marking, number only on noun
- Structure inferred from word order

Our hypothesis: English trains models to expect that structural information is *not* explicitly marked. This creates a prior against looking for explicit markers. When the model then sees French or Rust with explicit structural marking, that prior interferes.

## The Orthogonality Problem

We discovered something that concerns us about standard evaluation:

| Condition | Perplexity | Grammar Accuracy |
|-----------|------------|------------------|
| French 350M (step 200K) | 69.0 | 100% |
| English 350M (step 200K) | 84.1 | ~60% |
| Rust-only | 3.7 | 86.7% |
| Rust+English | 41.8 | 86.7% |

Perplexity and accuracy appear to be orthogonal dimensions:
- At 350M, PPX has nearly converged (~1.2x ratio) while grammar gap persists (60% vs 100%)
- Rust experiments show identical accuracy with 11x different perplexity

This means **standard metrics (loss, perplexity) miss structural learning deficits**. English can achieve nearly identical perplexity to French while remaining at chance on grammar.

## Scale Makes It Worse

We tested both 125M and 350M models through training completion. For French:
- 125M emerged at ~4.1M tokens (step ~4,000)
- 350M emerged at ~60.9M tokens (step ~119,000)

The larger model needed **14.9x more tokens** to achieve the same grammatical capability. This is backwards from standard scaling intuitions.

Our interpretation: more parameters means more capacity that needs to be filled before structure emerges. For morphologically rich languages, smaller models are more token-efficient.

For English, neither scale achieved grammar emergence:
- English 125M: Still at chance after 14.3M tokens
- English 350M: Still at ~60% (near chance) after 102.4M tokens

Scale might be the only path forward for English—brute-forcing your way past the sparse signal. Which would explain why the field has converged on "scale = capability."

## Limitations (the part where I tell you why this might be wrong)

- Only tested at 125M and 350M scale. Larger models might behave differently.
- Only two natural languages (French, English) and one programming language (Rust).
- Joint tokenizer might affect results. Language-specific tokenizers could show different patterns.
- Grammar probes are a limited evaluation. We might be missing something.
- We're an independent researcher, not a big lab. Limited compute means limited experiments.

## What We're Not Claiming

We're not claiming English is useless for training. English-trained models clearly work. We're not claiming you should train on French instead.

We are claiming:
1. Training efficiency varies dramatically by language morphology
2. Mixing English with structured languages can degrade performance on those languages
3. Standard metrics might be missing important structural deficits
4. The compute requirements derived from English training might not generalize

## Code and Data

Everything is public:
- Pre-registration: OSF 10.17605/OSF.IO/SJ48B
- Training logs: github.com/adamzwasserman/fractal-language
- Checkpoints available on request

We welcome replication attempts. If our methodology is flawed, we want to know.

## Questions I'm Genuinely Uncertain About

1. Would this replicate with other morphologically rich languages (Russian, Arabic, Finnish)?
2. Is there a model scale where the effect disappears?
3. Can staged training (French first, then English) preserve the French advantage?
4. What's actually happening in the weights? We've only measured behavior.

---

*Pre-registration: OSF 10.17605/OSF.IO/SJ48B*

*I'm happy to answer questions. The results surprised us and I'm genuinely uncertain about some of the implications.*
