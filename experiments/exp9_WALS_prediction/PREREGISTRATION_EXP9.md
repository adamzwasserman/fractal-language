# Pre-Registration: Exp9 — Can Engineered Morphology Outperform Natural Languages?

## Registration Date
2026-01-05

## Background

Exp8 (N=11 languages, 125M parameters) found that WALS morphological features predict LLM training efficiency:
- WALS_VerbSynth vs tokens_to_60%: r=-0.88, p<0.001
- WALS_Agreement vs val_ppl: r=-0.78, p<0.01

French (WALS_Predictive=9) reached 60% grammar in 6.1M tokens — 12× faster than Chinese (WALS_Predictive=2.5).

## Research Question

Can a synthetic language engineered for optimal WALS properties outperform natural high-WALS languages like French?

## Primary Hypothesis

**H1**: Synthetic languages with WALS_Predictive ≥ 10 will reach grammar competence faster and achieve lower perplexity than French (WALS_Predictive=9) when trained as 350M parameter models.

### Predictive Features (from Exp8)

| Feature | WALS Code | Optimal Value |
|---------|-----------|---------------|
| Verb Synthesis | 22A | 4+ categories/verb |
| Person/Number Marking | 29A | 2 (full paradigm) |
| TAM Exponence | 21B | 2 (polyexponential) |
| Fusion | 20A | 2 (concatenative) |

### Composite Score

```
WALS_Predictive = 22A + 29A + 21B + 20A
```

## Predictions

### P1: Synthetic Beats Natural
synth_α_a (WALS_Predictive=10) will reach 70% grammar accuracy faster than French (WALS_Predictive=9).

### P2: Emergence Speed
synth_α_a will reach 70% grammar accuracy in ≤40M tokens at 350M scale.

### P3: Final Perplexity
synth_α_a will achieve validation perplexity ≤30 by 500M tokens.

### P4: English Baseline
All synth_α variants will outperform English on both grammar emergence and perplexity.

### P5: Encoding Quality Matters (Ablation)
Among the synth_α variants:
- **synth_α_a** (highest morphological consistency) will emerge fastest
- **synth_α_c** (lowest consistency) will emerge slowest
- The ranking will be: synth_α_a > synth_α_b > synth_α_c

This tests whether *implementation fidelity* of morphological marking affects training, not just the presence of markers.

### P6: Consistency-Emergence Correlation
Across all synth_α variants, morphological consistency (measured as % of tokens with complete feature marking) will correlate negatively with tokens-to-emergence (r < -0.8).

## Test Languages

### Natural Languages (Baselines)

| Language | 22A | 29A | 21B | 20A | Predictive | Role |
|----------|-----|-----|-----|-----|------------|------|
| English | 2 | 1 | 1 | 2 | 6 | Standard baseline |
| French | 4 | 1 | 2 | 2 | 9 | High-WALS benchmark |

### Synthetic Languages (Experimental Conditions)

Three variants test whether encoding fidelity affects training:

| Variant | Encoding Method | Morphological Consistency | Expected |
|---------|-----------------|---------------------------|----------|
| synth_α_a | Regex + Lookup | High (verbs regularized) | Fastest |
| synth_α_b | Pattern Match | Medium (some errors) | Medium |
| synth_α_c | Dict + Fallback | Variable (sparse coverage) | Slowest |

WALS scores for synthetic variants:

| Variant | 22A | 29A | 21B | 20A | Predictive | Notes |
|---------|-----|-----|-----|-----|------------|-------|
| synth_α_a | 4 | 2 | 2 | 2 | **10** | Full verb morphology |
| synth_α_b | 3 | 1 | 2 | 2 | 8 | Some verb forms missed |
| synth_α_c | 2 | 1 | 2 | 2 | 7 | Coarse categories only |

**Note**: Synthetic language construction details are withheld due to their proprietary nature.

## Method

### Model
- Architecture: GPT-2 style transformer
- Size: 350M parameters (24 layers, d_model=1024, 16 heads)
- Training: Adam, lr=3e-4, cosine schedule

### Data
- Balanced corpora: up to 2B tokens per language
- Tokenizer: Re-use joint BPE tokenizer from Exp8 (50k vocabulary)
- Validation: 10M tokens held out per language

### Stopping Rule
Training stops at 2B tokens per language or until one language shows a sustained advantage over the others.

### Evaluation
- Grammar probes: Re-use probes from Exp8, every 1k steps
- Validation perplexity: Every 1k steps
- Primary metric: Tokens to 70% grammar accuracy

## Analysis Plan

1. **Primary comparison**: synth_α_a vs French emergence speed (one-tailed test)
2. **Ablation analysis**: Rank ordering of synth_α_a > synth_α_b > synth_α_c
3. **Baseline verification**: All synth_α variants outperform English
4. **Correlation**: Morphological consistency vs tokens-to-emergence across synth_α variants

## Success Criteria

The hypothesis is **supported** if:
- **P1**: synth_α_a reaches 70% grammar before French (primary outcome)
- **P4**: All synth_α variants outperform English
- **P5**: Rank ordering synth_α_a > synth_α_b > synth_α_c holds

The hypothesis is **partially supported** if:
- synth_α_a matches but does not beat French
- Rank ordering holds but magnitudes are small

The hypothesis is **refuted** if:
- French outperforms synth_α_a
- English outperforms any synth_α variant

## Proprietary Methods

Synthetic language construction methods are proprietary and patent-pending. Details will be disclosed only after the relevant patent is made public.

---

*Pre-registered by: Adam Zachary Wasserman*
*Date: 2026-01-05*
*Based on: Exp8 findings (r=-0.88 for VerbSynth, r=-0.78 for Agreement)*
