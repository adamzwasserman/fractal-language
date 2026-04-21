# The Language-Only Hypothesis: Results Summary

## Overview

This project tests whether morphological structure of natural language predicts LLM training efficiency. We conduct controlled experiments training identical transformers on different languages, holding architecture and data quantity constant.

**Core finding**: Languages with richer morphological agreement train more efficiently. WALS Verb Synthesis predicts grammar emergence speed with r=-0.88 (p<0.001).

---

## Experiment Progression

| Exp | Scale | Languages | Purpose | Status |
|-----|-------|-----------|---------|--------|
| Exp1 | 125M | EN, FR | Initial discovery | Complete |
| Exp8 | 125M | 11 languages | Exploratory: identify predictive features | Complete |
| Exp9 | 350M | 10+ languages | Confirmatory: test pre-registered predictions | Planned |

---

## Exp1: Initial Discovery (EN vs FR)

### Key Finding

French 125M achieves 100% grammar accuracy at step 12,000 (~197M tokens) while English 125M remains at chance (40%) through step 201,000 (~3.3B tokens).

This is a **>15x difference** in emergence threshold.

### Results

| Model | Steps | Tokens | PPL | Grammar |
|-------|-------|--------|-----|---------|
| French 125M | 12k | 197M | ~100 | **100%** |
| French 125M | 181k | 3B | 27 | **100%** |
| English 125M | 181k | 3B | 1,340 | 40% |
| English 125M | 400k | 4.3B | 777 | 40% |

---

## Exp8: Multilingual Exploration (11 Languages)

### Purpose

Exp8 is an **exploratory experiment** designed to identify which morphological features predict LLM training efficiency. Findings inform pre-registered predictions for Exp9.

### Languages Tested

| Code | Language | Type | WALS Agreement | WALS VerbSynth |
|------|----------|------|----------------|----------------|
| en | English | Natural | 4 | 2 |
| fr | French | Natural | 7 | 4 |
| es | Spanish | Natural | 7 | 4 |
| fi | Finnish | Natural | 5 | 2 |
| ru | Russian | Natural | 7 | 4 |
| vi | Vietnamese | Natural | 1 | 0 |
| zh | Chinese | Natural | 1 | 0 |
| synth_a-d | Synthetic variants | Constructed | 1-5 | 1-3 |

**Note**: Synthetic language construction details are withheld due to their proprietary nature. WALS values provided for reproducibility.

### Morphological Characterization

We characterize languages using WALS (World Atlas of Language Structures) features:

| Feature | Code | Scale | Rationale |
|---------|------|-------|-----------|
| Verb Synthesis | 22A | 0-5 categories/verb | More verb categories → more redundant signal |
| Person/Number | 29A | 0-2 | Agreement paradigm completeness |
| TAM Exponence | 21B | 1-2 | Tense-aspect-mood marking density |
| Fusion | 20A | 1-2 | Morpheme boundary transparency |

**Composite Score**: WALS_Predictive = 22A + 29A + 21B + 20A

### Key Results

#### Performance by Language

| Language | Grammar | Val PPL | Tokens to 60% |
|----------|---------|---------|---------------|
| fr | **87%** | **37.7** | **6.1M** |
| es | 78% | 42.8 | 16.4M |
| ru | 80% | 43.3 | 20.5M |
| en | 87% | 74.4 | 22.5M |
| fi | 72% | 52.9 | 22.5M |
| vi | 55% | 50.4 | 26.6M |
| zh | 55% | 97.1 | 75.8M |

**French reaches 60% grammar in 6.1M tokens — 12× faster than Chinese (75.8M).**

#### Morphology → LLM Success Correlations

| Predictor | Outcome | r | p |
|-----------|---------|---|---|
| WALS_VerbSynth | tokens_to_60% | **-0.880** | 0.0004*** |
| WALS_Agreement | val_ppl | **-0.784** | 0.0043** |
| WALS_Agreement | tokens_to_60% | **-0.782** | 0.0044** |
| WALS_Fusion | grammar_final | **+0.739** | 0.0094** |
| WALS_VerbSynth | val_ppl | **-0.740** | 0.0093** |

#### Null Result: Fractal Structure

Hurst exponent (H) does NOT predict any success metric:

| Predictor | Outcome | r | p |
|-----------|---------|---|---|
| H_word_freq | tokens_to_60% | -0.07 | 0.83 |
| H_token_freq | grammar_final | -0.51 | 0.11 |

The morphology → success relationship is **direct**, not mediated by statistical long-range dependence.

#### Reasoning: Unreliable at 125M Scale

All languages perform near chance (~20%) with high variance. Reasoning probes require larger models to assess.

### Predictive Features Identified

Four WALS features consistently predict LLM success:

1. **Verb Synthesis (22A)**: Strongest predictor of emergence speed
2. **Agreement composite**: Predicts both grammar and perplexity
3. **Fusion (20A)**: Predicts final grammar accuracy
4. **TAM Exponence (21B)**: Predicts perplexity

---

## Tokenizer Effects

### Exp1 Tokenizer (Valid)

Exp1 used a **custom 50k BPE tokenizer** trained jointly on English and French C4 data:
- Inflected forms stored as single tokens
- Grammar probes compare P("sits") vs P("sit") directly
- Preserves morphological boundaries

### XLM-R Tokenizer (Biased)

Experiments with XLM-R's pretrained tokenizer revealed critical issues:

```
XLM-R: "sits" → ['▁sit', 's']  (2 tokens)
       "sit"  → ['▁sit']       (1 token)
```

When probes only measured P(first_token), they tested nothing about verb agreement.

**Fix**: Compute joint probability of all target tokens.

**Implication**: Tokenizer choice affects both training efficiency AND evaluation validity. Fair cross-linguistic comparison requires language-appropriate tokenization.

---

## Next Steps: Exp9 (Pre-Registered Predictions)

Based on exp8 findings, we pre-register the following predictions for 350M scale:

### Primary Hypothesis

Languages with WALS_Predictive ≥ 7 will reach 70% grammar accuracy in ≤50M tokens.

### Specific Predictions

1. **P1 (Emergence)**: High-WALS languages reach 70% grammar in ≤50M tokens
2. **P2 (Perplexity)**: High-WALS languages achieve val PPL ≤30 by 500M tokens
3. **P3 (Ranking)**: Rank correlation with exp8 emergence order: r>0.7
4. **P4 (Scale)**: Morphological advantage preserved or amplified at 350M

### Success Criteria

- At least 75% of high-WALS languages meet P1-P2 thresholds
- Correlation between WALS_Predictive and emergence: |r|>0.6, p<0.05

---

## Files Included

### Exp1 Data
- `en_125M_training.csv` - English training metrics (201k steps)
- `fr_125M_training.csv` - French training metrics (201k steps)
- `grammar_probes_en.csv` - English grammar probe accuracy
- `grammar_probes_fr.csv` - French grammar probe accuracy
- `joint_tokenizer.json.gz` - Shared 50k BPE tokenizer
- `checkpoints/` - Model weights at key steps

### Exp8 Data
- `logs/{lang}_125M.csv` - Training logs for all 11 languages
- `correlation_summary.json` - All significant correlations
- `wals_features.json` - WALS morphological features
- `h_trajectories.json` - Hurst exponent over training
- `FINAL_REPORT.md` - Complete experimental report

### Documentation
- `Language_Only_Hypothesis_Paper.pdf` - Full paper
- `PREREGISTRATION_EXP9.md` - Exp9 pre-registration

---

## Pre-registration

- Exp1 DOI: https://osf.io/sj48b
- Project: https://osf.io/2pg8s

## Training Logs

Real-time training logs: https://github.com/adamzwasserman/fractal-language

## License

CC0 1.0 Universal
