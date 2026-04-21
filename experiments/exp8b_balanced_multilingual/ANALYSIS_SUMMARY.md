# Exp8b Comprehensive Correlation Analysis

**N = 11 languages** (en, fr, es, fi, ru, vi, zh, synth_a, synth_b, synth_c, synth_d)

## Significant Correlations (|r| >= 0.5, p < 0.1)

| Predictor | Outcome | r | p |
|-----------|---------|---|---|
| WALS_VerbSynth | tokens_to_60 | -0.880 | 0.0004*** |
| WALS_Agreement | val_ppl_final | -0.784 | 0.0043** |
| WALS_Agreement | tokens_to_60 | -0.782 | 0.0044** |
| WALS_Fusion | tokens_to_70 | -0.759 | 0.0068** |
| WALS_VerbSynth | val_ppl_final | -0.740 | 0.0093** |
| WALS_Fusion | grammar_final | +0.739 | 0.0094** |
| WALS_Fusion | tokens_to_60 | -0.664 | 0.0259* |
| WALS_Agreement | ppl_final | -0.656 | 0.0283* |
| WALS_VerbSynth | grammar_final | +0.611 | 0.0458* |
| WALS_VerbSynth | ppl_final | -0.594 | 0.0542. |
| WALS_Total | tokens_to_60 | -0.581 | 0.0611. |
| H_token_freq | learning_rate | -0.579 | 0.0622. |
| WALS_TAM | val_ppl_final | -0.538 | 0.0879. |

## Key Findings

### Morphology → LLM Success (Strong Evidence)

1. **Verb Synthesis (22A)** is the strongest predictor of emergence speed (r=-0.88, p<0.001)
   - Languages with more verb categories per word reach grammar threshold faster

2. **Agreement Composite** (VerbSynth + PersonNum + TAM) predicts both:
   - Emergence speed (r=-0.78, p=0.004)
   - Final perplexity (r=-0.78, p=0.004)

3. **Fusion (20A)** predicts final grammar quality (r=+0.74, p=0.009)
   - Concatenative languages achieve higher grammar accuracy

### Null Results (Important)

**Hurst exponent does NOT mediate morphology → LLM success:**
- H_word_freq vs tokens_to_60: r=-0.07, p=0.83
- H_word_freq vs grammar_final: r=-0.33, p=0.32
- H_entropy_early vs tokens_to_60: r=+0.35, p=0.30

**Non-predictive WALS features:**
- Cases (49A): No significant correlations
- Gender (30A): No significant correlations
- Person/Number (29A): Only significant as part of composite

## WALS Feature Definitions

| Code | Feature | Scale |
|------|---------|-------|
| 20A | Fusion | 1=isolating, 2=concatenative |
| 22A | Verb Synthesis | 0-4+ categories per verb |
| 29A | Person/Number | 0=none, 1=syncretic, 2=full |
| 30A | Gender | Number of genders |
| 49A | Cases | Number of cases |
| 21B | TAM | 1=monoexponential, 2=+agreement |

## Conclusion

**Morphological structure directly predicts LLM success.** The strongest predictors are:
1. Verb synthesis complexity
2. Agreement marking (composite)
3. Word-level fusion

Fractal structure (Hurst exponent) does NOT mediate these relationships. The one exception is a weak correlation between token-level H and learning rate (r=-0.58, p=0.06), suggesting that higher token-level fractal dimension may slow learning.
