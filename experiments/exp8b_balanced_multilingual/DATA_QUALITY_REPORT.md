# Data Quality Report: C4 Indonesian Corpus Issues

**Date:** 2026-01-05
**Experiment:** exp8b multilingual training
**Dataset:** allenai/c4 (Indonesian/id subset)

  https://huggingface.co/datasets/allenai/c4/discussions/new

## Summary

The C4 Indonesian (id) corpus contains primarily low-quality web spam and SEO content, rendering it unsuitable for language model pretraining. We replaced it with CC-100 Indonesian data.

## Evidence of Data Quality Issues

### 1. Anomalous Training Metrics

| Language | PPL at 25k steps | Grammar Accuracy | Token Diversity |
|----------|------------------|------------------|-----------------|
| EN       | 231.55          | (pending)        | ~25% unique     |
| FR       | 150.67          | 78.3%            | ~22% unique     |
| ES       | 126.24          | (pending)        | ~24% unique     |
| **ID**   | **41.74**       | **35%**          | **1.4% unique** |
| FI       | 205.59          | 68%              | ~20% unique     |
| RU       | 137.87          | 70%              | ~18% unique     |

Indonesian shows:
- **Suspiciously low perplexity** (41.74) - indicating highly repetitive/predictable content
- **Below-chance grammar accuracy** (35% vs 40% chance baseline) - model learning anti-patterns
- **Extreme token concentration** - only 1.4% unique tokens (13,859 out of 1M)

### 2. Content Analysis

Sample text from C4 Indonesian reveals SEO spam patterns:

```
thapki full serial | Cinta Sinopsis
Jual Obat Peninggi Badan Grow Up Super USA 081311069191 | Iklan Top Gratis
Download Lagu Terbaru Mp3 Gratis - Gudang Lagu
Jual Beli Mobil Bekas Murah - OLX Indonesia
```

The corpus consists primarily of:
- Product listings and advertisements
- Clickbait article titles
- SEO keyword stuffing
- Phone numbers and promotional content
- Low-quality user-generated content

### 3. Linguistic Structure Analysis

The content lacks:
- Complex sentence structures
- Proper grammatical agreement patterns
- Natural discourse flow
- Semantic coherence beyond keyword repetition

This explains why the model achieves low perplexity (easy to predict repetitive patterns) but fails grammar probes (no actual grammatical structure to learn).

## Resolution

### Replacement Corpus: CC-100 Indonesian

We replaced C4 Indonesian with the CC-100 Indonesian corpus from statmt.org:
- **Source:** https://data.statmt.org/cc-100/id.txt.xz
- **Size:** 36GB compressed
- **Quality:** Properly filtered web content with coherent text

Sample text from CC-100 Indonesian shows natural prose:
```
Kaisar Cheng Tung duduk seorang diri di atas permadani,
tenang dan diam, sedikitpun tidak tampak gugup atau ketakutan.
Di sekelilingnya tampak tubuh para pengawalnya bergelimpangan
bermandikan darah mereka sendiri.
```

This is coherent narrative Indonesian with proper grammar and complex sentence structures.

## Recommendations

### 1. For Our Experiment
- Continue using CC-100 Indonesian as replacement
- Monitor training metrics to confirm improvement
- Validate results against expected cross-linguistic patterns

### 2. For the Research Community

**Report to AllenAI (C4 maintainers):**

The C4 Indonesian subset appears to have insufficient filtering for web spam content. We recommend:

1. **Quality audit** of C4 non-English subsets, particularly lower-resource languages
2. **Additional filtering** using:
   - Token diversity thresholds (reject documents with <5% unique tokens)
   - Language model perplexity thresholds
   - Spam classifier filtering
3. **User warning** in documentation about potential quality issues in certain language subsets

**Contact:** AllenAI maintains C4 at https://huggingface.co/datasets/allenai/c4

Issues can be reported at: https://github.com/allenai/allennlp/issues or https://huggingface.co/datasets/allenai/c4/discussions

## Files

- `DATA_QUALITY_REPORT.md` - This report
- `logs/id_125M.csv` - Training logs showing anomalous metrics
- `download_cc100_id.sh` - Script for downloading replacement corpus

## References

- CC-100 Corpus: Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (2020)
- C4 Corpus: Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2020)
