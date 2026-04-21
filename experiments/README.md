# Experiments

This directory contains the experimental arc of the Language-Only Hypothesis research program. Each subdirectory is a self-contained experiment: training code, evaluation probes, logs, reports, and (where applicable) pre-registrations.

## Arc

The experiments progress from a direct English-vs-French comparison to progressively stronger tests of whether **morphological richness** — not model scale — drives training efficiency and emergence thresholds.

### exp1 — English vs French (central experiment)

**`exp1_125M_en_vs_fr/`** and **`exp1_350M_en_vs_fr/`**

Two identical GPT-2-style transformers trained on English and French C4 corpora with all variables held constant: same hyperparameters, same joint BPE tokenizer, same seed, same data volume. The 125M run is the paper's headline result.

**Finding:** French reaches 100% on grammatical-agreement probes at 197M tokens while English remains at chance level (40%) after 3B tokens. Perplexity ratio 50× at matched steps.

See `REPRODUCIBILITY.md`, `VALIDATION_SUMMARY.md`, and `logs/SCALING_ANALYSIS.md` inside the 125M directory.

### exp2 — Interleaved EN/FR training

**`exp2_interleaved/`**

Trains a single model on alternating English and French batches to test whether shared capacity and contaminating signal change the per-language emergence trajectory. Controls for the concern that exp1's effect is about model-language pairing rather than language structure itself.

### exp4_rust — Programming language control

**`exp4_rust/`**

Trains on Rust source code (HuggingFace `ammarnasr/the-stack-rust-clean`) as a morphologically minimal but syntactically rigid corpus. Two conditions: Rust-only and Rust+English mixed. Tests whether the "signal density" argument generalizes beyond natural languages.

*Note: "Rust" here refers to the programming language, not a synthetic linguistic construct.*

### exp8b — Multilingual balanced scaling

**`exp8b_balanced_multilingual/`**

12-language balanced training corpus. Measures per-language emergence thresholds, Hurst exponents of training curves and trained-model internals, and their correlation with WALS typological features. Establishes that the EN/FR result generalizes along a continuous morphological-richness dimension.

See `FINAL_REPORT.md`, `ANALYSIS_SUMMARY.md`, `DATA_QUALITY_REPORT.md`, and `OSF_WIKI_DRAFT.md`.

### exp9 — WALS-predicted emergence

**`exp9_WALS_prediction/`**

Pre-registered test of whether a quantitative combination of WALS morphological features predicts emergence-speed rankings across natural and synthetic-language variants. Pre-registration (`PREREGISTRATION_EXP9.md`) was filed before data collection.

## Synthetic languages

Several experiments include synthetic-language conditions designed to isolate specific morphological properties. The **construction recipes** for these variants are held proprietary pending patent filing and are not published. What is published for each synthetic condition:

- The hypothesis being tested
- The method class (e.g., rule-based morphological augmentation of English)
- The WALS-feature target for each variant
- Training curves, probe outcomes, and perplexity logs
- The pre-registered predictions and their outcomes

What is not published: generator code, lexicons, transformation rules, tokenizer files for synthetic variants, and the synthetic corpora themselves.

## Heavy data

Checkpoints, tokenized chunk files, raw corpora, and large validation sets live outside this repository on external storage. Each experiment's `REPRODUCIBILITY.md` (where present) specifies dataset provenance and hyperparameters sufficient to regenerate the artifacts from public sources.

## Contributing

If you want to replicate or extend an experiment, start by reading the `REPRODUCIBILITY.md` or equivalent report inside that experiment's directory. Open an issue to discuss before beginning substantial work — some experiments have unpublished follow-on analyses in flight.
