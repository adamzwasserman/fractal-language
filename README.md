# The Language-Only Hypothesis

Experimental codebase, training logs, and paper sources for a research program testing whether the capabilities of large language models are properties of **natural language structure** rather than of neural networks, parameter count, or training compute.

> "The transformer isn't generating structure — it's resolving structure that's already there. Like a microscope doesn't create cells."

## Core claim

All capabilities currently described as "emergent" in LLMs are predictable statistical consequences of deep, multi-scale regularities intrinsic to natural language corpora. The neural network functions as *instrumentation* — a replaceable measuring device — not as the source of the capability.

See [`HYPOTHESIS.md`](HYPOTHESIS.md) for the full statement and [`BLI_SCALING_FALSIFICATION.md`](BLI_SCALING_FALSIFICATION.md) for the embedding-geometry follow-up program.

**About.** The Language-Only Hypothesis is a pre-registered scientific hypothesis (OSF [SJ48B](https://osf.io/sj48b)) that the emergent capabilities of large language models are properties of natural-language structure rather than of neural-network architecture, parameter count, or training compute. It is distinct from linguistic-determinism or "language of thought" claims about human cognition; this is a claim about LLMs and language structure. By **Adam Zachary Wasserman** ([ORCID](https://orcid.org/0009-0002-8865-6583), [OSF](https://osf.io/user/8t64r)), part of the research program of the [Open Honest Foundation](https://openhonest.org); see also [The Linguistic Telescope](https://linguistictelescope.org).

## Method

Train identical transformer architectures on different natural languages (and on controlled variants) with all other variables held constant — model size, hyperparameters, tokenizer, seed, data volume. Measure whether emergence thresholds vary by language. If they do, scale cannot be the primary driver of emergence.

Two architecture families are used (GPT-2 style with LayerNorm/GELU/learned positions, and LLaMA style with RMSNorm/SwiGLU/RoPE) to demonstrate that the effect is a property of the language, not of the instrument.

## Pre-registration

The central experiment was pre-registered on OSF prior to data collection: [10.17605/OSF.IO/SJ48B](https://osf.io/sj48b).

## Repository layout

```
paper/           Paper drafts (EN/FR), compiled PDFs, figures, references
experiments/     Per-experiment code, reports, pre-registrations, and result artifacts
logs/            Training metrics and probe results from the 125M EN vs FR central experiment
HYPOTHESIS.md    Full statement of the Language-Only Hypothesis
BLI_SCALING_FALSIFICATION.md     Embedding-geometry follow-up program
REASON_TESTS_ARE_NOT_REASON_TESTS.md    Methodological note on "reasoning" benchmarks
LOTTERY_FRACTAL_INSTRUMENT_SYNTHESIS.md Synthesis across lottery-ticket, fractal, and instrumentation framings
CLAUDE.md        Orientation for AI-assisted contributors
```

## Experimental arc

See [`experiments/README.md`](experiments/README.md) for the full arc. Briefly:

- **exp1** — 125M and 350M GPT-2-style models, English vs French, the paper's central result
- **exp2** — Interleaved EN/FR training to test signal-contamination hypotheses
- **exp4_rust** — Programming language as a morphologically minimal control corpus
- **exp8b** — 12-language balanced multilingual scaling and Hurst-exponent analysis
- **exp9** — WALS-feature-based prediction of emergence thresholds

Some experiments include synthetic-language variants whose construction recipes are held proprietary pending patent filing. Results, training curves, and probe outcomes for those variants are published; generation code and corpora are not.

## Reproduction

Heavy data (checkpoints, tokenizer files, chunked corpora) lives outside the repo. The per-experiment `REPRODUCIBILITY.md` files inside `experiments/` specify data provenance (HuggingFace dataset names, seeds, hyperparameters).

Training code is PyTorch and expects CUDA (originally run on 4090s via vast.ai). Requirements live in each experiment's `requirements.txt`.

## Citation

Wasserman, A.Z. (2026). *The Scaling Hypothesis Is Language-Contingent: Evidence from Cross-Linguistic Training Dynamics*. OSF pre-registration: 10.17605/OSF.IO/SJ48B. Paper forthcoming — see [`paper/`](paper/).

## Contact

Adam Zachary Wasserman — Independent Researcher.
Collaboration inquiries welcome via GitHub issues.
