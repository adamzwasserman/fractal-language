# The Language-Only Hypothesis

Experimental codebase, training logs, and paper sources for a research program testing whether the capabilities of large language models are properties of **natural language structure** rather than of neural networks, parameter count, or training compute.

> The transformer resolves structure that is already in the language, the way a microscope resolves cells it did not create.

## The result, in one paragraph

Train two transformers that are identical in every way (same architecture, same compute, same hyperparameters, same tokenizer, same seed) and change only the training language. A 125M model trained on French reaches grammatical competence, 100% on agreement probes, at about 197 million tokens. The identical model trained on English is still at chance past three billion tokens. Same instrument, same scale, wildly different outcome, which locates the capability in the structure of the language rather than in the network or the compute. The effect holds across two different architecture families (GPT-2 style and LLaMA style), so it is a property of the language, not of one instrument. The central experiment was pre-registered on OSF before any data was collected.

## Go deeper

- **Pre-registration (OSF):** [10.17605/OSF.IO/SJ48B](https://osf.io/sj48b), filed before data collection.
- **Lead paper (Zenodo):** *The Scaling Hypothesis Is Language-Contingent* — [10.5281/zenodo.19423151](https://doi.org/10.5281/zenodo.19423151).
- **Companion deposits:** *English Considered Harmful* ([10.5281/zenodo.19443357](https://doi.org/10.5281/zenodo.19443357)) and *The 70% Rule* ([10.5281/zenodo.19423101](https://doi.org/10.5281/zenodo.19423101)).
- **Child-scale replication (in peer review):** *Right Tool, Right Job* (Wasserman & Beauchemin, BabyLM 2026 / EMNLP).
- **The wider program:** [The Linguistic Telescope](https://linguistictelescope.org).
- **Plain-language explainer:** forthcoming on [Essential Musings](https://emusings.substack.com).

## Follow the work

New results, papers, and essays go out first here:

- X: [@adamzwasserman](https://x.com/adamzwasserman)
- Substack: [Essential Musings](https://emusings.substack.com)
- Site: [adamzacharywasserman.com](https://adamzacharywasserman.com)
- Foundation: [Open Honest Foundation](https://openhonest.org)
- Watch or star this repo to be notified when new experiments land.

## Core claim

All capabilities currently described as "emergent" in LLMs are predictable statistical consequences of deep, multi-scale regularities intrinsic to natural language corpora. The neural network functions as *instrumentation*, a replaceable measuring device, not as the source of the capability. This is a claim about LLMs and language structure; it is distinct from linguistic-determinism or "language of thought" claims about human cognition.

See [`HYPOTHESIS.md`](HYPOTHESIS.md) for a summary; the full, canonical statement is the OSF pre-registration linked above.

## Method

Train identical transformer architectures on different natural languages (and on controlled variants) with all other variables held constant: model size, hyperparameters, tokenizer, seed, data volume. Measure whether emergence thresholds vary by language. If they do, scale cannot be the primary driver of emergence.

Two architecture families are used (GPT-2 style with LayerNorm/GELU/learned positions, and LLaMA style with RMSNorm/SwiGLU/RoPE) to demonstrate that the effect is a property of the language, not of the instrument.

## Repository layout

```
paper/           Paper drafts (EN/FR), compiled PDFs, figures, references
experiments/     Per-experiment code, reports, pre-registrations, and result artifacts
logs/            Training metrics and probe results from the 125M EN vs FR central experiment
HYPOTHESIS.md    Summary of the Language-Only Hypothesis (full statement on OSF)
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

Wasserman, A.Z. (2026). *The Scaling Hypothesis Is Language-Contingent: Evidence from Cross-Linguistic Training Dynamics.* OSF pre-registration 10.17605/OSF.IO/SJ48B; Zenodo [10.5281/zenodo.19423151](https://doi.org/10.5281/zenodo.19423151). Paper in `paper/`.

## Author

Adam Zachary Wasserman ([ORCID](https://orcid.org/0009-0002-8865-6583), [OSF](https://osf.io/user/8t64r)), Founder and Senior Fellow, [Open Honest Foundation](https://openhonest.org). Collaboration inquiries welcome via GitHub issues.
