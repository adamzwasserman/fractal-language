# The Scaling Hypothesis Is Language-Contingent

Training logs and data for the paper: *"The Scaling Hypothesis Is Language-Contingent: Evidence from Cross-Linguistic Training Dynamics"*

## Overview

This repository contains training logs from a controlled ablation study comparing identical 125M-parameter transformers trained on English vs French C4 corpora.

## Key Finding

French achieves grammatical competence (100% on agreement probes) at 197M tokens while English remains at chance level (40%) after 3B tokens. Perplexity shows a 50x ratio at matched training steps, suggesting 50-100x greater training efficiency for morphologically rich languages.

## Pre-registration

This experiment was pre-registered on OSF prior to data collection: [10.17605/OSF.IO/SJ48B](https://osf.io/sj48b)

## Contents

- `logs/` - Training metrics, grammar probe results, and analysis

## Citation

Paper forthcoming. Pre-registration: Wasserman, A.Z. (2024). The Language-Only Hypothesis. OSF: 10.17605/OSF.IO/SJ48B.

## Contact

Adam Zachary Wasserman - Independent Researcher
