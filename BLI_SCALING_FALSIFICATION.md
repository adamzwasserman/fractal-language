# Placeholder: Scaling Laws Are Language-Contingent at the Embedding Level

Follow-up paper to Wasserman (2026a), extending the scaling-hypothesis
falsification from the task level to the embedding-geometry level via
Bilingual Lexicon Induction across the 12-language exp8b corpus.

## The claim

Scaling laws are language-contingent not only in training dynamics and
task performance (already established in Wasserman 2026a) but also in
the geometry of the trained embedding matrix itself. The degree to
which two monolingual 125M models trained at matched compute share
embedding structure, measured by orthogonal Procrustes alignment on a
seed dictionary, is not predicted by compute budget or architecture
alone — it is predicted by whether each training language affords
enough morphological signal density for the model to acquire grammatical
competence in the available token budget.

## Evidence already in hand

The BabyLM 2026 paper (Wasserman, 2026c — in preparation) contains the
first instance of the measurement:

- FR BabyLM model ↔ GPT-2 (both competent, different languages): p@1 = 66.7%
- FR BabyLM model ↔ fractal-language 125M EN (matched arch, failed grammar): p@1 = 25.0%
- Alignment strength tracks acquired grammatical competence, not scale or architecture.

The BabyLM paper plants this as a single-language-pair result and
explicitly reserves the multi-language replication for this paper.

## Experimental design (draft)

### Corpus

All 12 monolingual exp8b checkpoints on `/Volumes/fractal/exp8b_balanced_multilingual/`.
Frozen. No retraining.

### Seed dictionaries

Bilingual lexicons for each language pair. Candidates:
- MUSE ground-truth dictionaries (where available for the 12 languages)
- PanLex for the less-resourced languages
- Parallel Wikipedia anchor links as a fallback

Seeds must cover all language pairs uniformly enough that per-pair sample
size does not confound the alignment matrix.

### Measurement

For each ordered language pair $(L_i, L_j)$:
1. Extract embedding matrix $E_i$ from $L_i$ checkpoint at matched training step
2. Look up each seed pair $(w_i, w_j)$ via the respective tokenizers
3. Build paired matrices $X \in \mathbb{R}^{n \times 768}$, $Y \in \mathbb{R}^{n \times 768}$
4. Hold out 20% of pairs
5. Solve $W = \arg\min \|XW - Y\|_F$ subject to $W^T W = I$ (orthogonal Procrustes, closed form via SVD)
6. Report: Procrustes fit, p@1 / p@5 / p@10 on held-out pairs, random-$W$ baseline

### Primary result

12×12 pairwise alignment matrix $A$ where $A_{ij}$ = p@1 of held-out word
translation from $L_i$ to $L_j$.

### The killer figure

Scatter plot: $x$-axis = WALS morphological-complexity index for each
language, $y$-axis = mean alignment strength of that language against
the other 11. Scaling-hypothesis prediction: flat line (alignment
depends on compute, which is matched). Language-Only Hypothesis
prediction: positive slope (alignment strength tracks the language's
own morphological signal density, because morphologically rich
languages reach competence in the token budget and morphologically
poor ones do not).

## Contribution structure

1. **Scaling-hypothesis falsification at the embedding level** — the headline.
2. **BLI alignment as an instrument for scaling-laws research** — a
   measurement tool the scaling community can apply to any pair of
   matched-architecture models.
3. **Typological calibration** — alignment-strength-vs-WALS as a new
   empirical signature for typologists, complementing the training-
   dynamics signatures from Wasserman 2026a.

## Relation to the program spine: Robot in the Dark

The scaling hypothesis is the conviction that if we make the lamp
brighter, we will eventually illuminate everything. Robot in the Dark
(Wasserman, in preparation) is the book-length philosophical argument
that says no: brighter lamps show more of what the lamp was designed
to show, but they do not reveal what the lamp cannot illuminate in
principle.

This paper is the specific empirical instance of that argument applied
to embedding geometry. The "lamp" is the transformer architecture. The
"cone of illumination" is determined by the training language's
morphological signal density. Scaling compute (making the lamp brighter)
does not extend the cone — the fractal-EN model at 6.5B tokens has a
WEAKER embedding geometry than the French model at 278M tokens because
English does not deposit the structure the lamp is designed to resolve.
BLI alignment is the measurement that makes this visible in frozen
parameters rather than in downstream task performance.

If Robot in the Dark is published (even via Zenodo) before this paper
is submitted, cite it as the framework: "We test a specific prediction
of the lamp-and-cone epistemology (Wasserman, 2026d) in the domain of
embedding-level scaling laws." If not yet published, the paper can
still stand on Wasserman 2026a's empirical basis alone, but the
epistemological spine is weaker without the citable framework reference.

## Positioning

- Builds directly on Wasserman 2026a; cite it as the task-level
  predecessor and this paper as the embedding-level extension.
- If Robot in the Dark is published: cite it as the epistemological
  framework the empirical result operationalizes.
- Target venues: ACL, COLING, Computational Linguistics (TACL), or
  ML scaling-laws workshops at NeurIPS / ICLR.
- Authorship: Wasserman & Beauchemin (David expressed interest in
  the 12-language study as of 2026-04-17).

## Framing constraints

- **Never use reasoning-language about the models.** The measurement is
  geometric; the interpretation is structural. See the no-reasoning-
  language feedback rule applied across the program.
- **"Probably, with a tractable path to stronger" is the calibration.**
  Do not overclaim indisputable falsification; do claim the result
  is the second independent measurement channel (after Wasserman 2026a)
  that fails the language-independence prediction and that a scaling-
  hypothesis defender must answer.
- **The paper is written for the scaling-laws community, not the cog
  sci community** — unlike BabyLM, which was written for the charter.
  Audience shift dictates framing.

## Cost estimate

- No training cost. Models exist.
- Seed-dictionary acquisition: a few dollars of API calls for MUSE
  curation and PanLex lookups.
- All BLI math runs on CPU. Whole experimental section is an afternoon
  of compute.
- Writing and figure design: standard paper timeline.

## Status

Placeholder — 2026-04-14. Planted after the BabyLM 2026 BLI finding
demonstrated the single-language-pair result is strong enough to
justify the multi-language follow-up. Writing begins after BabyLM
submission deadline.
