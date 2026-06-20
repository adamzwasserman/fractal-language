# MÉTRON Foundations: The Generic Levers and How Questions Fit Them

**Author.** Adam Z. Wasserman.
**Date.** 2026-06-19.
**Status.** Foundations note. Defines the conceptual basis of the MÉTRON no/low-code platform.
**Relationship to the programme.** The four levers named here are the ones demonstrated and defined in `NEWLY_TRACTABLE_HUMANITIES_QUESTIONS.md` (Section 3), grounded in the programme's results (`HYPOTHESIS.md`; the cross-linguistic ablation; the BLI structural-alignment result in `BLI_SCALING_FALSIFICATION.md` and `../babylm/`). The question portfolios that fit them are in `PREREGISTRATION_GRAMMAR_ARCHITECTURE.md` (the grammar E-series), `EXPERIMENT_CANDIDATES_AND_TRACTABLE_QUESTIONS.md` (further grammar candidates), and `NEWLY_TRACTABLE_HUMANITIES_QUESTIONS.md` plus `HUMANITIES_CHANNEL_SCOUTING.md` (the humanities H-series).

---

## 1. The idea: a small set of generic levers, fitted to many questions

The hard part of a no-code research platform is choosing the right level of abstraction. "Train a transformer and probe it" is too low: it asks a humanities researcher to be an ML engineer. "Answer my question" is too high: no platform can do that. The right level, and the reason a no-code platform is possible at all, is in between, and the programme has now found it.

The instrument has produced a handful of results that generalize into a small set of **reusable measurement operations**, the levers. The levers are **domain-independent**: they do not care whether the question is about grammar, ethics, mysticism, translation, or conceptual history. A particular research question becomes tractable by being **fitted** to one or more levers, with a falsifier and the discipline arms attached.

So MÉTRON exposes the levers, not the transformers. The researcher supplies the materials and the framing; the platform supplies the measurement. **Few levers, many questions** is what makes the no-code surface small enough to build and general enough to matter.

## 2. The four levers (the platform primitives)

Each lever is defined by: what it measures, the demonstrated result that validates it, what the researcher provides, what it returns, the class of question it answers, and the discipline it requires.

### L1. Controlled language ablation
- **What it measures.** The causal contribution of a single structural feature, by training matched models on corpora (real or synthetic) that differ in exactly that feature and comparing the structure each resolves.
- **Demonstrated by.** The cross-linguistic ablation (identical architecture, matched corpora, only the language varies; French reaches grammatical competence at 197M tokens, English does not through billions) and the Synth-A to Synth-D recipes.
- **Researcher provides.** Two corpora, or one corpus plus a feature to ablate.
- **Returns.** Whether and how the resolved structure shifts relative to the unablated control, against a random-orthogonal noise band.
- **Answers.** "Does feature X causally shape structure Y?" This is the randomized controlled experiment on language structure that cannot be run on human subjects.
- **Discipline.** Claim the representational consequence only, never a human behavioural outcome; that re-imports the confounds the ablation exists to remove.

### L2. Competence-gated Procrustes invariance
- **What it measures.** Whether two meaning spaces share structure, by fitting a single orthogonal map between frozen representations and scoring held-out retrieval, with per-concept residuals.
- **Demonstrated by.** The BLI result: French to a competent model aligns at p@1 of 66.7 percent, French to a failed English model at 25.0 percent, against a 2.1 percent chance floor.
- **Researcher provides.** Two spaces (languages, traditions, periods, registers) and a concept list.
- **Returns.** An alignment score and the residual catalogue: which concepts align (the shared core) and which do not (the bound residue).
- **Answers.** "Is this concept-domain universal or language/tradition/period-bound?" Translatability and universality become measurements rather than intuitions.
- **Discipline.** Competence-gate the measurement; only trust alignment from models that actually resolved the domain. Validate that per-concept residuals separate from the within-space and random baselines before ranking them. Negatively-defined concepts (apophatic primitives) are the hardest case and need the tightest controls.

### L3. Hurst and emergence-efficiency
- **What it measures.** The intrinsic long-range structure of a symbolic corpus, the depth signature that predicts whether and how fast competence emerges.
- **Demonstrated by.** The Hurst-emergence marker (grammar emerges as the Hurst exponent clears roughly 0.65; the corpus-to-model amplification chain).
- **Researcher provides.** A corpus.
- **Returns.** A structural fingerprint (Hurst exponent, emergence profile, which capabilities resolve).
- **Answers.** "Does this corpus carry the deep structure associated with competence, with orality, with literariness, or with a distinct authorial or source hand?"
- **Discipline.** The fingerprint detects a statistical signature, not a mental state or a historical fact. Control for structural richness when comparing corpora, not only for length.

### L4. Competence from structure, not volume (the enabling condition)
- **What it is.** Not a measurement but the **enabling condition** that makes the other three usable on humanities corpora. The headline result is that competence tracks structural richness rather than data quantity (the WALS composite correlated with emergence efficiency at roughly r = -0.88; competence at fewer than 200M tokens in structurally rich input).
- **Why it is a lever.** Many humanities corpora, a dead language, a single author, a single period, a single tradition, are far too small for web-scale methods. L4 removes that barrier: a small corpus stops being fatal, and in the right structural regime becomes a viable subject.
- **The caveat that makes it honest.** The benefit is conditional on structural richness. A structurally sparse small corpus (an isolating classical language, an oral-formulaic text) may not reach competence at small scale, exactly because the result is that sparse input needs more data. So L4 enables small rich corpora cleanly, and small sparse corpora only with explicit competence-gating, never assume it.

## 3. The fitting procedure (how a question becomes a measurement)

A research question is fitted to MÉTRON in five steps.

1. **Recast** the question as a claim about structure latent in a text corpus, the descriptive and structural version, never the normative one (is moral-concept-space universal, not which morality is right).
2. **Gate it on the tractability criterion** (`NEWLY_TRACTABLE_HUMANITIES_QUESTIONS.md`, Section 2): text-observable, manipulable or measurable, differential and falsifiable, and no world-knowledge leak.
3. **Select the lever or levers.** L1 for causal-shaping questions; L2 for universality, translatability, and conceptual drift; L3 for structural-signature, orality, and authorship; L4 as the enabler whenever the corpus is small.
4. **State the falsifier.** Name the null that kills each side, in advance.
5. **Attach the discipline arms.** A placebo or null arm on every candidate, and expert framing with native or specialist validation of stimuli before any model is probed.

## 4. The portfolios are configurations of the levers

The existing question slates are just the levers configured, which is the evidence that the levers are the foundation rather than a post-hoc gloss.

| Question | Lever(s) |
|---|---|
| E1 Learnability / poverty of the stimulus | L1 |
| E2 Universality of binding and islands | L2 |
| E3 One operation or a patchwork (passive) | L2 (alignment as one-operation test) |
| E4 Stored versus computed | L1, L3 |
| E5-E10 (auxiliary selection, dative-experiencer, ergativity, counterfactual, coercion, dative alternation) | L1 read-out, L2 for the alignment cases |
| H1 Linguistic relativity made causal | L1, read out by L2 |
| H2 Untranslatability / Quine indeterminacy | L2 |
| H3 Orality versus literacy, Homeric Question | L3 |
| H4 Comparative philosophy at civilizational scale | L1 and L2, enabled by L4 |
| H5 Conceptual history / legal originalism | L2 diachronic |
| H6 Literariness and the canon | L3 |
| H7 Episteme / paradigms | L1 (once a removable feature is defined) |

Every entry is a selection of levers plus a corpus or concept list plus a falsifier. That is precisely what the no-code interface needs to expose.

## 5. What the platform exposes versus what the researcher supplies

- **Platform (the ML, hidden):** the trained-from-scratch pipeline, the Procrustes alignment with its random-orthogonal and chance floors, the Hurst and emergence computation, the ablation and synthetic-variant rig, competence-gating, and the placebo scaffolding.
- **Researcher (the research design, theirs):** the corpus, concept list, or feature; the framing, validated by domain experts; the falsifier; the choice of lever.

This is the no-code division of labour: the machine learning lives in the levers, the research design lives with the researcher. Neither needs the other's skill set, which is the whole point.

## 6. The discipline, built into the platform

These are not optional add-ons; they are part of what a lever does.

- **A placebo or null arm on every candidate.** The dict-axioms result collapsed under a placebo control and was retracted; any configuration that cannot specify its placebo arm is not ready to run.
- **Competence-gating.** Only measure what the model actually resolved, or low-data noise masquerades as signal (semantic change, untranslatability, a distinct hand).
- **Native and specialist validation of stimuli**, before probing. This is the bottleneck the whole programme exists to lower, and the channel-scouting note (`HUMANITIES_CHANNEL_SCOUTING.md`) is in part a map of where those validators are.
- **The cone-of-light boundary.** Each lever measures structure; it does not settle normative or metaphysical truth, and it does not license a human-cognition claim from a model-representation result.

## 7. A measurement refinement: the frequency-stratified competence profile

Competence-gating asks one question (did the model resolve the domain). A finer readout stratifies the probe set by token frequency into bands and reports per-band accuracy, giving a competence *profile* rather than a single pass-or-fail. This refines E4 (the frequency-on-regulars signature) and competence-gating itself: instead of "resolved at 70 percent," the platform can say which frequency strata the model has acquired, a more honest and more informative signal, and one whose shape differs by language (frequency mass is distributed differently in morphologically rich versus sparse input, so a per-band acquisition curve is itself an L4 readout).

The caution is built in, from a known failure mode. A naive total of the form sum over bands of (accuracy in band times band size) is dominated by the largest, rarest band, so a few lucky responses in a huge low-frequency band can swamp the estimate and inflate it past any defensible ceiling. (A consumer vocabulary-size estimator surveyed in 2026 reported a per-user figure two to four times the native active-vocabulary range it itself cited, for exactly this reason.) Any band-weighted competence number therefore needs a chance-floor correction per band and the standard null arm before it is reported. The profile, not the single weighted total, is the trustworthy object.

## 8. Why this is the foundation

The levers are few and domain-independent; the questions are many and domain-specific. That asymmetry is what makes a no-code platform both buildable and worth building: a small, stable surface (four levers and a fitting procedure) with a large, open reach (any question that passes the criterion). MÉTRON is that surface. The grammar and humanities portfolios are the first proof that arbitrary previously-opaque questions reduce to configurations of these four operations, and the platform is the thing that lets a non-ML researcher do the configuring.
