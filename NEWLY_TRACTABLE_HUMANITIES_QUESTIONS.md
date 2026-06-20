# Newly Tractable Humanities Questions: A Scouting Note

**Author.** Adam Z. Wasserman.

**Date.** 2026-06-19.

**Status.** Scouting note, not a pre-registration. Nothing here is locked. Each candidate is recorded so it can later be promoted into the falsifier-bearing pre-registration format used in `PREREGISTRATION_GRAMMAR_ARCHITECTURE.md`, or discarded.

**Relationship to the programme.** This note generalizes the parent programme (`HYPOTHESIS.md`, OSF SJ48B block) past grammar theory. The parent treats the LLM as an instrument of observation that resolves structure already present in language (`INTELLIGENCE_AS_INTENTIONALITY_AND_AGENCY.md`), demonstrated by the controlled cross-linguistic ablation and the BLI structural-alignment result (`BLI_SCALING_FALSIFICATION.md`, `../babylm/`). The grammar sub-study (`PREREGISTRATION_GRAMMAR_ARCHITECTURE.md`, experiments E1 to E4) turns the instrument on the nativist-versus-constructionist axis. The geometry sub-study (`../wasserman-levin-2026/`, the polytope, sphere, and vector-invariance experiments) turns it on the Plato-versus-Wittgenstein axis. This note asks the prior question the programme implicitly raises: which previously opaque humanities questions, across disciplines and not only in grammar theory, become empirical by virtue of these instruments.

---

## 1. The question this note answers

The motivating question is not "what grammar experiment is next" but "what class of humanities questions, once thought permanently interpretive, is now measurable because of the instrument and its demonstrated results." The answer has two parts: a criterion that decides tractability, and a portfolio of concrete candidates that pass it.

## 2. The generalized tractability criterion

A humanities question is tractable on this instrument when it can be recast as a claim about **structure latent in a text corpus** that satisfies all four of the following.

1. **Text-observable.** The phenomenon leaves a signature in the symbol stream, not only in human behaviour, sensory experience, or social fact.
2. **Manipulable or measurable.** Either the structure can be manipulated by controlled or synthetic training (an ablation), or its geometry can be measured directly in the model's representations.
3. **Differential and falsifiable.** The recast question yields a contrast, across languages, periods, registers, traditions, or ablation arms, with a numeric prediction that can fail.
4. **No world-knowledge leak.** The question does not smuggle in human behaviour, social facts, or felicity conditions that a stateless model cannot access.

Criterion 4 is the recurring trap. It is the reason the negative-concord pragmatics candidate and the Labov hiring reinterpretation were ruled out of the grammar slate, and it is the reason the dict-axioms result had to be retracted once a placebo control was added (see Section 6). Every candidate below is selected to survive a null or placebo arm.

## 3. The four enabling levers

Each candidate rides one or more of these, and these specifically are the programme's own results, not generic NLP capabilities.

- **L1, controlled language ablation.** Train identical models on languages, or on synthetic variants (the Synth-A to Synth-D recipes), that differ in exactly one structural feature. This is the randomized controlled experiment on language structure that cannot be run on human subjects.
- **L2, competence-gated Procrustes invariance.** A number for whether two meaning spaces align under a single orthogonal map. The demonstrated instance: French to a competent model aligns at p@1 of 66.7 percent, French to a failed English model at 25.0 percent, against a 2.1 percent chance floor (`BLI_SCALING_FALSIFICATION.md`). Translatability and universality become measurements rather than intuitions.
- **L3, Hurst and emergence-efficiency.** A quantitative gauge of the intrinsic long-range structure of any symbolic corpus, with the working marker that grammar emerges when the Hurst exponent clears roughly 0.65.
- **L4, competence from structure, not volume.** The headline result, namely that competence tracks structural richness rather than data quantity (French grammar at 197M tokens, WALS composite correlated with emergence efficiency at roughly r = -0.88). This is the quiet liberator of the humanities specifically. Many humanities corpora, a dead language, a single author, a single period, a single tradition, are far too small for web-scale methods. L4 removes that barrier: a small corpus stops being fatal, and in some cases (Section 4, H4) becomes an advantage.

## 4. The portfolio

Each candidate is recorded as: the classic question, why it was opaque, the enabling lever, a design with an explicit falsifier, the trap to avoid, and de-confliction against existing slate items and prior art. Numbering is H1 to H7 to keep this family distinct from the grammar E-series.

### H1. Linguistic relativity, made causal (Sapir and Whorf)

**Classic question.** Does the grammar of a language shape the conceptual structure of its speakers.

**Why opaque.** Human subjects cannot be randomized into languages. Every correlational study (grammatical gender and object construal, futureless languages and saving behaviour) is confounded by culture, history, and biology.

**Enabling lever.** L1, with L2 as the read-out.

**Design.** Ablate exactly one feature (grammatical gender, evidentiality, future marking, aspect, counterfactual marking) in an otherwise identical corpus, using the Synth recipes, and measure whether the concept-space geometry the model resolves shifts relative to the unablated control. Culture, embodiment, and behaviour are held out by construction, isolating the structural contribution of the feature.

**Falsifier.** Ablating the feature leaves the relevant concept geometry within the random-orthogonal noise band of the unablated control.

**Trap.** Claim the representational consequence only. Never claim a human behavioural or cognitive outcome, which would re-import the confounds the design exists to remove.

**De-confliction.** This is the clean causal form of the most famous "untestable" claim in the twentieth-century humanities, and the ablation rig is the specific thing that makes it clean. It is adjacent to the geometry sub-study but distinct: the geometry study asks whether concepts are universal, this asks whether changing one grammatical feature deforms a concept.

### H2. Untranslatability and the indeterminacy of translation (Cassin, Quine, Benjamin)

**Classic question.** Are some concepts genuinely untranslatable, and is reference indeterminate across languages.

**Why opaque.** "Untranslatable" was connoisseurship; Quine's gavagai indeterminacy was a thought experiment with no measurement attached.

**Enabling lever.** L2.

**Design.** Compute competence-gated Procrustes p@1 per concept across language spaces. Rank concepts by alignment residual. The low-alignment tail is the empirically identified language-bound residue, that is, the Wittgensteinian remainder; the high-alignment head is the universal core.

**Falsifier.** Concept-level alignment residuals do not separate beyond the within-language and random baselines, that is, there is no stable structure to "untranslatability."

**Trap.** Do not treat low alignment as proof of metaphysical incommensurability; report it as a measured residual under a stated model and corpus.

**De-confliction.** This rides the vector-invariance machinery of `../wasserman-levin-2026/`. The distinct deliverable is the catalogue, the actual list of which concepts are the residue, rather than the mean cosine. Prior art: the Platonic Representation Hypothesis (Huh and colleagues, 2024), which the geometry sub-study already extends cross-linguistically.

### H3. Orality versus literacy and the Homeric Question (Parry and Lord, Ong)

**Classic question.** Do oral-formulaic and literate texts differ in deep structure, and was a disputed text (Homer, Ossian, an oral epic) orally composed.

**Why opaque.** Ong's psychodynamics of orality asserted a structural difference it could not quantify; the single-author-versus-tradition impasse is a literary-historical dead end.

**Enabling lever.** L3.

**Design.** Estimate the Hurst and emergence-profile signature of attested oral-tradition corpora versus literate corpora, establish the differential, then score disputed texts against it.

**Falsifier.** Oral-tradition and literate corpora do not separate on the structural marker.

**Trap.** The instrument detects the long-range statistical signature that oral-formulaic composition predicts, not "orality" as a mental state.

**De-confliction.** The Hurst marker is exactly a structural-depth measure, which makes this a natural and underexplored application of an instrument the programme already uses.

### H4. Comparative philosophy at civilizational scale (Graham, Hansen)

**Classic question.** Does the grammar of a classical language (for example classical Chinese, with its mass-noun syntax and absent copula) shape a different metaphysics from another (for example classical Greek).

**Why opaque.** The debate is interpretive, conducted by sinologists and classicists reading texts, with no measurement of the conceptual structures in question.

**Enabling lever.** L1 and L2, decisively enabled by L4.

**Design.** Train per-tradition models on classical Chinese and classical Greek philosophical corpora and compare resolved structure and concept geometry; vector invariance separates universal philosophical primitives from tradition-bound ones.

**Falsifier.** The two traditions' concept spaces align at the universal-core level with no tradition-specific residue beyond baseline.

**Trap.** Corpus provenance and editorial layering must be controlled; a measured difference must not be an artifact of canon formation.

**De-confliction.** This is the case where L4 turns the apparent weakness of small models into the reason the method works at all: these canons are tiny, and the programme's result that competence arrives at fewer than 200M tokens in structurally rich input is precisely what makes a dead, low-resource philosophical corpus a viable experimental subject.

### H5. Conceptual history and legal originalism (Koselleck)

**Classic question.** Did a concept (liberty, commerce, virtue, cruel, revolution) actually change meaning across periods, or do later readers project.

**Why opaque.** Intellectual historians argue interpretively about semantic change; originalist legal interpretation turns on an unmeasured claim about a term's original meaning.

**Enabling lever.** L2, diachronic.

**Design.** Train period-specific models and measure geometric drift of target concepts; Procrustes-align adjacent periods to localize ruptures, including Koselleck's Sattelzeit. The legal application asks directly whether a constitutional term's coordinate moved between ratification and now.

**Falsifier.** Target concepts show no drift beyond the within-period and random baselines across the periods in which historians claim change.

**Trap.** Competence-gate the measurement: only report drift for concepts the period model actually resolved, otherwise low-data noise masquerades as semantic change.

**De-confliction.** Prior art exists (diachronic word embeddings, Hamilton and colleagues). The differentiators are competence-gating and the high-stakes legal-originalism application, which is underexplored with this method.

### H6. Literariness and the canon (Russian Formalism)

**Classic question.** What distinguishes literary or canonical texts from the rest.

**Why opaque.** The judgment was impressionistic and entangled with aesthetic value.

**Enabling lever.** L3, with capability-resolution probes.

**Design.** Compute the structural fingerprint (Hurst, emergence efficiency, which capabilities resolve) of canonical works against forgotten contemporaries, with period held constant.

**Falsifier.** Canonical and forgotten contemporaries do not separate on the structural fingerprint.

**Trap.** Stated loudly: structural complexity is not aesthetic value. The claim under test is the Formalists' structural claim, offered as a proxy, not a verdict on quality. Honest framing is what keeps this from being crankery.

### H7. Foucault's episteme and Kuhn's paradigms, operationalized

**Classic question.** Does each era have an underlying order of the sayable that structures its discourse.

**Why opaque.** The claim is famously close to unfalsifiable.

**Enabling lever.** L1 on period or discipline-specific corpora.

**Design.** Operationalize an episteme as the resolvable latent structure of a period-corpus; ablating period-features should deform the concept space measurably.

**Falsifier.** Ablating period-features leaves the concept space within baseline, that is, there is no resolvable era-level structure to recover.

**Trap.** This is the highest overclaim risk in the portfolio. Present it as operationalization, not vindication, and run it only with the placebo discipline of Section 6.

### H8. Cultural transmission through language alone (Vygotsky, Tomasello, the orality debate)

**Classic question.** Is culture transmitted nonverbally, through embodied imitation, observation, and apprenticeship, or does language carry it? Mainstream documents the correlation (children absorb culture by observing) and treats the nonverbal account as primary, but has never isolated language's contribution.

**Why opaque.** You cannot ablate language from a developing human to see whether culture still transmits. Every nonverbal-transmission study leaves language operating in the background (ambient narration, naming, half-heard explanation, the child's own inner speech), so the outcome is attributed to observation while language's role goes unmeasured.

**Enabling lever.** L1 (controlled ablation), driven by the ablation asymmetry. The experiment impossible on humans (ablate language) is replaced by the one that is possible (ablate non-verbal observation). The LLM is that ablation taken to its limit: a system trained on text alone, with observation and embodiment removed by construction. A second, finer ablation removes specific cultural references from the training text.

**Design.** Two layers. First, train a model on a culture's textual record (observation and embodiment absent by construction) and test whether it produces culturally-appropriate responses to situations from that culture. Second, ablate specific cultural references (named practices, figures, markers) from the training text and test whether the model still responds appropriately to situations involving the ablated elements, which it can do only by inferring the underlying values and behavioural logic from the remaining linguistic structure. Cultural-insider validation of what counts as an appropriate response is fixed before any probing.

**Falsifier.** If the model cannot produce appropriate responses to the ablated elements above a placebo arm and a culture-naive baseline, language does not carry culture at the structural level and the nonverbal-primacy account survives. If it can, culture lives in the relational grammar, not only in surface references.

**Trap.** "Culturally appropriate" must be operationalized by cultural specialists in advance, against a placebo arm and a culture-naive null, or the result is unfalsifiable storytelling. Claim the structural result (culture is recoverable from the linguistic record), never a claim about how human children actually learn. Cone-of-light: the model shows what language can carry, not what humans do.

**De-confliction.** This is the cultural extension of the Language-Only Hypothesis: where the grammar work shows grammatical structure is in the language, H8 asks whether cultural structure is too. Distinct from H4 (concept geometry across traditions) and H5 (diachronic concept drift); H8 asks whether the appropriate-response logic of a single culture is recoverable from its text with the explicit markers removed.

## 5. Honorable mentions

- **Music as language** (Lerdahl and Jackendoff). Apply L3 and L2 to symbolic music corpora to test for language-like hierarchical structure and a cross-system "universal grammar of tonality." Extends the programme beyond text; borderline humanities.
- **Creole equal-complexity dogma** (McWhorter). L1 emergence-efficiency on creoles versus lexifiers gives a structural answer to a politically frozen question. Consolidation rather than new ground, given the Haitian Creole Oracle work already in `../babylm/`. Frame structurally.
- **Performatives and the agency argument** (Austin). Primarily an upgrade to the agency argument in `INTELLIGENCE_AS_INTENTIONALITY_AND_AGENCY.md`: performatives are the limit case where the saying constitutes the act, the strongest single instance of content-driven causation. Secondarily a scoped experiment on the grammar of illocutionary force (the hereby test, the first-person-present-simple restriction, cross-linguistic force-marking). Keep felicity conditions out, by Criterion 4.
- **Perennialism versus constructivism in mysticism** (Stace and Huxley versus Katz). Recorded here because it is the cleanest transfer of the Plato-versus-Wittgenstein axis to the domain most often called unfalsifiable: do apophatic concepts across traditions (sunyata, ein sof, the One, fana) align under L2, or are they language-constructed. See the channel scouting note for where this question is posed in public.

## 6. Methodological discipline

Two non-negotiable arms carry over from the parent programme.

1. **A placebo or null arm on every candidate.** The dict-axioms result collapsed under a placebo control and was retracted; it must not be cited as positive evidence. Any candidate above that cannot specify what its placebo arm looks like is not ready for promotion.
2. **Expert framing, native validation.** As in `PREREGISTRATION_GRAMMAR_ARCHITECTURE.md`, the question is framed by domain experts and the stimuli validated by speakers or specialists of the relevant language, period, or tradition, before any model is probed. This is the load-bearing version of the principle and the bottleneck the instrument programme exists to lower.

## 7. Ranking and provenance

**Ranking by distinctiveness to the instrument and tractability.** H1 and H2 are the standouts, because they are the two most celebrated "permanently untestable" questions in the humanities and the toolkit is the specific thing that cracks them: H1 needs the ablation, H2 needs the invariance metric. H4 is the sleeper, because it is where L4 converts the small-model regime from a limitation into the enabling condition. H5 has the clearest external stakes (legal originalism). H7 is the speculative frontier and should move last.

**Provenance.** This note was produced by reading the programme corpus directly (`fractal-language`, `../babylm`, `../linguistic-telescope`, `../wasserman-levin-2026`) and grounding the typological and disciplinary claims against the secondary literature (Aissen on differential object marking, Bonet and successors on the Person-Case Constraint, Aikhenvald on evidentiality, Corbett on agreement, Huh and colleagues on the Platonic Representation Hypothesis, Hamilton and colleagues on diachronic embeddings). It is a map of targets, not a result.
