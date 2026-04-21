# Placeholder: "No Test of Reason Tests Reason"

## Hypothesis

Every benchmark that claims to test *reasoning* in a language model is
demonstrably testing something else — lexical token bias, surface pattern
matching, template sensitivity, training-distribution overlap, morphological
cue density, or shared fractal structure across the prompt and the target.

If this holds, "reasoning" as an LLM property is an attribution error. The
apparent competence is already present in the structure of the input; the
network is resolving it, not performing it.

## Why this belongs here, not in babylm/

The BabyLM 2026 paper is a child-scale efficiency result framed around the
right-tool-right-job principle. It touches the reasoning-is-structure
question only obliquely — via the dict-axioms placebo correction (§6.2),
which showed that random axioms produce the same "gain" as targeted ones
on supposedly reasoning-heavy GLUE tasks. That is one receipt, not a
program.

The full program is a fractal-language project because the thesis is a
fractal-language thesis: what passes for reasoning in LLMs is the same
structural commonality across languages that was demonstrated in
Wasserman (2026a) and the 70% Rule paper. The experimental structure —
controlled ablations, placebo controls, cross-linguistic replications —
is the methodology this repo already runs on.

## Candidate experiments

1. **Placebo replication on a reasoning-labelled benchmark suite.** Run the
   dict-axioms placebo protocol across BIG-Bench reasoning tasks and LogiQA
   at 125M and 1B scales. Predict: the targeted-minus-placebo delta is near
   zero everywhere, matching the GLUE result already in babylm/.

2. **Structural isomorphism probe.** For each "reasoning" benchmark, construct
   a structural twin that preserves surface form and removes the entailment
   content. Predict: models score equivalently on the twin, revealing the
   structural cue as the load-bearing feature.

3. **Cross-lingual transfer gradient as a discriminator.** Tasks whose
   structure is language-independent (entailment, agreement) should transfer
   across languages; tasks that require genuine world-state tracking
   (MultiRC, EWoK) should not. The gradient documented in the BabyLM paper
   (§5.2) is the seed for the full cross-linguistic probe.

4. **Tokenizer-swap diagnostic.** The v3d result in the BabyLM paper showed
   that changing the tokenizer alone reproduced a 7.7pp "reasoning"
   regression. Generalize: if tokenizer choice moves a benchmark by more
   than task-data retraining does, the benchmark is not measuring what it
   claims.

## Relation to the program spine: Robot in the Dark

This paper is the most direct empirical application of Robot in the
Dark's central argument. The book diagnoses a general epistemological
error: scientists confuse the instrument for the phenomenon it measures.
This paper diagnoses the specific instance of that error in ML:
researchers confuse what the benchmark measures (structural cues,
lexical frequency, tokenizer distribution) for the phenomenon they
believe it measures ("reasoning").

In Robot in the Dark's lamp-and-cone metaphor: "reasoning benchmarks"
are built inside the lamp's cone of illumination. They test what the
lamp can see — structural regularities deposited in natural language —
and then attribute the illumination to the object rather than to the
lamp. This paper's job is to demonstrate that attribution error
empirically, benchmark by benchmark, via the same controlled-ablation
methodology (placebo controls, tokenizer swaps, structural twins) that
the BabyLM paper used on three individual cases.

If Robot in the Dark is published before this paper, cite it as the
framework that generates the specific prediction: "Every test of
'reasoning' is a test of structural resolution by the instrument, and
the structural-resolution interpretation is empirically distinguishable
from the reasoning interpretation via placebo controls."

## Framing constraints

- Never attribute reasoning, inference, understanding, or comprehension to
  the model. The thesis is that these attributions are the error being
  corrected — the specific error Robot in the Dark diagnoses at the general
  level. Describe mechanisms structurally.
- Maintain the Occam prior from the BabyLM paper: 200 million humans
  embedded cognition into languages over 100,000 years; the network reads
  it back.
- Use placebo controls on every positive result before interpretation.
- The paper's audience includes ML benchmarking researchers who will not
  have read Robot in the Dark. The epistemological argument must be made
  self-contained within this paper even if the framework is citable.

## Status

Placeholder only — 2026-04-14; Robot in the Dark connection added
2026-04-17. Full protocol, pre-registration, and experiment scripts to
be added when this project's slot opens after BabyLM 2026 submission.
