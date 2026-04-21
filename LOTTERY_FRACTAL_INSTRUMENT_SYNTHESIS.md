# Placeholder: Lottery Tickets Resolve Fractal Commonality

A synthesis paper unifying three independent lines of theory under one
mechanistic account of how transformer language models work.

**Note (2026-04-17):** This paper is NOT the program's capstone. Robot
in the Dark (Wasserman, in preparation) is the capstone — it makes the
general epistemological argument (scientists confuse the instrument for
the phenomenon; all instruments are lamps with bounded cones of
illumination) at book length. This paper is the ML-specific technical
formalization of that general argument, connecting LTH, fractal
commonality, and the instrument view into a single mechanistic account
that Robot in the Dark frames philosophically. If Robot in the Dark is
published first, this paper cites it as the epistemological framework
and focuses on the mechanistic details. If Robot in the Dark is not yet
published, this paper must re-derive enough of the epistemological
argument to stand alone, which costs several pages.

## The synthesis

Three positions from three different research traditions converge on the
same account of what a transformer is doing when it "learns language":

1. **Natural language has fractal structural commonality across languages**
   (Google's multilingual transfer work; fractal-structure-of-language
   research). Languages share self-similar organizational principles at
   multiple scales; the structural overlap is large enough that models
   trained on one language generalize to others through the shared fractal.

2. **A small subnetwork suffices to resolve low-effective-complexity
   functions** (Lottery Ticket Hypothesis, Frankle & Carbin 2018–2019).
   Inside any overparameterized dense network there exists a small
   "winning ticket" subnetwork that can match the full network's
   performance when trained from the same initialization. The rest of the
   network is scaffolding.

3. **The transformer is instrumentation, not a creative system**
   (Language-Only Hypothesis, Wasserman 2026a). The capabilities
   attributed to LLMs are properties of natural language structure, not
   of neural networks or scale. The model resolves structure already
   present in the training data; it does not generate that structure.

The three positions are usually discussed in different communities and
usually in isolation. Their conjunction is tighter than any pair alone
and produces a single account:

> Natural language has fractal structural commonality across languages.
> Fractal structure has low effective complexity per unit of scale.
> A small subnetwork therefore suffices to resolve it (LTH). That
> subnetwork resolves the structure it was trained on rather than
> creating new structure (instrument view). Two subnetworks trained
> on fractal-commonable languages end up embedding the same
> underlying structure, differing only by a rotation in representation
> space (BLI alignment, Wasserman 2026c). Compute budgets beyond what
> is needed to resolve the structure are waste; training a larger
> model on the same fractal yields diminishing returns because the
> signal it could resolve has already been resolved.

Every link is defensible and every link has independent empirical
support in the research program. The synthesis paper's job is to state
the chain, show how each piece of empirical evidence slots in, and
make the predictions the chain generates that no individual position
makes on its own.

## Why this paper matters

Each of the three positions has a built-in audience that does not
consistently talk to the other two:

- **Scaling-laws researchers** know LTH but usually treat language
  structure as an unspecified "target function."
- **Multilingual transfer researchers** know fractal commonality but
  usually treat it as a property of models rather than of languages.
- **Philosophers of language and typologists** have had no empirical
  handle on the instrument question at all; the debate has been
  conceptual.

The synthesis gives each community a frame for reading the other two's
work. It also makes the Language-Only Hypothesis testable in a form
the scaling-laws community can engage with: if the "target function" is
the fractal structure of the training language, scaling laws should be
language-contingent (already shown in Wasserman 2026a) and the small-
substrate mechanism should be detectable in embedding geometry
(already shown in Wasserman 2026c BLI triangulation).

## Cross-domain corroboration: construction over remediation

A fourth audience emerges once the three-way core is stated: researchers
who approach AI properties through **construction-time guarantees rather
than post-hoc empirical verification**. This stance is being developed
independently in at least three adjacent programs:

- **Parameter-efficient adaptation literature** (Hu et al. 2021, LoRA and
  successors). LoRA's guarantee that the base model cannot be overwritten
  is not an empirical claim subject to drift; it is a consequence of how
  the adaptation subspace is defined. Preservation is a mathematical
  invariant, not a test that sometimes passes. Cited in Wasserman 2026c
  §5.3 as a construction-time guarantee against catastrophic forgetting.
- **Design-time verification of trained models** (Haynes 2026). Numerical
  stability, type consistency, and computational correctness can be
  guaranteed at construction time via the type system and architectural
  constraints, rather than measured after training. This is the explicit
  programmatic statement of construction-over-remediation inside ML.
- **Honest Code** (Wasserman, software-engineering methodology series).
  Eliminate categories of software defects by architectural choice
  rather than by post-hoc remediation or mitigation. The same
  intellectual stance, developed in the software-engineering domain,
  with a distinct audience (developers, architects) and distinct
  empirical anchors (type systems, pure functions, SQL over caches,
  DOM as state).

The synthesis paper should name this parallel explicitly. It does two
things for the three-way core:

1. **It grounds the "scaffolding" tension.** Classical LTH leaves open
   whether the non-winning weights are scaffolding-as-overshoot or
   scaffolding-as-search-space. The construction-over-remediation reading
   favored by LoRA and Haynes 2026 resolves this: when the subnetwork is
   defined by construction (as in LoRA), there is no search space; the
   "scaffolding" is simply parameters that exist because the architecture
   does not yet have a way to exclude them. The instrument view inherits
   this reading: the winning subnetwork is not a discovered lucky
   initialization; it is the minimal substrate that the architecture
   cannot avoid converging on when the target fractal is resolvable.

2. **It establishes that the three-way core is not a ML-specific
   curiosity.** The Honest Code framework makes the same methodological
   move in software engineering, which means "prefer guarantees to
   measurements" is something multiple research traditions are converging
   on independently. This is weak evidence for the stance's generality but
   strong evidence against the charge that the three-way synthesis is
   motivated reasoning specific to one research program.

The cross-domain corroboration is a scope expansion, not a claim
expansion. The three-way core (LTH + fractal + instrument) is the
mechanistic account. The construction-over-remediation parallel is an
external validator of the stance the account rests on. Treat it as a
section rather than as a fourth synthesis leg.

## Tensions to address honestly

The synthesis does more work than any individual component, and the
paper should name the places a careful critic could push:

- **LTH is content-agnostic.** Classical LTH makes no claim about what
  the winning subnetwork resolves. Combining LTH with the instrument
  view adds a commitment (the content is in the data). That commitment
  is defensible from the controlled comparisons in the program but is
  not entailed by LTH itself. The paper must argue for the combination,
  not assume it.

- **Fractal commonality is statistical; cognition is semantic.** Google's
  multilingual work establishes that languages share structure the
  model can exploit. The instrument view makes the bigger claim: the
  shared structure is *cognitive* structure. The jump from statistical
  regularity to cognitive commonality is real and needs argument. The
  strongest single piece of evidence is BLI triangulation (alignment
  tracks competence, not scale). The paper must show that no weaker
  reading of the evidence would suffice.

- **"Scaffolding" is compatible with either creative or resolving
  networks.** LTH says the non-winning weights are scaffolding, but it
  does not say what the winning ticket does. The instrument reading of
  scaffolding (overparameterization that overshoots the minimal
  resolving substrate) is consistent with LTH but not uniquely
  implied. An alternative reading (scaffolding is search space during
  optimization) is also compatible with LTH. The paper should
  distinguish these readings and explain why the instrument reading is
  favored by the cross-linguistic evidence. The construction-over-
  remediation corroboration (Hu et al. 2021 LoRA, Haynes 2026) helps
  here: when the substrate is defined by construction rather than
  discovered by search (LoRA), the search-space reading of scaffolding
  no longer applies, and the instrument reading is the one that fits.
  LTH plus LoRA plus instrument is a triangulated account; LTH alone
  is underdetermined.

## Predictions the conjunction generates

The value of the synthesis is in the predictions no single position
makes on its own. Three candidates:

1. **Winning-ticket sparsity should correlate with the source language's
   fractal compressibility.** If a language has denser self-similar
   structure, a smaller subnetwork should suffice to resolve its
   grammatical core. Languages with lower morphological redundancy
   should require larger effective subnetworks. Testable by LTH
   experiments at matched architecture across typologically diverse
   languages.

2. **BLI alignment between models trained on matched-fractal pairs should
   be achievable with lower-rank transformations than BLI between
   matched-compute-but-fractal-divergent pairs.** The rank of the
   required alignment is a measurement of how much the two languages
   share fractally. Testable as a pairwise rank matrix across the
   exp8b 12-language corpus.

3. **Compute-scaling of capability should plateau at the point the
   fractal has been resolved and not continue thereafter.** Not a new
   prediction (Wasserman 2026a shows it for grammar), but the synthesis
   predicts the plateau is explainable: once the fractal is resolved,
   there is nothing left to scale. The corollary is that apparent
   capability gains at scales above the plateau point are capability
   claims about something other than the target fractal — they are
   either noise, different-target-fractal gains, or artifacts of
   evaluation protocol (as documented in Wasserman 2026c §6.4 for
   tokenizer-swap, §6.2 for dict-axioms placebo, §5.4 for
   Supplement lexical-frequency bias).

## Paper structure (draft)

- §1 The epistemological frame. If Robot in the Dark is citable: one
  paragraph citing it and summarizing the lamp-and-cone argument. If
  not: 1-2 pages re-deriving enough of the argument to motivate why
  instrument-vs-phenomenon is the load-bearing distinction.
- §2 The three ML-specific positions, stated in each community's terms
- §3 The conjunction and why it is tighter than any pair
- §4 The empirical stack already in place (Wasserman 2026a, 2026c; LTH
  literature; Google's multilingual work; BLI triangulation in Wasserman
  2026c §5.1)
- §5 The tensions named above and what resolves them
- §6 Cross-domain corroboration via construction-over-remediation (Hu et
  al. 2021 LoRA, Haynes 2026 design-time verification, Honest Code
  framework in software engineering). Scope-expanding, not claim-expanding.
- §7 Three predictions the conjunction generates that no single
  position makes on its own
- §8 Implications for compute budgets, benchmark design, and philosophy
  of language
- §9 Limitations and what the synthesis does not claim

## Framing constraints

- **No reasoning-language about the model.** The synthesis is
  mechanistic (resolving, aligning, embedding) and structural (fractal,
  lottery-ticket, linear rotation). Any sentence that attributes
  reasoning, understanding, or inference to the network undermines the
  thesis — and is the specific instance of the error Robot in the Dark
  diagnoses at the general level (confusing the instrument for the
  phenomenon). See the no-reasoning-language feedback rule applied
  across the program.
- **"Probably, with tractable path to stronger" is the calibration.**
  This is NOT the strongest claim in the program — Robot in the Dark
  makes the general philosophical claim. This paper makes the specific
  ML-technical formalization. Its strength level is "the evidence is
  convergent across three independent lines of research and consistent
  with the synthesis at a strength that makes it worth testing further."
- **Audience is all three ML communities simultaneously** (scaling-laws,
  multilingual-transfer, philosophy-of-language). Robot in the Dark
  addresses a fourth audience (general philosophy of science) that
  this paper does not need to serve. This paper should introduce each
  ML position in the language its community uses and translate between
  them, not assume fluency in any.
- **Synthesis paper, not experimental.** The paper's job is to connect
  existing evidence, not to run new experiments. Any new experiment
  that is needed to land a link should be its own paper — including
  the three predictions in §6 of the draft structure, which should be
  named as the three follow-up experimental papers rather than run in
  this one.
- **If Robot in the Dark is citable:** §1 can be short. The
  epistemological argument is "see Wasserman (2026d)" and the paper
  jumps directly to the three-way mechanistic synthesis. This saves
  2-3 pages. **If not citable:** the paper must self-contain enough of
  the lamp-and-cone argument to motivate why the three positions
  converge. This is doable but more expensive in page count.

## Status

Placeholder — 2026-04-14; cross-domain-corroboration section added
2026-04-15; Robot in the Dark reframing added 2026-04-17.

**Key strategic change (2026-04-17):** This paper was previously
positioned as the capstone of the program. It is not. Robot in the
Dark is the capstone — it makes the general epistemological argument
at book length. This paper is the ML-specific technical formalization
that hangs off that spine. The relationship is: Robot in the Dark
establishes the lamp-and-cone epistemology; this paper instantiates it
as a mechanistic account (LTH + fractal + instrument) and generates
specific predictions; the empirical papers (Wasserman 2026a, 2026c,
BLI_SCALING_FALSIFICATION, REASON_TESTS) test those predictions.

The synthesis is already operative in the BabyLM paper (§5.1 BLI,
§5.3 LoRA preservation, §7.4 structural-over-empirical pattern, §7.5
Platonic–Wittgensteinian bullet) but has never been stated in one
move. This paper's job is to state it in one move, to the ML audiences
that need to see it stated.

**Dependencies:**
- Robot in the Dark published (even via Zenodo) → §1 is short, paper
  is tighter, the epistemological frame is citable rather than
  re-derived. This is a strong dependency if timing permits.
- BLI multi-language follow-up (BLI_SCALING_FALSIFICATION.md) run →
  empirical anchor for predictions §7.
- David Beauchemin confirmed interest in the 12-language study
  (2026-04-17) → potential co-author on BLI paper, which feeds into
  this synthesis.
- An Honest Code synthesis paper on the software-engineering side
  would make a natural companion piece; the two synthesis papers could
  appear in parallel, each citing the other as cross-domain
  corroboration of the same methodological stance, without either
  carrying the load of both domains alone.
