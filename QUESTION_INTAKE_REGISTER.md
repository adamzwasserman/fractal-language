# Question Intake Register

**Author.** Adam Z. Wasserman.
**Date opened.** 2026-06-20.
**Status.** Living register. It captures under-explored questions as they arise, triages each, and routes it. The funnel upstream of the pipeline (`NEWLY_TRACTABLE_HUMANITIES_QUESTIONS.md`, `EXPERIMENT_CANDIDATES_AND_TRACTABLE_QUESTIONS.md`) and of the deductive material (`../linguistic-telescope/books/`, `INTELLIGENCE_AS_INTENTIONALITY_AND_AGENCY.md`).

## Why this register exists

The programme keeps generating questions that mainstream fields appear to treat as settled. The precise claim, the one that survives a specialist's scrutiny, is narrow: mainstream documents the *correlation* (for example, inner speech tracks executive function; observation tracks cultural transmission) and treats the *necessity and mechanism* as axiomatic, because the controlled experiment that would isolate cause from medium was infeasible on human subjects. The instrument changes that. The loose version ("no one is looking") is false and a reviewer will reject it; the precise version is the programme's thesis and ties directly to the discount bias and to textbook capture.

These questions are an asset only if they are routed rather than left in conversation. This register is the routing.

## The triage rule

Each question sorts into one of three:

1. **Tractable on the instrument.** Promote to a pipeline candidate, gated on the tractability criterion (text-observable; manipulable or measurable; differential and falsifiable; no world-knowledge leak) plus a placebo or null arm and the cone-of-light limit (claim the model-representation result, never the human-cognition result).
2. **Deductive, not empirical.** Route to the book and theory material; argued from premises, not tested.
3. **Empirical but not on this instrument.** Log honestly as tractable in principle, not by us, so the programme does not overclaim its reach.

## The ablation asymmetry (the engine, stated so it is not lost again)

The instrument's power is that it runs ablations impossible on humans. We never ablate language; we cannot, in a developing human. We ablate the other channels, observation and embodiment, and the LLM is the maximal case of that ablation by construction. When a question turns on language's contribution, the design is almost always: replace the impossible ablation (remove language) with the possible one (remove the non-linguistic channel), and read the result.

## Register

| # | Question | Mainstream status | Triage | Disposition |
|---|---|---|---|---|
| 1 | Is cultural transmission nonverbal, or does language carry it? | Correlation documented (observation, apprenticeship); nonverbal treated as primary; language's contribution never isolated | Tractable | Pipeline candidate **H8** in `NEWLY_TRACTABLE_HUMANITIES_QUESTIONS.md`, via the observation-ablation asymmetry plus a within-training reference ablation |
| 2 | Why is language necessary for executive function? | Correlation documented (inner speech and executive function); necessity treated as axiomatic | Deductive (demonstration) | Robot/Replicators material (`../linguistic-telescope/books/replicators_editorial_brief.md`, "Planning lives in the narrative loop"): planning is the narrative loop, shown in the one system where the loop can be removed and the capacity watched to vanish. Safe structural form added to `POSITIONING_FRONTIER_ABLATION_GAP.md` |
| 3 | The meta-pattern: fields document a correlation and treat the mechanism as axiomatic because the controlled test was infeasible | This is the diagnosis itself, not a single question | Book + grant significance | Discount bias and textbook capture (Replicators brief); grant significance framing |

## How to use

When a new question arises: add a row; tag the mainstream status precisely (name the documented correlation and why the mechanism is treated as axiomatic); assign a triage; route it. Promote tractable items into the pipeline docs with a falsifier and a placebo arm before they count as candidates. Keep the cone-of-light limit on every model result, and never let a model-representation finding be written as a claim about human cognition.
