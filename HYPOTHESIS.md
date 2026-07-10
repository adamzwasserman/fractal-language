# The Language-Only Hypothesis

**Full paper**: `paper/The Language-Only Hypothesis.pdf`

## Formal Statement

> *All capabilities currently described as "emergent" in large language models are predictable statistical consequences of deep, multi-scale regularities intrinsic to natural language corpora, and are not caused by neural-network architecture, parameter count, or training compute.*

## Core Research Question

Why do neural networks trained with gradient descent and attention mechanisms produce "unreasonably effective" results on language, but underwhelming results on other domains (vision, proteins, games)?

The same scaling recipe — decoder-only Transformers trained with next-token prediction — has been applied to vision, audio, proteins, and robotics with dramatically different outcomes. In every non-linguistic domain, scaling alone has failed to produce the sharp, broad, unpredictable capability jumps routinely observed in language models.

**Is the effectiveness due to the technology, or to language itself?**

## The Instrumentation Thesis

The neural network and optimizer serve as **instrumentation** — replaceable measuring devices that reveal structure already present in the data.

> "The transformer isn't generating structure - it's resolving structure that's already there. Like a microscope doesn't create cells."

Change the instrument and the revealed capabilities change little. Change the language and the revealed capabilities change dramatically.

This reframes scaling laws: they may measure how much compute is needed to *extract* structure already present in language, not how much compute is needed to *create* reasoning capability.

## Theoretical Foundation

Why does revealing the structure of language amount to revealing *intelligence*? Because intelligence, functionally, **is** intentionality (a world-model, in Brentano's sense of aboutness) coupled to agency (goal-directed action) — and language possesses both. On this account the instrument *inherits* the coupling rather than creating it. The formal scaffold — active inference's single free-energy objective, which splits into an epistemic term that improves the model and a pragmatic term that reaches goals; and Legg–Hutter's scalar definition of intelligence as goal-achievement across environments — is developed in [`INTELLIGENCE_AS_INTENTIONALITY_AND_AGENCY.md`](INTELLIGENCE_AS_INTENTIONALITY_AND_AGENCY.md).

That note yields a testable mechanism for this programme: morphological richness should predict faster free-energy / evidence reduction, and thus lower emergence thresholds. It is pre-registered as a companion mediation sub-study in [`PREREGISTRATION_FREE_ENERGY_MEDIATION.md`](PREREGISTRATION_FREE_ENERGY_MEDIATION.md).

## Experimental Design

Train transformer architectures on English and French C4 corpora with all variables held constant:
- Same model sizes (125M, 350M)
- Same hyperparameters
- Same random seed (42)
- Same joint tokenizer (50k BPE)
- Same data volume

### Two Architecture Variants (Instrument Replaceability Test)

To test whether the "instrument" is truly replaceable, we train both:

| | GPT-2 Style | LLaMA Style |
|---|-------------|-------------|
| **Normalization** | LayerNorm | RMSNorm |
| **Activation** | GELU | SwiGLU |
| **Position encoding** | Learned | Rotary (RoPE) |

If the French advantage holds across *both* architectures, this demonstrates:
1. Language effect is robust, not an artifact of architecture
2. The instrument is replaceable — different "microscopes" reveal the same structure
3. Pre-empts "your results are architecture-specific" critique

**Falsification target**: If emergence thresholds for various capabilities vary by language (across both architectures), this falsifies the claim that scale or training techniques alone explain emergent effectiveness.

## Pre-registered Predictions

### 1. Token-efficiency threshold prediction
The 125M French model will reach any fixed capability threshold (≥70% on held-out benchmarks) using **≥60% of the raw tokens** required by the identical 125M English model — a net advantage that cannot be explained by tokenizer effects alone.

### 2. Parameter-efficiency crossover prediction
At convergence, the 125M French model will outperform the 350M English model on at least three of four capability clusters:
- Syntactic recursion depth ≥7 on centre-embedded constructions
- 8-shot in-context learning on symbolic reasoning tasks
- Stable persona coherence across 20-turn dialogues
- Susceptibility to indirect suggestion (embedded-command acceptance rate)

### 3. Predictability-from-structure prediction
The observed token-efficiency ratio (French/English) for each capability will correlate strongly (r > 0.8) with independently measured linguistic fractal statistics (Hurst exponent, mean maximum embedding depth, clitic-to-content ratio, liaison density).

**A single confirmed prediction with statistical significance falsifies the claim that scale is the primary driver of emergent capabilities.**

## Rationale

French's explicit morphological marking will **lower emergence thresholds**.

French agreement markers provide redundant signals - error-correcting code built into the grammar:
- "les grandes maisons blanches" marks plurality and gender four times
- "the big white houses" marks plurality once

This redundancy gives the model multiple consistent gradients pointing at the same underlying abstraction. French provides stronger signal for the same semantic content.

### Open Question

Are there capabilities where English's analytic structure is advantageous? Where less morphological complexity lets the model see through to something deeper faster? This would manifest as crossed emergence curves - French earlier on some probes, English earlier on others.

## Related Work: Axiomatic Prompting

The `/Users/adam/dev/axioms/` project tests the same "instrumentation" thesis from a different angle: if transformers are instruments that resolve structure, then providing explicit formal axioms should dramatically improve performance in bounded expert domains.

Results from that project show that explicit axiomatic structure in the prompt measurably changes classification accuracy: where zero-shot accuracy is below 70%, axioms raised it by 15-24% (GoEmotions +14.7%, LEDGAR +24.0%, CADEC +15.3%), while above 70% they degraded it, and the 70% threshold predicted the direction of the effect on all six tasks tested (Wasserman 2026, *The 70% Rule*) - evidence that explicit structure in the *prompt*, not just the training data, affects whether the instrument functions correctly.

## Capability Probes

### Core Capabilities (13 probes)
- Arithmetic (basic, multi-step)
- Pattern recognition (sequence completion)
- Logic (syllogisms, negation, recursive reasoning)
- Analogy
- Categorization
- Reading comprehension
- Common sense
- Grammar
- World knowledge
- Creativity

### Language-Specific Syntactic Probes (3 probes)
1. **Relative clause attachment ambiguity** - English prefers high attachment, French prefers low attachment
2. **Gender agreement across long-distance dependencies** - Tests French morphological tracking
3. **Clitic pronoun placement** - French has strict preverbal ordering English lacks

## References

- Training code: `scripts/train_resumable.py`
- Capability probes: `scripts/enhanced_capability_probes.py`
- Validation: `scripts/validation_perplexity.py`
