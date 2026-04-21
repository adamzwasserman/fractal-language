# OSF Pre-registration: Experiment 4
## Rust as Synthetic Morphology: Testing the Pollution Hypothesis

**Project:** The Language-Only Hypothesis: Emergent Capabilities in Large Language Models
**Related Pre-registration:** OSF 10.17605/OSF.IO/SJ48B (Experiments 1-3)

---

### Study Information

#### 1. Title
Rust as Synthetic Morphology: Does English Interfere with Structural Pattern Learning in Code?

#### 2. Authors
Adam Zachary Wasserman

#### 3. Description
This experiment extends The Language-Only Hypothesis to programming languages, testing whether Rust's explicit structural markers function analogously to French morphological redundancy, and whether English text interferes with learning these patterns.

**Background:** Our previous experiments (pre-registered OSF 10.17605/OSF.IO/SJ48B) demonstrated that:
1. French achieves grammatical competence (100% on agreement probes) at 197M tokens while English remains at chance (40%) after 3B tokens
2. Interleaving English with French produces negative transfer—the pre-registered hypothesis of positive transfer was falsified

We interpret these findings through the "telescope framework": LLMs are measurement instruments that reveal structure in training data. Morphologically rich signals (French) provide clearer structure; mixing with morphologically impoverished signals (English) adds noise.

**Extension to Code:** Rust provides explicit structural markers analogous to French morphology:
- Lifetime annotations (`'a`) = agreement markers (must match across signatures)
- Ownership system (`&`, `&mut`, `Box<T>`) = redundant structural marking
- Borrow checker = grammar enforcer (rejects "ungrammatical" code)
- Type system = explicit categorical marking

If English "polluted" French learning, it should similarly pollute Rust learning.

#### 4. Hypotheses

**Primary Hypothesis (H1):** A model trained on Rust-only will achieve higher accuracy on Rust structural probes than an identical model trained on Rust+English (interleaved), when evaluated at matched training steps.

**Secondary Hypothesis (H2):** The Rust-only model will show faster acquisition of structural patterns, reaching threshold performance (>75% accuracy) at fewer training tokens than the Rust+English model.

**Exploratory Hypothesis (H3):** The pattern of results will parallel the French vs French+English findings: mixing a structurally explicit language (Rust/French) with a structurally implicit language (English) produces interference, not synthesis.

---

### Design Plan

#### 5. Study Type
Observational study / Controlled computational experiment

#### 6. Blinding
No blinding. This is a computational experiment with objective evaluation metrics.

#### 7. Study Design

**Conditions:**
1. **Rust-only:** 125M parameter transformer trained exclusively on Rust code
2. **Rust+English:** 125M parameter transformer trained on interleaved Rust code and English text (alternating documents)

**Architecture:** GPT-2 style transformer (identical to Experiments 1-3)
- 12 layers, d_model=768, 12 attention heads, d_ff=3072
- Batch size: 2, Sequence length: 512
- Learning rate: 6e-4 (Adam optimizer)
- Random seed: 42

**Data:**
- Rust: The Stack (cleaned Rust subset) - ~993K files, 3.67GB
- English: C4 English corpus (same as Experiments 1-3)
- Tokenizer: 50K BPE (Rust-only uses Rust tokenizer; Rust+EN uses joint tokenizer)

**Training:**
- Total steps: 200,000 per condition
- Checkpoints: Every 1,000 steps
- Evaluation: Every 10,000 steps

#### 8. Randomization
Fixed random seed (42) for reproducibility. Document order within each corpus is shuffled, but shuffling is deterministic given the seed.

---

### Sampling Plan

#### 9. Existing Data
**Registration prior to analysis of the data.**

The Rust corpus (The Stack) is publicly available but we have not analyzed it for this experiment. The English corpus is the same C4 data used in Experiments 1-3.

#### 10. Data Collection Procedures
- Rust data: Downloaded from HuggingFace (`ammarnasr/the-stack-rust-clean`)
- Quality filters: File size 100B-100KB, alphanum fraction >0.25
- Deduplication: MD5 hash-based
- No manual curation or selection

#### 11. Sample Size
Training will proceed for 200,000 steps per condition, matching Experiments 1-3. This represents approximately 200M tokens per condition at batch_size=2, seq_len=512.

#### 12. Sample Size Rationale
Matched to previous experiments for comparability. The French model achieved 100% grammar accuracy by step ~96,000 (197M tokens); we expect Rust structural patterns to emerge on a similar timescale if the morphological analogy holds.

#### 13. Stopping Rule
Training stops at 200,000 steps or when both conditions reach 95%+ accuracy on structural probes (whichever comes first).

---

### Variables

#### 14. Manipulated Variables
**Training data composition:**
- Condition 1: 100% Rust code
- Condition 2: 50% Rust code, 50% English text (interleaved by document)

#### 15. Measured Variables

**Primary outcome:** Accuracy on Rust Structural Probes (minimal pairs)

Probe categories:
1. **Lifetime Agreement** (4 probes): Do lifetime annotations match across signatures?
2. **Ownership Patterns** (4 probes): Correct use of `&`, `&mut`, ownership transfer
3. **Type Consistency** (4 probes): Generic types match instantiation
4. **Borrow Checker Patterns** (3 probes): Valid borrow patterns vs violations
5. **Trait Bounds** (2 probes): Required traits present for operations
6. **Mutability** (2 probes): `mut` keyword where required
7. **Expression vs Statement** (2 probes): Semicolon usage

Total: 21 minimal pairs

**Evaluation metric:** For each minimal pair, the model assigns higher log-probability to the correct Rust code vs the incorrect variant. Accuracy = proportion of pairs where model prefers correct.

**Secondary outcomes:**
- Training loss curves
- Per-category accuracy breakdown
- Steps to 75% threshold
- Perplexity on held-out Rust validation set

#### 16. Indices
Overall accuracy is the primary index. We will also report accuracy by probe category to identify which structural patterns are learned/not learned.

---

### Analysis Plan

#### 17. Statistical Models

**Primary analysis:** Compare accuracy between conditions at matched steps (10K, 50K, 100K, 150K, 200K).

Since this is a computational experiment with deterministic outcomes (given the random seed), we report:
- Accuracy difference: Rust-only minus Rust+English
- 95% bootstrap confidence intervals on accuracy (resampling across probes)

**Secondary analysis:**
- Learning curves: Accuracy vs training step for each condition
- Steps to threshold: First step where accuracy exceeds 75%
- Category breakdown: Which structural patterns show the largest condition difference?

#### 18. Transformations
None. Raw accuracy scores.

#### 19. Inference Criteria
H1 is supported if Rust-only accuracy > Rust+English accuracy at step 200,000, with non-overlapping 95% bootstrap CIs.

H2 is supported if Rust-only reaches 75% accuracy at an earlier step than Rust+English.

#### 20. Data Exclusion
No exclusion criteria. All probes contribute to accuracy calculation.

#### 21. Missing Data
Not applicable. Computational experiment with complete data.

#### 22. Exploratory Analysis
- Comparison with French vs French+English results from Experiments 1-3
- Analysis of which specific probes show largest interference effects
- Qualitative examination of model generations (code completion)
- Cross-evaluation: Test Rust-trained model on French probes and vice versa

---

### Other

#### 23. Other
**Falsification criteria:** If Rust+English outperforms Rust-only, or if there is no significant difference, this would falsify the pollution hypothesis and suggest that the French/English interference finding does not generalize to code.

**Relation to prior work:** This experiment was motivated by the unexpected negative transfer finding in Experiment 3 (interleaved French+English). We predicted positive transfer but observed interference. This follow-up tests whether the interference pattern generalizes beyond natural language to structured formal languages.

**Code and data availability:** All training code, evaluation scripts, and trained model checkpoints will be released upon completion. Pre-registration DOI will be included in any resulting publication.

---

### References
- Pre-registration for Experiments 1-3: OSF 10.17605/OSF.IO/SJ48B
- The Stack dataset: Kocetkov et al. (2022), "The Stack: 3TB of permissively licensed source code"
- Rust structural properties: Matsakis & Klock (2014), "The Rust Language"
