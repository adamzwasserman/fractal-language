# English Considered Harmful

We trained identical AI models on French vs English. Same architecture, same hyperparameters, same data source. The only variable was the language.

French achieved 100% grammar accuracy in 197M tokens. English remained at chance level (40%) after 3 billion tokens. A 15x efficiency gap.

But here's the part that surprised us: we thought English might just provide "less signal." So we mixed French and English in the same model to test this. If English were merely less helpful, French should still learn—just slower.

That's not what happened. French grammar dropped from 100% to 50-60%. English stayed at chance. English didn't just fail to help—it actively destroyed what French had learned.

We replicated this with code. Rust-only training achieved 3.7 perplexity. Rust mixed with English: 41.8 perplexity. Same accuracy on structural probes, but 11x worse at prediction. The model can still tell correct from incorrect, but its understanding is shallow.

Why does this happen? English is morphologically impoverished. "Les grandes portes sont ouvertes" marks feminine plural five times across five words. "The big doors are open" marks it once. French provides dense, redundant structural signal. English leaves structure implicit, inferred from word order.

When a model trains on English, it learns that structural information is NOT explicitly marked. This creates a prior against looking for explicit markers. When it then encounters French or Rust with explicit structure, that prior interferes. The signal gets scrambled.

We also found that perplexity and grammar accuracy are orthogonal—you can improve one indefinitely without affecting the other. At 350M parameters, the PPX gap between French and English has nearly converged (~1.2x ratio), yet the grammar gap persists: French stable at 100%, English fluctuating around 60%. Standard evaluation metrics miss structural learning deficits entirely.

Scale makes it worse for structured languages. Our 350M French model needed 14.9x more tokens to achieve grammar emergence than our 125M model (60.9M vs 4.1M tokens). Bigger models aren't more efficient for structured learning—they need more data before structure emerges. The field's focus on scale might be an artifact of training on English, where brute force is the only path forward.

The implication: the compute requirements we cite for AI training aren't universal laws. They're artifacts of English's morphological poverty. French doesn't need trillion-token training because its grammar is already explicit in the text.

We're not saying abandon English. But the assumption that more English is always better might be exactly wrong for structural capabilities.

Pre-registered on OSF. Code and training logs public. We welcome replication.

Paper: "English Considered Harmful: How Morphological Poverty Pollutes Language Model Training
