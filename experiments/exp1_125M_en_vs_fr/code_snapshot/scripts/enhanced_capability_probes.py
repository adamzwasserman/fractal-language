#!/usr/bin/env python
"""
Comprehensive capability probes for controlled French vs English experiment.
Includes 13 core capability probes plus 3 language-specific syntactic probes (16 total):

Core capabilities (13):
- Arithmetic, logic, reasoning, comprehension, knowledge, creativity
- Recursive reasoning, negation logic, multi-step arithmetic

Language-specific probes (3):
1. Relative clause attachment ambiguity (French vs English parsing preferences)
2. Gender agreement across long-distance dependencies (French morphology)
3. Clitic pronoun placement in complex sentences (French syntax)
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import json
from datetime import datetime
import sys
import re

# Set random seeds for reproducibility
mx.random.seed(42)
np.random.seed(42)

# Core capability suite (13 probes)
ORIGINAL_PROBES = [
    {
        "id": "arithmetic_basic",
        "prompt_en": "Calculate: 47 + 83 = ",
        "prompt_fr": "Calculer : 47 + 83 = ",
        "expected_pattern": r"130|cent trente",
        "category": "arithmetic"
    },
    {
        "id": "sequence_completion",
        "prompt_en": "Complete the sequence: 2, 4, 8, 16, ",
        "prompt_fr": "Compléter la séquence : 2, 4, 8, 16, ",
        "expected_pattern": r"32|trente-deux",
        "category": "pattern_recognition"
    },
    {
        "id": "logical_reasoning",
        "prompt_en": "If all birds can fly, and penguins are birds, then penguins can",
        "prompt_fr": "Si tous les oiseaux peuvent voler, et les pingouins sont des oiseaux, alors les pingouins peuvent",
        "expected_pattern": r"fly|voler|not fly|ne.*pas.*voler",
        "category": "logic"
    },
    {
        "id": "analogy",
        "prompt_en": "Cat is to meow as dog is to",
        "prompt_fr": "Chat est à miauler comme chien est à",
        "expected_pattern": r"bark|woof|aboyer|japper",
        "category": "analogy"
    },
    {
        "id": "categorization",
        "prompt_en": "Which doesn't belong: apple, banana, carrot, orange?",
        "prompt_fr": "Lequel n'appartient pas : pomme, banane, carotte, orange ?",
        "expected_pattern": r"carrot|carotte",
        "category": "categorization"
    },
    {
        "id": "reading_comprehension",
        "prompt_en": "John bought 3 apples. He ate 1 apple. How many apples does John have now?",
        "prompt_fr": "Jean a acheté 3 pommes. Il a mangé 1 pomme. Combien de pommes Jean a-t-il maintenant ?",
        "expected_pattern": r"2|two|deux",
        "category": "comprehension"
    },
    {
        "id": "common_sense",
        "prompt_en": "In winter, water becomes",
        "prompt_fr": "En hiver, l'eau devient",
        "expected_pattern": r"ice|frozen|glace|gelée",
        "category": "common_sense"
    },
    {
        "id": "grammar_basic",
        "prompt_en": "The correct form is: He _____ to the store yesterday. (go/went/gone)",
        "prompt_fr": "La forme correcte est : Il _____ au magasin hier. (va/est allé/allait)",
        "expected_pattern": r"went|est allé",
        "category": "grammar"
    },
    {
        "id": "world_knowledge",
        "prompt_en": "The capital of France is",
        "prompt_fr": "La capitale de la France est",
        "expected_pattern": r"Paris",
        "category": "knowledge"
    },
    {
        "id": "creativity",
        "prompt_en": "Write a creative sentence using the words: moon, dancing, silver",
        "prompt_fr": "Écrivez une phrase créative utilisant les mots : lune, dansant, argent",
        "expected_pattern": r".*(moon|lune).*(danc|dans).*(silver|argent).*",
        "category": "creativity"
    },
    {
        "id": "recursive_reasoning",
        "prompt_en": "If A is taller than B, and B is taller than C, who is the shortest?",
        "prompt_fr": "Si A est plus grand que B, et B est plus grand que C, qui est le plus petit ?",
        "expected_pattern": r"C",
        "category": "recursive_logic"
    },
    {
        "id": "negation_reasoning",
        "prompt_en": "It is not true that cats cannot climb trees. Therefore, cats",
        "prompt_fr": "Il n'est pas vrai que les chats ne peuvent pas grimper aux arbres. Par conséquent, les chats",
        "expected_pattern": r"can climb|peuvent grimper",
        "category": "negation_logic"
    },
    {
        "id": "multi_step_arithmetic",
        "prompt_en": "Sarah has 15 apples. She gives away 1/3 of them, then buys 8 more. How many apples does she have?",
        "prompt_fr": "Sarah a 15 pommes. Elle en donne 1/3, puis en achète 8 de plus. Combien de pommes a-t-elle ?",
        "expected_pattern": r"18|dix-huit",
        "category": "complex_arithmetic"
    }
]

# Enhanced language-specific probes
ENHANCED_PROBES = [
    {
        "id": "relative_clause_attachment",
        "prompt_en": "The servant of the actress who was on the balcony waved. Who was on the balcony?",
        "prompt_fr": "Le serviteur de l'actrice qui était sur le balcon a salué. Qui était sur le balcon ?",
        "expected_pattern_en": r"(actress|actrice)",  # English: high attachment preference
        "expected_pattern_fr": r"(servant|serviteur)",  # French: low attachment preference  
        "category": "syntactic_ambiguity",
        "notes": "Tests cross-linguistic differences in syntactic parsing preferences"
    },
    {
        "id": "gender_agreement_long_distance", 
        "prompt_en": "The tall woman who lives in the house with the red door is very intelligent",
        "prompt_fr": "La grande femme qui habite dans la maison avec la porte rouge est très",
        "expected_pattern_en": r"intelligent|smart|clever",
        "expected_pattern_fr": r"intelligente|astucieuse|maligne",  # Must agree with feminine "femme"
        "category": "morphological_agreement",
        "notes": "Tests long-distance gender agreement in French vs English"
    },
    {
        "id": "clitic_pronoun_placement",
        "prompt_en": "I want to give it to him tomorrow",
        "prompt_fr": "Je veux le lui donner demain",  # Clitics must precede infinitive
        "expected_pattern_en": r"give it to him|give him it",
        "expected_pattern_fr": r"le lui donner|lui donner ça",  # Wrong: *donner le lui
        "category": "clitic_placement",
        "notes": "Tests French clitic pronoun placement rules vs English"
    }
]

# Simple probes for 125M scale - designed to detect early emergence
SIMPLE_PROBES = [
    # === Basic Word Completion (high-frequency patterns) ===
    {
        "id": "simple_completion_1",
        "prompt_en": "The cat sat on the",
        "prompt_fr": "Le chat était assis sur le",
        "expected_pattern": r"mat|floor|chair|sofa|bed|couch|tapis|sol|canapé|fauteuil|lit",
        "category": "basic_completion",
        "notes": "Tests basic noun phrase completion"
    },
    {
        "id": "simple_completion_2",
        "prompt_en": "She opened the door and walked",
        "prompt_fr": "Elle a ouvert la porte et",
        "expected_pattern": r"in|out|inside|outside|through|away|est entrée|est sortie|a marché",
        "category": "basic_completion",
        "notes": "Tests action sequence completion"
    },
    {
        "id": "simple_completion_3",
        "prompt_en": "The sun rises in the",
        "prompt_fr": "Le soleil se lève à l'",
        "expected_pattern": r"east|morning|sky|est|matin|horizon",
        "category": "basic_completion",
        "notes": "Tests world knowledge completion"
    },

    # === Simple Sequence (easier than 2,4,8,16) ===
    {
        "id": "simple_sequence",
        "prompt_en": "1, 2, 3, 4,",
        "prompt_fr": "1, 2, 3, 4,",
        "expected_pattern": r"5",
        "category": "simple_pattern",
        "notes": "Simplest sequence completion"
    },
    {
        "id": "simple_sequence_2",
        "prompt_en": "Monday, Tuesday, Wednesday,",
        "prompt_fr": "Lundi, mardi, mercredi,",
        "expected_pattern": r"Thursday|jeudi",
        "category": "simple_pattern",
        "notes": "Day sequence completion"
    },

    # === French Gender Agreement (CRITICAL for hypothesis) ===
    {
        "id": "fr_gender_fem_adj",
        "prompt_en": "The house is very",
        "prompt_fr": "La maison est très",
        "expected_pattern_en": r"big|beautiful|old|nice|large|small",
        "expected_pattern_fr": r"grande|belle|vieille|jolie|petite",  # Must be feminine
        "category": "gender_agreement",
        "notes": "French feminine adjective agreement - key test"
    },
    {
        "id": "fr_gender_masc_adj",
        "prompt_en": "The book is very",
        "prompt_fr": "Le livre est très",
        "expected_pattern_en": r"good|interesting|old|long|new",
        "expected_pattern_fr": r"bon|intéressant|vieux|long|nouveau",  # Must be masculine
        "category": "gender_agreement",
        "notes": "French masculine adjective agreement"
    },

    # === French Plural Agreement (CRITICAL for hypothesis) ===
    {
        "id": "fr_plural_adj",
        "prompt_en": "The children are very",
        "prompt_fr": "Les enfants sont très",
        "expected_pattern_en": r"happy|young|smart|tired|good",
        "expected_pattern_fr": r"contents|jeunes|intelligents|fatigués|sages",  # Must be plural
        "category": "plural_agreement",
        "notes": "French plural adjective agreement"
    },
    {
        "id": "fr_plural_fem_adj",
        "prompt_en": "The girls are very",
        "prompt_fr": "Les filles sont très",
        "expected_pattern_en": r"happy|young|smart|pretty|nice",
        "expected_pattern_fr": r"contentes|jeunes|intelligentes|jolies|gentilles",  # Feminine plural
        "category": "plural_agreement",
        "notes": "French feminine plural agreement - complex morphology"
    },

    # === Subject-Verb Agreement (EN baseline) ===
    {
        "id": "en_sv_agreement_sing",
        "prompt_en": "The dog",
        "prompt_fr": "Le chien",
        "expected_pattern_en": r"is|was|runs|barks|likes|has",  # Singular verb
        "expected_pattern_fr": r"est|était|court|aboie|aime|a",
        "category": "sv_agreement",
        "notes": "Basic subject-verb agreement - singular"
    },
    {
        "id": "en_sv_agreement_plur",
        "prompt_en": "The dogs",
        "prompt_fr": "Les chiens",
        "expected_pattern_en": r"are|were|run|bark|like|have",  # Plural verb
        "expected_pattern_fr": r"sont|étaient|courent|aboient|aiment|ont",
        "category": "sv_agreement",
        "notes": "Basic subject-verb agreement - plural"
    },

    # === French Determiner Agreement ===
    {
        "id": "fr_det_fem",
        "prompt_en": "I see a table",
        "prompt_fr": "Je vois une",  # Must continue with feminine noun
        "expected_pattern_en": r"table|chair|house|car|woman",
        "expected_pattern_fr": r"table|chaise|maison|voiture|femme|fille",  # Feminine nouns
        "category": "determiner_agreement",
        "notes": "French feminine determiner primes feminine noun"
    },
    {
        "id": "fr_det_masc",
        "prompt_en": "I see a tree",
        "prompt_fr": "Je vois un",  # Must continue with masculine noun
        "expected_pattern_en": r"tree|car|man|dog|book",
        "expected_pattern_fr": r"arbre|homme|chien|livre|garçon|chat",  # Masculine nouns
        "category": "determiner_agreement",
        "notes": "French masculine determiner primes masculine noun"
    },

    # === Basic Coherence (topic maintenance) ===
    {
        "id": "coherence_food",
        "prompt_en": "At the restaurant, I ordered",
        "prompt_fr": "Au restaurant, j'ai commandé",
        "expected_pattern": r"food|meal|dish|pizza|steak|fish|chicken|wine|coffee|un plat|une pizza|du poisson|du poulet|du vin|un café",
        "category": "coherence",
        "notes": "Tests topic-appropriate completion"
    },
    {
        "id": "coherence_weather",
        "prompt_en": "The weather today is very",
        "prompt_fr": "Le temps aujourd'hui est très",
        "expected_pattern": r"cold|hot|warm|nice|bad|sunny|rainy|cloudy|froid|chaud|beau|mauvais|nuageux",
        "category": "coherence",
        "notes": "Tests weather vocabulary completion"
    },

    # === Simple Arithmetic (125M achievable) ===
    {
        "id": "simple_add",
        "prompt_en": "2 + 2 =",
        "prompt_fr": "2 + 2 =",
        "expected_pattern": r"4|four|quatre",
        "category": "simple_arithmetic",
        "notes": "Simplest arithmetic"
    },
    {
        "id": "simple_add_2",
        "prompt_en": "5 + 3 =",
        "prompt_fr": "5 + 3 =",
        "expected_pattern": r"8|eight|huit",
        "category": "simple_arithmetic",
        "notes": "Simple single-digit addition"
    }
]

class CapabilityProber:
    def __init__(self, model_path, tokenizer_path, lang):
        self.lang = lang
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        # Load model (same architecture as training)
        self.model = self._create_model()
        self._load_model(model_path)
        
    def _create_model(self):
        """Create model with identical architecture to training."""
        
        class TransformerBlock(nn.Module):
            def __init__(self, d_model, n_heads, d_ff):
                super().__init__()
                self.attention = nn.MultiHeadAttention(d_model, n_heads)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model)
                )
                self.ln1 = nn.LayerNorm(d_model)
                self.ln2 = nn.LayerNorm(d_model)
            
            def __call__(self, x):
                attn_out = self.attention(self.ln1(x), self.ln1(x), self.ln1(x))
                x = x + attn_out
                mlp_out = self.mlp(self.ln2(x))
                x = x + mlp_out
                return x

        vocab_size = self.vocab_size  # Capture from outer scope

        class Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                cfg = dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072)

                self.embed = nn.Embedding(vocab_size, cfg["d_model"])
                self.blocks = [TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["d_ff"])
                              for _ in range(cfg["n_layers"])]
                self.ln_f = nn.LayerNorm(cfg["d_model"])
                self.head = nn.Linear(cfg["d_model"], vocab_size, bias=False)
                self.head.weight = self.embed.weight

            def __call__(self, x):
                x = self.embed(x)
                for block in self.blocks:
                    x = block(x)
                return self.head(self.ln_f(x))
        
        return Transformer()
    
    def _load_model(self, model_path):
        """Load trained model weights."""
        try:
            weights = mx.load(str(model_path))
            self.model.update(weights)
            print(f"✅ Model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
            
    def generate_response(self, prompt, max_tokens=50, mode="greedy", temp=0.3, top_p=0.9):
        """Generate response with configurable sampling.

        Modes:
        - 'greedy': argmax (temp=0), deterministic but may loop
        - 'low_temp': temperature sampling with low temp (default 0.3)
        - 'nucleus': top-p nucleus sampling
        """
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt).ids

        for _ in range(max_tokens):
            x = mx.array([tokens])
            logits = self.model(x)[0, -1, :]

            if mode == "greedy":
                next_token = mx.argmax(logits).item()
            elif mode == "low_temp":
                probs = mx.softmax(logits / temp)
                next_token = mx.random.categorical(probs).item()
            elif mode == "nucleus":
                # Top-p nucleus sampling
                probs = mx.softmax(logits / temp)
                sorted_indices = mx.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                cumsum = mx.cumsum(sorted_probs)
                # Find cutoff
                cutoff_idx = int(mx.sum(cumsum < top_p).item()) + 1
                cutoff_idx = min(cutoff_idx, len(sorted_probs))
                # Zero out tokens below cutoff
                mask = mx.zeros_like(probs)
                mask[sorted_indices[:cutoff_idx]] = 1
                masked_probs = probs * mask
                masked_probs = masked_probs / mx.sum(masked_probs)
                next_token = mx.random.categorical(masked_probs).item()
            else:
                next_token = mx.argmax(logits).item()

            tokens.append(next_token)

            if next_token == self.tokenizer.token_to_id("<eos>"):
                break

        response = self.tokenizer.decode(tokens)
        return response[len(prompt):].strip()
    
    def check_match(self, response, pattern):
        """Check if response matches expected pattern."""
        if not response or not pattern:
            return False
        # Check first 100 chars of response for the pattern
        return bool(re.search(pattern, response[:100], re.IGNORECASE))

    def run_probes(self, mode="greedy"):
        """Run all capability probes with specified sampling mode."""
        results = {
            "language": self.lang,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "original_probes": {},
            "enhanced_probes": {},
            "simple_probes": {},
            "scores": {}
        }

        # Run simple probes first (125M scale)
        print(f"\n🔍 Running SIMPLE probes ({self.lang.upper()}, mode={mode})...")
        simple_correct = 0
        simple_total = 0
        for probe in SIMPLE_PROBES:
            prompt = probe[f"prompt_{self.lang}"]
            response = self.generate_response(prompt, max_tokens=20, mode=mode)

            # Get expected pattern (language-specific or shared)
            if f"expected_pattern_{self.lang}" in probe:
                pattern = probe[f"expected_pattern_{self.lang}"]
            else:
                pattern = probe.get("expected_pattern", "")

            matched = self.check_match(response, pattern)
            if matched:
                simple_correct += 1
            simple_total += 1

            results["simple_probes"][probe["id"]] = {
                "prompt": prompt,
                "response": response,
                "category": probe["category"],
                "expected_pattern": pattern,
                "matched": matched,
                "notes": probe.get("notes", "")
            }
            status = "✅" if matched else "❌"
            print(f"  {status} {probe['id']}: '{response[:40]}...'")

        results["scores"]["simple"] = {
            "correct": simple_correct,
            "total": simple_total,
            "accuracy": simple_correct / simple_total if simple_total > 0 else 0
        }
        print(f"\n  📊 Simple probes: {simple_correct}/{simple_total} ({100*simple_correct/simple_total:.1f}%)")

        # Run core capability suite (13 probes)
        print(f"\n🔍 Running core capability probes ({self.lang.upper()}, mode={mode})...")
        core_correct = 0
        core_total = 0
        for probe in ORIGINAL_PROBES:
            prompt = probe[f"prompt_{self.lang}"]
            response = self.generate_response(prompt, mode=mode)

            pattern = probe["expected_pattern"]
            matched = self.check_match(response, pattern)
            if matched:
                core_correct += 1
            core_total += 1

            results["original_probes"][probe["id"]] = {
                "prompt": prompt,
                "response": response,
                "category": probe["category"],
                "expected_pattern": pattern,
                "matched": matched
            }
            status = "✅" if matched else "❌"
            print(f"  {status} {probe['id']}: '{response[:40]}...'")

        results["scores"]["core"] = {
            "correct": core_correct,
            "total": core_total,
            "accuracy": core_correct / core_total if core_total > 0 else 0
        }
        print(f"\n  📊 Core probes: {core_correct}/{core_total} ({100*core_correct/core_total:.1f}%)")

        # Run enhanced language-specific probes
        print(f"\n🔍 Running enhanced syntactic probes ({self.lang.upper()}, mode={mode})...")
        enhanced_correct = 0
        enhanced_total = 0
        for probe in ENHANCED_PROBES:
            prompt = probe[f"prompt_{self.lang}"]
            response = self.generate_response(prompt, mode=mode)

            pattern = probe[f"expected_pattern_{self.lang}"]
            matched = self.check_match(response, pattern)
            if matched:
                enhanced_correct += 1
            enhanced_total += 1

            results["enhanced_probes"][probe["id"]] = {
                "prompt": prompt,
                "response": response,
                "category": probe["category"],
                "expected_pattern": pattern,
                "matched": matched,
                "notes": probe["notes"]
            }
            status = "✅" if matched else "❌"
            print(f"  {status} {probe['id']}: '{response[:40]}...'")

        results["scores"]["enhanced"] = {
            "correct": enhanced_correct,
            "total": enhanced_total,
            "accuracy": enhanced_correct / enhanced_total if enhanced_total > 0 else 0
        }
        print(f"\n  📊 Enhanced probes: {enhanced_correct}/{enhanced_total} ({100*enhanced_correct/enhanced_total:.1f}%)")

        # Overall summary
        total_correct = simple_correct + core_correct + enhanced_correct
        total_probes = simple_total + core_total + enhanced_total
        results["scores"]["overall"] = {
            "correct": total_correct,
            "total": total_probes,
            "accuracy": total_correct / total_probes if total_probes > 0 else 0
        }
        print(f"\n{'='*50}")
        print(f"📈 OVERALL: {total_correct}/{total_probes} ({100*total_correct/total_probes:.1f}%)")
        print(f"{'='*50}")

        return results

def run_capability_analysis(model_checkpoint_step, lang, mode="greedy"):
    """Run complete capability probe analysis."""
    base_path = Path("/Volumes/Misc Backup/fractal")

    # Paths
    model_path = base_path / "checkpoints" / lang / "125M" / f"model_{model_checkpoint_step}.safetensors"
    tokenizer_path = base_path / "joint_tokenizer.json"
    results_path = base_path / "results" / f"capability_probes_{lang}_{model_checkpoint_step}_{mode}.json"

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None

    # Run probes
    prober = CapabilityProber(model_path, tokenizer_path, lang)
    results = prober.run_probes(mode=mode)

    # Save results
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"💾 Capability analysis saved: {results_path}")
    return results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python enhanced_capability_probes.py <checkpoint_step> <language> [mode]")
        print("  mode: greedy (default), low_temp, nucleus")
        sys.exit(1)

    checkpoint_step = int(sys.argv[1])
    language = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "greedy"
    run_capability_analysis(checkpoint_step, language, mode)