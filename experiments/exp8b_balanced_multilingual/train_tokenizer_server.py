#!/usr/bin/env python3
"""
Train balanced 50k BPE tokenizer on server.

Run on server with: python3 train_tokenizer_server.py
"""

import sentencepiece as spm
from pathlib import Path
import random
from datetime import datetime

# Configuration
VOCAB_SIZE = 50000
CHARACTER_COVERAGE = 0.9999
NUM_THREADS = 32  # Use all server cores
RAW_TEXT_DIR = Path('/workspace/exp8/raw_text')
OUTPUT_DIR = Path('/workspace/exp8')

LANGUAGES = ['en', 'fr', 'es', 'fi', 'ru', 'id', 'vi', 'zh',
             'synth_a', 'synth_b', 'synth_c', 'synth_d']


def main():
    print("=" * 60)
    print("BALANCED TOKENIZER TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Threads: {NUM_THREADS}")
    print()

    # Combine all text files into one balanced corpus
    corpus_file = OUTPUT_DIR / 'balanced_corpus.txt'

    print("Combining corpus files...")
    all_lines = []

    for lang in LANGUAGES:
        text_file = RAW_TEXT_DIR / f'{lang}.txt'
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            print(f"  {lang}: {len(lines):,} lines")
            all_lines.extend(lines)
        else:
            print(f"  {lang}: MISSING!")

    print(f"\nTotal: {len(all_lines):,} lines")
    print("Shuffling...")
    random.seed(42)
    random.shuffle(all_lines)

    print(f"Writing to {corpus_file}...")
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')

    # Train tokenizer
    model_prefix = OUTPUT_DIR / 'balanced_tokenizer'
    print(f"\nTraining tokenizer...")
    print(f"  Output: {model_prefix}")

    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=str(model_prefix),
        vocab_size=VOCAB_SIZE,
        model_type='bpe',
        character_coverage=CHARACTER_COVERAGE,
        num_threads=NUM_THREADS,
        byte_fallback=True,
        normalization_rule_name='nfkc',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        input_sentence_size=10000000,
        shuffle_input_sentence=True,
        max_sentence_length=4192,
    )

    print("\nVerifying tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_prefix) + '.model')

    test_sentences = {
        'en': "The cat sits on the mat.",
        'fr': "Le chat est assis sur le tapis.",
        'zh': "猫坐在垫子上。",
        'synth_b': "The cat [SG] sits [SG] on the mat.",
    }

    print("\nToken fertility (tokens/word):")
    for lang, sent in test_sentences.items():
        tokens = sp.encode(sent, out_type=str)
        words = len(sent.split())
        print(f"  {lang}: {len(tokens)} tokens / {words} words = {len(tokens)/words:.2f}")

    print(f"\nCompleted: {datetime.now().isoformat()}")
    print(f"Model: {model_prefix}.model")
    print(f"Vocab: {model_prefix}.vocab")


if __name__ == '__main__':
    main()
