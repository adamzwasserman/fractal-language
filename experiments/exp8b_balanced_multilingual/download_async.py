#!/usr/bin/env python3
"""Async download of natural languages from C4 to 1B tokens each."""
import asyncio
import aiofiles
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("/workspace/exp8/raw_text_extended")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_CHARS = 4_000_000_000  # 1B tokens ≈ 4B chars

LANGS = [("en", "en"), ("fr", "fr"), ("es", "es"), ("fi", "fi"),
         ("ru", "ru"), ("vi", "vi"), ("zh", "zh")]

async def download_lang(lang, c4_name):
    filepath = OUTPUT_DIR / f"{lang}.txt"

    existing_chars = filepath.stat().st_size if filepath.exists() else 0
    if existing_chars >= TARGET_CHARS:
        print(f"{lang}: Already have {existing_chars/1e9:.1f}B chars")
        return

    print(f"{lang}: Starting (have {existing_chars/1e6:.0f}M)...")

    try:
        ds = load_dataset("allenai/c4", name=c4_name, split="train", streaming=True)
    except:
        try:
            ds = load_dataset("mc4", name=lang, split="train", streaming=True)
        except Exception as e:
            print(f"{lang}: Failed: {e}")
            return

    total_chars = existing_chars
    async with aiofiles.open(filepath, 'a', encoding='utf-8') as f:
        batch = []
        for i, example in enumerate(ds):
            text = example["text"].strip()
            if len(text) < 100:
                continue
            escaped = text.replace('\n', '\\n').replace('\r', '\\r')
            batch.append(escaped + '\n')
            total_chars += len(text)

            if len(batch) >= 1000:
                await f.write(''.join(batch))
                batch = []
                if i % 10000 == 0:
                    print(f"  {lang}: {total_chars/1e6:.0f}M chars")

            if total_chars >= TARGET_CHARS:
                break

            if i % 5000 == 0:
                await asyncio.sleep(0)

        if batch:
            await f.write(''.join(batch))

    print(f"{lang}: Done - {total_chars/1e9:.1f}B chars")

async def main():
    print("Async download: 7 languages to 1B tokens each\n")
    tasks = [download_lang(lang, c4) for lang, c4 in LANGS]
    await asyncio.gather(*tasks)
    print("\nAll done.")

if __name__ == "__main__":
    asyncio.run(main())
