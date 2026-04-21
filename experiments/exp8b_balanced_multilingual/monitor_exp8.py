#!/usr/bin/env python3
# EMA smoothing
EMA_ALPHA = 0.2
_ema_state = {}
def ema(lang, key, val):
    if key in ("step", "tokens_M") or val is None: return val
    try:
        val = float(val)
        if val == 0: return val
    except: return val
    k = f"{lang}_{key}"
    if k not in _ema_state: _ema_state[k] = val
    else: _ema_state[k] = EMA_ALPHA * val + (1-EMA_ALPHA) * _ema_state[k]
    return _ema_state[k]
"""
Monitor Exp8 training progress across all 12 languages.
Displays a table with one column per language, rows for metrics.

Usage:
    python3 monitor_exp8.py                     # One-shot display
    python3 monitor_exp8.py --watch             # Continuous refresh
    python3 monitor_exp8.py --remote            # Monitor remote server
"""

import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from datetime import datetime

# Handle Ctrl+C properly
def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Config
LANGUAGES = ["en", "fr", "es", "fi", "ru", "vi", "zh", "synth_a", "synth_b", "synth_c", "synth_d"]
# Auto-detect path: remote server vs local Mac
DEFAULT_PATH = "/workspace/exp8" if Path("/workspace/exp8").exists() else "/Volumes/fractal/exp8_multilingual"
LOCAL_PATH = Path(os.environ.get("DATA_PATH", DEFAULT_PATH))
MODEL_SIZE = "125M"

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Metrics to display: (key, label, formatter, higher_is_better)
METRICS = [
    ("step", "Step", lambda r: f"{r.get('step', 0):,}", True),
    ("tokens_M", "Tokens (M)", lambda r: f"{r.get('tokens_M', 0):.0f}", True),
    ("loss", "Loss", lambda r: f"{r.get('loss', float('nan')):.3f}", False),
    ("ppl", "PPL", lambda r: f"{math.exp(min(r.get('loss', 10), 10)):.0f}" if r.get('loss') else "-", False),
    ("val_ppl", "Val PPL", lambda r: f"{r.get('val_ppl', float('nan')):.1f}", False),
    ("grammar_acc", "Grammar %", lambda r: f"{r.get('grammar_acc', 0)*100:.0f}", True),
    ("reasoning_acc", "Reason %", lambda r: f"{r.get('reasoning_acc', 0)*100:.0f}", True),
    ("H_loss", "H(loss)", lambda r: f"{r.get('H_loss', float('nan')):.3f}", True),
    ("H_logit_entropy", "H(entropy)", lambda r: f"{r.get('H_logit_entropy', float('nan')):.3f}", True),
    ("H_top_k", "H(top_k)", lambda r: f"{r.get('H_top_k', float('nan')):.3f}", True),
]


def load_latest_probe(log_dir: Path, lang: str) -> dict:
    """Load most recent probe results for a language."""
    probe_file = log_dir / f"{lang}_{MODEL_SIZE}_probes.jsonl"

    if not probe_file.exists():
        # Try CSV fallback
        csv_file = log_dir / f"{lang}_{MODEL_SIZE}.csv"
        if csv_file.exists():
            with open(csv_file) as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Parse last line
                    header = lines[0].strip().split(',')
                    # Find last line with probe data (grammar_acc not empty)
                    grammar_idx = header.index('grammar_acc') if 'grammar_acc' in header else -1
                    values = None
                    for line in reversed(lines[1:]):
                        parts = line.strip().split(',')
                        if len(parts) > grammar_idx and parts[grammar_idx]:
                            values = parts
                            break
                    if not values:
                        values = lines[-1].strip().split(',')
                    result = {}
                    for h, v in zip(header, values):
                        try:
                            result[h] = float(v) if v and v != '' else float('nan')
                        except:
                            result[h] = v
                    return result
        return {}

    # Load last line of JSONL
    last_line = None
    with open(probe_file) as f:
        for line in f:
            if line.strip():
                last_line = line

    if last_line:
        try:
            return json.loads(last_line)
        except:
            pass
    return {}


def load_checkpoint_info(checkpoint_dir: Path, lang: str) -> dict:
    """Load info from latest checkpoint."""
    ckpt_dir = checkpoint_dir / lang / MODEL_SIZE
    if not ckpt_dir.exists():
        return {}

    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.json"),
                        key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0)

    if not checkpoints:
        return {}

    with open(checkpoints[-1]) as f:
        return json.load(f)


def get_numeric_value(data: dict, key: str) -> float:
    """Extract numeric value for comparison."""
    # Special case: PPL is computed from loss
    if key == "ppl":
        loss = data.get("loss", float('nan'))
        if isinstance(loss, (int, float)) and not math.isnan(loss):
            return math.exp(min(loss, 10))  # Cap to avoid overflow
        return float('nan')

    val = data.get(key, float('nan'))
    if isinstance(val, (int, float)):
        return float(val)
    return float('nan')


def format_table(data: dict) -> str:
    """Format metrics as a table with color coding."""
    # Column widths
    metric_width = 12
    col_width = 10

    lines = []

    # Header
    header = f"{'Metric':<{metric_width}}"
    for lang in LANGUAGES:
        header += f" {lang:>{col_width}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for key, label, formatter, higher_is_better in METRICS:
        # Get all numeric values for this metric
        values = {}
        for lang in LANGUAGES:
            result = data.get(lang, {})
            val = get_numeric_value(result, key)
            if not math.isnan(val) and val != 0:
                values[lang] = val

        # Find winners and losers (handles ties)
        winner_langs = set()
        loser_langs = set()
        if len(values) >= 2:
            max_val = max(values.values())
            min_val = min(values.values())
            if max_val != min_val:  # Only color if there's a difference
                if higher_is_better:
                    winner_langs = {lang for lang, val in values.items() if val == max_val}
                    loser_langs = {lang for lang, val in values.items() if val == min_val}
                else:
                    winner_langs = {lang for lang, val in values.items() if val == min_val}
                    loser_langs = {lang for lang, val in values.items() if val == max_val}

        row = f"{label:<{metric_width}}"
        for lang in LANGUAGES:
            result = data.get(lang, {})
            try:
                value = formatter(result)
            except:
                value = "-"

            # Pad first, then apply colors (so alignment isn't affected)
            padded = f"{value:>{col_width}}"
            if lang in winner_langs:
                padded = f"{GREEN}{padded}{RESET}"
            elif lang in loser_langs:
                padded = f"{RED}{padded}{RESET}"

            row += f" {padded}"
        lines.append(row)

    return "\n".join(lines)


def get_progress_summary(data: dict) -> str:
    """Summarize overall progress."""
    total_tokens = sum(d.get('tokens_M', 0) for d in data.values())
    active = sum(1 for d in data.values() if d.get('step', 0) > 0)

    # Find fastest and slowest
    steps = {lang: d.get('step', 0) for lang, d in data.items()}
    fastest = max(steps.items(), key=lambda x: x[1]) if steps else ("", 0)
    slowest = min((s for s in steps.items() if s[1] > 0), key=lambda x: x[1], default=("", 0))

    # Grammar leaders
    grammar = {lang: d.get('grammar_acc', 0) for lang, d in data.items() if d.get('step', 0) > 1000}
    if grammar:
        best_grammar = max(grammar.items(), key=lambda x: x[1])
    else:
        best_grammar = ("", 0)

    # Show in M if less than 1B, otherwise in B
    if total_tokens < 1000:
        tokens_str = f"{total_tokens:.0f}M"
    else:
        tokens_str = f"{total_tokens/1000:.1f}B"

    lines = [
        f"Total tokens: {tokens_str} across {active} active languages",
        f"Fastest: {fastest[0]} @ step {fastest[1]:,}",
        f"Slowest: {slowest[0]} @ step {slowest[1]:,}" if slowest[1] > 0 else "",
        f"Best grammar: {best_grammar[0]} @ {best_grammar[1]*100:.0f}%" if best_grammar[1] > 0 else "",
    ]
    return "\n".join(l for l in lines if l)


def main():
    watch_mode = "--watch" in sys.argv or "-w" in sys.argv

    log_dir = LOCAL_PATH / "logs"
    checkpoint_dir = LOCAL_PATH / "checkpoints"

    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        print("Make sure backhaul is running or set DATA_PATH env var")
        sys.exit(1)

    while True:
        # Clear screen in watch mode
        if watch_mode:
            print("\033[2J\033[H", end="")

        print(f"=== Exp8 Training Monitor ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
        print(f"Data: {LOCAL_PATH} [EMA α={EMA_ALPHA}]")
        print()

        # Load data for all languages
        data = {}
        for lang in LANGUAGES:
            probe_data = load_latest_probe(log_dir, lang)
            ckpt_data = load_checkpoint_info(checkpoint_dir, lang)
            # Merge with probe data taking priority
            data[lang] = {**ckpt_data, **probe_data}
            # Apply EMA smoothing
            for k, v in data[lang].items(): data[lang][k] = ema(lang, k, v)

        # Display table
        print(format_table(data))
        print()

        # Summary
        print(get_progress_summary(data))

        if not watch_mode:
            break

        print("\n[Refreshing every 10s - Ctrl+C to exit]")
        # Sleep in small increments so Ctrl+C works
        for _ in range(10):
            time.sleep(1)


if __name__ == "__main__":
    main()
