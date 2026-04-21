#!/usr/bin/env python
"""
Training logger - monitors running training jobs and records metrics to CSV.
Runs separately from training, doesn't interrupt anything.

Usage:
    uv run python scripts/training_logger.py [--once]

    --once: Just dump current state and exit (don't tail)
"""
import json
import csv
import re
import sys
import time
import math
from pathlib import Path
from datetime import datetime

# Paths
CHECKPOINT_BASE = Path("/Volumes/Misc Backup/fractal/checkpoints")
LOG_DIR = Path("/Volumes/Misc Backup/fractal/logs")
TASK_OUTPUT_DIR = Path("/tmp/claude/tasks")

# Task IDs for current runs (can have multiple for restarts)
TASK_IDS = {
    "en": ["en_training"],
    "fr": ["fr_training"]
}

def calculate_rolling_stats(values: list[float], window: int = 10) -> tuple[float, float, float]:
    """Calculate rolling mean, stddev, and coefficient of variation.

    Returns (mean, stddev, cv_percent) for the last `window` values.
    """
    if not values or len(values) < 2:
        return (None, None, None)

    recent = values[-window:] if len(values) >= window else values
    n = len(recent)

    mean = sum(recent) / n
    variance = sum((x - mean) ** 2 for x in recent) / n
    stddev = math.sqrt(variance)
    cv = (stddev / mean * 100) if mean != 0 else 0

    return (mean, stddev, cv)


def get_checkpoints(lang: str, model_size: str = "125M") -> list[dict]:
    """Read all checkpoint metadata for a language."""
    ckpt_dir = CHECKPOINT_BASE / lang / model_size
    if not ckpt_dir.exists():
        return []

    checkpoints = []
    for ckpt_file in sorted(ckpt_dir.glob("checkpoint_*.json")):
        try:
            with open(ckpt_file) as f:
                data = json.load(f)
                data["_path"] = str(ckpt_file)  # Store path for updates
                checkpoints.append(data)
        except (json.JSONDecodeError, IOError):
            continue

    return sorted(checkpoints, key=lambda x: x.get("step", 0))


def update_checkpoint_with_perplexity(lang: str, step: int, perplexity: float, passed: bool, model_size: str = "125M"):
    """Write perplexity back to checkpoint JSON file for permanent storage."""
    ckpt_path = CHECKPOINT_BASE / lang / model_size / f"checkpoint_{step}.json"
    if not ckpt_path.exists():
        return False

    try:
        with open(ckpt_path, 'r') as f:
            data = json.load(f)

        # Only update if not already present
        if "perplexity" not in data or data.get("perplexity") != perplexity:
            data["perplexity"] = perplexity
            data["emergence_passed"] = passed
            with open(ckpt_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
    except (json.JSONDecodeError, IOError):
        pass
    return False


def parse_task_output(task_id: str) -> list[dict]:
    """Parse smoke test results from task output file."""
    # Try .output first, then .log
    output_file = TASK_OUTPUT_DIR / f"{task_id}.output"
    if not output_file.exists():
        output_file = TASK_OUTPUT_DIR / f"{task_id}.log"
    if not output_file.exists():
        return []

    results = []
    # Pattern: "Smoke test: ppl=1262.1, passed=False"
    smoke_pattern = re.compile(r"Smoke test: ppl=([\d.]+), passed=(True|False)")
    # Pattern: "Checkpoint saved: step 1000, loss 6.6155, keep=False"
    ckpt_pattern = re.compile(r"Checkpoint saved: step (\d+), loss ([\d.]+)")

    current_step = None
    current_loss = None

    try:
        with open(output_file, 'r') as f:
            for line in f:
                # Check for checkpoint line (comes before smoke test)
                ckpt_match = ckpt_pattern.search(line)
                if ckpt_match:
                    current_step = int(ckpt_match.group(1))
                    current_loss = float(ckpt_match.group(2))

                # Check for smoke test line
                smoke_match = smoke_pattern.search(line)
                if smoke_match and current_step is not None:
                    results.append({
                        "step": current_step,
                        "loss": current_loss,
                        "perplexity": float(smoke_match.group(1)),
                        "passed": smoke_match.group(2) == "True"
                    })
                    current_step = None
                    current_loss = None
    except IOError:
        pass

    return results


def write_csv(lang: str, data: list[dict], model_size: str = "125M"):
    """Write training metrics to CSV."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / f"{lang}_{model_size}_training.csv"

    # Deduplicate by step
    seen_steps = set()
    unique_data = []
    for row in data:
        if row["step"] not in seen_steps:
            seen_steps.add(row["step"])
            unique_data.append(row)

    unique_data = sorted(unique_data, key=lambda x: x["step"])

    fieldnames = ["step", "loss", "perplexity", "passed",
                  "loss_mean_10", "loss_std_10", "loss_cv_10",
                  "ppl_mean_10", "ppl_std_10", "ppl_cv_10", "timestamp"]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in unique_data:
            writer.writerow({
                "step": row.get("step"),
                "loss": row.get("loss"),
                "perplexity": row.get("perplexity"),
                "passed": row.get("passed"),
                "loss_mean_10": row.get("loss_mean_10"),
                "loss_std_10": row.get("loss_std_10"),
                "loss_cv_10": row.get("loss_cv_10"),
                "ppl_mean_10": row.get("ppl_mean_10"),
                "ppl_std_10": row.get("ppl_std_10"),
                "ppl_cv_10": row.get("ppl_cv_10"),
                "timestamp": row.get("timestamp", "")
            })

    return csv_path, len(unique_data)


def collect_all_data(lang: str, model_size: str = "125M") -> list[dict]:
    """Combine checkpoint data with task output data."""
    # Get checkpoint data
    checkpoints = get_checkpoints(lang, model_size)
    ckpt_by_step = {c["step"]: c for c in checkpoints}

    # Get task output data from all task files (handles restarts)
    task_ids = TASK_IDS.get(lang, [])
    task_data = []
    for task_id in task_ids:
        task_data.extend(parse_task_output(task_id))

    # Merge: task output has perplexity, checkpoints have timestamps
    merged = []
    all_steps = set(ckpt_by_step.keys()) | {t["step"] for t in task_data}

    task_by_step = {t["step"]: t for t in task_data}

    # Collect values for rolling variance calculation
    loss_history = []
    ppl_history = []

    for step in sorted(all_steps):
        row = {"step": step}

        if step in ckpt_by_step:
            row["loss"] = ckpt_by_step[step].get("loss")
            row["timestamp"] = ckpt_by_step[step].get("timestamp", "")
            # Check if checkpoint already has perplexity saved
            if ckpt_by_step[step].get("perplexity"):
                row["perplexity"] = ckpt_by_step[step]["perplexity"]
                row["passed"] = ckpt_by_step[step].get("emergence_passed", False)

        if step in task_by_step:
            row["loss"] = task_by_step[step].get("loss", row.get("loss"))
            row["perplexity"] = task_by_step[step].get("perplexity")
            row["passed"] = task_by_step[step].get("passed")

            # Persist perplexity to checkpoint file for permanent storage
            if row.get("perplexity"):
                updated = update_checkpoint_with_perplexity(
                    lang, step, row["perplexity"], row["passed"], model_size
                )
                if updated:
                    print(f"       Saved ppl={row['perplexity']:.1f} to checkpoint_{step}.json")

        # Calculate rolling variance for loss (10-checkpoint window)
        if row.get("loss") is not None:
            loss_history.append(row["loss"])
            mean, stddev, cv = calculate_rolling_stats(loss_history, window=10)
            row["loss_mean_10"] = round(mean, 4) if mean else None
            row["loss_std_10"] = round(stddev, 4) if stddev else None
            row["loss_cv_10"] = round(cv, 2) if cv else None

        # Calculate rolling variance for perplexity (10-checkpoint window)
        if row.get("perplexity") is not None:
            ppl_history.append(row["perplexity"])
            mean, stddev, cv = calculate_rolling_stats(ppl_history, window=10)
            row["ppl_mean_10"] = round(mean, 2) if mean else None
            row["ppl_std_10"] = round(stddev, 2) if stddev else None
            row["ppl_cv_10"] = round(cv, 2) if cv else None

        merged.append(row)

    return merged


def print_summary(lang: str, data: list[dict]):
    """Print a summary of training progress."""
    if not data:
        print(f"  {lang.upper()}: No data yet")
        return

    latest = data[-1]
    print(f"  {lang.upper()}: step {latest['step']}", end="")
    if latest.get("loss"):
        print(f", loss={latest['loss']:.4f}", end="")
    if latest.get("perplexity"):
        print(f", ppl={latest['perplexity']:.1f}", end="")
    print()

    # Show variance metrics for loss
    if latest.get("loss_cv_10") is not None:
        cv = latest["loss_cv_10"]
        stability = "stable" if cv < 8 else "moderate" if cv < 12 else "volatile"
        print(f"       Loss variance (10-ckpt): CV={cv:.1f}% ({stability})")

    # Show variance metrics for perplexity
    if latest.get("ppl_cv_10") is not None:
        cv = latest["ppl_cv_10"]
        std = latest.get("ppl_std_10", 0)
        stability = "stable" if cv < 5 else "moderate" if cv < 10 else "volatile"
        print(f"       PPL variance (10-ckpt): CV={cv:.1f}%, std={std:.1f} ({stability})")

    # Show trend if we have enough data
    if len(data) >= 3:
        recent_ppl = [d.get("perplexity") for d in data[-3:] if d.get("perplexity")]
        if len(recent_ppl) >= 2:
            trend = "improving" if recent_ppl[-1] < recent_ppl[0] else "flat/rising"
            print(f"       Last 3 ppl: {' -> '.join(f'{p:.0f}' for p in recent_ppl)} ({trend})")


def main():
    once_mode = "--once" in sys.argv

    print("=" * 50)
    print("Training Logger")
    print("=" * 50)

    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now}] Collecting metrics...")

        for lang in ["en", "fr"]:
            data = collect_all_data(lang)
            csv_path, count = write_csv(lang, data)
            print_summary(lang, data)
            print(f"       Wrote {count} rows to {csv_path}")

        if once_mode:
            break

        print("\nWaiting 5 minutes... (Ctrl+C to stop)")
        try:
            time.sleep(300)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
