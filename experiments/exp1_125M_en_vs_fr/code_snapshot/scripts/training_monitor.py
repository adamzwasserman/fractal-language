#!/usr/bin/env python
"""
Training monitor that:
1. Watches for new checkpoints
2. Runs probes at EVERY checkpoint (every 1000 steps)
3. Records all results to disk
4. Restarts training if it stops before target

Usage:
    uv run python scripts/training_monitor.py
"""
import subprocess
import time
import json
import re
from pathlib import Path
from datetime import datetime

# Configuration
BASE_DIR = Path("/Volumes/Misc Backup/fractal")
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
PROJECT_DIR = Path("/Users/adam/dev/fractal-language")

CURRENT_TARGET = 300_000
CHECK_INTERVAL = 60  # seconds

MODELS = [
    {"lang": "en", "size": "125M"},
    {"lang": "fr", "size": "125M"},
]


def send_notification(title: str, message: str, sound: bool = False, persistent: bool = False):
    """Send macOS notification."""
    if persistent:
        script = f'display alert "{title}" message "{message}"'
    else:
        script = f'display notification "{message}" with title "{title}"'
        if sound:
            script += ' sound name "Glass"'
    subprocess.run(["osascript", "-e", script], capture_output=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] NOTIFY: {title} - {message}")


def get_current_step(lang: str, size: str) -> int:
    """Get current training step from training_state.json"""
    state_file = CHECKPOINTS_DIR / lang / size / "training_state.json"
    if not state_file.exists():
        return 0
    try:
        with open(state_file) as f:
            data = json.load(f)
        return data.get("step", 0)
    except:
        return 0


def get_checkpoint_steps(lang: str, size: str) -> set:
    """Get all checkpoint steps that exist."""
    ckpt_dir = CHECKPOINTS_DIR / lang / size
    steps = set()
    for f in ckpt_dir.glob("checkpoint_*.json"):
        try:
            step = int(f.stem.split("_")[1])
            steps.add(step)
        except:
            pass
    return steps


def is_training_running(lang: str) -> bool:
    """Check if training process is running."""
    result = subprocess.run(
        ["pgrep", "-f", f"training_with_eval.py {lang}"],
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def run_probes(step: int, lang: str, mode: str = "low_temp") -> dict:
    """Run capability probes."""
    print(f"    Running probes for {lang} step {step}...")

    result = subprocess.run(
        ["uv", "run", "python", "scripts/enhanced_capability_probes.py",
         str(step), lang, mode],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        timeout=300
    )

    output = result.stdout + result.stderr
    accuracy = None
    for line in output.split("\n"):
        if "OVERALL:" in line:
            try:
                pct = line.split("(")[1].split("%")[0]
                accuracy = float(pct)
            except:
                pass

    return {
        "step": step,
        "lang": lang,
        "mode": mode,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
    }


def run_validation_perplexity(step: int, lang: str) -> dict:
    """Run validation perplexity (skip if times out)."""
    print(f"    Running validation PPL for {lang} step {step}...")

    try:
        result = subprocess.run(
            ["uv", "run", "python", "scripts/validation_perplexity.py",
             str(step), lang],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=120  # Reduced timeout, skip if too slow
        )

        output = result.stdout + result.stderr
        perplexity = None
        for line in output.split("\n"):
            if "perplexity" in line.lower() or "ppl" in line.lower():
                numbers = re.findall(r'[\d.]+', line)
                for n in numbers:
                    try:
                        val = float(n)
                        if 1 < val < 100000:
                            perplexity = val
                            break
                    except:
                        pass

        return {"perplexity": perplexity}
    except subprocess.TimeoutExpired:
        print(f"    PPL timeout - skipping")
        return {"perplexity": None}


def save_checkpoint_metrics(lang: str, step: int, metrics: dict):
    """Save metrics for a checkpoint."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual file
    metrics_file = RESULTS_DIR / f"checkpoint_metrics_{lang}_{step}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Append to running log
    log_file = RESULTS_DIR / f"all_checkpoints_{lang}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    print(f"    Saved: {metrics_file.name}")


def start_training(lang: str, size: str, target: int):
    """Start training process."""
    print(f"  Starting {lang} training toward {target:,}...")

    cmd = f"env FRACTAL_MEMORY_LIMIT=4.0 nice -n 15 uv run python -u scripts/training_with_eval.py {lang} {size} auto {target}"
    log_file = f"/tmp/{lang}_training.log"

    subprocess.Popen(
        f"nohup {cmd} > {log_file} 2>&1 &",
        shell=True,
        cwd=PROJECT_DIR
    )
    print(f"  Training started, log: {log_file}")


def monitor():
    """Main monitoring loop - runs probes at every new checkpoint."""
    print(f"{'='*60}")
    print(f"Checkpoint Monitor Started")
    print(f"Target: {CURRENT_TARGET:,} steps")
    print(f"Runs probes at EVERY new checkpoint")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print(f"{'='*60}")

    # Track which checkpoints we've already processed
    processed = {m["lang"]: set() for m in MODELS}

    # Initialize with existing checkpoints (don't re-run probes on old ones)
    for model in MODELS:
        lang = model["lang"]
        size = model["size"]
        existing = get_checkpoint_steps(lang, size)
        processed[lang] = existing.copy()
        print(f"  {lang.upper()}: {len(existing)} existing checkpoints (will skip)")

    print()

    while True:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for model in MODELS:
            lang = model["lang"]
            size = model["size"]

            current_step = get_current_step(lang, size)
            running = is_training_running(lang)
            all_steps = get_checkpoint_steps(lang, size)
            new_steps = all_steps - processed[lang]

            status = "running" if running else "STOPPED"
            print(f"[{now}] {lang.upper()}: step {current_step:,} / {CURRENT_TARGET:,} - {status}")

            # Process new checkpoints
            if new_steps:
                for step in sorted(new_steps):
                    print(f"  New checkpoint: {step}")

                    # Run probes (lightweight, ~1GB)
                    probe_result = run_probes(step, lang, "low_temp")

                    # Skip validation perplexity - uses 6GB RAM and competes with training
                    # ppl_result = run_validation_perplexity(step, lang)

                    # Combine and save
                    metrics = {
                        "step": step,
                        "lang": lang,
                        "timestamp": datetime.now().isoformat(),
                        "probe_accuracy": probe_result.get("accuracy"),
                        "validation_ppl": None,  # Disabled to save memory
                    }
                    save_checkpoint_metrics(lang, step, metrics)

                    # Mark as processed
                    processed[lang].add(step)

                    print(f"    Probe: {metrics['probe_accuracy']}%")

            # Check if training stopped before target
            if not running and current_step < CURRENT_TARGET:
                print(f"  WARNING: {lang} stopped at {current_step}! Restarting...")
                start_training(lang, size, CURRENT_TARGET)
                time.sleep(5)

            # Check if target reached
            if current_step >= CURRENT_TARGET:
                print(f"  {lang.upper()} REACHED TARGET {CURRENT_TARGET:,}!")
                send_notification(f"{lang.upper()} Complete", f"Reached {current_step:,} steps", persistent=True)

        print()
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\nMonitor stopped by user")
