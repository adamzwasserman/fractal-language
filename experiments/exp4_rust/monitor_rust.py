#!/usr/bin/env python3
"""
Monitor for Rust Experiment 4: Side-by-side comparison
Shows loss, perplexity, probe accuracy by category for rust vs rust_en
"""
import subprocess
import time
import json
import math
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("/workspace/data/logs")
CHECKPOINT_DIR = Path("/workspace/data/checkpoints")

def get_gpu_stats():
    """Get GPU utilization and memory for both cards"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        gpus = {}
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                idx = int(parts[0])
                gpus[idx] = {
                    "util": int(parts[1]),
                    "mem_used": int(parts[2]),
                    "mem_total": int(parts[3])
                }
        return gpus
    except Exception:
        return {}

def get_latest_log_entry(mode):
    """Get latest training metrics from CSV log"""
    log_file = LOG_DIR / f"{mode}_125M_training.csv"
    if not log_file.exists():
        return None

    try:
        lines = log_file.read_text().strip().split("\n")
        if len(lines) < 2:
            return None

        for line in reversed(lines[1:]):
            parts = line.split(",")
            if len(parts) >= 2 and parts[0].isdigit():
                entry = {"step": int(parts[0]), "loss": float(parts[1])}
                if len(parts) >= 3 and parts[2]:
                    entry["probe_accuracy"] = float(parts[2])
                return entry
        return None
    except Exception:
        return None

def get_latest_checkpoint_meta(mode):
    """Get probe accuracy from latest checkpoint"""
    ckpt_dir = CHECKPOINT_DIR / mode / "125M"
    if not ckpt_dir.exists():
        return None

    checkpoints = list(ckpt_dir.glob("checkpoint_*.json"))
    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda x: int(x.stem.split("_")[1]))
    try:
        with open(latest) as f:
            return json.load(f)
    except Exception:
        return None

def get_probe_details(mode):
    """Get per-category probe results from latest probe run"""
    # Check for detailed probe results file
    results_dir = CHECKPOINT_DIR / mode / "125M"
    probe_files = list(results_dir.glob("probes_*.json"))
    if not probe_files:
        return None

    latest = max(probe_files, key=lambda x: int(x.stem.split("_")[1]))
    try:
        with open(latest) as f:
            return json.load(f)
    except Exception:
        return None

def bar(value, width=12):
    """Create a visual bar"""
    if value is None:
        return "?" * width
    filled = int(value * width)
    return "=" * filled + "-" * (width - filled)

def clear_screen():
    print("\033[2J\033[H", end="")

def main():
    COL = 35  # column width

    while True:
        clear_screen()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get data
        gpus = get_gpu_stats()
        rust_log = get_latest_log_entry("rust")
        rust_en_log = get_latest_log_entry("rust_en")
        rust_meta = get_latest_checkpoint_meta("rust")
        rust_en_meta = get_latest_checkpoint_meta("rust_en")

        rust_step = rust_log["step"] if rust_log else 0
        rust_en_step = rust_en_log["step"] if rust_en_log else 0
        rust_loss = rust_log["loss"] if rust_log else None
        rust_en_loss = rust_en_log["loss"] if rust_en_log else None
        rust_acc = rust_meta.get("probe_accuracy") if rust_meta else None
        rust_en_acc = rust_en_meta.get("probe_accuracy") if rust_en_meta else None

        # Perplexity = exp(loss)
        rust_ppx = math.exp(rust_loss) if rust_loss else None
        rust_en_ppx = math.exp(rust_en_loss) if rust_en_loss else None

        # Header
        print("=" * 75)
        print(f"  RUST EXPERIMENT 4: Synthetic Morphology        {now}")
        print("=" * 75)

        # Side-by-side headers
        left = "[GPU 0] RUST-ONLY"
        right = "[GPU 1] RUST+ENGLISH"
        print(f"  {left:<{COL}} | {right:<{COL}}")
        print("-" * 75)

        # GPU stats
        if 0 in gpus:
            left = f"GPU: {gpus[0]['util']:3d}%  Mem: {gpus[0]['mem_used']}/{gpus[0]['mem_total']}MB"
        else:
            left = "GPU: --"
        if 1 in gpus:
            right = f"GPU: {gpus[1]['util']:3d}%  Mem: {gpus[1]['mem_used']}/{gpus[1]['mem_total']}MB"
        else:
            right = "GPU: --"
        print(f"  {left:<{COL}} | {right:<{COL}}")

        # Step
        left = f"Step: {rust_step:,}"
        right = f"Step: {rust_en_step:,}"
        print(f"  {left:<{COL}} | {right:<{COL}}")

        # Loss
        left = f"Loss: {rust_loss:.4f}" if rust_loss else "Loss: --"
        right = f"Loss: {rust_en_loss:.4f}" if rust_en_loss else "Loss: --"
        print(f"  {left:<{COL}} | {right:<{COL}}")

        # Perplexity
        left = f"PPX:  {rust_ppx:.1f}" if rust_ppx else "PPX:  --"
        right = f"PPX:  {rust_en_ppx:.1f}" if rust_en_ppx else "PPX:  --"
        print(f"  {left:<{COL}} | {right:<{COL}}")

        print("-" * 75)

        # Probe accuracy header
        left = f"PROBE ACCURACY: {rust_acc*100:.1f}%" if rust_acc else "PROBE ACCURACY: --"
        right = f"PROBE ACCURACY: {rust_en_acc*100:.1f}%" if rust_en_acc else "PROBE ACCURACY: --"
        print(f"  {left:<{COL}} | {right:<{COL}}")

        # Per-category breakdown (from checkpoint meta or computed)
        # Categories: lifetime, ownership, type, borrow, mut, expr
        categories = ["lifetime", "ownership", "type", "borrow", "mut", "expr"]
        cat_labels = {
            "lifetime": "Lifetime ",
            "ownership": "Ownership",
            "type": "Type     ",
            "borrow": "Borrow   ",
            "mut": "Mutabilty",
            "expr": "Expr/Stmt",
        }

        # Get category data from meta if available
        rust_cats = rust_meta.get("probe_by_category", {}) if rust_meta else {}
        rust_en_cats = rust_en_meta.get("probe_by_category", {}) if rust_en_meta else {}

        for cat in categories:
            r_val = rust_cats.get(cat)
            re_val = rust_en_cats.get(cat)

            if r_val is not None:
                left = f"  {cat_labels[cat]}: {r_val*100:4.0f}% [{bar(r_val)}]"
            else:
                left = f"  {cat_labels[cat]}: --"

            if re_val is not None:
                right = f"  {cat_labels[cat]}: {re_val*100:4.0f}% [{bar(re_val)}]"
            else:
                right = f"  {cat_labels[cat]}: --"

            print(f"{left:<{COL+2}} | {right:<{COL}}")

        print("=" * 75)

        # Comparative section
        print("  COMPARISON")
        print("-" * 75)

        # PPX ratio
        if rust_ppx and rust_en_ppx:
            ppx_ratio = rust_en_ppx / rust_ppx
            print(f"  PPX Ratio (rust_en/rust): {ppx_ratio:.2f}x")

        # Loss comparison
        if rust_loss and rust_en_loss:
            loss_diff = rust_loss - rust_en_loss
            winner = "RUST" if loss_diff < 0 else "RUST+EN"
            print(f"  Loss Diff: {loss_diff:+.4f}  ({winner} better)")

        # Accuracy comparison
        if rust_acc is not None and rust_en_acc is not None:
            acc_diff = (rust_acc - rust_en_acc) * 100
            winner = "RUST" if acc_diff > 0 else "RUST+EN" if acc_diff < 0 else "TIE"
            print(f"  Accuracy Diff: {acc_diff:+.1f}%  ({winner} better)")

            # Log odds ratio
            if 0 < rust_acc < 1 and 0 < rust_en_acc < 1:
                rust_odds = rust_acc / (1 - rust_acc)
                rust_en_odds = rust_en_acc / (1 - rust_en_acc)
                log_or = math.log(rust_odds / rust_en_odds)
                print(f"  Log Odds Ratio: {log_or:+.3f}")

        print("-" * 75)

        # Hypothesis
        print("  HYPOTHESIS: Rust-only learns structure faster than Rust+EN")
        if rust_acc is not None and rust_en_acc is not None:
            diff = rust_acc - rust_en_acc
            if diff > 0.05:
                status = "SUPPORTED"
                icon = "+++"
            elif diff < -0.05:
                status = "FALSIFIED"
                icon = "---"
            else:
                status = "INCONCLUSIVE"
                icon = "???"
            print(f"  [{icon}] {status}")
        else:
            print(f"  [???] PENDING")

        print("=" * 75)
        print("  Refresh: 10s | Ctrl+C to exit")

        time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
