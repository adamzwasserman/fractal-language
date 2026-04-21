#!/usr/bin/env python
"""
Auto-restart wrapper for Rust experiment training.
Handles memory limits and restarts automatically.
"""
import subprocess
import sys
import time
from datetime import datetime

MODE = sys.argv[1] if len(sys.argv) > 1 else "rust"  # "rust" or "rust_en"
MODEL_SIZE = sys.argv[2] if len(sys.argv) > 2 else "125M"
DAYTIME_MODE = sys.argv[3] if len(sys.argv) > 3 else "auto"

MAX_RESTARTS = 1000
RESTART_DELAY = 10  # seconds

print(f"=== Rust Experiment Auto-Restart Wrapper ===")
print(f"Mode: {MODE} | Model: {MODEL_SIZE} | Daytime mode: {DAYTIME_MODE}")
print(f"Max restarts: {MAX_RESTARTS}")
print()

restart_count = 0
while restart_count < MAX_RESTARTS:
    restart_count += 1
    print(f"[{datetime.now().isoformat()}] Starting session {restart_count}...")

    result = subprocess.run(
        ["uv", "run", "python", "scripts/train_rust.py", MODE, MODEL_SIZE, DAYTIME_MODE],
        cwd="/Users/adam/dev/fractal-language"
    )

    exit_code = result.returncode

    if exit_code == 0:
        print(f"\n=== Training completed successfully! ===")
        break
    elif exit_code == 42:
        print(f"\n=== Memory limit reached, restarting in {RESTART_DELAY}s... ===")
        time.sleep(RESTART_DELAY)
    elif exit_code == 43:
        print(f"\n=== Training interrupted, restarting in {RESTART_DELAY}s... ===")
        time.sleep(RESTART_DELAY)
    else:
        print(f"\n=== Training failed with exit code {exit_code} ===")
        print("Check logs for errors. Not restarting.")
        sys.exit(exit_code)

print(f"\nTotal sessions: {restart_count}")
