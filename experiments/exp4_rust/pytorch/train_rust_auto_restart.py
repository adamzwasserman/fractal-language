#!/usr/bin/env python3
"""
Auto-restarting training wrapper for Rust Experiment 4.
Continuously runs training, restarting when it stops due to memory or interrupts.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 train_rust_auto_restart.py rust
  CUDA_VISIBLE_DEVICES=1 python3 train_rust_auto_restart.py rust_en
"""
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path


def log_message(msg):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


def run_training_session(mode, model_size):
    """Run a single training session"""
    script_path = Path(__file__).parent / "train_rust_ddp.py"

    cmd = ["python3", str(script_path), mode, model_size]
    log_message(f"Starting training: {' '.join(cmd)}")

    try:
        # Run training - don't capture output so tqdm works
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        log_message("Training interrupted by user")
        return -1
    except Exception as e:
        log_message(f"Training failed with exception: {e}")
        return -2


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rust"
    model_size = sys.argv[2] if len(sys.argv) > 2 else "125M"

    log_message(f"=== Rust Experiment Auto-Restart: {mode} {model_size} ===")
    log_message("Press Ctrl+C twice quickly to stop completely")

    session_count = 0
    max_sessions = 1000  # Safety limit

    try:
        while session_count < max_sessions:
            session_count += 1
            log_message(f"=== Training Session {session_count} ===")

            exit_code = run_training_session(mode, model_size)

            log_message(f"Session {session_count} ended with exit code: {exit_code}")

            if exit_code == 0:
                log_message("Training COMPLETE - reached 200k steps!")
                break
            elif exit_code == -1:
                log_message("User interrupt - stopping")
                break
            elif exit_code == 42:
                log_message("Memory limit reached - restarting in 10s...")
                time.sleep(10)
            elif exit_code == 43:
                log_message("Training interrupted - restarting in 10s...")
                time.sleep(10)
            elif exit_code == 1:
                log_message("Error occurred - restarting in 30s...")
                time.sleep(30)
            else:
                log_message(f"Unknown exit code {exit_code} - restarting in 30s...")
                time.sleep(30)

    except KeyboardInterrupt:
        log_message("Auto-restart stopped by user")

    log_message(f"Total sessions: {session_count}")
    log_message("Auto-restart finished")


if __name__ == "__main__":
    main()
