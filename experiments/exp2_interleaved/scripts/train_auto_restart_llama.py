#!/usr/bin/env python
"""
Auto-restarting training wrapper for LLaMA-style architecture
Continuously runs training, restarting when it stops due to memory limits
"""
import subprocess
import time
import sys
import os
from datetime import datetime
from pathlib import Path

def log_message(msg):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def run_training_session(lang, model_size, mode):
    """Run a single training session"""
    script_path = Path(__file__).parent / "train_resumable_llama.py"

    cmd = [
        sys.executable, str(script_path),
        lang, model_size, mode
    ]

    log_message(f"Starting LLAMA training: {' '.join(cmd[1:])}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None
        )
        return result.returncode, result.stdout, result.stderr

    except KeyboardInterrupt:
        log_message("Training interrupted by user")
        return -1, "", "Interrupted"
    except Exception as e:
        log_message(f"Training failed with exception: {e}")
        return -2, "", str(e)

def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    model_size = sys.argv[2] if len(sys.argv) > 2 else "125M"
    mode = sys.argv[3] if len(sys.argv) > 3 else "auto"

    log_message(f"Starting LLAMA auto-restart training: {lang} {model_size} {mode}")
    log_message("Press Ctrl+C to stop completely")

    session_count = 0
    total_steps = 0

    try:
        while True:
            session_count += 1
            log_message(f"=== LLAMA Training Session {session_count} ===")

            exit_code, stdout, stderr = run_training_session(lang, model_size, mode)

            # Parse output for step count
            lines = stdout.strip().split('\n')
            for line in reversed(lines):
                try:
                    if "Memory limit reached at step" in line:
                        step_part = line.split("step")[1].strip()
                        if step_part.isdigit():
                            total_steps = int(step_part)
                            break
                    elif "Stopping at step" in line:
                        step_part = line.split("step")[1].strip()
                        if step_part.isdigit():
                            total_steps = int(step_part)
                            break
                    elif "|" in line and "/200000" in line:
                        parts = line.split("|")
                        for part in parts:
                            if "/200000" in part:
                                step_part = part.split("/")[0].strip()
                                if step_part.isdigit():
                                    total_steps = int(step_part)
                                    break
                        if total_steps > 0:
                            break
                    elif "Starting from step" in line:
                        step_part = line.split("step")[1].strip()
                        if step_part.isdigit():
                            total_steps = int(step_part)
                            break
                except Exception as e:
                    pass

            log_message(f"Session {session_count} completed, exit code: {exit_code}")
            log_message(f"Total steps reached: {total_steps}")

            if exit_code == -1:
                log_message("Training stopped by user")
                break
            elif exit_code == 0:
                if total_steps >= 200000 or "reached target steps" in stdout:
                    log_message("Training reached target steps - COMPLETE!")
                    break
                else:
                    log_message("Session completed (likely memory limit), restarting...")
            else:
                log_message(f"Session ended with exit code {exit_code}")
                if stderr:
                    log_message(f"Error: {stderr[:500]}")
                log_message("Restarting after error...")

            restart_delay = 10
            log_message(f"Waiting {restart_delay} seconds before restart...")
            time.sleep(restart_delay)

            log_message(f"Progress: Session {session_count}, Steps: {total_steps}/200,000")

    except KeyboardInterrupt:
        log_message("Auto-restart stopped by user")
    except Exception as e:
        log_message(f"Auto-restart failed: {e}")

    log_message(f"Training sessions completed: {session_count}")
    log_message(f"Final step count: {total_steps}")
    log_message("LLAMA auto-restart training finished")

if __name__ == "__main__":
    main()
