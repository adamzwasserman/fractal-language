#!/usr/bin/env python3
"""
Dual-language training launcher with STRICT memory control.

Combined memory limit: 10GB (4GB per process, 2GB buffer)
Monitors every 5 seconds and kills processes preemptively.
"""

import subprocess
import time
import os
import sys
import signal
from pathlib import Path

# STRICT CONFIGURATION
MEMORY_PER_PROCESS = 4.0      # GB per process
TOTAL_MEMORY_LIMIT = 10.0     # GB combined hard limit
MEMORY_BUFFER = 2.0           # GB buffer for safety
CHECK_INTERVAL = 5            # seconds - frequent monitoring
MODEL_SIZE = "125M"

# Paths
SCRIPT_DIR = Path(__file__).parent
TRAINING_SCRIPT = SCRIPT_DIR / "training_with_eval.py"
BASE_PATH = Path("/Volumes/Misc Backup/fractal")

def get_process_memory_gb(pid):
    """Get memory usage of a process in GB using ps."""
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            rss_kb = int(result.stdout.strip())
            return rss_kb / (1024 * 1024)
    except:
        pass
    return 0.0

def get_all_child_memory(parent_pid):
    """Get total memory of parent and all children."""
    try:
        # Get all child PIDs
        result = subprocess.run(
            ["pgrep", "-P", str(parent_pid)],
            capture_output=True, text=True, timeout=5
        )
        pids = [parent_pid]
        if result.returncode == 0:
            pids.extend([int(p) for p in result.stdout.strip().split('\n') if p])

        total = sum(get_process_memory_gb(pid) for pid in pids)
        return total
    except:
        return get_process_memory_gb(parent_pid)

def get_checkpoint_step(lang):
    """Get the latest checkpoint step for a language."""
    ckpt_dir = BASE_PATH / "checkpoints" / lang / MODEL_SIZE
    if not ckpt_dir.exists():
        return 0

    checkpoints = list(ckpt_dir.glob("checkpoint_*.json"))
    if not checkpoints:
        return 0

    steps = []
    for cp in checkpoints:
        try:
            step = int(cp.stem.split("_")[1])
            steps.append(step)
        except:
            pass

    return max(steps) if steps else 0

def launch_training(lang):
    """Launch training process with nice priority and strict memory limit."""
    env = os.environ.copy()
    env["FRACTAL_MEMORY_LIMIT"] = str(MEMORY_PER_PROCESS)

    cmd = [
        "nice", "-n", "15",  # Lower priority
        sys.executable, str(TRAINING_SCRIPT), lang, MODEL_SIZE
    ]

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    return process

def main():
    print(f"=== Dual Training with STRICT Memory Control ===")
    print(f"Per-process limit: {MEMORY_PER_PROCESS}GB")
    print(f"Combined limit: {TOTAL_MEMORY_LIMIT}GB (buffer: {MEMORY_BUFFER}GB)")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print()

    processes = {"en": None, "fr": None}
    log_files = {}

    log_dir = BASE_PATH / "logs"
    log_dir.mkdir(exist_ok=True)

    shutdown_requested = False
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        print("\nShutdown requested...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while not shutdown_requested:
            # Check/launch each language
            for lang in ["en", "fr"]:
                step = get_checkpoint_step(lang)

                if step >= 200000:
                    if processes[lang] != "complete":
                        print(f"[{lang.upper()}] Training complete (step {step})")
                    processes[lang] = "complete"
                    continue

                # Check if process needs (re)starting
                if processes[lang] is None or (
                    processes[lang] != "complete" and
                    processes[lang].poll() is not None
                ):
                    if processes[lang] not in [None, "complete"]:
                        exit_code = processes[lang].returncode
                        if exit_code == 0:
                            print(f"[{lang.upper()}] Complete!")
                            processes[lang] = "complete"
                            continue
                        elif exit_code == 42:
                            print(f"[{lang.upper()}] Memory limit - restarting...")
                        else:
                            print(f"[{lang.upper()}] Exit {exit_code} - restarting...")

                    # Close old log file
                    if lang in log_files:
                        log_files[lang].close()

                    log_file = log_dir / f"{lang}_{MODEL_SIZE}_eval.log"
                    log_files[lang] = open(log_file, "a")

                    print(f"[{lang.upper()}] Starting from step {step}...")
                    processes[lang] = launch_training(lang)

            # Check if all complete
            if all(p == "complete" for p in processes.values()):
                print("\n=== All training complete! ===")
                break

            # STRICT memory monitoring
            memories = {}
            for lang in ["en", "fr"]:
                if processes[lang] not in [None, "complete"]:
                    mem = get_all_child_memory(processes[lang].pid)
                    memories[lang] = mem

            total_memory = sum(memories.values())

            # PREEMPTIVE action if approaching limit
            effective_limit = TOTAL_MEMORY_LIMIT - MEMORY_BUFFER
            if total_memory > effective_limit:
                print(f"\n!!! MEMORY WARNING: {total_memory:.2f}GB > {effective_limit:.1f}GB !!!")

                # Kill the larger process
                if memories:
                    largest = max(memories.items(), key=lambda x: x[1])
                    lang_to_kill = largest[0]
                    print(f"Killing {lang_to_kill.upper()} ({largest[1]:.2f}GB)")

                    if processes[lang_to_kill] not in [None, "complete"]:
                        processes[lang_to_kill].terminate()
                        try:
                            processes[lang_to_kill].wait(timeout=10)
                        except:
                            processes[lang_to_kill].kill()
                        processes[lang_to_kill] = None

                # Wait before restarting
                time.sleep(5)
                continue

            # Also check individual process limits
            for lang, mem in memories.items():
                if mem > MEMORY_PER_PROCESS * 1.1:  # 10% over individual limit
                    print(f"[{lang.upper()}] Over limit: {mem:.2f}GB > {MEMORY_PER_PROCESS}GB - killing")
                    processes[lang].terminate()
                    processes[lang] = None

            # Read and log output
            for lang in ["en", "fr"]:
                if processes[lang] not in [None, "complete"] and lang in log_files:
                    try:
                        import select
                        while select.select([processes[lang].stdout], [], [], 0)[0]:
                            line = processes[lang].stdout.readline()
                            if not line:
                                break
                            log_files[lang].write(line)
                            log_files[lang].flush()
                            # Print important messages
                            if any(x in line for x in ["Smoke test:", "Full eval:", "INTERESTING", "WARNING", "LIMIT", "Checkpoint saved"]):
                                print(f"[{lang.upper()}] {line.strip()}")
                    except:
                        pass

            # Status update
            status_parts = []
            for lang in ["en", "fr"]:
                if processes[lang] == "complete":
                    status_parts.append(f"{lang.upper()}:done")
                elif processes[lang] is not None:
                    step = get_checkpoint_step(lang)
                    mem = memories.get(lang, 0)
                    status_parts.append(f"{lang.upper()}:{step}@{mem:.1f}GB")

            print(f"  [{time.strftime('%H:%M:%S')}] {' | '.join(status_parts)} | Total: {total_memory:.2f}GB/{TOTAL_MEMORY_LIMIT}GB")

            time.sleep(CHECK_INTERVAL)

    finally:
        print("\nShutting down...")
        for lang in ["en", "fr"]:
            if processes[lang] not in [None, "complete"]:
                processes[lang].terminate()
                try:
                    processes[lang].wait(timeout=10)
                except:
                    processes[lang].kill()
            if lang in log_files:
                log_files[lang].close()
        print("Done.")

if __name__ == "__main__":
    main()
