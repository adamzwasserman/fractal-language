#!/usr/bin/env python
"""
Dual-language background training launcher.
Runs both English and French training as nice background processes with coordinated memory limits.

Usage:
    uv run python scripts/train_dual_background.py [architecture]

Arguments:
    architecture: gpt2 (default) or llama
"""
import subprocess
import sys
import os
import time
import signal
import psutil
from datetime import datetime
from pathlib import Path

# Each process gets 5GB max (10GB total combined)
MEMORY_LIMIT_PER_PROCESS = 5.0
TOTAL_MEMORY_LIMIT = 10.0

# Check interval for process health
CHECK_INTERVAL = 30  # seconds

processes = {}
shutdown_requested = False


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def signal_handler(signum, frame):
    global shutdown_requested
    log(f"Shutdown signal received ({signum})")
    shutdown_requested = True


def get_process_memory_gb(pid):
    """Get memory usage of a process in GB"""
    try:
        proc = psutil.Process(pid)
        return proc.memory_info().rss / (1024 ** 3)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def start_training(lang, architecture="gpt2"):
    """Start a training process for the given language"""
    script_dir = Path(__file__).parent

    if architecture == "llama":
        script = script_dir / "train_resumable_llama.py"
    else:
        script = script_dir / "train_resumable.py"

    # Use 'nice' to give lower priority and set memory mode to daytime (conservative)
    # We override the memory limit via environment variable
    env = os.environ.copy()
    env["FRACTAL_MEMORY_LIMIT"] = str(MEMORY_LIMIT_PER_PROCESS)

    cmd = [
        "nice", "-n", "10",  # Lower priority
        sys.executable, str(script),
        lang, "125M", "daytime"  # Always use daytime (5GB) mode for dual training
    ]

    log(f"Starting {lang.upper()} training: {' '.join(cmd[3:])}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    return proc


def check_total_memory():
    """Check combined memory usage of all training processes"""
    total = 0.0
    for lang, proc in processes.items():
        if proc and proc.poll() is None:
            total += get_process_memory_gb(proc.pid)
    return total


def monitor_output(proc, lang, lines_buffer, max_lines=20):
    """Non-blocking read of process output"""
    import select

    if proc.stdout is None:
        return

    # Non-blocking read using select
    while True:
        readable, _, _ = select.select([proc.stdout], [], [], 0)
        if not readable:
            break

        line = proc.stdout.readline()
        if not line:
            break

        line = line.strip()
        if line:
            # Keep recent lines for context
            lines_buffer.append(line)
            if len(lines_buffer) > max_lines:
                lines_buffer.pop(0)

            # Only print important messages
            if any(kw in line.lower() for kw in ['checkpoint', 'error', 'warning', 'step', 'loss']):
                log(f"[{lang.upper()}] {line}")


def main():
    global shutdown_requested

    # Parse architecture argument
    architecture = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    if architecture not in ("gpt2", "llama"):
        log(f"Unknown architecture: {architecture}. Use 'gpt2' or 'llama'")
        sys.exit(1)

    log(f"=== Dual Language Training Launcher ({architecture.upper()}) ===")
    log(f"Memory limit: {MEMORY_LIMIT_PER_PROCESS}GB per process, {TOTAL_MEMORY_LIMIT}GB total")
    log("Press Ctrl+C to stop")

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Output buffers for each language
    output_buffers = {"en": [], "fr": []}

    # Restart counters
    restart_counts = {"en": 0, "fr": 0}

    try:
        # Start both training processes
        for lang in ["en", "fr"]:
            processes[lang] = start_training(lang, architecture)
            time.sleep(2)  # Stagger starts slightly

        while not shutdown_requested:
            # Check process health
            for lang in ["en", "fr"]:
                proc = processes.get(lang)

                # Monitor output
                if proc and proc.poll() is None:
                    monitor_output(proc, lang, output_buffers[lang])

                # Check if process died
                if proc is None or proc.poll() is not None:
                    exit_code = proc.returncode if proc else -1

                    # Exit codes: 0=complete, 42=memory limit, 43=interrupted, other=error
                    if exit_code == 0:
                        log(f"{lang.upper()} training completed all 200k steps!")
                        # Check if the other one is also done
                        other = "fr" if lang == "en" else "en"
                        other_proc = processes.get(other)
                        if other_proc and other_proc.poll() is not None and other_proc.returncode == 0:
                            log("Both languages completed!")
                            return
                    elif exit_code in (42, 43):
                        # Memory limit or interrupted - normal restart
                        restart_counts[lang] += 1
                        reason = "memory limit" if exit_code == 42 else "interrupted"
                        log(f"{lang.upper()} hit {reason}, restarting... (restart #{restart_counts[lang]})")
                        time.sleep(5)
                        processes[lang] = start_training(lang, architecture)
                    else:
                        # Error - still restart but note it
                        restart_counts[lang] += 1
                        log(f"{lang.upper()} error (exit={exit_code}), restarting... (restart #{restart_counts[lang]})")
                        time.sleep(10)
                        processes[lang] = start_training(lang, architecture)

            # Check total memory
            total_mem = check_total_memory()
            if total_mem > TOTAL_MEMORY_LIMIT * 0.9:
                log(f"Warning: Total memory {total_mem:.1f}GB approaching limit")

            # Status update every 5 minutes
            if int(time.time()) % 300 == 0:
                en_mem = get_process_memory_gb(processes["en"].pid) if processes.get("en") and processes["en"].poll() is None else 0
                fr_mem = get_process_memory_gb(processes["fr"].pid) if processes.get("fr") and processes["fr"].poll() is None else 0
                log(f"Status: EN={en_mem:.1f}GB, FR={fr_mem:.1f}GB, Total={en_mem+fr_mem:.1f}GB")

            time.sleep(CHECK_INTERVAL)

    except Exception as e:
        log(f"Error in main loop: {e}")

    finally:
        # Graceful shutdown
        log("Shutting down training processes...")

        for lang, proc in processes.items():
            if proc and proc.poll() is None:
                log(f"Terminating {lang.upper()} training...")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log(f"Force killing {lang.upper()} training...")
                    proc.kill()

        log("=== Training stopped ===")
        log(f"Restart counts: EN={restart_counts['en']}, FR={restart_counts['fr']}")


if __name__ == "__main__":
    main()
