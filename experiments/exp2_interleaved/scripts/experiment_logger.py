#!/usr/bin/env python
"""
Comprehensive experiment logging for the Language-Only Hypothesis research.

Logs all training events in structured JSON Lines format for reproducibility
and analysis. Each log entry includes timestamp, event type, and relevant data.

Log files are stored at: /Volumes/Misc Backup/fractal/logs/experiments/
"""
import json
import os
import psutil
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import subprocess

# Base log directory
LOG_BASE = Path("/Volumes/Misc Backup/fractal/logs/experiments")
LOG_BASE.mkdir(parents=True, exist_ok=True)


class ExperimentLogger:
    """Structured logger for training experiments."""

    def __init__(self, experiment_name: str, architecture: str, lang: str, model_size: str):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Unique name for this experiment run
            architecture: 'gpt2' or 'llama'
            lang: 'en' or 'fr'
            model_size: '125M' or '350M'
        """
        self.experiment_name = experiment_name
        self.architecture = architecture
        self.lang = lang
        self.model_size = model_size

        # Create experiment-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOG_BASE / f"{lang}_{model_size}_{architecture}_{timestamp}.jsonl"
        self.summary_file = LOG_BASE / f"{lang}_{model_size}_{architecture}_summary.json"

        # Track experiment state
        self.start_time = datetime.now()
        self.total_steps = 0
        self.total_tokens = 0
        self.sessions = 0
        self.checkpoints_saved = 0

        # Log experiment start
        self._log_event("experiment_start", {
            "experiment_name": experiment_name,
            "architecture": architecture,
            "lang": lang,
            "model_size": model_size,
            "system_info": self._get_system_info(),
            "git_commit": self._get_git_commit(),
        })

    def _get_system_info(self) -> dict:
        """Collect system information for reproducibility."""
        try:
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "hostname": platform.node(),
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except:
            pass
        return None

    def _get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        process = psutil.Process()
        vm = psutil.virtual_memory()
        return {
            "process_memory_gb": round(process.memory_info().rss / (1024**3), 3),
            "system_memory_percent": vm.percent,
            "system_available_gb": round(vm.available / (1024**3), 2),
        }

    def _log_event(self, event_type: str, data: dict):
        """Write a structured log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "experiment": {
                "name": self.experiment_name,
                "architecture": self.architecture,
                "lang": self.lang,
                "model_size": self.model_size,
            },
            "data": data,
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_config(self, config: dict):
        """Log training configuration."""
        self._log_event("config", {
            "training_config": config,
            "memory": self._get_memory_stats(),
        })

    def log_session_start(self, session_num: int, start_step: int):
        """Log training session start."""
        self.sessions = session_num
        self._log_event("session_start", {
            "session": session_num,
            "start_step": start_step,
            "memory": self._get_memory_stats(),
        })

    def log_session_end(self, session_num: int, end_step: int, reason: str):
        """Log training session end."""
        self._log_event("session_end", {
            "session": session_num,
            "end_step": end_step,
            "reason": reason,  # 'memory_limit', 'completed', 'error', 'interrupted'
            "memory": self._get_memory_stats(),
        })

    def log_training_step(self, step: int, loss: float, learning_rate: float = None,
                          tokens_processed: int = None, batch_time_ms: float = None):
        """Log individual training step (call periodically, not every step)."""
        self.total_steps = step
        if tokens_processed:
            self.total_tokens = tokens_processed

        data = {
            "step": step,
            "loss": round(loss, 6),
            "memory": self._get_memory_stats(),
        }
        if learning_rate is not None:
            data["learning_rate"] = learning_rate
        if tokens_processed is not None:
            data["tokens_processed"] = tokens_processed
        if batch_time_ms is not None:
            data["batch_time_ms"] = round(batch_time_ms, 2)

        self._log_event("training_step", data)

    def log_checkpoint(self, step: int, loss: float, checkpoint_path: str,
                       save_time_seconds: float = None):
        """Log checkpoint save."""
        self.checkpoints_saved += 1
        data = {
            "step": step,
            "loss": round(loss, 6),
            "checkpoint_path": checkpoint_path,
            "checkpoint_number": self.checkpoints_saved,
        }
        if save_time_seconds is not None:
            data["save_time_seconds"] = round(save_time_seconds, 2)

        self._log_event("checkpoint_saved", data)

    def log_memory_warning(self, step: int, memory_gb: float, limit_gb: float):
        """Log memory warning before hitting limit."""
        self._log_event("memory_warning", {
            "step": step,
            "process_memory_gb": round(memory_gb, 3),
            "memory_limit_gb": limit_gb,
            "memory": self._get_memory_stats(),
        })

    def log_evaluation(self, step: int, eval_type: str, metrics: dict):
        """Log evaluation results (perplexity, capability probes, etc.)."""
        self._log_event("evaluation", {
            "step": step,
            "eval_type": eval_type,
            "metrics": metrics,
        })

    def log_error(self, step: int, error_type: str, error_message: str,
                  stack_trace: str = None):
        """Log errors during training."""
        data = {
            "step": step,
            "error_type": error_type,
            "error_message": error_message,
            "memory": self._get_memory_stats(),
        }
        if stack_trace:
            data["stack_trace"] = stack_trace

        self._log_event("error", data)

    def log_experiment_end(self, reason: str, final_step: int, final_loss: float = None):
        """Log experiment completion and write summary."""
        elapsed = datetime.now() - self.start_time

        self._log_event("experiment_end", {
            "reason": reason,
            "final_step": final_step,
            "final_loss": round(final_loss, 6) if final_loss else None,
            "total_sessions": self.sessions,
            "total_checkpoints": self.checkpoints_saved,
            "elapsed_seconds": elapsed.total_seconds(),
            "elapsed_human": str(elapsed),
        })

        # Write summary file
        summary = {
            "experiment_name": self.experiment_name,
            "architecture": self.architecture,
            "lang": self.lang,
            "model_size": self.model_size,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "final_step": final_step,
            "final_loss": final_loss,
            "total_sessions": self.sessions,
            "total_checkpoints": self.checkpoints_saved,
            "reason": reason,
            "log_file": str(self.log_file),
        }

        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


def get_active_experiments() -> list:
    """List all experiment log files."""
    return sorted(LOG_BASE.glob("*.jsonl"))


def load_experiment_log(log_file: Path) -> list:
    """Load all events from an experiment log."""
    events = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def get_experiment_summary(lang: str, model_size: str, architecture: str) -> Optional[dict]:
    """Get the latest summary for an experiment configuration."""
    summary_file = LOG_BASE / f"{lang}_{model_size}_{architecture}_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None


# Quick status check function
def print_all_experiments_status():
    """Print status of all experiments."""
    print("\n" + "="*70)
    print("EXPERIMENT STATUS")
    print("="*70)

    for arch in ['gpt2', 'llama']:
        for lang in ['en', 'fr']:
            for size in ['125M', '350M']:
                summary = get_experiment_summary(lang, size, arch)
                if summary:
                    print(f"\n{lang.upper()} {size} {arch.upper()}:")
                    print(f"  Steps: {summary.get('final_step', 'N/A'):,}")
                    print(f"  Loss: {summary.get('final_loss', 'N/A')}")
                    print(f"  Sessions: {summary.get('total_sessions', 'N/A')}")
                    print(f"  Status: {summary.get('reason', 'N/A')}")
                else:
                    print(f"\n{lang.upper()} {size} {arch.upper()}: No data")

    print("\n" + "="*70)


if __name__ == "__main__":
    print_all_experiments_status()
