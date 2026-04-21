#!/usr/bin/env python
"""
Plot training curves with forward projection based on velocity trends.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("/Volumes/Misc Backup/fractal/logs")
OUTPUT_DIR = Path("/Users/adam/dev/fractal-language/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

EMERGENCE_THRESHOLD = 150
TARGET_STEPS = 200_000


def load_data(lang: str, model_size: str = "125M") -> pd.DataFrame:
    """Load training CSV data."""
    csv_path = LOG_DIR / f"{lang}_{model_size}_training.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df = df.sort_values('step')

    # Use actual perplexity if available, otherwise estimate from loss
    # perplexity = exp(loss) for cross-entropy loss
    if 'loss' in df.columns:
        df['ppl_estimated'] = np.exp(df['loss'])
        # Prefer actual perplexity, fall back to estimated
        df['perplexity'] = df['perplexity'].fillna(df['ppl_estimated'])

    df = df.dropna(subset=['perplexity'])
    return df


def calculate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate perplexity velocity (change per 1k steps)."""
    df = df.copy()
    df['ppl_change'] = df['perplexity'].diff()
    df['step_change'] = df['step'].diff()
    df['velocity'] = df['ppl_change'] / (df['step_change'] / 1000)  # per 1k steps
    return df


def fit_decay_model(steps: np.ndarray, ppl: np.ndarray):
    """
    Fit an exponential decay model: ppl = a * exp(-b * step) + c
    Returns function for projection.
    """
    from scipy.optimize import curve_fit

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    try:
        # Initial guesses
        a0 = ppl[0] - ppl[-1]
        b0 = 1e-5
        c0 = ppl[-1] * 0.5

        popt, _ = curve_fit(
            exp_decay, steps, ppl,
            p0=[a0, b0, c0],
            bounds=([0, 0, 0], [np.inf, 1e-3, ppl[-1]]),
            maxfev=5000
        )
        return lambda x: exp_decay(x, *popt), popt
    except Exception as e:
        print(f"  Curve fit failed: {e}, using linear extrapolation")
        return None, None


def project_forward(df: pd.DataFrame, target_steps: int = TARGET_STEPS):
    """
    Project perplexity forward using exponential decay model.
    """
    steps = df['step'].values
    ppl = df['perplexity'].values

    # Fit decay model
    model_fn, params = fit_decay_model(steps, ppl)

    if model_fn is None:
        # Fallback: linear extrapolation from recent velocity
        recent_velocity = df['velocity'].iloc[-5:].mean()
        last_step = steps[-1]
        last_ppl = ppl[-1]

        future_steps = np.arange(last_step, target_steps + 1000, 1000)
        projected_ppl = last_ppl + recent_velocity * (future_steps - last_step) / 1000
        projected_ppl = np.maximum(projected_ppl, 10)  # Floor at 10
        return future_steps, projected_ppl, "linear"

    # Project using fitted model
    future_steps = np.arange(steps[-1], target_steps + 1000, 1000)
    projected_ppl = model_fn(future_steps)

    return future_steps, projected_ppl, "exponential"


def estimate_emergence_step(future_steps, projected_ppl, threshold=EMERGENCE_THRESHOLD):
    """Find step where perplexity crosses threshold."""
    below_threshold = projected_ppl < threshold
    if below_threshold.any():
        idx = np.argmax(below_threshold)
        return future_steps[idx]
    return None


def plot_curves():
    """Create training curves plot with projections."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'en': '#2563eb', 'fr': '#dc2626'}

    results = {}

    for lang in ['en', 'fr']:
        print(f"\nProcessing {lang.upper()}...")
        df = load_data(lang)

        if df.empty:
            print(f"  No data for {lang}")
            continue

        df = calculate_velocity(df)
        results[lang] = {
            'df': df,
            'current_step': df['step'].iloc[-1],
            'current_ppl': df['perplexity'].iloc[-1],
        }

        # Calculate average velocity trends
        if len(df) >= 5:
            early_vel = df['velocity'].iloc[1:len(df)//2].mean()
            late_vel = df['velocity'].iloc[len(df)//2:].mean()
            results[lang]['early_velocity'] = early_vel
            results[lang]['late_velocity'] = late_vel
            results[lang]['velocity_trend'] = 'accelerating' if late_vel < early_vel else 'decelerating'
            print(f"  Early velocity: {early_vel:.1f} ppl/1k steps")
            print(f"  Late velocity: {late_vel:.1f} ppl/1k steps")
            print(f"  Trend: {results[lang]['velocity_trend']}")

        # Project forward
        future_steps, projected_ppl, model_type = project_forward(df)
        results[lang]['projection'] = (future_steps, projected_ppl, model_type)

        # Estimate emergence
        emergence_step = estimate_emergence_step(future_steps, projected_ppl)
        results[lang]['emergence_step'] = emergence_step
        if emergence_step:
            print(f"  Projected emergence at step: {emergence_step:,}")
        else:
            print(f"  Emergence not projected within {TARGET_STEPS:,} steps")

    # Plot 1: Perplexity over steps (log scale)
    ax1 = axes[0, 0]
    for lang, data in results.items():
        df = data['df']
        ax1.plot(df['step'], df['perplexity'], 'o-',
                color=colors[lang], label=f'{lang.upper()} (actual)', linewidth=2, markersize=4)

        # Plot projection
        future_steps, projected_ppl, model_type = data['projection']
        ax1.plot(future_steps, projected_ppl, '--',
                color=colors[lang], alpha=0.5, label=f'{lang.upper()} (projected, {model_type})')

    ax1.axhline(y=EMERGENCE_THRESHOLD, color='green', linestyle=':', linewidth=2, label=f'Emergence threshold ({EMERGENCE_THRESHOLD})')
    ax1.set_yscale('log')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Perplexity (log scale)')
    ax1.set_title('Training Perplexity with Forward Projection')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, TARGET_STEPS)

    # Plot 2: Perplexity (linear scale, zoomed to current)
    ax2 = axes[0, 1]
    max_step = max(data['current_step'] for data in results.values()) * 1.5
    for lang, data in results.items():
        df = data['df']
        ax2.plot(df['step'], df['perplexity'], 'o-',
                color=colors[lang], label=f'{lang.upper()}', linewidth=2, markersize=4)

    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Current Training Progress (Linear Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_step)

    # Plot 3: Velocity over time
    ax3 = axes[1, 0]
    for lang, data in results.items():
        df = data['df']
        valid_vel = df.dropna(subset=['velocity'])
        ax3.plot(valid_vel['step'], valid_vel['velocity'], 'o-',
                color=colors[lang], label=f'{lang.upper()}', linewidth=2, markersize=4)

    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Velocity (ppl change per 1k steps)')
    ax3.set_title('Perplexity Velocity (negative = improving)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = "Training Summary\n" + "=" * 40 + "\n\n"

    for lang, data in results.items():
        summary_text += f"{lang.upper()}:\n"
        summary_text += f"  Current step: {data['current_step']:,}\n"
        summary_text += f"  Current perplexity: {data['current_ppl']:.1f}\n"
        if 'velocity_trend' in data:
            summary_text += f"  Velocity trend: {data['velocity_trend']}\n"
            summary_text += f"    Early: {data['early_velocity']:.1f} ppl/1k steps\n"
            summary_text += f"    Late: {data['late_velocity']:.1f} ppl/1k steps\n"
        if data['emergence_step']:
            summary_text += f"  Projected emergence: step {data['emergence_step']:,}\n"
        else:
            summary_text += f"  Projected emergence: >200k steps\n"
        summary_text += "\n"

    summary_text += f"\nEmergence threshold: ppl < {EMERGENCE_THRESHOLD}\n"
    summary_text += f"Target steps: {TARGET_STEPS:,}\n"
    summary_text += f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    output_path = OUTPUT_DIR / f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save a "latest" version
    latest_path = OUTPUT_DIR / "training_curves_latest.png"
    plt.savefig(latest_path, dpi=150, bbox_inches='tight')
    print(f"Latest plot: {latest_path}")

    plt.close()

    return results


if __name__ == "__main__":
    results = plot_curves()
