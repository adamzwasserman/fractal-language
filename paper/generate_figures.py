#!/usr/bin/env python3
"""
Generate all paper figures from experimental data.
Data source: /Volumes/fractal/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Data paths
DATA_ROOT = Path("/Volumes/fractal")
OUTPUT_DIR = Path("/Users/adam/dev/fractal-language/paper/figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Style settings
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11

COLORS = {
    'en': '#E74C3C',      # Red for English
    'fr': '#3498DB',      # Blue for French
    'rust': '#2ECC71',    # Green for Rust-only
    'rust_en': '#F39C12', # Orange for Rust+English
}


def load_125m_training():
    """Load 125M EN vs FR training data from regenerated checkpoint data."""
    dual_path = DATA_ROOT / "exp1_125M_en_vs_fr/logs/regenerated_training_dual.csv"

    df = pd.read_csv(dual_path)
    df = df.sort_values('step')

    # Filter out anomalies and NaN
    df = df.dropna(subset=['en_ppl', 'fr_ppl'])
    df = df[df['en_ppl'] > 10]
    df = df[df['fr_ppl'] > 5]

    # Create separate dataframes matching expected format
    en_df = pd.DataFrame({'step': df['step'], 'perplexity': df['en_ppl']})
    fr_df = pd.DataFrame({'step': df['step'], 'perplexity': df['fr_ppl']})

    return en_df, fr_df


def load_350m_training():
    """Load 350M EN vs FR training data from both checkpoint and training CSV sources."""
    # Source 1: Regenerated from checkpoints (steps 1k-66k)
    dual_path = DATA_ROOT / "exp1_350M_en_vs_fr/logs/regenerated_training_dual_350M.csv"
    df_ckpt = pd.read_csv(dual_path)

    # Source 2: Original training logs (steps 72.5k-200k)
    en_train_path = DATA_ROOT / "exp1_350M_en_vs_fr/logs/training_en_350M.csv"
    fr_train_path = DATA_ROOT / "exp1_350M_en_vs_fr/logs/training_fr_350M.csv"

    en_train = pd.read_csv(en_train_path)
    fr_train = pd.read_csv(fr_train_path)

    # Merge training logs
    en_train = en_train[['step', 'ppl']].rename(columns={'ppl': 'en_ppl'})
    fr_train = fr_train[['step', 'ppl']].rename(columns={'ppl': 'fr_ppl'})
    df_train = pd.merge(en_train, fr_train, on='step', how='outer')

    # Combine both sources
    df_ckpt_subset = df_ckpt[['step', 'en_ppl', 'fr_ppl']]
    df_combined = pd.concat([df_ckpt_subset, df_train], ignore_index=True)
    df_combined = df_combined.sort_values('step').drop_duplicates(subset='step', keep='last')

    # Filter out anomalies and NaN
    df_combined = df_combined.dropna(subset=['en_ppl', 'fr_ppl'])
    df_combined = df_combined[df_combined['en_ppl'] > 10]
    df_combined = df_combined[df_combined['fr_ppl'] > 10]

    # Create separate dataframes matching expected format
    en_df = pd.DataFrame({'step': df_combined['step'], 'perplexity': df_combined['en_ppl']})
    fr_df = pd.DataFrame({'step': df_combined['step'], 'perplexity': df_combined['fr_ppl']})

    return en_df, fr_df


def load_125m_grammar():
    """Load 125M grammar probe data."""
    en_path = DATA_ROOT / "exp1_125M_en_vs_fr/logs/grammar_probes_en.csv"
    fr_path = DATA_ROOT / "exp1_125M_en_vs_fr/logs/grammar_probes_fr.csv"

    en_df = pd.read_csv(en_path)
    fr_df = pd.read_csv(fr_path)

    return en_df, fr_df


def load_350m_grammar():
    """Load 350M grammar probe data."""
    en_path = DATA_ROOT / "exp1_350M_en_vs_fr/logs/grammar_probes_en.csv"
    fr_path = DATA_ROOT / "exp1_350M_en_vs_fr/logs/grammar_probes_fr.csv"

    en_df = pd.read_csv(en_path)
    fr_df = pd.read_csv(fr_path)

    return en_df, fr_df


def generate_ppx_trajectories():
    """Generate Figure 1: Perplexity Trajectories - 125M and 350M side by side."""
    print("Generating ppx_trajectories.png...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 125M EN vs FR - the dramatic result
    ax = axes[0]
    en_df, fr_df = load_125m_training()

    ax.plot(en_df['step'], en_df['perplexity'], color=COLORS['en'], label='English', linewidth=2, alpha=0.8)
    ax.plot(fr_df['step'], fr_df['perplexity'], color=COLORS['fr'], label='French', linewidth=2, alpha=0.8)

    ax.set_yscale('log')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Perplexity (log scale)', fontsize=12)
    ax.set_title('125M Parameters', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation for final values
    en_final = en_df.iloc[-1]
    fr_final = fr_df.iloc[-1]
    ax.annotate(f'EN: {en_final["perplexity"]:.0f}',
                xy=(en_final['step'], en_final['perplexity']),
                xytext=(10, 10), textcoords='offset points', fontsize=10, color=COLORS['en'])
    ax.annotate(f'FR: {fr_final["perplexity"]:.0f}',
                xy=(fr_final['step'], fr_final['perplexity']),
                xytext=(10, -15), textcoords='offset points', fontsize=10, color=COLORS['fr'])

    # 350M EN vs FR - convergence at larger scale
    ax = axes[1]
    en_df_350, fr_df_350 = load_350m_training()

    ax.plot(en_df_350['step'], en_df_350['perplexity'], color=COLORS['en'], label='English', linewidth=2, alpha=0.8)
    ax.plot(fr_df_350['step'], fr_df_350['perplexity'], color=COLORS['fr'], label='French', linewidth=2, alpha=0.8)

    ax.set_yscale('log')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Perplexity (log scale)', fontsize=12)
    ax.set_title('350M Parameters', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation for final values
    en_final_350 = en_df_350.iloc[-1]
    fr_final_350 = fr_df_350.iloc[-1]
    ax.annotate(f'EN: {en_final_350["perplexity"]:.0f}',
                xy=(en_final_350['step'], en_final_350['perplexity']),
                xytext=(10, 10), textcoords='offset points', fontsize=10, color=COLORS['en'])
    ax.annotate(f'FR: {fr_final_350["perplexity"]:.0f}',
                xy=(fr_final_350['step'], fr_final_350['perplexity']),
                xytext=(10, -15), textcoords='offset points', fontsize=10, color=COLORS['fr'])

    fig.suptitle('Perplexity Trajectories: English vs French on Matched C4 Corpora', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppx_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved ppx_trajectories.png")


def generate_accuracy_trajectories():
    """Generate Figure 2: Grammar Accuracy - 125M and 350M side by side."""
    print("Generating accuracy_trajectories.png...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 125M Grammar Accuracy
    ax = axes[0]
    en_df, fr_df = load_125m_grammar()

    ax.plot(en_df['step'], en_df['accuracy'], color=COLORS['en'], marker='o', markersize=4,
            label='English', linewidth=1.5, alpha=0.8)
    ax.plot(fr_df['step'], fr_df['accuracy'], color=COLORS['fr'], marker='o', markersize=4,
            label='French', linewidth=1.5, alpha=0.8)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100%')
    ax.axhline(y=40, color='gray', linestyle='--', alpha=0.5, label='Chance (40%)')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Grammar Accuracy (%)', fontsize=12)
    ax.set_title('125M: Grammar Accuracy (EN vs FR)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 350M Grammar Accuracy
    ax = axes[1]
    en_df_350, fr_df_350 = load_350m_grammar()

    ax.plot(en_df_350['step'], en_df_350['accuracy'], color=COLORS['en'], marker='o', markersize=3,
            label='English', linewidth=1.5, alpha=0.8)
    ax.plot(fr_df_350['step'], fr_df_350['accuracy'], color=COLORS['fr'], marker='o', markersize=3,
            label='French', linewidth=1.5, alpha=0.8)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100%')
    ax.axhline(y=40, color='gray', linestyle='--', alpha=0.5, label='Chance (40%)')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Grammar Accuracy (%)', fontsize=12)
    ax.set_title('350M: Grammar Accuracy (EN vs FR)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Accuracy Trajectories: Impact of Language on Structural Discrimination', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved accuracy_trajectories.png")


def load_interleaved_data():
    """Load interleaved EN/FR experiment data."""
    base_path = Path("/Volumes/fractal/exp2_125M_enfr_interleaved 2/en_training/logs")

    training = pd.read_csv(base_path / "training_enfr.csv")
    grammar_en = pd.read_csv(base_path / "grammar_probes_enfr_en.csv")
    grammar_fr = pd.read_csv(base_path / "grammar_probes_enfr_fr.csv")

    return training, grammar_en, grammar_fr


def generate_interleaved():
    """Generate Figure 3: Interleaved EN/FR experiment - perplexity and accuracy."""
    print("Generating interleaved.png...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    training, grammar_en, grammar_fr = load_interleaved_data()

    # Left panel: Perplexity trajectory
    ax = axes[0]
    ax.plot(training['step'], training['perplexity'], color='purple', linewidth=1.5, alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Perplexity (log scale)', fontsize=12)
    ax.set_title('Interleaved EN/FR: Perplexity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add final value annotation
    final = training.iloc[-1]
    ax.annotate(f'Final: {final["perplexity"]:.0f}',
                xy=(final['step'], final['perplexity']),
                xytext=(10, 10), textcoords='offset points', fontsize=10, color='purple')

    # Right panel: Grammar accuracy for both EN and FR probes
    ax = axes[1]
    ax.plot(grammar_en['step'], grammar_en['accuracy'], color=COLORS['en'], marker='o', markersize=3,
            label='EN Probes', linewidth=1.5, alpha=0.8)
    ax.plot(grammar_fr['step'], grammar_fr['accuracy'], color=COLORS['fr'], marker='o', markersize=3,
            label='FR Probes', linewidth=1.5, alpha=0.8)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100%')
    ax.axhline(y=40, color='gray', linestyle='--', alpha=0.5, label='Chance (40%)')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Grammar Accuracy (%)', fontsize=12)
    ax.set_title('Interleaved EN/FR: Grammar Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 3: Interleaved English/French Training (125M)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interleaved.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved interleaved.png")


def load_finetuning_data():
    """Load fine-tuning training data from trainer_state.json files."""
    import json
    base_path = Path("/Volumes/fractal/exp3_fr_chatbot/models")

    sft_data = {}
    dpo_data = {}

    # SFT models
    for size in [500, 1000, 3000]:
        pattern = base_path / f"sft_{size}" / "checkpoint-*" / "trainer_state.json"
        files = list(base_path.glob(f"sft_{size}/checkpoint-*/trainer_state.json"))
        if files:
            with open(files[-1]) as f:  # Get the last checkpoint
                data = json.load(f)
                sft_data[size] = [(h['step'], h['loss']) for h in data['log_history'] if 'loss' in h]

    # DPO models
    for size in [500, 1000, 3000, 5000]:
        files = list(base_path.glob(f"dpo_{size}/checkpoint-*/trainer_state.json"))
        if files:
            with open(files[-1]) as f:
                data = json.load(f)
                dpo_data[size] = [(h['step'], h['loss']) for h in data['log_history'] if 'loss' in h]

    return sft_data, dpo_data


def generate_finetuning():
    """Generate Figure 5: Fine-tuning efficiency from fr_chatbot experiment."""
    print("Generating finetuning.png...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sft_data, dpo_data = load_finetuning_data()

    # Left panel: SFT training loss
    ax = axes[0]
    colors_sft = {'500': '#9B59B6', '1000': '#3498DB', '3000': '#2ECC71'}
    for size, data in sorted(sft_data.items()):
        if data:
            steps, losses = zip(*data)
            ax.plot(steps, losses, label=f'SFT {size}', linewidth=2, alpha=0.8, color=colors_sft.get(str(size), 'gray'))

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('SFT Training Loss by Data Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right panel: DPO training loss
    ax = axes[1]
    colors_dpo = {'500': '#9B59B6', '1000': '#3498DB', '3000': '#2ECC71', '5000': '#E74C3C'}
    for size, data in sorted(dpo_data.items()):
        if data:
            steps, losses = zip(*data)
            ax.plot(steps, losses, label=f'DPO {size}', linewidth=2, alpha=0.8, color=colors_dpo.get(str(size), 'gray'))

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('DPO Training Loss by Data Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Fine-tuning Efficiency: French 125M Model (SFT + DPO)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "finetuning.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved finetuning.png")


def generate_emergence_timing():
    """Generate emergence timing chart - tokens to grammatical competence."""
    print("Generating emergence_timing.png...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Data: tokens to emergence (≥90% accuracy) or max tokens if not emerged
    # 125M: batch=32 (0-201k), then batch=10 (201k-400k), seq=512
    # 350M: batch=8, seq=512 → 4,096 tokens/step
    # French 125M: emerged at step 12k → 12k × 16,384 = 197M tokens
    # English 125M: never emerged through 400k steps → 3.3B + 1.0B = 4.3B tokens
    # French 350M: never emerged (max 70%), trained 200k steps → 200k × 4,096 = 819M tokens
    # English 350M: never emerged through 200k steps → 200k × 4,096 = 819M tokens

    experiments = ['French 125M', 'English 125M', 'French 350M', 'English 350M']
    tokens_millions = [197, 4300, 819, 819]  # in millions
    emerged = [True, False, False, False]

    colors = [COLORS['fr'] if 'French' in e else COLORS['en'] for e in experiments]
    hatches = ['' if em else '///' for em in emerged]

    bars = ax.barh(experiments, tokens_millions, color=colors, alpha=0.7)

    # Add hatching for non-emerged
    for bar, hatch, em in zip(bars, hatches, emerged):
        bar.set_hatch(hatch)
        if not em:
            bar.set_edgecolor('darkred')
            bar.set_linewidth(1.5)

    # Add annotations
    for i, (tokens, em) in enumerate(zip(tokens_millions, emerged)):
        label = f'{tokens:,}M tokens' if em else f'>{tokens:,}M (not emerged)'
        ax.annotate(label, xy=(tokens, i), xytext=(5, 0),
                    textcoords='offset points', va='center', fontsize=10)

    ax.set_xlabel('Tokens (millions)', fontsize=12)
    ax.set_title('Emergence Timing: Tokens to Grammatical Competence (≥90% accuracy)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(tokens_millions) * 1.15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['fr'], alpha=0.7, label='Emerged (≥90% accuracy)'),
        Patch(facecolor=COLORS['en'], alpha=0.7, hatch='///', edgecolor='darkred', label='Not emerged')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "emergence_timing.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved emergence_timing.png")


def generate_multifactor():
    """Generate multi-factor analysis figure (Language × Scale) showing orthogonality of PPL and accuracy."""
    print("Generating multifactor_analysis.png...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Multi-Factor Analysis: Language × Scale\nPerplexity and Accuracy are Orthogonal', fontsize=16, fontweight='bold')

    # Data based on verified experimental results:
    # - French 125M: PPL ~27, accuracy 100%, emerged at 197M tokens
    # - English 125M: PPL ~777 (extended to 4.3B tokens), accuracy 40%
    # - French 350M: PPL ~69, accuracy 70%, never emerged (819M tokens)
    # - English 350M: PPL ~84, accuracy 40%, never emerged (819M tokens)

    width = 0.35

    # Top-left: Perplexity by Language × Scale
    ax1 = axes[0, 0]
    x = np.arange(2)
    ax1.bar(x - width/2, [27, 69], width, color=COLORS['fr'], label='French')
    ax1.bar(x + width/2, [777, 84], width, color=COLORS['en'], label='English')

    ax1.set_ylabel('Perplexity')
    ax1.set_xlabel('Model Scale')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['125M', '350M'])
    ax1.set_title('Final Perplexity by Language × Scale')
    ax1.legend(loc='upper right', fontsize=9)
    # Add value labels
    ax1.annotate('27', (x[0] - width/2, 27), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.annotate('777', (x[0] + width/2, 777), ha='center', va='bottom', fontsize=10)
    ax1.annotate('69', (x[1] - width/2, 69), ha='center', va='bottom', fontsize=10)
    ax1.annotate('84', (x[1] + width/2, 84), ha='center', va='bottom', fontsize=10)

    # Top-middle: Grammar Accuracy by Language × Scale
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, [100, 70], width, color=COLORS['fr'], label='French')
    ax2.bar(x + width/2, [40, 40], width, color=COLORS['en'], label='English')
    ax2.axhline(y=40, color='gray', linestyle='--', alpha=0.5, label='Chance (40%)')
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Emergence (90%)')

    ax2.set_ylabel('Grammar Accuracy (%)')
    ax2.set_xlabel('Model Scale')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['125M', '350M'])
    ax2.set_title('Final Grammar Accuracy by Language × Scale')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(0, 110)
    # Add value labels
    ax2.annotate('100%', (x[0] - width/2, 100), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.annotate('40%', (x[0] + width/2, 40), ha='center', va='bottom', fontsize=10)
    ax2.annotate('70%', (x[1] - width/2, 70), ha='center', va='bottom', fontsize=10)
    ax2.annotate('40%', (x[1] + width/2, 40), ha='center', va='bottom', fontsize=10)

    # Top-right: PPL vs Accuracy scatter (ORTHOGONALITY)
    ax3 = axes[0, 2]
    # Data points: (PPL, Accuracy, label, color)
    points = [
        (27, 100, 'FR 125M', COLORS['fr']),
        (777, 40, 'EN 125M', COLORS['en']),
        (69, 70, 'FR 350M', COLORS['fr']),
        (84, 40, 'EN 350M', COLORS['en']),
    ]
    for ppx, acc, label, color in points:
        marker = 'o' if '125M' in label else 's'
        ax3.scatter(ppx, acc, c=color, s=200, marker=marker, edgecolors='black', linewidths=2)
        ax3.annotate(label, (ppx, acc), xytext=(10, 5), textcoords='offset points', fontsize=10)

    ax3.axhline(y=40, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=90, color='green', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Perplexity')
    ax3.set_ylabel('Grammar Accuracy (%)')
    ax3.set_title('ORTHOGONALITY: PPL vs Accuracy')
    ax3.set_ylim(0, 110)
    ax3.set_xscale('log')

    # Add annotation box for key insight
    ax3.annotate('Similar PPL\n(69 vs 84)\nbut 30% accuracy gap!',
                 xy=(75, 55), xytext=(200, 70),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Bottom-left: Token comparison
    ax4 = axes[1, 0]
    conditions = ['FR 125M', 'EN 125M', 'FR 350M', 'EN 350M']
    tokens = [197, 4300, 819, 819]
    colors_list = [COLORS['fr'], COLORS['en'], COLORS['fr'], COLORS['en']]
    emerged = [True, False, False, False]

    bars = ax4.bar(conditions, tokens, color=colors_list)
    for bar, em in zip(bars, emerged):
        if not em:
            bar.set_hatch('///')
            bar.set_alpha(0.7)

    ax4.set_ylabel('Tokens (millions)')
    ax4.set_title('Training Tokens by Condition')
    for i, (cond, tok, em) in enumerate(zip(conditions, tokens, emerged)):
        status = '✓' if em else '✗'
        ax4.annotate(f'{tok}M {status}', (i, tok), ha='center', va='bottom', fontsize=9)

    # Bottom-middle: Orthogonality table
    ax5 = axes[1, 1]
    ax5.axis('off')

    table_text = """
    ┌─────────────────────────────────────────────────┐
    │     ORTHOGONALITY OF PPL AND ACCURACY           │
    ├─────────────┬────────┬──────────┬───────────────┤
    │ Condition   │  PPL   │ Accuracy │ Status        │
    ├─────────────┼────────┼──────────┼───────────────┤
    │ FR 125M     │   27   │   100%   │ Emerged       │
    │ EN 125M     │  777   │    40%   │ Never emerged │
    │ FR 350M     │   69   │    70%   │ Never emerged │
    │ EN 350M     │   84   │    40%   │ Never emerged │
    ├─────────────┴────────┴──────────┴───────────────┤
    │ KEY FINDING: At 350M scale, EN and FR have      │
    │ similar PPL (84 vs 69) but 30% accuracy gap.    │
    │ PPL improvement ≠ grammar acquisition!          │
    └─────────────────────────────────────────────────┘
    """

    ax5.text(0.5, 0.5, table_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Bottom-right: Summary
    ax6 = axes[1, 2]
    ax6.axis('off')

    summary_text = """
KEY INSIGHTS
════════════════════════════

1. LANGUAGE > SCALE
   French 125M outperforms
   English at any scale tested

2. PPL ⊥ ACCURACY
   Models can improve PPL
   without learning grammar
   (EN 125M: PPL 1340→777,
   grammar stuck at 40%)

3. SCALE CONFOUNDED
   350M got fewer tokens
   (819M vs 3.3B-4.3B)
   Extended training planned

4. MORPHOLOGY IS KEY
   Explicit marking enables
   efficient grammar learning
"""

    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multifactor_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved multifactor_analysis.png")


def generate_all():
    """Generate all figures."""
    print("\n" + "="*60)
    print("Generating Paper Figures")
    print("="*60)
    print(f"Data source: {DATA_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    generate_ppx_trajectories()
    generate_accuracy_trajectories()
    generate_emergence_timing()
    generate_finetuning()
    generate_interleaved()
    generate_multifactor()

    print()
    print("="*60)
    print("Done! Generated figures:")
    for f in OUTPUT_DIR.glob("*.png"):
        print(f"  {f.name}")
    print("="*60)


if __name__ == "__main__":
    generate_all()
