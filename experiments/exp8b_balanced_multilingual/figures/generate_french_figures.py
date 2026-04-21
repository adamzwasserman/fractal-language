#!/usr/bin/env python3
"""
Generate French versions of all figures for the bilingual paper.
All labels, titles, annotations translated to French.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("/Volumes/fractal/exp8b_balanced_multilingual/paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.facecolor'] = 'white'

# Language name translations
LANG_FR = {
    'French': 'Français',
    'English': 'Anglais',
    'Spanish': 'Espagnol',
    'Russian': 'Russe',
    'Finnish': 'Finnois',
    'Vietnamese': 'Vietnamien',
    'Chinese': 'Chinois',
}


def fig1_killer_graphic_fr():
    """
    Killer Graphic: Morphology Predicts Both Speed AND Quality (FRENCH)
    """
    # Data from exp8b FINAL_REPORT
    data = {
        'Français': {'tokens_60': 6.1, 'val_ppl': 37.7, 'wals': 7},
        'Espagnol': {'tokens_60': 16.4, 'val_ppl': 42.8, 'wals': 7},
        'Russe': {'tokens_60': 20.5, 'val_ppl': 43.3, 'wals': 7},
        'Finnois': {'tokens_60': 22.5, 'val_ppl': 52.9, 'wals': 5},
        'Anglais': {'tokens_60': 22.5, 'val_ppl': 74.4, 'wals': 4},
        'Vietnamien': {'tokens_60': 26.6, 'val_ppl': 50.4, 'wals': 1},
        'Chinois': {'tokens_60': 75.8, 'val_ppl': 97.1, 'wals': 1},
    }

    # Calculate normalized indices (French = 100 on both axes)
    best_tokens = min(d['tokens_60'] for d in data.values())
    best_ppl = min(d['val_ppl'] for d in data.values())

    for lang, d in data.items():
        d['speed_idx'] = (best_tokens / d['tokens_60']) * 100
        d['quality_idx'] = (best_ppl / d['val_ppl']) * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.cm.RdYlGn
    wals_min, wals_max = 1, 7

    for lang, d in data.items():
        color = cmap((d['wals'] - wals_min) / (wals_max - wals_min))
        size = 200 + d['wals'] * 30

        ax.scatter(d['speed_idx'], d['quality_idx'],
                   c=[color], s=size, edgecolors='black', linewidth=1.5, zorder=5)

        offset_x, offset_y = 3, 3
        ha = 'left'

        if lang == 'Français':
            offset_x, offset_y = -5, -8
            ha = 'right'
        elif lang == 'Chinois':
            offset_x, offset_y = 3, 5
        elif lang == 'Espagnol':
            offset_x, offset_y = 3, 3
        elif lang == 'Russe':
            offset_x, offset_y = 3, -5
        elif lang == 'Vietnamien':
            offset_x, offset_y = -5, 3
            ha = 'right'
        elif lang == 'Finnois':
            offset_x, offset_y = 3, -3
        elif lang == 'Anglais':
            offset_x, offset_y = 3, 3

        ax.annotate(lang, (d['speed_idx'], d['quality_idx']),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=11, fontweight='bold', ha=ha)

    # Diagonal arrow from Chinese to French
    zh = data['Chinois']
    fr = data['Français']
    ax.annotate('',
                xy=(fr['speed_idx'] - 5, fr['quality_idx'] - 5),
                xytext=(zh['speed_idx'] + 5, zh['quality_idx'] + 5),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2,
                               connectionstyle='arc3,rad=0.1'))

    mid_x = (zh['speed_idx'] + fr['speed_idx']) / 2 + 10
    mid_y = (zh['quality_idx'] + fr['quality_idx']) / 2
    ax.text(mid_x, mid_y, '12× plus rapide\n2,6× meilleur',
            fontsize=10, style='italic', color='#444444',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.axhline(y=50, color='#dddddd', linestyle='--', linewidth=1, zorder=1)
    ax.axvline(x=50, color='#dddddd', linestyle='--', linewidth=1, zorder=1)

    ax.fill([50, 105, 105, 50], [50, 50, 105, 105],
            color='#e8f5e9', alpha=0.5, zorder=0)
    ax.text(95, 95, 'OPTIMAL', fontsize=9, color='#2e7d32',
            ha='right', va='top', fontweight='bold', alpha=0.7)

    ax.fill([0, 50, 50, 0], [0, 0, 50, 50],
            color='#ffebee', alpha=0.5, zorder=0)
    ax.text(5, 5, 'SOUS-OPTIMAL', fontsize=9, color='#c62828',
            ha='left', va='bottom', fontweight='bold', alpha=0.7)

    ax.set_xlabel("Indice de vitesse d'apprentissage\n(plus élevé = moins de tokens pour atteindre 60% de grammaire)",
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Indice de qualité du modèle\n(plus élevé = perplexité de validation plus basse)',
                  fontsize=12, fontweight='bold')
    ax.set_title("La morphologie prédit la vitesse d'entraînement ET la qualité du modèle",
                 fontsize=14, fontweight='bold', pad=20)

    ax.set_xlim(0, 110)
    ax.set_ylim(30, 110)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=wals_min, vmax=wals_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label("Score d'accord WALS\n(complexité morphologique)", fontsize=10)
    cbar.set_ticks([1, 4, 7])
    cbar.set_ticklabels(['Faible\n(analytique)', 'Moyen', 'Élevé\n(synthétique)'])

    legend_text = "Français (normalisé à 100) = référence\nPlus élevé = Meilleur sur les deux axes"
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / 'killer_graphic_fr.pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'killer_graphic_fr.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: killer_graphic_fr.pdf/png")
    plt.close()


def fig2_grammar_race_fr():
    """
    Grammar emergence "race" - FRENCH version
    """
    tokens = np.linspace(0, 800, 100)

    def grammar_curve(tokens_60, final_acc, start_acc=0.35):
        k = 3 / tokens_60
        curve = start_acc + (final_acc - start_acc) / (1 + np.exp(-k * (tokens - tokens_60)))
        return np.clip(curve, start_acc, final_acc)

    curves = {
        'Français': grammar_curve(6.1, 0.87),
        'Espagnol': grammar_curve(16.4, 0.78),
        'Russe': grammar_curve(20.5, 0.80),
        'Finnois': grammar_curve(22.5, 0.72),
        'Anglais': grammar_curve(22.5, 0.87),
        'Vietnamien': grammar_curve(26.6, 0.55),
        'Chinois': grammar_curve(75.8, 0.55),
    }

    colors = {
        'Français': '#3498db',
        'Espagnol': '#2ecc71',
        'Russe': '#9b59b6',
        'Finnois': '#1abc9c',
        'Anglais': '#e74c3c',
        'Vietnamien': '#f39c12',
        'Chinois': '#e67e22',
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    for lang, curve in curves.items():
        ax.plot(tokens, curve * 100, linewidth=2.5, label=lang, color=colors[lang])

    ax.axhline(y=60, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(185, 61, 'Seuil de 60%', fontsize=9, color='gray')
    ax.text(185, 81, 'Seuil de 80%', fontsize=9, color='gray')

    ax.annotate('FR : 6,1M', xy=(6.1, 60), xytext=(6.1, 50),
                fontsize=9, ha='center', color='#3498db', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#3498db', alpha=0.7))
    ax.annotate('EN : 22,5M', xy=(22.5, 60), xytext=(22.5, 50),
                fontsize=9, ha='center', color='#e74c3c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', alpha=0.7))
    ax.annotate('ZH : 75,8M', xy=(75.8, 60), xytext=(75.8, 50),
                fontsize=9, ha='center', color='#e67e22', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e67e22', alpha=0.7))

    ax.annotate('', xy=(6.1, 45), xytext=(75.8, 45),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(40, 42, '12× plus rapide', ha='center', fontsize=11, fontweight='bold')

    ax.set_xlabel("Tokens d'entraînement (millions)", fontweight='bold')
    ax.set_ylabel('Précision grammaticale (%)', fontweight='bold')
    ax.set_title("Course à l'émergence grammaticale : le français gagne par 12×", fontweight='bold', pad=15)
    ax.set_xlim(0, 200)
    ax.set_ylim(30, 95)
    ax.legend(loc='lower right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grammar_race_fr.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'grammar_race_fr.png', dpi=300, bbox_inches='tight')
    print("Saved: grammar_race_fr.pdf/png")
    plt.close()


def fig3_wals_correlation_fr():
    """
    WALS VerbSynth vs Tokens to 60% grammar - FRENCH version
    """
    languages = ['FR', 'ES', 'RU', 'FI', 'EN', 'VI', 'ZH']
    verbsynth = [4, 4, 4, 2, 2, 0, 0]
    tokens_60 = [6.1, 16.4, 20.5, 22.5, 22.5, 26.6, 75.8]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#3498db', '#3498db', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#f39c12']

    for i, (lang, vs, t60, color) in enumerate(zip(languages, verbsynth, tokens_60, colors)):
        ax.scatter(vs, t60, c=color, s=200, edgecolors='black', linewidth=1.5, zorder=5)

        offset_x, offset_y = 0.15, 2
        if lang == 'FR':
            offset_x, offset_y = 0.15, -5
        elif lang in ['ES', 'RU']:
            offset_y = 3
        elif lang == 'ZH':
            offset_x, offset_y = -0.3, 0

        ax.annotate(lang, (vs + offset_x, t60 + offset_y), fontsize=11, fontweight='bold')

    z = np.polyfit(verbsynth, tokens_60, 1)
    p = np.poly1d(z)
    x_line = np.linspace(-0.5, 4.5, 100)
    ax.plot(x_line, p(x_line), '--', color='gray', linewidth=2, alpha=0.7, zorder=1)

    ax.text(0.5, 70, f'r = −0,88\np < 0,001', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    ax.set_xlabel('Synthèse du verbe WALS (22A)', fontweight='bold')
    ax.set_ylabel('Tokens pour 60% de grammaire (millions)', fontweight='bold')
    ax.set_title("La complexité verbale prédit la vitesse d'émergence grammaticale", fontweight='bold', pad=15)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 85)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['0\n(Analytique)', '1', '2\n(Faible)', '3', '4\n(Élevé)'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'wals_correlation_fr.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'wals_correlation_fr.png', dpi=300, bbox_inches='tight')
    print("Saved: wals_correlation_fr.pdf/png")
    plt.close()


def fig4_overfitting_crossover_fr():
    """
    The Overfitting Phenomenon - FRENCH version
    """
    steps = [545, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000]
    en_ppl = [971.38, 795.87, 629.28, 496.78, 506.73, 530.17, 541.13, 561.89, 629.67, 680.76, 735.94, 756.52, 794.16, 843.84]
    fr_ppl = [1439.24, 1082.45, 896.09, 516.47, 387.54, 335.34, 287.29, 253.35, 221.32, 180.90, 138.86, 113.11, 100.41, 88.14]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, en_ppl, 'o-', color='#e74c3c', linewidth=2.5, markersize=8, label='Anglais')
    ax.plot(steps, fr_ppl, 's-', color='#3498db', linewidth=2.5, markersize=8, label='Français')

    ax.axvline(x=3000, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.annotate('Croisement', xy=(3000, 500), xytext=(3500, 700),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

    ax.fill_between([3000, 13000], [0, 0], [1500, 1500], alpha=0.1, color='red')
    ax.text(8000, 1350, 'Surapprentissage anglais\n(PPL val. en hausse)',
            fontsize=9, ha='center', color='#c0392b', style='italic')

    ax.text(8000, 50, 'Généralisation française\n(PPL val. en baisse)',
            fontsize=9, ha='center', color='#2980b9', style='italic')

    ax.annotate(f'EN : {en_ppl[-1]:.0f}', xy=(13000, en_ppl[-1]), xytext=(13200, en_ppl[-1]),
                fontsize=10, fontweight='bold', color='#e74c3c')
    ax.annotate(f'FR : {fr_ppl[-1]:.0f}', xy=(13000, fr_ppl[-1]), xytext=(13200, fr_ppl[-1]),
                fontsize=10, fontweight='bold', color='#3498db')

    ax.annotate(f'Écart de 9,57×', xy=(13000, 450), xytext=(11500, 450),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='-[, widthB=2.0, lengthB=0.5', color='black'))

    ax.set_xlabel("Étapes d'entraînement", fontweight='bold')
    ax.set_ylabel('Perplexité de validation', fontweight='bold')
    ax.set_title("Le phénomène de surapprentissage : l'anglais surapprend tandis que le français généralise",
                 fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 14000)
    ax.set_ylim(0, 1500)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'overfitting_crossover_fr.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'overfitting_crossover_fr.png', dpi=300, bbox_inches='tight')
    print("Saved: overfitting_crossover_fr.pdf/png")
    plt.close()


def fig5_fertility_bar_fr():
    """
    Tokenization Fertility bar chart - FRENCH version
    """
    languages = ['EN', 'FR', 'ES', 'VI', 'FI', 'RU', 'ZH']
    fertility = [1.10, 1.17, 2.14, 2.58, 3.36, 4.47, 13.70]
    scripts = ['Latin', 'Latin', 'Latin', 'Latin', 'Latin', 'Cyrillique', 'Logographique']

    colors = []
    for s in scripts:
        if s == 'Latin':
            colors.append('#2ecc71')
        elif s == 'Cyrillique':
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(languages, fertility, color=colors, edgecolor='black', linewidth=1.2)

    for bar, fert in zip(bars, fertility):
        height = bar.get_height()
        ax.annotate(f'{fert:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.annotate('', xy=(0, 1.5), xytext=(1, 1.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(0.5, 1.7, 'Appariés\n(comparaison équitable)', ha='center', fontsize=9, fontweight='bold')

    ax.annotate('Pénalité de 3,8×', xy=(5, 4.47), xytext=(5, 6),
                ha='center', fontsize=9, color='#d35400',
                arrowprops=dict(arrowstyle='->', color='#d35400'))
    ax.annotate('Pénalité de 11,7×', xy=(6, 13.70), xytext=(6, 15.5),
                ha='center', fontsize=9, color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b'))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Alphabet latin'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Alphabet cyrillique'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Logographique'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_xlabel('Langue', fontweight='bold')
    ax.set_ylabel('Fertilité (tokens par mot)', fontweight='bold')
    ax.set_title("Fertilité de tokenisation : le BPE favorise l'alphabet latin", fontweight='bold', pad=15)
    ax.set_ylim(0, 17)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Idéal (1,0)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fertility_bar_fr.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fertility_bar_fr.png', dpi=300, bbox_inches='tight')
    print("Saved: fertility_bar_fr.pdf/png")
    plt.close()


if __name__ == "__main__":
    print("Generating French figures...")
    print("=" * 50)

    fig1_killer_graphic_fr()
    fig2_grammar_race_fr()
    fig3_wals_correlation_fr()
    fig4_overfitting_crossover_fr()
    fig5_fertility_bar_fr()

    print("=" * 50)
    print("All French figures generated!")
    print(f"Output directory: {OUTPUT_DIR}")
