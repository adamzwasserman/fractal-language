#!/usr/bin/env python3
"""
Killer Graphic: Morphology Predicts Both Speed AND Quality

Inverted axes scatterplot showing:
- X: Learning Speed Index (higher = fewer tokens to 60% grammar = better)
- Y: Quality Index (higher = lower perplexity = better)
- Color: WALS Agreement Score
- Upper-right = BEST, Lower-left = WORST
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data from exp8b FINAL_REPORT
data = {
    'French': {'tokens_60': 6.1, 'val_ppl': 37.7, 'wals': 7},
    'Spanish': {'tokens_60': 16.4, 'val_ppl': 42.8, 'wals': 7},
    'Russian': {'tokens_60': 20.5, 'val_ppl': 43.3, 'wals': 7},
    'Finnish': {'tokens_60': 22.5, 'val_ppl': 52.9, 'wals': 5},
    'English': {'tokens_60': 22.5, 'val_ppl': 74.4, 'wals': 4},
    'Vietnamese': {'tokens_60': 26.6, 'val_ppl': 50.4, 'wals': 1},
    'Chinese': {'tokens_60': 75.8, 'val_ppl': 97.1, 'wals': 1},
}

# Calculate normalized indices (French = 100 on both axes)
best_tokens = min(d['tokens_60'] for d in data.values())  # 6.1 (French)
best_ppl = min(d['val_ppl'] for d in data.values())  # 37.7 (French)

for lang, d in data.items():
    d['speed_idx'] = (best_tokens / d['tokens_60']) * 100
    d['quality_idx'] = (best_ppl / d['val_ppl']) * 100

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 8))

# Color map based on WALS score
cmap = plt.cm.RdYlGn  # Red (low) -> Yellow (mid) -> Green (high)
wals_min, wals_max = 1, 7

# Plot each language
for lang, d in data.items():
    color = cmap((d['wals'] - wals_min) / (wals_max - wals_min))
    size = 200 + d['wals'] * 30  # Larger points for higher WALS

    ax.scatter(d['speed_idx'], d['quality_idx'],
               c=[color], s=size, edgecolors='black', linewidth=1.5, zorder=5)

    # Label positioning
    offset_x, offset_y = 3, 3
    ha = 'left'

    # Adjust specific labels to avoid overlap
    if lang == 'French':
        offset_x, offset_y = -5, -8
        ha = 'right'
    elif lang == 'Chinese':
        offset_x, offset_y = 3, 5
    elif lang == 'Spanish':
        offset_x, offset_y = 3, 3
    elif lang == 'Russian':
        offset_x, offset_y = 3, -5
    elif lang == 'Vietnamese':
        offset_x, offset_y = -5, 3
        ha = 'right'
    elif lang == 'Finnish':
        offset_x, offset_y = 3, -3
    elif lang == 'English':
        offset_x, offset_y = 3, 3

    ax.annotate(lang, (d['speed_idx'], d['quality_idx']),
                xytext=(offset_x, offset_y), textcoords='offset points',
                fontsize=11, fontweight='bold', ha=ha)

# Add diagonal arrow from Chinese to French
zh = data['Chinese']
fr = data['French']
ax.annotate('',
            xy=(fr['speed_idx'] - 5, fr['quality_idx'] - 5),
            xytext=(zh['speed_idx'] + 5, zh['quality_idx'] + 5),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=2,
                           connectionstyle='arc3,rad=0.1'))

# Add annotation for the gap
mid_x = (zh['speed_idx'] + fr['speed_idx']) / 2 + 10
mid_y = (zh['quality_idx'] + fr['quality_idx']) / 2
ax.text(mid_x, mid_y, '12× faster\n2.6× better',
        fontsize=10, style='italic', color='#444444',
        ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9))

# Shade the quadrants subtly
ax.axhline(y=50, color='#dddddd', linestyle='--', linewidth=1, zorder=1)
ax.axvline(x=50, color='#dddddd', linestyle='--', linewidth=1, zorder=1)

# Winner's corner highlight
ax.fill([50, 105, 105, 50], [50, 50, 105, 105],
        color='#e8f5e9', alpha=0.5, zorder=0)
ax.text(95, 95, 'OPTIMAL', fontsize=9, color='#2e7d32',
        ha='right', va='top', fontweight='bold', alpha=0.7)

# Loser's corner highlight
ax.fill([0, 50, 50, 0], [0, 0, 50, 50],
        color='#ffebee', alpha=0.5, zorder=0)
ax.text(5, 5, 'SUBOPTIMAL', fontsize=9, color='#c62828',
        ha='left', va='bottom', fontweight='bold', alpha=0.7)

# Axis labels and title
ax.set_xlabel('Learning Speed Index\n(higher = fewer tokens to reach 60% grammar)',
              fontsize=12, fontweight='bold')
ax.set_ylabel('Model Quality Index\n(higher = lower validation perplexity)',
              fontsize=12, fontweight='bold')
ax.set_title('Morphology Predicts Both Training Speed AND Model Quality',
             fontsize=14, fontweight='bold', pad=20)

# Set axis limits
ax.set_xlim(0, 110)
ax.set_ylim(30, 110)

# Add colorbar for WALS score
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=wals_min, vmax=wals_max))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
cbar.set_label('WALS Agreement Score\n(morphological complexity)', fontsize=10)
cbar.set_ticks([1, 4, 7])
cbar.set_ticklabels(['Low\n(analytic)', 'Medium', 'High\n(synthetic)'])

# Add legend for interpretation
legend_text = "French (normalized to 100) = baseline\nHigher = Better on both axes"
ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.9))

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Tight layout
plt.tight_layout()

# Save figures
plt.savefig('/Volumes/fractal/exp8b_balanced_multilingual/figures/killer_graphic.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Volumes/fractal/exp8b_balanced_multilingual/figures/killer_graphic.pdf',
            bbox_inches='tight', facecolor='white')

print("Saved: killer_graphic.png and killer_graphic.pdf")
print("\nData summary:")
print(f"{'Language':<12} {'Speed':<8} {'Quality':<8} {'WALS':<6}")
print("-" * 36)
for lang, d in sorted(data.items(), key=lambda x: -x[1]['speed_idx']):
    print(f"{lang:<12} {d['speed_idx']:<8.1f} {d['quality_idx']:<8.1f} {d['wals']:<6}")

plt.show()
