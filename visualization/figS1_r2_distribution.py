# -*- coding: utf-8 -*-
"""
Figure: R^2 distribution across all validation exercises.
NUTS-2 level (7 countries, total fleet) + municipality level (NO/SE BEV).
Germany is excluded from municipality-level validation - its KBA Kreis data
is used only to augment model training, not as an independent check.

Output: output/fig_r2_distribution.png (300 dpi)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

PROJ = Path(__file__).parent.parent
OUT_PNG = PROJ / "output" / "fig_r2_distribution.png"

rcParams['font.family']       = 'Arial'
rcParams['font.size']         = 7
rcParams['axes.linewidth']    = 0.6
rcParams['xtick.major.width'] = 0.6
rcParams['ytick.major.width'] = 0.6
rcParams['xtick.major.size']  = 2.5
rcParams['ytick.major.size']  = 2.5

COLOR_NUTS2 = "#4DBBD5"
COLOR_MUNI  = "#F39B7F"

nuts2 = [
    ("Spain", 0.984), ("Italy", 0.981), ("Germany", 0.951),
    ("Netherlands", 0.928), ("United Kingdom", 0.906),
    ("Belgium", 0.906), ("France", 0.872),
]
muni = [
    ("Norway BEV", 0.889), ("Sweden BEV", 0.820),
]

fig, ax = plt.subplots(figsize=(4.4, 3.4), facecolor='white')

labels = [f"{c} (NUTS-2)" for c, _ in nuts2] + [f"{c} (municipality)" for c, _ in muni]
values = [v for _, v in nuts2] + [v for _, v in muni]
colors = [COLOR_NUTS2] * len(nuts2) + [COLOR_MUNI] * len(muni)

order = np.argsort(values)
labels = [labels[i] for i in order]
values = [values[i] for i in order]
colors = [colors[i] for i in order]

y = np.arange(len(labels))
ax.scatter(values, y, c=colors, s=28, zorder=3, edgecolor='white', linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=6.5)
ax.set_xlabel("R²", fontsize=8)
pad = (max(values) - min(values)) * 0.15
ax.set_xlim(min(values) - pad, max(values) + pad)
ax.axvline(np.mean(values), color='grey', linewidth=0.7, linestyle=':', zorder=1)
ax.grid(axis='x', linestyle=':', linewidth=0.5, alpha=0.4, zorder=1)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title("R² across all validation exercises", fontsize=8, pad=8)

handles = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_NUTS2, markersize=5, label='NUTS-2'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_MUNI, markersize=5, label='Municipality'),
]
ax.legend(handles=handles, fontsize=6, frameon=False, loc='lower right')

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_PNG}")
plt.show()
