# -*- coding: utf-8 -*-
"""
Figure: total-fleet rebalancing factor per country (see
compute_rebalance_factor.py).

Output: output/fig_rebalancing_factor.png  (300 dpi)
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

PROJ = Path(__file__).parent.parent
REBAL_FACTOR_CSV = PROJ / "output" / "rebalance_factor_by_country_total.csv"
OUT_PNG = PROJ / "output" / "fig_rebalancing_factor.png"

rcParams['font.family']       = 'Arial'
rcParams['font.size']         = 7
rcParams['axes.linewidth']    = 0.6
rcParams['xtick.major.width'] = 0.6
rcParams['ytick.major.width'] = 0.6
rcParams['xtick.major.size']  = 2.5
rcParams['ytick.major.size']  = 2.5

COUNTRIES = {
    'Germany':        '#E64B35',
    'France':         '#00A087',
    'Italy':          '#91D1C2',
    'Spain':          '#E69F00',
    'United Kingdom': '#4DBBD5',
    'Netherlands':    '#F39B7F',
    'Belgium':        '#7E6148',
}

rebal_factor = pd.read_csv(REBAL_FACTOR_CSV).set_index('region')

country_order = ['Spain', 'Italy', 'Belgium', 'Netherlands', 'Germany', 'United Kingdom', 'France']

labels = country_order
vals   = [rebal_factor.loc[c, 'rescale_factor'] for c in country_order]
colors = [COUNTRIES[c] for c in country_order]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.0, 3.6), facecolor='white')

x = range(len(labels))
ax.bar(x, vals, 0.6, color=colors, edgecolor='white', linewidth=0.6, zorder=3)

ax.axhline(1.0, color='black', linewidth=0.7, linestyle='--', zorder=2)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha='right')
ax.set_ylabel('Total-fleet rebalancing factor', fontsize=8)
ax.set_title('Magnitude of the rebalancing adjustment', fontsize=8, pad=8)
ax.tick_params(labelsize=7)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT_PNG}')
