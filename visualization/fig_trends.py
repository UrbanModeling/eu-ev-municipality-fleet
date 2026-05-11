# -*- coding: utf-8 -*-
"""
Figure: Temporal trends in EV adoption (2011–2023)
Two-panel figure for Scientific Data manuscript — Nature style.

Panel a: BEV fleet by country (stacked area), 2011–2023
Panel b: EV fleet share by country (line), 2011–2023

Output: output/fig_trends.png  (300 dpi)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from pathlib import Path

PROJ = Path(__file__).parent.parent

# ── Nature-style global settings ──────────────────────────────────────────
rcParams['font.family']        = 'Arial'
rcParams['font.size']          = 7
rcParams['axes.linewidth']     = 0.6
rcParams['xtick.major.width']  = 0.6
rcParams['ytick.major.width']  = 0.6
rcParams['xtick.major.size']   = 2.5
rcParams['ytick.major.size']   = 2.5
rcParams['xtick.direction']    = 'out'
rcParams['ytick.direction']    = 'out'
rcParams['axes.unicode_minus'] = False
rcParams['pdf.fonttype']       = 42   # editable text in Illustrator

# ── Load data ──────────────────────────────────────────────────────────────
nat    = pd.read_csv(PROJ / 'output' / 'result_national.csv')
years  = sorted(nat['year'].unique())

# Country order: largest 2023 BEV fleet first
order_bev = (nat[nat['year'] == 2023]
             .sort_values('BEV', ascending=False)['region'].tolist())

# ── Nature-style colour palette (14 countries) ────────────────────────────
# Curated qualitative palette inspired by Nature / npg colour schemes
PALETTE = {
    'Germany':        '#E64B35',   # red
    'United Kingdom': '#4DBBD5',   # cyan
    'France':         '#00A087',   # teal
    'Norway':         '#3C5488',   # navy
    'Netherlands':    '#F39B7F',   # salmon
    'Sweden':         '#8491B4',   # lavender
    'Italy':          '#91D1C2',   # mint
    'Denmark':        '#DC0000',   # bright red
    'Belgium':        '#7E6148',   # brown
    'Switzerland':    '#B09C85',   # tan
    'Austria':        '#56B4E9',   # sky blue
    'Spain':          '#E69F00',   # amber
    'Portugal':       '#009E73',   # green
    'Finland':        '#CC79A7',   # mauve
}

# ── Panel data ─────────────────────────────────────────────────────────────
bev_wide   = nat.pivot(index='year', columns='region', values='BEV')[order_bev] / 1e6
share_wide = nat.pivot(index='year', columns='region', values='ev_share') * 100

# ── Figure layout (Nature double-column ≈ 183 mm wide) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), facecolor='white')
fig.subplots_adjust(wspace=0.42, left=0.08, right=0.88, top=0.93, bottom=0.15)

colors = [PALETTE[c] for c in order_bev]

# ── Panel a: BEV fleet (stacked area) ─────────────────────────────────────
ax = axes[0]
ax.stackplot(years,
             [bev_wide[c].values for c in order_bev],
             labels=order_bev,
             colors=colors,
             alpha=0.88,
             linewidth=0)

ax.set_xlim(2011, 2023)
ax.set_ylim(0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
ax.set_ylabel('BEV fleet (millions)', fontsize=7)
ax.set_xlabel('Year', fontsize=7)
ax.set_title('a', fontsize=8, fontweight='bold', loc='left', pad=3)
ax.tick_params(labelsize=6.5, which='both')
ax.tick_params(which='minor', length=1.5)
ax.spines[['top', 'right']].set_visible(False)

# Legend: reverse so stack order matches visual top-to-bottom
handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels_leg[::-1],
          fontsize=5.5, loc='upper left', frameon=False,
          ncol=1, handlelength=1.0, handleheight=0.85,
          borderpad=0, labelspacing=0.3)

# ── Panel b: EV fleet share (line) ────────────────────────────────────────
ax = axes[1]

top5 = share_wide.loc[2023].nlargest(5).index.tolist()

# Non-highlighted countries first (background)
for country in order_bev:
    if country not in top5:
        ax.plot(years, share_wide[country],
                color='#CCCCCC', lw=0.7, zorder=1)

# Highlighted top-5 countries on top
for country in top5:
    ax.plot(years, share_wide[country],
            color=PALETTE[country], lw=1.5,
            marker='o', markersize=2.2,
            markeredgewidth=0, zorder=3,
            label=country)

ax.set_xlim(2011, 2023)
ax.set_ylim(0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.set_ylabel('EV fleet share (%)', fontsize=7)
ax.set_xlabel('Year', fontsize=7)
ax.set_title('b', fontsize=8, fontweight='bold', loc='left', pad=3)
ax.tick_params(labelsize=6.5, which='both')
ax.tick_params(which='minor', length=1.5)
ax.spines[['top', 'right']].set_visible(False)

# Direct labels at right margin
ax_right = ax.get_position().x1
for country in top5:
    val = share_wide.loc[2023, country]
    fig.text(ax_right + 0.005, ax.get_position().y0 +
             (val / ax.get_ylim()[1]) * ax.get_position().height,
             country, fontsize=5.8, va='center',
             color=PALETTE[country],
             transform=fig.transFigure)

# ── Save ───────────────────────────────────────────────────────────────────
outpath = PROJ / 'output' / 'fig_trends.png'
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {outpath}")
plt.close()
