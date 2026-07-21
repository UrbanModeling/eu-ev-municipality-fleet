# -*- coding: utf-8 -*-
"""
Figure: NUTS-2 regional validation of total passenger vehicle stock
Single-panel scatter with OLS fit, coloured by country.

Output: output/fig_nuts2_validation.png  (300 dpi)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from scipy.stats import pearsonr
from pathlib import Path

PROJ = Path(__file__).parent.parent

# ── Nature-style settings ──────────────────────────────────────────────────
rcParams['font.family']        = 'Arial'
rcParams['font.size']          = 7
rcParams['axes.linewidth']     = 0.6
rcParams['xtick.major.width']  = 0.6
rcParams['ytick.major.width']  = 0.6
rcParams['xtick.major.size']   = 2.5
rcParams['ytick.major.size']   = 2.5
rcParams['xtick.direction']    = 'out'
rcParams['ytick.direction']    = 'out'

# ── Country palette (consistent with fig_trends) ───────────────────────────
COUNTRIES = {
    'deu': ('Germany',        '#E64B35'),
    'fra': ('France',         '#00A087'),
    'ita': ('Italy',          '#91D1C2'),
    'esp': ('Spain',          '#E69F00'),
    'gbr': ('United Kingdom', '#4DBBD5'),
    'nld': ('Netherlands',    '#F39B7F'),
    'bel': ('Belgium',        '#7E6148'),
}

# ── Load and concatenate ───────────────────────────────────────────────────
frames = []
for code, (name, color) in COUNTRIES.items():
    df = pd.read_csv(PROJ / 'output' / 'validation_nuts2' / f'{code}_nuts2_comparison.csv')
    df['country'] = name
    df['color']   = color
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
data = data.dropna(subset=['our_total', 'estat_total'])

actual = data['estat_total'].values
pred   = data['our_total'].values

# ── Overall R² and Pearson r ───────────────────────────────────────────────
ss_res = np.sum((actual - pred) ** 2)
ss_tot = np.sum((actual - actual.mean()) ** 2)
r2_all = 1.0 - ss_res / ss_tot
r_all, _ = pearsonr(actual, pred)

# ── Per-country R² ────────────────────────────────────────────────────────
summary = pd.read_csv(PROJ / 'output' / 'validation_nuts2' / 'validation_summary.csv')

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor='white')

xy_max = max(actual.max(), pred.max()) * 1.05
ax.plot([0, xy_max], [0, xy_max], color='black', lw=1.0, ls='--',
        zorder=2, label='1:1 line')

# Per-country OLS fit lines
for code, (name, color) in COUNTRIES.items():
    sub = data[data['country'] == name]
    a = sub['estat_total'].values
    p = sub['our_total'].values
    slope, intercept = np.polyfit(a, p, 1)
    x_fit = np.linspace(0, a.max(), 200)
    ax.plot(x_fit, slope * x_fit + intercept,
            color=color, lw=0.9, ls='-', zorder=3, alpha=0.8)

# Scatter points
for code, (name, color) in COUNTRIES.items():
    sub = data[data['country'] == name]
    r2_row = summary.loc[summary['country'] == name, 'R2'].values
    r2_val = r2_row[0] if len(r2_row) else np.nan
    ax.scatter(sub['estat_total'], sub['our_total'],
               s=18, alpha=0.55, color=color, edgecolors='none',
               zorder=4, label=f'{name} ($R^2$={r2_val:.2f})')

# Overall stats annotation
ax.text(0.97, 0.05,
        f'Overall $R^2$ = {r2_all:.3f}\nPearson $r$ = {r_all:.3f}',
        transform=ax.transAxes, fontsize=7,
        ha='right', va='bottom',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

ax.set_xlim(0, xy_max)
ax.set_ylim(0, xy_max)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Actual total fleet, millions (Eurostat NUTS-2)', fontsize=8)
ax.set_ylabel('Predicted total fleet, millions', fontsize=8)
ax.set_title('NUTS-2 regional validation — total passenger vehicle stock',
             fontsize=8, pad=6)
ax.tick_params(labelsize=7)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4, zorder=1)

ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.0f}'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.0f}'))

ax.legend(fontsize=6.5, frameon=False, loc='upper left',
          handlelength=0.8, labelspacing=0.4)

plt.tight_layout()

outpath = PROJ / 'output' / 'fig_nuts2_validation.png'
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {outpath}')
plt.close()
