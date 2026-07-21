# -*- coding: utf-8 -*-
"""
Supp. Fig. S2: (a) SHAP feature contribution of the full model, by vehicle
type. (b) NUTS-2 validation of the baseline (no-nightlight) model.

Output: output/fig7_feature_importance_ablation.png  (300 dpi)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from scipy.stats import pearsonr
from pathlib import Path

PROJ = Path(__file__).parent.parent
SHAP_CSV = PROJ / "output" / "shap" / "shap_importance_population_weighted.csv"
VAL_DIR = PROJ / "output" / "validation_nuts2_nolight"
OUT_PNG = PROJ / "output" / "fig7_feature_importance_ablation.png"

rcParams['font.family']       = 'Arial'
rcParams['font.size']         = 7
rcParams['axes.linewidth']    = 0.6
rcParams['xtick.major.width'] = 0.6
rcParams['ytick.major.width'] = 0.6
rcParams['xtick.major.size']  = 2.5
rcParams['ytick.major.size']  = 2.5

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.4, 3.8), facecolor='white')

# ═══════════════════════════════════════════════════════════════════════════
# Panel (a) - SHAP feature contribution
# ═══════════════════════════════════════════════════════════════════════════
FEATURES = ["gdp_per_cap", "light_per_cap", "light_mean_per_cap", "year"]
FEATURE_LABELS = {
    "gdp_per_cap":        "GDP per capita",
    "light_per_cap":      "Nightlight per capita",
    "light_mean_per_cap": "Nightlight mean per capita",
    "year":               "Year (temporal trend)",
}
FEATURE_COLORS = {
    "gdp_per_cap":        "#4DBBD5",
    "light_per_cap":      "#F39B7F",
    "light_mean_per_cap": "#00A087",
    "year":               "#3C5488",
}
VEHICLE_LABELS = {
    "EV stock_BEV":  "BEV",
    "EV stock_PHEV": "PHEV",
    "EV stock_ICEV": "Other vehicle",
}
VEHICLES = list(VEHICLE_LABELS.keys())

df = pd.read_csv(SHAP_CSV)
df = df[df["parameter"].isin(VEHICLES)]
pivot = (df.pivot(index="parameter", columns="feature", values="pct_contribution")
         .loc[VEHICLES, FEATURES])

x = np.arange(len(VEHICLES))
bar_w = 0.55
bottom = np.zeros(len(VEHICLES))
for feat in FEATURES:
    vals = pivot[feat].values
    ax_a.bar(x, vals, bar_w, bottom=bottom, color=FEATURE_COLORS[feat],
             edgecolor='white', linewidth=1.2, zorder=3,
             label=FEATURE_LABELS[feat])
    for xi, v, b in zip(x, vals, bottom):
        if v >= 6:
            ax_a.text(xi, b + v / 2, f'{v:.0f}%', ha='center', va='center',
                       fontsize=6.5, color='white', zorder=4)
    bottom += vals

ax_a.set_xticks(x)
ax_a.set_xticklabels([VEHICLE_LABELS[v] for v in VEHICLES], fontsize=8)
ax_a.set_ylabel('SHAP contribution (%)', fontsize=8)
ax_a.set_ylim(0, 100)
ax_a.set_title('Feature importance (SHAP)', fontsize=8, pad=8)
ax_a.tick_params(labelsize=7)
ax_a.spines[['top', 'right']].set_visible(False)
ax_a.spines['left'].set_visible(False)
ax_a.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.4, zorder=1)
ax_a.set_axisbelow(True)
ax_a.legend(fontsize=6, frameon=False, loc='upper center',
            bbox_to_anchor=(0.5, -0.16), ncol=2, handlelength=1.0, columnspacing=1.0)
ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes,
          fontsize=13, fontweight='bold', va='bottom')

# ═══════════════════════════════════════════════════════════════════════════
# Panel (b) - NUTS-2 baseline model validation
# ═══════════════════════════════════════════════════════════════════════════
COUNTRIES = {
    'deu': ('Germany',        '#E64B35'),
    'fra': ('France',         '#00A087'),
    'ita': ('Italy',          '#91D1C2'),
    'esp': ('Spain',          '#E69F00'),
    'gbr': ('United Kingdom', '#4DBBD5'),
    'nld': ('Netherlands',    '#F39B7F'),
    'bel': ('Belgium',        '#7E6148'),
}

frames = []
for code, (name, color) in COUNTRIES.items():
    d = pd.read_csv(VAL_DIR / f'{code}_nuts2_comparison.csv')
    d['country'] = name
    frames.append(d)
data = pd.concat(frames, ignore_index=True).dropna(subset=['our_total', 'estat_total'])

actual = data['estat_total'].values
pred = data['our_total'].values
ss_res = np.sum((actual - pred) ** 2)
ss_tot = np.sum((actual - actual.mean()) ** 2)
r2_all = 1.0 - ss_res / ss_tot
r_all, _ = pearsonr(actual, pred)

summary = pd.read_csv(VAL_DIR / 'validation_summary.csv')

xy_max = max(actual.max(), pred.max()) * 1.05
ax_b.plot([0, xy_max], [0, xy_max], color='black', lw=1.0, ls='--', zorder=2, label='1:1 line')

for code, (name, color) in COUNTRIES.items():
    sub = data[data['country'] == name]
    a = sub['estat_total'].values
    p = sub['our_total'].values
    slope, intercept = np.polyfit(a, p, 1)
    x_fit = np.linspace(0, a.max(), 200)
    ax_b.plot(x_fit, slope * x_fit + intercept, color=color, lw=0.9, ls='-', zorder=3, alpha=0.8)

for code, (name, color) in COUNTRIES.items():
    sub = data[data['country'] == name]
    r2_row = summary.loc[summary['country'] == name, 'R2'].values
    r2_val = r2_row[0] if len(r2_row) else np.nan
    ax_b.scatter(sub['estat_total'], sub['our_total'], s=18, alpha=0.55,
                 color=color, edgecolors='none', zorder=4,
                 label=f'{name} ($R^2$={r2_val:.2f})')

ax_b.text(0.97, 0.05, f'Overall $R^2$ = {r2_all:.3f}\nPearson $r$ = {r_all:.3f}',
          transform=ax_b.transAxes, fontsize=7, ha='right', va='bottom',
          bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

ax_b.set_xlim(0, xy_max)
ax_b.set_ylim(0, xy_max)
ax_b.set_aspect('equal', adjustable='box')
ax_b.set_xlabel('Actual total fleet, millions (Eurostat NUTS-2)', fontsize=8)
ax_b.set_ylabel('Predicted total fleet, millions', fontsize=8)
ax_b.set_title('NUTS-2 validation - baseline model', fontsize=8, pad=6)
ax_b.tick_params(labelsize=7)
ax_b.spines[['top', 'right']].set_visible(False)
ax_b.grid(True, linestyle=':', linewidth=0.5, alpha=0.4, zorder=1)
ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.0f}'))
ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.0f}'))
ax_b.legend(fontsize=6, frameon=False, loc='upper left', handlelength=0.8, labelspacing=0.35)
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes,
          fontsize=13, fontweight='bold', va='bottom')

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_PNG}")
