# -*- coding: utf-8 -*-
"""
Figure: Dataset construction workflow diagram
Nature-style schematic for Scientific Data manuscript.

Output: output/fig_workflow.png  (300 dpi)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams
from pathlib import Path

PROJ = Path(__file__).parent.parent

# ── Nature-style global settings ──────────────────────────────────────────
rcParams['font.family']    = 'Arial'
rcParams['font.size']      = 7
rcParams['axes.linewidth'] = 0.6
rcParams['pdf.fonttype']   = 42

# ── Colour scheme ─────────────────────────────────────────────────────────
C = {
    'input':      '#EBF2FA',   # light blue   – data source boxes
    'input_bd':   '#2166AC',   # blue border
    'proc':       '#F0F7F0',   # light green  – processing boxes
    'proc_bd':    '#1A7A3F',   # green border
    'model':      '#FEF5E7',   # light amber  – model box
    'model_bd':   '#E67E22',   # amber border
    'output':     '#F5EEF8',   # light purple – output boxes
    'output_bd':  '#7D3C98',   # purple border
    'valid':      '#FDF2F2',   # light red    – validation boxes
    'valid_bd':   '#C0392B',   # red border
    'arrow':      '#444444',
    'label_bg':   '#F8F8F8',
}

# ── Canvas ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor='white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# ── Helper functions ───────────────────────────────────────────────────────

def box(x, y, w, h, fc, ec, lw=0.9, radius=0.18):
    """Draw a rounded rectangle."""
    patch = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
    )
    ax.add_patch(patch)
    return patch


def label(x, y, text, size=6.8, bold=False, color='#222222', ha='center', va='center'):
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha=ha, va=va, fontsize=size,
            fontweight=weight, color=color, zorder=4,
            linespacing=1.35)


def arrow(x0, y0, x1, y1, color='#444444', lw=1.0,
          arrowstyle='->', mutation_scale=8):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=arrowstyle,
                                color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'),
                zorder=2)


def section_title(x, y, text):
    ax.text(x, y, text, ha='left', va='center', fontsize=6.5,
            fontweight='bold', color='#555555', style='italic', zorder=4)


# ══════════════════════════════════════════════════════════════════════════
# ROW 1 — Data Sources  (y = 9.0)
# ══════════════════════════════════════════════════════════════════════════
section_title(0.15, 9.55, 'Data sources')

src_y   = 9.0
src_w   = 2.0
src_h   = 0.82
src_xs  = [1.2, 3.4, 5.6, 7.8, 9.5]   # centres of 4+1 boxes

src_items = [
    ('IEA Global EV Outlook\n(BEV · PHEV · FCEV\nnational annual stocks)',   src_xs[0]),
    ('VIIRS / DMSP\nNighttime Light\ncomposites',                             src_xs[1]),
    ('Gridded GDP &\nPopulation\n(5-yr intervals)',                           src_xs[2]),
    ('GADM Level-4\nboundaries\n(65,300 municipalities)',                     src_xs[3]),
]

for text, cx in src_items:
    box(cx, src_y, src_w, src_h, C['input'], C['input_bd'])
    label(cx, src_y, text, size=6.3)

# ══════════════════════════════════════════════════════════════════════════
# ROW 2 — Feature Engineering  (y = 7.5)
# ══════════════════════════════════════════════════════════════════════════
feat_y = 7.5
feat_w = 8.2
feat_h = 0.78

box(5.0, feat_y, feat_w, feat_h, C['proc'], C['proc_bd'])
label(5.0, feat_y + 0.12, 'Feature Engineering', bold=True, size=7)
label(5.0, feat_y - 0.17,
      'GDP per capita  ·  NTL sum per capita  ·  NTL mean per capita  ·  Year',
      size=6.3, color='#444444')

# Arrows from each source box down to feature engineering
for cx in [src_xs[i] for i in range(4)]:
    arrow(cx, src_y - src_h/2 - 0.04,
          cx, feat_y + feat_h/2 + 0.04)

# ══════════════════════════════════════════════════════════════════════════
# ROW 3 — XGBoost Model  (y = 6.0)
# ══════════════════════════════════════════════════════════════════════════
model_y = 6.0

# Left sub-box: training
train_w, train_h = 3.4, 1.0
box(2.9, model_y, train_w, train_h, C['model'], C['model_bd'])
label(2.9, model_y + 0.22, 'XGBoost Training', bold=True, size=7)
label(2.9, model_y - 0.05,
      '14 countries × 13 years = 182 obs\nTarget: vehicles per capita',
      size=6.2, color='#444444')
label(2.9, model_y - 0.30,
      '4 models: BEV · PHEV · FCEV · Other',
      size=6.0, color='#888888')

# Right sub-box: city-level prediction
pred_w, pred_h = 3.4, 1.0
box(7.1, model_y, pred_w, pred_h, C['model'], C['model_bd'])
label(7.1, model_y + 0.22, 'City-level Prediction', bold=True, size=7)
label(7.1, model_y - 0.05,
      '65,300 municipalities × 13 years\nPredicted vehicles per capita',
      size=6.2, color='#444444')
label(7.1, model_y - 0.30,
      '→ multiply by city population',
      size=6.0, color='#888888')

# Arrows: feat eng → train and feat eng → predict
arrow(5.0, feat_y - feat_h/2 - 0.04, 2.9, model_y + train_h/2 + 0.04)
arrow(5.0, feat_y - feat_h/2 - 0.04, 7.1, model_y + pred_h/2 + 0.04)

# Arrow: train → predict (horizontal)
arrow(2.9 + train_w/2 + 0.05, model_y,
      7.1 - pred_w/2 - 0.05, model_y,
      arrowstyle='->')
ax.text(5.0, model_y + 0.10, 'apply', ha='center', va='center',
        fontsize=5.8, color='#888888', style='italic', zorder=4)

# ══════════════════════════════════════════════════════════════════════════
# ROW 4 — Proportional Rebalancing  (y = 4.6)
# ══════════════════════════════════════════════════════════════════════════
reb_y = 4.6
reb_w = 5.5
reb_h = 0.72

box(5.0, reb_y, reb_w, reb_h, C['proc'], C['proc_bd'])
label(5.0, reb_y + 0.13, 'Proportional Rebalancing', bold=True, size=7)
label(5.0, reb_y - 0.13,
      'City stocks scaled so Σ(city) = national total exactly  (per country, year, powertrain)',
      size=6.2, color='#444444')

arrow(7.1, model_y - pred_h/2 - 0.04,
      5.0, reb_y + reb_h/2 + 0.04)

# ══════════════════════════════════════════════════════════════════════════
# ROW 5 — Output Dataset  (y = 3.3)
# ══════════════════════════════════════════════════════════════════════════
out_y = 3.3
section_title(0.15, 3.72, 'Output dataset')

# City file
box(3.3, out_y, 3.6, 0.80, C['output'], C['output_bd'])
label(3.3, out_y + 0.17, 'result_city.csv', bold=True, size=7,
      color=C['output_bd'])
label(3.3, out_y - 0.12,
      '848,900 rows · 14 columns\n65,300 municipalities × 13 years',
      size=6.2, color='#444444')

# National file
box(7.3, out_y, 3.0, 0.80, C['output'], C['output_bd'])
label(7.3, out_y + 0.17, 'result_national.csv', bold=True, size=7,
      color=C['output_bd'])
label(7.3, out_y - 0.12,
      '182 rows · 10 columns\n14 countries × 13 years',
      size=6.2, color='#444444')

# Arrows: rebalancing → outputs
arrow(5.0, reb_y - reb_h/2 - 0.04, 3.3, out_y + 0.40 + 0.04)
arrow(5.0, reb_y - reb_h/2 - 0.04, 7.3, out_y + 0.40 + 0.04)

# ══════════════════════════════════════════════════════════════════════════
# ROW 6 — Validation  (y = 1.85 and 1.0)
# ══════════════════════════════════════════════════════════════════════════
section_title(0.15, 2.62, 'Technical validation')

val_items = [
    (2.2,  1.85, 'NUTS-2 Validation\n(Eurostat road_eqs_carage)',
              '7 countries · R² = 0.82 – 0.99'),
    (5.55, 1.85, 'Municipality Validation — Germany\n(KBA Landkreis, n = 398)',
              'BEV R² = 0.755 · PHEV R² = 0.754'),
    (5.55, 1.00, 'Municipality Validation — Norway\n(SSB, n = 356)',
              'BEV R² = 0.653'),
    (8.55, 1.85, 'Municipality Validation — Sweden\n(SCB, n = 290)',
              'BEV R² = 0.923 · PHEV R² = 0.948'),
]

for cx, cy, title_txt, stat_txt in val_items:
    box(cx, cy, 3.1, 0.72, C['valid'], C['valid_bd'])
    label(cx, cy + 0.14, title_txt, bold=True, size=6.3, color=C['valid_bd'])
    label(cx, cy - 0.16, stat_txt, size=6.0, color='#444444')

# Arrows from city output to validation boxes
for cx, cy, _, _ in val_items:
    arrow(3.3, out_y - 0.40 - 0.04, cx, cy + 0.36 + 0.04)

# Norway validation below Germany — connect sideways
arrow(5.55, 1.85 - 0.36 - 0.04, 5.55, 1.00 + 0.36 + 0.04)

# ══════════════════════════════════════════════════════════════════════════
# Row dividers (subtle horizontal rules)
# ══════════════════════════════════════════════════════════════════════════
for rule_y in [9.45, 8.15, 7.05, 5.45, 3.9, 2.55]:
    ax.axhline(rule_y, xmin=0.01, xmax=0.99,
               color='#DDDDDD', lw=0.5, zorder=1)

# ══════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════
outpath = PROJ / 'output' / 'fig_workflow.png'
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {outpath}")
plt.close()
