# -*- coding: utf-8 -*-
"""
Figure: Geographic distribution of EV fleet share and EV fleet size (2024)
Two-panel choropleth map for Scientific Data manuscript.

Panel A: EV fleet share (%) by municipality, 2024
Panel B: EV fleet (BEV+PHEV+FCEV) by municipality, 2024

Output: output/fig_geodist.png  (300 dpi, ~190mm wide)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from pathlib import Path

PROJ = Path(__file__).parent.parent
OUT  = PROJ / 'output'
OUT.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
city = pd.read_csv(PROJ / 'output' / 'result_city.csv')
LAST_YEAR = city['year'].max()
c23  = city[city['year'] == LAST_YEAR].copy()

gdf = gpd.read_file(str(PROJ / 'data' / 'boundary' / 'eu14_city.gpkg'),
                    columns=['UID', 'geometry'])
gdf['UID'] = gdf['UID'].astype(int)
c23['uid'] = c23['uid'].astype(int)
c23['ev_share'] = c23['ev_share'].fillna(0)
c23['EV'] = (c23['BEV'].fillna(0) + c23['PHEV'].fillna(0) + c23['FCEV'].fillna(0))
gdf = gdf.merge(c23[['uid', 'ev_share', 'EV']],
                left_on='UID', right_on='uid', how='left')
print(f"  Matched: {gdf['ev_share'].notna().sum()} / {len(gdf)}")

# Project to ETRS89-LAEA Europe (metric, equal-area)
gdf = gdf.to_crs('EPSG:3035')

# ── Clip to study area extent (remove overseas territories / outliers) ────
bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
# EU continental extent in EPSG:3035 (approx)
xmin, xmax = 2_500_000, 6_500_000
ymin, ymax = 1_400_000, 5_500_000
gdf_clip = gdf.cx[xmin:xmax, ymin:ymax]

# ── Colour scales ──────────────────────────────────────────────────────────
# Panel A: EV share — linear 0–25%, cap at 25th percentile of top values
ev_share_pct = gdf_clip['ev_share'] * 100
vmax_a = np.percentile(ev_share_pct.dropna(), 99)   # cap at 99th pct
vmax_a = round(vmax_a / 5) * 5  # round to nearest 5

# Panel B: EV fleet — log scale (many zeros → use +1 offset)
savings = gdf_clip['EV'].clip(lower=0)
vmax_b = np.percentile(savings[savings > 0], 99)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                         facecolor='white')

CMAP_A = 'YlOrRd'
CMAP_B = 'PuBuGn'
MISSING_COLOR = '#d9d9d9'
EDGE_COLOR    = 'none'
EDGE_WIDTH    = 0.0

# ── Panel A: EV fleet share ────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('white')

# Municipalities with data
has_data = gdf_clip['ev_share'].notna()
gdf_clip[~has_data].plot(ax=ax, color=MISSING_COLOR, linewidth=EDGE_WIDTH)
gdf_clip[has_data].plot(
    ax=ax,
    column='ev_share',
    cmap=CMAP_A,
    vmin=0,
    vmax=vmax_a / 100,
    linewidth=EDGE_WIDTH,
    edgecolor=EDGE_COLOR,
)

# Colorbar
sm_a = plt.cm.ScalarMappable(
    cmap=CMAP_A,
    norm=mcolors.Normalize(vmin=0, vmax=vmax_a)
)
sm_a.set_array([])
cbar_a = fig.colorbar(sm_a, ax=ax, fraction=0.03, pad=0.02, aspect=25)
cbar_a.set_label('EV fleet share (%)', fontsize=9)
cbar_a.ax.tick_params(labelsize=8)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_axis_off()
ax.set_title(f'(a)  EV fleet share, {LAST_YEAR}', fontsize=10, pad=6, loc='left')

# ── Panel B: EV fleet (log scale) ─────────────────────────────────────────
ax = axes[1]
ax.set_facecolor('white')

gdf_clip[~has_data].plot(ax=ax, color=MISSING_COLOR, linewidth=EDGE_WIDTH)

# Use log norm; municipalities with 0 EV fleet get minimum colour
savings_plot = gdf_clip['EV'].clip(lower=1)  # floor at 1 for log
norm_b = LogNorm(vmin=1, vmax=vmax_b)

gdf_clip[has_data].assign(_s=savings_plot[has_data]).plot(
    ax=ax,
    column='_s',
    cmap=CMAP_B,
    norm=norm_b,
    linewidth=EDGE_WIDTH,
    edgecolor=EDGE_COLOR,
)

sm_b = plt.cm.ScalarMappable(cmap=CMAP_B, norm=norm_b)
sm_b.set_array([])
cbar_b = fig.colorbar(sm_b, ax=ax, fraction=0.03, pad=0.02, aspect=25)
cbar_b.set_label('EV fleet', fontsize=9)
cbar_b.ax.tick_params(labelsize=8)
# Format log-scale ticks
import matplotlib.ticker as ticker
cbar_b.ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f'{x:,.0f}' if x >= 1000 else f'{x:.0f}')
)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_axis_off()
ax.set_title(f'(b)  EV fleet, {LAST_YEAR}', fontsize=10, pad=6, loc='left')

# ── Caption note ───────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    'Each polygon represents one LAU municipality (n = 71,715). '
    'Grey: municipalities with no model coverage. Projection: ETRS89-LAEA (EPSG:3035).',
    ha='center', fontsize=7.5, color='#555555'
)

plt.tight_layout(rect=[0, 0.03, 1, 1])

outpath = OUT / 'fig_geodist.png'
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {outpath}")
plt.close()
