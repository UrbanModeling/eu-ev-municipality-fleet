# -*- coding: utf-8 -*-
"""
Figure: Actual vs. predicted BEV stock, 2011-2023, for three representative
Swedish municipalities (small/medium/large by population) - a time-series
view of validation accuracy vs. municipality size.

Output: output/fig_sweden_timeseries_examples.png (300 dpi)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

PROJ = Path(__file__).parent.parent
VEH_CSV = PROJ / "output" / "eu14_city_vehicles_xgb_2011_2024.csv"
BND_CSV = PROJ / "data" / "boundary" / "eu14_city.csv"
ACT_CSV = PROJ / "data" / "validation" / "swe_actual_stock_2011_2023.csv"
OUT_PNG = PROJ / "output" / "fig_sweden_timeseries_examples.png"

rcParams['font.family']       = 'Arial'
rcParams['font.size']         = 7
rcParams['axes.linewidth']    = 0.6
rcParams['xtick.major.width'] = 0.6
rcParams['ytick.major.width'] = 0.6
rcParams['xtick.major.size']  = 2.5
rcParams['ytick.major.size']  = 2.5

COLOR_ACTUAL = "#3C5488"
COLOR_PRED   = "#F39B7F"

# (kommun name, kommunkod, subtitle) - tiers: <10,000 / 10,000-50,000 / 50,000+.
TARGETS = [
    ("Grastorp", 1444, "Small (pop. 5,693)"),
    ("Markaryd", 767,  "Medium (pop. 10,320)"),
    ("Uppsala",  380,  "Large (pop. 242,052)"),
]

print("Loading model predictions ...")
veh = pd.read_csv(VEH_CSV, low_memory=False)
veh_wide = veh.pivot_table(index=["region", "city", "year"], columns="parameter",
                           values="value", aggfunc="first").reset_index()
veh_wide.columns = [c.replace("EV stock_", "") if c.startswith("EV stock_") else c
                    for c in veh_wide.columns]
veh_sw = veh_wide[veh_wide["region"] == "Sweden"].copy()

print("Loading boundary (Sweden kommunkod) ...")
bnd = pd.read_csv(BND_CSV, low_memory=False, dtype={"LAU_ID": str})
bnd_sw = bnd[bnd["GID_0"] == "SWE"][["UID", "LAU_ID"]].copy()
bnd_sw["kommunkod"] = bnd_sw["LAU_ID"].str.zfill(4).astype(int)
veh_sw = veh_sw.merge(bnd_sw, left_on="city", right_on="UID", how="left")

pred = veh_sw.groupby(["kommunkod", "year"]).agg(BEV_pred=("BEV", "sum")).reset_index()

print("Loading Trafikanalys actual stock (2011-2023) ...")
act = pd.read_csv(ACT_CSV).rename(columns={"BEV": "BEV_actual"})

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), facecolor='white')

for ax, (name, kk, subtitle) in zip(axes, TARGETS):
    a = act[act["kommunkod"] == kk][["year", "BEV_actual"]]
    p = pred[pred["kommunkod"] == kk][["year", "BEV_pred"]]
    m = a.merge(p, on="year", how="inner").sort_values("year")
    m = m[m["year"].between(2011, 2023)]

    aa = m["BEV_actual"].values.astype(float)
    pp = m["BEV_pred"].values.astype(float)
    mask = aa > 0
    ss_tot = ((aa[mask] - aa[mask].mean()) ** 2).sum()
    ss_res = ((aa[mask] - pp[mask]) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    ax.plot(m["year"], m["BEV_actual"], color=COLOR_ACTUAL, linewidth=1.6,
            marker='o', markersize=3)
    ax.plot(m["year"], m["BEV_pred"], color=COLOR_PRED, linewidth=1.6,
            linestyle='--', marker='s', markersize=3)

    ax.set_title(f"{name}\n({subtitle}, R²={r2:.2f})", fontsize=7.5, pad=6)
    ax.set_xlabel("Year", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xticks([2011, 2015, 2019, 2023])

axes[0].set_ylabel("BEV stock (vehicles)", fontsize=7)

handles = [
    plt.Line2D([0], [0], color=COLOR_ACTUAL, linewidth=1.6, marker='o', markersize=3, label='Actual (registered)'),
    plt.Line2D([0], [0], color=COLOR_PRED, linewidth=1.6, linestyle='--', marker='s', markersize=3, label='Predicted (model)'),
]
fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
           fontsize=7, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_PNG}")
