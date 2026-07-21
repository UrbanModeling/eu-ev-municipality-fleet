# -*- coding: utf-8 -*-
"""
Figure: Municipality-level validation scatter plots, predicted vs actual
BEV stock with 1:1 line and OLS fit. Panel A: Norway (SSB, n=356, 2023).
Panel B: Sweden (SCB, n=290, 2023). Germany excluded - its KBA data
augments training, so it isn't an independent validation.

Output: output/fig_validation.png  (300 dpi)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr
from pathlib import Path

PROJ = Path(__file__).parent.parent

nor = pd.read_csv(PROJ / "output" / "validation_norway" / "norway_comparison.csv")
swe = pd.read_csv(PROJ / "output" / "validation_sweden" / "sweden_comparison.csv")

panels = [
    (nor, "BEV_actual", "BEV_pred",
     "Norway - BEV", "NOR", "SSB municipality  (n=356, 2023)"),
    (swe, "BEV_actual", "BEV_pred",
     "Sweden - BEV", "SWE", "SCB municipality  (n=290, 2023)"),
]

COUNTRY_COLOR = {
    "NOR": "#2E7D32",   # green
    "SWE": "#6A1B9A",   # purple
}
PANEL_LABELS = list("AB")


def compute_stats(actual, pred):
    mask = np.isfinite(actual) & np.isfinite(pred) & (actual >= 0) & (pred >= 0)
    a, p = actual[mask], pred[mask]
    r, _ = pearsonr(a, p)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((a - p) ** 2))
    n = mask.sum()
    return r2, r, rmse, n, a, p


def ols_slope_intercept(a, p):
    coeffs = np.polyfit(a, p, 1)
    return coeffs[0], coeffs[1]


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))

for ax, (df, x_col, y_col, title, ctry, src_label), label in zip(
        axes, panels, PANEL_LABELS):

    color = COUNTRY_COLOR[ctry]

    actual = df[x_col].values.astype(float)
    pred = df[y_col].values.astype(float)
    r2, r, rmse, n, a, p = compute_stats(actual, pred)

    xy_max = max(a.max(), p.max()) * 1.08
    xy_min = 0
    ax.plot([xy_min, xy_max], [xy_min, xy_max],
            color="black", lw=1.2, ls="--", zorder=2, label="1:1 line")

    slope, intercept = ols_slope_intercept(a, p)
    x_fit = np.linspace(xy_min, a.max(), 300)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color=color, lw=0.7, ls="-", zorder=3, label="OLS fit")

    ax.scatter(a, p, s=80, alpha=0.3, color=color, edgecolors="black", zorder=4)

    stats_txt = f"$R^2$ = {r2:.3f}\n"
    ax.text(0.97, 0.02, stats_txt, transform=ax.transAxes, fontsize=8,
            ha="right", va="bottom")

    ax.text(0.03, 0.97, src_label, transform=ax.transAxes, fontsize=7.5,
            ha="left", va="top", color="gray")

    ax.text(-0.18, 1.04, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("Actual (vehicles)", fontsize=9)
    ax.set_ylabel("Predicted (vehicles)", fontsize=9)
    ax.tick_params(labelsize=8)

    def _fmt_k(x, pos):
        if x >= 1e6:
            return f"{x/1e6:.0f}M"
        if x >= 1e3:
            return f"{x/1e3:.0f}k"
        return f"{x:.0f}"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))

    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    if label == "A":
        ax.legend(fontsize=7.5, frameon=False)

plt.tight_layout()
OUT = PROJ / "output" / "fig_validation.png"
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved -> {OUT}")
