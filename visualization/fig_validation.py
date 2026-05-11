# -*- coding: utf-8 -*-
"""
Figure: Municipality-level technical validation scatter plots
Five-panel figure for Scientific Data manuscript – Table 2.

Panels:
  A  Germany   BEV   (KBA Landkreis, n=398, 2023)
  B  Germany   PHEV  (KBA Landkreis, n=398, 2023)
  C  Norway    BEV   (SSB municipality, n=356, 2023)
  D  Sweden    BEV   (SCB municipality, n=290, 2023)
  E  Sweden    PHEV  (SCB municipality, n=290, 2023)

Each panel: predicted vs actual, 1:1 line, OLS fit, R² / Pearson r / RMSE.
Output: output/fig_validation.png  (300 dpi)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from pathlib import Path

PROJ = Path(__file__).parent.parent

# ── Load comparison tables ─────────────────────────────────────────────────
deu = pd.read_csv(PROJ / "output" / "validation_landkreis" / "landkreis_comparison.csv")
nor = pd.read_csv(PROJ / "output" / "validation_norway"   / "norway_comparison.csv")
swe = pd.read_csv(PROJ / "output" / "validation_sweden"   / "sweden_comparison.csv")

# ── Panel definitions ──────────────────────────────────────────────────────
# (df, x_col, y_col, title, country_label, source_label)
panels = [
    (deu, "BEV_actual",  "BEV_pred",
     "Germany – BEV",  "DEU", "KBA Landkreis  (n=398, 2023)"),
    (deu, "PHEV_actual", "PHEV_pred",
     "Germany – PHEV", "DEU", "KBA Landkreis  (n=398, 2023)"),
    (nor, "BEV_actual",  "BEV_pred",
     "Norway – BEV",   "NOR", "SSB municipality  (n=356, 2023)"),
    (swe, "BEV_actual",  "BEV_pred",
     "Sweden – BEV",   "SWE", "SCB municipality  (n=290, 2023)"),
    (swe, "PHEV_actual", "PHEV_pred",
     "Sweden – PHEV",  "SWE", "SCB municipality  (n=290, 2023)"),
]

COUNTRY_COLOR = {
    "DEU": "#1565C0",   # blue
    "NOR": "#2E7D32",   # green
    "SWE": "#6A1B9A",   # purple
}
PANEL_LABELS = list("ABCDE")

# ── Helper: compute stats ──────────────────────────────────────────────────

def compute_stats(actual, pred):
    mask = np.isfinite(actual) & np.isfinite(pred) & (actual >= 0) & (pred >= 0)
    a, p = actual[mask], pred[mask]
    r, _  = pearsonr(a, p)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2   = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((a - p) ** 2))
    n    = mask.sum()
    return r2, r, rmse, n, a, p


def ols_slope_intercept(a, p):
    """OLS fit: pred = slope * actual + intercept."""
    coeffs = np.polyfit(a, p, 1)
    return coeffs[0], coeffs[1]   # slope, intercept


# ── Layout ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 5))

# Top row: 3 panels (A B C); bottom row: 2 panels centered (D E)
gs_top = fig.add_gridspec(1, 3, left=0.05, right=0.98,
                           top=0.92, bottom=0.52,
                           wspace=0.38)
gs_bot = fig.add_gridspec(1, 4, left=0.05, right=0.98,
                           top=0.44, bottom=0.07,
                           wspace=0.38)

axes = [
    fig.add_subplot(gs_top[0, 0]),   # A – DEU BEV
    fig.add_subplot(gs_top[0, 1]),   # B – DEU PHEV
    fig.add_subplot(gs_top[0, 2]),   # C – NOR BEV
    fig.add_subplot(gs_bot[0, 1]),   # D – SWE BEV
    fig.add_subplot(gs_bot[0, 2]),   # E – SWE PHEV
]

# ── Draw each panel ────────────────────────────────────────────────────────
for ax, (df, x_col, y_col, title, ctry, src_label), label in zip(
        axes, panels, PANEL_LABELS):

    color = COUNTRY_COLOR[ctry]

    # Extract and clean
    actual = df[x_col].values.astype(float)
    pred   = df[y_col].values.astype(float)
    r2, r, rmse, n, a, p = compute_stats(actual, pred)

    # ── 1:1 reference line ────────────────────────────────────────
    xy_max = max(a.max(), p.max()) * 1.08
    xy_min = 0
    ax.plot([xy_min, xy_max], [xy_min, xy_max],
            color="black", lw=1.2, ls="--", zorder=2,
            label="1:1 line")

    # ── OLS fit line ──────────────────────────────────────────────
    slope, intercept = ols_slope_intercept(a, p)
    x_fit = np.linspace(xy_min, a.max(), 300)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color=color, lw=0.7, ls="-",
            zorder=3, label="OLS fit")

    # ── Scatter ───────────────────────────────────────────────────
    ax.scatter(a, p, s=80, alpha=0.3, color=color,
               edgecolors="black", zorder=4, )

    # ── Stats annotation ──────────────────────────────────────────
    stats_txt = (f"$R^2$ = {r2:.3f}\n")
    ax.text(0.97, 0.02, stats_txt,
            transform=ax.transAxes, fontsize=7.5,
            ha="right", va="bottom")
    
    # ── Source label (top-left) ───────────────────────────────────
    ax.text(0.03, 0.97, src_label,
            transform=ax.transAxes, fontsize=6.8,
            ha="left", va="top", color="gray")

    # ── Panel label (A/B/C/…) ─────────────────────────────────────
    ax.text(-0.18, 1.04, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")

    # ── Axes formatting ───────────────────────────────────────────
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.set_xlabel("Actual (vehicles)", fontsize=8)
    ax.set_ylabel("Predicted (vehicles)", fontsize=8)
    ax.tick_params(labelsize=7.5)

    # Compact tick formatter (e.g. 10000 → 10k)
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

    # Legend only on first panel
    if label == "A":
        ax.legend(fontsize=7, frameon=False)

# ── Save ───────────────────────────────────────────────────────────────────
OUT = PROJ / "output" / "fig_validation.png"
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved → {OUT}")
plt.show()
