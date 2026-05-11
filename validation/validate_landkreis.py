# -*- coding: utf-8 -*-
"""
Landkreis-level validation of XGBoost downscaling for Germany.
Compares model-predicted BEV/PHEV stock against KBA Zulassungsbezirk data
(~399 districts, much more granular than the existing 38 NUTS2 regions).

KBA data source:
  FZ Pkw mit Elektroantrieb Zulassungsbezirk (Berichtszeitpunkt=2023.10)
  BEV/PHEV counts derived from: Pkw Insgesamt × BEV/PHEV Anteil / 100

Output: output/validation_landkreis/
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.parent

# ── Paths ──────────────────────────────────────────────────────────────────
VEH_CSV  = PROJ / "output" / "eu14_city_vehicles_xgb_2011_2023.csv"
BND_CSV  = PROJ / "data" / "boundary" / "eu14_city.csv"
KBA_CSV  = PROJ / "data" / "validation" / "deu_FZ Pkw mit Elektroantrieb Zulassungsbezirk_2102999911177145523.csv"
OUT_DIR  = PROJ / "output" / "validation_landkreis"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load model predictions (Germany, 2023) ──────────────────────────────
print("Loading model predictions ...")
veh = pd.read_csv(VEH_CSV, low_memory=False)
# Pivot from long to wide
veh_wide = veh.pivot_table(index=["region","city","year"],
                            columns="parameter", values="value",
                            aggfunc="first").reset_index()
veh_wide.columns = [c.replace("EV stock_", "") if c.startswith("EV stock_") else c
                    for c in veh_wide.columns]
veh_de = veh_wide[(veh_wide["region"] == "Germany") & (veh_wide["year"] == 2023)].copy()
print(f"  Germany cities 2023: {len(veh_de)}")

# ── 2. Load boundary → CC_2 (5-digit AGS = Landkreis) ─────────────────────
print("Loading boundary data ...")
bnd = pd.read_csv(BND_CSV, low_memory=False)
bnd_de = bnd[bnd["GID_0"] == "DEU"][["UID", "NAME_2", "CC_2"]].copy()
bnd_de["CC_2"] = bnd_de["CC_2"].astype(str).str.zfill(5)

# Merge city → Landkreis
veh_de = veh_de.merge(bnd_de, left_on="city", right_on="UID", how="left")
missing = veh_de["CC_2"].isna().sum()
print(f"  Cities without Landkreis match: {missing} / {len(veh_de)}")

# ── 3. Aggregate to Landkreis ──────────────────────────────────────────────
lk_pred = (
    veh_de.groupby(["CC_2", "NAME_2"])
    .agg(BEV_pred=("BEV", "sum"), PHEV_pred=("PHEV", "sum"),
         ICEV_pred=("ICEV", "sum"), n_cities=("city", "count"))
    .reset_index()
)
lk_pred["EV_pred"] = lk_pred["BEV_pred"] + lk_pred["PHEV_pred"]
print(f"  Landkreise in model: {len(lk_pred)}")

# ── 4. Load KBA data (2023.10 = closest to year-end 2023) ─────────────────
print("Loading KBA validation data (2023.10) ...")
kba = pd.read_csv(KBA_CSV, encoding="utf-8-sig")
kba_23 = kba[kba["Berichtszeitpunkt"] == 2023.10].copy()

# Parse AGS key (may have leading zeros)
schluessel_col = [c for c in kba_23.columns if "Schl" in c][0]
kba_23["CC_2"] = kba_23[schluessel_col].astype(str).str.zfill(5)

# Derive BEV / PHEV counts from shares × total
kba_23["BEV_actual"]  = kba_23["Pkw Insgesamt"] * kba_23["Pkw BEV Anteil"]  / 100
kba_23["PHEV_actual"] = kba_23["Pkw Insgesamt"] * kba_23["Pkw Plug In Hybrid Anteil"] / 100
kba_23["EV_actual"]   = kba_23["BEV_actual"] + kba_23["PHEV_actual"]
kba_23["total_actual"]= kba_23["Pkw Insgesamt"]

print(f"  KBA districts: {len(kba_23)}")
print(f"  KBA total BEV (computed): {kba_23['BEV_actual'].sum():,.0f}")
print(f"  KBA total PHEV (computed): {kba_23['PHEV_actual'].sum():,.0f}")

# ── 5. Join ────────────────────────────────────────────────────────────────
merged = lk_pred.merge(
    kba_23[["CC_2", "Zulassungsbezirk", "BEV_actual", "PHEV_actual",
             "EV_actual", "total_actual"]],
    on="CC_2", how="inner"
)
print(f"\n  Matched districts: {len(merged)} / {len(lk_pred)} model, {len(kba_23)} KBA")

# ── 6. Metrics ────────────────────────────────────────────────────────────
def metrics(actual, pred, label):
    mask = (actual > 0) & np.isfinite(actual) & np.isfinite(pred)
    a, p = actual[mask], pred[mask]
    n = mask.sum()
    corr = np.corrcoef(a, p)[0,1]
    ss_res = ((a - p)**2).sum()
    ss_tot = ((a - a.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    mape = (np.abs(a - p) / a).mean() * 100
    mae  = np.abs(a - p).mean()
    bias = (p - a).mean()
    print(f"\n  [{label}]  n={n}")
    print(f"    R2   = {r2:.3f}")
    print(f"    r    = {corr:.3f}")
    print(f"    MAPE = {mape:.1f}%")
    print(f"    MAE  = {mae:,.0f} vehicles")
    print(f"    Bias = {bias:+,.0f} vehicles (pred - actual)")
    return {"label": label, "n": n, "r2": r2, "r": corr,
            "mape": mape, "mae": mae, "bias": bias}

results = []
print("\n" + "="*55)
print("VALIDATION RESULTS — Germany Landkreis (2023.10)")
print("="*55)
results.append(metrics(merged["BEV_actual"],  merged["BEV_pred"],  "BEV"))
results.append(metrics(merged["PHEV_actual"], merged["PHEV_pred"], "PHEV"))
results.append(metrics(merged["EV_actual"],   merged["EV_pred"],   "Total EV"))

# ── 7. Scale note ─────────────────────────────────────────────────────────
# KBA is 2023.10 (October); model uses annual ACEA total (≈ year-end)
# → expect model to overestimate by ~15% nationally
national_bev_ratio = merged["BEV_pred"].sum() / merged["BEV_actual"].sum()
print(f"\n  National BEV ratio (model/KBA): {national_bev_ratio:.3f}")
print(f"  (>1 expected: model uses year-end, KBA uses Oct snapshot)")

# ── 8. Scale-adjusted R² (remove national bias) ───────────────────────────
scale = merged["BEV_actual"].sum() / merged["BEV_pred"].sum()
merged["BEV_pred_scaled"] = merged["BEV_pred"] * scale
print()
metrics(merged["BEV_actual"], merged["BEV_pred_scaled"], "BEV (scale-adjusted)")

# ── 9. Save outputs ────────────────────────────────────────────────────────
summary = pd.DataFrame(results)
summary.to_csv(OUT_DIR / "validation_summary.csv", index=False)

merged_out = merged[["CC_2", "NAME_2", "Zulassungsbezirk",
                      "BEV_pred", "BEV_actual",
                      "PHEV_pred", "PHEV_actual",
                      "EV_pred", "EV_actual",
                      "total_actual", "n_cities"]].copy()
merged_out["BEV_ratio"] = merged_out["BEV_pred"] / merged_out["BEV_actual"]
merged_out.to_csv(OUT_DIR / "landkreis_comparison.csv", index=False, float_format="%.1f")

# Top over/under estimated
print("\n--- Top 10 overestimated Landkreise (model/actual) ---")
over = merged_out.nlargest(10, "BEV_ratio")[["NAME_2","BEV_pred","BEV_actual","BEV_ratio"]]
print(over.to_string(index=False))

print("\n--- Top 10 underestimated Landkreise ---")
under = merged_out.nsmallest(10, "BEV_ratio")[["NAME_2","BEV_pred","BEV_actual","BEV_ratio"]]
print(under.to_string(index=False))

print(f"\nOutputs saved to: {OUT_DIR}")
