# -*- coding: utf-8 -*-
"""
Municipality-level validation of XGBoost downscaling for Sweden.
Compares model-predicted BEV/PHEV stock against Trafikanalys
"Fordon i lan och kommuner" 2023 data (290 municipalities).

Data source:
  Tabell 4. Personbilar i trafik efter kommun och drivmedel. Ar 2023.
  Columns: El = BEV, Laddhybrider = PHEV

Output: output/validation_sweden/
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.parent

# -- Paths ------------------------------------------------------------------
VEH_CSV  = PROJ / "output" / "eu14_city_vehicles_xgb_2011_2023.csv"
BND_CSV  = PROJ / "data" / "boundary" / "eu14_city.csv"
SWE_XLS  = PROJ / "data" / "validation" / "swe_fordon_lan_och_kommuner_2023.xlsx"
OUT_DIR  = PROJ / "output" / "validation_sweden"
OUT_DIR.mkdir(exist_ok=True)

# -- 1. Load model predictions (Sweden, 2023) --------------------------------
print("Loading model predictions ...")
veh = pd.read_csv(VEH_CSV, low_memory=False)
veh_wide = veh.pivot_table(index=["region", "city", "year"],
                            columns="parameter", values="value",
                            aggfunc="first").reset_index()
veh_wide.columns = [c.replace("EV stock_", "") if c.startswith("EV stock_") else c
                    for c in veh_wide.columns]
veh_sw = veh_wide[(veh_wide["region"] == "Sweden") & (veh_wide["year"] == 2023)].copy()
print(f"  Sweden cities 2023: {len(veh_sw)}")

# -- 2. Load boundary -> NAME_2 (municipality name) -------------------------
print("Loading boundary data ...")
bnd = pd.read_csv(BND_CSV, low_memory=False)
bnd_sw = bnd[bnd["GID_0"] == "SWE"][["UID", "NAME_2"]].copy()
bnd_sw["NAME_2"] = bnd_sw["NAME_2"].astype(str).str.strip()

# Merge city -> municipality name
veh_sw = veh_sw.merge(bnd_sw, left_on="city", right_on="UID", how="left")
missing = veh_sw["NAME_2"].isna().sum()
print(f"  Cities without municipality match: {missing} / {len(veh_sw)}")

# -- 3. Aggregate to municipality -------------------------------------------
mun_pred = (
    veh_sw.groupby("NAME_2")
    .agg(BEV_pred=("BEV", "sum"), PHEV_pred=("PHEV", "sum"),
         ICEV_pred=("ICEV", "sum"), n_cities=("city", "count"))
    .reset_index()
)
mun_pred["EV_pred"] = mun_pred["BEV_pred"] + mun_pred["PHEV_pred"]
print(f"  Municipalities in model: {len(mun_pred)}")

# -- 4. Load Trafikanalys data (2023) ----------------------------------------
print("Loading Trafikanalys validation data (2023) ...")
df = pd.read_excel(SWE_XLS, sheet_name="Tabell 4 Personbil",
                   header=None, skiprows=7)
df.columns = ["Kommunkod", "Kommun", "Bensin", "Diesel", "El",
              "Elhybrider", "Laddhybrider", "Etanol", "Gas", "Ovriga", "Totalt"]
df = df[pd.to_numeric(df["Kommunkod"], errors="coerce").notna()].copy()
df["Kommunkod"] = df["Kommunkod"].astype(int)
df["Kommun"] = df["Kommun"].astype(str).str.strip()

# Convert El and Laddhybrider to numeric (some cells may have '-' for zero)
for col in ["El", "Laddhybrider", "Totalt"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df.rename(columns={"El": "BEV_actual", "Laddhybrider": "PHEV_actual",
                   "Totalt": "total_actual"}, inplace=True)
df["EV_actual"] = df["BEV_actual"] + df["PHEV_actual"]

print(f"  Municipalities in Trafikanalys: {len(df)}")
print(f"  Total BEV: {df['BEV_actual'].sum():,.0f}")
print(f"  Total PHEV: {df['PHEV_actual'].sum():,.0f}")

# -- 5. Name harmonisation (2 known mismatches) -----------------------------
name_map = {
    "Upplands Väsby": "Upplands-Väsby",   # hyphen variant in boundary
    "Malung-Sälen":   "Malung",           # post-merger name; boundary uses old name
}
df["Kommun_match"] = df["Kommun"].replace(name_map)

# -- 6. Join ----------------------------------------------------------------
merged = mun_pred.merge(
    df[["Kommun_match", "Kommun", "BEV_actual", "PHEV_actual",
        "EV_actual", "total_actual"]],
    left_on="NAME_2", right_on="Kommun_match", how="inner"
)
print(f"\n  Matched municipalities: {len(merged)} / {len(mun_pred)} model, {len(df)} Trafikanalys")

# -- 7. Metrics -------------------------------------------------------------
def metrics(actual, pred, label):
    mask = (actual > 0) & np.isfinite(actual) & np.isfinite(pred)
    a, p = actual[mask], pred[mask]
    n = mask.sum()
    corr = np.corrcoef(a, p)[0, 1]
    ss_res = ((a - p) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
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
print("\n" + "=" * 55)
print("VALIDATION RESULTS -- Sweden Municipalities (2023)")
print("=" * 55)
results.append(metrics(merged["BEV_actual"],  merged["BEV_pred"],  "BEV"))
results.append(metrics(merged["PHEV_actual"], merged["PHEV_pred"], "PHEV"))
results.append(metrics(merged["EV_actual"],   merged["EV_pred"],   "Total EV"))

# National ratio
national_bev_ratio = merged["BEV_pred"].sum() / merged["BEV_actual"].sum()
print(f"\n  National BEV ratio (model/Trafikanalys): {national_bev_ratio:.3f}")

# Scale-adjusted R2
scale = merged["BEV_actual"].sum() / merged["BEV_pred"].sum()
merged["BEV_pred_scaled"] = merged["BEV_pred"] * scale
print()
metrics(merged["BEV_actual"], merged["BEV_pred_scaled"], "BEV (scale-adjusted)")

# -- 8. Save outputs --------------------------------------------------------
summary = pd.DataFrame(results)
summary.to_csv(OUT_DIR / "validation_summary.csv", index=False)

merged_out = merged[["NAME_2", "Kommun",
                      "BEV_pred", "BEV_actual",
                      "PHEV_pred", "PHEV_actual",
                      "EV_pred", "EV_actual",
                      "total_actual", "n_cities"]].copy()
merged_out["BEV_ratio"] = merged_out["BEV_pred"] / merged_out["BEV_actual"]
merged_out.to_csv(OUT_DIR / "sweden_comparison.csv", index=False, float_format="%.1f")

print("\n--- Top 10 overestimated municipalities (model/actual) ---")
over = merged_out.nlargest(10, "BEV_ratio")[["NAME_2", "BEV_pred", "BEV_actual", "BEV_ratio"]]
print(over.to_string(index=False))

print("\n--- Top 10 underestimated municipalities ---")
under = merged_out.nsmallest(10, "BEV_ratio")[["NAME_2", "BEV_pred", "BEV_actual", "BEV_ratio"]]
print(under.to_string(index=False))

print(f"\nOutputs saved to: {OUT_DIR}")
