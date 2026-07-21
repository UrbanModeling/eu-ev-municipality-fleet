# -*- coding: utf-8 -*-
"""
Municipality-level validation of the model for Sweden, against
Trafikanalys "Fordon i lan och kommuner" 2023 data (290 municipalities).
Independent out-of-sample check - Sweden's data is not used in training.
Matched on Kommunkod (= LAU_ID for Sweden).

Output: output/validation_sweden/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.parent

VEH_CSV = PROJ / "output" / "eu14_city_vehicles_xgb_2011_2024.csv"
BND_CSV = PROJ / "data" / "boundary" / "eu14_city.csv"
SWE_XLS = PROJ / "data" / "validation" / "swe_fordon_lan_och_kommuner_2023.xlsx"
OUT_DIR = PROJ / "output" / "validation_sweden"
OUT_DIR.mkdir(exist_ok=True)

print("Loading model predictions ...")
veh = pd.read_csv(VEH_CSV, low_memory=False)
veh_wide = veh.pivot_table(index=["region", "city", "year"],
                            columns="parameter", values="value",
                            aggfunc="first").reset_index()
veh_wide.columns = [c.replace("EV stock_", "") if c.startswith("EV stock_") else c
                    for c in veh_wide.columns]
veh_sw = veh_wide[(veh_wide["region"] == "Sweden") & (veh_wide["year"] == 2023)].copy()
print(f"  Sweden cities 2023: {len(veh_sw)}")

print("Loading boundary data ...")
bnd = pd.read_csv(BND_CSV, low_memory=False, dtype={"LAU_ID": str})
bnd_sw = bnd[bnd["GID_0"] == "SWE"][["UID", "LAU_ID"]].copy()
bnd_sw["Kommunkod"] = bnd_sw["LAU_ID"].str.zfill(4)

veh_sw = veh_sw.merge(bnd_sw, left_on="city", right_on="UID", how="left")
missing = veh_sw["Kommunkod"].isna().sum()
print(f"  Cities without municipality match: {missing} / {len(veh_sw)}")

mun_pred = (
    veh_sw.groupby("Kommunkod")
    .agg(BEV_pred=("BEV", "sum"), PHEV_pred=("PHEV", "sum"),
         ICEV_pred=("ICEV", "sum"), n_cities=("city", "count"))
    .reset_index()
)
mun_pred["EV_pred"] = mun_pred["BEV_pred"] + mun_pred["PHEV_pred"]
print(f"  Municipalities in model: {len(mun_pred)}")

print("Loading Trafikanalys validation data (2023) ...")
df = pd.read_excel(SWE_XLS, sheet_name="Tabell 4 Personbil",
                   header=None, skiprows=7)
df.columns = ["Kommunkod", "Kommun", "Bensin", "Diesel", "El",
              "Elhybrider", "Laddhybrider", "Etanol", "Gas", "Ovriga", "Totalt"]
df = df[pd.to_numeric(df["Kommunkod"], errors="coerce").notna()].copy()
df["Kommunkod"] = df["Kommunkod"].astype(int).astype(str).str.zfill(4)

for col in ["El", "Laddhybrider", "Totalt"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df.rename(columns={"El": "BEV_actual", "Laddhybrider": "PHEV_actual",
                   "Totalt": "total_actual"}, inplace=True)
df["EV_actual"] = df["BEV_actual"] + df["PHEV_actual"]

print(f"  Municipalities in Trafikanalys: {len(df)}")
print(f"  Total BEV: {df['BEV_actual'].sum():,.0f}")

merged = mun_pred.merge(
    df[["Kommunkod", "BEV_actual", "PHEV_actual", "EV_actual", "total_actual"]],
    on="Kommunkod", how="inner"
)
print(f"\n  Matched municipalities: {len(merged)} / {len(mun_pred)} model, {len(df)} Trafikanalys")

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
    return {"label": label, "n": int(n), "r2": r2, "r": corr,
            "mape": mape, "mae": mae, "bias": bias}

print("\n" + "=" * 55)
print("VALIDATION RESULTS - Sweden Municipalities (2023)")
print("=" * 55)
results = []
results.append(metrics(merged["BEV_actual"], merged["BEV_pred"], "BEV"))
results.append(metrics(merged["PHEV_actual"], merged["PHEV_pred"], "PHEV"))
results.append(metrics(merged["EV_actual"], merged["EV_pred"], "Total EV"))

national_ratio = merged["BEV_pred"].sum() / merged["BEV_actual"].sum()
print(f"\n  National BEV ratio (model/Trafikanalys): {national_ratio:.3f}")

scale = merged["BEV_actual"].sum() / merged["BEV_pred"].sum()
merged["BEV_pred_scaled"] = merged["BEV_pred"] * scale
print()
r2 = metrics(merged["BEV_actual"], merged["BEV_pred_scaled"], "BEV (scale-adjusted)")
if r2: results.append(r2)

summary = pd.DataFrame(results)
summary.to_csv(OUT_DIR / "validation_summary.csv", index=False)
merged["BEV_ratio"] = merged["BEV_pred"] / merged["BEV_actual"]
merged.to_csv(OUT_DIR / "sweden_comparison.csv", index=False, float_format="%.1f")

print(f"\nOutputs saved to: {OUT_DIR}")
