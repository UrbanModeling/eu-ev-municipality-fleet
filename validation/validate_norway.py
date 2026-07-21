# -*- coding: utf-8 -*-
"""
Municipality-level validation of the model for Norway, against SSB
(Statistics Norway) registered BEV data (2023). Independent out-of-sample
check - Norway's data is not used in training.

Data: SSB Table 07849, matched on kommunenummer (= LAU_ID for Norway).

Output: output/validation_norway/
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.parent

VEH_CSV = PROJ / "output" / "eu14_city_vehicles_xgb_2011_2024.csv"
BND_CSV = PROJ / "data" / "boundary" / "eu14_city.csv"
NOR_CSV = PROJ / "data" / "validation" / "nor_07849_20260430-124846.csv"
OUT_DIR = PROJ / "output" / "validation_norway"
OUT_DIR.mkdir(exist_ok=True)

YEAR = 2023

print("Loading model predictions ...")
veh = pd.read_csv(VEH_CSV, low_memory=False)
veh_wide = veh.pivot_table(index=["region", "city", "year"],
                            columns="parameter", values="value",
                            aggfunc="first").reset_index()
veh_wide.columns = [c.replace("EV stock_", "") if c.startswith("EV stock_") else c
                    for c in veh_wide.columns]
veh_no = veh_wide[(veh_wide["region"] == "Norway") & (veh_wide["year"] == YEAR)].copy()
print(f"  Norway cities {YEAR}: {len(veh_no)}")

print("Loading boundary data ...")
# Bridges the LAU 2020 vintage used here to SSB's post-2024-split kommunenummer codes.
bnd_no = pd.read_csv(PROJ / "data" / "boundary" / "no_uid_kommunenummer_2024.csv",
                     dtype={"kommunenummer": str})
bnd_no["kommunenummer"] = bnd_no["kommunenummer"].str.zfill(4)

veh_no = veh_no.merge(bnd_no, left_on="city", right_on="UID", how="left")
print(f"  Cities without boundary match: {veh_no['kommunenummer'].isna().sum()}")

mun_pred = (
    veh_no.groupby("kommunenummer")
    .agg(BEV_pred=("BEV", "sum"), n_cities=("city", "count"))
    .reset_index()
)
print(f"  Kommuner in model: {len(mun_pred)}")

print(f"Loading SSB validation data ({YEAR}) ...")
nor = pd.read_csv(NOR_CSV, sep=";", skiprows=1, encoding="latin-1")
bev_col = f"Private cars {YEAR} Electricity"
total_cols = [c for c in nor.columns if c.startswith(f"Private cars {YEAR}")]

nor["BEV_actual"] = pd.to_numeric(nor[bev_col], errors="coerce").fillna(0)
nor["total_actual"] = nor[total_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

# "region" is like "K-1857 Werøy"; non-municipality rows won't match and are dropped below.
nor["kommunenummer"] = nor["region"].str.extract(r"^K-(\d{4})\b")
print(f"  SSB rows: {len(nor)}  (with 4-digit kommunenummer: {nor['kommunenummer'].notna().sum()})")
print(f"  Total BEV {YEAR}: {nor['BEV_actual'].sum():,.0f}")

merged = mun_pred.merge(
    nor[["kommunenummer", "region", "BEV_actual", "total_actual"]],
    on="kommunenummer", how="inner"
)
print(f"\n  Matched: {len(merged)} / {len(mun_pred)} model, "
      f"{nor['kommunenummer'].notna().sum()} SSB")
unmatched_ssb = nor[nor["kommunenummer"].notna() &
                     ~nor["kommunenummer"].isin(merged["kommunenummer"])]
if len(unmatched_ssb):
    names = [r.encode("ascii", "replace").decode() for r in unmatched_ssb["region"].tolist()]
    print(f"  Unmatched SSB kommuner: {names}")

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
print(f"VALIDATION RESULTS - Norway Municipalities ({YEAR})")
print("=" * 55)
results = []
r = metrics(merged["BEV_actual"], merged["BEV_pred"], "BEV")
if r: results.append(r)

national_ratio = merged["BEV_pred"].sum() / merged["BEV_actual"].sum()
print(f"\n  National BEV ratio (model/SSB): {national_ratio:.3f}")

scale = merged["BEV_actual"].sum() / merged["BEV_pred"].sum()
merged["BEV_pred_scaled"] = merged["BEV_pred"] * scale
print()
r2 = metrics(merged["BEV_actual"], merged["BEV_pred_scaled"], "BEV (scale-adjusted)")
if r2: results.append(r2)

summary = pd.DataFrame(results)
summary.to_csv(OUT_DIR / "validation_summary.csv", index=False)
merged["BEV_ratio"] = merged["BEV_pred"] / merged["BEV_actual"]
merged.to_csv(OUT_DIR / "norway_comparison.csv", index=False, float_format="%.1f")

print(f"\nOutputs saved to: {OUT_DIR}")
