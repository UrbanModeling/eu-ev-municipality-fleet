# -*- coding: utf-8 -*-
"""
Step 03 - Build two analysis-ready datasets.

Output 1: dataset_city_features.csv - city x year panel (light/GDP/pop
    features; GDP and population interpolated to annual from 5-year data)
Output 2: dataset_train.csv - national x year panel for model training

Inputs: data/road/road_light_{year}_stats.csv, data/gdp/city_gdp_{year}.csv,
data/pop/city_pop_{year}.csv (from 01/02), EV data by country 2026.xlsx
(IEA Global EV Outlook, see README)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

PROJ_DIR = Path(__file__).parent.parent
DATA_DIR = PROJ_DIR / "data"

ROAD_DIR = DATA_DIR / "road"
GDP_DIR  = DATA_DIR / "gdp"
POP_DIR  = DATA_DIR / "pop"
EV_XLSX  = Path("path/to/EV data by country 2026.xlsx")  # IEA Global EV Outlook 2026, see README

OUT_CITY  = DATA_DIR / "dataset_city_features.csv"
OUT_TRAIN = DATA_DIR / "dataset_train.csv"

YEAR_START, YEAR_END = 2011, 2024
TARGET_YEARS = np.arange(YEAR_START, YEAR_END + 1)
GDP_YEARS    = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
POP_YEARS    = [2000, 2005, 2010, 2015, 2020]
EU14_REGIONS = [
    "Austria", "Belgium", "Denmark", "Finland", "France", "Germany", "Italy",
    "Netherlands", "Norway", "Portugal", "Spain", "Sweden", "Switzerland",
    "United Kingdom",
]


def annual_interp(known_years, known_values, target_years):
    """Linear interpolation + extrapolation, clipped to >= 0."""
    f = interp1d(known_years, known_values, kind="linear",
                 fill_value="extrapolate", bounds_error=False)
    return np.clip(f(target_years), 0, None)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Nighttime lights  (annual, city level)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading nighttime lights ...")
light_frames = []
for yr in TARGET_YEARS:
    df = pd.read_csv(ROAD_DIR / f"road_light_{yr}_stats.csv",
                     usecols=["NAME_0", "UID", "light_sum", "light_mean"])
    for col in ["light_sum", "light_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    df["year"] = yr
    light_frames.append(df.rename(columns={"NAME_0": "region"}))

city_light = pd.concat(light_frames, ignore_index=True)
print(f"  {city_light['UID'].nunique():,} cities  x  {len(TARGET_YEARS)} years")
print(f"  nonzero light_sum fraction ({YEAR_END}): "
      f"{(city_light[city_light['year']==YEAR_END]['light_sum']>0).mean():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GDP  (5-yr intervals -> annual interpolation)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading GDP ...")
gdp_frames = []
for yr in GDP_YEARS:
    df = pd.read_csv(GDP_DIR / f"city_gdp_{yr}.csv",
                     usecols=["UID", f"gdp_total_{yr}"],
                     low_memory=False)
    df = df.rename(columns={f"gdp_total_{yr}": yr})
    gdp_frames.append(df.set_index("UID"))

gdp_wide = pd.concat(gdp_frames, axis=1)
gdp_wide = gdp_wide.fillna(0)

print("  Interpolating GDP to annual ...")
gdp_rows = []
for uid, row in gdp_wide.iterrows():
    vals = row.values.astype(float)
    mask = np.isfinite(vals)
    annual = annual_interp(np.array(GDP_YEARS)[mask], vals[mask], TARGET_YEARS)
    gdp_rows.append(
        pd.DataFrame({"UID": uid, "year": TARGET_YEARS, "gdp": annual})
    )

city_gdp = pd.concat(gdp_rows, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Population  (5-yr intervals -> annual interpolation)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading population ...")
pop_frames = []
for yr in POP_YEARS:
    df = pd.read_csv(POP_DIR / f"city_pop_{yr}.csv",
                     usecols=["UID", f"pop_total_{yr}"],
                     low_memory=False)
    df = df.rename(columns={f"pop_total_{yr}": yr})
    pop_frames.append(df.set_index("UID"))

pop_wide = pd.concat(pop_frames, axis=1)
pop_wide = pop_wide.fillna(0)

print("  Interpolating population to annual ...")
pop_rows = []
for uid, row in pop_wide.iterrows():
    vals = row.values.astype(float)
    mask = np.isfinite(vals)
    annual = annual_interp(np.array(POP_YEARS)[mask], vals[mask], TARGET_YEARS)
    pop_rows.append(
        pd.DataFrame({"UID": uid, "year": TARGET_YEARS, "pop": annual})
    )

city_pop = pd.concat(pop_rows, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Merge -> dataset_city_features.csv
# ══════════════════════════════════════════════════════════════════════════════
print("Merging city features ...")
city_feat = (
    city_light
    .merge(city_gdp, on=["UID", "year"], how="left")
    .merge(city_pop, on=["UID", "year"], how="left")
    [["region", "UID", "year", "light_sum", "light_mean", "gdp", "pop"]]
    .sort_values(["region", "UID", "year"])
    .reset_index(drop=True)
)

city_feat["gdp_per_cap"]        = np.where(city_feat["pop"] > 0, city_feat["gdp"]        / city_feat["pop"], 0.0)
city_feat["light_per_cap"]      = np.where(city_feat["pop"] > 0, city_feat["light_sum"]  / city_feat["pop"], 0.0)
city_feat["light_mean_per_cap"] = np.where(city_feat["pop"] > 0, city_feat["light_mean"] / city_feat["pop"], 0.0)
city_feat["log_pop"]            = np.log1p(city_feat["pop"])

city_feat.to_csv(OUT_CITY, index=False, float_format="%.6f")
print(f"  Saved: {OUT_CITY}")
print(f"  Shape: {city_feat.shape}  |  NA gdp: {city_feat['gdp'].isna().sum()}  "
      f"NA pop: {city_feat['pop'].isna().sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Aggregate city -> national, merge vehicle targets -> dataset_train.csv
# ══════════════════════════════════════════════════════════════════════════════
print("Building training dataset ...")

nat_feat = (
    city_feat
    .groupby(["region", "year"], as_index=False)[["light_sum", "light_mean", "gdp", "pop"]]
    .sum()
    .rename(columns={"light_sum": "nat_light", "light_mean": "nat_mean",
                     "gdp": "nat_gdp", "pop": "nat_pop"})
)

print("Loading vehicle stock targets (IEA GEVO 2026, raw) ...")
gevo = pd.read_excel(EV_XLSX, sheet_name="GEVO_EV_2026")
gevo = gevo[(gevo["mode"] == "Cars") & (gevo["category"] == "Historical") &
            (gevo["region_country"].isin(EU14_REGIONS))]

stock = gevo[gevo["parameter"] == "EV stock"]
share = gevo[gevo["parameter"] == "EV stock share"]

# BEV / PHEV / FCEV stock -> direct from the `EV stock` parameter
direct = stock[stock["powertrain"].isin(["BEV", "PHEV", "FCEV"])].copy()
direct["parameter_pt"] = "EV stock_" + direct["powertrain"]
direct = direct.rename(columns={"region_country": "region"})
direct = direct[["region", "parameter_pt", "year", "value"]]

# ICEV has no direct series; derive via total_stock = ev_total / (share/100).
ev_total = stock[stock["powertrain"] == "EV"][["region_country", "year", "value"]] \
    .rename(columns={"value": "ev_total"})
share_v = share[["region_country", "year", "value"]].rename(columns={"value": "share"})
icev = ev_total.merge(share_v, on=["region_country", "year"])
icev["value"] = icev["ev_total"] / (icev["share"] / 100) - icev["ev_total"]
icev = icev.rename(columns={"region_country": "region"})
icev["parameter_pt"] = "EV stock_ICEV"
icev = icev[["region", "parameter_pt", "year", "value"]]

veh = pd.concat([direct, icev], ignore_index=True)
veh = veh[veh["year"].between(YEAR_START, YEAR_END)]
veh_nat = (
    veh.groupby(["region", "parameter_pt", "year"], as_index=False)["value"].sum()
)

nat_feat["gdp_per_cap"]        = np.where(nat_feat["nat_pop"] > 0, nat_feat["nat_gdp"]   / nat_feat["nat_pop"], 0.0)
nat_feat["light_per_cap"]      = np.where(nat_feat["nat_pop"] > 0, nat_feat["nat_light"] / nat_feat["nat_pop"], 0.0)
nat_feat["light_mean_per_cap"] = np.where(nat_feat["nat_pop"] > 0, nat_feat["nat_mean"]  / nat_feat["nat_pop"], 0.0)
nat_feat["log_pop"]            = np.log1p(nat_feat["nat_pop"])

train_df = (
    veh_nat
    .merge(nat_feat, on=["region", "year"], how="inner")
    .dropna(subset=["nat_light", "nat_gdp", "nat_pop", "value"])
)
train_df["value_per_cap"] = np.where(
    train_df["nat_pop"] > 0,
    train_df["value"] / train_df["nat_pop"],
    0.0
)
train_df = (
    train_df[["region", "year", "parameter_pt", "value", "value_per_cap",
              "nat_light", "nat_mean", "nat_gdp", "nat_pop",
              "gdp_per_cap", "light_per_cap", "light_mean_per_cap", "log_pop"]]
    .sort_values(["region", "year", "parameter_pt"])
    .reset_index(drop=True)
)

train_df.to_csv(OUT_TRAIN, index=False, float_format="%.6f")
print(f"  Saved: {OUT_TRAIN}")
print(f"  Shape: {train_df.shape}")

print("\nPreview - per-capita features:")
print(train_df.groupby("parameter_pt")[["value_per_cap", "gdp_per_cap", "light_per_cap", "light_mean_per_cap"]]
      .describe().round(4).to_string())
