# -*- coding: utf-8 -*-
"""
Step 01 (v5) – Build two analysis-ready datasets.

Output 1: dataset_city_features.csv
    City × year panel with nighttime lights, GDP, and population.
    GDP and population are linearly interpolated (and extrapolated for
    2021-2023) from 5-year survey data to annual resolution.
    Rows: ~848,900  (65,300 cities × 13 years, 2011-2023)
    Columns: region, UID, year, light_sum, light_mean, gdp, pop,
             gdp_per_cap, light_per_cap, light_mean_per_cap, log_pop

Output 2: dataset_train.csv
    National-level panel ready for model training.
    City features aggregated to country level and merged with national
    vehicle stock counts.
    Rows: 14 countries × 13 years × 4 vehicle types = 728
    Columns: region, year, parameter_pt, value, value_per_cap,
             nat_light, nat_mean, nat_gdp, nat_pop,
             gdp_per_cap, light_per_cap, light_mean_per_cap, log_pop

Inputs
------
data/ntl_road_stats/road_light_{year}_stats.csv  (output of 01_process_ntl.py)
{GDP_DIR}/city_gdp_{year}.csv                    (5-yr gridded GDP, user-configured)
{POP_CSV}                                        (5-yr gridded population, user-configured)
{VEH_CSV}                                        (IEA national vehicle stocks, user-configured)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

# ── USER CONFIGURATION ────────────────────────────────────────────────────
# Set these paths to the locations of your downloaded external datasets.
# See README for data sources.
GDP_DIR = Path("path/to/gdp")                       # gridded GDP CSVs (city_gdp_{year}.csv)
POP_CSV = Path("path/to/eu14_population_.csv")      # gridded population CSV
VEH_CSV = Path("path/to/ev_eu14_with_icev_pt.csv")  # IEA national vehicle stocks
# ──────────────────────────────────────────────────────────────────────────

# NTL statistics are read from the output directory of 01_process_ntl.py.
ROAD_DIR = Path(__file__).parent.parent / "data" / "ntl_road_stats"

OUT_DIR   = Path(__file__).parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

OUT_CITY  = OUT_DIR / "dataset_city_features.csv"
OUT_TRAIN = OUT_DIR / "dataset_train.csv"

YEAR_START, YEAR_END = 2011, 2023
TARGET_YEARS = np.arange(YEAR_START, YEAR_END + 1)
GDP_YEARS    = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
POP_YEARS    = [2000, 2005, 2010, 2015, 2020]


# ── Helper ─────────────────────────────────────────────────────────────────
def annual_interp(known_years, known_values, target_years):
    """Linear interpolation + extrapolation, clipped to >= 0."""
    f = interp1d(known_years, known_values, kind="linear",
                 fill_value="extrapolate", bounds_error=False)
    return np.clip(f(target_years), 0, None)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Nighttime lights  (annual, city level)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading nighttime lights …")
light_frames = []
for yr in TARGET_YEARS:
    df = pd.read_csv(ROAD_DIR / f"road_light_{yr}_stats.csv",
                     usecols=["NAME_0", "UID", "light_sum", "light_mean"])
    for col in ["light_sum", "light_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    df["year"] = yr
    light_frames.append(df.rename(columns={"NAME_0": "region"}))

city_light = pd.concat(light_frames, ignore_index=True)
uid_region = city_light[["UID", "region"]].drop_duplicates("UID")   # UID → country name
print(f"  {city_light['UID'].nunique():,} cities  ×  {len(TARGET_YEARS)} years")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GDP  (5-yr intervals → annual interpolation)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading GDP …")
gdp_frames = []
for yr in GDP_YEARS:
    df = pd.read_csv(GDP_DIR / f"city_gdp_{yr}.csv",
                     usecols=["UID", f"gdp_total_{yr}"],
                     low_memory=False)
    df = df.rename(columns={f"gdp_total_{yr}": yr})
    gdp_frames.append(df.set_index("UID"))

gdp_wide = pd.concat(gdp_frames, axis=1)   # index=UID, cols=GDP years
# Cities with no overlapping pixels have all-NaN → treat as 0
gdp_wide = gdp_wide.fillna(0)

print("  Interpolating GDP to annual …")
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
# 3. Population  (5-yr intervals → annual interpolation)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading population …")
pop_raw = pd.read_csv(POP_CSV)
# Cities with no overlapping pixels have all-NaN → treat as 0
for yr in POP_YEARS:
    pop_raw[f"pop_total_{yr}"] = pop_raw[f"pop_total_{yr}"].fillna(0)

pop_rows = []
for _, row in pop_raw.iterrows():
    uid  = row["UID"]
    vals = np.array([row[f"pop_total_{yr}"] for yr in POP_YEARS], dtype=float)
    mask = np.isfinite(vals)
    annual = annual_interp(np.array(POP_YEARS)[mask], vals[mask], TARGET_YEARS)
    pop_rows.append(
        pd.DataFrame({"UID": uid, "year": TARGET_YEARS, "pop": annual})
    )

city_pop = pd.concat(pop_rows, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Merge → dataset_city_features.csv
# ══════════════════════════════════════════════════════════════════════════════
print("Merging city features …")
city_feat = (
    city_light
    .merge(city_gdp, on=["UID", "year"], how="left")
    .merge(city_pop, on=["UID", "year"], how="left")
    [["region", "UID", "year", "light_sum", "light_mean", "gdp", "pop"]]
    .sort_values(["region", "UID", "year"])
    .reset_index(drop=True)
)

# Per-capita features (set to 0 where pop=0 to avoid div-by-zero)
city_feat["gdp_per_cap"]        = np.where(city_feat["pop"] > 0, city_feat["gdp"]        / city_feat["pop"], 0.0)
city_feat["light_per_cap"]      = np.where(city_feat["pop"] > 0, city_feat["light_sum"]  / city_feat["pop"], 0.0)
city_feat["light_mean_per_cap"] = np.where(city_feat["pop"] > 0, city_feat["light_mean"] / city_feat["pop"], 0.0)
city_feat["log_pop"]            = np.log1p(city_feat["pop"])

city_feat.to_csv(OUT_CITY, index=False, float_format="%.6f")
print(f"  Saved: {OUT_CITY}")
print(f"  Shape: {city_feat.shape}  |  NA gdp: {city_feat['gdp'].isna().sum()}  "
      f"NA pop: {city_feat['pop'].isna().sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Aggregate city → national, merge vehicle targets → dataset_train.csv
# ══════════════════════════════════════════════════════════════════════════════
print("Building training dataset …")

# National feature aggregates
nat_feat = (
    city_feat
    .groupby(["region", "year"], as_index=False)[["light_sum", "light_mean", "gdp", "pop"]]
    .sum()
    .rename(columns={"light_sum": "nat_light", "light_mean": "nat_mean",
                     "gdp": "nat_gdp", "pop": "nat_pop"})
)

# National vehicle stock targets
veh = pd.read_csv(VEH_CSV)
veh = veh[veh["year"].between(YEAR_START, YEAR_END)]
veh_nat = (
    veh.groupby(["region", "parameter_pt", "year"], as_index=False)["value"].sum()
)

# Per-capita national features
nat_feat["gdp_per_cap"]        = np.where(nat_feat["nat_pop"] > 0, nat_feat["nat_gdp"]   / nat_feat["nat_pop"], 0.0)
nat_feat["light_per_cap"]      = np.where(nat_feat["nat_pop"] > 0, nat_feat["nat_light"] / nat_feat["nat_pop"], 0.0)
nat_feat["light_mean_per_cap"] = np.where(nat_feat["nat_pop"] > 0, nat_feat["nat_mean"]  / nat_feat["nat_pop"], 0.0)
nat_feat["log_pop"]            = np.log1p(nat_feat["nat_pop"])

# Merge
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

print("\nPreview – per-capita features:")
print(train_df.groupby("parameter_pt")[["value_per_cap", "gdp_per_cap", "light_per_cap", "light_mean_per_cap"]]
      .describe().round(4).to_string())
