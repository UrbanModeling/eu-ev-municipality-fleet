# -*- coding: utf-8 -*-
"""
Step 05 - Build analysis-ready result datasets (vehicle stocks only).

Output 1: output/result_city.gpkg - city x year panel with geometry,
    internal use only (LAU terms of use don't permit redistributing it)
Output 2: output/result_city.csv - same, without geometry; deposited to
    figshare, joined via gisco_id to a user's own copy of LAU boundaries

Inputs: output/eu14_city_vehicles_xgb_2011_2024.csv (from 04),
data/dataset_city_features.csv (from 03), data/boundary/eu14_city.gpkg
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent
OUT_DIR  = PROJ_DIR / "output"

VEH_CSV   = OUT_DIR  / "eu14_city_vehicles_xgb_2011_2024.csv"
FEAT_CSV  = PROJ_DIR / "data" / "dataset_city_features.csv"
BOUND_GPKG = PROJ_DIR / "data" / "boundary" / "eu14_city.gpkg"

OUT_CITY     = OUT_DIR / "result_city.gpkg"
OUT_CITY_CSV = OUT_DIR / "result_city.csv"

CNTR_MAP = {
    "Austria":        "AT",
    "Belgium":        "BE",
    "Switzerland":    "CH",
    "Germany":        "DE",
    "Denmark":        "DK",
    "Spain":          "ES",
    "Finland":        "FI",
    "France":         "FR",
    "United Kingdom": "UK",
    "Italy":          "IT",
    "Netherlands":    "NL",
    "Norway":         "NO",
    "Portugal":       "PT",
    "Sweden":         "SE",
}

# ══════════════════════════════════════════════════════════════════════════
# 1. Load source tables
# ══════════════════════════════════════════════════════════════════════════
print("Loading vehicle predictions ...")
veh = pd.read_csv(VEH_CSV, dtype={"city": int, "year": int})
veh_wide = veh.pivot_table(index=["region", "city", "year"], columns="parameter",
                            values="value", aggfunc="first").reset_index()
veh_wide.columns = [c.replace("EV stock_", "") if c.startswith("EV stock_") else c
                    for c in veh_wide.columns]

print("Loading city features ...")
feat = pd.read_csv(FEAT_CSV, dtype={"UID": int, "year": int})
feat = feat.rename(columns={"UID": "city"})

print("Loading boundary (municipality names + geometry) ...")
bound = gpd.read_file(BOUND_GPKG)[["UID", "NAME_3", "GISCO_ID", "geometry"]]
bound = bound.rename(columns={"UID": "city", "NAME_3": "muni_name"})
bound["city"] = bound["city"].astype(int)
bound = bound.drop_duplicates("city")
# cntr_code recovered from the GISCO_ID prefix (e.g. "FR_01001" -> "FR")
bound["cntr_code"] = bound["GISCO_ID"].str.split("_").str[0]

# ══════════════════════════════════════════════════════════════════════════
# 2. City-level dataset
# ══════════════════════════════════════════════════════════════════════════
print("Building city dataset ...")

city = veh_wide.merge(
    feat[["city", "year", "pop", "gdp", "light_sum", "light_mean"]],
    on=["city", "year"], how="left"
)
bound_join = bound[["city", "muni_name", "GISCO_ID", "cntr_code", "geometry"]].rename(
    columns={"GISCO_ID": "gisco_id"})
city = city.merge(bound_join, on="city", how="left")
city["cntr_code"] = city["cntr_code"].fillna(city["region"].map(CNTR_MAP))
city = city.rename(columns={"region": "country"})

for col in ["BEV", "PHEV", "FCEV", "ICEV"]:
    if col not in city.columns:
        city[col] = 0.0

city["ev_share"] = (city["BEV"] + city["PHEV"] + city["FCEV"]) / \
                   (city["BEV"] + city["PHEV"] + city["FCEV"] + city["ICEV"]).replace(0, np.nan)

city_cols = [
    "gisco_id", "muni_name", "cntr_code", "country", "year",
    "BEV", "PHEV", "FCEV", "ICEV", "ev_share",
    "pop", "gdp", "light_sum", "light_mean", "geometry",
]
city = gpd.GeoDataFrame(city[city_cols], geometry="geometry", crs=bound.crs)
city = city.sort_values(["cntr_code", "gisco_id", "year"]).reset_index(drop=True)

city.to_file(OUT_CITY, driver="GPKG")
print(f"  Saved: {OUT_CITY}  ({city.shape[0]:,} rows x {city.shape[1]} cols)")

# Public deliverable: no geometry column, per LAU boundary data terms of use.
city_no_geom = city.drop(columns="geometry")
city_no_geom.to_csv(OUT_CITY_CSV, index=False, float_format="%.6f")
print(f"  Saved: {OUT_CITY_CSV}  ({city_no_geom.shape[0]:,} rows x {city_no_geom.shape[1]} cols)")

# ══════════════════════════════════════════════════════════════════════════
# 3. National totals (for sanity check only, not exported)
# ══════════════════════════════════════════════════════════════════════════
sum_cols = ["BEV", "PHEV", "FCEV", "ICEV"]
nat = city.drop(columns="geometry").groupby(["cntr_code", "country", "year"])[sum_cols].sum().reset_index()
nat["ev_share"] = (nat["BEV"] + nat["PHEV"] + nat["FCEV"]) / \
                  (nat["BEV"] + nat["PHEV"] + nat["FCEV"] + nat["ICEV"]).replace(0, np.nan)
LATEST_YEAR = int(nat["year"].max())
print(f"\n-- National totals {LATEST_YEAR} --")
print(nat[nat["year"] == LATEST_YEAR].set_index("country")[["BEV", "ICEV", "ev_share"]].round(3).to_string())

print("\n-- City coverage --")
print(f"  Cities with muni_name: {city['muni_name'].notna().sum() / len(city) * 100:.1f}%")
print(f"  Cities with light_sum > 0 ({LATEST_YEAR}): "
      f"{(city[city['year']==LATEST_YEAR]['light_sum'] > 0).sum() / (city['year']==LATEST_YEAR).sum() * 100:.1f}%")
print("\nDone.")
