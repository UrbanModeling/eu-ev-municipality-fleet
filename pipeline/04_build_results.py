# -*- coding: utf-8 -*-
"""
Step 04 – Assemble final result datasets.

Reads XGBoost downscaling output, merges city features and boundary metadata,
and produces the two published result files.

Inputs (relative to project root)
------
output/eu14_city_vehicles_xgb_2011_2023.csv   (from 03_xgb_downscale.py)
data/dataset_city_features.csv                 (from 02_build_features.py)
data/boundary/eu14_city.csv                    (GADM boundary with city names)

Outputs
-------
output/eu_ev_municipality_fleet_2011_2023.csv
    City × year panel, 848,900 rows × 14 columns
output/eu_ev_national_fleet_2011_2023.csv
    Country × year panel, 182 rows × 10 columns
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent
OUT_DIR  = PROJ_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

VEH_CSV   = OUT_DIR  / "eu14_city_vehicles_xgb_2011_2023.csv"
FEAT_CSV  = PROJ_DIR / "data" / "dataset_city_features.csv"
BOUND_CSV = PROJ_DIR / "data" / "boundary" / "eu14_city.csv"

OUT_CITY = OUT_DIR / "eu_ev_municipality_fleet_2011_2023.csv"
OUT_NAT  = OUT_DIR / "eu_ev_national_fleet_2011_2023.csv"

# ── ISO3 mapping ───────────────────────────────────────────────────────────
GID_MAP = {
    "Austria": "AUT", "Belgium": "BEL", "Switzerland": "CHE",
    "Germany": "DEU", "Denmark": "DNK", "Spain": "ESP",
    "Finland": "FIN", "France": "FRA", "United Kingdom": "GBR",
    "Italy": "ITA", "Netherlands": "NLD", "Norway": "NOR",
    "Portugal": "PRT", "Sweden": "SWE",
}

# ── XGBoost parameter name → published column name ────────────────────────
PARAM_MAP = {
    "EV stock_BEV":  "BEV",
    "EV stock_PHEV": "PHEV",
    "EV stock_FCEV": "FCEV",
    "EV stock_ICEV": "other_veh",
}

# ══════════════════════════════════════════════════════════════════════════
# 1. Load and pivot XGBoost predictions (long → wide)
# ══════════════════════════════════════════════════════════════════════════
print("Loading XGBoost predictions …")
veh = pd.read_csv(VEH_CSV, dtype={"city": int, "year": int})
veh["value"] = pd.to_numeric(veh["value"], errors="coerce").fillna(0)
veh["parameter"] = veh["parameter"].map(PARAM_MAP)
veh = veh[veh["parameter"].notna()]

city = (veh.pivot_table(index=["region", "city", "year"],
                         columns="parameter", values="value",
                         aggfunc="sum")
           .reset_index()
           .rename(columns={"city": "uid"}))
city.columns.name = None
for col in ["BEV", "PHEV", "FCEV", "other_veh"]:
    if col not in city.columns:
        city[col] = 0.0
city[["BEV", "PHEV", "FCEV", "other_veh"]] = \
    city[["BEV", "PHEV", "FCEV", "other_veh"]].fillna(0)

# ══════════════════════════════════════════════════════════════════════════
# 2. Merge city features (pop, gdp, light_sum, light_mean)
# ══════════════════════════════════════════════════════════════════════════
print("Loading city features …")
feat = pd.read_csv(FEAT_CSV, dtype={"UID": int, "year": int})
feat = feat.rename(columns={"UID": "uid"})

city = city.merge(
    feat[["uid", "year", "pop", "gdp", "light_sum", "light_mean"]],
    on=["uid", "year"], how="left"
)

# ══════════════════════════════════════════════════════════════════════════
# 3. Merge boundary (city_name, gid_0)
# ══════════════════════════════════════════════════════════════════════════
print("Loading boundary data …")
bound = pd.read_csv(BOUND_CSV, low_memory=False,
                    usecols=["GID_0", "UID", "NAME_3"],
                    dtype={"UID": int})
bound = bound.rename(columns={"UID": "uid", "NAME_3": "city_name"})
bound = bound.drop_duplicates("uid")

city = city.merge(bound[["uid", "city_name", "GID_0"]], on="uid", how="left")
city["gid_0"] = city["GID_0"].fillna(city["region"].map(GID_MAP))
city = city.drop(columns=["GID_0"])

# ══════════════════════════════════════════════════════════════════════════
# 4. EV share and final column order
# ══════════════════════════════════════════════════════════════════════════
total = city["BEV"] + city["PHEV"] + city["FCEV"] + city["other_veh"]
city["ev_share"] = (city["BEV"] + city["PHEV"] + city["FCEV"]) / total.replace(0, np.nan)

city = (city[["uid", "city_name", "gid_0", "region", "year",
               "BEV", "PHEV", "FCEV", "other_veh",
               "pop", "gdp", "light_sum", "light_mean", "ev_share"]]
        .sort_values(["gid_0", "uid", "year"])
        .reset_index(drop=True))

city.to_csv(OUT_CITY, index=False, float_format="%.6f")
print(f"  Saved: {OUT_CITY.name}  ({city.shape[0]:,} rows × {city.shape[1]} cols)")

# ══════════════════════════════════════════════════════════════════════════
# 5. National aggregates
# ══════════════════════════════════════════════════════════════════════════
print("Building national dataset …")
nat = (city.groupby(["region", "year"], as_index=False)
           [["BEV", "PHEV", "FCEV", "other_veh", "pop", "gdp"]]
           .sum())
nat["gid_0"] = nat["region"].map(GID_MAP)
total_nat = nat["BEV"] + nat["PHEV"] + nat["FCEV"] + nat["other_veh"]
nat["ev_share"] = (nat["BEV"] + nat["PHEV"] + nat["FCEV"]) / total_nat.replace(0, np.nan)

nat = (nat[["gid_0", "region", "year",
             "BEV", "PHEV", "FCEV", "other_veh",
             "pop", "gdp", "ev_share"]]
       .sort_values(["gid_0", "year"])
       .reset_index(drop=True))

nat.to_csv(OUT_NAT, index=False, float_format="%.6f")
print(f"  Saved: {OUT_NAT.name}  ({nat.shape[0]:,} rows × {nat.shape[1]} cols)")

print("\nDone.")
