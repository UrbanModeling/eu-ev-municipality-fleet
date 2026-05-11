# -*- coding: utf-8 -*-
"""
Build two analysis-ready result datasets for the paper.

Output 1: output/result_city.csv
    City × year panel (65,300 cities × 13 years)
    Columns: uid, city_name, gid_0, region, year,
             BEV, PHEV, FCEV, ICEV,
             E_BEV_t, E_PHEV_t, E_FCEV_t, E_ICEV_t, E_EV_saved_t,
             pop, gdp, light_sum, light_mean, ev_share

Output 2: output/result_national.csv
    Country × year panel (14 countries × 13 years)
    Columns: gid_0, region, year,
             BEV, PHEV, FCEV, ICEV,
             E_BEV_t, E_PHEV_t, E_FCEV_t, E_ICEV_t, E_EV_saved_t,
             pop, gdp, ev_share,
             CI_electric, EF_FUEL

Inputs
------
output/eu14_city_lca_use_phase_2011_2023.csv   (from lca_use_phase.py)
data/dataset_city_features.csv                  (from build_datasets.py)
data/boundary/eu14_city.csv                     (GADM boundary with city names)
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent
OUT_DIR  = PROJ_DIR / "output"

LCA_CSV    = OUT_DIR  / "eu14_city_lca_use_phase_2011_2023.csv"
FEAT_CSV   = PROJ_DIR / "data" / "dataset_city_features.csv"
BOUND_CSV  = PROJ_DIR / "data" / "boundary" / "eu14_city.csv"

OUT_CITY = OUT_DIR / "result_city.csv"
OUT_NAT  = OUT_DIR / "result_national.csv"

# ── ISO3 mapping (from GADM GID_0) ────────────────────────────────────────
GID_MAP = {
    "Austria":        "AUT",
    "Belgium":        "BEL",
    "Switzerland":    "CHE",
    "Germany":        "DEU",
    "Denmark":        "DNK",
    "Spain":          "ESP",
    "Finland":        "FIN",
    "France":         "FRA",
    "United Kingdom": "GBR",
    "Italy":          "ITA",
    "Netherlands":    "NLD",
    "Norway":         "NOR",
    "Portugal":       "PRT",
    "Sweden":         "SWE",
}

# ══════════════════════════════════════════════════════════════════════════
# 1. Load source tables
# ══════════════════════════════════════════════════════════════════════════
print("Loading LCA use-phase results …")
lca = pd.read_csv(LCA_CSV, dtype={"city": int, "year": int})

print("Loading city features …")
feat = pd.read_csv(FEAT_CSV, dtype={"UID": int, "year": int})
feat = feat.rename(columns={"UID": "city"})

print("Loading boundary (city names) …")
bound = pd.read_csv(BOUND_CSV, low_memory=False,
                    usecols=["GID_0", "UID", "NAME_3"],
                    dtype={"UID": int})
bound = bound.rename(columns={"UID": "city", "NAME_3": "city_name"})
# Keep one row per city (NAME_3 is the finest level with good coverage)
bound = bound.drop_duplicates("city")

# ══════════════════════════════════════════════════════════════════════════
# 2. City-level dataset
# ══════════════════════════════════════════════════════════════════════════
print("Building city dataset …")

# Merge LCA + features
city = lca.merge(
    feat[["city", "year", "pop", "gdp", "light_sum", "light_mean"]],
    on=["city", "year"], how="left"
)

# Merge city names + GID_0 from boundary
city = city.merge(bound[["city", "city_name", "GID_0"]], on="city", how="left")

# Add gid_0 from region name (fallback if boundary GID_0 missing)
city["gid_0"] = city["GID_0"].fillna(city["region"].map(GID_MAP))
city = city.drop(columns=["GID_0"])

# EV share = (BEV+PHEV+FCEV) / total fleet
city["ev_share"] = (city["BEV"] + city["PHEV"] + city["FCEV"]) / \
                   (city["BEV"] + city["PHEV"] + city["FCEV"] + city["ICEV"]).replace(0, np.nan)

# Select and order columns
city_cols = [
    "city", "city_name", "gid_0", "region", "year",
    "BEV", "PHEV", "FCEV", "ICEV",
    "E_BEV_t", "E_PHEV_t", "E_FCEV_t", "E_ICEV_t", "E_EV_saved_t",
    "pop", "gdp", "light_sum", "light_mean", "ev_share",
]
city = city[city_cols].rename(columns={"city": "uid"})
city = city.sort_values(["gid_0", "uid", "year"]).reset_index(drop=True)

city.to_csv(OUT_CITY, index=False, float_format="%.6f")
print(f"  Saved: {OUT_CITY}  ({city.shape[0]:,} rows × {city.shape[1]} cols)")

# ══════════════════════════════════════════════════════════════════════════
# 3. National dataset
# ══════════════════════════════════════════════════════════════════════════
print("Building national dataset …")

# Additive columns (sum across cities) - vehicle + emissions from lca
sum_cols_lca = ["BEV", "PHEV", "FCEV", "ICEV",
                "E_BEV_t", "E_PHEV_t", "E_FCEV_t", "E_ICEV_t", "E_EV_saved_t"]

# National-level scalars (CI and EF_FUEL are uniform within region×year)
scalar_cols = ["CI_electric", "EF_FUEL"]

nat_sum    = lca.groupby(["region", "year"])[sum_cols_lca].sum()
nat_scalar = lca.groupby(["region", "year"])[scalar_cols].first()

# Pop and GDP from city_features (feat already has region column)
nat_feat = feat.groupby(["region", "year"])[["pop", "gdp"]].sum()

nat = nat_sum.join(nat_scalar).join(nat_feat).reset_index()

# EV share
nat["ev_share"] = (nat["BEV"] + nat["PHEV"] + nat["FCEV"]) / \
                  (nat["BEV"] + nat["PHEV"] + nat["FCEV"] + nat["ICEV"]).replace(0, np.nan)

# GID_0
nat["gid_0"] = nat["region"].map(GID_MAP)

nat_cols = [
    "gid_0", "region", "year",
    "BEV", "PHEV", "FCEV", "ICEV",
    "E_BEV_t", "E_PHEV_t", "E_FCEV_t", "E_ICEV_t", "E_EV_saved_t",
    "pop", "gdp", "ev_share",
    "CI_electric", "EF_FUEL",
]
nat = nat[nat_cols].sort_values(["gid_0", "year"]).reset_index(drop=True)

nat.to_csv(OUT_NAT, index=False, float_format="%.6f")
print(f"  Saved: {OUT_NAT}  ({nat.shape[0]:,} rows × {nat.shape[1]} cols)")

# ══════════════════════════════════════════════════════════════════════════
# 4. Quick sanity check
# ══════════════════════════════════════════════════════════════════════════
print("\n── National totals 2023 (MtCO2) ──")
check = nat[nat["year"] == 2023].set_index("region")[
    ["BEV", "ICEV", "E_ICEV_t", "E_EV_saved_t", "ev_share", "CI_electric", "EF_FUEL"]
].copy()
check["E_ICEV_t"] = check["E_ICEV_t"] / 1e6
check["E_EV_saved_t"] = check["E_EV_saved_t"] / 1e6
check.columns = ["BEV_veh", "ICEV_veh", "E_ICEV_Mt", "Saved_Mt", "EV_share", "CI_gCO2/kWh", "EF_gCO2/km"]
print(check.round(3).to_string())

print("\n── City coverage ──")
print(f"  Cities with city_name: {city['city_name'].notna().sum() / len(city) * 100:.1f}%")
print(f"  Cities with ev_share > 0: {(city['ev_share'] > 0).sum() / len(city) * 100:.1f}%")
print("\nDone.")
