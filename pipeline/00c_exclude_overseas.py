# -*- coding: utf-8 -*-
"""
Step 00c - Drop French overseas departments (NUTS2 FRY1-FRY5) from all
already-computed per-UID outputs; the nightlight raster doesn't cover them.
Filters in place, without renumbering UID - run 03/04/05 again afterwards.

Modifies: data/boundary/eu14_city.gpkg, eu14_city.csv, lau_nuts.csv,
data/pop/city_pop_*.csv, data/gdp/city_gdp_*.csv, data/road/road_light_*_stats.csv
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent
BND_GPKG = PROJ_DIR / "data" / "boundary" / "eu14_city.gpkg"
BND_CSV  = PROJ_DIR / "data" / "boundary" / "eu14_city.csv"
NUTS_CSV = PROJ_DIR / "data" / "boundary" / "lau_nuts.csv"

lau_nuts = pd.read_csv(NUTS_CSV)
exclude_uid = set(lau_nuts[lau_nuts["nuts2_code"].astype(str).str.startswith("FRY")]["UID"])
print(f"Excluding {len(exclude_uid)} overseas LAU units (FRY1-FRY5)")

print("Filtering boundary ...")
gdf = gpd.read_file(BND_GPKG)
gdf = gdf[~gdf["UID"].isin(exclude_uid)]
gdf.to_file(BND_GPKG, driver="GPKG")
gdf.drop(columns="geometry").to_csv(BND_CSV, index=False)
print(f"  {len(gdf):,} LAU units remain")

nuts_filtered = lau_nuts[~lau_nuts["UID"].isin(exclude_uid)]
nuts_filtered.to_csv(NUTS_CSV, index=False)

for subdir, pattern in [("pop", "city_pop_*.csv"), ("gdp", "city_gdp_*.csv"),
                         ("road", "road_light_*_stats.csv")]:
    for f in (PROJ_DIR / "data" / subdir).glob(pattern):
        df = pd.read_csv(f, low_memory=False)
        before = len(df)
        df = df[~df["UID"].isin(exclude_uid)]
        df.to_csv(f, index=False)
        print(f"  {f.name}: {before:,} -> {len(df):,} rows")

print("\nDone. Rerun pipeline/03_build_dataset.py onward.")
