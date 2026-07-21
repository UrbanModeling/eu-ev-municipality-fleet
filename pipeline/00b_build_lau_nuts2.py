# -*- coding: utf-8 -*-
"""
Step 00b - Map each LAU unit to its NUTS2/NUTS3 region via a spatial join
(representative point in polygon, nearest-neighbor fallback for gaps)
against Eurostat's NUTS 2021 boundary.

Source: https://gisco-services.ec.europa.eu/distribution/v2/nuts/gpkg/NUTS_RG_01M_2021_4326.gpkg
(downloaded to data/boundary/NUTS_RG_01M_2021_4326.gpkg)

Output: data/boundary/lau_nuts.csv   (UID, nuts3_code, nuts2_code)
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent
LAU_GPKG  = PROJ_DIR / "data" / "boundary" / "eu14_city.gpkg"
NUTS_GPKG = PROJ_DIR / "data" / "boundary" / "NUTS_RG_01M_2021_4326.gpkg"
OUT_CSV   = PROJ_DIR / "data" / "boundary" / "lau_nuts.csv"

EU14_CNTR = ["AT", "BE", "CH", "DE", "DK", "ES", "FI", "FR", "UK",
             "IT", "NL", "NO", "PT", "SE"]

print("Loading LAU boundary ...")
lau = gpd.read_file(LAU_GPKG)[["UID", "geometry"]].to_crs(epsg=4326)
print(f"  {len(lau):,} LAU units")

print("Loading NUTS3 boundary ...")
nuts = gpd.read_file(NUTS_GPKG)
nuts3 = nuts[(nuts["LEVL_CODE"] == 3) & (nuts["CNTR_CODE"].isin(EU14_CNTR))][
    ["NUTS_ID", "geometry"]
].to_crs(epsg=4326).rename(columns={"NUTS_ID": "nuts3_code"})
print(f"  {len(nuts3):,} NUTS3 regions")

print("Computing LAU representative points ...")
lau_pts = lau.copy()
lau_pts["geometry"] = lau_pts.geometry.representative_point()

print("Spatial join: LAU point within NUTS3 ...")
joined = gpd.sjoin(lau_pts, nuts3, how="left", predicate="within")
joined = joined.drop(columns=["index_right"])
n_missing = joined["nuts3_code"].isna().sum()
print(f"  Matched: {len(joined) - n_missing:,} / {len(joined):,}  (missing: {n_missing})")

if n_missing:
    print(f"  Filling {n_missing} unmatched via nearest NUTS3 ...")
    missing = joined[joined["nuts3_code"].isna()][["UID", "geometry"]]
    nearest = gpd.sjoin_nearest(missing, nuts3, how="left")
    nearest = nearest.drop_duplicates("UID")[["UID", "nuts3_code"]]
    joined = joined.set_index("UID")
    joined.loc[nearest["UID"], "nuts3_code"] = nearest.set_index("UID")["nuts3_code"]
    joined = joined.reset_index()

joined["nuts2_code"] = joined["nuts3_code"].str[:4]

out = joined[["UID", "nuts3_code", "nuts2_code"]]
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}  ({len(out):,} rows, {out['nuts3_code'].isna().sum()} still unmatched)")
