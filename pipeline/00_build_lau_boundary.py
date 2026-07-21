# -*- coding: utf-8 -*-
"""
Step 00 - Build data/boundary/eu14_city.gpkg + .csv from the Eurostat LAU
2020 boundary.

Output schema:
    UID, GID_0, NAME_0, NAME_3 (municipality name), CC_2 (Germany Kreis
    code, NaN elsewhere), LAU_ID, GISCO_ID, geometry

Output:
    data/boundary/eu14_city.gpkg
    data/boundary/eu14_city.csv
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent
LAU_GPKG = Path("path/to/LAU_14_Countries.gpkg")  # Eurostat LAU 2020, see README

OUT_GPKG = PROJ_DIR / "data" / "boundary" / "eu14_city.gpkg"
OUT_CSV  = PROJ_DIR / "data" / "boundary" / "eu14_city.csv"

CNTR_TO_GID0 = {
    "AT": "AUT", "BE": "BEL", "CH": "CHE", "DE": "DEU", "DK": "DNK",
    "ES": "ESP", "FI": "FIN", "FR": "FRA", "UK": "GBR", "IT": "ITA",
    "NL": "NLD", "NO": "NOR", "PT": "PRT", "SE": "SWE",
}
CNTR_TO_NAME0 = {
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland", "DE": "Germany",
    "DK": "Denmark", "ES": "Spain", "FI": "Finland", "FR": "France",
    "UK": "United Kingdom", "IT": "Italy", "NL": "Netherlands",
    "NO": "Norway", "PT": "Portugal", "SE": "Sweden",
}

print("Loading LAU boundary ...")
gdf = gpd.read_file(LAU_GPKG).to_crs(epsg=4326)
print(f"  {len(gdf):,} LAU units")

missing = set(gdf["CNTR_CODE"].unique()) - set(CNTR_TO_GID0)
assert not missing, f"Unmapped CNTR_CODE values: {missing}"

gdf = gdf.reset_index(drop=True)
gdf["UID"] = gdf.index.astype(int)
gdf["GID_0"] = gdf["CNTR_CODE"].map(CNTR_TO_GID0)
gdf["NAME_0"] = gdf["CNTR_CODE"].map(CNTR_TO_NAME0)
gdf["NAME_3"] = gdf["LAU_NAME"]

gdf["CC_2"] = pd.array([pd.NA] * len(gdf), dtype="string")
is_de = gdf["CNTR_CODE"] == "DE"
gdf.loc[is_de, "CC_2"] = gdf.loc[is_de, "LAU_ID"].astype(str).str.zfill(8).str[:5]
# "DE" prefix keeps this unambiguously non-numeric, so a CSV round-trip doesn't drop the leading zero.
gdf.loc[is_de, "CC_2"] = "DE" + gdf.loc[is_de, "CC_2"]

out_cols = ["UID", "GID_0", "NAME_0", "NAME_3", "CC_2", "LAU_ID", "GISCO_ID", "geometry"]
gdf = gdf[out_cols]

print(f"  Countries: {gdf['GID_0'].value_counts().to_dict()}")
print(f"  Germany CC_2 (Kreis) unique codes: {gdf.loc[is_de, 'CC_2'].nunique()}")

print("Saving ...")
gdf.to_file(OUT_GPKG, driver="GPKG")
gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)
print(f"  Saved: {OUT_GPKG}")
print(f"  Saved: {OUT_CSV}")
