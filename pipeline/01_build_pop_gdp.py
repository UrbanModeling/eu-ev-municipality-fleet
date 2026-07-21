# -*- coding: utf-8 -*-
"""
Step 01 - Population and GDP zonal statistics per municipality (UID). Each
raster's own `src.nodata` is read and passed explicitly to zonal_stats, to
avoid leaking nodata pixels (e.g. ocean) into the sum.

Inputs (external):
    gpw_v4_population_count_rev11_{year}_30_sec.tif  (NASA GPW v4, see README)
    rast_gdpTot_1990_2020_30arcsec.tif  (Kummu et al. gridded GDP, 7 bands, see README)

Inputs (project):
    data/boundary/eu14_city.gpkg  (UID, NAME_0, geometry)

Outputs:
    data/pop/city_pop_{year}.csv   (UID, GID_0, pop_total_{year})  for year in 2000..2020 step 5
    data/gdp/city_gdp_{year}.csv   (UID, gdp_total_{year})         for year in 1990..2020 step 5
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJ_DIR = Path(__file__).parent.parent
BOUNDARY_GPKG = PROJ_DIR / "data" / "boundary" / "eu14_city.gpkg"
POP_DIR = Path("path/to/gpw-v4-tiff")  # NASA GPW v4 population rasters, see README
GDP_TIF = Path("path/to/rast_gdpTot_1990_2020_30arcsec.tif")  # Kummu et al. GDP raster, see README

OUT_POP_DIR = PROJ_DIR / "data" / "pop"
OUT_POP_DIR.mkdir(parents=True, exist_ok=True)
OUT_GDP_DIR = PROJ_DIR / "data" / "gdp"
OUT_GDP_DIR.mkdir(parents=True, exist_ok=True)

POP_YEARS = [2000, 2005, 2010, 2015, 2020]
GDP_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020]

print("Loading city boundaries ...")
cities = gpd.read_file(BOUNDARY_GPKG)[["UID", "GID_0", "geometry"]].to_crs(epsg=4326)
print(f"  {len(cities):,} cities")


def zonal_sum(cities_gdf, raster_path, band=1):
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        data = src.read(band).astype(np.float64)
        affine = src.transform
        n_nodata = np.isnan(data).sum() if np.isnan(nodata) else (data == nodata).sum()
        print(f"    raster nodata={nodata}  nodata_fraction={n_nodata / data.size:.4f}")
    stats = zonal_stats(cities_gdf, data, affine=affine, stats=["sum"], nodata=nodata)
    return [s["sum"] if s["sum"] is not None else np.nan for s in stats]


# ══════════════════════════════════════════════════════════════════════════
# Population
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Population ===")
for year in POP_YEARS:
    out_csv = OUT_POP_DIR / f"city_pop_{year}.csv"
    if out_csv.exists():
        print(f"  {year}: already done, skipping")
        continue
    tif = POP_DIR / f"gpw_v4_population_count_rev11_{year}_30_sec.tif"
    print(f"  {year}: {tif.name}")
    df_year = cities[["UID", "GID_0"]].copy()
    df_year[f"pop_total_{year}"] = zonal_sum(cities, tif)
    n_nan = df_year[f"pop_total_{year}"].isna().sum()
    print(f"    NaN count: {n_nan} / {len(df_year)}")
    out_csv = OUT_POP_DIR / f"city_pop_{year}.csv"
    df_year.to_csv(out_csv, index=False)
    print(f"    Saved: {out_csv}")

# ══════════════════════════════════════════════════════════════════════════
# GDP (multi-band raster, one band per year)
# ══════════════════════════════════════════════════════════════════════════
print("\n=== GDP ===")
with rasterio.open(GDP_TIF) as src:
    n_bands = src.count
    print(f"  bands in file: {n_bands}, expected: {len(GDP_YEARS)}")

for i, year in enumerate(GDP_YEARS, start=1):
    out_csv = OUT_GDP_DIR / f"city_gdp_{year}.csv"
    if out_csv.exists():
        print(f"  {year} (band {i}): already done, skipping")
        continue
    print(f"  {year} (band {i})")
    gdp_vals = zonal_sum(cities, GDP_TIF, band=i)
    df_year = cities[["UID"]].copy()
    df_year[f"gdp_total_{year}"] = gdp_vals
    n_nan = df_year[f"gdp_total_{year}"].isna().sum()
    print(f"    NaN count: {n_nan} / {len(df_year)}")
    out_csv = OUT_GDP_DIR / f"city_gdp_{year}.csv"
    df_year.to_csv(out_csv, index=False)
    print(f"    Saved: {out_csv}")

print("\nDone.")
