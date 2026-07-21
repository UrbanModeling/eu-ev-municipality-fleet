# -*- coding: utf-8 -*-
"""
Step 02 - Road-masked nighttime light per municipality (UID) per year.
light_sum/light_mean are zonal stats of the NTL raster restricted to a
buffered OpenStreetMap road mask within each municipality polygon.

Inputs (external):
    nppviirs_like_V2_{year}.tif   (NPP-VIIRS-like nighttime light, 2011-2024, see README)
    <OSM highway roads gpkg/pbf, EU14 extract>                        (path TBD)

Inputs (project):
    data/boundary/eu14_city.gpkg     (UID, NAME_0, geometry)

Outputs:
    data/road/road_light_{year}_stats.csv   (NAME_0, UID, light_sum, light_mean)
"""

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterstats import zonal_stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJ_DIR = Path(__file__).parent.parent
BOUNDARY_GPKG = PROJ_DIR / "data" / "boundary" / "eu14_city.gpkg"
NIGHTLIGHT_DIR = Path("path/to/nighttime_light")  # NPP-VIIRS-like annual .tif files, see README
ROADS_PATH = Path("path/to/osm_road")   # OSM highway roads

OUT_DIR = PROJ_DIR / "data" / "road"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_START, YEAR_END = 2011, 2024
BBOX = (-25, 36, 35, 72)   # (min_lon, min_lat, max_lon, max_lat), covers all 14 countries — for the nightlight raster window only

HIGHWAY_CLASSES = [
    "motorway", "trunk", "primary", "secondary", "tertiary", "residential",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
]
BUFFER_M = 250   # road buffer half-width, metres

print("Loading city boundaries ...")
cities = gpd.read_file(BOUNDARY_GPKG)[["UID", "NAME_0", "geometry"]].to_crs(epsg=4326)
print(f"  {len(cities):,} cities")

print("Loading OSM highway roads and building buffer mask ...")
roads = gpd.read_file(ROADS_PATH)
roads = roads[roads["highway"].isin(HIGHWAY_CLASSES)]
roads_m = roads.to_crs(epsg=3857)
road_buffer = roads_m.buffer(BUFFER_M).union_all()
road_buffer = gpd.GeoSeries([road_buffer], crs=3857).to_crs(epsg=4326).iloc[0]

print("Intersecting road mask with municipality boundaries ...")
cities["geometry"] = cities["geometry"].intersection(road_buffer)
cities = cities[~cities["geometry"].is_empty]

min_lon, min_lat, max_lon, max_lat = BBOX

for year in range(YEAR_START, YEAR_END + 1):
    print(f"\n===== {year} =====")
    nightlight_path = NIGHTLIGHT_DIR / f"nppviirs_like_V2_{year}.tif"
    out_csv = OUT_DIR / f"road_light_{year}_stats.csv"

    if out_csv.exists():
        print(f"  Already done, skipping: {out_csv}")
        continue

    with rasterio.open(nightlight_path) as src:
        win = from_bounds(min_lon, min_lat, max_lon, max_lat, transform=src.transform)
        clipped_img = src.read(1, window=win)
        clipped_transform = src.window_transform(win)
        crs = src.crs

    cities_yr = cities.to_crs(crs) if cities.crs != crs else cities.copy()
    stats = zonal_stats(cities_yr, clipped_img, affine=clipped_transform,
                         stats=["sum", "mean"], nodata=0)
    cities_yr["light_sum"]  = [s["sum"]  if s["sum"]  is not None else 0.0 for s in stats]
    cities_yr["light_mean"] = [s["mean"] if s["mean"] is not None else 0.0 for s in stats]

    out_df = cities_yr[["NAME_0", "UID", "light_sum", "light_mean"]]
    out_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}  ({len(out_df):,} rows, "
          f"{(out_df['light_sum'] > 0).sum():,} nonzero)")

print("\nDone.")
