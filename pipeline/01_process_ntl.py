

import fiona
import numpy as np
import rasterio
import rasterio.features
from rasterio.windows import from_bounds
from pathlib import Path
import geopandas as gpd
from rasterstats import zonal_stats
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── USER CONFIGURATION ────────────────────────────────────────────────────
# Set these paths to the locations of your downloaded external datasets.
# See README for data sources.
NIGHTLIGHT_DIR    = Path("path/to/nighttime_light")       # NPP-VIIRS-like annual .tif files
ROADS_GPKG        = Path("path/to/osm_roads_eu14.geojson") # OSM road network (EU14)
CITY_BOUNDARY_SHP = Path("path/to/gadm_eu14.gpkg")        # GADM municipality boundaries

# Output directory for road-masked NTL statistics (one CSV per year).
# 02_build_features.py reads from this same directory.
NTL_OUT_DIR = Path(__file__).parent.parent / "data" / "ntl_road_stats"
# ──────────────────────────────────────────────────────────────────────────

# Spatial extent (lon/lat bounding box for EU14 region)
# Excludes Greece and Poland vs. the previous EU16 scope
BBOX = (-25, 36, 35, 72)   # (min_lon, min_lat, max_lon, max_lat)
# Covers 14 countries: Portugal(-9°W) to Finland(30°E), Spain(36°N) to Norway(71°N)

# Year range to process
YEAR_START = 2011
YEAR_END   = 2023


NTL_OUT_DIR.mkdir(parents=True, exist_ok=True)

min_lon, min_lat, max_lon, max_lat = BBOX


roads = gpd.read_file(ROADS_GPKG, columns=["geometry"])
if roads.crs.to_epsg() != 4326:
    roads = roads.to_crs("EPSG:4326")


cities = gpd.read_file(CITY_BOUNDARY_SHP)

# Retrieve fid from GeoPackage feature IDs via fiona (not exposed as a regular
# column by GeoPandas, but visible in QGIS as the SQLite primary key)
with fiona.open(str(CITY_BOUNDARY_SHP)) as f:
    cities["fid"] = [int(feat.id) for feat in f]

print("Cities columns:", cities.columns.tolist(), flush=True)


for year in range(YEAR_START, YEAR_END + 1):
    print(f"===== Processing {year} =====", flush=True)

    nightlight_path = Path(NIGHTLIGHT_DIR) / f"nppviirs_like_V2_{year}.tif"
    output_tiff = NTL_OUT_DIR / f"road_light_{year}.tif"
    output_csv  = NTL_OUT_DIR / f"road_light_{year}_stats.csv"

    # Step 1 – windowed read: only load EU14 bbox, no full-raster load
    with rasterio.open(nightlight_path) as src:
        win = from_bounds(min_lon, min_lat, max_lon, max_lat, transform=src.transform)
        clipped_img = src.read(1, window=win)
        clipped_transform = src.window_transform(win)
        crs = src.crs
        nodata = src.nodata

    profile = {
        "driver": "GTiff",
        "dtype": "float64",
        "width": clipped_img.shape[1],
        "height": clipped_img.shape[0],
        "count": 1,
        "crs": crs,
        "transform": clipped_transform,
    }

    # Step 2 – rasterise road network as binary mask
    road_mask = rasterio.features.rasterize(
        ((geom, 1) for geom in roads.geometry),
        out_shape=clipped_img.shape,
        transform=clipped_transform,
        fill=0,
        dtype=np.uint8,
    )
    road_light = (clipped_img * road_mask).astype(np.float64)

    # Step 3 – save road-light raster
    with rasterio.open(output_tiff, "w", **profile) as dst:
        dst.write(road_light, 1)

    # Step 4 – zonal statistics: sum and mean of road light per city
    cities_yr = cities.to_crs(crs) if cities.crs != crs else cities.copy()
    stats = zonal_stats(cities_yr, road_light, affine=clipped_transform, stats=["sum", "mean"], nodata=0)
    cities_yr["light_sum"]  = [s["sum"]  if s["sum"]  is not None else 0 for s in stats]
    cities_yr["light_mean"] = [s["mean"] if s["mean"] is not None else 0 for s in stats]

    # Step 5 – export (only keep essential columns)
    cities_yr[["CNTR_CODE", "fid", "light_sum", "light_mean"]].to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv.name}", flush=True)
