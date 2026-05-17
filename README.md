# EU Municipality-Level EV Fleet Dataset (2011–2023)

Code repository for the paper:

> **A municipality-level dataset of electric vehicle fleet composition across 65,300 European cities, 2011–2023**

This repository contains all code used to generate, validate, and visualise the dataset. The dataset itself is published separately (see *Data Availability* below).

---

## Repository structure

```
├── pipeline/
│   ├── 01_process_ntl.py       # Extract road-masked nighttime light statistics per municipality
│   ├── 02_build_features.py    # Build national training set and city-level feature panel
│   ├── 03_xgb_downscale.py     # Train XGBoost models and downscale vehicle stocks to city level
│   └── 04_build_results.py     # Assemble final result CSVs
├── validation/
│   ├── validate_nuts2.py       # NUTS-2 regional validation against Eurostat
│   ├── validate_landkreis.py   # Municipality-level validation against KBA (Germany)
│   ├── validate_norway.py      # Municipality-level validation against SSB (Norway)
│   └── validate_sweden.py      # Municipality-level validation against SCB (Sweden)
└── visualization/
    ├── fig_trends.py           # Fig. 2 – Temporal trends in national BEV fleet and EV share
    ├── fig_geodist.py          # Fig. 3 – Geographic distribution of EV fleet share and size
    ├── fig_nuts2_validation.py # Fig. 4 – NUTS-2 regional validation scatter plots
    └── fig_validation.py       # Fig. 5 – Municipality-level validation scatter plots
```

---

## Requirements

Python 3.9+ is recommended. Install dependencies with:

```bash
pip install pandas numpy scipy geopandas fiona rasterio rasterstats xgboost matplotlib
```

---

## External input data

The pipeline requires the following external datasets, which must be downloaded separately and placed at the paths configured at the top of each script:

| Dataset | Source | Used in |
|---|---|---|
| NPP-VIIRS-like nighttime light (annual, 1992–2024) | Chen et al. (2026), *Journal of Remote Sensing* | `01_process_ntl.py` |
| OpenStreetMap road network (EU14) | [openstreetmap.org](https://www.openstreetmap.org) | `01_process_ntl.py` |
| GADM municipality boundaries | [gadm.org](https://gadm.org) | `01_process_ntl.py`, `04_build_results.py` |
| Gridded GDP per capita PPP (1990–2022) | Kummu et al. (2025), *Scientific Data* | `02_build_features.py` |
| Gridded population (NASA GPW v4) | CIESIN / NASA SEDAC | `02_build_features.py` |
| IEA Global EV Outlook — national EV stocks and shares | [iea.org](https://www.iea.org/data-and-statistics/data-product/global-ev-outlook) | `02_build_features.py` |
| Eurostat regional passenger car stock (road_eqs_carage) | [ec.europa.eu/eurostat](https://ec.europa.eu/eurostat) | `validate_nuts2.py` |

Update the path constants at the top of each script to point to your local copies of these files.

---

## Running the pipeline

Execute the four pipeline scripts in order:

```bash
python pipeline/01_process_ntl.py       # ~hours; outputs road-masked NTL stats per city per year
python pipeline/02_build_features.py    # outputs data/dataset_city_features.csv and data/dataset_train.csv
python pipeline/03_xgb_downscale.py     # outputs city-level vehicle stock predictions
python pipeline/04_build_results.py     # outputs eu_ev_municipality_fleet_2011_2023.csv and eu_ev_national_fleet_2011_2023.csv
```

To reproduce the technical validation figures run the scripts in `validation/` after step 04.

To reproduce the paper figures run the scripts in `visualization/` after step 04.

---

## Output datasets

| File | Dimensions | Description |
|---|---|---|
| `eu_ev_municipality_fleet_2011_2023.csv` | 848,900 rows × 14 columns | Municipality × year panel for 65,300 cities across 14 countries |

Spatial boundaries can be joined using the `uid` column with GADM municipality polygons.

---

## Data availability

The dataset is deposited at [https://doi.org/10.6084/m9.figshare.32241864].

---

## Citation

If you use this code or dataset, please cite:

> [Citation to be added upon publication]

---

## License

Code: MIT License.
Data: CC BY 4.0.
