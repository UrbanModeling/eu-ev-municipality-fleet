# EU Municipality-Level EV Fleet Dataset (2011–2024)

Code repository for the paper:

> **Estimated municipality-level electric vehicle and passenger vehicle fleet composition across 14 European countries, 2011–2024**

This repository contains all code used to generate, validate, and visualise the dataset. The dataset itself is published separately (see *Data Availability* below).

---

## Repository structure

```
├── pipeline/
│   ├── 00_build_lau_boundary.py      # municipality boundary
│   ├── 00b_build_lau_nuts2.py        # map to NUTS-2/NUTS-3
│   ├── 00c_exclude_overseas.py       # drop French overseas depts
│   ├── 01_build_pop_gdp.py           # population + GDP stats
│   ├── 02_road_light.py              # nighttime-light stats
│   ├── 03_build_dataset.py           # build training/feature panels
│   ├── 04_xgb_downscale.py           # train model, predict city stocks
│   ├── 04_xgb_downscale_nolight.py   # ablation: no nightlight
│   └── 05_build_results.py          # assemble final result CSV
├── validation/
│   ├── validate_nuts2.py             # NUTS-2 vs Eurostat
│   ├── validate_nuts2_nolight.py     # NUTS-2, ablation model
│   ├── validate_norway.py            # municipality vs SSB
│   └── validate_sweden.py            # municipality vs Trafikanalys/SCB
├── analysis/
│   ├── compute_rebalance_factor.py   # rebalancing factor by country
│   └── shap_analysis.py              # SHAP feature attribution
└── visualization/
    ├── fig2_trends.py                     # Fig. 2
    ├── fig3_geodist.py                    # Fig. 3
    ├── fig4_nuts2_validation.py           # Fig. 4
    ├── fig5_validation.py                 # Fig. 5
    ├── figS1_r2_distribution.py           # Supp. Fig. S1
    ├── figS2_combined.py                  # Supp. Fig. S2
    ├── figS3_rebalancing_factor.py        # Supp. Fig. S3
    └── figS4_sweden_timeseries_examples.py # Supp. Fig. S4
```

---

## Requirements

Python 3.9+ is recommended. Install dependencies with:

```bash
pip install pandas numpy scipy geopandas fiona rasterio rasterstats xgboost shap matplotlib openpyxl
```

---

## External input data

The pipeline requires the following external datasets, which must be downloaded separately and placed at the paths configured at the top of each script:

| Dataset | Source | Used in |
|---|---|---|
| Eurostat LAU 2020 municipality boundaries | [ec.europa.eu/eurostat](https://ec.europa.eu/eurostat/web/gisco/geodata/administrative-units/local-administrative-units) | `00_build_lau_boundary.py` |
| Eurostat NUTS regions (2021, 1:1M) | [gisco-services.ec.europa.eu](https://gisco-services.ec.europa.eu/distribution/v2/nuts/gpkg/NUTS_RG_01M_2021_4326.gpkg) | `00b_build_lau_nuts2.py` |
| NPP-VIIRS-like nighttime light (annual, 1992–2024) | Chen et al. (2026), *Journal of Remote Sensing* | `02_road_light.py` |
| OpenStreetMap road network (EU14) | [openstreetmap.org](https://www.openstreetmap.org) | `02_road_light.py` |
| Gridded population (NASA GPW v4) | CIESIN / NASA SEDAC | `01_build_pop_gdp.py` |
| Gridded GDP per capita PPP (1990–2022) | Kummu et al. (2025), *Scientific Data* | `01_build_pop_gdp.py` |
| IEA Global EV Outlook — national EV stocks and shares | [iea.org](https://www.iea.org/data-and-statistics/data-product/global-ev-outlook) | `03_build_dataset.py` |
| Eurostat regional passenger car stock (`road_eqs_carage`) | [ec.europa.eu/eurostat](https://ec.europa.eu/eurostat) | `validate_nuts2.py`, `validate_nuts2_nolight.py` |
| KBA Zulassungsbezirk vehicle registrations (Germany) | [kba.de](https://www.kba.de) | `04_xgb_downscale.py`, `04_xgb_downscale_nolight.py`, `shap_analysis.py`, `compute_rebalance_factor.py` |
| SSB municipality vehicle registrations (Norway) | [ssb.no](https://www.ssb.no) | `validate_norway.py` |
| Trafikanalys / SCB municipality vehicle registrations (Sweden) | [trafa.se](https://www.trafa.se) | `validate_sweden.py`, `figS4_sweden_timeseries_examples.py` |

Update the path constants at the top of each script to point to your local copies of these files.

---

## Running the pipeline

Execute the pipeline scripts in order:

```bash
python pipeline/00_build_lau_boundary.py     # outputs data/boundary/eu14_city.gpkg + .csv
python pipeline/00b_build_lau_nuts2.py       # outputs data/boundary/lau_nuts.csv
python pipeline/00c_exclude_overseas.py      # drops French overseas departments from boundary/feature files
python pipeline/01_build_pop_gdp.py          # outputs data/pop/city_pop_{year}.csv, data/gdp/city_gdp_{year}.csv
python pipeline/02_road_light.py             # outputs data/road/road_light_{year}_stats.csv
python pipeline/03_build_dataset.py          # outputs data/dataset_city_features.csv and data/dataset_train.csv
python pipeline/04_xgb_downscale.py          # outputs output/eu14_city_vehicles_xgb_2011_2024.csv
python pipeline/05_build_results.py          # outputs output/result_city.csv
```


---

## Output datasets

| File | Dimensions | Description |
|---|---|---|
| `result_city.csv` | 1,004,010 rows × 14 columns | Municipality × year panel for 71,715 municipalities across 14 countries, 2011–2024 |

---

## Data availability

The dataset is deposited at [DOI to be assigned].

---

## Citation

If you use this code or dataset, please cite:

> [Citation to be added upon publication]

---

## License

Code: MIT License.
Data: CC BY 4.0.
