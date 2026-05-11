# -*- coding: utf-8 -*-
"""
Step 03 (v5) – XGBoost downscale: predict city-level vehicle stocks
using nighttime lights, GDP, and population as features.

Strategy
--------
Training:  read dataset_train.csv (national features + vehicle counts),
           fit one XGBoost per vehicle type.
Inference: read dataset_city_features.csv (city features), apply models
           → city-level vehicle stock predictions.
           Negative predictions are clipped to zero.

Run 01_build_datasets.py first to generate the two input datasets.

Inputs  (from data/ subfolder)
------
data/dataset_train.csv         national panel, features + targets
data/dataset_city_features.csv city × year panel, features only

Output
------
E:/Data/Project_A/Data/eu14/road/eu14_city_vehicles_xgb_2011_2023.csv
    Columns: region, city (UID), year, parameter, value
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

# ── Paths ──────────────────────────────────────────────────────────────────
PROJ_DIR   = Path(__file__).parent.parent
PROJ_DIR.joinpath("output").mkdir(exist_ok=True)
OUTPUT_CSV = PROJ_DIR / "output" / "eu14_city_vehicles_xgb_2011_2023.csv"
TRAIN_CSV  = PROJ_DIR / "data" / "dataset_train.csv"
CITY_CSV   = PROJ_DIR / "data" / "dataset_city_features.csv"

FEATURES_PC   = ["gdp_per_cap", "light_per_cap", "light_mean_per_cap", "year"]
FEATURES_CITY = ["gdp_per_cap", "light_per_cap", "light_mean_per_cap", "year"]

XGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load datasets
# ══════════════════════════════════════════════════════════════════════════════
print("Loading datasets ...")
train_full = pd.read_csv(TRAIN_CSV)
city_feat  = pd.read_csv(CITY_CSV)
print(f"  Training set : {train_full.shape}")
print(f"  City features: {city_feat.shape}")

train_full = train_full.dropna(subset=FEATURES_PC + ["value_per_cap"])

PARAMS = sorted(train_full["parameter_pt"].unique())
print(f"  Vehicle types to model: {PARAMS}")
print(f"  Training samples (per type): {len(train_full) // len(PARAMS)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Train one XGBoost model per vehicle type  (target: vehicles per capita)
# ══════════════════════════════════════════════════════════════════════════════
print("\nTraining XGBoost models (per-capita) ...")
models = {}
for param in PARAMS:
    sub = train_full[train_full["parameter_pt"] == param]
    X = sub[FEATURES_PC].values
    y = sub["value_per_cap"].values

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    models[param] = model

    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  {param:25s}  n={len(sub):3d}  in-sample R2={r2:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Predict city-level vehicle stocks
#    predicted_per_cap × city_pop = city vehicles
# ══════════════════════════════════════════════════════════════════════════════
print("\nPredicting city-level vehicle stocks ...")
city_pred_list = []

for param, model in models.items():
    sub = city_feat[["region", "UID", "pop"] + FEATURES_CITY].copy()
    per_cap = np.clip(model.predict(sub[FEATURES_CITY].values), 0, None)
    sub["value"] = per_cap * sub["pop"]
    sub["parameter"] = param
    city_pred_list.append(
        sub[["region", "UID", "year", "parameter", "value"]]
    )

out = (
    pd.concat(city_pred_list, ignore_index=True)
    .rename(columns={"UID": "city"})
    .sort_values(["region", "city", "year", "parameter"])
    .reset_index(drop=True)
)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Rescale city predictions to match national totals exactly
# ══════════════════════════════════════════════════════════════════════════════
print("\nRescaling city predictions to match national totals ...")

nat_totals = (
    train_full.groupby(["region", "year", "parameter_pt"])["value"]
    .sum()
    .reset_index()
    .rename(columns={"parameter_pt": "parameter", "value": "nat_total"})
)

city_sums = (
    out.groupby(["region", "year", "parameter"])["value"]
    .sum()
    .reset_index()
    .rename(columns={"value": "city_sum"})
)

out = (
    out
    .merge(nat_totals, on=["region", "year", "parameter"], how="left")
    .merge(city_sums,  on=["region", "year", "parameter"], how="left")
)

out["value"] = np.where(
    out["city_sum"] > 0,
    out["value"] * out["nat_total"] / out["city_sum"],
    0.0
)
out = out.drop(columns=["nat_total", "city_sum"])

# ── Verify rescaling ──────────────────────────────────────────────────────────
print("\nNational totals check — predicted vs. actual (2022, BEV):")
check_pred = (
    out[(out["year"] == 2022) & (out["parameter"] == "EV stock_BEV")]
    .groupby("region")["value"].sum().round(0)
)
check_actual = (
    train_full[
        (train_full["year"] == 2022) &
        (train_full["parameter_pt"] == "EV stock_BEV")
    ]
    .set_index("region")["value"].round(0)
)
comparison = pd.DataFrame({"predicted_sum": check_pred, "actual": check_actual})
print(comparison.to_string())

out.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
print(f"Shape: {out.shape}")
