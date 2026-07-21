# -*- coding: utf-8 -*-
"""
Compute the total-fleet rebalancing factor (national total / summed raw
city-level predictions) per country, for figS3_rebalancing_factor.py.
Mirrors 04_xgb_downscale.py but stops before the final rescaling step.

Output: output/rebalance_factor_by_country_total.csv  (region, rescale_factor)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

PROJ_DIR  = Path(__file__).parent.parent
TRAIN_CSV = PROJ_DIR / "data" / "dataset_train.csv"
CITY_CSV  = PROJ_DIR / "data" / "dataset_city_features.csv"
BND_CSV   = PROJ_DIR / "data" / "boundary" / "eu14_city.csv"
KBA_CSV   = PROJ_DIR / "data" / "validation" / "deu_FZ Pkw mit Elektroantrieb Zulassungsbezirk_2102999911177145523.csv"
OUT_CSV   = PROJ_DIR / "output" / "rebalance_factor_by_country_total.csv"

FEATURES = ["gdp_per_cap", "light_per_cap", "light_mean_per_cap", "year"]
NUTS2_COUNTRIES = ["Germany", "France", "Italy", "Spain",
                   "United Kingdom", "Netherlands", "Belgium"]

XGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=1, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)

print("Loading datasets ...")
train_full = pd.read_csv(TRAIN_CSV).dropna(subset=FEATURES + ["value_per_cap"])
city_feat  = pd.read_csv(CITY_CSV)

print("Building Germany NUTS3 (Kreis) augmentation rows ...")
bnd = pd.read_csv(BND_CSV, low_memory=False, usecols=["UID", "GID_0", "CC_2"], dtype={"CC_2": str})
bnd_de = bnd[bnd["GID_0"] == "DEU"].copy()
bnd_de["CC_2"] = bnd_de["CC_2"].str.replace("DE", "", regex=False).str.zfill(5)

feat_de_2023 = city_feat[(city_feat["region"] == "Germany") & (city_feat["year"] == 2023)].copy()
feat_de_2023 = feat_de_2023.merge(bnd_de, on="UID", how="left")
feat_de_2023["light_mean_x_pop"] = feat_de_2023["light_mean"] * feat_de_2023["pop"]

kreis = (
    feat_de_2023.groupby("CC_2")
    .agg(pop=("pop", "sum"), gdp=("gdp", "sum"), light_sum=("light_sum", "sum"),
         light_mean_x_pop=("light_mean_x_pop", "sum"))
    .reset_index()
)
kreis["gdp_per_cap"] = np.where(kreis["pop"] > 0, kreis["gdp"] / kreis["pop"], 0.0)
kreis["light_per_cap"] = np.where(kreis["pop"] > 0, kreis["light_sum"] / kreis["pop"], 0.0)
kreis["light_mean_per_cap"] = np.where(kreis["pop"] > 0, (kreis["light_mean_x_pop"] / kreis["pop"]) / kreis["pop"], 0.0)
kreis["year"] = 2023

kba = pd.read_csv(KBA_CSV, encoding="utf-8-sig")
kba_23 = kba[kba["Berichtszeitpunkt"] == 2023.10].copy()
schluessel_col = [c for c in kba_23.columns if "Schl" in c][0]
kba_23["CC_2"] = kba_23[schluessel_col].astype(str).str.zfill(5)
kba_23["BEV_actual"]  = kba_23["Pkw Insgesamt"] * kba_23["Pkw BEV Anteil"] / 100
kba_23["PHEV_actual"] = kba_23["Pkw Insgesamt"] * kba_23["Pkw Plug In Hybrid Anteil"] / 100
kba_23["ICEV_actual"] = kba_23["Pkw Insgesamt"] - kba_23["BEV_actual"] - kba_23["PHEV_actual"]

kreis = kreis.merge(kba_23[["CC_2", "BEV_actual", "PHEV_actual", "ICEV_actual"]], on="CC_2", how="inner")

aug_rows = []
for param, col in [("EV stock_BEV", "BEV_actual"), ("EV stock_PHEV", "PHEV_actual"), ("EV stock_ICEV", "ICEV_actual")]:
    df = kreis[["gdp_per_cap", "light_per_cap", "light_mean_per_cap", "year", "pop", col]].copy()
    df["value_per_cap"] = np.where(df["pop"] > 0, df[col] / df["pop"], 0.0)
    df["region"] = "Germany"
    df["parameter_pt"] = param
    aug_rows.append(df[["region", "parameter_pt", "year"] + FEATURES[:3] + ["value_per_cap"]])

aug_df = pd.concat(aug_rows, ignore_index=True)
train_aug = pd.concat([train_full, aug_df], ignore_index=True)
PARAMS = sorted(train_aug["parameter_pt"].unique())

print("Training models and generating RAW (pre-rebalance) city predictions ...")
raw_list = []
for param in PARAMS:
    sub = train_aug[train_aug["parameter_pt"] == param]
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(sub[FEATURES].values, sub["value_per_cap"].values)

    cf = city_feat[city_feat["region"].isin(NUTS2_COUNTRIES)][["region", "UID", "pop"] + FEATURES].copy()
    per_cap = np.clip(model.predict(cf[FEATURES].values), 0, None)
    cf["value"] = per_cap * cf["pop"]
    cf["parameter"] = param
    raw_list.append(cf[["region", "year", "parameter", "value"]])

raw = pd.concat(raw_list, ignore_index=True)
raw_sum = raw.groupby("region")["value"].sum()

nat_total = (
    train_full[train_full["region"].isin(NUTS2_COUNTRIES)]
    .groupby("region")["value"].sum()
)

rescale_factor = (nat_total / raw_sum).rename("rescale_factor").reset_index()
print(rescale_factor.to_string(index=False))

rescale_factor.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
