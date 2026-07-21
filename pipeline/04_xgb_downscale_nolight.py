# -*- coding: utf-8 -*-
"""
Step 04 (no-light ablation) - same as 04_xgb_downscale.py without the
nightlight features.

Output: output/eu14_city_vehicles_xgb_2011_2024_nolight.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

PROJ_DIR   = Path(__file__).parent.parent
PROJ_DIR.joinpath("output").mkdir(exist_ok=True)
OUTPUT_CSV = PROJ_DIR / "output" / "eu14_city_vehicles_xgb_2011_2024_nolight.csv"
TRAIN_CSV  = PROJ_DIR / "data" / "dataset_train.csv"
CITY_CSV   = PROJ_DIR / "data" / "dataset_city_features.csv"
BND_CSV    = PROJ_DIR / "data" / "boundary" / "eu14_city.csv"
KBA_CSV    = PROJ_DIR / "data" / "validation" / "deu_FZ Pkw mit Elektroantrieb Zulassungsbezirk_2102999911177145523.csv"

FEATURES_FULL = ["gdp_per_cap", "light_per_cap", "light_mean_per_cap", "year"]  # used to build aug rows
FEATURES = ["gdp_per_cap", "year"]  # used for training/prediction (no nightlight)

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


print("Loading datasets ...")
train_full = pd.read_csv(TRAIN_CSV)
city_feat  = pd.read_csv(CITY_CSV)
print(f"  Training set : {train_full.shape}")
print(f"  City features: {city_feat.shape}")

train_full = train_full.dropna(subset=FEATURES_FULL + ["value_per_cap"])


print("\nBuilding Germany NUTS3 (Kreis) augmentation rows from KBA 2023.10 ...")
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
kreis["light_mean_per_cap"] = np.where(
    kreis["pop"] > 0, (kreis["light_mean_x_pop"] / kreis["pop"]) / kreis["pop"], 0.0
)
kreis["year"] = 2023
print(f"  Kreise with features: {len(kreis)}")

kba = pd.read_csv(KBA_CSV, encoding="utf-8-sig")
kba_23 = kba[kba["Berichtszeitpunkt"] == 2023.10].copy()
schluessel_col = [c for c in kba_23.columns if "Schl" in c][0]
kba_23["CC_2"] = kba_23[schluessel_col].astype(str).str.zfill(5)
kba_23["BEV_actual"]  = kba_23["Pkw Insgesamt"] * kba_23["Pkw BEV Anteil"] / 100
kba_23["PHEV_actual"] = kba_23["Pkw Insgesamt"] * kba_23["Pkw Plug In Hybrid Anteil"] / 100
kba_23["ICEV_actual"] = kba_23["Pkw Insgesamt"] - kba_23["BEV_actual"] - kba_23["PHEV_actual"]

kreis = kreis.merge(kba_23[["CC_2", "BEV_actual", "PHEV_actual", "ICEV_actual"]], on="CC_2", how="inner")
print(f"  Kreise matched to KBA: {len(kreis)}")

aug_rows = []
for param, col in [("EV stock_BEV", "BEV_actual"), ("EV stock_PHEV", "PHEV_actual"), ("EV stock_ICEV", "ICEV_actual")]:
    df = kreis[["gdp_per_cap", "light_per_cap", "light_mean_per_cap", "year", "pop", col]].copy()
    df["value_per_cap"] = np.where(df["pop"] > 0, df[col] / df["pop"], 0.0)
    df["region"] = "Germany"
    df["parameter_pt"] = param
    aug_rows.append(df[["region", "parameter_pt", "year"] + FEATURES_FULL[:3] + ["value_per_cap"]])

aug_df = pd.concat(aug_rows, ignore_index=True)
print(f"  Augmentation rows added: {len(aug_df)}  (across BEV/PHEV/ICEV; FCEV uses national panel only)")

train_aug = pd.concat([train_full, aug_df], ignore_index=True)

PARAMS = sorted(train_aug["parameter_pt"].unique())
print(f"\n  Vehicle types to model: {PARAMS}")
print(f"  Features used (no nightlight): {FEATURES}")
print(f"  Total training rows (national panel + DE Kreis augmentation): {len(train_aug)}  (was {len(train_full)})")


print("\nTraining XGBoost models (pooled, per-capita, no nightlight, +DE NUTS3 augmentation) ...")
models = {}
for param in PARAMS:
    sub = train_aug[train_aug["parameter_pt"] == param]
    X = sub[FEATURES].values
    y = sub["value_per_cap"].values

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    models[param] = model

    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  {param:25s}  n={len(sub):4d}  in-sample R2={r2:.4f}")


print("\nPredicting city-level vehicle stocks ...")
city_pred_list = []
for param, model in models.items():
    sub = city_feat[["region", "UID", "pop"] + FEATURES].copy()
    per_cap = np.clip(model.predict(sub[FEATURES].values), 0, None)
    sub["value"] = per_cap * sub["pop"]
    sub["parameter"] = param
    city_pred_list.append(sub[["region", "UID", "year", "parameter", "value"]])

out = (
    pd.concat(city_pred_list, ignore_index=True)
    .rename(columns={"UID": "city"})
    .sort_values(["region", "city", "year", "parameter"])
    .reset_index(drop=True)
)

print("\nRescaling city predictions to match national totals ...")
nat_totals = (
    train_full.groupby(["region", "year", "parameter_pt"])["value"]
    .sum().reset_index().rename(columns={"parameter_pt": "parameter", "value": "nat_total"})
)
city_sums = (
    out.groupby(["region", "year", "parameter"])["value"]
    .sum().reset_index().rename(columns={"value": "city_sum"})
)
out = out.merge(nat_totals, on=["region", "year", "parameter"], how="left") \
         .merge(city_sums,  on=["region", "year", "parameter"], how="left")
out["value"] = np.where(out["city_sum"] > 0, out["value"] * out["nat_total"] / out["city_sum"], 0.0)
out = out.drop(columns=["nat_total", "city_sum"])

CHECK_YEAR = int(train_full["year"].max())
print(f"\nNational totals check - predicted vs. actual ({CHECK_YEAR}, BEV):")
check_pred = (
    out[(out["year"] == CHECK_YEAR) & (out["parameter"] == "EV stock_BEV")]
    .groupby("region")["value"].sum().round(0)
)
check_actual = (
    train_full[(train_full["year"] == CHECK_YEAR) & (train_full["parameter_pt"] == "EV stock_BEV")]
    .set_index("region")["value"].round(0)
)
print(pd.DataFrame({"predicted_sum": check_pred, "actual": check_actual}).to_string())

out.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
print(f"Shape: {out.shape}")
