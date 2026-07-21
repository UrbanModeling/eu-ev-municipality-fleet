# -*- coding: utf-8 -*-
"""
SHAP feature attribution for the model in pipeline/04_xgb_downscale.py, at
municipality level for the seven NUTS-2 validation countries (2024,
population-weighted).

Output: output/shap/shap_importance_population_weighted.csv, shap_summary_{param}.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_DIR  = Path(__file__).parent.parent
TRAIN_CSV = PROJ_DIR / "data" / "dataset_train.csv"
CITY_CSV  = PROJ_DIR / "data" / "dataset_city_features.csv"
BND_CSV   = PROJ_DIR / "data" / "boundary" / "eu14_city.csv"
KBA_CSV   = PROJ_DIR / "data" / "validation" / "deu_FZ Pkw mit Elektroantrieb Zulassungsbezirk_2102999911177145523.csv"
OUT_DIR   = PROJ_DIR / "output" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["gdp_per_cap", "light_per_cap", "light_mean_per_cap", "year"]
SNAPSHOT_YEAR = 2024
NUTS2_COUNTRIES = ["Germany", "France", "Italy", "Spain",
                   "United Kingdom", "Netherlands", "Belgium"]

XGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=1, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)

print("Loading datasets ...")
train_full = pd.read_csv(TRAIN_CSV).dropna(subset=FEATURES + ["value_per_cap"])
city_full  = pd.read_csv(CITY_CSV)

print("Building Germany NUTS3 (Kreis) augmentation rows from KBA 2023.10 ...")
bnd = pd.read_csv(BND_CSV, low_memory=False, usecols=["UID", "GID_0", "CC_2"], dtype={"CC_2": str})
bnd_de = bnd[bnd["GID_0"] == "DEU"].copy()
bnd_de["CC_2"] = bnd_de["CC_2"].str.replace("DE", "", regex=False).str.zfill(5)

feat_de_2023 = city_full[(city_full["region"] == "Germany") & (city_full["year"] == 2023)].copy()
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
    df["parameter_pt"] = param
    aug_rows.append(df[["parameter_pt"] + FEATURES + ["value_per_cap"]])

aug_df = pd.concat(aug_rows, ignore_index=True)
train_aug = pd.concat([train_full, aug_df], ignore_index=True)
PARAMS = sorted(train_aug["parameter_pt"].unique())
print(f"  Vehicle types: {PARAMS}")
print(f"  Training rows (with DE Kreis augmentation): {len(train_aug)}  (was {len(train_full)})")

# ── Municipality sample: 7 NUTS-2 validation countries, snapshot year ──────
city_sample = city_full[
    (city_full["region"].isin(NUTS2_COUNTRIES)) & (city_full["year"] == SNAPSHOT_YEAR)
][["region", "UID", "pop"] + FEATURES].reset_index(drop=True)
print(f"  City sample ({SNAPSHOT_YEAR}, 7 NUTS-2 countries): {city_sample.shape}")
print(city_sample["region"].value_counts().to_string())

pop_w = city_sample["pop"].values
pop_w = pop_w / pop_w.sum()

rows = []
for param in PARAMS:
    print(f"\n{'='*55}\n  {param}")
    sub = train_aug[train_aug["parameter_pt"] == param]
    X_train = sub[FEATURES]
    y_train = sub["value_per_cap"].values

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train.values, y_train)

    explainer = shap.TreeExplainer(model)
    X_city = city_sample[FEATURES]
    sv_city = explainer.shap_values(X_city)

    # population-weighted mean absolute SHAP per feature
    imp = (np.abs(sv_city) * pop_w[:, None]).sum(axis=0)
    imp_pct = imp / imp.sum() * 100
    for f, v, p in zip(FEATURES, imp, imp_pct):
        rows.append({"parameter": param, "feature": f,
                      "pop_weighted_mean_abs_shap": v, "pct_contribution": p})
    print("  [Pop-weighted, 7 NUTS-2 countries, " + str(SNAPSHOT_YEAR) + "]  " +
          ", ".join(f"{f}={p:.1f}%" for f, p in zip(FEATURES, imp_pct)))

    plt.figure()
    shap.summary_plot(sv_city, X_city, show=False)
    plt.title(f"{param} - SHAP ({SNAPSHOT_YEAR}, 7 NUTS-2 countries, pop-weighted)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"shap_summary_{param.replace(' ', '_')}.png", dpi=150)
    plt.close()

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_DIR / "shap_importance_population_weighted.csv", index=False)

print(f"\n{'='*55}\nSUMMARY - population-weighted % contribution by feature\n")
print(out_df.pivot(index="parameter", columns="feature", values="pct_contribution")
      [FEATURES].round(1).to_string())

print(f"\nOutputs saved to: {OUT_DIR}")
