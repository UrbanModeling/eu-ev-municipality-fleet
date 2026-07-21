# -*- coding: utf-8 -*-
"""
NUTS-2 level validation of the no-nightlight ablation model
(04_xgb_downscale_nolight.py). Same as validate_nuts2.py otherwise.

Output: output/validation_nuts2_nolight/
"""

import pandas as pd, numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.parent
DATA = PROJ / "data" / "validation"

VEH_CSV    = PROJ / "output" / "eu14_city_vehicles_xgb_2011_2024_nolight.csv"
BND_CSV    = PROJ / "data" / "boundary" / "eu14_city.csv"
LAU_NUTS_CSV = PROJ / "data" / "boundary" / "lau_nuts.csv"
ESTAT_TSV  = DATA / "eurostat_nuts3_vehicles.tsv"
LABELS_TSV = DATA / "eurostat_nuts_labels.tsv"
OUT_DIR    = PROJ / "output" / "validation_nuts2_nolight"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def r2_score(y, yp):
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading pre-computed predictions ...")
veh = pd.read_csv(VEH_CSV, low_memory=False)
veh["value"] = pd.to_numeric(veh["value"], errors="coerce").fillna(0)
city_total = (veh.groupby(["region", "city", "year"])["value"]
              .sum().reset_index(name="our_total"))
print(f"  City-year-total rows: {len(city_total):,}")

print("Loading boundary + LAU-NUTS2 mapping ...")
bnd = pd.read_csv(BND_CSV, low_memory=False, usecols=["UID", "GID_0", "NAME_0"])
lau_nuts = pd.read_csv(LAU_NUTS_CSV, low_memory=False)
bnd = bnd.merge(lau_nuts, on="UID", how="left")
uid_nuts2 = bnd.set_index("UID")["nuts2_code"]
city_total["nuts2_code"] = city_total["city"].map(uid_nuts2)

print("Loading Eurostat reference data ...")
df = pd.read_csv(ESTAT_TSV, sep="\t")
first = df.columns[0]
parts = df[first].str.split(",", expand=True)
parts.columns = ["freq", "vehicle", "unit", "geo"]
for c in parts.columns:
    parts[c] = parts[c].str.strip()
df2 = pd.concat([parts, df.drop(columns=[first])], axis=1)
year_cols = [c for c in df2.columns if c.strip().isdigit()]
df_long = df2.melt(id_vars=["freq", "vehicle", "unit", "geo"],
                   value_vars=year_cols, var_name="year", value_name="value")
df_long["year"]  = df_long["year"].str.strip().astype(int)
df_long["value"] = pd.to_numeric(
    df_long["value"].astype(str).str.replace(":", "").str.strip(), errors="coerce")
estat_all = (df_long[(df_long["vehicle"] == "CAR") & (df_long["unit"] == "NR")
                     & df_long["value"].notna()]
             [["geo", "year", "value"]].copy())

try:
    labels = pd.read_csv(LABELS_TSV, sep="\t", header=None,
                         names=["code", "name"]).dropna()
    code_name = labels.set_index("code")["name"].to_dict()
except Exception:
    code_name = {}

# ── Per-country validation ─────────────────────────────────────────────────
COUNTRIES = {
    "Germany":        ("DEU", (2011, 2024)),
    "France":         ("FRA", (2011, 2024)),
    "Italy":          ("ITA", (2011, 2024)),
    "Spain":          ("ESP", (2011, 2024)),
    "United Kingdom": ("GBR", (2011, 2024)),
    "Netherlands":    ("NLD", (2011, 2024)),
    "Belgium":        ("BEL", (2011, 2024)),
}

summary_rows = []
all_per_region = []

for cname, (gid0, yr_range) in COUNTRIES.items():
    print(f"\n{'='*55}")
    print(f"  {cname}")

    bnd_c = bnd[bnd["GID_0"] == gid0]
    total  = len(bnd_c)
    mapped = bnd_c["nuts2_code"].notna().sum()
    print(f"  Cities: {total}, mapped to NUTS-2: {mapped} ({mapped/total*100:.1f}%)")

    pred = (city_total[city_total["region"] == cname]
            .dropna(subset=["nuts2_code"])
            .query(f"year >= {yr_range[0]} and year <= {yr_range[1]}")
            .groupby(["nuts2_code", "year"])["our_total"]
            .sum().reset_index())

    nuts2_codes = bnd_c["nuts2_code"].dropna().unique()
    estat = (estat_all[estat_all["geo"].isin(nuts2_codes) &
                       estat_all["year"].between(*yr_range)]
             .rename(columns={"geo": "nuts2_code", "value": "estat_total"}))
    print(f"  Eurostat: {estat['nuts2_code'].nunique()} NUTS-2 x {estat['year'].nunique()} years")

    merged = pred.merge(estat, on=["nuts2_code", "year"]).dropna()
    print(f"  Matched: {merged['nuts2_code'].nunique()} NUTS-2 x {merged['year'].nunique()} years")

    y  = merged["estat_total"].values
    yp = merged["our_total"].values
    r2_val  = r2_score(y, yp)
    mae_val = float((np.abs(yp - y) / y * 100).mean())

    print(f"\n  R2  : {r2_val:.4f}")
    print(f"  MAE : {mae_val:.1f}%")

    summary_rows.append({
        "country":  cname,
        "nuts2_n":  merged["nuts2_code"].nunique(),
        "years":    f"{yr_range[0]}-{yr_range[1]}",
        "R2":       round(r2_val, 4),
        "MAE_pct":  round(mae_val, 1),
    })

    per_r = (merged.groupby("nuts2_code")
             .apply(lambda g: pd.Series({
                 "R2":  r2_score(g["estat_total"].values, g["our_total"].values),
                 "MAE": float((np.abs(g["our_total"] - g["estat_total"])
                               / g["estat_total"] * 100).mean())
             }), include_groups=False)
             .reset_index())
    per_r["name"]    = per_r["nuts2_code"].map(code_name)
    per_r["country"] = cname
    all_per_region.append(per_r)

    merged["nuts2_name"] = merged["nuts2_code"].map(code_name)
    merged.to_csv(OUT_DIR / f"{gid0.lower()}_nuts2_comparison.csv", index=False)

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("SUMMARY")
summary = pd.DataFrame(summary_rows)
print(summary[["country", "nuts2_n", "years", "R2", "MAE_pct"]].to_string(index=False))
summary.to_csv(OUT_DIR / "validation_summary.csv", index=False)

all_per_region_df = pd.concat(all_per_region, ignore_index=True)
all_per_region_df.to_csv(OUT_DIR / "per_nuts2_errors.csv", index=False)

# Overall pooled R2/MAE across all countries (single headline number)
all_merged = pd.concat(
    [pd.read_csv(OUT_DIR / f"{gid0.lower()}_nuts2_comparison.csv") for _, (gid0, _) in COUNTRIES.items()],
    ignore_index=True,
)
y, yp = all_merged["estat_total"].values, all_merged["our_total"].values
overall_r2 = r2_score(y, yp)
overall_mae = float((np.abs(yp - y) / y * 100).mean())
print(f"\nOverall pooled (n={len(all_merged)}): R2={overall_r2:.4f}  MAE={overall_mae:.1f}%")

print(f"\nOutputs saved to: {OUT_DIR}")
