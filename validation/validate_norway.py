# -*- coding: utf-8 -*-
"""
Municipality-level validation of XGBoost downscaling for Norway.
Compares model-predicted BEV stock against SSB (Statistics Norway)
municipality-level registered vehicle data (2011-2023).

Data source:
  SSB Table 07849: Registered vehicles by region, year and type of fuel
  File: nor_07849_20260430-124846.csv
  "Electricity" column = BEV (full electric private cars)

Matching challenges handled:
  1. Bilingual municipality names (Norwegian + Sami) -- extract Norwegian part
  2. Post-2020 municipal mergers -- aggregate model predictions for components
  3. Pre-2012 renames (Borre -> Horten, Nes(Numedal) -> Nesbyen)
  4. Disambiguation suffixes e.g. "Vaaler (Ostfold)"

Output: output/validation_norway/
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.parent

VEH_CSV  = PROJ / "output" / "eu14_city_vehicles_xgb_2011_2023.csv"
BND_CSV  = PROJ / "data" / "boundary" / "eu14_city.csv"
NOR_CSV  = PROJ / "data" / "validation" / "nor_07849_20260430-124846.csv"
OUT_DIR  = PROJ / "output" / "validation_norway"
OUT_DIR.mkdir(exist_ok=True)

YEAR = 2023

# ---------------------------------------------------------------------------
# Name mapping: SSB municipality name -> list of boundary NAME_2 values
# Two types:
#   mergers: new municipality = sum of several old ones
#   renames: 1-to-1 rename
# ---------------------------------------------------------------------------
# Format: ssb_name -> [list of boundary NAME_2]
# Merger map: SSB name -> list of component boundary UIDs (avoids name ambiguity)
# UIDs derived from eu14_city.csv GID_0=NOR rows
MERGER_MAP_UID = {
    "Lillestrøm":    [31002, 31019, 30996],           # Skedsmo, Fet, Sørum (Rælingen stayed separate)
    "Nordre Follo":  [31051, 31050],                  # Ski, Oppegård
    "Indre Østfold": [31095, 31124, 31077, 31078],   # Askim, Eidsberg, Hobøl, Spydeberg
    "Alver":         [30826, 30848, 30816],            # Lindås, Meland, Radøy
    "Bjørnafjorden": [30980, 30974],                   # Os(Hordaland), Fusa
    "Øygarden":      [30882, 30923, 30941],            # Øygarden, Fjell, Sund
    "Færder":        [32072, 32186],                   # Nøtterøy, Tjøme
    "Midt-Telemark": [31184, 31160],                  # Bø(Telemark), Sauherad
    "Kinn":          [30636, 30574],                   # Flora, Vågsøy
    "Sunnfjord":     [30670, 30613, 30664, 30632],    # Gaular, Jølster, Naustdal, Førde
    "Stad":          [30547, 30554, 30545],            # Selje, Eid, Hornindal
    "Fjord":         [30466, 30469],                   # Norddal, Stordal
    "Hustadvika":    [30385, 30411],                   # Eide, Fræna
    "Indre Fosen":   [30292, 30281],                   # Rissa, Leksvik
    "Heim":          [30366, 30326, 30320],            # Halsa, Hemne, Snillfjord
    "Orkland":       [30321, 30309, 30353],            # Orkdal, Agdenes, Meldal
    "Nærøysund":     [30132, 30127],                   # Nærøy, Vikna
    "Senja":              [29917, 29910, 29912, 29920],  # Torsken, Berg, Lenvik, Tranøy
    # Disambiguation via UID (boundary NAME_1 has encoding artifacts)
    "Våler (Østfold)":   [31192],                       # Våler in Østfold county
    "Våler (Hedmark)":   [30746],                       # Våler in Hedmark county
    "Herøy (Møre og Romsdal)": [30498],                # Herøy in Møre og Romsdal
    "Herøy (Nordland)":  [30039],                       # Herøy in Nordland
}
# Keep old name-based map as empty (replaced by UID version above)
MERGER_MAP = {}

# 1-to-1 renames (SSB current name -> boundary old name)
RENAME_MAP = {
    "Horten":   "Borre",   # Borre renamed to Horten in 2012
    "Nesbyen":  "Nes",     # Nes (Numedal/Buskerud) renamed to Nesbyen in 2020
}

# Disambiguation: boundary has two municipalities with same NAME_2
# Map: SSB name -> (NAME_2 in boundary, NAME_1 in boundary)
DISAMBIG_MAP = {
    "Våler (Østfold)": ("Våler", "Østfold"),
    "Våler (Hedmark)": ("Våler", "Hedmark"),
    "Herøy (Møre og Romsdal)": ("Herøy", "Møre og Romsdal"),
    "Herøy (Nordland)": ("Herøy", "Nordland"),
    "Os (Hordaland)":  ("Os", "Hordaland"),
    "Bø (Telemark)":   ("Bø", "Telemark"),
}

# ---------------------------------------------------------------------------
# Municipalities that in SSB 2023 refer to Os in Innlandet (Hedmark), not Hordaland.
# Must disambiguate because boundary has two "Os" rows.
# ---------------------------------------------------------------------------
# Add to DISAMBIG_MAP so "Os" -> Os in Hedmark (SSB K-3435)
DISAMBIG_MAP["Os"] = ("Os", "Hedmark")

# Known Sami-first bilingual names where Norwegian is the SECOND part
SAMI_FIRST = {
    "Hábmer - Hamarøy":                    "Hamarøy",
    "Dielddanuorri - Tjeldsund":           "Tjeldsund",
    "Gáivuotna - Kåfjord - Kaivuono":     "Kåfjord",
    "Deatnu - Tana":                       "Tana",
    "Unjárga - Nesseby":                   "Nesseby",
    "Kárásjohka - Karasjok":              "Karasjok",
    "Guovdageaidnu - Kautokeino":          "Kautokeino",
    "Raarvihke - Røyrvik":                 "Røyrvik",
    "Snåase - Snåsa":                      "Snåsa",
    "Porsanger - Porsángu - Porsanki":     "Porsanger",
    "Nordreisa - Ráisa - Raisi":           "Nordreisa",
    "Storfjord - Omasvuotna - Omasvuono":  "Storfjord",
    "Lyngen - Ivgu - Yykeä":              "Lyngen",
    "Loabák - Lavangen":                   "Lavangen",
    "Gratangen - Rivtták":                 "Gratangen",
    "Aarborte - Hattfjelldal":             "Hattfjelldal",
    "Rana - Raane":                        "Rana",
    "Fauske - Fuossko":                    "Fauske",
    "Sørfold - Fuolldá":                   "Sørfold",
    "Evenes - Evenássi":                   "Evenes",
    "Sortland - Suortá":                   "Sortland",
    "Harstad - Hárstták":                  "Harstad",
    "Hammerfest - Hámmerfeasta":           "Hammerfest",
    "Dielddanuorri - Tjeldsund":           "Tjeldsund",
    "K-21-22 Svalbard and Jan Mayen":      None,  # skip
    "K-23 Continental shelf":              None,   # skip
    "K-Rest Divided municalities and unknown": None,  # skip
}

# ===========================================================================
# 1. Load model predictions
# ===========================================================================
print("Loading model predictions ...")
veh = pd.read_csv(VEH_CSV, low_memory=False)
veh_wide = veh.pivot_table(index=["region", "city", "year"],
                            columns="parameter", values="value",
                            aggfunc="first").reset_index()
veh_wide.columns = [c.replace("EV stock_", "") if c.startswith("EV stock_") else c
                    for c in veh_wide.columns]
veh_no = veh_wide[(veh_wide["region"] == "Norway") & (veh_wide["year"] == YEAR)].copy()
print(f"  Norway cities {YEAR}: {len(veh_no)}")

# ===========================================================================
# 2. Load boundary
# ===========================================================================
print("Loading boundary data ...")
bnd = pd.read_csv(BND_CSV, low_memory=False)
bnd_no = bnd[bnd["GID_0"] == "NOR"][["UID", "NAME_2", "NAME_1"]].copy()
bnd_no["NAME_2"] = bnd_no["NAME_2"].astype(str).str.strip()
bnd_no["NAME_1"] = bnd_no["NAME_1"].astype(str).str.strip()

# Merge city UID -> municipality name
veh_no = veh_no.merge(bnd_no, left_on="city", right_on="UID", how="left")
print(f"  Cities without boundary match: {veh_no['NAME_2'].isna().sum()}")

# ===========================================================================
# 3. Aggregate model predictions to old-boundary municipalities
# ===========================================================================
# city-level -> NAME_2 level (old boundary)
mun_pred = (
    veh_no.groupby(["NAME_2", "NAME_1"])
    .agg(BEV_pred=("BEV", "sum"), PHEV_pred=("PHEV", "sum"),
         ICEV_pred=("ICEV", "sum"), n_cities=("city", "count"))
    .reset_index()
)
mun_pred["EV_pred"] = mun_pred["BEV_pred"] + mun_pred["PHEV_pred"]
print(f"  Old-boundary municipalities in model: {len(mun_pred)}")

# ===========================================================================
# 4. Load SSB data
# ===========================================================================
print(f"Loading SSB validation data ({YEAR}) ...")
nor = pd.read_csv(NOR_CSV, sep=";", skiprows=1, encoding="latin-1")
bev_col  = f"Private cars {YEAR} Electricity"
icev_col_petrol = f"Private cars {YEAR} Petrol"
icev_col_diesel = f"Private cars {YEAR} Diesel"
total_cols = [c for c in nor.columns if c.startswith(f"Private cars {YEAR}")]

nor["BEV_actual"] = pd.to_numeric(nor[bev_col], errors="coerce").fillna(0)
nor["total_actual"] = nor[total_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

# Strip K-XXXX prefix from region names
nor["ssb_name"] = nor["region"].str.replace(r"^K-[\d-]+\s*", "", regex=True).str.strip()
print(f"  SSB municipalities: {len(nor)}")
print(f"  Total BEV {YEAR}: {nor['BEV_actual'].sum():,.0f}")

# ===========================================================================
# 5. Build UID->BEV_pred lookup and track UIDs consumed by mergers
# ===========================================================================
# UID-level predictions (before aggregation to NAME_2)
uid_pred = (
    veh_no.groupby("UID")
    .agg(BEV_pred=("BEV", "sum"))
    .reset_index()
)
uid_bev = dict(zip(uid_pred["UID"], uid_pred["BEV_pred"]))

# UIDs consumed by mergers (should not appear in direct matching)
merger_uids = set()
for uids in MERGER_MAP_UID.values():
    merger_uids.update(uids)

def lookup_merger(uid_list):
    """Sum BEV predictions for a list of boundary UIDs."""
    return sum(uid_bev.get(u, 0.0) for u in uid_list)

# mun_pred excluding merger UIDs (for direct matching)
mun_pred_direct = (
    veh_no[~veh_no["UID"].isin(merger_uids)]
    .groupby(["NAME_2", "NAME_1"])
    .agg(BEV_pred=("BEV", "sum"))
    .reset_index()
)

# ===========================================================================
# 6. Match SSB -> model predictions
# ===========================================================================
records = []
skipped = []

for _, row in nor.iterrows():
    ssb_raw = row["ssb_name"]
    bev_actual = row["BEV_actual"]

    # ---- Skip non-municipality rows ----
    if ssb_raw in SAMI_FIRST and SAMI_FIRST[ssb_raw] is None:
        skipped.append(ssb_raw)
        continue

    # ---- Resolve SSB name ----
    # Priority 1: Sami-first bilingual exception
    if ssb_raw in SAMI_FIRST:
        nor_name = SAMI_FIRST[ssb_raw]
    # Priority 2: Regular bilingual (Norwegian first)
    elif " - " in ssb_raw:
        nor_name = ssb_raw.split(" - ")[0].strip()
    # Priority 3: Disambiguation suffix "(County)"
    elif "(" in ssb_raw:
        nor_name = ssb_raw
    else:
        nor_name = ssb_raw

    # ---- UID-based merger ----
    if nor_name in MERGER_MAP_UID:
        bev_pred = lookup_merger(MERGER_MAP_UID[nor_name])
        records.append({"ssb_name": ssb_raw, "nor_name": nor_name,
                        "BEV_actual": bev_actual, "BEV_pred": bev_pred,
                        "match_type": "merger"})
        continue

    # ---- Rename ----
    if nor_name in RENAME_MAP:
        boundary_name = RENAME_MAP[nor_name]
        row2 = mun_pred_direct[mun_pred_direct["NAME_2"] == boundary_name]
        bev_pred = row2["BEV_pred"].sum() if len(row2) > 0 else np.nan
        records.append({"ssb_name": ssb_raw, "nor_name": nor_name,
                        "BEV_actual": bev_actual, "BEV_pred": bev_pred,
                        "match_type": "rename"})
        continue

    # ---- Disambiguation (including "Os" -> Hedmark) ----
    if nor_name in DISAMBIG_MAP:
        n2, n1 = DISAMBIG_MAP[nor_name]
        row2 = mun_pred_direct[(mun_pred_direct["NAME_2"] == n2) &
                                (mun_pred_direct["NAME_1"] == n1)]
        bev_pred = row2["BEV_pred"].sum() if len(row2) > 0 else np.nan
        records.append({"ssb_name": ssb_raw, "nor_name": nor_name,
                        "BEV_actual": bev_actual, "BEV_pred": bev_pred,
                        "match_type": "disambig"})
        continue

    # ---- Direct name match (using non-merger municipalities only) ----
    row2 = mun_pred_direct[mun_pred_direct["NAME_2"] == nor_name]
    if len(row2) > 0:
        bev_pred = row2["BEV_pred"].sum()
        records.append({"ssb_name": ssb_raw, "nor_name": nor_name,
                        "BEV_actual": bev_actual, "BEV_pred": bev_pred,
                        "match_type": "direct"})
    else:
        records.append({"ssb_name": ssb_raw, "nor_name": nor_name,
                        "BEV_actual": bev_actual, "BEV_pred": np.nan,
                        "match_type": "unmatched"})

comp = pd.DataFrame(records)
matched = comp[comp["BEV_pred"].notna()].copy()
unmatched = comp[comp["BEV_pred"].isna()].copy()

print(f"\n  Skipped (non-municipality): {len(skipped)}")
print(f"  Matched: {len(matched)} ({matched['BEV_actual'].sum():,.0f} BEV actual)")
print(f"  Unmatched: {len(unmatched)}")
if len(unmatched) > 0:
    print(f"  Unmatched BEV: {unmatched['BEV_actual'].sum():,.0f}")
    print("  Unmatched names:", unmatched["ssb_name"].tolist())

# ===========================================================================
# 7. Metrics
# ===========================================================================
def metrics(actual, pred, label):
    mask = (actual > 0) & np.isfinite(actual) & np.isfinite(pred)
    a, p = actual[mask], pred[mask]
    n = mask.sum()
    if n < 2:
        print(f"\n  [{label}]  n={n} -- insufficient data")
        return None
    corr = np.corrcoef(a, p)[0, 1]
    ss_res = ((a - p) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    mape = (np.abs(a - p) / a).mean() * 100
    mae  = np.abs(a - p).mean()
    bias = (p - a).mean()
    print(f"\n  [{label}]  n={n}")
    print(f"    R2   = {r2:.3f}")
    print(f"    r    = {corr:.3f}")
    print(f"    MAPE = {mape:.1f}%")
    print(f"    MAE  = {mae:,.0f} vehicles")
    print(f"    Bias = {bias:+,.0f} vehicles (pred - actual)")
    return {"label": label, "n": int(n), "r2": r2, "r": corr,
            "mape": mape, "mae": mae, "bias": bias}

print("\n" + "=" * 55)
print(f"VALIDATION RESULTS -- Norway Municipalities ({YEAR})")
print("=" * 55)
results = []
r = metrics(matched["BEV_actual"], matched["BEV_pred"], "BEV")
if r: results.append(r)

national_ratio = matched["BEV_pred"].sum() / matched["BEV_actual"].sum()
print(f"\n  National BEV ratio (model/SSB): {national_ratio:.3f}")

# Scale-adjusted
scale = matched["BEV_actual"].sum() / matched["BEV_pred"].sum()
matched["BEV_pred_scaled"] = matched["BEV_pred"] * scale
print()
r2 = metrics(matched["BEV_actual"], matched["BEV_pred_scaled"], "BEV (scale-adjusted)")

# ===========================================================================
# 8. Match type breakdown
# ===========================================================================
print("\n  Match type breakdown:")
print(comp[comp["BEV_pred"].notna()].groupby("match_type").agg(
    n=("ssb_name","count"), bev_actual=("BEV_actual","sum")).to_string())

# ===========================================================================
# 9. Save outputs
# ===========================================================================
summary = pd.DataFrame(results)
summary.to_csv(OUT_DIR / "validation_summary.csv", index=False)

comp_out = comp[comp["BEV_pred"].notna()].copy()
comp_out["BEV_ratio"] = comp_out["BEV_pred"] / comp_out["BEV_actual"]
comp_out.to_csv(OUT_DIR / "norway_comparison.csv", index=False, float_format="%.1f")

print("\n--- Top 10 overestimated ---")
over = comp_out[comp_out["BEV_actual"] > 0].nlargest(10, "BEV_ratio")[["nor_name","BEV_pred","BEV_actual","BEV_ratio"]]
print(over.to_string(index=False))

print("\n--- Top 10 underestimated ---")
under = comp_out[comp_out["BEV_actual"] > 0].nsmallest(10, "BEV_ratio")[["nor_name","BEV_pred","BEV_actual","BEV_ratio"]]
print(under.to_string(index=False))

print(f"\nOutputs saved to: {OUT_DIR}")
