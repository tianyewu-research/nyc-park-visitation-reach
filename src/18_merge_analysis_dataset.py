"""
18_merge_analysis_dataset.py
Merge all interim variables into a single CBG-level analysis dataset.
Input:  data/interim/cbg_nyc.gpkg + all interim CSVs
Output: data/processed/analysis_cbg.gpkg
        outputs/tables/descriptive_stats.csv
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
PROCESSED.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg")

sources = {
    "acs_cbg.csv": None,
    "parks_cbg.csv": None,  # quality_index_cbg_svc used as default; quality_index_cbg (1km) kept as alternative
    "access_cbg.csv": None,
    "use_cbg.csv": None,
    "out_nbhd_cbg.csv": None,
    "local_ratio_cbg.csv": None,   # LPUR grid + mean_trip_km (new framework, 15b)
    "walkability_cbg.csv": None,
    "transit_cbg.csv": None,
}

for fname in sources:
    df = pd.read_csv(INTERIM / fname)
    df["GEOID"] = df["GEOID"].astype(str).str.zfill(12)
    sources[fname] = df
    print(f"  {fname}: {len(df)} rows, cols={list(df.columns)}")

for fname, df in sources.items():
    cbg = cbg.merge(df, on="GEOID", how="left")

# Headline park-use measure:
#   use_c = annual park visits by residents / resident population (visits per resident).
#   log_use_c = log1p(winsorized use_c) is the modelling form.
# V_c is Advan-SCALED (population-representative), so coverage/sampling is already corrected
# at source — no device normalization (V_c/D_c) is needed. Dividing by total_pop is per-capita
# (exposure) normalization, a separate and still-required step. See 14_ header / workflow doc.
# Extreme control (advisor 2026-06-26): use_c extremes come from small-population CBGs (the
# denominator), not low device observation (obs_devices>=10 for all CBGs). Winsorize both tails
# at WINSOR_P/(1-WINSOR_P); log already symmetrizes, winsorize removes the fat tails
# (kurtosis 3.9 -> 0.5 at 1%). 1% vs 2.5% compared: 1% caps only 2% of CBGs for near-normal
# kurtosis -> chosen. Switch by changing WINSOR_P.
WINSOR_P = 0.01
cbg["use_c"] = cbg["V_c"] / cbg["total_pop"]
cbg.loc[cbg["total_pop"] == 0, "use_c"] = None
_lo, _hi = cbg["use_c"].quantile(WINSOR_P), cbg["use_c"].quantile(1 - WINSOR_P)
cbg["use_c_w"] = cbg["use_c"].clip(_lo, _hi)        # winsorized (for modelling)
cbg["log_use_c"] = np.log1p(cbg["use_c_w"])

# Access (2SFCA, acres/person within 800m) is right-skewed; log fits the first stage far
# better (R2 0.062 vs 0.026). Floor the 1.6% zero-access CBGs (no park in 800m) at half the
# minimum positive value so they form the lowest-access tier rather than dropping out of log.
_acc = cbg["access_score"]
_pos_min = _acc[_acc > 0].min()
cbg["log_access"] = np.log(_acc.clip(lower=_pos_min / 2))

out_path = PROCESSED / "analysis_cbg.gpkg"
cbg.to_file(out_path, driver="GPKG")

analysis_vars = [
    "total_pop", "median_hh_income", "pct_nonwhite", "pop_density",
    "park_area_acres", "park_count", "quality_index_cbg",
    "access_score", "V_c", "obs_devices", "use_c", "log_use_c",
    "out_park_ratio", "walkability_c", "transit_freq", "transit_coverage",
    "quality_index_cbg_svc",
]

desc = cbg[analysis_vars].describe().T
desc["missing"] = cbg[analysis_vars].isna().sum()
desc["missing_pct"] = (desc["missing"] / len(cbg) * 100).round(1)
desc.to_csv(TABLES / "descriptive_stats.csv")

complete = cbg[["access_score", "use_c", "median_hh_income", "pct_nonwhite",
                "walkability_c", "transit_freq", "quality_index_cbg_svc", "out_park_ratio"]].dropna()

print(f"\nTotal CBGs: {len(cbg)}")
print(f"Complete cases (all key vars non-missing): {len(complete)} ({100*len(complete)/len(cbg):.1f}%)")
print(f"\nMissing values:")
for var in analysis_vars:
    n_miss = cbg[var].isna().sum()
    if n_miss > 0:
        print(f"  {var}: {n_miss} ({100*n_miss/len(cbg):.1f}%)")

print(f"\nSaved: {out_path}")
print(f"Descriptive stats: {TABLES / 'descriptive_stats.csv'}")

# --- self-check ---
_c = gpd.read_file(PROCESSED / "analysis_cbg.gpkg")
assert len(_c) == 6807 and {"log_use_c", "log_access"}.issubset(_c.columns)
print("[self-check] 18 merge OK")
