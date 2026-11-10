"""
diagnostics_predictor_collinearity.py  (standalone diagnostic, not part of the 10-23 chain)
Correlation matrix + VIF among RQ2/RQ3 predictors and controls.
Run twice for quality_index: svc (ParkServe service area) vs 1km buffer,
to quantify the quality–access collinearity discussed for RQ3.

Input:  data/processed/analysis_cbg_final.gpkg
Output: outputs/tables/predictor_correlation.csv
        outputs/tables/predictor_vif.csv
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"

cbg = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg")

predictors = [
    "access_score",
    "quality_index_cbg_svc",   # default
    "quality_index_cbg",       # 1km alternative
    "walkability_c",
    "transit_freq",
    "out_park_ratio",
    "park_area_acres",
    "median_hh_income",
    "pct_nonwhite",
    "pop_density",
]

# ── Correlation matrix (pairwise complete) ──
corr = cbg[predictors].corr(method="pearson")
corr.round(3).to_csv(TABLES / "predictor_correlation.csv")
print("=== Pearson correlation (key cells) ===")
print(f"quality_svc  vs access_score : {corr.loc['quality_index_cbg_svc','access_score']:+.3f}")
print(f"quality_1km  vs access_score : {corr.loc['quality_index_cbg','access_score']:+.3f}")
print(f"walkability  vs transit_freq : {corr.loc['walkability_c','transit_freq']:+.3f}")
print(f"walkability  vs pop_density  : {corr.loc['walkability_c','pop_density']:+.3f}")
print(f"income       vs nonwhite     : {corr.loc['median_hh_income','pct_nonwhite']:+.3f}")

# ── VIF, two model specs (svc vs 1km quality) ──
def compute_vif(cols, label):
    d = cbg[cols].dropna()
    X = add_constant(d)
    rows = []
    for i, c in enumerate(X.columns):
        if c == "const":
            continue
        rows.append({"variable": c, "VIF": variance_inflation_factor(X.values, i)})
    out = pd.DataFrame(rows)
    out["spec"] = label
    print(f"\n=== VIF [{label}]  (n={len(d)}) ===")
    print(out[["variable", "VIF"]].round(2).to_string(index=False))
    return out

rq3_base = ["walkability_c", "transit_freq", "out_park_ratio",
            "park_area_acres", "median_hh_income", "pct_nonwhite",
            "pop_density", "access_score"]

vif_svc = compute_vif(rq3_base + ["quality_index_cbg_svc"], "quality=svc")
vif_1km = compute_vif(rq3_base + ["quality_index_cbg"], "quality=1km")

pd.concat([vif_svc, vif_1km], ignore_index=True).round(3).to_csv(
    TABLES / "predictor_vif.csv", index=False)

print("\nRule of thumb: VIF > 5 = moderate concern, > 10 = serious collinearity.")
print(f"Saved: {TABLES/'predictor_correlation.csv'}, {TABLES/'predictor_vif.csv'}")
