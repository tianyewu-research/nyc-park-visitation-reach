"""
19_compute_mismatch.py
First-stage OLS: log_use_c ~ log_access → residual = mismatch_c.
Also creates 2×2 quadrant classification for visualization.

Input:  data/processed/analysis_cbg.gpkg
Output: data/processed/analysis_cbg_final.gpkg
        outputs/tables/first_stage_ols.txt
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

cbg = gpd.read_file(PROCESSED / "analysis_cbg.gpkg")
print(f"Total CBGs: {len(cbg)}")

# First stage: log(use) ~ log(2SFCA access). Both logged (see 18_). Residual = mismatch.
sample = cbg.dropna(subset=["log_access", "log_use_c"]).copy()
print(f"Analysis sample (access + use non-missing): {len(sample)}")

X = sm.add_constant(sample["log_access"])
y = sample["log_use_c"]

model = sm.OLS(y, X).fit()

sample["mismatch_c"] = model.resid

access_med = sample["log_access"].median()
use_med = sample["log_use_c"].median()

def classify_quad(row):
    ha = row["log_access"] >= access_med
    hu = row["log_use_c"] >= use_med
    if ha and hu:
        return "HH"
    elif ha and not hu:
        return "HL"
    elif not ha and hu:
        return "LH"
    else:
        return "LL"

sample["mismatch_quad"] = sample.apply(classify_quad, axis=1)

cbg = cbg.merge(
    sample[["GEOID", "mismatch_c", "mismatch_quad"]],
    on="GEOID",
    how="left",
)

out_path = PROCESSED / "analysis_cbg_final.gpkg"
cbg.to_file(out_path, driver="GPKG")

ols_report = model.summary().as_text()
with open(TABLES / "first_stage_ols.txt", "w") as f:
    f.write("First-stage OLS: log_use_c ~ log_access\n")
    f.write(f"N = {len(sample)}\n")
    f.write(f"Access median: {access_med:.4f}\n")
    f.write(f"Use median: {use_med:.4f}\n\n")
    f.write(ols_report)
    f.write(f"\n\nMismatch (residual) summary:\n")
    f.write(sample["mismatch_c"].describe().to_string())
    f.write(f"\n\nQuadrant distribution:\n")
    f.write(sample["mismatch_quad"].value_counts().to_string())

print(f"\nFirst-stage OLS:")
print(f"  R² = {model.rsquared:.4f}")
print(f"  β(access) = {model.params['log_access']:.4f} (p={model.pvalues['log_access']:.4e})")
print(f"  Access median: {access_med:.4f}")
print(f"  Use median: {use_med:.4f}")
print(f"\nMismatch (residual):")
print(f"  mean={sample['mismatch_c'].mean():.4f}, std={sample['mismatch_c'].std():.2f}")
print(f"  underused (ε<0): {(sample['mismatch_c']<0).sum()} CBGs")
print(f"  overused  (ε>0): {(sample['mismatch_c']>0).sum()} CBGs")
print(f"\nQuadrant:")
print(sample["mismatch_quad"].value_counts().to_string())
print(f"\nSaved: {out_path}")

# --- self-check ---
_c = gpd.read_file(PROCESSED / "analysis_cbg_final.gpkg")
_m = _c["mismatch_c"].dropna()
assert abs(_m.mean()) < 0.01 and (_m < 0).any() and (_m > 0).any()
assert _c["mismatch_quad"].dropna().nunique() == 4
print("[self-check] 19 mismatch OK")
