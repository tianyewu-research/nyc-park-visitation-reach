"""
25_single_stage_robustness.py   (robustness: generated-regressand defense)

The main analysis defines mismatch as the residual of a first-stage OLS (log_use ~ log_access)
and then models that residual (RQ2/RQ3). A residual used as a dependent variable is a
"generated regressand": the second-stage SEs ignore first-stage estimation uncertainty
(advisor point 8). This script re-estimates the social gradient (RQ2) and the mechanism
ordering (RQ3) in a SINGLE stage — log_access enters as a covariate directly in a spatial
error model of log_use_c — so no residual is generated. If the income gradient and RQ3
ordering match the two-stage results, the two-stage design is vindicated.

Model A (RQ2, single stage):
  log_use_c ~ log_access + income + nonwhite + popdens + park_area + borough  [ML error]
Model B (RQ3, single stage, standardized):
  log_use_c ~ log_access + quality + walkability + out_nbhd + transit + park_area
              + popdens + income + nonwhite + borough  [ML error]

Input:  data/processed/analysis_cbg_final.gpkg
Output: outputs/tables/single_stage_robustness.csv
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import scipy.stats as st
from libpysal.weights import Queen
import spreg
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

cbg = gpd.read_file(PROCESSED / "analysis_cbg_final.gpkg")
cbg["income_10k"] = cbg["median_hh_income"] / 10000.0
cbg["popdens_k"] = cbg["pop_density"] / 1000.0

nice = {
    "log_access": "Log park access", "quality_index_cbg_svc": "Park quality",
    "walkability_c": "Walkability", "out_park_ratio": "Out-of-nbhd mobility",
    "transit_freq": "Transit access", "park_area_acres": "Park area",
    "popdens_k": "Pop. density", "income_10k": "Income", "pct_nonwhite": "% non-white",
}


def fit_error(y, Xdf):
    d0 = pd.concat([y, Xdf], axis=1).dropna().reset_index(drop=True)
    # rebuild geometry-aligned weights on the retained sample via index into cbg
    return d0


def run(depvar, predictors, standardize, label):
    allv = [depvar] + predictors + ["borough", "geometry"]
    d = cbg.dropna(subset=[depvar] + predictors + ["borough"]).reset_index(drop=True).copy()
    w = Queen.from_dataframe(d, use_index=False)
    if w.islands:
        keep = [i for i in range(len(d)) if i not in set(w.islands)]
        d = d.iloc[keep].reset_index(drop=True)
        w = Queen.from_dataframe(d, use_index=False)
    w.transform = "r"

    cols = []
    for c in predictors:
        if standardize:
            d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
            cols.append(c + "_z")
        else:
            cols.append(c)
    boro = pd.get_dummies(d["borough"], prefix="boro", drop_first=True).astype(float)
    X_df = pd.concat([d[cols], boro], axis=1)
    X = X_df.values
    y = d[depvar].values.reshape(-1, 1)
    names = list(X_df.columns)

    sp = spreg.ML_Error(y, X, w=w, name_y=depvar, name_x=names)
    b = sp.betas.flatten()
    se = np.sqrt(np.diag(sp.vm))
    full = ["CONSTANT"] + names + ["lambda"]
    rows = []
    for i, nm in enumerate(full):
        if nm == "lambda":
            p = np.nan
        else:
            p = 2 * (1 - st.norm.cdf(abs(b[i] / se[i])))
        base = nm.replace("_z", "")
        rows.append({"model": label, "term": nice.get(base, nm),
                     "coef": round(b[i], 4), "p": round(p, 4) if not np.isnan(p) else np.nan,
                     "n": w.n, "pseudo_r2": round(sp.pr2, 3)})
    return pd.DataFrame(rows)


# Model A: RQ2 single-stage (raw units, interpretable)
A = run("log_use_c", ["log_access", "income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"],
        standardize=False, label="A_RQ2_single_stage")
# Model B: RQ3 single-stage (standardized, for ordering)
B = run("log_use_c", ["log_access", "quality_index_cbg_svc", "walkability_c", "out_park_ratio",
                      "transit_freq", "park_area_acres", "popdens_k", "income_10k", "pct_nonwhite"],
        standardize=True, label="B_RQ3_single_stage")

out = pd.concat([A, B], ignore_index=True)
out.to_csv(TABLES / "single_stage_robustness.csv", index=False)

print("=== Model A (RQ2 single-stage): social gradient net of access ===")
for _, r in A[A["term"].isin(["Income", "% non-white", "Log park access"])].iterrows():
    sig = "***" if r.p < .001 else "**" if r.p < .01 else "*" if r.p < .05 else "ns"
    print(f"  {r['term']:16s} coef={r['coef']:+.4f} p={r.p:.3f} {sig}")
print("\n=== Model B (RQ3 single-stage): mechanism ordering (standardized) ===")
mechlab = ["Out-of-nbhd mobility", "Park area", "Park quality", "Walkability", "Transit access", "Pop. density"]
bb = B[B["term"].isin(mechlab)].copy()
bb["abs"] = bb["coef"].abs()
for _, r in bb.sort_values("abs", ascending=False).iterrows():
    sig = "***" if r.p < .001 else "**" if r.p < .01 else "*" if r.p < .05 else "ns"
    print(f"  {r['term']:22s} coef={r['coef']:+.4f} p={r.p:.3f} {sig}")

# --- self-check ---
_t = pd.read_csv(TABLES / "single_stage_robustness.csv")
assert len(_t) > 0 and {"A_RQ2_single_stage", "B_RQ3_single_stage"}.issubset(set(_t["model"]))
print("\n[self-check] 25 single-stage robustness OK")
