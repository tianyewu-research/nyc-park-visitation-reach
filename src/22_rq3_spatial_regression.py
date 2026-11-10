"""
22_rq3_spatial_regression.py   (RQ3: what explains the mismatch?)
Diagnostic flow: OLS baseline -> Moran's I on residuals -> LM tests (lag/error/robust)
-> choose spatial lag or spatial error -> ML estimation.
Predictors are z-standardized so coefficients are comparable as relative contributions.

Main model:
  mismatch_c ~ quality + walkability + out_park_ratio + transit + park_area
               + pop_density + income + nonwhite + borough FE  [+ spatial term]

quality uses quality_index_cbg_svc (per collinearity diagnostic: VIF~1, corr w/ access +0.02).

Input:  data/processed/analysis_cbg_final.gpkg
Output: outputs/tables/rq3_regression.csv
        outputs/tables/rq3_lm_diagnostics.csv
        outputs/figures/rq3_mechanism_coefs.png
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from libpysal.weights import Queen
from esda.moran import Moran
import spreg
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"

cbg = gpd.read_file(PROCESSED / "analysis_cbg_final.gpkg")

# ── Sample + units ──
cbg["income_10k"] = cbg["median_hh_income"] / 10000.0
cbg["popdens_k"] = cbg["pop_density"] / 1000.0

mech = ["quality_index_cbg_svc", "walkability_c", "out_park_ratio", "transit_freq", "park_area_acres"]
ctrl = ["popdens_k", "income_10k", "pct_nonwhite"]
allvars = ["mismatch_c"] + mech + ctrl + ["borough"]
d = cbg.dropna(subset=allvars).reset_index(drop=True).copy()
print(f"RQ3 sample: {len(d)} CBGs")

# Drop islands so y/X/w align for spatial models
w = Queen.from_dataframe(d, use_index=False)
if w.islands:
    keep = [i for i in range(len(d)) if i not in set(w.islands)]
    d = d.iloc[keep].reset_index(drop=True)
    w = Queen.from_dataframe(d, use_index=False)
w.transform = "r"
print(f"Queen weights: n={w.n}, avg neighbors={w.mean_neighbors:.1f}")

# ── Standardize predictors (z-scores) for comparable coefficients ──
zcols = mech + ctrl
nice = {
    "quality_index_cbg_svc": "Park quality", "walkability_c": "Walkability",
    "out_park_ratio": "Out-of-nbhd mobility", "transit_freq": "Transit access",
    "park_area_acres": "Park area", "popdens_k": "Pop. density",
    "income_10k": "Income", "pct_nonwhite": "% non-white",
}
for c in zcols:
    d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()

boro_d = pd.get_dummies(d["borough"], prefix="boro", drop_first=True).astype(float)
X_df = pd.concat([d[[c + "_z" for c in zcols]], boro_d], axis=1)
X = X_df.values
y = d["mismatch_c"].values.reshape(-1, 1)
xnames = list(X_df.columns)

# ── Step 1-3: OLS baseline + spatial diagnostics ──
ols = spreg.OLS(y, X, w=w, spat_diag=True, moran=True, name_y="mismatch_c", name_x=xnames)
lm = {
    "Moran_resid": (ols.moran_res[0], ols.moran_res[2]),
    "LM_error": ols.lm_error, "RLM_error": ols.rlm_error,
    "LM_lag": ols.lm_lag, "RLM_lag": ols.rlm_lag,
}
pd.DataFrame([{"test": k, "stat": v[0], "p_value": v[1]} for k, v in lm.items()]).to_csv(
    TABLES / "rq3_lm_diagnostics.csv", index=False)
print("\n=== RQ3 spatial diagnostics ===")
print(f"Moran's I (resid) = {ols.moran_res[0]:.3f} (p={ols.moran_res[2]:.3f})")
print(f"Robust LM-error = {ols.rlm_error[0]:.1f} (p={ols.rlm_error[1]:.3f}) | "
      f"Robust LM-lag = {ols.rlm_lag[0]:.1f} (p={ols.rlm_lag[1]:.3f})")

# ── Step 4: choose + fit spatial model ──
use_error = ols.rlm_error[1] <= ols.rlm_lag[1]
chosen = "Spatial Error (ML)" if use_error else "Spatial Lag (ML)"
sp = (spreg.ML_Error(y, X, w=w, name_y="mismatch_c", name_x=xnames) if use_error
      else spreg.ML_Lag(y, X, w=w, name_y="mismatch_c", name_x=xnames))
print(f"-> chosen: {chosen}  (pseudo R2={sp.pr2:.3f})")

# ── Comparison table OLS vs spatial ──
sp_names = ["CONSTANT"] + xnames + (["lambda"] if use_error else ["rho"])
sp_b = sp.betas.flatten()
sp_se = np.sqrt(np.diag(sp.vm))
ols_b = ols.betas.flatten()
ols_se = np.sqrt(np.diag(ols.vm))

rows = []
for i, nm in enumerate(sp_names):
    if nm in ("rho", "lambda"):
        oc = op = np.nan
    else:
        oc = ols_b[i]
        op = 2 * (1 - st.norm.cdf(abs(ols_b[i] / ols_se[i])))
    z = sp_b[i] / sp_se[i]
    sp_p = 2 * (1 - st.norm.cdf(abs(z)))
    rows.append({"term": nm, "ols_coef": oc, "ols_p": op,
                 "spatial_coef": sp_b[i], "spatial_p": sp_p})
comp = pd.DataFrame(rows)
comp.to_csv(TABLES / "rq3_regression.csv", index=False)

# ── Report: hypothesized correlates ranked by |spatial coef| ──
print(f"\n=== RQ3 {chosen}: hypothesized correlates (standardized, ranked) ===")
mech_z = [m + "_z" for m in mech]
mtab = comp[comp["term"].isin(mech_z)].copy()
mtab["label"] = mtab["term"].str.replace("_z", "", regex=False).map(nice)
mtab["abs"] = mtab["spatial_coef"].abs()
mtab = mtab.sort_values("abs", ascending=False)
for _, r in mtab.iterrows():
    sig = "***" if r["spatial_p"] < 0.001 else "**" if r["spatial_p"] < 0.01 else "*" if r["spatial_p"] < 0.05 else "ns"
    print(f"  {r['label']:22s} beta={r['spatial_coef']:+.4f}  p={r['spatial_p']:.3f} {sig}")
sp_param = comp[comp["term"].isin(["rho", "lambda"])].iloc[0]
print(f"  [{sp_param['term']}] spatial param = {sp_param['spatial_coef']:+.3f} (p={sp_param['spatial_p']:.3f})")

# ── Figure: correlate coefficient plot (spatial model, with controls shown lighter) ──
plot_terms = mech_z + [c + "_z" for c in ctrl]
pf = comp[comp["term"].isin(plot_terms)].copy()
pf["label"] = pf["term"].str.replace("_z", "", regex=False).map(nice)
pf["is_mech"] = pf["term"].isin(mech_z)
pf = pf.sort_values("spatial_coef")
zc = 1.96
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#1f77b4" if m else "#bbbbbb" for m in pf["is_mech"]]
se = pf["spatial_coef"].abs() / pf["spatial_coef"].abs()  # placeholder, recompute CI properly
# proper CI from se
se_map = {nm: np.sqrt(np.diag(sp.vm))[i] for i, nm in enumerate(sp_names)}
errs = [zc * se_map[t] for t in pf["term"]]
ax.errorbar(pf["spatial_coef"], range(len(pf)), xerr=errs, fmt="o", color="black", ecolor="grey", capsize=3)
for i, (_, r) in enumerate(pf.iterrows()):
    ax.plot(r["spatial_coef"], i, "o", color="#1f77b4" if r["is_mech"] else "#bbbbbb", ms=8)
ax.axvline(0, color="red", lw=0.8, ls="--")
ax.set_yticks(range(len(pf)))
ax.set_yticklabels(pf["label"])
ax.set_xlabel("Standardized coefficient (spatial error model)")
ax.set_title(f"RQ3: correlates of access\u2013use mismatch\n(blue = hypothesized correlate, grey = control; {chosen}, n={w.n})")
plt.tight_layout()
plt.savefig(FIGS / "rq3_mechanism_coefs.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\nSaved: {TABLES/'rq3_regression.csv'}, {TABLES/'rq3_lm_diagnostics.csv'}, {FIGS/'rq3_mechanism_coefs.png'}")

# --- self-check ---
_t = pd.read_csv(TABLES / "rq3_regression.csv")
_l = pd.read_csv(TABLES / "rq3_lm_diagnostics.csv")
assert len(_t) > 0 and len(_l) > 0
print("[self-check] 22 RQ3 OK")
