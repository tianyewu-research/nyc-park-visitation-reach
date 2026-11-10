"""
21_rq2_social_pattern.py   (RQ2: is mismatch socially patterned?)
OLS: mismatch_c ~ income + nonwhite + pop_density + park_area + borough FE.
Descriptive association (NOT causal). Residual Moran's I reported as a diagnostic
(spatial dependence is handled in RQ3, script 22).

Input:  data/processed/analysis_cbg_final.gpkg
Output: outputs/tables/rq2_regression.csv
        outputs/figures/rq2_mismatch_social.png
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
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

# ── Build regression sample ──
mvars = ["mismatch_c", "median_hh_income", "pct_nonwhite",
         "pop_density", "park_area_acres", "borough"]
d = cbg.dropna(subset=mvars).reset_index(drop=True).copy()

# Interpretable units
d["income_10k"] = d["median_hh_income"] / 10000.0       # per $10,000
d["popdens_k"] = d["pop_density"] / 1000.0              # per 1,000 residents/km^2
print(f"RQ2 sample: {len(d)} CBGs (dropped {len(cbg)-len(d)} with missing covariates)")

# ── Model 1: natural units + borough FE ──
f = "mismatch_c ~ income_10k + pct_nonwhite + popdens_k + park_area_acres + C(borough)"
m = smf.ols(f, data=d).fit()

# ── Model 2: standardized predictors (compare income vs race effect sizes) ──
dz = d.copy()
for c in ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]:
    dz[c + "_z"] = (dz[c] - dz[c].mean()) / dz[c].std()
fz = "mismatch_c ~ income_10k_z + pct_nonwhite_z + popdens_k_z + park_area_acres_z + C(borough)"
mz = smf.ols(fz, data=dz).fit()
std_beta = {k.replace("_z", ""): v for k, v in mz.params.items() if k.endswith("_z")}

# ── Assemble coefficient table ──
ci = m.conf_int()
rows = []
for term in m.params.index:
    rows.append({
        "term": term,
        "coef": m.params[term],
        "std_err": m.bse[term],
        "t": m.tvalues[term],
        "p_value": m.pvalues[term],
        "ci_low": ci.loc[term, 0],
        "ci_high": ci.loc[term, 1],
        "std_beta": std_beta.get(term, np.nan),
    })
tbl = pd.DataFrame(rows)
tbl.to_csv(TABLES / "rq2_regression.csv", index=False)

# ── Residual spatial autocorrelation diagnostic ──
w = Queen.from_dataframe(d, use_index=False)
if w.islands:
    keep = [i for i in range(len(d)) if i not in set(w.islands)]
    d_w = d.iloc[keep].reset_index(drop=True)
    resid = m.resid.values[keep]
    w = Queen.from_dataframe(d_w, use_index=False)
else:
    resid = m.resid.values
w.transform = "r"
mi_resid = Moran(resid, w, permutations=999)

print(f"\n=== RQ2 OLS (n={int(m.nobs)}, R2={m.rsquared:.3f}, adjR2={m.rsquared_adj:.3f}) ===")
show = tbl[~tbl["term"].str.startswith("C(borough)")]
print(show[["term", "coef", "std_err", "p_value", "std_beta"]].round(4).to_string(index=False))
print(f"(borough fixed effects included, {sum(tbl['term'].str.startswith('C(borough)'))} dummies)")
print(f"\nResidual Moran's I = {mi_resid.I:.3f} (p={mi_resid.p_sim:.3f}) "
      f"-> OLS inference unreliable; fitting spatial models for corrected inference")

# ── Corrected inference: spatial regression (LM diagnostics -> lag vs error) ──
# Align y, X, w on an island-free sample.
d_sp = d.copy()
w_sp = Queen.from_dataframe(d_sp, use_index=False)
if w_sp.islands:
    keep = [i for i in range(len(d_sp)) if i not in set(w_sp.islands)]
    d_sp = d_sp.iloc[keep].reset_index(drop=True)
    w_sp = Queen.from_dataframe(d_sp, use_index=False)
w_sp.transform = "r"

xcols = ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]
boro_d = pd.get_dummies(d_sp["borough"], prefix="boro", drop_first=True).astype(float)
X_df = pd.concat([d_sp[xcols], boro_d], axis=1)
X = X_df.values
y_sp = d_sp["mismatch_c"].values.reshape(-1, 1)
xnames = list(X_df.columns)

ols_sp = spreg.OLS(y_sp, X, w=w_sp, spat_diag=True, moran=True,
                   name_y="mismatch_c", name_x=xnames)
lm_err = ols_sp.lm_error
lm_lag = ols_sp.lm_lag
rlm_err = ols_sp.rlm_error
rlm_lag = ols_sp.rlm_lag
print("\n=== LM spatial diagnostics ===")
print(f"LM-error  = {lm_err[0]:.1f} (p={lm_err[1]:.3f}) | Robust LM-error = {rlm_err[0]:.1f} (p={rlm_err[1]:.3f})")
print(f"LM-lag    = {lm_lag[0]:.1f} (p={lm_lag[1]:.3f}) | Robust LM-lag   = {rlm_lag[0]:.1f} (p={rlm_lag[1]:.3f})")

# choose model by the more significant robust LM
use_error = rlm_err[1] <= rlm_lag[1]
chosen = "Spatial Error (ML)" if use_error else "Spatial Lag (ML)"
sp_model = (spreg.ML_Error(y_sp, X, w=w_sp, name_y="mismatch_c", name_x=xnames) if use_error
            else spreg.ML_Lag(y_sp, X, w=w_sp, name_y="mismatch_c", name_x=xnames))
print(f"-> chosen by robust LM: {chosen}")

# ── OLS vs spatial comparison for the social variables ──
betas = sp_model.betas.flatten()
# spreg orders: [CONSTANT, x1..xk, (rho or lambda)]
sp_names = ["CONSTANT"] + xnames + (["lambda"] if use_error else ["rho"])
sp_se = np.sqrt(np.diag(sp_model.vm)) if hasattr(sp_model, "vm") else [np.nan] * len(betas)
import scipy.stats as st
comp_rows = []
for i, nm in enumerate(sp_names):
    if nm in ("CONSTANT", "rho", "lambda"):
        ols_c = ols_sp.betas.flatten()[0] if nm == "CONSTANT" else np.nan
        ols_p = np.nan
    else:
        j = xnames.index(nm) + 1
        ols_c = ols_sp.betas.flatten()[j]
        ols_p = 2 * (1 - st.norm.cdf(abs(ols_sp.betas.flatten()[j] /
                     np.sqrt(np.diag(ols_sp.vm))[j])))
    z = betas[i] / sp_se[i] if sp_se[i] and not np.isnan(sp_se[i]) else np.nan
    sp_p = 2 * (1 - st.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    comp_rows.append({"term": nm, "ols_coef": ols_c, "ols_p": ols_p,
                      "spatial_coef": betas[i], "spatial_p": sp_p})
comp = pd.DataFrame(comp_rows)
comp.to_csv(TABLES / "rq2_spatial.csv", index=False)

print(f"\n=== Corrected inference: {chosen} (n={len(y_sp)}) ===")
key = comp[comp["term"].isin(["income_10k", "pct_nonwhite", "popdens_k",
                              "park_area_acres", "rho", "lambda"])]
print(key.round(4).to_string(index=False))
print("\nVerdict on social patterning (spatially corrected):")
for v in ["income_10k", "pct_nonwhite"]:
    row = comp[comp["term"] == v].iloc[0]
    sig = "significant" if row["spatial_p"] < 0.05 else "NOT significant"
    print(f"  {v}: spatial coef={row['spatial_coef']:+.4f}, p={row['spatial_p']:.3f} -> {sig}")

# ── Figure: mismatch vs income and vs nonwhite ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
boro_colors = {"Manhattan": "#e41a1c", "Brooklyn": "#377eb8", "Queens": "#4daf4a",
               "Bronx": "#984ea3", "Staten Island": "#ff7f00"}

ax = axes[0]
for b, c in boro_colors.items():
    s = d[d["borough"] == b]
    ax.scatter(s["median_hh_income"] / 1000, s["mismatch_c"], s=4, alpha=0.35, color=c, label=b)
# fitted line on income (holding other vars at mean via simple bivariate fit for display)
xb = np.linspace(d["median_hh_income"].quantile(.02), d["median_hh_income"].quantile(.98), 50)
bi = np.polyfit(d["median_hh_income"], d["mismatch_c"], 1)
ax.plot(xb / 1000, np.polyval(bi, xb), "k-", lw=2)
ax.axhline(0, color="grey", lw=0.6, ls="--")
ax.set_xlabel("Median household income ($1,000s)")
ax.set_ylabel("Mismatch (residual)")
ax.set_title("Mismatch vs income")
ax.legend(fontsize=7, markerscale=2)

ax = axes[1]
for b, c in boro_colors.items():
    s = d[d["borough"] == b]
    ax.scatter(s["pct_nonwhite"] * 100, s["mismatch_c"], s=4, alpha=0.35, color=c)
xn = np.linspace(0, 1, 50)
bn = np.polyfit(d["pct_nonwhite"], d["mismatch_c"], 1)
ax.plot(xn * 100, np.polyval(bn, xn), "k-", lw=2)
ax.axhline(0, color="grey", lw=0.6, ls="--")
ax.set_xlabel("% non-white")
ax.set_ylabel("Mismatch (residual)")
ax.set_title("Mismatch vs % non-white")

plt.tight_layout()
plt.savefig(FIGS / "rq2_mismatch_social.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\nSaved: {TABLES/'rq2_regression.csv'}, {FIGS/'rq2_mismatch_social.png'}")

# --- self-check ---
_t = pd.read_csv(TABLES / "rq2_spatial.csv")
assert len(_t) > 0 and "spatial_coef" in _t.columns
print("[self-check] 21 RQ2 OK")
