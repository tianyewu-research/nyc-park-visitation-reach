"""Part 3 — RQ3 under partial SDM (n≈6,227). SEM/SAR/SDM/SDEM diagnostics + impacts; standardised
predictors, log regional access, borough FE (not lagged). Machine-readable equation inputs file.
Do not silently pick RQ3's family if it conflicts with RQ2 SDM — flag."""
import sys, numpy as np, pandas as pd, scipy.stats as st
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
sys.path.insert(0, "outputs/spatial_model_reassessment_20260803/scripts")
from slib import ROOT
from slib2 import trace_powers, sdm_impacts, innovation_moran, OUT
import geopandas as gpd, spreg
from libpysal.weights import Queen
from esda.moran import Moran
np.random.seed(42)

g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg").to_crs("EPSG:2263")
g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0
# log regional access
acc = g["access_score"]; pm = acc[acc > 0].min()
g["log_access"] = np.log(acc.clip(lower=pm / 2))
RAW = ["log_access", "quality_index_cbg_svc", "walkability_c", "transit_freq", "park_area_acres",
       "popdens_k", "income_10k", "pct_nonwhite"]
DEP = "lpur_2.0km"
d = g.dropna(subset=[DEP] + RAW + ["borough"]).reset_index(drop=True)
w = Queen.from_dataframe(d, use_index=False)
if w.islands:
    d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
    w = Queen.from_dataframe(d, use_index=False)
w.transform = "r"
n = len(d); y = d[DEP].values.reshape(-1, 1)
# standardise continuous predictors (z)
Z = {}
for c in RAW:
    Z[c] = (d[c] - d[c].mean()) / d[c].std()
boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
Zdf = pd.DataFrame(Z)
own = pd.concat([Zdf, boro], axis=1)
WX = pd.DataFrame(w.sparse @ Zdf.values, columns=["W_" + c for c in RAW], index=d.index)


def aicbic(m):
    ll = float(m.logll); k = len(m.betas.flatten()); return ll, 2*k-2*ll, k*np.log(n)-2*ll, k


diag = []
ols = spreg.OLS(y, own.values, w=w, spat_diag=True, moran=True, name_x=list(own.columns))
diag.append(dict(model="OLS", n=n, logLik="NA", AIC="NA", BIC="NA",
                 ols_resid_moran=round(ols.moran_res[0], 4), robustLM_error=round(ols.rlm_error[0], 2),
                 robustLM_error_p=ols.rlm_error[1], robustLM_lag=round(ols.rlm_lag[0], 2),
                 robustLM_lag_p=ols.rlm_lag[1]))
sem = spreg.ML_Error(y, own.values, w=w, name_x=list(own.columns))
sar = spreg.ML_Lag(y, own.values, w=w, name_x=list(own.columns))
Xsdm = pd.concat([own, WX], axis=1)
sdm = spreg.ML_Lag(y, Xsdm.values, w=w, name_x=list(Xsdm.columns), spat_impacts="full")
Xsdem = pd.concat([own, WX], axis=1)
sdem = spreg.ML_Error(y, Xsdem.values, w=w, name_x=list(Xsdem.columns))
for nm, m, kind, Xd in [("SEM", sem, "error", own), ("SAR", sar, "lag", own),
                        ("SDM(partial)", sdm, "lag", Xsdm), ("SDEM(partial)", sdem, "error", Xsdem)]:
    ll, aic, bic, k = aicbic(m)
    imi, imp_p = innovation_moran(m, y, w, kind, Xdes=Xd)
    sp = f"rho={m.betas.flatten()[-1]:.4f}" if kind == "lag" else f"lambda={m.betas.flatten()[-1]:.4f}"
    diag.append(dict(model=nm, n=n, logLik=round(ll, 3), AIC=round(aic, 3), BIC=round(bic, 3), spatial=sp,
                     resid_moran_I=round(imi, 4), resid_moran_p=round(imp_p, 4)))
lr = 2 * (float(sdm.logll) - float(sar.logll)); df = len(sdm.betas) - len(sar.betas)
p_lr = float(st.chi2.sf(lr, df))
diag.append(dict(model="LR SDM vs SAR (theta=0)", n=n, logLik=round(lr, 3), AIC=f"df={df}", BIC=f"p={p_lr:.3g}"))
pd.DataFrame(diag).to_csv(OUT / "RQ3_model_diagnostics.csv", index=False)

# impacts (standardised units) for the substantive predictors under SDM
T = trace_powers(w, Q=200, m_probe=4000)
scales = {c: [("per 1 SD (standardised)", 1.0)] for c in RAW}
rows, ret, pmult = sdm_impacts(sdm, Xsdm, w, T, RAW, scales, nsim=5000)
for r in rows:
    r["model"] = "RQ3_SDM"; r["weights"] = "Queen"; r["units"] = "standardised (per 1 SD)"
    r["beta_struct"] = round(pmult[r["variable"]]["beta"], 5); r["theta_struct"] = round(pmult[r["variable"]]["theta"], 5)
pd.DataFrame(rows).to_csv(OUT / "RQ3_final_impacts.csv", index=False)

# equation inputs
b = sdm.betas.flatten(); se = np.sqrt(np.diag(sdm.vm)); cols = ["CONSTANT"] + list(Xsdm.columns) + ["rho"]
lines = ["RQ3 partial-SDM equation inputs (machine-readable)\n",
         f"outcome: {DEP} (raw scale); n={n}; estimator: ML spatial Durbin (spreg.ML_Lag, slx on continuous only)",
         "spatial weights: Queen contiguity, row-standardised (islands dropped)",
         "borough fixed effects: reference = Bronx (drop_first on sorted boroughs); borough NOT spatially lagged",
         "standardisation: each continuous predictor z-scored (mean 0, SD 1) on the RQ3 sample;",
         "  log_access = log(access_score) FIRST, then z-scored.\n",
         "model: LPUR_i = a + sum_k beta_k*z(x_ki) + sum_k theta_k*(W z(x))_ki + sum_b g_b*Borough_bi",
         "       + rho*(W LPUR)_i + eps_i\n",
         "term, coefficient, se"]
for i, nm in enumerate(cols):
    if i < len(b):
        lines.append(f"{nm}, {b[i]:.6f}, {se[i]:.6f}")
means = {c: float(d[c].mean()) for c in RAW}; sds = {c: float(d[c].std()) for c in RAW}
lines.append("\nstandardisation constants (mean, sd) for back-transformation:")
for c in RAW:
    lines.append(f"{c}, mean={means[c]:.6f}, sd={sds[c]:.6f}")
(OUT / "RQ3_equation_inputs.txt").write_text("\n".join(lines) + "\n")

nw = [r for r in rows if r["variable"] == "pct_nonwhite"]
print("=== s3 RQ3 done (n=%d) ===" % n)
print(pd.DataFrame(diag)[["model", "logLik", "AIC", "BIC", "resid_moran_I", "resid_moran_p"]].to_string(index=False))
print(f"LR SDM vs SAR = {lr:.2f} df={df} p={p_lr:.3g} -> {'SDM supported (same family as RQ2)' if p_lr<0.05 else 'theta=0 not rejected — FLAG conflict with RQ2 SDM'}")
print("\nRQ3 Non-White impacts (standardised):")
print(pd.DataFrame(nw)[["effect", "estimate", "ci_lo", "ci_hi", "p"]].to_string(index=False))
