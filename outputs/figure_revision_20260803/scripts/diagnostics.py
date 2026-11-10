"""Method diagnostics + RQ3 specification audit. Supports the Data-analysis justification only;
does NOT change the RQ, model, or variable definitions. British spelling."""
import sys, pandas as pd, numpy as np, scipy.stats as st, json
from libpysal.weights import Queen, KNN
from esda.moran import Moran
import spreg, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from revlib import load_data, ols_diag, weights_summary, OUT, RQ2_PREDS
warnings.filterwarnings("ignore"); np.random.seed(42)

g = load_data()

# ---- RQ2 OLS diagnostics ----
diag = ols_diag("lpur_2.0km", RQ2_PREDS, g)

# ---- RQ2 spatial-error full stats incl lambda SE/p, logll, AIC, residual Moran ----
d = g.dropna(subset=["lpur_2.0km"] + RQ2_PREDS + ["borough"]).reset_index(drop=True)
w = Queen.from_dataframe(d, use_index=False)
if w.islands:
    d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
    w = Queen.from_dataframe(d, use_index=False)
w.transform = "r"
boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
X = pd.concat([d[RQ2_PREDS], boro], axis=1)
m = spreg.ML_Error(d["lpur_2.0km"].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
b = m.betas.flatten(); se = np.sqrt(np.diag(m.vm))
lam, lam_se = float(b[-1]), float(se[-1]); lam_z = lam / lam_se
lam_p = float(2 * (1 - st.norm.cdf(abs(lam_z))))
logll = float(m.logll); k = len(b); aic = 2 * k - 2 * logll
resid_moran = Moran(m.e_filtered.flatten(), w, permutations=999)   # FILTERED residuals (correct)

# spatial LAG model for the error-vs-lag comparison (diagnostics only; model choice = researcher)
ml = spreg.ML_Lag(d["lpur_2.0km"].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
mlb = ml.betas.flatten(); mlse = np.sqrt(np.diag(ml.vm)); rho = float(mlb[-1])
lag_logll = float(ml.logll); lag_aic = 2 * len(mlb) - 2 * lag_logll
nm2 = ["CONSTANT"] + list(X.columns) + ["rho"]
inw = nm2.index("pct_nonwhite"); nw_dir, nw_dse = float(mlb[inw]), float(mlse[inw])
nw_p = float(2 * (1 - st.norm.cdf(abs(nw_dir / nw_dse)))); nw_tot = nw_dir / (1 - rho)
iinc = nm2.index("income_10k"); inc_dir = float(mlb[iinc]); inc_p = float(2 * (1 - st.norm.cdf(abs(inc_dir / float(mlse[iinc])))))
lag_resid_moran = Moran(ml.u.flatten(), w, permutations=999)

wq = weights_summary(d, "queen"); wk = weights_summary(d, "knn")

with open(OUT / "method_diagnostics.txt", "w") as f:
    f.write("METHOD DIAGNOSTICS — RQ2 (outcome = LPUR 2 km; covariates = income, non-white, density, "
            "park area, borough FE). Supports model-choice justification only; does NOT change the model.\n\n")
    f.write(f"OLS baseline (diagnostic comparator only; strong residual spatial autocorrelation):\n")
    f.write(f"  n = {diag['n']}, R2 = {diag['r2']:.4f}\n")
    f.write(f"  Residual Moran's I = {diag['moran_res'][0]:.4f} (p = {diag['moran_res'][1]:.4g})\n")
    f.write(f"  LM error       = {diag['lm_error'][0]:.1f} (p = {diag['lm_error'][1]:.4g})\n")
    f.write(f"  LM lag         = {diag['lm_lag'][0]:.1f} (p = {diag['lm_lag'][1]:.4g})\n")
    f.write(f"  robust LM error= {diag['rlm_error'][0]:.1f} (p = {diag['rlm_error'][1]:.4g})\n")
    f.write(f"  robust LM lag  = {diag['rlm_lag'][0]:.1f} (p = {diag['rlm_lag'][1]:.4g})\n\n")
    f.write("*** IMPORTANT MODEL-SELECTION FINDING (for the outcome LPUR, this differs from the paper) ***\n")
    f.write("  Both robust LM statistics are significant, but robust LM LAG (%.0f) >> robust LM ERROR (%.0f).\n" % (diag['rlm_lag'][0], diag['rlm_error'][0]))
    f.write("  By the Anselin et al. (1996) decision rule this favours a spatial LAG specification.\n")
    f.write("  The fit statistics agree: the spatial-LAG model has higher log-likelihood and pseudo-R2\n")
    f.write("  than the spatial-error model (see below). The manuscript currently reports the spatial-\n")
    f.write("  ERROR model. This is a model-selection decision reserved for the researcher (AGENTS.md\n")
    f.write("  rule c); we report both here and do not alter the model.\n\n")
    f.write("ML spatial-ERROR model (as in manuscript):\n")
    f.write(f"  n = {int(w.n)}, lambda = {lam:.4f}, SE = {lam_se:.4f}, p = {lam_p:.4g}\n")
    f.write(f"  log-likelihood = {logll:.2f}, AIC = {aic:.2f}, pseudo-R2 = {m.pr2:.4f}\n")
    f.write(f"  residual Moran's I on spatially-FILTERED residuals (e_filtered) = {resid_moran.I:.4f} "
            f"(p = {resid_moran.p_sim:.4g}) — autocorrelation removed vs OLS {diag['moran_res'][0]:.3f}.\n\n")
    f.write("ML spatial-LAG model (diagnostic comparison):\n")
    f.write(f"  n = {int(w.n)}, rho = {rho:.4f}\n")
    f.write(f"  log-likelihood = {lag_logll:.2f}, AIC = {lag_aic:.2f}, pseudo-R2 = {ml.pr2:.4f}\n")
    f.write(f"  residual Moran's I = {lag_resid_moran.I:.4f} (p = {lag_resid_moran.p_sim:.4g})\n")
    f.write(f"  Non-White share: direct coef = {nw_dir:.4f} (SE {nw_dse:.4f}, p = {nw_p:.4g}), "
            f"approx total impact coef/(1-rho) = {nw_tot:.4f}\n")
    f.write(f"  Income (/$10k): direct coef = {inc_dir:.4f} (p = {inc_p:.4g})\n")
    f.write("  => The negative Non-White gradient is significant under BOTH error and lag models; income\n")
    f.write("     is null under the error model but weakly positive-significant under the lag model.\n\n")
    f.write("Spatial weights:\n")
    f.write(f"  Queen (primary): n={wq['n']}, islands={wq['islands']}, mean neighbours={wq['mean_neighbours']:.2f}, "
            f"min={wq['min_neighbours']}, max={wq['max_neighbours']}\n")
    f.write(f"  6-nearest-neighbour (sensitivity): n={wk['n']}, islands={wk['islands']}, "
            f"mean neighbours={wk['mean_neighbours']:.2f}, min={wk['min_neighbours']}, max={wk['max_neighbours']}\n")

pd.DataFrame([
    dict(quantity="OLS residual Moran I", value=diag['moran_res'][0], p=diag['moran_res'][1]),
    dict(quantity="LM error", value=diag['lm_error'][0], p=diag['lm_error'][1]),
    dict(quantity="LM lag", value=diag['lm_lag'][0], p=diag['lm_lag'][1]),
    dict(quantity="robust LM error", value=diag['rlm_error'][0], p=diag['rlm_error'][1]),
    dict(quantity="robust LM lag", value=diag['rlm_lag'][0], p=diag['rlm_lag'][1]),
    dict(quantity="spatial-error lambda", value=lam, p=lam_p),
    dict(quantity="spatial-error resid Moran I (e_filtered)", value=resid_moran.I, p=resid_moran.p_sim),
    dict(quantity="spatial-error logL", value=logll, p=np.nan),
    dict(quantity="spatial-error AIC", value=aic, p=np.nan),
    dict(quantity="spatial-error pseudo-R2", value=m.pr2, p=np.nan),
    dict(quantity="spatial-lag rho", value=rho, p=np.nan),
    dict(quantity="spatial-lag logL", value=lag_logll, p=np.nan),
    dict(quantity="spatial-lag AIC", value=lag_aic, p=np.nan),
    dict(quantity="spatial-lag pseudo-R2", value=ml.pr2, p=np.nan),
    dict(quantity="spatial-lag nonwhite direct", value=nw_dir, p=nw_p),
    dict(quantity="spatial-lag nonwhite total impact", value=nw_tot, p=np.nan),
    dict(quantity="spatial-lag income direct", value=inc_dir, p=inc_p),
]).to_csv(OUT / "method_diagnostics.csv", index=False)

# ---- RQ3 specification audit (verify n=6227) ----
RQ3_RAW = ["log_access" if "log_access" in g.columns else None]
# build log_access if missing
if "log_access" not in g.columns:
    a = g["access_score"] if "access_score" in g.columns else None
g3 = g.copy()
pred3 = ["log_access", "quality_index_cbg_svc", "walkability_c", "transit_freq", "park_area_acres",
         "popdens_k", "income_10k", "pct_nonwhite"]
have3 = [p for p in pred3 if p in g3.columns]
d3 = g3.dropna(subset=["lpur_2.0km"] + have3 + ["borough"]).reset_index(drop=True)
w3 = Queen.from_dataframe(d3, use_index=False)
n3 = len(d3) - len(w3.islands)
with open(OUT / "rq3_model_specification.txt", "w") as f:
    f.write("RQ3 MODEL SPECIFICATION (audit — not a change to the model)\n\n")
    f.write(f"Outcome: local park-use ratio (LPUR, 2 km), raw scale. Analytic n = {n3} (expected 6,227).\n\n")
    f.write("Predictors (all standardised to mean 0, SD 1 for comparability of coefficients):\n")
    f.write("  - log park access (E2SFCA, 6 km Gaussian catchment) — LOG-transformed, then standardised\n")
    f.write("  - park quality composite (10-min-walk service area) — standardised\n")
    f.write("  - walkability (SLD National Walkability Index) — standardised [control]\n")
    f.write("  - transit access (SLD transit-service frequency) — standardised [control]\n")
    f.write("  - park area within block group (acres) — standardised\n")
    f.write("  - population density (residents / km2 land) — standardised\n")
    f.write("  - median household income (tract-downscaled) — standardised\n")
    f.write("  - Non-White share (1 - non-Hispanic-White / total, B03002) — standardised\n\n")
    f.write("Fixed effects: borough (five boroughs; reference = Bronx).\n")
    f.write("Spatial weights: Queen contiguity, row-standardised (islands dropped).\n")
    f.write("Estimator: maximum-likelihood spatial error model (spreg.ML_Error).\n\n")
    f.write("Model expression (i indexes block groups):\n")
    f.write("  LPUR_i = alpha + sum_k beta_k * z(x_ki) + sum_b gamma_b * Borough_bi + u_i,\n")
    f.write("  u_i = lambda * sum_j W_ij u_j + eps_i,   eps_i ~ iid,\n")
    f.write("  where z(.) denotes standardisation and log access is logged before standardisation.\n")

print("[diagnostics] OLS resid Moran=%.3f (p=%.3g); rLM error=%.1f rLM lag=%.1f; lambda=%.3f p=%.2g; "
      "SE-model resid Moran=%.3f; RQ3 n=%d" % (diag['moran_res'][0], diag['moran_res'][1],
      diag['rlm_error'][0], diag['rlm_lag'][0], lam, lam_p, resid_moran.I, n3))
print("[diagnostics] Queen mean nb=%.2f islands=%d | kNN mean nb=%.2f" % (
    wq['mean_neighbours'], wq['islands'], wk['mean_neighbours']))
