"""Part 2 — final RQ2 under partial SDM: aggregate model, joint racial-composition model, and a
companion mean-distance model selection (not assumed). Impacts are direct/indirect/total (asymmetric-
valid multipliers, simulation SE/CI/p). Ecological association; no causal-spillover language."""
import sys, numpy as np, pandas as pd, scipy.stats as st
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
sys.path.insert(0, "outputs/spatial_model_reassessment_20260803/scripts")
from slib import build_sample, make_W, fit_ols, fit_ml, CONT, DEP, ROOT
from slib2 import (fit_sdm, design_sdm, trace_powers, sdm_impacts, innovation_moran, stat, OUT, boro_dummies)
import spreg, geopandas as gpd
from esda.moran import Moran

d, gid, checksum = build_sample()
n = len(d); y = d[DEP].values.reshape(-1, 1)
np.random.seed(42)
wq = make_W(d, "queen"); wk = make_W(d, "knn")
Tq = trace_powers(wq, Q=200, m_probe=4000)
Tk = trace_powers(wk, Q=200, m_probe=4000)


def iqr(col, data=d):
    return float(data[col].quantile(.75) - data[col].quantile(.25))

# ---------------- RQ2 aggregate SDM ----------------
struct_rows = []; agg_rows = []
for kind, w, T in [("Queen(primary)", wq, Tq), ("kNN6(sensitivity)", wk, Tk)]:
    m, X = fit_sdm(d, w)
    b = m.betas.flatten(); se = np.sqrt(np.diag(m.vm)); cols = ["CONSTANT"] + list(X.columns) + ["rho"]
    for i, nm in enumerate(cols):
        s = stat(b, se, i) if i < len(b) else None
        if s:
            struct_rows.append(dict(model="RQ2_aggregate_SDM", weights=kind, term=nm,
                                    coef=round(s["coef"], 5), se=round(s["se"], 5),
                                    ci_lo=round(s["ci_lo"], 5), ci_hi=round(s["ci_hi"], 5), p=s["p"]))
    scales = {"income_10k": [("per $10,000", 1.0), (f"IQR (${iqr('income_10k')*10:.0f}k)", iqr("income_10k"))],
              "pct_nonwhite": [("0->1 share", 1.0), ("+0.10 share", 0.10), (f"IQR ({iqr('pct_nonwhite'):.3f})", iqr("pct_nonwhite"))],
              "popdens_k": [("per 1,000 residents/km2", 1.0)],
              "park_area_acres": [("per acre", 1.0)]}
    rows, ret, pmult = sdm_impacts(m, X, w, T, CONT, scales, nsim=5000)
    rho_s = stat(b, se, len(b) - 1)
    for r in rows:
        r = dict(r); r["weights"] = kind; r["model"] = "RQ2_aggregate_SDM"
        r["rho"] = round(rho_s["coef"], 4); r["rho_ci"] = f"[{rho_s['ci_lo']:.4f},{rho_s['ci_hi']:.4f}]"
        r["beta_struct"] = round(pmult[r["variable"]]["beta"], 5)
        r["theta_struct"] = round(pmult[r["variable"]]["theta"], 5)
        agg_rows.append(r)
    imi, imp_p = innovation_moran(m, y, w, "lag", Xdes=X)
    agg_rows.append(dict(model="RQ2_aggregate_SDM", weights=kind, variable="(residual)", effect="innovation Moran I",
                         scale="reduced-not-eliminated" if imp_p < 0.05 else "n.s.",
                         estimate=round(imi, 4), p=f"{imp_p:.4f}", nsim_retained=ret,
                         rho=round(rho_s["coef"], 4)))
pd.DataFrame(agg_rows).to_csv(OUT / "RQ2_aggregate_SDM_impacts.csv", index=False)
pd.DataFrame(struct_rows).to_csv(OUT / "RQ2_aggregate_SDM_structural_coefs.csv", index=False)

# ---------------- RQ2 joint racial-composition SDM ----------------
RACE = ["pct_black", "pct_hispanic", "pct_asian"]
CONT_J = ["income_10k"] + RACE + ["popdens_k", "park_area_acres"]
joint_rows = []
for kind, w, T in [("Queen(primary)", wq, Tq), ("kNN6(sensitivity)", wk, Tk)]:
    X = design_sdm(d, w, cont=CONT_J)
    m = spreg.ML_Lag(y, X.values, w=w, name_x=list(X.columns), spat_impacts="full")
    scales = {v: [(f"IQR ({iqr(v):.3f})", iqr(v))] for v in RACE}
    scales.update({v: [("per unit", 1.0)] for v in ["income_10k", "popdens_k", "park_area_acres"]})
    rows, ret, pmult = sdm_impacts(m, X, w, T, CONT_J, scales, nsim=5000)
    rho = float(m.betas.flatten()[-1])
    for r in rows:
        if r["variable"] in RACE:
            r = dict(r); r["weights"] = kind; r["model"] = "RQ2_joint_race_SDM"; r["rho"] = round(rho, 4)
            r["beta_struct"] = round(pmult[r["variable"]]["beta"], 5)
            r["theta_struct"] = round(pmult[r["variable"]]["theta"], 5)
            joint_rows.append(r)
pd.DataFrame(joint_rows).to_csv(OUT / "RQ2_joint_race_SDM_impacts.csv", index=False)

# ---------------- Companion mean-distance: model selection (do not assume) ----------------
g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg").to_crs("EPSG:2263")
g["income_10k"] = g["median_hh_income"] / 10000.0; g["popdens_k"] = g["pop_density"] / 1000.0
DDEP = "mean_trip_km"
dd = g.dropna(subset=[DDEP] + CONT + ["borough"]).reset_index(drop=True)
np.random.seed(42)
from libpysal.weights import Queen
wdd = Queen.from_dataframe(dd, use_index=False)
if wdd.islands:
    dd = dd.iloc[[i for i in range(len(dd)) if i not in set(wdd.islands)]].reset_index(drop=True)
    wdd = Queen.from_dataframe(dd, use_index=False)
wdd.transform = "r"
nd = len(dd); yd = dd[DDEP].values.reshape(-1, 1)


def aicbic(m):
    ll = float(m.logll); k = len(m.betas.flatten()); return ll, 2*k-2*ll, k*np.log(nd)-2*ll, k


# fit on distance sample
def fit_on(kind, lag_cont=False):
    boro = pd.get_dummies(dd["borough"], prefix="b", drop_first=True).astype(float)
    base = pd.concat([dd[CONT], boro], axis=1)
    if lag_cont:
        WX = pd.DataFrame(wdd.sparse @ dd[CONT].values, columns=["W_"+c for c in CONT], index=dd.index)
        Xd = pd.concat([base, WX], axis=1)
    else:
        Xd = base
    if kind == "ols":
        return spreg.OLS(yd, Xd.values, w=wdd, spat_diag=True, moran=True, name_x=list(Xd.columns)), Xd
    if kind == "error":
        return spreg.ML_Error(yd, Xd.values, w=wdd, name_x=list(Xd.columns)), Xd
    return spreg.ML_Lag(yd, Xd.values, w=wdd, name_x=list(Xd.columns),
                        spat_impacts="full" if lag_cont else None), Xd


dist_diag = []
ols_d, _ = fit_on("ols")
dist_diag.append(dict(model="OLS", n=nd, logLik="NA", AIC="NA", BIC="NA",
                      ols_resid_moran=round(ols_d.moran_res[0], 4), ols_resid_moran_p=round(ols_d.moran_res[2], 4),
                      robustLM_error=round(ols_d.rlm_error[0], 2), robustLM_error_p=ols_d.rlm_error[1],
                      robustLM_lag=round(ols_d.rlm_lag[0], 2), robustLM_lag_p=ols_d.rlm_lag[1]))
mse, _ = fit_on("error"); msar, _ = fit_on("lag"); msdm, Xdd = fit_on("lag", lag_cont=True); msdem, _ = fit_on("error", lag_cont=True)
for nm, m in [("SEM", mse), ("SAR", msar), ("SDM(partial)", msdm), ("SDEM(partial)", msdem)]:
    ll, aic, bic, k = aicbic(m)
    sp = f"rho={m.betas.flatten()[-1]:.4f}" if "SD M" in nm or nm in ("SAR", "SDM(partial)") else f"lambda={m.betas.flatten()[-1]:.4f}"
    dist_diag.append(dict(model=nm, n=nd, logLik=round(ll, 3), AIC=round(aic, 3), BIC=round(bic, 3), spatial=sp))
lr_ds = 2 * (float(msdm.logll) - float(msar.logll)); df_ds = len(msdm.betas) - len(msar.betas)
p_ds = float(st.chi2.sf(lr_ds, df_ds))
dist_diag.append(dict(model="LR SDM vs SAR (theta=0)", n=nd, logLik=round(lr_ds, 3), AIC=f"df={df_ds}", BIC=f"p={p_ds:.3g}"))
pd.DataFrame(dist_diag).to_csv(OUT / "distance_model_diagnostics.csv", index=False)

# distance impacts under SDM (in km) — supported if LR rejects theta=0 and SDM best AIC
Tdd = trace_powers(wdd, Q=200, m_probe=4000)
scales = {"pct_nonwhite": [("0->1 share", 1.0), ("+0.10 share", 0.10), (f"IQR ({iqr('pct_nonwhite', dd):.3f})", iqr("pct_nonwhite", dd))],
          "income_10k": [("per $10,000", 1.0)], "popdens_k": [("per 1,000/km2", 1.0)], "park_area_acres": [("per acre", 1.0)]}
rows, ret, pmult = sdm_impacts(msdm, Xdd, wdd, Tdd, CONT, scales, nsim=5000)
for r in rows:
    r["outcome"] = "mean_trip_km (km)"; r["weights"] = "Queen"; r["model"] = "distance_SDM"
pd.DataFrame(rows).to_csv(OUT / "distance_final_impacts.csv", index=False)

print("=== s2 RQ2 done ===")
print("distance model selection (Queen):")
print(pd.DataFrame(dist_diag)[["model", "logLik", "AIC", "BIC"]].to_string(index=False))
print(f"  LR SDM vs SAR (distance) = {lr_ds:.2f} df={df_ds} p={p_ds:.3g}")
q = [r for r in agg_rows if r.get("weights") == "Queen(primary)" and r.get("variable") in ("pct_nonwhite", "income_10k")]
print("\nRQ2 aggregate SDM (Queen) Non-White & income impacts:")
print(pd.DataFrame(q)[["variable", "effect", "scale", "estimate", "ci_lo", "ci_hi", "p"]].to_string(index=False))
print("\nRQ2 joint race (Queen) per-IQR total impacts:")
jq = [r for r in joint_rows if r["weights"] == "Queen(primary)" and r["effect"] == "total"]
print(pd.DataFrame(jq)[["variable", "scale", "estimate", "ci_lo", "ci_hi", "p"]].to_string(index=False))
