"""Stages A-C: sample/weights audit, OLS spatial tests, expanded ML model set (OLS, SEM, SAR,
SDM, SDEM, full-Durbin diagnostic, SARAR-GMM) under Queen and kNN6, restriction/specification tests,
per-model innovation residual diagnostics. Writes 00,01,02,03,04,07. No manuscript edits."""
import sys, numpy as np, pandas as pd, scipy.stats as st
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from slib import (build_sample, make_W, design, fit_ols, fit_ml, fit_sarar, model_row, _stat,
                  OUT, CONT, DEP, SEED)
from esda.moran import Moran
np.random.seed(SEED)

d, geoids, checksum = build_sample()
n = len(d)
(OUT / "00_sample_geoids.txt").write_text("\n".join(geoids) + "\n")
y = d[DEP].values.reshape(-1, 1)

# ---------- 00 data + weights audit ----------
rows = []
for c in [DEP] + CONT:
    s = d[c]
    rows.append(dict(quantity=f"summary:{c}", mean=round(s.mean(), 4), sd=round(s.std(), 4),
                     min=round(s.min(), 4), p25=round(s.quantile(.25), 4), p50=round(s.quantile(.5), 4),
                     p75=round(s.quantile(.75), 4), max=round(s.max(), 4)))
aud = [dict(item="n_sample", value=n), dict(item="sample_checksum_sha256_16", value=checksum),
       dict(item="borough_reference_category", value="Bronx (drop_first on sorted boroughs)")]
for kind in ["queen", "knn6"]:
    w = make_W(d, "queen" if kind == "queen" else "knn")
    A = (w.sparse > 0).astype(int)
    card = np.asarray(A.sum(1)).flatten()
    sym = int((A != A.T).nnz)
    rowsums = np.asarray(w.sparse.sum(1)).flatten()
    aud += [dict(item=f"{kind}_islands", value=int(len(w.islands))),
            dict(item=f"{kind}_mean_neighbours", value=round(card.mean(), 3)),
            dict(item=f"{kind}_min_neighbours", value=int(card.min())),
            dict(item=f"{kind}_max_neighbours", value=int(card.max())),
            dict(item=f"{kind}_binary_asymmetric_entries(pre_rowstd)", value=sym),
            dict(item=f"{kind}_rowsum_min(post_rowstd)", value=round(float(rowsums.min()), 4)),
            dict(item=f"{kind}_rowsum_max(post_rowstd)", value=round(float(rowsums.max()), 4))]
pd.DataFrame(aud).to_csv(OUT / "00_data_and_weights_audit.csv", index=False)
pd.DataFrame(rows).to_csv(OUT / "00_covariate_summaries.csv", index=False)

# ---------- 01 OLS spatial tests (Queen primary; also kNN6) ----------
ols_rows = []
for kind in ["queen", "knn6"]:
    w = make_W(d, "queen" if kind == "queen" else "knn")
    m, X = fit_ols(d, w)
    ols_rows.append(dict(weights=kind, n=n, r2=round(m.r2, 4),
                         ols_resid_moran_I=round(m.moran_res[0], 4), ols_resid_moran_p=round(m.moran_res[2], 4),
                         LM_error=round(m.lm_error[0], 2), LM_error_p=m.lm_error[1],
                         LM_lag=round(m.lm_lag[0], 2), LM_lag_p=m.lm_lag[1],
                         robustLM_error=round(m.rlm_error[0], 2), robustLM_error_p=m.rlm_error[1],
                         robustLM_lag=round(m.rlm_lag[0], 2), robustLM_lag_p=m.rlm_lag[1]))
pd.DataFrame(ols_rows).to_csv(OUT / "01_ols_spatial_tests.csv", index=False)

# ---------- 02/03 model comparison + 07 residual diagnostics ----------
def Wy(w):
    return (w.sparse @ y).reshape(-1, 1)


def run_weights(kind):
    w = make_W(d, "queen" if kind == "queen" else "knn")
    wdef = "Queen row-standardised" if kind == "queen" else "kNN6 row-standardised"
    Wsp = w.sparse
    comp, resid, store = [], [], {}

    # OLS
    mo, Xo = fit_ols(d, w)
    comp.append(model_row("OLS", "y = Xb + e", mo, Xo, w, checksum, wdef, mo.u, "OLS residual", n, {}))
    # SEM
    me, Xe = fit_ml("error", d, w)
    lam = me.betas.flatten()[-1]
    comp.append(model_row("SEM", "y=Xb+u, u=lambda*Wu+eps", me, Xe, w, checksum, wdef, me.e_filtered,
                          "innovation eps=(I-lam W)u", n, {"lambda": _stat(me.betas.flatten(), np.sqrt(np.diag(me.vm)), len(me.betas) - 1)}))
    store["SEM"] = (me, Xe)
    # SAR
    ml, Xl = fit_ml("lag", d, w)
    rho = ml.betas.flatten()[-1]
    bx = ml.betas.flatten()[:-1]
    eps_sar = y - rho * Wy(w) - Xl.assign(CONSTANT=1)[["CONSTANT"] + list(Xl.columns)].values @ bx.reshape(-1, 1)
    comp.append(model_row("SAR", "y=rho*Wy+Xb+eps", ml, Xl, w, checksum, wdef, eps_sar,
                          "structural innovation eps=y-rho Wy-Xb", n, {"rho": _stat(ml.betas.flatten(), np.sqrt(np.diag(ml.vm)), len(ml.betas) - 1)}))
    store["SAR"] = (ml, Xl)
    # SDM (substantive: lag 4 continuous only)
    md, Xd = fit_ml("lag", d, w, lag_cont=True)
    rhod = md.betas.flatten()[-1]; bxd = md.betas.flatten()[:-1]
    eps_sdm = y - rhod * Wy(w) - Xd.assign(CONSTANT=1)[["CONSTANT"] + list(Xd.columns)].values @ bxd.reshape(-1, 1)
    comp.append(model_row("SDM", "y=rho*Wy+Xb+WXc*theta+eps (lag 4 continuous)", md, Xd, w, checksum, wdef, eps_sdm,
                          "structural innovation", n, {"rho": _stat(md.betas.flatten(), np.sqrt(np.diag(md.vm)), len(md.betas) - 1)}))
    store["SDM"] = (md, Xd)
    # SDEM (substantive: lag 4 continuous only)
    msd, Xsd = fit_ml("error", d, w, lag_cont=True)
    comp.append(model_row("SDEM", "y=Xb+WXc*theta+u, u=lambda*Wu+eps (lag 4 continuous)", msd, Xsd, w, checksum, wdef,
                          msd.e_filtered, "innovation", n, {"lambda": _stat(msd.betas.flatten(), np.sqrt(np.diag(msd.vm)), len(msd.betas) - 1)}))
    store["SDEM"] = (msd, Xsd)
    # Full-Durbin diagnostic (lag ALL non-constant incl borough) for common-factor test
    try:
        mfd, Xfd = fit_ml("lag", d, w, lag_all=True)
        store["FULLDURBIN"] = (mfd, Xfd)
        comp.append(model_row("FullDurbin(diagnostic)", "lag ALL non-constant incl borough (common-factor test only)",
                              mfd, Xfd, w, checksum, wdef, mfd.u, "reduced-form residual (diagnostic)", n,
                              {"rho": _stat(mfd.betas.flatten(), np.sqrt(np.diag(mfd.vm)), len(mfd.betas) - 1)}))
        fd_ok = True
    except Exception as ex:
        store["FULLDURBIN"] = None; fd_ok = False
        comp.append(dict(model="FullDurbin(diagnostic)", specification="lag ALL non-constant — FAILED",
                         n=n, checksum=checksum, W=wdef, note=f"could not fit: {ex}"))
    # SARAR / SAC via GMM (ML unavailable in spreg) — labelled GMM. vm covers up to rho; lambda has no GMM SE here.
    ms, Xs = fit_sarar(d, w)
    bs = ms.betas.flatten(); ses = np.sqrt(np.diag(ms.vm))
    rho_c = float(bs[-2]); rho_s = float(ses[-1])   # rho is last entry covered by vm
    rho_stat = dict(coef=rho_c, se=rho_s, ci_lo=rho_c - 1.96 * rho_s, ci_hi=rho_c + 1.96 * rho_s,
                    p=float(2 * (1 - st.norm.cdf(abs(rho_c / rho_s)))))
    lam_c = float(bs[-1])
    lam_stat = dict(coef=lam_c, se=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), p=float("nan"))
    comp.append(model_row("SARAR/SAC (GMM)", "y=rho*Wy+Xb+u, u=lambda*Wu+eps (GMM, not ML; lambda SE unavailable)",
                          ms, Xs, w, checksum, wdef, ms.e_filtered, "innovation (GMM)", n,
                          {"rho": rho_stat, "lambda": lam_stat}))
    store["SARAR"] = (ms, Xs)
    return comp, store, w


comp_q, store_q, wq = run_weights("queen")
comp_k, store_k, wk = run_weights("knn6")
pd.DataFrame(comp_q).to_csv(OUT / "02_model_comparison_queen.csv", index=False)
pd.DataFrame(comp_k).to_csv(OUT / "03_model_comparison_knn6.csv", index=False)

# consolidated residual diagnostics (07)
res_rows = []
for kind, comp in [("queen", comp_q), ("knn6", comp_k)]:
    for r in comp:
        if "resid_moran_I" in r:
            res_rows.append(dict(weights=kind, model=r["model"], residual=r["resid_name"],
                                 moran_I=r["resid_moran_I"], moran_p=r["resid_moran_p"]))
pd.DataFrame(res_rows).to_csv(OUT / "07_residual_diagnostics.csv", index=False)

# ---------- 04 restriction & specification tests ----------
def logl(store, key):
    return float(store[key][0].logll) if store.get(key) else np.nan


def lr(store, full, restricted, df):
    lf, lr_ = logl(store, full), logl(store, restricted)
    if not (np.isfinite(lf) and np.isfinite(lr_)):
        return dict(stat=np.nan, df=df, p=np.nan, note="likelihood unavailable")
    stat = 2 * (lf - lr_)
    return dict(stat=round(stat, 3), df=df, p=float(1 - st.chi2.cdf(stat, df)))


tests = []
for kind, store, comp in [("queen", store_q, comp_q), ("knn6", store_k, comp_k)]:
    # LR SDM vs SAR : theta=0 (df=4)
    t = lr(store, "SDM", "SAR", 4)
    tests.append(dict(weights=kind, test="LR SDM vs SAR (H0: theta=0, no omitted WX)", **t))
    # common-factor: SDM(substantive) vs SEM (df=4, H0: theta=-rho*beta)
    t = lr(store, "SDM", "SEM", 4)
    tests.append(dict(weights=kind, test="LR SDM vs SEM (common-factor restriction on 4 continuous)", **t))
    # full-Durbin common factor vs SEM (df = #lagged all) if fitted
    if store.get("FULLDURBIN"):
        k_lag = len([c for c in store["FULLDURBIN"][1].columns if c.startswith("W_")])
        t = lr(store, "FULLDURBIN", "SEM", k_lag)
        tests.append(dict(weights=kind, test=f"LR FullDurbin vs SEM (full common-factor, df={k_lag}, diagnostic)", **t))
    # SDEM vs SEM : theta=0 (df=4)
    t = lr(store, "SDEM", "SEM", 4)
    tests.append(dict(weights=kind, test="LR SDEM vs SEM (H0: theta=0, no omitted WX in error model)", **t))
    # SARAR restrictions: GMM -> LR invalid; report significance of rho & lambda instead
    ms = store["SARAR"][0]; bs = ms.betas.flatten(); ses = np.sqrt(np.diag(ms.vm))
    rho_p = float(2 * (1 - st.norm.cdf(abs(bs[-2] / ses[-2])))); lam_p = float(2 * (1 - st.norm.cdf(abs(bs[-1] / ses[-1]))))
    tests.append(dict(weights=kind, test="SARAR rho significance (GMM; LR restriction invalid for GMM)",
                      stat=round(float(bs[-2]), 4), df="-", p=rho_p))
    tests.append(dict(weights=kind, test="SARAR lambda significance (GMM; LR restriction invalid for GMM)",
                      stat=round(float(bs[-1]), 4), df="-", p=lam_p))
pd.DataFrame(tests).to_csv(OUT / "04_restriction_and_specification_tests.csv", index=False)

print("=== 01_models done ===")
print("sample n=%d checksum=%s" % (n, checksum))
print(pd.DataFrame(comp_q)[["model", "logLik", "AIC", "BIC", "resid_moran_I", "resid_moran_p"]].to_string(index=False))
print("\nRestriction tests (Queen):")
print(pd.DataFrame([t for t in tests if t["weights"] == "queen"])[["test", "stat", "df", "p"]].to_string(index=False))
