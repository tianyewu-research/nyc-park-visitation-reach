"""Part 1 — correct the model-selection package: full-rank diagnostic Durbin + valid LR tests,
corrected AIC/BIC reading, corrected (asymmetric-valid) impact validation. No manuscript edits."""
import sys, numpy as np, pandas as pd, scipy.stats as st
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
sys.path.insert(0, "outputs/spatial_model_reassessment_20260803/scripts")
from slib import build_sample, make_W, design, fit_ols, fit_ml, CONT, DEP
from slib2 import fit_sdm, trace_powers, multipliers, OUT
import spreg
from esda.moran import Moran

d, gid, checksum = build_sample()
n = len(d); y = d[DEP].values.reshape(-1, 1)
np.random.seed(42)


def aicbic(m):
    ll = float(m.logll); k = len(m.betas.flatten())
    return ll, 2 * k - 2 * ll, k * np.log(n) - 2 * ll, k


def fullrank_durbin(d, w):
    """Design [const, 4 cont, 4 boro, W(all 8 non-constant)]; drop redundant WX by incremental rank;
    fit ML_Lag on the maximal estimable set. Returns model, kept WX names, dropped WX names."""
    boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
    own = pd.concat([d[CONT], boro], axis=1)                     # 8 non-constant own
    allcols = list(own.columns)
    WXall = pd.DataFrame(w.sparse @ own.values, columns=["W_" + c for c in allcols], index=d.index)
    const = np.ones((n, 1))
    keep_own = own.values
    base = np.hstack([const, keep_own])                          # const + 8 own (assumed full rank)
    r0 = np.linalg.matrix_rank(base, tol=1e-8)
    kept, dropped = [], []
    cur = base
    for c in WXall.columns:
        cand = np.hstack([cur, WXall[[c]].values])
        if np.linalg.matrix_rank(cand, tol=1e-8) > cur.shape[1] - 0.5 and np.linalg.matrix_rank(cand, tol=1e-8) == cur.shape[1] + 1:
            cur = cand; kept.append(c)
        else:
            dropped.append(c)
    Xdes = pd.concat([own, WXall[kept]], axis=1)                 # spreg adds const
    m = spreg.ML_Lag(y, Xdes.values, w=w, name_x=list(Xdes.columns))
    return m, kept, dropped, Xdes


comp_rows, restr_rows, val_rows = [], [], []
mem = ["# Corrected model-selection memo (partial SDM adopted) — 2026-08-03\n",
       f"Sample n={n}, checksum {checksum}. Queen = prespecified PRIMARY W; kNN6 = sensitivity.",
       "Cross-sectional ecological association; rho and indirect impacts are NOT causal spillovers.\n"]

targets = {"queen": dict(ll=4540.298, k=16, lr=123.04, df=6, pf=0.459),
           "knn": dict(ll=4511.525, k=17, lr=86.20, df=7, pf=0.080)}

for kind in ["queen", "knn"]:
    w = make_W(d, kind); wlab = "Queen" if kind == "queen" else "kNN6"
    # base ML models on identical sample
    ols, Xo = fit_ols(d, w)
    me, Xe = fit_ml("error", d, w); ll_sem, aic_sem, bic_sem, k_sem = aicbic(me)
    ml, Xl = fit_ml("lag", d, w); ll_sar, aic_sar, bic_sar, k_sar = aicbic(ml)
    msdm, Xd = fit_sdm(d, w); ll_sdm, aic_sdm, bic_sdm, k_sdm = aicbic(msdm)
    msde, Xsde = fit_ml("error", d, w, lag_cont=True); ll_sde, aic_sde, bic_sde, k_sde = aicbic(msde)
    def struct_innov(mm, Xdes):
        """structural innovation for a lag/Durbin model using explicit design (spreg prepends const)."""
        b = mm.betas.flatten(); rho = b[-1]; b0 = b[0]; bx = b[1:-1]
        return y - rho * (w.sparse @ y) - (b0 + Xdes.values @ bx.reshape(-1, 1))

    for nm, mm, ll, aic, bic, kk, sp, resid in [
        ("OLS", ols, float("nan"), float("nan"), float("nan"), len(ols.betas), "", ols.u),
        ("SEM", me, ll_sem, aic_sem, bic_sem, k_sem, f"lambda={me.betas.flatten()[-1]:.4f}", me.e_filtered),
        ("SAR", ml, ll_sar, aic_sar, bic_sar, k_sar, f"rho={ml.betas.flatten()[-1]:.4f}", struct_innov(ml, Xl)),
        ("SDM(partial)", msdm, ll_sdm, aic_sdm, bic_sdm, k_sdm, f"rho={msdm.betas.flatten()[-1]:.4f}", struct_innov(msdm, Xd)),
        ("SDEM(partial)", msde, ll_sde, aic_sde, bic_sde, k_sde, f"lambda={msde.betas.flatten()[-1]:.4f}", msde.e_filtered)]:
        rm = Moran(np.asarray(resid).flatten(), w, permutations=999)
        comp_rows.append(dict(weights=wlab, model=nm, n=n, checksum=checksum, n_params=kk,
                              logLik=round(ll, 3) if np.isfinite(ll) else "NA",
                              AIC=round(aic, 3) if np.isfinite(aic) else "NA",
                              BIC=round(bic, 3) if np.isfinite(bic) else "NA",
                              spatial=sp, resid_moran_I=round(rm.I, 4), resid_moran_p=round(rm.p_sim, 4)))

    # full-rank diagnostic Durbin + valid LR
    mfd, kept, dropped, Xfd = fullrank_durbin(d, w); ll_fd, aic_fd, bic_fd, k_fd = aicbic(mfd)
    lr_fd_sem = 2 * (ll_fd - ll_sem); df_fd = k_fd - k_sem
    p_fd = float(st.chi2.sf(lr_fd_sem, df_fd))
    lr_full_partial = 2 * (ll_fd - ll_sdm); df_fp = k_fd - k_sdm
    p_fp = float(st.chi2.sf(lr_full_partial, df_fp)) if df_fp > 0 else float("nan")
    lr_sdm_sar = 2 * (ll_sdm - ll_sar); df_ss = k_sdm - k_sar
    p_ss = float(st.chi2.sf(lr_sdm_sar, df_ss))
    restr_rows += [
        dict(weights=wlab, test="LR partial-SDM vs SAR (H0 theta_cont=0; NESTED, valid)",
             stat=round(lr_sdm_sar, 3), df=df_ss, p=p_ss),
        dict(weights=wlab, test="LR full-rank Durbin vs SEM (valid; df=param diff)",
             stat=round(lr_fd_sem, 3), df=df_fd, p=p_fd),
        dict(weights=wlab, test="LR full-rank Durbin vs partial-SDM (do estimable W*boro add info?)",
             stat=round(lr_full_partial, 3), df=df_fp, p=round(p_fp, 4)),
        dict(weights=wlab, test="partial-SDM vs SEM likelihood difference (NON-NESTED: NOT an LR test)",
             stat=round(2 * (ll_sdm - ll_sem), 3), df="n/a (non-nested; report as raw 2*dLL only)", p="n/a")]
    # reproduction check
    tg = targets[kind]
    val_rows.append(dict(check=f"{wlab} full-rank Durbin logLik", reproduced=round(ll_fd, 3), target=tg["ll"],
                         ok=abs(ll_fd - tg["ll"]) < 0.5))
    val_rows.append(dict(check=f"{wlab} full-rank Durbin n_params", reproduced=k_fd, target=tg["k"],
                         ok=k_fd == tg["k"]))
    val_rows.append(dict(check=f"{wlab} LR full-Durbin vs SEM stat", reproduced=round(lr_fd_sem, 2), target=tg["lr"],
                         ok=abs(lr_fd_sem - tg["lr"]) < 1.0))
    val_rows.append(dict(check=f"{wlab} LR df", reproduced=df_fd, target=tg["df"], ok=df_fd == tg["df"]))
    val_rows.append(dict(check=f"{wlab} partial-vs-full p", reproduced=round(p_fp, 3), target=tg["pf"],
                         ok=abs(p_fp - tg["pf"]) < 0.03))

    # impact-validation: trace-power vs spreg native (asymmetric-valid), tolerance + tail
    rho = float(msdm.betas.flatten()[-1]); native = msdm.sp_multipliers["full"]
    Q, mp = 200, 4000
    T = trace_powers(w, Q=Q, m_probe=mp)
    M1, MW, MT = multipliers(rho, T, n)
    val_rows.append(dict(check=f"{wlab} SDM direct-multiplier trace-power vs spreg native",
                         reproduced=round(M1, 6), target=round(native[0], 6), ok=abs(M1 - native[0]) < 5e-4))
    val_rows.append(dict(check=f"{wlab} SDM total-multiplier closed-form vs native",
                         reproduced=round(MT, 6), target=round(native[2], 6), ok=abs(MT - native[2]) < 1e-6))
    val_rows.append(dict(check=f"{wlab} trace tail rho^{Q}*T_Q/n (convergence)",
                         reproduced=float(f"{abs((rho**Q)*T[-1]/n):.2e}"), target="~0", ok=abs((rho**Q)*T[-1]/n) < 1e-6))

    mem += [f"## {wlab}",
            f"- AIC/BIC among substantive ML models: SEM {aic_sem:.1f}/{bic_sem:.1f}, SAR {aic_sar:.1f}/{bic_sar:.1f}, "
            f"partial-SDM {aic_sdm:.1f}/{bic_sdm:.1f}, SDEM {aic_sde:.1f}/{bic_sde:.1f}.",
            f"- full-rank Durbin: logLik {ll_fd:.3f}, {k_fd} params; dropped redundant WX: {dropped or 'none'}; kept W*: {kept}.",
            f"- VALID LR full-rank Durbin vs SEM = {lr_fd_sem:.2f} on {df_fd} df (p={p_fd:.3g}).",
            f"- VALID LR partial-SDM vs SAR (theta_cont=0) = {lr_sdm_sar:.2f} on {df_ss} df (p={p_ss:.3g}).",
            f"- full-rank Durbin vs partial-SDM addition test: {lr_full_partial:.2f} on {df_fp} df (p={p_fp:.3f}) — "
            f"estimable lagged-borough terms {'do NOT' if p_fp>0.05 else 'DO'} add significant information.",
            f"- NON-TESTABLE: partial-SDM vs SEM is non-nested (partial SDM omits W*borough); reported only as a raw "
            f"likelihood difference 2*dLL={2*(ll_sdm-ll_sem):.2f}, NOT an LR test.\n"]

pd.DataFrame(comp_rows).to_csv(OUT / "corrected_model_comparison.csv", index=False)
pd.DataFrame(restr_rows).to_csv(OUT / "corrected_restriction_tests.csv", index=False)
pd.DataFrame(val_rows).to_csv(OUT / "corrected_impact_validation.csv", index=False)

# corrected AIC/BIC reading (Queen SDM best AIC&BIC; kNN SDM best AIC but SAR better BIC)
mem += ["## Corrected AIC/BIC reading",
        "- Queen (PRIMARY): partial-SDM has the best AIC and BIC among the substantive ML candidates.",
        "- kNN6 (sensitivity): partial-SDM has the best AIC, but SAR has the lower/better BIC. The valid nested LR of "
        "SDM vs SAR nevertheless rejects theta=0 under BOTH weights, and Queen is the prespecified primary W (kNN6 is "
        "not a post-hoc alternative).",
        "- The earlier claim that SDM had the best AIC AND BIC under both weights is corrected.\n"]
(OUT / "corrected_model_selection_memo.md").write_text("\n".join(mem) + "\n")

vr = pd.DataFrame(val_rows)
allok = bool(vr["ok"].all())
print("=== s1 corrected selection ===")
print(pd.DataFrame(restr_rows).to_string(index=False))
print("\nReproduction validation (all target checks):", "ALL PASS" if allok else "MISMATCH — DIAGNOSE")
print(vr.to_string(index=False))
