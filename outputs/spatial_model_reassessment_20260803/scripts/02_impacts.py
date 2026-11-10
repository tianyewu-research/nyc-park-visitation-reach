"""Stage D-E: LeSage-Pace average direct/indirect/total impacts (simulation SE/CI/p) for the lag-type
candidates (SAR, SDM, SARAR-GMM) under Queen and kNN6. Raw slopes are NOT effects. R spatialreg is
unavailable, so an independent numerical cross-check replaces spatialreg::impacts(). Writes 05,06,08,09."""
import sys, numpy as np, pandas as pd, scipy.stats as st
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from slib import (build_sample, make_W, design, fit_ml, fit_sarar, eig_rowstd,
                  impacts_from_draws, total_crosscheck, OUT, CONT, DEP, SEED)
np.random.seed(SEED)

d, geoids, checksum = build_sample()
n = len(d)
y = d[DEP].values.reshape(-1, 1)
IQR_nw = float(d["pct_nonwhite"].quantile(.75) - d["pct_nonwhite"].quantile(.25))
NSIM = 5000


def sar_params(m, X):
    b = m.betas.flatten()
    cols = ["CONSTANT"] + list(X.columns)                 # rho is appended after
    mean = b                                              # includes rho last
    idx = {v: cols.index(v) for v in CONT}
    return mean, m.vm, idx, None


def sdm_params(m, X):
    b = m.betas.flatten()
    cols = ["CONSTANT"] + list(X.columns)
    mean = b
    bidx = {v: cols.index(v) for v in CONT}
    tidx = {v: cols.index("W_" + v) for v in CONT}
    return mean, m.vm, bidx, tidx


def sarar_params(m, X):
    b = m.betas.flatten()                                 # [const, betas, boro, rho, lambda]
    cols = ["CONSTANT"] + list(X.columns) + ["rho", "lambda"]
    mean = b[:-1]                                          # drop lambda (impacts depend on rho only)
    vc = m.vm                                              # covers const..rho (matches mean length)
    idx = {v: (["CONSTANT"] + list(X.columns)).index(v) for v in CONT}
    return mean, vc, idx, None


def collect(model, weights, mean, vc, ev, bidx, tidx):
    # per-unit (0->1) impacts
    base = impacts_from_draws(bidx, mean, vc, ev, None, CONT, model, n, nsim=NSIM, theta_idx_map=tidx)
    rows = []
    for r in base:
        r = dict(r); r["weights"] = weights; r["scale"] = "per 1.0 share (0->1)"
        # attach structural beta/theta labels
        rows.append(r)
    # extra scales for pct_nonwhite only
    for scale_name, factor in [("+0.10 share", 0.10), (f"IQR ({IQR_nw:.3f})", IQR_nw)]:
        sub = impacts_from_draws({"pct_nonwhite": bidx["pct_nonwhite"]}, mean, vc, ev, None,
                                 ["pct_nonwhite"], model, n, nsim=NSIM,
                                 theta_idx_map=({"pct_nonwhite": tidx["pct_nonwhite"]} if tidx else None))
        for r in sub:
            r = dict(r)
            for kk in ["estimate", "se", "ci_lo", "ci_hi"]:
                r[kk] = round(r[kk] * factor, 6)
            r["weights"] = weights; r["scale"] = scale_name
            rows.append(r)
    return rows


allrows = []; cross = []; stability = []
for kind in ["queen", "knn6"]:
    w = make_W(d, "queen" if kind == "queen" else "knn")
    print(f"  eigenvalues for {kind} ...")
    ev = eig_rowstd(w)
    # SAR
    ml, Xl = fit_ml("lag", d, w)
    mean, vc, bidx, _ = sar_params(ml, Xl)
    allrows += collect("SAR", kind, mean, vc, ev, bidx, None)
    zc, closed = total_crosscheck(float(ml.betas.flatten()[-1]), w, n)
    cross.append(dict(check="SAR total-impact numerical vs closed-form (1/(1-rho))", weights=kind,
                      sparse_solve_mean=round(zc, 5), closed_form=round(closed, 5),
                      abs_diff=round(abs(zc - closed), 8)))
    # SDM
    md, Xd = fit_ml("lag", d, w, lag_cont=True)
    mean, vc, bidx, tidx = sdm_params(md, Xd)
    allrows += collect("SDM", kind, mean, vc, ev, bidx, tidx)
    # SARAR (GMM)
    ms, Xs = fit_sarar(d, w)
    mean, vc, bidx, _ = sarar_params(ms, Xs)
    allrows += collect("SARAR/SAC (GMM)", kind, mean, vc, ev, bidx, None)

imp = pd.DataFrame(allrows)
imp_q = imp[imp.weights == "queen"]; imp_k = imp[imp.weights == "knn6"]
cols = ["model", "weights", "variable", "effect", "scale", "estimate", "se", "ci_lo", "ci_hi", "p_two_sided", "nsim", "seed"]
imp_q[cols].to_csv(OUT / "05_impacts_queen.csv", index=False)
imp_k[cols].to_csv(OUT / "06_impacts_knn6.csv", index=False)
pd.DataFrame(cross).to_csv(OUT / "08_python_r_crosscheck.csv", index=False)

# 09 stability: compare leading candidate (SDM) total impacts of pct_nonwhite & income across Queen/kNN
stab = imp[(imp.effect == "total") & (imp.scale == "per 1.0 share (0->1)") &
           (imp.variable.isin(["pct_nonwhite", "income_10k"]))]
piv = stab.pivot_table(index=["model", "variable"], columns="weights", values="estimate")
piv.to_csv(OUT / "09_coefficient_and_impact_stability.csv")

print("=== 02_impacts done ===")
print("cross-check (should match to ~1e-6):")
print(pd.DataFrame(cross).to_string(index=False))
print("\nSDM total impacts (Queen), per 1.0 share:")
q = imp_q[(imp_q.model == "SDM") & (imp_q.scale == "per 1.0 share (0->1)")]
print(q[["variable", "effect", "estimate", "ci_lo", "ci_hi", "p_two_sided"]].to_string(index=False))
print("\nincome_10k total impact stability (SDM):")
print(piv.to_string())
