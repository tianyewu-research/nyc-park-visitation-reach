"""Corrected shared library for the final partial-SDM revision (2026-08-03).
Impacts use spreg native full multipliers (exact, on the TRUE directed row-standardised W — valid
for asymmetric kNN6) for point estimates, and a trace-of-powers (Neumann) method on the SAME directed
W for simulation uncertainty. No symmetrisation of kNN6. Cross-sectional ecological association:
rho and indirect impacts are NOT causal spillovers/diffusion/peer effects.
"""
import sys, numpy as np, pandas as pd, scipy.stats as st, hashlib
sys.path.insert(0, "outputs/spatial_model_reassessment_20260803/scripts")
from slib import build_sample, make_W, CONT, DEP, SEED   # reuse authoritative sample/weights
import spreg
from libpysal.weights import Queen, KNN
from esda.moran import Moran
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
OUT = Path("outputs/final_sdm_revision_20260803")


def boro_dummies(d):
    return pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)


def design_sdm(d, w, cont=CONT):
    """Own X (cont + borough) plus W*cont (borough NOT lagged)."""
    base = pd.concat([d[cont], boro_dummies(d)], axis=1)
    WX = pd.DataFrame(w.sparse @ d[cont].values, columns=["W_" + c for c in cont], index=d.index)
    return pd.concat([base, WX], axis=1)


def fit_sdm(d, w, cont=CONT, dep=DEP):
    X = design_sdm(d, w, cont)
    m = spreg.ML_Lag(d[dep].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns),
                     spat_impacts="full")
    return m, X


def trace_powers(w, Q=200, m_probe=400, seed=SEED):
    """Estimate T_q = tr(W^q), q=0..Q, on the TRUE directed row-standardised W via Hutchinson probes.
    Returns array T (len Q+1). Valid for asymmetric W."""
    rng = np.random.default_rng(seed)
    Wm = w.sparse.tocsr()
    n = Wm.shape[0]
    T = np.zeros(Q + 1); T[0] = n
    acc = np.zeros(Q + 1)
    Z = rng.integers(0, 2, size=(n, m_probe)).astype(float) * 2 - 1   # Rademacher
    V = Z.copy()
    for q in range(1, Q + 1):
        V = Wm @ V
        acc[q] = np.einsum("ij,ij->", Z, V)
    T[1:] = acc[1:] / m_probe
    return T


def multipliers(rho, T, n):
    """Average multipliers from the trace series (valid for asymmetric W):
    M1 = (1/n) tr[(I-rhoW)^-1] = (1/n) sum_q rho^q T_q ; MT = 1/(1-rho) ; MW = (M1-1)/rho."""
    q = np.arange(len(T))
    M1 = float(np.sum((rho ** q) * T) / n)
    MT = 1.0 / (1.0 - rho)
    MW = (M1 - 1.0) / rho if abs(rho) > 1e-9 else 0.0
    return M1, MW, MT


def sdm_impacts(m, X, w, T, variables, scales_by_var, nsim=5000, seed=SEED, rho_bounds=(-0.999, 0.999)):
    """Direct/indirect/total impacts per variable with simulation SE/CI/p. Draws (all betas incl rho)
    from the ML covariance; rejects draws with rho outside admissible bounds. Multipliers use the
    TRUE directed W: the point estimate uses spreg's exact native direct multiplier, and each draw's
    M1(rho) is the trace-power series ANCHORED so M1(rho_hat) equals the native value exactly
    (deterministic point estimate; series supplies only the rho-sensitivity). MT=1/(1-rho) exact."""
    cols = ["CONSTANT"] + list(X.columns)
    b = m.betas.flatten(); rho_i = len(b) - 1
    vc = m.vm
    n = w.n
    rho_hat = float(b[rho_i])
    native_M1 = float(m.sp_multipliers["full"][0])            # exact operator direct multiplier
    qexp = np.arange(len(T))
    M1_series_hat = float(np.sum((rho_hat ** qexp) * T) / n)
    anchor = native_M1 - M1_series_hat                        # deterministic anchoring correction

    def M1_of(rho):
        return float(np.sum((rho ** qexp) * T) / n) + anchor

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(b, vc, size=int(nsim * 1.6))
    adm = (draws[:, rho_i] > rho_bounds[0]) & (draws[:, rho_i] < rho_bounds[1])
    draws = draws[adm][:nsim]
    retained = len(draws)
    rows = []
    point_mult = {"_anchor_correction": anchor, "_native_M1": native_M1, "_series_M1_hat": M1_series_hat}
    for v in variables:
        bi = cols.index(v); ti = cols.index("W_" + v)
        D = np.empty(retained); I = np.empty(retained); Tot = np.empty(retained)
        for s in range(retained):
            p = draws[s]; rho = p[rho_i]; beta = p[bi]; theta = p[ti]
            M1 = M1_of(rho)
            MW = (M1 - 1.0) / rho
            MT = 1.0 / (1.0 - rho)
            d_eff = beta * M1 + theta * MW
            t_eff = (beta + theta) * MT
            D[s] = d_eff; Tot[s] = t_eff; I[s] = t_eff - d_eff
        # point estimate at MLE using exact native multiplier
        rho0 = rho_hat; beta0 = float(b[bi]); theta0 = float(b[ti])
        M1_0 = native_M1; MW_0 = (M1_0 - 1.0) / rho0; MT_0 = 1.0 / (1.0 - rho0)
        pt = dict(direct=beta0 * M1_0 + theta0 * MW_0, total=(beta0 + theta0) * MT_0)
        pt["indirect"] = pt["total"] - pt["direct"]
        point_mult[v] = dict(beta=float(beta0), theta=float(theta0), rho=float(rho0),
                             M1=M1_0, MW=MW_0, MT=MT_0, **pt)
        arrs = {"direct": D, "indirect": I, "total": Tot}
        for scale_name, factor in scales_by_var[v]:
            for eff in ["direct", "indirect", "total"]:
                a = arrs[eff] * factor
                est = pt[eff] * factor
                lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
                frac = min((a > 0).mean(), (a < 0).mean())
                p_res = 2.0 / retained
                p_txt = f"<{p_res:.4f}" if (2 * frac) < p_res else f"{2 * frac:.4f}"
                rows.append(dict(variable=v, effect=eff, scale=scale_name,
                                 estimate=round(est, 6), se=round(a.std(), 6),
                                 ci_lo=round(lo, 6), ci_hi=round(hi, 6), p=p_txt,
                                 nsim_retained=retained, seed=seed))
    return rows, retained, point_mult


def innovation_moran(m, y, w, kind, Xdes=None):
    """Correct innovation residual + its Moran's I (permutation). For lag/Durbin models pass the
    explicit design Xdes (without constant); spreg prepends the constant, so beta = [b0, coefs, rho]."""
    np.random.seed(SEED)
    if kind == "error":
        e = m.e_filtered
    elif kind == "lag":
        b = m.betas.flatten(); rho = b[-1]; b0 = b[0]; bx = b[1:-1]
        e = y - rho * (w.sparse @ y) - (b0 + Xdes.values @ bx.reshape(-1, 1))
    else:
        e = m.u
    mi = Moran(np.asarray(e).flatten(), w, permutations=999)
    return float(mi.I), float(mi.p_sim)


def stat(b, se, i):
    c, s = float(b[i]), float(se[i]); z = c / s
    return dict(coef=c, se=s, ci_lo=c - 1.96 * s, ci_hi=c + 1.96 * s,
                p=float(2 * (1 - st.norm.cdf(abs(z)))))
