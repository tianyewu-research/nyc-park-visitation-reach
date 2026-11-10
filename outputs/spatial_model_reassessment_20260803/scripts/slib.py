"""Shared library for the LPUR spatial-model reassessment (2026-08-03).
Same authoritative inputs and sample rule as outputs/figure_revision_20260803/scripts/revlib.py.
All model estimates produced here; the manuscript is NOT touched. R/spatialreg is unavailable on
this machine, so the required R cross-check cannot run; independent numerical cross-checks are used
and labelled as such. Cross-sectional associational analysis: a lagged outcome is NOT interpreted as
a causal spillover/diffusion/peer effect."""
import geopandas as gpd, pandas as pd, numpy as np, scipy.stats as st, hashlib
from libpysal.weights import Queen, KNN
from esda.moran import Moran
import spreg
from scipy.sparse import identity, csr_matrix
from scipy.sparse.linalg import spsolve
import scipy.linalg as sla
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(".")
OUT = ROOT / "outputs" / "spatial_model_reassessment_20260803"
SEED = 42
DEP = "lpur_2.0km"
CONT = ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]   # substantive continuous


def build_sample():
    """Exact n=6,252 RQ2 sample (Queen islands dropped), ordered. Returns d, geoids, checksum."""
    g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg").to_crs("EPSG:2263")
    g["income_10k"] = g["median_hh_income"] / 10000.0
    g["popdens_k"] = g["pop_density"] / 1000.0
    d = g.dropna(subset=[DEP] + CONT + ["borough"]).reset_index(drop=True).copy()
    np.random.seed(SEED)
    w = Queen.from_dataframe(d, use_index=False)
    if w.islands:
        keep = [i for i in range(len(d)) if i not in set(w.islands)]
        d = d.iloc[keep].reset_index(drop=True)
    geoids = list(d["GEOID"].values)
    checksum = hashlib.sha256("\n".join(geoids).encode()).hexdigest()[:16]
    return d, geoids, checksum


def make_W(d, kind="queen"):
    np.random.seed(SEED)
    w = Queen.from_dataframe(d, use_index=False) if kind == "queen" else KNN.from_dataframe(d, k=6)
    w.transform = "r"
    return w


def design(d, lag_cont=False, lag_all=False, w=None):
    """Return X design (DataFrame) and column names. lag_cont: append W*continuous (SDM/SDEM
    substantive). lag_all: append W*(all non-constant incl borough) for the diagnostic full Durbin."""
    boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
    base = pd.concat([d[CONT], boro], axis=1)
    if not (lag_cont or lag_all):
        return base
    Wmat = w.sparse
    if lag_cont:
        WX = pd.DataFrame(Wmat @ d[CONT].values, columns=["W_" + c for c in CONT], index=d.index)
        return pd.concat([base, WX], axis=1)
    if lag_all:
        allcols = base.columns
        WX = pd.DataFrame(Wmat @ base.values, columns=["W_" + c for c in allcols], index=d.index)
        return pd.concat([base, WX], axis=1)


def _stat(b, se, i):
    c, s = float(b[i]), float(se[i]); z = c / s
    return dict(coef=c, se=s, ci_lo=c - 1.96 * s, ci_hi=c + 1.96 * s, p=float(2 * (1 - st.norm.cdf(abs(z)))))


def fit_ols(d, w):
    X = design(d)
    m = spreg.OLS(d[DEP].values.reshape(-1, 1), X.values, w=w, spat_diag=True, moran=True, name_x=list(X.columns))
    return m, X


def fit_ml(kind, d, w, lag_cont=False, lag_all=False):
    """kind in {'error','lag'}; optional selective/all SLX lags -> SDEM/SDM (or full Durbin)."""
    X = design(d, lag_cont=lag_cont, lag_all=lag_all, w=w)
    y = d[DEP].values.reshape(-1, 1)
    if kind == "error":
        m = spreg.ML_Error(y, X.values, w=w, name_x=list(X.columns))
    else:
        m = spreg.ML_Lag(y, X.values, w=w, name_x=list(X.columns))
    return m, X


def fit_sarar(d, w):
    X = design(d)
    m = spreg.GM_Combo(d[DEP].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
    return m, X


def model_row(name, spec, m, X, w, checksum, wdef, resid_vec, resid_name, n, spatial):
    """spatial: dict of {'rho':(...), 'lambda':(...)} present ones. resid_vec: residual for Moran."""
    np.random.seed(SEED)
    rm = Moran(np.asarray(resid_vec).flatten(), w, permutations=999)
    b = m.betas.flatten(); k = len(b)
    logll = float(getattr(m, "logll", np.nan))
    aic = (2 * k - 2 * logll) if np.isfinite(logll) else np.nan
    bic = (k * np.log(n) - 2 * logll) if np.isfinite(logll) else np.nan
    row = dict(model=name, specification=spec, n=n, checksum=checksum, W=wdef, n_params=k,
               logLik=round(logll, 3) if np.isfinite(logll) else "NA (GMM)",
               AIC=round(aic, 3) if np.isfinite(aic) else "NA", BIC=round(bic, 3) if np.isfinite(bic) else "NA",
               resid_name=resid_name, resid_moran_I=round(rm.I, 4), resid_moran_p=round(rm.p_sim, 4))
    for key, tup in spatial.items():
        row[f"{key}"] = round(tup["coef"], 4); row[f"{key}_se"] = round(tup["se"], 4)
        row[f"{key}_ci_lo"] = round(tup["ci_lo"], 4); row[f"{key}_ci_hi"] = round(tup["ci_hi"], 4)
        row[f"{key}_p"] = tup["p"]
    return row


def eig_rowstd(w):
    """Real eigenvalues of the row-standardised W via the symmetric similarity D^-1/2 A D^-1/2."""
    A = (w.sparse > 0).astype(float)          # binary adjacency (Queen) or kNN adjacency
    deg = np.asarray(A.sum(1)).flatten()
    dinv = 1.0 / np.sqrt(deg)
    Dm = csr_matrix((dinv, (range(len(deg)), range(len(deg)))))
    S = (Dm @ A @ Dm).toarray()
    S = (S + S.T) / 2.0
    ev = sla.eigvalsh(S)
    return np.clip(ev, -0.9999, 0.9999)


def impacts_from_draws(beta_idx_map, mean_params, vc, ev, w, var_names, model, n, nsim=5000, theta_idx_map=None):
    """LeSage-Pace average direct/indirect/total impacts with simulation SE/CI/p.
    For row-standardised W (W1=1): total = (beta+theta)/(1-rho); direct uses eigenvalue traces.
    Draws full parameter vector ~ N(mean_params, vc). rho is the last element (SAR/SDM/SARAR)."""
    rng = np.random.default_rng(SEED)
    draws = rng.multivariate_normal(mean_params, vc, size=nsim)
    rho_i = len(mean_params) - 1
    out = []
    for v in var_names:
        bi = beta_idx_map[v]; ti = theta_idx_map[v] if theta_idx_map else None
        D = np.empty(nsim); I = np.empty(nsim); T = np.empty(nsim)
        for s in range(nsim):
            p = draws[s]; rho = p[rho_i]; beta = p[bi]; theta = p[ti] if ti is not None else 0.0
            tr1 = np.sum(1.0 / (1.0 - rho * ev)) / n            # (1/n) tr[(I-rhoW)^-1]
            trW = np.sum(ev / (1.0 - rho * ev)) / n             # (1/n) tr[(I-rhoW)^-1 W]
            d_eff = beta * tr1 + theta * trW
            t_eff = (beta + theta) / (1.0 - rho)
            D[s] = d_eff; T[s] = t_eff; I[s] = t_eff - d_eff
        for scale_name, arr in [("direct", D), ("indirect", I), ("total", T)]:
            m_, lo, hi = arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)
            p = 2 * min((arr > 0).mean(), (arr < 0).mean())
            out.append(dict(model=model, variable=v, effect=scale_name, estimate=round(m_, 5),
                            se=round(arr.std(), 5), ci_lo=round(lo, 5), ci_hi=round(hi, 5),
                            p_two_sided=round(p, 5), nsim=nsim, seed=SEED))
    return out


def total_crosscheck(rho, w, n):
    """Independent numerical check that (1/n)1'(I-rhoW)^-1 1 = 1/(1-rho) for row-std W (via sparse solve)."""
    Wm = w.sparse.tocsc()
    A = identity(n, format="csc") - rho * Wm
    z = spsolve(A, np.ones(n))
    return float(z.mean()), 1.0 / (1.0 - rho)
