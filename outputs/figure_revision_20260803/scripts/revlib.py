"""Shared helpers for the figure_revision_20260803 task. British spelling in outputs.
All model statistics are estimated fresh here; measure columns come from the current pipeline
(gpkg + interim CSVs), not from the forbidden old model-result tables."""
import geopandas as gpd, pandas as pd, numpy as np, scipy.stats as st
from libpysal.weights import Queen, KNN
import spreg, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(".")
OUT = ROOT / "outputs" / "figure_revision_20260803"
RQ2_PREDS = ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]


def load_data():
    """Authoritative gpkg + current-pipeline measure columns (radii already in gpkg)."""
    g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg").to_crs("EPSG:2263")
    g["income_10k"] = g["median_hh_income"] / 10000.0
    g["popdens_k"] = g["pop_density"] / 1000.0
    dwell = pd.read_csv(ROOT / "data/interim/lpur_robust.csv", dtype={"GEOID": str})
    prop = pd.read_csv(ROOT / "outputs/lpur_revision_calculations_20260730/T1_lpur_versions.csv", dtype={"GEOID": str})
    supply = pd.read_csv(ROOT / "data/interim/park_area_2km.csv", dtype={"GEOID": str})
    for df in (dwell, prop, supply):
        g = g.merge(df, on="GEOID", how="left")
    return g


def _queen(d, drop_islands=True):
    np.random.seed(42)
    w = Queen.from_dataframe(d, use_index=False)
    if drop_islands and w.islands:
        keep = [i for i in range(len(d)) if i not in set(w.islands)]
        d = d.iloc[keep].reset_index(drop=True)
        w = Queen.from_dataframe(d, use_index=False)
    w.transform = "r"
    return d, w


def fit_error(dep, preds, data, weights="queen", sample_mask=None, standardise=None):
    """ML spatial error. Returns coefs dict + n, lambda, logll, aic, pr2, sample."""
    np.random.seed(42)
    d = data if sample_mask is None else data[sample_mask]
    d = d.dropna(subset=[dep] + preds + ["borough"]).reset_index(drop=True).copy()
    d, w = _queen(d) if weights == "queen" else (d, None)
    if weights == "knn":
        d = d.dropna(subset=[dep] + preds + ["borough"]).reset_index(drop=True)
        w = KNN.from_dataframe(d, k=6); w.transform = "r"
    cols = []
    for c in preds:
        if standardise and c in standardise:
            d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std(); cols.append(c + "_z")
        else:
            cols.append(c)
    boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
    X = pd.concat([d[cols], boro], axis=1)
    m = spreg.ML_Error(d[dep].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
    b = m.betas.flatten(); se = np.sqrt(np.diag(m.vm)); nm = ["CONSTANT"] + list(X.columns) + ["lambda"]
    k = len(b)
    logll = float(getattr(m, "logll", np.nan))
    aic = (2 * k - 2 * logll) if np.isfinite(logll) else np.nan
    coefs = {}
    for i, name in enumerate(nm):
        base = name.replace("_z", "")
        if base in preds:
            c, s = float(b[i]), float(se[i]); z = c / s
            coefs[base] = dict(coef=c, se=s, z=z, p=float(2 * (1 - st.norm.cdf(abs(z)))),
                               ci_lo=c - 1.96 * s, ci_hi=c + 1.96 * s)
    return dict(coefs=coefs, n=int(w.n), lambda_=float(b[-1]), logll=logll, aic=aic,
                pr2=float(m.pr2), sample=d)


def ols_diag(dep, preds, data, sample_mask=None):
    """OLS baseline with spatial diagnostics (Moran on resid, LM tests)."""
    np.random.seed(42)
    d = data if sample_mask is None else data[sample_mask]
    d = d.dropna(subset=[dep] + preds + ["borough"]).reset_index(drop=True).copy()
    d, w = _queen(d)
    boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
    X = pd.concat([d[preds], boro], axis=1)
    m = spreg.OLS(d[dep].values.reshape(-1, 1), X.values, w=w, spat_diag=True, moran=True,
                  name_x=list(X.columns))
    return dict(n=int(w.n), moran_res=(float(m.moran_res[0]), float(m.moran_res[2])),
                lm_error=(float(m.lm_error[0]), float(m.lm_error[1])),
                lm_lag=(float(m.lm_lag[0]), float(m.lm_lag[1])),
                rlm_error=(float(m.rlm_error[0]), float(m.rlm_error[1])),
                rlm_lag=(float(m.rlm_lag[0]), float(m.rlm_lag[1])), r2=float(m.r2))


def weights_summary(d, kind="queen"):
    np.random.seed(42)
    if kind == "queen":
        w = Queen.from_dataframe(d, use_index=False)
    else:
        w = KNN.from_dataframe(d, k=6)
    card = np.array([len(w.neighbors[i]) for i in w.id_order])
    return dict(kind=kind, n=int(w.n), islands=int(len(w.islands)),
                mean_neighbours=float(card.mean()), min_neighbours=int(card.min()),
                max_neighbours=int(card.max()))

# colour-blind-friendly palette (Okabe–Ito subset)
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00",
          purple="#CC79A7", grey="#999999", black="#000000", sky="#56B4E9")
