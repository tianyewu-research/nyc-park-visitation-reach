"""
23_robustness.py   (RQ3 main-model robustness)
Re-fits the RQ3 spatial-error model under three perturbations and tabulates the
standardized coefficients of the hypothesized correlates for stability:
  (A) use_c winsorize 2.5% (vs main 1%)  -> re-derives mismatch, re-fits RQ3
  (B) kNN (k=6) weights (vs main Queen)
  (C) device-volume threshold obs_devices >= 20 (vs main, no binding threshold)
All inference uses ML spatial error (chosen by robust LM in RQ3). Descriptive — no causal claim.

Input:  data/processed/analysis_cbg_final.gpkg
Output: outputs/tables/rq3_robustness.csv
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
from libpysal.weights import Queen, KNN
import spreg
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

MECH = ["quality_index_cbg_svc", "walkability_c", "out_park_ratio", "transit_freq", "park_area_acres"]
CTRL = ["popdens_k", "income_10k", "pct_nonwhite"]
NICE = {"quality_index_cbg_svc": "Park quality", "walkability_c": "Walkability",
        "out_park_ratio": "Out-of-nbhd mobility", "transit_freq": "Transit access",
        "park_area_acres": "Park area", "popdens_k": "Pop. density",
        "income_10k": "Income", "pct_nonwhite": "% non-white"}

cbg = gpd.read_file(PROCESSED / "analysis_cbg_final.gpkg")
cbg["income_10k"] = cbg["median_hh_income"] / 10000.0
cbg["popdens_k"] = cbg["pop_density"] / 1000.0


def fit_rq3(d, weights_kind="queen"):
    """Fit ML spatial-error RQ3 on sample d (with a 'mismatch' column). Return {var: (coef, p)}."""
    d = d.reset_index(drop=True).copy()
    w = Queen.from_dataframe(d, use_index=False) if weights_kind == "queen" else KNN.from_dataframe(d, k=6)
    if w.islands:
        keep = [i for i in range(len(d)) if i not in set(w.islands)]
        d = d.iloc[keep].reset_index(drop=True)
        w = Queen.from_dataframe(d, use_index=False) if weights_kind == "queen" else KNN.from_dataframe(d, k=6)
    w.transform = "r"
    for c in MECH + CTRL:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    boro = pd.get_dummies(d["borough"], prefix="boro", drop_first=True).astype(float)
    Xdf = pd.concat([d[[c + "_z" for c in MECH + CTRL]], boro], axis=1)
    m = spreg.ML_Error(d["mismatch"].values.reshape(-1, 1), Xdf.values, w=w,
                       name_x=list(Xdf.columns))
    names = ["CONST"] + list(Xdf.columns) + ["lambda"]
    b = m.betas.flatten()
    se = np.sqrt(np.diag(m.vm))
    out = {}
    for i, nm in enumerate(names):
        if nm.endswith("_z"):
            p = 2 * (1 - st.norm.cdf(abs(b[i] / se[i])))
            out[nm[:-2]] = (b[i], p, len(d))
    return out


def derive_mismatch(df, winsor_p):
    """Re-derive mismatch under a given winsorize level (first stage log_use ~ log_access)."""
    df = df.copy()
    use = df["V_c"] / df["total_pop"]
    use[df["total_pop"] == 0] = np.nan
    lo, hi = use.quantile(winsor_p), use.quantile(1 - winsor_p)
    log_use = np.log1p(use.clip(lo, hi))
    s = df.assign(log_use=log_use).dropna(subset=["log_use", "log_access"])
    fit = sm.OLS(s["log_use"].values, sm.add_constant(s["log_access"].values)).fit()
    df.loc[s.index, "mismatch"] = fit.resid
    return df


# ── Main spec reference (winsor 1%, Queen) — read from RQ3 output ──
main = pd.read_csv(TABLES / "rq3_regression.csv")
main = main[main["term"].str.endswith("_z")].copy()
main["v"] = main["term"].str[:-2]
ref = {r["v"]: (r["spatial_coef"], r["spatial_p"]) for _, r in main.iterrows()}

# ── (A) winsorize 2.5% ──
dA = derive_mismatch(cbg, 0.025).dropna(subset=["mismatch"] + MECH + CTRL + ["borough"])
resA = fit_rq3(dA, "queen")

# ── (B) kNN weights (main mismatch) ──
cbg["mismatch"] = cbg["mismatch_c"]
dB = cbg.dropna(subset=["mismatch"] + MECH + CTRL + ["borough"])
resB = fit_rq3(dB, "knn")

# ── (C) device threshold obs_devices >= 20 ──
dC = cbg[cbg["obs_devices"] >= 20].dropna(subset=["mismatch"] + MECH + CTRL + ["borough"])
resC = fit_rq3(dC, "queen")

# ── Assemble comparison ──
def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

rows = []
for v in MECH + CTRL:
    rows.append({
        "variable": NICE[v],
        "main(1%,Queen)": f"{ref[v][0]:+.3f}{star(ref[v][1])}",
        "winsor2.5%": f"{resA[v][0]:+.3f}{star(resA[v][1])}",
        "kNN": f"{resB[v][0]:+.3f}{star(resB[v][1])}",
        "dev>=20": f"{resC[v][0]:+.3f}{star(resC[v][1])}",
    })
tbl = pd.DataFrame(rows)
tbl.to_csv(TABLES / "rq3_robustness.csv", index=False)

print(f"Samples: main n~6228 | winsor2.5 n={list(resA.values())[0][2]} | "
      f"kNN n={list(resB.values())[0][2]} | dev>=20 n={list(resC.values())[0][2]} "
      f"(dropped {len(cbg)-len(dC)} low-obs)")
print("\n=== RQ3 robustness: standardized coefficients (sign + significance) ===")
print(tbl.to_string(index=False))
print("\nNote: obs_devices>=5 and >=10 are non-binding (0 CBGs below 10), so device-threshold")
print("sensitivity only meaningfully tests >=20.")

# --- self-check ---
_t = pd.read_csv(TABLES / "rq3_robustness.csv")
assert len(_t) == len(MECH) + len(CTRL)
print("[self-check] 23 robustness OK")
