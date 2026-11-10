"""
29_matched_universe_robustness.py   (C14 robustness)
Recompute mismatch on the matched-park-universe use measure (14c) and rerun RQ2/RQ3 to check
that the income gradient and RQ3 ordering do not depend on POIs outside the access park universe.

Input:  data/processed/analysis_cbg_final.gpkg + data/interim/use_cbg_matched.csv
Output: outputs/tables/matched_universe_robustness.csv
"""
import geopandas as gpd, pandas as pd, numpy as np, scipy.stats as st
import statsmodels.api as sm
from libpysal.weights import Queen
import spreg, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); np.random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"; INTERIM = ROOT / "data" / "interim"; TABLES = ROOT / "outputs" / "tables"

g = gpd.read_file(PROC / "analysis_cbg_final.gpkg")
vm = pd.read_csv(INTERIM / "use_cbg_matched.csv", dtype={"GEOID": str})
g = g.merge(vm, on="GEOID", how="left")

# matched use -> winsor -> log -> first-stage residual = mismatch_m
g["use_m"] = g["V_c_matched"] / g["total_pop"]
g.loc[g["total_pop"] == 0, "use_m"] = np.nan
lo, hi = g["use_m"].quantile(0.01), g["use_m"].quantile(0.99)
g["log_use_m"] = np.log1p(g["use_m"].clip(lo, hi))
fs = g[["log_use_m", "log_access"]].dropna()
b = sm.OLS(fs["log_use_m"], sm.add_constant(fs["log_access"])).fit()
g.loc[fs.index, "mismatch_m"] = b.resid
print(f"first-stage (matched): R2={b.rsquared:.3f} beta={b.params.iloc[1]:.3f} n={len(fs)}")

g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0
nice = {"income_10k": "Income (per $10k)", "pct_nonwhite": "% non-white",
        "quality_index_cbg_svc": "Park quality", "walkability_c": "Walkability",
        "out_park_ratio": "Out-of-nbhd mobility", "transit_freq": "Transit access",
        "park_area_acres": "Park area", "popdens_k": "Pop. density"}


def run(preds, standardize, label):
    d = g.dropna(subset=["mismatch_m"] + preds + ["borough"]).reset_index(drop=True).copy()
    w = Queen.from_dataframe(d, use_index=False)
    if w.islands:
        d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
        w = Queen.from_dataframe(d, use_index=False)
    w.transform = "r"
    cols = []
    for c in preds:
        if standardize:
            d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std(); cols.append(c + "_z")
        else:
            cols.append(c)
    boro = pd.get_dummies(d["borough"], prefix="boro", drop_first=True).astype(float)
    X = pd.concat([d[cols], boro], axis=1)
    sp = spreg.ML_Error(d["mismatch_m"].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
    bt = sp.betas.flatten(); se = np.sqrt(np.diag(sp.vm)); names = ["CONSTANT"] + list(X.columns) + ["lambda"]
    rows = []
    for i, nm in enumerate(names):
        base = nm.replace("_z", "")
        if base in nice:
            p = 2 * (1 - st.norm.cdf(abs(bt[i] / se[i])))
            rows.append({"model": label, "term": nice[base], "coef": round(bt[i], 4),
                         "p": round(p, 4), "n": w.n})
    return pd.DataFrame(rows)


A = run(["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"], False, "RQ2_matched")
B = run(["quality_index_cbg_svc", "walkability_c", "out_park_ratio", "transit_freq",
         "park_area_acres", "popdens_k", "income_10k", "pct_nonwhite"], True, "RQ3_matched")
out = pd.concat([A, B], ignore_index=True)
out.to_csv(TABLES / "matched_universe_robustness.csv", index=False)

print("\n=== RQ2 (matched universe) ===")
for _, r in A.iterrows():
    if r.term in ("Income (per $10k)", "% non-white"):
        s = "***" if r.p < .001 else "**" if r.p < .01 else "*" if r.p < .05 else "ns"
        print(f"  {r.term:20s} {r.coef:+.4f} {s}")
print("=== RQ3 (matched universe), by |coef| ===")
bb = B.copy(); bb["a"] = bb.coef.abs()
for _, r in bb.sort_values("a", ascending=False).iterrows():
    s = "***" if r.p < .001 else "**" if r.p < .01 else "*" if r.p < .05 else "ns"
    print(f"  {r.term:20s} {r.coef:+.4f} {s}")

# --- self-check ---
assert len(pd.read_csv(TABLES / "matched_universe_robustness.csv")) > 0
print("\n[self-check] 29 matched-universe robustness OK")
