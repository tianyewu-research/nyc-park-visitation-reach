"""
27_max_geofence_robustness.py
Robustness: re-run RQ2/RQ3 on the single-busiest-geofence use measure (14b) to check the
sum-per-park main measure is not driven by within-park geofence overlap.

Input:  data/processed/analysis_cbg_final.gpkg + data/interim/use_cbg_max.csv
Output: outputs/tables/max_geofence_robustness.csv
"""
import geopandas as gpd, pandas as pd, numpy as np, scipy.stats as st
from libpysal.weights import Queen
import spreg, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); np.random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"; INTERIM = ROOT / "data" / "interim"; TABLES = ROOT / "outputs" / "tables"

g = gpd.read_file(PROC / "analysis_cbg_final.gpkg")
vmax = pd.read_csv(INTERIM / "use_cbg_max.csv", dtype={"GEOID": str})
g = g.merge(vmax, on="GEOID", how="left")

# max-variant use -> winsorized log -> first-stage residual = mismatch_max
g["use_max"] = g["V_c_max"] / g["total_pop"]
g.loc[g["total_pop"] == 0, "use_max"] = np.nan
lo, hi = g["use_max"].quantile(0.01), g["use_max"].quantile(0.99)
g["log_use_max"] = np.log1p(g["use_max"].clip(lo, hi))
fs = g[["log_use_max", "log_access"]].dropna()
import statsmodels.api as sm
b = sm.OLS(fs["log_use_max"], sm.add_constant(fs["log_access"])).fit()
g.loc[fs.index, "mismatch_max"] = b.resid
print(f"first-stage (max): R2={b.rsquared:.3f} beta={b.params.iloc[1]:.3f}")

g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0
nice = {"income_10k": "Income (per $10k)", "pct_nonwhite": "% non-white",
        "quality_index_cbg_svc": "Park quality", "walkability_c": "Walkability",
        "out_park_ratio": "Out-of-nbhd mobility", "transit_freq": "Transit access",
        "park_area_acres": "Park area", "popdens_k": "Pop. density"}


def run(preds, standardize, label):
    d = g.dropna(subset=["mismatch_max"] + preds + ["borough"]).reset_index(drop=True).copy()
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
    sp = spreg.ML_Error(d["mismatch_max"].values.reshape(-1, 1), X.values, w=w,
                        name_x=list(X.columns))
    bt = sp.betas.flatten(); se = np.sqrt(np.diag(sp.vm)); names = ["CONSTANT"] + list(X.columns) + ["lambda"]
    rows = []
    for i, nm in enumerate(names):
        base = nm.replace("_z", "")
        if base in nice:
            p = 2 * (1 - st.norm.cdf(abs(bt[i] / se[i])))
            rows.append({"model": label, "term": nice[base], "coef": round(bt[i], 4), "p": round(p, 4)})
    return pd.DataFrame(rows)


A = run(["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"], False, "RQ2_max")
B = run(["quality_index_cbg_svc", "walkability_c", "out_park_ratio", "transit_freq",
         "park_area_acres", "popdens_k", "income_10k", "pct_nonwhite"], True, "RQ3_max")
out = pd.concat([A, B], ignore_index=True)
out.to_csv(TABLES / "max_geofence_robustness.csv", index=False)

print("\n=== RQ2 (max geofence) ===")
for _, r in A.iterrows():
    if r.term in ("Income (per $10k)", "% non-white"):
        s = "***" if r.p < .001 else "**" if r.p < .01 else "*" if r.p < .05 else "ns"
        print(f"  {r.term:20s} {r.coef:+.4f} {s}")
print("=== RQ3 (max geofence), by |coef| ===")
bb = B.copy(); bb["a"] = bb.coef.abs()
for _, r in bb.sort_values("a", ascending=False).iterrows():
    s = "***" if r.p < .001 else "**" if r.p < .01 else "*" if r.p < .05 else "ns"
    print(f"  {r.term:20s} {r.coef:+.4f} {s}")

# --- self-check ---
assert len(pd.read_csv(TABLES / "max_geofence_robustness.csv")) > 0
print("\n[self-check] 27 max-geofence robustness OK")
