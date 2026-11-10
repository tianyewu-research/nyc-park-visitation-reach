"""
30_respec_comparison.py   (advisor 2026-07-26 respecification)
Produces the new-vs-old coefficient comparison the researcher will review before prose edits:

#1 single-stage: RQ2/RQ3 estimated as  log(use) ~ log(access) + covariates  (spatial error),
   which is equivalent to regressing the mismatch residual but avoids the generated-regressand
   and mismatch≈use critiques. Two-stage (mismatch residual) kept as a reference column.
#5 race: pct_nonwhite now from B03002 (non-Hispanic white); B02001 old definition as comparison.
#4 out-of-nbhd mobility dropped from RQ3 main (circular + near-zero variance); a with-it column
   is reported to show other coefficients are unchanged.

Output: outputs/tables/respec_comparison.csv  (+ console table)
"""
import geopandas as gpd, pandas as pd, numpy as np, scipy.stats as st
from libpysal.weights import Queen
import spreg, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); np.random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg")
g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0
TAB = ROOT / "outputs" / "tables"

nice = {"log_access": "Log access", "income_10k": "Income (/$10k)", "popdens_k": "Pop density (/1k)",
        "pct_nonwhite": "% non-white", "pct_nonwhite_b02001": "% non-white",
        "quality_index_cbg_svc": "Park quality", "walkability_c": "Walkability",
        "out_park_ratio": "Out-of-nbhd mobility", "transit_freq": "Transit", "park_area_acres": "Park area"}


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def spatial_error(depvar, preds, standardize, race_col=None):
    """Fit ML spatial error; return {label: (coef, p)} for preds (race_col swaps the race var)."""
    pl = [race_col if p == "pct_nonwhite" else p for p in preds]
    d = g.dropna(subset=[depvar] + pl + ["borough"]).reset_index(drop=True).copy()
    w = Queen.from_dataframe(d, use_index=False)
    if w.islands:
        d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
        w = Queen.from_dataframe(d, use_index=False)
    w.transform = "r"
    cols = []
    for c in pl:
        if standardize:
            d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std(); cols.append(c + "_z")
        else:
            cols.append(c)
    boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
    X = pd.concat([d[cols], boro], axis=1)
    sp = spreg.ML_Error(d[depvar].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
    b = sp.betas.flatten(); se = np.sqrt(np.diag(sp.vm)); names = ["CONST"] + list(X.columns) + ["lambda"]
    out = {}
    for i, nm in enumerate(names):
        base = nm.replace("_z", "")
        if base in nice:
            p = 2 * (1 - st.norm.cdf(abs(b[i] / se[i])))
            out[nice[base]] = (round(b[i], 4), round(p, 4))
    out["_n"] = w.n
    return out


rows = []

def add(model, res):
    for k, v in res.items():
        if k == "_n":
            continue
        rows.append({"model": model, "term": k, "coef": v[0], "p": v[1], "sig": stars(v[1]), "n": res["_n"]})

# ---------- RQ2 ----------
rq2 = ["log_access", "income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]
add("RQ2 single-stage (NEW race B03002)", spatial_error("log_use_c", rq2, False, "pct_nonwhite"))
add("RQ2 single-stage (OLD race B02001)", spatial_error("log_use_c", rq2, False, "pct_nonwhite_b02001"))
add("RQ2 two-stage residual (ref, NEW race)", spatial_error("mismatch_c", ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"], False, "pct_nonwhite"))

# ---------- RQ3 ----------
rq3_main = ["log_access", "quality_index_cbg_svc", "walkability_c", "transit_freq",
            "park_area_acres", "popdens_k", "income_10k", "pct_nonwhite"]           # out-nbhd DROPPED (#4)
rq3_with = rq3_main + ["out_park_ratio"]
add("RQ3 single-stage MAIN (no out-nbhd, NEW race)", spatial_error("log_use_c", rq3_main, True, "pct_nonwhite"))
add("RQ3 single-stage +out-nbhd (appendix)", spatial_error("log_use_c", rq3_with, True, "pct_nonwhite"))
add("RQ3 single-stage (OLD race)", spatial_error("log_use_c", rq3_main, True, "pct_nonwhite_b02001"))

df = pd.DataFrame(rows)
df.to_csv(TAB / "respec_comparison.csv", index=False)

# pretty print
for mdl in df["model"].unique():
    print(f"\n=== {mdl} ===")
    for _, r in df[df.model == mdl].iterrows():
        print(f"  {r['term']:22s} {r['coef']:+.4f} {r['sig']:>3s}   (n={r['n']})")

print("\n[self-check] 30 respec comparison OK")
assert len(df) > 0
