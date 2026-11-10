"""
31_lpur_framework.py   (new framework results, 2026-07-28)
Local Park-Use Ratio (LPUR, 2 km) as the primary outcome, replacing the access-use mismatch
residual. Outputs:
  RQ1  Global Moran's I + LISA on LPUR
  RQ2  spatial-error model of LPUR on SES, with a racial decomposition (all groups vs NH-white
       reference, plus one-at-a-time) and a companion model on mean trip distance
  RQ3  spatial-error model of LPUR on urban form + access (covariate) + park attributes
  SPEC income coefficient across the 0.5-3 km radius grid (specification curve)
Tables -> outputs/tables/lpur_*.csv ; figure -> outputs/figures/lpur_spec_curve.png
"""
import geopandas as gpd, pandas as pd, numpy as np, scipy.stats as st
import matplotlib.pyplot as plt
from libpysal.weights import Queen, KNN
from esda.moran import Moran, Moran_Local
import spreg, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); np.random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg")
TAB = ROOT / "outputs" / "tables"; FIG = ROOT / "outputs" / "figures"
g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0
g["log_access"] = g["log_access"]


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def build(dep, preds, std):
    d = g.dropna(subset=[dep] + preds + ["borough"]).reset_index(drop=True).copy()
    w = Queen.from_dataframe(d, use_index=False)
    if w.islands:
        d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
        w = Queen.from_dataframe(d, use_index=False)
    w.transform = "r"
    cols = []
    for c in preds:
        if std:
            d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std(); cols.append(c + "_z")
        else:
            cols.append(c)
    boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
    X = pd.concat([d[cols], boro], axis=1)
    return d, w, X, cols


def sperr(dep, preds, std=False, label=""):
    d, w, X, cols = build(dep, preds, std)
    sp = spreg.ML_Error(d[dep].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
    b = sp.betas.flatten(); se = np.sqrt(np.diag(sp.vm)); nm = ["C"] + list(X.columns) + ["lam"]
    res = {}
    for i, n in enumerate(nm):
        base = n.replace("_z", "")
        if base in preds:
            p = 2 * (1 - st.norm.cdf(abs(b[i] / se[i])))
            res[base] = (round(b[i], 4), round(p, 4))
    if label:
        print(f"\n=== {label} (n={w.n}, pr2={sp.pr2:.3f}) ===")
        for k, (c, p) in res.items():
            print(f"  {k:20s} {c:+.4f} {stars(p)}")
    return res, w.n

# ---------- RQ1 ----------
d1 = g.dropna(subset=["lpur_2.0km"]).reset_index(drop=True)
wq = Queen.from_dataframe(d1, use_index=False); wq.transform = "r"
y = d1["lpur_2.0km"].values
mi = Moran(y, wq, permutations=999)
wk = KNN.from_dataframe(d1, k=6); wk.transform = "r"
mik = Moran(y, wk, permutations=999)
print(f"=== RQ1 LPUR spatial clustering ===\n  Moran's I Queen={mi.I:.3f} (p={mi.p_sim:.3f}) | kNN={mik.I:.3f} (p={mik.p_sim:.3f})")
lm = Moran_Local(y, wq, permutations=999); sig = lm.p_sim < 0.05
lab = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
cl = pd.Series(["ns"] * len(d1))
for c, nmc in lab.items():
    cl[sig & (lm.q == c)] = nmc
print("  LISA:", cl.value_counts().to_dict())
pd.DataFrame([{"weights": "Queen", "I": mi.I, "p": mi.p_sim},
             {"weights": "kNN6", "I": mik.I, "p": mik.p_sim}]).to_csv(TAB / "lpur_morans.csv", index=False)

# ---------- RQ2: SES + racial decomposition ----------
sperr("lpur_2.0km", ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"], False, "RQ2 LPUR ~ income + non-white")
# racial decomposition: all groups together, NH-white omitted as reference
sperr("lpur_2.0km", ["income_10k", "pct_black", "pct_hispanic", "pct_asian", "popdens_k", "park_area_acres"],
      False, "RQ2 LPUR racial decomposition (ref = NH-white)")
# one-at-a-time
rows = []
for grp in ["pct_black", "pct_hispanic", "pct_asian", "pct_white_nh", "pct_nonwhite"]:
    res, n = sperr("lpur_2.0km", ["income_10k", grp, "popdens_k", "park_area_acres"], False)
    rows.append({"group": grp, "coef": res[grp][0], "p": res[grp][1],
                 "income_coef": res["income_10k"][0], "n": n})
pd.DataFrame(rows).to_csv(TAB / "lpur_racial_decomp.csv", index=False)
print("\n=== RQ2 one-at-a-time racial groups (LPUR 2km) ===")
for r in rows:
    print(f"  {r['group']:14s} {r['coef']:+.4f} {stars(r['p'])}   (income {r['income_coef']:+.4f})")

# companion: mean trip distance
sperr("mean_trip_km", ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"], False,
      "Companion: mean trip distance ~ income + non-white")

# ---------- RQ3 ----------
sperr("lpur_2.0km", ["log_access", "quality_index_cbg_svc", "walkability_c", "transit_freq",
                     "park_area_acres", "popdens_k", "income_10k", "pct_nonwhite"], True, "RQ3 LPUR correlates (standardized)")

# ---------- SPEC CURVE: income coef vs radius ----------
grid = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
sc = []
for r in grid:
    dep = f"lpur_{r}km"
    d, w, X, cols = build(dep, ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"], False)
    sp = spreg.ML_Error(d[dep].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
    b = sp.betas.flatten(); se = np.sqrt(np.diag(sp.vm)); nm = ["C"] + list(X.columns) + ["lam"]
    i = nm.index("income_10k")
    sc.append({"radius_km": r, "income_coef": b[i], "se": se[i],
               "lo": b[i] - 1.96 * se[i], "hi": b[i] + 1.96 * se[i]})
scd = pd.DataFrame(sc); scd.to_csv(TAB / "lpur_spec_curve.csv", index=False)
print("\n=== SPEC CURVE: income coef on LPUR across radius ===")
print(scd[["radius_km", "income_coef", "lo", "hi"]].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color="red", lw=0.8, ls="--")
ax.fill_between(scd.radius_km, scd.lo, scd.hi, alpha=0.2, color="#1f77b4")
ax.plot(scd.radius_km, scd.income_coef, "-o", color="#1f77b4")
ax.axvline(2.0, color="grey", lw=0.8, ls=":")
ax.set_xlabel("Neighborhood radius (km)")
ax.set_ylabel("Income coefficient on LPUR (per $10k)")
ax.set_title("Specification curve: income gradient in local park-use ratio\nacross neighborhood radius (spatial error model, 95% CI)")
plt.tight_layout(); plt.savefig(FIG / "lpur_spec_curve.png", dpi=200, bbox_inches="tight"); plt.close()
print(f"\nSaved figure {FIG/'lpur_spec_curve.png'}")

# --- self-check ---
assert (TAB / "lpur_spec_curve.csv").exists() and (TAB / "lpur_racial_decomp.csv").exists()
print("[self-check] 31 LPUR framework OK")
