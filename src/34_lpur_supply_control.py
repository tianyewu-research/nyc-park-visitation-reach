"""
34_lpur_supply_control.py   (advisor 2026-07-29: same-scale supply control + bounded-outcome +
low-volume robustness)
LPUR is defined at 2 km, but the RQ3 supply controls (log E2SFCA access at 6 km, park area within
the CBG) are at other scales. This computes a SAME-SCALE supply measure — total park land within
2 km of the block-group centroid — and re-estimates RQ2/RQ3 controlling for it, to test whether
the racial gradient in LPUR is just same-scale supply geometry. Also:
  - logit-transformed LPUR (bounded [0,1] outcome robustness)
  - minimum weekly-visitor threshold (low-volume-CBG noise) + count of LPUR at 0/1
Output: data/interim/park_area_2km.csv + outputs/tables/lpur_supply_control.csv
"""
import geopandas as gpd, pandas as pd, numpy as np, scipy.stats as st
from libpysal.weights import Queen
import spreg, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); np.random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"; INT = ROOT / "data" / "interim"; TAB = ROOT / "outputs" / "tables"
M2FT = 1 / 0.3048

# ---- same-scale supply: park land (acres) within 2 km of CBG centroid ----
cbg = gpd.read_file(INT / "cbg_nyc.gpkg").to_crs("EPSG:2263")
parks = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623").to_crs("EPSG:2263")
parks = parks[parks["acres"] >= 0.1].reset_index(drop=True)
parks_u = parks.geometry.unary_union
from shapely.strtree import STRtree
tree = STRtree(list(parks_u.geoms) if hasattr(parks_u, "geoms") else [parks_u])
cent = cbg.geometry.centroid
rows = []
SQFT_PER_ACRE = 43560.0
for i in range(len(cbg)):
    buf = cent.iloc[i].buffer(2000 * M2FT)
    idx = tree.query(buf)
    a = 0.0
    for gi in idx:
        g = tree.geometries[gi] if hasattr(tree, "geometries") else list(parks_u.geoms)[gi]
        a += buf.intersection(g).area
    rows.append({"GEOID": cbg["GEOID"].iloc[i], "park_area_2km": round(a / SQFT_PER_ACRE, 3)})
pa = pd.DataFrame(rows); pa.to_csv(INT / "park_area_2km.csv", index=False)
print(f"park_area_2km: mean={pa.park_area_2km.mean():.1f} ac, median={pa.park_area_2km.median():.1f}, "
      f"corr note computed")

# ---- assemble ----
g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg").merge(pa, on="GEOID", how="left")
if "total_visits" not in g.columns:
    lr = pd.read_csv(INT / "local_ratio_cbg.csv", dtype={"GEOID": str})[["GEOID", "total_visits"]]
    g = g.merge(lr, on="GEOID", how="left")
g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0
g["park2k_z"] = (g["park_area_2km"] - g["park_area_2km"].mean()) / g["park_area_2km"].std()
# logit LPUR (clip to avoid inf at 0/1)
lp = g["lpur_2.0km"].clip(0.005, 0.995)
g["logit_lpur"] = np.log(lp / (1 - lp))
print(f"LPUR exactly 0: {(g['lpur_2.0km']==0).sum()} | exactly 1: {(g['lpur_2.0km']==1).sum()} "
      f"(of {g['lpur_2.0km'].notna().sum()} with LPUR)")
print(f"among LPUR=0/1: median weekly visitors = "
      f"{g.loc[g['lpur_2.0km'].isin([0,1]),'total_visits'].median()/52:.1f} vs overall "
      f"{g['total_visits'].median()/52:.1f}")


def rq(dep, preds, sub=None, label=""):
    d = g if sub is None else g[sub]
    d = d.dropna(subset=[dep] + preds + ["borough"]).reset_index(drop=True).copy()
    w = Queen.from_dataframe(d, use_index=False)
    if w.islands:
        d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
        w = Queen.from_dataframe(d, use_index=False)
    w.transform = "r"
    boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
    X = pd.concat([d[preds], boro], axis=1)
    sp = spreg.ML_Error(d[dep].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
    b = sp.betas.flatten(); se = np.sqrt(np.diag(sp.vm)); nm = ["C"] + list(X.columns) + ["l"]
    res = {"label": label, "n": w.n}
    for want in ["pct_nonwhite", "income_10k", "park_area_2km", "park2k_z"]:
        if want in nm:
            i = nm.index(want); p = 2 * (1 - st.norm.cdf(abs(b[i] / se[i])))
            res[want] = (round(b[i], 4), round(p, 4))
    return res


base = ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]
out = []
out.append(rq("lpur_2.0km", base, label="RQ2 baseline (no 2km supply)"))
out.append(rq("lpur_2.0km", base + ["park_area_2km"], label="RQ2 + 2km park-area control"))
out.append(rq("logit_lpur", base + ["park_area_2km"], label="logit(LPUR) + 2km control"))
for thr in [5, 20]:
    out.append(rq("lpur_2.0km", base + ["park_area_2km"],
                  sub=(g["total_visits"] / 52 >= thr), label=f"weekly visitors >= {thr}"))

res = pd.DataFrame([{**{k: v for k, v in r.items() if k in ("label", "n")},
                     "nonwhite": r.get("pct_nonwhite"), "income": r.get("income_10k"),
                     "park2km": r.get("park_area_2km")} for r in out])
res.to_csv(TAB / "lpur_supply_control.csv", index=False)


def fmt(t):
    if not t:
        return "—"
    c, p = t; s = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
    return f"{c:+.4f}{s}"


print("\n=== LPUR RQ2 under same-scale supply control / logit / volume thresholds ===")
for r in out:
    print(f"  {r['label']:34s} nonwhite {fmt(r.get('pct_nonwhite')):>12s} | "
          f"income {fmt(r.get('income_10k')):>10s} | park2km {fmt(r.get('park_area_2km')):>10s} (n={r['n']})")

assert (TAB / "lpur_supply_control.csv").exists()
print("[self-check] 34 supply control OK")
