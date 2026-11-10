"""
33_lpur_robustness.py   (LPUR measure-construction robustness, 2026-07-29)
One 52-file pass recomputing the local park-use ratio (2 km) under alternative construction
choices, then re-estimating the RQ2 non-white coefficient (the key finding) under each:
  lpur_ge5      main measure (dwell >= 5 min)          -- reproduces 15b
  lpur_none     no dwell filter (all visit records)
  lpur_ge21     stricter dwell (>= 21 min)
  lpur_matched  dwell >= 5 min, restricted to POIs inside the 1,676 municipal park boundaries
Output: data/interim/lpur_robust.csv + outputs/tables/lpur_robust_coef.csv
"""
import pandas as pd, geopandas as gpd, numpy as np, json, glob, scipy.stats as st
from collections import defaultdict
from pathlib import Path
from libpysal.weights import Queen
import spreg, warnings

warnings.filterwarnings("ignore"); np.random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"; INT = ROOT / "data" / "interim"; TAB = ROOT / "outputs" / "tables"
NYC = ("36005", "36047", "36061", "36081", "36085"); M2FT = 1 / 0.3048
TWO_KM_FT = 2000 * M2FT
DWELL = {"none": ("<5", "5-20", "21-60", "61-240", ">240"),
         "ge5": ("5-20", "21-60", "61-240", ">240"),
         "ge21": ("21-60", "61-240", ">240")}


def qf(bj, keep):
    if not isinstance(bj, str):
        return 1.0
    try:
        b = json.loads(bj)
    except Exception:
        return 1.0
    t = sum(b.values())
    return sum(b.get(k, 0) for k in keep) / t if t > 0 else 0.0


cbg = gpd.read_file(INT / "cbg_nyc.gpkg")
cx = cbg.geometry.centroid.x.values; cy = cbg.geometry.centroid.y.values
gid = cbg["GEOID"].values; gidx = {g: i for i, g in enumerate(gid)}
# municipal park boundaries for matched-universe membership
parks = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623").to_crs("EPSG:2263")
parks = parks[parks["acres"] >= 0.1].reset_index(drop=True)
from shapely.strtree import STRtree
ptree = STRtree(parks.geometry.values)

within = {v: defaultdict(float) for v in DWELL}      # within-2km
total = {v: defaultdict(float) for v in DWELL}
within_m = defaultdict(float); total_m = defaultdict(float)   # matched (ge5, municipal only)
poi_matched = {}                                     # ID_STORE -> bool inside municipal boundary

files = sorted(glob.glob(str(RAW / "nypark_2325" / "2023-*.csv.gz")))
for k, f in enumerate(files):
    df = pd.read_csv(f, usecols=["ID_STORE", "POI_CBG", "VISITOR_HOME_CBGS", "VISITOR_COUNTS",
                                 "BUCKETED_DWELL_TIMES", "LATITUDE", "LONGITUDE"])
    df["pc"] = df["POI_CBG"].dropna().astype("Int64").astype(str).str.zfill(12)
    df = df[df["pc"].str[:5].isin(NYC)].dropna(subset=["LATITUDE", "LONGITUDE"])
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE),
                           crs="EPSG:4326").to_crs("EPSG:2263")
    px = pts.geometry.x.values; py = pts.geometry.y.values
    geoms = pts.geometry.values
    sid = df["ID_STORE"].values; vc = df["VISITOR_COUNTS"].values
    bdt = df["BUCKETED_DWELL_TIMES"].values; vhc = df["VISITOR_HOME_CBGS"].values
    for j in range(len(df)):
        v = vhc[j]
        if not isinstance(v, str):
            continue
        try:
            hd = json.loads(v)
        except Exception:
            continue
        tot = sum(hd.values())
        if tot <= 0:
            continue
        base = vc[j] if pd.notna(vc[j]) else 0
        if base <= 0:
            continue
        qvals = {vv: base * qf(bdt[j], keep) for vv, keep in DWELL.items()}
        s = sid[j]
        if s not in poi_matched:
            near = ptree.query_nearest(geoms[j], max_distance=100 * M2FT)
            poi_matched[s] = len(near) > 0
        matched = poi_matched[s]
        for hc, dc in hd.items():
            if hc.startswith("CA:"):
                continue
            hc = hc.zfill(12); i = gidx.get(hc)
            if i is None:
                continue
            sh = dc / tot
            d = np.hypot(cx[i] - px[j], cy[i] - py[j])
            near2 = d <= TWO_KM_FT
            for vv in DWELL:
                q = qvals[vv] * sh
                total[vv][hc] += q
                if near2:
                    within[vv][hc] += q
            qm = qvals["ge5"] * sh
            if matched:
                total_m[hc] += qm
                if near2:
                    within_m[hc] += qm
    if (k + 1) % 13 == 0:
        print(f"  {k+1}/52")

rows = []
for g in gid:
    row = {"GEOID": g}
    for vv in DWELL:
        t = total[vv][g]; row[f"lpur_{vv}"] = round(within[vv][g] / t, 4) if t > 0 else None
    tm = total_m[g]; row["lpur_matched"] = round(within_m[g] / tm, 4) if tm > 0 else None
    rows.append(row)
lp = pd.DataFrame(rows); lp.to_csv(INT / "lpur_robust.csv", index=False)

# re-estimate RQ2 non-white coefficient under each variant
g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg").merge(lp, on="GEOID", how="left")
g["income_10k"] = g["median_hh_income"] / 10000.0; g["popdens_k"] = g["pop_density"] / 1000.0
out = []
for col, lab in [("lpur_ge5", "dwell >=5 min (main)"), ("lpur_none", "no dwell filter"),
                 ("lpur_ge21", "dwell >=21 min"), ("lpur_matched", "municipal-park universe")]:
    d = g.dropna(subset=[col, "income_10k", "pct_nonwhite", "popdens_k", "park_area_acres", "borough"]).reset_index(drop=True)
    w = Queen.from_dataframe(d, use_index=False)
    if w.islands:
        d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
        w = Queen.from_dataframe(d, use_index=False)
    w.transform = "r"
    boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
    X = pd.concat([d[["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]], boro], axis=1)
    sp = spreg.ML_Error(d[col].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
    b = sp.betas.flatten(); se = np.sqrt(np.diag(sp.vm)); nm = ["C"] + list(X.columns) + ["l"]
    inw = nm.index("pct_nonwhite"); inc = nm.index("income_10k")
    pnw = 2 * (1 - st.norm.cdf(abs(b[inw] / se[inw]))); pin = 2 * (1 - st.norm.cdf(abs(b[inc] / se[inc])))
    out.append({"variant": lab, "nonwhite_coef": round(b[inw], 4), "nonwhite_p": round(pnw, 4),
                "income_coef": round(b[inc], 4), "income_p": round(pin, 4), "n": w.n})
res = pd.DataFrame(out); res.to_csv(TAB / "lpur_robust_coef.csv", index=False)
print("\n=== LPUR RQ2 coefficients under measure-construction variants ===")
for r in out:
    sn = "***" if r["nonwhite_p"] < .001 else "**" if r["nonwhite_p"] < .01 else "*" if r["nonwhite_p"] < .05 else "ns"
    si = "***" if r["income_p"] < .001 else "**" if r["income_p"] < .01 else "*" if r["income_p"] < .05 else "ns"
    print(f"  {r['variant']:26s} non-white {r['nonwhite_coef']:+.4f} {sn:3s} | income {r['income_coef']:+.4f} {si}")

assert (TAB / "lpur_robust_coef.csv").exists()
print("[self-check] 33 LPUR robustness OK")
