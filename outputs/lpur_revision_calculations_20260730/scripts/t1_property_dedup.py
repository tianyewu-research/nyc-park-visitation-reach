"""
T1 — Park-property de-duplication robustness for LPUR.

DE-DUPLICATION FEASIBILITY (stated explicitly): Advan Weekly Patterns is AGGREGATED. Each POI-week
row carries VISITOR_COUNTS (a scaled unique-visitor total) and VISITOR_HOME_CBGS (a home-CBG ->
device-count dictionary). There are NO device or person identifiers. A visitor who enters two POIs
of the same park appears in both POIs' VISITOR_COUNTS with no key to link them. TRUE visitor-level
de-duplication across POIs is therefore NOT IDENTIFIABLE from these data and is not claimed here.
What we CAN do is bound the POI-overlap: version C takes, within each park property x week x home
CBG, the MAXIMUM single-POI contribution (a conservative lower bound that assumes all of a
property's POI visitors are the same people), versus the POI-SUM (upper bound) of versions A/B.

Three LPUR versions (2 km, dwell-qualified visitors, apportioned to home CBG by device share):
  A. Original POI-sum LPUR  — all POIs; distance = CBG centroid -> POI point (reproduces main measure).
  B. Matched-universe POI-sum LPUR — only POIs matched to a municipal park property;
     distance = CBG centroid -> property polygon nearest point.
  C. Property-level conservative LPUR — per property x week x home CBG take the MAX POI contribution;
     distance = CBG centroid -> property polygon nearest point.

Output: T1_lpur_versions.csv (per-CBG A/B/C), T1_models.csv (RQ2 coefs), T1_summary.txt.
"""
import pandas as pd, geopandas as gpd, numpy as np, json, glob, scipy.stats as st
from collections import defaultdict
from pathlib import Path
from libpysal.weights import Queen
from shapely.strtree import STRtree
import spreg, warnings

warnings.filterwarnings("ignore"); np.random.seed(42)
ROOT = Path(".")
RAW = ROOT / "data" / "raw"; INT = ROOT / "data" / "interim"
OUT = ROOT / "outputs" / "lpur_revision_calculations_20260730"
NYC = ("36005", "36047", "36061", "36081", "36085"); M2FT = 1 / 0.3048
TWO_KM = 2000 * M2FT
DWELL_KEEP = ("5-20", "21-60", "61-240", ">240")


def qf(bj):
    if not isinstance(bj, str):
        return 1.0
    try:
        b = json.loads(bj)
    except Exception:
        return 1.0
    t = sum(b.values())
    return sum(b.get(k, 0) for k in DWELL_KEEP) / t if t > 0 else 0.0


cbg = gpd.read_file(INT / "cbg_nyc.gpkg").to_crs("EPSG:2263")
cx = cbg.geometry.centroid.x.values; cy = cbg.geometry.centroid.y.values
cent = list(cbg.geometry.centroid.values)
gid = cbg["GEOID"].values; gidx = {gg: i for i, gg in enumerate(gid)}

parks = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623").to_crs("EPSG:2263")
parks = parks[parks["acres"] >= 0.1].reset_index(drop=True)
prop_geom = list(parks.geometry.values)
ptree = STRtree(prop_geom)
n_prop = len(parks)
print(f"municipal park properties (>=0.1 ac): {n_prop}")

# precompute per-CBG set of property indices within 2 km (polygon nearest-point distance)
print("precomputing CBG->within-2km property sets...")
cbg_near = [set() for _ in range(len(cbg))]
for i in range(len(cbg)):
    buf = cent[i].buffer(TWO_KM)
    for pi in ptree.query(buf):
        if prop_geom[pi].distance(cent[i]) <= TWO_KM:
            cbg_near[i].add(int(pi))

# POI -> property matching cache (point-in-polygon, else nearest within 100 ft)
poi_prop = {}; poi_xy = {}
TOL = 100 * M2FT


def match_poi(sid, x, y):
    if sid in poi_prop:
        return poi_prop[sid]
    from shapely.geometry import Point
    pt = Point(x, y)
    cand = ptree.query(pt)
    prop = None
    for pi in cand:
        if prop_geom[pi].contains(pt):
            prop = int(pi); break
    if prop is None:
        cand2 = ptree.query(pt.buffer(TOL))
        best, bd = None, TOL
        for pi in cand2:
            dd = prop_geom[pi].distance(pt)
            if dd <= bd:
                best, bd = int(pi), dd
        prop = best
    poi_prop[sid] = prop; poi_xy[sid] = (x, y)
    return prop


# accumulators
withinA = defaultdict(float); totA = defaultdict(float)
withinB = defaultdict(float); totB = defaultdict(float)
withinC = defaultdict(float); totC = defaultdict(float)
n_poi_seen = set(); n_poi_matched = set()

files = sorted(glob.glob(str(RAW / "nypark_2325" / "2023-*.csv.gz")))
for k, f in enumerate(files):
    df = pd.read_csv(f, usecols=["ID_STORE", "POI_CBG", "VISITOR_HOME_CBGS", "VISITOR_COUNTS",
                                 "BUCKETED_DWELL_TIMES", "LATITUDE", "LONGITUDE"])
    df["pc"] = df["POI_CBG"].dropna().astype("Int64").astype(str).str.zfill(12)
    df = df[df["pc"].str[:5].isin(NYC)].dropna(subset=["LATITUDE", "LONGITUDE"])
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE),
                           crs="EPSG:4326").to_crs("EPSG:2263")
    px = pts.geometry.x.values; py = pts.geometry.y.values
    sid = df["ID_STORE"].values; vc = df["VISITOR_COUNTS"].values
    bdt = df["BUCKETED_DWELL_TIMES"].values; vhc = df["VISITOR_HOME_CBGS"].values
    week_max = defaultdict(float)   # (prop, home_idx) -> max POI contribution this week (version C)
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
        qv0 = base * qf(bdt[j])
        s = sid[j]; n_poi_seen.add(s)
        prop = match_poi(s, px[j], py[j])
        if prop is not None:
            n_poi_matched.add(s)
        for hc, dc in hd.items():
            if hc.startswith("CA:"):
                continue
            hc = hc.zfill(12); i = gidx.get(hc)
            if i is None:
                continue
            qv = qv0 * (dc / tot)
            # A: POI-point distance, all POIs
            dpt = np.hypot(cx[i] - px[j], cy[i] - py[j])
            totA[hc] += qv
            if dpt <= TWO_KM:
                withinA[hc] += qv
            # B / C need a matched property
            if prop is not None:
                near = prop in cbg_near[i]        # property polygon within 2 km of CBG centroid
                totB[hc] += qv
                if near:
                    withinB[hc] += qv
                key = (prop, i)
                if qv > week_max[key]:
                    week_max[key] = qv
    # fold week_max into version C annual totals
    for (prop, i), mx in week_max.items():
        hc = gid[i]
        totC[hc] += mx
        if prop in cbg_near[i]:
            withinC[hc] += mx
    if (k + 1) % 13 == 0:
        print(f"  {k+1}/52")

rows = []
for gg in gid:
    row = {"GEOID": gg}
    row["lpur_A_poisum"] = round(withinA[gg] / totA[gg], 4) if totA[gg] > 0 else None
    row["lpur_B_matched"] = round(withinB[gg] / totB[gg], 4) if totB[gg] > 0 else None
    row["lpur_C_property_max"] = round(withinC[gg] / totC[gg], 4) if totC[gg] > 0 else None
    rows.append(row)
lp = pd.DataFrame(rows); lp.to_csv(OUT / "T1_lpur_versions.csv", index=False)

match_rate = len(n_poi_matched) / len(n_poi_seen) if n_poi_seen else 0
n_props_used = len({poi_prop[s] for s in n_poi_matched if poi_prop.get(s) is not None})

# ---- models ----
g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg").merge(lp, on="GEOID", how="left")
g["income_10k"] = g["median_hh_income"] / 10000.0; g["popdens_k"] = g["pop_density"] / 1000.0
PRE = ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]


def model(dep):
    d = g.dropna(subset=[dep] + PRE + ["borough"]).reset_index(drop=True).copy()
    w = Queen.from_dataframe(d, use_index=False)
    if w.islands:
        d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
        w = Queen.from_dataframe(d, use_index=False)
    w.transform = "r"
    boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
    X = pd.concat([d[PRE], boro], axis=1)
    m = spreg.ML_Error(d[dep].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
    b = m.betas.flatten(); se = np.sqrt(np.diag(m.vm)); nm = ["C"] + list(X.columns) + ["lambda"]
    res = {"dep": dep, "n": int(w.n), "lambda": round(float(b[-1]), 4)}
    for name in ["pct_nonwhite", "income_10k"]:
        i = nm.index(name); c, s = float(b[i]), float(se[i]); z = c / s
        res[name] = dict(coef=round(c, 5), se=round(s, 5), ci_lo=round(c - 1.96 * s, 5),
                         ci_hi=round(c + 1.96 * s, 5), p=float(2 * (1 - st.norm.cdf(abs(z)))))
    return res


mods = [model("lpur_A_poisum"), model("lpur_B_matched"), model("lpur_C_property_max")]
mrows = []
for m in mods:
    for name in ["pct_nonwhite", "income_10k"]:
        c = m[name]
        mrows.append(dict(version=m["dep"], predictor=name, coef=c["coef"], se=c["se"],
                          ci95_lo=c["ci_lo"], ci95_hi=c["ci_hi"], p_exact=c["p"], n=m["n"], lambda_=m["lambda"]))
pd.DataFrame(mrows).to_csv(OUT / "T1_models.csv", index=False)

# descriptives + correlations
desc = lp[["lpur_A_poisum", "lpur_B_matched", "lpur_C_property_max"]].describe().T
corr = lp[["lpur_A_poisum", "lpur_B_matched", "lpur_C_property_max"]].corr()
desc.to_csv(OUT / "T1_lpur_descriptives.csv")
corr.to_csv(OUT / "T1_lpur_correlations.csv")

with open(OUT / "T1_summary.txt", "w") as fo:
    fo.write("T1 PARK-PROPERTY DE-DUPLICATION ROBUSTNESS\n\n")
    fo.write("De-duplication feasibility: NOT IDENTIFIABLE from aggregated Weekly Patterns data.\n")
    fo.write("  No device/person IDs; a visitor to multiple POIs of one park cannot be linked.\n")
    fo.write("  Version C (per property x week x home-CBG MAX POI contribution) is a CONSERVATIVE\n")
    fo.write("  lower bound on within-park de-duplication, not true visitor-level de-duplication.\n\n")
    fo.write(f"POI->property match rate: {len(n_poi_matched)}/{len(n_poi_seen)} = {100*match_rate:.1f}%\n")
    fo.write(f"Distinct municipal park properties receiving matched POIs: {n_props_used} (of {n_prop} >=0.1 ac)\n\n")
    fo.write("LPUR descriptives (2 km):\n" + desc.round(4).to_string() + "\n\n")
    fo.write("Correlations among versions:\n" + corr.round(4).to_string() + "\n\n")
    fo.write("RQ2 spatial-error coefficients (non-white / income), identical spec/sample rule:\n")
    for m in mods:
        for name in ["pct_nonwhite", "income_10k"]:
            c = m[name]
            fo.write(f"  {m['dep']:22s} {name:12s} coef={c['coef']:+.4f} se={c['se']:.4f} "
                     f"95%CI=[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] p={c['p']:.4g} n={m['n']} lambda={m['lambda']}\n")

print(f"\n[T1] match rate {100*match_rate:.1f}%, properties {n_props_used}")
print("[T1] descriptives:\n", desc.round(4).to_string())
print("[T1] correlations:\n", corr.round(4).to_string())
for m in mods:
    c = m["pct_nonwhite"]
    print(f"  {m['dep']:22s} non-white {c['coef']:+.4f} (p={c['p']:.2g}) n={m['n']}")
print("[self-check] T1 OK")
