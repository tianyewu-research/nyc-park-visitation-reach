"""SECTION C+D build — LPUR A/B/C per CBG and property-level destination records, from the CANONICAL
crosswalk only (no independent re-matching).

Reads `poi_property_crosswalk_canonical_100ft.csv` (ID_STORE -> matched gispropnum). One pass over the
52 weekly files attributes dwell-qualified, home-apportioned visitor weight and accumulates:
  A. all-POI POI-sum LPUR (distance = home-CBG centroid -> POI point) — reproduction check only;
  B. matched-universe POI-sum LPUR (canonical matched POIs; "within 2 km" = matched property is within
     2 km of the home-CBG centroid by polygon nearest-point, per t1_property_dedup.py);
  C. property-level maximum LPUR (per property x week x home-CBG max POI contribution, summed over weeks);
  D. destination records = C's annual (property, home-CBG) weight; property distance = home-CBG centroid
     -> property REPRESENTATIVE POINT (the definition used by the prior destination analysis p5);
     non-local = that distance > 2.0 km.

No POI is re-matched here: matched_pid comes only from the crosswalk. Property geometry is used solely
to compute centroid-to-property distances (not to assign POIs to properties)."""
import sys, json, glob, numpy as np, pandas as pd, geopandas as gpd
from collections import defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, RAW, INT, OUT, NYC, M2FT, TWO_KM_FT, qfrac, sha256
from shapely.strtree import STRtree

np.random.seed(42)
files = sorted(glob.glob(str(RAW / "nypark_2325" / "2023-*.csv.gz")))

# ---- canonical crosswalk (matched POIs only) ----
xw = pd.read_csv(OUT / "poi_property_crosswalk_canonical_100ft.csv")
poi2pid = dict(zip(xw.loc[xw.match_status == "matched", "ID_STORE"],
                   xw.loc[xw.match_status == "matched", "matched_pid"].astype(str)))
print(f"crosswalk matched POIs: {len(poi2pid)}")

# ---- geography: CBG centroids; dissolved properties for distance only ----
cbg = gpd.read_file(INT / "cbg_nyc.gpkg").to_crs("EPSG:2263")
cx = cbg.geometry.centroid.x.values; cy = cbg.geometry.centroid.y.values
cent = list(cbg.geometry.centroid.values)
gid = cbg["GEOID"].astype(str).values; gidx = {g: i for i, g in enumerate(gid)}

parks = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623").to_crs("EPSG:2263")
parks = parks[parks["acres"] >= 0.1].copy(); parks["pid"] = parks["gispropnum"].astype(str)
prop = parks.dissolve(by="pid", aggfunc={"acres": "sum", "signname": "first", "borough": "first",
                                         "typecatego": "first", "waterfront": "max"}).reset_index()
pid_list = prop["pid"].values
pid_row = {p: i for i, p in enumerate(pid_list)}
prop_geom = list(prop.geometry.values)
rep = prop.geometry.representative_point()
prop_rx = rep.x.values; prop_ry = rep.y.values     # representative-point coords (destination distance)
BOROMAP = {"B": "Brooklyn", "X": "Bronx", "M": "Manhattan", "Q": "Queens", "R": "Staten Island"}
prop_attr = prop.assign(borough_name=prop["borough"].map(lambda b: BOROMAP.get(str(b), str(b))),
                        is_wf=(prop["waterfront"].fillna(0).astype(float) > 0))[
    ["pid", "signname", "borough_name", "acres", "typecatego", "is_wf"]].set_index("pid")

# per-CBG set of property-rows within 2 km (polygon nearest-point) — for LPUR B/C "within"
print("precomputing CBG -> within-2km property sets (nearest boundary) ...")
ptree = STRtree(prop_geom)
cbg_near = [set() for _ in range(len(cbg))]
for i in range(len(cbg)):
    buf = cent[i].buffer(TWO_KM_FT)
    for pj in ptree.query(buf):
        if prop_geom[pj].distance(cent[i]) <= TWO_KM_FT:
            cbg_near[i].add(int(pj))

# ---- one pass: attribute weight ----
withinA = defaultdict(float); totA = defaultdict(float)
withinB = defaultdict(float); totB = defaultdict(float)
annualC = defaultdict(float)                    # (pid_row, home_idx) -> annual summed weekly-max weight
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
    week_max = defaultdict(float)
    for j in range(len(df)):
        v = vhc[j]
        if not isinstance(v, str):
            continue
        try:
            hd = json.loads(v)
        except Exception:
            continue
        tot = sum(hd.values()); base = vc[j] if pd.notna(vc[j]) else 0
        if tot <= 0 or base <= 0:
            continue
        qv0 = base * qfrac(bdt[j])
        if qv0 <= 0:
            continue
        pid = poi2pid.get(sid[j]); prow = pid_row.get(str(pid)) if pid is not None else None
        for hc, dc in hd.items():
            if hc.startswith("CA:"):
                continue
            hc = hc.zfill(12); i = gidx.get(hc)
            if i is None:
                continue
            qv = qv0 * (dc / tot)
            # A: all POIs, centroid->POI-point distance
            dpt = np.hypot(cx[i] - px[j], cy[i] - py[j]); totA[hc] += qv
            if dpt <= TWO_KM_FT:
                withinA[hc] += qv
            # B/C: matched properties only
            if prow is not None:
                totB[hc] += qv
                if prow in cbg_near[i]:
                    withinB[hc] += qv
                key = (prow, i)
                if qv > week_max[key]:
                    week_max[key] = qv
    for (prow, i), mx in week_max.items():
        annualC[(prow, i)] += mx
    if (k + 1) % 13 == 0:
        print(f"  pass {k+1}/52")

# ---- per-CBG LPUR A/B/C ----
withinC = defaultdict(float); totC = defaultdict(float)
for (prow, i), wgt in annualC.items():
    hc = gid[i]; totC[hc] += wgt
    if prow in cbg_near[i]:
        withinC[hc] += wgt
rows = []
for g in gid:
    rows.append(dict(GEOID=g,
                     lpur_A_poisum=round(withinA[g] / totA[g], 6) if totA[g] > 0 else np.nan,
                     lpur_B_matched=round(withinB[g] / totB[g], 6) if totB[g] > 0 else np.nan,
                     lpur_C_property_max=round(withinC[g] / totC[g], 6) if totC[g] > 0 else np.nan,
                     totA=round(totA[g], 3), totB=round(totB[g], 3), totC=round(totC[g], 3)))
lp = pd.DataFrame(rows)
lp.to_csv(OUT / "lpur_representation_canonical_100ft_by_cbg.csv", index=False)

# ---- destination records (property, home-CBG) with representative-point distance ----
recs = []
for (prow, i), wgt in annualC.items():
    hc = gid[i]; pid = pid_list[prow]
    dist_km = np.hypot(cx[i] - prop_rx[prow], cy[i] - prop_ry[prow]) / M2FT / 1000.0
    at = prop_attr.loc[pid]
    recs.append(dict(pid=pid, home_cbg=hc, attributed_weight=wgt,
                     dist_km=round(dist_km, 4), non_local=dist_km > 2.0,
                     dest_name=at["signname"], dest_borough=at["borough_name"],
                     dest_acres=float(at["acres"]), dest_type=at["typecatego"],
                     is_waterfront=bool(at["is_wf"])))
R = pd.DataFrame(recs)
R.to_csv(OUT / "destination_records_property_level_canonical_100ft.csv", index=False)

# update provenance manifest with new checksums
man = json.loads((OUT / "provenance_manifest.json").read_text())
man["checksums"]["lpur_representation_canonical_100ft_by_cbg.csv"] = sha256(OUT / "lpur_representation_canonical_100ft_by_cbg.csv")
man["checksums"]["destination_records_property_level_canonical_100ft.csv"] = sha256(OUT / "destination_records_property_level_canonical_100ft.csv")
man["destination_records"] = dict(n_rows=len(R), attributed_total=round(float(R.attributed_weight.sum()), 1),
                                  pct_nonlocal_cells=round(100 * float(R.non_local.mean()), 2),
                                  pct_nonlocal_weight=round(100 * float(R.loc[R.non_local, "attributed_weight"].sum() /
                                                                        R.attributed_weight.sum()), 2))
(OUT / "provenance_manifest.json").write_text(json.dumps(man, indent=2))

# quick reproduction check vs frozen primary lpur_2.0km
gpk = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg")[["GEOID", "lpur_2.0km"]]
gpk["GEOID"] = gpk["GEOID"].astype(str)
chk = lp.merge(gpk, on="GEOID")
corrA = chk[["lpur_A_poisum", "lpur_2.0km"]].dropna().corr().iloc[0, 1]
print("=== s1 done ===")
print(f"LPUR A vs frozen lpur_2.0km corr = {corrA:.4f} (A reproduces the all-POI primary construction)")
print(f"records: {len(R):,} (pid,home) cells; attributed {R.attributed_weight.sum():,.0f}; "
      f"non-local cells {100*R.non_local.mean():.1f}%, non-local weight "
      f"{100*R.loc[R.non_local,'attributed_weight'].sum()/R.attributed_weight.sum():.1f}%")
print(lp[["lpur_A_poisum", "lpur_B_matched", "lpur_C_property_max"]].describe().round(4).to_string())
