"""
15b_compute_local_ratio.py   (new framework, 2026-07-28)
Local Park-Use Ratio (LPUR): the share of a block group's dwell-qualified park visits that go to
parks within a neighborhood radius of home, measured with the true home-centroid-to-park-POI
distance (not the coarse CBG-membership test in the deprecated 15_). Introduced to replace the
access-use mismatch residual, which was near-collinear with use (r=0.96); LPUR is a direct,
self-varying behavioral construct: high LPUR = residents rely on nearby parks, low LPUR = they
travel out. Companion measure: mean home-to-park trip distance per block group (km).

Weighting matches the use measure: each POI-week contributes VISITOR_COUNTS x dwell-qualified
fraction (>=5 min) apportioned to home block groups by device share. Computes LPUR on a radius
GRID (0.5-3 km) so the income gradient can be shown as a specification curve, plus the primary
2 km ratio and the visit-distance CDF (share of visits within each radius, city-wide).

Input:  data/raw/nypark_2325/2023-*.csv.gz ; data/interim/cbg_nyc.gpkg
Output: data/interim/local_ratio_cbg.csv     (GEOID, lpur_<r>km ..., mean_trip_km, total_visits)
        outputs/tables/visit_distance_cdf.csv (radius_km, share_within)
"""
import pandas as pd, geopandas as gpd, numpy as np, json, glob
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"; INT = ROOT / "data" / "interim"; TAB = ROOT / "outputs" / "tables"
NYC = ("36005", "36047", "36061", "36081", "36085"); M2FT = 1 / 0.3048
DWELL_KEEP = ("5-20", "21-60", "61-240", ">240")
GRID_KM = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
GRID_FT = [r * 1000 * M2FT for r in GRID_KM]


def qfrac(bj):
    if not isinstance(bj, str):
        return 1.0
    try:
        b = json.loads(bj)
    except Exception:
        return 1.0
    t = sum(b.values())
    return sum(b.get(k, 0) for k in DWELL_KEEP) / t if t > 0 else 0.0


cbg = gpd.read_file(INT / "cbg_nyc.gpkg")
cx = cbg.geometry.centroid.x.values; cy = cbg.geometry.centroid.y.values
gid = cbg["GEOID"].values; gidx = {g: i for i, g in enumerate(gid)}

within = [defaultdict(float) for _ in GRID_FT]   # per-radius within-count (dwell-qualified visitors)
total = defaultdict(float)                         # total dwell-qualified visitors
wdist = defaultdict(float)                         # visitor-weighted sum of distance (ft)
cdf_within = np.zeros(len(GRID_FT)); cdf_total = 0.0

files = sorted(glob.glob(str(RAW / "nypark_2325" / "2023-*.csv.gz")))
for k, f in enumerate(files):
    df = pd.read_csv(f, usecols=["POI_CBG", "VISITOR_HOME_CBGS", "VISITOR_COUNTS",
                                 "BUCKETED_DWELL_TIMES", "LATITUDE", "LONGITUDE"])
    df["pc"] = df["POI_CBG"].dropna().astype("Int64").astype(str).str.zfill(12)
    df = df[df["pc"].str[:5].isin(NYC)].dropna(subset=["LATITUDE", "LONGITUDE"])
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE),
                           crs="EPSG:4326").to_crs("EPSG:2263")
    px = pts.geometry.x.values; py = pts.geometry.y.values
    vc = df["VISITOR_COUNTS"].values; bdt = df["BUCKETED_DWELL_TIMES"].values
    vhc = df["VISITOR_HOME_CBGS"].values
    for j in range(len(df)):
        v = vhc[j]
        if not isinstance(v, str):
            continue
        try:
            hd = json.loads(v)
        except Exception:
            continue
        visitors = (vc[j] if pd.notna(vc[j]) else 0) * qfrac(bdt[j])
        tot = sum(hd.values())
        if tot <= 0 or visitors <= 0:
            continue
        for hc, dc in hd.items():
            if hc.startswith("CA:"):
                continue
            hc = hc.zfill(12); i = gidx.get(hc)
            if i is None:
                continue
            qv = visitors * (dc / tot)
            d = np.hypot(cx[i] - px[j], cy[i] - py[j])
            total[hc] += qv; wdist[hc] += qv * d; cdf_total += qv
            for r, rr in enumerate(GRID_FT):
                if d <= rr:
                    within[r][hc] += qv; cdf_within[r] += qv
    if (k + 1) % 13 == 0:
        print(f"  {k+1}/52")

rows = []
for g in gid:
    t = total.get(g, 0.0)
    if t <= 0:
        rows.append({"GEOID": g, "total_visits": 0.0, "mean_trip_km": None,
                     **{f"lpur_{r}km": None for r in GRID_KM}})
        continue
    row = {"GEOID": g, "total_visits": round(t, 1),
           "mean_trip_km": round((wdist[g] / t) / M2FT / 1000, 3)}
    for r, rk in enumerate(GRID_KM):
        row[f"lpur_{rk}km"] = round(within[r][g] / t, 4)
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(INT / "local_ratio_cbg.csv", index=False)

cdf = pd.DataFrame({"radius_km": GRID_KM, "share_within": (cdf_within / cdf_total).round(4)})
cdf.to_csv(TAB / "visit_distance_cdf.csv", index=False)

d = out[out.total_visits > 0]
print(f"\nCBGs with visits: {len(d)}")
print(f"LPUR 2km: mean={d['lpur_2.0km'].mean():.3f} sd={d['lpur_2.0km'].std():.3f} "
      f"IQR=[{d['lpur_2.0km'].quantile(.25):.3f},{d['lpur_2.0km'].quantile(.75):.3f}]")
print(f"mean_trip_km: mean={d['mean_trip_km'].mean():.2f} sd={d['mean_trip_km'].std():.2f}")
print("City-wide visit-distance CDF (share within radius):")
print(cdf.to_string(index=False))

# --- self-check ---
_c = pd.read_csv(INT / "local_ratio_cbg.csv")
assert len(_c) == 6807 and _c["lpur_2.0km"].dropna().between(0, 1).all()
print("[self-check] 15b local ratio OK")
