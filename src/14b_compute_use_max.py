"""
14b_compute_use_max.py   (robustness: single-busiest-geofence use measure)

Advisor point: summing all points of interest within a park can over-count visitors where
overlapping sub-area geofences (e.g. Central Park's 70+ POIs) record the same device more than
once. This script builds an alternative V_c that keeps only each park-site's SINGLE BUSIEST
geofence, removing within-park overlap, so the RQ conclusions can be checked against it (14_ is
the main sum-per-park measure).

Method: POIs are stable across weeks (ID_STORE). We (1) accumulate each POI's annual
dwell-qualified unique visitors and its home-CBG apportionment; (2) cluster POIs into park-sites
by 100 m proximity (DBSCAN, unaffected by the incomplete park-boundary file); (3) keep, per
site, the POI with the largest annual volume; (4) sum the kept POIs' home apportionment to V_c_max.

Input:  data/raw/nypark_2325/2023-*.csv.gz ; data/interim/cbg_nyc.gpkg
Output: data/interim/use_cbg_max.csv   (GEOID, V_c_max)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import json, glob
from collections import defaultdict
from pathlib import Path
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
NYC_FIPS = ("36005", "36047", "36061", "36081", "36085")
DWELL_KEEP = ("5-20", "21-60", "61-240", ">240")


def qfrac(bj):
    if not isinstance(bj, str):
        return 1.0
    try:
        b = json.loads(bj)
    except Exception:
        return 1.0
    t = sum(b.values())
    return sum(b.get(k, 0) for k in DWELL_KEEP) / t if t > 0 else 0.0


cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg")
nyc_geoids = set(cbg["GEOID"].values)
files = sorted(glob.glob(str(RAW / "nypark_2325" / "2023-*.csv.gz")))
print(f"weeks: {len(files)}")

poi_total = defaultdict(float)                 # ID_STORE -> annual qualified visitors
poi_home = defaultdict(lambda: defaultdict(float))  # ID_STORE -> {home_cbg -> visitors}
poi_xy = {}                                    # ID_STORE -> (lat, lon)

for i, f in enumerate(files):
    df = pd.read_csv(f, usecols=["ID_STORE", "POI_CBG", "VISITOR_HOME_CBGS",
                                 "VISITOR_COUNTS", "BUCKETED_DWELL_TIMES", "LATITUDE", "LONGITUDE"])
    df["cbg"] = df["POI_CBG"].dropna().astype("Int64").astype(str).str.zfill(12)
    df = df[df["cbg"].str[:5].isin(NYC_FIPS)]
    for _, r in df.iterrows():
        vhc = r["VISITOR_HOME_CBGS"]
        if pd.isna(vhc):
            continue
        try:
            hd = json.loads(vhc)
        except Exception:
            continue
        vis = r["VISITOR_COUNTS"] if pd.notna(r["VISITOR_COUNTS"]) else 0
        qv = vis * qfrac(r["BUCKETED_DWELL_TIMES"])
        tot = sum(hd.values())
        if tot <= 0:
            continue
        sid = r["ID_STORE"]
        poi_total[sid] += qv
        if sid not in poi_xy and pd.notna(r["LATITUDE"]):
            poi_xy[sid] = (r["LATITUDE"], r["LONGITUDE"])
        for hcbg, dc in hd.items():
            if hcbg.startswith("CA:"):
                continue
            hcbg = hcbg.zfill(12)
            if hcbg in nyc_geoids:
                poi_home[sid][hcbg] += qv * (dc / tot)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(files)}")

# cluster POIs into park-sites by 100 m proximity, keep argmax-volume POI per site
sids = [s for s in poi_total if s in poi_xy]
coords = np.radians([poi_xy[s] for s in sids])
labels = DBSCAN(eps=100 / 6371000, min_samples=1, metric="haversine").fit(coords).labels_
site_best = {}
for s, lab in zip(sids, labels):
    if lab not in site_best or poi_total[s] > poi_total[site_best[lab]]:
        site_best[lab] = s
kept = set(site_best.values())
print(f"POIs {len(sids)} -> park-sites {len(kept)} (kept busiest per site)")

V_max = defaultdict(float)
for sid in kept:
    for hcbg, v in poi_home[sid].items():
        V_max[hcbg] += v

out = pd.DataFrame({"GEOID": list(nyc_geoids)})
out["V_c_max"] = out["GEOID"].map(lambda g: round(V_max.get(g, 0.0), 2))
out.to_csv(INTERIM / "use_cbg_max.csv", index=False)
print(f"V_c_max: median(>0)={out.loc[out.V_c_max>0,'V_c_max'].median():.0f}  saved {INTERIM/'use_cbg_max.csv'}")

# --- self-check ---
_c = pd.read_csv(INTERIM / "use_cbg_max.csv")
assert len(_c) == 6807 and (_c["V_c_max"] >= 0).all()
print("[self-check] 14b use_max OK")
