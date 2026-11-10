"""
14c_compute_use_matched.py   (robustness C14: matched-park-universe use measure)

Advisor point (C14): use is summed over all ~6,300 POIs, but access (E2SFCA supply) is built
only from the 1,676 mapped NYC park properties. ~28% of POIs (state/federal/POPS open spaces)
fall outside that boundary set and account for ~20% of visitor volume, so a positive mismatch
could partly reflect access not covering a park residents actually use, rather than genuine
use-above-access. This script rebuilds use from ONLY the POIs that spatially match the NYC park
boundaries used for access (within 100 ft), so the use and access park universes coincide, and
the RQ conclusions can be re-checked on that matched universe.

Input:  data/raw/nypark_2325/2023-*.csv.gz ; data/interim/cbg_nyc.gpkg
        data/raw/NYC公园边界/Parks Properties_20260623/
Output: data/interim/use_cbg_matched.csv   (GEOID, V_c_matched)
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import json, glob
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
NYC_FIPS = ("36005", "36047", "36061", "36081", "36085")
DWELL_KEEP = ("5-20", "21-60", "61-240", ">240")
CRS = "EPSG:2263"
MATCH_FT = 100.0


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

# accumulate per-POI annual dwell-qualified visitors + home apportionment + coordinates
poi_home = defaultdict(lambda: defaultdict(float))
poi_xy = {}
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

# spatial match POIs to NYC park boundaries (same universe as the access supply)
parks = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623").to_crs(CRS)
parks = parks[parks["acres"] >= 0.1].reset_index(drop=True)
sids = [s for s in poi_home if s in poi_xy]
pts = gpd.GeoDataFrame({"ID_STORE": sids},
                       geometry=gpd.points_from_xy([poi_xy[s][1] for s in sids],
                                                   [poi_xy[s][0] for s in sids]),
                       crs="EPSG:4326").to_crs(CRS)
nn = gpd.sjoin_nearest(pts, parks[["geometry"]], distance_col="dist")
nn = nn.drop_duplicates("ID_STORE")
matched = set(nn.loc[nn["dist"] <= MATCH_FT, "ID_STORE"])
print(f"POIs {len(sids)} -> matched to NYC park boundary (<={MATCH_FT:.0f}ft): {len(matched)} "
      f"({100*len(matched)/len(sids):.0f}%)")

V = defaultdict(float)
for sid in matched:
    for hcbg, v in poi_home[sid].items():
        V[hcbg] += v

out = pd.DataFrame({"GEOID": list(nyc_geoids)})
out["V_c_matched"] = out["GEOID"].map(lambda g: round(V.get(g, 0.0), 2))
out.to_csv(INTERIM / "use_cbg_matched.csv", index=False)
print(f"V_c_matched median(>0)={out.loc[out.V_c_matched>0,'V_c_matched'].median():.0f}  "
      f"saved {INTERIM/'use_cbg_matched.csv'}")

# --- self-check ---
_c = pd.read_csv(INTERIM / "use_cbg_matched.csv")
assert len(_c) == 6807 and (_c["V_c_matched"] >= 0).all()
print("[self-check] 14c use_matched OK")
