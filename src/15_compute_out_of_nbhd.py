"""
15_compute_out_of_nbhd.py
Compute out-of-neighborhood park visit ratio per CBG.
Neighborhood = CBGs whose centroids are within 1 km of the home CBG centroid (Q7, revised).
out_park_ratio = visits to parks outside neighborhood / total visits.

CRS note: EPSG:2263 uses US survey feet. The earlier script passed 1,000 directly
to the KD-tree and therefore implemented 1,000 feet, not 1 km. This version
converts 1,000 metres to feet explicitly.

Input:  data/raw/nypark_2325/2023-*.csv.gz
        data/interim/cbg_nyc.gpkg
Output: data/interim/out_nbhd_cbg.csv
"""

import pandas as pd
import geopandas as gpd
import json
import glob
from pathlib import Path
from collections import defaultdict
from scipy.spatial import cKDTree
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

NYC_FIPS_PREFIXES = ("36005", "36047", "36061", "36081", "36085")
METERS_TO_FEET = 1 / 0.3048
BUFFER_FT = 1000 * METERS_TO_FEET

cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg")  # EPSG:2263, US survey feet
assert cbg.crs is not None and cbg.crs.to_epsg() == 2263
nyc_geoids = set(cbg["GEOID"].values)

centroids = cbg.copy()
centroids["cx"] = centroids.geometry.centroid.x
centroids["cy"] = centroids.geometry.centroid.y
coords = centroids[["cx", "cy"]].values
geoid_list = centroids["GEOID"].values

print("Building 1km neighborhood lookup via KD-tree...")
tree = cKDTree(coords)
nbhd = {}
for i, geoid in enumerate(geoid_list):
    idx = tree.query_ball_point(coords[i], r=BUFFER_FT)
    nbhd[geoid] = set(geoid_list[idx])

avg_size = np.mean([len(v) for v in nbhd.values()])
print(f"Avg neighborhood size (1km buffer): {avg_size:.1f} CBGs")

files_2023 = sorted(glob.glob(str(RAW / "nypark_2325" / "2023-*.csv.gz")))
print(f"Weekly files: {len(files_2023)}")

cbg_in_nbhd = defaultdict(float)
cbg_total = defaultdict(float)

for i, fpath in enumerate(files_2023):
    df = pd.read_csv(fpath, usecols=["POI_CBG", "VISITOR_HOME_CBGS"])

    df["POI_CBG_str"] = df["POI_CBG"].dropna().astype(int).astype(str).str.zfill(12)
    df = df[df["POI_CBG_str"].str[:5].isin(NYC_FIPS_PREFIXES)]

    for _, row in df.iterrows():
        vhc = row["VISITOR_HOME_CBGS"]
        if pd.isna(vhc):
            continue
        try:
            home_dict = json.loads(vhc)
        except (json.JSONDecodeError, TypeError):
            continue

        park_cbg = row["POI_CBG_str"]

        for home_cbg, device_count in home_dict.items():
            if home_cbg.startswith("CA:"):
                continue
            home_cbg = home_cbg.zfill(12)
            if home_cbg not in nyc_geoids:
                continue

            cbg_total[home_cbg] += device_count
            if park_cbg in nbhd.get(home_cbg, set()):
                cbg_in_nbhd[home_cbg] += device_count

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Processed {i+1}/{len(files_2023)} weeks")

rows = []
for geoid in nyc_geoids:
    total = cbg_total.get(geoid, 0)
    in_nb = cbg_in_nbhd.get(geoid, 0)
    ratio = 1 - (in_nb / total) if total > 0 else None
    rows.append({"GEOID": geoid, "out_park_ratio": round(ratio, 4) if ratio is not None else None})

result = pd.DataFrame(rows)
result.to_csv(INTERIM / "out_nbhd_cbg.csv", index=False)

has_data = result["out_park_ratio"].notna()
print(f"\nRows: {len(result)}")
print(f"CBGs with data: {has_data.sum()}")
print(f"out_park_ratio: mean={result['out_park_ratio'].mean():.3f}, median={result['out_park_ratio'].median():.3f}")
print(f"Saved: {INTERIM / 'out_nbhd_cbg.csv'}")

# --- self-check ---
_c = pd.read_csv(INTERIM / "out_nbhd_cbg.csv")
assert len(_c) == 6807 and _c["out_park_ratio"].dropna().between(0, 1).all()
print("[self-check] 15 out-of-nbhd OK")
