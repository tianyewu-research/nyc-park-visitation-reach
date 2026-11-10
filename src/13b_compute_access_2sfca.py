"""
13b_compute_access_2sfca.py   (replaces the degenerate coverage-based access)
Two-Step Floating Catchment Area (2SFCA) accessibility.
Supply = park area (acres); Demand = population within catchment; catchment = 800m (10-min walk).
Polygon-aware: a park is reachable from a CBG if the park boundary lies within 800m of the
CBG centroid (so large parks are handled correctly).

Step 1: each park j -> R_j = area_j / (sum of population of CBGs whose 800m catchment reaches j)
Step 2: each CBG i -> access_2sfca_i = sum of R_j over parks reaching i

Input:  data/raw/NYC公园边界/Parks Properties_20260623/
        data/interim/cbg_nyc.gpkg
        data/interim/acs_cbg.csv   (total_pop)
Output: data/interim/access_cbg.csv   (access_score = 2SFCA; coverage kept as access_score_cov)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

TARGET_CRS = "EPSG:2263"          # US survey feet
CATCHMENT_FT = 800 / 0.3048       # 800 m in feet
MIN_PARK_ACRES = 0.1

cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg").to_crs(TARGET_CRS)
acs = pd.read_csv(INTERIM / "acs_cbg.csv", dtype={"GEOID": str})
cbg = cbg.merge(acs[["GEOID", "total_pop"]], on="GEOID", how="left")
cbg["total_pop"] = cbg["total_pop"].fillna(0)
cbg["cx"] = cbg.geometry.centroid.x
cbg["cy"] = cbg.geometry.centroid.y
cbg_pts = gpd.GeoDataFrame(
    cbg[["GEOID", "total_pop"]],
    geometry=gpd.points_from_xy(cbg["cx"], cbg["cy"]), crs=TARGET_CRS,
)
print(f"CBGs: {len(cbg_pts)}")

parks = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623").to_crs(TARGET_CRS)
parks = parks[parks["acres"] >= MIN_PARK_ACRES].reset_index(drop=True)
parks["park_id"] = parks.index
parks_geom = parks[["park_id", "acres", "geometry"]].copy()
print(f"Parks (>= {MIN_PARK_ACRES} acre): {len(parks_geom)}")

# Reachability: CBG centroid within CATCHMENT_FT of a park boundary.
# Buffer CBG centroids by catchment, then intersect with park polygons.
cbg_buf = cbg_pts.copy()
cbg_buf["geometry"] = cbg_buf.geometry.buffer(CATCHMENT_FT)
print("Spatial join (CBG 800m catchment x parks)...")
pairs = gpd.sjoin(cbg_buf[["GEOID", "total_pop", "geometry"]],
                  parks_geom[["park_id", "acres", "geometry"]],
                  predicate="intersects", how="inner")
print(f"Reachable CBG-park pairs: {len(pairs)}")

# Step 1: supply-to-demand ratio per park
demand = pairs.groupby("park_id")["total_pop"].sum().rename("demand")
supply = parks_geom.set_index("park_id")["acres"]
R = (supply / demand.replace(0, np.nan)).rename("R_j")        # acres per person
R = R.fillna(0)

# Step 2: sum R_j over reachable parks per CBG
pairs = pairs.merge(R, left_on="park_id", right_index=True, how="left")
access = pairs.groupby("GEOID")["R_j"].sum().rename("access_2sfca")

out = cbg[["GEOID"]].merge(access, on="GEOID", how="left")
out["access_2sfca"] = out["access_2sfca"].fillna(0)

# keep prior coverage-based access for reference if present
cov_path = INTERIM / "access_cbg.csv"
if cov_path.exists():
    prev = pd.read_csv(cov_path, dtype={"GEOID": str})
    if "access_score" in prev.columns:
        out = out.merge(prev[["GEOID", "access_score"]].rename(
            columns={"access_score": "access_score_cov"}), on="GEOID", how="left")

out["access_score"] = out["access_2sfca"]                     # 2SFCA is now the primary measure
out.to_csv(cov_path, index=False)

a = out["access_score"]
print(f"\n=== 2SFCA access_score (acres per person within 800m) ===")
print(f"mean={a.mean():.4f}, median={a.median():.4f}, std={a.std():.4f}")
print(f"CV={a.std()/a.mean():.2f}  (coverage version had CV~0.13 -> no signal)")
print(f"zeros (no reachable park): {(a==0).sum()} ({100*(a==0).mean():.1f}%)")
print(f"P5={a.quantile(.05):.4f}, P95={a.quantile(.95):.4f}, max={a.max():.3f}")
print(f"Saved: {cov_path}")

# --- self-check ---
_c = pd.read_csv(INTERIM / "access_cbg.csv")
assert len(_c) == 6807 and (_c["access_score"] >= 0).all()
assert _c["access_score"].std() / _c["access_score"].mean() > 1   # has real variation
print("[self-check] 13b 2SFCA access OK")
