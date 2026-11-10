"""
13_compute_access.py
Compute CBG-level park access score from ParkServe 10-min walk service areas.
access_score = fraction of CBG area covered by any Open-access park service area.
Input:  ParkServe_shapefiles_05152026.zip (AllServiceAreas + Parks)
        data/interim/cbg_nyc.gpkg
Output: data/interim/access_cbg.csv
"""

import geopandas as gpd
import pandas as pd
import zipfile
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

TARGET_CRS = "EPSG:2263"
PS_ZIP = RAW / "ParkServe可达性数据" / "空间可达性数据" / "ParkServe_shapefiles_05152026.zip"

cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg")
print(f"CBG loaded: {len(cbg)}, CRS={cbg.crs}")

# ──────────────────────────────────────
# 1. Extract and load Parks → get Open-access ParkIDs
# ──────────────────────────────────────
print("Extracting Parks shapefile to get Open-access ParkIDs...")
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(PS_ZIP) as z:
        for name in z.namelist():
            if name.startswith("Shapefiles/Parks.") and not name.startswith("Shapefiles/Parks_"):
                z.extract(name, tmpdir)

    parks = gpd.read_file(f"{tmpdir}/Shapefiles/Parks.shp")

open_parkids = set(parks[parks["Park_Acces"] == 3]["ParkID"].astype(str))
print(f"Open-access parks nationwide: {len(open_parkids)}")

# ──────────────────────────────────────
# 2. Extract and load AllServiceAreas, clip to NYC bbox
#    AllServiceAreas uses ESRI:102008 (Albers Equal Area), so convert bbox
# ──────────────────────────────────────
ALBERS_CRS = "ESRI:102008"
nyc_bounds_albers = cbg.to_crs(ALBERS_CRS).total_bounds  # in Albers coordinates

print("Extracting AllServiceAreas shapefile (this may take a minute)...")
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(PS_ZIP) as z:
        for name in z.namelist():
            if name.startswith("Shapefiles/AllServiceAreas."):
                z.extract(name, tmpdir)

    svc = gpd.read_file(
        f"{tmpdir}/Shapefiles/AllServiceAreas.shp",
        bbox=tuple(nyc_bounds_albers),
    )

print(f"Service areas loaded (NYC bbox clip): {len(svc)}")

svc["ParkID_str"] = svc["ParkID"].astype(str)
svc = svc[svc["ParkID_str"].isin(open_parkids)]
print(f"After filtering Open-access: {len(svc)}")

# ──────────────────────────────────────
# 3. Dissolve service areas and compute overlay
# ──────────────────────────────────────
svc = svc.to_crs(TARGET_CRS)

print("Dissolving service areas (union of all polygons)...")
svc_dissolved = svc.dissolve().geometry.iloc[0]

print("Computing CBG × service area overlay...")
cbg["cbg_area"] = cbg.geometry.area
cbg["covered_area"] = cbg.geometry.intersection(svc_dissolved).area
cbg["access_score"] = (cbg["covered_area"] / cbg["cbg_area"]).clip(upper=1.0)
cbg["access_binary"] = (cbg["access_score"] > 0).astype(int)

# ──────────────────────────────────────
# 4. Save
# ──────────────────────────────────────
out = cbg[["GEOID", "access_score", "access_binary"]]
out.to_csv(INTERIM / "access_cbg.csv", index=False)

pct_any = 100 * cbg["access_binary"].mean()
print(f"\nRows: {len(out)}")
print(f"access_score: mean={cbg['access_score'].mean():.3f}, median={cbg['access_score'].median():.3f}")
print(f"CBGs with any coverage: {cbg['access_binary'].sum()} ({pct_any:.1f}%)")
print(f"Fully covered (>=0.99): {(cbg['access_score'] >= 0.99).sum()}")
print(f"Zero coverage: {(cbg['access_score'] == 0).sum()}")
print(f"Saved: {INTERIM / 'access_cbg.csv'}")
