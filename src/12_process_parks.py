"""
12_process_parks.py
Build park-level quality index and aggregate to CBG level.
Components (equal weight): has_restroom, playground_count, canopy_pct, is_active_rec.
Input:  data/raw/NYC公园边界/Parks Properties_20260623/
        data/raw/公园属性数据/NYC_Parks_Active_and_Passive_Recreation_20260623.csv
        data/raw/公园属性数据/Directory_Of_Toilets_In_Public_Parks_20260624.csv
        ParkServe AdditionalData.csv (extracted from zip)
        data/interim/cbg_nyc.gpkg
Output: data/interim/parks_quality.csv   (park-level)
        data/interim/parks_cbg.csv       (CBG-level)
"""

import pandas as pd
import geopandas as gpd
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

TARGET_CRS = "EPSG:2263"

# ──────────────────────────────────────
# 1. Load park boundaries (B01)
# ──────────────────────────────────────
parks = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623")
parks = parks.to_crs(TARGET_CRS)
parks["gispropnum"] = parks["gispropnum"].str.strip()
print(f"B01 parks loaded: {len(parks)}")

# ──────────────────────────────────────
# 2. Recreation data → is_active_rec
# ──────────────────────────────────────
rec = pd.read_csv(RAW / "公园属性数据" / "NYC_Parks_Active_and_Passive_Recreation_20260623.csv")
rec["GISPROPNUM"] = rec["GISPROPNUM"].str.strip()
rec_agg = rec.groupby("GISPROPNUM")["RECREATION CATEGORY"].agg(
    lambda x: "Active" if "Active" in x.values else ("Mixed" if "Mixed" in x.values else "Passive")
).reset_index()
rec_agg.columns = ["gispropnum", "rec_category"]
rec_agg["is_active_rec"] = (rec_agg["rec_category"].isin(["Active", "Mixed"])).astype(int)

parks = parks.merge(rec_agg[["gispropnum", "is_active_rec"]], on="gispropnum", how="left")
parks["is_active_rec"] = parks["is_active_rec"].fillna(0).astype(int)
print(f"Recreation matched: {parks['is_active_rec'].sum()} active/mixed parks")

# ──────────────────────────────────────
# 3. Toilets → has_restroom (name matching)
# ──────────────────────────────────────
toilets = pd.read_csv(RAW / "公园属性数据" / "Directory_Of_Toilets_In_Public_Parks_20260624.csv")
toilet_names = set(toilets["Name"].str.strip().str.lower())

parks["_name_lower"] = parks["signname"].fillna("").str.strip().str.lower()

def match_restroom(park_name):
    if not park_name:
        return 0
    if park_name in toilet_names:
        return 1
    for tn in toilet_names:
        if tn in park_name or park_name in tn:
            return 1
    return 0

parks["has_restroom"] = parks["_name_lower"].apply(match_restroom)
print(f"Restroom matched: {parks['has_restroom'].sum()} parks")

# ──────────────────────────────────────
# 4. ParkServe AdditionalData → playground_count, canopy_pct
# ──────────────────────────────────────
ps_zip = RAW / "ParkServe可达性数据" / "空间可达性数据" / "ParkServe_shapefiles_05152026.zip"
with zipfile.ZipFile(ps_zip) as z:
    with z.open("Shapefiles/AdditionalData.csv") as f:
        ps_add = pd.read_csv(f)

ps_nyc = ps_add[ps_add["ParkID"].astype(str).str.startswith("3651000")].copy()
ps_nyc["_name_lower"] = ps_nyc["Park_Name"].str.strip().str.lower()

parks_with_ps = parks.merge(
    ps_nyc[["_name_lower", "playground_count", "canopy_pct"]],
    on="_name_lower",
    how="left",
)
# deduplicate: keep first match per park
parks_with_ps = parks_with_ps.drop_duplicates(subset="gispropnum", keep="first")

parks_with_ps["playground_count"] = parks_with_ps["playground_count"].fillna(0)
parks_with_ps["canopy_pct"] = parks_with_ps["canopy_pct"].clip(upper=100).fillna(parks_with_ps["canopy_pct"].median())

matched_ps = parks_with_ps["canopy_pct"].notna().sum()
print(f"ParkServe name matched: {matched_ps} / {len(parks)} parks have canopy/playground data")

# ──────────────────────────────────────
# 5. Build quality_index (equal weight, min-max normalized)
# ──────────────────────────────────────
def minmax(s):
    r = s.max() - s.min()
    if r == 0:
        return s * 0
    return (s - s.min()) / r

parks_q = parks_with_ps.copy()
parks_q["n_restroom"] = minmax(parks_q["has_restroom"].astype(float))
parks_q["n_playground"] = minmax(parks_q["playground_count"].astype(float))
parks_q["n_canopy"] = minmax(parks_q["canopy_pct"].astype(float))
parks_q["n_active"] = minmax(parks_q["is_active_rec"].astype(float))

parks_q["quality_index"] = (
    parks_q["n_restroom"] + parks_q["n_playground"] + parks_q["n_canopy"] + parks_q["n_active"]
) / 4

park_quality_out = parks_q[
    ["gispropnum", "signname", "acres", "typecatego", "has_restroom",
     "playground_count", "canopy_pct", "is_active_rec", "quality_index"]
]
park_quality_out.to_csv(INTERIM / "parks_quality.csv", index=False)
print(f"\nPark-level quality saved: {len(park_quality_out)} parks")
print(f"quality_index: mean={parks_q['quality_index'].mean():.3f}, median={parks_q['quality_index'].median():.3f}")

# ──────────────────────────────────────
# 6. Aggregate to CBG level
#    Part A: park_area and park_count (from overlay, only parks inside CBG)
#    Part B: quality_index_cbg via 1km buffer (default, independent of access_score)
#            quality_index_cbg_svc via ParkServe service area (alternative, for comparison)
# ──────────────────────────────────────
from scipy.spatial import cKDTree
import numpy as np

# EPSG:2263 uses US survey feet. Convert the intended 1 km buffer explicitly.
METERS_TO_FEET = 1 / 0.3048
BUFFER_FT = 1000 * METERS_TO_FEET

cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg")

# --- Part A: park_area, park_count (intersection-based) ---
parks_geo = parks_q[["gispropnum", "acres", "quality_index", "geometry"]].copy()
parks_geo = parks_geo[parks_geo.geometry.notna() & ~parks_geo.geometry.is_empty]

overlay = gpd.overlay(parks_geo, cbg[["GEOID", "geometry"]], how="intersection")
# EPSG:2263 units are US survey feet; 1 acre = 43,560 sq ft.
# park_area_acres must be the area of the park *within* the CBG (intersection),
# NOT the full park acreage — otherwise a park spanning N CBGs is counted N times.
overlay["overlap_acres"] = overlay.geometry.area / 43560.0

park_cbg = overlay.groupby("GEOID").apply(
    lambda g: pd.Series({
        "park_area_acres": g["overlap_acres"].sum(),
        "park_count": g["gispropnum"].nunique(),
    }),
    include_groups=False,
).reset_index()

all_cbg = cbg[["GEOID"]].merge(park_cbg, on="GEOID", how="left")
all_cbg["park_area_acres"] = all_cbg["park_area_acres"].fillna(0)
all_cbg["park_count"] = all_cbg["park_count"].fillna(0).astype(int)

# --- Part B: quality_index via 1km buffer (default) ---
print("\nComputing quality_index_cbg via 1km buffer...")
cbg_centroids = cbg.geometry.centroid
cbg_coords = np.column_stack([cbg_centroids.x, cbg_centroids.y])

park_centroids = parks_geo.geometry.centroid
park_coords = np.column_stack([park_centroids.x, park_centroids.y])
park_quality = parks_geo["quality_index"].values

park_tree = cKDTree(park_coords)

qi_1km = []
for i in range(len(cbg)):
    idx = park_tree.query_ball_point(cbg_coords[i], r=BUFFER_FT)
    if len(idx) == 0:
        qi_1km.append(None)
    else:
        dists = np.array([np.linalg.norm(cbg_coords[i] - park_coords[j]) for j in idx])
        weights = 1.0 / np.maximum(dists, 1.0)  # inverse distance, floor at 1m
        vals = park_quality[idx]
        mask = ~np.isnan(vals)
        if mask.any():
            qi_1km.append(np.average(vals[mask], weights=weights[mask]))
        else:
            qi_1km.append(None)

all_cbg["quality_index_cbg"] = qi_1km

# --- Part B alt: quality_index via ParkServe service area (for comparison) ---
print("Computing quality_index_cbg_svc via ParkServe service area...")
import zipfile, tempfile

PS_ZIP = RAW / "ParkServe可达性数据" / "空间可达性数据" / "ParkServe_shapefiles_05152026.zip"
ALBERS_CRS = "ESRI:102008"
nyc_bounds_albers = cbg.to_crs(ALBERS_CRS).total_bounds

with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(PS_ZIP) as z:
        for name in z.namelist():
            if name.startswith("Shapefiles/AllServiceAreas."):
                z.extract(name, tmpdir)
    svc = gpd.read_file(f"{tmpdir}/Shapefiles/AllServiceAreas.shp", bbox=tuple(nyc_bounds_albers))

svc = svc.to_crs(TARGET_CRS)

ps_quality = parks_q[["_name_lower", "quality_index"]].drop_duplicates(subset="_name_lower")
svc["_name_lower"] = svc["Park_Name"].str.strip().str.lower()
svc = svc.merge(ps_quality, on="_name_lower", how="left")
svc = svc[svc["quality_index"].notna()]

svc_overlay = gpd.overlay(svc[["quality_index", "geometry"]], cbg[["GEOID", "geometry"]], how="intersection")
svc_overlay["overlap_area"] = svc_overlay.geometry.area

qi_svc = svc_overlay.groupby("GEOID").apply(
    lambda g: (g["quality_index"] * g["overlap_area"]).sum() / g["overlap_area"].sum()
    if g["overlap_area"].sum() > 0 else None,
    include_groups=False,
).reset_index()
qi_svc.columns = ["GEOID", "quality_index_cbg_svc"]

all_cbg = all_cbg.merge(qi_svc, on="GEOID", how="left")

# --- Save ---
all_cbg.to_csv(INTERIM / "parks_cbg.csv", index=False)

no_park = (all_cbg["park_count"] == 0).sum()
qi_1km_miss = all_cbg["quality_index_cbg"].isna().sum()
qi_svc_miss = all_cbg["quality_index_cbg_svc"].isna().sum()

print(f"\nCBG-level saved: {len(all_cbg)} rows")
print(f"CBGs with no park inside: {no_park} ({100*no_park/len(all_cbg):.1f}%)")
print(f"quality_index_cbg (1km buffer):   NaN={qi_1km_miss} ({100*qi_1km_miss/len(all_cbg):.1f}%)")
print(f"quality_index_cbg_svc (svc area): NaN={qi_svc_miss} ({100*qi_svc_miss/len(all_cbg):.1f}%)")
print(f"park_area_acres: mean={all_cbg['park_area_acres'].mean():.2f}")

# --- self-check ---
_c = pd.read_csv(INTERIM / "parks_cbg.csv")
assert len(_c) == 6807 and (_c["park_area_acres"] >= 0).all()
assert _c["quality_index_cbg_svc"].dropna().between(0, 1).all()
print("[self-check] 12 parks OK")
