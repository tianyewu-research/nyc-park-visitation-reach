"""
10_prepare_base_geography.py
Build NYC CBG base layer: filter to NYC, add borough, project, compute area.
Input:  data/raw/CBG地理边界/tl_2023_36_bg/
        data/raw/NYC行政边界/Borough_Boundaries_20260623.geojson
Output: data/interim/cbg_nyc.gpkg
"""

import geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

NYC_COUNTY_FIPS = {"005", "047", "061", "081", "085"}
BOROUGH_MAP = {
    "005": "Bronx",
    "047": "Brooklyn",
    "061": "Manhattan",
    "081": "Queens",
    "085": "Staten Island",
}
TARGET_CRS = "EPSG:2263"

cbg = gpd.read_file(RAW / "CBG地理边界" / "tl_2023_36_bg")
cbg = cbg[cbg["COUNTYFP"].isin(NYC_COUNTY_FIPS)].copy()
cbg["borough"] = cbg["COUNTYFP"].map(BOROUGH_MAP)

# TIGER ALAND = land area in square meters (official, excludes water).
# Used for population density so waterfront CBGs are not understated.
cbg["land_area_sqm"] = cbg["ALAND"].astype(float)

cbg = cbg.to_crs(TARGET_CRS)
# geometry area in EPSG:2263 (US survey feet) — kept for geometric checks only.
cbg["geom_area_sqft"] = cbg.geometry.area

cbg = cbg[["GEOID", "borough", "land_area_sqm", "geom_area_sqft", "geometry"]].reset_index(drop=True)

out_path = INTERIM / "cbg_nyc.gpkg"
cbg.to_file(out_path, driver="GPKG")

print(f"Rows: {len(cbg)}")
print(f"CRS:  {cbg.crs}")
print(f"Borough counts:\n{cbg['borough'].value_counts().to_string()}")
print(f"Land area (sqm): min={cbg['land_area_sqm'].min():.0f}, max={cbg['land_area_sqm'].max():.0f}, median={cbg['land_area_sqm'].median():.0f}")
print(f"Land area = 0 (all-water CBGs): {(cbg['land_area_sqm'] == 0).sum()}")
print(f"Saved: {out_path}")

# --- self-check ---
_c = gpd.read_file(INTERIM / "cbg_nyc.gpkg")
assert len(_c) == 6807 and _c["GEOID"].is_unique and _c["borough"].notna().all()
print("[self-check] 10 base geography OK")
