"""
16_process_walkability.py
Map EPA Walkability Index from 2020 Census GEOID to 2023 TIGER GEOID.
Direct match for 87.3% of CBGs; areal interpolation for the remaining 12.7%.

Input:  data/raw/EPA步行性数据/WalkabilityIndex/Natl_WI.gdb
        data/raw/CBG地理边界/tl_2023_36_bg/
        data/interim/cbg_nyc.gpkg
Output: data/interim/walkability_cbg.csv
"""

import geopandas as gpd
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

TARGET_CRS = "EPSG:2263"
NYC_COUNTIES = {"005", "047", "061", "081", "085"}

cbg_2023 = gpd.read_file(INTERIM / "cbg_nyc.gpkg")
nyc_geoids_2023 = set(cbg_2023["GEOID"].values)
print(f"2023 NYC CBGs: {len(nyc_geoids_2023)}")

wi = gpd.read_file(
    RAW / "EPA步行性数据" / "WalkabilityIndex" / "Natl_WI.gdb",
    where="STATEFP='36'",
)
wi = wi[wi["COUNTYFP"].isin(NYC_COUNTIES)].copy()
wi = wi.to_crs(TARGET_CRS)
print(f"EPA WI NYC CBGs (2020 vintage): {len(wi)}")

# ──────────────────────────────────────
# Direct match on GEOID20
# ──────────────────────────────────────
direct = wi[wi["GEOID20"].isin(nyc_geoids_2023)][["GEOID20", "NatWalkInd"]].copy()
direct = direct.rename(columns={"GEOID20": "GEOID", "NatWalkInd": "walkability_c"})
matched_geoids = set(direct["GEOID"])
print(f"Direct match: {len(direct)}")

# ──────────────────────────────────────
# Areal interpolation for unmatched 2023 CBGs
# ──────────────────────────────────────
unmatched_geoids = nyc_geoids_2023 - matched_geoids
print(f"Need interpolation: {len(unmatched_geoids)}")

if unmatched_geoids:
    cbg_unmatched = cbg_2023[cbg_2023["GEOID"].isin(unmatched_geoids)].copy()
    wi_geo = wi[["GEOID20", "NatWalkInd", "geometry"]].copy()

    overlay = gpd.overlay(cbg_unmatched[["GEOID", "geometry"]], wi_geo, how="intersection")
    overlay["overlap_area"] = overlay.geometry.area

    interpolated = overlay.groupby("GEOID").apply(
        lambda g: (g["NatWalkInd"] * g["overlap_area"]).sum() / g["overlap_area"].sum()
        if g["overlap_area"].sum() > 0 else None,
        include_groups=False,
    ).reset_index()
    interpolated.columns = ["GEOID", "walkability_c"]
    interpolated = interpolated.dropna()
    print(f"Interpolated: {len(interpolated)}")
else:
    interpolated = pd.DataFrame(columns=["GEOID", "walkability_c"])

# ──────────────────────────────────────
# Combine and save
# ──────────────────────────────────────
result = pd.concat([direct, interpolated], ignore_index=True)

all_cbg = cbg_2023[["GEOID"]].merge(result, on="GEOID", how="left")

out_path = INTERIM / "walkability_cbg.csv"
all_cbg.to_csv(out_path, index=False)

print(f"\nRows: {len(all_cbg)}")
print(f"walkability_c: NaN={all_cbg['walkability_c'].isna().sum()}, "
      f"mean={all_cbg['walkability_c'].mean():.1f}, "
      f"median={all_cbg['walkability_c'].median():.1f}, "
      f"range=[{all_cbg['walkability_c'].min():.1f}, {all_cbg['walkability_c'].max():.1f}]")
print(f"Saved: {out_path}")

# --- self-check ---
_c = pd.read_csv(INTERIM / "walkability_cbg.csv")
assert len(_c) == 6807 and _c["walkability_c"].between(1, 20).all()
print("[self-check] 16 walkability OK")
