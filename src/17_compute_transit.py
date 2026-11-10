"""
17_compute_transit.py
Extract CBG-level transit accessibility from EPA Smart Location Database.

Original plan: compute CBG→park travel time via r5py + GTFS.
Fallback (active): JPype/JVM SIGBUS crash on macOS Tahoe prevents r5py from running.
Using EPA SLD D4A (aggregate transit frequency per sq mi) as proxy.
Same GEOID 2020→2023 interpolation logic as 16_process_walkability.py.

If r5py becomes runnable (macOS/JPype fix), revert to the GTFS-based computation
documented in data_processing_workflow.md §17.

Input:  data/raw/EPA步行性数据/SmartLocationDatabaseV3/SmartLocationDatabase.gdb
        data/raw/CBG地理边界/tl_2023_36_bg/
        data/interim/cbg_nyc.gpkg
Output: data/interim/transit_cbg.csv
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

sld = gpd.read_file(
    RAW / "EPA步行性数据" / "SmartLocationDatabaseV3" / "SmartLocationDatabase.gdb",
    where="STATEFP='36'",
)
sld = sld[sld["COUNTYFP"].isin(NYC_COUNTIES)].copy()
sld = sld.to_crs(TARGET_CRS)
print(f"EPA SLD NYC CBGs (2020 vintage): {len(sld)}")

transit_cols = ["D4A", "D4B025", "D4C", "D4D"]
for col in transit_cols:
    bad = (sld[col] == -99999)
    sld.loc[bad, col] = None
    print(f"  {col}: valid={sld[col].notna().sum()}, -99999 replaced={bad.sum()}")

# ──────────────────────────────────────
# Direct match
# ──────────────────────────────────────
direct = sld[sld["GEOID20"].isin(nyc_geoids_2023)][["GEOID20"] + transit_cols].copy()
direct = direct.rename(columns={"GEOID20": "GEOID"})
matched_geoids = set(direct["GEOID"])
print(f"Direct match: {len(direct)}")

# ──────────────────────────────────────
# Areal interpolation for unmatched
# ──────────────────────────────────────
unmatched_geoids = nyc_geoids_2023 - matched_geoids
print(f"Need interpolation: {len(unmatched_geoids)}")

if unmatched_geoids:
    cbg_unmatched = cbg_2023[cbg_2023["GEOID"].isin(unmatched_geoids)].copy()
    sld_geo = sld[["GEOID20"] + transit_cols + ["geometry"]].copy()

    overlay = gpd.overlay(cbg_unmatched[["GEOID", "geometry"]], sld_geo, how="intersection")
    overlay["overlap_area"] = overlay.geometry.area

    interpolated_rows = []
    for geoid, g in overlay.groupby("GEOID"):
        total_area = g["overlap_area"].sum()
        if total_area == 0:
            continue
        row = {"GEOID": geoid}
        for col in transit_cols:
            valid = g[g[col].notna()]
            if len(valid) > 0:
                row[col] = (valid[col] * valid["overlap_area"]).sum() / valid["overlap_area"].sum()
            else:
                row[col] = None
        interpolated_rows.append(row)

    interpolated = pd.DataFrame(interpolated_rows)
    print(f"Interpolated: {len(interpolated)}")
else:
    interpolated = pd.DataFrame(columns=["GEOID"] + transit_cols)

# ──────────────────────────────────────
# Combine and save
# ──────────────────────────────────────
result = pd.concat([direct, interpolated], ignore_index=True)
all_cbg = cbg_2023[["GEOID"]].merge(result, on="GEOID", how="left")

all_cbg = all_cbg.rename(columns={"D4A": "transit_freq", "D4B025": "transit_coverage"})

out_path = INTERIM / "transit_cbg.csv"
all_cbg.to_csv(out_path, index=False)

print(f"\nRows: {len(all_cbg)}")
print(f"transit_freq (D4A): NaN={all_cbg['transit_freq'].isna().sum()}, "
      f"mean={all_cbg['transit_freq'].mean():.1f}, median={all_cbg['transit_freq'].median():.1f}")
print(f"transit_coverage (D4B025): NaN={all_cbg['transit_coverage'].isna().sum()}, "
      f"mean={all_cbg['transit_coverage'].mean():.3f}")
print(f"Saved: {out_path}")

# --- self-check ---
_c = pd.read_csv(INTERIM / "transit_cbg.csv")
assert len(_c) == 6807 and (_c["transit_freq"].dropna() >= 0).all()
print("[self-check] 17 transit OK")
