"""
14_process_mobile_visits.py

USE MEASURE (rev. 2026-07-17, after advisor magnitude review). The headline measure is
DWELL-QUALIFIED UNIQUE PARK VISITORS per resident:
    use_c = V_c / total_pop,  modelled as log(use_c) in 18_.
where V_c = annual dwell-qualified unique park visitors by residents of CBG c.

Three fixes over the earlier VISIT_COUNTS sum (which gave 2.9B visits/yr = 350/resident, an
order of magnitude too high — advisor red-flag #1):
  1. DWELL FILTER. BUCKETED_DWELL_TIMES shows 83.8% of park "visits" dwell <5 min — geofence
     pass-throughs (walking past/through), not park use (MEDIAN_DWELL median = 2 min). We keep
     only the >=5-min share: q = (visits in 5-20,21-60,61-240,>240 buckets) / all visits, and
     count VISITOR_COUNTS * q as the POI's dwell-qualified unique visitors.
  2. UNIQUE VISITORS. Numerator is VISITOR_COUNTS (unique visitors), not VISIT_COUNTS (total
     visits incl. repeats; VISIT/VISITOR ~= 1.30). Matches the earlier Q1 "唯一设备数" choice.
  3. POI NESTING. Main measure SUMS all park POIs (advisor decision 2026-07-17: sum-per-park,
     retains large parks' genuine multiple sub-destinations e.g. Central Park's 76 POIs).
     Overlapping geofences mean a device can be counted at >1 sub-POI of one park -> potential
     over-count; disclosed in limitations. A max-per-park variant (drops nested POIs, keeps the
     dominant geofence) is produced by 14b_ as robustness.

Counts are Advan-SCALED (population-representative), so no device-panel D_c normalization is
needed. Dividing by total_pop is per-capita (exposure) normalization — separate and required.

Outputs (mobile-side only):
  - V_c            : annual dwell-qualified unique park visitors by residents of CBG c.
  - obs_devices    : mean weekly VISITOR_COUNTS-instances from CBG c (observation-volume proxy
                     for the >=5/10/20 threshold sensitivity; NOT unique devices, double-counts
                     multi-park visitors).
  - weeks_observed
Per-capita use_c = V_c / total_pop and log(use_c) are built in 18_ (needs ACS population).

Input:  data/raw/nypark_2325/2023-*.csv.gz (52 files)
        data/interim/cbg_nyc.gpkg
Output: data/interim/use_cbg.csv
"""

import pandas as pd
import geopandas as gpd
import json
import glob
from pathlib import Path
from collections import defaultdict

DWELL_KEEP = ("5-20", "21-60", "61-240", ">240")   # >=5 min buckets (drop "<5" pass-throughs)


def dwell_qualified_fraction(bucket_json):
    """Share of a POI's visit-time in >=5-min dwell buckets. 1.0 if unparseable (keep all)."""
    if not isinstance(bucket_json, str):
        return 1.0
    try:
        b = json.loads(bucket_json)
    except (json.JSONDecodeError, TypeError):
        return 1.0
    total = sum(b.values())
    if total <= 0:
        return 0.0
    return sum(b.get(k, 0) for k in DWELL_KEEP) / total

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

NYC_FIPS_PREFIXES = ("36005", "36047", "36061", "36081", "36085")

cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg")
nyc_geoids = set(cbg["GEOID"].values)
print(f"NYC CBGs: {len(nyc_geoids)}")

files_2023 = sorted(glob.glob(str(RAW / "nypark_2325" / "2023-*.csv.gz")))
print(f"Weekly files: {len(files_2023)}")

cbg_V_annual = defaultdict(float)
cbg_D_weekly = defaultdict(list)

for i, fpath in enumerate(files_2023):
    df = pd.read_csv(fpath, usecols=["POI_CBG", "VISITOR_HOME_CBGS",
                                     "VISITOR_COUNTS", "BUCKETED_DWELL_TIMES"])

    df["POI_CBG_str"] = df["POI_CBG"].dropna().astype(int).astype(str).str.zfill(12)
    df = df[df["POI_CBG_str"].str[:5].isin(NYC_FIPS_PREFIXES)]

    week_devices = defaultdict(float)

    for _, row in df.iterrows():
        vhc = row["VISITOR_HOME_CBGS"]
        if pd.isna(vhc):
            continue

        try:
            home_dict = json.loads(vhc)
        except (json.JSONDecodeError, TypeError):
            continue

        # dwell-qualified unique visitors for this POI-week
        visitors = row["VISITOR_COUNTS"] if pd.notna(row["VISITOR_COUNTS"]) else 0
        q = dwell_qualified_fraction(row["BUCKETED_DWELL_TIMES"])
        qualified_visitors = visitors * q

        home_total = sum(home_dict.values())
        if home_total <= 0:
            continue

        for home_cbg, device_count in home_dict.items():
            if home_cbg.startswith("CA:"):
                continue
            home_cbg = home_cbg.zfill(12)
            if home_cbg not in nyc_geoids:
                continue

            cbg_V_annual[home_cbg] += qualified_visitors * (device_count / home_total)
            week_devices[home_cbg] += device_count

    for geoid, dev in week_devices.items():
        cbg_D_weekly[geoid].append(dev)

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Processed {i+1}/{len(files_2023)} weeks")

print(f"CBGs with data: {len(cbg_V_annual)}")

rows = []
for geoid in nyc_geoids:
    V_c = cbg_V_annual.get(geoid, 0)
    weekly_devs = cbg_D_weekly.get(geoid, [])
    obs_devices = sum(weekly_devs) / len(weekly_devs) if weekly_devs else 0
    rows.append({
        "GEOID": geoid,
        "V_c": round(V_c, 2),
        "obs_devices": round(obs_devices, 2),   # observation-volume proxy (NOT unique devices)
        "weeks_observed": len(weekly_devs),
    })

result = pd.DataFrame(rows)
result.to_csv(INTERIM / "use_cbg.csv", index=False)

has_data = result["V_c"] > 0
print(f"\nRows: {len(result)}")
print(f"CBGs with data: {has_data.sum()} ({100*has_data.mean():.1f}%)")
print(f"V_c: mean={result.loc[has_data,'V_c'].mean():.1f}, median={result.loc[has_data,'V_c'].median():.1f}")
print(f"obs_devices (volume proxy): median={result.loc[has_data,'obs_devices'].median():.1f}")
print(f"obs_devices < 5: {result['obs_devices'].between(0.01,5,inclusive='right').sum()} | "
      f"< 10: {result['obs_devices'].between(0.01,10,inclusive='right').sum()} | "
      f"< 20: {result['obs_devices'].between(0.01,20,inclusive='right').sum()} CBGs")
print("use_c = V_c/total_pop and log(use_c) are computed in 18_.")
print(f"Saved: {INTERIM / 'use_cbg.csv'}")

# --- self-check ---
_c = pd.read_csv(INTERIM / "use_cbg.csv")
assert len(_c) == 6807 and (_c["V_c"] >= 0).all() and (_c["obs_devices"] >= 0).all()
print("[self-check] 14 mobile use OK")
