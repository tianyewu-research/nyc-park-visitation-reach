"""
11_process_acs.py
Clean ACS data: total population, median household income, race composition.
Input:  data/raw/ACS人口 : 收入 : 种族数据/ACSDT5Y2023.B01003_*/
        data/raw/ACS人口 : 收入 : 种族数据/ACSDT5Y2023.B19013_*/
        data/raw/ACS人口 : 收入 : 种族数据/ACSDT5Y2023.B02001_*/
        data/interim/cbg_nyc.gpkg
Output: data/interim/acs_cbg.csv
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "ACS人口 : 收入 : 种族数据"
INTERIM = ROOT / "data" / "interim"

NYC_FIPS_PREFIXES = ("36005", "36047", "36061", "36081", "36085")


def read_acs_table(folder_pattern, value_col):
    folder = next(RAW.glob(folder_pattern))
    data_file = next(folder.glob("*-Data.csv"))
    df = pd.read_csv(data_file, dtype=str)
    df = df.iloc[1:]  # skip description row
    df["GEOID"] = df["GEO_ID"].str.split("US").str[1]
    df = df[df["GEOID"].str[:5].isin(NYC_FIPS_PREFIXES)].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df[["GEOID", value_col]]


pop = read_acs_table("ACSDT5Y2023.B01003_*", "B01003_001E")
pop = pop.rename(columns={"B01003_001E": "total_pop"})

# Income: TRACT level downscaled to CBG (advisor 2026-06-26). CBG B19013 had 21.9% missing;
# tract B19013 has 6.4%. Each CBG inherits its parent tract's median income (GEOID first 11 digits).
tract_folder = next(RAW.glob("ACSDT5Y2023.B19013_TRACT_*"))
tinc = pd.read_csv(next(tract_folder.glob("*-Data.csv")), dtype=str).iloc[1:]
tinc["tract"] = tinc["GEO_ID"].str.split("US").str[1]
tinc = tinc[tinc["tract"].str[:5].isin(NYC_FIPS_PREFIXES)].copy()
tinc["inc"] = pd.to_numeric(tinc["B19013_001E"], errors="coerce")
tinc.loc[tinc["inc"] < 0, "inc"] = None
tract_income = tinc.set_index("tract")["inc"]

# pct_nonwhite uses B03002 (Hispanic-or-Latino by race) so that Hispanic residents who report
# White alone are counted as non-white: pct_nonwhite = 1 - (Not-Hispanic-White-alone / total).
# This is the standard EJ definition and differs from the earlier B02001 "1 - White alone",
# which classified Hispanic whites as white (see advisor comment, 2026-07-26). The old B02001
# variant is retained as pct_nonwhite_b02001 for an appendix comparison.
race_file = next(RAW.glob("**/ACSDT5Y2023.B03002-Data.csv"))
race = pd.read_csv(race_file, dtype=str).iloc[1:]
race["GEOID"] = race["GEO_ID"].str.split("US").str[1]
race = race[race["GEOID"].str[:5].isin(NYC_FIPS_PREFIXES)].copy()
race_total = pd.to_numeric(race["B03002_001E"], errors="coerce")
race_nhwhite = pd.to_numeric(race["B03002_003E"], errors="coerce")
race["pct_nonwhite"] = 1 - (race_nhwhite / race_total)
# racial subgroup shares (of total population) for the RQ2 decomposition
race["pct_white_nh"] = race_nhwhite / race_total
race["pct_black"] = pd.to_numeric(race["B03002_004E"], errors="coerce") / race_total
race["pct_asian"] = pd.to_numeric(race["B03002_006E"], errors="coerce") / race_total
race["pct_hispanic"] = pd.to_numeric(race["B03002_012E"], errors="coerce") / race_total
# old B02001 definition for appendix comparison
b02_folder = next(RAW.glob("ACSDT5Y2023.B02001_*"))
b02 = pd.read_csv(next(b02_folder.glob("*-Data.csv")), dtype=str).iloc[1:]
b02["GEOID"] = b02["GEO_ID"].str.split("US").str[1]
b02t = pd.to_numeric(b02["B02001_001E"], errors="coerce")
b02w = pd.to_numeric(b02["B02001_002E"], errors="coerce")
b02["pct_nonwhite_b02001"] = 1 - (b02w / b02t)
race = race.merge(b02[["GEOID", "pct_nonwhite_b02001"]], on="GEOID", how="left")
race = race[["GEOID", "pct_nonwhite", "pct_nonwhite_b02001",
             "pct_white_nh", "pct_black", "pct_asian", "pct_hispanic"]]

acs = pop.merge(race, on="GEOID", how="outer")

cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg")[["GEOID", "land_area_sqm"]]
acs = cbg[["GEOID"]].merge(acs, on="GEOID", how="left")

# Downscale tract income to each CBG via its parent tract (GEOID first 11 digits).
acs["median_hh_income"] = acs["GEOID"].str[:11].map(tract_income)

# Population density per sq km, using TIGER land area (excludes water).
# All-water CBGs (land_area_sqm == 0) have no population and get NaN density.
land_km2 = cbg.set_index("GEOID")["land_area_sqm"] / 1e6
acs["pop_density"] = acs["total_pop"] / acs["GEOID"].map(land_km2)
acs.loc[acs["GEOID"].map(land_km2) == 0, "pop_density"] = None

acs = acs[["GEOID", "total_pop", "median_hh_income", "pct_nonwhite", "pct_nonwhite_b02001",
           "pct_white_nh", "pct_black", "pct_asian", "pct_hispanic", "pop_density"]]

out_path = INTERIM / "acs_cbg.csv"
acs.to_csv(out_path, index=False)

print(f"Rows: {len(acs)}")
print(f"total_pop: NaN={acs['total_pop'].isna().sum()}, zero={( acs['total_pop']==0).sum()}, mean={acs['total_pop'].mean():.0f}")
print(f"median_hh_income: NaN={acs['median_hh_income'].isna().sum()}, mean={acs['median_hh_income'].mean():.0f}")
print(f"pct_nonwhite: NaN={acs['pct_nonwhite'].isna().sum()}, range=[{acs['pct_nonwhite'].min():.3f}, {acs['pct_nonwhite'].max():.3f}]")
print(f"pop_density (per sq km): NaN={acs['pop_density'].isna().sum()}, "
      f"mean={acs['pop_density'].mean():.0f}, median={acs['pop_density'].median():.0f}, "
      f"max={acs['pop_density'].max():.0f}")
print(f"Saved: {out_path}")

# --- self-check ---
_c = pd.read_csv(INTERIM / "acs_cbg.csv")
assert len(_c) == 6807
assert _c["pct_nonwhite"].dropna().between(0, 1).all()
assert _c["median_hh_income"].isna().mean() < 0.10
print("[self-check] 11 ACS OK")
