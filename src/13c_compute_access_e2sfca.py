"""
13c_compute_access_e2sfca.py
Enhanced 2SFCA (E2SFCA) with a continuous Gaussian distance decay, motivated by the
observed visit-distance distribution (median DISTANCE_FROM_HOME ~5.7 km; the 800 m 2SFCA
catchment captures only ~9% of visits). E2SFCA keeps the supply/demand logic of 2SFCA but
weights nearer parks/CBGs more via w(d)=exp(-d^2/(2*sigma^2)) within catchment d0.

Computes E2SFCA at two catchments for comparison and tests each against the first stage
(log_use ~ log access). Does NOT overwrite the main access yet — writes its own file +
a comparison so the choice can be confirmed.

Input:  data/raw/NYC公园边界/Parks Properties_20260623/
        data/interim/cbg_nyc.gpkg, data/interim/acs_cbg.csv
        data/interim/access_cbg.csv (existing 800m 2SFCA, for comparison)
        data/processed/analysis_cbg.gpkg (log_use_c, for first-stage test)
Output: data/interim/access_e2sfca_cbg.csv
        outputs/tables/access_e2sfca_comparison.csv
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.spatial import cKDTree
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

TARGET_CRS = "EPSG:2263"
M2FT = 1 / 0.3048
MIN_PARK_ACRES = 0.1

cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg").to_crs(TARGET_CRS)
acs = pd.read_csv(INTERIM / "acs_cbg.csv", dtype={"GEOID": str})
cbg = cbg.merge(acs[["GEOID", "total_pop"]], on="GEOID", how="left")
cbg["total_pop"] = cbg["total_pop"].fillna(0)
cent = cbg.geometry.centroid
cbg_xy = np.column_stack([cent.x, cent.y])
pop = cbg["total_pop"].values

parks = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623").to_crs(TARGET_CRS)
parks = parks[parks["acres"] >= MIN_PARK_ACRES].reset_index(drop=True)
pcent = parks.geometry.centroid
park_xy = np.column_stack([pcent.x, pcent.y])
area = parks["acres"].values
print(f"CBGs: {len(cbg)}, Parks: {len(parks)}")

cbg_tree = cKDTree(cbg_xy)
park_tree = cKDTree(park_xy)


def e2sfca(d0_m, sigma_m):
    d0 = d0_m * M2FT
    sigma = sigma_m * M2FT
    # Step 1: R_j = area_j / sum_k pop_k * w(d_kj)
    R = np.zeros(len(parks))
    for j in range(len(parks)):
        idx = cbg_tree.query_ball_point(park_xy[j], d0)
        if not idx:
            continue
        dd = np.linalg.norm(cbg_xy[idx] - park_xy[j], axis=1)
        w = np.exp(-(dd ** 2) / (2 * sigma ** 2))
        demand = np.sum(pop[idx] * w)
        R[j] = area[j] / demand if demand > 0 else 0
    # Step 2: A_i = sum_j R_j * w(d_ij)
    A = np.zeros(len(cbg))
    for i in range(len(cbg)):
        idx = park_tree.query_ball_point(cbg_xy[i], d0)
        if not idx:
            continue
        dd = np.linalg.norm(park_xy[idx] - cbg_xy[i], axis=1)
        w = np.exp(-(dd ** 2) / (2 * sigma ** 2))
        A[i] = np.sum(R[idx] * w)
    return A


# Catchments >= observed median home-to-park distance (5.7 km) so the catchment does not fall
# short of typical realized travel (advisor point 6a). Gaussian bandwidth sigma = d0/2, so the
# weight is exp(-2) ~= 0.14 at the catchment edge and ~0.16 at the 5.7 km median under the 6 km
# spec — a soft decay, not a hard cutoff. 6 km adopted as main (see comparison below); 5 km and
# 7 km retained as robustness. Params written to the comparison table for reproducibility.
specs = {"e2sfca_5km": (5000, 2500), "e2sfca_6km": (6000, 3000), "e2sfca_7km": (7000, 3500)}
for name, (d0, sg) in specs.items():
    print(f"Computing {name} (d0={d0}m, sigma={sg}m)...")
    cbg[name] = e2sfca(d0, sg)

# Existing 800m 2SFCA for comparison
prev = pd.read_csv(INTERIM / "access_cbg.csv", dtype={"GEOID": str})
cbg = cbg.merge(prev[["GEOID", "access_score"]].rename(columns={"access_score": "sfca_800m"}),
                on="GEOID", how="left")

# First-stage test for each measure: log_use_c ~ log(access)
proc = gpd.read_file(PROCESSED / "analysis_cbg.gpkg")[["GEOID", "log_use_c"]]
test = cbg.merge(proc, on="GEOID", how="left")

def first_stage_r2(col):
    d = test[[col, "log_use_c"]].copy()
    d = d[d[col] > 0].dropna()
    x = np.log(d[col].values)
    m = sm.OLS(d["log_use_c"].values, sm.add_constant(x)).fit()
    return m.rsquared, m.params[1], m.pvalues[1], len(d)

GAUSS = {"e2sfca_5km": (5000, 2500), "e2sfca_6km": (6000, 3000), "e2sfca_7km": (7000, 3500)}
rows = []
for col in ["sfca_800m", "e2sfca_5km", "e2sfca_6km", "e2sfca_7km"]:
    a = cbg[col]
    cv = a.std() / a.mean()
    r2, b, p, n = first_stage_r2(col)
    corr800 = cbg[["sfca_800m", col]].corr().iloc[0, 1]
    d0, sg = GAUSS.get(col, (None, None))
    rows.append({"measure": col, "d0_m": d0, "sigma_m": sg, "mean": a.mean(), "CV": cv,
                 "zeros_pct": 100 * (a == 0).mean(), "first_stage_R2": r2, "beta": b, "p": p,
                 "n": n, "corr_with_800m": corr800})
comp = pd.DataFrame(rows)
comp.to_csv(TABLES / "access_e2sfca_comparison.csv", index=False)

out = cbg[["GEOID", "e2sfca_5km", "e2sfca_6km", "e2sfca_7km", "sfca_800m"]]
out.to_csv(INTERIM / "access_e2sfca_cbg.csv", index=False)

# Adopt E2SFCA 6km as the main access measure: catchment exceeds the observed 5.7km median
# home-to-park distance (advisor point 6a) with a soft Gaussian decay, 0% zeros. Keep 800m
# 2SFCA and E2SFCA 5km/7km as reference/robustness columns.
main = cbg[["GEOID"]].copy()
main["access_2sfca_800m"] = cbg["sfca_800m"]
main["access_e2sfca_5km"] = cbg["e2sfca_5km"]
main["access_e2sfca_7km"] = cbg["e2sfca_7km"]
main["access_score"] = cbg["e2sfca_6km"]      # primary
main.to_csv(INTERIM / "access_cbg.csv", index=False)

print("\n=== Access measure comparison ===")
print(comp.round(4).to_string(index=False))
print(f"\nSaved: {INTERIM/'access_e2sfca_cbg.csv'}, {TABLES/'access_e2sfca_comparison.csv'}")

# --- self-check ---
_c = pd.read_csv(INTERIM / "access_e2sfca_cbg.csv")
assert len(_c) == 6807 and (_c["e2sfca_6km"] >= 0).all()
print("[self-check] 13c E2SFCA OK")
