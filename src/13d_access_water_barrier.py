"""
13d_access_water_barrier.py   (advisor #6b: water-barrier E2SFCA sensitivity)
Euclidean 6 km catchments can place parks across the East/Hudson/Harlem Rivers into a block
group's catchment even where no bridge or tunnel connects them. Without a routing network
(r5py abandoned), we build a conservative barrier variant: a CBG-park pair is dropped when the
two lie in different boroughs AND the straight segment between them leaves the NYC land polygon
(i.e., crosses open water). This treats every water crossing as a barrier — an upper bound on the
correction, since some crossings do follow bridges — and is reported as a robustness column only.

Output: data/interim/access_water_barrier.csv (GEOID, access_score_nowater)
"""
import geopandas as gpd, pandas as pd, numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.prepared import prep
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"; INTERIM = ROOT / "data" / "interim"
CRS = "EPSG:2263"; M2FT = 1 / 0.3048
D0, SIGMA = 6000 * M2FT, 3000 * M2FT
MIN_ACRES = 0.1

cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg").to_crs(CRS)
acs = pd.read_csv(INTERIM / "acs_cbg.csv", dtype={"GEOID": str})
cbg = cbg.merge(acs[["GEOID", "total_pop"]], on="GEOID", how="left")
cbg["total_pop"] = cbg["total_pop"].fillna(0)
cbg_cent = cbg.geometry.centroid
cbg_xy = np.column_stack([cbg_cent.x, cbg_cent.y])
pop = cbg["total_pop"].values
cbg_boro = cbg["borough"].values

# NYC shoreline (LineString layer). A straight CBG-park segment between two on-land points
# "crosses water" iff it intersects the shoreline (going land->water->land). We test this only
# for cross-borough pairs (same-borough pairs are treated as land-connected). CBG geometries
# themselves include water out to jurisdictional limits, so land cannot be derived from them;
# the shoreline layer is the correct barrier source.
shore = gpd.read_file("zip://" + str(RAW / "CBG地理边界" / "NYC_Shoreline_2012_EPSG4326.zip")).to_crs(CRS)
shore_u = shore.geometry.unary_union
shore_p = prep(shore_u)

parks = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623").to_crs(CRS)
parks = parks[parks["acres"] >= MIN_ACRES].reset_index(drop=True)
park_cent = parks.geometry.centroid
park_xy = np.column_stack([park_cent.x, park_cent.y])
area = parks["acres"].values
# borough of each park via spatial join of centroid to borough polygons
boro = cbg.dissolve(by="borough").reset_index()[["borough", "geometry"]]
pj = gpd.sjoin(gpd.GeoDataFrame(geometry=park_cent, crs=CRS), boro, how="left", predicate="within")
park_boro = pj["borough"].reindex(range(len(parks))).values

cbg_tree = cKDTree(cbg_xy); park_tree = cKDTree(park_xy)


def connected(ax, ay, bx, by, same_boro):
    if same_boro:
        return True
    return not shore_p.intersects(LineString([(ax, ay), (bx, by)]))


def w(d):
    return np.exp(-(d ** 2) / (2 * SIGMA ** 2))


# Step 1: R_j = area_j / sum_k pop_k w(d_kj), only land-connected k
R = np.zeros(len(parks))
for j in range(len(parks)):
    idx = cbg_tree.query_ball_point(park_xy[j], D0)
    if not idx:
        continue
    demand = 0.0
    for k in idx:
        if connected(cbg_xy[k, 0], cbg_xy[k, 1], park_xy[j, 0], park_xy[j, 1],
                     cbg_boro[k] == park_boro[j]):
            d = np.hypot(cbg_xy[k, 0] - park_xy[j, 0], cbg_xy[k, 1] - park_xy[j, 1])
            demand += pop[k] * w(d)
    R[j] = area[j] / demand if demand > 0 else 0

# Step 2: A_i = sum_j R_j w(d_ij), only land-connected j
A = np.zeros(len(cbg))
for i in range(len(cbg)):
    idx = park_tree.query_ball_point(cbg_xy[i], D0)
    for j in idx:
        if connected(cbg_xy[i, 0], cbg_xy[i, 1], park_xy[j, 0], park_xy[j, 1],
                     cbg_boro[i] == park_boro[j]):
            d = np.hypot(park_xy[j, 0] - cbg_xy[i, 0], park_xy[j, 1] - cbg_xy[i, 1])
            A[i] += R[j] * w(d)

out = cbg[["GEOID"]].copy()
out["access_score_nowater"] = A
out.to_csv(INTERIM / "access_water_barrier.csv", index=False)
main = pd.read_csv(INTERIM / "access_cbg.csv", dtype={"GEOID": str})[["GEOID", "access_score"]]
cmp = out.merge(main, on="GEOID")
cmp = cmp[(cmp.access_score > 0) & (cmp.access_score_nowater > 0)]
print(f"parks={len(parks)} cbgs={len(cbg)}")
print(f"corr(main, no-water) = {np.corrcoef(np.log(cmp.access_score), np.log(cmp.access_score_nowater))[0,1]:.3f}")
print(f"mean access main={main.access_score.mean():.5f}  no-water={A.mean():.5f}")
print(f"saved {INTERIM/'access_water_barrier.csv'}")
assert len(out) == 6807
print("[self-check] 13d water-barrier access OK")
