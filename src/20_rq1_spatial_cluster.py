"""
20_rq1_spatial_cluster.py   (RQ1: is mismatch spatially clustered?)
Queen contiguity weights -> Global Moran's I (999 perm) -> LISA (HH/LL/HL/LH).
Robustness: kNN (k=6) weights. Maps: LISA cluster map + 2x2 access-use quadrant map.

Input:  data/processed/analysis_cbg_final.gpkg
Output: outputs/tables/morans_i_global.csv
        outputs/figures/lisa_cluster_map.png
        outputs/figures/mismatch_quadrant_map.png
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libpysal.weights import Queen, KNN
from esda.moran import Moran, Moran_Local
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)
SEED = 42
PERM = 999

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Load and restrict to CBGs with a valid mismatch value ──
cbg_all = gpd.read_file(PROCESSED / "analysis_cbg_final.gpkg")
gdf = cbg_all.dropna(subset=["mismatch_c"]).reset_index(drop=True).copy()
print(f"CBGs with mismatch: {len(gdf)} (dropped {cbg_all['mismatch_c'].isna().sum()} missing)")

# ── Build Queen weights; drop islands (disconnected polygons) ──
w_queen = Queen.from_dataframe(gdf, use_index=False)
if w_queen.islands:
    print(f"Queen islands (no contiguity neighbor): {len(w_queen.islands)} -> dropped")
    keep = [i for i in range(len(gdf)) if i not in set(w_queen.islands)]
    gdf = gdf.iloc[keep].reset_index(drop=True)
    w_queen = Queen.from_dataframe(gdf, use_index=False)
w_queen.transform = "r"
print(f"Queen weights: n={w_queen.n}, avg neighbors={w_queen.mean_neighbors:.1f}")

y = gdf["mismatch_c"].values

# ── Global Moran's I (Queen + kNN robustness) ──
mi_queen = Moran(y, w_queen, permutations=PERM)

w_knn = KNN.from_dataframe(gdf, k=6)
w_knn.transform = "r"
mi_knn = Moran(y, w_knn, permutations=PERM)

global_tbl = pd.DataFrame([
    {"weights": "Queen contiguity", "n": w_queen.n, "morans_I": mi_queen.I,
     "expected_I": mi_queen.EI, "z_score": mi_queen.z_sim, "p_value": mi_queen.p_sim},
    {"weights": "kNN (k=6)", "n": w_knn.n, "morans_I": mi_knn.I,
     "expected_I": mi_knn.EI, "z_score": mi_knn.z_sim, "p_value": mi_knn.p_sim},
])
global_tbl.to_csv(TABLES / "morans_i_global.csv", index=False)
print("\n=== Global Moran's I ===")
print(global_tbl.round(4).to_string(index=False))

# ── LISA (Local Moran's I) on Queen ──
lisa = Moran_Local(y, w_queen, permutations=PERM)
sig = lisa.p_sim < 0.05
# quadrant codes from esda: 1 HH, 2 LH, 3 LL, 4 HL
labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
gdf["lisa_cluster"] = ["ns"] * len(gdf)
for code, name in labels.items():
    gdf.loc[sig & (lisa.q == code), "lisa_cluster"] = name

print("\n=== LISA clusters (p<0.05) ===")
print(gdf["lisa_cluster"].value_counts().to_string())

# borough outlines + labels (shared by both maps)
import matplotlib.patches as mpatches
BORO = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island"}
_ba = cbg_all.copy()
_ba["_cty"] = _ba["GEOID"].str[2:5]
boro = _ba.dissolve(by="_cty").reset_index()


def add_boroughs(ax):
    boro.boundary.plot(ax=ax, color="#333333", linewidth=0.7)
    for _, r in boro.iterrows():
        c = r.geometry.representative_point()
        ax.annotate(BORO.get(r["_cty"], ""), (c.x, c.y), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#111111")


# ── Map 1: LISA cluster map ──
lisa_colors = {"HH": "#d7191c", "LL": "#2c7bb6", "HL": "#fdae61", "LH": "#abd9e9", "ns": "#e8e8e8"}
lisa_lab = {"HH": "High–High (over-use hotspot)", "LL": "Low–Low (under-use coldspot)",
            "HL": "High–Low (spatial outlier)", "LH": "Low–High (spatial outlier)",
            "ns": "Not significant"}
fig, ax = plt.subplots(figsize=(10, 10))
for cat, color in lisa_colors.items():
    sub = gdf[gdf["lisa_cluster"] == cat]
    if len(sub):
        sub.plot(ax=ax, color=color, edgecolor="white", linewidth=0.05)
add_boroughs(ax)
ax.set_title("LISA clusters of access–use mismatch (Queen, 999 permutations, p < 0.05)", fontsize=13)
ax.axis("off")
handles = [mpatches.Patch(facecolor=lisa_colors[c], edgecolor="#999",
           label=f"{lisa_lab[c]} (n={int((gdf['lisa_cluster']==c).sum())})")
           for c in ["HH", "LL", "HL", "LH", "ns"]]
ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)
plt.tight_layout()
plt.savefig(FIGS / "lisa_cluster_map.png", dpi=200, bbox_inches="tight")
plt.close()

# ── Map 2: 2x2 access-use quadrant map ──
quad_colors = {"HH": "#1a9641", "HL": "#d7191c", "LH": "#2c7bb6", "LL": "#fdae61"}
quad_label = {"HH": "High access / High use", "HL": "High access / Low use (underused)",
              "LH": "Low access / High use", "LL": "Low access / Low use"}
fig, ax = plt.subplots(figsize=(10, 10))
cbg_all.plot(ax=ax, color="#f5f5f5", edgecolor="none")  # base incl. block groups with no data
for cat, color in quad_colors.items():
    sub = gdf[gdf["mismatch_quad"] == cat]
    if len(sub):
        sub.plot(ax=ax, color=color, edgecolor="white", linewidth=0.05)
add_boroughs(ax)
ax.set_title("Access–use quadrants (split at medians of access and use)", fontsize=13)
ax.axis("off")
qhandles = [mpatches.Patch(facecolor=quad_colors[c], edgecolor="#999",
            label=f"{quad_label[c]} (n={int((gdf['mismatch_quad']==c).sum())})")
            for c in ["HH", "HL", "LH", "LL"]]
qhandles.append(mpatches.Patch(facecolor="#f5f5f5", edgecolor="#999", label="No park-use data"))
ax.legend(handles=qhandles, loc="upper left", fontsize=9, frameon=True)
plt.tight_layout()
plt.savefig(FIGS / "mismatch_quadrant_map.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\nSaved tables: {TABLES/'morans_i_global.csv'}")
print(f"Saved figures: {FIGS/'lisa_cluster_map.png'}, {FIGS/'mismatch_quadrant_map.png'}")

# --- self-check ---
_t = pd.read_csv(TABLES / "morans_i_global.csv")
assert len(_t) == 2 and (_t["p_value"] < 0.05).all()
print("[self-check] 20 RQ1 OK")
