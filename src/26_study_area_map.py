"""
26_study_area_map.py   (Figure 1: study area)
NYC study area — five boroughs with all mapped parks, for locating the over/under-use zones.

Input:  data/interim/cbg_nyc.gpkg
        data/raw/NYC公园边界/Parks Properties_20260623/
Output: outputs/figures/study_area_map.png
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw"
FIGS = ROOT / "outputs" / "figures"

CRS = "EPSG:2263"
BORO = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island"}

cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg").to_crs(CRS)
cbg["cty"] = cbg["GEOID"].str[2:5]
boro = cbg.dissolve(by="cty").reset_index()

parks = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623").to_crs(CRS)
parks = parks[parks["acres"] >= 0.1]
print(f"Boroughs: {len(boro)}, Parks (>=0.1 ac): {len(parks)}")

fig, ax = plt.subplots(figsize=(9, 10))
boro.plot(ax=ax, facecolor="#f2f2ef", edgecolor="#555555", linewidth=0.8)
parks.plot(ax=ax, facecolor="#2e8b57", edgecolor="none", alpha=0.85)

# borough labels at centroids
for _, r in boro.iterrows():
    c = r.geometry.representative_point()
    ax.annotate(BORO.get(r["cty"], r["cty"]), (c.x, c.y), ha="center", va="center",
                fontsize=12, fontweight="bold", color="#222222",
                path_effects=None)

ax.set_axis_off()
# no in-figure title: the manuscript caption already states this
green = mpatches.Patch(facecolor="#2e8b57", label=f"Parks (n = {len(parks)})")
land = mpatches.Patch(facecolor="#f2f2ef", edgecolor="#555555", label="Borough land area")
ax.legend(handles=[green, land], loc="lower left", frameon=True, fontsize=10)

# simple north arrow + note
ax.annotate("N", xy=(0.96, 0.94), xytext=(0.96, 0.88), xycoords="axes fraction",
            ha="center", fontsize=12, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="black"))
plt.tight_layout()
# original raster settings kept unchanged (dpi=200); "_no_comma" suffix so the
# previous study_area_map.png is never overwritten
plt.savefig(FIGS / "study_area_map_no_comma.png", dpi=200, bbox_inches="tight")
# submission exports of the same figure: 300 dpi raster + vector
plt.savefig(FIGS / "study_area_map_no_comma_300dpi.png", dpi=300, bbox_inches="tight")
plt.savefig(FIGS / "study_area_map_no_comma.pdf", bbox_inches="tight")
plt.savefig(FIGS / "study_area_map_no_comma.svg", bbox_inches="tight")
plt.close()
print(f"Saved: {FIGS/'study_area_map_no_comma.png'}")

# --- self-check ---
for f in ("study_area_map_no_comma.png", "study_area_map_no_comma_300dpi.png",
          "study_area_map_no_comma.pdf", "study_area_map_no_comma.svg"):
    assert (FIGS / f).exists(), f
print("[self-check] 26 study-area map OK")
