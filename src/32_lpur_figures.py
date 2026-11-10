"""
32_lpur_figures.py   (new-framework figures, 2026-07-28)
  A. LPUR LISA cluster map (high vs low local reliance) with borough outlines
  B. LPUR chorophleth (continuous 2km local ratio)
  C. Specification curves: income coef (null) + non-white coef (robust) vs radius
  D. Racial-decomposition coefficient plot (LPUR ~ each group, spatial error)
Outputs -> outputs/figures/lpur_*.png
"""
import geopandas as gpd, pandas as pd, numpy as np, scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from libpysal.weights import Queen
from esda.moran import Moran_Local
import warnings
from pathlib import Path

warnings.filterwarnings("ignore"); np.random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs" / "figures"; TAB = ROOT / "outputs" / "tables"
g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg").to_crs("EPSG:2263")

BORO = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island"}
g["_cty"] = g["GEOID"].str[2:5]
boro = g.dissolve(by="_cty").reset_index()


def add_boro(ax, labels=True):
    """Borough outlines; `labels=False` draws boundaries only (Fig. 3 keeps the
    outlines but drops the names, which sat on top of the CBG fills)."""
    boro.boundary.plot(ax=ax, color="#333", linewidth=0.7)
    if not labels:
        return
    for _, r in boro.iterrows():
        c = r.geometry.representative_point()
        ax.annotate(BORO.get(r["_cty"], ""), (c.x, c.y), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#111")

# ---------- A. LISA map of LPUR ----------
d = g.dropna(subset=["lpur_2.0km"]).reset_index(drop=True)
w = Queen.from_dataframe(d, use_index=False); w.transform = "r"
lm = Moran_Local(d["lpur_2.0km"].values, w, permutations=999); sig = lm.p_sim < 0.05
lab = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
d["cl"] = "ns"
for c, nm in lab.items():
    d.loc[sig & (lm.q == c), "cl"] = nm
colors = {"HH": "#1a9641", "LL": "#d7191c", "HL": "#fdae61", "LH": "#abd9e9", "ns": "#e8e8e8"}
txt = {"HH": "High–High (high local reliance)", "LL": "Low–Low (travel out for parks)",
       "HL": "High–Low outlier", "LH": "Low–High outlier", "ns": "Not significant"}
fig, ax = plt.subplots(figsize=(10, 10))
for c in ["HH", "LL", "HL", "LH", "ns"]:
    sub = d[d.cl == c]
    if len(sub):
        sub.plot(ax=ax, color=colors[c], edgecolor="white", linewidth=0.05)
add_boro(ax)
ax.set_title("Local park-use reliance clusters (LPUR, 2 km; Queen, 999 perm, p < 0.05)", fontsize=13)
ax.axis("off")
h = [mpatches.Patch(facecolor=colors[c], edgecolor="#999", label=f"{txt[c]} (n={int((d.cl==c).sum())})")
     for c in ["HH", "LL", "HL", "LH", "ns"]]
ax.legend(handles=h, loc="upper left", fontsize=9, frameon=True)
plt.tight_layout(); plt.savefig(FIG / "lpur_lisa_map.png", dpi=200, bbox_inches="tight"); plt.close()

# ---------- B. LPUR choropleth ----------
fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
g.plot(ax=ax, color="#f5f5f5", edgecolor="none")
d.plot(column="lpur_2.0km", ax=ax, cmap="viridis", edgecolor="white", linewidth=0.05,
       legend=True, legend_kwds={"label": "LPUR (share of visits within 2 km)",
                                 "shrink": 0.5, "pad": 0.04})
# boundaries only: no in-map borough names (Figure 1 already locates them), and no
# in-figure title (the Figure 3 caption carries it)
add_boro(ax, labels=False)
ax.axis("off")
plt.savefig(FIG / "lpur_choropleth_clean.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG / "lpur_choropleth_clean.pdf", bbox_inches="tight")
plt.savefig(FIG / "lpur_choropleth_clean.svg", bbox_inches="tight")
plt.close()

# ---------- C. Specification curves ----------
inc = pd.read_csv(TAB / "lpur_spec_curve.csv")
rac = pd.read_csv(TAB / "lpur_spec_curve_race.csv").rename(columns={"r": "radius_km"})
# race curve needs CI; recompute se from p? we stored coef+p; approximate se via coef/z
from scipy.stats import norm
rac["z"] = rac["p"].apply(lambda p: norm.ppf(1 - p / 2))
rac["se"] = (rac["nonwhite_coef"].abs() / rac["z"]).replace([np.inf, -np.inf], np.nan)
rac["lo"] = rac["nonwhite_coef"] - 1.96 * rac["se"]; rac["hi"] = rac["nonwhite_coef"] + 1.96 * rac["se"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
for ax, df, col, lo, hi, ttl, yl in [
    (axes[0], inc, "income_coef", "lo", "hi", "Income (per $10k): null across all radii", "Coefficient on LPUR"),
    (axes[1], rac, "nonwhite_coef", "lo", "hi", "Share non-white: negative and significant throughout", "Coefficient on LPUR")]:
    ax.axhline(0, color="red", lw=0.8, ls="--")
    ax.fill_between(df["radius_km"], df[lo], df[hi], alpha=0.2, color="#1f77b4")
    ax.plot(df["radius_km"], df[col], "-o", color="#1f77b4")
    ax.axvline(2.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Neighborhood radius (km)"); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=11)
fig.suptitle("Specification curves: SES gradients in local park-use ratio across neighborhood radius\n(spatial error model, 95% CI; dotted line = 2 km primary)", fontsize=12, y=1.02)
plt.tight_layout(); plt.savefig(FIG / "lpur_spec_curves.png", dpi=200, bbox_inches="tight"); plt.close()

# ---------- D. Racial-decomposition coefficient plot ----------
rd = pd.read_csv(TAB / "lpur_racial_decomp.csv")
name = {"pct_black": "% Black", "pct_hispanic": "% Hispanic", "pct_asian": "% Asian",
        "pct_white_nh": "% White (NH)", "pct_nonwhite": "% non-white"}
rd = rd[rd.group.isin(["pct_black", "pct_hispanic", "pct_asian", "pct_white_nh"])].copy()
rd["z"] = rd["p"].apply(lambda p: norm.ppf(1 - p / 2))
rd["se"] = (rd["coef"].abs() / rd["z"]).replace([np.inf, -np.inf], np.nan)
rd["lab"] = rd.group.map(name); rd = rd.sort_values("coef")
fig, ax = plt.subplots(figsize=(8, 5))
ax.axvline(0, color="red", lw=0.8, ls="--")
ax.errorbar(rd["coef"], range(len(rd)), xerr=1.96 * rd["se"], fmt="o", color="black", ecolor="grey", capsize=3)
ax.set_yticks(range(len(rd))); ax.set_yticklabels(rd["lab"])
ax.set_xlabel("Coefficient on LPUR (2 km), per 1-unit share, one group at a time")
ax.set_title("Racial patterning of local park-use reliance\n(each group vs remainder; spatial error model, 95% CI)", fontsize=11)
plt.tight_layout(); plt.savefig(FIG / "lpur_racial_decomp.png", dpi=200, bbox_inches="tight"); plt.close()

print("Saved: lpur_lisa_map.png, lpur_choropleth_clean.png, lpur_spec_curves.png, lpur_racial_decomp.png")
for f in ["lpur_lisa_map", "lpur_choropleth_clean", "lpur_spec_curves", "lpur_racial_decomp"]:
    assert (FIG / f"{f}.png").exists()
for f in ["lpur_choropleth_clean.pdf", "lpur_choropleth_clean.svg"]:
    assert (FIG / f).exists(), f
print("[self-check] 32 LPUR figures OK")
