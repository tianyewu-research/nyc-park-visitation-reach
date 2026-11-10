"""Figure B (supplementary) — descriptive relationship between LPUR and mean park distance.
Both from the same origin-destination records; relationship is partly constructive. No causal
claim, no naive correlation p-value (spatial dependence). British spelling."""
import sys, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from revlib import load_data, OUT, CB

plt.rcParams["axes.unicode_minus"] = True          # true minus U+2212 in tick labels
MINUS = "\u2212"                                   # U+2212 MINUS SIGN, not hyphen U+002D


def format_signed_value(value, digits=2):
    """Render a number with a real minus sign (U+2212) rather than a hyphen."""
    text = f"{abs(value):.{digits}f}"
    return f"{MINUS}{text}" if value < 0 else text


def format_count(n):
    """Thousands separator only from five digits up: 6576, but 10,000."""
    return str(n) if abs(n) < 10000 else f"{n:,}"


g = load_data()
d = g[["lpur_2.0km", "mean_trip_km"]].dropna()
x = d["lpur_2.0km"].values; y = d["mean_trip_km"].values
n = len(d)
r = pearsonr(x, y)[0]; rho = spearmanr(x, y)[0]

stats = pd.DataFrame([
    dict(variable="lpur_2.0km", n=n, mean=x.mean(), sd=x.std(), p25=np.percentile(x, 25),
         p50=np.percentile(x, 50), p75=np.percentile(x, 75)),
    dict(variable="mean_trip_km", n=n, mean=y.mean(), sd=y.std(), p25=np.percentile(y, 25),
         p50=np.percentile(y, 50), p75=np.percentile(y, 75)),
])
stats["pearson_r"] = round(r, 4); stats["spearman_rho"] = round(rho, 4)
stats.round(4).to_csv(OUT / "F2_lpur_distance_relationship_stats.csv", index=False)

# LOWESS line + bootstrap 95% band
grid = np.linspace(x.min(), x.max(), 100)
base = lowess(y, x, frac=0.3, return_sorted=True)
line = np.interp(grid, base[:, 0], base[:, 1])
np.random.seed(42)
boot = np.empty((200, len(grid)))
for b in range(200):
    idx = np.random.randint(0, n, n)
    lo = lowess(y[idx], x[idx], frac=0.3, return_sorted=True)
    boot[b] = np.interp(grid, lo[:, 0], lo[:, 1])
lo_band, hi_band = np.percentile(boot, [2.5, 97.5], axis=0)

fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")
hb = ax.hexbin(x, y, gridsize=45, cmap="Blues", mincnt=1, linewidths=0.1)
cb = fig.colorbar(hb, ax=ax, shrink=0.8); cb.set_label("Block groups per bin", fontsize=9)
ax.fill_between(grid, lo_band, hi_band, color=CB["vermillion"], alpha=0.25, lw=0)
ax.plot(grid, line, color=CB["vermillion"], lw=2, label="LOWESS (95% bootstrap band)")
ax.set_xlabel("Local park-use ratio (LPUR, 2 km)", fontsize=10)
ax.set_ylabel("Mean centroid-to-park Euclidean distance (km)", fontsize=10)
ax.text(0.97, 0.95,
        f"Pearson $r$ = {format_signed_value(r)}\n"
        f"Spearman $\\rho$ = {format_signed_value(rho)}\n"
        f"$n$ = {format_count(n)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ccc"))
ax.legend(loc="lower left", fontsize=8, frameon=True)
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"F2_lpur_distance_relationship_clean.{ext}", dpi=600,
                bbox_inches="tight", pad_inches=0.02)
plt.close()

with open(OUT / "F2_caption_clean.txt", "w") as f:
    f.write(
        "Figure B (Supplementary). Descriptive relationship between the local park-use ratio "
        "(LPUR, 2 km) and the mean centroid-to-park Euclidean distance (km) across census block "
        f"groups (n = {format_count(n)}). Hexbin shading shows block-group density; the red line is a LOWESS "
        "smoother with a 95% bootstrap band. Pearson r = "
        f"{format_signed_value(r)} and Spearman ρ = {format_signed_value(rho)}. Both indicators are constructed from the same "
        "origin-destination visit distribution, so their relationship is partly constructive; this "
        "figure is shown to display the residual spread between the two measures, not as an "
        "independent validation or a causal test. No ordinary correlation p-value is reported "
        "because spatial dependence violates the independence assumption.\n")

print("=== F2 done ===")
print(stats.round(3).to_string(index=False))
print(f"Pearson r={r:.4f}  Spearman rho={rho:.4f}  n={n}")
