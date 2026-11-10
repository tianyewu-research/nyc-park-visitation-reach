"""CORRECTION TASK 4 (Figure 6) — reword and relabel (NO model rerun).

Redraws Figure 6 from the archived tidy impacts (radius_group_impacts_tidy.csv): 'LPUR catchment radii'
instead of 'neighborhood radii', a shorter title, the 2-km reference label moved off the panel boundary,
and a caption stating connected points are correlated scale specifications (not a dose-response curve).
The underlying estimates are unchanged."""
import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
BASE = Path("outputs/final_additional_checks_20260807")
OUT = BASE / "corrections_20260807"; FIG = OUT / "figures"
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00", grey="#999999",
          black="#000000")
PRIMARY = 2.0
gdf = pd.read_csv(BASE / "radius_group_impacts_tidy.csv")


def profile_ax(ax, sub, colr, title):
    sub = sub.sort_values("radius_km")
    ax.errorbar(sub.radius_km, sub.estimate, yerr=[sub.estimate - sub.ci_lo, sub.ci_hi - sub.estimate],
                fmt="o-", color=colr, ecolor=colr, capsize=2.5, ms=4.5, lw=1.3)
    ax.axhline(0, color=CB["grey"], lw=0.9, ls="--")
    ax.axvline(PRIMARY, color=CB["black"], lw=0.9, ls=":")
    ax.set_title(title, fontsize=9, loc="left"); ax.tick_params(labelsize=8)


races = ["Black", "Hispanic", "Asian"]
colmap = {"Black": CB["blue"], "Hispanic": CB["orange"], "Asian": CB["green"]}
fig, axes = plt.subplots(3, 2, figsize=(9.5, 9.0), sharex=True)
for i, race in enumerate(races):
    for j, eff in enumerate(["direct", "total"]):
        profile_ax(axes[i, j], gdf[(gdf.race == race) & (gdf.effect == eff)], colmap[race],
                   f"{race} — average {eff} impact")
        if i == 2:
            axes[i, j].set_xlabel("LPUR catchment radius (km)", fontsize=9)
        if j == 0:
            axes[i, j].set_ylabel("per-IQR impact on LPUR", fontsize=8.5)
# move the 2-km label to mid-panel (away from the lower boundary)
ymid = np.mean(axes[0, 0].get_ylim())
axes[0, 0].text(PRIMARY + 0.03, ymid, "2 km (primary)", fontsize=7, ha="left", va="center",
                color=CB["black"], rotation=90)
fig.suptitle("Scale sensitivity of racial-composition associations with LPUR", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
FIG.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(FIG / f"Fig6_scale_sensitivity.{ext}", dpi=300, bbox_inches="tight")
plt.close()

cap = ("Figure 6. Scale sensitivity of neighborhood racial-composition associations with the local "
       "park-use ratio (LPUR). Each panel plots the average direct (left column) or total (right column) "
       "impact of Black, Hispanic and Asian share (per interquartile range, joint partial spatial Durbin "
       "model, Queen weights) against the LPUR catchment radius from 0.5 to 3.0 km; the dotted line marks "
       "the prespecified 2-km radius. Bars are 95% confidence intervals. Connected points are CORRELATED "
       "SCALE SPECIFICATIONS of the same measure — a larger catchment mechanically classifies more visits "
       "as local — and must NOT be read as a continuous dose-response relationship. The profiles are shown "
       "to assess stability of sign and significance around 2 km. These are model-implied ecological "
       "associations, not causal effects.\n")
(OUT / "Fig6_scale_sensitivity_caption.txt").write_text(cap)
print("=== c4 (Fig6) done ===  ->", (FIG / "Fig6_scale_sensitivity.png").name)
