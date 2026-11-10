"""Figure A — adjusted neighbourhood racial-composition associations with LPUR and mean distance.
Two panels (own x-axis/units), aggregate vs joint-composition models labelled separately.
Contrasts reported per IQR increase of each share. British spelling."""
import sys, pandas as pd, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from revlib import load_data, fit_error, OUT, CB, RQ2_PREDS

g = load_data()
DEP_L, DEP_D = "lpur_2.0km", "mean_trip_km"
# common analysis sample: RQ2 complete cases with all race shares AND both outcomes present
race = ["pct_nonwhite", "pct_black", "pct_hispanic", "pct_asian"]
need = [DEP_L, DEP_D, "income_10k", "popdens_k", "park_area_acres", "borough"] + race
mask = g[need].notna().all(axis=1)
gm = g[mask].copy()

AGG = ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]
JOINT = ["income_10k", "pct_black", "pct_hispanic", "pct_asian", "popdens_k", "park_area_acres"]

rows = []
def iqr_stats(col):
    v = gm[col]; p25, p75 = v.quantile(.25), v.quantile(.75); return p25, p75, p75 - p25

for dep, outlab, unit in [(DEP_L, "LPUR (2 km)", "ratio"), (DEP_D, "mean distance", "km")]:
    fA = fit_error(dep, AGG, gm, sample_mask=None)
    fJ = fit_error(dep, JOINT, gm, sample_mask=None)
    sd_out = gm[dep].std()
    for model, fit, preds in [("aggregate", fA, ["pct_nonwhite"]),
                              ("joint composition", fJ, ["pct_black", "pct_hispanic", "pct_asian"])]:
        for pr in preds:
            c = fit["coefs"][pr]; p25, p75, iqr = iqr_stats(pr)
            rows.append(dict(outcome=outlab, outcome_unit=unit, model=model, predictor=pr,
                             coef=c["coef"], se=c["se"], ci_lo=c["ci_lo"], ci_hi=c["ci_hi"],
                             p_exact=c["p"], n=fit["n"], lambda_=fit["lambda_"],
                             p25=p25, p75=p75, iqr=iqr,
                             iqr_contrast=c["coef"] * iqr, iqr_ci_lo=c["ci_lo"] * iqr, iqr_ci_hi=c["ci_hi"] * iqr,
                             outcome_sd=sd_out, iqr_contrast_in_sd=(c["coef"] * iqr) / sd_out))
df = pd.DataFrame(rows)
df.to_csv(OUT / "F1_adjusted_racial_associations.csv", index=False)

# ---- forest plot: two panels ----
labels = {"pct_nonwhite": "Non-White share\n(aggregate model)", "pct_black": "Black share\n(joint model)",
          "pct_hispanic": "Hispanic share\n(joint model)", "pct_asian": "Asian share\n(joint model)"}
order = ["pct_nonwhite", "pct_black", "pct_hispanic", "pct_asian"]
colour = {"aggregate": CB["black"], "joint composition": CB["blue"]}

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
for ax, outlab, unit in [(axes[0], "LPUR (2 km)", "ratio units"), (axes[1], "mean distance", "km")]:
    sub = df[df.outcome == outlab].set_index("predictor")
    ys = list(range(len(order)))[::-1]
    for y, pr in zip(ys, order):
        r = sub.loc[pr]
        col = colour[r["model"]]
        ax.errorbar(r["iqr_contrast"], y, xerr=[[r["iqr_contrast"] - r["iqr_ci_lo"]], [r["iqr_ci_hi"] - r["iqr_contrast"]]],
                    fmt="o", color=col, ecolor=col, capsize=3, ms=6, lw=1.5)
    ax.axvline(0, color=CB["vermillion"], lw=0.9, ls="--")
    ax.set_yticks(ys[::-1]); ax.set_yticklabels([labels[p] for p in order[::-1]], fontsize=8.5)
    ax.set_xlabel(f"Adjusted contrast per IQR increase ({unit})", fontsize=9)
    ax.set_title(f"Panel {'A' if unit.startswith('ratio') else 'B'}: {outlab}", fontsize=10, loc="left")
    ax.tick_params(labelsize=8.5)
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CB["black"], label="Aggregate model (Non-White share)", ms=6),
       Line2D([0], [0], marker="o", color="w", markerfacecolor=CB["blue"], label="Joint composition model (Black/Hispanic/Asian, ref. remainder)", ms=6)]
fig.legend(handles=leg, loc="lower center", ncol=1, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Adjusted associations of neighbourhood racial composition with local park use", fontsize=11)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"F1_adjusted_racial_associations.{ext}", dpi=600, bbox_inches="tight")
plt.close()

_n = int(df.n.iloc[0])
with open(OUT / "F1_caption.txt", "w") as f:
    f.write(
        "Figure A. Adjusted associations of neighbourhood racial composition with the local park-use "
        "ratio (LPUR, 2 km; Panel A) and mean centroid-to-park Euclidean distance (km; Panel B). "
        "Points are ML spatial-error coefficients expressed as the adjusted contrast for an "
        "interquartile-range increase in each composition share, with 95% confidence intervals. "
        "The Non-White share is from an aggregate model; the Black, Hispanic and Asian shares are "
        "from a single joint composition model (non-Hispanic-White plus small non-Hispanic groups "
        "as the omitted reference) and are ecological neighbourhood-level associations, not "
        f"individual-level racial effects. Both panels use the same sample (n = {_n}) and Queen "
        "row-standardised weights; each panel has its own x-axis and units. Covariates: median "
        "household income, population density, park area, and borough fixed effects.\n")

print("=== F1 done ===")
print(df[["outcome", "model", "predictor", "iqr_contrast", "iqr_ci_lo", "iqr_ci_hi", "p_exact", "n", "lambda_"]].round(4).to_string(index=False))
# collinearity note for joint distance model
from numpy.linalg import cond
Xc = gm[["pct_black", "pct_hispanic", "pct_asian"]].values
print("\njoint-model race-share condition number:", round(float(cond(Xc)), 1),
      "| pairwise corr:\n", gm[["pct_black", "pct_hispanic", "pct_asian"]].corr().round(3).to_string())
