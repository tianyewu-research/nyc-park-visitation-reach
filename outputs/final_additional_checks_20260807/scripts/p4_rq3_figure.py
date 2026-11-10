"""PHASE 4 — RQ3 impact figure. Uses the archived final RQ3 impacts (RQ3_final_impacts.csv, Table 5);
no refit. Plots average DIRECT and TOTAL impacts with 95% CIs for all eight standardized predictors.
No significance stars or visual devices — in particular the regional-accessibility (log_access) TOTAL
estimate sits at p=0.049 and must not be visually overstated."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUTDIR, CB, savefig, FINAL
import matplotlib.pyplot as plt

r = pd.read_csv(FINAL / "RQ3_final_impacts.csv")
LABEL = {"log_access": "Regional accessibility (6 km, log)", "quality_index_cbg_svc": "Park quality",
         "walkability_c": "Walkability", "transit_freq": "Transit frequency",
         "park_area_acres": "Park area (local supply)", "popdens_k": "Population density",
         "income_10k": "Median household income", "pct_nonwhite": "Non-White share"}
ORDER = ["pct_nonwhite", "income_10k", "log_access", "quality_index_cbg_svc", "park_area_acres",
         "walkability_c", "transit_freq", "popdens_k"]

plot = r[r.effect.isin(["direct", "total"])].copy()
plot = plot[plot.variable.isin(ORDER)]
plot["label"] = plot.variable.map(LABEL)
plot.to_csv(OUTDIR / "rq3_impacts_plot_data.csv", index=False)

yy = np.arange(len(ORDER))[::-1]
fig, ax = plt.subplots(figsize=(8.6, 5.4))
for eff, off, col, mk, fc in [("direct", 0.16, CB["blue"], "o", CB["blue"]),
                              ("total", -0.16, CB["vermillion"], "s", "white")]:
    for y, var in zip(yy, ORDER):
        row = plot[(plot.variable == var) & (plot.effect == eff)]
        if not len(row):
            continue
        row = row.iloc[0]
        ax.errorbar(row["estimate"], y + off,
                    xerr=[[row["estimate"] - row["ci_lo"]], [row["ci_hi"] - row["estimate"]]],
                    fmt=mk, mfc=fc, mec=col, ecolor=col, capsize=3, ms=6.5, lw=1.3)
ax.axvline(0, color=CB["grey"], lw=1.0, ls="--")
ax.set_yticks(np.arange(len(ORDER))); ax.set_yticklabels([LABEL[v] for v in ORDER[::-1]], fontsize=9.5)
ax.set_xlabel("Average impact on LPUR per 1 SD of the standardized predictor", fontsize=10)
ax.set_title("RQ3 — average direct and total impacts (partial SDM, standardized predictors)",
             fontsize=11, loc="left")
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", mfc=CB["blue"], mec=CB["blue"],
              label="Average direct impact", ms=6.5),
       Line2D([0], [0], marker="s", color="w", mfc="white", mec=CB["vermillion"],
              label="Average total impact", ms=6.5)]
ax.legend(handles=leg, loc="lower left", fontsize=8.5, frameon=True)
ax.text(0.01, -0.16, "Predictors are standardized (z-scored). Bars are 95% CIs. No significance "
        "thresholds are marked; the regional-accessibility total estimate is borderline (p≈0.049).",
        transform=ax.transAxes, fontsize=7, va="top", color="#555")
plt.tight_layout()
p8 = savefig(fig, "Fig8_RQ3_impacts"); plt.close()

print("=== PHASE 4 done ===")
print(f"Fig8: {p8[0]} (+pdf/svg); data: rq3_impacts_plot_data.csv")
la = r[(r.variable == 'log_access') & (r.effect == 'total')].iloc[0]
print(f"regional accessibility TOTAL = {la['estimate']:.4f}, p={la['p']} (rendered without stars)")
