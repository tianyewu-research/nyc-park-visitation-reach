"""CORRECTION TASK 2 — Figure 7 per-IQR rescale (NO refit).

The original Figure 7 is on the theoretical 0-to-1 Non-White-share contrast, whereas the primary
manuscript result is per IQR. Multiply every raw-scale estimate, SE, and CI limit in
robustness_nonradius_tidy.csv by the fixed primary IQR 0.541316 (p values unchanged; clipped-logit NOT
rescaled). Redraw with a per-IQR x-axis."""
import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
BASE = Path("outputs/final_additional_checks_20260807")
OUT = BASE / "corrections_20260807"; FIG = OUT / "figures"
CB = dict(orange="#E69F00", vermillion="#D55E00", black="#000000")
IQR = 0.541316  # fixed primary-sample Non-White IQR (reproduced in PHASE 0)

tidy = pd.read_csv(BASE / "robustness_nonradius_tidy.csv")
for col in ["estimate", "se", "ci_lo", "ci_hi"]:
    tidy[col] = tidy[col] * IQR          # p unchanged
tidy["scale"] = f"per IQR ({IQR:.6f})"
tidy.to_csv(OUT / "robustness_nonradius_IQR_tidy.csv", index=False)

specs = tidy[tidy.effect == "total"].reset_index(drop=True)
ylabels = list(specs["display_label"]); yy = np.arange(len(specs))[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
for ax, eff, title in [(axes[0], "direct", "(a) Average direct impact"),
                       (axes[1], "total", "(b) Average total impact")]:
    dsub = tidy[tidy.effect == eff].set_index("display_label").loc[ylabels].reset_index()
    for y, (_, r) in zip(yy, dsub.iterrows()):
        prim = r["group"] == "Primary"
        col = CB["black"] if prim else CB["orange"]
        ax.errorbar(r["estimate"], y, xerr=[[r["estimate"] - r["ci_lo"]], [r["ci_hi"] - r["estimate"]]],
                    fmt="o", color=col, ecolor=col, capsize=2.6, ms=8 if prim else 6, lw=1.4,
                    zorder=3 if prim else 2)
    grp = specs["group"].tolist()
    for i in range(1, len(grp)):
        if grp[i] != grp[i - 1]:
            ax.axhline(len(grp) - i - 0.5, color="#D9D9D9", lw=0.8, zorder=0)
    ax.axvline(0, color=CB["vermillion"], lw=1.0, ls="--")
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel("Average impact on LPUR per IQR increase in Non-White share", fontsize=10)
    ax.tick_params(labelsize=9)
axes[0].set_yticks(yy); axes[0].set_yticklabels(ylabels, fontsize=9)
fig.suptitle("Robustness of the Non-White-share LPUR association per IQR (non-radius checks, partial SDM)",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
FIG.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(FIG / f"Fig7_grouped_nonradius_robustness_IQR.{ext}", dpi=300, bbox_inches="tight")
plt.close()

prim = specs[specs.group == "Primary"].iloc[0]
cap = (f"Figure 7. Robustness of the neighborhood Non-White-share association with the local park-use "
       f"ratio (LPUR) under the partial spatial Durbin model, expressed per interquartile-range increase "
       f"in Non-White share (IQR = {IQR:.4f}). Panel (a) shows the average direct impact and panel (b) "
       f"the average total impact, each with 95% confidence intervals, across dwell-rule, sample-"
       f"threshold, spatial-weights, local-supply and park-representation specifications; radius "
       f"specifications are shown separately in Figure 6. The primary specification (filled dark marker) "
       f"appears once. Estimates, SEs and confidence limits were rescaled from the 0-to-1 contrast by the "
       f"fixed primary-sample IQR; p values are unchanged. The clipped-logit outcome is on a different "
       f"scale and is not shown here. These are model-implied average impacts, not causal effects.\n")
(OUT / "Fig7_IQR_caption.txt").write_text(cap)

print("=== c2 done ===")
print(f"IQR = {IQR}; primary total per-IQR = {prim['estimate']:.4f} [{prim['ci_lo']:.4f}, {prim['ci_hi']:.4f}]")
print("(sanity vs reproduced primary total per-IQR -0.064606)")
