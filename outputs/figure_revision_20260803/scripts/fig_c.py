"""Figure C — unified robustness forest for the Non-White-share coefficient on LPUR.
All specifications re-estimated here (ML spatial error, RQ2 covariates). Raw-scale specs share one
x-axis; the clipped-logit outcome is on a different scale and is shown in a separate inset panel.
British spelling. Property-level maximum = conservative property-level aggregation (NOT true
visitor-level de-duplication; aggregated data have no device/person identifiers)."""
import sys, pandas as pd, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from revlib import load_data, fit_error, OUT, CB, RQ2_PREDS

g = load_data()
g["wk_visitors"] = g["total_visits"] / 52.0
CLIP = 0.005
g["logit_lpur"] = np.log(g["lpur_2.0km"].clip(CLIP, 1 - CLIP) / (1 - g["lpur_2.0km"].clip(CLIP, 1 - CLIP)))
NW = "pct_nonwhite"


def spec(category, specification, dep, preds=RQ2_PREDS, weights="queen", mask=None,
         outcome="LPUR (2 km)", outcome_scale="raw", radius_km=2.0, dwell_rule="dwell >=5 min",
         sample_rule="none", supply_control="baseline", park_representation="POI sum",
         transform="raw", is_primary=False):
    f = fit_error(dep, preds, g, weights=weights, sample_mask=mask)
    c = f["coefs"][NW]
    return dict(category=category, specification=specification, outcome=outcome, outcome_scale=outcome_scale,
                radius_km=radius_km, dwell_rule=dwell_rule, sample_rule=sample_rule, weights=weights,
                supply_control=supply_control, park_representation=park_representation, transform=transform,
                n=f["n"], coef=c["coef"], se=c["se"], ci_lo=c["ci_lo"], ci_hi=c["ci_hi"],
                p_exact=c["p"], lambda_=f["lambda_"], is_primary=is_primary)


rows = []
# 1. Radius 0.5-3.0 every 0.25
for rk in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
    col = f"lpur_{rk}km"
    rows.append(spec("radius", f"radius {rk} km", col, radius_km=rk, is_primary=(rk == 2.0)))
# 2. Dwell
rows.append(spec("dwell", "no dwell filter", "lpur_none", dwell_rule="none"))
rows.append(spec("dwell", "dwell >=5 min (primary)", "lpur_ge5", dwell_rule="dwell >=5 min", is_primary=True))
rows.append(spec("dwell", "dwell >=21 min", "lpur_ge21", dwell_rule="dwell >=21 min"))
# 3. Sample restriction
rows.append(spec("sample", "no minimum volume (primary)", "lpur_2.0km", sample_rule="none", is_primary=True))
rows.append(spec("sample", "weekly visitors >=5", "lpur_2.0km", mask=(g.wk_visitors >= 5), sample_rule=">=5/wk"))
rows.append(spec("sample", "weekly visitors >=20", "lpur_2.0km", mask=(g.wk_visitors >= 20), sample_rule=">=20/wk"))
# 4. Weights
rows.append(spec("weights", "Queen contiguity (primary)", "lpur_2.0km", weights="queen", is_primary=True))
rows.append(spec("weights", "six-nearest-neighbour", "lpur_2.0km", weights="knn"))
# 5. Local supply
rows.append(spec("supply", "baseline controls (primary)", "lpur_2.0km", supply_control="baseline", is_primary=True))
rows.append(spec("supply", "+ park acreage within 2 km", "lpur_2.0km", preds=RQ2_PREDS + ["park_area_2km"],
                 supply_control="+2km park acreage"))
# 6. Park-destination representation
rows.append(spec("representation", "original POI sum (primary)", "lpur_A_poisum", park_representation="POI sum", is_primary=True))
rows.append(spec("representation", "matched municipal-park universe", "lpur_B_matched", park_representation="matched universe"))
rows.append(spec("representation", "conservative property-level maximum", "lpur_C_property_max", park_representation="property-level maximum"))
# 7. Outcome transform (logit on separate scale)
rows.append(spec("transform", "raw LPUR (primary)", "lpur_2.0km", transform="raw", outcome_scale="raw", is_primary=True))
rows.append(spec("transform", "clipped logit LPUR [0.005,0.995]", "logit_lpur", transform="logit",
                 outcome="logit LPUR", outcome_scale="logit"))

M = pd.DataFrame(rows)
M.to_csv(OUT / "F3_robustness_master.csv", index=False)

# ---------- plot: Panel A radius, Panel B (dwell/sample/weights/supply/representation), Panel C logit inset ----------
raw = M[M.outcome_scale == "raw"].copy()
radA = raw[raw.category == "radius"].sort_values("radius_km")
panelB_cats = ["dwell", "sample", "weights", "supply", "representation"]
B = raw[raw.category.isin(panelB_cats)].copy()

fig = plt.figure(figsize=(13, 6.8))
gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.25, 0.42], wspace=0.9)
axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1]); axC = fig.add_subplot(gs[2])

# Panel A: radius — plot at y=i (ascending: 0.5 km bottom, 3.0 km top); labels match points
radA = radA.reset_index(drop=True)
for i, r in radA.iterrows():
    col = CB["black"] if r.is_primary else CB["sky"]
    ms = 8 if r.is_primary else 5
    axA.errorbar(r.coef, i, xerr=[[r.coef - r.ci_lo], [r.ci_hi - r.coef]], fmt="o", color=col,
                 ecolor=col, capsize=2.5, ms=ms, lw=1.3, zorder=3 if r.is_primary else 2)
axA.axvline(0, color=CB["vermillion"], lw=0.9, ls="--")
axA.set_yticks(range(len(radA))); axA.set_yticklabels([f"{r} km" for r in radA.radius_km], fontsize=8.5)
axA.set_title("Panel A: neighbourhood radius", fontsize=10, loc="left")
axA.set_xlabel("Non-White share coefficient (raw LPUR)", fontsize=9)

# Panel B: grouped by category, category order top->bottom
cat_order = ["dwell", "sample", "weights", "supply", "representation"]
B = B.set_index("category").loc[cat_order].reset_index()
labels, ypos = [], []
y = len(B) - 1
for i, r in B.iterrows():
    labels.append(r.specification); ypos.append(y)
    col = CB["black"] if r.is_primary else CB["orange"]
    ms = 8 if r.is_primary else 5
    axB.errorbar(r.coef, y, xerr=[[r.coef - r.ci_lo], [r.ci_hi - r.coef]], fmt="o", color=col,
                 ecolor=col, capsize=2.5, ms=ms, lw=1.3, zorder=3 if r.is_primary else 2)
    y -= 1
axB.axvline(0, color=CB["vermillion"], lw=0.9, ls="--")
axB.set_yticks(ypos); axB.set_yticklabels(labels, fontsize=8)
axB.set_ylim(-0.6, len(B) - 0.4)
axB.set_title("Panel B: dwell / sample / weights / supply / representation", fontsize=10, loc="left")
axB.set_xlabel("Non-White share coefficient (raw LPUR)", fontsize=9)

# share x-lims across A and B
xl = min(raw.ci_lo.min(), 0) * 1.05; xr = max(raw.ci_hi.max(), 0) * 1.05
axA.set_xlim(xl, xr); axB.set_xlim(xl, xr)

# Panel C: logit inset (separate scale)
lg = M[M.outcome_scale == "logit"].iloc[0]
axC.errorbar(lg.coef, 0, xerr=[[lg.coef - lg.ci_lo], [lg.ci_hi - lg.coef]], fmt="o",
             color=CB["purple"], ecolor=CB["purple"], capsize=3, ms=7)
axC.axvline(0, color=CB["vermillion"], lw=0.9, ls="--")
axC.set_yticks([0]); axC.set_yticklabels(["clipped logit\nLPUR"], fontsize=8)
axC.set_title("Panel C: logit scale\n(separate x-axis)", fontsize=9, loc="left")
axC.set_xlabel("coef (logit)", fontsize=9)

fig.suptitle("Robustness of the Non-White-share association with the local park-use ratio\n"
             "(ML spatial-error models; filled dark = primary specification; 95% CI)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"F3_robustness_forest.{ext}", dpi=600, bbox_inches="tight")
plt.close()

with open(OUT / "F3_caption.txt", "w") as f:
    f.write(
        "Figure C. Robustness of the adjusted association between neighbourhood Non-White share and "
        "the local park-use ratio (LPUR). Each point is the Non-White-share coefficient from an ML "
        "spatial-error model with the RQ2 covariates (median household income, population density, "
        "park area, borough fixed effects) and 95% confidence interval. Panel A varies the "
        "neighbourhood radius defining LPUR from 0.5 to 3.0 km (2.0 km primary). Panel B varies the "
        "dwell-time rule, a minimum weekly-visitor sample restriction, the spatial weights (Queen "
        "vs six-nearest-neighbour), a local-supply control (park acreage within 2 km), and the "
        "park-destination representation (original POI sum; matched municipal-park universe; and a "
        "conservative property-level maximum, which bounds within-park overlap but is NOT true "
        "visitor-level de-duplication, since the aggregated data carry no device or person "
        "identifiers). Filled dark symbols mark the primary specification. Panel C shows the "
        "clipped-logit outcome on its own scale because its coefficient is not comparable to the "
        "raw-ratio coefficients. Full statistics: F3_robustness_master.csv.\n")

print("=== F3 done: %d specifications ===" % len(M))
print(M[["category", "specification", "coef", "ci_lo", "ci_hi", "p_exact", "n", "outcome_scale"]].round(4).to_string(index=False))
