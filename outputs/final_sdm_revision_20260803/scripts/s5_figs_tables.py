"""Part 5 — figures and tables (British spelling, colour-blind-safe). Figure A (per-IQR racial-
composition impacts, LPUR + distance, direct vs total), Figure C (SDM robustness forest: direct
& total panels, raw vs logit separated), model-selection table, primary-impact table."""
import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT = Path("outputs/final_sdm_revision_20260803")
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00", grey="#999999", black="#000000")
np.random.seed(42)
# negative tick labels must use U+2212 MINUS SIGN, never U+002D hyphen. This is the
# matplotlib default; set it explicitly so a user matplotlibrc cannot silently flip it.
plt.rcParams["axes.unicode_minus"] = True


def format_count(n):
    """Thousands separator only from five digits up: 6252, but 10,000."""
    return str(int(n)) if abs(int(n)) < 10000 else f"{int(n):,}"


def pnum(p):
    return 0.0002 if str(p).startswith("<") else float(p)

# ---- joint-race distance impacts (for Figure A symmetry) ----
# This model is fitted and validated upstream. Re-fitting a large ML spatial model inside a
# presentation-only script is both slow and a source of platform-specific crashes; consume the
# versioned result table instead.
djr = pd.read_csv(OUT / "distance_joint_race_SDM_impacts.csv")

# ---- assemble Figure A data (per-IQR, direct & total) ----
agg = pd.read_csv(OUT / "RQ2_aggregate_SDM_impacts.csv")
jl = pd.read_csv(OUT / "RQ2_joint_race_SDM_impacts.csv")
dl = pd.read_csv(OUT / "distance_final_impacts.csv")


def grab(df, var, eff, scale_contains, outcome):
    s = df[(df["variable"] == var) & (df["effect"] == eff) & (df["scale"].astype(str).str.contains(scale_contains))]
    if len(s) == 0:
        return None
    r = s.iloc[0]
    return dict(var=var, outcome=outcome, effect=eff, est=float(r["estimate"]), lo=float(r["ci_lo"]), hi=float(r["ci_hi"]))


order = [("pct_nonwhite", "Non-White"), ("pct_black", "Black"), ("pct_hispanic", "Hispanic"), ("pct_asian", "Asian")]
figA = []
for var, lab in order:
    for eff in ["direct", "total"]:
        src_l = agg if var == "pct_nonwhite" else jl
        rL = grab(src_l[src_l.get("weights", "Queen(primary)").astype(str).str.contains("Queen")] if "weights" in src_l else src_l, var, eff, "IQR", "LPUR")
        src_d = dl if var == "pct_nonwhite" else djr
        rD = grab(src_d, var, eff, "IQR", "distance")
        if rL:
            rL["label"] = lab; figA.append(rL)
        if rD:
            rD["label"] = lab; figA.append(rD)
figAdf = pd.DataFrame(figA)
figAdf.to_csv(OUT / "Figure_A_data.csv", index=False)

# ---- Figure A plot ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
labs = [l for _, l in order]
for ax, outcome, unit in [(axes[0], "LPUR", "per-IQR change in LPUR (ratio)"), (axes[1], "distance", "per-IQR change in distance (km)")]:
    sub = figAdf[figAdf.outcome == outcome]
    yb = np.arange(len(labs))[::-1]
    for y, lab in zip(yb, labs):
        for eff, off, mk, fc in [("direct", 0.14, "o", "white"), ("total", -0.14, "o", CB["blue"])]:
            r = sub[(sub.label == lab) & (sub.effect == eff)]
            if len(r):
                r = r.iloc[0]
                ax.errorbar(r["est"], y + off, xerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt=mk,
                            mfc=fc, mec=CB["blue"], ecolor=CB["blue"], capsize=2.5, ms=6, lw=1.2)
    ax.axvline(0, color=CB["vermillion"], lw=0.9, ls="--")
    ax.set_yticks(np.arange(len(labs))); ax.set_yticklabels(labs[::-1], fontsize=9)
    panel = "(a) LPUR (2 km)" if outcome == "LPUR" else "(b) Mean park distance"
    ax.set_xlabel(unit, fontsize=9); ax.set_title(panel, fontsize=10, loc="left", fontweight="bold")
leg = [Line2D([0], [0], marker="o", mfc="white", mec=CB["blue"], color="w", label="Average direct impact", ms=6),
       Line2D([0], [0], marker="o", mfc=CB["blue"], mec=CB["blue"], color="w", label="Average total impact", ms=6)]
fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=[0, 0.03, 1, 1])
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"Figure_A_adjusted_associations_clean.{ext}", dpi=600, bbox_inches="tight")
plt.close()
(OUT / "Figure_A_caption_clean.txt").write_text(
    "Figure A. Adjusted per-interquartile-range (IQR) associations of neighbourhood racial composition with the "
    "local park-use ratio (LPUR, 2 km; left) and the mean centroid-to-park Euclidean distance (km; right), from "
    "partial spatial Durbin models (Queen weights). Open markers are average direct impacts; filled markers are "
    "average total impacts; bars are 95% confidence intervals. Vertical dashed lines mark the null value of "
    "zero, indicating no adjusted association. Non-White is from the aggregate model; Black, "
    "Hispanic and Asian are from the joint composition model (non-Hispanic-White plus small non-Hispanic groups "
    "as the omitted reference). These are ecological neighbourhood-level associations, not individual-level racial "
    "effects, and are not causal spillovers. Structural coefficients are tabulated separately and are not shown as "
    f"effects. Aggregate models included {format_count(6252)} block groups. Joint racial-composition "
    f"models included {format_count(6252)} block groups for LPUR and {format_count(6246)} for mean "
    "distance.\n")

# ---- Figure C robustness forest (direct & total panels; raw vs logit separated) ----
M = pd.read_csv(OUT / "robustness_SDM_master.csv")
raw_all = M[(M.status == "OK") & (M.outcome_scale == "raw")].copy()
lg = M[(M.status == "OK") & (M.outcome_scale == "logit")].copy()
cats = ["radius", "dwell", "sample", "weights", "supply", "representation"]
# Show the primary model once; all other rows change one component relative to it.
primary = raw_all[(raw_all.category == "radius") & raw_all.specification.str.contains("primary")].copy()
alternatives = raw_all[
    raw_all.category.isin(cats) & ~raw_all.specification.str.contains("primary")
].copy()
raw = pd.concat([primary, alternatives], ignore_index=True)
group_label = {
    "radius": "Radius", "dwell": "Dwell rule", "sample": "Sample",
    "weights": "Weights", "supply": "Local supply", "representation": "Park representation"
}
raw["display_label"] = raw.apply(
    lambda r: "Primary specification" if "primary" in str(r["specification"])
    else (
        f"Radius: {str(r['specification']).replace('radius ', '')}"
        if r["category"] == "radius"
        else f"{group_label[r['category']]}: {r['specification']}"
    ), axis=1)
specs = raw[raw.effect == "total"].reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)
ylabels = list(specs["display_label"])
yy = np.arange(len(specs))[::-1]
for ax, eff, title in [(axes[0], "direct", "(a) Average direct impact"), (axes[1], "total", "(b) Average total impact")]:
    dsub = raw[raw.effect == eff].reset_index(drop=True)
    for y, (_, r) in zip(yy, dsub.iterrows()):
        prim = "(primary)" in str(r["specification"])
        col = CB["black"] if prim else CB["orange"]
        ax.errorbar(r["estimate"], y, xerr=[[r["estimate"] - r["ci_lo"]], [r["ci_hi"] - r["estimate"]]], fmt="o",
                    color=col, ecolor=col, capsize=2.2, ms=7 if prim else 5, lw=1.2, zorder=3 if prim else 2)
    # Separate groups without repeating the primary reference row.
    meta = specs[["category"]].reset_index(drop=True)
    for i in range(1, len(meta)):
        if meta.loc[i, "category"] != meta.loc[i - 1, "category"]:
            ax.axhline(len(meta) - i - 0.5, color="#D9D9D9", lw=0.7, zorder=0)
    ax.axvline(0, color=CB["vermillion"], lw=0.9, ls="--")
    ax.set_title(title, fontsize=10, loc="left"); ax.set_xlabel("Non-White share impact on raw LPUR", fontsize=9)
axes[0].set_yticks(yy); axes[0].set_yticklabels(ylabels, fontsize=7.4)
plt.tight_layout()
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"Figure_C_SDM_robustness.{ext}", dpi=600, bbox_inches="tight")
plt.close()
lgt = lg.set_index("effect")
(OUT / "Figure_C_caption.txt").write_text(
    "Figure C. Robustness of the neighbourhood Non-White-share association with the local park-use ratio under the "
    "partial spatial Durbin model. Panel A shows the average DIRECT impact (own-area) and Panel B the average TOTAL "
    "impact (own plus neighbouring areas) of the full 0-to-1 Non-White share, each with 95% confidence intervals, "
    "across radius, dwell, sample, weights, local-supply and park-representation specifications. The primary model "
    "is shown once; every remaining row changes one component relative to it. The 19 raw-scale specifications "
    "comprise the primary model and 18 distinct alternatives. Filled dark markers mark the primary specification. "
    "Direct and total impacts are LeSage-Pace average impacts, not causal effects. "
    "The clipped-logit outcome is on a different scale and is NOT plotted on these axes; its impacts are direct = "
    f"{lgt.loc['direct','estimate']:.3f}, indirect = {lgt.loc['indirect','estimate']:.3f}, total = "
    f"{lgt.loc['total','estimate']:.3f} (logit units; robustness_SDM_master.csv).\n")

# ---- Table: model selection (Queen + kNN panels) ----
cmp = pd.read_csv(OUT / "corrected_model_comparison.csv")
restr = pd.read_csv(OUT / "corrected_restriction_tests.csv")
cmp.to_csv(OUT / "Table_model_selection.csv", index=False)   # already has both weights
# ---- Table: primary SDM impacts (Non-White + income) ----
agg2 = agg[agg["weights"].astype(str).str.contains("Queen")]
prim = agg2[agg2["variable"].isin(["pct_nonwhite", "income_10k"])].copy()
prim = prim[["model", "weights", "variable", "effect", "scale", "estimate", "se", "ci_lo", "ci_hi", "p", "rho", "beta_struct", "theta_struct"]]
prim.to_csv(OUT / "Table_primary_SDM_impacts.csv", index=False)

print("=== s5 figures + tables done ===")
print("Figure A, Figure C (png/svg/pdf), Table_model_selection.csv, Table_primary_SDM_impacts.csv written")
print("distance joint-race per-IQR total:", djr[djr.effect == "total"][["variable", "estimate", "ci_lo", "ci_hi", "p"]].to_string(index=False))
