from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


NEW = Path("outputs/destination_per_cbg_and_match_alignment_20260808")
OLD = Path("outputs/final_additional_checks_20260807")
OUT = Path("outputs/manuscript_figures/figures")
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "black": "#111111",
    "grey": "#777777",
    "lightgrey": "#d9d9d9",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.titlesize": 10.2,
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": True,   # negative ticks use U+2212, never U+002D
})


def save(fig, stem):
    for ext, dpi in [("png", 600), ("pdf", None), ("svg", None)]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Figure 2: concise, internally consistent workflow for later Illustrator refinement.
fig, ax = plt.subplots(figsize=(8.25, 5.7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

row_y = [0.83, 0.59, 0.35, 0.11]
row_labels = ["DATA SOURCES", "PROCESSING", "MEASURES /\nSAMPLES", "ANALYSIS"]
fills = ["#E9F2F8", "#FBF0E2", "#E8F4ED", "#EEEAF7"]
for y, lab in zip(row_y, row_labels):
    ax.text(0.012, y + 0.015, lab, ha="left", va="center", fontsize=8.4,
            fontweight="bold", color="#52616d")


def box(x, y, w, h, text, fill, fs=8.2):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
                           linewidth=0.8, edgecolor="#52616d", facecolor=fill)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#1f2933", linespacing=1.16)
    return (x, y, w, h)


def arrow(src, dst, sx=0.5, dx=0.5):
    x1 = src[0] + src[2] * sx
    y1 = src[1]
    x2 = dst[0] + dst[2] * dx
    y2 = dst[1] + dst[3]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6.5,
                                 linewidth=0.65, color="#657786", shrinkA=2, shrinkB=2))


xs4 = [0.14, 0.355, 0.57, 0.785]
w4 = 0.19
src = [
    box(xs4[0], 0.775, w4, 0.12, "Advan Weekly Patterns\n52 weeks, 2023\npark POIs + home CBGs", fills[0]),
    box(xs4[1], 0.775, w4, 0.12, "NYC Parks + ParkServe\nproperties, supply, quality", fills[0]),
    box(xs4[2], 0.775, w4, 0.12, "ACS 2019-2023\nincome, composition, density", fills[0]),
    box(xs4[3], 0.775, w4, 0.12, "EPA SLD 2021\nwalkability + transit", fills[0]),
]

proc = [
    box(0.14, 0.525, 0.25, 0.13, "Dwell-qualified attribution\nvisitors x dwell share x\nhome-CBG share; POI distance", fills[1]),
    box(0.425, 0.525, 0.25, 0.13, "Park processing\n1,675 properties; canonical\n100-ft POI crosswalk; park measures", fills[1]),
    box(0.71, 0.525, 0.25, 0.13, "Neighborhood harmonization\nACS + EPA measures to\n2023 CBGs (n = 6,807)", fills[1]),
]
arrow(src[0], proc[0])
arrow(src[0], proc[1], sx=0.72, dx=0.18)
arrow(src[1], proc[1])
arrow(src[1], proc[2], sx=0.8, dx=0.18)
arrow(src[2], proc[2], sx=0.45, dx=0.55)
arrow(src[3], proc[2], sx=0.45, dx=0.82)

meas = [
    box(0.14, 0.285, 0.25, 0.125, "Park-use outcomes\nLPUR within 2 km + mean distance\n(n = 6,576)", fills[2]),
    box(0.425, 0.285, 0.25, 0.125, "RQ2 common sample\ncomposition + income + controls\n(n = 6,252)", fills[2]),
    box(0.71, 0.285, 0.25, 0.125, "RQ3 complete cases\npark + built environment\n(n = 6,227)", fills[2]),
]
arrow(proc[0], meas[0])
arrow(proc[0], meas[1], sx=0.7, dx=0.18)
arrow(proc[1], meas[1])
arrow(proc[1], meas[2], sx=0.8, dx=0.2)
arrow(proc[2], meas[1], sx=0.25, dx=0.82)
arrow(proc[2], meas[2])

ana = [
    box(0.14, 0.035, 0.19, 0.125, "RQ1\nGlobal Moran's I\nBH-FDR LISA", fills[3]),
    box(0.355, 0.035, 0.19, 0.125, "RQ2\npartial SDM impacts\n+ destination concentration", fills[3]),
    box(0.57, 0.035, 0.19, 0.125, "RQ3\nstandardized partial SDM\ncorrelate impacts", fills[3]),
    box(0.785, 0.035, 0.19, 0.125, "Robustness\n11 correlated radii +\nconstruction/model checks", fills[3]),
]
arrow(meas[0], ana[0])
arrow(meas[1], ana[1])
arrow(meas[2], ana[2])
arrow(meas[1], ana[3], sx=0.76, dx=0.22)
arrow(meas[2], ana[3], sx=0.64, dx=0.72)
fig.subplots_adjust(left=0.01, right=0.995, bottom=0.01, top=0.995)
save(fig, "Figure2_workflow_0808")


# Figure 7: aggregate and group-specific radius profiles.
agg = pd.read_csv(NEW / "aggregate_nonwhite_radius_SDM_impacts.csv")
grp = pd.read_csv(OLD / "radius_group_impacts_tidy.csv")
grp = grp[grp["effect"].eq("total")]

fig, axes = plt.subplots(2, 2, figsize=(8.25, 6.0), sharex=True, sharey=True)
axes = axes.ravel()

ax = axes[0]
for effect, color, marker, label in [
    ("direct", COL["blue"], "o", "Direct"),
    ("total", COL["black"], "s", "Total"),
]:
    d = agg[agg["effect"].eq(effect)].sort_values("radius_km")
    ax.errorbar(
        d["radius_km"], d["estimate"],
        yerr=[d["estimate"] - d["ci_lo"], d["ci_hi"] - d["estimate"]],
        fmt=marker, color=color, ecolor=color, mfc="white" if effect == "direct" else color,
        mec=color, markersize=3.8, capsize=2.2, elinewidth=0.85, lw=0,
        label=label,
    )
ax.set_title("(a) Aggregate Non-White share", loc="left")
ax.legend(frameon=False, loc="lower left", ncol=2, handletextpad=0.35, columnspacing=0.8)

for ax, race, panel, color in zip(
    axes[1:], ["Black", "Hispanic", "Asian"], ["b", "c", "d"],
    ["#2C7FB8", COL["orange"], COL["green"]],
):
    d = grp[grp["race"].eq(race)].sort_values("radius_km")
    ax.errorbar(
        d["radius_km"], d["estimate"],
        yerr=[d["estimate"] - d["ci_lo"], d["ci_hi"] - d["estimate"]],
        fmt="o", color=color, ecolor=color, markersize=3.8, capsize=2.2,
        elinewidth=0.85, lw=0,
    )
    ax.set_title(f"({panel}) {race} share: total impact", loc="left")

for ax in axes:
    ax.axhline(0, color="#8c8c8c", lw=0.8, ls="--", zorder=0)
    ax.axvline(2.0, color="#4d4d4d", lw=0.8, ls=":", zorder=0)
    ax.set_xlim(0.4, 3.1)
    ax.set_ylim(-0.115, 0.052)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.grid(axis="y", color="#ededed", lw=0.55, zorder=0)
axes[0].set_ylabel("Average LPUR impact per IQR")
axes[2].set_ylabel("Average LPUR impact per IQR")
fig.supxlabel("LPUR catchment radius (km)", y=0.02)
fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.98, hspace=0.31, wspace=0.17)
save(fig, "Figure7_radius_sensitivity_0808")


# Figure 8: non-radius robustness with the canonical 100-ft representation rows.
rob = pd.read_csv(NEW / "figure8_replacement_tidy.csv")
rob = rob[rob["effect"].isin(["direct", "total"])].copy()
order = [
    "Primary specification",
    "Dwell rule: dwell >=21 min",
    "Dwell rule: no dwell filter",
    "Sample threshold: weekly visitors >=20",
    "Sample threshold: weekly visitors >=5",
    "Spatial weights: kNN6",
    "Local supply: + park acreage within 2 km",
    "Park representation: conservative property-level maximum",
    "Park representation: matched municipal-park universe",
]
labels = {
    "Primary specification": "Primary (2 km)",
    "Dwell rule: dwell >=21 min": "Dwell \u2265 21 min",
    "Dwell rule: no dwell filter": "No dwell filter",
    "Sample threshold: weekly visitors >=20": "Weekly visitors \u2265 20",
    "Sample threshold: weekly visitors >=5": "Weekly visitors \u2265 5",
    "Spatial weights: kNN6": "kNN6 weights",
    "Local supply: + park acreage within 2 km": "+ park area within 2 km",
    "Park representation: conservative property-level maximum": "Property-level maximum",
    "Park representation: matched municipal parks": "Matched municipal parks",
    "Park representation: matched municipal-park universe": "Matched municipal parks",
}
rob["row"] = rob["display_label"].map({x: i for i, x in enumerate(order)})

fig, axes = plt.subplots(1, 2, figsize=(8.25, 4.75), sharey=True)
for ax, effect, panel in zip(axes, ["direct", "total"], ["a", "b"]):
    d = rob[rob["effect"].eq(effect)].sort_values("row")
    for _, row in d.iterrows():
        if row["display_label"] == "Primary specification":
            color = COL["black"]
        elif row["universe"] == "canonical 100ft":
            color = COL["vermillion"]
        else:
            color = COL["orange"]
        ax.errorbar(
            row["estimate"], row["row"],
            xerr=[[row["estimate"] - row["ci_lo"]], [row["ci_hi"] - row["estimate"]]],
            fmt="o", color=color, ecolor=color, capsize=2.3, markersize=4.5, lw=1.0,
        )
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    ax.set_title(f"({panel}) Average {effect} impact", loc="left")
    ax.grid(axis="y", color="#ededed", lw=0.55)
    for sep in [0.5, 2.5, 4.5, 5.5, 6.5]:
        ax.axhline(sep, color="#e2e2e2", lw=0.6, zorder=0)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([labels[x] for x in order])
axes[0].invert_yaxis()
fig.supxlabel("Average impact on LPUR per IQR increase in Non-White share", y=0.092)
legend = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["black"], markeredgecolor=COL["black"], label="Primary"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["orange"], markeredgecolor=COL["orange"], label="Other checks"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["vermillion"], markeredgecolor=COL["vermillion"], label="Canonical 100-ft representation"),
]
fig.legend(handles=legend, frameon=False, loc="lower center", bbox_to_anchor=(0.60, 0.008), ncol=3,
           handletextpad=0.35, columnspacing=0.9)
fig.subplots_adjust(left=0.34, right=0.99, bottom=0.225, top=0.96, wspace=0.14)
save(fig, "Figure8_robustness_0808")


# Figure 9: per-CBG destination concentration plus pooled size composition.
conc = pd.read_csv(NEW / "destination_concentration_per_cbg.csv", dtype={"home_cbg": str})
rec = pd.read_csv(NEW / "destination_records_property_level_canonical_100ft.csv", dtype={"home_cbg": str})
qorder = ["Q1 (lowest Non-White)", "Q2", "Q3", "Q4 (highest Non-White)"]
qlab = ["Q1", "Q2", "Q3", "Q4"]


def box_panel(ax, column, ylabel, title):
    arrays = [conc.loc[conc["origin_quartile"].eq(q), column].dropna().to_numpy() for q in qorder]
    bp = ax.boxplot(arrays, positions=np.arange(4), widths=0.56, showfliers=False, patch_artist=True,
                    medianprops={"color": COL["black"], "lw": 1.5})
    for box in bp["boxes"]:
        box.set(facecolor=COL["sky"], alpha=0.55, edgecolor=COL["blue"])
    for i, vals in enumerate(arrays):
        jitter = (np.random.default_rng(42 + i).random(len(vals)) - 0.5) * 0.22
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=1.7, color=COL["grey"], alpha=0.12,
                   edgecolors="none", rasterized=True, zorder=1)
        ax.text(i, 0.012, f"n = {len(vals)}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6.5, color="#555555")
    ax.set_xticks(range(4), qlab)
    ax.set_xlabel("Origin Non-White-share quartile")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color="#ededed", lw=0.55)


fig = plt.figure(figsize=(8.25, 6.15))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78], hspace=0.43, wspace=0.28)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
box_panel(ax1, "top10_share", "Within-CBG top-10 share", "(a) Top-10 non-local destination share")
box_panel(ax2, "effective_destinations", "Within-CBG effective destinations (1/HHI)",
          "(b) Effective non-local destinations")


def size_class(acres):
    if acres < 1:
        return "Very small (<1 ac)"
    if acres < 5:
        return "Small (1\u20135 ac)"
    if acres < 20:
        return "Medium (5\u201320 ac)"
    if acres < 100:
        return "Large (20\u2013100 ac)"
    return "Very large (\u2265 100 ac)"


nwmap = conc.set_index("home_cbg")["origin_quartile"].to_dict()
rn = rec[rec["non_local"]].copy()
rn["origin_quartile"] = rn["home_cbg"].map(nwmap)
rn["size_class"] = rn["dest_acres"].map(size_class)
sc = rn.groupby(["origin_quartile", "size_class"])["attributed_weight"].sum().unstack(fill_value=0).reindex(qorder)
sc = sc.div(sc.sum(axis=1), axis=0)
classes = ["Very small (<1 ac)", "Small (1\u20135 ac)", "Medium (5\u201320 ac)",
           "Large (20\u2013100 ac)", "Very large (\u2265 100 ac)"]
colors = [COL["sky"], COL["blue"], COL["green"], COL["orange"], COL["vermillion"]]
bottom = np.zeros(4)
for cls, color in zip(classes, colors):
    vals = sc[cls].to_numpy()
    ax3.bar(np.arange(4), vals, bottom=bottom, width=0.62, color=color, label=cls)
    bottom += vals
ax3.set_xticks(range(4), qlab)
ax3.set_ylim(0, 1)
ax3.set_xlabel("Origin Non-White-share quartile")
ax3.set_ylabel("Pooled non-local attributed-weight share")
ax3.set_title("(c) Destination-size composition (pooled descriptive summary)", loc="left")
ax3.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3,
           handlelength=1.25, columnspacing=0.9)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.98)
save(fig, "Figure9_destination_patterns_0808")

print("Wrote revised Figures 7-9 to", OUT)
