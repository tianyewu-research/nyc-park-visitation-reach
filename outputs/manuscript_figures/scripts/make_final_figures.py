from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs/final_additional_checks_20260807")
CORR = ROOT / "corrections_20260807"
OUT = Path("outputs/manuscript_figures/figures")
OUT.mkdir(parents=True, exist_ok=True)
print(f"Writing figures to {OUT}", flush=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": True,   # negative ticks use U+2212, never U+002D
    }
)


def save(fig, stem):
    print(f"Saving {stem}", flush=True)
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}", flush=True)


# Figure 7: total impacts only. Direct impacts are already shown at 2 km in Figure 6.
radius = pd.read_csv(ROOT / "radius_group_impacts_tidy.csv")
radius = radius[radius["effect"].eq("total")]
colors = {"Black": "#2C7FB8", "Hispanic": "#E69500", "Asian": "#009E73"}
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.7), sharex=True, sharey=True)
for ax, race in zip(axes, ["Black", "Hispanic", "Asian"]):
    d = radius[radius["race"].eq(race)].sort_values("radius_km")
    ax.axhline(0, color="#8c8c8c", lw=0.9, ls="--", zorder=0)
    ax.axvline(2.0, color="#4d4d4d", lw=0.9, ls=":", zorder=0)
    ax.errorbar(
        d["radius_km"],
        d["estimate"],
        yerr=[d["estimate"] - d["ci_lo"], d["ci_hi"] - d["estimate"]],
        fmt="o-",
        color=colors[race],
        ecolor=colors[race],
        lw=1.5,
        elinewidth=1.0,
        capsize=2.5,
        markersize=4,
    )
    ax.set_title(race)
    ax.set_xlim(0.4, 3.1)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
axes[0].set_ylabel("Average total impact on LPUR per IQR")
fig.supxlabel("LPUR catchment radius (km)", y=0.01, fontsize=10)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "Figure7_radius_sensitivity")


# Figure 8: non-radius robustness checks, with compact labels and separate panel scales.
rob = pd.read_csv(CORR / "robustness_nonradius_IQR_tidy.csv")
rob = rob[rob["effect"].isin(["direct", "total"])].copy()
label_map = {
    "Primary specification": "Primary (2 km)",
    "Dwell rule: dwell >=21 min": "Dwell ≥21 min",
    "Dwell rule: no dwell filter": "No dwell filter",
    "Sample threshold: weekly visitors >=20": "Weekly visitors ≥20",
    "Sample threshold: weekly visitors >=5": "Weekly visitors ≥5",
    "Spatial weights: kNN6": "kNN6 weights",
    "Local supply: + park acreage within 2 km": "+ park area within 2 km",
    "Park representation: conservative property-level maximum": "Property-level maximum",
    "Park representation: matched municipal-park universe": "Matched municipal parks",
}
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
rob["display"] = rob["display_label"].map(label_map)
rob["row"] = rob["display_label"].map({v: i for i, v in enumerate(order)})
fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.1), sharey=True)
for ax, effect, panel in zip(axes, ["direct", "total"], ["a", "b"]):
    d = rob[rob["effect"].eq(effect)].sort_values("row")
    y = np.arange(len(d))
    point_colors = ["#111111" if x == "Primary specification" else "#D98200" for x in d["display_label"]]
    for yi, (_, row), color in zip(y, d.iterrows(), point_colors):
        ax.errorbar(
            row["estimate"], yi,
            xerr=[[row["estimate"] - row["ci_lo"]], [row["ci_hi"] - row["estimate"]]],
            fmt="o", color=color, ecolor=color, capsize=2.5, markersize=5, lw=1.2,
        )
    ax.axvline(0, color="#777777", lw=0.9, ls="--")
    ax.set_title(f"({panel}) Average {effect} impact", loc="left")
    ax.set_xlabel("LPUR impact per IQR")
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
axes[0].set_yticks(np.arange(len(order)))
axes[0].set_yticklabels([label_map[x] for x in order])
axes[0].invert_yaxis()
fig.subplots_adjust(left=0.36, right=0.99, bottom=0.11, top=0.95, wspace=0.18)
save(fig, "Figure8_robustness")


# Figure 9: descriptive non-local destination structure.
metrics = pd.read_csv(CORR / "destination_metrics_corrected.csv")
metrics = metrics[metrics["scope"].eq("non_local_gt2km")].copy()
metrics["q"] = ["Q1", "Q2", "Q3", "Q4"]
size = pd.read_csv(CORR / "destination_size_class_shares_corrected.csv")
size["q"] = size["origin_quartile"].str.extract(r"(Q[1-4])", expand=False)
size["size_label"] = size["size_class"].str.replace(r"^\d+\s+", "", regex=True)
size_order = ["very small (<1 ac)", "small (1-5 ac)", "medium (5-20 ac)", "large (20-100 ac)", "very large (>=100 ac)"]
palette = ["#56B4E9", "#0072B2", "#009E73", "#E69F00", "#D55E00"]

fig = plt.figure(figsize=(8.4, 6.0))
grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.35)
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
x = np.arange(4)
axes[0].errorbar(
    x, metrics["top10_share"],
    yerr=[metrics["top10_share"] - metrics["top10_share_ci_lo"], metrics["top10_share_ci_hi"] - metrics["top10_share"]],
    fmt="o-", color="#0072B2", capsize=2.5, lw=1.5, markersize=4.5,
)
axes[0].set_title("(a) Top-10 destination share", loc="left")
axes[0].set_ylabel("Share of non-local attributed weight")
axes[0].set_xticks(x, metrics["q"])
axes[0].set_xlabel("Origin Non-White-share quartile")
axes[0].grid(axis="y", color="#eeeeee", lw=0.6)

axes[1].errorbar(
    x, metrics["effective_destinations"],
    yerr=[metrics["effective_destinations"] - metrics["effective_destinations_ci_lo"], metrics["effective_destinations_ci_hi"] - metrics["effective_destinations"]],
    fmt="s-", color="#D55E00", capsize=2.5, lw=1.5, markersize=4.5,
)
axes[1].set_title("(b) Effective destinations", loc="left")
axes[1].set_ylabel("Effective number (1/HHI)")
axes[1].set_xticks(x, metrics["q"])
axes[1].set_xlabel("Origin Non-White-share quartile")
axes[1].grid(axis="y", color="#eeeeee", lw=0.6)

bottom = np.zeros(4)
for lab, color in zip(size_order, palette):
    vals = size[size["size_label"].eq(lab)].set_index("q").reindex(["Q1", "Q2", "Q3", "Q4"])["share"].to_numpy()
    axes[2].bar(x, vals, bottom=bottom, color=color, width=0.72, label=lab)
    bottom += vals
axes[2].set_title("(c) Park-size composition", loc="left")
axes[2].set_ylabel("Share of non-local attributed weight")
axes[2].set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
axes[2].set_xlabel("Origin Non-White-share quartile")
axes[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, handlelength=1.4)
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.96)
save(fig, "Figure9_destination_patterns")


# Figure 10: RQ3 impacts. Flag accessibility in the label rather than in a footnote.
rq3 = pd.read_csv(ROOT / "rq3_impacts_plot_data.csv")
label_order = [
    "Non-White share", "Median household income", "Regional accessibility (6 km, log)",
    "Park quality", "Park area (local supply)", "Walkability", "Transit frequency", "Population density",
]
short = {
    "Regional accessibility (6 km, log)": "Regional accessibility*",
    "Park area (local supply)": "Local park area",
}
rq3["ord"] = rq3["label"].map({x: i for i, x in enumerate(label_order)})
fig, ax = plt.subplots(figsize=(7.4, 4.9))
offsets = {"direct": -0.10, "total": 0.10}
styles = {
    "direct": ("#0072B2", "o", "Average direct impact"),
    "total": ("#D55E00", "s", "Average total impact"),
}
for effect in ["direct", "total"]:
    d = rq3[rq3["effect"].eq(effect)].sort_values("ord")
    y = d["ord"].to_numpy() + offsets[effect]
    color, marker, label = styles[effect]
    ax.errorbar(
        d["estimate"], y,
        xerr=[d["estimate"] - d["ci_lo"], d["ci_hi"] - d["estimate"]],
        fmt=marker, mfc="white" if effect == "total" else color, mec=color,
        color=color, ecolor=color, capsize=2.5, lw=1.2, markersize=5.5, label=label,
    )
ax.axvline(0, color="#777777", lw=0.9, ls="--")
ax.set_yticks(np.arange(len(label_order)))
ax.set_yticklabels([short.get(x, x) for x in label_order])
ax.invert_yaxis()
ax.set_xlabel("Average impact on LPUR per 1 SD")
ax.grid(axis="y", color="#eeeeee", lw=0.6)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.64, 1.08), ncol=2)
ax.text(0.0, -0.14, "* Model-sensitive because the local term and spatial lag are nearly collinear.", transform=ax.transAxes, fontsize=7.5)
fig.subplots_adjust(left=0.30, right=0.98, bottom=0.20, top=0.90)
save(fig, "Figure10_RQ3_impacts")
