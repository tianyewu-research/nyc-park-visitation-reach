from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path("outputs/manuscript_figures/figures")
OUT.mkdir(parents=True, exist_ok=True)

WIDTH_MM = 180
HEIGHT_MM = 125
MM_PER_INCH = 25.4

COLORS = {
    "input_fill": "#EAF2F8",
    "input_edge": "#9CBFD6",
    "process_fill": "#FFF3E0",
    "process_edge": "#D5B681",
    "data_fill": "#E8F5E9",
    "data_edge": "#94BE9A",
    "analysis_fill": "#F1ECF8",
    "analysis_edge": "#B2A2C9",
    "bar_fill": "#EEF2F4",
    "bar_edge": "#90A4AE",
    "text": "#263238",
    "secondary": "#52616D",
    "arrow": "#607D8B",
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def rounded_box(ax, x, y, w, h, title, details, fill, edge):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.014",
        linewidth=0.65,
        edgecolor=edge,
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(patch)
    pad_x = 0.012
    ax.text(
        x + pad_x,
        y + h - 0.023,
        title,
        ha="left",
        va="top",
        fontsize=8.3,
        fontweight="bold",
        color=COLORS["text"],
        zorder=4,
    )
    ax.text(
        x + pad_x,
        y + h - 0.052,
        "\n".join(details),
        ha="left",
        va="top",
        fontsize=7.05,
        linespacing=1.17,
        color=COLORS["text"],
        zorder=4,
    )
    return {"x": x, "y": y, "w": w, "h": h}


def stage_label(ax, y, label, edge):
    ax.add_line(
        Line2D(
            [0.115, 0.115],
            [y - 0.043, y + 0.043],
            lw=2.2,
            color=edge,
            solid_capstyle="round",
            zorder=2,
        )
    )
    ax.text(
        0.018,
        y,
        label,
        ha="left",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["secondary"],
        linespacing=1.08,
        zorder=4,
    )


def line(ax, points, lw=0.75):
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        ax.add_line(
            Line2D(
                [x1, x2],
                [y1, y2],
                lw=lw,
                color=COLORS["arrow"],
                solid_capstyle="round",
                zorder=1,
            )
        )


def arrow(ax, start, end, lw=0.75):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=lw,
            color=COLORS["arrow"],
            shrinkA=0,
            shrinkB=1.8,
            zorder=2,
        )
    )


fig = plt.figure(
    figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH),
    facecolor="white",
)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Stage labels form a redundant cue when the pastel fills are printed in grayscale.
stage_label(ax, 0.855, "DATA\nINPUTS", COLORS["input_edge"])
stage_label(ax, 0.657, "SOURCE\nPROCESSING", COLORS["process_edge"])
stage_label(ax, 0.382, "ANALYSIS-\nREADY DATA", COLORS["data_edge"])
stage_label(ax, 0.100, "RQ-SPECIFIC\nANALYSES", COLORS["analysis_edge"])

# Row 1: four data-source families.
source_y, source_h = 0.790, 0.130
source_x = [0.135, 0.3515, 0.568, 0.7845]
source_w = 0.2005
sources = [
    rounded_box(
        ax,
        source_x[0],
        source_y,
        source_w,
        source_h,
        "Advan Weekly Patterns",
        ["52 weeks, 2023", "park POIs + home CBG shares"],
        COLORS["input_fill"],
        COLORS["input_edge"],
    ),
    rounded_box(
        ax,
        source_x[1],
        source_y,
        source_w,
        source_h,
        "NYC Parks + ParkServe",
        ["boundaries, acreage", "service areas + amenities"],
        COLORS["input_fill"],
        COLORS["input_edge"],
    ),
    rounded_box(
        ax,
        source_x[2],
        source_y,
        source_w,
        source_h,
        "ACS 2019–2023",
        ["population, income", "racial composition"],
        COLORS["input_fill"],
        COLORS["input_edge"],
    ),
    rounded_box(
        ax,
        source_x[3],
        source_y,
        source_w,
        source_h,
        "EPA SLD 2021",
        ["walkability", "transit frequency"],
        COLORS["input_fill"],
        COLORS["input_edge"],
    ),
]

# Row 2: source-specific processing. Box centers align with their source branches.
process_y, process_h, process_w = 0.595, 0.125, 0.2005
process_x = [0.135, 0.3515, 0.67625]
processes = [
    rounded_box(
        ax,
        process_x[0],
        process_y,
        process_w,
        process_h,
        "Visitation attribution",
        ["visitors × dwell-qualified share", "× home-CBG share"],
        COLORS["process_fill"],
        COLORS["process_edge"],
    ),
    rounded_box(
        ax,
        process_x[1],
        process_y,
        process_w,
        process_h,
        "Park harmonization",
        ["1,676 records → 1,675 properties", "canonical 100-ft POI crosswalk"],
        COLORS["process_fill"],
        COLORS["process_edge"],
    ),
    rounded_box(
        ax,
        process_x[2],
        process_y,
        process_w,
        process_h,
        "Neighborhood harmonization",
        ["ACS + EPA measures", "aligned to 2023 CBGs"],
        COLORS["process_fill"],
        COLORS["process_edge"],
    ),
]

# Source-to-processing connections. ACS and EPA share one collector to avoid
# duplicated diagonal arrows.
source_centers = [b["x"] + b["w"] / 2 for b in sources]
process_centers = [b["x"] + b["w"] / 2 for b in processes]
arrow(ax, (source_centers[0], source_y), (process_centers[0], process_y + process_h))
arrow(ax, (source_centers[1], source_y), (process_centers[1], process_y + process_h))
collector_y = 0.755
line(ax, [(source_centers[2], source_y), (source_centers[2], collector_y)])
line(ax, [(source_centers[3], source_y), (source_centers[3], collector_y)])
line(ax, [(source_centers[2], collector_y), (source_centers[3], collector_y)])
arrow(
    ax,
    ((source_centers[2] + source_centers[3]) / 2, collector_y),
    (process_centers[2], process_y + process_h),
)

# Integration bar and the three short source-specific feeds.
bar_x, bar_y, bar_w, bar_h = 0.205, 0.505, 0.710, 0.045
bar = FancyBboxPatch(
    (bar_x, bar_y),
    bar_w,
    bar_h,
    boxstyle="round,pad=0.003,rounding_size=0.018",
    linewidth=0.65,
    edgecolor=COLORS["bar_edge"],
    facecolor=COLORS["bar_fill"],
    zorder=3,
)
ax.add_patch(bar)
ax.text(
    bar_x + bar_w / 2,
    bar_y + bar_h / 2,
    "Integrate by 2023 CBG and calculate distances",
    ha="center",
    va="center",
    fontsize=7.5,
    fontweight="bold",
    color=COLORS["text"],
    zorder=4,
)
for center in process_centers:
    arrow(ax, (center, process_y), (center, bar_y + bar_h))

# Row 3: the three analysis-ready samples.
data_y, data_h, data_w = 0.325, 0.115, 0.266
data_x = [0.135, 0.427, 0.719]
datasets = [
    rounded_box(
        ax,
        data_x[0],
        data_y,
        data_w,
        data_h,
        "Park-use outcomes",
        ["LPUR (2 km) + mean distance", "n = 6,576"],
        COLORS["data_fill"],
        COLORS["data_edge"],
    ),
    rounded_box(
        ax,
        data_x[1],
        data_y,
        data_w,
        data_h,
        "RQ2 complete cases",
        ["social composition + controls", "n = 6,252"],
        COLORS["data_fill"],
        COLORS["data_edge"],
    ),
    rounded_box(
        ax,
        data_x[2],
        data_y,
        data_w,
        data_h,
        "RQ3 complete cases",
        ["park + built environment", "n = 6,227"],
        COLORS["data_fill"],
        COLORS["data_edge"],
    ),
]
data_centers = [b["x"] + b["w"] / 2 for b in datasets]
spine_x, bus_y = data_centers[1], 0.470
line(ax, [(spine_x, bar_y), (spine_x, bus_y)])
line(ax, [(data_centers[0], bus_y), (data_centers[2], bus_y)])
for center in data_centers:
    arrow(ax, (center, bus_y), (center, data_y + data_h))

# Row 4: analyses. The RQ2 branch is split once; the compact property tag
# indicates why destination analyses use the canonical park-property crosswalk.
analysis_y, analysis_h = 0.035, 0.130
analysis_x = [0.135, 0.3515, 0.568, 0.7845]
analysis_w = 0.2005
analyses = [
    rounded_box(
        ax,
        analysis_x[0],
        analysis_y,
        analysis_w,
        analysis_h,
        "RQ1 | Spatial pattern",
        ["Global Moran's I", "BH-FDR local Moran clusters"],
        COLORS["analysis_fill"],
        COLORS["analysis_edge"],
    ),
    rounded_box(
        ax,
        analysis_x[1],
        analysis_y,
        analysis_w,
        analysis_h,
        "RQ2 | Social patterning",
        ["partial SDM: LPUR + distance", "radius + construction checks"],
        COLORS["analysis_fill"],
        COLORS["analysis_edge"],
    ),
    rounded_box(
        ax,
        analysis_x[2],
        analysis_y,
        analysis_w,
        analysis_h,
        "RQ2 | Destinations",
        ["within-CBG top-10 share", "HHI + effective count"],
        COLORS["analysis_fill"],
        COLORS["analysis_edge"],
    ),
    rounded_box(
        ax,
        analysis_x[3],
        analysis_y,
        analysis_w,
        analysis_h,
        "RQ3 | Park context",
        ["partial SDM impacts", "[X, WX] collinearity checks"],
        COLORS["analysis_fill"],
        COLORS["analysis_edge"],
    ),
]
analysis_centers = [b["x"] + b["w"] / 2 for b in analyses]

# RQ1 and RQ3 use tidy elbow connectors rather than long diagonal arrows.
elbow_y = 0.245
line(ax, [(data_centers[0], data_y), (data_centers[0], elbow_y)])
line(ax, [(data_centers[0], elbow_y), (analysis_centers[0], elbow_y)])
arrow(ax, (analysis_centers[0], elbow_y), (analysis_centers[0], analysis_y + analysis_h))

line(ax, [(data_centers[2], data_y), (data_centers[2], elbow_y)])
line(ax, [(data_centers[2], elbow_y), (analysis_centers[3], elbow_y)])
arrow(ax, (analysis_centers[3], elbow_y), (analysis_centers[3], analysis_y + analysis_h))

# RQ2 social and property-level destination branches.
rq2_bus_y = 0.287
line(ax, [(data_centers[1], data_y), (data_centers[1], rq2_bus_y)])
line(ax, [(analysis_centers[1], rq2_bus_y), (analysis_centers[2], rq2_bus_y)])
arrow(
    ax,
    (analysis_centers[1], rq2_bus_y),
    (analysis_centers[1], analysis_y + analysis_h),
)

tag_w, tag_h = 0.245, 0.050
tag_x = analysis_centers[2] - tag_w / 2
tag_y = 0.213
arrow(ax, (analysis_centers[2], rq2_bus_y), (analysis_centers[2], tag_y + tag_h))
tag = FancyBboxPatch(
    (tag_x, tag_y),
    tag_w,
    tag_h,
    boxstyle="round,pad=0.002,rounding_size=0.016",
    linewidth=0.65,
    linestyle=(0, (2.2, 1.8)),
    edgecolor=COLORS["process_edge"],
    facecolor=COLORS["process_fill"],
    zorder=3,
)
ax.add_patch(tag)
ax.text(
    analysis_centers[2],
    tag_y + tag_h / 2,
    "Property-level branch\nrepresentation checks + destinations",
    ha="center",
    va="center",
    fontsize=6.25,
    linespacing=1.05,
    color=COLORS["text"],
    zorder=4,
)
arrow(
    ax,
    (analysis_centers[2], tag_y),
    (analysis_centers[2], analysis_y + analysis_h),
)

stem = OUT / "Figure2_workflow_0813"
for extension, dpi in (("svg", None), ("pdf", None), ("png", 600)):
    fig.savefig(
        stem.with_suffix(f".{extension}"),
        dpi=dpi,
        facecolor="white",
        edgecolor="none",
        metadata={"Title": "Figure 2. Data-processing and analytical workflow"},
    )
plt.close(fig)

print(f"Wrote {stem}.svg")
print(f"Wrote {stem}.pdf")
print(f"Wrote {stem}.png")
