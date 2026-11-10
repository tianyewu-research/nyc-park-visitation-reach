"""
28_methodology_diagram.py   (Figure 2: methodological workflow)
A stage-by-stage chain of the methods used, for reviewer orientation.
Output: outputs/figures/methodology_diagram.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIGS = Path(__file__).resolve().parent.parent / "outputs" / "figures"

stages = [
    ("1 · Data",
     "Advan mobile location data (52 weeks, 2023)  ·  NYC Parks / ParkServe boundaries\n"
     "ACS 2019–2023 5-year estimates (income, race B03002)  ·  EPA Smart Location Database",
     "#e8eef7"),
    ("2 · Measures",
     "Dwell-qualified (≥5 min) unique visitors apportioned to home CBG, with home→park distance\n"
     "LPUR  =  share of a CBG's visits to parks within 2 km  (primary outcome)\n"
     "Mean travel distance  =  visitor-weighted mean home→park distance  (companion)",
     "#e3f0e7"),
    ("3 · Question",
     "How locally is each neighborhood's park use oriented — nearby parks vs. travel out?\n"
     "Covariates: income, racial composition (Black / Hispanic / Asian), park quality,\n"
     "log access, walkability, transit, density, borough",
     "#fdf3e3"),
    ("4 · Spatial analysis",
     "Row-standardized Queen weights (kNN robustness)\n"
     "RQ1:  Global Moran's I (999 permutations)  +  LISA clusters of LPUR\n"
     "RQ2 / RQ3:  OLS → LM diagnostics → ML spatial error model of LPUR",
     "#f3e8f5"),
    ("5 · Robustness",
     "Neighborhood radius specification curve (0.5–3 km)  ·  Queen / kNN weights\n"
     "racial decomposition  ·  travel-distance companion measure\n"
     "within-neighborhood ratio cancels panel-intensity differences",
     "#eeeeee"),
]

fig, ax = plt.subplots(figsize=(9, 11))
ax.set_xlim(0, 10); ax.set_ylim(0, len(stages) * 2.2 + 0.5); ax.axis("off")
n = len(stages)
box_h = 1.6
for i, (title, body, color) in enumerate(stages):
    y = (n - 1 - i) * 2.2 + 0.5
    box = FancyBboxPatch((0.5, y), 9, box_h, boxstyle="round,pad=0.08,rounding_size=0.12",
                         linewidth=1.2, edgecolor="#555555", facecolor=color)
    ax.add_patch(box)
    ax.text(0.75, y + box_h - 0.28, title, fontsize=13, fontweight="bold", va="top", color="#222")
    ax.text(0.75, y + box_h - 0.72, body, fontsize=10.5, va="top", color="#333", linespacing=1.35)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((5, y), (5, y - 0.6), arrowstyle="-|>",
                                     mutation_scale=18, linewidth=1.6, color="#555"))

ax.set_title("Methodological workflow: from data to inference", fontsize=14, pad=14)
plt.tight_layout()
plt.savefig(FIGS / "methodology_diagram.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGS/'methodology_diagram.png'}")
assert (FIGS / "methodology_diagram.png").exists()
print("[self-check] 28 methodology diagram OK")
