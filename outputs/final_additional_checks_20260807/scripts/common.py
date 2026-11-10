"""Shared helpers for the final additional checks (2026-08-07). Reuses the frozen final-pipeline
libraries (slib, slib2) rather than reimplementing sample/weights/impacts. Writes ONLY into
outputs/final_additional_checks_20260807. No manuscript edits.

Cross-sectional ecological association: rho and indirect/total impacts are model-implied quantities
conditional on the SDM and W — NOT causal spillovers, diffusion, or behavioral propagation.
American English in all new labels/reports."""
import sys
from pathlib import Path

ROOT = Path(".")
FINAL = ROOT / "outputs" / "final_sdm_revision_20260803"
REASSESS = ROOT / "outputs" / "spatial_model_reassessment_20260803"
OUTDIR = ROOT / "outputs" / "final_additional_checks_20260807"
FIGDIR = OUTDIR / "figures"
sys.path.insert(0, str(REASSESS / "scripts"))
sys.path.insert(0, str(FINAL / "scripts"))

SEED = 42
# Okabe-Ito color-blind-safe palette
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00",
          skyblue="#56B4E9", yellow="#F0E442", purple="#CC79A7", grey="#999999", black="#000000")

RACE = ["pct_black", "pct_hispanic", "pct_asian"]
RACE_LABEL = {"pct_black": "Black", "pct_hispanic": "Hispanic", "pct_asian": "Asian",
              "pct_nonwhite": "Non-White"}


def savefig(fig, stem):
    """Save a figure to figures/ as 300-dpi PNG plus vector PDF and SVG."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, dpi in (("png", 300), ("pdf", 300), ("svg", 300)):
        p = FIGDIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(str(p.relative_to(ROOT)))
    return paths
