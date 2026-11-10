"""Shared constants/paths for the destination-per-CBG + match-alignment rerun (2026-08-08).
Writes ONLY into outputs/destination_per_cbg_and_match_alignment_20260808/. No manuscript edits;
no overwriting of existing outputs. Canonical matching threshold is EXACTLY 100 EPSG:2263 feet."""
import sys
from pathlib import Path

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
INT = ROOT / "data" / "interim"
GPKG = ROOT / "data" / "processed" / "analysis_cbg_final.gpkg"
OUT = ROOT / "outputs" / "destination_per_cbg_and_match_alignment_20260808"
FIG = OUT / "figures"
REASSESS = ROOT / "outputs" / "spatial_model_reassessment_20260803" / "scripts"
FINAL = ROOT / "outputs" / "final_sdm_revision_20260803" / "scripts"
sys.path.insert(0, str(REASSESS))
sys.path.insert(0, str(FINAL))

SEED = 42
NYC = ("36005", "36047", "36061", "36081", "36085")
M2FT = 1 / 0.3048
MATCH_FT = 100.0            # canonical threshold: EXACTLY 100 feet in EPSG:2263 (NOT 100*M2FT)
OLD_TOL_FT = 100.0 * M2FT   # the old t1 threshold: ~328.08 ft (mislabelled "100 ft")
TWO_KM_FT = 2000.0 * M2FT
DWELL_KEEP = ("5-20", "21-60", "61-240", ">240")
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00", skyblue="#56B4E9",
          yellow="#F0E442", purple="#CC79A7", grey="#999999", black="#000000")


def qfrac(bj):
    import json
    if not isinstance(bj, str):
        return 1.0
    try:
        b = json.loads(bj)
    except Exception:
        return 1.0
    t = sum(b.values())
    return sum(b.get(k, 0) for k in DWELL_KEEP) / t if t > 0 else 0.0


def savefig(fig, stem, dpi=600):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    return [str((FIG / f"{stem}.{e}").relative_to(ROOT)) for e in ("png", "pdf", "svg")]


def sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
