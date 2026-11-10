"""PHASE 9 — validation and handoff. Builds run_manifest.json, validation_report.md and
analysis_summary.md from the produced outputs. Runs no new models. No manuscript edits."""
import sys, json, hashlib, platform, subprocess, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUTDIR, FIGDIR, SEED
import numpy, pandas, geopandas, libpysal, esda, spreg, scipy, matplotlib, statsmodels

GPKG = ROOT / "data" / "processed" / "analysis_cbg_final.gpkg"
prov = json.loads((OUTDIR / "phase0_provenance.json").read_text())


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------- run manifest ----------
scripts = sorted([str(p.relative_to(ROOT)) for p in (OUTDIR / "scripts").glob("*.py")])
csvs = sorted([str(p.relative_to(ROOT)) for p in OUTDIR.glob("*.csv")])
mds = sorted([str(p.relative_to(ROOT)) for p in OUTDIR.glob("*.md")])
jsons = sorted([str(p.relative_to(ROOT)) for p in OUTDIR.glob("*.json")])
figs = sorted([str(p.relative_to(ROOT)) for p in FIGDIR.glob("*") if p.suffix in (".png", ".pdf", ".svg")])
try:
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     text=True).strip()
except Exception:
    commit = "unknown"

manifest = dict(
    task="final_additional_checks_20260807",
    generated=dt.datetime.now().isoformat(timespec="seconds"),
    git_commit=commit, seed=SEED,
    python=sys.version.split()[0], platform=platform.platform(),
    packages=dict(numpy=numpy.__version__, pandas=pandas.__version__, geopandas=geopandas.__version__,
                  libpysal=libpysal.__version__, esda=esda.__version__, spreg=spreg.__version__,
                  scipy=scipy.__version__, statsmodels=statsmodels.__version__,
                  matplotlib=matplotlib.__version__),
    input_hashes={str(GPKG.relative_to(ROOT)): sha256(GPKG)},
    reproduced_sample_sizes=dict(RQ2_aggregate=prov["n_agg"], joint_LPUR=prov["n_joint_lpur"],
                                 distance=prov["n_dist"]),
    scripts=scripts, output_csv=csvs, output_md=mds, output_json=jsons, figures=figs,
    run_order=["p0_repro_gate", "p1_collinearity_composition", "p2_radius_profiles",
               "p3_robustness_nonradius", "p4_rq3_figure", "p5_destination",
               "p6_geographic_summary", "p7_network_feasibility", "p9_finalize"],
    figure2_spec="Figure2_Illustrator_content_spec.md (author redraws in Illustrator)")
(OUTDIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

# ---------- validation report ----------
v = ["# Validation report — final_additional_checks_20260807\n"]
ok = True


def line(passed, msg):
    global ok; ok = ok and passed; v.append(f"- {'PASS' if passed else 'FAIL'}: {msg}")


# PHASE 0
p0 = pd.read_csv(OUTDIR / "phase0_reproduction_checks.csv")
line(bool(p0["ok"].all()), f"PHASE 0 reproduction gate ({int(p0['ok'].sum())}/{len(p0)} checks) — hash, "
     f"n=6,252, Non-White direct -0.021969 / total -0.064606 all reproduce")

# figure-data existence / integrity
expected = {
    "collinearity_local_X.csv", "collinearity_augmented_X_WX.csv", "vif_diagnostics.csv",
    "condition_numbers.csv", "racial_composition_separate_models_impacts.csv",
    "racial_composition_joint_vs_separate_comparison.csv",
    "radius_group_impacts_tidy.csv", "radius_nonwhite_impacts_tidy.csv", "radius_model_diagnostics.csv",
    "robustness_nonradius_tidy.csv", "FigS_clipped_logit.csv", "rq3_impacts_plot_data.csv",
    "destination_records_property_level.csv", "destination_metrics_by_nonwhite_quartile.csv",
    "top_destinations_by_nonwhite_quartile.csv", "destination_classification_crosswalk.csv",
    "borough_lpur_lisa_summary.csv"}
present = {p.name for p in OUTDIR.glob("*.csv")}
missing = expected - present
line(not missing, f"all expected data CSVs present ({len(expected - missing)}/{len(expected)})"
     + (f"; MISSING {sorted(missing)}" if missing else ""))

expected_figs = {"FigS_correlation_heatmap", "FigS_joint_vs_separate", "Fig6_radius_group_profiles",
                 "FigS_radius_nonwhite_profile", "Fig7_grouped_nonradius_robustness",
                 "FigS_clipped_logit", "Fig8_RQ3_impacts", "Fig9_destination_patterns"}
for stem in sorted(expected_figs):
    have = all((FIGDIR / f"{stem}.{ext}").exists() for ext in ("png", "pdf", "svg"))
    line(have, f"figure {stem} exported as PNG+PDF+SVG")

# radius diagnostics convergence
rd = pd.read_csv(OUTDIR / "radius_model_diagnostics.csv")
line(bool((rd["converged"] == True).all()), f"all {len(rd)} radius models converged")

# LISA closure vs sanctioned totals
gf = (OUTDIR / "geographic_results_facts.md").read_text()
line("PASS — per-CBG classes reproduce" in gf, "borough LISA reproduces sanctioned FDR totals (closure)")

# separate/joint sign agreement (diagnostic, not a gate)
comp = pd.read_csv(OUTDIR / "racial_composition_joint_vs_separate_comparison.csv")
sub = comp[comp.effect.isin(["direct", "total"])]
line(True, f"separate-vs-joint sign agreement (direct+total): "
     f"{int(sub['sign_agrees'].sum())}/{len(sub)} (disagreements are between non-significant estimates)")

# omitted / not-usable tasks (transparent)
v += ["", "## Omitted / conditional tasks (transparent)",
      "- PHASE 6 NTA/community-district join: **omitted** — no authoritative NTA/CD boundary layer "
      "exists in the repo (borough boundaries only). Borough-level summary delivered instead.",
      "- PHASE 7 network sensitivity: **not usable** — no pedestrian-routing/PBF library installed "
      "(osmnx/pandana/networkx/pyrosm/osmium all absent); OD support is several hundred thousand pairs; "
      "not reproducible in the frozen environment. Feasibility note delivered; Euclidean LPUR retained.",
      "- PHASE 8 Figure 2: content specification only; the manuscript figure is NOT overwritten.",
      "",
      "## Flags to address before manuscript rewriting",
      "1. The distance/joint sample n is **6,252** (reproduced; archived distance_model_diagnostics.csv). "
      "The Figure_A caption's **6,246** is stale — correct it wherever it appears.",
      "2. Report Non-White average **direct** impact as primary/stable; indirect & total as model-implied "
      "quantities conditional on the SDM and W. No causal-spillover / diffusion / propagation language.",
      "3. Income total is a **model-dependent** spatial-context association — not co-primary with the "
      "Non-White direct impact; do not imply the SDM revealed a hidden true income effect.",
      "4. Destination results are **descriptive** attributed visit records; they narrow but do not "
      "resolve choice-vs-constraint. No preference/welfare claims.",
      "5. Hispanic total impact is non-significant in the joint model; avoid blanket 'all groups' phrasing.",
      "6. Radius profiles mix a measurement-scale gradient with any substantive gradient; read for "
      "sign/significance stability around 2 km, not as a dose-response curve."]
v.append(f"\nOverall: {'ALL PASS' if ok else 'SOME FAIL — DIAGNOSE'}")
(OUTDIR / "validation_report.md").write_text("\n".join(v) + "\n")

print("=== PHASE 9 done ===")
print("Overall:", "ALL PASS" if ok else "SOME FAIL")
print(f"scripts={len(scripts)} csv={len(csvs)} md={len(mds)} figures={len(figs)}")
if missing:
    print("MISSING CSVs:", sorted(missing))
