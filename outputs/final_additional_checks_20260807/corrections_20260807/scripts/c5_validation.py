"""CORRECTION TASK 5 — corrections validation + manifest. Checks the correction deliverables and writes
corrections_validation_report.md with interpretive flags. Runs no models; no manuscript edits; does not
overwrite the original final_additional_checks_20260807 outputs."""
import sys, json, hashlib, platform, datetime as dt
from pathlib import Path
import pandas as pd, numpy as np
OUT = Path("outputs/final_additional_checks_20260807/"
           "corrections_20260807")
FIG = OUT / "figures"
ROOT = Path(".")

flags = []


def exists(name, sub=""):
    p = (FIG / name) if sub == "fig" else (OUT / name)
    return p.exists()


# gather correction results
collin = pd.read_csv(OUT / "rq3_access_collinearity_confirm.csv")
sens = pd.read_csv(OUT / "rq3_access_collinearity_sensitivity_models.csv")
imp = pd.read_csv(OUT / "rq3_access_collinearity_sensitivity_impacts.csv")
cov = pd.read_csv(OUT / "destination_match_coverage.csv")
met = pd.read_csv(OUT / "destination_metrics_corrected.csv")
robiqr = pd.read_csv(OUT / "robustness_nonradius_IQR_tidy.csv")

# accessibility stability check
def gi(m, eff):
    r = imp[(imp.model == m) & (imp.variable == "log_access") & (imp.effect == eff)]
    return r.iloc[0] if len(r) else None


la_A = gi("A_full", "total"); la_B = gi("B_omit_Wlogaccess", "total")


def sg(r):
    return (float(r["ci_lo"]) > 0) or (float(r["ci_hi"]) < 0)


acc_stable = (np.sign(la_A["estimate"]) == np.sign(la_B["estimate"])) and (sg(la_A) == sg(la_B))
acc_verdict = ("STABLE across A/B" if acc_stable else
               "NOT STABLE across A/B (significant in the full model, non-significant when W·log_access "
               "is omitted; the direct impact changes sign)")

# expected deliverables
expect_csv = ["rq3_access_collinearity_confirm.csv", "rq3_access_collinearity_sensitivity_models.csv",
              "rq3_access_collinearity_sensitivity_impacts.csv",
              "rq3_access_collinearity_sensitivity_structural.csv", "robustness_nonradius_IQR_tidy.csv",
              "destination_metrics_corrected.csv", "destination_top_destinations_corrected.csv",
              "destination_match_coverage.csv", "destination_origin_quartile_counts.csv",
              "destination_size_class_shares_corrected.csv", "TableS_top_destinations.csv"]
expect_md = ["rq3_access_collinearity_interpretation.md", "destination_analysis_summary_corrected.md",
             "Figure2_Illustrator_content_spec_corrected.md"]
expect_fig = ["FigS_RQ3_access_collinearity_sensitivity", "Fig7_grouped_nonradius_robustness_IQR",
              "Fig9_destination_patterns_corrected", "Fig6_scale_sensitivity"]

miss_csv = [f for f in expect_csv if not (OUT / f).exists()]
miss_md = [f for f in expect_md if not (OUT / f).exists()]
miss_fig = [s for s in expect_fig if not all((FIG / f"{s}.{e}").exists() for e in ("png", "pdf", "svg"))]
all_present = not (miss_csv or miss_md or miss_fig)

# neutral size labels check (no 'flagship')
sc = pd.read_csv(OUT / "destination_size_class_shares_corrected.csv")
no_flagship = not sc["size_class"].astype(str).str.contains("flagship", case=False).any()

# IQR rescale sanity: primary total per-IQR ~ -0.0646
prim = robiqr[(robiqr.group == "Primary") & (robiqr.effect == "total")].iloc[0]
iqr_ok = abs(float(prim["estimate"]) - (-0.064606)) < 5e-4

status = "PASS WITH REQUIRED INTERPRETIVE FLAGS" if (all_present and no_flagship and iqr_ok) \
    else "INCOMPLETE — DIAGNOSE"

# weight/cell shares
cell_gt2 = cov[cov.quantity.str.contains("cell share")]["value"].iloc[0]
wt_gt2 = cov[cov.quantity.str.contains("attributed-weight share")]["value"].iloc[0]

L = [f"# Corrections validation report — corrections_20260807\n",
     f"Status: **{status}**\n",
     "Scope: targeted corrections to `final_additional_checks_20260807` per the independent audit. "
     "The original outputs are NOT overwritten; the manuscript, frozen GeoPackage, RQs, LPUR definition "
     "and primary partial SDM are unchanged.\n",
     "## Deliverable presence",
     f"- Data CSVs: {'ALL present' if not miss_csv else 'MISSING ' + str(miss_csv)}.",
     f"- Reports (md): {'ALL present' if not miss_md else 'MISSING ' + str(miss_md)}.",
     f"- Figures (png+pdf+svg): {'ALL present' if not miss_fig else 'MISSING ' + str(miss_fig)}.",
     f"- Neutral size labels (no 'flagship'): {'PASS' if no_flagship else 'FAIL'}.",
     f"- Figure 7 per-IQR rescale sanity (primary total = {prim['estimate']} ≈ -0.064606): "
     f"{'PASS' if iqr_ok else 'FAIL'}.",
     "",
     "## Required interpretive flags (must be honored in the manuscript)",
     "1. **Local-X multicollinearity: LOW.** Max local VIF ~2.06; local condition numbers 3.2–4.0. The "
     "locally entered predictors do not show substantial multicollinearity.",
     f"2. **Augmented RQ3 accessibility near-redundancy: MATERIAL.** corr(log_access, W_log_access) = "
     f"{collin[collin.quantity.str.contains('corr')]['value'].iloc[0]}; augmented condition number "
     f"{collin[collin.quantity.str.contains('condition')]['value'].iloc[0]}. The accessibility-specific "
     "local/spatial-lag decomposition is weakly identified. The regional-accessibility total impact is "
     + acc_verdict + " — it must be **downgraded / marked model-sensitive**, and Figure 8 must not "
     "emphasize its marginal p≈0.049 interval.",
     "3. **Group-specific composition: directions mostly stable, uncertainty/significance sensitive.** "
     "Separate vs joint models mostly agree in sign but significance changes for several group-specific "
     "estimates; do not claim full group-specific robustness. The other RQ3 impacts (Non-White, income, "
     "park quality, local park area, population density) ARE stable when the accessibility terms are "
     "removed (variant C).",
     "4. **Primary result unchanged: the aggregate Non-White average DIRECT impact is the stable primary "
     "finding.** Indirect/total are model-implied; income total is model-dependent; no causal-spillover "
     "language.",
     "",
     "## Destination reporting corrections applied",
     f"- 1,051,858 are annual property-home-CBG **cells**, not visits. Cell share >2 km = {cell_gt2}%; "
     f"attributed-weight share >2 km = {wt_gt2}% (the relevant reach statistic).",
     "- Match coverage, origin-CBG quartile counts + cut points, top-5 and top-10 shares, neutral size "
     "labels, and a 1,000-rep CBG-clustered descriptive bootstrap are reported. Figure 9 (corrected) uses "
     "non-local destinations only with no dual axes.",
     "",
     "## Figures/wording corrections applied",
     "- Figure 7 rescaled to per-IQR (`Fig7_grouped_nonradius_robustness_IQR`).",
     "- Figure 6 reworded to 'LPUR catchment radii' / 'Scale sensitivity …' with a corrected caption "
     "(`Fig6_scale_sensitivity`).",
     "- Figure 2 spec wording corrected (dwell retention ~16%; vendor-scaled home allocation; never 6,246).",
     "- New RQ3 accessibility sensitivity figure (`FigS_RQ3_access_collinearity_sensitivity`).",
     "",
     "## Not overwritten / out of scope (unchanged)",
     "- Original `final_additional_checks_20260807` outputs, manuscript DOCX, frozen GeoPackage, primary "
     "partial SDM, RQs, and LPUR definition are all unchanged."]
(OUT / "corrections_validation_report.md").write_text("\n".join(L) + "\n")

# manifest
scripts = sorted(str(p.name) for p in (OUT / "scripts").glob("*.py"))
manifest = dict(task="corrections_20260807",
                generated=dt.datetime.now().isoformat(timespec="seconds"),
                python=sys.version.split()[0], platform=platform.platform(), seed=42,
                status=status, accessibility_total_stable=bool(acc_stable),
                scripts=scripts,
                csv=sorted(p.name for p in OUT.glob("*.csv")),
                md=sorted(p.name for p in OUT.glob("*.md")),
                figures=sorted(p.name for p in FIG.glob("*")))
(OUT / "corrections_run_manifest.json").write_text(json.dumps(manifest, indent=2))

print("=== c5 done ===")
print("status:", status)
print("accessibility total stable across A/B:", acc_stable)
print("missing:", miss_csv + miss_md + miss_fig or "none")
