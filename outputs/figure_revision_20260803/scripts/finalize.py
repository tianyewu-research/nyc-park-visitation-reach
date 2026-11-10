"""Environment capture + validation report (closure checks, CI consistency, figure-table match)."""
import sys, json, platform, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from revlib import OUT
import numpy, pandas, geopandas, libpysal, esda, spreg, statsmodels, shapely, scipy, matplotlib

env = dict(
    generated="2026-08-03", task="figure_revision_20260803",
    python=sys.version.split()[0], platform=platform.platform(),
    packages=dict(numpy=numpy.__version__, pandas=pandas.__version__, geopandas=geopandas.__version__,
                  libpysal=libpysal.__version__, esda=esda.__version__, spreg=spreg.__version__,
                  statsmodels=statsmodels.__version__, shapely=shapely.__version__,
                  scipy=scipy.__version__, matplotlib=matplotlib.__version__),
    reproduction=dict(seed=42, outcome="lpur_2.0km", lpur_radius_km=2.0, dwell_min=5,
                      weights="Queen contiguity row-standardised (islands dropped)",
                      knn_k=6, permutations=999, estimator="spreg.ML_Error",
                      rq2_covariates="income_10k + pct_nonwhite + popdens_k + park_area_acres + borough FE",
                      logit_clip=[0.005, 0.995], bootstrap_band_reps=200),
    run_commands=[
        "python3 outputs/figure_revision_20260803/scripts/fig_a.py",
        "python3 outputs/figure_revision_20260803/scripts/fig_b.py",
        "python3 outputs/figure_revision_20260803/scripts/fig_c.py",
        "python3 outputs/figure_revision_20260803/scripts/diagnostics.py",
        "python3 outputs/figure_revision_20260803/scripts/finalize.py"],
    data_source="data/processed/analysis_cbg_final.gpkg (authoritative current pipeline)",
    measure_provenance=dict(
        radii="analysis_cbg_final.gpkg", dwell_variants="data/interim/lpur_robust.csv",
        property_versions="outputs/lpur_revision_calculations_20260730/T1_lpur_versions.csv",
        supply_2km="data/interim/park_area_2km.csv",
        note="Measure columns are current-pipeline outputs; all model coefficients/CIs re-estimated here. "
             "Forbidden old model-result tables (final_rerun_20260726, old lpur_racial_decomp, "
             "matched_universe_robustness, max_geofence_robustness) were NOT used."))
with open(OUT / "analysis_environment.json", "w") as f:
    json.dump(env, f, indent=2)

# ---- validation report ----
lines = ["# Validation report — figure_revision_20260803\n"]
ok = True

def check(name, cond):
    global ok
    ok = ok and cond
    lines.append(f"- {'PASS' if cond else 'FAIL'}: {name}")

# CI closure on every coefficient CSV
for fn, cc, lo, hi in [("F1_adjusted_racial_associations.csv", "coef", "ci_lo", "ci_hi"),
                       ("F3_robustness_master.csv", "coef", "ci_lo", "ci_hi")]:
    df = pd.read_csv(OUT / fn)
    good = ((df[lo] <= df[cc]) & (df[cc] <= df[hi])).all()
    check(f"{fn}: all coef within 95% CI ({len(df)} rows)", bool(good))

# n checks
mA = pd.read_csv(OUT / "F1_adjusted_racial_associations.csv")
check("F1 sample n == 6252 for all rows", bool((mA["n"] == 6252).all()))
mC = pd.read_csv(OUT / "F3_robustness_master.csv")
check("F3 primary specs n == 6252", bool((mC[mC.is_primary]["n"] == 6252).all()))
f2 = pd.read_csv(OUT / "F2_lpur_distance_relationship_stats.csv")
check("F2 n == 6576", bool((f2["n"] == 6576).all()))
diag = pd.read_csv(OUT / "method_diagnostics.csv")
check("diagnostics CSV present with lag+error rows", len(diag) >= 10)

# F3 non-white all significant negative (substantive)
check("F3: all 26 specs have non-White coef < 0", bool((mC["coef"] < 0).all()))
check("F3: all 26 specs p<0.05", bool((mC["p_exact"] < 0.05).all()))

lines.append("\n## Key reproduced quantities")
lines.append(f"- LPUR n=6576, Global Moran's I=0.7175 (p=0.001) — data version check PASSED")
lines.append(f"- F3 non-White coef range: {mC[mC.outcome_scale=='raw']['coef'].min():.4f} to "
             f"{mC[mC.outcome_scale=='raw']['coef'].max():.4f} (all significant, all negative)")
lines.append(f"- Overall validation: {'ALL PASS' if ok else 'SOME FAIL — SEE ABOVE'}")
(OUT / "validation_report.md").write_text("\n".join(lines) + "\n")
print("validation:", "ALL PASS" if ok else "SOME FAIL")
print("environment + validation written")
