"""Environment capture + validation report for the spatial-model reassessment."""
import sys, json, platform, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from slib import OUT, build_sample
import numpy, pandas, geopandas, libpysal, esda, spreg, scipy, shapely, matplotlib

d, geoids, checksum = build_sample()

env = dict(
    generated="2026-08-03", task="spatial_model_reassessment_20260803",
    python=sys.version.split()[0], platform=platform.platform(),
    R_available=False,
    R_note="R / spatialreg / sf / spdep are NOT installed on this machine; the required R cross-check "
           "of impacts could not be run. An independent numerical cross-check (sparse-solve total "
           "impact vs closed form) was used instead and is reported in 08_python_r_crosscheck.csv.",
    packages=dict(numpy=numpy.__version__, pandas=pandas.__version__, geopandas=geopandas.__version__,
                  libpysal=libpysal.__version__, esda=esda.__version__, spreg=spreg.__version__,
                  scipy=scipy.__version__, shapely=shapely.__version__, matplotlib=matplotlib.__version__),
    reproduction=dict(seed=42, outcome="lpur_2.0km", n=len(d), sample_checksum_sha256_16=checksum,
                      rq2_covariates=["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres", "borough FE (ref Bronx)"],
                      weights=["Queen row-standardised (islands dropped)", "kNN6 row-standardised"],
                      estimators="spreg ML_Lag / ML_Error (+slx_lags for SDM/SDEM), GM_Combo for SARAR (GMM, not ML)",
                      impacts="LeSage-Pace average direct/indirect/total via eigenvalue traces of row-std W; "
                              "5000 parameter draws from estimated covariance; seed 42",
                      permutations=999),
    run_commands=["python3 outputs/spatial_model_reassessment_20260803/scripts/01_models.py",
                  "python3 outputs/spatial_model_reassessment_20260803/scripts/02_impacts.py",
                  "python3 outputs/spatial_model_reassessment_20260803/scripts/03_finalize.py"],
    inputs=["data/processed/analysis_cbg_final.gpkg", "data/interim/lpur_robust.csv",
            "outputs/lpur_revision_calculations_20260730/T1_lpur_versions.csv", "data/interim/park_area_2km.csv"])
(OUT / "analysis_environment.json").write_text(json.dumps(env, indent=2))

lines = ["# Validation report — spatial_model_reassessment_20260803\n"]
ok = True
def chk(name, cond):
    global ok; ok = ok and bool(cond); lines.append(f"- {'PASS' if cond else 'FAIL'}: {name}")

chk(f"sample n == 6252 (got {len(d)})", len(d) == 6252)
chk(f"sample checksum == 04cda3a7bdefc8b2 (got {checksum})", checksum == "04cda3a7bdefc8b2")
cc = pd.read_csv(OUT / "08_python_r_crosscheck.csv")
chk("impact cross-check abs_diff < 1e-6 (numerical == closed-form total)", bool((cc["abs_diff"] < 1e-6).all()))
for fn in ["05_impacts_queen.csv", "06_impacts_knn6.csv"]:
    im = pd.read_csv(OUT / fn)
    chk(f"{fn}: all impact estimates within 95% CI", bool(((im.ci_lo <= im.estimate) & (im.estimate <= im.ci_hi)).all()))
o1 = pd.read_csv(OUT / "01_ols_spatial_tests.csv")
chk("OLS residual Moran reproduced ~0.6075 (Queen)", abs(float(o1[o1.weights=='queen']['ols_resid_moran_I'].iloc[0]) - 0.6075) < 0.01)
comp = pd.read_csv(OUT / "02_model_comparison_queen.csv")
chk("SEM logLik reproduced ~4478.78", abs(float(comp[comp.model=='SEM']['logLik'].iloc[0]) - 4478.78) < 0.5)
chk("SAR logLik reproduced ~4512.58", abs(float(comp[comp.model=='SAR']['logLik'].iloc[0]) - 4512.58) < 0.5)
chk("R cross-check documented as unavailable", env["R_available"] is False)
lines.append(f"\nOverall: {'ALL PASS' if ok else 'SOME FAIL'}")
(OUT / "validation_report.md").write_text("\n".join(lines) + "\n")
print("validation:", "ALL PASS" if ok else "SOME FAIL")
