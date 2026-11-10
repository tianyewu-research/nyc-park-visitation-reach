"""PHASE 0 — reproduction gate. Verify frozen GeoPackage hash, reproduce the primary RQ2 sample
sizes and the primary Non-White LPUR average direct/total impacts BEFORE any new analysis. If any
check fails beyond rounding tolerance, write BLOCKED_reproduction_mismatch.md and stop."""
import sys, json, hashlib
import numpy as np, pandas as pd, geopandas as gpd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUTDIR, SEED
from slib import build_sample, make_W, CONT, DEP
from slib2 import fit_sdm, trace_powers, sdm_impacts, design_sdm
import spreg

EXPECT_HASH = "51dd9f29d17fb9c93376c9cd665023689682789c9422d22159c038b11b300927"
GPKG = ROOT / "data" / "processed" / "analysis_cbg_final.gpkg"
TOL = 5e-4  # rounding tolerance on impact reproduction

np.random.seed(SEED)
checks = []


def chk(name, passed, reproduced, target, note=""):
    checks.append(dict(check=name, reproduced=reproduced, target=target,
                       ok=bool(passed), note=note))


# 1) GeoPackage hash
h = hashlib.sha256(GPKG.read_bytes()).hexdigest()
chk("frozen GeoPackage SHA-256", h == EXPECT_HASH, h, EXPECT_HASH)

# 2) Primary aggregate RQ2 sample
d, gid, checksum = build_sample()
n_agg = len(d)
chk("primary aggregate RQ2 sample n", n_agg == 6252, n_agg, 6252,
    "build_sample(): dropna on income/nonwhite/popdens/park_area + Queen islands dropped")
chk("sample GEOID checksum (sha256[:16])", checksum == "04cda3a7bdefc8b2", checksum,
    "04cda3a7bdefc8b2")

# 3) Joint-composition LPUR sample (Black/Hispanic/Asian complete case on the LPUR sample)
g = gpd.read_file(GPKG).to_crs("EPSG:2263")
g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0
CONT_J = ["income_10k", "pct_black", "pct_hispanic", "pct_asian", "popdens_k", "park_area_acres"]
dj = g.dropna(subset=[DEP] + CONT_J + ["borough"]).reset_index(drop=True)
from libpysal.weights import Queen
wj = Queen.from_dataframe(dj, use_index=False)
if wj.islands:
    dj = dj.iloc[[i for i in range(len(dj)) if i not in set(wj.islands)]].reset_index(drop=True)
n_joint_lpur = len(dj)
# Archived output documents joint LPUR n = 6252 (same as aggregate: race completeness == nonwhite
# completeness) and joint DISTANCE n = 6246. The prompt's "6246" is the distance-sample figure.
chk("joint-composition LPUR sample n (archived documents 6,252)", n_joint_lpur == 6252,
    n_joint_lpur, 6252, "race shares share ACS B03002 completeness with pct_nonwhite")

# 4) Distance sample (mean_trip_km complete case).
# The prompt/Figure_A caption cite 6,246, but the archived distance_model_diagnostics.csv explicitly
# documents n=6,252 for every distance model. Per the prompt ("unless the archived final output
# explicitly documents otherwise") the authoritative target is the archived 6,252, which reproduces
# exactly. The 6,246 caption figure is stale and is flagged for correction.
dd = g.dropna(subset=["mean_trip_km"] + CONT + ["borough"]).reset_index(drop=True)
wd = Queen.from_dataframe(dd, use_index=False)
if wd.islands:
    dd = dd.iloc[[i for i in range(len(dd)) if i not in set(wd.islands)]].reset_index(drop=True)
n_dist = len(dd)
arch_dist = int(pd.read_csv(ROOT / "outputs" / "final_sdm_revision_20260803" /
                            "distance_model_diagnostics.csv")["n"].dropna().iloc[0])
chk("distance sample n (archived distance_model_diagnostics.csv documents this)",
    n_dist == arch_dist == 6252, n_dist, arch_dist,
    "reproduces archived 6,252; Figure_A caption's 6,246 is stale — must be corrected")

# 5) Reproduce the primary Non-White LPUR average direct & total impacts (IQR scale, Queen)
wq = make_W(d, "queen")
Tq = trace_powers(wq, Q=200, m_probe=4000)
m, X = fit_sdm(d, wq)
iqr_nw = float(d["pct_nonwhite"].quantile(.75) - d["pct_nonwhite"].quantile(.25))
scales = {c: [("unit", 1.0)] for c in CONT}
scales["pct_nonwhite"] = [("IQR", iqr_nw)]
rows, ret, pmult = sdm_impacts(m, X, wq, Tq, CONT, scales, nsim=5000)
nw = {r["effect"]: r for r in rows if r["variable"] == "pct_nonwhite"}
dir_est = float(nw["direct"]["estimate"]); tot_est = float(nw["total"]["estimate"])
chk("Non-White LPUR average DIRECT impact (IQR, Queen)", abs(dir_est - (-0.021969)) < TOL,
    round(dir_est, 6), -0.021969)
chk("Non-White LPUR average TOTAL impact (IQR, Queen)", abs(tot_est - (-0.064606)) < TOL,
    round(tot_est, 6), -0.064606)

df = pd.DataFrame(checks)
df.to_csv(OUTDIR / "phase0_reproduction_checks.csv", index=False)
all_ok = bool(df["ok"].all())

print("=== PHASE 0 reproduction gate ===")
print(df.to_string(index=False))
print("\nOverall:", "ALL PASS — proceed" if all_ok else "MISMATCH — STOP")

if not all_ok:
    failed = df[~df["ok"]]
    lines = ["# BLOCKED — reproduction mismatch\n",
             "PHASE 0 reproduction gate failed. New analysis was NOT started. The frozen pipeline "
             "must be reconciled before continuing.\n", "## Failed checks\n"]
    for _, r in failed.iterrows():
        lines.append(f"- {r['check']}: reproduced={r['reproduced']} target={r['target']} ({r['note']})")
    (OUTDIR / "BLOCKED_reproduction_mismatch.md").write_text("\n".join(lines) + "\n")
    sys.exit(1)

# stash reproduced sample sizes for downstream provenance
prov = dict(n_agg=n_agg, n_joint_lpur=n_joint_lpur, n_dist=n_dist, checksum=checksum,
            nw_direct_iqr=round(dir_est, 6), nw_total_iqr=round(tot_est, 6),
            iqr_nonwhite=round(iqr_nw, 6), gpkg_sha256=h)
(OUTDIR / "phase0_provenance.json").write_text(json.dumps(prov, indent=2))
print("\nprovenance stashed:", OUTDIR / "phase0_provenance.json")
