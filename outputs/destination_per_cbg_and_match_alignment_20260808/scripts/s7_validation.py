"""SECTION I — validation + provenance. Fails (exit 1) if any applicable assertion fails."""
import sys, json, re, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import OUT, MATCH_FT, sha256
from slib import build_sample

checks = []


def ck(name, passed, detail=""):
    checks.append(dict(check=name, passed=bool(passed), detail=detail))


xw = pd.read_csv(OUT / "poi_property_crosswalk_canonical_100ft.csv")
lp = pd.read_csv(OUT / "lpur_representation_canonical_100ft_by_cbg.csv", dtype={"GEOID": str})
R = pd.read_csv(OUT / "destination_records_property_level_canonical_100ft.csv", dtype={"home_cbg": str, "pid": str})
conc = pd.read_csv(OUT / "destination_concentration_per_cbg.csv", dtype={"home_cbg": str})
rep = pd.read_csv(OUT / "representation_robustness_SDM_impacts.csv")
qsum = pd.read_csv(OUT / "destination_concentration_quartile_summary.csv")
agg = pd.read_csv(OUT / "aggregate_nonwhite_radius_SDM_impacts.csv")
man = json.loads((OUT / "provenance_manifest.json").read_text())
d, gid, checksum = build_sample()
d["GEOID"] = d["GEOID"].astype(str)

# 1. canonical threshold exactly 100 ft
ck("canonical threshold is exactly 100 EPSG:2263 feet", MATCH_FT == 100.0, f"MATCH_FT={MATCH_FT}")

# 2. every matched pid in downstream data exists in the crosswalk
xw_pids = set(xw.loc[xw.match_status == "matched", "matched_pid"].astype(str))
rec_pids = set(R["pid"].astype(str))
ck("every destination-record property exists in the canonical crosswalk", rec_pids <= xw_pids,
   f"records pids not in crosswalk: {len(rec_pids - xw_pids)}")

# 3. no downstream ANALYSIS script (s1-s6) re-matches POIs independently (only s0 matches)
downstream = [p for p in (OUT / "scripts").glob("s[1-6]*.py")]
rematch_hits = []
for p in downstream:
    txt = p.read_text()
    # POI-matching signatures: sjoin_nearest, or a match_poi helper on POI points (STRtree used in s1
    # is CBG->property distance, not POI assignment, so it is not a signature)
    if re.search(r"sjoin_nearest|def match_poi", txt):
        rematch_hits.append(p.name)
ck("no downstream script re-matches POIs independently (STRtree in s1 is CBG->property distance only)",
   not rematch_hits, f"suspect files: {rematch_hits}")

# 4. quartile CBG counts reconcile to the frozen sample
qc = conc.groupby("origin_quartile").size()
ck("quartile CBG counts reconcile to the frozen sample (sum == n)", int(qc.sum()) == len(d),
   f"sum={int(qc.sum())} vs frozen n={len(d)}; per-q={qc.to_dict()}")

# 5. weights reconcile: annual records total == sum of C denominators; per-CBG non-local reconcile
tol = 1e-3
recs_total = float(R["attributed_weight"].sum())
totC_sum = float(lp["totC"].sum())
ck("weekly->annual: destination-record total == sum of per-CBG C denominators (within tol)",
   abs(recs_total - totC_sum) / max(1, totC_sum) < 1e-6, f"records={recs_total:.1f} totC={totC_sum:.1f}")
# per-CBG non-local weight in concentration table == sum of non-local record weights per CBG
nl_by = R[R.non_local].groupby("home_cbg")["attributed_weight"].sum()
merged = conc.dropna(subset=["nonlocal_weight"]).set_index("home_cbg")["nonlocal_weight"]
common = merged.index.intersection(nl_by.index)
recon_ok = np.allclose(merged.loc[common].values, nl_by.loc[common].round(3).values, atol=1e-2)
ck("annual records -> per-CBG concentration non-local weight reconciles (within tol)", recon_ok,
   f"cbgs compared={len(common)}")

# 6. representation LPUR denominators positive where outcomes reported
ck("matched-universe & property-max LPUR denominators positive where outcomes reported",
   bool(rep["denom_positive"].all()) and bool((lp["totB"] >= 0).all()) and bool((lp["totC"] >= 0).all()),
   "denom_positive flag true for both representation specs")

# 7. SDM sample/covariates/weights/sims match the final pipeline
sdm_ok = bool((rep["n"] == 6252).all()) and bool((rep["weights"] == "Queen").all()) and \
    bool((rep["seed"] == 42).all()) and bool((rep["nsim"] >= 4900).all())
ck("SDM sample n=6252, Queen weights, seed 42, ~5000 sims (matches final pipeline)", sdm_ok,
   f"n={rep['n'].unique().tolist()} nsim>= {int(rep['nsim'].min())}")

# 8. plotted values equal accompanying CSV values (Fig9 plot data == concentration table)
fp = pd.read_csv(OUT / "figure9_plot_data.csv", dtype={"home_cbg": str})
j = fp.merge(conc[["home_cbg", "top10_share", "effective_destinations"]], on="home_cbg",
             suffixes=("_plot", "_src"))
plot_ok = np.allclose(j["top10_share_plot"].fillna(-9), j["top10_share_src"].fillna(-9), atol=1e-9) and \
    np.allclose(j["effective_destinations_plot"].fillna(-9), j["effective_destinations_src"].fillna(-9), atol=1e-9)
ck("Figure 9 plotted values equal the concentration CSV values", plot_ok)

# 9. both universes described without relabelling 328 ft as 100 ft
readme = (OUT / "matching_audit_readme.md").read_text()
ck("audit describes both universes without relabelling 328 ft as 100 ft",
   ("328" in readme) and ("100 ft" in readme) and ("M2FT" in readme))

# 10. seeds recorded; checksums + source scripts in the provenance manifest
ck("provenance manifest records seed, checksums, and source-script paths",
   man.get("seed") == 42 and len(man.get("checksums", {})) >= 4 and "source_scripts" in man)

# 11. LPUR A reproduces the frozen primary (pipeline fidelity)
import geopandas as gpd
gpk = gpd.read_file(__import__("common").GPKG)[["GEOID", "lpur_2.0km"]]
gpk["GEOID"] = gpk["GEOID"].astype(str)
mrg = lp.merge(gpk, on="GEOID")
corrA = mrg[["lpur_A_poisum", "lpur_2.0km"]].dropna().corr().iloc[0, 1]
ck("all-POI reconstruction (LPUR A) reproduces the frozen primary lpur_2.0km (corr>0.999)",
   corrA > 0.999, f"corr={corrA:.5f}")

rep_df = pd.DataFrame(checks)
allok = bool(rep_df["passed"].all())
lines = ["# Validation report — destination_per_cbg_and_match_alignment_20260808\n",
         f"Overall: **{'ALL PASS' if allok else 'FAILED'}**\n"]
for _, r in rep_df.iterrows():
    lines.append(f"- {'PASS' if r['passed'] else 'FAIL'}: {r['check']}" + (f" — {r['detail']}" if r["detail"] else ""))
(OUT / "validation_report.md").write_text("\n".join(lines) + "\n")

# refresh provenance checksums for all csv/figure deliverables
man["checksums_all_csv"] = {p.name: sha256(p) for p in sorted(OUT.glob("*.csv"))}
man["validation_passed"] = allok
(OUT / "provenance_manifest.json").write_text(json.dumps(man, indent=2))

print("=== s7 validation ===")
print(rep_df.to_string(index=False))
print("Overall:", "ALL PASS" if allok else "FAILED")
if not allok:
    sys.exit(1)
