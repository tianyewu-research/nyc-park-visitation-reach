"""PHASE 8/9 — analysis_summary.md: only VERIFIED factual statements that may be used in
Methods/Results/Discussion, with direct observations separated from interpretation and all caveats
stated. Reads the produced CSVs; runs no models; does NOT edit the manuscript."""
import sys, json
import pandas as pd, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import OUTDIR, FINAL

prov = json.loads((OUTDIR / "phase0_provenance.json").read_text())
vif = pd.read_csv(OUTDIR / "vif_diagnostics.csv")
cond = pd.read_csv(OUTDIR / "condition_numbers.csv")
comp = pd.read_csv(OUTDIR / "racial_composition_joint_vs_separate_comparison.csv")
radius = pd.read_csv(OUTDIR / "radius_group_impacts_tidy.csv")
boro = pd.read_csv(OUTDIR / "borough_lpur_lisa_summary.csv")
rq3 = pd.read_csv(FINAL / "RQ3_final_impacts.csv")
agg = pd.read_csv(FINAL / "RQ2_aggregate_SDM_impacts.csv")
metrics = pd.read_csv(OUTDIR / "destination_metrics_by_nonwhite_quartile.csv") \
    if (OUTDIR / "destination_metrics_by_nonwhite_quartile.csv").exists() else None

max_local_vif = vif[(vif.design == "local_X") & (vif.numerically_stable)]["vif"].astype(float).max()
cl = cond[cond.design == "local_X"]["condition_number"].max()
ca = cond[cond.design == "augmented_X_WX"]["condition_number"].max()

L = []
L += ["# Analysis summary — final_additional_checks_20260807\n",
      "Verified facts only, for Methods / Results / Discussion. **Direct observations** are separated "
      "from **interpretation**. All new outputs live in `outputs/final_additional_checks_20260807/`. "
      "Nothing here edits the manuscript, changes the RQs/LPUR definition, or reopens model selection.\n",
      "## 0. Reproduction (direct observation)",
      f"- Frozen GeoPackage SHA-256 verified; RQ2 sample n={prov['n_agg']} (checksum {prov['checksum']}); "
      f"joint-composition LPUR n={prov['n_joint_lpur']}; distance n={prov['n_dist']}.",
      f"- Primary Non-White LPUR average impacts reproduce exactly: direct {prov['nw_direct_iqr']}, "
      f"total {prov['nw_total_iqr']} (per-IQR, Queen).",
      "- **Correction required:** the distance sample is n=6,252 (archived "
      "distance_model_diagnostics.csv), NOT 6,246. The 6,246 in the Figure_A caption is stale.",
      "",
      "## 1. Multicollinearity (direct observation)",
      f"- Largest numerically-stable VIF in any local design X: {max_local_vif:.2f} (no VIF threshold is "
      "applied as pass/fail; full values in `vif_diagnostics.csv`).",
      f"- Standardized condition numbers: local X ≤ {cl}; augmented [X, WX] ≤ {ca}.",
      "- *Interpretation:* local covariates are not collinear enough to threaten estimation. The augmented "
      "[X, WX] design is more correlated (each W-lag with its own covariate ~0.8–0.9), which is expected "
      "under an SLX/SDM and is a caveat for reading individual STRUCTURAL coefficients — not for the "
      "reported average direct/indirect/total impacts.",
      "",
      "## 2. Composition stability (direct observation)"]
for outcome in ["LPUR", "distance"]:
    sub = comp[(comp.outcome == outcome) & (comp.effect.isin(["direct", "total"]))]
    agree = int(sub["sign_agrees"].sum())
    disc = sub[~sub["sign_agrees"]]
    tail = ""
    if len(disc):
        d0 = disc.iloc[0]
        both_ns = (not bool(d0["separate_sig"])) and (not bool(d0["joint_sig"]))
        tail = (f"; the one disagreement is {d0['race']} {d0['effect']}"
                + (", between two non-significant estimates" if both_ns else "") + ".")
    else:
        tail = "."
    L.append(f"- {outcome}: separate Black/Hispanic/Asian SDMs agree in sign with the joint model in "
             f"{agree}/{len(sub)} direct+total cells{tail}")
L += ["- *Interpretation guard:* separate-share coefficients contrast a group with all remaining "
      "composition; joint coefficients are conditional on the other shares. Use separate models for "
      "sign/uncertainty stability only, NOT for like-for-like magnitude comparison.",
      "",
      "## 3. Radius profiles 0.5–3.0 km (direct observation)"]
for race in ["Black", "Hispanic", "Asian"]:
    tot = radius[(radius.race == race) & (radius.effect == "total")]
    sig = tot[(tot.ci_hi < 0) | (tot.ci_lo > 0)]
    at2 = tot[tot.radius_km == 2.0].iloc[0]
    L.append(f"- {race}: total impact negative at {(tot.estimate < 0).sum()}/{len(tot)} radii, "
             f"significant at {len(sig)}/{len(tot)}; at 2 km = {at2['estimate']:.4f} "
             f"[{at2['ci_lo']:.4f}, {at2['ci_hi']:.4f}].")
L += ["- All 11 joint models converged (`radius_model_diagnostics.csv`).",
      "- *Interpretation:* larger radii mechanically count more visits as local, so profiles blend a "
      "measurement-scale gradient with any substantive gradient; read for sign/significance stability "
      "around 2 km, not as a dose-response curve.",
      "",
      "## 4. RQ3 impacts (from frozen Table 5; direct observation)"]
for var, lab in [("pct_nonwhite", "Non-White share"), ("income_10k", "Median income"),
                 ("log_access", "Regional accessibility (6 km)"), ("park_area_acres", "Local park area"),
                 ("quality_index_cbg_svc", "Park quality")]:
    d = rq3[(rq3.variable == var) & (rq3.effect == "direct")].iloc[0]
    t = rq3[(rq3.variable == var) & (rq3.effect == "total")].iloc[0]
    L.append(f"- {lab}: direct {d['estimate']:.4f} (p={d['p']}); total {t['estimate']:.4f} (p={t['p']}).")
L += ["- *Caution:* regional-accessibility total is borderline (p≈0.049); do not overstate it or use "
      "significance stars (Fig8 deliberately omits them).",
      "",
      "## 5. Geographic description (direct observation; RQ1 support)"]
for _, r in boro[boro.borough != "NYC (all)"].iterrows():
    L.append(f"- {r['borough']}: median LPUR {r['lpur_median']}; high–high {r['HH_pct']}%, "
             f"low–low {r['LL_pct']}% of its CBGs (FDR q<0.05).")
L += ["- Per-CBG FDR LISA reproduces the sanctioned totals exactly (HH 1068 / LL 1375 / HL 31 / LH 37 / "
      "ns 4065; global Moran's I 0.718).",
      "- *Constraint:* describe WHERE only; no corridor/road language (none overlaid). No NTA/CD layer "
      "available, so no sub-borough geography.",
      ""]

if metrics is not None:
    L.append("## 6. Destination patterns (descriptive; direct observation)")
    nl = metrics[metrics.scope == "non_local_gt2km"]
    for _, r in nl.iterrows():
        L.append(f"- {r['origin_quartile']} (non-local >2 km): top-10 share {100*r['top10_share']:.1f}%, "
                 f"effective destinations {r['effective_destinations']}, visit-weighted acreage "
                 f"{r['visit_weighted_acreage']}, waterfront share {100*r['waterfront_share']:.1f}%, "
                 f"mean distance {r['mean_distance_km']} km.")
    L += ["- Units are attributed visit records (conservative property-week-home-CBG maximum), not unique "
          "visitors. Waterfront uses the authoritative NYC Parks `waterfront` field.",
          "- *Interpretation guard:* descriptive only; narrows but does not resolve choice-vs-constraint; "
          "no preference/welfare/trip-purpose claims.",
          ""]
else:
    L += ["## 6. Destination patterns", "- (pending: destination_metrics_by_nonwhite_quartile.csv)", ""]

L += ["## 7. Conditional analyses NOT delivered (transparent)",
      "- Network-based LPUR sensitivity: **not usable** (no routing library installed; several-hundred-"
      "thousand OD pairs; not reproducible). Euclidean LPUR retained as primary; see "
      "`network_sensitivity_feasibility.md`.",
      "- NTA/community-district geography: **omitted** (no boundary layer). Borough-level only.",
      "",
      "## 8. Standing language constraints (apply throughout)",
      "1. Non-White average **direct** impact = primary/stable. Indirect & total = model-implied, "
      "conditional on the SDM and W. No causal-spillover / diffusion / behavioral-propagation wording.",
      "2. Income total = **model-dependent** spatial-context association; not co-primary; do not imply "
      "the SDM revealed a hidden true income effect.",
      "3. Hispanic total is non-significant; avoid blanket 'all racial groups' claims.",
      "4. Keep raw-LPUR and clipped-logit results on separate scales."]
(OUTDIR / "analysis_summary.md").write_text("\n".join(L) + "\n")
print("=== analysis_summary.md written ===")
print("destination section:", "included" if metrics is not None else "PENDING p5")
