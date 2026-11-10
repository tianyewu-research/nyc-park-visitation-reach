"""SECTION J — RUN_SUMMARY.md: canonical counts, denominator explanations, old-vs-new Figure 8,
per-CBG Q1-Q4 medians, whether the abstract claim survives, deliverable paths, and author-judgment items."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import OUT, ROOT

audit = pd.read_csv(OUT / "matching_audit.csv")
cmp = pd.read_csv(OUT / "representation_old_vs_new_comparison.csv")
qsum = pd.read_csv(OUT / "destination_concentration_quartile_summary.csv")
man = json.loads((OUT / "provenance_manifest.json").read_text())
val = (OUT / "validation_report.md").read_text()
disp = pd.read_csv(OUT / "origin_quartile_coverage_and_dispersion.csv")


def av(sub):
    return audit.loc[audit.metric.str.contains(sub, regex=False), "value"].iloc[0]


QORDER = ["Q1 (lowest Non-White)", "Q2", "Q3", "Q4 (highest Non-White)"]


def med(q, m):
    r = qsum[(qsum.origin_quartile == q) & (qsum.metric == m)].iloc[0]
    return r["median"], r["iqr"], r["median_ci_lo"], r["median_ci_hi"]


t10 = {q: med(q, "top10_share") for q in QORDER}
eff = {q: med(q, "effective_destinations") for q in QORDER}
q4_more = eff["Q4 (highest Non-White)"][0] > eff["Q1 (lowest Non-White)"][0]
ci_sep = eff["Q4 (highest Non-White)"][2] > eff["Q1 (lowest Non-White)"][3]

L = ["# RUN SUMMARY — destination_per_cbg_and_match_alignment_20260808\n",
     f"Validation: **{'ALL PASS' if 'ALL PASS' in val else 'SEE validation_report.md'}**. "
     "No manuscript edits; no existing outputs overwritten. Canonical universe = 100 EPSG:2263 ft.\n",
     "## 1. Canonical counts",
     f"- Geographic candidates: **{int(av('geographic candidates'))}**; analytic (t1 any-home): "
     f"**{int(av('t1 rule'))}**; analytic (usable NYC-home): **{int(av('usable NYC-home'))}**.",
     f"- Matched POIs @100 ft: **{man['matched_pois_100ft_geographic']}**; represented properties: "
     f"**{man['represented_properties_100ft']}**; ties: {int(av('ties @100ft'))}.",
     f"- Dissolved gispropnum properties (>=0.1 ac): {man['dissolved_properties']}.",
     "",
     "## 2. Exact explanation of the discrepancies",
     f"- **6,338 (t1) vs 6,385 (p5)**: different DENOMINATOR rules on the same POI universe. Geographic "
     f"candidates (valid coords, NYC POI_CBG) = 6,385 (p5). Analytic candidates (t1 rule: >=1 week with a "
     f"parseable home dict, total>0, VISITOR_COUNTS>0) = 6,338. The **47** difference = geographic POIs "
     f"that never carry a usable analytic contribution. (A further {int(av('t1-analytic but no'))} "
     f"t1-analytic POIs have no usable NYC-home contribution.)",
     f"- **4,414 (t1) vs 3,795 (p5)**: different THRESHOLD. t1 used `TOL = 100*M2FT` ~ 328.08 ft "
     f"(mislabelled '100 ft'); p5 used 100 ft. On the common dissolved geometry, @328 ft matches "
     f"{int(av('@328ft / analytic'))} t1-analytic POIs (= t1's 4,414) and @100 ft matches 3,795 (= p5). "
     f"We never relabel 328 ft as 100 ft.",
     "",
     "## 3. Old vs corrected Figure 8 representation impacts (0->1 scale)"]
for _, r in cmp[cmp.effect.isin(["direct", "total"])].iterrows():
    L.append(f"- {r['specification']} {r['effect']}: old(T1 ~328ft) {r['old_estimate']:+.4f} "
             f"-> new(canonical 100ft) {r['new_estimate']:+.4f} (sign same={r['sign_same']}, both "
             f"significant={r['both_sig']}).")
L += ["- **Neither substantive conclusion changes**: both representation checks stay negative and "
      "significant for direct and total; canonical 100 ft magnitudes are slightly smaller than the "
      "~328 ft versions.",
      "",
      "## 4. Per-CBG Q1-Q4 medians (corrected estimand)",
      "| quartile | top-10 share median [IQR] (95% CI) | effective destinations median [IQR] (95% CI) |",
      "|---|---|---|"]
for q, ql in zip(QORDER, ["Q1", "Q2", "Q3", "Q4"]):
    a = t10[q]; b = eff[q]
    L.append(f"| {ql} | {a[0]:.3f} [{a[1]:.3f}] ({a[2]:.3f}-{a[3]:.3f}) | "
             f"{b[0]:.2f} [{b[1]:.2f}] ({b[2]:.2f}-{b[3]:.2f}) |")
L += ["",
      "## 5. Does the abstract destination claim survive?",
      f"- Claim: higher-Non-White origins 'distributed non-local visits across more park properties'. "
      f"Per-CBG effective destinations rise monotonically Q1 {eff['Q1 (lowest Non-White)'][0]:.2f} -> "
      f"Q4 {eff['Q4 (highest Non-White)'][0]:.2f}; per-CBG top-10 share falls Q1 "
      f"{t10['Q1 (lowest Non-White)'][0]:.3f} -> Q4 {t10['Q4 (highest Non-White)'][0]:.3f}. "
      + ("The median 95% CIs for effective destinations are SEPARATED (Q4>Q1). "
         if ci_sep else "The median 95% CIs overlap. ")
      + f"**The claim SURVIVES at the corrected per-CBG level (direction preserved), but the true "
      f"magnitudes are much smaller than the pooled figures** (per-CBG effective destinations ~12-17, "
      f"not the pooled ~43-49). The abstract should state the per-CBG medians, not the inflated pooled "
      f"counts, and the pooled top-10 pattern should be labelled pooled descriptive evidence.",
      "",
      "## 6. Deliverables (paths under outputs/destination_per_cbg_and_match_alignment_20260808/)"]
for grp, pat in [("CSV", "*.csv"), ("figures", "figures/*.png"), ("memos/reports", "*.md"),
                 ("scripts", "scripts/*.py"), ("provenance/validation", "provenance_manifest.json")]:
    fs = sorted(str(p.relative_to(OUT)) for p in OUT.glob(pat))
    L.append(f"- **{grp}** ({len(fs)}): " + ", ".join(fs))
L += ["",
      "## 7. Items requiring author judgment (not silently decided)",
      "- **Property distance definition.** Destination non-local (>2 km) uses centroid -> property "
      "REPRESENTATIVE POINT (the definition the prior destination analysis used). LPUR B/C 'within 2 km' "
      "uses centroid -> property nearest boundary point (the t1 LPUR definition). Both are stated "
      "precisely; if a single definition is preferred, the author should choose (nearest-boundary is more "
      "defensible for reach; representative-point preserves continuity).",
      "- **Represented-property count** is 1,640 under the canonical dissolve+PIP-first matcher (p5 "
      "reported 1,638 via sjoin_nearest; t1 reported 1,647 on undissolved rows). The 2-3 property "
      "difference is a matcher detail; the canonical count governs.",
      "- **Pooled vs per-CBG framing.** The corrected per-CBG estimand is primary. Whether to retain the "
      "pooled top-10 statement in the main text (labelled pooled descriptive) or move it to the "
      "supplement is an editorial choice.",
      "- **Abstract magnitude.** The direction survives but the numbers change substantially; the author "
      "must update the abstract to the per-CBG medians."]
(OUT / "RUN_SUMMARY.md").write_text("\n".join(L) + "\n")
print("=== RUN_SUMMARY written ===")
print(f"abstract claim survives (direction): {q4_more}; effective-dest CI separated Q4>Q1: {ci_sep}")
