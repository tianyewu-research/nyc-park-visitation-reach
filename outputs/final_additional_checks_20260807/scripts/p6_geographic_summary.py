"""PHASE 6 — verified geographic summary for Results writing.

Recomputes the FDR-adjusted LISA per CBG on the LPUR-observed sample (n=6,576) using the sanctioned
recipe (Queen row-standardized, Moran_Local 999 perm seed 42, Benjamini-Hochberg q<0.05) from
outputs/lpur_revision_calculations_20260730/scripts/rev_fast.py, then breaks results down by borough.
No NTA / community-district boundary layer is available in the repo (only borough boundaries), so the
optional NTA/CD join is NOT feasible and is documented as such — no corridor/road inference is made."""
import sys, numpy as np, pandas as pd, geopandas as gpd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUTDIR, SEED
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
from statsmodels.stats.multitest import multipletests

np.random.seed(SEED)
GPKG = ROOT / "data" / "processed" / "analysis_cbg_final.gpkg"
BORO = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island"}
# sanctioned FDR totals (lpur_revision_calculations_20260730/T2_lisa_fdr_counts.csv)
SANCTIONED = {"HH": 1068, "LL": 1375, "HL": 31, "LH": 37, "ns": 4065, "TOTAL": 6576}

g = gpd.read_file(GPKG).to_crs("EPSG:2263")
d = g.dropna(subset=["lpur_2.0km"]).reset_index(drop=True)
w = Queen.from_dataframe(d, use_index=False); w.transform = "r"
gmi = Moran(d["lpur_2.0km"].values, w, permutations=999)
assert abs(gmi.I - 0.718) < 0.02, f"Global Moran's I {gmi.I} != 0.718 — data version issue, STOP"

lisa = Moran_Local(d["lpur_2.0km"].values, w, permutations=999, seed=SEED)
qmap = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
p = lisa.p_sim.copy()
fdr_sig, q_fdr, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
d["lisa_fdr"] = ["ns"] * len(d)
for i in range(len(d)):
    if fdr_sig[i]:
        d.loc[i, "lisa_fdr"] = qmap[lisa.q[i]]
d["borough"] = d["GEOID"].str[2:5].map(BORO)

# closure check against sanctioned totals
counts = {c: int((d.lisa_fdr == c).sum()) for c in ["HH", "LL", "HL", "LH", "ns"]}
closure_ok = all(counts[c] == SANCTIONED[c] for c in counts) and len(d) == SANCTIONED["TOTAL"]

# ---- borough-level summary ----
rows = []
for boro in ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]:
    sub = d[d.borough == boro]
    nb = len(sub)
    row = dict(borough=boro, n_cbg=nb,
               lpur_mean=round(sub["lpur_2.0km"].mean(), 4),
               lpur_median=round(sub["lpur_2.0km"].median(), 4),
               lpur_p25=round(sub["lpur_2.0km"].quantile(.25), 4),
               lpur_p75=round(sub["lpur_2.0km"].quantile(.75), 4))
    for c in ["HH", "LL", "HL", "LH", "ns"]:
        cc = int((sub.lisa_fdr == c).sum())
        row[f"{c}_n"] = cc
        row[f"{c}_pct"] = round(100 * cc / nb, 1) if nb else 0.0
    rows.append(row)
# city-wide row
allrow = dict(borough="NYC (all)", n_cbg=len(d),
              lpur_mean=round(d["lpur_2.0km"].mean(), 4), lpur_median=round(d["lpur_2.0km"].median(), 4),
              lpur_p25=round(d["lpur_2.0km"].quantile(.25), 4), lpur_p75=round(d["lpur_2.0km"].quantile(.75), 4))
for c in ["HH", "LL", "HL", "LH", "ns"]:
    allrow[f"{c}_n"] = counts[c]; allrow[f"{c}_pct"] = round(100 * counts[c] / len(d), 1)
rows.append(allrow)
summ = pd.DataFrame(rows)
summ.to_csv(OUTDIR / "borough_lpur_lisa_summary.csv", index=False)

# ---- facts md ----
lines = ["# Geographic results facts (PHASE 6)\n",
         f"Recomputed on the LPUR-observed sample n={len(d)} (Queen row-standardized, "
         f"Moran_Local 999 permutations, seed {SEED}, Benjamini-Hochberg q<0.05).",
         f"Global Moran's I (LPUR, 2 km) = {gmi.I:.3f} (p_sim={gmi.p_sim:.3f}).",
         f"Closure against sanctioned FDR totals (T2_lisa_fdr_counts.csv): "
         f"{'PASS — per-CBG classes reproduce the sanctioned counts exactly' if closure_ok else 'MISMATCH — DIAGNOSE'}.",
         f"  Reproduced HH={counts['HH']} LL={counts['LL']} HL={counts['HL']} LH={counts['LH']} ns={counts['ns']}.\n",
         "## Borough-level facts (verified; describe WHERE, reserve WHY for Discussion)"]
for _, r in summ[summ.borough != "NYC (all)"].iterrows():
    lines.append(f"- **{r['borough']}** (n={r['n_cbg']}): mean LPUR {r['lpur_mean']}, median {r['lpur_median']}; "
                 f"FDR high–high (locally-oriented) {r['HH_n']} ({r['HH_pct']}%), "
                 f"low–low (travel-farther) {r['LL_n']} ({r['LL_pct']}%).")
hh_lead = summ[summ.borough != "NYC (all)"].sort_values("HH_pct", ascending=False).iloc[0]
ll_lead = summ[summ.borough != "NYC (all)"].sort_values("LL_pct", ascending=False).iloc[0]
hi_lpur = summ[summ.borough != "NYC (all)"].sort_values("lpur_median", ascending=False).iloc[0]
lo_lpur = summ[summ.borough != "NYC (all)"].sort_values("lpur_median").iloc[0]
lines += ["", "## Supported summary statements",
          f"- Highest median LPUR: {hi_lpur['borough']} ({hi_lpur['lpur_median']}); "
          f"lowest: {lo_lpur['borough']} ({lo_lpur['lpur_median']}).",
          f"- Largest share of high–high (locally-oriented) clusters: {hh_lead['borough']} "
          f"({hh_lead['HH_pct']}% of its CBGs).",
          f"- Largest share of low–low (travel-farther) clusters: {ll_lead['borough']} "
          f"({ll_lead['LL_pct']}% of its CBGs).",
          "",
          "## NTA / community-district join",
          "NOT feasible: the repository contains only a borough boundary layer "
          "(`data/raw/NYC行政边界/Borough_Boundaries_20260623.geojson`); no authoritative NTA or "
          "community-district boundary is available. No sub-borough spatial join was performed, and no "
          "`nta_or_community_district_lpur_lisa_summary.csv` is emitted. Do NOT describe patterns "
          "'along a corridor' or road — no corridor was overlaid or tested.",
          "",
          "Caveat: HH = high LPUR surrounded by high LPUR (locally-oriented neighborhoods); "
          "LL = low LPUR surrounded by low LPUR (residents travel farther). These are spatial-association "
          "clusters of an observed ratio, not causal statements."]
(OUTDIR / "geographic_results_facts.md").write_text("\n".join(lines) + "\n")

print("=== PHASE 6 done ===")
print("Global Moran I:", round(float(gmi.I), 4), "| closure:", "PASS" if closure_ok else "MISMATCH")
print(summ[["borough", "n_cbg", "lpur_median", "HH_n", "HH_pct", "LL_n", "LL_pct"]].to_string(index=False))
