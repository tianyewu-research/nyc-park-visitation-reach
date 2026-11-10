"""SECTION E (+ pooled comparison) — make the CBG, not the pooled quartile, the unit of destination
concentration.

For each origin CBG in the frozen RQ2 sample, restrict to its NON-LOCAL (>2 km) destination records,
sum annual attributed weight by park property, normalize within the CBG, and compute concentration
metrics (top-5/top-10 share, HHI, effective destinations = 1/HHI, weighted mean/median distance, plus
size/waterfront composition). CBGs with zero non-local weight are retained with NA metrics and counted
by quartile. Quartile summaries use CBGs as the resampling unit for a bootstrap 95% CI of the median.
The old pooled-quartile metrics are kept only as a clearly labelled supplementary comparison."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUT, SEED, sha256
from slib import build_sample
import scipy.stats as st

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

R = pd.read_csv(OUT / "destination_records_property_level_canonical_100ft.csv", dtype={"home_cbg": str, "pid": str})

# frozen RQ2 sample + Non-White quartiles (same cut points as the corrections)
d, gid, checksum = build_sample()
d["GEOID"] = d["GEOID"].astype(str)
qs = d["pct_nonwhite"].quantile([.25, .5, .75]).values
cutpoints = dict(q25=round(float(qs[0]), 6), q50=round(float(qs[1]), 6), q75=round(float(qs[2]), 6))
nw_map = dict(zip(d["GEOID"], d["pct_nonwhite"]))


def quartile(x):
    if x <= qs[0]:
        return "Q1 (lowest Non-White)"
    if x <= qs[1]:
        return "Q2"
    if x <= qs[2]:
        return "Q3"
    return "Q4 (highest Non-White)"


QORDER = ["Q1 (lowest Non-White)", "Q2", "Q3", "Q4 (highest Non-White)"]
# keep only origins in the frozen sample
R = R[R.home_cbg.isin(set(d["GEOID"]))].copy()
Rn = R[R.non_local].copy()

# ---- per-CBG concentration on non-local records ----
def wmedian(v, w):
    o = np.argsort(v); v = v[o]; w = w[o]; c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2)])


per = []
for hc, sub in Rn.groupby("home_cbg"):
    w = sub["attributed_weight"].values.astype(float)
    tot = w.sum()
    by = sub.groupby("pid")["attributed_weight"].sum().values
    shares = np.sort(by / by.sum())[::-1]
    hhi = float((shares ** 2).sum())
    dist = sub["dist_km"].values.astype(float)
    wf = sub["is_waterfront"].values
    per.append(dict(home_cbg=hc, nonlocal_weight=round(tot, 3), n_nonlocal_properties=len(shares),
                    top5_share=round(float(shares[:5].sum()), 6), top10_share=round(float(shares[:10].sum()), 6),
                    HHI=round(hhi, 6), effective_destinations=round(1.0 / hhi, 4) if hhi > 0 else np.nan,
                    wmean_distance_km=round(float((w * dist).sum() / tot), 4),
                    wmedian_distance_km=round(wmedian(dist, w), 4),
                    waterfront_share=round(float(w[wf].sum() / tot), 4)))
per = pd.DataFrame(per)

# all frozen-sample CBGs (retain zero-non-local with NA metrics)
base = d[["GEOID", "pct_nonwhite"]].rename(columns={"GEOID": "home_cbg"}).copy()
base["origin_quartile"] = base["pct_nonwhite"].map(quartile)
conc = base.merge(per, on="home_cbg", how="left")
conc["has_nonlocal"] = conc["nonlocal_weight"].notna() & (conc["nonlocal_weight"] > 0)
conc.to_csv(OUT / "destination_concentration_per_cbg.csv", index=False)

METRICS = ["top5_share", "top10_share", "effective_destinations", "HHI", "wmean_distance_km",
           "wmedian_distance_km", "n_nonlocal_properties", "waterfront_share"]


def boot_median_ci(vals, reps=1000):
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3:
        return (np.nan, np.nan)
    meds = [np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(reps)]
    return (round(float(np.percentile(meds, 2.5)), 4), round(float(np.percentile(meds, 97.5)), 4))


# ---- quartile summary (distributional; median primary) ----
qsum = []
for q in QORDER:
    sub = conc[conc.origin_quartile == q]
    n_cbg = len(sub)
    for mtr in METRICS:
        vals = sub[mtr].values.astype(float)
        obs = vals[~np.isnan(vals)]
        lo, hi = boot_median_ci(vals)
        qsum.append(dict(origin_quartile=q, metric=mtr, n_cbg=n_cbg, n_observed=len(obs),
                         n_missing=n_cbg - len(obs),
                         median=round(float(np.median(obs)), 4) if len(obs) else np.nan,
                         iqr=round(float(np.percentile(obs, 75) - np.percentile(obs, 25)), 4) if len(obs) else np.nan,
                         q25=round(float(np.percentile(obs, 25)), 4) if len(obs) else np.nan,
                         q75=round(float(np.percentile(obs, 75)), 4) if len(obs) else np.nan,
                         mean=round(float(np.mean(obs)), 4) if len(obs) else np.nan,
                         sd=round(float(np.std(obs, ddof=1)), 4) if len(obs) > 1 else np.nan,
                         median_ci_lo=lo, median_ci_hi=hi))
qsum = pd.DataFrame(qsum)
qsum.to_csv(OUT / "destination_concentration_quartile_summary.csv", index=False)

# ---- continuous Spearman sensitivity (descriptive, non-causal, unadjusted) ----
cont_rows = []
cc = conc.dropna(subset=["top10_share"])   # rows with observed concentration
for mtr in METRICS:
    x = cc["pct_nonwhite"].values; y = cc[mtr].values
    ok = ~np.isnan(y)
    rho, p = st.spearmanr(x[ok], y[ok])
    cont_rows.append(dict(metric=mtr, n=int(ok.sum()), spearman_rho=round(float(rho), 4),
                          p_unadjusted=round(float(p), 6),
                          note="descriptive; NOT causal; NOT adjusted for spatial dependence"))
pd.DataFrame(cont_rows).to_csv(OUT / "destination_concentration_continuous_sensitivity.csv", index=False)

# ---- pooled-quartile metrics (OLD estimand) for the supplementary comparison ----
pooled_rows = []
for q in QORDER:
    sub = Rn[Rn.home_cbg.map(lambda h: quartile(nw_map.get(h, -1)) == q if h in nw_map else False)]
    if not len(sub):
        continue
    by = sub.groupby("pid")["attributed_weight"].sum()
    shares = np.sort((by / by.sum()).values)[::-1]
    hhi = float((shares ** 2).sum())
    pooled_rows.append(dict(origin_quartile=q, scope="pooled_all_CBGs_in_quartile",
                            pooled_top10_share=round(float(shares[:10].sum()), 4),
                            pooled_effective_destinations=round(1.0 / hhi, 2),
                            pooled_HHI=round(hhi, 5), n_properties=len(shares)))
pooled = pd.DataFrame(pooled_rows)
# attach per-CBG medians for side-by-side
per_med = qsum[qsum.metric.isin(["top10_share", "effective_destinations"])].pivot_table(
    index="origin_quartile", columns="metric", values="median").reindex(QORDER)
pooled = pooled.merge(per_med.rename(columns={"top10_share": "perCBG_median_top10_share",
                      "effective_destinations": "perCBG_median_effective_destinations"}),
                      left_on="origin_quartile", right_index=True, how="left")
pooled.to_csv(OUT / "pooled_vs_per_cbg_destination_metrics.csv", index=False)

# ---- memo ----
def med(q, m):
    r = qsum[(qsum.origin_quartile == q) & (qsum.metric == m)].iloc[0]
    return r["median"], r["iqr"], r["median_ci_lo"], r["median_ci_hi"]


t10 = {q: med(q, "top10_share") for q in QORDER}
eff = {q: med(q, "effective_destinations") for q in QORDER}
t10_seq = [t10[q][0] for q in QORDER]; eff_seq = [eff[q][0] for q in QORDER]
mono_t10 = all(np.diff(t10_seq) <= 0) or all(np.diff(t10_seq) >= 0)
mono_eff = all(np.diff(eff_seq) <= 0) or all(np.diff(eff_seq) >= 0)
q1e, q4e = eff["Q1 (lowest Non-White)"], eff["Q4 (highest Non-White)"]
q4_more_div = q4e[0] > q1e[0]
ci_overlap_eff = not (q4e[2] > q1e[3] or q1e[2] > q4e[3])
sp_t10 = pd.DataFrame(cont_rows); sp10 = sp_t10[sp_t10.metric == "top10_share"].iloc[0]
sp_eff = sp_t10[sp_t10.metric == "effective_destinations"].iloc[0]
pooled_t10 = dict(zip(pooled.origin_quartile, pooled.pooled_top10_share))
pooled_mono = all(np.diff([pooled_t10[q] for q in QORDER]) <= 0)

L = ["# Per-CBG destination-concentration results memo (SECTION E)\n",
     "Estimand correction: the destination-concentration metrics are now computed PER ORIGIN CBG on its "
     "non-local (>2 km) records and then summarized by Non-White quartile, instead of pooling all CBGs in "
     "a quartile (pooling conflates behavior with the geographic dispersion of origins). Medians, IQRs and "
     "bootstrap CIs are primary; p-values are descriptive only.\n",
     f"Frozen RQ2 sample n={len(d)} (checksum {checksum}); Non-White cut points q25={cutpoints['q25']}, "
     f"q50={cutpoints['q50']}, q75={cutpoints['q75']}. Canonical 100 ft universe. Seed {SEED}.\n",
     "## Per-CBG top-10 non-local share (median [IQR], 95% CI of median)"]
for q in QORDER:
    m, i, lo, hi = t10[q]
    L.append(f"- {q}: {m} [IQR {i}], 95% CI [{lo}, {hi}] (n_obs "
             f"{int(qsum[(qsum.origin_quartile==q)&(qsum.metric=='top10_share')]['n_observed'].iloc[0])}).")
L.append("\n## Per-CBG effective non-local destinations (1/HHI) (median [IQR], 95% CI of median)")
for q in QORDER:
    m, i, lo, hi = eff[q]
    L.append(f"- {q}: {m} [IQR {i}], 95% CI [{lo}, {hi}].")
L += ["", "## Direct answers",
      f"1. Per-CBG top-10 share monotonic across Q1-Q4? **{'YES' if mono_t10 else 'NO'}** "
      f"(medians {', '.join(f'{v:.3f}' for v in t10_seq)}).",
      f"2. Per-CBG effective destination count monotonic across Q1-Q4? **{'YES' if mono_eff else 'NO'}** "
      f"(medians {', '.join(f'{v:.2f}' for v in eff_seq)}).",
      f"3. Are Q4 origins more diversified than Q1 after removing the pooling artifact? "
      f"Q4 median effective destinations {q4e[0]} [IQR {q4e[1]}], Q1 {q1e[0]} [IQR {q1e[1]}]; "
      f"Q4 {'>' if q4_more_div else '<='} Q1 in median, and the median 95% CIs "
      f"{'OVERLAP' if ci_overlap_eff else 'DO NOT overlap'}. "
      + ("The per-CBG evidence does NOT show Q4 as clearly more diversified once pooling is removed."
         if (not q4_more_div or ci_overlap_eff) else
         "The per-CBG evidence supports Q4 being more diversified."),
      f"4. Abstract claim ('distributed non-local visits across more park properties'): "
      + ("**UNSUPPORTED / not supported at the per-CBG level**" if (not q4_more_div or ci_overlap_eff)
         else "**SUPPORTED at the per-CBG level**")
      + f". Per-CBG effective-destination medians are Q1 {q1e[0]} vs Q4 {q4e[0]} (CI "
      f"{'overlapping' if ci_overlap_eff else 'separated'}); Spearman(Non-White, effective destinations) "
      f"= {sp_eff['spearman_rho']} (descriptive, unadjusted).",
      f"5. Pooled top-10 claim: pooled top-10 shares are "
      f"{', '.join(f'{pooled_t10[q]:.3f}' for q in QORDER)} ({'monotonically decreasing' if pooled_mono else 'not monotonic'}). "
      "This pooled pattern is still true as a POOLED DESCRIPTIVE statistic, but it partly reflects origin "
      "dispersion (see origin_quartile_coverage_and_dispersion.csv) and should be labelled pooled "
      "descriptive evidence, not per-CBG behavioral diversification, and moved out of the main "
      "inferential claim.",
      "",
      "Supplementary: `pooled_vs_per_cbg_destination_metrics.csv` (pooled vs per-CBG medians); "
      "`destination_concentration_continuous_sensitivity.csv` (Spearman, descriptive)."]
(OUT / "destination_concentration_results_memo.md").write_text("\n".join(L) + "\n")

man = json.loads((OUT / "provenance_manifest.json").read_text())
man["checksums"]["destination_concentration_per_cbg.csv"] = sha256(OUT / "destination_concentration_per_cbg.csv")
(OUT / "provenance_manifest.json").write_text(json.dumps(man, indent=2))

print("=== s3 done ===")
print("per-CBG top-10 median by quartile:", [round(v, 3) for v in t10_seq])
print("per-CBG effective-dest median by quartile:", [round(v, 2) for v in eff_seq])
print("pooled top-10 by quartile:", {q: pooled_t10[q] for q in QORDER})
print(f"mono top10={mono_t10} mono eff={mono_eff} Q4>Q1 eff={q4_more_div} CI overlap eff={ci_overlap_eff}")
