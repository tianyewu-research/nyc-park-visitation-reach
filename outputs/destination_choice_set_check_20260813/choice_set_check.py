"""Descriptive robustness check: is the higher per-CBG effective destination count in higher-Non-White
quartiles explained by those CBGs having a LARGER municipal-park CHOICE SET within feasible non-local
range? Descriptive only; changes no frozen input, model, or result.

Frozen inputs:
  - per-CBG effective_destination_count (1/HHI) and top10_share from the 0808 final destination pipeline;
  - Non-White quartiles (1,563 CBGs each), RQ2 complete-case sample (6,252 CBGs);
  - canonical municipal-park universe: NYC Parks boundaries >=0.1 ac dissolved by gispropnum (1,675 IDs);
  - property representative points and CBG centroids as in the final destination analysis (cbg_nyc.gpkg);
  - distances in EPSG:2263, feet -> km.

Choice set (primary): unique canonical properties with centroid->rep-point distance >2 km and <=10 km.
All 1,675 canonical properties are used (NOT restricted to Advan-observed parks; NOT POIs)."""
import sys, json, platform, numpy as np, pandas as pd, geopandas as gpd, scipy, scipy.stats as st
from pathlib import Path
from scipy.spatial import cKDTree
import statsmodels, statsmodels.formula.api as smf

SEED = 42
np.random.seed(SEED)
ROOT = Path(".")
OUT = ROOT / "outputs" / "destination_choice_set_check_20260813"
FROZEN = ROOT / "outputs" / "destination_per_cbg_and_match_alignment_20260808" / "destination_concentration_per_cbg.csv"
M2FT = 1 / 0.3048
KM2FT = 1000.0 * M2FT
QO = ["Q1 (lowest Non-White)", "Q2", "Q3", "Q4 (highest Non-White)"]
QSHORT = dict(zip(QO, ["Q1", "Q2", "Q3", "Q4"]))
TARGET_EFF = {"Q1 (lowest Non-White)": 11.70, "Q2": 14.06, "Q3": 16.11, "Q4 (highest Non-White)": 17.35}

val = []   # validation log lines


def log(s):
    val.append(s); print(s)


# ---- frozen per-CBG concentration (effective count, top10) + quartiles ----
conc = pd.read_csv(FROZEN, dtype={"home_cbg": str})
conc = conc[["home_cbg", "pct_nonwhite", "origin_quartile", "effective_destinations", "top10_share"]].copy()
conc = conc.rename(columns={"effective_destinations": "effective_destination_count"})
assert conc["home_cbg"].nunique() == 6252, "expected 6,252 unique CBGs"
qc = conc["origin_quartile"].value_counts().reindex(QO)
assert (qc == 1563).all(), f"expected 1,563 per quartile, got {qc.to_dict()}"
frozen_med = conc.groupby("origin_quartile")["effective_destination_count"].median().reindex(QO)
repro_ok = all(abs(frozen_med[q] - TARGET_EFF[q]) < 0.01 for q in QO)
log(f"[repro] frozen effective medians {frozen_med.round(2).tolist()} vs target "
    f"{[TARGET_EFF[q] for q in QO]} -> {'OK' if repro_ok else 'MISMATCH'}")
assert repro_ok, "frozen effective-count medians did not reproduce; STOP"

# ---- canonical municipal-park universe: dissolve by gispropnum, representative points ----
parks = gpd.read_file(ROOT / "data" / "raw" / "NYC公园边界" / "Parks Properties_20260623").to_crs("EPSG:2263")
n_rows_all = len(parks)
parks = parks[parks["acres"] >= 0.1].copy()
n_rows_acre = len(parks)
parks["pid"] = parks["gispropnum"].astype(str)
prop = parks.dissolve(by="pid").reset_index()
n_pid = prop["pid"].nunique()
log(f"[universe] Parks rows all={n_rows_all}, >=0.1ac={n_rows_acre}; unique gispropnum={n_pid}")
assert n_pid == 1675, f"expected 1,675 canonical property IDs, got {n_pid}"
rep = prop.geometry.representative_point()
prop_xy = np.column_stack([rep.x.values, rep.y.values])

# ---- CBG centroids (same geometry as the final destination analysis: cbg_nyc.gpkg) ----
cbg = gpd.read_file(ROOT / "data" / "interim" / "cbg_nyc.gpkg").to_crs("EPSG:2263")
cbg["GEOID"] = cbg["GEOID"].astype(str)
cent = cbg.set_index("GEOID").geometry.centroid
conc["cx"] = conc["home_cbg"].map(cent.x)
conc["cy"] = conc["home_cbg"].map(cent.y)
assert conc[["cx", "cy"]].notna().all().all(), "missing CBG centroid for some frozen CBG"
cbg_xy = conc[["cx", "cy"]].values

# ---- choice-set counts via KDTree (count within R minus within 2 km) ----
tree = cKDTree(prop_xy)
n_pairs = len(conc) * n_pid
R = {r: r * KM2FT for r in (2, 6, 10, 15)}
cnt = {r: np.array(tree.query_ball_point(cbg_xy, R[r], return_length=True)) for r in (2, 6, 10, 15)}
conc["choice_count_2_10km"] = cnt[10] - cnt[2]
conc["choice_count_2_6km"] = cnt[6] - cnt[2]
conc["choice_count_2_15km"] = cnt[15] - cnt[2]
assert (conc[["choice_count_2_10km", "choice_count_2_6km", "choice_count_2_15km"]] >= 0).all().all(), \
    "negative choice count encountered"
conc["normalized_effective_count"] = np.where(conc["choice_count_2_10km"] > 0,
                                              conc["effective_destination_count"] / conc["choice_count_2_10km"],
                                              np.nan)
zero_choice = int((conc["choice_count_2_10km"] == 0).sum())
log(f"[choice] CBG-property distance pairs = {n_pairs:,}; CBGs with zero 2-10 km choice set = {zero_choice}")

cbg_level = conc[["home_cbg", "origin_quartile", "pct_nonwhite", "choice_count_2_10km",
                  "choice_count_2_6km", "choice_count_2_15km", "effective_destination_count",
                  "top10_share", "normalized_effective_count"]].copy()
cbg_level.to_csv(OUT / "choice_set_cbg_level.csv", index=False)

# ---- quartile summary (n, median, IQR, bootstrap 95% CI of median, seed 42) ----
rng = np.random.default_rng(SEED)


def boot_median_ci(v, reps=1000):
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return (np.nan, np.nan)
    meds = [np.median(rng.choice(v, size=len(v), replace=True)) for _ in range(reps)]
    return (round(float(np.percentile(meds, 2.5)), 4), round(float(np.percentile(meds, 97.5)), 4))


METRICS = ["choice_count_2_10km", "effective_destination_count", "top10_share",
           "normalized_effective_count"]
rows = []
for q in QO:
    sub = conc[conc.origin_quartile == q]
    for m in METRICS:
        v = sub[m].values.astype(float); obs = v[~np.isnan(v)]
        lo, hi = boot_median_ci(v)
        rows.append(dict(origin_quartile=QSHORT[q], metric=m, n=len(sub), n_observed=len(obs),
                         median=round(float(np.median(obs)), 4),
                         iqr=round(float(np.percentile(obs, 75) - np.percentile(obs, 25)), 4),
                         q25=round(float(np.percentile(obs, 25)), 4), q75=round(float(np.percentile(obs, 75)), 4),
                         median_ci_lo=lo, median_ci_hi=hi))
qsum = pd.DataFrame(rows)
qsum.to_csv(OUT / "choice_set_quartile_summary.csv", index=False)

# ---- descriptive associations ----
sp_eff = st.spearmanr(conc["choice_count_2_10km"], conc["effective_destination_count"])
sp_t10 = st.spearmanr(conc["choice_count_2_10km"], conc["top10_share"])
kw = st.kruskal(*[conc.loc[conc.origin_quartile == q, "choice_count_2_10km"].values for q in QO])
assoc = pd.DataFrame([
    dict(test="Spearman(choice_count_2_10km, effective_destination_count)", statistic=round(float(sp_eff.correlation), 4),
         p_value=float(sp_eff.pvalue), label="descriptive, not causal"),
    dict(test="Spearman(choice_count_2_10km, top10_share)", statistic=round(float(sp_t10.correlation), 4),
         p_value=float(sp_t10.pvalue), label="descriptive, not causal"),
    dict(test="Kruskal-Wallis(choice_count_2_10km across Q1-Q4)", statistic=round(float(kw.statistic), 3),
         p_value=float(kw.pvalue), label="descriptive, not causal"),
])
assoc.to_csv(OUT / "choice_set_associations.csv", index=False)

# ---- diagnostic regression (attenuation), HC3 ----
reg = conc.copy()
reg["Q2"] = (reg.origin_quartile == "Q2").astype(int)
reg["Q3"] = (reg.origin_quartile == "Q3").astype(int)
reg["Q4"] = (reg.origin_quartile == "Q4 (highest Non-White)").astype(int)
reg["log_eff"] = np.log(reg["effective_destination_count"])
reg["log1p_choice"] = np.log1p(reg["choice_count_2_10km"])
# baseline (no choice-set) for attenuation comparison
m_base = smf.ols("log_eff ~ Q2 + Q3 + Q4", data=reg).fit(cov_type="HC3")
m_adj = smf.ols("log_eff ~ Q2 + Q3 + Q4 + log1p_choice", data=reg).fit(cov_type="HC3")
mrows = []
for label, m in [("baseline (quartile only)", m_base), ("adjusted (+log1p choice set)", m_adj)]:
    ci = m.conf_int()
    for term in m.params.index:
        mrows.append(dict(model=label, term=term, coef=round(float(m.params[term]), 5),
                          ci_lo=round(float(ci.loc[term, 0]), 5), ci_hi=round(float(ci.loc[term, 1]), 5),
                          p_value=float(m.pvalues[term]), se_type="HC3"))
pd.DataFrame(mrows).to_csv(OUT / "choice_set_diagnostic_model.csv", index=False)

# ---- range sensitivity (2-6 km and 2-15 km): quartile medians + Q4-Q1 contrast ----
rng_rows = []
for col, lab in [("choice_count_2_6km", "2-6 km"), ("choice_count_2_10km", "2-10 km"),
                 ("choice_count_2_15km", "2-15 km")]:
    meds = {QSHORT[q]: float(conc.loc[conc.origin_quartile == q, col].median()) for q in QO}
    # normalized effective under this range
    norm = np.where(conc[col] > 0, conc["effective_destination_count"] / conc[col], np.nan)
    conc["_n"] = norm
    nmeds = {QSHORT[q]: float(np.nanmedian(conc.loc[conc.origin_quartile == q, "_n"])) for q in QO}
    rng_rows.append(dict(range=lab, choice_median_Q1=round(meds["Q1"], 2), choice_median_Q2=round(meds["Q2"], 2),
                         choice_median_Q3=round(meds["Q3"], 2), choice_median_Q4=round(meds["Q4"], 2),
                         choice_Q4_minus_Q1=round(meds["Q4"] - meds["Q1"], 2),
                         norm_eff_median_Q1=round(nmeds["Q1"], 5), norm_eff_median_Q4=round(nmeds["Q4"], 5),
                         norm_eff_Q4_minus_Q1=round(nmeds["Q4"] - nmeds["Q1"], 5)))
pd.DataFrame(rng_rows).to_csv(OUT / "choice_set_range_sensitivity.csv", index=False)
conc.drop(columns=["_n"], inplace=True)

# ---- validation file ----
log(f"[assert] unique CBGs=6252 ({conc['home_cbg'].nunique()==6252}); 1563/quartile ({(qc==1563).all()})")
log(f"[assert] canonical property IDs before distance expansion = {n_pid} (==1675: {n_pid==1675})")
log(f"[assert] no negative counts: "
    f"{(conc[['choice_count_2_10km','choice_count_2_6km','choice_count_2_15km']]>=0).all().all()}")
log(f"[report] CBGs with zero 2-10 km choice set: {zero_choice}")
log(f"[env] CRS=EPSG:2263 (US survey feet); 1 km = {KM2FT:.4f} ft; pairs={n_pairs:,}")
log(f"[env] python={sys.version.split()[0]} numpy={np.__version__} pandas={pd.__version__} "
    f"geopandas={gpd.__version__} scipy={scipy.__version__} statsmodels={statsmodels.__version__}")
(OUT / "choice_set_validation.txt").write_text("\n".join(val) + "\n")

# ---- decision logic + interpretation ----
def med(metric):
    return {QSHORT[q]: float(qsum[(qsum.origin_quartile == QSHORT[q]) & (qsum.metric == metric)]["median"].iloc[0]) for q in QO}


choice_med = med("choice_count_2_10km"); norm_med = med("normalized_effective_count")
eff_med = med("effective_destination_count")
choice_q4_q1 = choice_med["Q4"] - choice_med["Q1"]
choice_ratio = choice_med["Q4"] / choice_med["Q1"] if choice_med["Q1"] else np.nan
norm_seq = [norm_med[q] for q in ["Q1", "Q2", "Q3", "Q4"]]
norm_ordered_up = all(np.diff(norm_seq) > 0)
norm_ordered_down = all(np.diff(norm_seq) < 0)
norm_gradient_present = norm_ordered_up or norm_ordered_down
# attenuation of Q4 coef when choice added
q4_base = float(m_base.params["Q4"]); q4_adj = float(m_adj.params["Q4"])
q4_adj_sig = float(m_adj.pvalues["Q4"]) < 0.05
attenuation = (q4_base - q4_adj) / q4_base if q4_base != 0 else np.nan
choice_similar = abs(choice_ratio - 1) < 0.15  # within ~15%
q4_not_larger = choice_med["Q4"] <= choice_med["Q1"]

if (choice_similar or q4_not_larger) and norm_gradient_present and abs(attenuation) < 0.10:
    _dir = ("is not larger in Q4 (it is slightly smaller: "
            f"median {choice_med['Q4']:.0f} vs {choice_med['Q1']:.0f} in Q1)" if q4_not_larger
            else "is similar across quartiles")
    verdict = (f"STRENGTHENED: the municipal-park choice set {_dir}, yet the normalized effective-count "
               "gradient persists and even steepens with quartile, and the quartile gap in log effective "
               "count is not attenuated by the choice set. Higher-Non-White origins reach more effective "
               "destinations despite no larger opportunity set, so the destination-diversity "
               "interpretation is strengthened, not explained away by choice-set size.")
elif (not choice_similar) and norm_gradient_present:
    verdict = ("PARTLY EXPLAINED: Q4 has a materially larger choice set and the normalized gradient "
               "attenuates but remains ordered, so destination diversity is partly explained by the "
               "opportunity structure.")
else:
    verdict = ("REDUCES TO CHOICE-SET SIZE: the normalized gradient disappears or reverses, so the raw "
               "pattern reflects choice-set size and the abstract-level destination-diversity claim "
               "should be removed.")

L = ["# Choice-set robustness for destination diversity — interpretation\n",
     "Descriptive check only. It does not change the manuscript framework, primary models, LPUR "
     "construction, quartiles, or the canonical crosswalk, and it identifies no mechanism: this analysis "
     "cannot identify preference, constraint, behavioral effect, or causal relationships.\n",
     "## Choice set (>2 km, <=10 km; all 1,675 canonical municipal park properties)"]
for q in ["Q1", "Q2", "Q3", "Q4"]:
    r = qsum[(qsum.origin_quartile == q) & (qsum.metric == "choice_count_2_10km")].iloc[0]
    L.append(f"- {q}: median {r['median']:.0f} [IQR {r['iqr']:.0f}], 95% CI of median "
             f"[{r['median_ci_lo']:.1f}, {r['median_ci_hi']:.1f}].")
L.append(f"- Q4 - Q1 median choice set = {choice_q4_q1:.0f} (Q4/Q1 = {choice_ratio:.2f}).")
L.append("\n## Effective destinations and NORMALIZED effective count (effective / choice set)")
for q in ["Q1", "Q2", "Q3", "Q4"]:
    L.append(f"- {q}: effective median {eff_med[q]:.2f}; normalized median {norm_med[q]:.4f}.")
L.append(f"- Normalized gradient Q1->Q4: {'monotonic' if norm_gradient_present else 'NOT monotonic'} "
         f"({', '.join(f'{v:.4f}' for v in norm_seq)}).")
L += ["", "## Descriptive associations (not causal)",
      f"- Spearman(choice set, effective destinations) = {sp_eff.correlation:.3f} (p={sp_eff.pvalue:.2g}).",
      f"- Spearman(choice set, top-10 share) = {sp_t10.correlation:.3f} (p={sp_t10.pvalue:.2g}).",
      f"- Kruskal-Wallis choice set across Q1-Q4 = {kw.statistic:.1f} (p={kw.pvalue:.2g}).",
      "", "## Diagnostic attenuation (HC3; residual spatial dependence NOT modeled)",
      f"- log(effective) ~ quartile: Q4 coef {q4_base:+.4f}. Adding log1p(choice set): Q4 coef "
      f"{q4_adj:+.4f} (p={m_adj.pvalues['Q4']:.2g}), log1p(choice) coef {m_adj.params['log1p_choice']:+.4f}. "
      f"Q4 attenuation {100*attenuation:.0f}% ({'remains' if q4_adj_sig else 'no longer'} significant). "
      "This is a descriptive adjustment only, not a causal or spatially-adjusted model.",
      "", "## Verdict", verdict,
      "", "## Range checks (quartile medians; Q4-Q1 contrast)"]
rs = pd.read_csv(OUT / "choice_set_range_sensitivity.csv")
for _, r in rs.iterrows():
    L.append(f"- {r['range']}: choice medians Q1={r['choice_median_Q1']:.0f}, Q4={r['choice_median_Q4']:.0f} "
             f"(Q4-Q1={r['choice_Q4_minus_Q1']:.0f}); normalized effective Q1={r['norm_eff_median_Q1']:.4f}, "
             f"Q4={r['norm_eff_median_Q4']:.4f} (Q4-Q1={r['norm_eff_Q4_minus_Q1']:+.4f}).")

# paste-ready paragraphs
methods = (f"**Methods (paste-ready).** To test whether the higher per-CBG effective destination count in "
           f"higher-Non-White origin quartiles reflects a larger set of available parks rather than more "
           f"dispersed use, we computed, for each of the 6,252 RQ2 block groups, the number of canonical "
           f"municipal park properties (all 1,675 dissolved NYC Parks records >=0.1 acre) whose "
           f"representative-point distance from the block-group centroid was greater than 2 km and no more "
           f"than 10 km (EPSG:2263; feet converted to kilometres). We summarised this choice set and a "
           f"normalised effective count (effective destinations divided by choice set) by quartile with "
           f"medians, IQRs and 1,000-resample block-group bootstrap intervals (seed 42), reported Spearman "
           f"correlations and a Kruskal-Wallis test (descriptive), and fitted a diagnostic HC3 regression "
           f"of log effective count on quartile indicators with and without log1p(choice set). Residual "
           f"spatial dependence was not modelled; the analysis is descriptive and identifies no mechanism.")
_dirtxt = ("was not larger in higher-Non-White quartiles (if anything slightly smaller)"
           if q4_not_larger else
           ("was similar across quartiles" if choice_similar else "was larger in higher-Non-White quartiles"))
results = (f"**Results/Discussion (paste-ready).** The 2-10 km municipal-park choice set "
           f"{_dirtxt} (median {choice_med['Q1']:.0f} in Q1 vs {choice_med['Q4']:.0f} in Q4; Kruskal-Wallis "
           f"p={kw.pvalue:.2g}), and across block groups it was "
           f"{'positively' if sp_eff.correlation>0 else 'essentially un'}associated with the effective "
           f"destination count (Spearman {sp_eff.correlation:.2f}). "
           + ("After normalising the effective count by the choice set, the quartile ordering "
              + ("persisted and steepened" if norm_gradient_present else "did not persist")
              + f" (normalised medians {', '.join(f'{norm_med[q]:.3f}' for q in ['Q1','Q2','Q3','Q4'])}), and "
              f"the quartile gap in log effective count "
              + (f"attenuated by {100*attenuation:.0f}% but remained ordered"
                 if (norm_gradient_present and not (choice_similar or q4_not_larger)) else
                 ("was essentially unchanged" if (choice_similar or q4_not_larger) else "was absorbed"))
              + f" when the choice set was added (Q4 {q4_base:+.3f} to {q4_adj:+.3f}). ")
           + (lambda s: s[0].upper() + s[1:])(verdict.split(":", 1)[1].strip())
           + " These patterns are descriptive and cannot identify preference, constraint, or any "
           "behavioural or causal effect.")
L += ["", methods, "", results]
(OUT / "choice_set_interpretation.md").write_text("\n".join(L) + "\n")

# optional figure only if the normalized metric changes the substantive interpretation
make_fig = ("REDUCES TO CHOICE-SET SIZE" in verdict) or ("PARTLY EXPLAINED" in verdict)
fig_note = "figure created (normalized metric alters interpretation)" if make_fig else \
    "no figure (normalized metric does not change the substantive interpretation)"
if make_fig:
    import matplotlib.pyplot as plt
    CB = dict(blue="#0072B2", vermillion="#D55E00", skyblue="#56B4E9", blueedge="#0072B2")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, col, ylab, title in [(axes[0], "choice_count_2_10km", "Choice set (2-10 km, # properties)",
                                  "(a) Municipal-park choice set"),
                                 (axes[1], "normalized_effective_count", "Effective count / choice set",
                                  "(b) Normalized effective destinations")]:
        data = [conc.loc[conc.origin_quartile == q, col].dropna().values for q in QO]
        bp = ax.boxplot(data, positions=range(4), widths=0.6, showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", lw=1.6))
        for patch in bp["boxes"]:
            patch.set(facecolor=CB["skyblue"], alpha=0.6, edgecolor=CB["blue"])
        ax.set_xticks(range(4)); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel("Origin Non-White quartile", fontsize=9); ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
    fig.suptitle("Destination diversity vs municipal-park choice set (descriptive)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"choice_set_by_quartile.{ext}", dpi=600, bbox_inches="tight")
    plt.close()

log(f"[figure] {fig_note}")
(OUT / "choice_set_validation.txt").write_text("\n".join(val) + "\n")

print("\n=== choice-set check done ===")
print("verdict:", verdict.split(":", 1)[0])
print("choice medians:", {q: round(choice_med[q]) for q in ['Q1', 'Q2', 'Q3', 'Q4']})
print("normalized medians:", {q: round(norm_med[q], 4) for q in ['Q1', 'Q2', 'Q3', 'Q4']})
print("Q4 log-eff coef base->adj:", round(q4_base, 4), "->", round(q4_adj, 4), f"({100*attenuation:.0f}% atten)")
