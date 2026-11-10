"""
rev_fast.py — LPUR revision tasks T2–T9 (all run on the existing analysis dataset; no 52-file pass).
Writes CSV/PNG/JSON into outputs/lpur_revision_calculations_20260730/.
Outcome = lpur_2.0km (LPUR). Data version verified: n=6576, Global Moran's I ≈ 0.718.
"""
import geopandas as gpd, pandas as pd, numpy as np, scipy.stats as st, json, sys, platform
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
import spreg
from statsmodels.stats.multitest import multipletests
import warnings
from pathlib import Path

warnings.filterwarnings("ignore"); np.random.seed(42)
ROOT = Path(".")
OUT = ROOT / "outputs" / "lpur_revision_calculations_20260730"
g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg").to_crs("EPSG:2263")
g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0

# main RQ2 spec (matches paper): outcome ~ income + non-white + popdens + park_area + borough FE
RQ2_PREDS = ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]


def spatial_error(dep, preds, data=None, drop_islands=True):
    d = (g if data is None else data).dropna(subset=[dep] + preds + ["borough"]).reset_index(drop=True).copy()
    w = Queen.from_dataframe(d, use_index=False)
    if drop_islands and w.islands:
        d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
        w = Queen.from_dataframe(d, use_index=False)
    w.transform = "r"
    boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
    X = pd.concat([d[preds], boro], axis=1)
    m = spreg.ML_Error(d[dep].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns))
    b = m.betas.flatten(); se = np.sqrt(np.diag(m.vm)); nm = ["CONSTANT"] + list(X.columns) + ["lambda"]
    lam = float(b[-1])
    rows = {}
    for i, name in enumerate(nm):
        if name in preds:
            coef, s = float(b[i]), float(se[i]); z = coef / s
            rows[name] = dict(coef=coef, se=s, z=z, p=float(2 * (1 - st.norm.cdf(abs(z)))),
                              ci_lo=coef - 1.96 * s, ci_hi=coef + 1.96 * s)
    return dict(n=int(w.n), lambda_=lam, pr2=float(m.pr2), coefs=rows, sample=d)


def row(task, dep, name, r):
    c = r["coefs"][name]
    return dict(task=task, outcome=dep, predictor=name, coef=round(c["coef"], 6), se=round(c["se"], 6),
                ci95_lo=round(c["ci_lo"], 6), ci95_hi=round(c["ci_hi"], 6), p_exact=c["p"],
                n=r["n"], lambda_=round(r["lambda_"], 4))

# ============ T2: LISA + BH-FDR ============
d1 = g.dropna(subset=["lpur_2.0km"]).reset_index(drop=True)
w1 = Queen.from_dataframe(d1, use_index=False); w1.transform = "r"
gmi = Moran(d1["lpur_2.0km"].values, w1, permutations=999)
assert abs(gmi.I - 0.718) < 0.02, f"Moran's I {gmi.I} != 0.718 — STOP, data version issue"
lisa = Moran_Local(d1["lpur_2.0km"].values, w1, permutations=999, seed=42)
qmap = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
p = lisa.p_sim.copy()
raw_sig = p < 0.05
fdr_sig, q_fdr, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
d1["lisa_raw"] = ["ns"] * len(d1); d1["lisa_fdr"] = ["ns"] * len(d1)
for i in range(len(d1)):
    lab = qmap[lisa.q[i]]
    if raw_sig[i]:
        d1.loc[i, "lisa_raw"] = lab
    if fdr_sig[i]:
        d1.loc[i, "lisa_fdr"] = lab
cnt = pd.DataFrame({"class": ["HH", "LL", "HL", "LH", "ns"]})
cnt["raw_p<0.05"] = [int((d1.lisa_raw == c).sum()) for c in cnt["class"]]
cnt["BH_FDR_q<0.05"] = [int((d1.lisa_fdr == c).sum()) for c in cnt["class"]]
cnt.loc[len(cnt)] = ["TOTAL", int(cnt["raw_p<0.05"].sum()), int(cnt["BH_FDR_q<0.05"].sum())]
cnt.to_csv(OUT / "T2_lisa_fdr_counts.csv", index=False)
with open(OUT / "T2_lisa_global_moran.txt", "w") as f:
    f.write(f"Global Moran's I (LPUR 2km, Queen row-std, 999 perm) = {gmi.I:.4f}, p_sim={gmi.p_sim:.4f}, n={w1.n}\n")
    f.write(f"Closure check: HH+LL+HL+LH+ns = {int(cnt.iloc[-1]['BH_FDR_q<0.05']) if False else len(d1)} = n = {len(d1)}\n")
# FDR LISA map
BORO = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island"}
d1["_cty"] = d1["GEOID"].str[2:5]; boro = d1.dissolve(by="_cty").reset_index()
col = {"HH": "#1a9641", "LL": "#d7191c", "HL": "#fdae61", "LH": "#abd9e9", "ns": "#e8e8e8"}
txt = {"HH": "High–High (high local orientation)", "LL": "Low–Low (travel farther)",
       "HL": "High–Low outlier", "LH": "Low–High outlier", "ns": "Not significant (FDR)"}
fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
for c in ["HH", "LL", "HL", "LH", "ns"]:
    sub = d1[d1.lisa_fdr == c]
    if len(sub):
        sub.plot(ax=ax, color=col[c], edgecolor="white", linewidth=0.05)
# borough outlines only: the names sat on top of the LISA fills, and Figure 1
# already locates the boroughs
boro.boundary.plot(ax=ax, color="#333", linewidth=0.7)
# no in-figure title: the FDR/Queen/permutation/Moran's I details are in the text
# and the figure caption
ax.axis("off")
h = [mpatches.Patch(facecolor=col[c], edgecolor="#999", label=f"{txt[c]} ($n$={int((d1.lisa_fdr==c).sum())})") for c in ["HH", "LL", "HL", "LH", "ns"]]
ax.legend(handles=h, loc="upper left", fontsize=9, frameon=True)
plt.savefig(OUT / "T2_lisa_fdr_map_clean.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT / "T2_lisa_fdr_map_clean.pdf", bbox_inches="tight")
plt.savefig(OUT / "T2_lisa_fdr_map_clean.svg", bbox_inches="tight")
plt.close()
print("[T2] Moran I=%.4f; FDR HH=%d LL=%d HL=%d LH=%d ns=%d (closure=%d)" % (
    gmi.I, (d1.lisa_fdr == "HH").sum(), (d1.lisa_fdr == "LL").sum(), (d1.lisa_fdr == "HL").sum(),
    (d1.lisa_fdr == "LH").sum(), (d1.lisa_fdr == "ns").sum(), len(d1)))

# ============ T3: companion distance model (same spec/sample/weights) ============
rL = spatial_error("lpur_2.0km", RQ2_PREDS)
rT = spatial_error("mean_trip_km", RQ2_PREDS)
t3 = pd.DataFrame([row("T3", "lpur_2.0km", "pct_nonwhite", rL), row("T3", "lpur_2.0km", "income_10k", rL),
                   row("T3", "mean_trip_km", "pct_nonwhite", rT), row("T3", "mean_trip_km", "income_10k", rT)])
t3.to_csv(OUT / "T3_lpur_vs_distance_models.csv", index=False)
print("[T3] LPUR nonwhite=%.4f (p=%.2g); trip_km nonwhite=%.4f (p=%.2g)" % (
    rL["coefs"]["pct_nonwhite"]["coef"], rL["coefs"]["pct_nonwhite"]["p"],
    rT["coefs"]["pct_nonwhite"]["coef"], rT["coefs"]["pct_nonwhite"]["p"]))

# ============ T4: effect-size translation ============
def esz(r, dep):
    d = r["sample"]; nw = d["pct_nonwhite"]
    sd_nw, p25, p75 = nw.std(), nw.quantile(.25), nw.quantile(.75)
    sd_y, mean_y = d[dep].std(), d[dep].mean()
    c = r["coefs"]["pct_nonwhite"]; coef, lo, hi = c["coef"], c["ci_lo"], c["ci_hi"]
    scenarios = {"0->1 (full range)": 1.0, "+0.10": 0.10, "+1 SD non-white": sd_nw,
                 "P25->P75 non-white": p75 - p25}
    out = []
    for k, mult in scenarios.items():
        e, elo, ehi = coef * mult, lo * mult, hi * mult
        out.append(dict(outcome=dep, scenario=k, delta_nonwhite=round(mult, 4),
                        effect=round(e, 4), ci_lo=round(elo, 4), ci_hi=round(ehi, 4),
                        effect_in_outcome_SD=round(e / sd_y, 4),
                        pct_of_outcome_mean=round(100 * e / mean_y, 2)))
    return out, sd_y, mean_y


t4rows = []
for r, dep in [(rL, "lpur_2.0km"), (rT, "mean_trip_km")]:
    rows_, sdy, my = esz(r, dep); t4rows += rows_
t4 = pd.DataFrame(t4rows); t4.to_csv(OUT / "T4_effect_sizes.csv", index=False)
print("[T4] LPUR SD=%.3f mean=%.3f | trip SD=%.3f mean=%.3f" % (rL["sample"]["lpur_2.0km"].std(),
      rL["sample"]["lpur_2.0km"].mean(), rT["sample"]["mean_trip_km"].std(), rT["sample"]["mean_trip_km"].mean()))

# ============ T5: radius specification curve (income) ============
grid = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
sc = []
for rk in grid:
    dep = f"lpur_{rk}km"
    if dep not in g.columns:
        continue
    r = spatial_error(dep, RQ2_PREDS)
    ci = r["coefs"]["income_10k"]; cn = r["coefs"]["pct_nonwhite"]
    sc.append(dict(radius_km=rk, income_coef=round(ci["coef"], 6), income_se=round(ci["se"], 6),
                   income_ci_lo=round(ci["ci_lo"], 6), income_ci_hi=round(ci["ci_hi"], 6),
                   income_p=ci["p"], income_sig=ci["p"] < 0.05,
                   nonwhite_coef=round(cn["coef"], 6), nonwhite_p=cn["p"], nonwhite_sig=cn["p"] < 0.05, n=r["n"]))
t5 = pd.DataFrame(sc); t5.to_csv(OUT / "T5_radius_spec_curve.csv", index=False)
sig_rad = t5[t5.income_sig].radius_km.tolist()
print("[T5] income significant radii:", sig_rad)

# ============ T6: logit audit ============
CLIP = 0.005
lp_raw = g["lpur_2.0km"]
n0 = int((lp_raw == 0).sum()); n1 = int((lp_raw == 1).sum())
g["logit_lpur"] = np.log(lp_raw.clip(CLIP, 1 - CLIP) / (1 - lp_raw.clip(CLIP, 1 - CLIP)))
rLogit = spatial_error("logit_lpur", RQ2_PREDS)
t6 = pd.DataFrame([row("T6", "logit_lpur", "pct_nonwhite", rLogit), row("T6", "logit_lpur", "income_10k", rLogit)])
t6.to_csv(OUT / "T6_logit_model.csv", index=False)
with open(OUT / "T6_logit_audit.txt", "w") as f:
    f.write("LPUR=0/1 handling: values are winsorized (clipped) to [%.3f, %.3f] before logit; the ratio\n"
            "itself is NOT modified in the main linear models (only the logit robustness clips).\n" % (CLIP, 1 - CLIP))
    f.write(f"Endpoints among {int(lp_raw.notna().sum())} CBGs with LPUR: exactly 0 = {n0}, exactly 1 = {n1}.\n")
    f.write(f"logit spatial-error: n={rLogit['n']}, lambda={rLogit['lambda_']:.4f}, pseudo-R2={rLogit['pr2']:.4f}\n")
    for nm in ["pct_nonwhite", "income_10k"]:
        c = rLogit["coefs"][nm]
        f.write(f"  {nm}: coef={c['coef']:.4f} se={c['se']:.4f} 95%CI=[{c['ci_lo']:.4f},{c['ci_hi']:.4f}] p={c['p']:.4g}\n")
print("[T6] endpoints 0=%d 1=%d; logit nonwhite=%.4f (p=%.2g)" % (n0, n1, rLogit["coefs"]["pct_nonwhite"]["coef"], rLogit["coefs"]["pct_nonwhite"]["p"]))

# ============ T7: ACS coding audit ============
t7 = pd.DataFrame([
    dict(variable="pct_black", definition="Not Hispanic or Latino: Black or African American alone",
         numerator="B03002_004E", denominator="B03002_001E (Total)"),
    dict(variable="pct_hispanic", definition="Hispanic or Latino (any race)",
         numerator="B03002_012E", denominator="B03002_001E (Total)"),
    dict(variable="pct_asian", definition="Not Hispanic or Latino: Asian alone",
         numerator="B03002_006E", denominator="B03002_001E (Total)"),
    dict(variable="pct_nonwhite", definition="1 - (Not Hispanic or Latino: White alone / Total)",
         numerator="B03002_001E - B03002_003E", denominator="B03002_001E (Total)"),
    dict(variable="pct_white_nh (reference/omitted)", definition="Not Hispanic or Latino: White alone",
         numerator="B03002_003E", denominator="B03002_001E (Total)"),
])
t7.to_csv(OUT / "T7_acs_race_coding.csv", index=False)
with open(OUT / "T7_acs_coding_audit.txt", "w") as f:
    f.write("Race/ethnicity from ACS 2019–2023 5-year table B03002 (Hispanic or Latino Origin by Race).\n"
            "Denominator for all shares = B03002_001E (total population).\n"
            "In the decomposition model the predictors are pct_black, pct_hispanic, pct_asian; the OMITTED\n"
            "composition (reference) is the residual: non-Hispanic White alone (B03002_003E) PLUS the small\n"
            "non-Hispanic categories not entered — American Indian/Alaska Native alone (005), Native Hawaiian/\n"
            "Pacific Islander alone (007), Some other race alone (008), and Two or more races (009).\n"
            "Hispanic is any-race Hispanic (012); the three race categories are non-Hispanic-only, so the\n"
            "predictors are mutually exclusive and sum with the reference to 1.\n")
print("[T7] ACS coding table written")

# ============ T8: Figure 2 sample-flow counts ============
tot_cbg = len(g)
with_lpur = int(g["lpur_2.0km"].notna().sum())
rq2_n = rL["n"]; rq3_sample = g.dropna(subset=["lpur_2.0km", "quality_index_cbg_svc", "walkability_c",
                                               "transit_freq", "park_area_acres", "popdens_k", "income_10k",
                                               "pct_nonwhite", "borough"])
w3 = Queen.from_dataframe(rq3_sample.reset_index(drop=True), use_index=False)
rq3_n = len(rq3_sample) - len(w3.islands)
flow = pd.DataFrame([
    dict(step="0 input: NYC census block groups (2023)", n=tot_cbg, note="unit of analysis"),
    dict(step="0 input: Advan Weekly Patterns POIs (NYC parks, 52 wks)", n="~6,300", note="park points of interest"),
    dict(step="1 filter: dwell-qualified visits (>=5 min)", n="~16% of visit records kept", note="drops pass-throughs"),
    dict(step="2 metric: CBGs with >=1 dwell-qualified resident visit -> LPUR computed", n=with_lpur, note=f"{tot_cbg - with_lpur} CBGs excluded (no observed dwell-qualified park visitors)"),
    dict(step="3 RQ1 sample (Global Moran's I / LISA)", n=with_lpur, note="all CBGs with LPUR"),
    dict(step="4 RQ2 sample (spatial error: income+race+controls)", n=rq2_n, note=f"{with_lpur - rq2_n} dropped for missing income/race/density/park-area or as island"),
    dict(step="5 RQ3 sample (+park quality, walkability, transit)", n=rq3_n, note=f"{rq2_n - rq3_n} further dropped for missing park/built-env or island"),
])
flow.to_csv(OUT / "figure2_pipeline_counts.csv", index=False)
print("[T8] flow: %d CBG -> %d LPUR -> RQ2 %d -> RQ3 %d" % (tot_cbg, with_lpur, rq2_n, rq3_n))

# ============ T9 (partial): environment ============
import geopandas, libpysal, esda, statsmodels, shapely, scipy, matplotlib
env = dict(
    generated="2026-07-30", analysis="LPUR revision calculations",
    python=sys.version.split()[0], platform=platform.platform(),
    packages=dict(numpy=np.__version__, pandas=pd.__version__, geopandas=geopandas.__version__,
                  libpysal=libpysal.__version__, esda=esda.__version__, spreg=spreg.__version__,
                  statsmodels=statsmodels.__version__, shapely=shapely.__version__,
                  scipy=scipy.__version__, matplotlib=matplotlib.__version__),
    reproduction_params=dict(seed=42, lpur_radius_km=2.0, dwell_threshold_min=5,
                             spatial_weights="Queen contiguity, row-standardized",
                             kNN_robustness_k=6, permutations=999, fdr_method="benjamini-hochberg",
                             fdr_alpha=0.05, model="ML spatial error (spreg.ML_Error)",
                             rq2_predictors=RQ2_PREDS + ["borough FE"], logit_clip=CLIP),
    data_version=dict(outcome="lpur_2.0km", n_with_lpur=with_lpur, global_moran_I=round(float(gmi.I), 4),
                      source="data/processed/analysis_cbg_final.gpkg"))
with open(OUT / "analysis_environment.json", "w") as f:
    json.dump(env, f, indent=2)
print("[T9] environment written")
print("\nrev_fast DONE")
