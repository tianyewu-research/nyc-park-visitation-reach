"""PHASE 1 — multicollinearity diagnostics + racial-composition coefficient stability.

For the RQ2 aggregate, RQ2 joint-composition, and RQ3 models:
  * Pearson/Spearman correlations among continuous LOCAL predictors;
  * VIFs for the local design X (borough FE included, reference = Bronx);
  * standardized condition number for X;
  * the same for the estimable augmented exogenous design [X, WX] (excluding Wy).
Then, on the fixed joint-composition sample, refit three SEPARATE partial SDMs (one race share at a
time) for both LPUR and mean distance, and compare average impacts with the joint model.

VIF thresholds are NOT used as pass/fail; values are reported with the variables involved. Separate
models are sign/uncertainty diagnostics only: a separate-share coefficient contrasts the named group
with ALL remaining composition, whereas a joint coefficient is conditional on the other shares and the
omitted reference — the estimands differ, so magnitudes are not compared like-for-like."""
import sys, numpy as np, pandas as pd, geopandas as gpd, scipy.stats as st
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUTDIR, SEED, RACE, RACE_LABEL, CB, savefig
from slib import build_sample, make_W, CONT, DEP
from slib2 import fit_sdm, design_sdm, trace_powers, sdm_impacts, boro_dummies
import spreg
from libpysal.weights import Queen
import matplotlib.pyplot as plt

np.random.seed(SEED)
GPKG = ROOT / "data" / "processed" / "analysis_cbg_final.gpkg"

# ---------------- assemble the three model samples/designs ----------------
d, gid, checksum = build_sample()                       # RQ2 aggregate/joint LPUR sample (n=6252)
wq = make_W(d, "queen")
Tq = trace_powers(wq, Q=200, m_probe=4000)

CONT_AGG = CONT                                          # income, nonwhite, popdens, park_area
CONT_JOINT = ["income_10k"] + RACE + ["popdens_k", "park_area_acres"]

# RQ3 sample + standardized predictors (replicate s3_rq3.py exactly)
g = gpd.read_file(GPKG).to_crs("EPSG:2263")
g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0
acc = g["access_score"]; pm = acc[acc > 0].min()
g["log_access"] = np.log(acc.clip(lower=pm / 2))
RQ3_RAW = ["log_access", "quality_index_cbg_svc", "walkability_c", "transit_freq",
           "park_area_acres", "popdens_k", "income_10k", "pct_nonwhite"]
d3 = g.dropna(subset=[DEP] + RQ3_RAW + ["borough"]).reset_index(drop=True)
w3 = Queen.from_dataframe(d3, use_index=False)
if w3.islands:
    d3 = d3.iloc[[i for i in range(len(d3)) if i not in set(w3.islands)]].reset_index(drop=True)
    w3 = Queen.from_dataframe(d3, use_index=False)
w3.transform = "r"
d3z = d3.copy()
for c in RQ3_RAW:
    d3z[c] = (d3[c] - d3[c].mean()) / d3[c].std()       # z-scored (RQ3 predictors are standardized)


def local_and_augmented(data, cont, w):
    """Return (X_local DataFrame, X_aug DataFrame) matching the SDM design (borough FE local only,
    W*cont appended for the augmented exogenous design; Wy excluded)."""
    boro = boro_dummies(data)
    Xloc = pd.concat([data[cont].reset_index(drop=True), boro.reset_index(drop=True)], axis=1)
    WX = pd.DataFrame(w.sparse @ data[cont].values, columns=["W_" + c for c in cont])
    Xaug = pd.concat([Xloc, WX], axis=1)
    return Xloc, Xaug


def vif_table(X, focus_cols, model, design):
    """VIF = 1/(1-R2) regressing each focus column on all other columns of X (intercept added).
    Numerically-unstable (near-perfect) fits are reported as such rather than as a huge finite VIF."""
    rows = []
    cols = list(X.columns)
    Xv = X.values.astype(float)
    for c in focus_cols:
        j = cols.index(c)
        others = np.delete(Xv, j, axis=1)
        A = np.column_stack([np.ones(len(others)), others])
        yj = Xv[:, j]
        beta, *_ = np.linalg.lstsq(A, yj, rcond=None)
        resid = yj - A @ beta
        ss_tot = float(((yj - yj.mean()) ** 2).sum())
        r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
        stable = np.isfinite(r2) and r2 < 0.9995
        vif = 1.0 / (1.0 - r2) if stable else np.nan
        rows.append(dict(model=model, design=design, variable=c,
                         vif=round(vif, 3) if np.isfinite(vif) else "unstable (R2>0.9995)",
                         r2_on_others=round(r2, 5), numerically_stable=bool(stable)))
    return rows


def cond_number(X):
    """Standardized (z-scored) condition number of the design (Belsley-style): column-standardize to
    unit variance, add intercept, and take the ratio of the largest to smallest singular value."""
    Z = X.values.astype(float).copy()
    sd = Z.std(axis=0); sd[sd == 0] = 1.0
    Z = (Z - Z.mean(axis=0)) / sd
    Z = np.column_stack([np.ones(len(Z)), Z])
    s = np.linalg.svd(Z, compute_uv=False)
    return float(s.max() / s.min())


def corr_long(data, cont, model, method):
    m = data[cont].corr(method=method)
    out = []
    for i, a in enumerate(cont):
        for b in cont[i + 1:]:
            out.append(dict(model=model, method=method, var1=a, var2=b, corr=round(float(m.loc[a, b]), 4)))
    return out


# ---------------- run diagnostics per model ----------------
MODELS = [("RQ2_aggregate", d, CONT_AGG, wq),
          ("RQ2_joint", d, CONT_JOINT, wq),
          ("RQ3_standardized", d3z, RQ3_RAW, w3)]

corr_local, corr_aug, vif_rows, cond_rows = [], [], [], []
for name, data, cont, w in MODELS:
    Xloc, Xaug = local_and_augmented(data, cont, w)
    # correlations among continuous local predictors
    corr_local += corr_long(data, cont, name, "pearson")
    corr_local += corr_long(data, cont, name, "spearman")
    # correlations among the augmented continuous set (own + W-lagged)
    aug_cont = cont + ["W_" + c for c in cont]
    aug_df = pd.concat([data[cont].reset_index(drop=True),
                        pd.DataFrame(w.sparse @ data[cont].values,
                                     columns=["W_" + c for c in cont])], axis=1)
    corr_aug += corr_long(aug_df, aug_cont, name, "pearson")
    corr_aug += corr_long(aug_df, aug_cont, name, "spearman")
    # VIFs: continuous only (borough dummies are controls, not of interest)
    vif_rows += vif_table(Xloc, cont, name, "local_X")
    vif_rows += vif_table(Xaug, cont + ["W_" + c for c in cont], name, "augmented_X_WX")
    # condition numbers
    cond_rows.append(dict(model=name, design="local_X", condition_number=round(cond_number(Xloc), 2),
                          n_cols=Xloc.shape[1] + 1))
    cond_rows.append(dict(model=name, design="augmented_X_WX",
                          condition_number=round(cond_number(Xaug), 2), n_cols=Xaug.shape[1] + 1))

pd.DataFrame(corr_local).to_csv(OUTDIR / "collinearity_local_X.csv", index=False)
pd.DataFrame(corr_aug).to_csv(OUTDIR / "collinearity_augmented_X_WX.csv", index=False)
pd.DataFrame(vif_rows).to_csv(OUTDIR / "vif_diagnostics.csv", index=False)
pd.DataFrame(cond_rows).to_csv(OUTDIR / "condition_numbers.csv", index=False)

# ---------------- separate vs joint racial-composition models ----------------
def iqr(col, data):
    return float(data[col].quantile(.75) - data[col].quantile(.25))


# distance sample (mean_trip_km) for the companion outcome
dd = g.dropna(subset=["mean_trip_km"] + CONT + ["borough"]).reset_index(drop=True)
wdd = Queen.from_dataframe(dd, use_index=False)
if wdd.islands:
    dd = dd.iloc[[i for i in range(len(dd)) if i not in set(wdd.islands)]].reset_index(drop=True)
    wdd = Queen.from_dataframe(dd, use_index=False)
wdd.transform = "r"
Tdd = trace_powers(wdd, Q=200, m_probe=4000)

sep_rows, comp_rows = [], []
for outcome, dep, data, w, T in [("LPUR", DEP, d, wq, Tq),
                                 ("distance", "mean_trip_km", dd, wdd, Tdd)]:
    # JOINT model
    contJ = CONT_JOINT
    Xj = design_sdm(data, w, cont=contJ)
    mj = spreg.ML_Lag(data[dep].values.reshape(-1, 1), Xj.values, w=w,
                      name_x=list(Xj.columns), spat_impacts="full")
    scalesJ = {v: [("IQR", iqr(v, data))] for v in RACE}
    scalesJ.update({v: [("unit", 1.0)] for v in ["income_10k", "popdens_k", "park_area_acres"]})
    rowsJ, retJ, pmJ = sdm_impacts(mj, Xj, w, T, contJ, scalesJ, nsim=5000)
    joint_imp = {(r["variable"], r["effect"]): r for r in rowsJ if r["variable"] in RACE}

    for race in RACE:
        # SEPARATE model: replace the three joint race terms with this one share
        contS = ["income_10k", race, "popdens_k", "park_area_acres"]
        Xs = design_sdm(data, w, cont=contS)
        ms = spreg.ML_Lag(data[dep].values.reshape(-1, 1), Xs.values, w=w,
                          name_x=list(Xs.columns), spat_impacts="full")
        scalesS = {race: [("IQR", iqr(race, data))]}
        scalesS.update({v: [("unit", 1.0)] for v in ["income_10k", "popdens_k", "park_area_acres"]})
        rowsS, retS, pmS = sdm_impacts(ms, Xs, w, T, contS, scalesS, nsim=5000)
        for r in rowsS:
            if r["variable"] == race:
                rr = dict(r); rr["outcome"] = outcome; rr["model"] = "separate"
                rr["specification"] = f"{RACE_LABEL[race]} share alone (vs all remaining composition)"
                sep_rows.append(rr)
        # comparison of sign/uncertainty (NOT magnitude): separate vs joint
        for eff in ["direct", "indirect", "total"]:
            sepr = [r for r in rowsS if r["variable"] == race and r["effect"] == eff][0]
            jr = joint_imp[(race, eff)]
            def sig(r):  # significant if CI excludes 0
                return (float(r["ci_lo"]) > 0) or (float(r["ci_hi"]) < 0)
            comp_rows.append(dict(
                outcome=outcome, race=RACE_LABEL[race], effect=eff,
                separate_estimate=sepr["estimate"], separate_ci=f"[{sepr['ci_lo']},{sepr['ci_hi']}]",
                separate_sig=sig(sepr), joint_estimate=jr["estimate"],
                joint_ci=f"[{jr['ci_lo']},{jr['ci_hi']}]", joint_sig=sig(jr),
                sign_agrees=bool(np.sign(float(sepr["estimate"])) == np.sign(float(jr["estimate"]))),
                note="estimands differ (separate=vs-all-remaining; joint=conditional); compare sign/uncertainty only"))

pd.DataFrame(sep_rows).to_csv(OUTDIR / "racial_composition_separate_models_impacts.csv", index=False)
pd.DataFrame(comp_rows).to_csv(OUTDIR / "racial_composition_joint_vs_separate_comparison.csv", index=False)

# ---------------- FigS: correlation heatmap (RQ2 joint augmented set) ----------------
aug_cont_j = CONT_JOINT + ["W_" + c for c in CONT_JOINT]
augdf = pd.concat([d[CONT_JOINT].reset_index(drop=True),
                   pd.DataFrame(wq.sparse @ d[CONT_JOINT].values,
                                columns=["W_" + c for c in CONT_JOINT])], axis=1)
C = augdf[aug_cont_j].corr(method="pearson")
short = {c: c.replace("pct_", "").replace("_10k", "").replace("_acres", "").replace("popdens_k", "popdens")
         .replace("park_area", "parkarea").replace("W_", "W·") for c in aug_cont_j}
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(aug_cont_j))); ax.set_yticks(range(len(aug_cont_j)))
ax.set_xticklabels([short[c] for c in aug_cont_j], rotation=90, fontsize=8)
ax.set_yticklabels([short[c] for c in aug_cont_j], fontsize=8)
for i in range(len(aug_cont_j)):
    for j in range(len(aug_cont_j)):
        v = C.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(v) > 0.6 else "black")
cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Pearson correlation", fontsize=9)
ax.set_title("RQ2 joint-composition augmented exogenous design [X, WX] (Pearson)", fontsize=10)
plt.tight_layout()
p_heat = savefig(fig, "FigS_correlation_heatmap"); plt.close()

# ---------------- FigS: joint vs separate racial-composition impacts (LPUR + distance) ----------------
comp = pd.DataFrame(comp_rows)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, outcome, unit in [(axes[0], "LPUR", "per-IQR impact on LPUR"),
                          (axes[1], "distance", "per-IQR impact on distance (km)")]:
    sub = comp[(comp.outcome == outcome) & (comp.effect.isin(["direct", "total"]))]
    labs = ["Black", "Hispanic", "Asian"]
    yb = np.arange(len(labs))[::-1]
    for y, lab in zip(yb, labs):
        for eff, off, col in [("direct", 0.16, CB["blue"]), ("total", -0.16, CB["vermillion"])]:
            for mdl, mk, fc in [("joint", "o", None), ("separate", "s", "white")]:
                key = f"{mdl}_estimate"; cikey = f"{mdl}_ci"
                r = sub[(sub.race == lab) & (sub.effect == eff)]
                if not len(r):
                    continue
                r = r.iloc[0]
                est = float(r[key]); lo, hi = [float(x) for x in r[cikey].strip("[]").split(",")]
                yy = y + off + (0.06 if mdl == "separate" else -0.06)
                ax.errorbar(est, yy, xerr=[[est - lo], [hi - est]], fmt=mk,
                            mfc=(fc if fc else col), mec=col, ecolor=col, capsize=2, ms=6, lw=1.1)
    ax.axvline(0, color=CB["grey"], lw=0.9, ls="--")
    ax.set_yticks(np.arange(len(labs))); ax.set_yticklabels(labs[::-1], fontsize=9)
    ax.set_xlabel(unit, fontsize=9)
    ax.set_title(f"({'a' if outcome=='LPUR' else 'b'}) {outcome}", fontsize=10, loc="left", fontweight="bold")
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", mfc=CB["blue"], mec=CB["blue"], label="Direct impact (blue)", ms=6),
       Line2D([0], [0], marker="o", color="w", mfc=CB["vermillion"], mec=CB["vermillion"], label="Total impact (orange)", ms=6),
       Line2D([0], [0], marker="o", color="w", mfc=CB["grey"], mec=CB["grey"], label="● filled = joint model", ms=6),
       Line2D([0], [0], marker="s", color="w", mfc="white", mec=CB["grey"], label="□ open = separate model", ms=6)]
fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Racial-composition average impacts: joint vs separate SDMs (sign/uncertainty diagnostic)",
             fontsize=10)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
p_js = savefig(fig, "FigS_joint_vs_separate"); plt.close()

# ---------------- diagnostic report ----------------
vdf = pd.DataFrame(vif_rows); cdf = pd.DataFrame(cond_rows)
max_local_vif = vdf[(vdf.design == "local_X") & vdf.numerically_stable]["vif"].max()
lines = ["# Collinearity and composition-stability diagnostic (PHASE 1)\n",
         f"Samples: RQ2 aggregate/joint n={len(d)} (checksum {checksum}); RQ3 n={len(d3z)}; "
         f"distance n={len(dd)}. Reference borough = Bronx. Seed {SEED}.\n",
         "## Multicollinearity",
         "- Continuous-predictor correlations (Pearson/Spearman) are in `collinearity_local_X.csv`; "
         "the augmented own+W-lagged set is in `collinearity_augmented_X_WX.csv`.",
         f"- Largest numerically-stable VIF in any LOCAL design X: {max_local_vif}. VIFs are reported, "
         "not thresholded; see `vif_diagnostics.csv` for every variable and design.",
         "- Standardized condition numbers are in `condition_numbers.csv`. The augmented [X, WX] design "
         "carries higher condition numbers than the local X because each W-lagged term is strongly "
         "correlated with its own covariate; this is expected under an SLX/SDM specification and is a "
         "known interpretive caveat for the individual structural coefficients — NOT for the average "
         "direct/indirect/total impacts, which are the reported quantities.",
         "\n## Composition stability (separate vs joint)",
         "- `racial_composition_separate_models_impacts.csv` fits Black/Hispanic/Asian shares one at a "
         "time (partial SDM, same controls/sample/weights) for LPUR and mean distance.",
         "- `racial_composition_joint_vs_separate_comparison.csv` compares SIGN and uncertainty only.",
         "- Interpretation guard: a separate-share coefficient contrasts that group with ALL remaining "
         "composition; a joint coefficient is conditional on the other shares and the omitted reference. "
         "The estimands are not identical, so magnitude differences are expected and must not be read as "
         "instability. Use the separate models only to check that signs and significance are stable.\n",
         "## Sign/uncertainty summary"]
for outcome in ["LPUR", "distance"]:
    sub = comp[(comp.outcome == outcome) & (comp.effect.isin(["direct", "total"]))]
    agree = int(sub["sign_agrees"].sum()); tot = len(sub)
    lines.append(f"- {outcome}: sign agrees between separate and joint in {agree}/{tot} race×effect (direct+total) cells.")
    disc = sub[~sub["sign_agrees"]]
    for _, r in disc.iterrows():
        both_ns = (not r["separate_sig"]) and (not r["joint_sig"])
        lines.append(f"  - sign flip: {r['race']} {r['effect']} (separate {r['separate_estimate']} "
                     f"vs joint {r['joint_estimate']})"
                     + (" — both non-significant, so not a substantive conflict" if both_ns else
                        " — at least one estimate significant; inspect"))
lines.append("")
lines.append(f"Figures: `{p_heat[0]}` (+pdf/svg), `{p_js[0]}` (+pdf/svg).")
(OUTDIR / "collinearity_and_composition_diagnostic.md").write_text("\n".join(lines) + "\n")

print("=== PHASE 1 done ===")
print("VIF (local_X, stable max):", max_local_vif)
print(cdf.to_string(index=False))
print("\nsign agreement (direct+total):")
print(comp[comp.effect.isin(["direct", "total"])].groupby("outcome")["sign_agrees"].mean().round(3).to_string())
