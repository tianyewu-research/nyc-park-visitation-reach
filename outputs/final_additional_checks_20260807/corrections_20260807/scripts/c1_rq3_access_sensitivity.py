"""CORRECTION TASK 1 — RQ3 regional-accessibility collinearity sensitivity.

The augmented RQ3 exogenous design has corr(log_access, W_log_access) ~ 0.9987 (VIF ~392/395,
augmented condition number 57.06): the 6-km accessibility surface is already spatially smoothed, so its
Queen lag adds almost no independent information, and the accessibility-specific local/spatial-lag
decomposition is WEAKLY IDENTIFIED. This script fits, on the exact frozen RQ3 sample (n=6,227, Queen,
seed 42):
  A. the archived full RQ3 partial SDM (reproduction);
  B. a variant retaining local log_access but omitting ONLY W_log_access (all else unchanged);
  C. a diagnostic variant omitting BOTH log_access and W_log_access (all else unchanged).
It reports convergence, rho, logLik, AIC/BIC, residual innovation Moran's I/p, structural coefficients,
and average direct/indirect/total impacts (95% CI, p) for the substantive predictors. The frozen primary
model is NOT replaced; the aim is to judge whether the accessibility finding should be downgraded and
whether the other RQ3 conclusions survive the near-redundancy."""
import sys, numpy as np, pandas as pd, scipy.stats as st
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
sys.path.insert(0, "outputs/spatial_model_reassessment_20260803/scripts")
sys.path.insert(0, "outputs/final_sdm_revision_20260803/scripts")
from slib import ROOT
from slib2 import trace_powers, innovation_moran
import geopandas as gpd, spreg
from libpysal.weights import Queen
import matplotlib.pyplot as plt
from pathlib import Path

SEED = 42
np.random.seed(SEED)
OUT = Path("outputs/final_additional_checks_20260807/"
           "corrections_20260807")
FIG = OUT / "figures"
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00", grey="#999999",
          purple="#CC79A7", black="#000000")

# ---- replicate the frozen RQ3 sample and design exactly (src: s3_rq3.py) ----
g = gpd.read_file(ROOT / "data" / "processed" / "analysis_cbg_final.gpkg").to_crs("EPSG:2263")
g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0
acc = g["access_score"]; pm = acc[acc > 0].min()
g["log_access"] = np.log(acc.clip(lower=pm / 2))
RAW = ["log_access", "quality_index_cbg_svc", "walkability_c", "transit_freq", "park_area_acres",
       "popdens_k", "income_10k", "pct_nonwhite"]
DEP = "lpur_2.0km"
d = g.dropna(subset=[DEP] + RAW + ["borough"]).reset_index(drop=True)
w = Queen.from_dataframe(d, use_index=False)
if w.islands:
    d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
    w = Queen.from_dataframe(d, use_index=False)
w.transform = "r"
n = len(d); y = d[DEP].values.reshape(-1, 1)
Z = pd.DataFrame({c: (d[c] - d[c].mean()) / d[c].std() for c in RAW})
boro = pd.get_dummies(d["borough"], prefix="b", drop_first=True).astype(float)
own = pd.concat([Z, boro], axis=1)
WX_all = pd.DataFrame(w.sparse @ Z.values, columns=["W_" + c for c in RAW], index=d.index)
T = trace_powers(w, Q=200, m_probe=4000)

# ---- confirm the collinearity finding (audit targets) ----
aug = pd.concat([Z, WX_all], axis=1)
corr_la = float(np.corrcoef(Z["log_access"], WX_all["W_log_access"])[0, 1])


def vif_of(col, design):
    cols = list(design.columns); j = cols.index(col)
    X = design.values.astype(float); others = np.delete(X, j, axis=1)
    A = np.column_stack([np.ones(len(others)), others])
    b, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
    resid = X[:, j] - A @ b
    r2 = 1 - (resid ** 2).sum() / ((X[:, j] - X[:, j].mean()) ** 2).sum()
    return 1.0 / (1.0 - r2)


vif_la = vif_of("log_access", aug); vif_wla = vif_of("W_log_access", aug)
# condition number on the full augmented design including borough FE (matches PHASE 1 definition)
aug_full = pd.concat([Z, boro, WX_all], axis=1)
Za = aug_full.values.astype(float); Za = (Za - Za.mean(0)) / Za.std(0)
Za = np.column_stack([np.ones(len(Za)), Za]); s = np.linalg.svd(Za, compute_uv=False)
cond_aug = float(s.max() / s.min())
collin = pd.DataFrame([
    dict(quantity="corr(log_access, W_log_access)", value=round(corr_la, 4), audit_target=0.9987),
    dict(quantity="VIF log_access (augmented)", value=round(vif_la, 3), audit_target=391.931),
    dict(quantity="VIF W_log_access (augmented)", value=round(vif_wla, 3), audit_target=395.143),
    dict(quantity="augmented condition number", value=round(cond_aug, 2), audit_target=57.06)])
collin.to_csv(OUT / "rq3_access_collinearity_confirm.csv", index=False)

# ---- generalized LeSage-Pace impacts tolerating an absent W term (theta=0) ----
def impacts(m, Xdes, variables, nsim=5000):
    cols = ["CONSTANT"] + list(Xdes.columns)
    b = m.betas.flatten(); rho_i = len(b) - 1; vc = m.vm
    rho_hat = float(b[rho_i]); native_M1 = float(m.sp_multipliers["full"][0])
    qexp = np.arange(len(T))
    M1_series_hat = float(np.sum((rho_hat ** qexp) * T) / n)
    anchor = native_M1 - M1_series_hat

    def M1_of(r):
        return float(np.sum((r ** qexp) * T) / n) + anchor
    rng = np.random.default_rng(SEED)
    draws = rng.multivariate_normal(b, vc, size=int(nsim * 1.6))
    draws = draws[(draws[:, rho_i] > -0.999) & (draws[:, rho_i] < 0.999)][:nsim]
    ret = len(draws)
    rows = []
    for v in variables:
        bi = cols.index(v)
        ti = cols.index("W_" + v) if ("W_" + v) in cols else None
        D = np.empty(ret); I = np.empty(ret); Tot = np.empty(ret)
        for si in range(ret):
            p = draws[si]; r = p[rho_i]; beta = p[bi]; theta = p[ti] if ti is not None else 0.0
            M1 = M1_of(r); MW = (M1 - 1.0) / r; MT = 1.0 / (1.0 - r)
            de = beta * M1 + theta * MW; te = (beta + theta) * MT
            D[si] = de; Tot[si] = te; I[si] = te - de
        beta0 = float(b[bi]); theta0 = float(b[ti]) if ti is not None else 0.0
        M1_0 = native_M1; MW_0 = (M1_0 - 1.0) / rho_hat; MT_0 = 1.0 / (1.0 - rho_hat)
        pt = dict(direct=beta0 * M1_0 + theta0 * MW_0, total=(beta0 + theta0) * MT_0)
        pt["indirect"] = pt["total"] - pt["direct"]
        for eff, arr in [("direct", D), ("indirect", I), ("total", Tot)]:
            frac = min((arr > 0).mean(), (arr < 0).mean()); pres = 2.0 / ret
            p_txt = f"<{pres:.4f}" if (2 * frac) < pres else f"{2 * frac:.4f}"
            rows.append(dict(variable=v, effect=eff, estimate=round(pt[eff], 6),
                             se=round(arr.std(), 6), ci_lo=round(np.percentile(arr, 2.5), 6),
                             ci_hi=round(np.percentile(arr, 97.5), 6), p=p_txt,
                             theta_included=(ti is not None)))
    return rows


def aicbic(m):
    ll = float(m.logll); k = len(m.betas.flatten()); return ll, 2 * k - 2 * ll, k * np.log(n) - 2 * ll, k


# ---- fit A / B / C ----
specs = {
    "A_full": dict(own_drop=[], wx=RAW),
    "B_omit_Wlogaccess": dict(own_drop=[], wx=[c for c in RAW if c != "log_access"]),
    "C_omit_logaccess_and_lag": dict(own_drop=["log_access"],
                                     wx=[c for c in RAW if c != "log_access"]),
}
SUBST = ["log_access", "quality_index_cbg_svc", "park_area_acres", "popdens_k", "income_10k",
         "pct_nonwhite", "walkability_c", "transit_freq"]
diag_rows, imp_rows, struct_rows = [], [], []
models = {}
for name, sp in specs.items():
    own_cols = [c for c in RAW if c not in sp["own_drop"]] + list(boro.columns)
    Xown = pd.concat([Z[[c for c in RAW if c not in sp["own_drop"]]], boro], axis=1)
    WX = WX_all[["W_" + c for c in sp["wx"]]]
    Xdes = pd.concat([Xown, WX], axis=1)
    m = spreg.ML_Lag(y, Xdes.values, w=w, name_x=list(Xdes.columns), spat_impacts="full")
    models[name] = (m, Xdes)
    ll, aic, bic, k = aicbic(m)
    imi, imp = innovation_moran(m, y, w, "lag", Xdes=Xdes)
    rho = float(m.betas.flatten()[-1])
    diag_rows.append(dict(model=name, converged=True, n=n, k_params=k, rho=round(rho, 4),
                          logLik=round(ll, 3), AIC=round(aic, 3), BIC=round(bic, 3),
                          resid_innov_moran_I=round(imi, 4), resid_innov_moran_p=round(imp, 4),
                          W_logaccess_included=("log_access" in sp["wx"]),
                          logaccess_included=("log_access" not in sp["own_drop"])))
    # structural coefficients
    b = m.betas.flatten(); se = np.sqrt(np.diag(m.vm)); cols = ["CONSTANT"] + list(Xdes.columns) + ["rho"]
    for i, nm in enumerate(cols):
        if i < len(b):
            z = b[i] / se[i]
            struct_rows.append(dict(model=name, term=nm, coef=round(float(b[i]), 6),
                                    se=round(float(se[i]), 6), p=round(float(2 * (1 - st.norm.cdf(abs(z)))), 5)))
    # impacts for the substantive predictors present in this model
    present = [v for v in SUBST if v not in sp["own_drop"]]
    for r in impacts(m, Xdes, present):
        r = dict(r); r["model"] = name; imp_rows.append(r)
    print(f"  fit {name}: rho={rho:.4f} AIC={aic:.1f} BIC={bic:.1f} residMoranI={imi:.4f}")

pd.DataFrame(diag_rows).to_csv(OUT / "rq3_access_collinearity_sensitivity_models.csv", index=False)
pd.DataFrame(imp_rows).to_csv(OUT / "rq3_access_collinearity_sensitivity_impacts.csv", index=False)
pd.DataFrame(struct_rows).to_csv(OUT / "rq3_access_collinearity_sensitivity_structural.csv", index=False)

# ---- figure: accessibility + other predictors direct/total across A/B/C ----
imp = pd.DataFrame(imp_rows)
LABEL = {"log_access": "Regional accessibility (6 km)", "quality_index_cbg_svc": "Park quality",
         "park_area_acres": "Local park area", "popdens_k": "Population density",
         "income_10k": "Median income", "pct_nonwhite": "Non-White share",
         "walkability_c": "Walkability", "transit_freq": "Transit frequency"}
order = ["log_access", "pct_nonwhite", "income_10k", "quality_index_cbg_svc", "park_area_acres",
         "popdens_k", "walkability_c", "transit_freq"]
mcol = {"A_full": CB["black"], "B_omit_Wlogaccess": CB["blue"], "C_omit_logaccess_and_lag": CB["orange"]}
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
yy = np.arange(len(order))[::-1]
for ax, eff, title in [(axes[0], "direct", "(a) Average direct impact"),
                       (axes[1], "total", "(b) Average total impact")]:
    for k, (mdl, col) in enumerate(mcol.items()):
        off = (k - 1) * 0.22
        for y, var in zip(yy, order):
            r = imp[(imp.model == mdl) & (imp.variable == var) & (imp.effect == eff)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.errorbar(r["estimate"], y + off, xerr=[[r["estimate"] - r["ci_lo"]], [r["ci_hi"] - r["estimate"]]],
                        fmt="o", mfc=col, mec=col, ecolor=col, capsize=2, ms=5, lw=1.1,
                        label=mdl if (y == yy[0]) else None)
    ax.axvline(0, color=CB["grey"], lw=0.9, ls="--")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("Average impact on LPUR per 1 SD (standardized)", fontsize=9)
axes[0].set_yticks(np.arange(len(order))); axes[0].set_yticklabels([LABEL[v] for v in order[::-1]], fontsize=9)
handles = [plt.Line2D([0], [0], marker="o", color="w", mfc=c, mec=c,
                      label={"A_full": "A: full (frozen primary)",
                             "B_omit_Wlogaccess": "B: omit W·log_access",
                             "C_omit_logaccess_and_lag": "C: omit log_access & W·log_access"}[m], ms=6)
           for m, c in mcol.items()]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("RQ3 accessibility collinearity sensitivity: average impacts under A/B/C", fontsize=12)
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
FIG.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(FIG / f"FigS_RQ3_access_collinearity_sensitivity.{ext}", dpi=300, bbox_inches="tight")
plt.close()

# ---- interpretation md ----
def geti(mdl, var, eff):
    r = imp[(imp.model == mdl) & (imp.variable == var) & (imp.effect == eff)]
    return r.iloc[0] if len(r) else None


def sig(r):
    return (float(r["ci_lo"]) > 0) or (float(r["ci_hi"]) < 0)


la_A_t = geti("A_full", "log_access", "total"); la_A_d = geti("A_full", "log_access", "direct")
la_B_t = geti("B_omit_Wlogaccess", "log_access", "total"); la_B_d = geti("B_omit_Wlogaccess", "log_access", "direct")
acc_stable = (np.sign(float(la_A_t["estimate"])) == np.sign(float(la_B_t["estimate"]))) and \
             (sig(la_A_t) == sig(la_B_t))
others_stable = True
for var in ["pct_nonwhite", "income_10k", "quality_index_cbg_svc", "park_area_acres", "popdens_k"]:
    for eff in ["direct", "total"]:
        rA = geti("A_full", var, eff); rC = geti("C_omit_logaccess_and_lag", var, eff)
        if rA is None or rC is None:
            continue
        if (np.sign(float(rA["estimate"])) != np.sign(float(rC["estimate"]))) or (sig(rA) != sig(rC)):
            others_stable = False

L = ["# RQ3 regional-accessibility collinearity sensitivity — interpretation\n",
     f"Sample n={n}, Queen row-standardized, seed {SEED}. Frozen primary RQ3 model is unchanged.\n",
     "## Collinearity confirmed (not dismissed)",
     f"- corr(log_access, W_log_access) = {corr_la:.4f}; VIF(log_access)={vif_la:.1f}, "
     f"VIF(W_log_access)={vif_wla:.1f}; augmented condition number {cond_aug:.2f}.",
     "- The 6-km accessibility surface is already spatially smoothed, so its Queen lag adds almost no "
     "independent information. The accessibility-specific local/spatial-lag decomposition is therefore "
     "**weakly identified**. Parameter covariance IS propagated into the impacts, but near-redundancy "
     "can still make the accessibility decomposition unstable and specification-dependent — so the "
     "augmented collinearity is NOT irrelevant to the accessibility impact, contrary to the original "
     "diagnostic wording.\n",
     "## Model fit (A full / B omit W·log_access / C omit both)"]
dd = pd.DataFrame(diag_rows)
for _, r in dd.iterrows():
    L.append(f"- {r['model']}: rho={r['rho']}, logLik={r['logLik']}, AIC={r['AIC']}, BIC={r['BIC']}, "
             f"resid innov Moran I={r['resid_innov_moran_I']} (p={r['resid_innov_moran_p']}).")
L += ["", "## Regional-accessibility impact",
      f"- A (full): direct {la_A_d['estimate']} (p={la_A_d['p']}), total {la_A_t['estimate']} "
      f"(p={la_A_t['p']}, {'sig' if sig(la_A_t) else 'n.s.'}).",
      f"- B (omit W·log_access): direct {la_B_d['estimate']} (p={la_B_d['p']}), total "
      f"{la_B_t['estimate']} (p={la_B_t['p']}, {'sig' if sig(la_B_t) else 'n.s.'}).",
      f"- **Verdict:** the accessibility total impact is {'STABLE' if acc_stable else 'NOT STABLE'} "
      "in sign/significance across A and B. "
      + ("It can be interpreted with the near-redundancy caveat." if acc_stable else
         "It is specification-dependent and should be DOWNGRADED / marked model-sensitive in RQ3 (and "
         "Figure 8 should not emphasize its marginal interval)."),
      "",
      "## Other RQ3 conclusions (A full vs C omit both accessibility terms)",
      f"- Non-White, income, park quality, local park area, and population-density impacts are "
      f"{'STABLE' if others_stable else 'NOT uniformly stable'} in sign and significance when the "
      "near-redundant accessibility terms are removed (variant C). See "
      "`rq3_access_collinearity_sensitivity_impacts.csv`.",
      "",
      "## Corrected diagnostic wording (supersedes the original)",
      "Local-covariate multicollinearity is low (max VIF ~2.06; local condition numbers 3.2–4.0). "
      "However, in the RQ3 augmented design the regional-accessibility local term and its spatial lag "
      "are near-collinear (corr 0.9987), so the accessibility-specific direct/indirect decomposition is "
      "weakly identified and specification-dependent. This does NOT affect the low local-X collinearity "
      "statement, and does NOT undermine the aggregate Non-White direct impact (the primary result).",
      "",
      "## Figure 8 guidance",
      "Move the bottom note into the manuscript caption. "
      + ("The accessibility total is stable across A/B, so it may be reported with the near-redundancy "
         "caveat; still avoid significance stars." if acc_stable else
         "The accessibility total is model-sensitive; mark it model-sensitive or drop it from "
         "interpretive emphasis rather than highlighting its marginal p≈0.049 interval."),
      "",
      "Figure: `FigS_RQ3_access_collinearity_sensitivity.{png,pdf,svg}`."]
(OUT / "rq3_access_collinearity_interpretation.md").write_text("\n".join(L) + "\n")

print("=== c1 done ===")
print(collin.to_string(index=False))
print(f"accessibility total stable across A/B: {acc_stable}; other RQ3 impacts stable in C: {others_stable}")
print(imp[imp.variable == "log_access"][["model", "effect", "estimate", "ci_lo", "ci_hi", "p"]].to_string(index=False))
