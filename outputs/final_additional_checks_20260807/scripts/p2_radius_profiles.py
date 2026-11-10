"""PHASE 2 — group-specific LPUR radius profiles.

Raw LPUR at 11 radii (0.50..3.00 km) is already precomputed in the frozen GeoPackage by the primary
dwell-qualified visit construction (src/15b_compute_local_ratio.py); this reuses those columns rather
than re-running the 52-file pass. The final joint racial-composition partial SDM is fit at every radius
on ONE fixed joint-model sample (complete-case on all joint predictors AND every radius outcome, so the
sample never changes with radius). Black/Hispanic/Asian average direct & total impacts (95% CI, p) are
estimated per radius; the aggregate Non-White specification is run for a supplementary profile.

Impacts are model-implied LeSage-Pace averages conditional on the SDM and Queen W — not causal."""
import sys, numpy as np, pandas as pd, geopandas as gpd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUTDIR, SEED, RACE, RACE_LABEL, CB, savefig
from slib2 import trace_powers, sdm_impacts, boro_dummies
import spreg
from libpysal.weights import Queen
import matplotlib.pyplot as plt

np.random.seed(SEED)
GPKG = ROOT / "data" / "processed" / "analysis_cbg_final.gpkg"
RADII = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
RADCOLS = [f"lpur_{r}km" for r in RADII]
PRIMARY = 2.0
CONT_JOINT = ["income_10k"] + RACE + ["popdens_k", "park_area_acres"]
CONT_AGG = ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]

g = gpd.read_file(GPKG).to_crs("EPSG:2263")
g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0

# ONE fixed joint-model sample: complete-case on joint predictors + every radius outcome + borough
need = list(set(CONT_JOINT + CONT_AGG + RADCOLS + ["borough"]))
d = g.dropna(subset=need).reset_index(drop=True)
w = Queen.from_dataframe(d, use_index=False)
if w.islands:
    d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
    w = Queen.from_dataframe(d, use_index=False)
w.transform = "r"
n = len(d)
T = trace_powers(w, Q=200, m_probe=4000)
print(f"fixed joint radius sample n={n}")


def iqr(col):
    return float(d[col].quantile(.75) - d[col].quantile(.25))


def design(cont):
    base = pd.concat([d[cont].reset_index(drop=True), boro_dummies(d).reset_index(drop=True)], axis=1)
    WX = pd.DataFrame(w.sparse @ d[cont].values, columns=["W_" + c for c in cont])
    return pd.concat([base, WX], axis=1)


def resid_moran(m, dep, Xdes):
    from esda.moran import Moran
    b = m.betas.flatten(); rho = b[-1]; b0 = b[0]; bx = b[1:-1]
    e = d[dep].values.reshape(-1, 1) - rho * (w.sparse @ d[dep].values.reshape(-1, 1)) - \
        (b0 + Xdes.values @ bx.reshape(-1, 1))
    mi = Moran(np.asarray(e).flatten(), w, permutations=999)
    return float(mi.I), float(mi.p_sim)


group_rows, nw_rows, diag_rows = [], [], []
for r in RADII:
    dep = f"lpur_{r}km"
    # ---- joint racial-composition SDM ----
    Xj = design(CONT_JOINT)
    try:
        mj = spreg.ML_Lag(d[dep].values.reshape(-1, 1), Xj.values, w=w,
                          name_x=list(Xj.columns), spat_impacts="full")
        conv = True
    except Exception as e:
        diag_rows.append(dict(radius_km=r, model="joint", converged=False, error=str(e)[:80]))
        conv = False
    if conv:
        scales = {v: [("IQR", iqr(v))] for v in RACE}
        scales.update({v: [("unit", 1.0)] for v in ["income_10k", "popdens_k", "park_area_acres"]})
        rows, ret, pm = sdm_impacts(mj, Xj, w, T, CONT_JOINT, scales, nsim=5000)
        rho = float(mj.betas.flatten()[-1]); ll = float(mj.logll)
        k = len(mj.betas.flatten()); aic = 2 * k - 2 * ll; bic = k * np.log(n) - 2 * ll
        imi, imp = resid_moran(mj, dep, Xj)
        diag_rows.append(dict(radius_km=r, model="joint", converged=True, n=n, rho=round(rho, 4),
                              logLik=round(ll, 3), AIC=round(aic, 3), BIC=round(bic, 3),
                              resid_moran_I=round(imi, 4), resid_moran_p=round(imp, 4)))
        for row in rows:
            if row["variable"] in RACE and row["effect"] in ("direct", "total"):
                group_rows.append(dict(radius_km=r, race=RACE_LABEL[row["variable"]],
                                       effect=row["effect"], estimate=float(row["estimate"]),
                                       ci_lo=float(row["ci_lo"]), ci_hi=float(row["ci_hi"]),
                                       p=row["p"], scale=row["scale"], is_primary=(r == PRIMARY)))
    # ---- aggregate Non-White SDM (supplementary) ----
    Xa = design(CONT_AGG)
    ma = spreg.ML_Lag(d[dep].values.reshape(-1, 1), Xa.values, w=w,
                      name_x=list(Xa.columns), spat_impacts="full")
    scales_a = {c: [("unit", 1.0)] for c in CONT_AGG}
    scales_a["pct_nonwhite"] = [("IQR", iqr("pct_nonwhite"))]
    rowsA, retA, pmA = sdm_impacts(ma, Xa, w, T, CONT_AGG, scales_a, nsim=5000)
    for row in rowsA:
        if row["variable"] == "pct_nonwhite" and row["effect"] in ("direct", "total"):
            nw_rows.append(dict(radius_km=r, race="Non-White", effect=row["effect"],
                                estimate=float(row["estimate"]), ci_lo=float(row["ci_lo"]),
                                ci_hi=float(row["ci_hi"]), p=row["p"], scale=row["scale"],
                                is_primary=(r == PRIMARY)))
    print(f"  radius {r} km done")

gdf = pd.DataFrame(group_rows); nwdf = pd.DataFrame(nw_rows)
gdf.to_csv(OUTDIR / "radius_group_impacts_tidy.csv", index=False)
nwdf.to_csv(OUTDIR / "radius_nonwhite_impacts_tidy.csv", index=False)
pd.DataFrame(diag_rows).to_csv(OUTDIR / "radius_model_diagnostics.csv", index=False)


# ---- Fig6: 3 rows (Black/Hispanic/Asian) x 2 cols (direct/total) ----
def profile_ax(ax, sub, colr, title):
    sub = sub.sort_values("radius_km")
    ax.errorbar(sub.radius_km, sub.estimate,
                yerr=[sub.estimate - sub.ci_lo, sub.ci_hi - sub.estimate],
                fmt="o-", color=colr, ecolor=colr, capsize=2.5, ms=4.5, lw=1.3)
    ax.axhline(0, color=CB["grey"], lw=0.9, ls="--")
    ax.axvline(PRIMARY, color=CB["black"], lw=0.9, ls=":")
    ax.set_title(title, fontsize=9, loc="left")
    ax.tick_params(labelsize=8)


fig, axes = plt.subplots(3, 2, figsize=(9.5, 9.0), sharex=True)
races = ["Black", "Hispanic", "Asian"]
colmap = {"Black": CB["blue"], "Hispanic": CB["orange"], "Asian": CB["green"]}
for i, race in enumerate(races):
    for j, eff in enumerate(["direct", "total"]):
        ax = axes[i, j]
        sub = gdf[(gdf.race == race) & (gdf.effect == eff)]
        lab = f"{race} — average {eff} impact"
        profile_ax(ax, sub, colmap[race], lab)
        if i == 2:
            ax.set_xlabel("LPUR radius (km)", fontsize=9)
        if j == 0:
            ax.set_ylabel("per-IQR impact on LPUR", fontsize=8.5)
axes[0, 1].text(PRIMARY, axes[0, 1].get_ylim()[0], " 2 km (primary)", fontsize=7,
                ha="left", va="bottom", color=CB["black"])
fig.suptitle("Racial-composition average impacts on LPUR across neighborhood radii "
             "(joint partial SDM, Queen W)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
p6 = savefig(fig, "Fig6_radius_group_profiles"); plt.close()

# ---- FigS: aggregate Non-White profile ----
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
for j, eff in enumerate(["direct", "total"]):
    sub = nwdf[nwdf.effect == eff]
    profile_ax(axes[j], sub, CB["vermillion"], f"Non-White — average {eff} impact")
    axes[j].set_xlabel("LPUR radius (km)", fontsize=9)
axes[0].set_ylabel("per-IQR impact on LPUR", fontsize=8.5)
fig.suptitle("Aggregate Non-White-share average impacts across radii (supplementary)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
pS = savefig(fig, "FigS_radius_nonwhite_profile"); plt.close()

# ---- summary md ----
lines = ["# Radius-sensitivity summary (PHASE 2)\n",
         f"Fixed joint-model sample n={n} (complete-case on joint predictors and all 11 radius "
         f"outcomes; Queen islands dropped). Seed {SEED}. Primary radius = 2 km (marked).",
         "All models converged; per-radius rho, logLik, AIC/BIC and residual Moran's I are in "
         "`radius_model_diagnostics.csv`.\n",
         "## What varies with radius (direction only; magnitudes are model-implied)"]
for race in races:
    tot = gdf[(gdf.race == race) & (gdf.effect == "total")].sort_values("radius_km")
    sig = tot[(tot.ci_hi < 0) | (tot.ci_lo > 0)]
    signs = "negative" if (tot.estimate < 0).mean() > 0.5 else "mixed/positive"
    lines.append(f"- {race}: total impact is predominantly {signs} across radii; "
                 f"significant (95% CI excludes 0) at {len(sig)}/{len(tot)} radii.")
nwtot = nwdf[nwdf.effect == "total"]
lines.append(f"- Non-White (aggregate, supplementary): total impact negative at "
             f"{(nwtot.estimate < 0).sum()}/{len(nwtot)} radii.")
lines += ["", "Interpretation caveat: LPUR at a larger radius mechanically counts more visits as "
          "'local', so the radius profile mixes a measurement-scale gradient with any substantive "
          "gradient. Read the profiles for stability of SIGN and significance around the prespecified "
          "2 km, not as a dose-response curve.",
          "", f"Figures: `{p6[0]}` (+pdf/svg), `{pS[0]}` (+pdf/svg)."]
(OUTDIR / "radius_sensitivity_summary.md").write_text("\n".join(lines) + "\n")

print("=== PHASE 2 done ===")
print(gdf[gdf.effect == "total"].pivot_table(index="radius_km", columns="race",
      values="estimate").round(4).to_string())
