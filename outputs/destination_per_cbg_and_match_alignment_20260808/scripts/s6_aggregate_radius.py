"""SECTION H — supply the missing aggregate Non-White radius evidence.

Using the final partial-SDM pipeline (frozen RQ2 covariates + continuous-covariate spatial lags, Queen
weights, seed 42, 5,000 sims), compute the AGGREGATE Non-White-share average direct and total impacts
for LPUR at 0.5-3.0 km, on one fixed sample. These are correlated SCALE specifications of the same
measure, not independent tests or a dose-response function."""
import sys, numpy as np, pandas as pd, geopandas as gpd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUT, SEED, CB, GPKG, savefig
from slib2 import trace_powers, sdm_impacts, boro_dummies
import spreg
from libpysal.weights import Queen
import matplotlib.pyplot as plt

np.random.seed(SEED)
RADII = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
RADCOLS = [f"lpur_{r}km" for r in RADII]
CONT = ["income_10k", "pct_nonwhite", "popdens_k", "park_area_acres"]
PRIMARY = 2.0

g = gpd.read_file(GPKG).to_crs("EPSG:2263")
g["income_10k"] = g["median_hh_income"] / 10000.0
g["popdens_k"] = g["pop_density"] / 1000.0
# one fixed sample: complete-case on covariates + every radius outcome
d = g.dropna(subset=list(set(CONT + RADCOLS + ["borough"]))).reset_index(drop=True)
w = Queen.from_dataframe(d, use_index=False)
if w.islands:
    d = d.iloc[[i for i in range(len(d)) if i not in set(w.islands)]].reset_index(drop=True)
    w = Queen.from_dataframe(d, use_index=False)
w.transform = "r"
n = len(d)
T = trace_powers(w, Q=200, m_probe=4000)
iqr_nw = float(d["pct_nonwhite"].quantile(.75) - d["pct_nonwhite"].quantile(.25))


def design(cont):
    base = pd.concat([d[cont].reset_index(drop=True), boro_dummies(d).reset_index(drop=True)], axis=1)
    WX = pd.DataFrame(w.sparse @ d[cont].values, columns=["W_" + c for c in cont])
    return pd.concat([base, WX], axis=1)


rows = []
for r in RADII:
    dep = f"lpur_{r}km"
    X = design(CONT)
    m = spreg.ML_Lag(d[dep].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns),
                     spat_impacts="full")
    rho = float(m.betas.flatten()[-1])
    scales = {c: [("unit", 1.0)] for c in CONT}
    scales["pct_nonwhite"] = [("IQR", iqr_nw)]
    imp, ret, pm = sdm_impacts(m, X, w, T, CONT, scales, nsim=5000)
    for row in imp:
        if row["variable"] == "pct_nonwhite" and row["effect"] in ("direct", "total"):
            rows.append(dict(radius_km=r, effect=row["effect"], estimate=float(row["estimate"]),
                             ci_lo=float(row["ci_lo"]), ci_hi=float(row["ci_hi"]), p=row["p"],
                             rho=round(rho, 4), n=n, converged=(-0.999 < rho < 0.999),
                             scale=f"per IQR ({iqr_nw:.4f})", seed=SEED, nsim=ret, is_primary=(r == PRIMARY)))
    print(f"  radius {r} km done (rho={rho:.4f})")
agg = pd.DataFrame(rows)
agg.to_csv(OUT / "aggregate_nonwhite_radius_SDM_impacts.csv", index=False)

# ---- compact panel ----
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
for ax, eff, title in [(axes[0], "direct", "(a) Average direct impact"),
                       (axes[1], "total", "(b) Average total impact")]:
    sub = agg[agg.effect == eff].sort_values("radius_km")
    ax.errorbar(sub.radius_km, sub.estimate, yerr=[sub.estimate - sub.ci_lo, sub.ci_hi - sub.estimate],
                fmt="o-", color=CB["vermillion"], ecolor=CB["vermillion"], capsize=2.5, ms=4.5, lw=1.3)
    ax.axhline(0, color=CB["grey"], lw=0.9, ls="--")
    ax.axvline(PRIMARY, color=CB["black"], lw=0.9, ls=":")
    ax.set_title(title, fontsize=10, loc="left"); ax.set_xlabel("LPUR catchment radius (km)", fontsize=9)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("Aggregate Non-White impact (per IQR)", fontsize=8.5)
axes[0].text(PRIMARY, axes[0].get_ylim()[0], " 2 km (primary)", fontsize=7, ha="left", va="bottom", color=CB["black"])
fig.suptitle("Aggregate Non-White-share average impacts on LPUR across catchment radii "
             "(correlated scale specifications)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.94])
paths = savefig(fig, "aggregate_nonwhite_radius_panel"); plt.close()

d2 = agg[(agg.effect == "direct")].sort_values("radius_km")
tot = agg[(agg.effect == "total")].sort_values("radius_km")
dir_neg = int((d2.ci_hi < 0).sum()); tot_neg = int((tot.ci_hi < 0).sum())
at2 = agg[(agg.radius_km == 2.0)]
cap = (f"Aggregate Non-White-share average impacts on LPUR across 0.5-3.0 km catchment radii (per IQR, "
       f"IQR={iqr_nw:.4f}; partial SDM, Queen weights, one fixed sample n={n}). The average DIRECT impact "
       f"is negative with a 95% CI excluding zero at {dir_neg}/{len(d2)} radii; the average TOTAL impact "
       f"at {tot_neg}/{len(tot)}. At the prespecified 2 km: direct "
       f"{at2[at2.effect=='direct']['estimate'].iloc[0]:+.4f}, total "
       f"{at2[at2.effect=='total']['estimate'].iloc[0]:+.4f}. These are CORRELATED SCALE SPECIFICATIONS "
       f"of one measure, not independent tests or a dose-response function; a larger catchment mechanically "
       f"classifies more visits as local. Model-implied ecological associations, not causal effects.\n")
(OUT / "aggregate_radius_caption_numbers.txt").write_text(cap)

print("=== s6 done ===")
print(f"aggregate Non-White DIRECT neg&sig at {dir_neg}/11 radii; TOTAL at {tot_neg}/11 radii")
print(agg[agg.effect == "direct"][["radius_km", "estimate", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
