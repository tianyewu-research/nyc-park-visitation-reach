"""Part 4 — 26-specification robustness under the partial SDM. For each estimable spec, the Non-White
average DIRECT/INDIRECT/TOTAL impacts with 95% CI and p. Raw-LPUR and logit outcomes kept on separate
scales. Failures (rank/convergence/inadmissible rho) reported transparently, never silently SEM."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
sys.path.insert(0, "outputs/spatial_model_reassessment_20260803/scripts")
from slib import build_sample, make_W, CONT, ROOT
from slib2 import design_sdm, trace_powers, sdm_impacts, OUT
import spreg
np.random.seed(42)

d0, gid, checksum = build_sample()
# merge measure columns (current pipeline)
dwell = pd.read_csv(ROOT / "data/interim/lpur_robust.csv", dtype={"GEOID": str})
prop = pd.read_csv(ROOT / "outputs/lpur_revision_calculations_20260730/T1_lpur_versions.csv", dtype={"GEOID": str})
supply = pd.read_csv(ROOT / "data/interim/park_area_2km.csv", dtype={"GEOID": str})
d0 = d0.merge(dwell, on="GEOID", how="left").merge(prop, on="GEOID", how="left").merge(supply, on="GEOID", how="left")
d0["wk_visitors"] = d0["total_visits"] / 52.0
CLIP = 0.005
for c in [c for c in d0.columns if c.startswith("lpur_")]:
    pass
Tcache = {}


def get_T(w, key):
    if key not in Tcache:
        Tcache[key] = trace_powers(w, Q=200, m_probe=4000)
    return Tcache[key]


def run_spec(category, spec, outcome_col, weights="queen", mask=None, cont=CONT, outcome_scale="raw",
             logit=False):
    dd = d0 if mask is None else d0[mask]
    dep = outcome_col
    dd = dd.dropna(subset=[dep] + cont + ["borough"]).reset_index(drop=True).copy()
    if logit:
        lp = dd[dep].clip(CLIP, 1 - CLIP); dd["_logit"] = np.log(lp / (1 - lp)); dep = "_logit"
    w = make_W(dd, "queen" if weights == "queen" else "knn")
    if w.islands:
        dd = dd.iloc[[i for i in range(len(dd)) if i not in set(w.islands)]].reset_index(drop=True)
        w = make_W(dd, "queen" if weights == "queen" else "knn")
    n = w.n
    try:
        X = design_sdm(dd, w, cont=cont)
        m = spreg.ML_Lag(dd[dep].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns), spat_impacts="full")
        rho = float(m.betas.flatten()[-1])
        if not (-0.999 < rho < 0.999):
            raise ValueError(f"inadmissible rho={rho:.3f}")
        T = get_T(w, f"{weights}_{n}")
        rows, ret, pmult = sdm_impacts(m, X, w, T, ["pct_nonwhite"],
                                       {"pct_nonwhite": [("0->1 share", 1.0)]}, nsim=5000)
        out = []
        for r in rows:
            out.append(dict(category=category, specification=spec, weights=weights, outcome_scale=outcome_scale,
                            n=n, rho=round(rho, 4), effect=r["effect"], estimate=r["estimate"],
                            se=r["se"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p=r["p"],
                            beta_struct=round(pmult["pct_nonwhite"]["beta"], 5),
                            theta_struct=round(pmult["pct_nonwhite"]["theta"], 5),
                            status="OK", nsim_retained=ret))
        return out
    except Exception as ex:
        return [dict(category=category, specification=spec, weights=weights, outcome_scale=outcome_scale,
                     n=n, effect=e, estimate=None, status=f"FAILED: {ex}") for e in ("direct", "indirect", "total")]


ROWS = []
# radius 0.5-3.0
for rk in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
    ROWS += run_spec("radius", f"radius {rk} km" + (" (primary)" if rk == 2.0 else ""), f"lpur_{rk}km")
# dwell
ROWS += run_spec("dwell", "no dwell filter", "lpur_none")
ROWS += run_spec("dwell", "dwell >=5 min (primary)", "lpur_ge5")
ROWS += run_spec("dwell", "dwell >=21 min", "lpur_ge21")
# sample
ROWS += run_spec("sample", "no minimum volume (primary)", "lpur_2.0km")
ROWS += run_spec("sample", "weekly visitors >=5", "lpur_2.0km", mask=(d0.wk_visitors >= 5))
ROWS += run_spec("sample", "weekly visitors >=20", "lpur_2.0km", mask=(d0.wk_visitors >= 20))
# weights
ROWS += run_spec("weights", "Queen (primary)", "lpur_2.0km", weights="queen")
ROWS += run_spec("weights", "kNN6", "lpur_2.0km", weights="knn")
# local supply control (add park_area_2km to continuous set)
ROWS += run_spec("supply", "baseline (primary)", "lpur_2.0km")
ROWS += run_spec("supply", "+ park acreage within 2 km", "lpur_2.0km", cont=CONT + ["park_area_2km"])
# park representation
ROWS += run_spec("representation", "original POI sum (primary)", "lpur_A_poisum")
ROWS += run_spec("representation", "matched municipal-park universe", "lpur_B_matched")
ROWS += run_spec("representation", "conservative property-level maximum", "lpur_C_property_max")
# transform
ROWS += run_spec("transform", "raw LPUR (primary)", "lpur_2.0km")
ROWS += run_spec("transform", "clipped logit LPUR", "lpur_2.0km", outcome_scale="logit", logit=True)

M = pd.DataFrame(ROWS)
M.to_csv(OUT / "robustness_SDM_master.csv", index=False)

ok = M[M.status == "OK"]
raw_tot = ok[(ok.effect == "total") & (ok.outcome_scale == "raw")]
raw_dir = ok[(ok.effect == "direct") & (ok.outcome_scale == "raw")]
neg_tot = int((raw_tot.estimate < 0).sum()); neg_dir = int((raw_dir.estimate < 0).sum())
print("=== s4 robustness under SDM done ===")
print(f"raw-scale specs: {len(raw_tot)} | negative TOTAL impact: {neg_tot} | negative DIRECT impact: {neg_dir}")
print(f"failed specs: {int((M.status!='OK').sum())//3}")
print(raw_tot[["category", "specification", "weights", "estimate", "ci_lo", "ci_hi", "p"]].to_string(index=False))
