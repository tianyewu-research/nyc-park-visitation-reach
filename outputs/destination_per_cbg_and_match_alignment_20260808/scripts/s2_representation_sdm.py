"""SECTION C (SDM) — recompute the two park-representation robustness impacts from the CANONICAL 100 ft
crosswalk, using the final partial-SDM specification and impact procedure (identical to the frozen
`s4_robustness.py run_spec`: frozen RQ2 sample, dropna on outcome+CONT+borough, Queen with islands
dropped, design_sdm, spreg.ML_Lag spat_impacts='full', sdm_impacts with 5,000 sims, seed 42). Reports
Non-White average direct/indirect/total impacts. The primary all-POI result is left unchanged.

Also emits a replacement Figure 8 forest plot that keeps the frozen non-radius robustness rows but
swaps the two representation rows for their canonical 100 ft values (per-IQR scale, matching the
corrections figure)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUT, FIG, SEED, CB, savefig, sha256
from slib import build_sample, make_W, CONT
from slib2 import design_sdm, trace_powers, sdm_impacts
import spreg
import matplotlib.pyplot as plt

np.random.seed(SEED)
IQR = 0.541316  # fixed primary-sample Non-White IQR (per corrections)

d0, gid, checksum = build_sample()
canon = pd.read_csv(OUT / "lpur_representation_canonical_100ft_by_cbg.csv", dtype={"GEOID": str})
d0 = d0.merge(canon[["GEOID", "lpur_B_matched", "lpur_C_property_max"]], on="GEOID", how="left")


def run_rep(outcome_col, spec):
    dd = d0.dropna(subset=[outcome_col] + CONT + ["borough"]).reset_index(drop=True).copy()
    w = make_W(dd, "queen")
    if w.islands:
        dd = dd.iloc[[i for i in range(len(dd)) if i not in set(w.islands)]].reset_index(drop=True)
        w = make_W(dd, "queen")
    n = w.n
    pos = bool((dd[outcome_col] > 0).any()) and bool((dd[outcome_col] >= 0).all())
    X = design_sdm(dd, w, cont=CONT)
    m = spreg.ML_Lag(dd[outcome_col].values.reshape(-1, 1), X.values, w=w, name_x=list(X.columns),
                     spat_impacts="full")
    rho = float(m.betas.flatten()[-1])
    conv = -0.999 < rho < 0.999
    T = trace_powers(w, Q=200, m_probe=4000)
    rows, ret, pm = sdm_impacts(m, X, w, T, ["pct_nonwhite"],
                                {"pct_nonwhite": [("0->1 share", 1.0), (f"IQR ({IQR:.4f})", IQR)]},
                                nsim=5000)
    out = []
    for r in rows:
        out.append(dict(specification=spec, outcome=outcome_col, universe="canonical_100ft",
                        weights="Queen", n=n, rho=round(rho, 4), converged=conv,
                        effect=r["effect"], scale=r["scale"], estimate=r["estimate"], se=r["se"],
                        ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p=r["p"], seed=SEED, nsim=ret,
                        denom_positive=pos))
    return out, n, rho, conv


rows_all = []
for col, spec in [("lpur_B_matched", "matched municipal-park universe"),
                  ("lpur_C_property_max", "conservative property-level maximum")]:
    r, n, rho, conv = run_rep(col, spec)
    rows_all += r
    print(f"  {spec}: n={n} rho={rho:.4f} converged={conv}")
rep = pd.DataFrame(rows_all)
rep.to_csv(OUT / "representation_robustness_SDM_impacts.csv", index=False)

# ---- compare to the frozen (T1 328ft) representation impacts (raw 0->1 scale) ----
frozen = pd.read_csv(ROOT / "outputs/final_sdm_revision_20260803/robustness_SDM_master.csv")
fz = frozen[(frozen.category == "representation") & (frozen.outcome_scale == "raw")]
cmp_rows = []
for spec in ["matched municipal-park universe", "conservative property-level maximum"]:
    for eff in ["direct", "indirect", "total"]:
        old = fz[(fz.specification == spec) & (fz.effect == eff)]
        new = rep[(rep.specification == spec) & (rep.effect == eff) & (rep.scale == "0->1 share")]
        if len(old) and len(new):
            o = old.iloc[0]; nw = new.iloc[0]
            cmp_rows.append(dict(specification=spec, effect=eff,
                                 old_universe="T1 ~328ft", old_estimate=round(float(o["estimate"]), 6),
                                 old_ci=f"[{o['ci_lo']},{o['ci_hi']}]",
                                 new_universe="canonical 100ft", new_estimate=nw["estimate"],
                                 new_ci=f"[{nw['ci_lo']},{nw['ci_hi']}]",
                                 sign_same=bool(np.sign(float(o["estimate"])) == np.sign(float(nw["estimate"]))),
                                 both_sig=bool((float(o["ci_hi"]) < 0 or float(o["ci_lo"]) > 0) and
                                               (float(nw["ci_hi"]) < 0 or float(nw["ci_lo"]) > 0))))
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(OUT / "representation_old_vs_new_comparison.csv", index=False)

# ---- replacement Figure 8: frozen non-radius rows, representation rows swapped, per-IQR ----
tidy = pd.read_csv(ROOT / "outputs/final_additional_checks_20260807/corrections_20260807/"
                          "robustness_nonradius_IQR_tidy.csv")
# swap the two representation rows' estimate/se/ci with canonical per-IQR values
rep_iqr = rep[rep.scale.str.startswith("IQR")]
for spec in ["matched municipal-park universe", "conservative property-level maximum"]:
    for eff in ["direct", "indirect", "total"]:
        nw = rep_iqr[(rep_iqr.specification == spec) & (rep_iqr.effect == eff)]
        if not len(nw):
            continue
        nw = nw.iloc[0]
        mask = (tidy.specification == spec) & (tidy.effect == eff)
        tidy.loc[mask, ["estimate", "se", "ci_lo", "ci_hi", "p"]] = [nw["estimate"], nw["se"],
                                                                     nw["ci_lo"], nw["ci_hi"], nw["p"]]
tidy["universe"] = np.where(tidy.category == "representation", "canonical 100ft", "frozen")
tidy.to_csv(OUT / "figure8_replacement_tidy.csv", index=False)

specs = tidy[tidy.effect == "total"].reset_index(drop=True)
ylabels = list(specs["display_label"]); yy = np.arange(len(specs))[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
for ax, eff, title in [(axes[0], "direct", "(a) Average direct impact"),
                       (axes[1], "total", "(b) Average total impact")]:
    dsub = tidy[tidy.effect == eff].set_index("display_label").loc[ylabels].reset_index()
    for y, (_, r) in zip(yy, dsub.iterrows()):
        prim = r["group"] == "Primary"
        canon_rep = r["category"] == "representation"
        col = CB["black"] if prim else (CB["vermillion"] if canon_rep else CB["orange"])
        ax.errorbar(r["estimate"], y, xerr=[[r["estimate"] - r["ci_lo"]], [r["ci_hi"] - r["estimate"]]],
                    fmt="o", color=col, ecolor=col, capsize=2.6, ms=8 if prim else 6, lw=1.4,
                    zorder=3 if prim else 2)
    grp = specs["group"].tolist()
    for i in range(1, len(grp)):
        if grp[i] != grp[i - 1]:
            ax.axhline(len(grp) - i - 0.5, color="#D9D9D9", lw=0.8, zorder=0)
    ax.axvline(0, color=CB["grey"], lw=1.0, ls="--")
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel("Average impact on LPUR per IQR increase in Non-White share", fontsize=10)
    ax.tick_params(labelsize=9)
axes[0].set_yticks(yy); axes[0].set_yticklabels(ylabels, fontsize=9)
from matplotlib.lines import Line2D
lg = [Line2D([0], [0], marker="o", color="w", mfc=CB["black"], mec=CB["black"], label="primary", ms=7),
      Line2D([0], [0], marker="o", color="w", mfc=CB["orange"], mec=CB["orange"], label="other checks (frozen)", ms=6),
      Line2D([0], [0], marker="o", color="w", mfc=CB["vermillion"], mec=CB["vermillion"], label="representation (canonical 100 ft)", ms=6)]
fig.legend(handles=lg, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Figure 8 (replacement): non-radius robustness with canonical-100ft representation rows",
             fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
paths = savefig(fig, "figure8_replacement_robustness"); plt.close()

# caption numbers
prim = specs[specs.group == "Primary"].iloc[0]
mm = rep_iqr[(rep_iqr.specification == "matched municipal-park universe")]
cm = rep_iqr[(rep_iqr.specification == "conservative property-level maximum")]


def g(df, eff):
    r = df[df.effect == eff].iloc[0]
    return f"{r['estimate']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"


cap = (f"Figure 8 (replacement). Non-radius robustness of the Non-White-share association with LPUR under "
       f"the partial spatial Durbin model, per interquartile-range increase in Non-White share "
       f"(IQR={IQR:.4f}). All rows are the frozen non-radius specifications except the two park-"
       f"representation rows, which are recomputed on the CANONICAL 100 ft matched-park universe. "
       f"Matched municipal-park universe: direct {g(mm,'direct')}, total {g(mm,'total')}. Conservative "
       f"property-level maximum: direct {g(cm,'direct')}, total {g(cm,'total')}. Primary (all-POI) total "
       f"per IQR {prim['estimate']:+.4f} [{prim['ci_lo']:+.4f}, {prim['ci_hi']:+.4f}]. Direct impact is "
       f"primary; total is a model-implied quantity. Not causal.\n")
(OUT / "figure8_replacement_caption_numbers.txt").write_text(cap)

# provenance
man = json.loads((OUT / "provenance_manifest.json").read_text())
man["checksums"]["representation_robustness_SDM_impacts.csv"] = sha256(OUT / "representation_robustness_SDM_impacts.csv")
(OUT / "provenance_manifest.json").write_text(json.dumps(man, indent=2))

print("=== s2 done ===")
print(rep[rep.scale == "0->1 share"][["specification", "effect", "estimate", "ci_lo", "ci_hi", "p", "n"]].to_string(index=False))
print("\nold(T1 328ft) vs new(canonical 100ft):")
print(cmp.to_string(index=False))
