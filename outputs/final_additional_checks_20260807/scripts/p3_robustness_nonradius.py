"""PHASE 3 — reorganize the non-radius robustness figure.

Consumes the already-validated robustness_SDM_master.csv (no refit). Removes all radius rows (the new
Fig 6 covers radius), keeps the primary model exactly once, and groups the remaining checks by dwell
rule / sample threshold / spatial weights / local supply / park representation. Direct and total
impacts go in separate panels with 95% CIs. The clipped-logit result is on a different scale and is
kept in a SEPARATE supplementary panel/table, never mixed with the raw-LPUR estimates."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUTDIR, CB, savefig, FINAL
import matplotlib.pyplot as plt

M = pd.read_csv(FINAL / "robustness_SDM_master.csv")
raw = M[(M.status == "OK") & (M.outcome_scale == "raw")].copy()
logit = M[(M.status == "OK") & (M.outcome_scale == "logit")].copy()

GROUPS = ["dwell", "sample", "weights", "supply", "representation"]
GLABEL = {"dwell": "Dwell rule", "sample": "Sample threshold", "weights": "Spatial weights",
          "supply": "Local supply", "representation": "Park representation"}

# primary once (it appears tagged '(primary)' inside every family); take it from radius category
primary = raw[(raw.category == "radius") & raw.specification.str.contains("primary")].copy()
primary["group"] = "Primary"; primary["display_label"] = "Primary specification"
# non-radius, non-primary alternatives
alt = raw[raw.category.isin(GROUPS) & ~raw.specification.str.contains("primary")].copy()
alt["group"] = alt.category.map(GLABEL)
alt["display_label"] = alt.apply(lambda r: f"{GLABEL[r['category']]}: {r['specification']}", axis=1)

tidy = pd.concat([primary, alt], ignore_index=True)
# order: primary first, then by group
group_order = ["Primary"] + [GLABEL[g] for g in GROUPS]
tidy["group_rank"] = tidy["group"].map({g: i for i, g in enumerate(group_order)})
tidy = tidy.sort_values(["group_rank", "specification"]).reset_index(drop=True)
tidy.to_csv(OUTDIR / "robustness_nonradius_tidy.csv", index=False)

# ---- Fig7: direct + total panels ----
specs = tidy[tidy.effect == "total"].reset_index(drop=True)   # one row per spec for labels
ylabels = list(specs["display_label"])
yy = np.arange(len(specs))[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
for ax, eff, title in [(axes[0], "direct", "(a) Average direct impact"),
                       (axes[1], "total", "(b) Average total impact")]:
    dsub = tidy[tidy.effect == eff].reset_index(drop=True)
    # keep spec order aligned with 'specs'
    dsub = dsub.set_index("display_label").loc[ylabels].reset_index()
    for y, (_, r) in zip(yy, dsub.iterrows()):
        prim = r["group"] == "Primary"
        col = CB["black"] if prim else CB["orange"]
        ax.errorbar(r["estimate"], y, xerr=[[r["estimate"] - r["ci_lo"]], [r["ci_hi"] - r["estimate"]]],
                    fmt="o", color=col, ecolor=col, capsize=2.6, ms=8 if prim else 6,
                    lw=1.4, zorder=3 if prim else 2)
    # group separators
    grp = specs["group"].tolist()
    for i in range(1, len(grp)):
        if grp[i] != grp[i - 1]:
            ax.axhline(len(grp) - i - 0.5, color="#D9D9D9", lw=0.8, zorder=0)
    ax.axvline(0, color=CB["vermillion"], lw=1.0, ls="--")
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel("Non-White share average impact on raw LPUR (0→1)", fontsize=10)
    ax.tick_params(labelsize=9)
axes[0].set_yticks(yy); axes[0].set_yticklabels(ylabels, fontsize=9)
fig.suptitle("Robustness of the Non-White-share LPUR association (non-radius checks, partial SDM)",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
p7 = savefig(fig, "Fig7_grouped_nonradius_robustness"); plt.close()

# ---- FigS: clipped-logit (separate scale) ----
logit_tidy = logit[logit.effect.isin(["direct", "indirect", "total"])][
    ["effect", "estimate", "se", "ci_lo", "ci_hi", "p"]].copy()
logit_tidy.insert(0, "specification", "clipped-logit LPUR (winsorized to [0.005,0.995])")
logit_tidy.to_csv(OUTDIR / "FigS_clipped_logit.csv", index=False)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
lt = logit_tidy.set_index("effect")
effs = ["direct", "total"]
yy2 = np.arange(len(effs))[::-1]
for y, eff in zip(yy2, effs):
    r = lt.loc[eff]
    ax.errorbar(r["estimate"], y, xerr=[[r["estimate"] - r["ci_lo"]], [r["ci_hi"] - r["estimate"]]],
                fmt="s", color=CB["purple"], ecolor=CB["purple"], capsize=3, ms=7, lw=1.4)
ax.axvline(0, color=CB["vermillion"], lw=1.0, ls="--")
ax.set_yticks(np.arange(len(effs))); ax.set_yticklabels(effs[::-1], fontsize=9)
ax.set_xlabel("Non-White share average impact on clipped-LOGIT LPUR (logit units — not comparable "
              "to raw scale)", fontsize=8.5)
ax.set_title("Supplementary: clipped-logit outcome (separate scale)", fontsize=10, loc="left")
plt.tight_layout()
pSl = savefig(fig, "FigS_clipped_logit"); plt.close()

print("=== PHASE 3 done ===")
print(f"non-radius specs (per effect): {len(specs)} (1 primary + {len(specs)-1} alternatives)")
print(specs[["group", "specification"]].to_string(index=False))
print(f"\nFig7: {p7[0]}  | FigS_clipped_logit: {pSl[0]} + CSV")
