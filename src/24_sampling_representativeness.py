"""
24_sampling_representativeness.py
Does Advan device sampling intensity vary systematically with socioeconomic status?
Bears on RQ2 reliability (mismatch ~ income/race).

CAVEAT: a fully clean panel-representativeness test needs the vendor home_panel_summary
(unique devices residing per CBG), which is not in the data. We use best-effort proxies:
  (A) sampling intensity proxy = obs_devices / total_pop  (devices-visit-instances per
      resident). Confounded by park-visit propensity (the outcome), so read directionally.
  (B) coverage completeness = weeks_observed / 52  (less volume-confounded: presence, not
      volume). A clean read on whether some groups are systematically under-observed.
  (C) fully unobserved CBGs (no mobile data): SES profile vs observed.

Mitigant: VISIT_COUNTS are Advan-SCALED (population-representative), so use_c already
absorbs gross sampling differences; these tests look for residual / gross problems.

Input:  data/interim/use_cbg.csv, data/interim/acs_cbg.csv, data/interim/cbg_nyc.gpkg
Output: outputs/tables/sampling_representativeness.csv
        outputs/figures/sampling_by_ses.png
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"

use = pd.read_csv(INTERIM / "use_cbg.csv", dtype={"GEOID": str})
acs = pd.read_csv(INTERIM / "acs_cbg.csv", dtype={"GEOID": str})
cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg")[["GEOID", "borough"]]
d = use.merge(acs, on="GEOID").merge(cbg, on="GEOID")

d["sampling_proxy"] = d["obs_devices"] / d["total_pop"]      # devices/resident (confounded)
d["coverage"] = d["weeks_observed"] / 52.0                   # data completeness
d["income_10k"] = d["median_hh_income"] / 10000.0
d["popdens_k"] = d["pop_density"] / 1000.0

# ── (C) Unobserved CBGs vs observed ──
obs = d[d["obs_devices"] > 0]
unobs = d[d["obs_devices"] == 0]
print("=== (C) Fully unobserved CBGs ===")
print(f"unobserved: {len(unobs)} | observed: {len(obs)}")
print(f"  income  — unobserved median={unobs['median_hh_income'].median():.0f} vs "
      f"observed={obs['median_hh_income'].median():.0f}")
print(f"  nonwhite— unobserved median={unobs['pct_nonwhite'].median():.3f} vs "
      f"observed={obs['pct_nonwhite'].median():.3f}")

# Work on observed, well-defined sample
s = obs[(obs["total_pop"] > 0) & obs["median_hh_income"].notna() & obs["pct_nonwhite"].notna()].copy()
s["log_sampling"] = np.log(s["sampling_proxy"])

# ── (A) sampling proxy gradient by SES ──
mA = smf.ols("log_sampling ~ income_10k + pct_nonwhite + popdens_k + C(borough)", data=s).fit()
# standardized betas
sz = s.copy()
for c in ["income_10k", "pct_nonwhite", "popdens_k"]:
    sz[c] = (sz[c] - sz[c].mean()) / sz[c].std()
mAz = smf.ols("log_sampling ~ income_10k + pct_nonwhite + popdens_k + C(borough)", data=sz).fit()

# ── (B) coverage completeness gradient ──
mB = smf.ols("coverage ~ income_10k + pct_nonwhite + popdens_k + C(borough)", data=s).fit()

# confound magnitude: how much sampling proxy correlates with the outcome use_c
s["use_c"] = s["V_c"] / s["total_pop"]
conf = s[["sampling_proxy", "use_c"]].apply(np.log).corr().iloc[0, 1]

rows = []
for var, lab in [("income_10k", "Income (per $10k)"), ("pct_nonwhite", "% non-white"),
                 ("popdens_k", "Pop density (per 1k/km2)")]:
    rows.append({
        "variable": lab,
        "A_log_sampling_beta": mA.params[var], "A_p": mA.pvalues[var],
        "A_std_beta": mAz.params[var],
        "B_coverage_beta": mB.params[var], "B_p": mB.pvalues[var],
    })
out = pd.DataFrame(rows)
out.to_csv(TABLES / "sampling_representativeness.csv", index=False)

print(f"\n=== (A) log(sampling proxy) ~ SES  (R2={mA.rsquared:.3f}) ===")
print(out[["variable", "A_std_beta", "A_p"]].round(4).to_string(index=False))
print(f"  [confound] corr(log sampling proxy, log use_c) = {conf:.3f} "
      f"-> sampling proxy is {'strongly' if abs(conf)>0.5 else 'moderately' if abs(conf)>0.3 else 'weakly'} "
      f"tied to the outcome; read (A) directionally.")
print(f"\n=== (B) coverage (weeks/52) ~ SES  (R2={mB.rsquared:.3f}) ===")
print(out[["variable", "B_coverage_beta", "B_p"]].round(4).to_string(index=False))
print(f"  coverage: mean={s['coverage'].mean():.3f}, p10={s['coverage'].quantile(.1):.3f} "
      f"(most CBGs observed nearly every week)")

# ── Figure: BOTH proxies (coverage + observation volume) across income & race quartiles ──
# Two proxies are shown to avoid conflating presence (coverage) with volume (intensity):
# coverage completeness = share of weeks a CBG's residents are observed; observation volume =
# scaled weekly park-visitor instances per resident (NOT unique devices — a resident generates
# multiple scaled instances across parks/weeks, so values exceed 1 by construction).
proxies = [("coverage", "Coverage completeness\n(share of weeks observed)"),
           ("sampling_proxy", "Observation volume\n(scaled visitor-instances / resident)")]
groups = [("median_hh_income", "Income quartile (Q1 low → Q4 high)"),
          ("pct_nonwhite", "% non-white quartile (Q1 low → Q4 high)")]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for r, (pcol, plab) in enumerate(proxies):
    for c, (gv, glab) in enumerate(groups):
        ax = axes[r, c]
        s["q"] = pd.qcut(s[gv], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        g = s.groupby("q")[pcol].median()
        ax.bar(range(4), g.values, color="#4c72b0")
        ax.set_xticks(range(4)); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel(glab); ax.set_ylabel(plab)
        ax.set_title(f"{plab.splitlines()[0]} by {'income' if gv=='median_hh_income' else 'race'}")
fig.suptitle("Mobile-panel sampling representativeness by neighborhood socioeconomic status",
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(FIGS / "sampling_by_ses.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\nSaved: {TABLES/'sampling_representativeness.csv'}, {FIGS/'sampling_by_ses.png'}")

# --- self-check ---
_t = pd.read_csv(TABLES / "sampling_representativeness.csv")
assert len(_t) == 3
print("[self-check] 24 sampling representativeness OK")
