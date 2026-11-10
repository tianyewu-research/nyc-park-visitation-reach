#!/usr/bin/env python3
"""
31_run_unified_final_analysis.py

Unified post-audit analysis for the park access-use manuscript.

This script does not overwrite the historical tables and figures. It reads the
existing analysis inputs, applies the researcher-selected natural-log
specification, and writes a self-contained result bundle to:

    outputs/final_rerun_20260726/
    data/processed/analysis_cbg_final_log.gpkg

Main specification
------------------
Outcome:        log(winsorized annual visits per resident)
Access:         log(E2SFCA access)
RQ2 and RQ3:   single-stage ML spatial-error models with Queen weights
Local Moran:   Benjamini-Hochberg FDR correction at q=0.05

The script also runs:
  * transformation provenance checks;
  * corrected 1 km out-of-neighborhood sensitivity;
  * water-barrier and dwell-threshold sensitivities;
  * denominator-decoupled visitor-volume sensitivity;
  * low-population exclusions;
  * tract-clustered OLS, direct-CBG-income and tract-aggregated checks;
  * regenerated Tables 4-6 and Figures 5-6.

Important boundary
------------------
The park boundary and amenity inputs remain the locally available 2026
downloads. Results therefore remain provisional until matched 2023 park inputs
are supplied and the upstream park/access scripts are rerun.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import spreg
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN, Queen
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
PERMUTATIONS = 999
WINSOR_P = 0.01

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RAW_ACS = ROOT / "data" / "raw" / "ACS人口 : 收入 : 种族数据"
OUT = ROOT / "outputs" / "final_rerun_20260726"
OUT.mkdir(parents=True, exist_ok=True)

DISPLAY = {
    "log_access_final": "Log park access",
    "quality_index_cbg_svc": "Park quality",
    "walkability_c": "Walkability",
    "transit_freq": "Transit access",
    "park_area_acres": "Park area",
    "popdens_k": "Population density",
    "income_10k": "Median household income",
    "pct_nonwhite": "Share not non-Hispanic White",
    "out_park_ratio": "Out-of-neighborhood mobility",
    "log_total_pop": "Log population",
}


def winsorized_log(series: pd.Series, p: float = WINSOR_P) -> tuple[pd.Series, float, float]:
    """Natural log after two-sided winsorization; non-positive values are missing."""
    x = pd.to_numeric(series, errors="coerce").where(lambda z: z > 0)
    lo, hi = x.quantile(p), x.quantile(1 - p)
    return np.log(x.clip(lo, hi)), float(lo), float(hi)


def build_weights(d: gpd.GeoDataFrame, kind: str):
    """Return an island-free sample and row-standardized Queen or kNN weights."""
    d = d.reset_index(drop=True).copy()
    if kind == "queen":
        w = Queen.from_dataframe(d, use_index=False)
        if w.islands:
            keep = [i for i in range(len(d)) if i not in set(w.islands)]
            d = d.iloc[keep].reset_index(drop=True)
            w = Queen.from_dataframe(d, use_index=False)
    elif kind == "knn":
        w = KNN.from_dataframe(d, k=6)
    else:
        raise ValueError(f"Unknown weights: {kind}")
    w.transform = "r"
    return d, w


def make_design(
    d: gpd.GeoDataFrame,
    predictors: list[str],
    standardize: bool,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Create continuous predictors plus borough fixed effects."""
    cols = []
    scale = {}
    for col in predictors:
        if standardize:
            mu, sd = float(d[col].mean()), float(d[col].std())
            if not np.isfinite(sd) or sd == 0:
                raise ValueError(f"Cannot standardize {col}: sd={sd}")
            zcol = f"{col}_z"
            d[zcol] = (d[col] - mu) / sd
            cols.append(zcol)
            scale[col] = (mu, sd)
        else:
            cols.append(col)
    borough = pd.get_dummies(d["borough"], prefix="borough", drop_first=True).astype(float)
    return pd.concat([d[cols], borough], axis=1), scale


def fit_sem(
    frame: gpd.GeoDataFrame,
    depvar: str,
    predictors: list[str],
    *,
    standardize: bool,
    weights: str = "queen",
    label: str,
):
    """Fit one ML spatial-error model and return model objects plus a tidy table."""
    needed = [depvar, "borough", "geometry"] + predictors
    d = frame.dropna(subset=needed).copy()
    d, w = build_weights(d, weights)
    X, scale = make_design(d, predictors, standardize)
    y = d[depvar].to_numpy().reshape(-1, 1)
    model = spreg.ML_Error(y, X.to_numpy(), w=w, name_y=depvar, name_x=list(X.columns))

    beta = model.betas.flatten()
    se = np.sqrt(np.diag(model.vm))
    names = ["CONSTANT"] + list(X.columns) + ["lambda"]
    rows = []
    for i, name in enumerate(names):
        z = beta[i] / se[i] if se[i] > 0 else np.nan
        p = 2 * (1 - st.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        base = name.removesuffix("_z")
        rows.append(
            {
                "model": label,
                "term": base,
                "display": DISPLAY.get(base, name),
                "coef": beta[i],
                "std_err": se[i],
                "ci_low": beta[i] - 1.96 * se[i],
                "ci_high": beta[i] + 1.96 * se[i],
                "p_value": p,
                "n": w.n,
                "pseudo_r2": float(model.pr2),
                "weights": weights,
                "standardized_predictors": standardize,
            }
        )
    return {
        "model": model,
        "data": d,
        "weights": w,
        "X": X,
        "scale": scale,
        "table": pd.DataFrame(rows),
        "predictors": predictors,
        "depvar": depvar,
        "label": label,
    }


def transform_audit(g: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compare the code, manuscript-description and selected log/log specifications."""
    d = g.dropna(subset=["use_c_w", "access_score"]).copy()
    d = d[(d["use_c_w"] > 0) & (d["access_score"] > 0)]
    specs = {
        "historical_code_log1p_use_log_access": (
            np.log1p(d["use_c_w"]),
            np.log(d["access_score"]),
        ),
        "selected_log_use_log_access": (
            np.log(d["use_c_w"]),
            np.log(d["access_score"]),
        ),
        "incorrect_manuscript_log1p_both": (
            np.log1p(d["use_c_w"]),
            np.log1p(d["access_score"]),
        ),
    }
    rows = []
    for label, (y, x) in specs.items():
        model = sm.OLS(y, sm.add_constant(x)).fit()
        rows.append(
            {
                "specification": label,
                "n": int(model.nobs),
                "r_squared": model.rsquared,
                "access_beta": model.params.iloc[1],
                "residual_sd": model.resid.std(),
            }
        )
    return pd.DataFrame(rows)


def rq1_outputs(g: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = g.dropna(subset=["mismatch_final"]).copy()
    d, wq = build_weights(d, "queen")
    wk = KNN.from_dataframe(d, k=6)
    wk.transform = "r"
    y = d["mismatch_final"].to_numpy()

    np.random.seed(SEED)
    mq = Moran(y, wq, permutations=PERMUTATIONS)
    mk = Moran(y, wk, permutations=PERMUTATIONS)
    local = Moran_Local(y, wq, permutations=PERMUTATIONS)

    global_table = pd.DataFrame(
        [
            {
                "weights": "Queen contiguity",
                "n": wq.n,
                "morans_I": mq.I,
                "expected_I": mq.EI,
                "z_score": mq.z_sim,
                "p_value": mq.p_sim,
            },
            {
                "weights": "kNN (k=6)",
                "n": wk.n,
                "morans_I": mk.I,
                "expected_I": mk.EI,
                "z_score": mk.z_sim,
                "p_value": mk.p_sim,
            },
        ]
    )

    reject, qvals, _, _ = multipletests(local.p_sim, alpha=0.05, method="fdr_bh")
    labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    d["lisa_p_raw"] = local.p_sim
    d["lisa_q_fdr"] = qvals
    d["lisa_cluster_fdr"] = "ns"
    for code, label in labels.items():
        d.loc[reject & (local.q == code), "lisa_cluster_fdr"] = label

    counts = (
        d["lisa_cluster_fdr"]
        .value_counts()
        .rename_axis("cluster")
        .reset_index(name="n")
    )
    counts["fdr_method"] = "Benjamini-Hochberg"
    counts["q_threshold"] = 0.05
    d[["GEOID", "lisa_p_raw", "lisa_q_fdr", "lisa_cluster_fdr", "geometry"]].to_file(
        OUT / "lisa_clusters_fdr.gpkg", driver="GPKG"
    )
    return global_table, counts


def clustered_ols_from_sem(sem, label: str) -> pd.DataFrame:
    """OLS comparison with tract-clustered standard errors on the SEM sample."""
    X = sm.add_constant(sem["X"], has_constant="add")
    groups = sem["data"]["GEOID"].astype(str).str[:11]
    model = sm.OLS(sem["data"][sem["depvar"]].to_numpy(), X.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    ci = model.conf_int()
    names = ["CONSTANT"] + list(sem["X"].columns)
    rows = []
    for i, name in enumerate(names):
        base = name.removesuffix("_z")
        rows.append(
            {
                "model": label,
                "term": base,
                "display": DISPLAY.get(base, name),
                "coef": model.params[i],
                "std_err": model.bse[i],
                "ci_low": ci[i, 0],
                "ci_high": ci[i, 1],
                "p_value": model.pvalues[i],
                "n": int(model.nobs),
                "clusters": int(groups.nunique()),
            }
        )
    return pd.DataFrame(rows)


def read_direct_cbg_income() -> pd.DataFrame:
    """Read the original CBG B19013 estimate for the reliable-income subset check."""
    folders = sorted(RAW_ACS.glob("ACSDT5Y2023.B19013_20*"))
    if not folders:
        return pd.DataFrame(columns=["GEOID", "median_hh_income_cbg_direct"])
    data_file = next(folders[0].glob("*-Data.csv"))
    d = pd.read_csv(data_file, dtype=str).iloc[1:]
    d["GEOID"] = d["GEO_ID"].str.split("US").str[1]
    d["median_hh_income_cbg_direct"] = pd.to_numeric(d["B19013_001E"], errors="coerce")
    d.loc[d["median_hh_income_cbg_direct"] < 0, "median_hh_income_cbg_direct"] = np.nan
    return d[["GEOID", "median_hh_income_cbg_direct"]]


def weighted_mean(group: pd.DataFrame, value: str, weight: str = "total_pop") -> float:
    mask = group[value].notna() & group[weight].notna() & (group[weight] > 0)
    if not mask.any():
        return np.nan
    return float(np.average(group.loc[mask, value], weights=group.loc[mask, weight]))


def aggregate_to_tract(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Aggregate CBG variables to parent tracts for a tract-level SEM check."""
    d = g.copy()
    d["tract"] = d["GEOID"].astype(str).str[:11]
    geoms = d[["tract", "geometry"]].dissolve(by="tract").reset_index()
    rows = []
    for tract, s in d.groupby("tract", sort=True):
        pop = s["total_pop"].sum(min_count=1)
        visits = s["V_c"].sum(min_count=1)
        land = s["land_area_sqm"].sum(min_count=1)
        rows.append(
            {
                "tract": tract,
                "GEOID": tract,
                "total_pop": pop,
                "V_c": visits,
                "use_tract": visits / pop if pd.notna(pop) and pop > 0 else np.nan,
                "access_score": weighted_mean(s, "access_score"),
                "median_hh_income": s["median_hh_income"].dropna().iloc[0]
                if s["median_hh_income"].notna().any()
                else np.nan,
                "pct_nonwhite": weighted_mean(s, "pct_nonwhite"),
                "pop_density": pop / (land / 1e6)
                if pd.notna(pop) and pd.notna(land) and land > 0
                else np.nan,
                "park_area_acres": s["park_area_acres"].sum(min_count=1),
                "quality_index_cbg_svc": weighted_mean(s, "quality_index_cbg_svc"),
                "walkability_c": weighted_mean(s, "walkability_c"),
                "transit_freq": weighted_mean(s, "transit_freq"),
                "borough": s["borough"].mode().iloc[0] if s["borough"].notna().any() else np.nan,
            }
        )
    out = gpd.GeoDataFrame(pd.DataFrame(rows).merge(geoms, on="tract"), geometry="geometry", crs=g.crs)
    out["log_use_final"], _, _ = winsorized_log(out["use_tract"])
    out["log_access_final"] = np.log(out["access_score"].where(out["access_score"] > 0))
    out["income_10k"] = out["median_hh_income"] / 10000.0
    out["popdens_k"] = out["pop_density"] / 1000.0
    return out


def sem_prediction_figure(rq2):
    """Figure 5: average adjusted predictions from the final RQ2 SEM."""
    model = rq2["model"]
    X = rq2["X"]
    d = rq2["data"]
    names = list(X.columns)
    beta = model.betas.flatten()[: len(names) + 1]
    cov = model.vm[: len(names) + 1, : len(names) + 1]
    avg = X.mean(axis=0).to_numpy()

    panels = [
        ("income_10k", "Median household income ($10,000s)"),
        ("pct_nonwhite", "Share not non-Hispanic White"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (var, xlabel) in zip(axes, panels):
        grid = np.linspace(d[var].quantile(0.05), d[var].quantile(0.95), 80)
        idx = names.index(var)
        pred, low, high = [], [], []
        for value in grid:
            row = avg.copy()
            row[idx] = value
            design = np.r_[1.0, row]
            mu = float(design @ beta)
            se = float(np.sqrt(design @ cov @ design))
            pred.append(mu)
            low.append(mu - 1.96 * se)
            high.append(mu + 1.96 * se)
        ax.plot(grid, pred, color="#1f5a7a", linewidth=2.2)
        ax.fill_between(grid, low, high, color="#8ebad1", alpha=0.35, linewidth=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Adjusted log park use")
        ax.grid(alpha=0.2)
    fig.suptitle("Adjusted park use by neighborhood social composition")
    fig.tight_layout()
    fig.savefig(OUT / "figure5_rq2_adjusted_predictions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def coefficient_sensitivity_figure(rq3_main: pd.DataFrame, robustness: pd.DataFrame):
    """Figure 6: main RQ3 coefficients and density/income/race sensitivity."""
    main_terms = [
        "quality_index_cbg_svc",
        "walkability_c",
        "transit_freq",
        "park_area_acres",
        "popdens_k",
        "income_10k",
        "pct_nonwhite",
    ]
    main = rq3_main[rq3_main["term"].isin(main_terms)].copy()
    main["order"] = main["term"].map({v: i for i, v in enumerate(main_terms)})
    main = main.sort_values("order", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    y = np.arange(len(main))
    ax.errorbar(
        main["coef"],
        y,
        xerr=[main["coef"] - main["ci_low"], main["ci_high"] - main["coef"]],
        fmt="o",
        color="#1f5a7a",
        ecolor="#6b9bb4",
        capsize=3,
    )
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y, main["display"])
    ax.set_xlabel("Coefficient (predictors standardized)")
    ax.set_title("a. Final RQ3 spatial-error model")
    ax.grid(axis="x", alpha=0.2)

    focus_specs = [
        "Main log/log, Queen",
        "Visitor volume + log population",
        "Population >=100",
        "Population >=500",
        "Dwell: no threshold",
        "Dwell: >=21 min",
        "Water-barrier access",
    ]
    focus_terms = ["popdens_k", "income_10k", "pct_nonwhite"]
    sens = robustness[
        robustness["model"].isin(focus_specs) & robustness["term"].isin(focus_terms)
    ].copy()
    colors = {
        "popdens_k": "#b2182b",
        "income_10k": "#2166ac",
        "pct_nonwhite": "#4d9221",
    }
    ax = axes[1]
    ypos = {spec: i for i, spec in enumerate(reversed(focus_specs))}
    offsets = {"popdens_k": -0.18, "income_10k": 0.0, "pct_nonwhite": 0.18}
    for term in focus_terms:
        s = sens[sens["term"] == term]
        yy = [ypos[m] + offsets[term] for m in s["model"]]
        ax.errorbar(
            s["coef"],
            yy,
            xerr=[s["coef"] - s["ci_low"], s["ci_high"] - s["coef"]],
            fmt="o",
            color=colors[term],
            ecolor=colors[term],
            capsize=2,
            label=DISPLAY[term],
        )
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(list(ypos.values()), list(ypos.keys()))
    ax.set_xlabel("Coefficient")
    ax.set_title("b. Social-gradient sensitivity")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "figure6_rq3_coefficients_and_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    np.random.seed(SEED)
    g = gpd.read_file(PROCESSED / "analysis_cbg.gpkg")
    g["GEOID"] = g["GEOID"].astype(str).str.zfill(12)

    # Replace the historical 1,000-foot out-of-neighborhood measure with the
    # corrected 1 km result produced by the revised script 15.
    corrected_mobility = pd.read_csv(INTERIM / "out_nbhd_cbg.csv", dtype={"GEOID": str})
    g = g.drop(columns=["out_park_ratio"], errors="ignore").merge(
        corrected_mobility, on="GEOID", how="left"
    )

    # Freeze the selected transformation specification.
    g["use_c"] = g["V_c"] / g["total_pop"]
    g.loc[g["total_pop"] <= 0, "use_c"] = np.nan
    g["log_use_final"], use_lo, use_hi = winsorized_log(g["use_c"])
    g["log_access_final"] = np.log(g["access_score"].where(g["access_score"] > 0))
    first = g.dropna(subset=["log_use_final", "log_access_final"]).copy()
    fs = sm.OLS(first["log_use_final"], sm.add_constant(first["log_access_final"])).fit()
    g["mismatch_final"] = np.nan
    g.loc[first.index, "mismatch_final"] = fs.resid
    g["income_10k"] = g["median_hh_income"] / 10000.0
    g["popdens_k"] = g["pop_density"] / 1000.0
    g["log_total_pop"] = np.log(g["total_pop"].where(g["total_pop"] > 0))

    transform_audit(g).to_csv(OUT / "transform_specification_audit.csv", index=False)

    global_moran, lisa_counts = rq1_outputs(g)
    global_moran.to_csv(OUT / "table3_global_moran.csv", index=False)
    lisa_counts.to_csv(OUT / "lisa_counts_fdr.csv", index=False)

    rq2_predictors = [
        "log_access_final",
        "income_10k",
        "pct_nonwhite",
        "popdens_k",
        "park_area_acres",
    ]
    rq3_predictors = [
        "log_access_final",
        "quality_index_cbg_svc",
        "walkability_c",
        "transit_freq",
        "park_area_acres",
        "popdens_k",
        "income_10k",
        "pct_nonwhite",
    ]

    rq2 = fit_sem(
        g,
        "log_use_final",
        rq2_predictors,
        standardize=False,
        label="RQ2 main SEM",
    )
    rq3 = fit_sem(
        g,
        "log_use_final",
        rq3_predictors,
        standardize=True,
        label="Main log/log, Queen",
    )

    rq2_cluster = clustered_ols_from_sem(rq2, "RQ2 OLS, tract-clustered SE")

    # Direct CBG-income subset check.
    direct = read_direct_cbg_income()
    g_direct = g.merge(direct, on="GEOID", how="left")
    g_direct["income_10k"] = g_direct["median_hh_income_cbg_direct"] / 10000.0
    rq2_direct = fit_sem(
        g_direct,
        "log_use_final",
        rq2_predictors,
        standardize=False,
        label="RQ2 direct CBG-income subset",
    )

    # Tract-aggregated SEM check.
    tract = aggregate_to_tract(g)
    rq2_tract = fit_sem(
        tract,
        "log_use_final",
        rq2_predictors,
        standardize=False,
        label="RQ2 tract-aggregated SEM",
    )

    table4 = pd.concat(
        [rq2["table"], rq2_cluster, rq2_direct["table"], rq2_tract["table"]],
        ignore_index=True,
        sort=False,
    )
    table4.to_csv(OUT / "table4_rq2_main_and_tract_inference.csv", index=False)
    rq3["table"].to_csv(OUT / "table5_rq3_main.csv", index=False)

    # Unified RQ3 robustness.
    robustness = [rq3["table"]]

    g25 = g.copy()
    g25["log_use_final"], _, _ = winsorized_log(g25["use_c"], p=0.025)
    robustness.append(
        fit_sem(
            g25,
            "log_use_final",
            rq3_predictors,
            standardize=True,
            label="Winsor 2.5%, Queen",
        )["table"]
    )
    robustness.append(
        fit_sem(
            g,
            "log_use_final",
            rq3_predictors,
            standardize=True,
            weights="knn",
            label="Main log/log, kNN",
        )["table"]
    )
    robustness.append(
        fit_sem(
            g[g["obs_devices"] >= 20].copy(),
            "log_use_final",
            rq3_predictors,
            standardize=True,
            label="Observed devices >=20",
        )["table"]
    )

    water_path = INTERIM / "access_water_barrier.csv"
    if water_path.exists():
        water = pd.read_csv(water_path, dtype={"GEOID": str})
        gw = g.merge(water, on="GEOID", how="left")
        gw["log_access_final"] = np.log(
            gw["access_score_nowater"].where(gw["access_score_nowater"] > 0)
        )
        robustness.append(
            fit_sem(
                gw,
                "log_use_final",
                rq3_predictors,
                standardize=True,
                label="Water-barrier access",
            )["table"]
        )

    dwell_path = INTERIM / "use_dwell_variants.csv"
    if dwell_path.exists():
        dwell = pd.read_csv(dwell_path, dtype={"GEOID": str})
        for col, label in [
            ("V_none", "Dwell: no threshold"),
            ("V_ge5", "Dwell: >=5 min"),
            ("V_ge21", "Dwell: >=21 min"),
        ]:
            gd = g.merge(dwell[["GEOID", col]], on="GEOID", how="left")
            gd["_use_variant"] = gd[col] / gd["total_pop"]
            gd.loc[gd["total_pop"] <= 0, "_use_variant"] = np.nan
            gd["log_use_final"], _, _ = winsorized_log(gd["_use_variant"])
            robustness.append(
                fit_sem(
                    gd,
                    "log_use_final",
                    rq3_predictors,
                    standardize=True,
                    label=label,
                )["table"]
            )

    # Denominator-decoupled sensitivity: model visit volume and adjust for log population.
    gv = g.copy()
    gv["log_visit_volume"], _, _ = winsorized_log(gv["V_c"])
    volume_predictors = rq3_predictors + ["log_total_pop"]
    robustness.append(
        fit_sem(
            gv,
            "log_visit_volume",
            volume_predictors,
            standardize=True,
            label="Visitor volume + log population",
        )["table"]
    )
    for threshold in [100, 500]:
        robustness.append(
            fit_sem(
                g[g["total_pop"] >= threshold].copy(),
                "log_use_final",
                rq3_predictors,
                standardize=True,
                label=f"Population >={threshold}",
            )["table"]
        )

    rq3_with_mobility = rq3_predictors + ["out_park_ratio"]
    robustness.append(
        fit_sem(
            g,
            "log_use_final",
            rq3_with_mobility,
            standardize=True,
            label="Corrected 1 km mobility included",
        )["table"]
    )

    table6 = pd.concat(robustness, ignore_index=True)
    table6.to_csv(OUT / "table6_rq3_unified_robustness.csv", index=False)

    sem_prediction_figure(rq2)
    coefficient_sensitivity_figure(rq3["table"], table6)

    # Explicit sample-loss chain.
    rq1_n = int(global_moran.loc[global_moran["weights"] == "Queen contiguity", "n"].iloc[0])
    sample_flow = pd.DataFrame(
        [
            {"stage": "All NYC CBGs", "n": len(g), "loss_from_previous": 0},
            {
                "stage": "Positive-population CBGs with use and access",
                "n": len(first),
                "loss_from_previous": len(g) - len(first),
            },
            {
                "stage": "RQ1 Queen sample after islands",
                "n": rq1_n,
                "loss_from_previous": len(first) - rq1_n,
            },
            {
                "stage": "RQ2 complete-case Queen sample",
                "n": int(rq2["weights"].n),
                "loss_from_previous": rq1_n - int(rq2["weights"].n),
            },
            {
                "stage": "RQ3 complete-case Queen sample",
                "n": int(rq3["weights"].n),
                "loss_from_previous": int(rq2["weights"].n) - int(rq3["weights"].n),
            },
        ]
    )
    sample_flow.to_csv(OUT / "sample_flow.csv", index=False)

    metadata = {
        "created_by": "src/31_run_unified_final_analysis.py",
        "random_seed": SEED,
        "permutations": PERMUTATIONS,
        "main_specification": {
            "outcome": "natural log of 1% winsorized annual visits per resident",
            "access": "natural log of E2SFCA access",
            "rq2_rq3_model": "single-stage ML spatial error with Queen contiguity",
            "local_moran_multiple_testing": "Benjamini-Hochberg FDR q=0.05",
        },
        "winsor_limits_use_c": {"lower": use_lo, "upper": use_hi},
        "first_stage": {
            "n": int(fs.nobs),
            "r_squared": float(fs.rsquared),
            "access_beta": float(fs.params["log_access_final"]),
        },
        "known_blocker": (
            "Park boundaries and amenity inputs are 2026 downloads. "
            "Matched 2023 park inputs are not present locally."
        ),
        "corrected_distance": {
            "crs": "EPSG:2263 (US survey feet)",
            "one_kilometre_feet": 1000 / 0.3048,
        },
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))

    keep = [
        "GEOID",
        "use_c",
        "log_use_final",
        "log_access_final",
        "mismatch_final",
        "geometry",
    ]
    g[keep].to_file(PROCESSED / "analysis_cbg_final_log.gpkg", driver="GPKG")

    print("\nUnified final-analysis bundle written to:", OUT)
    print("First stage:", metadata["first_stage"])
    print("\nSample flow:")
    print(sample_flow.to_string(index=False))
    print("\nFDR LISA counts:")
    print(lisa_counts.to_string(index=False))
    print("\nRQ2 income rows:")
    print(
        table4[table4["term"].eq("income_10k")][
            ["model", "coef", "std_err", "p_value", "n"]
        ].to_string(index=False)
    )
    print("\nRQ3 main:")
    print(
        rq3["table"][
            rq3["table"]["term"].isin(
                ["popdens_k", "income_10k", "pct_nonwhite", "quality_index_cbg_svc"]
            )
        ][["display", "coef", "ci_low", "ci_high", "p_value", "n"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
