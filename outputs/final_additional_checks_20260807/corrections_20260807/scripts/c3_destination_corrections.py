"""CORRECTION TASK 3 — destination reporting + Figure 9 correction.

Reuses the cached property-level records (the conservative property-week-home-CBG MAXIMUM construction
is unchanged). Corrects the REPORTING and the figure:
 - 1,051,858 are annual property-home-CBG CELLS, not visits;
 - report both the cell share >2 km and the attributed-weight share >2 km;
 - recover match coverage (candidate POIs, matched within 100 ft, represented properties, weight retained);
 - report origin-CBG n per quartile + exact cut points; add top-5 share;
 - neutral size labels (no 'flagship');
 - 1,000-rep CBG-clustered descriptive bootstrap (seed 42);
 - redesign Figure 9 on NON-LOCAL destinations only (no dual axes); all-destination / waterfront /
   distance / named destinations go to supplementary tables."""
import sys, json, glob, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "outputs/spatial_model_reassessment_20260803/scripts")
from slib import build_sample
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
ROOT = Path(".")
BASE = ROOT / "outputs" / "final_additional_checks_20260807"
OUT = BASE / "corrections_20260807"; FIG = OUT / "figures"
RECORDS = BASE / "destination_records_property_level.csv"
CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00", skyblue="#56B4E9",
          grey="#999999", black="#000000", purple="#CC79A7")
NYC = ("36005", "36047", "36061", "36081", "36085"); CRS = "EPSG:2263"; MATCH_FT = 100.0
QORDER = ["Q1 (lowest Non-White)", "Q2", "Q3", "Q4 (highest Non-White)"]
NBOOT = 1000

R = pd.read_csv(RECORDS, dtype={"home_cbg": str, "pid": str})
R["is_waterfront"] = R["dest_waterfront"].astype(str).str.lower().isin(["yes", "true", "1", "1.0"])

# ---- neutral size labels ----
def size_class(a):
    if a < 1:
        return "1 very small (<1 ac)"
    if a < 5:
        return "2 small (1-5 ac)"
    if a < 20:
        return "3 medium (5-20 ac)"
    if a < 100:
        return "4 large (20-100 ac)"
    return "5 very large (>=100 ac)"


R["size_class"] = R["dest_acres"].map(size_class)

# ---- origin quartile CBG counts + exact cut points ----
d, gid, checksum = build_sample()
qs = d["pct_nonwhite"].quantile([.25, .5, .75]).values
cutpoints = dict(q25=round(float(qs[0]), 6), q50=round(float(qs[1]), 6), q75=round(float(qs[2]), 6))
oc = R.groupby("origin_quartile")["home_cbg"].nunique().reindex(QORDER)
qcounts = pd.DataFrame({"origin_quartile": QORDER, "n_origin_cbg": oc.values,
                        "nonwhite_q25": cutpoints["q25"], "nonwhite_q50": cutpoints["q50"],
                        "nonwhite_q75": cutpoints["q75"]})
qcounts.to_csv(OUT / "destination_origin_quartile_counts.csv", index=False)

# ---- overall reporting facts: cell share and weight share > 2 km ----
cell_share_gt2 = float(R["non_local"].mean())
weight_share_gt2 = float(R.loc[R["non_local"], "attributed_visit_records"].sum() /
                         R["attributed_visit_records"].sum())
n_cells = len(R); n_properties = R["pid"].nunique()

# ---- match coverage: light Pass-1 (POI coords) + sjoin to properties (candidate/matched/%) ----
import geopandas as gpd
parks = gpd.read_file(ROOT / "data" / "raw" / "NYC公园边界" / "Parks Properties_20260623").to_crs(CRS)
parks = parks[parks["acres"] >= 0.1].copy(); parks["pid"] = parks["gispropnum"].astype(str)
prop = parks.dissolve(by="pid").reset_index()
files = sorted(glob.glob(str(ROOT / "data" / "raw" / "nypark_2325" / "2023-*.csv.gz")))
poi_xy = {}
for f in files:
    df = pd.read_csv(f, usecols=["ID_STORE", "POI_CBG", "LATITUDE", "LONGITUDE"])
    df["pc"] = df["POI_CBG"].dropna().astype("Int64").astype(str).str.zfill(12)
    df = df[df["pc"].str[:5].isin(NYC)].dropna(subset=["LATITUDE", "LONGITUDE"])
    for sid, la, lo in zip(df["ID_STORE"], df["LATITUDE"], df["LONGITUDE"]):
        poi_xy.setdefault(sid, (la, lo))
sids = list(poi_xy)
pts = gpd.GeoDataFrame({"ID_STORE": sids},
                       geometry=gpd.points_from_xy([poi_xy[s][1] for s in sids],
                                                   [poi_xy[s][0] for s in sids]),
                       crs="EPSG:4326").to_crs(CRS)
nn = gpd.sjoin_nearest(pts, prop[["pid", "geometry"]], distance_col="dist").drop_duplicates("ID_STORE")
n_candidate = len(sids)
n_matched = int((nn["dist"] <= MATCH_FT).sum())

# weight-retained proxy from the interim use pipeline (sum construction; documented)
try:
    ua = pd.read_csv(ROOT / "data" / "interim" / "use_cbg.csv")          # all POIs, sum
    um = pd.read_csv(ROOT / "data" / "interim" / "use_cbg_matched.csv")  # matched properties, sum
    ua["GEOID"] = ua["GEOID"].astype(str).str.zfill(12)
    um["GEOID"] = um["GEOID"].astype(str).str.zfill(12)
    ins = set(d["GEOID"].astype(str))
    ua_in = ua[ua["GEOID"].isin(ins)]["V_c"].sum()
    um_in = um[um["GEOID"].isin(ins)]["V_c_matched"].sum()
    weight_retained_proxy = round(float(um_in / ua_in), 4)
except Exception:
    weight_retained_proxy = np.nan

cov = pd.DataFrame([
    dict(quantity="candidate park POIs (NYC, any week)", value=n_candidate),
    dict(quantity="POIs matched to a municipal park property within 100 ft", value=n_matched),
    dict(quantity="match rate (%)", value=round(100 * n_matched / n_candidate, 1)),
    dict(quantity="represented park properties (in analysis records)", value=n_properties),
    dict(quantity="annual property-home-CBG cells", value=n_cells),
    dict(quantity="cell share beyond 2 km (%)", value=round(100 * cell_share_gt2, 1)),
    dict(quantity="attributed-weight share beyond 2 km (%)", value=round(100 * weight_share_gt2, 1)),
    dict(quantity="visit-weight retained by matched universe (sum-construction proxy, %)",
         value=round(100 * weight_retained_proxy, 1) if weight_retained_proxy == weight_retained_proxy else "n/a")])
cov.to_csv(OUT / "destination_match_coverage.csv", index=False)
print(f"coverage: candidate {n_candidate}, matched {n_matched} ({100*n_matched/n_candidate:.1f}%), "
      f"properties {n_properties}; cell>2km {100*cell_share_gt2:.1f}%, weight>2km {100*weight_share_gt2:.1f}%")


# ---- metrics (efficient) with 1000-rep CBG-clustered descriptive bootstrap ----
def metrics_arr(pid, acres, wf, dist, wgt):
    tot = wgt.sum()
    order = np.argsort(pid, kind="stable"); p2 = pid[order]; w2 = wgt[order]
    uniq, idx = np.unique(p2, return_index=True); dest_w = np.add.reduceat(w2, idx)
    shares = np.sort(dest_w / tot)[::-1]; hhi = float((shares ** 2).sum())
    o = np.argsort(dist); cwt = np.cumsum(wgt[o]); med = float(dist[o][np.searchsorted(cwt, cwt[-1] / 2)])
    return dict(n_cells=len(pid), attributed_total=round(float(tot), 1),
                top5_share=round(float(shares[:5].sum()), 4), top10_share=round(float(shares[:10].sum()), 4),
                HHI=round(hhi, 5), effective_destinations=round(float(1 / hhi) if hhi > 0 else np.nan, 2),
                visit_weighted_acreage=round(float((wgt * acres).sum() / tot), 2),
                waterfront_share=round(float(wgt[wf].sum() / tot), 4),
                mean_distance_km=round(float((wgt * dist).sum() / tot), 3), median_distance_km=round(med, 3))


BOOT_KEYS = ["top5_share", "top10_share", "effective_destinations", "visit_weighted_acreage",
             "waterfront_share", "mean_distance_km"]
rng = np.random.default_rng(SEED)
metric_rows, top_rows = [], []
for scope, filt in [("all_destinations", R), ("non_local_gt2km", R[R.non_local])]:
    for q in QORDER:
        sub = filt[filt.origin_quartile == q]
        if not len(sub):
            continue
        pid = sub["pid"].values.astype(str); acr = sub["dest_acres"].values.astype(float)
        wf = sub["is_waterfront"].values; dist = sub["dist_km"].values.astype(float)
        wgt = sub["attributed_visit_records"].values.astype(float); hc = sub["home_cbg"].values.astype(str)
        m = metrics_arr(pid, acr, wf, dist, wgt)
        cbgs = np.unique(hc); order = np.argsort(hc, kind="stable"); hc_s = hc[order]
        ustart = np.searchsorted(hc_s, cbgs); uend = np.append(ustart[1:], len(hc_s))
        cbg_rows = {c: order[ustart[k]:uend[k]] for k, c in enumerate(cbgs)}
        boot = {k: np.empty(NBOOT) for k in BOOT_KEYS}
        for bidx in range(NBOOT):
            pick = rng.choice(cbgs, size=len(cbgs), replace=True)
            ridx = np.concatenate([cbg_rows[c] for c in pick])
            mb = metrics_arr(pid[ridx], acr[ridx], wf[ridx], dist[ridx], wgt[ridx])
            for k in BOOT_KEYS:
                boot[k][bidx] = mb[k]
        for k in BOOT_KEYS:
            m[f"{k}_ci_lo"] = round(float(np.percentile(boot[k], 2.5)), 4)
            m[f"{k}_ci_hi"] = round(float(np.percentile(boot[k], 97.5)), 4)
        m.update(scope=scope, origin_quartile=q, n_origin_cbg=int(oc[q]), n_boot=NBOOT,
                 interval_type="descriptive CBG-clustered bootstrap (percentile)")
        metric_rows.append(m)
        by_dest = sub.groupby(["pid", "dest_name", "dest_borough", "dest_acres", "dest_waterfront"])[
            "attributed_visit_records"].sum().sort_values(ascending=False)
        tot = by_dest.sum()
        for rank, ((pid_, name, boro, acr_, wf_), val) in enumerate(by_dest.head(10).items(), 1):
            top_rows.append(dict(scope=scope, origin_quartile=q, rank=rank, dest_name=name,
                                 dest_borough=boro, dest_acres=acr_, dest_waterfront=wf_,
                                 attributed_weight=round(val, 1), share=round(val / tot, 4)))
    print(f"  metrics+bootstrap({NBOOT}) done: {scope}")

MET = pd.DataFrame(metric_rows); MET.to_csv(OUT / "destination_metrics_corrected.csv", index=False)
TOP = pd.DataFrame(top_rows); TOP.to_csv(OUT / "destination_top_destinations_corrected.csv", index=False)
TOP.to_csv(OUT / "TableS_top_destinations.csv", index=False)
sc = R.groupby(["origin_quartile", "size_class"])["attributed_visit_records"].sum().reset_index()
sct = R.groupby("origin_quartile")["attributed_visit_records"].sum()
sc["share"] = sc.apply(lambda r: r["attributed_visit_records"] / sct[r["origin_quartile"]], axis=1)
sc.to_csv(OUT / "destination_size_class_shares_corrected.csv", index=False)

# ---- Figure 9 (corrected): NON-LOCAL only, 3 panels, no dual axes ----
nl = MET[MET.scope == "non_local_gt2km"].set_index("origin_quartile").reindex(QORDER)
xi = np.arange(4); qlab = ["Q1", "Q2", "Q3", "Q4"]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
# (a) top-10 share
axes[0].errorbar(xi, nl["top10_share"], yerr=[nl["top10_share"] - nl["top10_share_ci_lo"],
                 nl["top10_share_ci_hi"] - nl["top10_share"]], fmt="o-", color=CB["blue"],
                 ecolor=CB["blue"], capsize=3, ms=6)
axes[0].set_title("(a) Top-10 destination share", fontsize=10, loc="left")
axes[0].set_ylabel("Share of non-local attributed weight", fontsize=9)
# (b) effective destinations
axes[1].errorbar(xi, nl["effective_destinations"], yerr=[nl["effective_destinations"] - nl["effective_destinations_ci_lo"],
                 nl["effective_destinations_ci_hi"] - nl["effective_destinations"]], fmt="s-",
                 color=CB["vermillion"], ecolor=CB["vermillion"], capsize=3, ms=6)
axes[1].set_title("(b) Effective # destinations (1/HHI)", fontsize=10, loc="left")
axes[1].set_ylabel("Effective destinations", fontsize=9)
# (c) neutral acreage-class composition (non-local)
scn = R[R.non_local].groupby(["origin_quartile", "size_class"])["attributed_visit_records"].sum().reset_index()
scnt = R[R.non_local].groupby("origin_quartile")["attributed_visit_records"].sum()
scn["share"] = scn.apply(lambda r: r["attributed_visit_records"] / scnt[r["origin_quartile"]], axis=1)
piv = scn.pivot(index="origin_quartile", columns="size_class", values="share").reindex(QORDER).fillna(0)
bottom = np.zeros(4); colors = [CB["skyblue"], CB["blue"], CB["green"], CB["orange"], CB["vermillion"]]
for k, cls in enumerate(sorted(piv.columns)):
    axes[2].bar(xi, piv[cls].values, bottom=bottom, width=0.6, color=colors[k % len(colors)],
                label=cls.split(" ", 1)[1])
    bottom += piv[cls].values
axes[2].set_title("(c) Destination acreage-class mix", fontsize=10, loc="left")
axes[2].set_ylabel("Share of non-local weight", fontsize=9)
axes[2].legend(fontsize=6.5, frameon=False, loc="upper right")
for ax in axes:
    ax.set_xticks(xi); ax.set_xticklabels(qlab); ax.set_xlabel("Origin Non-White quartile", fontsize=9)
fig.suptitle("Non-local (>2 km) destination patterns by origin Non-White quartile "
             "(attributed visitor-weight; conservative property maximum)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
FIG.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(FIG / f"Fig9_destination_patterns_corrected.{ext}", dpi=300, bbox_inches="tight")
plt.close()

# ---- corrected summary md ----
L = ["# Descriptive destination analysis — corrected reporting (TASK 3)\n",
     "Descriptive extension of RQ2 — NOT causal, NOT a new RQ. The conservative property-week-home-CBG "
     "MAXIMUM construction is unchanged; only the reporting and figure are corrected.\n",
     "## Observation unit and weight (precise definitions)",
     f"- The analysis has **{n_cells:,} annual property-home-CBG cells** (a cell is one origin CBG paired "
     "with one destination park property). These are NOT visits and NOT unique people.",
     "- The analytical **weight** is the annual sum, over the 52 weeks, of the dwell-qualified, "
     "home-apportioned visitor-equivalents AFTER taking the maximum across a property's POIs within each "
     "property-week-home-CBG cell (vendor-scaled; not a count of unique individuals).",
     "",
     "## Coverage (see destination_match_coverage.csv)",
     f"- Candidate NYC park POIs: {n_candidate:,}; matched to a municipal park property within 100 ft: "
     f"{n_matched:,} ({100*n_matched/n_candidate:.1f}%). Represented properties: {n_properties:,}.",
     f"- Visit-weight retained by the matched-property universe (sum-construction coverage proxy from the "
     f"interim use pipeline): {100*weight_retained_proxy:.1f}%."
     if weight_retained_proxy == weight_retained_proxy else
     "- Visit-weight retained by the matched universe: not computed.",
     f"- **{100*cell_share_gt2:.1f}%** of cells are beyond 2 km; **{100*weight_share_gt2:.1f}%** of the "
     "attributed visitor-weight is beyond 2 km (the latter is the more relevant reach statistic).",
     "",
     "## Origin quartiles (cut points on the frozen RQ2 sample)",
     f"- Non-White cut points: q25={cutpoints['q25']}, q50={cutpoints['q50']}, q75={cutpoints['q75']}.",
     "- Origin-CBG counts per quartile: " + ", ".join(f"{q.split(' ')[0]} n={int(oc[q])}" for q in QORDER) + ".",
     "",
     "## Non-local (>2 km) concentration by quartile (descriptive; 1,000-rep CBG-clustered bootstrap)"]
for q in QORDER:
    r = MET[(MET.scope == "non_local_gt2km") & (MET.origin_quartile == q)].iloc[0]
    L.append(f"- {q}: top-5 {100*r['top5_share']:.1f}%, top-10 {100*r['top10_share']:.1f}%, effective "
             f"destinations {r['effective_destinations']} "
             f"[{r['effective_destinations_ci_lo']}, {r['effective_destinations_ci_hi']}].")
L += ["", "## Interpretation guard",
      "Descriptive concentration/size patterns of attributed visitor-weight. They can narrow, but not "
      "resolve, the choice-versus-constraint ambiguity: even if farther visits concentrate in large or "
      "waterfront parks, the data do not reveal preference, trip purpose, or welfare. Intervals are "
      "descriptive CBG-clustered bootstrap percentiles, not inferential tests.",
      "",
      "## Deliverables",
      "- Main Figure 9 (corrected): `Fig9_destination_patterns_corrected.{png,pdf,svg}` — non-local only, "
      "no dual axes: (a) top-10 share, (b) effective destinations, (c) acreage-class mix.",
      "- Supplementary: `destination_metrics_corrected.csv` (all-destination + non-local, incl. waterfront "
      "share, mean/median distance, visit-weighted acreage), `TableS_top_destinations.csv` (named top-10), "
      "`destination_size_class_shares_corrected.csv`, `destination_origin_quartile_counts.csv`.",
      "",
      "Size labels are neutral (very small / small / medium / large / very large); 'flagship' is reserved "
      "for the official NYC Parks classification field and is not used here."]
(OUT / "destination_analysis_summary_corrected.md").write_text("\n".join(L) + "\n")

print("=== c3 done ===")
print(MET[MET.scope == "non_local_gt2km"][["origin_quartile", "n_origin_cbg", "top5_share",
      "top10_share", "effective_destinations"]].to_string(index=False))
