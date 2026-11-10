"""PHASE 5 — descriptive destination analysis (extension of RQ2; NOT causal, NOT a new RQ).

Two passes over the 52 weekly Advan files (cached after first run):
  Pass 1 (light): collect each POI's coordinates -> spatially match POIs to NYC municipal park
     PROPERTIES (nearest within 100 ft, same universe as the access supply / matched-use robustness).
  Pass 2 (heavy): attribute dwell-qualified visits to home CBGs; within each (property, week, home CBG)
     cell take the conservative MAXIMUM across the property's POIs (limits multiple-POI over-weighting);
     sum those weekly maxima over the year. Units are 'attributed visit records', NOT unique visitors.

Origins are grouped by quartile of CBG Non-White share on the frozen RQ2 analysis sample (documented cut
points). For each quartile, all-destination and non-local (>2 km) concentration/size/type metrics are
computed, with CBG-clustered bootstrap 95% intervals. Descriptive only: these patterns cannot separate
preference from constraint."""
import sys, json, glob, numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUTDIR, SEED, savefig, CB
from slib import build_sample
import matplotlib.pyplot as plt

np.random.seed(SEED)
RAW = ROOT / "data" / "raw"
NYC = ("36005", "36047", "36061", "36081", "36085")
DWELL_KEEP = ("5-20", "21-60", "61-240", ">240")
CRS = "EPSG:2263"; M2FT = 1 / 0.3048; MATCH_FT = 100.0
LOCAL_KM = 2.0
RECORDS = OUTDIR / "destination_records_property_level.csv"
files = sorted(glob.glob(str(RAW / "nypark_2325" / "2023-*.csv.gz")))

# ---------- origin grouping on the frozen RQ2 sample (always needed) ----------
d, gid, checksum = build_sample()
qs = d["pct_nonwhite"].quantile([.25, .5, .75]).values
cutpoints = dict(q25=round(float(qs[0]), 4), q50=round(float(qs[1]), 4), q75=round(float(qs[2]), 4))


def quartile(x):
    if x <= qs[0]:
        return "Q1 (lowest Non-White)"
    if x <= qs[1]:
        return "Q2"
    if x <= qs[2]:
        return "Q3"
    return "Q4 (highest Non-White)"


def qfrac(bj):
    if not isinstance(bj, str):
        return 1.0
    try:
        b = json.loads(bj)
    except Exception:
        return 1.0
    t = sum(b.values())
    return sum(b.get(k, 0) for k in DWELL_KEEP) / t if t > 0 else 0.0


poi_match_line = ""
if RECORDS.exists():
    # ---------- reuse cached property-level records (skip the 52-file passes) ----------
    R = pd.read_csv(RECORDS, dtype={"home_cbg": str, "pid": str})
    print(f"reloaded cached records: {len(R):,} rows")
else:
    import geopandas as gpd
    cbg = gpd.read_file(ROOT / "data" / "interim" / "cbg_nyc.gpkg").to_crs(CRS)
    cbg_cx = dict(zip(cbg["GEOID"], cbg.geometry.centroid.x))
    cbg_cy = dict(zip(cbg["GEOID"], cbg.geometry.centroid.y))
    parks = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623").to_crs(CRS)
    parks = parks[parks["acres"] >= 0.1].reset_index(drop=True)
    parks["pid"] = parks["gispropnum"].astype(str)
    prop = parks.dissolve(by="pid", aggfunc={"acres": "sum", "signname": "first", "borough": "first",
                                             "typecatego": "first", "class": "first",
                                             "waterfront": "first"}).reset_index()
    prp = prop.geometry.representative_point()
    prop["px"] = prp.x.values; prop["py"] = prp.y.values
    BOROMAP = {"B": "Brooklyn", "X": "Bronx", "M": "Manhattan", "Q": "Queens", "R": "Staten Island"}
    prop["borough_name"] = prop["borough"].map(lambda b: BOROMAP.get(str(b), str(b)))
    pid_attr = prop.set_index("pid")[["signname", "borough_name", "acres", "typecatego", "class",
                                      "waterfront", "px", "py"]].to_dict("index")
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
    poi_pid = dict(zip(nn.loc[nn["dist"] <= MATCH_FT, "ID_STORE"], nn.loc[nn["dist"] <= MATCH_FT, "pid"]))
    poi_match_line = f"{len(poi_pid)}/{len(sids)}"
    annual = defaultdict(float)
    for i, f in enumerate(files):
        df = pd.read_csv(f, usecols=["ID_STORE", "POI_CBG", "VISITOR_HOME_CBGS", "VISITOR_COUNTS",
                                     "BUCKETED_DWELL_TIMES"])
        df["pc"] = df["POI_CBG"].dropna().astype("Int64").astype(str).str.zfill(12)
        df = df[df["pc"].str[:5].isin(NYC)]
        week = defaultdict(float)
        for sid, vhc, vc, bdt in zip(df["ID_STORE"], df["VISITOR_HOME_CBGS"], df["VISITOR_COUNTS"],
                                     df["BUCKETED_DWELL_TIMES"]):
            pid = poi_pid.get(sid)
            if pid is None or not isinstance(vhc, str):
                continue
            try:
                hd = json.loads(vhc)
            except Exception:
                continue
            vis = (vc if pd.notna(vc) else 0) * qfrac(bdt)
            tot = sum(hd.values())
            if tot <= 0 or vis <= 0:
                continue
            for hc, dc in hd.items():
                if hc.startswith("CA:"):
                    continue
                hc = hc.zfill(12)
                if hc not in cbg_cx:
                    continue
                av = vis * (dc / tot)
                key = (pid, hc)
                if av > week[key]:
                    week[key] = av
        for key, v in week.items():
            annual[key] += v
    d_nw = dict(zip(d["GEOID"], d["pct_nonwhite"]))
    recs = []
    for (pid, hc), v in annual.items():
        if hc not in d_nw:
            continue
        at = pid_attr.get(pid)
        if at is None:
            continue
        dist_km = np.hypot(cbg_cx[hc] - at["px"], cbg_cy[hc] - at["py"]) / M2FT / 1000
        recs.append(dict(pid=pid, home_cbg=hc, attributed_visit_records=v,
                         origin_nonwhite=d_nw[hc], origin_quartile=quartile(d_nw[hc]),
                         dest_name=at["signname"], dest_borough=at["borough_name"], dest_acres=at["acres"],
                         dest_type=at["typecatego"], dest_class=at["class"],
                         dest_waterfront=str(at["waterfront"]), dist_km=round(dist_km, 3),
                         non_local=dist_km > LOCAL_KM))
    R = pd.DataFrame(recs)
    R.to_csv(RECORDS, index=False)
    print(f"built records: {len(R):,}  matched POIs: {poi_match_line}")

R["home_cbg"] = R["home_cbg"].astype(str)

# ---- authoritative re-derivation of destination attributes by property id (pid) ----
# Robust to any earlier borough-code / waterfront-encoding issues baked into the cached records:
# borough and waterfront are taken from the NYC Parks Properties table keyed on gispropnum.
import geopandas as _gpd
_p = _gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623")
_p = _p[_p["acres"] >= 0.1].copy()
_p["pid"] = _p["gispropnum"].astype(str)
_BORO = {"B": "Brooklyn", "X": "Bronx", "M": "Manhattan", "Q": "Queens", "R": "Staten Island"}
_agg = _p.dissolve(by="pid", aggfunc={"acres": "sum", "signname": "first", "borough": "first",
                                      "typecatego": "first", "class": "first",
                                      "waterfront": "max"}).reset_index()
_agg["borough_name"] = _agg["borough"].map(lambda b: _BORO.get(str(b), str(b)))
_agg["is_wf"] = _agg["waterfront"].fillna(0).astype(float) > 0
attr = _agg.set_index("pid")
R["pid"] = R["pid"].astype(str)
R["dest_borough"] = R["pid"].map(attr["borough_name"]).fillna(R["dest_borough"])
R["dest_acres"] = R["pid"].map(attr["acres"]).fillna(R["dest_acres"])
R["dest_name"] = R["pid"].map(attr["signname"]).fillna(R["dest_name"])
R["is_waterfront"] = R["pid"].map(attr["is_wf"]).fillna(False).astype(bool)
R["dest_waterfront"] = R["is_waterfront"].map({True: "Yes", False: "No"})
# re-persist the corrected records (authoritative borough/waterfront/acres) as the deliverable
R.drop(columns=["is_waterfront"]).to_csv(RECORDS, index=False)
print(f"records in-sample: {len(R):,}  attributed total: {R.attributed_visit_records.sum():,.0f}")
print(f"waterfront records: {100*R.is_waterfront.mean():.1f}% of rows")


# ---------- size classes (transparent, reproducible) ----------
def size_class(a):
    if a < 1:
        return "1 pocket (<1 ac)"
    if a < 5:
        return "2 small (1-5 ac)"
    if a < 20:
        return "3 medium (5-20 ac)"
    if a < 100:
        return "4 large (20-100 ac)"
    return "5 flagship (>=100 ac)"


R["size_class"] = R["dest_acres"].map(size_class)  # is_waterfront set authoritatively above

cw = R.drop_duplicates("pid")[["pid", "dest_name", "dest_borough", "dest_acres", "dest_type",
                               "dest_class", "dest_waterfront", "size_class"]].sort_values(
    "dest_acres", ascending=False)
cw.to_csv(OUTDIR / "destination_classification_crosswalk.csv", index=False)


# ---------- efficient metrics (vectorized weighted median; no np.repeat) ----------
def wmedian(values, weights):
    o = np.argsort(values)
    v = values[o]; w = weights[o]
    cw_ = np.cumsum(w)
    cut = cw_[-1] / 2.0
    return float(v[np.searchsorted(cw_, cut)])


def metrics_arr(pid, name, acres, wf, dist, wgt):
    tot = wgt.sum()
    # per-destination shares
    order = np.argsort(pid, kind="stable")
    p2 = pid[order]; w2 = wgt[order]
    uniq, idx = np.unique(p2, return_index=True)
    dest_w = np.add.reduceat(w2, idx)
    shares = np.sort(dest_w / tot)[::-1]
    hhi = float((shares ** 2).sum())
    return dict(n_records=len(pid), attributed_total=round(float(tot), 1),
                top10_share=round(float(shares[:10].sum()), 4), HHI=round(hhi, 5),
                effective_destinations=round(float(1.0 / hhi) if hhi > 0 else np.nan, 2),
                visit_weighted_acreage=round(float((wgt * acres).sum() / tot), 2),
                waterfront_share=round(float(wgt[wf].sum() / tot), 4),
                mean_distance_km=round(float((wgt * dist).sum() / tot), 3),
                median_distance_km=round(wmedian(dist, wgt), 3))


BOOT_KEYS = ["top10_share", "effective_destinations", "visit_weighted_acreage",
             "waterfront_share", "mean_distance_km"]
QORDER = ["Q1 (lowest Non-White)", "Q2", "Q3", "Q4 (highest Non-White)"]


def subset_arrays(sub):
    return (sub["pid"].values.astype(str), None, sub["dest_acres"].values.astype(float),
            sub["is_waterfront"].values, sub["dist_km"].values.astype(float),
            sub["attributed_visit_records"].values.astype(float), sub["home_cbg"].values.astype(str))


metric_rows, top_rows = [], []
rng = np.random.default_rng(SEED)
for scope, filt in [("all_destinations", R), ("non_local_gt2km", R[R.non_local])]:
    for q in QORDER:
        sub = filt[filt.origin_quartile == q]
        if not len(sub):
            continue
        pid, _, acr, wf, dist, wgt, hc = subset_arrays(sub)
        m = metrics_arr(pid, None, acr, wf, dist, wgt)
        # single-loop CBG-clustered bootstrap for ALL metrics at once
        cbgs = np.unique(hc)
        # index records by CBG once
        order = np.argsort(hc, kind="stable")
        hc_s = hc[order]; ustart = np.searchsorted(hc_s, cbgs)
        uend = np.append(ustart[1:], len(hc_s))
        cbg_rows = {c: order[ustart[k]:uend[k]] for k, c in enumerate(cbgs)}
        boot = {k: [] for k in BOOT_KEYS}
        for _ in range(200):
            pick = rng.choice(cbgs, size=len(cbgs), replace=True)
            ridx = np.concatenate([cbg_rows[c] for c in pick])
            mb = metrics_arr(pid[ridx], None, acr[ridx], wf[ridx], dist[ridx], wgt[ridx])
            for k in BOOT_KEYS:
                boot[k].append(mb[k])
        for k in BOOT_KEYS:
            arr = np.array(boot[k], dtype=float)
            m[f"{k}_ci_lo"] = round(float(np.percentile(arr, 2.5)), 4)
            m[f"{k}_ci_hi"] = round(float(np.percentile(arr, 97.5)), 4)
        m.update(scope=scope, origin_quartile=q, origin_nonwhite_q25=cutpoints["q25"],
                 origin_nonwhite_q50=cutpoints["q50"], origin_nonwhite_q75=cutpoints["q75"])
        metric_rows.append(m)
        # top destinations
        by_dest = sub.groupby(["pid", "dest_name", "dest_borough", "dest_acres", "dest_waterfront"])[
            "attributed_visit_records"].sum().sort_values(ascending=False)
        tot = by_dest.sum()
        for rank, ((pid_, name, boro, acr_, wf_), val) in enumerate(by_dest.head(10).items(), 1):
            top_rows.append(dict(scope=scope, origin_quartile=q, rank=rank, dest_name=name,
                                 dest_borough=boro, dest_acres=acr_, dest_waterfront=wf_,
                                 attributed=round(val, 1), share=round(val / tot, 4)))
    print(f"  metrics+bootstrap done: {scope}")

MET = pd.DataFrame(metric_rows); MET.to_csv(OUTDIR / "destination_metrics_by_nonwhite_quartile.csv", index=False)
TOP = pd.DataFrame(top_rows); TOP.to_csv(OUTDIR / "top_destinations_by_nonwhite_quartile.csv", index=False)

sc = R.groupby(["origin_quartile", "size_class"])["attributed_visit_records"].sum().reset_index()
sc_tot = R.groupby("origin_quartile")["attributed_visit_records"].sum()
sc["share"] = sc.apply(lambda r: r["attributed_visit_records"] / sc_tot[r["origin_quartile"]], axis=1)
sc.to_csv(OUTDIR / "destination_size_class_shares.csv", index=False)

# ---------- Fig9: 3-panel destination figure ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
qlab = ["Q1", "Q2", "Q3", "Q4"]; xi = np.arange(4)
axa = axes[0]
for scope, col, mk in [("all_destinations", CB["blue"], "o"), ("non_local_gt2km", CB["vermillion"], "s")]:
    subm = MET[MET.scope == scope].set_index("origin_quartile").reindex(QORDER)
    axa.errorbar(xi, subm["effective_destinations"],
                 yerr=[subm["effective_destinations"] - subm["effective_destinations_ci_lo"],
                       subm["effective_destinations_ci_hi"] - subm["effective_destinations"]],
                 fmt=mk + "-", color=col, ecolor=col, capsize=3, ms=6,
                 label=("all destinations" if scope == "all_destinations" else "non-local (>2 km)"))
axa.set_xticks(xi); axa.set_xticklabels(qlab); axa.set_xlabel("Origin Non-White quartile", fontsize=9)
axa.set_ylabel("Effective # destinations (1/HHI)", fontsize=9)
axa.set_title("(a) Destination concentration", fontsize=10, loc="left"); axa.legend(fontsize=7.5, frameon=False)
axb = axes[1]
subn = MET[MET.scope == "non_local_gt2km"].set_index("origin_quartile").reindex(QORDER)
axb.bar(xi - 0.2, subn["visit_weighted_acreage"], width=0.4, color=CB["green"])
axb.set_xticks(xi); axb.set_xticklabels(qlab); axb.set_xlabel("Origin Non-White quartile", fontsize=9)
axb.set_ylabel("Visit-weighted acreage (non-local)", fontsize=9, color=CB["green"])
axb2 = axb.twinx()
axb2.plot(xi, subn["waterfront_share"], "o-", color=CB["orange"])
axb2.set_ylabel("Waterfront visit share", fontsize=9, color=CB["orange"])
axb.set_title("(b) Destination size & waterfront (non-local)", fontsize=10, loc="left")
axc = axes[2]
piv = sc.pivot(index="origin_quartile", columns="size_class", values="share").reindex(QORDER).fillna(0)
bottom = np.zeros(4)
colors = [CB["skyblue"], CB["blue"], CB["green"], CB["orange"], CB["vermillion"]]
for k, cls in enumerate(sorted(piv.columns)):
    axc.bar(xi, piv[cls].values, bottom=bottom, width=0.6, color=colors[k % len(colors)],
            label=cls.split(" ", 1)[1])
    bottom += piv[cls].values
axc.set_xticks(xi); axc.set_xticklabels(qlab); axc.set_xlabel("Origin Non-White quartile", fontsize=9)
axc.set_ylabel("Visit share", fontsize=9)
axc.set_title("(c) Destination size-class mix (all)", fontsize=10, loc="left")
axc.legend(fontsize=6.5, frameon=False, loc="upper right")
fig.suptitle("Descriptive destination patterns by origin Non-White quartile "
             "(attributed visit records; conservative property maximum)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
p9 = savefig(fig, "Fig9_destination_patterns"); plt.close()

# ---------- summary md ----------
match_txt = poi_match_line or "(reused cached records; see destination_classification_crosswalk.csv)"
lines = ["# Descriptive destination analysis summary (PHASE 5)\n",
         "Descriptive extension of RQ2 — NOT causal and NOT a new research question. Units are "
         "**attributed visit records** (dwell-qualified, home-apportioned, conservative property-week-"
         "home-CBG maximum), not unique visitors.\n",
         f"- Universe: NYC municipal park PROPERTIES matched to Advan POIs within {MATCH_FT:.0f} ft "
         f"(matched POIs: {match_txt}).",
         f"- Origins: frozen RQ2 sample (checksum {checksum}); Non-White quartile cut points "
         f"q25={cutpoints['q25']}, q50={cutpoints['q50']}, q75={cutpoints['q75']}.",
         f"- {len(R):,} (property, home-CBG) records; {R.attributed_visit_records.sum():,.0f} attributed "
         f"visit records total; {(100*R.non_local.mean()):.1f}% of records are non-local (>2 km).",
         "- Waterfront classification uses the authoritative NYC Parks `waterfront` field only.\n",
         "## Concentration (non-local >2 km)"]
for q in QORDER:
    r = MET[(MET.scope == "non_local_gt2km") & (MET.origin_quartile == q)]
    if len(r):
        r = r.iloc[0]
        lines.append(f"- {q}: top-10 destinations = {100*r['top10_share']:.1f}% of non-local records; "
                     f"effective destinations {r['effective_destinations']}; visit-weighted acreage "
                     f"{r['visit_weighted_acreage']}; waterfront share {100*r['waterfront_share']:.1f}%; "
                     f"mean distance {r['mean_distance_km']} km.")
lines += ["", "## Interpretation guard",
          "These are descriptive concentration/size/reach patterns of observed attributed visit records. "
          "They can narrow, but not eliminate, the choice-versus-constraint ambiguity: even if farther "
          "visits concentrate in large or waterfront parks, the data do not reveal visitor preference, "
          "trip purpose, or welfare. Bootstrap intervals are CBG-clustered and descriptive; no "
          "inferential test is attached.",
          "", f"Figure: `{p9[0]}` (+pdf/svg)."]
(OUTDIR / "destination_analysis_summary.md").write_text("\n".join(lines) + "\n")

print("=== PHASE 5 done ===")
print(MET[MET.scope == "non_local_gt2km"][["origin_quartile", "top10_share", "effective_destinations",
      "visit_weighted_acreage", "waterfront_share", "mean_distance_km"]].to_string(index=False))
