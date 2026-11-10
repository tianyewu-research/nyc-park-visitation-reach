"""SECTIONS A + B — canonical POI->property crosswalk (100 ft) + matching audit.

One pass over the 52 weekly files records, per unique POI (ID_STORE): first-seen coordinates and
POI_CBG, and whether it ever carries a usable positive dwell-qualified contribution to at least one
NYC home CBG (analytic_eligible). Parks Properties (>=0.1 ac) are dissolved by gispropnum. Every
GEOGRAPHIC candidate (valid coords, NYC POI_CBG) is then matched: point-in-polygon first, else the
nearest dissolved property when the boundary distance is <= 100 EPSG:2263 FEET (NOT 100*M2FT). Ties
(equal nearest distance) are resolved by minimum distance then sorted gispropnum, and counted.

The canonical crosswalk is saved for every downstream script to read; no downstream script re-matches.
The audit also computes the OLD ~328 ft match on the SAME geometry to reconcile the two pipelines and
to explain the 6,338 vs 6,385 and 4,414 vs 3,795 discrepancies without relabelling 328 ft as 100 ft."""
import sys, json, glob, numpy as np, pandas as pd, geopandas as gpd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import (ROOT, RAW, INT, OUT, NYC, M2FT, MATCH_FT, OLD_TOL_FT, DWELL_KEEP, qfrac, sha256)
from shapely.geometry import Point

np.random.seed(42)
files = sorted(glob.glob(str(RAW / "nypark_2325" / "2023-*.csv.gz")))

# NYC home-CBG universe (origins), as used by the LPUR/destination attribution
cbg = gpd.read_file(INT / "cbg_nyc.gpkg")
nyc_home = set(cbg["GEOID"].astype(str).values)

# ---- one pass: per-POI first-seen coords/POI_CBG + analytic eligibility ----
poi_lat, poi_lon, poi_cbg = {}, {}, {}
analytic_any = {}   # t1 rule: >=1 week with parseable home dict, tot>0, base>0 (any home)
analytic_nyc = {}   # our rule: >=1 usable positive dwell-qualified contribution to an NYC home CBG
for k, f in enumerate(files):
    df = pd.read_csv(f, usecols=["ID_STORE", "POI_CBG", "VISITOR_HOME_CBGS", "VISITOR_COUNTS",
                                 "BUCKETED_DWELL_TIMES", "LATITUDE", "LONGITUDE"])
    df["pc"] = df["POI_CBG"].dropna().astype("Int64").astype(str).str.zfill(12)
    df = df[df["pc"].str[:5].isin(NYC)].dropna(subset=["LATITUDE", "LONGITUDE"])
    for sid, la, lo, pc, vhc, vc, bdt in zip(df["ID_STORE"], df["LATITUDE"], df["LONGITUDE"], df["pc"],
                                             df["VISITOR_HOME_CBGS"], df["VISITOR_COUNTS"],
                                             df["BUCKETED_DWELL_TIMES"]):
        if sid not in poi_lat:
            poi_lat[sid] = la; poi_lon[sid] = lo; poi_cbg[sid] = pc
        if analytic_nyc.get(sid):
            continue
        if not isinstance(vhc, str):
            continue
        try:
            hd = json.loads(vhc)
        except Exception:
            continue
        tot = sum(hd.values()); base = vc if pd.notna(vc) else 0
        if tot <= 0 or base <= 0:
            continue
        analytic_any[sid] = True                        # t1 candidate rule (any home)
        qv0 = base * qfrac(bdt)
        if qv0 <= 0:
            continue
        for hc in hd:                                   # usable NYC-home contribution
            if hc.startswith("CA:"):
                continue
            if hc.zfill(12) in nyc_home:
                analytic_nyc[sid] = True
                break
    if (k + 1) % 13 == 0:
        print(f"  pass {k+1}/52")

sids = list(poi_lat)
geo = pd.DataFrame({"ID_STORE": sids,
                    "longitude": [poi_lon[s] for s in sids], "latitude": [poi_lat[s] for s in sids],
                    "POI_CBG": [poi_cbg[s] for s in sids],
                    "analytic_eligible": [bool(analytic_nyc.get(s, False)) for s in sids],
                    "analytic_any_home": [bool(analytic_any.get(s, False)) for s in sids]})
n_geo = len(geo); n_analytic = int(geo["analytic_eligible"].sum())
n_analytic_any = int(geo["analytic_any_home"].sum())
print(f"geographic candidates: {n_geo}; analytic (t1, any-home): {n_analytic_any}; "
      f"analytic (usable NYC-home): {n_analytic}; geographic-only(not any-home): {n_geo - n_analytic_any}")

# ---- parks: acreage filter + dissolve by gispropnum ----
parks_raw = gpd.read_file(RAW / "NYC公园边界" / "Parks Properties_20260623").to_crs("EPSG:2263")
n_rows_all = len(parks_raw)
parks = parks_raw[parks_raw["acres"] >= 0.1].copy()
n_rows_acre = len(parks)
parks["pid"] = parks["gispropnum"].astype(str)
n_pid_before = parks["pid"].nunique()
prop = parks.dissolve(by="pid").reset_index()
n_pid_after = len(prop)
print(f"park polygon rows: all={n_rows_all}, >=0.1ac={n_rows_acre}; unique gispropnum after dissolve={n_pid_after}")

pts = gpd.GeoDataFrame(geo.copy(),
                       geometry=gpd.points_from_xy(geo["longitude"], geo["latitude"]),
                       crs="EPSG:4326").to_crs("EPSG:2263")


# ---- deterministic PIP-first-then-nearest matcher on dissolved properties ----
def match(points, prop, tol_ft):
    """Return per-point matched_pid, distance_ft, method, tie_flag. PIP first; else nearest <= tol_ft.
    Ties (equal nearest distance) resolved by min distance then sorted pid; tie_flag marks them."""
    prop_sindex = prop.sindex
    geoms = prop.geometry.values; pids = prop["pid"].values
    mp = np.empty(len(points), dtype=object); dft = np.full(len(points), np.nan)
    meth = np.array(["none"] * len(points), dtype=object); tie = np.zeros(len(points), dtype=bool)
    px = points.geometry.values
    for i in range(len(points)):
        pt = px[i]
        # point-in-polygon (containment)
        cand = list(prop_sindex.query(pt, predicate="intersects"))
        hits = [j for j in cand if geoms[j].covers(pt)]
        if hits:
            j = min(hits, key=lambda jj: pids[jj])   # deterministic if overlapping polygons
            mp[i] = pids[j]; dft[i] = 0.0; meth[i] = "point_in_polygon"
            tie[i] = len(hits) > 1
            continue
        # nearest within tolerance
        cand2 = list(prop_sindex.query(pt.buffer(tol_ft), predicate="intersects"))
        if not cand2:
            continue
        dists = np.array([geoms[j].distance(pt) for j in cand2])
        dmin = dists.min()
        if dmin <= tol_ft:
            winners = [cand2[t] for t in np.where(np.isclose(dists, dmin, atol=1e-6))[0]]
            j = min(winners, key=lambda jj: pids[jj])
            mp[i] = pids[j]; dft[i] = float(dmin); meth[i] = f"nearest_within_{int(round(tol_ft))}ft"
            tie[i] = len(winners) > 1
    return mp, dft, meth, tie


print(f"matching at canonical {MATCH_FT:.0f} ft ...")
mp100, d100, meth100, tie100 = match(pts, prop, MATCH_FT)
print(f"matching at old {OLD_TOL_FT:.1f} ft (audit) ...")
mp328, d328, meth328, tie328 = match(pts, prop, OLD_TOL_FT)

geo["match_status"] = np.where(pd.notna(mp100), "matched", "unmatched")
geo["matched_pid"] = pd.Series(mp100).astype("object")
geo["distance_ft"] = np.round(d100, 3)
geo["match_method"] = meth100
geo["tie_flag"] = tie100
geo["matched_pid_328ft"] = pd.Series(mp328).astype("object")
geo["match_method_328ft"] = meth328
# canonical crosswalk (100 ft) — the required columns first
XW_COLS = ["ID_STORE", "longitude", "latitude", "POI_CBG", "analytic_eligible", "match_status",
           "matched_pid", "distance_ft", "match_method", "tie_flag"]
xw = geo[XW_COLS + ["matched_pid_328ft", "match_method_328ft"]].copy()
xw.to_csv(OUT / "poi_property_crosswalk_canonical_100ft.csv", index=False)

# ---- matching audit: reconciliation ----
m100 = geo["match_status"] == "matched"
m328 = pd.notna(geo["matched_pid_328ft"])
ae = geo["analytic_eligible"]
aa = geo["analytic_any_home"]
audit_rows = [
    dict(metric="geographic candidates (valid coords, NYC POI_CBG) [= p5 denominator 6,385]", value=n_geo),
    dict(metric="analytic candidates, t1 rule (any-home parseable dwell/base>0) [= t1 denominator ~6,338]", value=n_analytic_any),
    dict(metric="analytic candidates, usable NYC-home dwell-qualified", value=n_analytic),
    dict(metric="geographic-only (not t1-analytic) = the 6,385 vs 6,338 gap", value=n_geo - n_analytic_any),
    dict(metric="t1-analytic but no usable NYC-home = further 6,338 vs NYC-usable gap", value=n_analytic_any - n_analytic),
    dict(metric="matched POIs @328ft / analytic(any-home) denominator [~ t1's 4,414]", value=int((m328 & aa).sum())),
    dict(metric="park polygon rows (all)", value=n_rows_all),
    dict(metric="park polygon rows (>=0.1 acre)", value=n_rows_acre),
    dict(metric="unique gispropnum (>=0.1 ac, before dissolve grouping)", value=n_pid_before),
    dict(metric="unique gispropnum after dissolve", value=n_pid_after),
    dict(metric="matched POIs @100ft / geographic denominator", value=int(m100.sum())),
    dict(metric="matched POIs @100ft / analytic denominator", value=int((m100 & ae).sum())),
    dict(metric="match rate @100ft (geographic)", value=round(100 * m100.mean(), 1)),
    dict(metric="match rate @100ft (analytic)", value=round(100 * (m100 & ae).sum() / n_analytic, 1)),
    dict(metric="matched POIs @328ft / geographic denominator", value=int(m328.sum())),
    dict(metric="matched POIs @328ft / analytic denominator", value=int((m328 & ae).sum())),
    dict(metric="represented properties @100ft (unique matched gispropnum)",
         value=int(geo.loc[m100, "matched_pid"].nunique())),
    dict(metric="represented properties @328ft (unique matched gispropnum)",
         value=int(geo.loc[m328, "matched_pid_328ft"].nunique())),
    dict(metric="ties @100ft (equal-nearest, resolved deterministically)", value=int(geo["tie_flag"].sum())),
    dict(metric="point-in-polygon matches @100ft", value=int((geo["match_method"] == "point_in_polygon").sum())),
    dict(metric="nearest matches @100ft", value=int(geo["match_method"].astype(str).str.startswith("nearest").sum())),
]
pd.DataFrame(audit_rows).to_csv(OUT / "matching_audit.csv", index=False)

# POI-ID level reconciliation: old(328) vs new(100) match sets, categories sum to the union
cat = pd.Series("both_unmatched", index=geo.index, dtype=object)
cat[m100 & m328] = "matched_both"
cat[m100 & ~m328] = "new_only_100ft"   # (empty by construction: 100<=328)
cat[~m100 & m328] = "old_only_328ft"
recon = cat.value_counts().rename_axis("category").reset_index(name="n_pois")
recon["note"] = recon["category"].map({
    "matched_both": "matched under both thresholds",
    "old_only_328ft": "matched only at ~328 ft (100-300+ ft from boundary): the threshold difference",
    "new_only_100ft": "matched only at 100 ft (should be 0: 100<=328)",
    "both_unmatched": "unmatched under both (>328 ft from any municipal park property)"})
# property-ID disagreements among matched_both
mb = geo[m100 & m328]
pid_disagree = int((mb["matched_pid"].astype(str) != mb["matched_pid_328ft"].astype(str)).sum())
recon = pd.concat([recon, pd.DataFrame([dict(category="(of matched_both) property-ID disagreements",
                                             n_pois=pid_disagree, note="different property assigned at 100 vs 328 ft")])],
                  ignore_index=True)
recon.to_csv(OUT / "matching_reconciliation.csv", index=False)
assert int(recon.loc[recon.category.isin(["matched_both", "old_only_328ft", "new_only_100ft",
                                          "both_unmatched"]), "n_pois"].sum()) == n_geo, \
    "reconciliation categories must sum to the geographic union"

readme = f"""# Matching audit — canonical 100 ft vs old ~328 ft

The two prior pipelines used DIFFERENT "matched municipal-park universe" definitions. This audit
reconciles them on one dissolved-by-gispropnum park geometry. The canonical threshold is EXACTLY
**100 EPSG:2263 feet**; the old `t1_property_dedup.py` used `TOL = 100 * M2FT` = **{OLD_TOL_FT:.1f} ft**
(~100 m), although its comments said "100 ft". We NEVER relabel 328 ft as 100 ft.

## Candidate denominators (the 6,385 vs 6,338 gap, fully reconciled)
- Geographic candidates (unique POIs with valid coordinates and an NYC POI_CBG across 52 weeks): **{n_geo}**
  — reproduces p5's denominator (6,385).
- Analytic candidates, t1 rule (>=1 week with a parseable home dictionary, total>0 and VISITOR_COUNTS>0,
  ANY home): **{n_analytic_any}** — reproduces t1's denominator (~6,338).
- Analytic candidates, usable NYC-home dwell-qualified (>=1 positive contribution to an NYC home CBG):
  **{n_analytic}**.
- Reconciliation (categories sum to the geographic union {n_geo}):
  geographic-only, never t1-analytic = **{n_geo - n_analytic_any}** (this IS the 6,385 vs 6,338 gap);
  t1-analytic but no usable NYC-home = **{n_analytic_any - n_analytic}**; usable NYC-home = {n_analytic}.
  The two pipelines used different DENOMINATOR rules, not different POI universes.

## Park universe
- Polygon rows: all {n_rows_all}; >=0.1 acre {n_rows_acre}. Unique gispropnum after dissolve: **{n_pid_after}**.

## Matched counts (same dissolved geometry, PIP-first then nearest)
- @100 ft: {int(m100.sum())}/{n_geo} geographic ({100*m100.mean():.1f}%); {int((m100 & ae).sum())}/{n_analytic} analytic.
- @328 ft: {int(m328.sum())}/{n_geo} geographic; {int((m328 & ae).sum())}/{n_analytic} analytic.
- The **4,414 (t1) vs 3,795 (p5)** matched-POI gap is driven by the ~328 ft vs 100 ft threshold. On the
  common dissolved geometry, @328 ft matches {int(m328.sum())} geographic POIs and
  {int((m328 & aa).sum())} t1-analytic POIs (~ t1's 4,414; the small residual is t1 matching to
  UNDISSOLVED polygon rows, which the canonical pipeline deliberately does not do). @100 ft matches
  {int(m100.sum())} (= p5's 3,795). The reconciliation table `matching_reconciliation.csv` gives the exact
  matched-both / old-only-328ft / both-unmatched split, summing exactly to the {n_geo} geographic union.
- Represented properties (unique matched gispropnum): @100 ft **{int(geo.loc[m100, 'matched_pid'].nunique())}**,
  @328 ft {int(geo.loc[m328, 'matched_pid_328ft'].nunique())}.
- Ties at 100 ft (equal-nearest, resolved by min distance then sorted gispropnum): {int(geo['tie_flag'].sum())}.
- Property-ID disagreements among POIs matched under both thresholds: {pid_disagree}.

All downstream scripts read `poi_property_crosswalk_canonical_100ft.csv`; none re-match independently.
"""
(OUT / "matching_audit_readme.md").write_text(readme)

# provenance manifest (seed + checksums + source scripts)
manifest = dict(task="destination_per_cbg_and_match_alignment_20260808", seed=42,
                canonical_threshold_ft=MATCH_FT, old_threshold_ft=round(OLD_TOL_FT, 4),
                geographic_candidates=n_geo, analytic_candidates=n_analytic,
                dissolved_properties=n_pid_after,
                matched_pois_100ft_geographic=int(m100.sum()),
                represented_properties_100ft=int(geo.loc[m100, "matched_pid"].nunique()),
                source_scripts=dict(old_328ft="outputs/lpur_revision_calculations_20260730/scripts/t1_property_dedup.py",
                                    old_100ft_destination="outputs/final_additional_checks_20260807/scripts/p5_destination.py"),
                checksums={"poi_property_crosswalk_canonical_100ft.csv":
                           sha256(OUT / "poi_property_crosswalk_canonical_100ft.csv")})
(OUT / "provenance_manifest.json").write_text(json.dumps(manifest, indent=2))

print("=== s0 done ===")
print(f"geographic {n_geo} | analytic {n_analytic} | matched@100ft {int(m100.sum())} | "
      f"properties {int(geo.loc[m100, 'matched_pid'].nunique())} | ties {int(geo['tie_flag'].sum())}")
print(recon.to_string(index=False))
