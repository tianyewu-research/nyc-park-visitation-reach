"""SECTION F — quantify origin geography so the pooling issue is visible.

Per Non-White quartile: frozen-sample CBG count; CBGs with positive non-local weight; total all-
destination and non-local attributed weight; borough composition of origins; unweighted radius of
gyration of CBG centroids (km) = sqrt(mean squared distance from the quartile centroid); and a visit-
weighted radius of gyration as a sensitivity descriptor. These are diagnostics of pooling, not
behavioral outcomes."""
import sys, json, numpy as np, pandas as pd, geopandas as gpd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUT, GPKG, M2FT
from slib import build_sample

d, gid, checksum = build_sample()
d["GEOID"] = d["GEOID"].astype(str)
qs = d["pct_nonwhite"].quantile([.25, .5, .75]).values


def quartile(x):
    if x <= qs[0]:
        return "Q1 (lowest Non-White)"
    if x <= qs[1]:
        return "Q2"
    if x <= qs[2]:
        return "Q3"
    return "Q4 (highest Non-White)"


QORDER = ["Q1 (lowest Non-White)", "Q2", "Q3", "Q4 (highest Non-White)"]
d["origin_quartile"] = d["pct_nonwhite"].map(quartile)

# CBG centroids (EPSG:2263 feet -> km)
g = gpd.read_file(GPKG).to_crs("EPSG:2263")
g["GEOID"] = g["GEOID"].astype(str)
cent = g.set_index("GEOID").geometry.centroid
d["cx_km"] = d["GEOID"].map(cent.x) / M2FT / 1000.0
d["cy_km"] = d["GEOID"].map(cent.y) / M2FT / 1000.0
d["borough"] = d["borough"].astype(str)

# per-CBG all-dest and non-local weight
R = pd.read_csv(OUT / "destination_records_property_level_canonical_100ft.csv", dtype={"home_cbg": str})
R = R[R.home_cbg.isin(set(d["GEOID"]))]
all_w = R.groupby("home_cbg")["attributed_weight"].sum()
nl_w = R[R.non_local].groupby("home_cbg")["attributed_weight"].sum()
d["all_weight"] = d["GEOID"].map(all_w).fillna(0.0)
d["nonlocal_weight"] = d["GEOID"].map(nl_w).fillna(0.0)


def radius_of_gyration(sub, weights=None):
    x = sub["cx_km"].values; y = sub["cy_km"].values
    if weights is None:
        cxm, cym = x.mean(), y.mean()
        return float(np.sqrt(((x - cxm) ** 2 + (y - cym) ** 2).mean()))
    w = weights.values.astype(float)
    if w.sum() <= 0:
        return np.nan
    cxm = np.average(x, weights=w); cym = np.average(y, weights=w)
    return float(np.sqrt(np.average((x - cxm) ** 2 + (y - cym) ** 2, weights=w)))


rows = []
for q in QORDER:
    sub = d[d.origin_quartile == q]
    boro = sub["borough"].value_counts(normalize=True).round(4).to_dict()
    rows.append(dict(origin_quartile=q, n_cbg=len(sub),
                     n_cbg_positive_nonlocal=int((sub["nonlocal_weight"] > 0).sum()),
                     total_all_weight=round(float(sub["all_weight"].sum()), 1),
                     total_nonlocal_weight=round(float(sub["nonlocal_weight"].sum()), 1),
                     radius_of_gyration_km=round(radius_of_gyration(sub), 3),
                     radius_of_gyration_km_weighted=round(radius_of_gyration(sub, sub["nonlocal_weight"]), 3),
                     borough_composition=json.dumps(boro)))
disp = pd.DataFrame(rows)
disp.to_csv(OUT / "origin_quartile_coverage_and_dispersion.csv", index=False)

print("=== s4 done ===")
print(disp[["origin_quartile", "n_cbg", "n_cbg_positive_nonlocal", "radius_of_gyration_km",
            "radius_of_gyration_km_weighted"]].to_string(index=False))
print("borough composition by quartile:")
for _, r in disp.iterrows():
    print(f"  {r['origin_quartile']}: {r['borough_composition']}")
