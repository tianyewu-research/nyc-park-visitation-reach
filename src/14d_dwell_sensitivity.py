"""
14d_dwell_sensitivity.py   (advisor #7: dwell-threshold sensitivity)
The main use measure keeps only visitors dwelling >=5 min (drops 84% of records as pass-throughs).
This builds three use variants in ONE pass over the 52 weekly files to bracket that choice:
  V_none : no dwell filter (q=1, all records)
  V_ge5  : >=5 min  (buckets 5-20,21-60,61-240,>240)  -- the MAIN measure
  V_ge21 : >=21 min (buckets 21-60,61-240,>240)        -- stricter (bucket edges preclude 3/10 min)
Output: data/interim/use_dwell_variants.csv (GEOID, V_none, V_ge5, V_ge21)
"""
import pandas as pd, geopandas as gpd, json, glob
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"; INTERIM = ROOT / "data" / "interim"
NYC_FIPS = ("36005", "36047", "36061", "36081", "36085")
KEEP = {"none": ("<5", "5-20", "21-60", "61-240", ">240"),
        "ge5": ("5-20", "21-60", "61-240", ">240"),
        "ge21": ("21-60", "61-240", ">240")}


def qfrac(bj, keep):
    if not isinstance(bj, str):
        return 1.0
    try:
        b = json.loads(bj)
    except Exception:
        return 1.0
    t = sum(b.values())
    return sum(b.get(k, 0) for k in keep) / t if t > 0 else 0.0


cbg = gpd.read_file(INTERIM / "cbg_nyc.gpkg")
nyc = set(cbg["GEOID"].values)
files = sorted(glob.glob(str(RAW / "nypark_2325" / "2023-*.csv.gz")))
V = {k: defaultdict(float) for k in KEEP}
for i, f in enumerate(files):
    df = pd.read_csv(f, usecols=["POI_CBG", "VISITOR_HOME_CBGS", "VISITOR_COUNTS", "BUCKETED_DWELL_TIMES"])
    df["cbg"] = df["POI_CBG"].dropna().astype("Int64").astype(str).str.zfill(12)
    df = df[df["cbg"].str[:5].isin(NYC_FIPS)]
    for _, r in df.iterrows():
        vhc = r["VISITOR_HOME_CBGS"]
        if pd.isna(vhc):
            continue
        try:
            hd = json.loads(vhc)
        except Exception:
            continue
        vis = r["VISITOR_COUNTS"] if pd.notna(r["VISITOR_COUNTS"]) else 0
        tot = sum(hd.values())
        if tot <= 0:
            continue
        qs = {k: vis * qfrac(r["BUCKETED_DWELL_TIMES"], keep) for k, keep in KEEP.items()}
        for hcbg, dc in hd.items():
            if hcbg.startswith("CA:"):
                continue
            hcbg = hcbg.zfill(12)
            if hcbg in nyc:
                sh = dc / tot
                for k in KEEP:
                    V[k][hcbg] += qs[k] * sh
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(files)}")

out = pd.DataFrame({"GEOID": list(nyc)})
out["V_none"] = out["GEOID"].map(lambda x: round(V["none"].get(x, 0.0), 2))
out["V_ge5"] = out["GEOID"].map(lambda x: round(V["ge5"].get(x, 0.0), 2))
out["V_ge21"] = out["GEOID"].map(lambda x: round(V["ge21"].get(x, 0.0), 2))
out.to_csv(INTERIM / "use_dwell_variants.csv", index=False)
for k in ["V_none", "V_ge5", "V_ge21"]:
    print(f"{k}: median(>0)={out.loc[out[k]>0,k].median():.0f}")
print(f"saved {INTERIM/'use_dwell_variants.csv'}")
assert len(out) == 6807
print("[self-check] 14d dwell variants OK")
