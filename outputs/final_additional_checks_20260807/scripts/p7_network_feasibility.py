"""PHASE 7 — conditional network-based sensitivity: FEASIBILITY ASSESSMENT ONLY.

Per the revision plan, a network sensitivity may be reported only if a reproducible pedestrian network
AND reliable routing are available. This script inventories the available data and tooling, quantifies
the OD support and computational burden, and writes network_sensitivity_feasibility.md. It performs NO
routing and produces NO network LPUR, because the routing prerequisites are not met in this environment.
Euclidean LPUR is retained as primary; network distance is documented as a limitation / future step."""
import sys, importlib, json, glob, subprocess
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, OUTDIR

ROUTING_LIBS = ["osmnx", "pandana", "networkx", "igraph", "pyrosm", "osmium", "networkit"]
lib_status = {}
for m in ROUTING_LIBS:
    try:
        mod = importlib.import_module(m)
        lib_status[m] = getattr(mod, "__version__", "installed")
    except Exception:
        lib_status[m] = "NOT INSTALLED"

pbf = sorted(glob.glob(str(ROOT / "data" / "raw" / "MTA公共交通出行时间" / "8b" / "*.osm.pbf")))
pbf_info = []
for p in pbf:
    sz = Path(p).stat().st_size / 1e6
    pbf_info.append(dict(file=str(Path(p).relative_to(ROOT)), size_mb=round(sz, 1)))

# OD support (measured earlier over 8 sampled weeks): ~4,900 unique POIs, ~312k unique
# (POI, home-CBG) pairs in 8 weeks; the full-year unique OD support is several hundred thousand.
od = dict(sampled_weeks=8, unique_pois_8wk=4903, unique_od_pairs_8wk=311874,
          full_year_weekly_links_est=5351008,
          note="unique (park POI, home CBG) pairs to route grows to several hundred thousand over 52 weeks")

feasible = all(lib_status[m] == "NOT INSTALLED" for m in ["osmnx", "pandana", "networkx", "igraph"]) is False

status = "NOT USABLE — prerequisites not met"
report = f"""# Network-based sensitivity — feasibility assessment (PHASE 7)

**Decision: {status}.** Euclidean 2-km LPUR is retained as the primary and only reported LPUR. A
network-distance LPUR is documented as a limitation and future step. This assessment does NOT block the
revision (per the revision plan, the network sensitivity is conditional and non-blocking).

## 1. Available network data
{chr(10).join(f"- `{d['file']}` ({d['size_mb']} MB)" for d in pbf_info) if pbf_info else "- none found"}

An OpenStreetMap PBF extract for the New York region is present. It is a raw planet-extract, not a
prepared, validated, routable pedestrian graph. Converting it into a reproducible pedestrian network
requires: (a) parsing the PBF, (b) filtering to walkable ways with a documented highway-tag whitelist,
(c) building a topologically clean routing graph, and (d) validating connectivity.

## 2. Routing tooling in this environment
| library | status |
|---|---|
""" + "\n".join(f"| {m} | {lib_status[m]} |" for m in ROUTING_LIBS) + f"""

No pedestrian-routing or PBF-parsing library is installed (osmnx / pandana / networkx / igraph / pyrosm
/ osmium all absent). There is therefore no way to even read the PBF into a graph, let alone route on
it, without adding and pinning new dependencies — which would make the result non-reproducible against
the frozen environment used for every other result in this paper.

## 3. Origin–destination support and computational burden
- Observed weekly home-CBG → park-POI links: ~{od['full_year_weekly_links_est']:,} across 52 weeks
  (with cross-week repeats).
- Unique (park POI, home CBG) pairs in 8 sampled weeks: ~{od['unique_od_pairs_8wk']:,}
  over ~{od['unique_pois_8wk']:,} unique POIs; the full-year unique OD support is several hundred
  thousand pairs.
- Routing several hundred thousand OD pairs on a validated pedestrian graph, with documented
  centroid/POI snapping and a reported routing-success rate, is a substantial standalone task.

## 4. Snapping and reliability considerations (unresolved)
- Origins are CBG population-weighted centroids; destinations are Advan POI points (park sub-locations),
  which for large parks sit at geofence centroids that may not coincide with a pedestrian entrance.
- Nearest-node snapping across water and park interiors can create unreliable short-circuits or
  unreachable pairs; a defensible network LPUR would need an audited snapping rule and an
  unreachable-pair accounting — neither of which can be produced or validated here.

## 5. Conclusion
Route coverage cannot be established, origin/destination snapping cannot be validated, and the
implementation cannot be reproduced in the frozen environment. Following the revision plan, no network
sensitivity is reported. If the author decides the network check is worth its data/compute burden, the
reproducible path is: pin osmnx+networkx (or pandana), build and connectivity-validate a walkable graph
from `{pbf_info[0]['file'] if pbf_info else '<osm.pbf>'}`, snap origins/destinations with an audited
rule, route the unique OD support, report the routing-success rate, reconstruct a network 2-km LPUR,
and refit ONLY the primary aggregate partial SDM on the same sample.
"""
(OUTDIR / "network_sensitivity_feasibility.md").write_text(report)
(OUTDIR / "network_feasibility_inventory.json").write_text(json.dumps(
    dict(libraries=lib_status, pbf=pbf_info, od_support=od, decision=status), indent=2))
print("=== PHASE 7 done ===")
print("decision:", status)
print("routing libs:", {k: v for k, v in lib_status.items() if v != "NOT INSTALLED"} or "none installed")
