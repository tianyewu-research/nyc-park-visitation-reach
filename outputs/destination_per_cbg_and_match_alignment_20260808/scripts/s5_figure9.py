"""SECTION G — Figure 9 redesigned around the corrected per-CBG estimand.

Panel A: per-CBG top-10 non-local destination share by quartile (box + jittered spread).
Panel B: per-CBG effective non-local destination count (1/HHI) by quartile.
Panel C: POOLED attributed-weight destination-size composition (clearly labelled pooled; not a per-CBG
concentration result). Medians/IQRs shown; sample n noted unobtrusively. No forced monotonic narrative."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import OUT, CB, savefig
import matplotlib.pyplot as plt

QORDER = ["Q1 (lowest Non-White)", "Q2", "Q3", "Q4 (highest Non-White)"]
QLAB = ["Q1", "Q2", "Q3", "Q4"]
conc = pd.read_csv(OUT / "destination_concentration_per_cbg.csv", dtype={"home_cbg": str})
conc = conc[conc.origin_quartile.isin(QORDER)]
R = pd.read_csv(OUT / "destination_records_property_level_canonical_100ft.csv", dtype={"home_cbg": str})

# exact plotted data
plot_data = conc[["home_cbg", "origin_quartile", "top10_share", "effective_destinations",
                  "n_nonlocal_properties", "has_nonlocal"]].copy()
plot_data.to_csv(OUT / "figure9_plot_data.csv", index=False)


def box_panel(ax, col, ylabel, title):
    data = [conc[(conc.origin_quartile == q)][col].dropna().values for q in QORDER]
    bp = ax.boxplot(data, positions=range(4), widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color=CB["black"], lw=1.6))
    for patch in bp["boxes"]:
        patch.set(facecolor=CB["skyblue"], alpha=0.6, edgecolor=CB["blue"])
    for k, q in enumerate(QORDER):
        v = conc[(conc.origin_quartile == q)][col].dropna().values
        jitter = (np.random.default_rng(42 + k).random(len(v)) - 0.5) * 0.25
        ax.scatter(np.full(len(v), k) + jitter, v, s=2, color=CB["grey"], alpha=0.15, zorder=1)
        ax.annotate(f"n={len(v)}", (k, ax.get_ylim()[0]), fontsize=6.5, ha="center", va="bottom",
                    color="#555")
    ax.set_xticks(range(4)); ax.set_xticklabels(QLAB)
    ax.set_xlabel("Origin Non-White quartile", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10, loc="left")


np.random.seed(42)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
box_panel(axes[0], "top10_share", "Per-CBG top-10 non-local share", "(a) Within-CBG top-10 share")
box_panel(axes[1], "effective_destinations", "Per-CBG effective destinations (1/HHI)",
          "(b) Within-CBG effective destinations")

# Panel C: pooled attributed-weight size composition (non-local), labelled pooled
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


nwmap = conc.set_index("home_cbg")["origin_quartile"].to_dict()
Rn = R[R.non_local].copy()
Rn["origin_quartile"] = Rn["home_cbg"].map(nwmap)
Rn = Rn[Rn.origin_quartile.isin(QORDER)]
Rn["size_class"] = Rn["dest_acres"].map(size_class)
sc = Rn.groupby(["origin_quartile", "size_class"])["attributed_weight"].sum().reset_index()
sct = Rn.groupby("origin_quartile")["attributed_weight"].sum()
sc["share"] = sc.apply(lambda r: r["attributed_weight"] / sct[r["origin_quartile"]], axis=1)
piv = sc.pivot(index="origin_quartile", columns="size_class", values="share").reindex(QORDER).fillna(0)
bottom = np.zeros(4); xi = np.arange(4)
colors = [CB["skyblue"], CB["blue"], CB["green"], CB["orange"], CB["vermillion"]]
for k, cls in enumerate(sorted(piv.columns)):
    axes[2].bar(xi, piv[cls].values, bottom=bottom, width=0.6, color=colors[k % len(colors)],
                label=cls.split(" ", 1)[1])
    bottom += piv[cls].values
axes[2].set_xticks(xi); axes[2].set_xticklabels(QLAB); axes[2].set_xlabel("Origin Non-White quartile", fontsize=9)
axes[2].set_ylabel("Pooled non-local weight share", fontsize=9)
axes[2].set_title("(c) POOLED size composition", fontsize=10, loc="left")
axes[2].legend(fontsize=6.5, frameon=False, loc="upper right")
fig.suptitle("Figure 9 (replacement): per-CBG non-local destination concentration by origin Non-White "
             "quartile (canonical 100 ft)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
paths = savefig(fig, "figure9_replacement_per_cbg"); plt.close()


# caption numbers
def stat(q, col):
    v = conc[conc.origin_quartile == q][col].dropna()
    return v.median(), v.quantile(.25), v.quantile(.75)


cap = ["Figure 9 (replacement). Per-CBG destination concentration on non-local (>2 km) records, by origin "
       "Non-White quartile (canonical 100 ft matched-park universe; frozen RQ2 sample, 1,563 CBGs per "
       "quartile). Panels (a) and (b) show WITHIN-CBG distributions (box = median and IQR; points = CBGs) "
       "of the top-10 destination share and the effective destination count (1/HHI); these are the "
       "corrected per-CBG estimand, not pooled-quartile metrics. Panel (c) is the POOLED attributed-weight "
       "size composition and is descriptive only.\n"]
for q, ql in zip(QORDER, QLAB):
    m, lo, hi = stat(q, "top10_share"); me, loe, hie = stat(q, "effective_destinations")
    cap.append(f"{ql}: top-10 share median {m:.3f} (IQR {lo:.3f}-{hi:.3f}); effective destinations "
               f"median {me:.2f} (IQR {loe:.2f}-{hie:.2f}).")
(OUT / "figure9_replacement_caption_numbers.txt").write_text("\n".join(cap) + "\n")

print("=== s5 done ===")
print("Fig9 per-CBG:", paths[0])
print(conc.groupby("origin_quartile")[["top10_share", "effective_destinations"]].median().reindex(QORDER).round(3).to_string())
