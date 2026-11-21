"""pd3 module_theory MT1 figure: figs/pd3/modules_partition.png (CPU).
(a) module size vs isolated branching threshold g_lin,k (top-30 modules of each partition + lumped rest), coloured by the
    module's dominant superclass, marker by partition; (b) equitability errors (CV of the input from module l over the
    neurons of module k) for self (k = l) and cross (k != l) pairs, excitatory and inhibitory; (c) dead-zone screen: share of
    target neurons (u^k-weighted) with net-positive input from the source's Perron pattern vs r, for the 10 cross pairs with
    the largest I^PW_kl (P0), with the AL pair for reference.
Inputs: results/pd3/modules/modules_table.json, quotient_<P>.npz, deadzone_<P>.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/MALECNS_ROOT"
OK = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2", "verm": "#D55E00",
      "purple": "#CC79A7", "grey": "#999999"}
tab = json.load(open(f"{ROOT}/results/pd3/modules/modules_table.json"))
Ps = [P for P in ["P0", "P1", "P2"] if P in tab]
SC = {"cb_intrinsic": OK["blue"], "vnc_intrinsic": OK["orange"], "ol_intrinsic": OK["green"], "vnc_motor": OK["verm"],
      "ascending_neuron": OK["purple"], "descending_neuron": OK["sky"], "other": OK["grey"]}
MK = {"P0": "o", "P1": "s", "P2": "^"}
fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), dpi=130, gridspec_kw={"wspace": 0.3})
ax = axs[0]
for P in Ps:
    for row in tab[P]["rows"]:
        top = max(row["superclass_mix"], key=row["superclass_mix"].get)
        col = SC.get(top, SC["other"]); gl = row["g_lin"]
        if not np.isfinite(gl):
            gl = 300.0
        ax.scatter(gl, row["size"], marker=MK[P], s=34, facecolor=col if P == "P0" else "none", edgecolor=col, lw=1.1, zorder=3)
        if P == "P0" and row["name"] in ("AL-L", "AL-R", "isl1", "isl2", "isl3", "isl5", "rest"):
            off = {"AL-L": (5, -10), "AL-R": (5, 4), "isl3": (5, 4), "isl5": (5, -10)}.get(row["name"], (4, 3))
            ax.annotate(row["name"], (gl, row["size"]), xytext=off, textcoords="offset points", fontsize=7.5)
ax.axvline(0.1632, color="k", lw=1); ax.text(0.15, 30, "whole-brain g_lin = 0.1632", fontsize=7.5, rotation=90, ha="right")
ax.axvline(30, color="0.6", lw=0.8, ls=":"); ax.text(33, 12, "g_lin > 30\n(none: at 300)", fontsize=7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.1, 500); ax.set_ylim(8, 2e6)
ax.set_xlabel("isolated branching threshold g_lin,k of the module alone"); ax.set_ylabel("module size (neurons)")
h = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in SC.items()]
h += [plt.Line2D([], [], marker=MK[P], ls="", mfc="none" if P != "P0" else "0.3", mec="0.3",
                 label={"P0": "P0 spectral islands", "P1": "P1 Leiden", "P2": "P2 superclass x side"}[P]) for P in Ps]
ax.legend(handles=h, fontsize=6.8, loc="upper right", framealpha=0.9, title="dominant superclass / partition", title_fontsize=7)
ax.set_title("(a) top-30 modules + lumped rest: size vs own threshold", fontsize=10)
# (b) equitability
ax = axs[1]
pos = 0; ticks = []
for P in Ps:
    q = np.load(f"{ROOT}/results/pd3/modules/quotient_{P}.npz", allow_pickle=True)
    for typ, Mx, col in [("E", q["CV_E"], OK["blue"]), ("I", q["CV_I"], OK["verm"])]:
        diag = np.diag(Mx); off = Mx[~np.eye(len(Mx), dtype=bool)]
        for lab, v in [("self", diag), ("cross", off)]:
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            parts = ax.violinplot(np.log10(v), positions=[pos], widths=0.8, showmedians=True)
            for b in parts["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.45)
            for k_ in ("cbars", "cmins", "cmaxes", "cmedians"):
                parts[k_].set_color(col)
            ticks.append((pos, f"{P}\n{typ}\n{lab}")); pos += 1
    pos += 0.6
q0 = np.load(f"{ROOT}/results/pd3/modules/quotient_P0.npz", allow_pickle=True); n0 = list(q0["names"])
iL, iR = n0.index("AL-L"), n0.index("AL-R")
ax.scatter([0.0, 1.0], np.log10([q0["CV_E"][iL, iL], q0["CV_E"][iL, iR]]), color="k", marker="*", s=60, zorder=4,
           label="AL-L <- AL-L / <- AL-R (E)")
ax.axhline(0, color="0.5", lw=0.8, ls=":")
ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks], fontsize=6.5)
ax.set_ylabel("log10 CV over i in k of sum_{j in l} c w_ij  (0 = equitable)")
ax.set_title("(b) equitability error (Schaub et al. 2016): de Candia's\nreduction is exact only at CV = 0", fontsize=10)
ax.legend(fontsize=7, loc="upper right")
# (c) dead zone
ax = axs[2]
dz = json.load(open(f"{ROOT}/results/pd3/modules/deadzone_P0.json"))["pairs"]
cross = sorted([p for p in dz if p["k"] != "rest" or True], key=lambda p: -p["I_PW"])[:10]
RS = [0, 0.5, 1, 2, 4, 6]
cols = [OK["blue"], OK["orange"], OK["green"], OK["verm"], OK["purple"], OK["sky"], "#555555", "#8c6d31", "#7b4173", "#3182bd"]
for i, p in enumerate(cross):
    ax.plot(RS, [p["share_u"][f"{r:g}"] for r in RS], marker="o", ms=4, lw=1.4, color=cols[i % len(cols)],
            label=f"{p['k']} <- {p['l']}  (I_PW {p['I_PW']:.2g}, r1* {p['r1_PW']:.2g})")
for p in dz:
    if (p["k"], p["l"]) == ("AL-L", "AL-R"):
        ax.plot(RS, [p["share_u"][f"{r:g}"] for r in RS], color="k", lw=2.4, ls="--", label="AL-L <- AL-R (reference)")
ax.axhline(0.05, color="0.4", lw=0.9, ls=":"); ax.text(5.9, 0.06, "5% (dead-zone threshold)", fontsize=7, ha="right")
ax.set_xlabel("inhibition r"); ax.set_ylabel("u^k-weighted share of k's neurons with net-positive input from l")
ax.set_ylim(-0.02, 1.02)
ax.set_title("(c) dead-zone screen, P0: the 10 pairs with the strongest\ninhibitory coupling I^PW_kl", fontsize=10)
ax.legend(fontsize=6.3, loc="upper right", framealpha=0.9)
fig.suptitle("MT1: connectome modules (P0 spectral islands" + (", P1 Leiden" if "P1" in Ps else "") +
             (", P2 superclass x side" if "P2" in Ps else "") + ")", fontsize=12, y=1.0)
fig.savefig(f"{ROOT}/figs/pd3/modules_partition.png", bbox_inches="tight")
print("saved figs/pd3/modules_partition.png", Ps)
