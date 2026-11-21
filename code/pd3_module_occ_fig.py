"""pd3 module_theory MT4 figure: figs/pd3/modules_sim_occupancy.png (CPU) from results/pd3/modules/module_occ_P0.json.
(a, b) simulated module occupancy among survivors X_k / surv2 (only g with surv2 >= 0.5) at r = 0 and r = 1, modules ordered by
       their isolated threshold g_lin,k (white tick = g_lin,k of the module alone, MT1);
(c) number of modules with X_k >= 1% among survivors over (g, r) (grey: fewer than half the replicas alive), with the simulated
    onset and D crossovers (flow_sim_boundaries.json) and g_lin;
(d) coupled activation point (first g with X_k >= 1%) against the isolated threshold g_lin,k, simulation (filled) and QMF
    all-active start (open), r = 0, 1, 4: points below the diagonal are switched on by drive from already active modules
    (imperfect bifurcation) rather than by their own instability.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = "/MALECNS_ROOT"
OK = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
occ = json.load(open(f"{ROOT}/results/pd3/modules/module_occ_P0.json"))
tab = {r["name"]: r for r in json.load(open(f"{ROOT}/results/pd3/modules/modules_table.json"))["P0"]["rows"]}
names = occ["modules"]; M = len(names)
gl = np.array([tab[n]["g_lin"] if n in tab else np.inf for n in names]); order = np.argsort(gl, kind="stable")
fsb = json.load(open(f"{ROOT}/results/analysis2/flow_sim_boundaries.json"))["rows"]
fig = plt.figure(figsize=(16, 10.5), dpi=130)
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.22)
for j, r in enumerate(["0", "1"]):
    ax = fig.add_subplot(gs[0, j]); s = occ["sim"][r]
    g = np.array(s["g"]); X = np.array(s["X_among_survivors"]); ok = np.array(s["surv2"]) >= 0.5
    Z = np.where(ok[:, None], np.maximum(X, 1e-6), np.nan)[:, order].T
    im = ax.pcolormesh(g, np.arange(M), Z, norm=LogNorm(1e-5, 1), cmap="viridis", shading="nearest")
    for row, k in enumerate(order):
        if np.isfinite(gl[k]) and 0.1 <= gl[k] <= 30:
            ax.plot(gl[k], row, marker="|", color="w", ms=7, mew=1.6)
    ax.axvline(0.1632, color="k", lw=1)
    ax.set_xscale("log"); ax.set_yticks(np.arange(M)); ax.set_yticklabels([names[k] for k in order], fontsize=6)
    ax.set_xlabel("coupling g"); ax.set_title(f"({'ab'[j]}) simulation r = {r}: module occupancy among survivors\n(blank: < half the replicas alive; white tick: module's own g_lin,k)", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01).set_label("X_k (fraction of time active)", fontsize=8)
ax = fig.add_subplot(gs[1, 0])
for r, s in occ["sim"].items():
    g = np.array(s["g"]); ok = np.array(s["surv2"]) >= 0.5
    X = np.array(s["X_among_survivors"]); n1 = (X >= 0.01).sum(1)
    sc = ax.scatter(g[ok], np.full(ok.sum(), float(r)), c=n1[ok], cmap="viridis", vmin=0, vmax=M, s=9, marker="s")
    ax.scatter(g[~ok], np.full((~ok).sum(), float(r)), color="0.85", s=5, marker="s")
ax.plot([row["on"] for row in fsb], [row["r"] for row in fsb], color=OK[5], lw=1.6, ls="--", label="sim onset (tau = 3000)")
Dg = [(row["D"], row["r"]) for row in fsb if row["D"]]
ax.plot([x for x, _ in Dg], [y for _, y in Dg], color=OK[6], lw=1.6, ls="--", label="sim D (PR = N/4)")
ax.axvline(0.1632, color="k", lw=1.2, label="g_lin")
ax.set_xscale("log"); ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition r")
ax.set_title("(c) number of P0 modules with occupancy >= 1% among survivors (simulation;\ngrey: not sustained)", fontsize=9.5)
fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01).set_label("# modules >= 1% (of 34)", fontsize=8)
ax.legend(fontsize=7.5, loc="upper left")
ax = fig.add_subplot(gs[1, 1])
for r, col in [("0", OK[4]), ("1", OK[5]), ("4", OK[3])]:
    cs = occ["sim"][r]["cross_1pct_survivors"]
    xs = [gl[names.index(k)] for k in names if cs.get(k) and np.isfinite(gl[names.index(k)])]
    ys = [cs[k] for k in names if cs.get(k) and np.isfinite(gl[names.index(k)])]
    ax.scatter(xs, ys, color=col, s=22, label=f"simulation r = {r}")
    if r in occ["qmf"]:
        cq = occ["qmf"][r]["cross_1pct"]
        xq = [gl[names.index(k)] for k in names if cq.get(k) and np.isfinite(gl[names.index(k)])]
        yq = [cq[k] for k in names if cq.get(k) and np.isfinite(gl[names.index(k)])]
        ax.scatter(xq, yq, facecolors="none", edgecolors=col, s=30, label=f"QMF r = {r}")
xx = np.geomspace(0.1, 400, 10); ax.plot(xx, xx, color="0.4", lw=1, ls=":", label="activation = own threshold")
ax.axhline(0.1632, color="k", lw=1)
for k in ["AL-L", "isl1", "isl2", "rest"]:
    i = names.index(k)
    if k in occ["sim"]["0"]["cross_1pct_survivors"] and occ["sim"]["0"]["cross_1pct_survivors"][k]:
        ax.annotate(k, (gl[i], occ["sim"]["0"]["cross_1pct_survivors"][k]), xytext=(4, -9), textcoords="offset points", fontsize=7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.12, 400); ax.set_ylim(0.12, 30)
ax.set_xlabel("isolated threshold g_lin,k (module alone, MT1)"); ax.set_ylabel("coupled activation: first g with X_k >= 1%")
ax.set_title("(d) modules switch on far below their own thresholds: driven, imperfect\nactivation (no sharp island lines)", fontsize=9.5)
ax.legend(fontsize=7, loc="lower right", ncol=2)
fig.suptitle("MT4: which modules are active where (existing T = 3000 simulations, 23 r values; QMF for comparison)", fontsize=12, y=0.97)
fig.savefig(f"{ROOT}/figs/pd3/modules_sim_occupancy.png", bbox_inches="tight")
print("saved figs/pd3/modules_sim_occupancy.png")
