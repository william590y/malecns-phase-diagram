"""pd3 module_theory MT2 figure: de Candia's closure on connectome modules -> figs/pd3/modules_coarse_phase.png (CPU).
(a) (g, r) diagram of the P0 closure (30 modules + rest): colour = number of active modules (X_k > 1e-3) in the up-sweep state
    (light -> dark); cells with >= 2 stable attractors outlined (multistable); cells where some start did not settle marked;
    solid line = SL edge of the closure g = alpha_eff / mu_+(W(r)) (transcritical, second order); dashed markers = first-order
    boundaries found along the sweeps (state jump > 0.02 or up/down hysteresis); thin grey = neuron-level g_lin = 0.1632.
(b) number of coexisting stable attractors from all 520 starts per (g, r).
(c) the same as (a) for P1 (Leiden) if available.
Multistability, SL stability and first-order intervals come from the corrected post-processing (pd3_coarse_post.py).
(d) our (g, r) rays in de Candia's (w0, w1) plane (w0 = g (E_kk - r I_kk)/alpha_eff, w1 = g (E_kl - r I_kl)/alpha_eff,
    Perron-weighted, mean-1 v): AL-L <- AL-R (theory_module2.json PW_*_mean1) and AL-L <- rest (quotient_P0.npz), for
    r = 0, 1, 2, 4, 6, from g = 0 to the simulated onset g_on(r) (flow_sim_boundaries.json; dot = g_lin), with de Candia's
    M = 2 lines: w0 = 1 for w1 <= 0 (SL-B1) and w0 + w1 = 1 for w1 >= 0 (SL-SH).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = "/MALECNS_ROOT"
OK = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
a_eff = (1 - np.exp(-0.2)) / 0.2
Ps = [P for P in ["P0", "P1"] if os.path.exists(f"{ROOT}/results/pd3/modules/coarse_mf_{P}.npz")]
fig = plt.figure(figsize=(16, 10.5), dpi=130)
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.27)


def diagram(ax, P, title):
    d = np.load(f"{ROOT}/results/pd3/modules/coarse_mf_{P}.npz", allow_pickle=True); js = json.load(open(f"{ROOT}/results/pd3/modules/coarse_mf_{P}.json"))
    post = np.load(f"{ROOT}/results/pd3/modules/coarse_post_{P}.npz", allow_pickle=True)
    pj = json.load(open(f"{ROOT}/results/pd3/modules/coarse_post_{P}.json"))
    g, r = d["g"], d["r"]; M = d["up"].shape[2]
    m_up = (d["up"] > 1e-3).sum(2)
    im = ax.pcolormesh(g, r, m_up, cmap="viridis", vmin=0, vmax=M, shading="nearest")
    yy, xx = np.where(post["n_attr_corr"] == 0)
    ax.scatter(g[xx], r[yy], s=3, marker=".", color="w", alpha=0.8)
    gsl = np.where(d["g_SL_edge"] < 30, d["g_SL_edge"], np.nan)
    ax.plot(gsl, r, color="k", lw=2.4)
    for b in pj["hysteresis_intervals"]:
        ax.plot([b["g_lo"], b["g_hi"]], [b["r"], b["r"]], color=OK[1], lw=1.6, solid_capstyle="butt")
    ax.axvline(0.1632, color="0.6", lw=1)
    ax.set_xscale("log"); ax.set_xlim(0.12, 30); ax.set_ylim(-0.15, 6.15)
    ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition r"); ax.set_title(title, fontsize=10)
    return im, d, js


axa = fig.add_subplot(gs[0, 0])
im, d0, js0 = diagram(axa, "P0", "(a) closure, P0 (30 spectral modules + rest):\nup-sweep, number of active modules")
hl = [plt.Line2D([], [], color="k", lw=2.4, label="closure SL edge (transcritical, 2nd order)"),
      plt.Line2D([], [], color=OK[1], lw=1.6, label="up/down hysteresis interval (1st order)"),
      plt.Line2D([], [], color="0.6", lw=1, label="neuron-level g_lin = 0.1632"),
      plt.Line2D([], [], color="0.3", marker=".", ls="", label="white dots: no stable state (non-stationary)")]
axa.legend(handles=hl, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, framealpha=0.95)
cb = fig.colorbar(im, ax=fig.axes[-1], fraction=0.046, pad=0.02); cb.set_label("# modules with X_k > 1e-3 (of 31)", fontsize=8)
ax = fig.add_subplot(gs[0, 1])
cmap = ListedColormap(["#f0f0f0", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]); bn = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 4.5, 1e9], 5)
ax.pcolormesh(d0["g"], d0["r"], np.load(f"{ROOT}/results/pd3/modules/coarse_post_P0.npz")["n_attr_corr"], cmap=cmap, norm=bn,
              shading="nearest")
ax.plot(d0["g_SL_edge"], d0["r"], color="k", lw=2)
ax.axvline(0.1632, color="0.6", lw=1)
ax.set_xscale("log"); ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition r")
ax.set_title("(b) closure P0: number of coexisting stable attractors\n(0 = none settled: non-stationary; from 520 starts)", fontsize=10)
for lab, col in [("0 (none settled)", "#f0f0f0"), ("1", "#c6dbef"), ("2", "#6baed6"), ("3-4", "#2171b5"), (">= 5", "#08306b")]:
    ax.scatter([], [], marker="s", s=30, color=col, edgecolors="0.5", label=lab)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, title="stable attractors per cell", title_fontsize=7)
ax.set_xlim(0.12, 30)
if "P1" in Ps:
    im1, d1, js1 = diagram(fig.add_subplot(gs[0, 2]), "P1", "(c) closure, P1 (30 Leiden modules + rest):\nup-sweep, number of active modules")
    cb = fig.colorbar(im1, ax=fig.axes[-1], fraction=0.046, pad=0.02); cb.set_label("# modules with X_k > 1e-3 (of 31)", fontsize=8)
# (d) rays
ax = fig.add_subplot(gs[1, 0])
t2 = json.load(open(f"{ROOT}/results/pd3/theory_module2.json")); E2, I2 = np.array(t2["PW_E_mean1"]), np.array(t2["PW_I_mean1"])
q = np.load(f"{ROOT}/results/pd3/modules/quotient_P0.npz", allow_pickle=True); nm = list(q["names"]); iL, iR = nm.index("AL-L"), nm.index("rest")
fsb = {round(row["r"], 3): row for row in json.load(open(f"{ROOT}/results/analysis2/flow_sim_boundaries.json"))["rows"]}
for j, r in enumerate([0, 1, 2, 4, 6]):
    gon = fsb[float(r)]["on"]; gg = np.linspace(0, gon, 50)
    w0 = gg * (E2[0, 0] - r * I2[0, 0]) / a_eff; w1 = gg * (E2[0, 1] - r * I2[0, 1]) / a_eff
    ax.plot(w1, w0, color=OK[[0, 4, 3, 5, 6][j]], lw=1.8, label=f"r = {r}: AL-L <- AL-R" if True else None)
    gl = 0.1632; ax.plot(gl * (E2[0, 1] - r * I2[0, 1]) / a_eff, gl * (E2[0, 0] - r * I2[0, 0]) / a_eff, "o", color=OK[[0, 4, 3, 5, 6][j]], ms=4)
    w0b = gg * (q["E_PW"][iL, iL] - r * q["I_PW"][iL, iL]) / a_eff; w1b = gg * (q["E_PW"][iL, iR] - r * q["I_PW"][iL, iR]) / a_eff
    ax.plot(w1b, w0b, color=OK[[0, 4, 3, 5, 6][j]], lw=1.2, ls=":")
ax.plot([], [], color="0.3", ls=":", label="dotted: AL-L <- rest (same r)")
xs = np.linspace(-0.6, 0, 10); ax.plot(xs, np.ones_like(xs), color="k", lw=2.5)
xs2 = np.linspace(0, 1, 10); ax.plot(xs2, 1 - xs2, color="k", lw=2.5)
ax.text(-0.55, 0.5, "SL", fontsize=11); ax.text(-0.55, 2.2, "B1 (+ SH)", fontsize=10); ax.text(0.3, 2.2, "SH", fontsize=11)
ax.axvline(0, color="0.7", lw=0.8)
ax.set_xlim(-0.6, 1.0); ax.set_ylim(0, 3.6)
ax.set_xlabel("w1 = g (E_kl - r I_kl) / alpha_eff  (inter-module imbalance)"); ax.set_ylabel("w0 = g (E_kk - r I_kk) / alpha_eff  (intra)")
ax.set_title("(d) (g, r) rays in de Candia's (w0, w1) plane, g = 0 .. g_on(r)\n(dot: g_lin; thick: de Candia's M = 2 SL lines)", fontsize=9.5)
ax.legend(fontsize=6.8, loc="lower right")
# (e) up vs down example at r = 4 (P0): module activities
ax = fig.add_subplot(gs[1, 1])
ri = int(np.argmin(np.abs(d0["r"] - 4.0))); names = [str(x) for x in d0["names"]]
for k, col in [("AL-L", OK[4]), ("AL-R", OK[2]), ("isl1", OK[1]), ("isl3", OK[6]), ("isl2", OK[3]), ("rest", OK[5])]:
    if k in names and max(d0["up"][ri, :, names.index(k)].max(), d0["dn"][ri, :, names.index(k)].max()) > 1e-6:
        i = names.index(k)
        ax.plot(d0["g"], np.maximum(d0["up"][ri, :, i], 1e-7), color=col, lw=1.6, label=f"{k} up")
        ax.plot(d0["g"], np.maximum(d0["dn"][ri, :, i], 1e-7), color=col, lw=1.1, ls="--", label=f"{k} down")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-6, 1.5); ax.axvline(0.1632, color="0.6", lw=1)
ax.set_xlabel("coupling g"); ax.set_ylabel("module activity X_k (closure)")
ax.set_title(f"(e) closure P0 at r = {d0['r'][ri]:g}: up (solid) and down (dashed) sweeps", fontsize=10)
ax.legend(fontsize=6.5, ncol=2, loc="lower right")
# (f) text summary
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
pj0 = json.load(open(f"{ROOT}/results/pd3/modules/coarse_post_P0.json"))
lines = [f"P0 closure ({js0['n_cells']} cells, 520 starts each):",
         f"  multistable cells {pj0['n_cells_multistable_corrected']}, max {pj0['max_n_attractors_corrected']} attractors",
         f"  hysteresis intervals (first order) {pj0['n_hysteresis_intervals']}",
         f"  cells with no stable attractor {pj0['n_cells_no_stable_attractor']}",
         f"  cells where some start did not settle {js0['n_cells_with_unconverged']}"]
if js0.get("dtcheck_r1"):
    lines.append(f"  r = 1 non-settling columns: dt 0.2 {js0['dtcheck_r1']['n_dt0.2']},")
    lines.append(f"  dt 0.05 {js0['dtcheck_r1']['n_dt0.05']} (of {js0['dtcheck_r1']['n_columns_each']})")
if "P1" in Ps:
    pj1 = json.load(open(f"{ROOT}/results/pd3/modules/coarse_post_P1.json"))
    lines += ["", "P1 closure:", f"  multistable cells {pj1['n_cells_multistable_corrected']}, max {pj1['max_n_attractors_corrected']}",
              f"  hysteresis intervals {pj1['n_hysteresis_intervals']}", f"  no stable attractor {pj1['n_cells_no_stable_attractor']}"]
ax.text(0, 1, "\n".join(lines), va="top", fontsize=8, family="monospace")
ax.set_title("(f) summary (coarse_mf_<P>.json)", fontsize=10, loc="left")
fig.suptitle("MT2: de Candia's closure (Perron-weighted, net input W = E^PW - r I^PW) on connectome modules", fontsize=12, y=0.96)
fig.savefig(f"{ROOT}/figs/pd3/modules_coarse_phase.png", bbox_inches="tight")
print("saved figs/pd3/modules_coarse_phase.png", Ps)
