"""pd3 verification of module_theory: regenerate figs/pd3/modules_coarse_phase.png and modules_skeleton.png with the verifier's
corrections (adapted copies of pd3_coarse_fig.py and pd3_skeleton_fig.py, which are left unchanged; originals of the PNGs kept in
figs/pd3/verify_orig/). Changes: (1) closure SL edge and the 'no stable attractor' census use the SL edge computed directly
(pd3_verify_module_theory_sledge.py; the original rectified power iteration failed for P1 at r >= 1.25 and gave no edge at r >= 5.75);
(2) 'no stable state (non-stationary)' -> 'no stable fixed point found' (in those cells some starts converge to unstable or silent
states); (3) skeleton legend: the SL edge is labelled as the exact linear instability, not 'MF-DP' (the criticality program rejects
the MF-DP class), and 'rounded critical band' -> 'finite-K rounded onset'; (4) skeleton: QMF fixed points coexisting at > 1%
relative-L1 difference added as small dots (the 10% criterion hides those at r = 0.25 and 1).
"""

# =================== part 1: closure figure (adapted from pd3_coarse_fig.py) ===================
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
# VERIFICATION: SL edge computed directly (pd3_verify_module_theory_sledge.py) and the census recounted with it
_sl = json.load(open(f"{ROOT}/results/pd3/verify_module_theory_sledge.json"))
SLDIR, NCORR = {}, {}
for _P in Ps:
    _d = np.load(f"{ROOT}/results/pd3/modules/coarse_mf_{_P}.npz", allow_pickle=True)
    SLDIR[_P] = np.array([_sl[_P][f"{x:g}"][0] if _sl[_P][f"{x:g}"][0] is not None else np.inf for x in _d["r"]])
    _nA, _AX, _gg = _d["n_attr"], _d["attr_X"], _d["g"]          # load once (npz access decompresses every time)
    _n = np.zeros(_nA.shape, int)
    for _ri in range(_n.shape[0]):
        for _gi in range(_n.shape[1]):
            for _j in range(_nA[_ri, _gi]):
                _x = _AX[_ri, _gi, _j]
                _n[_ri, _gi] += (_gg[_gi] < SLDIR[_P][_ri]) if np.nanmax(_x) < 1e-12 else 1
    NCORR[_P] = _n
_vj = json.load(open(f"{ROOT}/results/pd3/verify_module_theory.json"))
fig = plt.figure(figsize=(16, 10.5), dpi=130)
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.27)


def diagram(ax, P, title):
    d = np.load(f"{ROOT}/results/pd3/modules/coarse_mf_{P}.npz", allow_pickle=True); js = json.load(open(f"{ROOT}/results/pd3/modules/coarse_mf_{P}.json"))
    post = np.load(f"{ROOT}/results/pd3/modules/coarse_post_{P}.npz", allow_pickle=True)
    pj = json.load(open(f"{ROOT}/results/pd3/modules/coarse_post_{P}.json"))
    g, r = d["g"], d["r"]; M = d["up"].shape[2]
    m_up = (d["up"] > 1e-3).sum(2)
    im = ax.pcolormesh(g, r, m_up, cmap="viridis", vmin=0, vmax=M, shading="nearest")
    gdir, ncorr = SLDIR[P], NCORR[P]
    yy, xx = np.where(ncorr == 0)
    ax.scatter(g[xx], r[yy], s=3, marker=".", color="w", alpha=0.8)
    gsl = np.where(gdir < 30, gdir, np.nan)
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
      plt.Line2D([], [], color="0.3", marker=".", ls="", label="white dots: no stable fixed point found")]
axa.legend(handles=hl, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, framealpha=0.95)
cb = fig.colorbar(im, ax=fig.axes[-1], fraction=0.046, pad=0.02); cb.set_label("# modules with X_k > 1e-3 (of 31)", fontsize=8)
ax = fig.add_subplot(gs[0, 1])
cmap = ListedColormap(["#f0f0f0", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]); bn = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 4.5, 1e9], 5)
ax.pcolormesh(d0["g"], d0["r"], NCORR["P0"], cmap=cmap, norm=bn,
              shading="nearest")
ax.plot(SLDIR["P0"], d0["r"], color="k", lw=2)
ax.axvline(0.1632, color="0.6", lw=1)
ax.set_xscale("log"); ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition r")
ax.set_title("(b) closure P0: number of coexisting stable attractors\n(0 = no stable fixed point found; from 520 starts)", fontsize=10)
for lab, col in [("0 (no stable f.p.)", "#f0f0f0"), ("1", "#c6dbef"), ("2", "#6baed6"), ("3-4", "#2171b5"), (">= 5", "#08306b")]:
    ax.scatter([], [], marker="s", s=30, color=col, edgecolors="0.5", label=lab)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, title="stable attractors per cell", title_fontsize=7)
ax.set_xlim(0.12, 30)
if "P1" in Ps:
    im1, d1, js1 = diagram(fig.add_subplot(gs[0, 2]), "P1", "(c) closure, P1 (30 Leiden modules + rest): up-sweep,\n# active modules (SL edge recomputed directly, verification)")
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
         f"  cells where some start did not settle {js0['n_cells_with_unconverged']}",
         f"  with the QMF's 10% rel-L1 criterion: {_vj['V11']['P0']['multistable_cells_10pct_relL1']} multistable",
         f"    cells, max {_vj['V11']['P0']['max_attractors_10pct_relL1']} attractors (verification)"]
if js0.get("dtcheck_r1"):
    lines.append(f"  r = 1 non-settling columns: dt 0.2 {js0['dtcheck_r1']['n_dt0.2']},")
    lines.append(f"  dt 0.05 {js0['dtcheck_r1']['n_dt0.05']} (of {js0['dtcheck_r1']['n_columns_each']})")
if "P1" in Ps:
    pj1 = json.load(open(f"{ROOT}/results/pd3/modules/coarse_post_P1.json"))
    lines += ["", "P1 closure:", f"  multistable cells {pj1['n_cells_multistable_corrected']}, max {pj1['max_n_attractors_corrected']}",
              f"  hysteresis intervals {pj1['n_hysteresis_intervals']}",
              f"  no stable attractor {int((NCORR['P1'] == 0).sum())} (with the directly",
              f"    computed SL edge; {pj1['n_cells_no_stable_attractor']} with the original edge)"]
ax.text(0, 1, "\n".join(lines), va="top", fontsize=8, family="monospace")
ax.set_title("(f) summary (coarse_mf_<P>.json)", fontsize=10, loc="left")
fig.suptitle("MT2: de Candia's closure (Perron-weighted, net input W = E^PW - r I^PW) on connectome modules", fontsize=12, y=0.96)
fig.savefig(f"{ROOT}/figs/pd3/modules_coarse_phase.png", bbox_inches="tight")  # verified version
print("saved figs/pd3/modules_coarse_phase.png", Ps)


# =================== part 2: skeleton figure (adapted from pd3_skeleton_fig.py) ===================
"""pd3 module_theory deliverable: module skeleton of the (g, r) phase diagram next to de Candia's closure (CPU).
Left (neuron level, this connectome):
  solid black   g_lin = 0.1632: the SL edge of the exact process (MF-DP transcritical, r-independent; phase_complete.json);
  thin dotted   QMF SL edge (net-input closure at neuron level, r-dependent): up-sweep leaves silence (qmf_analysis.json);
  dashed        simulated crossovers tau = 3000 (onset) and PR = N/4 (D) (results/analysis2/flow_sim_boundaries.json);
  dotted, light imperfect island activation: first g where island 1 / the median island / the rest reaches 1% activity
                (simulation, among survivors; module_occ_P0.json) - smeared recruitment, not sharp lines;
  markers       neuron-level QMF: g with >= 2 coexisting stable fixed points (MT3-multi), hysteresis with both branches fixed
                points (type A) or fixed point vs non-stationary branch (type B); hatched band: g where the all-active QMF does
                not settle (non-stationary mean field). These are MEAN-FIELD candidates, not confirmed first-order lines.
Right (de Candia's closure, P0 modules; coarse_mf_P0.npz): SL edge (solid), multistable cells, first-order sweep jumps.
Output: figs/pd3/modules_skeleton.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/MALECNS_ROOT"
OK = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
fsb = json.load(open(f"{ROOT}/results/analysis2/flow_sim_boundaries.json"))["rows"]
occ = json.load(open(f"{ROOT}/results/pd3/modules/module_occ_P0.json"))
qa = json.load(open(f"{ROOT}/results/pd3/module_theory/qmf_analysis.json"))["per_r"]
fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.6), dpi=130, gridspec_kw={"wspace": 0.18})
ax = axs[0]
R = [row["r"] for row in fsb]
ax.axvline(0.1632, color="k", lw=2.4, label="g_lin = 0.1632: linear instability of silence (exact, r-independent)")
ax.plot([row["on"] for row in fsb], R, color=OK[4], lw=1.8, ls="--", label="tau = 3000 crossover (sim onset; B_AL sustained)")
Dg = [(row["D"], row["r"]) for row in fsb if row["D"]]
ax.plot([x for x, _ in Dg], [y for _, y in Dg], color=OK[5], lw=1.8, ls="--", label="PR = N/4 crossover (sim D)")
nr = [(row["D_not_reached_up_to"], row["r"]) for row in fsb if not row["D"] and row["D_not_reached_up_to"]]
ax.scatter([x for x, _ in nr], [y for _, y in nr], marker=">", color=OK[5], s=18, label="D not reached up to this g (sim)")
rs = sorted(float(r) for r in occ["sim"])
i1 = [occ["sim"][f"{r:g}"]["cross_1pct_survivors"].get("isl1") for r in rs]
isl = [k for k in occ["modules"] if k.startswith("isl")]
med = [np.nanmedian([occ["sim"][f"{r:g}"]["cross_1pct_survivors"][k] or np.nan for k in isl]) for r in rs]
rest = [occ["sim"][f"{r:g}"]["cross_1pct_survivors"].get("rest") for r in rs]
ax.plot([x if x else np.nan for x in i1], rs, color=OK[1], lw=1.5, ls=":", label="island 1 reaches 1% (sim; imperfect)")
ax.plot(med, rs, color=OK[3], lw=1.5, ls=":", label="median island reaches 1% (sim)")
ax.plot([x if x else np.nan for x in rest], rs, color=OK[6], lw=1.5, ls=":", label="rest reaches 1% (sim)")
qr = sorted(float(r) for r in qa)
# VERIFICATION: g with >= 2 stable QMF fixed points whose 34-module vectors differ by > 1% (single linkage), from qmf_r<r>.npz
import scipy.sparse as _sp
from scipy.sparse.csgraph import connected_components as _cc
def _rl(a, b): return np.abs(a - b).sum() / max(np.abs(a).sum(), np.abs(b).sum(), 1e-300)
M1PCT = {}
for _r in qr:
    _d = np.load(f"{ROOT}/results/pd3/module_theory/qmf_r{_r:g}.npz", allow_pickle=True)
    _res, _rho, _X = _d["p1_resid"], _d["p1_rhoJ"], _d["p1_X"].copy(); _X[_d["p1_mean"] < 1e-8] = 0
    _conv = (_res < 1e-7) & (_res / np.maximum(1 - _rho, 1e-6) < 1e-6) & (_rho < 1)
    for _gi, _g in enumerate(_d["g1"]):
        _v = [_X[_s, _gi] for _s in range(_X.shape[0]) if _conv[_s, _gi]]
        if len(_v) >= 2 and _cc(_sp.csr_matrix(np.array([[_rl(a, b) <= 0.01 for b in _v] for a in _v])), directed=False)[0] >= 2:
            M1PCT.setdefault(f"{_r:g}", []).append(float(_g))
for r in qr:
    a = qa[f"{r:g}"]
    ns = [x for x in a["nonstationary_g_all_active_pass1"] if x in a["pass3_dt0.05_oscillating_g"]] or \
        (a["pass3_dt0.05_oscillating_g"] if a["nonstationary_g_all_active_pass1"] else [])
    if ns:
        ax.plot([min(ns), max(ns)], [r, r], color="0.6", lw=6, alpha=0.35, solid_capstyle="butt")
    if a["g_multistable_fixed_points"]:
        ax.scatter(a["g_multistable_fixed_points"], [r] * len(a["g_multistable_fixed_points"]), marker="s", s=16, facecolors="none",
                   edgecolors=OK[0], lw=0.8)
    g1pct = M1PCT.get(f"{r:g}", [])
    if g1pct:
        ax.scatter(g1pct, [r - 0.12] * len(g1pct), marker=".", s=10, color="0.35")
    for h in a["hysteresis_runs"]:
        if h["n_type_A_both_fixed_points"] >= 2:
            ax.plot([h["g_lo"], h["g_hi"]], [r + 0.08, r + 0.08], color=OK[5], lw=2.5)
        elif h["n_type_B_fixed_point_vs_nonstationary"] >= 2:
            ax.plot([h["g_lo"], h["g_hi"]], [r + 0.08, r + 0.08], color=OK[2], lw=2.5)
    if a.get("SL_edge_up_sweep"):
        ax.plot(a["SL_edge_up_sweep"], r, marker="|", color="0.3", ms=9)
ax.plot([], [], color="0.6", lw=6, alpha=0.35, label="QMF does not settle (all-active start, dt = 0.2 and 0.05)")
ax.scatter([], [], marker="s", s=16, facecolors="none", edgecolors=OK[0], label="QMF: >= 2 stable fixed points (> 10% apart)")
ax.scatter([], [], marker=".", s=10, color="0.35", label="QMF: >= 2 stable fixed points (> 1% apart; verification)")
ax.plot([], [], color=OK[5], lw=2.5, label="QMF hysteresis, both branches fixed points")
ax.plot([], [], color=OK[2], lw=2.5, label="QMF hysteresis, fixed point vs non-stationary")
ax.plot([], [], marker="|", color="0.3", ls="", ms=9, label="QMF SL edge (net-input closure)")
ax.text(0.125, 5.6, "SL", fontsize=13); ax.text(0.19, 5.6, "finite-K\nrounded\nonset", fontsize=8)
ax.text(1.1, 2.35, "B_AL + island\ncascade", fontsize=8); ax.text(6, 0.15, "D", fontsize=13)
ax.set_xscale("log"); ax.set_xlim(0.12, 32); ax.set_ylim(-0.2, 6.3)
ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition r")
ax.set_title("(a) neuron level: module skeleton (simulation crossovers + QMF structure)", fontsize=10)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, framealpha=0.95)
ax = axs[1]
if os.path.exists(f"{ROOT}/results/pd3/modules/coarse_mf_P0.npz"):
    d = np.load(f"{ROOT}/results/pd3/modules/coarse_mf_P0.npz", allow_pickle=True); js = json.load(open(f"{ROOT}/results/pd3/modules/coarse_mf_P0.json"))
    m_up = (d["up"] > 1e-3).sum(2)
    im = ax.pcolormesh(d["g"], d["r"], m_up, cmap="viridis", vmin=0, vmax=d["up"].shape[2], shading="nearest", alpha=0.85)
    post = np.load(f"{ROOT}/results/pd3/modules/coarse_post_P0.npz", allow_pickle=True)
    pj = json.load(open(f"{ROOT}/results/pd3/modules/coarse_post_P0.json"))
    yy, xx = np.where(post["n_attr_corr"] == 0)
    ax.scatter(d["g"][xx], d["r"][yy], s=3, marker=".", color="w", alpha=0.8)
    ax.plot(np.where(d["g_SL_edge"] < 30, d["g_SL_edge"], np.nan), d["r"], color="k", lw=2.4, label="closure SL edge (second order)")
    for b in pj["hysteresis_intervals"]:
        ax.plot([b["g_lo"], b["g_hi"]], [b["r"], b["r"]], color=OK[1], lw=1.6)
    ax.plot([], [], color=OK[1], lw=1.6, label="first order: up/down hysteresis interval")
    ax.scatter([], [], s=8, marker=".", color="0.5", label="white dots: no stable fixed point found")
    labs_ = post["labels_corr"]; rv, gv_ = d["r"], d["g"]
    bs = [(float(rv[ri]), float(gv_[gi])) for ri in range(len(rv)) for gi in range(len(gv_))
          if any("AL-L" in x and "AL-R" not in x for x in labs_[ri, gi].split("|"))
          and any("AL-R" in x and "AL-L" not in x for x in labs_[ri, gi].split("|"))]
    if bs:
        rr_ = sorted(set(x for x, _ in bs)); gmin = [min(g_ for r_, g_ in bs if r_ == x) for x in rr_]
        ax.plot(gmin, rr_, color=OK[6], lw=2, ls="--", label="B1L | B1R coexist from here (AL lobes, closure)")
    ax.axvline(0.1632, color="0.6", lw=1, label="neuron-level g_lin")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("# active modules (up-sweep)", fontsize=8)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, framealpha=0.95)
ax.set_xscale("log"); ax.set_xlim(0.12, 30); ax.set_ylim(-0.15, 6.15); ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition r")
ax.set_title("(b) de Candia's closure on the same 31 modules (P0)", fontsize=10)
fig.suptitle("Module skeleton of the (g, r) diagram: neuron level vs de Candia's coarse closure", fontsize=12, y=0.99)
fig.savefig(f"{ROOT}/figs/pd3/modules_skeleton.png", bbox_inches="tight")  # verified version
print("saved figs/pd3/modules_skeleton.png")
