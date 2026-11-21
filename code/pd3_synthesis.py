#!/usr/bin/env python
"""pd3 synthesis (workflow wf_98fb990d-4fb, synthesis step, 2026-11-09).

Reads ONLY the verified outputs of the four pd3 programs (redraw_fss, module_theory, criticality, multistability) and
results/analysis2/phase_complete.json. Runs no simulation. Writes
  results/pd3/synthesis.json            every number quoted in results/pd3/REPORT.md that is not copied verbatim from a
                                        program md, with the file and key it was read from (or the computation done here);
  figs/pd3/phase_diagram_v3.png         THE (g, r) phase diagram of the real connectome in de Candia style, simulation with
                                        theory overlaid, and de Candia's closure on the same modules against the tests;
  figs/pd3/q2_second_order_evidence.png the evidence on "is the steep but smooth crossover a second-order transition?";
  figs/pd3/fss_summary.png              finite-size check: null onsets sharpen with N, the real AL crossover does not.

Line convention of the de Candia-style diagram (de Candia et al. 2026, Fig. 2): solid = second-order (continuous) line,
dashed = first-order / coexistence line, dotted = crossover or convention (tau = 200, tau = 3000, PR = N/4, 1% activation).
(The user-specified heat map figs/pd3/phase_lifetime_heatmap.png draws the same crossovers dashed; it is not redrawn here.)

Usage: python code/pd3_synthesis.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = "/MALECNS_ROOT"
FIG = f"{ROOT}/figs/pd3"
OK = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#7F7F7F")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 8,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})

NUM = {}


def rec(name, value, source):
    """Record a number used in the report, with where it came from."""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    NUM[name] = {"value": value, "source": source}
    return value


def J(p):
    with open(f"{ROOT}/{p}") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------- load
pc = J("results/analysis2/phase_complete.json")
GLIN = rec("g_lin_real", pc["g_lin"]["real"], "results/analysis2/phase_complete.json g_lin.real")
rkeys = sorted(pc["sim"]["real"], key=float)
R = np.array([float(k) for k in rkeys])


def simb(name, part="g"):
    out = []
    for k in rkeys:
        b = pc["sim"]["real"][k][name]
        if b.get("status") == "ok" and b.get("g") is not None:
            out.append(b["g"] if part == "g" else b["total"][0 if part == "lo" else 1])
        else:
            out.append(np.nan)
    return np.array(out, float)


S, S_lo, S_hi = simb("S"), simb("S", "lo"), simb("S", "hi")
ON, ON_lo, ON_hi = simb("on"), simb("on", "lo"), simb("on", "hi")
D, D_lo, D_hi = simb("D"), simb("D", "lo"), simb("D", "hi")
GMAX = np.array([pc["sim"]["real"][k]["g_range"][1] for k in rkeys])
TH_ON = np.array([pc["theory"]["real"][k]["on"]["g"] or np.nan for k in rkeys], float)
TH_S = np.array([pc["theory"]["real"][k]["S"]["g"] or np.nan for k in rkeys], float)
TH_D = np.array([pc["theory"]["real"][k]["D"]["g"] if pc["theory"]["real"][k]["D"]["status"] == "ok" else np.nan
                 for k in rkeys], float)
for rr, a, b, c in zip(R, S, ON, D):
    if rr in (0.0, 1.0, 6.0):
        rec(f"sim_S_r{rr:g}", a, f"phase_complete.json sim.real['{rr:g}'].S.g")
        rec(f"sim_on_r{rr:g}", b, f"phase_complete.json sim.real['{rr:g}'].on.g")
        rec(f"sim_xon_r{rr:g}", b / GLIN, "computed: sim on / g_lin (phase_complete.json)")
        rec(f"sim_xS_r{rr:g}", a / GLIN, "computed: sim S / g_lin (phase_complete.json)")
rec("sim_D_by_r", {f"{rr:g}": (None if np.isnan(d) else round(float(d), 3)) for rr, d in zip(R, D)},
    "phase_complete.json sim.real[r].D.g (None = PR < N/4 up to the largest simulated g)")
rec("sim_gmax_by_r", {f"{rr:g}": float(g) for rr, g in zip(R, GMAX)}, "phase_complete.json sim.real[r].g_range[1]")
rec("on_over_S_window_r0_r1", [float(ON[0] / S[0]), float(ON[R == 1][0] / S[R == 1][0])], "computed from phase_complete.json")

occ = J("results/pd3/modules/module_occ_P0.json")
ISL1, ISLMED, REST = [], [], []
for k in rkeys:
    c = occ["sim"][k]["cross_1pct_survivors"]
    isl = [c[m] for m in c if m.startswith("isl") and c[m] is not None]
    ISL1.append(c.get("isl1") if c.get("isl1") is not None else np.nan)
    ISLMED.append(float(np.median(isl)) if len(isl) == 30 else np.nan)  # median only where all 30 islands crossed
    REST.append(c.get("rest") if c.get("rest") is not None else np.nan)
ISL1, ISLMED, REST = (np.array(a, float) for a in (ISL1, ISLMED, REST))
rec("isl1_1pct_r0_r1_r6", [float(ISL1[R == v][0]) for v in (0, 1, 6)],
    "results/pd3/modules/module_occ_P0.json sim[r].cross_1pct_survivors.isl1")
rec("rest_1pct_r0_r1_r6", [float(REST[R == v][0]) for v in (0, 1, 6)],
    "results/pd3/modules/module_occ_P0.json sim[r].cross_1pct_survivors.rest")

qa = J("results/pd3/module_theory/qmf_analysis.json")
QR = np.array([float(k) for k in qa["per_r"]])
QSL = np.array([qa["per_r"][k]["SL_edge_up_sweep"] for k in qa["per_r"]], float)

m3b = J("results/pd3/multistab/analysis_m3b.json")
M3B = np.array([(c["r"], c["g"], max(c["alive"].values())) for c in m3b["cells"].values()], float)
rec("m3b_cells_any_coexistence", [len(M3B), bool(m3b["any_coexistence"])], "results/pd3/multistab/analysis_m3b.json")

m1 = J("results/pd3/multistab/analysis_m1.json")
M1 = []
for rk, v in m1["per_r"].items():
    for row in v["rows"]:
        ns = [row["starts"][s]["n_surv"] for s in ("AL-L", "AL-R")]
        M1.append((float(rk), row["g"], min(ns), bool(row["broken_symmetry"]), bool(row["closure_predicts_B1"])))
M1 = np.array(M1, dtype=object)
rec("m1_summary", m1["summary"], "results/pd3/multistab/analysis_m1.json summary")
rec("m1_n_closure_window_points", int(sum(1 for p in M1 if p[4])), "results/pd3/multistab/analysis_m1.json rows[].closure_predicts_B1")

cm = np.load(f"{ROOT}/results/pd3/modules/coarse_mf_P0.npz", allow_pickle=True)
cp = np.load(f"{ROOT}/results/pd3/modules/coarse_post_P0.npz", allow_pickle=True)
cpj = J("results/pd3/modules/coarse_post_P0.json")
CG, CR = cm["g"], cm["r"]


def closure_cat(label, n):
    labs = str(label).split("|")
    if n == 0 or labs == ["none"]:
        return 5  # no stable fixed point found
    if labs == ["SL"]:
        return 0
    lobe = any("AL-L" in x and "AL-R" not in x for x in labs) and any("AL-R" in x and "AL-L" not in x for x in labs)
    if lobe:
        return 4  # coexisting left-only and right-only AL states (B1L | B1R)
    if n >= 2:
        return 3  # other multistability (islands on/off)
    if labs == ["SH"]:
        return 2
    return 1  # one active attractor: B_m with the AL (+ islands)


_LAB, _NAT = cp["labels_corr"], cp["n_attr_corr"]  # load once (each NpzFile access re-reads the array)
CAT = np.array([[closure_cat(_LAB[i, j], _NAT[i, j]) for j in range(len(CG))] for i in range(len(CR))])
CM_SL_EDGE = cm["g_SL_edge"]
rec("closure_P0_cells_by_category", {n: int((CAT == i).sum()) for i, n in enumerate(
    ["SL", "single_active_Bm", "SH", "multistable_islands", "B1L_B1R_AL_lobes", "no_stable_fixed_point"])},
    "computed here from results/pd3/modules/coarse_post_P0.npz labels_corr / n_attr_corr (4000 cells)")
rec("closure_P0_B1_first_g", {k: round(v, 4) for k, v in J("results/pd3/module_theory.json")["MT2"]["P0"]["corrected"][
    "AL_lobe_broken_first_g_by_r"].items() if k in ("2.5", "4", "6")},
    "results/pd3/module_theory.json MT2.P0.corrected.AL_lobe_broken_first_g_by_r")
rec("closure_P0_SL_edge_r0_r6", [cpj["g_SL_edge"]["0"], cpj["g_SL_edge"]["6"]], "results/pd3/modules/coarse_post_P0.json g_SL_edge")
rec("closure_P0_n_hysteresis_intervals", cpj["n_hysteresis_intervals"], "results/pd3/modules/coarse_post_P0.json")

fss = J("results/pd3/redraw_fss/fss_analysis.json")
crit = J("results/pd3/criticality/aval_analysis.json")
qs = J("results/pd3/criticality/qs_analysis.json")
m2a = J("results/pd3/multistab/analysis_m2a.json")
isl = J("results/pd3/module_theory/island_lifetimes.json")


# ================================================================================================ figure 1: diagram
def figure_phase_diagram():
    fig = plt.figure(figsize=(17.5, 10.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.36], hspace=0.08, wspace=0.13, left=0.045, right=0.985,
                          top=0.94, bottom=0.03)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    lega = fig.add_subplot(gs[1, 0]); lega.axis("off")
    legb = fig.add_subplot(gs[1, 1]); legb.axis("off")
    GL, GH = 0.1, 30.0

    # ---------------- (a) regions
    C_SL, C_M, C_B, C_D = "#E6E6E6", "#FBEFC7", "#D8E8F5", "#CFEBDD"
    ax.fill_betweenx(R, GL, GLIN, color=C_SL, lw=0)
    ax.fill_betweenx(R, GLIN, ON, color=C_M, lw=0)
    dfill = np.where(np.isnan(D), GH, D)
    ax.fill_betweenx(R, ON, dfill, color=C_B, lw=0)
    md = ~np.isnan(D)
    ax.fill_betweenx(R[md], D[md], GH, color=C_D, lw=0)
    # rows simulated only to g = 2.6: hatch the unsimulated part of each row band
    for i, rr in enumerate(R):
        if GMAX[i] < 29:
            lo = (R[i] + R[i - 1]) / 2 if i > 0 else rr
            hi = (R[i] + R[i + 1]) / 2 if i < len(R) - 1 else rr
            ax.fill_between([GMAX[i], GH], lo, hi, facecolor="none", edgecolor="#B0B0B0", hatch="////", lw=0)
    # ---------------- (a) lines
    ax.axvline(GLIN, color="k", lw=2.8, zorder=6)
    ax.fill_betweenx(R, S_lo, S_hi, color=OK["verm"], alpha=0.18, lw=0)
    ax.fill_betweenx(R, ON_lo, ON_hi, color=OK["blue"], alpha=0.18, lw=0)
    ax.plot(S, R, ls=":", lw=2.2, color=OK["verm"], marker="o", ms=3, zorder=5)
    ax.plot(ON, R, ls=":", lw=2.6, color=OK["blue"], marker="o", ms=3.5, zorder=5)
    ax.fill_betweenx(R[md], D_lo[md], D_hi[md], color=OK["purple"], alpha=0.2, lw=0)
    ax.plot(D[md], R[md], ls=":", lw=2.4, color=OK["purple"], marker="o", ms=3.5, zorder=5)
    m = ~np.isnan(ISL1)
    ax.plot(ISL1[m], R[m], ls=(0, (1, 2)), lw=1.4, color=OK["orange"], zorder=4)
    m = ~np.isnan(ISLMED)
    ax.plot(ISLMED[m], R[m], ls=(0, (1, 2)), lw=1.4, color=OK["green"], marker=".", ms=4, zorder=4)
    m = ~np.isnan(REST)
    ax.plot(REST[m], R[m], ls=(0, (1, 2)), lw=1.4, color="0.35", marker=".", ms=4, zorder=4)
    # D not reached up to g = 30
    nod = np.isnan(D) & (GMAX > 29)
    ax.plot(np.full(nod.sum(), 27.5), R[nod], ls="", marker=">", ms=7, color=OK["purple"], zorder=6)
    # theory overlays
    ax.plot(TH_ON, R, ls=(0, (1, 3)), lw=0.9, color=OK["sky"], marker="s", ms=4.5, mfc="none", mew=1.1, zorder=5)
    mt = ~np.isnan(TH_D)
    ax.plot(TH_D[mt], R[mt], ls=(0, (1, 3)), lw=0.9, color=OK["purple"], marker="D", ms=4.5, mfc="none", mew=1.1,
            zorder=5)
    ax.plot(QSL, QR, ls="", marker="|", ms=11, mew=1.6, color="0.25", zorder=7)
    ax.scatter(M3B[:, 1], M3B[:, 0], marker="x", s=22, color="0.30", lw=1.0, zorder=6)
    # labels
    ax.text(0.118, 3.0, "SL\nsilence\nstable", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.205, 5.55, "M", fontsize=12, fontweight="bold", ha="center", color="#7A5A00")
    ax.text(0.215, 5.1, "metastable\nAL (finite-K\nrounded\nonset)", fontsize=7.5, ha="center", va="top", color="#5A4300")
    ax.text(0.73, 5.62, r"$\mathbf{B_{AL}}$", fontsize=12, ha="center")
    ax.text(0.73, 5.32, "AL on,\nL+R", fontsize=7.5, ha="center", va="top")
    ax.text(2.6, 3.05, "AL + islands\n(cascade of smeared\nmodule activations)", fontsize=7.5, ha="center")
    ax.text(10.5, 1.35, "brain-wide activity,\nstill localized\n(PR < N/4)", fontsize=7.5, ha="center")
    ax.text(9.0, 0.28, "D (SH-like):\nglobal, PR ≥ N/4", fontsize=9, ha="center", fontweight="bold", color="#00563B")
    ax.text(15.0, 5.05, "hatched rows (r = 2.25-5.5):\nsimulated only to g = 2.6", fontsize=7, ha="center", color="0.35",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    ax.set_xscale("log"); ax.set_xlim(GL, GH); ax.set_ylim(-0.12, 6.25)
    ax.set_xticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30])
    ax.set_xticklabels(["0.1", "0.2", "0.3", "0.5", "1", "2", "3", "5", "10", "20", "30"])
    ax.set_xlabel("coupling g (raw-normalised units)")
    ax.set_ylabel("inhibition scale r")
    ax.set_title("(a) Real connectome, stochastic model (h = 0): phases and lines from simulation, theory overlaid",
                 loc="left")

    hand_a = [
        Line2D([], [], color="k", lw=2.8, label="g_lin = 0.1632: SL edge, the only critical line"),
        Line2D([], [], color="k", lw=1.6, ls="--", label="first-order / coexistence line: none found (box)"),
        Line2D([], [], color=OK["blue"], lw=2.6, ls=":", marker="o", ms=3.5, label="τ = 3000 crossover (sim.; 95% band)"),
        Line2D([], [], color=OK["verm"], lw=2.2, ls=":", marker="o", ms=3, label="τ = 200 crossover (sim.; slow window)"),
        Line2D([], [], color=OK["purple"], lw=2.4, ls=":", marker="o", ms=3.5, label="PR = N/4 crossover (sim.; D)"),
        Line2D([], [], color=OK["purple"], ls="", marker=">", ms=7, label="D not reached up to g = 30 (sim.)"),
        Line2D([], [], color=OK["orange"], lw=1.4, ls=(0, (1, 2)), label="island 1 (2,292 neurons) at 1% (sim.)"),
        Line2D([], [], color=OK["green"], lw=1.4, ls=(0, (1, 2)), marker=".", label="median of 30 islands at 1% (sim.)"),
        Line2D([], [], color="0.35", lw=1.4, ls=(0, (1, 2)), marker=".", label="rest of brain at 1% (sim.)"),
        Line2D([], [], color=OK["sky"], lw=0.9, ls=(0, (1, 3)), marker="s", mfc="none", ms=4.5,
               label="theory: τ = 3000, one-mode chain, computed K"),
        Line2D([], [], color=OK["purple"], lw=0.9, ls=(0, (1, 3)), marker="D", mfc="none", ms=4.5,
               label="theory: QMF PR = N/4"),
        Line2D([], [], color="0.25", ls="", marker="|", ms=11, mew=1.6, label="theory: QMF silent edge (net-input)"),
        Line2D([], [], color="0.3", ls="", marker="x", ms=6, label="QMF ≥ 2 fixed points (noise: 1 state)"),
        Patch(fc=C_SL, label="SL: quiescent"), Patch(fc=C_M, label="M: metastable AL (rounded onset)"),
        Patch(fc=C_B, label="B_AL (+ islands): localized, active"), Patch(fc=C_D, label="D: global (SH-like)"),
    ]
    lega.legend(handles=hand_a, loc="upper left", ncol=2, frameon=False, fontsize=7.8, handlelength=3.2,
                columnspacing=1.5, bbox_to_anchor=(0.0, 1.0))

    # ---------------- (b) de Candia's closure on the P0 modules
    cmap = ListedColormap(["#E6E6E6", "#D8E8F5", "#CFEBDD", "#F3DCA9", "#E8A27C", "#FFFFFF"])
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    axb.pcolormesh(CG, CR, CAT, cmap=cmap, norm=norm, shading="nearest")
    yy, xx = np.where(CAT == 5)
    axb.scatter(CG[xx], CR[yy], s=2, color="0.55", marker=".", lw=0)
    sl = np.where(CM_SL_EDGE < 30, CM_SL_EDGE, np.nan)
    axb.plot(sl, CR, color="k", lw=2.6, zorder=6)
    for b in cpj["hysteresis_intervals"]:
        axb.plot([b["g_lo"], b["g_hi"]], [b["r"], b["r"]], color="k", lw=1.5, ls=(0, (4, 2)), zorder=5)
    axb.axvline(GLIN, color="0.4", lw=1.2, zorder=4)
    axb.plot(ON, R, ls=":", lw=2.4, color=OK["blue"], zorder=6)
    for rr, g, nsurv, broken, inwin in M1:
        if nsurv >= 10:
            axb.scatter(g, rr, s=46 if inwin else 30, facecolor="white" if not inwin else OK["yellow"], edgecolor="k",
                        lw=1.0, zorder=8, marker="o")
        else:
            axb.scatter(g, rr, s=44, color=OK["verm"], marker="X", lw=0.6, edgecolor="k", zorder=8)
    axb.text(0.122, 5.75, "SL", fontsize=11, fontweight="bold")
    axb.text(0.80, 3.5, "B1L | B1R\n(one AL lobe on)", fontsize=8.5, ha="center", va="center",
             bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    axb.text(0.55, 1.75, "B_m: AL,\nthen + islands", fontsize=8.5, ha="center", va="center")
    axb.text(9.0, 0.2, "SH", fontsize=11, fontweight="bold", ha="center")
    axb.set_xscale("log"); axb.set_xlim(0.12, GH); axb.set_ylim(-0.12, 6.25)
    axb.set_xticks([0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30])
    axb.set_xticklabels(["0.2", "0.3", "0.5", "1", "2", "3", "5", "10", "20", "30"])
    axb.set_xlabel("coupling g (raw-normalised units)")
    axb.set_ylabel("inhibition scale r")
    axb.set_title("(b) de Candia's closure on the same modules (theory, P0: AL-L, AL-R, 28 islands, rest) vs the "
                  "stochastic test", loc="left")
    hand_b = [
        Patch(fc="#E6E6E6", label="closure: SL"),
        Patch(fc="#D8E8F5", label="closure: one active state (B_m)"),
        Patch(fc="#CFEBDD", label="closure: SH (all 31 modules on)"),
        Patch(fc="#F3DCA9", label="closure: ≥ 2 states (islands on/off)"),
        Patch(fc="#E8A27C", label="closure: B1L | B1R (one AL lobe on)"),
        Patch(fc="white", ec="0.6", label="closure: no stable fixed point found"),
        Line2D([], [], color="k", lw=2.6, label="closure SL edge (second order; moves with r)"),
        Line2D([], [], color="k", lw=1.5, ls=(0, (4, 2)), label="closure first-order interval (hysteresis)"),
        Line2D([], [], color="0.4", lw=1.2, label="exact g_lin (neuron level)"),
        Line2D([], [], color=OK["blue"], lw=2.4, ls=":", label="simulated τ = 3000 crossover"),
        Line2D([], [], color="k", ls="", marker="o", mfc="white", ms=6, label="M1 test: L and R co-active (no B1)"),
        Line2D([], [], color="k", ls="", marker="o", mfc=OK["yellow"], ms=7, label="M1 test inside a closure B1 window"),
        Line2D([], [], color=OK["verm"], ls="", marker="X", mec="k", ms=7, label="M1 untestable: < 10 survivors"),
    ]
    legb.legend(handles=hand_b, loc="upper left", ncol=2, frameon=False, fontsize=7.8, handlelength=3.0,
                columnspacing=1.5, bbox_to_anchor=(0.0, 1.0))
    fig.text(0.80, 0.035,
             "Searched for, not found, in the stochastic model:\n"
             "• broken L/R symmetry: 0 of 48 (g, r) points (M1)\n"
             "• sweep hysteresis that persists as the dwell grows\n   (6 r, h = 1e-5 and 1e-4, D = 50-800; M2a)\n"
             "• coexistence at 14 loop centres (M2b)\n"
             "• > 1 active attractor from 13 starts in 23 cells (M3)\n"
             "• the 40 mean-field coexistence cells (M3b)",
             fontsize=7.8, va="bottom", ha="left", bbox=dict(fc="#F7F7F7", ec="0.6", pad=4))
    fig.suptitle("MaleCNS stochastic Wilson-Cowan: de Candia-style (g, r) phase diagram.  Solid = continuous "
                 "(second-order, finite-size rounded) line; dashed = first-order line; dotted = crossover",
                 fontsize=12, y=0.985)
    out = f"{FIG}/phase_diagram_v3.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


# ================================================================================ figure 2: Q2 evidence (6 panels)
def figure_q2():
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.9, bottom=0.07, wspace=0.27, hspace=0.38)
    xon = {0: ON[R == 0][0] / GLIN, 1: ON[R == 1][0] / GLIN}
    # (a) order parameter: QS n_AL vs x, finite-K chain, deterministic mean field
    ax = axs[0, 0]
    for key, col, r_ in (("real_r0", OK["black"], 0), ("real_r1", OK["blue"], 1)):
        rows = qs["C2"][key]["rows"]
        x = np.array([q["x"] for q in rows]); n = np.array([q["n_AL"] for q in rows])
        se = np.array([q["n_AL_se"] for q in rows]); ch = np.array([q["chain_nQS_neurons"] for q in rows], float)
        mf = np.array([q["MF_amp_nAL"] if q["MF_amp_nAL"] is not None else np.nan for q in rows], float)
        K = qs["C2"][key]["K"]
        ax.errorbar(x, n, yerr=1.96 * se, color=col, marker="o", ms=4, lw=1.4, label=f"simulation r = {r_}: QS active AL neurons")
        ax.plot(x, ch, color=col, ls="--", lw=1.2, label=f"finite-K chain, K = {K:g} (r = {r_})")
        xm = np.concatenate([[0.9, 1.0], x[x > 1]]); mm = np.concatenate([[0, 0], mf[x > 1]])
        ax.plot(xm, mm, color=col, ls="-", lw=0.9, alpha=0.7, label=f"deterministic mean field, K → ∞ shape (r = {r_})")
        ax.axvline(xon[r_], color=col, ls=":", lw=1.5)
    ax.axvline(1.0, color="k", lw=2.4)
    ax.text(1.008, 45, "g_lin", fontsize=8.5)
    ax.text(xon[0] + 0.01, 2, "τ = 3000\n(r = 0)", fontsize=7.5)
    ax.text(xon[1] + 0.01, 2, "τ = 3000\n(r = 1)", fontsize=7.5, color=OK["blue"])
    ax.set_xlim(0.9, 2.3); ax.set_ylim(0, 95)
    ax.set_xlabel("x = g / g_lin"); ax.set_ylabel("active AL neurons (quasi-stationary mean)")
    ax.set_title("(a) Order parameter: continuous in mean field,\nrounded at finite K; 'onset' far above g_lin", loc="left")
    ax.legend(handles=[Line2D([], [], color="k", marker="o", ms=4, label="simulation: QS mean active AL neurons"),
                       Line2D([], [], color="k", ls="--", label="finite-K chain at K_needed (75.8 / 36.9)"),
                       Line2D([], [], color="k", lw=0.9, label="mean field A_r (x − 1) (K → ∞ normal form)"),
                       Patch(fc="k", label="r = 0"), Patch(fc=OK["blue"], label="r = 1")],
              fontsize=7, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    # (b) susceptibility
    ax = axs[0, 1]
    for key, col, r_ in (("real_r0", OK["black"], 0), ("real_r1", OK["blue"], 1)):
        rows = qs["C2"][key]["rows"]
        x = np.array([q["x"] for q in rows]); c = np.array([q["chi_AL"] for q in rows])
        ci = np.array([q["chi_AL_ci"] for q in rows], float)
        ch = np.array([q["chain_chi_neurons"] for q in rows], float)
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=col, alpha=0.15, lw=0)
        ax.plot(x, c, color=col, marker="o", ms=3.5, lw=1.4, label=f"simulation r = {r_}")
        ax.plot(x, ch, color=col, ls="--", lw=1.2, label=f"finite-K chain, K = {qs['C2'][key]['K']:g}")
        pk = qs["C2"][key]["chi_AL_peak"]
        ax.plot(pk["x"], pk["height"], marker="v", color=col, ms=8, ls="")
    ax.axvline(1.0, color="k", lw=2.4)
    ax.set_xscale("log"); ax.set_xlim(0.93, 3.6)
    ax.set_xticks([1, 1.5, 2, 3]); ax.set_xticklabels(["1", "1.5", "2", "3"])
    ax.set_xlabel("x = g / g_lin"); ax.set_ylabel("χ_AL = var(n_AL) / ⟨n_AL⟩")
    ax.set_title("(b) Fluctuations: one smooth, finite peak\n(no divergence; r = 0 at the chain's place, r = 1 18% above)",
                 loc="left")
    ax.legend(fontsize=7)
    # (c) growth-rate zero x*
    ax = axs[0, 2]
    fam = crit["critical_point"]["families"]; chp = crit["critical_point"]["chain_pred"]
    rows = [("real brain, r = 0", "real_r0", "real_r0_vw", "K75.8_real", None),
            ("real brain, r = 1", "real_r1", "real_r1_vw", "K36.6_real", "K36.6_real_netinput_closure_r1"),
            ("real brain, r = 6", "real_r6", "real_r6_vw", "K16.2_real", "K16.2_real_netinput_closure_r6"),
            ("one-mode control, K = 76", "complete76", None, "K76_complete", None),
            ("one-mode control, K = 37", "complete37", None, "K37_complete", None),
            ("configuration null, N = 167k, r = 0", "cnull_N167216_s0_r0", None, None, None)]
    for i, (lab, f, fv, c, cl) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = fam[f]["x_star_ci"]
        ax.errorbar(fam[f]["x_star"], y, xerr=[[fam[f]["x_star"] - lo], [hi - fam[f]["x_star"]]], color="k",
                    marker="o", ms=7, ls="", zorder=5)
        if fv:
            ax.plot(fam[fv]["x_star"], y, marker="^", color=OK["orange"], ms=8, ls="", zorder=5)
        if c:
            ax.plot(chp[c]["x_star"], y, marker="s", mfc="none", mec=OK["blue"], mew=1.8, ms=9, ls="", zorder=4)
        if cl:
            ax.plot(chp[cl]["x_star"], y, marker="X", color=OK["verm"], ms=9, ls="", zorder=4)
        ax.text(1.02, y + 0.25, lab, fontsize=7.8, ha="left")
    ax.axvline(1.0, color="k", lw=2.4)
    ax.set_yticks([]); ax.set_xlim(0.97, 2.2); ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("x* = zero of the growth rate of E[n(age)]  (x = g / g_lin)")
    ax.legend(handles=[Line2D([], [], color="k", marker="o", ls="", label="simulation, uniform seeds (pre-registered)"),
                       Line2D([], [], color=OK["orange"], marker="^", ls="", label="simulation, mode-weighted seeds (post hoc)"),
                       Line2D([], [], color=OK["blue"], marker="s", mfc="none", ls="", label="chain, exact linearisation (threshold at g_lin)"),
                       Line2D([], [], color=OK["verm"], marker="X", ls="", label="chain, net-input closure (threshold moves with r)")],
              fontsize=6.8, loc="upper right")
    ax.set_title("(c) Location: x* > 1 is finite-K saturation, as the chain\npredicts; threshold at g_lin "
                 "(r = 6: within ~5%)", loc="left")
    # (d) FSS onset vs N
    ax = axs[1, 0]
    for key, col, mk in (("cnull_r0", OK["black"], "o"), ("cnull_r1", OK["blue"], "s")):
        sm = fss["seed_means"][key]
        N = np.array([s["N"] for s in sm]); xm = np.array([s["x_on_mean"] for s in sm])
        ax.plot(N, xm - 1, color=col, marker=mk, ms=6, lw=1.6,
                label=f"configuration nulls, r = {key[-1]} (seed mean)")
        for s in sm:
            ax.plot([s["N"]] * len(s["x_on_per_seed"]), np.array(s["x_on_per_seed"]) - 1, color=col, marker=mk,
                    ms=3, ls="", alpha=0.4)
    for key, col, mk in (("realsub_r0", OK["black"], "o"), ("realsub_r1", OK["blue"], "s")):
        sm = fss["seed_means"][key]
        N = np.array([s["N"] for s in sm]); xm = np.array([s["x_on_mean"] for s in sm])
        ax.plot(N, xm - 1, color=col, marker=mk, ms=7, mfc=OK["verm"], lw=1.2, ls="--",
                label=f"real subsamples keeping the AL, r = {key[-1]}")
    for r_, col, mk in ((0, OK["black"], "o"), (1, OK["blue"], "s")):
        ax.plot(526, isl[f"AL_r{r_}"]["x_tau3000"] - 1, marker="*", ms=13, color=OK["verm"], mec=col, ls="")
        ax.plot(167216, xon[r_] - 1, marker=mk, ms=9, mfc="none", mec=OK["verm"], mew=2, ls="")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("network size N (neurons)"); ax.set_ylabel("x_on − 1   (onset at τ = 3000, x = g / g_lin)")
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 3]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "3"])
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [
        Line2D([], [], marker="*", color=OK["verm"], ls="", ms=11, label="AL module alone (N = 526)"),
        Line2D([], [], marker="o", mfc="none", mec=OK["verm"], ls="", ms=8, label="real full brain")], fontsize=6.8,
        loc="lower left")
    ax.set_title("(d) Finite size: onsets of delocalized nulls approach g_lin as N grows;\nthe real AL crossover "
                 "does not move from N = 526 to 167k", loc="left")
    # (e) <S>(T) local slope
    ax = axs[1, 1]
    curves = [("real_r0_x1.00", "real brain r = 0", OK["black"], "-", 2.0),
              ("real_r1_x1.00", "real brain r = 1", OK["blue"], "-", 2.0),
              ("module_r0_x1.00", "AL module alone r = 0", OK["orange"], "-", 1.6),
              ("cnull_N167216_s0_r0_x1.00", "configuration null 167k r = 0", OK["purple"], "-", 1.6),
              ("complete37_x1.00", "one-mode control K = 37", "0.70", "--", 1.3),
              ("complete76_x1.00", "one-mode control K = 76", "0.55", "--", 1.3),
              ("complete300_x1.00", "one-mode control K = 300", "0.40", "--", 1.3),
              ("complete2000_x1.00", "one-mode control K = 2000", "0.20", "--", 1.3)]
    for k, lab, col, ls, lw in curves:
        sl = np.array(crit["runs"][k]["ST"]["slope"], float)
        ok = np.isfinite(sl[:, 1])
        ax.plot(sl[ok, 0], sl[ok, 1], color=col, ls=ls, lw=lw, label=f"{lab} (peak {crit['runs'][k]['ST']['peak']:.2f})")
    ax.axhline(2.0, color="k", lw=1.0, ls=":")
    ax.text(0.45, 2.05, "MF-DP asymptote ⟨S⟩ ∝ T²", fontsize=7.5, zorder=10)
    ax.set_xscale("log"); ax.set_xlim(0.4, 100); ax.set_ylim(0, 2.9)
    ax.set_xlabel("avalanche duration T (model time)"); ax.set_ylabel("local slope d ln⟨S⟩ / d ln T")
    ax.legend(fontsize=6.5, loc="lower right", ncol=1)
    ax.set_title("(e) Universality: avalanches at g_lin grow faster than one-mode\nMF-DP (peak 2.1-2.35 vs 1.57-1.92); "
                 "nulls even more", loc="left")
    # (f) hysteresis loop areas vs dwell
    ax = axs[1, 2]
    rlist = [0, 0.5, 1, 2, 4, 6]
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(rlist)))
    Dw = np.array([50, 200, 800])
    for rr, col in zip(rlist, cols):
        k = f"r{rr:g}_h1e-05_area_all"
        v = m2a["decisions"][k]
        A = np.array([v["A50"], v["A200"], v["A800"]])
        ax.plot(Dw, A, color=col, marker="o", lw=1.6, label=f"r = {rr:g}")
        ax.errorbar(800 * (1 + 0.04 * rlist.index(rr)), v["A800"], yerr=v["SE800"], color=col, ls="", capsize=2)
        ax.plot([600, 1060], [v["floor800"]] * 2, color=col, lw=2.6, alpha=0.6)
    ax.text(1100, 0.065, "noise floors\n(D = 800)", fontsize=7.5, va="top")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(40, 1800)
    ax.set_xticks([50, 200, 800]); ax.set_xticklabels(["50", "200", "800"])
    ax.set_xlabel("dwell per g step D (model time)"); ax.set_ylabel("loop area ∫|ln ρ_up − ln ρ_down| d ln g")
    ax.legend(fontsize=7, title="h = 1e-5", title_fontsize=7.5, loc="lower left")
    ax.set_title("(f) First-order test: sweep loops shrink 6.5-15.5x from D = 50\nto 800; r = 0 still open at 800 "
                 "but shrinking", loc="left")
    fig.suptitle("Is the steep but smooth crossover a second-order transition?  Evidence from the four pd3 programs "
                 "(x = g / g_lin; thick black line = g_lin)", fontsize=12.5)
    out = f"{FIG}/q2_second_order_evidence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


# ================================================================================== figure 3: finite-size summary
def figure_fss():
    fig, axs = plt.subplots(2, 2, figsize=(14, 10.5))
    fig.subplots_adjust(left=0.07, right=0.88, top=0.92, bottom=0.07, wspace=0.25, hspace=0.3)
    Ns = sorted({g["N"] for g in fss["per_graph"] if g["kind"] == "cnull"})
    norm = matplotlib.colors.LogNorm(vmin=min(Ns), vmax=max(Ns))
    cmap = plt.cm.viridis
    for ax, r_ in ((axs[0, 0], 0), (axs[0, 1], 1)):
        for gph in fss["per_graph"]:
            if gph["r"] != r_:
                continue
            pts = sorted(gph["points"], key=lambda p: p["x"])
            x = np.array([p["x"] for p in pts]); t = np.array([p["tau_sim"] if p["deaths"] > 0 else np.nan for p in pts])
            if gph["kind"] == "cnull":
                ax.plot(x, t, color=cmap(norm(gph["N"])), lw=1.0, marker=".", ms=3, alpha=0.9)
            else:
                ax.plot(x, t, color=OK["verm"], lw=1.6, marker="o", ms=3, alpha=0.9, zorder=5)
        # real full brain (pooled simulation, phase_complete)
        pts = pc["sim"]["real"][f"{r_}"]["points"]
        xg = np.array([p["g"] / GLIN for p in pts]); tg = np.array([p["tau_sim"] if p["deaths"] > 0 else np.nan for p in pts])
        ax.plot(xg, tg, ls="", marker="o", ms=4, color="k", zorder=6)
        ax.plot(isl[f"AL_r{r_}"]["x"], np.where(np.array(isl[f"AL_r{r_}"]["deaths"]) > 0, isl[f"AL_r{r_}"]["tau"], np.nan),
                ls="--", color="k", lw=1.2, zorder=6)
        ax.axhline(3000, color="0.3", lw=1, ls=":"); ax.axhline(200, color="0.3", lw=1, ls=":")
        ax.text(1.03, 3400, "τ = 3000", fontsize=8); ax.text(1.03, 225, "τ = 200", fontsize=8)
        ax.axvline(1.0, color="k", lw=2.4)
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_xlim(0.9, 2.5 if r_ == 0 else 5.0); ax.set_ylim(8, 3e5)
        tk = [0.9, 1, 1.2, 1.5, 2, 2.5] if r_ == 0 else [0.9, 1, 1.5, 2, 3, 4, 5]
        ax.set_xticks(tk); ax.set_xticklabels([f"{v:g}" for v in tk])
        ax.set_xlabel("x = g / g_lin (each graph's own exact g_lin)"); ax.set_ylabel("lifetime τ_sim from fully active (model time)")
        ax.set_title(f"({'a' if r_ == 0 else 'b'}) r = {r_}: lifetime curves (points with ≥ 1 death)"
                     + ("" if r_ == 0 else "; legend as in (a)"), loc="left")
        if r_ == 0:
            ax.legend(handles=[Line2D([], [], color=cmap(0.0), lw=1, label="configuration nulls (colour = N)"),
                           Line2D([], [], color=OK["verm"], lw=1.6, marker="o", ms=3, label="real subsamples keeping the AL (5k, 20k, 80k)"),
                           Line2D([], [], color="k", ls="", marker="o", ms=4, label="real full brain (167k)"),
                           Line2D([], [], color="k", ls="--", lw=1.2, label="AL module alone (526)")],
                      fontsize=7.5, loc="lower right")
    cax = fig.add_axes([0.90, 0.56, 0.015, 0.34])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("null size N (neurons)")
    # (c) onset vs N
    ax = axs[1, 0]
    for key, col, mk, lab in (("cnull_r0", OK["black"], "o", "nulls r = 0"), ("cnull_r1", OK["blue"], "s", "nulls r = 1"),
                              ("realsub_r0", OK["verm"], "o", "real subsamples r = 0"),
                              ("realsub_r1", OK["verm"], "s", "real subsamples r = 1")):
        sm = fss["seed_means"][key]
        N = np.array([s["N"] for s in sm]); xm = np.array([s["x_on_mean"] for s in sm])
        sd = np.array([s["x_on_seed_sd"] for s in sm])
        ax.errorbar(N, xm - 1, yerr=sd, color=col, marker=mk, ms=6, lw=1.5, capsize=2, label=lab + " (seed mean ± SD)",
                    mfc=col if key.startswith("cnull") else "white")
    for r_, mk in ((0, "o"), (1, "s")):
        ax.plot(526, isl[f"AL_r{r_}"]["x_tau3000"] - 1, marker="*", ms=13, color=OK["verm"], ls="")
        ax.plot(167216, ON[R == r_][0] / GLIN - 1, marker=mk, ms=10, mfc="none", mec="k", mew=1.8, ls="")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 3]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "3"])
    ax.set_xlabel("network size N (neurons)"); ax.set_ylabel("x_on − 1  (τ = 3000 onset)")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [
        Line2D([], [], marker="*", color=OK["verm"], ls="", ms=11, label="AL module alone (526)"),
        Line2D([], [], marker="o", mfc="none", mec="k", ls="", ms=8, label="real full brain")], fontsize=7.5, loc="lower left")
    ax.set_title("(c) Distance of the onset from g_lin: nulls shrink with N\n(Spearman −1 at r = 0 and 1); the real "
                 "crossover stays put (within 1.8%)", loc="left")
    # (d) survivor fluctuations at the crossover
    ax = axs[1, 1]
    for key, col, mk, lab in (("cnull_r0", OK["black"], "o", "nulls r = 0"), ("cnull_r1", OK["blue"], "s", "nulls r = 1"),
                              ("realsub_r0", OK["verm"], "o", "real subsamples r = 0"),
                              ("realsub_r1", OK["verm"], "s", "real subsamples r = 1")):
        h = fss["supplementary_chi_win"][key]["heights_by_N"]
        N = np.array(sorted(int(n) for n in h)); hm = np.array([np.mean(h[str(n)]) for n in N])
        for n in N:
            ax.plot([n] * len(h[str(n)]), h[str(n)], color=col, marker=mk, ms=3, ls="", alpha=0.45)
        sl = fss["supplementary_chi_win"][key]["loglog_slope_vs_N"]; se = fss["supplementary_chi_win"][key]["se"]
        ax.plot(N, hm, color=col, marker=mk, ms=6, lw=1.5, mfc=col if key.startswith("cnull") else "white",
                label=f"{lab}: slope {sl:.2f} ± {se:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("network size N (neurons)"); ax.set_ylabel("max χ_surv = var(n)/⟨n⟩ in [x_S, 1.25 x_on]")
    ax.legend(fontsize=7.5, loc="upper left", title="effective log-log slope vs N (≤ 1.8 decades)", title_fontsize=7.5)
    ax.set_title("(d) Fluctuations at the crossover grow with N in the r = 0 nulls,\nbarely in the real subsamples "
                 "(supplementary, not pre-registered)", loc="left")
    fig.suptitle("Finite-size check: configuration nulls sharpen toward g_lin as N grows; the real AL crossover, set "
                 "by a fixed-size module, does not", fontsize=12.5)
    out = f"{FIG}/fss_summary.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


# ============================================================================================ numbers for REPORT
def collect_numbers():
    fam = crit["critical_point"]["families"]; chp = crit["critical_point"]["chain_pred"]
    for f in ("real_r0", "real_r1", "real_r6", "real_r0_vw", "real_r1_vw", "real_r6_vw", "complete37", "complete76",
              "cnull_N167216_s0_r0"):
        rec(f"xstar_{f}", [round(fam[f]["x_star"], 4)] + [round(v, 4) for v in fam[f]["x_star_ci"]],
            f"results/pd3/criticality/aval_analysis.json critical_point.families.{f}.x_star(_ci)")
    for c in chp:
        rec(f"xstar_chain_{c}", round(chp[c]["x_star"], 4), f"aval_analysis.json critical_point.chain_pred.{c}.x_star")
    for k in ("real_r0_x1.00", "real_r1_x1.00", "real_r6_x1.00", "module_r0_x1.00", "cnull_N167216_s0_r0_x1.00",
              "complete37_x1.00", "complete76_x1.00", "complete300_x1.00", "complete2000_x1.00"):
        rec(f"ST_peak_{k}", round(crit["runs"][k]["ST"]["peak"], 3), f"aval_analysis.json runs.{k}.ST.peak")
    for key in ("real_r0", "real_r1", "real_r6"):
        v = qs["C2"][key]
        rec(f"QS_{key}", {"chi_AL_peak_g": round(v["chi_AL_peak"]["g"], 4), "chain_chi_peak_g": round(v["chain_chi_peak"]["g"], 4),
                          "beta": round(v["beta"], 3), "beta_chain": round(v["beta_chain_same_window"], 3),
                          "all_unimodal": v["all_unimodal"], "K": v["K"]},
            f"results/pd3/criticality/qs_analysis.json C2.{key}")
    for key in ("cnull_r0", "cnull_r1", "realsub_r0", "realsub_r1"):
        rec(f"fss_seed_means_{key}", {str(s["N"]): round(s["x_on_mean"], 4) for s in fss["seed_means"][key]},
            f"results/pd3/redraw_fss/fss_analysis.json seed_means.{key}[].x_on_mean")
        rec(f"fss_chi_win_slope_{key}", [round(fss["supplementary_chi_win"][key]["loglog_slope_vs_N"], 3),
                                         round(fss["supplementary_chi_win"][key]["se"], 3)],
            f"fss_analysis.json supplementary_chi_win.{key}")
    for r_ in (0, 1):
        rec(f"AL_alone_x_tau3000_r{r_}", round(isl[f"AL_r{r_}"]["x_tau3000"], 4),
            f"results/pd3/module_theory/island_lifetimes.json AL_r{r_}.x_tau3000")
        rec(f"AL_alone_tau_x1_r{r_}", round(isl[f"AL_r{r_}"]["tau_x1"], 2), f"island_lifetimes.json AL_r{r_}.tau_x1")
    for rr in (0, 0.5, 1, 2, 4, 6):
        v = m2a["decisions"][f"r{rr:g}_h1e-05_area_all"]
        rec(f"m2a_loop_r{rr:g}_h1e-5", {"A50": round(v["A50"], 3), "A200": round(v["A200"], 3), "A800": round(v["A800"], 3),
                                        "SE800": round(v["SE800"], 3), "floor800": round(v["floor800"], 3),
                                        "fold_50_to_800": round(v["A50"] / v["A800"], 1)},
            f"results/pd3/multistab/analysis_m2a.json decisions.r{rr:g}_h1e-05_area_all")
    rec("qmf_SL_edge_up_sweep", {f"{r_:g}": round(float(e), 4) for r_, e in zip(QR, QSL)},
        "results/pd3/module_theory/qmf_analysis.json per_r[r].SL_edge_up_sweep")
    rec("theory_onset_P2_r0_r1_r6", [round(float(TH_ON[R == v][0]), 4) for v in (0, 1, 6)],
        "phase_complete.json theory.real[r].on.g")
    rec("theory_on_over_sim_on_r0_r1_r6", [round(float(TH_ON[R == v][0] / ON[R == v][0]), 3) for v in (0, 1, 6)],
        "computed from phase_complete.json theory/sim on")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    figure_phase_diagram()
    figure_q2()
    figure_fss()
    collect_numbers()
    with open(f"{ROOT}/results/pd3/synthesis.json", "w") as f:
        json.dump({"generated_by": "code/pd3_synthesis.py", "date": "2026-11-09", "numbers": NUM}, f, indent=1)
    print("wrote results/pd3/synthesis.json with", len(NUM), "entries")
