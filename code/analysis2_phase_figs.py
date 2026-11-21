"""Figures for the complete phase diagram, from results/analysis2/phase_complete.json (code/analysis2_phase.py).
  figs/pd2/phase_real.png        (g, r) plane, MaleCNS: theory regions + curves, simulated boundaries with uncertainty
  figs/pd2/phase_nulls.png       same for the configuration null and the weight-shuffled null
  figs/pd2/phase_driven.png      (g, h) plane at r = 1: simulated density and fluctuation maps + theory P5 peak curves
  figs/pd2/tau_sim_vs_theory.png simulated vs predicted module lifetime for every (g, r) with tau_sim > 200
Colours: boundary type = categorical slots blue / orange / aqua (validated with the dataviz palette checker: CVD and
normal-vision separation pass; aqua is below 3:1 contrast, so every curve is also labelled and has its own marker);
magnitudes = single-hue ramps.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/analysis2"; FIG = f"{ROOT}/figs/pd2"
J = json.load(open(f"{OUT}/phase_complete.json"))
T = J["T"]
C = {"S": "#2a78d6", "on": "#eb6834", "D": "#1baf7a"}
REG = {"Q": "#eeeeec", "S": "#d6e6fa", "L": "#fde2d5", "D": "#d2f0e3"}
MK = {"S": "o", "on": "s", "D": "D"}
LAB = {"S": "S starts (τ = 200)", "on": "sustained onset (τ = T = 3000)", "D": "delocalized (PR = N/4)"}
BLUE = LinearSegmentedColormap.from_list("blue", ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ORNG = LinearSegmentedColormap.from_list("orng", ["#fde2d5", "#f7b08f", "#eb6834", "#b8431a", "#7a2a0e"])
BLUE_SEQ = LinearSegmentedColormap.from_list("blueseq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "legend.frameon": False})


def theory_curves(kind, gmax_edge):
    th = J["theory"][kind]
    rs = sorted(th, key=float)
    out = {}
    for key in ["S", "on", "D"]:
        rr, gg = [], []
        for rk in rs:
            e = th[rk].get(key)
            if e is None:
                continue
            g = e.get("g")
            rr.append(float(rk)); gg.append(g if g is not None else np.nan)
        out[key] = (np.array(rr), np.array(gg, float))
    return out


def draw_plane(ax, kind, xlim, title, region_labels=True):
    tc = theory_curves(kind, xlim[1])
    rfine = np.linspace(0, 6, 601)
    def interp(key, fill_right):
        r, g = tc[key]
        ok = np.isfinite(g)
        if not ok.any():
            return np.full_like(rfine, fill_right)
        lg = np.where(ok, np.log(np.where(ok, g, 1)), np.log(fill_right))
        return np.exp(np.interp(rfine, r, lg))
    edge_r = xlim[1] * 1.5
    gS, gon, gD = interp("S", edge_r), interp("on", edge_r), interp("D", edge_r)
    gD = np.maximum(gD, gon)
    x0 = xlim[0] / 2
    ax.fill_betweenx(rfine, x0, gS, color=REG["Q"], lw=0)
    ax.fill_betweenx(rfine, gS, gon, color=REG["S"], lw=0)
    ax.fill_betweenx(rfine, gon, gD, color=REG["L"], lw=0)
    ax.fill_betweenx(rfine, gD, edge_r, color=REG["D"], lw=0)
    for key in ["S", "on", "D"]:
        r, g = tc[key]
        ok = np.isfinite(g)
        ax.plot(g[ok], r[ok], "-", color=C[key], lw=2.2, zorder=3)
    # critic: QMF D is only known to its grid cell (32 log points in 0.1-30, 20% steps); show the cell as a bar
    for rk, e in J["theory"][kind].items():
        d = e.get("D") or {}
        b = d.get("grid_bracket")
        if d.get("g") is None or not b or b[0] is None:
            continue
        ax.plot([b[0], b[1]], [float(rk)] * 2, color=C["D"], lw=5, alpha=0.55, solid_capstyle="butt", zorder=2.5)
    gl = J["g_lin"][kind]
    ax.axvline(gl, color=INK, lw=1.0, ls=":", zorder=2)
    ax.text(gl * 1.03, 6.2, f"g_lin = {gl:.3g}", ha="left", va="center", fontsize=8.5, color=INK)
    # simulation
    sim = J["sim"][kind]
    for key in ["S", "on", "D"]:
        rr, gg, lo, hi = [], [], [], []
        for rk in sorted(sim, key=float):
            e = sim[rk][key]
            if e["g"] is None:
                if key == "D" and e["status"] in ("none", "no_sustained_points"):
                    gmax = e["bracket"][0] if e["bracket"][0] else sim[rk]["g_range"][1]
                    ax.plot(min(gmax, xlim[1] * 0.97), float(rk), marker=">", ms=7, mfc="white", mec=C["D"], mew=1.4,
                            zorder=5, ls="none")
                continue
            tot = e["total"] or [e["g"], e["g"]]
            rr.append(float(rk)); gg.append(e["g"])
            lo.append(e["g"] - (tot[0] if tot[0] else e["g"])); hi.append((tot[1] if tot[1] else e["g"]) - e["g"])
        if not rr:
            continue
        rr, gg = np.array(rr), np.array(gg)
        ax.plot(gg, rr, "--", color=C[key], lw=1.3, zorder=4)
        if key == "D":                         # D at the onset: hollow, larger, so the onset square stays visible
            at = np.array([sim[rk]["D"]["status"] == "at_onset" for rk in sorted(sim, key=float) if sim[rk]["D"]["g"] is not None])
            if at.any():
                ax.plot(gg[at], rr[at], "D", ms=11, mfc="none", mec=C["D"], mew=2.0, zorder=7, ls="none")
            if (~at).any():
                ax.errorbar(gg[~at], rr[~at], xerr=[np.array(lo)[~at], np.array(hi)[~at]], fmt=MK[key], ms=5.5,
                            color=C[key], mec=INK, mew=0.6, elinewidth=1.2, capsize=2.5, zorder=6, ls="none")
            continue
        ax.errorbar(gg, rr, xerr=[lo, hi], fmt=MK[key], ms=5.5, color=C[key], mec=INK, mew=0.6, elinewidth=1.2,
                    capsize=2.5, zorder=6, ls="none")
    ax.set_xscale("log"); ax.set_xlim(*xlim); ax.set_ylim(-0.15, 6.35)
    ax.set_xlabel("coupling g  (raw synapse-count weights / mean in-strength)")
    ax.set_ylabel("inhibition ratio r")
    ax.set_title(title, fontsize=11, loc="left")
    return gS, gon, gD, rfine


def legend_handles(include_regions=True):
    h = [Line2D([], [], color=C[k], lw=2.2, label=f"theory: {LAB[k]}") for k in ["S", "on", "D"]]
    h += [Line2D([], [], color=C["D"], lw=5, alpha=0.55, label="theory D: QMF grid cell (20% steps in g)")]
    h += [Line2D([], [], color=C[k], lw=1.3, ls="--", marker=MK[k], mec=INK, mew=0.6, ms=5.5,
                 label=f"simulation: {LAB[k]}") for k in ["S", "on", "D"]]
    h += [Line2D([], [], color=C["D"], marker="D", mfc="none", mew=2.0, ls="none", ms=10,
                 label="simulation: delocalized already at the onset (no L phase)")]
    h += [Line2D([], [], color=C["D"], marker=">", mfc="white", mew=1.4, ls="none", ms=7,
                 label="simulation: not delocalized up to the largest simulated g")]
    h += [Line2D([], [], color=INK, ls=":", lw=1, label="g_lin (silence unstable, theory P1)")]
    if include_regions:
        h += [Patch(color=REG[k], label=f"theory region {k}: {n}") for k, n in
              [("Q", "quiescent"), ("S", "slow window"), ("L", "localized active"), ("D", "delocalized active")]]
    return h


# ------------------------------------------------------------------------------------------------------- phase_real.png
fig, ax = plt.subplots(figsize=(11.5, 6.6))
gS, gon, gD, rf = draw_plane(ax, "real", (0.12, 32), "MaleCNS: theory (regions, solid) vs simulation (points, dashed), T = 3000")
for txt, x, y in [("Q", 0.15, 3.0), ("S", np.sqrt(gS[300] * gon[300]), 3.0), ("L", 3.0, 3.0), ("D", 20, 0.12)]:
    ax.text(x, y, txt, fontsize=15, fontweight="bold", color=INK2, ha="center", va="center", alpha=0.8)
ax.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
ax.text(1.01, 0.02, "error bars: envelope of grid bracket\nand replica-bootstrap 95% CI", transform=ax.transAxes,
        fontsize=8, color=INK2, va="bottom")
fig.tight_layout(); fig.savefig(f"{FIG}/phase_real.png"); plt.close(fig)

# ------------------------------------------------------------------------------------------------------ phase_nulls.png
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
for ax, kind in zip(axs, ["null", "wnull"]):
    gS, gon, gD, rf = draw_plane(ax, kind, (0.4, 32), {"null": "Configuration null (wiring shuffled)",
                                                       "wnull": "Weight-shuffled null (synapse counts shuffled)"}[kind])
    ax.text(0.5, 3.0, "Q", fontsize=15, fontweight="bold", color=INK2, ha="center", va="center", alpha=0.8)
    ax.text(15, 5.0, "L", fontsize=15, fontweight="bold", color=INK2, ha="center", va="center", alpha=0.8)
    ax.text(20, 0.15, "D", fontsize=15, fontweight="bold", color=INK2, ha="center", va="center", alpha=0.8)
    ax.text(2.2, 4.4, "theory L band: the branching theory says sustained,\nbut the QMF fixed point is still silent or\n"
            "below PR = N/4 (mean-field inhibition)", fontsize=8.5, color=INK2, ha="left", va="center")
axs[1].set_ylabel("")
fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(f"{FIG}/phase_nulls.png"); plt.close(fig)

# ---------------------------------------------------------------------------------------------- tau_sim_vs_theory.png
fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.3))
norm = Normalize(0, 6)
counts = {}
for ax, kind in zip(axs, ["real", "null", "wnull"]):
    n_fin = n_cens = n_off = n_few = 0
    for rk, e in sorted(J["sim"][kind].items(), key=lambda kv: float(kv[0])):
        r = float(rk)
        for p in e["points"]:
            lt = p["ln_tau_theory"]
            if lt is None:
                continue
            x = np.exp(min(lt, 700))
            if x > 1e10:
                n_off += 1; continue
            if p["deaths"] >= 3 and p["tau_sim"] is not None and p["tau_sim"] > 200:
                lo, hi = p["tau_ci"]
                ax.plot([x, x], [lo, hi if hi else lo], color="#c3c2b7", lw=0.8, zorder=1)
                ax.scatter([x], [p["tau_sim"]], c=[r], cmap=BLUE, norm=norm, s=26, edgecolors=INK, linewidths=0.4,
                           zorder=3)
                n_fin += 1
            elif 0 < p["deaths"] < 3 and p["tau_sim"] is not None and p["tau_sim"] > 200:
                ax.plot(x, p["tau_sim"], marker="o", ms=4, mfc="none", mec="#a8a7a0", ls="none", zorder=2)
                n_few += 1
            elif p["deaths"] == 0 and x < T:      # theory says it should have died by T; no replica did
                ax.plot(x, p["tau_ci"][0], marker="^", ms=5, color="#8a8984", ls="none", zorder=2)
                n_cens += 1
    counts[kind] = (n_fin, n_few, n_cens, n_off)
    xx = np.array([1, 1e10])
    ax.plot(xx, xx, color=INK, lw=1.2)
    for f, ls in [(10, "--"), (100, ":")]:
        ax.plot(xx, f * xx, color=INK2, lw=0.9, ls=ls)
    ax.axhline(200, color="#d0cfc9", lw=0.8, zorder=0); ax.axhline(T, color="#d0cfc9", lw=0.8, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(3, 1e6); ax.set_ylim(150, 3e5)
    ax.set_xlabel("theory τ (exact birth–death recursion, same g and r)")
    ax.set_title(f"{ {'real': 'MaleCNS', 'null': 'configuration null', 'wnull': 'weight-shuffled null'}[kind]}  "
                 f"({n_fin} points with ≥ 3 deaths)", fontsize=10, loc="left")
axs[0].set_ylabel("simulated τ_sim (censored-exponential MLE, T = 3000)")
sm = plt.cm.ScalarMappable(cmap=BLUE, norm=norm)
cb = fig.colorbar(sm, ax=axs, fraction=0.02, pad=0.01); cb.set_label("inhibition ratio r")
handles = [Line2D([], [], color=INK, lw=1.2, label="τ_sim = τ_theory"),
           Line2D([], [], color=INK2, lw=0.9, ls="--", label="τ_sim = 10 τ_theory"),
           Line2D([], [], color=INK2, lw=0.9, ls=":", label="τ_sim = 100 τ_theory"),
           Line2D([], [], color="#c3c2b7", lw=0.8, label="95% CI of τ_sim"),
           Line2D([], [], marker="o", mfc="none", mec="#a8a7a0", ls="none", label="1–2 deaths (τ_sim very uncertain)"),
           Line2D([], [], marker="^", color="#8a8984", ls="none", label="no deaths by T although τ_theory < T:\n95% lower bound of τ_sim")]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.45, -0.1))
fig.savefig(f"{FIG}/tau_sim_vs_theory.png", bbox_inches="tight"); plt.close(fig)
print("tau plot counts (>=3 deaths, 1-2 deaths, censored with tau_th < T, theory > 1e10 not shown):", counts)

# ---------------------------------------------------------------------------------------------------- phase_driven.png
# (critic 2026-11-09: 2 x 2 layout; added (d) simulated density vs the driven QMF mean field (results/pd/qmf_drive_real.npz)
#  and the configuration-null h = 1e-3 fluctuation curve in (c); h tick labels as 1e-5 ... 1e-1)
D = J["driven"]
hs = sorted([k for k in D["runs"] if D["runs"][k].get("graph", "real") == "real"], key=lambda k: D["runs"][k]["h"])
hv = np.array([D["runs"][k]["h"] for k in hs])
hlab = lambda x: f"1e{int(round(np.log10(x)))}"
fig, axs2 = plt.subplots(2, 2, figsize=(15.5, 11))
axs = [axs2[0, 0], axs2[0, 1], axs2[1, 0], axs2[1, 1]]
he = np.r_[hv[0] / np.sqrt(10), np.sqrt(hv[1:] * hv[:-1]), hv[-1] * np.sqrt(10)]
for ax, key, cmap, lab in [(axs[0], "rho", BLUE_SEQ, "stationary fraction active ρ"),
                           (axs[1], "F_rep", ORNG, "replica fluctuation χ_rep / ρ  (variance-to-mean)")]:
    allv = [np.array(D["runs"][k][key]) for k in hs]
    vmin = min(np.nanmin(v[v > 0]) for v in allv); vmax = max(np.nanmax(v) for v in allv)
    nrm = LogNorm(vmin, vmax)
    for j, k in enumerate(hs):
        g = np.array(D["runs"][k]["g"]); v = np.array(D["runs"][k][key])
        ge = np.r_[g[0] / np.sqrt(g[1] / g[0]), np.sqrt(g[1:] * g[:-1]), g[-1] * np.sqrt(g[-1] / g[-2])]
        im = ax.pcolormesh(ge, [he[j], he[j + 1]], v[None, :], cmap=cmap, norm=nrm, shading="flat")
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label(lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("coupling g"); ax.set_ylabel("external drive h")
    ax.set_yticks(hv); ax.set_yticklabels([hlab(x) for x in hv]); ax.minorticks_off()
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2"])
for ax in axs[:2]:
    p = D["p5_theory_r1"]; ax.plot(p["g"], p["h"], color=INK, lw=2.0)
    p = D["p5_measured_tau"]; ax.plot(p["g"], p["h"], color=INK, lw=2.0, ls="--")
    ax.axvline(J["g_lin"]["real"], color=INK, lw=1, ls=":")
    xl = (min(min(D["runs"][k]["g"]) for k in hs) / 1.05, max(max(D["runs"][k]["g"]) for k in hs) * 1.05)
    ax.set_xlim(*xl); ax.set_ylim(he[0], he[-1])
    for j, k in enumerate(hs):
        run = D["runs"][k]
        if run["peak_resolved"]:
            cell = run["peak_cell"]
            ax.errorbar([run["peak_g"]], [run["h"]], xerr=[[run["peak_g"] - cell[0]], [cell[1] - run["peak_g"]]],
                        fmt="o", ms=7, mfc="white", mec=INK, ecolor=INK, mew=1.3, capsize=3, zorder=6)
axs[0].set_title("(a) simulated density (MaleCNS, r = 1, T = 1000, late half)", fontsize=10, loc="left")
axs[1].set_title("(b) simulated fluctuations with theory P5 peak curves", fontsize=10, loc="left")
for j, k in enumerate(hs):
    run = D["runs"][k]
    c = BLUE(j / max(len(hs) - 1, 1))
    axs[2].plot(run["g"], run["F_rep"], "-o", ms=3, lw=1.4, color=c, label=f"MaleCNS, h = {hlab(run['h'])}")
    if run["peak_resolved"]:
        axs[2].plot(run["peak_g"], run["peak_F"], marker="o", ms=9, mfc="none", mec=INK, mew=1.3)
if "null_1e-3" in D["runs"]:
    rn = D["runs"]["null_1e-3"]
    axs[2].plot(rn["g"], rn["F_rep"], "--s", ms=3, lw=1.3, color="#8a8984", label="configuration null, h = 1e-3")
axs[2].axvline(J["g_lin"]["real"], color=INK, lw=1, ls=":")
axs[2].axvspan(0.1, 1.0, color="#f0efec", zorder=0)
axs[2].set_xscale("log"); axs[2].set_yscale("log")
axs[2].set_xlabel("coupling g"); axs[2].set_ylabel("χ_rep / ρ  (variance-to-mean across replicas)")
axs[2].set_title("(c) fluctuation curves (shaded: MaleCNS peak-search window)", fontsize=10, loc="left")
axs[2].legend(fontsize=8, loc="lower right")
# (d) density, simulation vs driven QMF (same graph, r, T, dt; QMF value at the end of the run)
qd = np.load(f"{ROOT}/results/pd/qmf_drive_real.npz")
for j, k in enumerate(hs):
    run = D["runs"][k]; c = BLUE(j / max(len(hs) - 1, 1))
    axs[3].plot(run["g"], run["rho"], "-o", ms=3, lw=1.4, color=c, label=f"simulation, h = {hlab(run['h'])}")
    jj = int(np.argmin(np.abs(np.log(qd["h"]) - np.log(run["h"]))))
    axs[3].plot(qd["g"], qd["rho_T"][0, jj], "--", lw=1.6, color=c)
axs[3].plot([], [], "--", color=INK2, lw=1.6, label="QMF mean field (same h, dashed)")
axs[3].axvline(J["g_lin"]["real"], color=INK, lw=1, ls=":")
axs[3].set_xscale("log"); axs[3].set_yscale("log"); axs[3].set_xlim(0.09, 3.3)
axs[3].set_xlabel("coupling g"); axs[3].set_ylabel("stationary fraction active ρ")
axs[3].set_title("(d) density: simulation vs quenched mean field (theory)", fontsize=10, loc="left")
axs[3].legend(fontsize=8, loc="lower right", ncol=1)
for ax in axs[2:]:
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2"])
hh = [Line2D([], [], color=INK, lw=2, label="theory P5 peak, τ from theory (r = 1), n = 526"),
      Line2D([], [], color=INK, lw=2, ls="--", label="theory P5 peak, measured h = 0 τ_sim, n = 526"),
      Line2D([], [], color=INK, ls=":", lw=1, label="g_lin (theory P1)"),
      Line2D([], [], marker="o", mfc="white", mec=INK, color=INK, ls="none", ms=7,
             label="simulated χ_rep/ρ peak, resolved in 0.1 ≤ g ≤ 1 (bar: grid cell)")]
fig.legend(handles=hh, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.045))
src = sorted(set(D["runs"][k]["file"] for k in D["runs"])) + ["pd/qmf_drive_real.npz"]
fig.text(0.01, -0.07, "data: " + ", ".join(src), fontsize=7.5, color=INK2)
fig.tight_layout(); fig.savefig(f"{FIG}/phase_driven.png", bbox_inches="tight"); plt.close(fig)
print("figures written to", FIG)
