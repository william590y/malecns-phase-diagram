"""Critic figure figs/pd2/critic_boundaries.png from results/analysis2/critic.json + phase_complete.json (MaleCNS, h = 0).
(a) matched tau criteria (tau = 200 / 3000): model-free 95% data brackets of the simulated crossing (bars), the assembler's
    interpolated simulated boundary (markers), the branching theory (solid) and the pair-operator theory variant (hollow).
(b) the operational phase.py criteria on the B = 32 pd2 runs (S_op: last replica dies after t = 200; L_op: all 32 replicas
    alive at T at this and every larger g), as grid cells (bars), against the theory evaluated at the tau these
    B-dependent criteria correspond to (S_op <-> tau 52, L_op <-> tau 1.38e5 for exponential lifetimes).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "/MALECNS_ROOT"
C = json.load(open(f"{ROOT}/results/analysis2/critic.json"))
J = json.load(open(f"{ROOT}/results/analysis2/phase_complete.json"))
COL = {"S": "#2a78d6", "on": "#eb6834"}
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "legend.frameon": False})
fig, axs = plt.subplots(1, 2, figsize=(14.5, 7.6), sharey=True)
ax = axs[0]
rs = sorted(C["sim"]["real"], key=float)
for key, bk in [("S", "S_data_bracket"), ("on", "on_data_bracket")]:
    off = -0.03 if key == "S" else 0.03
    for rk in rs:
        b = C["sim"]["real"][rk][bk]
        if b["lo"] and b["hi"]:
            ax.plot([b["lo"], b["hi"]], [float(rk) + off] * 2, color=COL[key], lw=4, alpha=0.45, solid_capstyle="butt")
    ax.plot([J["sim"]["real"][rk][key]["g"] for rk in rs], [float(rk) + off for rk in rs], "o" if key == "S" else "s",
            ms=4.5, color=COL[key], mec=INK, mew=0.5, ls="none")
    th = C["theory"]["real"]["per_r"]
    rr = sorted(th, key=float)
    ax.plot([th[r][f"{key}_out"]["g"] for r in rr], [float(r) for r in rr], "-", color=COL[key], lw=2)
    pv = C["theory_versions"]["branch_real.json:pair"]["per_r"]
    pts = [(float(r), pv[r][key if key == "S" else "on"]["g"]) for r in pv if pv[r][key if key == "S" else "on"]["g"]]
    ax.plot([p[1] for p in pts], [p[0] for p in pts], "D", ms=6, mfc="white", mec=COL[key], mew=1.5, ls="none")
ax.axvline(J["g_lin"]["real"], color=INK, ls=":", lw=1)
ax.set_xscale("log"); ax.set_xlim(0.15, 2.6)
ax.set_xticks([0.2, 0.3, 0.5, 1, 2]); ax.set_xticklabels(["0.2", "0.3", "0.5", "1", "2"])
ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition ratio r")
ax.set_title("(a) matched τ criteria: S at τ = 200, onset at τ = T = 3000", loc="left", fontsize=10.5)
h = [Line2D([], [], color=COL["S"], lw=4, alpha=0.45, label="simulated S: 95% data bracket (model-free)"),
     Line2D([], [], color=COL["on"], lw=4, alpha=0.45, label="simulated onset: 95% data bracket"),
     Line2D([], [], color=INK2, marker="o", ls="none", ms=4.5, label="simulated, interpolated (analysis2_phase.py)"),
     Line2D([], [], color=INK2, lw=2, label="theory, branching operator (dense rows)"),
     Line2D([], [], color=INK2, marker="D", mfc="white", ls="none", ms=6, label="theory, pair-operator variant"),
     Line2D([], [], color=INK, ls=":", lw=1, label="g_lin")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8.5)

ax = axs[1]
TO = C["theory_operational"]["B32"]
for key, tk, mk in [("S_op", "S", "o"), ("L_op", "L", "s")]:
    col = COL["S"] if key == "S_op" else COL["on"]
    for rk in rs:
        o = C["operational"].get(f"pd2/onset_real_r{rk}.npz")
        if not o or o[key] is None:
            continue
        lo = o[f"{key}_prev"] or o[key]
        ax.plot([lo, o[key]], [float(rk)] * 2, color=col, lw=4, alpha=0.45, solid_capstyle="butt")
    rr = sorted(TO["real"], key=float)
    pts = [(float(r), TO["real"][r][tk]["g"]) for r in rr if TO["real"][r][tk]["g"]]
    ax.plot([p[1] for p in pts], [p[0] for p in pts], "-", color=col, lw=2)
    miss = [float(r) for r in rr if TO["real"][r][tk]["g"] is None]
    if miss:
        ax.plot([2.5] * len(miss), miss, ">", ms=7, mfc="white", mec=col, mew=1.4, ls="none")
ax.axvline(J["g_lin"]["real"], color=INK, ls=":", lw=1)
ax.set_xscale("log"); ax.set_xlim(0.15, 2.6)
ax.set_xticks([0.2, 0.3, 0.5, 1, 2]); ax.set_xticklabels(["0.2", "0.3", "0.5", "1", "2"])
ax.set_xlabel("coupling g")
ax.set_title("(b) operational phase.py criteria, B = 32 runs", loc="left", fontsize=10.5)
h = [Line2D([], [], color=COL["S"], lw=4, alpha=0.45, label="simulated S_op cell (last of 32 replicas dies after t = 200)"),
     Line2D([], [], color=COL["on"], lw=4, alpha=0.45, label="simulated L_op cell (all 32 alive at T, here and above)"),
     Line2D([], [], color=COL["S"], lw=2, label=f"theory at the matched τ = {TO['tau_S_equiv']:.0f}"),
     Line2D([], [], color=COL["on"], lw=2, label=f"theory at the matched τ = {TO['tau_L_equiv']:.3g}"),
     Line2D([], [], color=COL["on"], marker=">", mfc="white", ls="none", ms=7, label="theory: not reached by g = 4.89 (last row)")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8.5)
fig.suptitle("MaleCNS, h = 0: the theory lies above the simulated boundaries under either definition", x=0.01, ha="left",
             fontsize=11.5)
fig.tight_layout(); fig.savefig(f"{ROOT}/figs/pd2/critic_boundaries.png"); plt.close(fig)
print("wrote figs/pd2/critic_boundaries.png")
