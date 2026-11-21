"""m2_mf_figs.py -- figures of the M2 mean-field predictions (lane m2/theory, meanfield).
Reads results/m2/theory/meanfield_{predictions,lines,points,validation,glin,chain}.json; writes
figs/m2/meanfield_phase.png, meanfield_driven.png, meanfield_modes.png."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = "/MALECNS_ROOT"
TH = f"{ROOT}/results/m2/theory"; FG = f"{ROOT}/figs/m2"
PR = json.load(open(f"{TH}/meanfield_predictions.json"))
L = json.load(open(f"{TH}/meanfield_lines.json")); P = json.load(open(f"{TH}/meanfield_points.json"))
CH = json.load(open(f"{TH}/meanfield_chain.json")); VAL = json.load(open(f"{TH}/meanfield_validation.json"))
GL = json.load(open(f"{TH}/meanfield_glin.json"))
SIM1 = {row["r"]: row for row in json.load(open(f"{ROOT}/results/analysis2/flow_sim_boundaries.json"))["rows"]}
prim = PR["primary_closure"]; alt = "qmf" if prim == "cqmf" else "cqmf"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 10, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2})
RS = [float(r) for r in PR["P3_decay_M2"].keys()]
RC = [0, 0.25, 0.5, 1, 2, 4, 6]


def col(lines, key):
    return np.array([np.nan if lines.get(r) is None or lines[r].get(key) is None else lines[r][key] for r in lines])


# ------------------------------------------------------------------------------------------------ phase diagram
fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1, 1, 0.75]})
for ax, cl, ttl in ((axs[0], "primary", f"M2 decay, h = 0 — {prim.upper()} (primary)"), (axs[1], "alternate", f"M2 decay, h = 0 — {alt.upper()}")):
    blk = {r: PR["P3_decay_M2"][r][cl] for r in PR["P3_decay_M2"]}
    r = np.array(RS)
    for key, lab, c, ls in (("S", "S, cutoff mean field (lower bound)", C1, ":"), ("on", "onset, cutoff mean field (lower bound)", C1, "-"),
                            ("on_equal_size", "onset, equal-size rule (M1-calibrated)", C3, "-"), ("D", "D (PR = N/4)", C4, "-.")):
        y = col(blk, key)
        if np.isfinite(y).any():
            ax.plot(y, r, ls=ls, color=c, label=lab, marker="o" if key == "on_equal_size" else None, ms=3)
    chon = np.array([(PR["P3_decay_M2"][k]["chain_AL_mode"]["frozen_kcal"] or {}).get("on_frozen") or np.nan for k in PR["P3_decay_M2"]])
    chad = np.array([(PR["P3_decay_M2"][k]["chain_AL_mode"]["frozen_kcal"] or {}).get("on_adiabatic") or np.nan for k in PR["P3_decay_M2"]])
    ax.plot(chon, r, color=C2, label="onset, AL-mode chain, frozen (upper bound)")
    ax.plot(chad, r, color=C2, ls="--", label="onset, AL-mode chain, adiabatic")
    hop = col(blk, "hopf_first_g")
    ax.plot(hop, r, ls="none", marker="^", color=C5, mec=INK2, mew=0.5, ms=6, label="first mean-field oscillation (Hopf)")
    bis = [((blk[k] or {}).get("bistable_intervals") or []) for k in blk]
    first = True
    for rr, iv in zip(r, bis):
        for s in iv:
            ax.plot([s["from"], s["to"]], [rr, rr], color=INK, lw=4, solid_capstyle="butt", label="bistable (2 ICs)" if first else None)
            first = False
    ax.axvline(GL["glin"]["real"]["g_lin"], color=INK2, lw=1)
    ax.text(GL["glin"]["real"]["g_lin"] * 1.03, 6.15, "g_lin 0.163", color=INK2, fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(0.1, 30); ax.set_ylim(-0.2, 6.4)
    ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition scale r"); ax.set_title(ttl, color=INK, loc="left")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=7.5, loc="lower center", ncol=4, frameon=False)
# M1 validation
ax = axs[2]
for cl, c in (("qmf", C1), ("cqmf", C2)):
    rows = VAL["V1"][cl]["rows"]
    rr = [float(k) for k in rows]; dm = [rows[k]["D_MF"] if rows[k]["D_MF"] else np.nan for k in rows]
    ax.plot(dm, rr, marker="o", color=c, label=f"M1 {cl.upper()} D (score {VAL['V1'][cl]['score']:.2f})")
rs1 = sorted(SIM1); ds = [SIM1[k]["D"] if SIM1[k]["D"] else np.nan for k in rs1]
ax.plot(ds, rs1, color=INK, lw=1.5, marker="s", ms=3, label="M1 simulated D")
ax.set_xscale("log"); ax.set_xlim(0.5, 30); ax.set_ylim(-0.2, 6.4); ax.set_xlabel("g"); ax.set_ylabel("r")
ax.set_title("Closure check on M1 (V1)", color=INK, loc="left"); ax.legend(fontsize=7, loc="upper left", facecolor=SURF, edgecolor=GRID)
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(f"{FG}/meanfield_phase.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------------------------------------ driven
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})
cm = plt.get_cmap("Blues")
for i, r in enumerate(RC):
    nm = f"driven_M2_{prim if r else 'qmf'}_r{r:g}"
    if nm not in P:
        continue
    pp = sorted([p for p in P[nm]["points"] if p["ic"] == "quiet"], key=lambda p: p["g"])
    g = [p["g"] for p in pp]; c = cm(0.35 + 0.6 * i / (len(RC) - 1))
    axs[0].plot(g, [p["G"] for p in pp], color=c, label=f"r = {r:g}")
    axs[1].plot(g, [p["rho_ORN"] for p in pp], color=c, label=f"r = {r:g}")
axs[0].axhspan(0.29, 1.45, color=C3, alpha=0.12, lw=0); axs[0].text(0.055, 0.33, "gain window 0.29-1.45", color=INK2, fontsize=7)
axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("g"); axs[0].set_ylabel("G = rho_uPN / rho_ORN")
axs[0].set_title(f"Gain, ORN 0.04 ({prim.upper()})", loc="left", color=INK); axs[0].legend(fontsize=7, ncol=2, facecolor=SURF, edgecolor=GRID)
axs[1].axhline(0.04, color=INK2, lw=1, ls=":"); axs[1].set_xscale("log"); axs[1].set_xlabel("g"); axs[1].set_ylabel("measured rho_ORN")
axs[1].set_title("ORN activity under recurrent input", loc="left", color=INK)
# phase tiles
cats = ["DQ", "DA", "DT", "SS-L", "SS-D"]
grid = np.full((len(RS), 40), np.nan); allow = np.zeros((len(RS), 40), bool); osc = np.zeros((len(RS), 40), bool); G = None
for i, r in enumerate(RS):
    nm = f"driven_M2_{prim if r else 'qmf'}_r{r:g}"
    if nm not in P:
        continue
    pp = sorted([p for p in P[nm]["points"] if p["ic"] == "quiet"], key=lambda p: p["g"]); G = [p["g"] for p in pp]
    for j, p in enumerate(pp):
        base = p["phase"].split("-O")[0].replace("-NS", "").replace("-D", "") if not p["phase"].startswith("SS") else p["phase"][:4]
        grid[i, j] = cats.index(base); allow[i, j] = p["allowed"]; osc[i, j] = "-O" in p["phase"]
if G is not None:
    ramp = ListedColormap(["#dfe9f7", "#9cc0ec", "#5e97dc", "#2a78d6", "#123e7a"])
    ax = axs[2]
    lg = np.log10(G); edges = np.r_[lg[0] - (lg[1] - lg[0]) / 2, (lg[1:] + lg[:-1]) / 2, lg[-1] + (lg[1] - lg[0]) / 2]
    re = np.arange(len(RS) + 1) - 0.5
    im = ax.pcolormesh(10 ** edges, re, grid, cmap=ramp, norm=BoundaryNorm(np.arange(-0.5, 5.5), 5), edgecolors=SURF, linewidth=0.4)
    yy, xx = np.where(allow); ax.plot(np.array(G)[xx], yy, ls="none", marker="o", ms=2.5, color=C5, mec=INK, mew=0.4, label="data-allowed (cutoff removal)")
    sse = [PR["P6_driven_M2_ORN0.04"][f"{r:g}"]["primary"]["SS_equal_size"] if PR["P6_driven_M2_ORN0.04"][f"{r:g}"]["primary"] else np.nan for r in RS]
    ax.plot([s if s is not None else np.nan for s in sse], range(len(RS)), color="#ffffff", lw=2.5)
    ax.plot([s if s is not None else np.nan for s in sse], range(len(RS)), color=C2, lw=1.5, ls="--", label="removal SS, equal-size rule")
    g029 = [PR["P6_driven_M2_ORN0.04"][f"{r:g}"]["primary"]["DQ_DA_G029"] if PR["P6_driven_M2_ORN0.04"][f"{r:g}"]["primary"] else np.nan for r in RS]
    ax.plot([s if s is not None else np.nan for s in g029], range(len(RS)), color=C3, lw=1.5, label="G = 0.29")
    yy, xx = np.where(osc); ax.plot(np.array(G)[xx], yy, ls="none", marker="x", ms=3, color=C2, label="mean-field oscillation (-O)")
    ax.set_xscale("log"); ax.set_yticks(range(len(RS))[::2]); ax.set_yticklabels([f"{r:g}" for r in RS][::2])
    ax.set_xlabel("g"); ax.set_ylabel("r"); ax.set_title("Driven phases (tiles: cutoff removal = lower bounds)", loc="left", color=INK, fontsize=9)
    cb = fig.colorbar(im, ax=ax, ticks=range(5)); cb.ax.set_yticklabels(cats)
    ax.legend(fontsize=7, loc="upper left", facecolor=SURF, edgecolor=GRID)
fig.tight_layout(); fig.savefig(f"{FG}/meanfield_driven.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------------------------------------ modes / sigma
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
for i, r in enumerate(RC):
    nm = f"decay_M2_{prim if r else 'qmf'}_r{r:g}"
    if nm not in P:
        continue
    pp = sorted([p for p in P[nm]["points"] if p["ic"] == "decay" and "ring" in p and p["tau_MF"] is None], key=lambda p: p["g"])
    good = lambda p: (p.get("ring_robust_P") and not p["osc"]["osc"] and p.get("ring_resid", 1) < 1e-6
                      and p["ring"]["demog"].get("osc_absz", 2) < 1)
    c = cm(0.35 + 0.6 * i / (len(RC) - 1))
    g = [p["g"] for p in pp]
    Pp = [p["ring"]["demog"].get("osc_P", np.nan) if good(p) else np.nan for p in pp]
    Dp = [p["ring"]["demog"].get("osc_DP", np.nan) if good(p) else np.nan for p in pp]
    axs[0].plot(g, Pp, color=c, marker="o", ms=3, label=f"r = {r:g}")
    axs[1].plot(g, Dp, color=c, marker="o", ms=3)
    axs[2].plot(g, [p["sigma_x"] for p in pp], color=c, marker="o", ms=3, label=f"r = {r:g}")
axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("g"); axs[0].set_ylabel("period of leading damped mode (units)")
axs[0].set_title("Leading damped mode: period (robust, stationary)", loc="left", color=INK); axs[0].legend(fontsize=7, ncol=2, facecolor=SURF, edgecolor=GRID)
axs[1].axhline(0.7, color=C2, lw=1, ls="--"); axs[1].text(0.23, 1.0, "OSC-strong threshold 0.7", color=INK2, fontsize=7)
axs[1].axhline(0.591, color=INK2, lw=1, ls=":"); axs[1].text(0.23, 0.33, "slow-fast bound 0.59 (r = 0)", color=INK2, fontsize=7)
axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_ylim(1e-6, 2.5); axs[1].set_xlabel("g"); axs[1].set_ylabel("damping per period |z|^P")
axs[1].set_title("Coherence of the leading damped mode", loc="left", color=INK)
axs[2].axhline(1, color=INK2, lw=1); axs[2].set_xscale("log"); axs[2].set_yscale("log"); axs[2].set_xlabel("g")
axs[2].set_ylabel("sigma_x (frozen-resource branching ratio)"); axs[2].set_title("Renormalised branching at the QS state", loc="left", color=INK)
fig.tight_layout(); fig.savefig(f"{FG}/meanfield_modes.png", dpi=150); plt.close(fig)
print("figures written")
