"""E3 figures -> figs/aldef/depression_*.png, from results/aldef/depression/{points.json, summary.json, meso.json}."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "/MALECNS_ROOT"
RES = f"{ROOT}/results/aldef/depression"; FIG = f"{ROOT}/figs/aldef"
os.makedirs(FIG, exist_ok=True)
P = json.load(open(f"{RES}/points.json")); SUM = json.load(open(f"{RES}/summary.json"))
A, DRV, TH = SUM["arms"], SUM["driven"], SUM.get("theory", {})
MESO = json.load(open(f"{RES}/meso.json")) if os.path.exists(f"{RES}/meso.json") else None
G_LIN = 0.16315274988719286
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#9ec5f4", "#5c9ce6", "#2a78d6", "#1a5aa6", "#0d3b70"]          # one hue, light -> dark
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "lines.linewidth": 2, "figure.dpi": 150})
LITNAME = {"KW09": "Kazama & Wilson 2009 (f .72, 2.4 s)", "N15": "Nagel 2015 (f .78, 0.89 s)",
           "NW16": "Nagel & Wilson 2016 (f .75, 1.57 s)", "slow": "slow component (f .91, 0.63 s)"}
LITCOL = {"KW09": CAT[0], "N15": CAT[1], "NW16": CAT[2], "slow": CAT[3]}


def logx(ax):
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    lo, hi = ax.get_xlim()
    t = [x for x in (0.06, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 3, 5, 8, 15, 30) if lo <= x <= hi]
    ax.xaxis.set_major_locator(FixedLocator(t)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def grid(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)


def tau_curve(ax, arm, color, label, ls="-", marker="o"):
    if arm not in P:
        return
    pts = P[arm]
    g = np.array([p["g"] for p in pts]); tau = np.array([p["tau"] if p["tau"] is not None else np.inf for p in pts])
    lo = np.array([p["tau_ci"][0] for p in pts]); hi = np.array([p["tau_ci"][1] if p["tau_ci"][1] is not None else np.inf for p in pts])
    fin = np.isfinite(tau) & (tau < 3000.0)              # tau >= T is not resolved by a T = 3000 run: drawn at T
    y = np.where(fin, tau, 3000.0)
    ax.plot(g, y, ls=ls, color=color, lw=1.6, label=label, zorder=3)
    ax.errorbar(g[fin], tau[fin], yerr=[np.maximum(tau[fin] - lo[fin], 0), np.maximum(np.minimum(hi[fin], 3e4) - tau[fin], 0)], fmt=marker, ms=3.5,
                color=color, mec="white", mew=0.6, elinewidth=0.8, capsize=0, zorder=4)
    ax.plot(g[~fin], np.full((~fin).sum(), 3000.0), marker="^", ls="none", ms=5, color=color, mec="white", mew=0.6, zorder=4)


def tau_axes(ax, title):
    ax.set_xscale("log"); ax.set_yscale("log"); grid(ax)
    ax.axhline(200, color=MUTED, ls="--", lw=0.8); ax.axhline(3000, color=MUTED, ls="--", lw=0.8)
    ax.axvline(G_LIN, color=MUTED, ls=":", lw=1)
    ax.text(0.152, 230, "τ = 200 (S)", color=INK2, fontsize=7); ax.text(0.152, 3400, "τ = 3000 = T (sustained)", color=INK2, fontsize=7)
    ax.set_xlim(0.14, 9); ax.set_ylim(5, 8000); logx(ax)
    ax.set_xlabel("coupling g"); ax.set_ylabel("lifetime τ from a fully active start")
    ax.set_title(title, loc="left")


def fig_tau():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, kind, title in ((axs[0], "sp", "Depression per spike (Tsodyks–Markram), all synapses"),
                            (axs[1], "rt", "Depression during activity ('rate' mode), all synapses")):
        tau_curve(ax, "base_r1", INK, "no depression")
        for k in ("KW09", "N15", "NW16", "slow"):
            tau_curve(ax, f"{kind}_{k}_r1", LITCOL[k], LITNAME[k])
        tau_axes(ax, title); ax.legend(fontsize=7, loc="lower right")
    ax = axs[2]
    tau_curve(ax, "base_r1", INK, "no depression")
    for c, tau in zip(SEQ[:3] + [SEQ[4]], (3, 10, 30)):
        tau_curve(ax, f"sp_U25_tau{tau}_r1", c, f"U 0.25, τ_rec {tau} ({10 * tau} ms)")
    tau_curve(ax, "sp_NW16_r1", SEQ[4], "U 0.25, τ_rec 157 (1.57 s)")
    tau_axes(ax, "Weaker depression (per spike, U 0.25)"); ax.legend(fontsize=7, loc="lower right")
    fig.text(0.01, 0.01, "r = 1, B = 64, T = 3000. Triangles: τ ≥ 3000 (not resolved by T = 3000 runs). Dotted: g_lin = 0.163. Time unit 10 ms "
             "(τ_rec in units: 240, 89, 157, 63).", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/depression_tau.png"); plt.close(fig)


def qs_curve(ax, arm, key, color, label, region="AL_module", ls="-"):
    if arm not in P:
        return
    pts = [p for p in P[arm] if p["alive"] > 0]
    if key == "qs":
        pts = [p for p in pts if p.get("qs") and p["qs"].get(region) is not None]
        y = [p["qs"][region] for p in pts]
    elif key == "ratio":                                  # AL-module rate / rest-of-CNS rate (survivors)
        pts = [p for p in pts if p.get("qs") and p["qs"].get("CNS_rest")]
        y = [p["qs"]["AL_module"] / p["qs"]["CNS_rest"] for p in pts]
    else:
        pts = [p for p in pts if p.get(key) is not None and p["alive"] >= 0.25]
        y = [p[key] for p in pts]
    ax.plot([p["g"] for p in pts], y, "o", ls=ls, color=color, label=label, ms=3.5, mec="white", mew=0.6, lw=1.6)


def fig_amp():
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.4)); axs = axs.ravel()
    for arm, c, lab in [("base_r1", INK, "no depression")] + [(f"sp_{k}_r1", LITCOL[k], LITNAME[k]) for k in LITCOL]:
        qs_curve(axs[0], arm, "qs", c, lab)
        qs_curve(axs[2], arm, "AL_share", c, lab)
        qs_curve(axs[3], arm, "ratio", c, lab)
        if arm != "base_r1":
            qs_curve(axs[1], arm, "xact_AL", c, lab)
    for k in LITCOL:
        qs_curve(axs[0], f"rt_{k}_r1", "qs", LITCOL[k], None, ls=":")
    axs[0].set_yscale("log"); axs[0].set_ylabel("AL-module active fraction (survivors, t 1500–3000)")
    axs[0].set_title("Quasi-stationary AL activity", loc="left")
    axs[0].legend(handles=[Line2D([], [], color=c, marker="o", label=lab) for c, lab in
                           [(INK, "no depression")] + [(LITCOL[k], LITNAME[k]) for k in LITCOL]] +
                  [Line2D([], [], color=INK2, ls="-", label="per spike"), Line2D([], [], color=INK2, ls=":", label="rate mode")],
                  fontsize=7, loc="upper left")
    axs[3].set_yscale("log"); axs[3].set_ylabel("AL-module rate / rest-of-CNS rate (survivors)")
    axs[3].set_title("How much the AL stands out per neuron", loc="left")
    axs[1].set_ylabel("mean efficacy of active AL-module neurons"); axs[1].set_ylim(0, 1.02)
    axs[1].set_title("How depleted the active AL synapses are", loc="left")
    axs[2].set_ylabel("share of all activity in the AL module"); axs[2].set_ylim(0, 1.02)
    axs[2].set_title("Does the AL still carry the activity?", loc="left")
    for ax in axs:
        ax.set_xscale("log"); grid(ax); ax.set_xlabel("coupling g"); ax.set_xlim(0.14, 9); logx(ax)
    fig.text(0.01, 0.005, "r = 1. Points with at least one surviving replica (share: ≥ 25 % alive). Per-spike depression, "
             "all synapses; AL module = the 526-neuron lLN1_bc/PN module.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/depression_amplitude.png"); plt.close(fig)


def fig_groups():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    G1 = [("spg_ORN_r1", CAT[3], "ORNs only"), ("spg_PN_r1", CAT[1], "PNs only"), ("spg_LN_r1", CAT[0], "LNs only"),
          ("spg_lLN1bc_r1", CAT[4], "lLN1_bc only"), ("spg_AL_r1", CAT[6], "whole AL only")]
    G2 = [("spg_exc_r1", CAT[7], "excitatory (ACh) outputs only"), ("spg_notAL_r1", CAT[5], "everything except the AL")]
    for ax, grp, title in ((axs[0], G1, "Depression in one AL population"), (axs[1], G2, "Excitatory-only / outside the AL")):
        tau_curve(ax, "base_r1", INK, "no depression")
        tau_curve(ax, "sp_NW16_r1", MUTED, "all synapses", ls="--")
        for arm, c, lab in grp:
            tau_curve(ax, arm, c, lab)
        tau_axes(ax, title); ax.legend(fontsize=7, loc="lower right")
    ax = axs[2]
    for arm, c, lab in [("base_r1", INK, "no depression"), ("sp_NW16_r1", MUTED, "all synapses")] + G1 + G2:
        qs_curve(ax, arm, "AL_share", c, lab)
    ax.set_xscale("log"); grid(ax); ax.set_xlim(0.14, 9); logx(ax); ax.set_ylim(0, 1.02); ax.set_xlabel("coupling g")
    ax.set_ylabel("share of all activity in the AL module"); ax.set_title("Who carries the sustained activity", loc="left")
    ax.legend(fontsize=6.5, loc="lower left", ncol=2)
    fig.text(0.01, 0.01, "Per-spike depression U 0.25, τ_rec 157 (Nagel & Wilson 2016) on the named presynaptic population only; r = 1.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/depression_groups.png"); plt.close(fig)


def fig_adapt():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    tau_curve(ax, "base_r1", INK, "none")
    for c, dt in zip(SEQ[1:], (0.1, 0.3, 1, 3)):
        tau_curve(ax, f"ad_DT{dt:g}_ta50_r1", c, f"D·τ_a = {dt:g}, τ_a 50")
    tau_curve(ax, "ad_DT1_ta10_r1", CAT[1], "D·τ_a = 1, τ_a 10")
    tau_curve(ax, "adg_LN_DT1_ta50_r1", CAT[4], "LNs only, D·τ_a = 1, τ_a 50")
    tau_axes(ax, "Spike-frequency adaptation"); ax.legend(fontsize=7, loc="lower right")
    ax = axs[1]
    tau_curve(ax, "base_r1", INK, "none")
    for c, n in zip(SEQ[1:], (5, 10, 25, 50)):
        tau_curve(ax, f"ref{n}_r1", c, f"refractory {n} steps ({n * 0.2:g} units)")
    tau_axes(ax, "Absolute refractory period"); ax.legend(fontsize=7, loc="lower right")
    ax = axs[2]
    for arm, c, lab in [("base_r1", INK, "none"), ("ad_DT1_ta50_r1", SEQ[3], "adaptation D·τ_a 1"),
                        ("ad_DT3_ta50_r1", SEQ[4], "adaptation D·τ_a 3"), ("ref25_r1", CAT[1], "refractory 25"),
                        ("ref50_r1", CAT[7], "refractory 50"), ("sp_NW16_r1", CAT[2], "depression (NW16)")]:
        qs_curve(ax, arm, "qs", c, lab)
    ax.set_xscale("log"); ax.set_yscale("log"); grid(ax); ax.set_xlim(0.14, 9); logx(ax); ax.set_xlabel("coupling g")
    ax.set_ylabel("AL-module active fraction (survivors)"); ax.set_title("Quasi-stationary AL activity", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    fig.text(0.01, 0.01, "r = 1. Adaptation: θ_i rises by D per unit active time and decays with τ_a; stationary θ = D τ_a ρ "
             "is subtracted from the input. 1 unit = 10 ms.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/depression_adapt_refr.png"); plt.close(fig)


FAMILIES = [("Reference", ["base_r1", "base_r0"]),
            ("Depression per spike, all synapses", ["sp_KW09_r1", "sp_N15_r1", "sp_NW16_r1", "sp_slow_r1", "sp_U25_tau30_r1",
                                                     "sp_U25_tau10_r1", "sp_U25_tau3_r1", "sp_NW16_r0"]),
            ("Depression rate mode, all synapses", ["rt_KW09_r1", "rt_N15_r1", "rt_NW16_r1", "rt_slow_r1", "rt_NW16_r0"]),
            ("Depression (NW16) on one population", ["spg_exc_r1", "spg_notAL_r1", "spg_AL_r1", "spg_LN_r1", "spg_lLN1bc_r1",
                                                     "spg_PN_r1", "spg_ORN_r1"]),
            ("Adaptation", ["ad_DT0.1_ta50_r1", "ad_DT0.3_ta50_r1", "ad_DT1_ta50_r1", "ad_DT3_ta50_r1", "ad_DT1_ta10_r1",
                            "adg_LN_DT1_ta50_r1", "ad_DT1_ta50_r0"]),
            ("Refractoriness", ["ref5_r1", "ref10_r1", "ref25_r1", "ref50_r1", "ref10_r0"]),
            ("Isolation: AL alone / CNS without the AL", ["iso_base_ALonly_r1", "iso_base_noAL_r1", "iso_sp_NW16_ALonly_r1",
                                                           "iso_sp_NW16_noAL_r1", "iso_sp_KW09_ALonly_r1", "iso_sp_KW09_noAL_r1"])]


def fig_bounds():
    rows = [(f, a) for f, arms in FAMILIES for a in arms if a in A]
    fig, ax = plt.subplots(figsize=(10, 0.26 * len(rows) + 1.6))
    y = 0; yt, yl = [], []
    gmax = 9.0
    for fam, arms in FAMILIES:
        arms = [a for a in arms if a in A]
        if not arms:
            continue
        ax.text(0.141, y - 0.15, fam, fontsize=8, fontweight="bold", color=INK, va="bottom")
        y += 0.8
        for a in arms:
            s = A[a]
            for key, mk, c in (("S", "o", CAT[0]), ("on", "s", CAT[1])):
                b = s[key]
                if b["g"] is not None:
                    tot = b.get("total") or [b["g"], b["g"]]
                    ax.plot([tot[0], tot[1] if tot[1] else gmax], [y, y], color=c, lw=2.2, alpha=0.45, solid_capstyle="butt")
                    ax.plot(b["g"], y, mk, color=c, mec=c, mew=1.2, ms=6, mfc=c if key == "on" else "white", zorder=5)
                elif b["status"] == "none":
                    ax.annotate("", xy=(gmax * 1.05, y), xytext=(gmax * 0.85, y), arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
                elif b["status"] == "below_grid":
                    ax.plot(0.15, y, "<", color=c, ms=5)
            ta = {"iso_base_ALonly_r1": "base_r1", "iso_sp_NW16_ALonly_r1": "sp_NW16_r1",
                  "iso_sp_KW09_ALonly_r1": "sp_KW09_r1"}.get(a, None if a.startswith("iso_") else a)
            t = TH.get("curves", {}).get(ta, {}).get("cal") if ta else None
            if t and t.get("on_frozen"):
                ax.plot(t["on_frozen"], y, "|", color=INK, ms=9, mew=1.6)
            if t and t.get("on_adiabatic"):
                ax.plot(t["on_adiabatic"], y, "|", color=MUTED, ms=9, mew=1.6)
            if MESO and ta in MESO["arms"] and MESO["arms"][ta].get("on"):
                ax.plot(MESO["arms"][ta]["on"], y, "x", color=CAT[6], ms=6, mew=1.4)
            yt.append(y); yl.append(a); y += 1
        y += 0.4
    for a, ls in (("base_r1", "-"),):
        for key, c in (("S", CAT[0]), ("on", CAT[1])):
            if A.get(a, {}).get(key, {}).get("g"):
                ax.axvline(A[a][key]["g"], color=c, ls=ls, lw=0.6, alpha=0.5)
    ax.axvline(G_LIN, color=MUTED, ls=":", lw=1)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(0.14, gmax * 1.15); logx(ax); grid(ax); ax.set_xlabel("coupling g")
    h = [Line2D([], [], marker="o", color=CAT[0], mfc="white", ls="-", label="S: τ = 200 (bar = grid bracket ∪ bootstrap 95 %)"),
         Line2D([], [], marker="s", color=CAT[1], ls="-", label="sustained onset: τ = 3000"),
         Line2D([], [], marker="|", color=INK, ls="none", ms=9, mew=1.6, label="theory onset, frozen resource (K calibrated)"),
         Line2D([], [], marker="|", color=MUTED, ls="none", ms=9, mew=1.6, label="theory onset, adiabatic resource"),
         Line2D([], [], marker="x", color=CAT[6], ls="none", label="mesoscopic (n, x) model onset"),
         Line2D([], [], color=MUTED, ls=":", label="g_lin = 0.163")]
    ax.legend(handles=h, fontsize=7, loc="upper right", frameon=True, facecolor="white", edgecolor=GRID)
    ax.set_title("Q/S/L boundaries per arm (r = 1 unless _r0): S = τ 200, sustained onset = τ 3000", loc="left")
    ax.text(0.99, 0.005, "Theory / mesoscopic markers describe the AL mode alone (on iso_*_ALonly rows: the matching depression arm).",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK2)
    fig.tight_layout(); fig.savefig(f"{FIG}/depression_boundaries.png"); plt.close(fig)


def fig_theory():
    th = TH.get("curves", {}); g = np.array(TH.get("g", []))
    if not th:
        return
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    panels = ((axs[0], "r = 1, no depression and NW16", [("base_r1", "base_r1", INK, "no depression, whole CNS"),
                                                          ("iso_base_ALonly_r1", None, MUTED, "no depression, AL alone"),
                                                          ("sp_NW16_r1", None, CAT[2], "NW16, whole CNS"),
                                                          ("iso_sp_NW16_ALonly_r1", "sp_NW16_r1", CAT[0], "NW16, AL alone")]),
              (axs[1], "r = 1, KW09 and weak depression", [("sp_KW09_r1", None, CAT[1], "KW09, whole CNS"),
                                                            ("iso_sp_KW09_ALonly_r1", "sp_KW09_r1", CAT[6], "KW09, AL alone"),
                                                            ("sp_U25_tau10_r1", "sp_U25_tau10_r1", CAT[3], "U .25, τ_rec 10"),
                                                            ("sp_U25_tau30_r1", "sp_U25_tau30_r1", CAT[4], "U .25, τ_rec 30")]),
              (axs[2], "r = 0", [("base_r0", "base_r0", INK, "no depression"), ("sp_NW16_r0", "sp_NW16_r0", CAT[2], "NW16 per spike"),
                                 ("rt_NW16_r0", "rt_NW16_r0", CAT[1], "NW16 rate mode")]))
    for ax, title, arms in panels:
        for a, ta, c, lab in arms:
            tau_curve(ax, a, c, lab)
            if ta is None:
                continue
            t = th.get(ta, {}).get("cal")
            if t:
                ax.plot(g, np.minimum(np.exp(np.minimum(t["lt_frozen"], 10)), 3000.0), color=c, ls="--", lw=1.1)
                ax.plot(g, np.minimum(np.exp(np.minimum(t["lt_adiabatic"], 10)), 3000.0), color=c, ls=":", lw=1.1)
            if MESO and ta in MESO["arms"]:
                rr = MESO["arms"][ta]["rows"]
                ax.plot([x["g"] for x in rr], [min(x["tau_reg"], 3000.0) for x in rr], color=c, ls="-.", lw=1.0)
        tau_axes(ax, title)
        h = [Line2D([], [], color=c, marker="o", ls="-", label=lab) for a, ta, c, lab in arms]
        h += [Line2D([], [], color=INK2, ls="--", lw=1.1, label="theory, frozen resource"),
              Line2D([], [], color=INK2, ls=":", lw=1.1, label="theory, adiabatic resource"),
              Line2D([], [], color=INK2, ls="-.", lw=1.0, label="mesoscopic (n, x) model")]
        ax.legend(handles=h, fontsize=7, loc="lower right")
    k = TH.get("kappa", {})
    fig.text(0.01, 0.01, f"Theory K = κ·K_eff with κ fitted to this experiment's base onset (κ: {', '.join(f'r={a}: {float(b):.2f}' for a, b in k.items())}); "
             "theory grid ends at g = 4.9. Lines are theories of the AL mode alone; compare them with the AL-alone runs.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/depression_theory.png"); plt.close(fig)


def fig_driven():
    arms = [("base", INK, "no depression"), ("sp_NW16", CAT[2], "per spike NW16"), ("sp_KW09", CAT[0], "per spike KW09"),
            ("rt_NW16", CAT[1], "rate mode NW16")]
    if not any(f"drv_{a}_h1e-5_r1" in DRV for a, _, _ in arms):
        return
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for j, h in enumerate(("1e-5", "1e-4")):
        ax = axs[j]
        for a, c, lab in arms:
            d = DRV.get(f"drv_{a}_h{h}_r1")
            if not d:
                continue
            m, se = np.array(d["AL_module"]["gain"]), np.array(d["AL_module"]["se"]) / d["rho_iso"]
            ax.errorbar(d["g"], m, yerr=se, fmt="o-", color=c, ms=3.5, mec="white", mew=0.6, lw=1.6, elinewidth=0.8, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); grid(ax); ax.axvline(G_LIN, color=MUTED, ls=":", lw=1); logx(ax)
        ax.set_xlabel("coupling g"); ax.set_ylabel("AL-module activity / isolated-unit activity")
        ax.set_title(f"Response to weak drive h = {h} on every neuron", loc="left"); ax.legend(fontsize=7, loc="upper left")
    ax = axs[2]
    for h, ls in (("1e-5", "-"), ("1e-4", "--")):
        b = DRV.get(f"drv_base_h{h}_r1")
        for a, c, lab in arms[1:]:
            d = DRV.get(f"drv_{a}_h{h}_r1")
            if not (b and d):
                continue
            x, y = np.array(d["AL_module"]["mean"]), np.array(b["AL_module"]["mean"])
            sx, sy = np.array(d["AL_module"]["se"]), np.array(b["AL_module"]["se"])
            rt = x / y; se = rt * np.sqrt((sx / x) ** 2 + (sy / y) ** 2)
            ax.errorbar(d["g"], rt, yerr=se, fmt="o", ls=ls, color=c, ms=3.5, mec="white", mew=0.6, lw=1.4, elinewidth=0.8,
                        label=f"{lab}, h {h}")
    ax.axhline(1, color=MUTED, lw=0.8); ax.axvline(G_LIN, color=MUTED, ls=":", lw=1)
    ax.set_xscale("log"); grid(ax); logx(ax); ax.set_xlabel("coupling g"); ax.set_ylabel("AL activity with / without depression")
    ax.set_title("Linear regime: is the gain unchanged?", loc="left"); ax.legend(fontsize=6.5, loc="lower left")
    fig.text(0.01, 0.01, "Quiet start, r = 1, B = 64, mean over 1000 < t ≤ 3000. Error bars: replica SE. Dotted: g_lin.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/depression_driven.png"); plt.close(fig)


def fig_removal():
    arms = [("rm_base_r1", INK, "no depression"), ("rm_sp_NW16_r1", CAT[2], "per spike NW16, all"),
            ("rm_sp_KW09_r1", CAT[0], "per spike KW09, all"), ("rm_spg_exc_r1", CAT[4], "per spike NW16, excitatory only"),
            ("rm_rt_NW16_r1", CAT[1], "rate mode NW16, all")]
    if not any(a in DRV for a, _, _ in arms):
        return
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, c, lab in arms:
        d = DRV.get(a)
        if not d:
            continue
        R = d["rows"]; g = [x["g"] for x in R]
        axs[0].plot(g, [x["G"] for x in R], "o-", color=c, ms=3.5, mec="white", mew=0.6, lw=1.6, label=lab)
        axs[1].plot(g, [max(x["resid_100_300"] or 0, 1e-4) for x in R], "o-", color=c, ms=3.5, mec="white", mew=0.6, lw=1.6, label=lab)
        axs[2].plot(g, [x["p_alive_end"] for x in R], "o-", color=c, ms=3.5, mec="white", mew=0.6, lw=1.6, label=lab)
        ok = [x["g"] for x in R if x["allowed"]]
        if ok:
            axs[0].plot(ok, [x["G"] for x in R if x["allowed"]], "o", color=c, ms=8, mfc="none", mew=1.2)
    axs[0].axhspan(0.29, 1.45, color=CAT[2], alpha=0.12, lw=0)
    axs[0].text(2.7, 0.33, "PN/ORN rate ratio of real flies\n(1–5 Hz over 3.44 Hz)", fontsize=7, color=INK2, ha="right", va="bottom")
    axs[0].set_yscale("log"); axs[0].set_ylabel("driven gain G = ρ(antennal uPN) / ρ(ORN)")
    axs[0].set_title("Driven AL: PN gain over ORNs (circles: all E1 criteria met)", loc="left")
    axs[1].axhline(0.1, color=MUTED, ls="--", lw=0.8); axs[1].set_yscale("log")
    axs[1].set_ylabel("uPN activity 100–300 units after removal / driven (0 drawn at 1e-4)")
    axs[1].set_title("After all ORNs are removed", loc="left")
    axs[2].axhline(0.05, color=MUTED, ls="--", lw=0.8); axs[2].set_ylabel("fraction of replicas with AL activity at the end")
    axs[2].set_title("Still active 3000 units after removal", loc="left")
    for ax in axs:
        ax.set_xscale("log"); grid(ax); logx(ax); ax.set_xlabel("coupling g"); ax.legend(fontsize=7, loc="upper left")
    fig.text(0.01, 0.01, "E1 protocol (results/aldef/drive/plan.md): quiet start, all 2,639 ORNs driven at ρ_iso 0.04 (3.6 Hz at 10 ms), "
             "r = 1, B = 64; at t = 1500 all ORNs clamped to 0.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/depression_removal.png"); plt.close(fig)


def fig_iso():
    if "iso_base_ALonly_r1" not in P:
        return
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (k, full, title) in zip(axs, (("base", "base_r1", "No depression"), ("sp_NW16", "sp_NW16_r1", "Per-spike depression NW16"),
                                          ("sp_KW09", "sp_KW09_r1", "Per-spike depression KW09"))):
        tau_curve(ax, full, INK, "whole CNS")
        tau_curve(ax, f"iso_{k}_ALonly_r1", CAT[0], "AL alone (rest of CNS clamped off)")
        tau_curve(ax, f"iso_{k}_noAL_r1", CAT[1], "CNS without the AL (AL clamped off)")
        tau_axes(ax, title); ax.set_xlim(0.14, 30); logx(ax); ax.legend(fontsize=7, loc="lower right")
    fig.text(0.01, 0.01, "r = 1, decay from a fully active start, B = 64. 'AL' = AL classes ∪ the 526-neuron module.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/depression_isolation.png"); plt.close(fig)


for f in (fig_iso, fig_tau, fig_amp, fig_groups, fig_adapt, fig_bounds, fig_theory, fig_driven, fig_removal):
    try:
        f(); print("ok", f.__name__)
    except Exception as e:
        import traceback; traceback.print_exc(); print("FAILED", f.__name__, e)
