"""E1 figures (2026-11-09): figs/aldef/drive_*.png from results/aldef/drive/{rows,summary}.json and runs/*.npz.
  drive_gain.png      AL gain G = rho_uPN_ant / rho_ORN vs g (both r, every ORN level) with the empirical windows
  drive_allowed.png   where each criterion holds (gain / silent after removal / both) on the (g, ORN level) strip, per r
  drive_removal.png   uPN activity after removal (relative to driven) and AL survival, vs time since removal
  drive_regions.png   driven activity by region vs g, and what is left 1000-3000 units after removal
  drive_persist.png   median time from removal to the last AL activity vs g, with the h = 0 lifetimes (r = 1)
  drive_fluct.png     AL-module fluctuations in the driven window vs g (driven-plane peak g = 0.30 marked)
  drive_palp.png      residual antennal uPN activity after removal: palps kept (ant) vs all ORNs removed (all)
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, LogLocator, FuncFormatter

ROOT = "/MALECNS_ROOT"
D = f"{ROOT}/results/aldef/drive"
F = f"{ROOT}/figs/aldef"
os.makedirs(F, exist_ok=True)
rows = json.load(open(f"{D}/rows.json"))
summ = json.load(open(f"{D}/summary.json"))
levels = json.load(open(f"{D}/levels.json"))
LEV = list(levels)
H0 = {0: {"g_lin": 0.1632, "S": 0.234, "onset": 0.266}, 1: {"g_lin": 0.1632, "S": 0.283, "onset": 0.344}}
GAIN_P, GAIN_B = (0.29, 1.45), (0.077, 1.67)
T1 = 1500.0
# palette (dataviz reference): ordinal blue ramp for ORN level, categorical slots for regions, neutral inks
BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.titlesize": 10})


def sel(proto, r, lev):
    return [x for x in rows if x["proto"] == proto and x["r"] == r and x["level"] == lev]


def lev_label(lev):
    v = levels[lev]
    return f"ORN {v['rho_iso']:.2f} ({v['ORN_Hz_at_tau10ms']:.1f} Hz @10 ms)"


def gticks(ax):
    """Readable tick labels on a log g axis."""
    t = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    ax.xaxis.set_major_locator(FixedLocator(t)); ax.xaxis.set_major_formatter(FixedFormatter([f"{x:g}" for x in t]))
    ax.xaxis.set_minor_locator(NullLocator())


def vlines(ax, r, ymin=None, text=True):
    b = H0[r]
    ax.axvline(b["g_lin"], color=INK2, lw=1.0, ls="-", zorder=1)
    ax.axvline(b["S"], color=INK2, lw=1.0, ls="--", zorder=1)
    ax.axvline(b["onset"], color=INK2, lw=1.0, ls=":", zorder=1)
    if text:
        tr = ax.get_xaxis_transform()
        for x, s, ha in ((b["g_lin"], "g_lin ", "right"), (b["S"], "τ=200 ", "right"), (b["onset"], " onset", "left")):
            ax.text(x, 1.01, s, transform=tr, ha=ha, va="bottom", fontsize=7, color=INK2)


# --------------------------------------------------------------------------------------------- gain
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, r in zip(axs, (0, 1)):
    ax.axhspan(*GAIN_P, color="#f0efec", zorder=0)
    for y in GAIN_B:
        ax.axhline(y, color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=1)
    for k, lev in enumerate(LEV):
        A = [x for x in sel("all", r, lev) if x["g"] > 0]
        g = np.array([x["g"] for x in A]); G = np.array([x["G"] for x in A])
        lo = np.array([x["G_ci"][0] for x in A]); hi = np.array([x["G_ci"][1] for x in A])
        ok = np.array([x["gain_primary"] and x["S1"] and x["S2"] for x in A])
        okb = np.array([x["gain_broad"] and x["S1"] and x["S2"] for x in A])
        ax.fill_between(g, lo, hi, color=BLUE[k], alpha=0.18, lw=0)
        ax.plot(g, G, color=BLUE[k], lw=2, label=lev_label(lev), zorder=3)
        ax.plot(g[okb], G[okb], "o", ms=5, mfc="white", mec=BLUE[k], mew=1.5, zorder=4)
        ax.plot(g[ok], G[ok], "o", ms=5, color=BLUE[k], zorder=5)
    vlines(ax, r)
    if r == 1:
        ax.axvline(0.30, color=CAT[1], lw=1.2, ls="-.", zorder=1)
        ax.text(0.305, 0.93, "fluct. peak\nh = 1e-5", transform=ax.get_xaxis_transform(), color=INK2, fontsize=7, va="top")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.035, 1.05); gticks(ax)
    ax.set_xlabel("coupling g"); ax.set_title(f"r = {r}", pad=14)
axs[0].set_ylabel("AL gain  G = ρ(antennal uPN) / ρ(ORN)")
axs[0].text(0.98, GAIN_P[0] * 1.06, "PN 1–5 Hz / ORN 3.44 Hz", fontsize=7, color=INK2, ha="right",
            transform=axs[0].get_yaxis_transform())
h, l = axs[1].get_legend_handles_labels()
h += [Line2D([], [], marker="o", ls="", color=INK2, ms=5), Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5),
      Patch(color="#f0efec"), Line2D([], [], color=INK2, ls=(0, (2, 2)), lw=0.8)]
l += ["allowed (primary gain + silent)", "allowed (broad gain + silent)", "primary gain window", "broad gain window"]
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.13))
fig.suptitle("ORN-driven AL gain across coupling (shaded: 95% bootstrap CI over 64 replicas)", y=1.03)
fig.savefig(f"{F}/drive_gain.png"); plt.close(fig)

# --------------------------------------------------------------------------------------------- allowed strip
cols = {"none": "#ffffff", "gain": "#b7d3f6", "silent": "#f5c9b5", "both": "#2a78d6"}
fig, axs = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
gs_all = sorted({x["g"] for x in rows if x["g"] > 0})
edges = np.sqrt(np.array(gs_all[:-1]) * np.array(gs_all[1:]))
edges = np.concatenate([[gs_all[0] / np.sqrt(gs_all[1] / gs_all[0])], edges, [gs_all[-1] * np.sqrt(gs_all[-1] / gs_all[-2])]])
for ax, r in zip(axs, (0, 1)):
    for k, lev in enumerate(LEV):
        A = {x["g"]: x for x in sel("all", r, lev)}
        P = {x["g"]: x for x in sel("ant", r, lev)}
        for i, g in enumerate(gs_all):
            if g not in A:
                continue
            x = A[g]
            gain, sil = x["gain_primary"], x["S1"] and x["S2"]
            c = cols["both" if gain and sil else "gain" if gain else "silent" if sil else "none"]
            ax.add_patch(plt.Rectangle((edges[i], k - 0.42), edges[i + 1] - edges[i], 0.84, color=c, ec=GRID, lw=0.5))
            if gain and sil and g in P and not (P[g]["S1p"] and P[g]["S2p"]):
                ax.plot(g, k, "x", color="white", ms=5, mew=1.5)
            if not gain and x["gain_broad"] and sil:
                ax.plot(g, k, "o", ms=3.5, mfc="white", mec="#2a78d6")
    vlines(ax, r, text=(r == 0))
    if r == 1:
        ax.axvline(0.30, color=CAT[1], lw=1.2, ls="-.")
    ax.set_yticks(range(len(LEV))); ax.set_yticklabels([f"{levels[l]['rho_iso']:.2f}" for l in LEV])
    ax.set_ylim(-0.6, len(LEV) - 0.4); ax.set_ylabel(f"r = {r}\nORN level ρ_iso"); ax.grid(False)
    ax.set_xscale("log"); ax.set_xlim(edges[0], edges[-1]); gticks(ax)
axs[1].set_xlabel("coupling g")
leg = [Patch(color=cols["both"], label="allowed: realistic gain AND silent after removal"),
       Patch(color=cols["gain"], label="realistic gain only (persists after removal)"),
       Patch(color=cols["silent"], label="silent after removal, gain outside primary window"),
       Patch(facecolor="white", edgecolor=GRID, label="neither"),
       Line2D([], [], marker="o", ls="", mfc="white", mec="#2a78d6", ms=4, label="silent + broad gain window"),
       Line2D([], [], marker="x", ls="", color=INK2, ms=5, label="white ×: allowed, but fails palp-intact test")]
fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.16))
fig.suptitle("Where the model matches the ORN-driven AL (pre-registered criteria, plan.md)", y=1.0)
fig.savefig(f"{F}/drive_allowed.png"); plt.close(fig)

# --------------------------------------------------------------------------------------------- removal traces
LEV_SHOW = "0.04"
GSHOW = [0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.46]
gcol = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(GSHOW)))
fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
for ri, r in enumerate((0, 1)):
    for proto, ax in (("all", axs[ri, 0]), ("ant", axs[ri, 1])):
        for gi, gv in enumerate(GSHOW):
            f = [x for x in sel(proto, r, LEV_SHOW) if abs(x["g"] - gv) < 1e-9]
            if not f:
                continue
            d = np.load(f"{D}/runs/{f[0]['file']}")
            j = int(np.argmin(abs(d["g"] - gv))); names = list(d["reg_names"]); iu = names.index("uPN_ant")
            t = d["t"]; m = t > T1
            y = d["reg_rho"][j, :, iu, :].mean(1) / max(f[0]["driven"]["uPN_ant"], 1e-12)
            ax.plot(t[m] - T1, np.maximum(y[m], 1e-4), color=gcol[gi], lw=1.4, label=f"g = {gv:g}")
        ax.axhline(0.1, color=INK2, lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(8e-5, 3)
        ax.set_title(("all ORNs removed" if proto == "all" else "antennae removed, palps kept") + f" (r = {r})")
    ax = axs[ri, 2]
    for gi, gv in enumerate(GSHOW):
        f = [x for x in sel("all", r, LEV_SHOW) if abs(x["g"] - gv) < 1e-9]
        if not f:
            continue
        d = np.load(f"{D}/runs/{f[0]['file']}")
        j = int(np.argmin(abs(d["g"] - gv))); names = list(d["reg_names"])
        la = d["reg_last"][j, names.index("ALsys")] - T1
        tt = np.logspace(-1, np.log10(2999), 200)
        ax.plot(tt, [(la >= x).mean() for x in tt], color=gcol[gi], lw=1.4, label=f"g = {gv:g}")
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03); ax.set_title(f"P(AL still active), all ORNs removed (r = {r})")
for ax in axs[:, 0]:
    ax.set_ylabel("antennal uPN activity / driven level")
for ax in axs[1]:
    ax.set_xlabel("time since removal (model units; ×τ_u ≈ 10 ms)")
for ax in axs[:, :2].ravel():
    ax.text(0.02, 0.02, "1e-4 = floor for 0", transform=ax.transAxes, ha="left", fontsize=7, color=INK2)
h, l = axs[0, 0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.legend(h, l, loc="lower center", ncol=len(GSHOW), fontsize=8)
fig.suptitle(f"After ORN removal (ORN level {LEV_SHOW}): how fast and how far PN activity falls; dashed = 10 % criterion")
fig.savefig(f"{F}/drive_removal.png"); plt.close(fig)

# --------------------------------------------------------------------------------------------- regions
REG = [("ORN", "ORN"), ("uPN_ant", "antennal uPN"), ("LN_exc", "LN (ACh)"), ("LN_inh", "LN (GABA/Glu)"),
       ("AL_module", "AL module"), ("CNS_rest", "rest of CNS")]
fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for ri, r in enumerate((0, 1)):
    A = [x for x in sel("all", r, LEV_SHOW) if x["g"] > 0]
    g = np.array([x["g"] for x in A])
    for k, (key, lab) in enumerate(REG):
        axs[ri, 0].plot(g, [max(x["driven"][key], 1e-7) for x in A], color=CAT[k], lw=2, label=lab, marker="o", ms=3)
        if key != "ORN":
            axs[ri, 1].plot(g, [max(x["after_abs"][key]["1000-3000"], 1e-7) for x in A], color=CAT[k], lw=2,
                            label=lab, marker="o", ms=3)
    for ax in axs[ri]:
        vlines(ax, r, text=(ri == 0)); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-7, 1); gticks(ax)
    axs[ri, 0].set_ylabel(f"r = {r}\nactive fraction")
    axs[ri, 0].set_title("ORN drive on (500–1500)", pad=14 if ri == 0 else 4)
    axs[ri, 1].set_title("1000–3000 units after removing all ORNs", pad=14 if ri == 0 else 4)
axs[0, 0].legend(fontsize=7.5)
for ax in axs[1]:
    ax.set_xlabel("coupling g")
fig.suptitle(f"Activity by region (ORN level {LEV_SHOW}; 1e-7 = floor for 0)", y=1.0)
fig.tight_layout(); fig.savefig(f"{F}/drive_regions.png"); plt.close(fig)

# --------------------------------------------------------------------------------------------- persistence vs h=0 lifetimes
lr = json.load(open(f"{ROOT}/results/analysis2/longruns.json"))
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, r in zip(axs, (0, 1)):
    for k, lev in enumerate(LEV):
        A = [x for x in sel("all", r, lev) if x["g"] > 0]
        ax.plot([x["g"] for x in A], [max(x["ext_med"], 0.2) for x in A], color=BLUE[k], lw=2, marker="o", ms=3,
                label=lev_label(lev))
    if r == 1:
        try:
            L = lr["lifetimes"]["real"]
            pts = [(e["g"], e["tau_sim"]) for e in L if isinstance(e, dict) and e.get("tau_sim") is not None]
            if pts:
                gg, tt = zip(*sorted(pts))
                ax.plot(gg, np.array(tt) * np.log(2), color=CAT[1], lw=1.5, ls="--", label="h = 0 median lifetime (τ ln 2), full start")
        except Exception as e:
            print("longruns overlay skipped:", e)
    ax.axhline(3000, color=INK2, lw=0.8, ls=":"); ax.text(0.04, 3300, "end of run (censored)", fontsize=7, color=INK2)
    ax.axhline(100, color=INK2, lw=0.8, ls="--"); ax.text(0.04, 110, "≈1 s at τ_u = 10 ms", fontsize=7, color=INK2)
    vlines(ax, r); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("coupling g"); ax.set_title(f"r = {r}", pad=14)
    ax.set_xlim(0.035, 1.05); gticks(ax)
axs[0].set_ylabel("median time from removal to last AL activity")
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.14))
fig.suptitle("How long the AL keeps going after all ORNs are removed", y=1.04)
fig.savefig(f"{F}/drive_persist.png"); plt.close(fig)

# --------------------------------------------------------------------------------------------- fluctuations
fig, axs = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
for ri, r in enumerate((0, 1)):
    for k, lev in enumerate(LEV):
        A = [x for x in sel("all", r, lev) if x["g"] > 0]
        g = [x["g"] for x in A]
        axs[ri, 0].plot(g, [x["chi_time"] for x in A], color=BLUE[k], lw=2, marker="o", ms=3, label=lev_label(lev))
        axs[ri, 1].plot(g, [x["chi_rep"] for x in A], color=BLUE[k], lw=2, marker="o", ms=3)
    for ax in axs[ri]:
        vlines(ax, r, text=(ri == 0)); ax.axvline(0.30, color=CAT[1], lw=1.2, ls="-.")
        ax.set_xscale("log"); ax.set_yscale("log"); gticks(ax)
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5))); ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    axs[ri, 0].set_ylabel(f"r = {r}")
axs[0, 0].set_title("χ_time = n ⟨var_t ρ_module⟩ / ρ", pad=14); axs[0, 1].set_title("χ_rep = n var_rep(ρ̄_module) / ρ", pad=14)
for ax in axs[1]:
    ax.set_xlabel("coupling g")
h, l = axs[0, 0].get_legend_handles_labels()
fig.suptitle("AL-module fluctuations under ORN drive (driven window); dash-dot = driven-plane peak g = 0.30 (h = 1e-5, r = 1)")
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.legend(h, l, loc="lower center", ncol=5, fontsize=7.5)
fig.savefig(f"{F}/drive_fluct.png"); plt.close(fig)
print("figures written")

# --------------------------------------------------------------------------------------------- palp-intact residual
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, r in zip(axs, (0, 1)):
    for k, lev in enumerate(LEV):
        A = [x for x in sel("ant", r, lev) if x["g"] > 0]
        B = [x for x in sel("all", r, lev) if x["g"] > 0]
        ax.plot([x["g"] for x in A], [max(x["resid"]["1000-3000"], 1e-4) for x in A], color=BLUE[k], lw=2, marker="o",
                ms=3, label=lev_label(lev))
        ax.plot([x["g"] for x in B], [max(x["resid"]["100-300"], 1e-4) for x in B], color=BLUE[k], lw=1, ls="--")
    ax.axhline(0.1, color=INK2, lw=0.8, ls=(0, (2, 2))); ax.axhline(0.01, color=INK2, lw=0.8, ls=":")
    ax.text(0.98, 0.11, "10 % criterion", transform=ax.get_yaxis_transform(), ha="right", fontsize=7, color=INK2)
    ax.text(0.98, 0.011, "1 %", transform=ax.get_yaxis_transform(), ha="right", fontsize=7, color=INK2)
    vlines(ax, r); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.035, 1.05); ax.set_ylim(8e-5, 2); gticks(ax)
    ax.set_xlabel("coupling g"); ax.set_title(f"r = {r}", pad=14)
axs[0].set_ylabel("antennal uPN activity after removal / driven")
h, l = axs[0].get_legend_handles_labels()
h += [Line2D([], [], color=INK2, lw=2), Line2D([], [], color=INK2, lw=1, ls="--")]
l += ["antennae removed, palps kept (1000–3000 after)", "all ORNs removed (100–300 after)"]
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
fig.suptitle("Residual antennal PN activity after removal (1e-4 = floor for 0)", y=1.04)
fig.savefig(f"{F}/drive_palp.png"); plt.close(fig)
print("palp figure written")
