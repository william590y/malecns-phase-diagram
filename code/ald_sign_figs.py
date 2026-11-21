"""E2 (transmitter signs): figures from results/aldef/signs/summary.json (ald_sign_analyze.py) and the spectra.
-> figs/aldef/signs_phase.png, signs_curves.png, signs_ignition.png, signs_ensemble.png, signs_spectrum.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = "/MALECNS_ROOT"
RES = f"{ROOT}/results/aldef/signs"; FIG = f"{ROOT}/figs/aldef"
S = json.load(open(f"{RES}/summary.json"))
# palette (dataviz reference instance, light mode)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PHASE = {"Q": "#ecebe7", "S": "#b7d3f6", "L": "#5598e7", "D": "#184f95"}
PHASE_NAME = {"Q": "quiescent (dies fast, tau < 200)", "S": "slow window (200 <= tau < 3000)",
              "L": "sustained, localized (PR < N/4)", "D": "sustained, delocalized (PR >= N/4)"}
DET = ["base", "lLN1bc_inh", "lLN1bc_zero", "lLN1bc_lLN2_inh"]
LAB = {"base": "baseline", "lLN1bc_inh": "(i) lLN1_bc → −r", "lLN1bc_zero": "(ii) lLN1_bc → 0",
       "lLN1bc_lLN2_inh": "(iii) lLN1_bc + ACh lLN2 → −r"}
SPEC_OF = {"base": "base", "lLN1bc_inh": "lLN1bc_off", "lLN1bc_zero": "lLN1bc_off", "lLN1bc_lLN2_inh": "lLN1bc_lLN2ach_off"}
COL = {c: CAT[i] for i, c in enumerate(DET)}
GMIN, GMAX = 0.12, 30.0
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})


def segs(b):
    """Phase segments [(phase, g0, g1)] on [GMIN, GMAX] from boundaries S, on, D."""
    S_, on, D = b.get("S"), b.get("on"), b.get("D")
    cuts = [("Q", GMIN)]
    if S_ is not None:
        cuts.append(("S", max(S_, GMIN)))
    if on is not None:
        cuts.append(("L", max(on, GMIN)))
        if D is not None:
            cuts.append(("D", max(D, GMIN)))
    out = []
    for i, (ph, g0) in enumerate(cuts):
        g1 = cuts[i + 1][1] if i + 1 < len(cuts) else GMAX
        if g1 > g0:
            out.append((ph, g0, g1))
    return out


def strip(ax, y, b, h=0.62):
    for ph, g0, g1 in segs(b):
        ax.barh(y, np.log10(g1) - np.log10(g0), left=np.log10(g0), height=h, color=PHASE[ph], edgecolor=SURF, linewidth=1.5)


def logx(ax):
    ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 30]
    ax.set_xticks(np.log10(ticks)); ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(np.log10(GMIN), np.log10(GMAX))


# ------------------------------------------------------------------------------------------------- 1. phase strips
ens_rows = {}
for r in (0, 1, 3):
    e = S["ens"].get(f"ens_type_r{r}")
    if e:
        ens_rows[r] = e
fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.4), sharex=True)
for ax, r in zip(axes, (0, 1, 3)):
    labels, y = [], 0
    rows = []
    for c in DET:
        k = f"{c}_r{r}"
        if k in S["det"]:
            rows.append((LAB[c], S["det"][k]["bounds"], S["spec"].get(SPEC_OF[c], {}).get("g_lin"), COL[c]))
        elif c == "lLN1bc_inh" and r == 0:
            rows.append((LAB[c] + "  (= ii at r = 0)", S["det"].get(f"lLN1bc_zero_r0", {}).get("bounds", {}),
                         S["spec"].get("lLN1bc_off", {}).get("g_lin"), COL[c]))
    pb = S["pd2_baseline"].get(str(r))
    if pb:
        rows.insert(1 if f"base_r{r}" in S["det"] else 0, ("baseline, earlier pd2 runs (B = 32, pooled)", pb, None, INK2))
    for cname, lab0 in (("ens_type", "(iv) type-level ensemble"), ("ens_neuron", "(iv') neuron-level ensemble")):
      e = S["ens"].get(f"{cname}_r{r}")
      if e:
        ons = [x["bounds"]["on"] for x in e["runs"] if x["bounds"]["on"] is not None]
        med = {"S": np.median([x["bounds"]["S"] for x in e["runs"] if x["bounds"]["S"] is not None]),
               "on": float(np.median(ons)) if ons else None,
               "D": (float(np.median([x["bounds"]["D"] if x["bounds"]["D"] is not None else np.inf for x in e["runs"]]))
                     if ons else None)}
        if med["D"] is not None and not np.isfinite(med["D"]):
            med["D"] = None
        rows.append((f"{lab0}, median of {e['n_draws']} draws", med, None, INK2))
    for i, (lab, b, gl, col) in enumerate(rows[::-1]):
        strip(ax, i, b)
        if gl:
            ax.plot([np.log10(gl)] * 2, [i - 0.36, i + 0.36], color=INK, lw=2, solid_capstyle="butt")
        labels.append(lab)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(f"r = {r}  (inhibition strength)", loc="left", fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
logx(axes[-1]); axes[-1].set_xlabel("coupling g (raw units)")
hand = [Patch(color=PHASE[p], label=PHASE_NAME[p]) for p in "QSLD"] + [
    Line2D([], [], color=INK, lw=2, label="g_lin of the excitatory operator (spectral)")]
fig.legend(handles=hand, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Coarse phase diagram with the AL clique's transmitter changed (T = 3000, fully active start, h = 0)",
             x=0.02, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
fig.savefig(f"{FIG}/signs_phase.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------------------------------------ 2. curves
fig, axes = plt.subplots(3, 3, figsize=(11, 8.2), sharex=True)
for j, r in enumerate((0, 1, 3)):
    for c in DET:
        k = f"{c}_r{r}"
        if k not in S["det"]:
            continue
        d = S["det"][k]; g = np.log10(d["g"])
        al = np.array(d["alive_curve"]); pr = np.maximum(np.array(d["pr_curve"]), 1e-7)
        ms = np.array([p["occ_module_share"] if p["occ_module_share"] is not None else np.nan for p in d["points"]])
        kw = dict(color=COL[c], lw=2, marker="o", ms=3.5, label=LAB[c])
        axes[0, j].plot(g, al, **kw); axes[1, j].plot(g, pr, **kw); axes[2, j].plot(g, ms, **kw)
    axes[0, j].set_title(f"r = {r}", loc="left", fontsize=10)
    axes[1, j].set_yscale("log"); axes[1, j].axhline(0.25, color=INK2, lw=1, ls="--")
    axes[1, j].text(np.log10(GMIN) + 0.02, 0.3, "N/4 (delocalized)", fontsize=7.5, color=INK2)
    axes[1, j].set_ylim(1e-4, 1)
    axes[2, j].set_ylim(-0.03, 1.03)
    for i in range(3):
        axes[i, j].grid(color=GRID, lw=0.6); axes[i, j].set_axisbelow(True)
    logx(axes[2, j]); axes[2, j].set_xlabel("coupling g")
axes[0, 0].set_ylabel("fraction of replicas\nalive at T = 3000")
axes[1, 0].set_ylabel("participation ratio / N\n(occupancy, second half)")
axes[2, 0].set_ylabel("share of activity in the\n526-neuron AL module")
fig.legend(handles=[Line2D([], [], color=COL[c], lw=2, marker="o", ms=3.5, label=LAB[c]) for c in DET], ncol=4,
           frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 0.955))
fig.suptitle("Survival, spread and location of activity vs coupling, per transmitter condition (coarse + refinement sweeps)",
             x=0.02, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f"{FIG}/signs_curves.png", dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------------------------- 3. ignition site
REG = ["AL", "VNC", "AN/DN", "CB other", "OL intrinsic", "MB", "LH"]
RCOL = dict(zip(REG + ["other"], CAT[:7] + ["#b9b7b0"]))
bars = []
for c in DET:
    sp = S["spec"].get(SPEC_OF[c])
    for r in (0, 1, 3):
        k = f"{c}_r{r}"
        if k in S["det"] and "ignition" in S["det"][k]:
            ig = S["det"][k]["ignition"]
            bars.append((f"{LAB[c]}, r = {r}  (g = {ig['g']:.3g})", ig["region"], ig.get("PR_neurons"), ig.get("cos_v1")))
    if sp:
        bars.append((f"{LAB[c]}: leading eigenvector |v1|²", sp["mode0"]["region"], sp["mode0"]["inv_ipr"], None))
fig, ax = plt.subplots(figsize=(10, 0.36 * len(bars) + 1.6))
for i, (lab, reg, prn, cosv) in enumerate(bars[::-1]):
    left = 0.0
    tot = sum(reg.values())
    vals = {k: reg.get(k, 0.0) for k in REG}
    vals["other"] = max(1.0 - sum(vals.values()), 0.0)          # composition() lists only the top regions
    for k in REG + ["other"]:
        if vals[k] > 0:
            ax.barh(i, vals[k], left=left, color=RCOL[k], edgecolor=SURF, linewidth=1.2, height=0.7)
            left += vals[k]
    txt = (f"PR {prn:.0f}" if cosv is not None else f"PR (1/IPR) {prn:.0f}") if prn else ""
    if cosv is not None:
        txt += f",  cos(occ, |v1|) {cosv:.2f}"
    ax.text(1.01, i, txt, va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars[::-1]], fontsize=8)
ax.set_xlim(0, 1); ax.set_xlabel("share of occupancy at the first sustained g (eigenvector rows: share of |v1|²)")
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(color=RCOL[k], label=k) for k in REG + ["other"]], ncol=8, frameon=False, fontsize=8,
          loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.set_title("Where activity first sustains itself (ignition site), by region", loc="left", fontsize=11, pad=24)
fig.tight_layout()
fig.savefig(f"{FIG}/signs_ignition.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------------------------------------------- 4. ensemble
import glob
fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.6))
axes = axs.ravel()
ax = axes[0]
rows = []
for f in glob.glob(f"{RES}/spec/ens_type_ignore_*.json"):
    rows += json.load(open(f))["rows"]
lam = np.array([x["lam"][0] for x in rows]); exc = np.array([x["lLN1_bc_signs"].get("1", 0) for x in rows])
alm = np.array([x["mode0"]["AL_class_share"] >= 0.5 for x in rows])
bins = np.linspace(1.5, 7.5, 25)
ax.hist(lam[exc == 30], bins=bins, color=CAT[0], label=f"lLN1_bc excitatory ({(exc == 30).sum()} draws)", edgecolor=SURF)
ax.hist(lam[exc == 0], bins=bins, color=CAT[1], label=f"lLN1_bc not excitatory ({(exc == 0).sum()})", edgecolor=SURF)
ytop = ax.get_ylim()[1]; ax.set_ylim(0, ytop * 1.45)
for c, ls, tt in (("base", "-", "baseline"), ("lLN1bc_off", "--", "(i)/(ii)"), ("lLN1bc_lLN2ach_off", ":", "(iii)")):
    ax.axvline(S["spec"][c]["lam"][0], color=INK, lw=1.2, ls=ls)
    ax.text(S["spec"][c]["lam"][0] + 0.05, ytop * 1.38, tt, fontsize=7.5, color=INK, va="top")
ax.set_xlabel("λ1 of the whole-CNS excitatory operator"); ax.set_ylabel("draws")
ax.set_title(f"a  spectra of {len(rows)} type-level sign draws: the AL carries mode 1 in {alm.mean():.0%}", loc="left",
             fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.6, 0.86))


def excol(n):
    return CAT[0] if n == 30 else (CAT[1] if n == 0 else CAT[2])


tk = [0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
for j, (cname, title) in enumerate((("ens_type", "b  type-level draws (one sign per cell type)"),
                                    ("ens_neuron", "c  neuron-level draws (one sign per neuron)"))):
    ax = axes[1 + j]
    for r, mk in zip((0, 1, 3), ("o", "s", "^")):
        e = S["ens"].get(f"{cname}_r{r}")
        if not e:
            continue
        for x in e["runs"]:
            on = x["bounds"]["on"]; gl = x["g_lin_naive"]
            if gl is None or on is None:
                continue
            ex = x["lLN1_bc_signs"].get("1", 0) if x["lLN1_bc_signs"] else 0
            ax.scatter(gl, on, marker=mk, s=36, color=excol(ex), edgecolor=SURF, lw=0.8, zorder=3)
    for c, mkc in (("base", "*"),):
        for r, mk in zip((0, 1, 3), ("o", "s", "^")):
            k = f"{c}_r{r}"
            if k in S["det"]:
                ax.scatter(S["spec"]["base"]["g_lin_naive"], S["det"][k]["bounds"]["on"], marker=mk, s=60, facecolor="none",
                           edgecolor=INK, lw=1.3, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(tk); ax.set_xticklabels([f"{t:g}" for t in tk]); ax.set_yticks(tk); ax.set_yticklabels([f"{t:g}" for t in tk])
    ax.minorticks_off(); ax.set_xlim(0.12, 0.5); ax.set_ylim(0.15, 1.1)
    ax.plot([0.1, 3], [0.1, 3], color=GRID, lw=1, zorder=0)
    ax.text(0.40, 0.37, "onset = g_lin", fontsize=7, color=INK2, rotation=33)
    ax.set_xlabel("α_eff / λ1 of the draw's excitatory operator (linear threshold)")
    ax.set_ylabel("simulated sustained onset g (τ = 3000)")
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax = axes[3]
for cname, fill in (("ens_type", True), ("ens_neuron", False)):
    for r, mk in zip((0, 1, 3), ("o", "s", "^")):
        e = S["ens"].get(f"{cname}_r{r}")
        if not e:
            continue
        for x in e["runs"]:
            ig = x.get("ignition")
            if not ig:
                continue
            ex = x["lLN1_bc_signs"].get("1", 0) if x["lLN1_bc_signs"] else 0
            col = excol(ex)
            ax.scatter(x["bounds"]["on"], ig["module_share"], marker=mk, s=36, zorder=3,
                       color=col if fill else "none", edgecolor=SURF if fill else col, lw=0.8 if fill else 1.3)
SHORT = {"base": "base", "lLN1bc_inh": "i", "lLN1bc_zero": "ii", "lLN1bc_lLN2_inh": "iii"}
for c in DET:
    for r, mk in zip((0, 1, 3), ("o", "s", "^")):
        k = f"{c}_r{r}"
        if k in S["det"] and "ignition" in S["det"][k]:
            xx, yy = S["det"][k]["bounds"]["on"], S["det"][k]["ignition"]["module_share"]
            ax.scatter(xx, yy, marker=mk, s=70, color="white", edgecolor=INK, lw=1.3, zorder=4)
            off = {"base": (0, 9), "lLN1bc_inh": (-10, 7), "lLN1bc_zero": (0, 9), "lLN1bc_lLN2_inh": (11, 7)}[c]
            ax.annotate(SHORT[c], (xx, yy), xytext=off, textcoords="offset points", ha="center", fontsize=7.5,
                        color=INK, zorder=5)
ax.set_xscale("log"); ax.set_xticks([0.2, 0.3, 0.5, 0.7, 1, 1.5]); ax.set_xticklabels(["0.2", "0.3", "0.5", "0.7", "1", "1.5"])
ax.minorticks_off()
ax.set_xlabel("sustained onset g"); ax.set_ylabel("share of activity in the AL module at ignition")
ax.set_title("d  where it ignites: AL-module share at the first sustained g", loc="left", fontsize=9.5)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
hand = [Line2D([], [], marker=m, ls="", color=INK2, label=f"r = {r}") for r, m in zip((0, 1, 3), ("o", "s", "^"))] + [
    Patch(color=CAT[0], label="draw: all 30 lLN1_bc excitatory"), Patch(color=CAT[1], label="draw: no lLN1_bc excitatory"),
    Patch(color=CAT[2], label="draw: 16-28 of 30 lLN1_bc excitatory"),
    Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK2, label="hollow (d): neuron-level draw"),
    Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markersize=8,
           label="black ring: fixed-sign runs (b, c: baseline; d: base / i / ii / iii as labelled)")]
fig.legend(handles=hand, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Sign ensembles drawn from the release's transmitter confidences (24 type-level, 16 neuron-level draws; "
             "r = 0, 1, 3)", x=0.02, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.1, 1, 0.96))
fig.savefig(f"{FIG}/signs_ensemble.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------------------------------------------- 5. spectrum
fig, ax = plt.subplots(figsize=(8.5, 3.6))
conds = [("base", "baseline"), ("lLN1bc_off", "(i)/(ii) lLN1_bc not exc."), ("lLN1bc_lLN2ach_off", "(iii) + ACh lLN2 not exc."),
         ("allALLN_ach_off", "all 137 ACh AL LNs not exc.")]
RC2 = {"AL": CAT[0], "VNC": CAT[1], "AN/DN": CAT[2], "CB other": CAT[3], "CB motor/endocrine": CAT[4]}
for i, (c, lab) in enumerate(conds):
    sp = S["spec"].get(c)
    if not sp:
        continue
    for j, (lv, rg) in enumerate(zip(sp["lam"], sp["modes_region"])):
        ax.scatter(i + (j % 2) * 0.12 - 0.06, lv, s=46, color=RC2.get(rg, "#b9b7b0"), edgecolor=SURF, lw=0.8, zorder=3)
    ax.text(i + 0.12, sp["lam"][0], f" {sp['lam'][0]:.2f}  (g_lin {sp['g_lin']:.3f})", fontsize=7.5, va="center", color=INK)
ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[1] for c in conds], fontsize=8)
ax.set_ylabel("eigenvalue (top 8, real part)")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
used = {rg for c, _ in conds if c in S["spec"] for rg in S["spec"][c]["modes_region"]}
ax.legend(handles=[Patch(color=v, label=f"mode carried by {k}") for k, v in RC2.items() if k in used] +
          ([Patch(color="#b9b7b0", label="other region")] if used - set(RC2) else []), frameon=False, fontsize=7.5,
          loc="upper right")
ax.set_title("Top excitatory eigenvalues of the whole CNS under each sign change", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIG}/signs_spectrum.png", dpi=150); plt.close(fig)
print("figures written")
