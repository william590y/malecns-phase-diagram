"""m2_driven_figs.py -- figures of lane m2/driven (reads results/m2/driven/curves.json + summary.json; 2026-11-09).
figs/m2/driven_phase_M2.png      main (g, r) driven phase diagram of M2 (ORN 0.04), allowed region + lines
figs/m2/driven_phase_compare.png M1 / M2 / M2s / configuration null / weight-shuffled null at R_CORE
figs/m2/driven_curves.png        observables vs g at r = 0, 1, 4 for M1 / M2 / M2s
figs/m2/driven_orn_levels.png    allowed region and SS line vs ORN level; field slope s(g); static susceptibility
figs/m2/driven_variants.png      lines of the depression-sensitivity variants at r = 0, 1, 4
figs/m2/driven_peaks.png         fluctuation / susceptibility peak lines against the instability lines (M2, ORN 0.04)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_common as M

ROOT = M.ROOT
RES = f"{ROOT}/results/m2/driven"
FIG = f"{ROOT}/figs/m2"
os.makedirs(FIG, exist_ok=True)

# ordinal phase ramp (validated: dataviz validate_palette --ordinal, light): quiet = neutral, then one blue ramp
PH_COL = {"DQ": "#e4e3df", "DA": "#86b6ef", "DT": "#3987e5", "SS-L": "#1c5cab", "SS-D": "#0d366b"}
PH_NAME = {"DQ": "DQ driven, quiet (G < 0.29)", "DA": "DA driven, amplifying", "DT": "DT outlives input (transient)",
           "SS-L": "SS-L self-sustaining, localized", "SS-D": "SS-D self-sustaining, delocalized"}
ALLOW = "#eb6834"          # categorical slot 2 (orange): the data-allowed region
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160})

C = json.load(open(f"{RES}/curves.json"))
S = json.load(open(f"{RES}/summary.json"))


def curves(set_, model, graph, orn, palp=False):
    out = {}
    for k, c in C.items():
        kk = c["key"]
        if kk["set"] == set_ and kk["model"] == model and kk["graph"] == graph and abs(kk["orn"] - orn) < 1e-9 \
                and kk["palp"] == palp:
            out[kk["r"]] = c
    return dict(sorted(out.items()))


def any_curve(model, graph, orn, r, sets=("A1", "A2", "A3", "A4", "B1", "B2", "B3")):
    for s in sets:
        cs = curves(s, model, graph, orn)
        if r in cs:
            return cs[r]
    return None


def edges(g):
    lg = np.log(np.asarray(g, float))
    mid = (lg[1:] + lg[:-1]) / 2
    return np.exp(np.r_[lg[0] - (mid[0] - lg[0]), mid, lg[-1] + (lg[-1] - mid[-1])])


DEC = S.get("decay_lane", {}).get("rows", {})


def phase_map(ax, cs, title, lines=True, peaks=True, legend=False, ylabel=True, dec_key=None, gain_lines=True):
    rs = list(cs)
    for row, r in enumerate(rs):
        c = cs[r]; g = np.array(c["g"]); e = edges(g)
        cv = c["curve"]
        for i in range(len(g)):
            base = cv["phase_base"][i]
            ax.add_patch(Rectangle((e[i], row - 0.5), e[i + 1] - e[i], 1, facecolor=PH_COL.get(base, "#ffffff"),
                                   edgecolor=PH_COL.get(base, "#ffffff"), linewidth=0.3))
            ph = cv["phase"][i] or ""
            if "-O" in ph:
                ax.plot(g[i], row, "o", ms=5, mfc="#eda100", mec=INK, mew=0.7, zorder=8)
            if "-NS" in ph:
                ax.plot(g[i], row, "x", ms=3.5, color="#eda100" if base in ("DQ", "DA") else "white", mew=0.9, zorder=5)
        # allowed points: one orange outline per contiguous run of allowed grid points in this row
        al = [bool(x) and gain_lines for x in cv["allowed"]]
        i = 0
        while i < len(g):
            if al[i]:
                j = i
                while j + 1 < len(g) and al[j + 1]:
                    j += 1
                ax.add_patch(Rectangle((e[i], row - 0.42), e[j + 1] - e[i], 0.84, facecolor="none", edgecolor=ALLOW,
                                       linewidth=1.5, zorder=4))
                i = j + 1
            else:
                i += 1
    for row in range(len(rs) - 1):
        ax.axhline(row + 0.5, color="white", lw=0.8, zorder=3)
    if lines:
        def line(key, **kw):
            pts = [(cs[r]["lines"].get(key), row) for row, r in enumerate(rs) if cs[r]["lines"].get(key) is not None]
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, **kw)
        line("DT_SS", color="white", lw=1.4, ls="-", marker="D", ms=3, zorder=6)
        if gain_lines:
            line("allowed_hi", color=ALLOW, lw=1.4, ls="--", marker="|", ms=6, zorder=6)
            line("DQ_DA", color=INK, lw=0.9, ls=":", zorder=6)
        if dec_key:
            pts = [(DEC.get(f"{dec_key}_r{r:g}", {}).get("on"), row) for row, r in enumerate(rs)]
            pts = [p for p in pts if p[0] is not None]
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, ls="none", marker="o", ms=7, mfc="none", mec="#1baf7a", mew=1.6, zorder=7)
    if peaks:
        for row, r in enumerate(rs):
            p = cs[r]["peaks"]["F_AL"]
            if p["g"] is not None and p["resolved"]:
                ax.plot(p["g"], row, marker="*", ms=8, color="#e87ba4", mec=INK, mew=0.5, zorder=7)
    ax.axvline(M.G_LIN, color="#7f7f7f", lw=1.0, ls="-", zorder=3)
    ax.set_xscale("log")
    allg = np.concatenate([edges(cs[r]["g"]) for r in rs])
    ax.set_xlim(allg.min(), allg.max())
    ax.set_ylim(-0.5, len(rs) - 0.5)
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([f"{r:g}" for r in rs], fontsize=7 if len(rs) > 10 else 8)
    if ylabel:
        ax.set_ylabel("inhibition scale r")
    ax.set_xlabel("coupling g")
    ax.set_title(title, loc="left")


def legend_handles(peaks=True):
    h = [Patch(facecolor=PH_COL[k], edgecolor="none", label=PH_NAME[k]) for k in PH_COL]
    h += [Patch(facecolor="none", edgecolor=ALLOW, linewidth=1.5, label="data-allowed grid points (G 0.29-1.45, S1, S2)"),
          Line2D([], [], color=ALLOW, ls="--", lw=1.4, label="top of allowed region (interpolated)"),
          Line2D([], [], color=INK, ls=":", lw=0.9, label="G = 0.29 (DQ -> DA)"),
          Line2D([], [], color="#7f7f7f", marker="D", ms=3, lw=1.4, label="tau_post = 3000 (DT -> SS), white on map"),
          Line2D([], [], color="#7f7f7f", ls="-", lw=1.0, label="g_lin = 0.163 (silence linearly unstable)"),
          Line2D([], [], ls="none", marker="o", ms=7, mfc="none", mec="#1baf7a", mew=1.6,
                 label="setting (a) onset, h = 0 (lane m2/decay)"),
          Line2D([], [], ls="none", marker="o", ms=5, mfc="#eda100", mec=INK, label="-O oscillation flag (none strong)"),
          Line2D([], [], ls="none", marker="x", ms=4, color="#eda100", label="-NS non-stationary (10 % rule)")]
    if peaks:
        h.append(Line2D([], [], ls="none", marker="*", ms=8, color="#e87ba4", mec=INK, mew=0.5,
                        label="resolved AL-module Fano peak (chi_AL/rho_AL)"))
    return h


def fig_main():
    cs = curves("A1", "M2", "real", 0.04)
    if not cs:
        return
    fig = plt.figure(figsize=(11, 7.2))
    ax = fig.add_axes([0.07, 0.09, 0.58, 0.83])
    phase_map(ax, cs, "M2 driven phase diagram (ORN drive 0.04, 64 replicas per cell)", dec_key="M2_real")
    fig.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(0.67, 0.92), fontsize=8)
    n = sum(len(c["g"]) for c in cs.values())
    fig.text(0.67, 0.30, f"{len(cs)} r values, 40 coarse g each (0.05-30) + 8-point\nrefinements at r = 0, .25, .5, 1, 2, 4, 6 ({n} points).\n"
             "Driven window 500-1500; every ORN removed at 1500;\n"
             "tau_post = lifetime of non-ORN activity after removal.\n"
             "Allowed = PN/ORN gain 0.29-1.45 and PNs silent\nafter removal (E1 criteria).", fontsize=8, color=INK2,
             va="top")
    fig.savefig(f"{FIG}/driven_phase_M2.png")
    plt.close(fig)


def fig_compare():
    panels = [("A2", "M1", "real", "M1 (old model, no depression)"), ("A1", "M2", "real", "M2 (default)"),
              ("A4", "M2s", "real", "M2s (per-target lLN1_bc signs)"),
              ("A3", "M2", "cnull", "M2, configuration null"), ("A3", "M2", "wnull", "M2, weight-shuffled null")]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.4))
    axs = axs.ravel()
    for i, (s, m, gr, t) in enumerate(panels):
        cs = curves(s, m, gr, 0.04)
        cs = {r: c for r, c in cs.items() if r in M.R_CORE}
        if cs:
            phase_map(axs[i], cs, t, ylabel=(i % 3 == 0), dec_key=f"{m}_{gr}", gain_lines=(gr != "cnull"))
            axs[i].set_xlim(0.045, 33)
    axs[5].axis("off")
    axs[5].legend(handles=legend_handles(), loc="center left", fontsize=7.5)
    fig.suptitle("Driven phases at ORN drive 0.04: old model, new model, sign variant and the two nulls "
                 "(configuration null: ORN->PN wiring destroyed, so gain lines are not drawn)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{FIG}/driven_phase_compare.png")
    plt.close(fig)


def fig_curves():
    rs = (0, 1, 4)
    models = [("M1", "#7f7f7f", "A2"), ("M2", CAT[0], "A1"), ("M2s", CAT[1], "A4")]
    obs = [("rho_nonORN", "non-ORN activity rho_drv", "log"), ("G", "gain G = rho_uPN / rho_ORN", "log"),
           ("tau_post", "tau_post after ORN removal", "log"), ("F_AL", "AL-module Fano chi_AL / rho_AL", "log"),
           ("F_non", "non-ORN Fano chi_non / rho_non", "log"), ("tau_int", "autocorr. time tau_int (non-ORN)", "log"),
           ("PR_non_n", "PR_nonORN / n", "log")]
    fig, axs = plt.subplots(len(obs), len(rs), figsize=(12, 15), sharex=True)
    for j, r in enumerate(rs):
        c2 = curves("A1", "M2", "real", 0.04).get(r)
        for i, (f, lab, sc) in enumerate(obs):
            ax = axs[i, j]
            if c2 is not None:     # M2 phase background
                g = np.array(c2["g"]); e = edges(g)
                for k in range(len(g)):
                    ax.axvspan(e[k], e[k + 1], color=PH_COL.get(c2["curve"]["phase_base"][k], "white"), alpha=0.35, lw=0)
            for m, col, s in models:
                c = curves(s, m, "real", 0.04).get(r)
                if c is None:
                    continue
                y = np.array([np.nan if v is None else v for v in c["curve"][f]], float)
                if f == "tau_post":
                    y = np.where(np.isfinite(y), np.minimum(y, 2e4), 2e4)
                    y = np.maximum(y, 0.1)
                ax.plot(c["g"], y, "-o", ms=2.5, lw=1.5, color=col, label=m)
            if f == "G":
                ax.axhspan(0.29, 1.45, color=ALLOW, alpha=0.18, lw=0)
                ax.axhline(0.29, color=ALLOW, lw=0.8); ax.axhline(1.45, color=ALLOW, lw=0.8)
            if f == "tau_post":
                for v in (200, 3000):
                    ax.axhline(v, color=INK2, lw=0.8, ls="--")
                ax.text(0.05, 2.3e4, " >= 2e4 drawn at 2e4 (0-1 of 64 replicas died)", fontsize=6.5, color=INK2, va="bottom")
            ax.axvline(M.G_LIN, color=INK, lw=0.7, ls=(0, (2, 2)))
            ax.set_xscale("log"); ax.set_yscale(sc)
            ax.grid(True, color=GRID, lw=0.5)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            if i == 0:
                ax.set_title(f"r = {r}", loc="left")
            if i == len(obs) - 1:
                ax.set_xlabel("coupling g")
    axs[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Driven observables vs g (ORN 0.04; background = M2 phase: grey DQ, light blue DA, blue DT, "
                 "dark SS)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(f"{FIG}/driven_curves.png")
    plt.close(fig)


def fig_levels():
    levels = [(0.01, "B1"), (0.02, "B1"), (0.04, "A1"), (0.07, "B1"), (0.10, "B1")]
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.8))
    ax = axs[0]
    rows = []
    for row, r in enumerate(M.R_CORE):
        for li, (o, s) in enumerate(levels):
            c = curves(s, "M2", "real", o).get(r)
            if c is None:
                continue
            y = row + (li - 2) * 0.15
            L = c["lines"]
            if L.get("allowed_lo") is not None and L.get("allowed_hi") is not None:
                ax.plot([L["allowed_lo"], L["allowed_hi"]], [y, y], color=ramp[li], lw=3.2, solid_capstyle="butt")
            if L.get("DT_SS") is not None:
                ax.plot(L["DT_SS"], y, "D", ms=4, color=ramp[li], mec=INK, mew=0.5)
    ax.axvline(M.G_LIN, color=INK, lw=0.7, ls=(0, (2, 2)))
    ax.set_xscale("log"); ax.set_yticks(range(len(M.R_CORE))); ax.set_yticklabels([f"{r:g}" for r in M.R_CORE])
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 3]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "3"])
    ax.minorticks_off()
    ax.set_ylabel("inhibition scale r"); ax.set_xlabel("coupling g"); ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.set_title("(a) allowed region and DT->SS by ORN level", loc="left", fontsize=9)
    ax.legend(handles=[Line2D([], [], color=ramp[i], lw=3.2, label=f"ORN level {o:g}") for i, (o, _) in enumerate(levels)]
              + [Line2D([], [], ls="none", marker="D", color="#7f7f7f", mec=INK, label="DT->SS (tau_post 3000)")],
              fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    fs = S.get("field_scaling", {})
    for k, (key, lab) in enumerate((("s_eff", "(b) field slope vs transmitted ORN input"),
                                     ("chi_h", "(c) static susceptibility d rho_nonORN / d h_ORN"))):
        ax = axs[k + 1]
        for i, r in enumerate((0, 1, 4)):
            d = fs.get(str(float(r))) or fs.get(str(r))
            if d is None:
                continue
            g = np.array(d["g"]); y = np.array(d[key], float)
            lo = np.array(d[f"{key}_lo"], float); hi = np.array(d[f"{key}_hi"], float)
            ax.fill_between(g, lo, hi, color=CAT[i], alpha=0.18, lw=0)
            ax.plot(g, y, "-o", ms=2.5, lw=1.5, color=CAT[i], label=f"r = {r}")
            if key == "s_eff":
                ax.plot(g, np.array(d["s"], float), ls="--", lw=0.9, color=CAT[i], alpha=0.8)
            c = curves("A1", "M2", "real", 0.04).get(r)
            if c is not None and c["lines"].get("DT_SS") is not None:
                ax.axvline(c["lines"]["DT_SS"], color=CAT[i], lw=0.9, ls="--")
        if key == "s_eff":
            ax.axhline(1, color=INK2, lw=0.7); ax.axhline(0.5, color=INK2, lw=0.7, ls=":")
            ax.text(0.052, 1.02, "linear response", fontsize=7, color=INK2, va="bottom")
            ax.text(0.052, 0.52, "mean-field critical (1/delta)", fontsize=7, color=INK2, va="bottom")
            ax.set_ylabel("dln rho_nonORN / dln(rho_ORN e_ORN)\n(dashed: vs rho_ORN alone)", fontsize=8)
            ax.set_ylim(-0.05, 1.15)
        else:
            ax.set_yscale("log")
        ax.axvline(M.G_LIN, color=INK, lw=0.7, ls=(0, (2, 2)))
        ax.set_xscale("log"); ax.set_xlabel("coupling g"); ax.grid(True, color=GRID, lw=0.5)
        ax.set_title(lab, loc="left", fontsize=9)
        ax.legend(fontsize=7.5, title="dashed: DT->SS line", title_fontsize=7)
    fig.suptitle("M2: dependence on the ORN drive level (levels 0.01-0.10; 95 % replica bootstrap bands)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{FIG}/driven_orn_levels.png")
    plt.close(fig)


def fig_variants():
    rows = [("M1", 0.04, "M1 (no depression)"), ("M1s", 0.04, "M1s (signs fix only)"), ("M2", 0.04, "M2 (default)"),
            ("M2s", 0.04, "M2s"), ("M2_allsyn", 0.04, "M2 all-synapse depression"), ("M2_KW09", 0.04, "M2 KW09 (U .28, TAU 240)"),
            ("M2_N15", 0.04, "M2 N15 (U .22, TAU 89)"), ("M2_tau30", 0.04, "M2 TAU 30 (tau_u 50 ms)"),
            ("M2_tau30", 0.10, "M2 TAU 30, ORN 0.10")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
    for j, r in enumerate((0, 1, 4)):
        ax = axs[j]
        for i, (m, o, lab) in enumerate(rows):
            c = any_curve(m, "real", o, r)
            y = len(rows) - 1 - i
            if c is None:
                continue
            L = c["lines"]
            if L.get("allowed_lo") is not None and L.get("allowed_hi") is not None:
                ax.plot([L["allowed_lo"], L["allowed_hi"]], [y, y], color=ALLOW, lw=4, solid_capstyle="butt")
            if L.get("silent_lost") is not None:
                ax.plot(L["silent_lost"], y, "|", ms=10, mew=1.6, color=CAT[0])
            if L.get("DT_SS") is not None:
                ax.plot(L["DT_SS"], y, "D", ms=5, color="#0d366b")
            p = c["peaks"]["F_AL"]
            if p["g"] is not None and p["resolved"]:
                ax.plot(p["g"], y, "*", ms=8, color="#e87ba4", mec=INK, mew=0.5)
        ax.axvline(M.G_LIN, color=INK, lw=0.7, ls=(0, (2, 2)))
        ax.set_xscale("log"); ax.set_xlim(0.05, 12); ax.grid(True, axis="x", color=GRID, lw=0.5)
        ax.set_title(f"r = {r}", loc="left"); ax.set_xlabel("coupling g")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab for _, _, lab in rows][::-1])
    fig.legend(handles=[Line2D([], [], color=ALLOW, lw=4, label="data-allowed region"),
                           Line2D([], [], ls="none", marker="|", ms=10, mew=1.6, color=CAT[0], label="PNs no longer silent after removal"),
                           Line2D([], [], ls="none", marker="D", ms=5, color="#0d366b", label="DT -> SS (tau_post = 3000)"),
                           Line2D([], [], ls="none", marker="*", ms=8, color="#e87ba4", mec=INK, mew=0.5, label="resolved AL Fano peak")],
                  fontsize=8, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=4)
    fig.suptitle("Driven lines under the synapse-model variants (ORN 0.04 unless stated)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.07, 1, 0.95))
    fig.savefig(f"{FIG}/driven_variants.png")
    plt.close(fig)


def fig_peaks():
    cs = curves("A1", "M2", "real", 0.04)
    if not cs:
        return
    rs = list(cs)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    mk = {"F_AL": ("*", "#e87ba4", 9, "AL-module Fano chi_AL/rho_AL"), "F_non": ("o", CAT[2], 5, "non-ORN Fano chi/rho"),
          "chi_non": ("s", CAT[3], 5, "non-ORN chi"), "tau_int": ("^", CAT[6], 5, "autocorrelation time tau_int")}
    for k, (m, col, ms, lab) in mk.items():
        for row, r in enumerate(rs):
            p = cs[r]["peaks"][k]
            if p["g"] is None:
                continue
            ax.plot(p["g"], row + {"F_AL": 0.2, "F_non": 0.07, "chi_non": -0.07, "tau_int": -0.2}[k], m, ms=ms,
                    color=col if p["resolved"] else "none", mec=col, mew=1.0)
    def line(key, **kw):
        pts = [(cs[r]["lines"].get(key), row) for row, r in enumerate(rs) if cs[r]["lines"].get(key) is not None]
        if pts:
            x, y = zip(*pts); ax.plot(x, y, **kw)
    line("DT_SS", color="#0d366b", lw=1.6, label="DT -> SS (tau_post = 3000)")
    line("silent_lost", color=CAT[0], lw=1.2, ls="--", label="PNs no longer silent after removal")
    line("allowed_hi", color=ALLOW, lw=1.2, ls="--", label="top of allowed region")
    fs = S.get("field_scaling", {})
    pts = [(fs[k]["s_eff_half"], rs.index(float(k)) if float(k) in rs else None) for k in fs if fs[k].get("s_eff_half")]
    pts = [p for p in pts if p[1] is not None]
    if pts:
        x, y = zip(*sorted(pts, key=lambda t: t[1])); ax.plot(x, y, color=CAT[4], lw=1.4, ls="-.", marker="o", ms=4,
                                                                label="field slope vs transmitted input = 0.5 (post hoc)")
    for k in fs:
        if float(k) not in rs:
            continue
        p = fs[k]["chi_h_peak"]
        if p["g"] is not None:
            ax.plot(p["g"], rs.index(float(k)) + 0.3, "D", ms=5, color=CAT[7] if p["resolved"] else "none", mec=CAT[7])
    ax.plot([], [], "D", ms=5, color=CAT[7], label="static susceptibility d rho / d h peak (filled = resolved)")
    ax.text(29, -0.9, "open markers at g = 30: still rising\nat the grid end (no peak)", fontsize=7, color=INK2,
            ha="right", va="bottom")
    pts = [(DEC.get(f"M2_real_r{r:g}", {}).get("on"), row) for row, r in enumerate(rs)]
    pts = [p for p in pts if p[0] is not None]
    if pts:
        x, y = zip(*pts); ax.plot(x, y, color="#1baf7a", lw=1.4, marker="o", ms=3, label="setting (a) onset, h = 0 (lane m2/decay)")
    ax.axvline(M.G_LIN, color="#7f7f7f", lw=1.0, label="g_lin = 0.163")
    for k, (m, col, ms, lab) in mk.items():
        ax.plot([], [], m, ms=ms, color=col, mec=col, label=f"{lab} peak (filled = resolved)")
    ax.set_xscale("log"); ax.set_yticks(range(len(rs))); ax.set_yticklabels([f"{r:g}" for r in rs], fontsize=7)
    tk = [0.2, 0.5, 1, 2, 5, 10, 20]
    ax.set_xticks(tk); ax.set_xticklabels([f"{t:g}" for t in tk]); ax.minorticks_off()
    ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition scale r"); ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.set_title("M2 (ORN 0.04): where the fluctuation peaks sit relative to the instability lines", loc="left")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(f"{FIG}/driven_peaks.png")
    plt.close(fig)


def fig_palp():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for j, r in enumerate((0, 1)):
        ax = axs[j]
        for i, (m, col) in enumerate((("M2", CAT[0]), ("M2s", CAT[1]))):
            ca = curves({"M2": "A1", "M2s": "A4"}[m], m, "real", 0.04).get(r)
            if ca is not None and ca["lines"].get("allowed_lo") and ca["lines"].get("allowed_hi"):
                y0 = 1.25 if m == "M2" else 1.05
                ax.plot([ca["lines"]["allowed_lo"], ca["lines"]["allowed_hi"]], [y0, y0], color=col, lw=5,
                        solid_capstyle="butt", alpha=0.55)
                ax.text(ca["lines"]["allowed_hi"] * 1.04, y0, f"{m} allowed region", fontsize=7, color=INK2, va="center")
            c = curves("B4", m, "real", 0.04, palp=True).get(r)
            if c is None:
                continue
            ax.plot(c["g"], c["curve"]["S1p_resid"], "-o", ms=3, color=col, label=f"{m}: 100-300 after removal")
            ax.plot(c["g"], c["curve"]["S2p_resid"], "--s", ms=3, color=col, label=f"{m}: 1000-3000 after removal")
        ax.axhline(0.1, color=INK2, lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(8e-3, 1.6)
        ax.grid(True, color=GRID, lw=0.5); ax.set_xlabel("coupling g"); ax.set_title(f"r = {r}", loc="left")
    axs[0].set_ylabel("antennal PN activity / driven value\n(palp ORNs still driven)")
    axs[1].legend(fontsize=7.5)
    fig.suptitle("Palp-intact removal test (Olsen 2007): antennal ORNs removed, palp ORNs still driven; pass = both <= 0.1.\n"
                 "Bars at the top: the data-allowed region (gain + all-ORN removal test) of the same model", x=0.01,
                 ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"{FIG}/driven_palp.png")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["main", "compare", "curves", "levels", "variants", "peaks", "palp"]
    for w in which:
        try:
            {"main": fig_main, "compare": fig_compare, "curves": fig_curves, "levels": fig_levels,
             "variants": fig_variants, "peaks": fig_peaks, "palp": fig_palp}[w]()
            print("ok", w)
        except Exception as ex:      # partial data during the runs
            import traceback; traceback.print_exc()
            print("FAILED", w, ex)
