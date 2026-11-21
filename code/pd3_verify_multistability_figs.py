"""VERIFICATION COPY (pd3_verify_multistability, 2026-11-09) of pd3_multistab_figs.py: identical except that the panel-(f) "no survivors"
label of multistab_AL_symmetry.png is computed from the raw extinction times (the original hardcoded "before t = 500", which is
wrong for r = 6, g = 0.43: 16 of 256 replicas lived past t = 500). Run: python pd3_verify_multistability_figs.py m1

Figures of the pd3 multistability program (reads results/pd3/multistab/analysis_*.json and the raw npz files).
  figs/pd3/multistab_AL_symmetry.png     M1: A(t) traces, t_sym against r, keep fraction / KS, late A per start against g (closure
                                         B1 windows shaded), late-A histograms in the closure's B1 windows
  figs/pd3/multistab_hysteresis.png      M2a/M2c: up/down rho and PR/N against g (D = 50 / 200 / 800, h = 1e-5), loop area against D,
                                         h = 0 extinction, r-sweep loops
  figs/pd3/multistab_ic_attractors.png   M3: late module-activity heat strips per start, 4 r x 6 g
  figs/pd3/multistab_qmf_cells.png       M3b: late projections onto the QMF attractor-difference vectors at the QMF multistable cells
Colours: Okabe-Ito for categories (validated subset; the 4-start palette has a CVD-floor pair, so every start also has its own
marker); cividis for sequential maps.
Usage: python pd3_multistab_figs.py [m1] [m2] [m3] [m3b]
"""
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm

ROOT = "/MALECNS_ROOT"
D_ = f"{ROOT}/results/pd3/multistab"
FIG = f"{ROOT}/figs/pd3"
OI = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "purple": "#CC79A7", "orange": "#E69F00", "sky": "#56B4E9",
      "black": "#000000", "grey": "#7F7F7F"}
START_C = {"AL-L": OI["blue"], "AL-R": OI["verm"], "AL": OI["green"], "all": OI["purple"]}
START_M = {"AL-L": "o", "AL-R": "s", "AL": "^", "all": "D"}
GON = {0: 0.266, 0.5: 0.306, 1: 0.344, 2: 0.402, 3: 0.456, 4: 0.508, 6: 0.579}
CLOSURE_B1 = {3: 0.2471, 4: 0.2871, 6: 0.4280}
G_LIN = 0.1632
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def J(p):
    return json.load(open(f"{D_}/analysis_{p}.json"))


# ------------------------------------------------------------------------------------------------------------------- M1
def fig_m1():
    from matplotlib.ticker import LogLocator, FuncFormatter
    A = J("m1")["per_r"]
    rs = sorted(A, key=float)
    fig = plt.figure(figsize=(17, 19))
    sf = fig.subfigures(4, 1, height_ratios=[1, 1, 1.05, 0.95], hspace=0.04)
    gfmt = FuncFormatter(lambda v, _: f"{v:g}")
    # row 1: traces
    sel = [("0", 0.266), ("1", 0.5504), ("4", 0.32), ("6", 0.58)]
    axs = sf[0].subplots(1, 4)
    for k, (r, g) in enumerate(sel):
        ax = axs[k]
        if r not in A:
            ax.set_visible(False); continue
        rows = A[r]["rows"]; row = min(rows, key=lambda x: abs(x["g"] - g))
        t = np.array(row["trace_t"])
        for s in ("AL-L", "AL-R"):
            n = np.array(row[f"trace_{s}_n_AL_active"]); ok = n >= 8
            med = np.where(ok, np.array(row[f"trace_{s}_median"], float), np.nan)
            q1 = np.where(ok, np.array(row[f"trace_{s}_q25"], float), np.nan); q3 = np.where(ok, np.array(row[f"trace_{s}_q75"], float), np.nan)
            ax.fill_between(t, q1, q3, color=START_C[s], alpha=0.2, lw=0)
            ax.plot(t, med, color=START_C[s], lw=2, label=f"start {s} only (median, IQR)")
            if (~ok).any() and ok.any():
                tend = t[np.where(ok)[0][-1]]
                ax.axvline(tend, color=START_C[s], lw=0.8, ls=":")
        for y in (0.3, -0.3):
            ax.axhline(y, color=OI["grey"], lw=0.8, ls=":")
        for y in (0.6, -0.6):
            ax.axhline(y, color=OI["grey"], lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_ylim(-1.05, 1.05); ax.set_xlim(0.2, 1000)
        x = row["g"] / GON[float(r)]
        cl = "\nclosure predicts B1L | B1R here" if row.get("closure_predicts_B1") else ""
        surv = row["starts"]["AL-L"]["alive_T"]
        ax.set_title(f"r = {r}, g = {row['g']:.3g} ({x:.2f} x onset){cl}\nAL-L start: {surv:.0%} of replicas alive at t = 1000", fontsize=9.5)
        ax.set_xlabel("model time t"); ax.set_ylabel("lobe asymmetry A(t)" if k == 0 else "")
        if k == 0:
            ax.legend(loc="lower left", frameon=False)
            ax.text(0.25, 0.34, "|A| = 0.3 (t_sym)", fontsize=8, color=OI["grey"])
            ax.text(0.25, 0.64, "A = 0.6 (keep threshold)", fontsize=8, color=OI["grey"])
        if k == 2:
            ax.text(0.03, 0.03, "curves end (dotted) when < 8 replicas\nstill have an active AL", transform=ax.transAxes, fontsize=7.5)
    sf[0].subplots_adjust(top=0.78, wspace=0.25)
    sf[0].suptitle("(a) Left/right AL asymmetry A = (f_L - f_R)/(f_L + f_R) after starting one lobe only (64 replicas per start; whole brain, h = 0)",
                   fontsize=11, weight="bold", x=0.02, ha="left")
    # row 2
    axs = sf[1].subplots(1, 3)
    mults = [1.0, 1.25, 1.6, 2.5, 4, 8]
    cmap = plt.get_cmap("cividis")
    ax = axs[0]
    for mi, mu in enumerate(mults):
        xs, ys, lo, hi = [], [], [], []
        for r in rs:
            row = min(A[r]["rows"], key=lambda x: abs(x["x_gon"] - mu))
            if abs(row["x_gon"] - mu) > 0.01:
                continue
            v = [row["starts"][s]["t_sym"] for s in ("AL-L", "AL-R")]
            meds = [x["median"] for x in v if x["median"] is not None]
            if not meds:
                continue
            xs.append(float(r) + (mi - 2.5) * 0.05); ys.append(np.mean(meds))
            lo.append(min(x["q25"] for x in v if x["q25"] is not None)); hi.append(max(x["q75"] for x in v if x["q75"] is not None))
        c = cmap(mi / (len(mults) - 1))
        ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], color=c, marker="o", ms=5, lw=1.5,
                    capsize=2, label=f"{mu:g} x onset")
    ax.set_yscale("log"); ax.set_ylim(0.7, 30)
    ax.yaxis.set_major_formatter(gfmt); ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v in (2, 5, 20) else ""))
    ax.set_xlabel("inhibition scale r"); ax.set_ylabel("t_sym (model time)\nmean of the L- and R-start medians; bars: IQR")
    ax.set_title("(b) time to symmetrise (first |A| < 0.3)")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", title="g =", title_fontsize=8)
    ax = axs[1]
    for mi, mu in enumerate(mults):
        xs, ys = [], []
        for r in rs:
            row = min(A[r]["rows"], key=lambda x: abs(x["x_gon"] - mu))
            if abs(row["x_gon"] - mu) > 0.01:
                continue
            fl = row["starts"]["AL-L"]["frac_surv_keeping_start_lobe"]
            if fl is None:
                continue
            xs.append(float(r) + (mi - 2.5) * 0.05); ys.append(fl)
        ax.plot(xs, ys, color=cmap(mi / (len(mults) - 1)), marker="o", ms=5, lw=1.5)
    for r in rs:
        for row in A[r]["rows"]:
            if row["closure_predicts_B1"] and min(abs(row["x_gon"] - m) for m in mults) > 0.01:
                fl = row["starts"]["AL-L"]["frac_surv_keeping_start_lobe"]
                if fl is not None:
                    ax.plot(float(r), fl, marker="*", ms=12, color=OI["orange"], mec="k", mew=0.5, ls="none")
    ax.plot([], [], marker="*", ms=12, color=OI["orange"], mec="k", mew=0.5, ls="none", label="closure B1-window points with survivors")
    ax.axhline(0.25, color=OI["verm"], ls="--", lw=1); ax.text(0.05, 0.27, "criterion c1: 25%", color=OI["verm"], fontsize=8)
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("inhibition scale r"); ax.set_ylabel("fraction of surviving AL-L-start replicas\nwith late mean A > 0.6")
    ax.set_title("(c) does a one-lobe state persist? (t in [500, 1000])"); ax.legend(frameon=False, loc="upper right")
    ax = axs[2]
    for r in rs:
        for row in A[r]["rows"]:
            k = row["ks_L_vs_R"]
            if k is None:
                continue
            mu = row["x_gon"]
            onm = min(range(len(mults)), key=lambda i: abs(mults[i] - mu))
            star = abs(mults[onm] - mu) >= 0.01
            col = OI["orange"] if star else cmap(onm / (len(mults) - 1))
            ax.plot(float(r) + (0 if star else (onm - 2.5) * 0.05), max(k["p"], 1e-12), marker="*" if star else "o", ms=12 if star else 6,
                    color=col, mec="k" if star else col, mew=0.5, ls="none")
    ax.axhline(1e-3, color=OI["verm"], ls="--", lw=1); ax.text(0.05, 1.6e-3, "criterion c2: p = 0.001", color=OI["verm"], fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(1e-6, 2); ax.set_xlabel("inhibition scale r")
    ax.set_ylabel("KS p-value, late A:\nAL-L start vs AL-R start")
    ax.set_title("(d) do the two one-lobe starts end in different states?")
    sf[1].suptitle("Symmetrisation statistics (colour: g as a multiple of the simulated onset g_on(r); stars: closure B1-window points)",
                   fontsize=11, weight="bold", x=0.02, ha="left")
    # row 3: late A per start vs g
    axs = sf[2].subplots(1, len(rs), sharey=True)
    for k, r in enumerate(rs):
        ax = axs[k]
        rows = A[r]["rows"]; G = np.array([x["g"] for x in rows])
        if float(r) in CLOSURE_B1:
            ax.axvspan(CLOSURE_B1[float(r)], 3.0, color=OI["orange"], alpha=0.15, lw=0)
            ax.text(CLOSURE_B1[float(r)] * 1.05, 0.95, "closure:\nB1L | B1R", fontsize=7.5, color="#8a5a00", va="top")
        for si, s in enumerate(("AL-L", "AL-R", "AL", "all")):
            m = np.array([x["starts"][s]["late_A_mean"] if x["starts"][s]["late_A_mean"] is not None else np.nan for x in rows], float)
            sd = np.array([x["starts"][s]["late_A_sd"] if x["starts"][s]["late_A_sd"] is not None else np.nan for x in rows], float)
            ax.errorbar(G * (1 + 0.03 * (si - 1.5)), m, yerr=sd, color=START_C[s], marker=START_M[s], ms=5, lw=1, capsize=1.5,
                        ls="none", label=f"start {s}")
        for x in rows:
            if all(x["starts"][s]["n_surv"] == 0 for s in ("AL-L", "AL-R", "AL", "all")):
                ax.plot(x["g"], -0.9, marker="x", color="k", ms=6, ls="none")
        ax.axhline(0, color=OI["grey"], lw=0.6)
        ax.set_xscale("log"); ax.set_ylim(-1, 1); ax.set_title(f"r = {r}")
        ax.xaxis.set_major_formatter(gfmt); ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_xticks([t for t in (0.3, 1, 3) if G.min() * 0.8 <= t <= G.max() * 1.25])
        ax.set_xlabel("g")
        if k == 0:
            ax.set_ylabel("late A (t in [500, 1000]):\nmean +- SD over survivors")
            ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 0.08))
            ax.text(0.03, 0.02, "x: no survivors", transform=ax.transAxes, fontsize=7.5)
    sf[2].suptitle("(e) Late asymmetry per start against g (shaded: g range where de Candia's closure has coexisting one-lobe states B1L | B1R, "
                   "theory_module2.json)", fontsize=11, weight="bold", x=0.02, ha="left")
    # row 4: histograms in the closure windows
    pts = [("4", 0.29), ("4", 0.32), ("4", 0.40), ("6", 0.43), ("6", 0.58), ("6", 0.70)]
    axs = sf[3].subplots(1, 6)
    bins = np.linspace(-1, 1, 41)
    for k, (r, g) in enumerate(pts):
        ax = axs[k]
        if r not in A:
            ax.set_visible(False); continue
        row = min(A[r]["rows"], key=lambda x: abs(x["g"] - g))
        nsv = sum(len(row["late_A_values"][s]) for s in ("AL-L", "AL-R", "AL", "all"))
        for s in ("AL-L", "AL-R", "AL", "all"):
            v = np.array(row["late_A_values"][s], float)
            if len(v):
                ax.hist(v, bins=bins, histtype="step", lw=1.8, color=START_C[s], label=f"{s} (n = {len(v)})")
        ax.set_title(f"r = {r}, g = {row['g']:.3g} ({row['x_gon']:.2f} x onset)", fontsize=9.5)
        ax.set_xlabel("late A per replica"); ax.set_xlim(-1, 1)
        if k == 0:
            ax.set_ylabel("replicas")
        if nsv == 0:
            _z = np.load(f"{D_}/al_sym_r{r}.npz", allow_pickle=True); _G = list(json.loads(str(_z["meta"]))["g"])
            _e = _z["ext_step"][_G.index(row["g"])] * 0.2          # verification fix: label from the raw extinction times
            _n500 = int((_e >= 500).sum())
            ax.text(0.5, 0.5, f"no survivors:\nall 256 replicas\nextinct by t = {_e.max():.0f}\n({_n500} of 256 alive\npast t = 500)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9)
            ax.set_yticks([])
        else:
            ax.legend(frameon=False, fontsize=7, loc="upper left")
            ax.text(0.98, 0.70, f"KS(L, R)\np = {row['ks_L_vs_R']['p']:.2g}" if row["ks_L_vs_R"] else "KS: < 2\nsurvivors\nper start",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    sf[3].suptitle("(f) Late-A histograms at the closure's B1-window points (closure: r = 4 B1L | B1R from g = 0.287; r = 6 from 0.428)",
                   fontsize=11, weight="bold", x=0.02, ha="left")
    fig.savefig(f"{FIG}/multistab_AL_symmetry.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote multistab_AL_symmetry.png")


# ------------------------------------------------------------------------------------------------------------------- M2
def fig_m2():
    from matplotlib.lines import Line2D
    A = J("m2a"); runs = A["runs"]; h0 = A["h0"]
    C = J("m2c") if os.path.exists(f"{D_}/analysis_m2c.json") else {}
    rs = ["0", "0.5", "1", "2", "4", "6"]
    fig = plt.figure(figsize=(18, 19))
    sf = fig.subfigures(4, 1, height_ratios=[1.1, 1, 1, 1], hspace=0.03)
    ls = {50: ":", 200: "--", 800: "-"}
    for row_i, (obs, ylab) in enumerate([("rho", "mean fraction active rho\n(last half of each dwell)"),
                                          ("PR", "participation ratio / N\n(replica-mean occupancy)")]):
        axs = sf[row_i].subplots(1, 6)
        for k, r in enumerate(rs):
            ax = axs[k]
            for D in (50, 200, 800):
                key = f"r{r}_h1e-05_D{D}"
                if key not in runs:
                    continue
                v = runs[key]; G = np.array(v["g"])
                up = np.array(v["rho_up" if obs == "rho" else "PR_up"]); dn = np.array(v["rho_down" if obs == "rho" else "PR_down"])
                ax.plot(G, up, color=OI["verm"], ls=ls[D], lw=1.7)
                ax.plot(G, dn, color=OI["blue"], ls=ls[D], lw=1.7)
            if obs == "rho" and r in h0 and h0[r]["g_ext_median"] is not None:
                q = h0[r]["g_ext_q25_q75"]
                ax.axvspan(q[0] * 0.995, q[1] * 1.005, color=OI["grey"], alpha=0.3, lw=0)
                ax.axvline(h0[r]["g_ext_median"], color=OI["orange"], lw=1.8)
                if h0[r]["g_tau_eq_D_heatmap"]:
                    ax.axvline(h0[r]["g_tau_eq_D_heatmap"], color=OI["green"], lw=1.4, ls="-.")
            ax.axvline(G_LIN, color="k", lw=0.8, alpha=0.5)
            ax.axvline(GON[float(r)], color=OI["grey"], lw=0.9, ls="--")
            ax.set_xscale("log"); ax.set_yscale("log")
            if obs == "rho":
                ax.set_xlim(0.14, 3.0)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v_, _: f"{v_:g}"))
            ax.set_title(f"r = {r}")
            ax.set_xlabel("g")
            if k == 0:
                ax.set_ylabel(ylab)
        if row_i == 0:
            hl = [Line2D([], [], color=OI["verm"], lw=1.7, label="up-sweep (from silence)"),
                  Line2D([], [], color=OI["blue"], lw=1.7, label="down-sweep (from all-active at g = 30)")]
            hl += [Line2D([], [], color="k", ls=ls[D], lw=1.5, label=f"dwell D = {D}") for D in (50, 200, 800)]
            hl += [Line2D([], [], color=OI["orange"], lw=1.8, label="h = 0 down-sweep (D = 200): median extinction g (grey: IQR)"),
                   Line2D([], [], color=OI["green"], lw=1.4, ls="-.", label="lifetime map: tau = 200 crossing"),
                   Line2D([], [], color=OI["grey"], lw=0.9, ls="--", label="simulated onset (tau = 3000)"),
                   Line2D([], [], color="k", lw=0.8, alpha=0.5, label="g_lin = 0.1632")]
            sf[0].legend(handles=hl, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
            sf[0].subplots_adjust(bottom=0.27, top=0.85, wspace=0.3)
            sf[0].suptitle("(a) Adiabatic g sweeps at h = 1e-5, zoomed on g <= 3 (above, the branches coincide; panel b and the areas in c cover the "
                           "whole sweep 0.15-30); 32 replicas per branch", fontsize=11, weight="bold", x=0.02, ha="left")
        else:
            sf[1].subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)
            sf[1].suptitle("(b) Participation ratio of the activity on the same sweeps (whole g range; colours and line styles as in a)",
                           fontsize=11, weight="bold", x=0.02, ha="left")
    # row 3: loop areas vs D
    axs = sf[2].subplots(1, 4)
    cmap = plt.get_cmap("cividis")
    for k, (win, h, ttl) in enumerate([("area_all", "1e-05", "whole sweep, h = 1e-5"), ("area_onset_window", "1e-05", "onset window g <= 2.5 g_on, h = 1e-5"),
                                       ("area_all", "0.0001", "whole sweep, h = 1e-4"), ("area_onset_window", "0.0001", "onset window, h = 1e-4")]):
        ax = axs[k]
        for ri, r in enumerate(rs):
            Ds, a, se, fl = [], [], [], []
            for D in (50, 200, 800):
                key = f"r{r}_h{h}_D{D}"
                if key in runs:
                    w = runs[key][win]; Ds.append(D); a.append(w["A_rho"]); se.append(w["A_rho_se"]); fl.append(w["A_rho_noise_floor"])
            if not Ds:
                continue
            c = cmap(ri / (len(rs) - 1))
            ax.errorbar(np.array(Ds) * (1 + 0.03 * (ri - 2.5)), a, yerr=se, color=c, marker="o", ms=5, lw=1.5, capsize=2, label=f"r = {r}")
            ax.plot(Ds, fl, color=c, lw=0.9, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("dwell D per g point (model time)")
        ax.set_xticks([50, 200, 800]); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v_, _: f"{v_:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("loop area\nint |ln rho_up - ln rho_down| d ln g" if k == 0 else "")
        ax.set_title(ttl, fontsize=10)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
        if k == 2:
            ax.text(0.03, 0.03, "dotted: noise floor (area between random\nhalves of one branch, scaled to 32 vs 32)", transform=ax.transAxes, fontsize=8)
    sf[2].subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)
    sf[2].suptitle("(c) Loop area against dwell: a first-order loop would keep its area as D grows; lag loops shrink toward the noise floor "
                   "(bars: replica-bootstrap SE)", fontsize=11, weight="bold", x=0.02, ha="left")
    # row 4: r-sweeps
    axs = sf[3].subplots(1, 3)
    for k, g in enumerate(["0.6", "2", "8"]):
        ax = axs[k]
        if g not in C:
            ax.set_visible(False); continue
        v = C[g]; R = np.array(v["r"])
        for gname, lsx in (("fwd", "-"), ("rev", "--")):
            ax.plot(R, v[gname]["rho_r_up"], color=OI["verm"], ls=lsx, lw=1.7, label=f"r increasing ({'0 -> 6 -> 0' if gname == 'fwd' else '6 -> 0 -> 6'} run)")
            ax.plot(R, v[gname]["rho_r_down"], color=OI["blue"], ls=lsx, lw=1.7, label=f"r decreasing ({'0 -> 6 -> 0' if gname == 'fwd' else '6 -> 0 -> 6'} run)")
        ax.set_yscale("log"); ax.set_xlabel("inhibition scale r"); ax.set_title(f"g = {g}  (D = 200, h = 1e-5)")
        a = v["fwd"]["area"]; b = v["rev"]["area"]
        ax.text(0.98, 0.97, f"loop area int |d ln rho| dr:\n0->6->0 run {a['A_rho']:.3f} +- {a['A_rho_se']:.3f} (floor {a['A_rho_noise_floor']:.3f})\n"
                            f"6->0->6 run {b['A_rho']:.3f} +- {b['A_rho_se']:.3f} (floor {b['A_rho_noise_floor']:.3f})",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
        if k == 0:
            ax.set_ylabel("mean fraction active rho"); ax.legend(frameon=False, fontsize=8, loc="lower left")
    sf[3].subplots_adjust(top=0.85, bottom=0.15, wspace=0.25)
    sf[3].suptitle("(d) Sweeps in r at fixed g (de Candia's lines are crossed by changing w1, i.e. r); 32 replicas per run, state carried over",
                   fontsize=11, weight="bold", x=0.02, ha="left")
    fig.savefig(f"{FIG}/multistab_hysteresis.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote multistab_hysteresis.png")


# ------------------------------------------------------------------------------------------------------------------- M3
def fig_m3():
    A = J("m3")["cells"]
    part = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz", allow_pickle=True)
    names = [str(x) for x in part["names"]]
    rs = ["0", "1", "2", "4"]; gs = ["0.4", "0.7", "1.2", "2.5", "6", "15"]
    fig, axs = plt.subplots(4, 6, figsize=(22, 16), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.92, bottom=0.07, left=0.06, right=0.88, hspace=0.32, wspace=0.22)
    norm = LogNorm(1e-4, 1)
    im = None
    for i, r in enumerate(rs):
        for j, g in enumerate(gs):
            ax = axs[i, j]; key = f"r{r}_g{g}"
            if key not in A:
                ax.set_visible(False); continue
            c = A[key]; starts = list(c["starts"])
            M = np.full((len(starts), len(names)), np.nan)
            for si, s in enumerate(starts):
                mv = c["starts"][s]["mean_vec"]
                if mv is not None:
                    M[si] = np.maximum(np.array(mv), 1e-5)
            cm = plt.get_cmap("cividis").copy(); cm.set_bad("#e8e8e8")
            im = ax.imshow(M, aspect="auto", cmap=cm, norm=norm, interpolation="nearest")
            for si, s in enumerate(starts):
                ax.text(len(names) - 0.4, si, f"{c['starts'][s]['alive_T']:.2f}", fontsize=7.5, va="center", ha="left")
            nd = sum(1 for s in starts if c['starts'][s]['alive_T'] == 0)
            if nd == len(starts):
                ax.set_title(f"r = {r}, g = {g}\nall 416 replicas extinct before T = 2000", fontsize=9)
            else:
                ax.set_title(f"r = {r}, g = {g}\n{'MULTISTABLE' if c['multistable'] else 'one non-silent attractor'}; "
                             f"max relL1 {c['max_rel_l1_between_start_means']:.2f}", fontsize=9)
            if j == 0:
                ax.set_yticks(range(len(starts))); ax.set_yticklabels(starts, fontsize=7)
            if i == 3:
                tk = [names.index(n) for n in ("AL-L", "AL-R", "isl1", "isl5", "isl10", "isl20", "isl30", "rest")]
                ax.set_xticks(tk); ax.set_xticklabels([names[t] for t in tk], rotation=90, fontsize=8.5)
    if im is not None:
        cb = fig.colorbar(im, cax=fig.add_axes([0.91, 0.3, 0.012, 0.4]))
        cb.set_label("late fraction active per module (t in [1000, 2000]; survivors; grey: none survived)")
    fig.suptitle("M3: late module activity per start (rows) and P0 module (columns: AL-L, AL-R, AL-mid, islands 1-30, rest); "
                 "right margin: fraction of the 32 replicas alive at T = 2000 (h = 0)", fontsize=11, y=0.97)
    fig.savefig(f"{FIG}/multistab_ic_attractors.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote multistab_ic_attractors.png")


# ------------------------------------------------------------------------------------------------------------------ M3b
def fig_m3b():
    A = J("m3b")["cells"]
    rs = sorted(set(v["r"] for v in A.values()))
    fig, axs = plt.subplots(len(rs), 1, figsize=(17, 3.3 * len(rs)), squeeze=False)
    rng = np.random.default_rng(0)
    for i, r in enumerate(rs):
        ax = axs[i, 0]
        cells = sorted([v for v in A.values() if v["r"] == r], key=lambda v: v["g"])
        xt, xl = [], []
        x = 0
        for v in cells:
            dead = all(a == 0 for a in v["alive"].values())
            for pl, res in v["pairs"].items():
                if dead:
                    ax.plot(x + 0.17, 0.5, marker="x", color=OI["grey"], ms=8, mew=1.5, ls="none")
                for kind, dx in (("mt3", 0.0), ("fp", 0.35)):
                    q = res[kind]
                    for lab, col, sh in (("a", OI["blue"], -0.08), ("b", OI["verm"], 0.08)):
                        zz = np.array(q[f"z_{lab}"], float)
                        if len(zz):
                            ax.plot(x + dx + sh + rng.uniform(-0.03, 0.03, len(zz)), np.clip(zz, -0.6, 1.6), ls="none", ms=3.5, color=col,
                                    alpha=0.6, marker="o" if kind == "mt3" else "^")
                xt.append(x + 0.17); xl.append(f"g = {v['g']:.3g}: {pl}")
                x += 1
        ax.axhline(0, color=OI["blue"], lw=0.8, ls="--"); ax.axhline(1, color=OI["verm"], lw=0.8, ls="--")
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8, rotation=35, ha="right")
        ax.set_xlim(-0.5, max(x, 4) - 0.3)
        ax.set_ylim(-0.6, 1.6); ax.set_ylabel("late projection z\n(0 = QMF state a, 1 = b)")
        nd = sum(1 for v in cells if all(a == 0 for a in v["alive"].values()))
        ax.set_title(f"r = {r:g}: {len(cells)} QMF multistable cells; coexistence in the stochastic model at "
                     f"{sum(v['coexistence'] for v in cells)}; every replica extinct (x) at {nd}", fontsize=10)
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], color=OI["blue"], marker="o", ls="none", label="started at QMF state a: MT3 module start (circles, 64 replicas)"),
          Line2D([], [], color=OI["blue"], marker="^", ls="none", label="started at a: Bernoulli sample of the QMF fixed point (triangles, 32)"),
          Line2D([], [], color=OI["verm"], marker="o", ls="none", label="started at QMF state b (same two kinds)"),
          Line2D([], [], color=OI["grey"], marker="x", ls="none", label="no replica alive at T = 5000")]
    axs[0, 0].legend(handles=hl, frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("M3b: do the QMF's coexisting fixed points survive demographic noise? Late (t in [3000, 5000]) projection of each surviving replica "
                 "onto the difference of the two QMF states, z = (x - rho_a).d/|d|^2", fontsize=10.5, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIG}/multistab_qmf_cells.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote multistab_qmf_cells.png")


def fig_m2b():
    A = J("m2b")
    fs = sorted(glob.glob(f"{D_}/m2b_*.npz")); fs = [f for f in fs if ".tmp" not in f]
    n = len(fs); nc = 4; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(17, 4.2 * nr), squeeze=False)
    for k in range(n, nr * nc):
        axs.flat[k].set_visible(False)
    for k, f in enumerate(fs):
        ax = axs.flat[k]
        z = np.load(f, allow_pickle=True); m = json.loads(str(z["meta"])); B = m["B"]
        tot = z["modc"].sum(1).astype(float); t = np.arange(tot.shape[0]) * float(z["rec_dt"])
        # 10-model-time block means for readability
        nb = len(t) // 10; tb = t[1:nb * 10 + 1].reshape(nb, 10).mean(1)
        for lab, cols, c in (("from up-branch end-state", slice(0, B), OI["verm"]), ("from down-branch end-state", slice(B, 2 * B), OI["blue"])):
            x = tot[1:nb * 10 + 1, cols].reshape(nb, 10, -1).mean(1)
            mu = x.mean(1); se = x.std(1) / np.sqrt(x.shape[1])
            ax.fill_between(tb, mu - 2 * se, mu + 2 * se, color=c, alpha=0.25, lw=0)
            ax.plot(tb, mu, color=c, lw=1.6, label=lab)
        ax.set_xscale("log")
        a = A[os.path.basename(f)]
        kind = "r-sweep (M2c) loop centre, D = 200" if m.get("mode") == "m2b_rsweep" else f"g-sweep (M2a) loop centre, D = {m['D']:g}"
        ax.set_title(f"r = {m['r']:g}, h = {m['h']:g}, g = {m['g']:.3g}\n({kind})\nlate KS p: rho {a['ks_p_rho']:.2g}, PR {a['ks_p_PR']:.2g}", fontsize=9)
        ax.set_xlabel("model time")
        if k % nc == 0:
            ax.set_ylabel("mean number of active neurons\n(10-time blocks; band: +- 2 SE)")
        if k == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("M2b: 64 replicas from each branch's end-state at the loop centre, run for T = 5000 at fixed g, r, h\n"
                 "(coexisting states would keep the two curves apart; late window for the KS tests: t in [2500, 5000])", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{FIG}/multistab_m2b.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote multistab_m2b.png")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["m1", "m2", "m3", "m3b", "m2b"]
    os.makedirs(FIG, exist_ok=True)
    for p in parts:
        {"m1": fig_m1, "m2": fig_m2, "m3": fig_m3, "m3b": fig_m3b, "m2b": fig_m2b}[p]()
