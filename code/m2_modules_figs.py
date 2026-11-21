"""m2_modules_figs.py -- figures of lane m2/modules (figs/m2/modules_*.png). Reads only the lane's JSON outputs + the
decay / driven / analysis2 boundary files.  Usage: python m2_modules_figs.py [phase] [lr] [ms] [iso]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


def plainlog(ax, axis="x", ticks=(1, 1.5, 2, 3, 4, 6, 8)):
    """log axis with plain-number tick labels at the given values"""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks)); a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/m2/modules"; FIG = f"{ROOT}/figs/m2"
os.makedirs(FIG, exist_ok=True)
C_M2, C_M1, C_M2S, C_T30 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd7"
G_LIN = 0.16315
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 130})
R_ALL = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6]


def lines_a():
    b = json.load(open(f"{ROOT}/results/m2/decay/boundaries.json"))["rows"]
    a2 = json.load(open(f"{ROOT}/results/analysis2/phase_complete.json"))["sim"]["real"]
    m2 = {r: (b[f"M2_real_r{r:g}"]["on"]["g"], b[f"M2_real_r{r:g}"]["D"]["g"]) for r in R_ALL if f"M2_real_r{r:g}" in b}
    m1 = {float(k): (v["on"]["g"], v["D"]["g"]) for k, v in a2.items()}
    return {"M2": m2, "M1": m1}


def lines_b():
    c = json.load(open(f"{ROOT}/results/m2/driven/curves.json"))
    out = {"M2": {}, "M1": {}}
    for k, v in c.items():
        s, model, graph, orn, r, palp = k.split("|")
        if graph == "real" and orn == "0.04" and palp == "False" and model in out and s in ("A1", "A2"):
            out[model][float(r)] = v["lines"].get("DT_SS")
    return out


def row_edges(vals):
    v = np.asarray(vals, float)
    mid = (v[1:] + v[:-1]) / 2
    return np.r_[v[0] - (mid[0] - v[0]) if len(v) > 1 else v[0] - 0.1, mid, v[-1] + (v[-1] - mid[-1]) if len(v) > 1 else v[-1] + 0.1]


def fig_phase():
    A = json.load(open(f"{OUT}/phase_A.json")); R = A["rows"]
    la, lb = lines_a(), lines_b()
    panels = [("a", "M1_pd3", "(a) h = 0, fully active start: M1"), ("a", "M2_lane", "(a) h = 0, fully active start: M2"),
              ("b", "M1", "(b) ORN drive 0.04 (window 500-1500): M1"), ("b", "M2", "(b) ORN drive 0.04: M2")]
    cmap = plt.get_cmap("Blues")
    cols = [cmap(x) for x in np.linspace(0.18, 1.0, 34)]
    cm = ListedColormap(["#ecebe6"] + cols); norm = BoundaryNorm(np.arange(-0.5, 35.5, 1), cm.N)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, (st, model, title) in zip(axs.ravel(), panels):
        rs = sorted({float(k.split("|")[2]) for k in R if k.startswith(f"{st}|{model}|")})
        redges = row_edges(rs)
        ax.add_patch(plt.Rectangle((0.05, -1), 30, 8, facecolor="white", edgecolor=GRID, hatch="////", lw=0, zorder=0))
        for i, r in enumerate(rs):
            row = R[f"{st}|{model}|{r:g}"]; g = np.array(row["g"])
            n = np.array(row["parts"]["P0"]["theta"]["0.01"]["n_on"])
            ge = np.exp(row_edges(np.log(g)))
            ax.pcolormesh(ge, [redges[i], redges[i + 1]], n[None, :], cmap=cm, norm=norm, shading="flat", rasterized=True)
        key = "M2" if "M2" in model else "M1"
        if st == "a":
            L = la[key]; rr = sorted(L)
            ax.plot([L[r][0] for r in rr], rr, color=INK, lw=1.4, label="onset (tau = 3000)")
            rd = [r for r in rr if L[r][1]]
            ax.plot([L[r][1] for r in rd], rd, color=INK, lw=1.4, ls=":", label="D (PR = N/4)")
        else:
            L = lb[key]; rr = sorted(r for r in L if L[r])
            ax.plot([L[r] for r in rr], rr, color=INK, lw=1.4, label="DT->SS (driven lane)")
        shr = [(R[f"{st}|{model}|{r:g}"]["parts"]["P0"]["theta"]["0.01"]["first_SH_g"], r) for r in rs]
        if any(g_ for g_, _ in shr):
            ax.plot([x[0] if x[0] else np.nan for x in shr], [x[1] for x in shr], color="#eda100", lw=1.6, ls="-.", label="SH: all 34 modules >= 1%")
        ax.axvline(G_LIN, color=INK2, lw=1, ls="--", label="g_lin = 0.163")
        ax.set_xscale("log"); ax.set_xlim(0.05, 30); ax.set_ylim(redges[0], 6.15)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.legend(loc="upper left", fontsize=7.5)
    for ax in axs[1]:
        ax.set_xlabel("coupling g")
    for ax in axs[:, 0]:
        ax.set_ylabel("inhibition scale r")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=axs, shrink=0.8, ticks=[0, 5, 10, 15, 20, 25, 30, 34])
    cb.set_label("P0 modules with occupancy >= 1% (of 34); grey = SL (not sustained / none on); hatched = not simulated")
    fig.suptitle("Module phase map: how many of the 34 pd3 modules are on (1% occupancy; (a) among survivors)", fontsize=11, color=INK)
    fig.savefig(f"{FIG}/modules_phase_map.png"); plt.close(fig)


def fig_activation():
    A = json.load(open(f"{OUT}/phase_A.json")); R = A["rows"]
    la = lines_a()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    ax = axs[0]
    for model, key, col in (("M1_pd3", "M1", C_M1), ("M2_lane", "M2", C_M2), ("M2s_lane", "M2", C_M2S)):
        for th, ls in (("0.01", "-"), ("0.001", "--")):
            rs, ns = [], []
            for r in R_ALL:
                k = f"a|{model}|{r:g}"
                if k not in R:
                    continue
                n = np.array(R[k]["parts"]["P0"]["theta"][th]["n_on"])
                if (n > 0).any():
                    rs.append(r); ns.append(n[np.argmax(n > 0)])
            ax.plot(rs, ns, color=col, ls=ls, marker="o", ms=3.5, lw=1.6,
                    label=f"{model.split('_')[0]}, theta {float(th) * 100:g}%")
    ax.set_xlabel("r"); ax.set_ylabel("modules on at the first sustained g (of 34)")
    ax.set_title("A. Modules on at the onset (setting a)", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16)); ax.set_ylim(0, 35)
    ax = axs[1]
    for model, key, col, dx in (("M1_pd3", "M1", C_M1, -0.04), ("M2_lane", "M2", C_M2, 0.04)):
        for r in R_ALL:
            k = f"a|{model}|{r:g}"
            if k not in R:
                continue
            on = la[key].get(float(r), (None,))[0]
            cr = R[k]["parts"]["P0"]["theta"]["0.01"]["crossing"]
            v = np.array([x for x in cr.values() if x is not None]) / on
            if not len(v):
                continue
            ax.plot([r + dx] * 2, [v.min(), v.max()], color=col, lw=0.8, alpha=0.6)
            ax.plot([r + dx] * 2, np.percentile(v, [25, 75]), color=col, lw=3.5, solid_capstyle="butt")
            ax.plot(r + dx, np.median(v), "o", color="white", mec=col, ms=4)
    ax.plot([], [], color=C_M1, lw=3.5, label="M1 (IQR, line = range, dot = median)"); ax.plot([], [], color=C_M2, lw=3.5, label="M2")
    ax.set_yscale("log"); ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("r"); ax.set_ylabel("module's 1% crossing / onset g")
    ax.set_title("B. Where each module switches on, relative to the onset", loc="left", fontsize=10); ax.legend(fontsize=7.5)
    ax = axs[2]
    mods = [("AL-L", "-"), ("isl1", "--"), ("rest", ":")]
    for model, key, col in (("M1_pd3", "M1", C_M1), ("M2_lane", "M2", C_M2)):
        for mname, ls in mods:
            rs, gs = [], []
            for r in R_ALL:
                k = f"a|{model}|{r:g}"
                if k in R and R[k]["parts"]["P0"]["theta"]["0.01"]["crossing"].get(mname):
                    rs.append(r); gs.append(R[k]["parts"]["P0"]["theta"]["0.01"]["crossing"][mname])
            ax.plot(rs, gs, color=col, ls=ls, lw=1.6, label=f"{key} {mname}")
        rr = sorted(la[key]); ax.plot(rr, [la[key][r][0] for r in rr], color=col, lw=0.8, alpha=0.5)
    ax.set_yscale("log"); ax.set_xlabel("r"); ax.set_ylabel("g at which the module reaches 1% (survivors)")
    ax.set_title("C. Activation g: AL-L, island 1, rest (thin = onset)", loc="left", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(f"{FIG}/modules_activation.png"); plt.close(fig)


ISO_STYLE = {"AL": ("#2a78d6", "-"), "AL_L": ("#2a78d6", "--"), "AL_R": ("#2a78d6", ":"), "isl1": ("#eb6834", "-"),
             "isl3": ("#1baf7a", "-"), "isl5": ("#eda100", "-"), "CB": ("#e87ba4", "-"), "VNC": ("#008300", "-"),
             "OL": ("#6250d6", "-")}


def fig_iso():
    J = json.load(open(f"{OUT}/iso.json"))
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, sharey=True, constrained_layout=True)
    CAP = 2e5
    for i, model in enumerate(("M2", "M1")):
        for j, r in enumerate((0, 1, 4)):
            ax = axs[i, j]
            fb = None
            for mod, (col, ls) in ISO_STYLE.items():
                k = f"{model}_r{r}_{mod}"
                if k not in J:
                    continue
                v = J[k]; g = np.array(v["g"]); tau = np.array([x if x is not None else np.inf for x in v["tau"]], float)
                fin = np.isfinite(tau)
                ax.plot(g[fin], tau[fin], color=col, ls=ls, lw=1.6, label=f"{mod} ({v['N']:,})")
                if (~fin).any():
                    ax.plot(g[~fin], np.full((~fin).sum(), CAP * 1.25 ** list(ISO_STYLE).index(mod)), color=col, ls="none",
                            marker="^", ms=3.5)
                fb = v["full_brain_on"]
            if fb:
                ax.axvline(fb, color=INK, lw=1.2, label="full-brain onset")
                ax.text(fb * 1.06, 7, f"full brain {fb:.3g}", fontsize=7.5, color=INK, rotation=90, va="bottom")
            ax.axvline(G_LIN, color=INK2, lw=0.8, ls="--")
            for y in (200, 3000):
                ax.axhline(y, color=GRID, lw=0.8, zorder=0)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.05, 30); ax.set_ylim(5, 2e6)
            ax.set_title(f"{model}, r = {r}: modules alone", loc="left", fontsize=10)
            if i == 1:
                ax.set_xlabel("coupling g")
            if j == 0:
                ax.set_ylabel("lifetime tau (MLE; triangles = no deaths by T 3000)")
            if j == 2:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Which modules can stay on alone? Isolated subgraphs, h = 0, fully active start (grey lines: tau = 200, 3000; dashed: g_lin)",
                 fontsize=10.5)
    fig.savefig(f"{FIG}/modules_iso.png"); plt.close(fig)



MODEL_COL = {"M1": C_M1, "M2": C_M2, "M2s": C_M2S, "M2_tau30": C_T30}


def ref_g(setting, model, r):
    import m2_modules_args as MA
    return MA.onset(model, r) if setting == "a" else MA.dtss(model, r)


def fig_lr():
    sys.path.insert(0, f"{ROOT}/code")
    J = json.load(open(f"{OUT}/lr.json" if os.path.exists(f"{OUT}/lr.json") else f"{OUT}/lr_a.json"))["cells"]
    for setting, fname, xl in (("a", "modules_lr.png", "g / onset (tau = 3000) of the row"),
                               ("b", "modules_lr_driven.png", "g / DT->SS of the row")):
        fig, axs = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True, constrained_layout=True)
        for i, (pair, sL) in enumerate((("AL", "AL_L"), ("HEM", "hemiL"))):
            for key, v in J.items():
                st, model, r = key.split("|")
                if st != setting:
                    continue
                r = float(r); ref = ref_g(setting, model, r); col = MODEL_COL.get(model, INK2)
                x = np.array([rw["g"] / ref for rw in v["rows"]])
                ts, tq, la, lsd, cr = [], [], [], [], []
                for rw in v["rows"]:
                    s_ = rw[pair]["starts"].get(sL, {}); t_ = s_.get("t_sym") or {}
                    ts.append(t_.get("median", np.nan) if t_.get("median") is not None else np.nan)
                    tq.append((t_.get("q25") or np.nan, t_.get("q75") or np.nan))
                    la.append(s_.get("late_A_mean") if s_.get("late_A_mean") is not None else np.nan)
                    lsd.append(s_.get("late_A_sd") if s_.get("late_A_sd") is not None else np.nan)
                    a_ = rw[pair]["starts"].get("all", {})
                    cr.append(a_.get("corr_LR_median") if a_.get("corr_LR_median") is not None else np.nan)
                ts, tq, la, lsd, cr = map(lambda z: np.array(z, float), (ts, tq, la, lsd, cr))
                if sL in v["starts"]:
                    axs[i, 0].plot(x, ts, color=col, lw=1, marker="o", ms=3, alpha=0.85)
                    axs[i, 0].vlines(x, tq[:, 0], tq[:, 1], color=col, lw=0.8, alpha=0.6)
                    axs[i, 1].errorbar(x, la, yerr=lsd, color=col, lw=1, marker="o", ms=3, alpha=0.85, capsize=0)
                if "all" in v["starts"]:
                    axs[i, 2].plot(x, cr, color=col, lw=1, marker="o", ms=3, alpha=0.85)
            axs[i, 0].set_yscale("log"); axs[i, 0].set_ylabel(f"{pair}: t_sym after a {sL} start (median, IQR)")
            axs[i, 1].axhline(0.6, color=INK2, ls="--", lw=0.8); axs[i, 1].axhline(0, color=GRID, lw=0.8)
            axs[i, 1].set_ylim(-0.3, 1.0); axs[i, 1].set_ylabel(f"{pair}: late mean A, {sL} start (mean, SD)")
            axs[i, 2].axhline(-0.2, color=INK2, ls="--", lw=0.8); axs[i, 2].axhline(0, color=GRID, lw=0.8)
            axs[i, 2].set_ylim(-1, 1); axs[i, 2].set_ylabel(f"{pair}: corr_t(f_L, f_R), all start (median)")
        for ax in axs[1]:
            ax.set_xlabel(xl); ax.set_xscale("log")
        for ax in axs.ravel():
            plainlog(ax, "x", (1, 1.5, 2, 3, 4, 6, 8) if setting == "a" else (0.5, 0.8, 1, 1.25, 1.6, 2.5, 4))
        for ax in axs[:, 0]:
            plainlog(ax, "y", (0.5, 0.8, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40))
        present = {k.split("|")[1] for k in J if k.startswith(setting + "|")}
        for m, c in MODEL_COL.items():
            if m in present:
                axs[0, 2].plot([], [], color=c, marker="o", ms=3, label=m)
        axs[0, 2].legend(fontsize=8, loc="lower right")
        axs[0, 1].text(0.98, 0.62, "c1 threshold 0.6", transform=axs[0, 1].get_yaxis_transform(), ha="right", fontsize=7, color=INK2)
        axs[0, 2].text(0.98, -0.27, "c4 threshold -0.2", transform=axs[0, 2].get_yaxis_transform(), ha="right", fontsize=7, color=INK2)
        ttl = "h = 0" if setting == "a" else "ORN drive 0.04 (more ORNs on the right: 1,344 R vs 884 L)"
        rr = ("r in R_CORE for M2; 0, 1, 4 for M1, M2s, M2_tau30" if setting == "a" else "r = 0, 1, 4 for M2; 0, 1 for M1")
        fig.suptitle(f"Left/right symmetry, {ttl}. One point per (r, g); {rr}. A = (f_L - f_R)/(f_L + f_R)", fontsize=10)
        fig.savefig(f"{FIG}/{fname}"); plt.close(fig)


def heat(ax, M, rows, cols, title, vmax=1.0, fmt="{:.2f}", cmap="Blues"):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 0.6 * vmax else INK)
    ax.set_title(title, loc="left", fontsize=9.5)
    return im


def fig_ms():
    sys.path.insert(0, f"{ROOT}/code")
    fn = f"{OUT}/multistab.json" if os.path.exists(f"{OUT}/multistab.json") else f"{OUT}/multistab_a.json"
    J = json.load(open(fn))["cells"]
    fig = plt.figure(figsize=(14, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    for key, v in J.items():
        st, model, r = key.split("|")
        if st != "a":
            continue
        ref = ref_g("a", model, float(r)); x = [rw["g"] / ref for rw in v["rows"]]
        ax.plot(x, [rw["P0"]["max_rel_l1_start_means"] for rw in v["rows"]], color=MODEL_COL[model], marker="o", ms=3, lw=1)
    ax.axhline(0.2, color=INK2, ls="--", lw=0.8); ax.set_xscale("log")
    ax.set_xlabel("g / onset"); ax.set_ylabel("largest relative L1 between start means (P0)")
    plainlog(ax, "x", (1, 1.5, 2, 3, 4, 6, 8))
    ax.set_title("A. h = 0: do the starts end in different states?", loc="left", fontsize=9.5)
    for m in ("M1", "M2", "M2s", "M2_tau30"):
        ax.plot([], [], color=MODEL_COL[m], marker="o", ms=3, label=m + (" (AL_L, AL_R starts only)" if m in ("M2s", "M2_tau30") else ""))
    ax.legend(fontsize=7.5)
    for k_, (gpos, qty, ttl) in enumerate(((gs[0, 1], "alive_T", "B. M2, r = 1, h = 0: fraction alive at T = 2000"),
                                           (gs[0, 2], "confinement_median", "C. M2, r = 1, h = 0: activity share inside start set"))):
        ax = fig.add_subplot(gpos)
        v = J.get("a|M2|1")
        if v:
            starts = [s for s in ("all", "AL_L", "AL_R", "AL", "isl1", "isl3", "isl5", "hemiL", "hemiR", "CB", "VNC", "OL", "nonAL",
                                  "rand10", "rand1") if s in v["starts"]]
            M = np.array([[rw["starts"].get(s, {}).get(qty) if rw["starts"].get(s, {}).get(qty) is not None else np.nan
                           for rw in v["rows"]] for s in starts], float)
            ref = ref_g("a", "M2", 1.0)
            heat(ax, M, starts, [f"{rw['g'] / ref:.3g}" for rw in v["rows"]], ttl)
            ax.set_xlabel("g / onset")
    ax = fig.add_subplot(gs[1, 0])
    for key, v in J.items():
        st, model, r = key.split("|")
        if st != "b":
            continue
        ref = ref_g("b", model, float(r)); x = [rw["g"] / ref for rw in v["rows"]]
        ax.plot(x, [rw["P0"]["max_rel_l1_start_means"] for rw in v["rows"]], color=MODEL_COL[model], marker="o", ms=3, lw=1)
    ax.axhline(0.2, color=INK2, ls="--", lw=0.8); ax.set_xscale("log")
    ax.set_xlabel("g / DT->SS"); ax.set_ylabel("largest relative L1 between start means (P0)")
    plainlog(ax, "x", (0.5, 0.8, 1, 1.25, 1.6, 2.5, 4))
    ax.set_title("D. ORN drive: do the starts end in different states?", loc="left", fontsize=9.5)
    ax = fig.add_subplot(gs[1, 1])
    for key, v in J.items():
        st, model, r = key.split("|")
        if st != "b":
            continue
        ref = ref_g("b", model, float(r))
        rows = [rw for rw in v["rows"] if "quiet_vs_all" in rw]
        x = np.array([rw["g"] / ref for rw in rows]); y = np.array([rw["quiet_vs_all"]["ratio"] for rw in rows])
        p = np.array([rw["quiet_vs_all"]["ks_p"] for rw in rows])
        ax.plot(x, y, color=MODEL_COL[model], lw=1, alpha=0.8)
        ax.plot(x[p < 1e-3], y[p < 1e-3], "o", color=MODEL_COL[model], ms=4)
        ax.plot(x[p >= 1e-3], y[p >= 1e-3], "o", mfc="white", mec=MODEL_COL[model], ms=4)
    for yy in (1.5, 1 / 1.5):
        ax.axhline(yy, color=INK2, ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("g / DT->SS")
    ax.set_ylabel("late nonORN activity: median(all start) / median(quiet start)")
    plainlog(ax, "x", (0.5, 0.8, 1, 1.25, 1.6, 2.5, 4)); plainlog(ax, "y", (0.67, 0.8, 0.9, 1, 1.1, 1.25, 1.5))
    for m in ("M1", "M2"):
        ax.plot([], [], color=MODEL_COL[m], marker="o", ms=4, label=m)
    ax.legend(fontsize=8)
    ax.set_title("E. ORN drive: quiet vs fully active start (filled: KS p < 0.001)", loc="left", fontsize=9.5)
    ax = fig.add_subplot(gs[1, 2])
    v = J.get("b|M2|1")
    if v:
        starts = [s for s in ("quiet", "all", "AL_L", "AL_R", "hemiL", "hemiR", "nonAL", "CB", "VNC", "OL", "isl1", "rand10") if s in v["starts"]]
        ref = ref_g("b", "M2", 1.0)
        x = np.array([rw["g"] / ref for rw in v["rows"]])
        sty = {"quiet": (INK, "-", 2.2), "all": (C_M2, "-", 2.2)}
        for s in starts:
            c, ls, lw = sty.get(s, (GRID if s not in ("AL_L", "hemiL") else INK2, "-", 1.0))
            ax.plot(x, [rw["starts"].get(s, {}).get("rho_late_mean", np.nan) for rw in v["rows"]], color=c, ls=ls, lw=lw,
                    label=s if s in sty else None)
        ax.plot([], [], color=GRID, label="other 10 starts")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("g / DT->SS")
        ax.set_ylabel("late mean active fraction (P0 total)"); ax.legend(fontsize=8)
        ax.set_title("F. M2, r = 1, ORN drive: late activity, all 12 starts overlap", loc="left", fontsize=9.5)
        plainlog(ax, "x", (0.5, 0.8, 1, 1.25, 1.6, 2.5, 4))
    fig.suptitle("Multistability across initial conditions (pd3 M3 rule: clusters of late 34-module vectors, relative L1 cut 0.2)", fontsize=10.5)
    fig.savefig(f"{FIG}/modules_multistab.png"); plt.close(fig)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["phase"]
    if "phase" in parts:
        fig_phase(); fig_activation()
    if "iso" in parts:
        fig_iso()
    if "lr" in parts:
        fig_lr()
    if "ms" in parts:
        fig_ms()
