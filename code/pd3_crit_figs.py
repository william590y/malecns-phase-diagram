"""pd3 criticality figures (CPU). Reads aval_analysis.json, qs_analysis.json, the aval_*.npz records (distributions, samples).
Okabe-Ito colours (validated subset #0072B2 #D55E00 #009E73 #CC79A7 #E69F00 #56B4E9 + black); every series also differs by
marker or line style and is labelled (CVD-safe secondary encoding). Writes figs/pd3/aval_*.png, qs_*.png, field_scaling.png,
lifetime_vs_K_criticality.png.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
import pd3_aval_analyze as AA
R_ = f"{ROOT}/results/pd3/criticality"; FIG = f"{ROOT}/figs/pd3"
DT = AA.DT; A_EFF = AA.A_EFF
OI = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "purple": "#CC79A7", "orange": "#E69F00", "sky": "#56B4E9", "black": "#000000"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 8.5, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 1.8, "figure.dpi": 110})
A = json.load(open(f"{R_}/aval_analysis.json"))
Q = json.load(open(f"{R_}/qs_analysis.json")) if os.path.exists(f"{R_}/qs_analysis.json") else {}
RUNS = A["runs"]
STY = {  # tag -> (label, colour, marker, linestyle)
    "real_r0_x1.00": ("real brain r=0", OI["blue"], "o", "-"),
    "real_r1_x1.00": ("real brain r=1", OI["verm"], "s", "-"),
    "real_r6_x1.00": ("real brain r=6", OI["purple"], "^", "-"),
    "module_r0_x1.00": ("AL module alone r=0", OI["green"], "D", "--"),
    "module_r1_x1.00": ("AL module alone r=1", OI["orange"], "v", "--"),
    "cnull_N167216_s0_r0_x1.00": ("config. null (full) r=0", OI["black"], "x", ":"),
    "cnull_N167216_s0_r1_x1.00": ("config. null (full) r=1", OI["sky"], "+", ":"),
    "complete37_x1.00": ("complete graph K=37", "#666666", None, "-."),
    "complete76_x1.00": ("complete graph K=76", "#999999", None, "-."),
}
_cache = {}
from matplotlib.ticker import FuncFormatter, NullFormatter, LogLocator


def plain_log_x(a, ticks=None):
    if ticks is not None:
        a.set_xticks(ticks)
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); a.xaxis.set_minor_formatter(NullFormatter())


def run(tag):
    if tag not in _cache:
        _cache[tag] = AA.load(tag)
    return _cache[tag]


def save(fig, name):
    p = f"{FIG}/{name}"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved", p)


# ------------------------------------------------------------------------------------------- 1. sizes and durations
def fig_size_duration():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    tags = ["real_r0_x1.00", "real_r1_x1.00", "module_r0_x1.00", "cnull_N167216_s0_r0_x1.00", "complete76_x1.00"]
    for t in tags:
        r = run(t)
        if r is None:
            continue
        lab, c, mk, ls = STY[t]
        x, p, _ = AA.size_density(r); ax[0].plot(x, x ** 1.5 * p, ls=ls, marker=mk, ms=4, color=c, label=lab, markevery=2)
        x, p, _ = AA.dur_density(r); ax[1].plot(x, x ** 2 * p, ls=ls, marker=mk, ms=4, color=c, label=lab, markevery=2)
        tt = np.geomspace(0.2, 300, 120); Pe, _ = AA.survival_emp(r, tt)
        ax[2].plot(tt, (1 + A_EFF * tt) * Pe, ls=ls, color=c, label=lab)
    # chain, R = 1
    for K, c in [(37.0, "#555555"), (75.8, "#aaaaaa")]:
        P = AA.chain_size_dist(K, 1.0, 4000); S = np.arange(1, len(P) + 1)
        ax[0].plot(1.94 * S, S ** 1.5 * P / 1.94 * 1.94 ** 1.5 / 1.94 ** 0, color=c, lw=1.2, ls=(0, (1, 1)), label=f"chain K={K:g} (S x 1.94)")
        tt = np.geomspace(0.05, 300, 300); Sv, _ = AA.chain_surv_mean(K, 1.0, tt)
        dens = -np.gradient(Sv, tt); ax[1].plot(tt, tt ** 2 * dens, color=c, lw=1.2, ls=(0, (1, 1)), label=f"chain K={K:g}")
        ax[2].plot(tt, (1 + A_EFF * tt) * Sv, color=c, lw=1.2, ls=(0, (1, 1)), label=f"chain K={K:g}")
    ax[0].set(xscale="log", yscale="log", xlabel="avalanche size S (spikes)", ylabel=r"$S^{3/2}\,P(S)$",
              title="(a) sizes at g = g_lin, compensated by MF-DP $S^{-3/2}$")
    ax[1].set(xscale="log", yscale="log", xlabel="duration T (model time, 1/alpha)", ylabel=r"$T^{2}\,P(T)$",
              title="(b) durations, compensated by MF-DP $T^{-2}$")
    ax[2].set(xscale="log", yscale="log", xlabel="t (model time)", ylabel=r"$(1+\alpha_{eff} t)\,P(T>t)$",
              title="(c) survival, compensated (flat = critical branching)")
    rr = run("real_r0_x1.00")
    if rr is not None:
        wv = AA.seed_weights(rr); tt = np.geomspace(0.2, 300, 120); Pe, _ = AA.survival_emp(rr, tt, wv)
        ax[2].plot(tt, (1 + A_EFF * tt) * Pe, color=OI["sky"], ls="-", lw=2.2, label="real brain r=0, seeds weighted by v (post hoc)")
        x, p, _ = AA.dur_density(rr, wv); ax[1].plot(x, x ** 2 * p, color=OI["sky"], ls="-", lw=2.2, label="real r=0, seeds ~ v (post hoc)")
    ax[0].set_ylim(1e-4, 3); ax[1].set_ylim(1e-3, 3); ax[2].set_ylim(1e-3, 3)
    for a in ax:
        a.legend(frameon=False, loc="lower left")
    fig.suptitle("Single-seed avalanches at x = g/g_lin = 1 (seed: one of the 366 excitatory AL-module neurons; nulls: leading-mode support)", y=1.02)
    save(fig, "aval_size_duration.png")


# --------------------------------------------------------------------------------------------------- 2. <S>(T)
def fig_ST():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for t in ["real_r0_x1.00", "real_r1_x1.00", "module_r0_x1.00", "cnull_N167216_s0_r0_x1.00", "complete76_x1.00", "complete37_x1.00"]:
        if t not in RUNS:
            continue
        lab, c, mk, ls = STY[t]; st = RUNS[t]["ST"]
        rows = np.array(st["rows"]); sl = np.array(st["slope"])
        ax[0].plot(rows[:, 0], rows[:, 1], ls=ls, marker=mk, ms=4, color=c, label=lab)
        if len(sl):
            ax[1].plot(sl[:, 0], sl[:, 1], ls=ls, marker=mk, ms=4, color=c, label=lab)
    tt = np.geomspace(0.2, 300, 10)
    ax[0].plot(tt, 3 * tt ** 2, color="#888888", lw=1, ls=":", label=r"$\propto T^2$ (MF-DP)")
    ax[1].axhline(2, color="#888888", lw=1, ls=":"); ax[1].text(0.25, 2.03, "MF-DP asymptote 2", fontsize=8, color="#555555")
    ax[0].set(xscale="log", yscale="log", xlabel="duration T (model time)", ylabel=r"$\langle S\rangle(T)$ (spikes)", title=r"(a) mean size vs duration")
    ax[1].set(xscale="log", xlabel="duration T (model time)", ylabel=r"local slope d ln<S>/d ln T", title="(b) local slope (3-bin regression)")
    ax[1].set_ylim(0.8, 3.2)
    # (c) peak slope vs K
    pts = []
    for t, row in RUNS.items():
        if abs(row["x"] - 1) > 1e-6 or row["ST"].get("peak") is None:
            continue
        g = row["graph"]
        if g == "complete":
            K = row["K_ref"]; grp = "complete graph (one-mode control)"
        elif g == "cnull":
            K = row.get("K_nc_glin"); grp = "configuration nulls (K_nc)"
        elif g == "module":
            K = {0.0: 70.0, 1.0: 36.5}.get(row["r"]); grp = "AL module alone (K_fit realsub)"
        else:
            K = row["K_ref"]; grp = "real brain (K_needed)"
        if K:
            pts.append((grp, K, row["ST"]["peak"], row["ST"].get("peak_ci"), t))
    styles = {"complete graph (one-mode control)": ("#777777", "o"), "configuration nulls (K_nc)": (OI["black"], "x"),
              "AL module alone (K_fit realsub)": (OI["green"], "D"), "real brain (K_needed)": (OI["verm"], "s")}
    for grp, (c, mk) in styles.items():
        P = [p for p in pts if p[0] == grp]
        if P:
            Ks = np.array([p[1] for p in P]); pk = np.array([p[2] for p in P])
            ci = np.array([p[3] if p[3] else [np.nan, np.nan] for p in P])
            ax[2].errorbar(Ks, pk, yerr=[pk - ci[:, 0], ci[:, 1] - pk], fmt=mk, color=c, ms=7, capsize=2, label=grp, lw=1)
    cq = A["chain_ST_peaks_quoted"]
    ax[2].plot([37, 75.8, 300, 2000], [cq["37"], cq["75.8"], cq["300"], cq["2000"]], color="#888888", ls="--", marker="*", ms=9,
               label="exact chain prediction (plan)")
    ax[2].axhline(2, color="#888888", lw=1, ls=":")
    for t, row in RUNS.items():
        if row.get("vw") and abs(row["x"] - 1) < 1e-6 and row["vw"]["ST"].get("peak") and row["graph"] == "real":
            ax[2].plot([row["K_ref"]], [row["vw"]["ST"]["peak"]], "s", mfc="none", color=OI["verm"], ms=8,
                       label="real brain, seeds ~ v (post hoc)" if t == "real_r0_x1.00" else None)
    ax[2].set(xscale="log", xlabel="effective module size K", ylabel="peak local slope of <S>(T)", title="(c) peak slope vs K (x = 1)")
    ax[0].legend(frameon=False, fontsize=8); ax[1].legend(frameon=False, fontsize=8)
    ax[2].legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.42))
    save(fig, "aval_ST_slope.png")


# --------------------------------------------------------------------------------------------- 3. critical point
def fig_critical():
    cp = A["critical_point"]; fam = cp["families"]; cc = cp["chain_curves"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))
    for key, lab, c, mk in [("real_r0", "real brain r=0", OI["blue"], "o"), ("real_r1", "real brain r=1", OI["verm"], "s"),
                            ("real_r6", "real brain r=6", OI["purple"], "^"), ("complete76", "complete graph K=76", "#888888", "D"),
                            ("complete37", "complete graph K=37", "#444444", "v")]:
        if key not in fam:
            continue
        f = fam[key]; se = [s if s is not None else 0 for s in f["gamma_se"]]
        ax[0].errorbar(f["x"], f["gamma"], yerr=1.96 * np.array(se), fmt=mk + "-", color=c, ms=5, capsize=2, label=lab, lw=1.3)
        if f["x_star"]:
            ax[0].plot([f["x_star"]], [0], marker="|", ms=18, color=c, mew=2)
    for key, lab, c, mk in [("real_r0_vw", "real r=0, seeds ~ v (post hoc)", OI["sky"], "o"), ("real_r1_vw", "real r=1, seeds ~ v (post hoc)", OI["orange"], "s"),
                            ("real_r6_vw", "real r=6, seeds ~ v (post hoc)", OI["purple"], "^")]:
        if key in fam:
            f = fam[key]; ax[0].plot(f["x"], f["gamma"], mk + "--", color=c, ms=5, mfc="none", lw=1.2, label=lab)
            if f["x_star"]:
                ax[0].plot([f["x_star"]], [0], marker="|", ms=18, color=c, mew=2)
    for key, lab, c in [("K75.8_real", "chain K=75.8 (r=0 K_needed)", OI["blue"]), ("K36.6_real", "chain K=36.6 (r=1)", OI["verm"]),
                        ("K16.2_real", "chain K=16.2 (r=6)", OI["purple"])]:
        ax[0].plot(cc[key]["x"], cc[key]["gamma"], color=c, ls=":", lw=1.4, label=lab)
    for key, lab, c in [("K16.2_real_netinput_closure_r6", "net-input closure, r=6 (K=16.2)", OI["purple"]),
                        ("K36.6_real_netinput_closure_r1", "net-input closure, r=1 (K=36.6)", OI["verm"])]:
        if key in cc:
            ax[0].plot(cc[key]["x"], cc[key]["gamma"], color=c, ls=(0, (6, 2, 1, 2)), lw=1.0, label=lab)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].axvline(1, color="k", lw=1.6)
    ax[0].text(1.005, -0.29, "g_lin", fontsize=8)
    ax[0].set(xlabel="x = g / g_lin", ylabel=r"growth rate $\gamma$ of ln E[n(age)] (1/model time)",
              title="(a) growth rate over ages 5-50 steps; | = zero x*")
    ax[0].set_xlim(0.88, 2.35)
    # (b) gamma at x = 1 vs r
    rd = cp["r_dependence_x1"]
    rr = [d["r"] for d in rd]
    ax[1].errorbar(rr, [d["gamma"] for d in rd], yerr=[1.96 * (d["gamma_se"] or 0) for d in rd], fmt="o-", color=OI["verm"], capsize=3,
                   label="simulation (real brain, x = 1)")
    if all("gamma_vw" in d for d in rd):
        ax[1].errorbar(rr, [d["gamma_vw"] for d in rd], yerr=[1.96 * (d["gamma_vw_se"] or 0) for d in rd], fmt="o--", mfc="none",
                       color=OI["orange"], capsize=3, label="simulation, seeds ~ v (post hoc)")
    ax[1].plot(rr, [d["chain_gamma_exact_linearisation"] for d in rd], "s--", color="#555555",
               label="exact linearisation (R = 1) + chain saturation at K_needed(r)")
    ax[1].plot(rr, [d["chain_gamma_closure"] for d in rd], "^:", color=OI["purple"], label="net-input closure: R = mu(r)/mu(0), same K")
    ax[1].set(xlabel="inhibition scale r", ylabel=r"$\gamma$ at x = 1 (1/model time)", title="(b) is the critical point r-independent?")
    # (c) E[n(age)] at x = 1 vs r
    cols = {0: OI["blue"], 0.5: OI["sky"], 1: OI["verm"], 2: OI["orange"], 4: OI["green"], 6: OI["purple"]}
    lss = {0: "-", 0.5: "--", 1: "-", 2: "-.", 4: "--", 6: ":"}
    for r in [0, 0.5, 1, 2, 4, 6]:
        t = f"real_r{r:g}_x1.00"; ru = run(t)
        if ru is None:
            continue
        a = np.arange(0, 401); En = ru["agg_n"][a] / ru["den"][a]
        ok = ru["agg_alive"][a] >= 30                                    # drop ages with < 30 avalanches alive
        ax[2].plot(a[ok] * DT, En[ok], color=cols[r], ls=lss[r], label=f"r = {r:g}")
    for K, c in [(75.8, "#555555"), (16.2, "#aaaaaa")]:
        tt = np.arange(0, 401) * DT; _, En = AA.chain_surv_mean(K, 1.0, tt)
        ax[2].plot(tt, En, color=c, lw=1.2, ls=(0, (1, 1)), label=f"chain R=1, K={K:g}")
    ax[2].set(yscale="log", xlabel="age (model time)", ylabel="E[n_active(age)] (extinct = 0)", title="(c) mean activity after one seed, x = 1")
    ax[0].legend(frameon=False, fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax[1].legend(frameon=False, fontsize=7.8); ax[2].legend(frameon=False, fontsize=7.8)
    ax[2].set_title("(c) mean activity after one seed, x = 1 (ages with >= 30 alive)")
    save(fig, "aval_critical_point.png")


# ------------------------------------------------------------------------------------------------------- 4. shapes
def fig_shapes():
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))
    r = run("real_r0_x1.00")
    if r is not None:
        cols = [OI["blue"], OI["verm"], OI["green"], OI["purple"], OI["orange"]]
        for j, (Tx, c) in enumerate(zip(r["tset"][:5], cols)):
            n = r["ex_cnt"][j]
            if n < 30:
                continue
            prof = r["ex_sum"][j, :Tx].sum(1) / n
            u = (np.arange(Tx) + 0.5) / Tx
            ax[0].plot(u, prof, color=c, label=f"T = {Tx*DT:g} ({Tx} steps; {int(n)} avalanches)")
        ax[0].set(xlabel="t / T", ylabel="mean spikes per step", title="(a) mean profiles at fixed duration (real r=0, x=1)")
        # (b) collapse of the log-bin profiles: s(t/T) / T^(gamma_ST - 1), gamma_ST = 2 (MF-DP)
        e = r["bin_edges"]; nb = len(e) - 1
        cm = plt.get_cmap("viridis")
        sel = [k for k in range(nb) if r["prof_cnt"][k] >= 300 and e[k] >= 25]
        u40 = (np.arange(40) + 0.5) / 40
        for i, k in enumerate(sel):
            Tm = r["prof_T"][k] / r["prof_cnt"][k] * DT
            prof = r["prof_sum"][k, :, 0] / r["prof_cnt"][k]; prof = prof / prof.mean()
            ax[1].plot(u40[1:], prof[1:], color=cm(i / max(len(sel) - 1, 1)), lw=1.2, label=f"T = {Tm:.1f}" if i % 2 == 0 else None)
        ax[1].set(xlabel="t / T", ylabel="profile / its mean over the avalanche",
                  title="(b) normalised shapes, real r=0, x=1, T >= 5 (colour = T)")
    for t in ["real_r0_x1.00", "real_r1_x1.00", "real_r6_x1.00", "module_r0_x1.00", "cnull_N167216_s0_r0_x1.00", "complete76_x1.00"]:
        if t not in RUNS:
            continue
        lab, c, mk, ls = STY[t]; sk = RUNS[t]["skew"]
        T = np.array([q["T_mean"] for q in sk]); m = np.array([q["skew_sp"] for q in sk]); se = np.array([q["skew_sp_se"] for q in sk])
        ok = np.array([q["n"] for q in sk]) >= 100
        ax[2].errorbar(T[ok], m[ok], yerr=1.96 * se[ok], fmt=(mk or "o") + ls, color=c, ms=4, capsize=1.5, lw=1.2, label=lab)
    vw = RUNS.get("real_r0_x1.00", {}).get("vw", {}).get("skew", [])
    if vw:
        T = np.array([q["T_mean"] for q in vw]); m = np.array([q["skew_sp"] for q in vw]); se = np.array([q["skew_sp_se"] for q in vw])
        ok = T >= 1
        ax[2].errorbar(T[ok], m[ok], yerr=1.96 * se[ok], fmt="o--", mfc="none", color=OI["sky"], ms=4, capsize=1.5, lw=1.2,
                       label="real r=0, seeds ~ v (post hoc)")
    ax[2].axhspan(-0.02, 0.02, color="#dddddd", label="continuous one-mode chain: 0 +- 0.02")
    ax[2].plot([40], [0.4], marker="*", ms=14, color=OI["orange"], ls="none", label="de Candia 2026, SL-B1 line (~0.4 at T = 40/alpha)")
    ax[2].axhline(0, color="k", lw=0.7)
    ax[2].set(xscale="log", xlabel="duration T (model time)", ylabel="mean skewness (+ = fast rise, slow decay)",
              title="(c) shape skewness vs duration, x = 1 (bars: 95% CI)")
    ax[2].set_ylim(-0.15, 0.48)
    ax[0].legend(frameon=False, fontsize=7.6); ax[1].legend(frameon=False, fontsize=7.6, loc="upper right")
    ax[2].legend(frameon=False, fontsize=7.3, loc="upper left", ncol=1)
    save(fig, "aval_shapes.png")


# ------------------------------------------------------------------------------------------- 5. example avalanches
def fig_examples():
    tags = ["real_r0_x1.00", "real_r1_x1.00", "real_r6_x1.00", "real_r0_x1.20"]
    fig, ax = plt.subplots(len(tags), 3, figsize=(15, 2.5 * len(tags)), sharex=False)
    for i, t in enumerate(tags):
        r = run(t)
        if r is None:
            continue
        meta = r["samp_meta"]; off = r["samp_off"]
        idx = [j for j, m in enumerate(meta) if m["status"] == 0 and m["T"] >= 100]
        idx = sorted(idx, key=lambda j: meta[j]["T"])
        pick = [idx[int(q * (len(idx) - 1))] for q in (0.5, 0.9, 1.0)] if len(idx) >= 3 else idx
        for k, j in enumerate(pick):
            x = r["samp_data"][off[j]:off[j + 1]].astype(float)
            ker = np.ones(5) / 5 / DT                                       # spikes per model time, 1-model-time window
            x = np.stack([np.convolve(x[:, q], ker, mode="same") for q in range(3)], 1); tt = np.arange(len(x)) * DT
            a = ax[i, k]
            a.plot(tt, x[:, 0], color=OI["blue"], lw=1.1, label="AL left")
            a.plot(tt, x[:, 1], color=OI["verm"], lw=1.1, ls="--", label="AL right")
            a.plot(tt, x[:, 2], color=OI["green"], lw=1.1, ls=":", label="rest of brain")
            a.set_title(f"{t}: T = {len(x)*DT:.0f}, S = {meta[j]['S']}, seed {'L' if meta[j]['seed_chan']==0 else 'R'}", fontsize=8.5)
            if k == 0:
                a.set_ylabel("spikes per model time")
            if i == len(tags) - 1:
                a.set_xlabel("time since seed (model time)")
            if i == 0 and k == 0:
                a.legend(frameon=False, fontsize=8)
    fig.suptitle("Sampled long avalanches (reservoir sample of extinct avalanches with T >= 20 model time; median / 90th pct / longest): "
                 "spike rate by region, 1-model-time moving average", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, "aval_examples.png")


def fig_recruitment():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    cols = {0: OI["blue"], 1: OI["verm"], 2: OI["orange"], 4: OI["green"], 6: OI["purple"]}
    lss = {0: "-", 1: "--", 2: "-.", 4: ":", 6: "-"}
    for r in [0, 1, 2, 4, 6]:
        t = f"real_r{r:g}_x1.00"
        if t not in RUNS or not RUNS[t].get("recruit_50_250"):
            continue
        rc = RUNS[t]["recruit_50_250"]; u = rc["t_over_T"]
        ax[0].plot(u, rc["share_minority"], color=cols[r], ls=lss[r], label=f"r={r} (n={int(rc['n_al'])})")
        ax[1].plot(u, rc["share_seedlobe"], color=cols[r], ls=lss[r], label=f"r={r}")
        ax[2].plot(u, rc["share_other"], color=cols[r], ls=lss[r], label=f"r={r}")
    ax[0].set(xlabel="t / T", ylabel="minority-lobe share of AL spikes", title="(a) minority lobe (by total spikes), T = 10-50")
    ax[1].set(xlabel="t / T", ylabel="seed-lobe share of AL spikes", title="(b) lobe of the seed")
    ax[2].set(xlabel="t / T", ylabel="share of spikes outside the AL", title="(c) recruitment of the rest of the brain")
    for a in ax:
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("Recruitment inside long avalanches at x = 1 (T = 10-50 model time); de Candia's competition mechanism predicts a late fall of the minority share growing with r", y=1.02, fontsize=9.5)
    save(fig, "aval_recruitment.png")


# ---------------------------------------------------------------------------------------------------- 6. QS / field
def fig_qs():
    if "C2" not in Q:
        return
    c2 = Q["C2"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    sty = {"real_r0": ("real r=0", OI["blue"], "o"), "real_r1": ("real r=1", OI["verm"], "s"), "real_r6": ("real r=6", OI["purple"], "^"),
           "module_r0": ("AL module alone r=0", OI["green"], "D"), "module_r1": ("AL module alone r=1", OI["orange"], "v")}
    for k, (lab, c, mk) in sty.items():
        if k not in c2:
            continue
        rows = c2[k]["rows"]; x = np.array([q["x"] for q in rows])
        n = np.array([q["n_AL"] for q in rows]); se = np.array([q["n_AL_se"] for q in rows]); ch = np.array([q["chain_nQS_neurons"] for q in rows])
        ls = "--" if k.startswith("module") else "-"
        ax[0].errorbar(x, n, yerr=1.96 * se, fmt=mk + ls, color=c, ms=4, capsize=1.5, label=lab, lw=1.2)
        if k.startswith("real"):
            ax[0].plot(x, ch, color=c, ls=":", lw=1.3, label=f"chain K={c2[k]['K']:g} (x1.94), no fit")
            Rm1 = np.array([q["R"] for q in rows]) - 1; ok = Rm1 > 0
            ax[1].plot(Rm1[ok], n[ok], mk + ls, color=c, ms=4, label=f"{lab}: beta {c2[k]['beta']:.2f}" if c2[k]["beta"] else lab)
            ax[1].plot(Rm1[ok], ch[ok], ":", color=c, lw=1.3, label=f"chain K={c2[k]['K']:g}: {c2[k]['beta_chain_same_window']:.2f}" if c2[k]["beta_chain_same_window"] else None)
        chi = np.array([q["chi_AL"] for q in rows]); g = x * c2[k]["g_lin"]
        ax[2].plot(g, chi, mk + ls, color=c, ms=4, label=lab, lw=1.2)
        if k.startswith("real"):
            ax[2].plot(g, [q["chain_chi_neurons"] for q in rows], ":", color=c, lw=1.3)
            ax[2].axvline(c2[k]["chain_chi_peak"]["g"], color=c, lw=0.8, ls=(0, (2, 4)))
    rr = np.geomspace(0.02, 3, 10); ax[1].plot(rr, 30 * rr, color="#888", lw=1, ls="-.", label="slope 1 (MF-DP beta = 1)")
    ax[1].axvspan(2 / np.sqrt(75.8), 8 / np.sqrt(75.8), color=OI["blue"], alpha=0.08, lw=0, label="fit window r=0: (R-1) sqrt(K) in [2, 8]")
    ax[0].set(xscale="log", yscale="log", xlabel="x = g / g_lin", ylabel="QS mean active AL neurons <n_AL>", title="(a) quasi-stationary order parameter")
    ax[1].set(xscale="log", yscale="log", xlabel="R - 1 (branching distance)", ylabel="<n_AL>", title="(b) order-parameter exponent")
    ax[2].set(xscale="log", xlabel="g", ylabel=r"$\chi_{AL}$ = var(n_AL)/<n_AL>", title="(c) QS susceptibility (dotted: chain; vertical: chain peak g)")
    plain_log_x(ax[0], [1, 1.5, 2, 3]); plain_log_x(ax[2], [0.15, 0.2, 0.3, 0.4, 0.6])
    for a in ax:
        a.legend(frameon=False, fontsize=7.5)
    save(fig, "qs_order_parameter.png")
    # susceptibility figure: chi_AL, chi_tot, histograms
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for k, (lab, c, mk) in sty.items():
        if k not in c2:
            continue
        rows = c2[k]["rows"]; x = np.array([q["x"] for q in rows]); g = x * c2[k]["g_lin"]
        lo = np.array([q["chi_AL_ci"][0] for q in rows]); hi = np.array([q["chi_AL_ci"][1] for q in rows]); m = np.array([q["chi_AL"] for q in rows])
        ax[0].fill_between(g, lo, hi, color=c, alpha=0.2, lw=0); ax[0].plot(g, m, mk + ("--" if k.startswith("module") else "-"), color=c, ms=4, label=lab)
        mt = np.array([q["chi_tot"] for q in rows]); ax[1].plot(g, mt, mk + "-", color=c, ms=4, label=lab)
    for k, gp in [("real_r0", 0.195), ("real_r1", 0.209)]:
        ax[0].axvline(gp, color=sty[k][1], ls=":", lw=1.2, label=f"predicted peak g {gp} ({sty[k][0]})")
    ax[0].set(xscale="log", xlabel="g", ylabel=r"$\chi_{AL}$ (95% replica bootstrap band)", title="(a) module susceptibility")
    ax[1].set(xscale="log", yscale="log", xlabel="g", ylabel=r"$\chi_{QS}$ = N var(rho)/<rho>", title="(b) whole-brain QS susceptibility")
    plain_log_x(ax[0], [0.15, 0.2, 0.3, 0.4, 0.6]); plain_log_x(ax[1], [0.15, 0.2, 0.3, 0.4, 0.6])
    if "real_r0" in c2:
        rows = c2["real_r0"]["rows"]; cm = plt.get_cmap("viridis")
        pick = [0, 3, 6, 9, 12, 15, 18, 21]
        for i, j in enumerate(pick):
            if j >= len(rows):
                continue
            h = np.array(rows[j]["hist_AL"]); h = h / h.sum()
            ax[2].plot(np.arange(len(h)), h, color=cm(i / (len(pick) - 1)), lw=1.2, label=f"x = {rows[j]['x']:.2f}")
        ax[2].set(xscale="symlog", yscale="log", xlabel="n_AL", ylabel="QS probability", title="(c) n_AL histograms, real r=0 (unimodal?)")
        ax[2].set_ylim(1e-6, 1)
    for a in ax:
        a.legend(frameon=False, fontsize=7.5)
    save(fig, "qs_susceptibility.png")


def fig_field():
    if "C3" not in Q or not Q["C3"]:
        return
    c3 = Q["C3"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cols = {"0.90": OI["blue"], "1.00": OI["verm"], "1.10": OI["green"]}
    for key, d in c3.items():
        r, x = key.split("_"); x = x[1:]
        ls = "-" if r == "r0" else "--"; mk = "o" if r == "r0" else "s"
        h = np.array(d["h"]); n = np.array(d["n_AL"]); se = np.array(d["n_AL_se"])
        ok = n > 0
        ax[0].errorbar(h[ok], n[ok], yerr=1.96 * se[ok], fmt=mk + ls, color=cols[x], ms=4, capsize=1.5, label=f"{r}, x={x}", lw=1.2)
        sl = np.array(d["slope"]); nm = np.array(d["n_mid"]); ok = np.isfinite(sl) & (nm > 0) & (np.array(d["slope_se"]) < 0.25)
        ax[1].errorbar(nm[ok], sl[ok], yerr=1.96 * np.array(d["slope_se"])[ok], fmt=mk + ls, color=cols[x], ms=4, capsize=1.5, label=f"{r}, x={x}", lw=1.2)
        ch = d["chain"]; cn = np.sqrt(np.array(ch["n"])[1:] * np.array(ch["n"])[:-1])
        ax[1].plot(cn, ch["slope"], color=cols[x], ls=":", lw=1.1)
    ax[1].axhline(0.5, color="#888", lw=1, ls="-."); ax[1].axhline(1, color="#888", lw=0.8)
    ax[1].text(4e-3, 0.45, "MF-DP critical: 1/2", fontsize=8, color="#555"); ax[1].text(4e-3, 1.03, "linear response: 1", fontsize=8, color="#555")
    ax[0].set(xscale="log", yscale="log", xlabel="drive h", ylabel="<n_AL> (active AL neurons)", title="(a) driven response of the AL module")
    ax[1].set(xscale="log", xlabel="<n_AL> (geometric mean of the two h)", ylabel="local slope d ln n_AL / d ln h (SE < 0.25 only)",
              title="(b) field exponent; dotted = exact driven chain at K_needed")
    ax[1].set_xlim(3e-3, 150)
    ax[1].set_ylim(-0.1, 1.4)
    for a in ax:
        a.legend(frameon=False, fontsize=8)
    save(fig, "field_scaling.png")


def fig_lifetime_K():
    if "C4" not in Q:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.7))
    KK = np.geomspace(8, 8000, 50)
    ax[0].plot(KK, 1.90 * KK ** 0.503 / A_EFF, color="#888", ls="--", label=r"prediction $1.90\,K^{0.503}/\alpha_{eff}$")
    for r, c, mk in [(0, OI["blue"], "o"), (1, OI["verm"], "s")]:
        sub = Q["C4"].get(f"r{r}", {}).get("rows", [])
        ax[0].plot([q["K_n"] for q in sub], [q["tau_x1"] for q in sub], mk, color=c, ms=6, mfc="none", label=f"nested AL sub-modules r={r} (K_n, unfitted)")
        ax[0].plot([q["K_fit"] for q in sub], [q["tau_x1"] for q in sub], mk, color=c, ms=6, label=f"nested AL sub-modules r={r} (K_fit)")
        nl = Q["C4_merged"][f"r{r}"]["nulls"]
        ax[0].plot([q["K_nc"] for q in nl], [q["tau_x1"] for q in nl], "x" if r == 0 else "+", color=c, ms=7, label=f"configuration nulls r={r} (K_nc)")
        ax[0].plot([q["K_fit"] for q in nl if q["K_fit"]], [q["tau_x1"] for q in nl if q["K_fit"]], "^" if r == 0 else "v", color=c, ms=5, mfc="none",
                   label=f"configuration nulls r={r} (K_fit)")
        for i, x in enumerate(["1.5", "2"]):
            Kn = np.array([q["K_n"] for q in sub]); tv = np.array([q["tau"][x] for q in sub]); dd = np.array([q["deaths"][x] for q in sub])
            ax[1].plot(Kn, tv, ("-" if x == "1.5" else "--"), color=c, lw=1.2, label=f"r={r}, x={x}")
            ax[1].plot(Kn[dd >= 20], tv[dd >= 20], mk, color=c, ms=5)
            ax[1].plot(Kn[dd < 20], tv[dd < 20], mk, color=c, ms=6, mfc="white")
    ax[0].set(xscale="log", yscale="log", xlabel="module size K", ylabel="mean lifetime from fully active at g_lin (model time)",
              title="(a) lifetime at the critical point vs K")
    ax[1].set(yscale="log", xlabel="K_n (nested AL sub-module)", ylabel="lifetime (model time)", title="(b) nested sub-modules above g_lin (hollow: < 20 of 4096 replicas died)")
    for a in ax:
        a.legend(frameon=False, fontsize=7.3)
    save(fig, "lifetime_vs_K_criticality.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["sd", "st", "crit", "shapes", "ex", "rec", "qs", "field", "lk"]
    F = {"sd": fig_size_duration, "st": fig_ST, "crit": fig_critical, "shapes": fig_shapes, "ex": fig_examples, "rec": fig_recruitment,
         "qs": fig_qs, "field": fig_field, "lk": fig_lifetime_K}
    for w in which:
        try:
            F[w]()
        except Exception as ex:
            import traceback; traceback.print_exc(); print("FAILED", w, ex)
