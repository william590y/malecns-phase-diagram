"""Figure for the pd3 theory audit (results/pd3/theory_notes.md): figs/pd3/theory_audit.png.
(a) de Candia's coarse-grained 2-module mean field for the left/right antennal-lobe pair in the simulation's (g, r) plane,
    with the exact linear threshold g_lin, the neuron-level module mean-field onset and the simulated tau = 3000 onset.
(b)-(e) finite-K MF-DP predictions of the one-mode birth-death chain at g_lin (R = 1); (f) configuration-null family.
Inputs: results/pd3/theory_module.json, theory_module2.json, theory_bd.json, theory_bd2.json,
results/analysis2/flow_sim_boundaries.json (simulated boundaries).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

R3 = "/MALECNS_ROOT/results/pd3"
FIG = "/MALECNS_ROOT/figs/pd3/theory_audit.png"
m1 = json.load(open(f"{R3}/theory_module.json")); m2 = json.load(open(f"{R3}/theory_module2.json"))
bd = json.load(open(f"{R3}/theory_bd.json")); bd2 = json.load(open(f"{R3}/theory_bd2.json"))
sim = json.load(open("/MALECNS_ROOT/results/analysis2/flow_sim_boundaries.json"))["rows"]
a_eff = bd["alpha_eff"]; g_lin = 0.16315274988719286
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = {37: "#7fb2ef", 54.7: "#3f88dc", 75.8: "#1d5aa6", 300: "#123f78", 2000: "#0a2448"}
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
fig, ax = plt.subplots(2, 3, figsize=(15, 10.2))

# (a) coarse 2-module phase diagram
a = ax[0, 0]; cm = m2["coarse_2module"]; G = np.array(cm["g"]); RS = np.array(cm["r"])
cat = np.zeros((len(RS), len(G)))
for i, r in enumerate(RS):
    for j, l in enumerate(cm["phase"][f"{r:g}"]["labels"]):
        s = set(l.split("+"))
        cat[i, j] = 0 if s == {"SL"} else 1 if s == {"SH"} else 3 if "SH" in s else 2
cols = ["#f2f1ee", "#cfe0f6", "#f8d2c0", "#c9ecdc"]
ge = np.sqrt(G[1:] * G[:-1]); ge = np.r_[G[0] ** 2 / ge[0], ge, G[-1] ** 2 / ge[-1]]
re_ = np.r_[RS[0] - 0.125, (RS[1:] + RS[:-1]) / 2, RS[-1] + 0.125]
a.pcolormesh(ge, re_, cat, cmap=ListedColormap(cols), vmin=-0.5, vmax=3.5, shading="flat")
hat = np.ma.masked_where(cat != 2, cat)
a.pcolor(ge, re_, hat, hatch="///", alpha=0.0, shading="flat")
edge = [cm["phase"][f"{r:g}"]["g_SL_edge"] for r in RS]
a.plot(edge, RS, color=ORANGE, lw=2, label="coarse MF SL edge (de Candia, 2nd order)")
a.axvline(g_lin, color=INK, lw=2, label="g_lin (exact linear, r-independent)")
q = m1["D_module_qmf"]; Gq = np.array(q["g"]); qon = []
for r in q["r"]:
    X = np.array(q["XL"][f"{r:g}_both"]) + np.array(q["XR"][f"{r:g}_both"]); k = np.argmax(X > 1e-6); qon.append(np.sqrt(Gq[k] * Gq[k - 1]))
a.plot(qon, q["r"], ls=":", marker="s", ms=4, color=INK2, lw=1.5, label="neuron-level MF onset (SH only)")
a.plot([x["on"] for x in sim], [x["r"] for x in sim], ls="--", marker="o", ms=3.5, color=BLUE, lw=1.6,
       label="simulated onset (tau = 3000)")
for rs in m1["A_AL_pair"]["r1_star_PW"].values():
    a.axhline(rs, color=INK2, lw=0.8, ls=(0, (1, 2)))
a.text(0.62, 2.42, "w1 < 0 above r1* = 2.0-2.3", fontsize=8, color=INK2, va="bottom", ha="center")
a.text(0.55, 1.6, "SH", fontsize=11, color=INK, ha="center"); a.text(0.172, 4.6, "SL", fontsize=11, color=INK)
a.text(0.47, 5.2, "B1L | B1R\nbistable", fontsize=9, color=INK, ha="center")
a.text(0.75, 4.1, "B1 + SH\ncoexist", fontsize=9, color=INK, ha="center")
a.set_xscale("log"); a.set_xlim(0.15, 1.0); a.set_ylim(-0.125, 6.125)
a.set_xticks([0.15, 0.2, 0.3, 0.5, 0.7, 1.0]); a.set_xticklabels(["0.15", "0.2", "0.3", "0.5", "0.7", "1"])
a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xlabel("coupling g"); a.set_ylabel("inhibition scale r")
a.set_title("(a) AL pair: de Candia closure vs neuron level")
handles = a.get_legend_handles_labels()[0] + [Patch(facecolor=cols[0], edgecolor=INK2, label="coarse: SL (silent)"),
    Patch(facecolor=cols[1], edgecolor=INK2, label="coarse: SH (both ALs)"),
    Patch(facecolor=cols[2], hatch="///", edgecolor=INK2, label="coarse: B1 (one AL), no SH"),
    Patch(facecolor=cols[3], edgecolor=INK2, label="coarse: B1 + SH coexist")]
a.legend(handles=handles, fontsize=6.8, loc="upper left", bbox_to_anchor=(-0.02, -0.14), ncol=2, framealpha=0.95)

# (b) compensated size distribution
b = ax[0, 1]; av = {r["K"]: r for r in bd["aval_R1"]["rows"]}
for K in [37, 54.7, 75.8]:
    P = np.array(av[K]["P_S"]); S = np.arange(1, len(P) + 1)
    b.plot(S, P * S ** 1.5, color=SEQ[K], lw=2, label=f"K = {K:g} (S_c = {av[K]['S_c']})")
    b.plot(av[K]["S_c"], np.interp(av[K]["S_c"], S, P * S ** 1.5), "o", color=SEQ[K], ms=5)
from scipy.special import gammaln
S = np.arange(1, 4001); Pinf = np.exp(gammaln(2 * (S - 1) + 1) - gammaln(S) - gammaln(S + 1) - (2 * S - 1) * np.log(2))
b.plot(S, Pinf * S ** 1.5, color=INK2, lw=1.2, ls="--", label="K = inf (critical branching)")
b.set_xscale("log"); b.set_yscale("log"); b.set_ylim(1e-3, 1.0); b.set_xlim(1, 4000)
b.set_xlabel("avalanche size S (chain units; x1.94 = neuron spikes)"); b.set_ylabel("S^1.5 P(S)")
b.set_title("(b) sizes at g_lin: S^-3/2 up to S_c = 3.4 K")
b.legend(fontsize=7.5, loc="lower left")

# (c) compensated survival (durations)
c = ax[0, 2]; tg = np.array(bd["aval_R1"]["tgrid"])
for K in [37, 54.7, 75.8]:
    Sv = np.array(av[K]["surv"]); c.plot(tg / a_eff, Sv * (1 + tg), color=SEQ[K], lw=2, label=f"K = {K:g} (T_c = {av[K]['T_c']/a_eff:.1f})")
c.axhline(1, color=INK2, ls="--", lw=1.2, label="K = inf: P(T > t) = 1/(1 + alpha t)")
c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(0.05, 200); c.set_ylim(1e-2, 2)
c.set_xlabel("duration t (model time)"); c.set_ylabel("(1 + alpha t) P(T > t)")
c.set_title("(c) durations at g_lin: T^-2 up to T_c")
c.legend(fontsize=7.5, loc="lower left")

# (d) local slope of <S>(T)
d = ax[1, 0]
series = [(37, bd["mc_R1"]["37"]["rows"]), (75.8, bd["mc_R1"]["75.8"]["rows"]), (300, bd2["mc"]["300"]["rows"]), (2000, bd2["mc"]["2000"]["rows"])]
for K, rows in series:
    T = np.array([r["mean_T"] for r in rows]); Sm = np.array([r["mean_S"] for r in rows])
    lt, ls = np.log(T), np.log(Sm); sl = np.gradient(ls, lt)
    sl = np.convolve(sl, np.ones(3) / 3, mode="same"); keep = slice(1, len(sl) - 1)
    d.plot(T[keep] / a_eff, sl[keep], color=SEQ[K], lw=2, label=f"K = {K:g}")
d.axhline(2, color=INK2, ls="--", lw=1.2, label="MF-DP: <S> ~ T^2")
d.set_xscale("log"); d.set_ylim(0, 2.3); d.set_xlim(0.3, 300)
d.set_xlabel("avalanche duration T (model time)"); d.set_ylabel("local slope d ln<S> / d ln T")
d.set_title("(d) <S>(T): exponent 2 needs K >> 100")
d.legend(fontsize=7.5, loc="lower right")

# (e) QS susceptibility
e = ax[1, 1]
for K in [37, 75.8, 300, 3000]:
    ch = bd2["chi"][f"{K:g}"]; col = SEQ.get(K, "#0a2448" if K == 3000 else "#123f78")
    e.plot(ch["x"], np.array(ch["chi"]) / np.sqrt(K), color=col, lw=2, label=f"K = {K:g} (peak R = {ch['R_peak']:.3f})")
e.set_xlabel("x = (R - 1) sqrt(K)"); e.set_ylabel("chi_QS / sqrt(K),  chi_QS = var(n)/<n>")
e.set_title("(e) quasi-stationary fluctuations peak at x = 1.7")
e.legend(fontsize=7.5, loc="lower center")

# (f) null family
f = ax[1, 2]; nf = bd["null_family"]
for lab, col, nm in [("r0_theory", BLUE, "r = 0, K/N from theory K_eff"), ("r1_needed", ORANGE, "r = 1, K/N from K needed")]:
    rows = [x for x in nf["rows"][lab] if x["on_over_glin"]]
    f.plot([x["N"] for x in rows], [x["on_over_glin"] for x in rows], marker="o", ms=4, lw=2, color=col, label="null family: " + nm)
real0 = sim[0]["on"] / g_lin; real1 = [x for x in sim if x["r"] == 1.0][0]["on"] / g_lin
f.axhline(real0, color=BLUE, ls="--", lw=1.2, label=f"real brain r = 0 (onset/g_lin = {real0:.2f})")
f.axhline(real1, color=ORANGE, ls="--", lw=1.2, label=f"real brain r = 1 ({real1:.2f})")
f.set_xscale("log"); f.set_yscale("log"); f.set_ylim(1.0, 6.5)
f.set_yticks([1, 1.5, 2, 3, 5]); f.set_yticklabels(["1", "1.5", "2", "3", "5"])
f.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
f.set_xlabel("null size N (neurons)"); f.set_ylabel("predicted onset / g_lin (tau = 3000)")
f.set_title("(f) null onset window shrinks with N")
f.legend(fontsize=7.2, loc="upper right")
for axx in ax.ravel():
    axx.grid(True, color=GRID, lw=0.6); axx.set_axisbelow(True)
    for s in ["top", "right"]:
        axx.spines[s].set_visible(False)
fig.suptitle("MaleCNS stochastic Wilson-Cowan: what de Candia's modular theory and finite-K MF-DP predict (theory, no new simulations)",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.05, right=0.985, top=0.92, bottom=0.06, wspace=0.27, hspace=0.62)
fig.savefig(FIG, dpi=150, facecolor="white")
print("saved", FIG)
