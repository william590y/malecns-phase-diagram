"""pd3 module_theory MT5 analysis + figure: lifetimes of single P0 modules simulated alone (CPU).
Input: results/pd3/module_theory/island_<m>_r<r>.npz (pd3_island_sim.py). Lifetime = censored-exponential MLE
tau = sum(min(t_ext, T)) / #deaths; with 0 deaths the 95% lower bound 2 E / chi2_{0.95}(2) = E / 2.996 is reported.
Crossings x(tau = 200), x(tau = 3000) by linear interpolation of ln tau in ln x (first crossing).
Outputs: results/pd3/module_theory/island_lifetimes.json, figs/pd3/modules_island_lifetimes.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/MALECNS_ROOT"
MODS = ["AL", "isl1", "isl2", "isl3", "isl4", "isl5"]
COL = {"AL": "#000000", "isl1": "#0072B2", "isl2": "#E69F00", "isl3": "#009E73", "isl4": "#CC79A7", "isl5": "#D55E00"}
out = {}


def cross(x, lt, thr):
    i = np.where(lt >= np.log(thr))[0]
    if not len(i):
        return None
    i = i[0]
    if i == 0:
        return float(x[0])
    a, b = np.log(x[i - 1]), np.log(x[i])
    return float(np.exp(a + (np.log(thr) - lt[i - 1]) / (lt[i] - lt[i - 1]) * (b - a)))


fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=130, sharey=True, gridspec_kw={"wspace": 0.08})
for j, r in enumerate([0, 1]):
    ax = axs[j]
    for m in MODS:
        d = np.load(f"{ROOT}/results/pd3/module_theory/island_{m}_r{r}.npz")
        e = d["ext_step"]; T = float(d["T"]); dt = float(d["dt"]); steps = int(round(T / dt))
        tt = np.where(e >= 0, e, steps) * dt; nd = (e >= 0).sum(1); Ex = tt.sum(1)
        tau = np.where(nd > 0, Ex / np.maximum(nd, 1), Ex / 2.996); cens = nd == 0
        x = d["x"]; lt = np.log(tau)
        out[f"{m}_r{r}"] = {"n": int(d["n"]), "Lambda": float(d["Lambda"]), "g_lin": float(d["g_lin"]), "x": x.tolist(),
                            "tau": tau.tolist(), "censored": cens.tolist(), "deaths": nd.tolist(),
                            "tau_x1": float(tau[0]), "x_tau200": cross(x, lt, 200), "x_tau3000": cross(x, lt, 3000)}
        o = out[f"{m}_r{r}"]
        o["g_tau3000"] = o["x_tau3000"] * o["g_lin"] if o["x_tau3000"] else None
        o["g_tau200"] = o["x_tau200"] * o["g_lin"] if o["x_tau200"] else None
        lab = f"{m} (n={int(d['n'])}, g_lin={float(d['g_lin']):.3g})"
        ax.plot(x[~cens], tau[~cens], "o-", ms=3.5, lw=1.4, color=COL[m], label=lab)
        if cens.any():
            ax.plot(x[cens], np.full(cens.sum(), 1.3e5), marker="^", ls="", ms=5, color=COL[m], mfc="none")
    ax.axhline(200, color="0.4", lw=0.8, ls="--"); ax.axhline(3000, color="0.4", lw=0.8, ls="--")
    ax.text(1.02, 230, "tau = 200", fontsize=7.5, color="0.3"); ax.text(1.02, 3400, "tau = 3000 (= T)", fontsize=7.5, color="0.3")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3, 3e5); ax.set_xlim(0.97, 4.1)
    ax.set_xticks([1, 1.5, 2, 3, 4]); ax.set_xticklabels(["1", "1.5", "2", "3", "4"])
    ax.set_xlabel("x = g / g_lin,k  (each module's own branching threshold)")
    ax.set_title(f"r = {r}: module simulated ALONE (internal synapses, B = 64, T = 3000)", fontsize=9.5)
    if j == 0:
        ax.set_ylabel("mean lifetime from fully active (model time, censored MLE)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
axs[1].text(0.99, 0.03, "open triangles: no replica died by T (lower bound off-scale)", transform=axs[1].transAxes, fontsize=7,
            ha="right")
fig.suptitle("MT5: is any spectral island 'a second AL'? Lifetimes of P0 modules alone vs their own threshold", fontsize=11, y=1.0)
fig.savefig(f"{ROOT}/figs/pd3/modules_island_lifetimes.png", bbox_inches="tight")
json.dump(out, open(f"{ROOT}/results/pd3/module_theory/island_lifetimes.json", "w"), indent=1)
for k, v in out.items():
    print(k, "n", v["n"], "g_lin %.4f" % v["g_lin"], "tau(x=1) %.3g" % v["tau_x1"], "x200", v["x_tau200"] and round(v["x_tau200"], 3),
          "x3000", v["x_tau3000"] and round(v["x_tau3000"], 3), "g3000", v["g_tau3000"] and round(v["g_tau3000"], 3))
