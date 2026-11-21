"""m2_stoch_figs.py -- figures of lane m2/theory-stochastic (theory only; no experiment data). figs/m2/stoch_*.png"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/MALECNS_ROOT/code")

ROOT = "/MALECNS_ROOT"
T = f"{ROOT}/results/m2/theory"
F = f"{ROOT}/figs/m2"
os.makedirs(F, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED = "#222222", "#888888"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "lines.linewidth": 2})


def J(p):
    return json.load(open(p))


def lt_series(prof_rows, rows_full, key):
    return None


def fig_lifetime():
    import m2_stoch_predict as P
    on = J(f"{T}/stochastic_onset.json")
    qs = J(f"{T}/stochastic_qs.json") if os.path.exists(f"{T}/stochastic_qs.json") else {}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (m, title) in zip(axs, (("M2", "Model 2 (r = 0, PF)"), ("M1", "Model 1 (validation, r = 0)"))):
        U, TAU = (0.25, 157.0) if m == "M2" else (0.0, np.inf)
        d = P.decay_summary(f"{ROOT}/results/m2/theory/stochastic/decay/bqmf_{m}.json", U, TAU, rlist=(0.0,))
        rows = d["_rows"]; g = np.array([r["g"] for r in rows])
        for j, (lab, getter) in enumerate((("single mode, 2-D", lambda r: r["_single"].get("PF_r0", {}).get("lt_2D")),
                                           ("single mode, frozen", lambda r: r["_single"].get("PF_r0", {}).get("lt_F")),
                                           ("single mode, adiabatic", lambda r: r["_single"].get("PF_r0", {}).get("lt_A")))):
            y = np.array([np.nan if getter(r) is None else getter(r) for r in rows], float)
            ax.plot(g, y / np.log(10), color=C[j], lw=2 if j == 0 else 1.2, ls="-" if j == 0 else "--", label=lab)
        if m == "M2":
            pp = d["_pooled"]
            y = np.array([np.nan if (p.get("PF_r0") is None or p["PF_r0"].get("lt_2D") is None) else p["PF_r0"]["lt_2D"] for p in pp], float)
            ax.plot(g, y / np.log(10), color=C[3], lw=2, label="pooled, 2-D")
            if "M2" in qs:
                q = [x for x in qs["M2"]["rows"] if np.isfinite(x["lt_LNA_gauss"])]
                ax.plot([x["g"] for x in q], [x["lt_LNA_gauss"] / np.log(10) for x in q], "o", ms=6, color=C[4],
                        mec="white", mew=1.5, label="coherent LNA (Gaussian)")
        for thr in (200, 3000):
            ax.axhline(np.log10(thr), color=MUTED, lw=0.8, ls=":")
        ax.axvline(0.16315, color=MUTED, lw=0.8)
        ax.set_xscale("log"); ax.set_ylim(0, 8); ax.set_xlim(0.16, 6 if m == "M2" else 1.2)
        ax.set_xlabel("coupling g"); ax.set_ylabel("log10 lifetime (model units)"); ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{F}/stoch_lifetime.png", dpi=150); plt.close(fig)


def fig_modes():
    mo = J(f"{T}/stochastic_modes.json")["modes"]
    ms = [m for m in mo if not m.get("complex") and m.get("purity_v", 0) >= 0.9]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for m in ms:
        ax.annotate("", xy=(m["K0"], m["PF_M2_2D_on"] or 40), xytext=(m["K0"], m["PF_M1_on"] or 40),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
        ax.plot(m["K0"], m["PF_M1_on"], "o", color=C[0], ms=8, mec="white", mew=1.5)
        ax.plot(m["K0"], m["PF_M2_2D_on"], "s", color=C[1], ms=8, mec="white", mew=1.5)
        ax.text(m["K0"] * 1.08, m["PF_M2_2D_on"], f"mode {m['k']} (AL {m['AL_v']:.2f})", fontsize=7, color=INK)
    ax.plot([], [], "o", color=C[0], label="M1 single-mode onset"); ax.plot([], [], "s", color=C[1], label="M2 single-mode onset (2-D)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("mode size K0"); ax.set_ylabel("onset g (tau = 3000)")
    ax.legend(frameon=False, fontsize=7); ax.set_title("Depression ranks modes by size", loc="left")
    fig.tight_layout(); fig.savefig(f"{F}/stoch_modes.png", dpi=150); plt.close(fig)


def fig_driven():
    dv = J(f"{T}/stochastic_driven.json")
    keys = [k for k in ("M2_orn0.04", "M1_orn0.04", "M2_orn0.1") if k in dv]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    for j, k in enumerate(keys):
        rows = dv[k]["rows"]; g = [r["g"] for r in rows]
        axs[0, 0].plot(g, [r["G_gain"] for r in rows], color=C[j], label=k)
        axs[0, 1].plot(g, [r["fano_nonORN"] for r in rows], color=C[j], label=k)
        axs[1, 0].plot(g, [r["tau_int"] for r in rows], color=C[j], label=k)
        axs[1, 1].plot(g, [r["m_unit"] for r in rows], color=C[j], label=k)
    axs[0, 0].axhspan(0.29, 1.45, color="#dddddd", alpha=0.5, lw=0)
    for ax, yl in zip(axs.ravel(), ("PN/ORN gain G", "Fano chi/rho (nonORN)", "tau_int nonORN (units)", "MR branching ratio m per unit")):
        ax.set_xscale("log"); ax.set_xlabel("coupling g"); ax.set_ylabel(yl); ax.legend(frameon=False, fontsize=7)
    axs[0, 1].set_yscale("log")
    axs[1, 1].axhline(0.99, color=MUTED, ls=":", lw=0.8)
    fig.suptitle("Driven state, linear-noise approximation (r = 0 closure)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(f"{F}/stoch_driven.png", dpi=150); plt.close(fig)


def fig_aval():
    av = J(f"{T}/stochastic_aval.json")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    j = 0
    for nm, d in av.items():
        if nm == "notes":
            continue
        for gk, gd in d["by_gap"].items():
            rows = gd["rows"]; g = [r["g"] for r in rows]
            axs[0].plot(g, [max(r["eps"], 1e-9) for r in rows], color=C[j % 6], label=f"{nm} {gk}")
            axs[1].plot(g, [min(r["S_eps"], r["S_dep"]) for r in rows], color=C[j % 6], label=f"{nm} {gk}")
            j += 1
    axs[0].set_yscale("log"); axs[0].set_ylabel("self-organised eps = 1 - m")
    axs[1].set_yscale("log"); axs[1].axhline(1000, color=MUTED, ls=":"); axs[1].set_ylabel("power-law cutoff min(S_eps, S_dep)")
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("coupling g"); ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(f"{F}/stoch_aval.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_lifetime, fig_modes, fig_driven, fig_aval):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("fail", f.__name__, repr(e))
