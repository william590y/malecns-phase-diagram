"""Figures for the MaleCNS Griffiths investigation (static PNGs into figs/). Skips any input that is missing.
  fig1_decay_curves.png      rho(t) from a fully active start, log-log, one curve per coupling g (raw norm; real vs null)
  fig2_slope_map.png         local decay exponent d log rho / d log t over (g, t): the Griffiths signature is a band of g
                             with slow, g-dependent power-law decay between fast death and sustained activity
  fig3_order_parameter.png   stationary density, survival and lifetime vs g (raw real vs null; in-norm r = 0.5/1/2 + null)
  fig4_spreading.png         single-seed spreading: survival P(t) and mean active count N(t)
  fig5_spectrum.png          IPR vs eigenvalue (top 100) real vs null, and the leading eigenvector on soma positions
  fig6_activity_maps.png     where activity lives: per-neuron occupancy on soma positions (top view) at several g
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = "/MALECNS_ROOT"
R, F = f"{ROOT}/results", f"{ROOT}/figs"
C = "/MALECNS_CACHE"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
load = lambda n: np.load(f"{R}/{n}") if os.path.exists(f"{R}/{n}") else None
real_raw = load("fine_decay_raw_r1.npz")
if real_raw is None:
    real_raw = load("decay_raw_r1.npz")
null_raw = load("fine_decay_raw_r1_null.npz")
if null_raw is None:
    null_raw = load("decay_raw_r1_null.npz")


def curves(ax, d, title):
    t, g, rho = d["t"], d["g"], d["rho"].mean(-1)
    cm = plt.cm.viridis((g - g.min()) / (np.ptp(g) or 1))
    for i in range(len(g)):
        m = rho[i] > 0
        ax.loglog(t[m], rho[i][m], color=cm[i], lw=1)
    ax.set_xlabel("time (1/α)"); ax.set_ylabel("ρ(t), fraction active"); ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(g.min(), g.max())); sm.set_array([])
    plt.colorbar(sm, ax=ax, label="coupling g")


# fig1
if real_raw is not None:
    fig, axs = plt.subplots(1, 2 if null_raw is not None else 1, figsize=(10, 4), squeeze=False)
    curves(axs[0, 0], real_raw, "MaleCNS (raw synapse weights)")
    if null_raw is not None:
        curves(axs[0, 1], null_raw, "configuration-model null")
    fig.suptitle("Decay of activity from a fully active start (stochastic Wilson–Cowan, h = 0)")
    fig.tight_layout(); fig.savefig(f"{F}/fig1_decay_curves.png"); plt.close(fig)


def slope_map(ax, d, title):
    t, g, rho = d["t"], d["g"], d["rho"].mean(-1)
    lt = np.log(t)
    S = np.full((len(g), len(t) - 4), np.nan)
    for i in range(len(g)):
        lr = np.log(np.where(rho[i] > 0, rho[i], np.nan))
        for k in range(2, len(t) - 2):
            seg = slice(k - 2, k + 3)
            if np.isfinite(lr[seg]).all():
                S[i, k - 2] = np.polyfit(lt[seg], lr[seg], 1)[0]
    im = ax.pcolormesh(t[2:-2], g, S, cmap="magma", vmin=-4, vmax=0, shading="nearest")
    ax.set_xscale("log"); ax.set_xlabel("time (1/α)"); ax.set_ylabel("coupling g"); ax.set_title(title)
    plt.colorbar(im, ax=ax, label="local exponent d log ρ / d log t  (white = extinct)")


if real_raw is not None:
    fig, axs = plt.subplots(1, 2 if null_raw is not None else 1, figsize=(10, 4), squeeze=False)
    slope_map(axs[0, 0], real_raw, "MaleCNS")
    if null_raw is not None:
        slope_map(axs[0, 1], null_raw, "null")
    fig.suptitle("Local decay exponent: ~0 = sustained, −0.1…−2 = slow power-law-like, white = all replicas extinct")
    fig.tight_layout(); fig.savefig(f"{F}/fig2_slope_map.png"); plt.close(fig)


# fig3
def stats(d):
    t, g, rho, surv = d["t"], d["g"], d["rho"].mean(-1), d["surv"].mean(-1)
    life = np.array([t[np.argmax(r == 0)] if (r == 0).any() else np.nan for r in rho])
    return g, rho[:, -1], surv[:, -1], life


fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for d, lab, c in [(real_raw, "MaleCNS raw", "C0"), (null_raw, "null raw", "C1")]:
    if d is None:
        continue
    g, rhoT, sT, life = stats(d)
    axs[0].semilogx(g, rhoT, "o-", ms=3, c=c, label=lab)
    axs[1].semilogx(g, life, "o-", ms=3, c=c, label=lab)
for n, lab in [("decay_in_r05.npz", "in-norm r=0.5"), ("decay_in_r1.npz", "in-norm r=1"), ("decay_in_r2.npz", "in-norm r=2"),
               ("decay_in_r1_null.npz", "in-norm null r=1")]:
    d = load(n)
    if d is not None:
        g, rhoT, sT, life = stats(d)
        axs[2].plot(g, rhoT, "o-", ms=3, label=lab)
axs[0].set_xlabel("g"); axs[0].set_ylabel("ρ at end of run"); axs[0].set_yscale("log"); axs[0].legend()
axs[0].set_title("order parameter (raw weights)")
axs[1].set_xlabel("g"); axs[1].set_ylabel("time until all replicas extinct"); axs[1].set_yscale("log")
axs[1].set_title("lifetime below onset (NaN = survives)")
axs[2].set_xlabel("g"); axs[2].set_ylabel("ρ at end of run"); axs[2].set_title("in-normalized: effect of inhibition r"); axs[2].legend()
fig.tight_layout(); fig.savefig(f"{F}/fig3_order_parameter.png"); plt.close(fig)

# fig4 spreading
sp_r, sp_n = load("spread_raw_r1.npz"), load("spread_raw_r1_null.npz")
if sp_r is not None or sp_n is not None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for d, ls, lab in [(sp_r, "-", "MaleCNS"), (sp_n, "--", "null")]:
        if d is None:
            continue
        t, g = d["t"], d["g"]
        N = 167216
        for i in range(len(g)):
            P = d["surv"][i].mean(-1); n = d["rho"][i].mean(-1) * N
            axs[0].loglog(t, np.where(P > 0, P, np.nan), ls, label=f"{lab} g={g[i]:.2f}")
            axs[1].loglog(t, np.where(n > 0, n, np.nan), ls, label=f"{lab} g={g[i]:.2f}")
    axs[0].set_xlabel("time"); axs[0].set_ylabel("survival P(t)"); axs[0].legend(fontsize=6, ncol=2)
    axs[1].set_xlabel("time"); axs[1].set_ylabel("mean active neurons N(t)")
    fig.suptitle("Spreading from one random seed neuron (512 replicas per g)")
    fig.tight_layout(); fig.savefig(f"{F}/fig4_spreading.png"); plt.close(fig)

# fig5 spectrum
pos = np.load(C + "/positions.npy")
if os.path.exists(f"{R}/spectrum.json"):
    s = json.load(open(f"{R}/spectrum.json"))
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 3, 1)
    for k, lab in [("sym_exc_weighted", "MaleCNS"), ("null_sym_exc_weighted", "null")]:
        ax.scatter(s[k]["top_eigs"], s[k]["ipr"], s=8, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("eigenvalue (symmetrized excitatory synapse matrix)"); ax.set_ylabel("IPR")
    ax.axhline(1 / 167216, c="gray", ls=":", lw=0.8); ax.text(ax.get_xlim()[0], 1.3 / 167216, " fully delocalized 1/N", fontsize=7)
    ax.legend(); ax.set_title("localization of the top 100 modes")
    V = np.load(f"{R}/sym_exc_weighted_vecs_top10.npy")
    ok = np.isfinite(pos).all(1)
    for j, (a, b, nm) in enumerate([(0, 1, "top view (x, y)"), (0, 2, "side view (x, z)")]):
        ax = fig.add_subplot(1, 3, 2 + j)
        ax.scatter(pos[ok, a], pos[ok, b], s=0.05, c="0.85", rasterized=True)
        p = V[:, 0] ** 2; o = np.argsort(p)
        sel = o[-3000:]; sel = sel[ok[sel]]
        sc = ax.scatter(pos[sel, a], pos[sel, b], s=2, c=p[sel], cmap="inferno_r", norm=LogNorm(), rasterized=True)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_title(f"leading mode |v|², {nm}"); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc, ax=ax, label="|v_i|²")
    fig.tight_layout(); fig.savefig(f"{F}/fig5_spectrum.png"); plt.close(fig)

# fig6 activity maps
d = real_raw
if d is not None and "occ" in d.files:
    g, occ = d["g"], d["occ"].astype(np.float32)
    ok = np.isfinite(pos).all(1)
    picks = [i for i in np.linspace(0, len(g) - 1, 6).astype(int) if occ[i].max() > 0] or [len(g) - 1]
    fig, axs = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.4), squeeze=False)
    for ax, i in zip(axs[0], picks):
        o = occ[i]
        ax.scatter(pos[ok, 0], pos[ok, 1], s=0.03, c="0.88", rasterized=True)
        act = ok & (o > 1e-4)
        ax.scatter(pos[act, 0], pos[act, 1], s=0.6, c=o[act], cmap="viridis", norm=LogNorm(1e-4, 1), rasterized=True)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"g = {g[i]:.3f}\n{act.sum()} neurons active >1e-4")
    fig.suptitle("Per-neuron occupancy (second half of run, mean over replicas) on soma positions, top view")
    fig.tight_layout(); fig.savefig(f"{F}/fig6_activity_maps.png"); plt.close(fig)
# fig7 localization of the active state vs g (from analysis.json)
if os.path.exists(f"{R}/analysis.json"):
    A = json.load(open(f"{R}/analysis.json"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, lab, c in [("fine_decay_raw_r1", "MaleCNS", "C0"), ("fine_decay_raw_r1_null", "null", "C1")]:
        rows = [r for r in A.get(k, []) + (A.get("upper_decay_raw_r1", []) if k == "fine_decay_raw_r1" else []) if "PR_occ" in r]
        if not rows:
            continue
        g = [r["g"] for r in rows]
        axs[0].loglog(g, [r["PR_occ"] for r in rows], "o-", ms=3, c=c, label=lab)
        axs[1].semilogx(g, [r["share_AL_module"] for r in rows], "o-", ms=3, c=c, label=lab)
    axs[0].axhline(167216, c="gray", ls=":", lw=0.8)
    axs[0].set_xlabel("coupling g"); axs[0].set_ylabel("participation ratio of activity (neurons)")
    axs[0].set_title("how many neurons carry the surviving activity"); axs[0].legend()
    axs[1].set_xlabel("coupling g"); axs[1].set_ylabel("share of activity in the antennal-lobe module")
    axs[1].set_title("share in the antennal-lobe module (leading eigenvector)")
    fig.tight_layout(); fig.savefig(f"{F}/fig7_localization.png"); plt.close(fig)
print("figures in", F, sorted(os.listdir(F)))
