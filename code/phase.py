"""Phase diagram of the stochastic Wilson-Cowan model on MaleCNS (raw weights) from the results/pd sweeps.

Per run (one coupling g, one inhibition ratio r, B replicas from a fully active start, h = 0) we measure
  life     time at which the last replica went silent (inf if any replica is alive at T)
  surv_T   fraction of replicas alive at T
  rho_T    fraction of neurons active at T
  PR       participation ratio of the per-neuron occupancy over the second half of the run, as a fraction of N
  AL       share of the occupancy inside the antennal-lobe module (top-1000 neurons of the leading excitatory eigenvector)
and classify (operational definitions, chosen before looking at the stage-1 sweep and kept fixed):
  Q  quiescent          all replicas silent, life < 200       (fast, exponential-like death)
  S  slow / Griffiths   all replicas silent with life >= 200, or 0 < surv_T < 1 (metastable: long, broadly distributed
                        lifetimes, the Griffiths-window signature in a finite network)
  L  localized active   surv_T = 1 and PR < 0.25
  D  delocalized active surv_T = 1 and PR >= 0.25            (0.25 N: the configuration null jumps to ~0.45 N at onset)
Driven runs (h > 0, r = 1): stationary mean density, replica-to-replica fluctuation N * var, and the response to h.
Writes results/pd/phase.json and figs/pd_*.png. Skips whatever is missing, so it can be rerun as jobs finish.
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = "/MALECNS_ROOT"
R, F, C = f"{ROOT}/results", f"{ROOT}/figs", "/MALECNS_CACHE"
N = 167216
LIFE_SLOW, PR_DELOC = 200.0, 0.25
sup = np.load(C + "/superclasses.npy")
V = np.load(f"{R}/sym_exc_weighted_vecs_top10.npy")
mod = np.zeros(N, bool); mod[np.argsort(-V[:, 0] ** 2)[:1000]] = True
PH = ["Q", "S", "L", "D"]
PH_NAME = {"Q": "quiescent", "S": "slow (Griffiths window)", "L": "localized active", "D": "delocalized active"}
PH_COL = ["#d9dde3", "#e8a33d", "#2a78b5", "#7b2d8e"]     # fixed order: Q, S, L, D
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def run_stats(path):
    d = np.load(path)
    t, g, rho, surv = d["t"], d["g"], d["rho"], d["surv"]
    occ = d["occ"].astype(np.float64)
    rows = []
    for i, gg in enumerate(g):
        m = rho[i].mean(-1)
        alive = surv[i, -1].mean()
        life = float(t[np.argmax(m == 0)]) if (m == 0).any() else float("inf")
        o = occ[i]
        pr = float(o.sum() ** 2 / (o ** 2).sum() / N) if o.sum() > 0 else 0.0
        al = float(o[mod].sum() / o.sum()) if o.sum() > 0 else float("nan")
        if alive == 0:
            ph = "Q" if life < LIFE_SLOW else "S"
        elif alive < 1:
            ph = "S"
        else:
            ph = "D" if pr >= PR_DELOC else "L"
        rows.append({"g": float(gg), "life": life, "surv_T": float(alive), "rho_T": float(m[-1]), "PR": pr, "AL": al,
                     "phase": ph})
    return rows, occ, g


out = {"real": {}, "null": {}, "wnull": {}, "drive": {}}
occs = {}
for p in sorted(glob.glob(f"{R}/pd/decay_r*_*.npz")):
    r, kind = re.match(r".*decay_r([\d.]+)_(\w+)\.npz", p).groups()
    rows, occ, g = run_stats(p)
    out[kind][r] = rows
    if kind == "real" and r == "1":
        occs["r1"] = (g, occ)
for p in sorted(glob.glob(f"{R}/pd/drive_h*_*.npz")):
    h, kind = re.match(r".*drive_h([\de.+-]+)_(\w+)\.npz", p).groups()
    d = np.load(p)
    t, g, rho = d["t"], d["g"], d["rho"]
    late = t >= t[-1] / 2
    rows = []
    for i, gg in enumerate(g):
        x = rho[i][late]                      # [late times, B]
        rows.append({"g": float(gg), "rho": float(x.mean()), "fluct": float(N * x.mean(0).var()),
                     "fluct_t": float(N * x.var())})
    out["drive"][f"{kind}_h{h}"] = {"h": float(h), "rows": rows}
json.dump(out, open(f"{R}/pd/phase.json", "w"), indent=1)


def phase_map(ax, block, title):
    rs = sorted(block, key=float)
    if not rs:
        ax.set_visible(False); return
    g = np.array([x["g"] for x in block[rs[0]]])
    Z = np.array([[PH.index(x["phase"]) for x in block[r]] for r in rs])
    cmap, norm = ListedColormap(PH_COL), BoundaryNorm(np.arange(-0.5, 4.5), 4)
    ge = np.r_[g[0] / 1.1, np.sqrt(g[1:] * g[:-1]), g[-1] * 1.1]
    ax.pcolormesh(ge, np.arange(len(rs) + 1) - 0.5, Z, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.6)
    ax.set_xscale("log"); ax.set_yticks(range(len(rs))); ax.set_yticklabels(rs)
    ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition ratio r"); ax.set_title(title)


os.makedirs(F, exist_ok=True)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [2.2, 1, 1]})
phase_map(axs[0], out["real"], "MaleCNS")
phase_map(axs[1], out["null"], "configuration null (wiring shuffled)")
phase_map(axs[2], out["wnull"], "weight-shuffled null (strengths shuffled)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PH_COL]
fig.legend(handles, [PH_NAME[p] for p in PH], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{F}/pd_phase_gr.png"); plt.close(fig)

# continuous order parameters on the (g, r) plane, real connectome
if out["real"]:
    rs = sorted(out["real"], key=float)
    g = np.array([x["g"] for x in out["real"][rs[0]]])
    ge = np.r_[g[0] / 1.1, np.sqrt(g[1:] * g[:-1]), g[-1] * 1.1]
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, key, lab, f in [(axs[0], "rho_T", "log10 fraction active", lambda v: np.log10(max(v, 1e-7))),
                            (axs[1], "PR", "participation ratio / N", lambda v: v),
                            (axs[2], "AL", "share in antennal-lobe module", lambda v: v)]:
        Z = np.array([[f(x[key]) if x["surv_T"] > 0 else np.nan for x in out["real"][r]] for r in rs])
        im = ax.pcolormesh(ge, np.arange(len(rs) + 1) - 0.5, Z, cmap="viridis", shading="flat")
        ax.set_xscale("log"); ax.set_yticks(range(len(rs))); ax.set_yticklabels(rs)
        ax.set_xlabel("coupling g"); ax.set_ylabel("r"); plt.colorbar(im, ax=ax, label=lab)
    fig.suptitle("MaleCNS order parameters at the end of the run (blank = all replicas silent)")
    fig.tight_layout(); fig.savefig(f"{F}/pd_order_gr.png"); plt.close(fig)

# driven plane (r = 1): density and fluctuations vs g for each h
dr = {k: v for k, v in out["drive"].items() if k.startswith("real_")}
if dr:
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    hs = sorted(dr, key=lambda k: dr[k]["h"])
    cm = plt.cm.viridis(np.linspace(0.1, 0.9, len(hs)))
    for c, k in zip(cm, hs):
        rows = dr[k]["rows"]; g = [x["g"] for x in rows]
        axs[0].loglog(g, [x["rho"] for x in rows], "o-", ms=2.5, c=c, label=f"h = {dr[k]['h']:g}")
        axs[1].loglog(g, [x["fluct_t"] for x in rows], "o-", ms=2.5, c=c)
    if "1" in out["real"]:
        rows = [x for x in out["real"]["1"] if x["rho_T"] > 0]
        axs[0].loglog([x["g"] for x in rows], [x["rho_T"] for x in rows], "k--", lw=1, label="h = 0")
    axs[0].set_xlabel("coupling g"); axs[0].set_ylabel("stationary fraction active"); axs[0].legend(fontsize=7)
    axs[1].set_xlabel("coupling g"); axs[1].set_ylabel("fluctuations  N·var(ρ)")
    axs[0].set_title("driven network (r = 1): activity"); axs[1].set_title("fluctuations (peak = critical crossover)")
    fig.tight_layout(); fig.savefig(f"{F}/pd_drive.png"); plt.close(fig)

# recruitment of brain divisions vs g (r = 1)
if "r1" in occs:
    g, occ = occs["r1"]
    cls = [c for c in np.unique(sup) if (sup == c).sum() >= 200]
    M = np.array([[occ[i][sup == c].mean() for i in range(len(g))] for c in cls])
    order = np.argsort([np.argmax(row > 1e-3) if (row > 1e-3).any() else 99 for row in M])
    ge = np.r_[g[0] / 1.1, np.sqrt(g[1:] * g[:-1]), g[-1] * 1.1]
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(cls) + 1.2))
    im = ax.pcolormesh(ge, np.arange(len(cls) + 1), np.log10(np.maximum(M[order], 1e-6)), cmap="magma", vmin=-6, vmax=-0.5)
    ax.set_xscale("log"); ax.set_yticks(np.arange(len(cls)) + 0.5); ax.set_yticklabels([cls[i] for i in order])
    ax.set_xlabel("coupling g"); ax.set_title("Which parts of the CNS are active (mean occupancy per superclass, r = 1, h = 0)")
    plt.colorbar(im, ax=ax, label="log10 mean occupancy")
    fig.tight_layout(); fig.savefig(f"{F}/pd_recruitment.png"); plt.close(fig)

for kind in ["real", "null", "wnull"]:
    for r in sorted(out[kind], key=float):
        s = "".join(x["phase"] for x in out[kind][r])
        print(f"{kind:5s} r={r:>4s}  {s}")
print("g grid:", " ".join("%.3g" % x["g"] for x in next(iter(out["real"].values()), [])))
