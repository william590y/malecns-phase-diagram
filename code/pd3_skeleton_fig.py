"""pd3 module_theory deliverable: module skeleton of the (g, r) phase diagram next to de Candia's closure (CPU).
Left (neuron level, this connectome):
  solid black   g_lin = 0.1632: the SL edge of the exact process (MF-DP transcritical, r-independent; phase_complete.json);
  thin dotted   QMF SL edge (net-input closure at neuron level, r-dependent): up-sweep leaves silence (qmf_analysis.json);
  dashed        simulated crossovers tau = 3000 (onset) and PR = N/4 (D) (results/analysis2/flow_sim_boundaries.json);
  dotted, light imperfect island activation: first g where island 1 / the median island / the rest reaches 1% activity
                (simulation, among survivors; module_occ_P0.json) - smeared recruitment, not sharp lines;
  markers       neuron-level QMF: g with >= 2 coexisting stable fixed points (MT3-multi), hysteresis with both branches fixed
                points (type A) or fixed point vs non-stationary branch (type B); hatched band: g where the all-active QMF does
                not settle (non-stationary mean field). These are MEAN-FIELD candidates, not confirmed first-order lines.
Right (de Candia's closure, P0 modules; coarse_mf_P0.npz): SL edge (solid), multistable cells, first-order sweep jumps.
Output: figs/pd3/modules_skeleton.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/MALECNS_ROOT"
OK = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
fsb = json.load(open(f"{ROOT}/results/analysis2/flow_sim_boundaries.json"))["rows"]
occ = json.load(open(f"{ROOT}/results/pd3/modules/module_occ_P0.json"))
qa = json.load(open(f"{ROOT}/results/pd3/module_theory/qmf_analysis.json"))["per_r"]
fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.6), dpi=130, gridspec_kw={"wspace": 0.18})
ax = axs[0]
R = [row["r"] for row in fsb]
ax.axvline(0.1632, color="k", lw=2.4, label="g_lin = 0.1632: SL edge (MF-DP, exact, r-independent)")
ax.plot([row["on"] for row in fsb], R, color=OK[4], lw=1.8, ls="--", label="tau = 3000 crossover (sim onset; B_AL sustained)")
Dg = [(row["D"], row["r"]) for row in fsb if row["D"]]
ax.plot([x for x, _ in Dg], [y for _, y in Dg], color=OK[5], lw=1.8, ls="--", label="PR = N/4 crossover (sim D)")
nr = [(row["D_not_reached_up_to"], row["r"]) for row in fsb if not row["D"] and row["D_not_reached_up_to"]]
ax.scatter([x for x, _ in nr], [y for _, y in nr], marker=">", color=OK[5], s=18, label="D not reached up to this g (sim)")
rs = sorted(float(r) for r in occ["sim"])
i1 = [occ["sim"][f"{r:g}"]["cross_1pct_survivors"].get("isl1") for r in rs]
isl = [k for k in occ["modules"] if k.startswith("isl")]
med = [np.nanmedian([occ["sim"][f"{r:g}"]["cross_1pct_survivors"][k] or np.nan for k in isl]) for r in rs]
rest = [occ["sim"][f"{r:g}"]["cross_1pct_survivors"].get("rest") for r in rs]
ax.plot([x if x else np.nan for x in i1], rs, color=OK[1], lw=1.5, ls=":", label="island 1 reaches 1% (sim; imperfect)")
ax.plot(med, rs, color=OK[3], lw=1.5, ls=":", label="median island reaches 1% (sim)")
ax.plot([x if x else np.nan for x in rest], rs, color=OK[6], lw=1.5, ls=":", label="rest reaches 1% (sim)")
qr = sorted(float(r) for r in qa)
for r in qr:
    a = qa[f"{r:g}"]
    ns = [x for x in a["nonstationary_g_all_active_pass1"] if x in a["pass3_dt0.05_oscillating_g"]] or \
        (a["pass3_dt0.05_oscillating_g"] if a["nonstationary_g_all_active_pass1"] else [])
    if ns:
        ax.plot([min(ns), max(ns)], [r, r], color="0.6", lw=6, alpha=0.35, solid_capstyle="butt")
    if a["g_multistable_fixed_points"]:
        ax.scatter(a["g_multistable_fixed_points"], [r] * len(a["g_multistable_fixed_points"]), marker="s", s=16, facecolors="none",
                   edgecolors=OK[0], lw=0.8)
    for h in a["hysteresis_runs"]:
        if h["n_type_A_both_fixed_points"] >= 2:
            ax.plot([h["g_lo"], h["g_hi"]], [r + 0.08, r + 0.08], color=OK[5], lw=2.5)
        elif h["n_type_B_fixed_point_vs_nonstationary"] >= 2:
            ax.plot([h["g_lo"], h["g_hi"]], [r + 0.08, r + 0.08], color=OK[2], lw=2.5)
    if a.get("SL_edge_up_sweep"):
        ax.plot(a["SL_edge_up_sweep"], r, marker="|", color="0.3", ms=9)
ax.plot([], [], color="0.6", lw=6, alpha=0.35, label="QMF does not settle (all-active start, dt = 0.2 and 0.05)")
ax.scatter([], [], marker="s", s=16, facecolors="none", edgecolors=OK[0], label="QMF: >= 2 stable fixed points")
ax.plot([], [], color=OK[5], lw=2.5, label="QMF hysteresis, both branches fixed points")
ax.plot([], [], color=OK[2], lw=2.5, label="QMF hysteresis, fixed point vs non-stationary")
ax.plot([], [], marker="|", color="0.3", ls="", ms=9, label="QMF SL edge (net-input closure)")
ax.text(0.125, 5.6, "SL", fontsize=13); ax.text(0.19, 5.6, "rounded\ncritical\nband", fontsize=8)
ax.text(0.75, 0.15, "B_AL + island\ncascade", fontsize=9); ax.text(6, 0.15, "D", fontsize=13)
ax.set_xscale("log"); ax.set_xlim(0.12, 32); ax.set_ylim(-0.2, 6.3)
ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition r")
ax.set_title("(a) neuron level: module skeleton (simulation crossovers + QMF structure)", fontsize=10)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, framealpha=0.95)
ax = axs[1]
if os.path.exists(f"{ROOT}/results/pd3/modules/coarse_mf_P0.npz"):
    d = np.load(f"{ROOT}/results/pd3/modules/coarse_mf_P0.npz", allow_pickle=True); js = json.load(open(f"{ROOT}/results/pd3/modules/coarse_mf_P0.json"))
    m_up = (d["up"] > 1e-3).sum(2)
    im = ax.pcolormesh(d["g"], d["r"], m_up, cmap="viridis", vmin=0, vmax=d["up"].shape[2], shading="nearest", alpha=0.85)
    post = np.load(f"{ROOT}/results/pd3/modules/coarse_post_P0.npz", allow_pickle=True)
    pj = json.load(open(f"{ROOT}/results/pd3/modules/coarse_post_P0.json"))
    yy, xx = np.where(post["n_attr_corr"] == 0)
    ax.scatter(d["g"][xx], d["r"][yy], s=3, marker=".", color="w", alpha=0.8)
    ax.plot(np.where(d["g_SL_edge"] < 30, d["g_SL_edge"], np.nan), d["r"], color="k", lw=2.4, label="closure SL edge (second order)")
    for b in pj["hysteresis_intervals"]:
        ax.plot([b["g_lo"], b["g_hi"]], [b["r"], b["r"]], color=OK[1], lw=1.6)
    ax.plot([], [], color=OK[1], lw=1.6, label="first order: up/down hysteresis interval")
    ax.scatter([], [], s=8, marker=".", color="0.5", label="white dots: no stable state (non-stationary)")
    labs_ = post["labels_corr"]; rv, gv_ = d["r"], d["g"]
    bs = [(float(rv[ri]), float(gv_[gi])) for ri in range(len(rv)) for gi in range(len(gv_))
          if any("AL-L" in x and "AL-R" not in x for x in labs_[ri, gi].split("|"))
          and any("AL-R" in x and "AL-L" not in x for x in labs_[ri, gi].split("|"))]
    if bs:
        rr_ = sorted(set(x for x, _ in bs)); gmin = [min(g_ for r_, g_ in bs if r_ == x) for x in rr_]
        ax.plot(gmin, rr_, color=OK[6], lw=2, ls="--", label="B1L | B1R coexist from here (AL lobes, closure)")
    ax.axvline(0.1632, color="0.6", lw=1, label="neuron-level g_lin")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("# active modules (up-sweep)", fontsize=8)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, framealpha=0.95)
ax.set_xscale("log"); ax.set_xlim(0.12, 30); ax.set_ylim(-0.15, 6.15); ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition r")
ax.set_title("(b) de Candia's closure on the same 31 modules (P0)", fontsize=10)
fig.suptitle("Module skeleton of the (g, r) diagram: neuron level vs de Candia's coarse closure", fontsize=12, y=0.99)
fig.savefig(f"{ROOT}/figs/pd3/modules_skeleton.png", bbox_inches="tight")
print("saved figs/pd3/modules_skeleton.png")
