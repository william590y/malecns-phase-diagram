"""pd3 module_theory MT3 analysis: attractor census, hysteresis, oscillations of the neuron-level QMF (CPU) + figure.

Inputs: results/pd3/module_theory/qmf_r<r>.npz (pd3_qmf_ic.py), results/pd3/modules/partition_P0.npz, modules_table.json.
Decisions (plan.md 3.3):
  MT3-multi: two starts converged (residual < 1e-7 AND estimated distance to the fixed point residual / (1 - rho(J)) < 1e-6;
             the second condition, added after inspection, removes slowly drifting columns next to the SL edge, e.g. r = 0.25,
             g = 0.168, rho(J) = 0.9998) to fixed points with rho(J) < 1 whose 34-module vectors differ by > 10%
             relative L1 (|a - b|_1 / max(|a|_1, |b|_1)). States with mean activity < 1e-8 (< 0.002 active neurons) are set to
             the silent state before comparing (otherwise slowly decaying sub-threshold columns count as distinct states).
             Non-stationary columns (residual > 1e-5 and a dominant period <= 512 steps in the 1024-step record) are compared
             through their time-averaged module vectors and reported separately (they are not fixed points); the rest of the
             unconverged columns (residual 1e-7..1e-5, or monotone transients near the SL edge) are 'slow' and excluded.
  MT3-order: first order = up/down branches (time averages) differ by > 5% (relative) on >= 2 consecutive g points, with a
             jump on a branch and rho(J) -> 1 at the branch end; otherwise continuous. Points where either branch is still
             converging (residual 1e-7..1e-5), |rho(J) - 1| < 0.01 (critical slowing) or max(up, down) < 1e-6 are excluded.
Also: SL edge of the QMF (up-sweep leaves silence; silent-state growth rate crosses 1), QMF D (PR/N = 1/4), the oscillation
region (all-active start not converged), the dt = 0.05 check of pass 3, and which neurons oscillate (all_active_std > 1e-3).
Outputs: results/pd3/module_theory/qmf_analysis.json, figs/pd3/modules_qmf_phase.png
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap

ROOT = "/MALECNS_ROOT"
C = "/MALECNS_CACHE"
OK = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
part = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz", allow_pickle=True)
memb = part["membership"].astype(int); names = [str(x) for x in part["names"]]; M = len(names)
sup = np.load(C + "/superclasses.npy", allow_pickle=True)
tab = {r["name"]: r for r in json.load(open(f"{ROOT}/results/pd3/modules/modules_table.json"))["P0"]["rows"]}
gl = np.array([tab[n]["g_lin"] if n in tab else np.inf for n in names])
order = np.argsort(gl, kind="stable")
files = sorted([f for f in glob.glob(f"{ROOT}/results/pd3/module_theory/qmf_r*.npz") if "smoke" not in f and "tmp" not in f],
               key=lambda f: float(json.loads(str(np.load(f, allow_pickle=True)["meta"]))["r"]))


def rel_l1(a, b):
    return float(np.abs(a - b).sum() / max(np.abs(a).sum(), np.abs(b).sum(), 1e-300))


def clusters(vecs, thr=0.1):
    reps = []
    for v in vecs:
        if not any(rel_l1(v, u) <= thr for u in reps):
            reps.append(v)
    return reps


def first_cross(g, y, thr):
    idx = np.where(y >= thr)[0]
    if not len(idx) or idx[0] == 0:
        return None
    i = idx[0]; x0, x1 = np.log(g[i - 1]), np.log(g[i]); y0, y1 = y[i - 1], y[i]
    return float(np.exp(x0 + (thr - y0) / (y1 - y0) * (x1 - x0)))


res = {"files": [], "per_r": {}}
D = {}
for f in files:
    d = np.load(f, allow_pickle=True); m = json.loads(str(d["meta"])); r = float(m["r"]); D[r] = d
    res["files"].append({"file": f.replace(ROOT + "/", ""), "r": r, "gpu": m["gpu"], "seconds": m["seconds"]})
    g1, g2, st = d["g1"], d["g2"], list(d["starts"])
    conv = (d["p1_resid"] < 1e-7) & (d["p1_resid"] / np.maximum(1 - d["p1_rhoJ"], 1e-6) < 1e-6)   # residual AND est. distance
    stab = d["p1_rhoJ"] < 1
    osc = (d["p1_resid"] > 1e-5) & (d["p1_period"] <= 512)    # non-stationary with a dominant period < half the 1024-step window
    slow = ~conv & ~osc                                       # still converging (residual 1e-7..1e-5, or a monotone transient)
    Xfp = d["p1_X"].copy(); Xavg = d["p1_tavg_X"].copy()
    zero = d["p1_mean"] < 1e-8                                # < 0.002 active neurons in total: the silent state (SL)
    Xfp[zero] = 0; Xavg[zero & ~osc] = 0
    census = []
    n_fp, n_osc, multi_g, osc_g, any_diff = [], [], [], [], []
    for gi, g in enumerate(g1):
        fp = [Xfp[s, gi] for s in range(len(st)) if conv[s, gi] and stab[s, gi]]
        ov = [Xavg[s, gi] for s in range(len(st)) if osc[s, gi]]
        fpc = clusters(fp); ovc = clusters(ov)
        n_fp.append(len(fpc)); n_osc.append(len(ovc))
        ref = Xavg[0, gi]
        dif = [st[s] for s in range(len(st)) if rel_l1(Xavg[s, gi], ref) > 0.1]
        any_diff.append(dif)
        if len(fpc) >= 2:
            multi_g.append(float(g))
        census.append({"g": float(g), "n_fixed_point_attractors": len(fpc), "n_oscillatory_states": len(ovc),
                       "n_nonstationary_starts": int(osc[:, gi].sum()), "n_slow_starts": int(slow[:, gi].sum()),
                       "n_conv_unstable": int((conv[:, gi] & ~stab[:, gi]).sum()),
                       "starts_differing_from_all_active_by_gt10pct": dif,
                       "max_rel_l1_to_all_active": float(max(rel_l1(Xavg[s, gi], ref) for s in range(len(st))))})
    iL, iR = names.index("AL-L"), names.index("AL-R")
    b1 = [float(g) for gi, g in enumerate(g1) if Xavg[st.index("AL-L"), gi, iL] > 1e-6 and
          Xavg[st.index("AL-L"), gi, iR] < 0.1 * Xavg[st.index("AL-L"), gi, iL]]
    # pass 2
    up, dn = d["up_tavg_mean"], d["dn_tavg_mean"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rd = np.abs(up - dn) / np.maximum(np.maximum(up, dn), 1e-300)
    def ok_(res_, rj):                                        # comparable: converged or clearly non-stationary, not critical
        return ((res_ < 1e-7) | (res_ > 1e-5)) & (np.abs(rj - 1) > 0.01)
    comp_ok = ok_(d["up_resid"], d["up_rhoJ"]) & ok_(d["dn_resid"], d["dn_rhoJ"]) & (np.maximum(up, dn) >= 1e-6)
    rd[~comp_ok] = 0
    big = rd > 0.05
    runs = []; i = 0
    while i < len(g2):
        if big[i]:
            j = i
            while j + 1 < len(g2) and big[j + 1]:
                j += 1
            if j > i:
                seg = slice(max(i - 1, 0), min(j + 2, len(g2)))
                jump_up = float(np.max(np.abs(np.diff(np.log(np.maximum(up[seg], 1e-12))))))
                jump_dn = float(np.max(np.abs(np.diff(np.log(np.maximum(dn[seg], 1e-12))))))
                runs.append({"g_lo": float(g2[i]), "g_hi": float(g2[j]), "n_points": int(j - i + 1), "max_rel_diff": float(rd[i:j + 1].max()),
                             "max_dlnrho_step_up": jump_up, "max_dlnrho_step_down": jump_dn,
                             "up_oscillating_frac": float((d["up_resid"][i:j + 1] > 1e-5).mean()),
                             "down_oscillating_frac": float((d["dn_resid"][i:j + 1] > 1e-5).mean()),
                             "n_type_A_both_fixed_points": int(sum(1 for q_ in range(i, j + 1) if d["up_resid"][q_] < 1e-7 and d["dn_resid"][q_] < 1e-7
                                                                   and d["up_rhoJ"][q_] < 1 and d["dn_rhoJ"][q_] < 1)),
                             "n_type_B_fixed_point_vs_nonstationary": int(sum(1 for q_ in range(i, j + 1) if (d["up_resid"][q_] < 1e-7) != (d["dn_resid"][q_] < 1e-7))),
                             "n_type_C_both_nonstationary": int(sum(1 for q_ in range(i, j + 1) if d["up_resid"][q_] > 1e-5 and d["dn_resid"][q_] > 1e-5)),
                             "rhoJ_up_range": [float(d["up_rhoJ"][i:j + 1].min()), float(d["up_rhoJ"][i:j + 1].max())],
                             "rhoJ_down_range": [float(d["dn_rhoJ"][i:j + 1].min()), float(d["dn_rhoJ"][i:j + 1].max())]})
            i = j + 1
        else:
            i += 1
    step_up = np.abs(np.diff(np.log(np.maximum(up, 1e-12)))); step_dn = np.abs(np.diff(np.log(np.maximum(dn, 1e-12))))
    oscA = (d["p1_resid"][0] > 1e-5) & (d["p1_period"][0] <= 512)
    osc_range = [float(g1[oscA].min()), float(g1[oscA].max())] if oscA.any() else None
    up_osc = d["up_resid"] > 1e-5; dn_osc = d["dn_resid"] > 1e-5
    # SL edge and D
    g_sl_up = first_cross(g2, (up > 1e-8).astype(float), 0.5)
    sl_idx = [gi for gi in range(len(g1)) if d["p1_mean"][0, gi] == 0]
    # neurons oscillating (all-active start)
    sd = d["all_active_std"].astype(np.float32)
    nosc = (sd > 1e-3).sum(1)
    gi_rep = int(np.argmax(nosc)) if nosc.max() > 0 else None
    comp = None
    if gi_rep is not None:
        sel = sd[gi_rep] > 1e-3
        mm = np.bincount(memb[sel], minlength=M); ss = {}
        for s_, c_ in zip(*np.unique(sup[sel], return_counts=True)):
            ss[str(s_) or "?"] = int(c_)
        comp = {"g": float(g1[gi_rep]), "n": int(sel.sum()), "by_module": {names[k]: int(mm[k]) for k in np.argsort(-mm)[:6] if mm[k]},
                "by_superclass": dict(sorted(ss.items(), key=lambda x: -x[1])[:6])}
    p3_osc = d["p3_resid"] > 1e-5 / 4                         # dt = 0.05: per-step changes are ~4x smaller
    res["per_r"][f"{r:g}"] = {
        "census": census, "g_multistable_fixed_points": multi_g, "MT3_multi": bool(multi_g),
        "max_n_fixed_point_attractors": int(max(n_fp)), "max_n_oscillatory_states": int(max(n_osc)),
        "g_B1_like_AL_L_start": b1,
        "hysteresis_runs": runs, "MT3_first_order": bool(any(rr_["n_points"] >= 2 for rr_ in runs)),
        "MT3_first_order_fixed_points_only": bool(any(rr_["n_type_A_both_fixed_points"] >= 2 for rr_ in runs)),
        "hysteresis_points_by_type": {"A_both_fixed_points": int(sum(rr_["n_type_A_both_fixed_points"] for rr_ in runs)),
                                      "B_fixed_point_vs_nonstationary": int(sum(rr_["n_type_B_fixed_point_vs_nonstationary"] for rr_ in runs)),
                                      "C_both_nonstationary": int(sum(rr_["n_type_C_both_nonstationary"] for rr_ in runs))},
        "max_rel_up_down_diff": float(rd.max()), "g_at_max_rel_diff": float(g2[np.argmax(rd)]),
        "n_g2_points_excluded_from_hysteresis_test": int((~comp_ok).sum()),
        "max_dlnrho_step_up": float(step_up.max()), "max_dlnrho_step_down": float(step_dn.max()),
        "oscillation_range_all_active_pass1": osc_range,
        "nonstationary_g_all_active_pass1": [float(x) for x in g1[oscA]],
        "n_g_oscillating_pass1_all_active": int(oscA.sum()),
        "up_oscillating_g_range": [float(g2[up_osc].min()), float(g2[up_osc].max())] if up_osc.any() else None,
        "down_oscillating_g_range": [float(g2[dn_osc].min()), float(g2[dn_osc].max())] if dn_osc.any() else None,
        "pass3_dt0.05_oscillating_g": [float(x) for x in g1[p3_osc]],
        "pass3_dt0.05_vs_dt0.2_oscillating_agree": int((p3_osc == oscA).sum()), "n_g": int(len(g1)),
        "pass3_amp_vs_pass1_amp": [[float(g1[i]), float(d["p1_amp"][0, i]), float(d["p3_amp"][i]), float(d["p1_period"][0, i]),
                                    float(d["p3_period"][i])] for i in range(len(g1)) if oscA[i] or p3_osc[i]],
        "n_neurons_oscillating_all_active": nosc.tolist(), "oscillating_composition_at_max": comp,
        "SL_edge_up_sweep": g_sl_up, "silent_all_active_g": [float(g1[i]) for i in sl_idx],
        "D_up_PR_quarter": first_cross(g2, d["up_pr"], 0.25), "D_down_PR_quarter": first_cross(g2, d["dn_pr"], 0.25),
        "D_pass1_all_active_PR_quarter": first_cross(g1, d["p1_pr"][0], 0.25),
        "rhoJ_up_max_below_osc": float(d["up_rhoJ"][~up_osc].max()) if (~up_osc).any() else None}
    pr = res["per_r"][f"{r:g}"]
    print(f"r={r:g}: MT3_multi={pr['MT3_multi']} (max fp attractors {pr['max_n_fixed_point_attractors']}), "
          f"hysteresis runs {len(runs)}, max up/down rel diff {pr['max_rel_up_down_diff']:.3g} at g={pr['g_at_max_rel_diff']:.3g}; "
          f"osc pass1 {osc_range}; SL edge {g_sl_up}; D up/down {pr['D_up_PR_quarter']}/{pr['D_down_PR_quarter']}; "
          f"B1-like {b1[:3]}")
R = sorted(D)
res["summary"] = {"r": R, "MT3_multi_any": any(res["per_r"][f"{r:g}"]["MT3_multi"] for r in R),
                  "MT3_first_order_any": any(res["per_r"][f"{r:g}"]["MT3_first_order"] for r in R),
                  "B1_like_any": any(res["per_r"][f"{r:g}"]["g_B1_like_AL_L_start"] for r in R)}
json.dump(res, open(f"{ROOT}/results/pd3/module_theory/qmf_analysis.json", "w"), indent=1)
print(json.dumps(res["summary"]))

# ---------------- figure ----------------
show = [r for r in [0.0, 1.0, 4.0] if r in D]
fig = plt.figure(figsize=(15, 11.5), dpi=130)
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05], hspace=0.42, wspace=0.28)
mnames = [names[k] for k in order]
norm = LogNorm(1e-6, 1)
for col, r in enumerate(show):
    d = D[r]
    for row, (br, key, rk) in enumerate([("up-sweep (from silence, +1e-6 per step in g)", "up_tavg_X", "up_resid"),
                                         ("down-sweep (from all active at g = 30)", "dn_tavg_X", "dn_resid")]):
        ax = fig.add_subplot(gs[row, col])
        Z = np.maximum(d[key][:, order].T, 1e-7)
        im = ax.pcolormesh(d["g2"], np.arange(M), Z, norm=norm, cmap="viridis", shading="nearest")
        oscm = d[rk] > 1e-5
        if oscm.any():
            ax.scatter(d["g2"][oscm], np.full(oscm.sum(), M + 0.2), marker="|", s=18, color=OK[5], clip_on=False)
        ax.axvline(0.1632, color="w", lw=1.2, ls="-"); ax.set_xscale("log")
        ax.set_yticks(np.arange(M)); ax.set_yticklabels(mnames if col == 0 else [], fontsize=5.5)
        ax.set_ylim(-0.5, M + 0.6)
        ax.set_title(f"r = {r:g}: {br}", fontsize=9)
        if row == 1:
            ax.set_xlabel("coupling g")
        if col == 0:
            ax.set_ylabel("P0 module (ordered by isolated g_lin,k)", fontsize=8)
cb = fig.colorbar(im, ax=fig.axes[:6], fraction=0.015, pad=0.01); cb.set_label("module mean activity X_k (QMF)", fontsize=9)
fig.text(0.13, 0.905, "white line: g_lin = 0.1632; vermilion ticks above a strip: g where that branch does not converge "
         "(oscillating QMF; strip shows the time average)", fontsize=8.5)
# (g, r) attractor map
ax = fig.add_subplot(gs[2, 0])
g1 = D[R[0]]["g1"]
nfp = np.array([[c["n_fixed_point_attractors"] for c in res["per_r"][f"{r:g}"]["census"]] for r in R])
nos = np.array([[c["n_oscillatory_states"] for c in res["per_r"][f"{r:g}"]["census"]] for r in R])
cmap = ListedColormap(["#ffffff", "#c6dbef", "#2171b5", "#08306b"]); bn = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
yy = np.arange(len(R))
ax.pcolormesh(g1, yy, np.minimum(nfp, 3), cmap=cmap, norm=bn, shading="nearest")
oy, ox = np.where(nos > 0)
ax.scatter(g1[ox], yy[oy], s=9, marker="x", color=OK[5], lw=0.8, label="no fixed point reached (oscillating)")
m2y, m2x = np.where(nos > 1)
if len(m2y):
    ax.scatter(g1[m2x], yy[m2y], s=26, facecolors="none", edgecolors=OK[0], lw=0.8, label=">1 distinct oscillatory time average")
ax.set_xscale("log"); ax.set_yticks(yy); ax.set_yticklabels([f"{r:g}" for r in R]); ax.set_xlabel("coupling g"); ax.set_ylabel("inhibition r")
ax.axvline(0.1632, color=OK[0], lw=1)
ax.set_title("QMF from 32 starts: distinct stable fixed points\n(white 0, light 1, blue 2, navy >= 3)", fontsize=9)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, framealpha=0.9)
# rho(J) along branches
ax = fig.add_subplot(gs[2, 1])
for i, r in enumerate(show):
    d = D[r]
    ax.plot(d["g2"], d["up_rhoJ"], color=OK[[4, 5, 3][i]], lw=1.5, label=f"r = {r:g} up")
    ax.plot(d["g2"], d["dn_rhoJ"], color=OK[[4, 5, 3][i]], lw=1.2, ls="--", label=f"r = {r:g} down")
ax.axhline(1, color="k", lw=0.8); ax.axvline(0.1632, color="0.5", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("coupling g"); ax.set_ylabel("rho(J) of the linearised QMF map")
ax.set_ylim(0.8, max(1.6, min(3, max(np.nanmax(D[r]["up_rhoJ"]) for r in show) + 0.05)))
ax.set_title("growth rate along the branches\n(rho(J) > 1: not a stable fixed point)", fontsize=9); ax.legend(fontsize=7, ncol=2)
# up vs down mean activity
ax = fig.add_subplot(gs[2, 2])
for i, r in enumerate(show):
    d = D[r]
    ax.plot(d["g2"], np.maximum(d["up_tavg_mean"], 1e-10), color=OK[[4, 5, 3][i]], lw=1.5, label=f"r = {r:g} up")
    ax.plot(d["g2"], np.maximum(d["dn_tavg_mean"], 1e-10), color=OK[[4, 5, 3][i]], lw=1.2, ls="--", label=f"r = {r:g} down")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-6, 0.3); ax.axvline(0.1632, color="0.5", lw=0.8)
ax.set_xlabel("coupling g"); ax.set_ylabel("mean activity (time average)"); ax.legend(fontsize=7, ncol=2)
ax.set_title("up- vs down-sweep: hysteresis test", fontsize=9)
fig.suptitle("MT3: neuron-level quenched mean field of the full MaleCNS (167,216 neurons), P0 modules", fontsize=12, y=0.95)
fig.savefig(f"{ROOT}/figs/pd3/modules_qmf_phase.png", bbox_inches="tight")
print("saved figs/pd3/modules_qmf_phase.png")
