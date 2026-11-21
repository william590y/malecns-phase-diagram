"""Q3 cross-check: the order in which regions actually switch on in the simulations, from the per-neuron occupancy
(occ[g, N] = activity averaged over replicas and the second half of the run, T = 3000, h = 0, fully active start) of the
existing r = 0 and r = 1 sweeps, compared with the spectral order (tail modes, g_on ~ alpha_eff / lambda_k).
Writes results/al_lit/grounding/recruit.json.
"""
import json
import numpy as np
from al_common import labels, ROOT, OUT

L = labels(); N = L["N"]
t = np.load(f"{ROOT}/results/pd/tail_real_r0_dir.npz", allow_pickle=True)
supports = {}
for k in range(20):
    p = t["P"][:, k].astype(np.float64); p /= p.sum(); o = np.argsort(-p)
    supports[f"mode{k+1}"] = o[: np.searchsorted(np.cumsum(p[o]), 0.9) + 1]
FILES = {0.0: ["pd/decay_r0_real.npz", "pd2/onset_real_r0.npz", "pd2/deloc_real_r0.npz"],
         1.0: ["pd/decay_r1_real.npz", "pd2/onset_real_r1.npz", "pd2/deloc_real_r1.npz"]}
regions = ["AL", "LH", "MB", "CB other", "CX", "VNC", "AN/DN", "OL intrinsic", "VPN/VCN", "JO/mechano (brain)"]
out = {}
for r, files in FILES.items():
    G, O = [], []
    for f in files:
        try:
            z = np.load(f"{ROOT}/results/{f}", allow_pickle=True)
        except FileNotFoundError:
            continue
        G += list(np.round(z["g"], 4)); O += list(z["occ"].astype(np.float32))
    G = np.array(G); O = np.array(O)
    ug = np.unique(G)
    occ = np.array([O[G == g].mean(0) for g in ug])          # average duplicates
    res = {"g": ug.tolist(), "region_occ": {}, "mode_occ": {}, "first_g_occ_ge_0.05": {}, "first_g_mode_ge_0.05": {}}
    for rg in regions:
        m = L["reg"] == rg
        y = occ[:, m].mean(1)
        res["region_occ"][rg] = y.tolist()
        hit = np.where(y >= 0.05)[0]
        res["first_g_occ_ge_0.05"][rg] = float(ug[hit[0]]) if len(hit) else None
    for k, s in supports.items():
        y = occ[:, s].mean(1)
        res["mode_occ"][k] = y.tolist()
        hit = np.where(y >= 0.05)[0]
        res["first_g_mode_ge_0.05"][k] = float(ug[hit[0]]) if len(hit) else None
    tot = occ.mean(1)
    res["total_occ"] = tot.tolist()
    # share of all activity carried by each region at a few g
    res["activity_share"] = {}
    for g in [0.3, 0.5, 0.8, 1.2, 2.0]:
        i = np.argmin(np.abs(ug - g))
        a = occ[i]
        if a.sum() > 0:
            res["activity_share"][f"{ug[i]:.3f}"] = {rg: float(a[L["reg"] == rg].sum() / a.sum()) for rg in regions}
    out[str(r)] = res
    print(f"r={r}: first g with mean occupancy >= 0.05 by region:",
          dict(sorted(((k, v) for k, v in res["first_g_occ_ge_0.05"].items() if v), key=lambda kv: kv[1])))
    print("   by tail-mode support:", dict(sorted(((k, v) for k, v in res["first_g_mode_ge_0.05"].items() if v),
                                                key=lambda kv: kv[1])))
    for g, sh in res["activity_share"].items():
        print("   share at g", g, {k: round(v, 3) for k, v in sh.items() if v > 0.01})
json.dump(out, open(f"{OUT}/recruit.json", "w"))
