"""pd3 module_theory MT3b (added after MT3 found QMF multistability at r >= 0.5): what differs between coexisting QMF fixed points?

For every r and pass-1 g at which MT3 found >= 2 distinct converged stable fixed points (results/pd3/module_theory/
qmf_analysis.json; 34-module vectors differing by > 10% relative L1), the representative start of each distinct attractor
(up to 4 per g, the first start in pass-1 order of each cluster) is re-run with exactly pd3_qmf_ic.py's map, start and
convergence policy (float32 phase to 1e-7, float64 polish to 1e-9) and the full 167,216-neuron vectors are kept. For every
pair of attractors (A, B) at the same (r, g):
  n_diff(>0.05), n_diff(>0.01): neurons with |rho_A - rho_B| above the threshold; their P0 modules, superclasses, signs
  (exc / inh / modulatory); the connected components of the undirected synaptic graph (E + I, both directions) induced on the
  neurons with |d rho| > 0.05: count and the largest component (one large switching region vs many local switches);
  total activity difference sum_i (rho_A - rho_B) in neurons.
Sanity: the recomputed mean activity must equal the pass-1 value (stored p1_mean) to 1e-6 relative.
Outputs: results/pd3/module_theory/qmf_multi.json, qmf_multi_vectors.npz
Usage: python pd3_qmf_multi.py
"""
import json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch

C = "/MALECNS_CACHE"; ROOT = "/MALECNS_ROOT"
dev = "cuda"; t0 = time.time(); dt = 0.2; decay = float(np.exp(-dt)); DT = torch.float64; F32 = torch.float32
log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
sup = np.load(C + "/superclasses.npy", allow_pickle=True)
ins = np.bincount(post, weights=w, minlength=N); c = 1.0 / ins[ins > 0].mean()
nz = sign[pre] != 0
A = sp.csr_matrix((np.ones(int(nz.sum())), (post[nz], pre[nz])), shape=(N, N)); A = ((A + A.T) > 0).astype(np.int8).tocsr()
part = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz", allow_pickle=True)
memb = part["membership"].astype(np.int64); names = [str(x) for x in part["names"]]; MOD = len(names)
AL = np.isin(memb, [0, 1, 2])


def start_vec(nm):
    if nm == "all":
        return np.ones(N)
    if nm == "AL":
        return AL * 1.0
    if nm in ("AL-L", "AL-R"):
        return (memb == names.index(nm)) * 1.0
    if nm == "rest_only":
        return (~AL) * 1.0
    if nm.startswith("isl"):
        return (memb == names.index(nm)) * 1.0
    if nm.startswith("half"):
        i = int(nm[4:]); sel = np.sort(np.random.default_rng(2026 + i).choice(MOD, MOD // 2, replace=False))
        return np.isin(memb, sel) * 1.0
    if nm == "AL+isl1":
        return (AL | (memb == names.index("isl1"))) * 1.0
    if nm == "uniform1e-3":
        return np.full(N, 1e-3)
    if nm == "AL@0.05":
        return AL * 0.05
    raise ValueError(nm)


def rel_l1(a, b):
    return float(np.abs(a - b).sum() / max(np.abs(a).sum(), np.abs(b).sum(), 1e-300))


ana = json.load(open(f"{ROOT}/results/pd3/module_theory/qmf_analysis.json"))
out = {"per_r": {}}; vecs = {}
for rk, pr in ana["per_r"].items():
    if not pr["g_multistable_fixed_points"]:
        continue
    r = float(rk); d = np.load(f"{ROOT}/results/pd3/module_theory/qmf_r{rk}.npz", allow_pickle=True)
    st = [str(x) for x in d["starts"]]; g1 = d["g1"]
    conv = (d["p1_resid"] < 1e-7) & (d["p1_resid"] / np.maximum(1 - d["p1_rhoJ"], 1e-6) < 1e-6); stab = d["p1_rhoJ"] < 1
    Xfp = d["p1_X"].copy(); Xfp[d["p1_mean"] < 1e-8] = 0
    jobs = []                                                   # (g index, [start indices of cluster representatives])
    for gi, g in enumerate(g1):
        if float(g) not in [round(x, 12) for x in pr["g_multistable_fixed_points"]] and not any(abs(g - x) < 1e-9 for x in pr["g_multistable_fixed_points"]):
            continue
        reps = []
        for s in range(len(st)):
            if conv[s, gi] and stab[s, gi] and not any(rel_l1(Xfp[s, gi], Xfp[t, gi]) <= 0.1 for t in reps):
                reps.append(s)
        jobs.append((gi, reps[:4]))
    s = np.where(sign[pre] > 0, 1.0, np.where(sign[pre] < 0, -r, 0.0)); keep = s != 0
    idx = torch.tensor(np.vstack([post[keep], pre[keep]]), device=dev)
    W64 = torch.sparse_coo_tensor(idx, torch.tensor(s[keep] * w[keep] * c, device=dev, dtype=DT), (N, N)).coalesce().to_sparse_csr()
    W = W64.to(F32); del idx
    cols = [(gi, si) for gi, reps in jobs for si in reps]
    X = torch.tensor(np.array([start_vec(st[si]) for _, si in cols]).T, device=dev, dtype=DT)
    gv = torch.tensor([g1[gi] for gi, _ in cols], device=dev, dtype=F32)[None, :]
    live = torch.arange(len(cols), device=dev); steps = 0
    while steps < 25000 and len(live):
        Xl = X[:, live]; gl = gv[:, live]
        for _ in range(500):
            Xn = Xl * decay + (1 - Xl) * (-torch.expm1(-torch.tanh((torch.sparse.mm(W, Xl.float()) * gl).clamp(min=0)) * dt)).double()
            res = (Xn - Xl).abs().amax(0); Xl = Xn
        X[:, live] = Xl; steps += 500; live = live[res >= 1e-7]
    for _ in range(1000):                                        # float64 polish
        Xn = X * decay + (1 - X) * (-torch.expm1(-torch.tanh((torch.sparse.mm(W64, X) * gv.double()).clamp(min=0)) * dt))
        res = (Xn - X).abs().amax(0); X = Xn
        if (res < 1e-9).all():
            break
    Xc = X.cpu().numpy(); resid = res.cpu().numpy()
    log(f"r={rk}: {len(cols)} columns recomputed at {len(jobs)} g, max resid {resid.max():.2e}")
    rows = []
    for gi, reps in jobs:
        ci = [k for k, (g_, s_) in enumerate(cols) if g_ == gi]
        for k in ci:
            vecs[f"r{rk}_g{g1[gi]:.4f}_{st[cols[k][1]]}"] = Xc[:, k].astype(np.float32)
        for a_ in range(len(ci)):
            for b_ in range(a_ + 1, len(ci)):
                ka, kb = ci[a_], ci[b_]; xa, xb = Xc[:, ka], Xc[:, kb]; D = np.abs(xa - xb)
                sel = D > 0.05; sel01 = D > 0.01
                ncomp, lab = connected_components(A[sel][:, sel], directed=False) if sel.any() else (0, np.array([]))
                comp_sizes = np.bincount(lab) if sel.any() else np.array([0])
                mods = np.bincount(memb[sel], minlength=MOD); scs = {}
                for s_, n_ in zip(*np.unique(sup[sel], return_counts=True)):
                    scs[str(s_) or "?"] = int(n_)
                rows.append({"g": float(g1[gi]), "A": st[cols[ka][1]], "B": st[cols[kb][1]],
                             "mean_A": float(xa.mean()), "mean_B": float(xb.mean()),
                             "mean_check_A": float(abs(xa.mean() / d["p1_mean"][cols[ka][1], gi] - 1)),
                             "mean_check_B": float(abs(xb.mean() / d["p1_mean"][cols[kb][1], gi] - 1)),
                             "resid_A": float(resid[ka]), "resid_B": float(resid[kb]),
                             "n_diff_0.05": int(sel.sum()), "n_diff_0.01": int(sel01.sum()), "sum_diff_neurons": float((xa - xb).sum()),
                             "max_abs_diff": float(D.max()),
                             "n_components_0.05": int(ncomp), "largest_component_0.05": int(comp_sizes.max()),
                             "component_sizes_top10": sorted(comp_sizes.tolist(), reverse=True)[:10],
                             "by_module": {names[k]: int(mods[k]) for k in np.argsort(-mods)[:8] if mods[k]},
                             "by_superclass": dict(sorted(scs.items(), key=lambda x: -x[1])[:6]),
                             "by_sign": {"exc": int((sign[sel] > 0).sum()), "inh": int((sign[sel] < 0).sum()), "mod": int((sign[sel] == 0).sum())},
                             "A_higher_count": int(((xa - xb) > 0.05).sum()), "B_higher_count": int(((xb - xa) > 0.05).sum())})
    out["per_r"][rk] = {"n_g": len(jobs), "pairs": rows,
                        "median_n_diff_0.05": float(np.median([x["n_diff_0.05"] for x in rows])) if rows else None,
                        "max_n_diff_0.05": int(max(x["n_diff_0.05"] for x in rows)) if rows else None,
                        "max_mean_check": float(max(max(x["mean_check_A"], x["mean_check_B"]) for x in rows)) if rows else None}
    log(f"r={rk}: pairs {len(rows)}, n_diff(>0.05) median {out['per_r'][rk]['median_n_diff_0.05']}, max {out['per_r'][rk]['max_n_diff_0.05']}, "
        f"mean check {out['per_r'][rk]['max_mean_check']:.2e}")
    del W, W64, X; torch.cuda.empty_cache()
json.dump(out, open(f"{ROOT}/results/pd3/module_theory/qmf_multi.json", "w"), indent=1)
np.savez_compressed(f"{ROOT}/results/pd3/module_theory/qmf_multi_vectors.npz", **vecs)
log("saved qmf_multi.json")
