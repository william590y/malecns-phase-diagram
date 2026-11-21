"""E3 theory input (GPU): right/left Perron vectors of the branching operator B(g) = f(g E) (excitatory edges, raw norm)
at the g values of the existing dense theory rows (results/pd/branch_real_dense_part*.json, which hold lam, K_r, logtau_r),
so that the depression-extended birth-death chain (code/ald_dep_theory.py) can weight per-neuron depression factors by the
mode. Same operator and power iteration as code/theory_branch_gpu.py (copied, not imported: that file runs on import).
Stores for every g: lam, and the top entries of the mode (largest u_j v_j covering >= 99.99 % of <u, v> = 1, at most 20,000):
idx, u, v (u normalised so that <u, v> = 1, v with unit 2-norm, the convention of theory_branch_gpu.py), plus <u^2, v>, <u, v^2>.
Output: results/aldef/depression/theory_vecs.npz
"""
import glob, json, time
import numpy as np
import scipy.sparse as sp
import torch

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
dev = "cuda"; t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:.0f}s]", *a, flush=True)
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
dt = 0.2; a_eff = (1 - np.exp(-dt)) / dt
exc = sign[pre] > 0
Ex = sp.csr_matrix((c * w[exc], (post[exc], pre[exc])), shape=(N, N)); Ex.sum_duplicates(); Ex.sort_indices()
ExT = Ex.T.tocsr(); ExT.sort_indices()
T = lambda a, dtype=torch.float64: torch.tensor(a, device=dev, dtype=dtype)
crow, col, base = T(Ex.indptr, torch.int64), T(Ex.indices, torch.int64), T(Ex.data)
Ecoo = Ex.tocoo(); order = np.lexsort((Ecoo.row, Ecoo.col))
assert np.allclose(ExT.data, Ex.data[order]) and (ExT.indices == Ecoo.row[order]).all()
permT = T(order, torch.int64); crowT, colT = T(ExT.indptr, torch.int64), T(ExT.indices, torch.int64)
f = lambda x: (1 - torch.exp(-torch.tanh(torch.clamp(x, min=0)) * dt)) / dt


def mats(g):
    vals = f(g * base)
    return (torch.sparse_csr_tensor(crow, col, vals, (N, N)), torch.sparse_csr_tensor(crowT, colT, vals[permT], (N, N)))


def perron(M, x=None, iters=20000, tol=1e-11):
    x = torch.ones(N, 1, device=dev, dtype=torch.float64) if x is None else x.clone()
    x /= x.norm(); lam = 0.0
    for it in range(iters):
        y = torch.sparse.mm(M, x); lam_new = float(y.norm() / x.norm())
        x = x + y / lam_new; x /= x.norm()
        if it > 50 and abs(lam_new - lam) < tol * lam_new:
            break
        lam = lam_new
    return lam_new, x


rows = []
for p in sorted(glob.glob(f"{ROOT}/results/pd/branch_real_dense_part*.json")):
    rows += json.load(open(p))["rows"]
rows.sort(key=lambda x: x["g"])
gs = np.array([x["g"] for x in rows])
out = {"g": gs, "lam": [], "u2v": [], "uv2": [], "idx": [], "u": [], "v": [], "cover": []}
v = None
for i, g in enumerate(gs):
    B, BT = mats(g)
    lam, v = perron(B, v)
    _, u = perron(BT, v)
    u = u / float((u * v).sum())
    uu, vv = u.cpu().numpy().ravel(), v.cpu().numpy().ravel()
    uv = uu * vv
    o = np.argsort(-uv); cs = np.cumsum(uv[o])
    k = min(int(np.searchsorted(cs, 0.9999)) + 1, 20000)
    sel = o[:k]
    out["lam"].append(lam); out["u2v"].append(float((uu ** 2 * vv).sum())); out["uv2"].append(float((uu * vv ** 2).sum()))
    out["idx"].append(sel.astype(np.int64)); out["u"].append(uu[sel]); out["v"].append(vv[sel]); out["cover"].append(float(cs[k - 1]))
    log(f"g {g:.4f} lam {lam:.4f} (json {rows[i]['lam']:.4f}) u2v {out['u2v'][-1]:.5f} (json {rows[i]['u2v']:.5f}) support {k}")
L = max(len(x) for x in out["idx"])
pad = lambda arrs, fill: np.array([np.r_[a, np.full(L - len(a), fill)] for a in arrs])
np.savez_compressed(f"{ROOT}/results/aldef/depression/theory_vecs.npz", g=gs, lam=np.array(out["lam"]),
                    u2v=np.array(out["u2v"]), uv2=np.array(out["uv2"]), idx=pad(out["idx"], -1), u=pad(out["u"], 0.0),
                    v=pad(out["v"], 0.0), n=np.array([len(x) for x in out["idx"]]), cover=np.array(out["cover"]),
                    lam_json=np.array([x["lam"] for x in rows]), alpha_eff=a_eff)
log("saved")
