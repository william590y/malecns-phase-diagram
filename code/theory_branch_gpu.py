"""GPU version of theory_branch.py (branching variant), parallel over the coupling grid as a Slurm array.

Same theory, same definitions (see theory_branch.py): branching operator B_ij(g) = f(g E_ij) on excitatory edges, Perron
vectors by shifted power iteration on the GPU, coincidence term D(g, r) evaluated for every r at once per neuron, K_eff and the
exact birth-death extinction time. Task k of n takes grid points k, k+n, k+2n, ... of g = g_lin * logspace(0, log10(FMAX), NG).
Usage: theory_branch_gpu.py <kind> <r,r,...> <FMAX> <NG> <task> <ntasks> <suffix>
       -> results/pd/branch_<kind><suffix>_part<task>.json ; merge with theory_branch_merge.py
"""
import json, sys, time
import numpy as np
import scipy.sparse as sp
import torch
from scipy.special import logsumexp

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
kind, RS, FMAX, NG, TASK, NT, SUF = (sys.argv[1], [float(x) for x in sys.argv[2].split(",")], float(sys.argv[3]),
                                     int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
TS = [3000.0, 8000.0, 30000.0]
dev = "cuda"; t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:.0f}s]", *a, flush=True)
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
rng = np.random.default_rng(0)
if kind == "null":
    post = post[rng.permutation(len(post))]
elif kind == "wnull":
    w = w[rng.permutation(len(w))]
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
dt = 0.2; a_eff = (1 - np.exp(-dt)) / dt
exc, inh = sign[pre] > 0, sign[pre] < 0
Ex = sp.csr_matrix((c * w[exc], (post[exc], pre[exc])), shape=(N, N)); Ex.sum_duplicates(); Ex.sort_indices()
Ih = sp.csr_matrix((c * w[inh], (post[inh], pre[inh])), shape=(N, N)); Ih.sum_duplicates(); Ih.sort_indices()
ExT = Ex.T.tocsr(); ExT.sort_indices()
T = lambda a, dtype=torch.float64: torch.tensor(a, device=dev, dtype=dtype)
crow, col, base = T(Ex.indptr, torch.int64), T(Ex.indices, torch.int64), T(Ex.data)
# map each transposed entry to its value in Ex (to build B^T from the same values)
Ecoo = Ex.tocoo(); order = np.lexsort((Ecoo.row, Ecoo.col))           # sort by (col, row) = CSR order of Ex^T
assert np.allclose(ExT.data, Ex.data[order]) and (ExT.indices == Ecoo.row[order]).all()
permT = T(order, torch.int64); crowT, colT = T(ExT.indptr, torch.int64), T(ExT.indices, torch.int64)
f = lambda x: (1 - torch.exp(-torch.tanh(torch.clamp(x, min=0)) * dt)) / dt


def mats(g):
    vals = f(g * base)
    B = torch.sparse_csr_tensor(crow, col, vals, (N, N))
    BT = torch.sparse_csr_tensor(crowT, colT, vals[permT], (N, N))
    return B, BT


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


lamE, _ = perron(torch.sparse_csr_tensor(crow, col, base, (N, N)))
lo, hi = a_eff / lamE * 0.999, a_eff / lamE * 6; v0 = None
for _ in range(18):
    mid = np.sqrt(lo * hi); lam, v0 = perron(mats(mid)[0], v0)
    lo, hi = (mid, hi) if lam < a_eff else (lo, mid)
g_lin = float(np.sqrt(lo * hi))
log(kind, "lambda_PF(E)", lamE, "g_lin", g_lin, "task", TASK, "/", NT)
grid = g_lin * np.logspace(0, np.log10(FMAX), NG)[1:]
mine = grid[TASK::NT]
Rt = T(RS)


def coincidence(g, u, v):
    """D(g, r) for every r at once."""
    u = u.cpu().numpy().ravel(); v = v.cpu().numpy().ravel()
    ou = np.argsort(-u); I_set = ou[: min(np.searchsorted(np.cumsum(u[ou]) / u.sum(), 0.99) + 1, 20000)]
    ov = np.argsort(-v); vmask = np.zeros(N, bool); vmask[ov[: np.searchsorted(np.cumsum(v[ov]) / v.sum(), 0.99) + 1]] = True
    D = torch.zeros(len(RS), device=dev, dtype=torch.float64)
    for i in I_set:
        je = Ex.indices[Ex.indptr[i]:Ex.indptr[i + 1]]; xe = Ex.data[Ex.indptr[i]:Ex.indptr[i + 1]]
        m = vmask[je]; je, xe = je[m], xe[m]
        if len(je) == 0:
            continue
        ki = Ih.indices[Ih.indptr[i]:Ih.indptr[i + 1]]; yi = Ih.data[Ih.indptr[i]:Ih.indptr[i + 1]]
        m = vmask[ki]; ki, yi = ki[m], yi[m]
        xe_t, ve_t = T(xe), T(v[je]); fe = f(g * xe_t)
        t = torch.zeros(len(RS), device=dev, dtype=torch.float64)
        if len(ki):
            yi_t, vi_t = T(yi), T(v[ki])
            dei = f(g * xe_t[None, :, None] - g * Rt[:, None, None] * yi_t[None, None, :]) - fe[None, :, None]
            t -= (dei * ve_t[None, :, None] * vi_t[None, None, :]).sum((1, 2))
        if len(je) > 1:
            dee = f(g * (xe_t[:, None] + xe_t[None, :])) - fe[:, None] - fe[None, :]
            dee.fill_diagonal_(0.0)
            t -= 0.5 * (dee * (ve_t[:, None] * ve_t[None, :])).sum()
        D += u[i] * t
    return D.cpu().numpy()


def log_tau(lam, K):
    n_top = int(np.floor(K))
    if n_top < 1 or lam <= 0:
        return -np.inf
    n = np.arange(1, n_top + 1, dtype=float)
    lb = lam * n * (1 - n / K); lb[-1] = 0.0
    ld = a_eff * n
    logD = np.empty(n_top); logD[-1] = -np.log(ld[-1])
    for k in range(n_top - 2, -1, -1):
        logD[k] = np.logaddexp(0.0, np.log(lb[k]) + logD[k + 1]) - np.log(ld[k])
    return float(logsumexp(logD))


rows = []
v = v0
for g in mine:
    B, BT = mats(g)
    lam, v = perron(B, v)
    _, u = perron(BT, v)
    u = u / float((u * v).sum())
    base_ = float(((u ** 2) * v).sum()); uv2 = float((u * v ** 2).sum())
    D = coincidence(g, u, v)
    row = {"g": float(g), "lam": lam, "R": lam / a_eff, "uv2": uv2, "u2v": base_}
    for r, d in zip(RS, D):
        K = 1 / (base_ * (uv2 + d / lam))
        row[f"D_r{r:g}"] = float(d); row[f"K_r{r:g}"] = float(K); row[f"logtau_r{r:g}"] = log_tau(lam, K)
    rows.append(row)
    log(f"g {g:.4f} R {lam/a_eff:.3f}", " ".join(f"r{r:g}:K{row[f'K_r{r:g}']:.0f}" for r in RS[::4]))
json.dump({"kind": kind, "g_lin": g_lin, "lambda_PF_E": lamE, "alpha_eff": a_eff, "r": RS, "rows": rows},
          open(f"{ROOT}/results/pd/branch_{kind}{SUF}_part{TASK}.json", "w"))
log("saved")
