"""pd3 module_theory MT5: is any spectral island 'a second AL'? Stochastic Wilson-Cowan on single P0 modules ALONE (GPU).

module_sim.py convention: internal synapses of the module only (the rest of the brain silent), full-brain scaling
c = 1/mean in-strength, synchronous dt = 0.2, rates as swc.py (off 1 - e^{-dt}; on 1 - exp(-tanh(s)_+ dt)), h = 0.
Start fully active, B = 64 replicas, T = 3000; the exact extinction step of every replica is recorded (-1 = alive at T).
g at 24 ln points in [1, 4] x g_lin,k, where g_lin,k is the module's own exact branching threshold (MT1,
results/pd3/modules/modules_table.json, P0 rows; AL = the 526 module, threshold recomputed here the same way).
Lifetime tau = censored-exponential MLE (sum of extinction times censored at T / number of deaths).
Usage: python pd3_island_sim.py --module isl1|...|AL --r R  ->  results/pd3/module_theory/island_<module>_r<R>.npz
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import brentq
import torch

C = "/MALECNS_CACHE"; ROOT = "/MALECNS_ROOT"
ap = argparse.ArgumentParser(); ap.add_argument("--module", required=True); ap.add_argument("--r", type=float, required=True)
ap.add_argument("--B", type=int, default=64); ap.add_argument("--T", type=float, default=3000.0)
a = ap.parse_args()
dev = "cuda"; t0 = time.time(); dt = 0.2; a_eff = (1 - np.exp(-dt)) / dt
log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
part = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz", allow_pickle=True)
memb = part["membership"]; names = [str(x) for x in part["names"]]
if a.module == "AL":
    sel = np.isin(memb, [names.index("AL-L"), names.index("AL-R"), names.index("AL-mid")])
else:
    sel = memb == names.index(a.module)
idx = np.where(sel)[0]; n = len(idx); loc = -np.ones(N, np.int64); loc[idx] = np.arange(n)
m = sel[pre] & sel[post] & (sign[pre] != 0)
pi, po, ww, sg = loc[pre[m]], loc[post[m]], w[m], sign[pre[m]]
Em = sp.csr_matrix((c * ww * (sg > 0), (po, pi)), shape=(n, n))
fdt = lambda x: (1 - np.exp(-np.tanh(np.clip(x, 0, None)) * dt)) / dt


def lam(A):
    if A.shape[0] <= 400:
        return float(np.linalg.eigvals(A.toarray()).real.max())
    ev = sla.eigs(A, k=3, which="LR", tol=1e-9, maxiter=20000, v0=np.ones(A.shape[0]), return_eigenvectors=False)
    return float(ev.real.max())


Lam = lam(Em)
def Bf(g):
    B = Em.copy(); B.data = fdt(g * B.data); return B
glin = float(np.exp(brentq(lambda lg: np.log(lam(Bf(np.exp(lg)))) - np.log(a_eff), np.log(a_eff / Lam) - 1e-9,
                           np.log(a_eff / Lam) + 3, xtol=1e-9, rtol=1e-7)))
tabrow = None
tp = f"{ROOT}/results/pd3/modules/modules_table.json"
if os.path.exists(tp) and "P0" in json.load(open(tp)):
    tabrow = next((r_ for r_ in json.load(open(tp))["P0"]["rows"] if r_["name"] == a.module), None)
log(f"module {a.module}: n={n}, internal edges {int(m.sum())}, Lambda={Lam:.4f}, g_lin={glin:.5f}"
    f" (MT1 table: {tabrow['g_lin'] if tabrow else 'n/a'})")
coef = np.where(sg > 0, 1.0, -a.r) * ww * c
W = torch.sparse_coo_tensor(torch.tensor(np.vstack([po, pi]), device=dev), torch.tensor(coef, device=dev, dtype=torch.float32),
                            (n, n)).coalesce().to_sparse_csr()
G = glin * np.geomspace(1, 4, 24); steps = int(round(a.T / dt)); p_off = 1 - np.exp(-dt)
ext = np.full((len(G), a.B), -1, np.int32); occ = np.zeros((len(G), n), np.float32)
for gi, g in enumerate(G):
    X = torch.ones(n, a.B, device=dev); alive = torch.ones(a.B, dtype=torch.bool, device=dev)
    ex = torch.full((a.B,), -1, dtype=torch.int32, device=dev); oc = torch.zeros(n, device=dev); no = 0
    for s in range(1, steps + 1):
        inp = g * torch.sparse.mm(W, X)
        p_on = -torch.expm1(-torch.tanh(inp.clamp(min=0)) * dt)
        u = torch.rand_like(X); X = torch.where(X > 0, (u >= p_off).float(), (u < p_on).float())
        if s > steps // 2:
            oc += X.mean(1); no += 1
        if s % 5 == 0 or s < 50:
            dead = (X.sum(0) == 0) & alive
            if dead.any():
                ex[dead] = s; alive &= ~dead
                if not alive.any():
                    break
    # extinction recorded at the first check after death: exact for s < 50, within 5 steps (1 model time) after
    ext[gi] = ex.cpu().numpy(); occ[gi] = (oc / max(no, 1)).cpu().numpy()
    tt = np.where(ext[gi] >= 0, ext[gi], steps) * dt; d = int((ext[gi] >= 0).sum())
    log(f"g={g:.4f} (x={g/glin:.3f}): deaths {d}/{a.B}, tau_MLE={tt.sum()/max(d,1e-9):.4g}")
tmp = f"{ROOT}/results/pd3/module_theory/island_{a.module}_r{a.r:g}.tmp.npz"
np.savez_compressed(tmp, g=G, x=G / glin, ext_step=ext, occ=occ, dt=dt, T=a.T, B=a.B, n=n, Lambda=Lam, g_lin=glin,
                    idx=idx, meta=json.dumps(vars(a)))
os.replace(tmp, tmp.replace(".tmp.npz", ".npz"))
log("saved")
