"""Stochastic Wilson-Cowan on the antennal-lobe module ALONE (526 neurons: union of the 99% supports of the two leading
excitatory modes; internal synapses only; the rest of the brain silent), same rates, scaling c and synchronous dt as swc.py.
Also writes results/module/module_graph.npz for the browser simulation on the explainer's flow page.
Question: does the module by itself reproduce the whole-brain onsets (slow window and sustained activity)?
Usage: python module_sim.py  (GPU if available) -> results/module/module_decay.json
"""
import json, time
import numpy as np
import torch

C = "/MALECNS_CACHE"; ROOT = "/MALECNS_ROOT"
dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
pos = np.load(C + "/positions.npy"); typ = np.load(C + "/types.npy"); sides = np.load(C + "/sides.npy")
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
t = np.load(f"{ROOT}/results/pd/tail_real_r0_dir.npz", allow_pickle=True)
M = np.zeros(N, bool)
for k in range(2):
    p = t["P"][:, k].astype(float); o = np.argsort(-p); M[o[: np.searchsorted(np.cumsum(p[o]) / p.sum(), 0.99) + 1]] = True
idx = np.where(M)[0]; loc = -np.ones(N, np.int64); loc[idx] = np.arange(len(idx))
m = M[pre] & M[post] & (sign[pre] != 0)
pi, po, ww, sg = loc[pre[m]], loc[post[m]], w[m], sign[pre[m]]
n = len(idx)
P = pos[idx].copy(); bad = ~np.isfinite(P).all(1)
for i in np.where(bad)[0]:                      # place unlocated somata at their strongest partners' mean position
    nb = np.r_[po[pi == i], pi[po == i]]; nb = nb[~bad[nb]] if len(nb) else nb
    P[i] = P[nb].mean(0) if len(nb) else np.nanmean(P[~bad], 0)
np.savez_compressed(f"{ROOT}/results/module/module_graph.npz", idx=idx, pre=pi.astype(np.int16), post=po.astype(np.int16),
                    w=ww.astype(np.uint16), sign=sign[idx].astype(np.int8), pos=P.astype(np.float32), c=c,
                    types=typ[idx], sides=sides[idx], unplaced=bad)
print(f"module n={n}, edges={len(pi)} (exc {int((sg>0).sum())}, inh {int((sg<0).sum())}), c={c:.6g}", flush=True)
dt, T, B = 0.2, 3000.0, 64
steps = int(T / dt); p_off = 1 - np.exp(-dt)
rec = np.unique(np.round(np.logspace(0, np.log10(steps), 100)).astype(int))
out = {"n": n, "B": B, "T": T, "rows": []}
for r in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
    coef = np.where(sg > 0, 1.0, -r) * ww * c
    W = torch.sparse_coo_tensor(torch.tensor(np.vstack([po, pi]), device=dev), torch.tensor(coef, device=dev, dtype=torch.float32), (n, n)).coalesce().to_sparse_csr()
    for g in np.round(np.geomspace(0.15, 1.2, 22), 4):
        X = torch.ones(n, B, device=dev); alive_t = np.full(B, np.inf); k = 0; dens = []
        for s in range(1, steps + 1):
            inp = g * torch.sparse.mm(W, X)
            p_on = 1 - torch.exp(-torch.tanh(inp.clamp(min=0)) * dt)
            u = torch.rand_like(X); X = torch.where(X > 0, (u >= p_off).float(), (u < p_on).float())
            if s == rec[k]:
                na = X.sum(0).cpu().numpy(); dens.append(float(na.mean() / n))
                newly = (na == 0) & ~np.isfinite(alive_t); alive_t[newly] = s * dt; k += 1
                if k == len(rec) or (na == 0).all():
                    break
        surv = float(np.isinf(alive_t).mean())
        life = float(np.max(alive_t[np.isfinite(alive_t)])) if np.isfinite(alive_t).any() else None
        med = float(np.median(np.where(np.isinf(alive_t), T * 10, alive_t)))
        out["rows"].append({"r": r, "g": float(g), "surv_T": surv, "last_death": life, "median_life": med, "rho_end": dens[-1]})
        print(f"[{time.time()-t0:.0f}s] r={r} g={g:.4f} surv(T)={surv:.2f} median life={med:.0f} rho_end={dens[-1]:.3f}", flush=True)
    json.dump(out, open(f"{ROOT}/results/module/module_decay.json", "w"), indent=1)
print("saved")
