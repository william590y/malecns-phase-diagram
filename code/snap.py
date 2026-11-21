"""Record full-network activity snapshots of the stochastic Wilson-Cowan model for movies.
One replica per coupling g, from a fully active start (h = 0); state saved (bit-packed) at ~90 times (linear to t=10,
then log-spaced to T). Same model and options as swc.py.
Usage: python snap.py --norm raw --g 0.25 0.36 1.0 2.5 [--null] --T 3000 --out results/snaps_raw.npz
"""
import argparse, json, time
import numpy as np
import torch
C = "/MALECNS_CACHE"
ap = argparse.ArgumentParser()
ap.add_argument("--norm", choices=["in", "raw"], default="raw")
ap.add_argument("--g", type=float, nargs="+", required=True)
ap.add_argument("--r", type=float, default=1.0)
ap.add_argument("--T", type=float, default=3000.0)
ap.add_argument("--dt", type=float, default=0.2)
ap.add_argument("--null", action="store_true")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args(); dev = "cuda"; t0 = time.time()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float32); sign = np.load(C + "/sign.npy"); N = len(sign)
if a.null:
    post = post[np.random.default_rng(0).permutation(len(post))]
ins = np.bincount(post, weights=w, minlength=N).astype(np.float32)
scale = np.where(ins > 0, 1.0 / np.maximum(ins, 1e-9), 0.0) if a.norm == "in" else np.full(N, 1.0 / ins[ins > 0].mean())
coef = np.where(sign[pre] > 0, 1.0, np.where(sign[pre] < 0, -a.r, 0.0)).astype(np.float32) * w * scale[post].astype(np.float32)
keep = coef != 0
Wm = torch.sparse_coo_tensor(torch.tensor(np.vstack([post[keep], pre[keep]]), device=dev),
                             torch.tensor(coef[keep], device=dev), (N, N)).coalesce().to_sparse_csr()
steps = int(round(a.T / a.dt))
lin = np.arange(1, int(10 / a.dt) + 1)
rec = np.unique(np.r_[lin, np.round(np.logspace(np.log10(lin[-1] + 1), np.log10(steps), 60)).astype(int)])
p_off = 1 - np.exp(-a.dt)
frames, times = {}, rec * a.dt
for g in a.g:
    X = torch.ones(N, 1, device=dev); snaps = [np.packbits(np.ones(N, bool))]; k = 0
    for s in range(1, steps + 1):
        inp = g * torch.sparse.mm(Wm, X)
        p_on = 1 - torch.exp(-torch.tanh(inp.clamp(min=0)) * a.dt)
        u = torch.rand_like(X)
        X = torch.where(X > 0, (u >= p_off).float(), (u < p_on).float())
        if s == rec[k]:
            snaps.append(np.packbits(X[:, 0].bool().cpu().numpy())); k += 1
            if k == len(rec):
                break
    while len(snaps) < len(rec) + 1:
        snaps.append(np.packbits(np.zeros(N, bool)))
    frames[f"g{g:.4f}"] = np.array(snaps)
    print(f"[{time.time()-t0:.0f}s] g={g} final active={int(X.sum())}", flush=True)
np.savez_compressed(a.out, times=np.r_[0, times], N=N, meta=json.dumps(vars(a)), **frames)
print("saved", a.out)
