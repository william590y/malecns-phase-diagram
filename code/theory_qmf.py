"""Theory for the phase diagram, computed on the same graphs and grids as the simulations (results/pd).

1. Rectified Perron root mu(r): the growth rate of the linearised, rectified dynamics x <- [W~(r) x]_+ (x >= 0). Near the
   silent state a neuron whose summed input is negative has rate 0, so this, not the ordinary eigenvalue of the signed
   matrix, is the spectral quantity that sets the deterministic onset. For r = 0 it is the Perron eigenvalue of the
   excitatory operator. Discrete-time onset (synchronous updates, dt): g_QMF = (1 - e^{-alpha dt}) / (dt * mu).
2. Quenched mean-field (QMF) dynamics: the exact expectation of the simulation's update under the independence
   (mean-field) closure, iterated from the same fully active start for the same number of steps:
       rho_i <- rho_i e^{-alpha dt} + (1 - rho_i) (1 - exp(-beta tanh(g sum_j W~_ij rho_j + h)_+ dt))
   Gives rho(t), the end-of-run density, participation ratio and antennal-lobe share for every (g, r), with no fitting.
   It ignores fluctuations, so it cannot see extinction of a small active region: the gap between QMF and simulation is
   the finite-size (rare-region) effect.
Usage: python theory_qmf.py --kind real|null|wnull --r 0 0.25 ... [--h H] --out results/pd/qmf_<kind>.npz
"""
import argparse, json, time
import numpy as np
import torch

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["real", "null", "wnull"], required=True)
ap.add_argument("--r", type=float, nargs="+", required=True)
ap.add_argument("--h", type=float, nargs="+", default=[0.0])
ap.add_argument("--T", type=float, default=3000.0)
ap.add_argument("--dt", type=float, default=0.2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"; t0 = time.time()
G = np.logspace(-1, np.log10(30), 32)                     # same grid as pd_submit.sh
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float32); sign = np.load(C + "/sign.npy"); N = len(sign)
rng = np.random.default_rng(0)                            # same permutation as swc.py --null (seed 0)
if a.kind == "null":
    post = post[rng.permutation(len(post))]
elif a.kind == "wnull":
    w = w[rng.permutation(len(w))]
ins = np.bincount(post, weights=w, minlength=N)
scale = np.float32(1.0 / ins[ins > 0].mean())
V = np.load(f"{ROOT}/results/sym_exc_weighted_vecs_top10.npy")
mod = torch.zeros(N, dtype=torch.bool, device=dev); mod[torch.tensor(np.argsort(-V[:, 0] ** 2)[:1000], device=dev)] = True
steps = int(round(a.T / a.dt))
rec = np.unique(np.round(np.logspace(0, np.log10(steps), 120)).astype(int))
decay = float(np.exp(-a.dt))
gv = torch.tensor(G, dtype=torch.float32, device=dev)
out = {"g": G, "t": rec * a.dt, "r": np.array(a.r), "h": np.array(a.h), "kind": a.kind}
mu_l, rho_l, end_l, pr_l, al_l = [], [], [], [], []
for r in a.r:
    s = np.where(sign[pre] > 0, 1.0, np.where(sign[pre] < 0, -r, 0.0)).astype(np.float32)
    keep = s != 0
    idx = torch.tensor(np.vstack([post[keep], pre[keep]]), device=dev)
    Wm = torch.sparse_coo_tensor(idx, torch.tensor(s[keep] * w[keep] * scale, device=dev), (N, N)).coalesce().to_sparse_csr()
    del idx
    # rectified Perron root by shifted power iteration (x <- (x + [Wx]_+ / mu) / 2 converges without oscillation)
    x = torch.ones(N, 1, device=dev); x /= x.norm(); mu = 1.0
    for it in range(4000):
        y = torch.clamp(torch.sparse.mm(Wm, x), min=0)
        mu_new = float(y.norm() / x.norm())
        x = x + y / max(mu_new, 1e-12); x /= x.norm()
        if it > 50 and abs(mu_new - mu) < 1e-7 * mu_new:
            mu = mu_new; break
        mu = mu_new
    xv = x[:, 0] ** 2; xv /= xv.sum()
    mu_l.append([mu, float(1 / (xv ** 2).sum()), float(xv[mod].sum())])
    print(f"[{time.time()-t0:.0f}s] {a.kind} r={r}: rectified mu={mu:.4f} (1/IPR {1/float((xv**2).sum()):.0f}, AL {float(xv[mod].sum()):.2f}) "
          f"g_QMF={(1-decay)/(a.dt*mu):.4f}", flush=True)
    for h in a.h:
        X = torch.ones(N, len(G), device=dev); rho, k = [], 0
        for st in range(1, steps + 1):
            inp = torch.sparse.mm(Wm, X) * gv + h
            p_on = 1 - torch.exp(-torch.tanh(inp.clamp(min=0)) * a.dt)
            X = X * decay + (1 - X) * p_on
            if st == rec[k]:
                rho.append(X.mean(0).cpu().numpy()); k += 1
                if k == len(rec):
                    break
        rho = np.array(rho).T                              # [G, t]
        Xs = X.double()
        pr = ((Xs.sum(0) ** 2) / (Xs ** 2).sum(0).clamp(min=1e-300) / N).cpu().numpy()
        al = (Xs[mod].sum(0) / Xs.sum(0).clamp(min=1e-300)).cpu().numpy()
        rho_l.append(rho); end_l.append(rho[:, -1]); pr_l.append(pr); al_l.append(al)
        print(f"[{time.time()-t0:.0f}s]   h={h:g} rho_T: " + " ".join("%.2g" % v for v in rho[:, -1]), flush=True)
    del Wm; torch.cuda.empty_cache()
out["mu"] = np.array(mu_l)                                   # [r, (mu, 1/IPR, AL share)]
out["rho"] = np.array(rho_l).reshape(len(a.r), len(a.h), len(G), -1)
out["rho_T"] = np.array(end_l).reshape(len(a.r), len(a.h), len(G))
out["PR"] = np.array(pr_l).reshape(len(a.r), len(a.h), len(G))
out["AL"] = np.array(al_l).reshape(len(a.r), len(a.h), len(G))
out["meta"] = json.dumps(vars(a))
np.savez_compressed(a.out, **out)
print("saved", a.out, flush=True)
