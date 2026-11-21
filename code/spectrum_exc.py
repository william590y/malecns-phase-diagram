"""Directed excitatory-only operator (raw scaling, as swc.py --norm raw with inhibitory/modulatory inputs dropped), real vs
the same configuration null as swc.py --null. Rationale: the rate f(s) = beta tanh(s)_+ is rectified, so near the silent
state an active inhibitory neuron cannot lower anyone's rate below zero; for sparse activity the growth operator is closer to
the excitatory part than to the signed matrix. Writes results/spectrum_exc.json.
"""
import json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

C = "/MALECNS_CACHE"
OUT = "/MALECNS_ROOT/results"
t0 = time.time()
pre = np.load(C + "/pre.npy").astype(np.int64); post0 = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
exc = sign[pre] > 0
res = {}
for null in [False, True]:
    post = post0[np.random.default_rng(0).permutation(len(post0))] if null else post0
    ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
    M = sp.csr_matrix((w[exc] * c, (post[exc], pre[exc])), shape=(N, N))
    vals, vecs = sla.eigs(M, k=10, which="LR", tol=1e-6, maxiter=50000)
    o = np.argsort(-vals.real); vals, vecs = vals[o], vecs[:, o]
    p = np.abs(vecs) ** 2; p /= p.sum(0, keepdims=True)
    k = "null_raw_exc" if null else "raw_exc"
    res[k] = {"eig_re": vals.real.tolist(), "ipr": (p ** 2).sum(0).tolist(), "g_star_meanfield": float(1 / vals[0].real)}
    print(f"[{time.time()-t0:.0f}s] {k} Re lam_max {vals[0].real:.4f} g* {1/vals[0].real:.3f} IPR {(p[:,0]**2).sum():.3g}", flush=True)
json.dump(res, open(f"{OUT}/spectrum_exc.json", "w"), indent=1)
