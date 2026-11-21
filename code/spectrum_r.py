"""Mean-field theory lines for the (g, r) phase diagram: leading eigenvalue of the raw-scaled signed operator
W~(r) = c * w_ij * s_j(r) (s = +1 ACh, -r GABA/Glu/His, 0 modulatory) as a function of the inhibition ratio r, for the real
connectome, the configuration null (targets permuted) and the weight-shuffled null (synapse counts permuted across edges),
each with the same permutation (default_rng(0)) as swc.py --null [--null_kind weights]. r = 0 is the excitatory-only operator.
Also the IPR and the 200-neuron composition of the r = 0 leading mode of the weight-shuffled null.
Writes results/spectrum_r.json.
"""
import json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

C = "/MALECNS_CACHE"
OUT = "/MALECNS_ROOT/results"
t0 = time.time()
pre = np.load(C + "/pre.npy").astype(np.int64); post0 = np.load(C + "/post.npy").astype(np.int64)
w0 = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
sup = np.load(C + "/superclasses.npy"); typ = np.load(C + "/types.npy")
R = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6]
res = {"r": R}
for kind in ["real", "null", "wnull"]:
    post, w = post0, w0
    if kind == "null":
        post = post0[np.random.default_rng(0).permutation(len(post0))]
    if kind == "wnull":
        w = w0[np.random.default_rng(0).permutation(len(w0))]
    c = 1 / np.bincount(post, weights=w, minlength=N)[np.bincount(post, weights=w, minlength=N) > 0].mean()
    out = []
    for r in R:
        s = np.where(sign[pre] > 0, 1.0, np.where(sign[pre] < 0, -r, 0.0))
        m = s != 0
        M = sp.csr_matrix((s[m] * w[m] * c, (post[m], pre[m])), shape=(N, N))
        vals, vecs = sla.eigs(M, k=6, which="LR", tol=1e-6, maxiter=50000)
        o = np.argsort(-vals.real); vals, vecs = vals[o], vecs[:, o]
        p = np.abs(vecs[:, 0]) ** 2; p /= p.sum()
        d = {"r": r, "lam": float(vals[0].real), "lam2": float(vals[1].real), "g_star": float(1 / vals[0].real),
             "ipr": float((p ** 2).sum())}
        if r == 0:
            top = np.argsort(-p)[:200]
            sc, cnt = np.unique(sup[top], return_counts=True); ty, tc = np.unique(typ[top], return_counts=True)
            d["lead_superclasses"] = dict(sorted(zip(sc.tolist(), cnt.tolist()), key=lambda x: -x[1])[:5])
            d["lead_types"] = dict(sorted(zip(ty.tolist(), tc.tolist()), key=lambda x: -x[1])[:6])
            d["share_top100"] = float(np.sort(p)[::-1][:100].sum())
        out.append(d)
        print(f"[{time.time()-t0:.0f}s] {kind} r={r}: lam {d['lam']:.4f} g* {d['g_star']:.3f} 1/IPR {1/d['ipr']:.0f}", flush=True)
    res[kind] = out
    json.dump(res, open(f"{OUT}/spectrum_r.json", "w"), indent=1)
print("saved")
