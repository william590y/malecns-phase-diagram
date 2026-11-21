"""Spectrum of the exact operator the simulation uses (swc.py), real vs configuration null, plus the mean-field onset.
W~_ij = c_i w_ij s_j with s_j = +1 (ACh), -r (GABA/Glu/His), 0 (modulatory); c_i = 1/mean in-strength (raw) or
1/in-strength_i (in). The null permutes targets with the same seed as swc.py --null (default_rng(0)), so it is the same
null graph the dynamics ran on. Linearising the mean field at rho = 0 (tanh'(0) = 1) gives onset g* = alpha/(beta Re lambda_max).
For the in-norm operator, also lists the in-degree / top input share of the neurons that carry the leading modes, to test
whether the 0.9 ladder comes from neurons dominated by one presynaptic partner.
Writes results/spectrum_sim.json.
"""
import json, time, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

C = "/MALECNS_CACHE"
OUT = "/MALECNS_ROOT/results"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:.0f}s]", *a, flush=True)
pre0 = np.load(C + "/pre.npy").astype(np.int64); post0 = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy")
sup = np.load(C + "/superclasses.npy"); typ = np.load(C + "/types.npy")
N = len(sign); r = 1.0
s = np.where(sign[pre0] > 0, 1.0, np.where(sign[pre0] < 0, -r, 0.0))
res = {}


def top(v, n=200):
    p = np.abs(v) ** 2; p = p / p.sum(); o = np.argsort(-p)
    sc, cnt = np.unique(sup[o[:n]], return_counts=True); ty, tc = np.unique(typ[o[:n]], return_counts=True)
    return o, {"share_top10": float(p[o[:10]].sum()), "share_top100": float(p[o[:100]].sum()),
               "superclasses": dict(sorted(zip(sc.tolist(), cnt.tolist()), key=lambda x: -x[1])[:6]),
               "types": dict(sorted(zip(ty.tolist(), tc.tolist()), key=lambda x: -x[1])[:8])}


for null in [False, True]:
    post = post0[np.random.default_rng(0).permutation(len(post0))] if null else post0
    ins = np.bincount(post, weights=w, minlength=N)
    indeg = np.bincount(post, minlength=N)
    maxin = np.zeros(N); np.maximum.at(maxin, post, w)
    for norm in ["raw", "in"]:
        c = np.where(ins > 0, 1 / np.maximum(ins, 1e-9), 0) if norm == "in" else np.full(N, 1 / ins[ins > 0].mean())
        M = sp.csr_matrix((s * w * c[post], (post, pre0)), shape=(N, N))
        name = f"{'null_' if null else ''}{norm}"
        vals, vecs = sla.eigs(M, k=20, which="LR", tol=1e-6, maxiter=50000)
        o = np.argsort(-vals.real); vals, vecs = vals[o], vecs[:, o]
        p = np.abs(vecs) ** 2; p /= p.sum(0, keepdims=True); ipr = (p ** 2).sum(0)
        oi, lead = top(vecs[:, 0])
        d = {"eig_re": vals.real.tolist(), "eig_im": vals.imag.tolist(), "ipr": ipr.tolist(),
             "g_star_meanfield": float(1 / vals[0].real), "lead": lead}
        # carriers of the top 5 modes: in-degree and largest-single-input share
        car = [int(np.argmax(p[:, j])) for j in range(5)]
        d["carriers"] = [{"mode": j, "neuron": int(i), "type": str(typ[i]), "superclass": str(sup[i]),
                          "in_degree": int(indeg[i]), "max_input_share": float(maxin[i] / max(ins[i], 1e-9)),
                          "weight_share": float(p[i, j])} for j, i in enumerate(car)]
        res[name] = d
        log(name, "Re lam_max %.4f" % vals[0].real, "g* %.3f" % (1 / vals[0].real), "IPR %.3g" % ipr[0],
            "top-5 Re", np.round(vals.real[:5], 3).tolist())
json.dump(res, open(f"{OUT}/spectrum_sim.json", "w"), indent=1)
log("saved", f"{OUT}/spectrum_sim.json")
