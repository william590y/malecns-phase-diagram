"""Stage 1: structure and spectrum of the MaleCNS v1.0 neuron graph (167,216 neurons, 25.6M edges).

Questions for the Griffiths-phase hypothesis (Moretti & Munoz 2013; Odor et al.):
  - how heavy-tailed are in/out degrees and strengths;
  - is the leading eigenvector of the (excitatory) connectivity LOCALIZED (inverse participation ratio IPR = O(1))
    or delocalized (IPR ~ 1/N), and where (superclass / cell type);
  - how many top eigenvalues are localized and how dense the top of the spectrum is (rare-region / Lifshitz tail);
  - does a degree-preserving null graph (directed configuration model by permuting edge targets) lose the localization.
A[i, j] = synapse count from neuron j (pre) onto neuron i (post). Signs from consensus neurotransmitter (cache sign.npy:
ACh +1, GABA/Glu/His -1, modulatory/unclear 0).
Usage: python spectrum.py [k_sym=100]
"""
import json, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

C = "/MALECNS_CACHE"
OUT = "/MALECNS_ROOT/results"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

pre = np.load(C + "/pre.npy"); post = np.load(C + "/post.npy")
w = np.load(C + "/weight.npy").astype(np.float64)
sign = np.load(C + "/sign.npy").astype(np.float64)
sup = np.load(C + "/superclasses.npy"); typ = np.load(C + "/types.npy")
N = len(sign)
log("loaded", N, "neurons", len(w), "edges")


def mat(m, ww):
    return sp.csr_matrix((ww[m], (post[m], pre[m])), shape=(N, N))


allm = np.ones(len(w), bool)
exc = sign[pre] > 0
A = mat(allm, w)
AE = mat(exc, w)
W = mat(allm, w * sign[pre])
B = mat(allm, np.ones_like(w))          # binary
res = {"N": N, "edges": int(len(w)), "frac_edges_exc": float(exc.mean()),
       "frac_syn_exc": float(w[exc].sum() / w.sum()),
       "frac_syn_inh": float(w[sign[pre] < 0].sum() / w.sum())}


def dist(x):
    q = np.percentile(x, [50, 90, 99, 99.9, 100])
    return {"mean": float(x.mean()), "p50": q[0], "p90": q[1], "p99": q[2], "p99.9": q[3], "max": q[4],
            "cv": float(x.std() / max(x.mean(), 1e-12))}


res["in_degree"] = dist(np.asarray(B.sum(1)).ravel()); res["out_degree"] = dist(np.asarray(B.sum(0)).ravel())
res["in_strength"] = dist(np.asarray(A.sum(1)).ravel()); res["out_strength"] = dist(np.asarray(A.sum(0)).ravel())
res["edge_weight"] = dist(w)
log("degree stats done")


def ipr(V):
    V = np.abs(V) ** 2
    V = V / V.sum(0, keepdims=True)
    return (V ** 2).sum(0)


def where(v, n=200):
    """Composition of the n highest-|v|^2 neurons and the share of |v|^2 they hold."""
    p = np.abs(v) ** 2; p = p / p.sum()
    o = np.argsort(-p)
    top = o[:n]
    sc, cnt = np.unique(sup[top], return_counts=True)
    ty, tcnt = np.unique(typ[top], return_counts=True)
    return {"share_top10": float(p[o[:10]].sum()), "share_top100": float(p[o[:100]].sum()),
            "share_top1000": float(p[o[:1000]].sum()),
            "superclasses_top%d" % n: dict(sorted(zip(sc.tolist(), cnt.tolist()), key=lambda x: -x[1])[:8]),
            "types_top%d" % n: dict(sorted(zip(ty.tolist(), tcnt.tolist()), key=lambda x: -x[1])[:10])}


def sym_spec(M, name, k):
    S = ((M + M.T) * 0.5).tocsr()
    vals, vecs = sla.eigsh(S, k=k, which="LA", tol=1e-6, maxiter=20000)
    o = np.argsort(-vals); vals, vecs = vals[o], vecs[:, o]
    ip = ipr(vecs)
    res[name] = {"top_eigs": vals.tolist(), "ipr": ip.tolist(), "N_ipr_top": float(N * ip[0]),
                 "lead": where(vecs[:, 0]), "second": where(vecs[:, 1]),
                 "n_localized_ipr_gt_0.01": int((ip > 0.01).sum())}
    np.save(f"{OUT}/{name}_vecs_top10.npy", vecs[:, :10].astype(np.float32))
    log(name, "top eig %.2f" % vals[0], "IPR %.4g (N*IPR %.1f)" % (ip[0], N * ip[0]), "#IPR>0.01:", int((ip > 0.01).sum()))


def nonsym_spec(M, name, k=20):
    vals, vecs = sla.eigs(M, k=k, which="LR", tol=1e-6, maxiter=20000)
    o = np.argsort(-vals.real); vals, vecs = vals[o], vecs[:, o]
    ip = ipr(vecs)
    res[name] = {"top_eigs_real": vals.real.tolist(), "top_eigs_imag": vals.imag.tolist(), "ipr": ip.tolist(),
                 "lead": where(vecs[:, 0])}
    log(name, "top Re eig %.2f" % vals[0].real, "IPR %.4g" % ip[0])


sym_spec(AE, "sym_exc_weighted", K)
sym_spec(mat(exc, np.ones_like(w)), "sym_exc_binary", min(K, 50))
sym_spec(mat(exc, np.log1p(w)), "sym_exc_logweight", min(K, 50))
nonsym_spec(AE, "dir_exc_weighted")
nonsym_spec(W, "dir_signed_weighted")
# in-strength-normalized signed operator (rate models with afferent normalization): row i scaled by 1/in_strength_i
ins = np.asarray(A.sum(1)).ravel(); ins[ins == 0] = 1
nonsym_spec(sp.diags(1 / ins) @ W, "dir_signed_innorm")

# null: directed configuration model (permute targets; keeps every in/out degree and the weight multiset)
rng = np.random.default_rng(0)
perm = rng.permutation(len(post))
post0 = post
post = post0[perm]
sym_spec(mat(exc, w), "null_sym_exc_weighted", min(K, 50))
post = post0
json.dump(res, open(f"{OUT}/spectrum.json", "w"), indent=1, default=float)
log("done ->", f"{OUT}/spectrum.json")
