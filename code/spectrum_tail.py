"""Tail of the spectrum: top-K eigenpairs of the raw-scaled operator the simulation uses, for the 'island-counting' theory of
the localized active phase. If each strongly localized eigenvector is a rare region that switches on by itself when
g * lambda_k crosses the threshold, then the number of active islands, and the participation ratio of the activity, follow
from counting eigenvalues above 1/g weighted by their sizes 1/IPR_k. Delocalization happens when 1/g reaches the
'mobility edge', where eigenvectors stop being localized.
Usage: spectrum_tail.py <kind: real|null|wnull> <r> <K> <mode: dir|sym>
Writes results/pd/tail_<kind>_r<r>_<mode>.npz: eigenvalues, IPR, AL share, superclass of the top neuron, and |v|^2 (float16).
"""
import sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
kind, r, K, mode = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
t0 = time.time()
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
sup = np.load(C + "/superclasses.npy")
rng = np.random.default_rng(0)
if kind == "null":
    post = post[rng.permutation(len(post))]
elif kind == "wnull":
    w = w[rng.permutation(len(w))]
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
s = np.where(sign[pre] > 0, 1.0, np.where(sign[pre] < 0, -r, 0.0)); m = s != 0
M = sp.csr_matrix((s[m] * w[m] * c, (post[m], pre[m])), shape=(N, N))
if mode == "sym":
    vals, vecs = sla.eigsh(((M + M.T) * 0.5).tocsr(), k=K, which="LA", tol=1e-5, maxiter=100000)
else:
    vals, vecs = sla.eigs(M, k=K, which="LR", tol=1e-5, maxiter=100000)
o = np.argsort(-vals.real); vals, vecs = vals[o], vecs[:, o]
P = np.abs(vecs) ** 2; P /= P.sum(0, keepdims=True)
V0 = np.load(f"{ROOT}/results/sym_exc_weighted_vecs_top10.npy")
mod = np.zeros(N, bool); mod[np.argsort(-V0[:, 0] ** 2)[:1000]] = True
np.savez_compressed(f"{ROOT}/results/pd/tail_{kind}_r{sys.argv[2]}_{mode}.npz", lam=vals.real, lam_im=vals.imag,
                    ipr=(P ** 2).sum(0), AL=P[mod].sum(0), topsup=sup[P.argmax(0)], P=P.astype(np.float16))
print(f"[{time.time()-t0:.0f}s] {kind} r={r} {mode}: top {vals.real[:5].round(3)}  K-th {vals.real[-1]:.3f}  "
      f"median 1/IPR {np.median(1/(P**2).sum(0)):.0f}", flush=True)
