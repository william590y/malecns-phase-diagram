"""pd3 theory audit, part 3: does the configuration null's leading excitatory mode have K proportional to N?
(The finite-size prediction "the null's onset window shrinks as N^-1/2" requires K_null ~ N.) CPU, a few minutes.

Size-N configuration null: a uniformly random subset of N neurons; every edge whose PRESYNAPTIC neuron is in the subset is an
out-stub (it keeps its weight and the presynaptic sign, as in swc.py --null targets, where post is permuted); the edges whose
POSTSYNAPTIC neuron is in the subset give the in-stub multiset. Out-stubs are matched to a random sample (without replacement)
of in-stubs (if there are more in-stubs than out-stubs; otherwise a random subset of out-stubs is used), so out-degrees and
out-strengths are kept and in-degrees are kept up to that subsampling. c = 1 / mean in-strength of the resulting graph
(the simulator's raw normalisation). N = 167216 with the full edge list is the existing configuration null (same seed-0
permutation as swc.py / theory_branch_gpu.py).
For each graph: lambda_PF(E) (weak-synapse), right/left Perron vectors v, u (<u, v> = 1), K_nc = 1 / (<u^2, v> <u, v^2>),
1/IPR(v), and the weak-synapse threshold g = alpha_eff / lambda_PF(E).
Writes results/pd3/theory_nullK.json.
"""
import json, time
import numpy as np
import scipy.sparse as sp

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:.0f}s]", *a, flush=True)
dt = 0.2; a_eff = (1 - np.exp(-dt)) / dt
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); NF = len(sign)


def perron(M, iters=5000, tol=1e-10):
    n = M.shape[0]; x = np.ones(n) / np.sqrt(n); lam = 0.0
    for it in range(iters):
        y = M @ x; lam_new = np.linalg.norm(y) / np.linalg.norm(x)
        x = x + y / lam_new; x /= np.linalg.norm(x)
        if it > 50 and abs(lam_new - lam) < tol * lam_new:
            break
        lam = lam_new
    return lam_new, x


def analyse(pr, po, ww, sg, n):
    ins = np.bincount(po, weights=ww, minlength=n); c = 1 / ins[ins > 0].mean()
    e = sg > 0
    E = sp.csr_matrix((c * ww[e], (po[e], pr[e])), shape=(n, n)); E.sum_duplicates()
    lam, v = perron(E); lamT, u = perron(E.T.tocsr())
    v = np.abs(v); u = np.abs(u); u = u / (u @ v)
    K = 1 / (((u ** 2) @ v) * (u @ (v ** 2)))
    p = v ** 2 / (v ** 2).sum()
    return {"lambda_PF": float(lam), "lambda_left": float(lamT), "g_weak": float(a_eff / lam), "K_nc": float(K),
            "inv_ipr": float(1 / (p ** 2).sum()), "c": float(c), "n_edges": int(len(pr)), "K_over_N": float(K / n)}


out = {"rows": []}
rng0 = np.random.default_rng(0)
post_full_null = post[rng0.permutation(len(post))]
r = analyse(pre, post_full_null, w, sign[pre], NF); r.update({"N": NF, "seed": 0, "kind": "full null (swc.py seed 0)"})
out["rows"].append(r); log(r)
for N in [5000, 10000, 20000, 40000, 80000]:
    for seed in [1, 2]:
        rng = np.random.default_rng(1000 * seed + N % 997)
        S = np.sort(rng.choice(NF, N, replace=False)); loc = -np.ones(NF, np.int64); loc[S] = np.arange(N)
        out_e = np.where(loc[pre] >= 0)[0]; in_e = np.where(loc[post] >= 0)[0]
        m = min(len(out_e), len(in_e))
        oe = rng.choice(out_e, m, replace=False) if len(out_e) > m else out_e
        ie = rng.choice(in_e, m, replace=False) if len(in_e) > m else in_e
        ie = rng.permutation(ie)
        r = analyse(loc[pre[oe]], loc[post[ie]], w[oe], sign[pre[oe]], N)
        r.update({"N": N, "seed": seed, "kind": "subset stub-matching"}); out["rows"].append(r); log(r)
sub = [x for x in out["rows"]]
for key in ["K_nc", "inv_ipr"]:
    out[f"fit_{key}_vs_N"] = np.polyfit(np.log([x["N"] for x in sub]), np.log([x[key] for x in sub]), 1).tolist()
json.dump(out, open(f"{ROOT}/results/pd3/theory_nullK.json", "w"), indent=1)
log("saved", out["fit_K_nc_vs_N"], out["fit_inv_ipr_vs_N"])
