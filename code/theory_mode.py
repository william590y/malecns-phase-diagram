"""Leading (rectified) mode of each graph and its effective size K, for the finite-size onset theory (see theory page).
For each r: right vector v of the rectified map x -> [W~(r) x]_+ (shifted power iteration), its growth rate mu, the active
set S = {i: (W~ v)_i > 0}, the left vector u of the linearisation J = 1_S W~ (power iteration on J^T), normalised <u, v> = 1,
and the effective number of independent units of the mode
    K = 1 / ( <u^2, v> <u, v^2> )          (= n for a mode spread evenly over n neurons)
plus the symmetric proxy K_sym = (sum v^2)^3 / (sum v^3)^2, 1/IPR = (sum v^2)^2 / sum v^4 and PR_L1 = (sum v)^2 / sum v^2.
Prediction of the sustained onset over an observation time T (alpha = beta = 1, synchronous steps dt):
    K Phi(R_on) = ln(alpha_eff T) - ln( sqrt(2 pi / K) R_on / (R_on - 1)^2 ),  Phi(R) = ln R - 1 + 1/R,
    g_QMF = alpha_eff / mu, alpha_eff = (1 - e^{-dt}) / dt,  g_on = R_on g_QMF.
Usage: theory_mode.py <kind> <r ...>   -> results/pd/mode_<kind>.json
"""
import json, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
kind, RS = sys.argv[1], [float(x) for x in sys.argv[2:]]
t0 = time.time()
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
sup = np.load(C + "/superclasses.npy"); typ = np.load(C + "/types.npy")
rng = np.random.default_rng(0)
if kind == "null":
    post = post[rng.permutation(len(post))]
elif kind == "wnull":
    w = w[rng.permutation(len(w))]
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
V0 = np.load(f"{ROOT}/results/sym_exc_weighted_vecs_top10.npy")
mod = np.zeros(N, bool); mod[np.argsort(-V0[:, 0] ** 2)[:1000]] = True
dt, T = 0.2, 3000.0
a_eff = (1 - np.exp(-dt)) / dt
Phi = lambda R: np.log(R) - 1 + 1 / R


def onset(K, T):
    f = lambda R: K * Phi(R) - (np.log(a_eff * T) - np.log(np.sqrt(2 * np.pi / K) * R / (R - 1) ** 2))
    try:
        return brentq(f, 1 + 1e-6, 50)
    except ValueError:
        return float("nan")


res = {"kind": kind, "alpha_eff": a_eff, "T": T, "rows": []}
for r in RS:
    s = np.where(sign[pre] > 0, 1.0, np.where(sign[pre] < 0, -r, 0.0)); m = s != 0
    M = sp.csr_matrix((s[m] * w[m] * c, (post[m], pre[m])), shape=(N, N)); MT = M.T.tocsr()
    x = np.ones(N) / np.sqrt(N); mu = 0
    for it in range(6000):
        y = np.maximum(M @ x, 0); mu_new = np.linalg.norm(y) / np.linalg.norm(x)
        x = x + y / mu_new; x /= np.linalg.norm(x)
        if it > 100 and abs(mu_new - mu) < 1e-9 * mu_new:
            break
        mu = mu_new
    mu = mu_new; v = x
    S = (M @ v) > 0
    u = v.copy()
    for it2 in range(6000):
        y = MT @ (S * u); lam = np.linalg.norm(y) / np.linalg.norm(u)
        u_new = u + y / lam; u_new /= np.linalg.norm(u_new)
        if it2 > 100 and np.linalg.norm(u_new - u) < 1e-9:
            u = u_new; break
        u = u_new
    u = u / (u @ v)
    K = 1 / ((u ** 2) @ v * (u @ (v ** 2)))
    Ks = (v ** 2).sum() ** 3 / ((v ** 3).sum()) ** 2
    top = np.argsort(-v)[:200]
    sc, cnt = np.unique(sup[top], return_counts=True); ty, tc = np.unique(typ[top], return_counts=True)
    row = {"r": r, "mu": mu, "lam_left": lam, "iters": [it, it2], "K": K, "K_sym": Ks,
           "inv_ipr": (v ** 2).sum() ** 2 / (v ** 4).sum(), "PR_L1": v.sum() ** 2 / (v ** 2).sum(),
           "AL_share_v": float(v[mod].sum() / v.sum()), "u_neg_mass": float(-u[u < 0].sum() / np.abs(u).sum()),
           "g_QMF": a_eff / mu, "R_on": onset(K, T), "R_on_sym": onset(Ks, T),
           "superclasses": dict(sorted(zip(sc.tolist(), cnt.tolist()), key=lambda z: -z[1])[:4]),
           "types": dict(sorted(zip(ty.tolist(), tc.tolist()), key=lambda z: -z[1])[:5])}
    row["g_on_pred"] = row["g_QMF"] * row["R_on"]
    res["rows"].append(row)
    np.save(f"{ROOT}/results/pd/mode_{kind}_r{r:g}_v.npy", v.astype(np.float32))
    print(f"[{time.time()-t0:.0f}s] {kind} r={r:g}: mu {mu:.4f} (left {lam:.4f}, it {it}/{it2}) g_QMF {row['g_QMF']:.4f} "
          f"K {K:.1f} K_sym {Ks:.1f} 1/IPR {row['inv_ipr']:.0f} PR_L1 {row['PR_L1']:.0f} AL {row['AL_share_v']:.2f} "
          f"u_neg {row['u_neg_mass']:.3f} -> R_on {row['R_on']:.3f} g_on {row['g_on_pred']:.4f}", flush=True)
    json.dump(res, open(f"{ROOT}/results/pd/mode_{kind}.json", "w"), indent=1)
