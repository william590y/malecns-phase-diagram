"""Linear response of the driven network below the linear threshold g_lin (r = 1, raw norm), computed on the real graph.

Companion of code/analysis2_driven.py (which reads the cache written here). CPU, scipy sparse; about 10 minutes.

With drive h > 0 every silent neuron switches on spontaneously with per-step probability q0 = 1 - exp(-tanh(h) dt);
baseline density rho0 = q0 / (q0 + p_off), p_off = 1 - exp(-dt). To first order in the input densities (independent
inputs, i.e. the closure of section 2 of explainer/theory.html but with EXACT single-input increments), neuron i switches
on with probability
    Q_i = q0 + sum_j a_ij rho_j ,   a_ij = exp(-F(h) dt) - exp(-F(h + g W~_ij) dt),   F(s) = tanh(s)_+ ,
and the stationary density solves rho_i = Q_i / (Q_i + p_off) (Q_i clipped at 0). Variants (all r = 1):
  exact   : a_ij over excitatory AND inhibitory inputs. An inhibitory input onto a silent neuron removes at most the
            spontaneous rate (a_ij >= -q0): for g c w_ij >> h it is clipped (acts as -q0), for g c w_ij << h it acts
            linearly (-g c w_ij). So the drive itself decides how much inhibition counts.
  B_only  : inhibitory a_ij = 0 and excitatory a_ij = 1 - exp(-F(g E_ij) dt) = dt * B_ij(g): the h -> 0 limit, the
            branching operator B(g) of theory section 3 (lambda_PF(B(g_lin)) = alpha_eff); evaluated with the
            h = 1e-5 baseline q0 (its relative excess rho_bar/rho0 - 1 is then h-independent to O(h)).
Leading-mode pole: right/left Perron vectors v, u of B(g_lin) (shifted power iteration), <u, v> = 1; for h -> 0,
    rho_bar - rho0 ~= rho0 (sum v)(sum u) / N * lam/(alpha_eff - lam),   lam(g) = alpha_eff g / g_lin
(lambda_PF(B(g)) is proportional to g to within the 0.34% tanh saturation at g <= g_lin; dense theory row at g = 0.1751
gives lam 0.97263 vs alpha_eff g/g_lin = 0.97275).
Output: results/analysis2/driven_linresp.json
"""
import json, time
import numpy as np
import scipy.sparse as sp

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/analysis2/driven_linresp.json"
DT = 0.2; P_OFF = 1 - np.exp(-DT); A_EFF = P_OFF / DT
G_LIN = 0.16315274988719286          # results/pd/branch_real_dense_part0.json "g_lin"
HS = [1e-5, 1e-4, 1e-3, 1e-2]        # h = 0.1 left out: rho0 ~ 0.1, first order in rho is not controlled
# coupling values: every simulated drive-grid point below g_lin (stage 1 + pd2 grids) plus points close to g_lin
GS = sorted({0.1, 0.1202, 0.1445, 0.1091, 0.1191, 0.1299, 0.1417, 0.1547, 0.08, 0.158, 0.1615})
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:.0f}s]", *a, flush=True)

pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
keep = sign[pre] != 0
Wt = sp.csr_matrix((np.where(sign[pre[keep]] > 0, 1.0, -1.0) * c * w[keep], (post[keep], pre[keep])), shape=(N, N))
Wt.sum_duplicates(); Wt.sort_indices()
base = Wt.data.copy(); isexc = base > 0
log("graph", N, Wt.nnz)
F = lambda s: np.tanh(np.clip(s, 0, None))


def solve(g, h, variant, tol=1e-7, itmax=3000):
    q0 = 1 - np.exp(-F(h) * DT)
    if variant == "exact":
        a = np.exp(-F(h) * DT) - np.exp(-F(h + g * base) * DT)
    else:
        a = np.where(isexc, 1 - np.exp(-F(g * base) * DT), 0.0)
    A = sp.csr_matrix((a, Wt.indices, Wt.indptr), shape=(N, N))
    rho0 = q0 / (q0 + P_OFF)
    rho = np.full(N, rho0)
    for it in range(itmax):
        Q = np.clip(q0 + A @ rho, 0, None)
        new = Q / (Q + P_OFF)
        err = np.abs(new - rho).max() / rho0
        rho = new
        if err < tol:
            break
    return {"g": g, "h": h, "variant": variant, "rho0": float(rho0), "rho_bar": float(rho.mean()),
            "excess_rel": float(rho.mean() / rho0 - 1), "iters": it + 1, "converged": bool(err < tol)}


def perron(M, iters=6000, tol=1e-10):
    x = np.ones(N) / np.sqrt(N); lam = 0.0
    for it in range(iters):
        y = M @ x; lam_new = np.linalg.norm(y) / np.linalg.norm(x)
        x = x + y / lam_new; x /= np.linalg.norm(x)
        if it > 50 and abs(lam_new - lam) < tol * lam_new:
            break
        lam = lam_new
    return lam_new, x, it + 1


res = {"definitions": __doc__, "g_lin": G_LIN, "alpha_eff": A_EFF, "h": HS, "g": GS, "rows": []}
# leading-mode pole amplitude from the Perron vectors of B(g_lin)
Bd = sp.csr_matrix((np.where(isexc, (1 - np.exp(-F(G_LIN * base) * DT)) / DT, 0.0), Wt.indices, Wt.indptr), shape=(N, N))
lam, v, itv = perron(Bd)
lamT, u, itu = perron(Bd.T.tocsr())
v = np.abs(v); u = np.abs(u); u = u / (u @ v)
amp = float(v.sum() * u.sum() / N)
res["pole"] = {"lam_at_g_lin": float(lam), "lam_left": float(lamT), "iters": [itv, itu], "sum_v": float(v.sum()),
               "sum_u": float(u.sum()), "amp_sumv_sumu_over_N": amp,
               "inv_ipr_v": float(1 / ((v / np.linalg.norm(v)) ** 4).sum()),
               "note": "rho_bar/rho0 - 1 (pole part) = amp * x/(1-x), x = g/g_lin"}
log("pole: lam", lam, lamT, "amp", amp, "iters", itv, itu)
for g in GS:
    r = solve(g, 1e-5, "B_only"); res["rows"].append(r)
    log(f"g={g:.4f} B_only excess {r['excess_rel']:.4f} iters {r['iters']}")
    for h in HS:
        r = solve(g, h, "exact"); res["rows"].append(r)
        log(f"g={g:.4f} h={h:g} exact excess {r['excess_rel']:.4f} iters {r['iters']}")
    json.dump(res, open(OUT, "w"), indent=1)
log("saved", OUT)
