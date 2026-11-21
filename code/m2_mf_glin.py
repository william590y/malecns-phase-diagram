"""m2_mf_glin.py -- silent-state thresholds of the M2 operators (lane m2/theory, meanfield). GPU.

(1) g_lin: exact linearisation of the stochastic process about the RESTED silent state. Spike-mode depression gives a
    rested unit full efficacy, so the branching operator is B(g)_ij = f(g E_ij)/dt (E = positive part of the operator,
    f(s) = 1 - exp(-beta tanh(s) dt)) exactly as in M1; g_lin solves lambda_PF(B(g)) = alpha_eff (bisection, 30 steps).
    Computed for the operators of M2 (= M1 = M2_allsyn = KW09 = N15 = tau30), M2s (= M1s), and the two nulls (seed 0).
    Also the frozen-resource threshold for a uniform resource x: g_lin(x) = g_lin / x exactly (f depends on g x only);
    checked numerically at x = 0.5.
(2) mu(r): the rectified Perron root of the signed operator W(r) (x <- [W x]_+, shifted power iteration as
    theory_qmf.py) -> the QMF silent edge g_QMF(r) = (1 - e^{-dt}) / (dt mu(r)) = alpha_eff / mu(r), for R_ALL (real, M2s)
    and R_CORE (nulls).
Output: results/m2/theory/meanfield_glin.json
"""
import json, sys, time
import numpy as np
import torch

sys.path.insert(0, "/MALECNS_ROOT/code")
import m2_mf_core as M          # noqa: E402
import m2_common as MC          # noqa: E402

OUT = "/MALECNS_ROOT/results/m2/theory/meanfield_glin.json"
t0 = time.time()
log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
dev = "cuda"


def glin(mdl):
    one = torch.ones(mdl.N, device=dev, dtype=torch.float64)
    lo, hi = 0.02, 5.0
    v = None
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        lam, sg, v, _ = mdl.perron_branch(one, mid, v0=v, iters=20000, tol=1e-11)
        lo, hi = (mid, hi) if sg < 1 else (lo, mid)
    g = float(np.sqrt(lo * hi))
    lamE, _, vE, _ = mdl.perron_branch(one, 1e-6, iters=20000, tol=1e-11)     # ~ linear regime -> lambda_PF(E) * g
    lamE = lamE / 1e-6
    # localisation of the threshold mode
    vv = (v[:, 0] ** 2); vv = vv / vv.sum()
    mod = torch.tensor(mdl.reg_np["AL_module"], device=dev)
    return {"g_lin": g, "lambda_PF_E": float(lamE), "alpha_over_lambdaE": float(M.ALPHA / lamE),
            "mode_inv_IPR": float(1 / (vv ** 2).sum()), "mode_AL_module_share": float(vv[mod].sum())}


def mu_rect(mdl):
    x = torch.ones(mdl.N, 1, device=dev); x /= x.norm(); mu = 1.0; mu_new = 1.0
    for it in range(6000):
        y = torch.clamp(torch.sparse.mm(mdl.W, x), min=0)
        mu_new = float(y.norm() / x.norm())
        x = x + y / max(mu_new, 1e-12); x /= x.norm()
        if it > 50 and abs(mu_new - mu) < 1e-7 * mu_new:
            break
        mu = mu_new
    xv = x[:, 0] ** 2; xv /= xv.sum()
    mod = torch.tensor(mdl.reg_np["AL_module"], device=dev)
    return {"mu": mu_new, "g_QMF": M.ALPHA / mu_new, "iters": it, "inv_IPR": float(1 / (xv ** 2).sum()),
            "AL_module_share": float(xv[mod].sum())}


res = {"alpha_eff": M.ALPHA, "glin": {}, "mu": {}}
for name, var, null in [("real", "M2", None), ("M2s", "M2s", None), ("null_targets", "M2", "targets"),
                        ("null_weights", "M2", "weights")]:
    mdl = M.Model(var, 1.0, null=null, null_seed=0)
    res["glin"][name] = glin(mdl)
    log(name, res["glin"][name])
    if name == "real":       # frozen uniform resource check: g_lin(x = 0.5) = 2 g_lin
        half = torch.full((mdl.N,), 0.5, device=dev, dtype=torch.float64)
        lam, sg, _, _ = mdl.perron_branch(half, 2 * res["glin"]["real"]["g_lin"], iters=20000, tol=1e-11)
        res["glin"]["real"]["sigma_at_2glin_x0.5"] = sg
    rs = MC.R_ALL if null is None else MC.R_CORE
    res["mu"][name] = {}
    for r in rs:
        m_r = M.Model(var, r, null=null, null_seed=0)
        res["mu"][name][f"{r:g}"] = mu_rect(m_r)
        log(name, r, res["mu"][name][f"{r:g}"])
        del m_r; torch.cuda.empty_cache()
    del mdl; torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"), indent=1)
res["seconds"] = time.time() - t0
json.dump(res, open(OUT, "w"), indent=1)
log("saved", OUT)
