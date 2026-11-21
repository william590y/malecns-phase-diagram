"""pd3 shared helpers (phase diagram v3, workflow wf_98fb990d-4fb). Importable; no side effects on import.

Copies (NOT imports) of the verified helpers of code/analysis2_phase.py (that file has no __main__ guard):
  iso, first_cross, censored_mle, log_tau, log_tau_smooth, k_needed, A_EFF.
Plus:
  load_cache()                    -> pre, post, w, sign (read-only connectome cache)
  c_full()                        -> raw-normalisation constant of the full brain, 1 / mean in-strength (1/742.7)
  exc_csr_torch(...)              -> excitatory coupling matrix E (E_ij = c w_ij for excitatory pre j) on the GPU (CSR, float64)
  perron_t(M, x)                  -> Perron root / vector by the shifted power iteration of theory_branch_gpu.py
  g_lin_exact(E)                  -> exact g_lin by bisection on lambda_PF(B(g)) = alpha_eff, B_ij = f(g E_ij),
                                     f(x) = (1 - exp(-tanh(x) dt)) / dt (relative tolerance 1e-6 on g)
  module_idx()                    -> the 526 antennal-lobe module neurons (results/module/module_graph.npz idx)
"""
import numpy as np
from scipy import stats
from scipy.optimize import isotonic_regression
from scipy.special import logsumexp

CACHE = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
DT = 0.2
A_EFF = (1 - np.exp(-DT)) / DT


# ----------------------------------------------------------------------------------------- copies from analysis2_phase.py
def iso(y, w):
    return isotonic_regression(np.asarray(y, float), weights=np.asarray(w, float), increasing=True).x


def first_cross(x, y, thr, log_y=False):
    """First upward crossing of thr by y(x), x increasing; interpolated linearly in (ln x, y or ln y)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    yy = np.log(np.maximum(y, 1e-300)) if log_y else y
    tt = np.log(thr) if log_y else thr
    hit = np.where(yy >= tt)[0]
    if len(hit) == 0:
        return {"g": None, "bracket": [float(x[-1]), None], "status": "none"}
    k = int(hit[0])
    if k == 0:
        return {"g": None, "bracket": [None, float(x[0])], "status": "below_grid"}
    fr = (tt - yy[k - 1]) / (yy[k] - yy[k - 1]) if yy[k] > yy[k - 1] else 1.0
    g = float(np.exp(np.log(x[k - 1]) + fr * np.log(x[k] / x[k - 1])))
    return {"g": g, "bracket": [float(x[k - 1]), float(x[k])], "status": "ok", "k": k}


def censored_mle(times, died):
    """Censored-exponential MLE as in analysis2_phase.load_sim: tau = E/d; chi-square 95% CI; with d = 0: tau = inf and the
    one-sided 95% lower bound 2E/chi2_.95(2). Returns dict(deaths, E, tau, tau_reg (= E/max(d, 0.5)), tau_ci)."""
    times = np.asarray(times, float); died = np.asarray(died, bool)
    d, E = int(died.sum()), float(times.sum())
    if d > 0:
        ci = [2 * E / stats.chi2.ppf(0.975, 2 * d + 2), 2 * E / stats.chi2.ppf(0.025, 2 * d)]
        tau = E / d
    else:
        ci = [2 * E / stats.chi2.ppf(0.95, 2), float("inf")]
        tau = float("inf")
    return {"deaths": d, "E": E, "tau": tau, "tau_reg": E / max(d, 0.5), "tau_ci": ci}


def log_tau(lam, K, n_top=None):
    """Exact mean extinction time from the top state of the logistic birth-death chain (theory_branch_gpu.py)."""
    n_top = int(np.floor(K)) if n_top is None else int(n_top)
    if n_top < 1 or lam <= 0:
        return -np.inf
    n = np.arange(1, n_top + 1, dtype=float)
    lb = lam * n * (1 - n / K); lb[-1] = 0.0
    if (lb[:-1] <= 0).any():                       # n_top above K: drop non-positive birth rates (smoothK variant only)
        lb = np.where(lb > 0, lb, 0.0)
    ld = A_EFF * n
    logD = np.empty(n_top); logD[-1] = -np.log(ld[-1])
    for k in range(n_top - 2, -1, -1):
        logD[k] = (np.logaddexp(0.0, np.log(lb[k]) + logD[k + 1]) if lb[k] > 0 else 0.0) - np.log(ld[k])
    return float(logsumexp(logD))


def log_tau_smooth(lam, K):
    """Same chain, truncation level interpolated: (1 - f) * [top state floor(K)] + f * [top state floor(K) + 1], f = K - floor(K),
    same lam and K in both (the extra top state has its birth rate set to 0 as in the original). Continuous in K."""
    k0 = int(np.floor(K)); f = K - k0
    a = log_tau(lam, K, k0)
    b = log_tau(lam, K, k0 + 1) if f > 0 else a
    return float((1 - f) * a + f * b)


def k_needed(lam, lt_target, kmax=50000.0):
    """K for which the theory's own recursion (smooth in K) with the theory's lam(g) gives ln tau = lt_target."""
    if not np.isfinite(lam) or lam <= A_EFF:
        return None
    lo, hi = 1.01, 4.0
    while log_tau_smooth(lam, hi) < lt_target:
        hi *= 2
        if hi > kmax:
            return None
    if log_tau_smooth(lam, lo) > lt_target:
        return None
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if log_tau_smooth(lam, mid) < lt_target else (lo, mid)
        if hi / lo < 1 + 1e-5:
            break
    return float(np.sqrt(lo * hi))


# ------------------------------------------------------------------------------------------------------- data / graphs
def load_cache():
    pre = np.load(CACHE + "/pre.npy").astype(np.int64); post = np.load(CACHE + "/post.npy").astype(np.int64)
    w = np.load(CACHE + "/weight.npy").astype(np.float64); sign = np.load(CACHE + "/sign.npy")
    return pre, post, w, sign


def c_of(post, w, n):
    ins = np.bincount(post, weights=w, minlength=n)
    return float(1.0 / ins[ins > 0].mean())


def c_full():
    pre, post, w, sign = load_cache()
    return c_of(post, w, len(sign))


def module_idx():
    return np.load(f"{ROOT}/results/module/module_graph.npz")["idx"].astype(np.int64)


# ------------------------------------------------------------------------------------------------- spectral (torch, GPU)
def _f_torch(x, torch):
    return (1 - torch.exp(-torch.tanh(torch.clamp(x, min=0)) * DT)) / DT


def exc_csr_torch(pre, post, w, esign, c, n, dev="cuda"):
    """E (rows = post, cols = pre), excitatory edges only, duplicates summed; returns dict with CSR pieces and the transpose
    permutation, as in theory_branch_gpu.py. esign = per-edge presynaptic sign."""
    import torch
    import scipy.sparse as sp
    e = esign > 0
    Ex = sp.csr_matrix((c * w[e], (post[e], pre[e])), shape=(n, n)); Ex.sum_duplicates(); Ex.sort_indices()
    ExT = Ex.T.tocsr(); ExT.sort_indices()
    Ecoo = Ex.tocoo(); order = np.lexsort((Ecoo.row, Ecoo.col))
    assert np.allclose(ExT.data, Ex.data[order]) and (ExT.indices == Ecoo.row[order]).all()
    T = lambda a, dt=torch.float64: torch.tensor(a, device=dev, dtype=dt)
    return {"n": n, "crow": T(Ex.indptr, torch.int64), "col": T(Ex.indices, torch.int64), "base": T(Ex.data),
            "crowT": T(ExT.indptr, torch.int64), "colT": T(ExT.indices, torch.int64), "permT": T(order, torch.int64),
            "nnz": int(Ex.nnz)}


def B_mats(Ed, g):
    """Branching operator B(g) = f(g E) and its transpose (torch sparse CSR). g = None -> E itself (weak-synapse limit)."""
    import torch
    vals = Ed["base"] if g is None else _f_torch(g * Ed["base"], torch)
    n = Ed["n"]
    B = torch.sparse_csr_tensor(Ed["crow"], Ed["col"], vals, (n, n))
    BT = torch.sparse_csr_tensor(Ed["crowT"], Ed["colT"], vals[Ed["permT"]], (n, n))
    return B, BT


def perron_t(M, x=None, iters=20000, tol=1e-11):
    """Shifted power iteration (x <- x + Mx/lam), as in theory_branch_gpu.py / pd3_theory_nullK.py (float64)."""
    import torch
    n = M.shape[0]
    x = torch.ones(n, 1, device=M.device, dtype=torch.float64) if x is None else x.clone()
    x /= x.norm(); lam = 0.0; lam_new = 0.0
    for it in range(iters):
        y = torch.sparse.mm(M, x); lam_new = float(y.norm() / x.norm())
        x = x + y / lam_new; x /= x.norm()
        if it > 50 and abs(lam_new - lam) < tol * lam_new:
            break
        lam = lam_new
    return lam_new, x, it + 1


def k_nc(u, v):
    """u, v Perron left/right vectors (torch, any sign convention); returns K = 1/(<u^2,v><u,v^2>) with <u,v> = 1, and 1/IPR(v)."""
    u = u.abs().double().ravel(); v = v.abs().double().ravel()
    u = u / float((u * v).sum())
    K = 1.0 / (float(((u ** 2) * v).sum()) * float((u * v ** 2).sum()))
    p = v ** 2 / float((v ** 2).sum())
    return float(K), float(1.0 / float((p ** 2).sum()))


def g_lin_exact(Ed, lamE=None, rtol=1e-6, v0=None):
    """Exact g_lin: lambda_PF(f(g E)) = alpha_eff, by bisection in ln g between alpha_eff/lambda_PF(E) (B(g) <= g E
    elementwise, so g_lin >= the weak-synapse value) and 4x that value. Returns g_lin, the Perron vector at g_lin."""
    if lamE is None:
        lamE, v0, _ = perron_t(B_mats(Ed, None)[0])
    gw = A_EFF / lamE
    lo, hi = gw * (1 - 1e-9), gw * 4.0
    v = v0
    while hi / lo > 1 + rtol:
        mid = np.sqrt(lo * hi); lam, v, _ = perron_t(B_mats(Ed, mid)[0], v)
        lo, hi = (mid, hi) if lam < A_EFF else (lo, mid)
    return float(np.sqrt(lo * hi)), v


# ---------------------------------------------------------------------------------- vectorised chain (same mathematics)
def log_tau_fast(lam, K, n_top=None):
    """Vectorised closed form of log_tau (identical chain): D[k] = (1 + b_k D[k+1]) / d_k unrolled,
    log D[k] = -C[k] + logsumexp_{j >= k} (C[j] - log d_j), C[j] = sum_{i<j} log(b_i/d_i); log tau = logsumexp_k log D[k]."""
    n_top = int(np.floor(K)) if n_top is None else int(n_top)
    if n_top < 1 or lam <= 0:
        return -np.inf
    n = np.arange(1, n_top + 1, dtype=float)
    lb = lam * n * (1 - n / K); lb[-1] = 0.0
    ld = A_EFF * n
    with np.errstate(divide="ignore"):
        lr = np.where(lb[:-1] > 0, np.log(np.maximum(lb[:-1], 1e-300)) - np.log(ld[:-1]), -1e10)
    C = np.r_[0.0, np.cumsum(lr)]
    t = C - np.log(ld)
    rev = np.logaddexp.accumulate(t[::-1])[::-1]            # logsumexp_{j >= k} t_j
    logD = rev - C
    return float(logsumexp(logD))


def log_tau_smooth_fast(lam, K):
    k0 = int(np.floor(K)); f = K - k0
    a = log_tau_fast(lam, K, k0)
    b = log_tau_fast(lam, K, k0 + 1) if f > 0 else a
    return float((1 - f) * a + f * b)
