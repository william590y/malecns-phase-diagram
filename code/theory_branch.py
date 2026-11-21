"""Revised onset theory (see explainer/theory.html, sections 3-5): exact linearisation of the stochastic process around silence.

Expanding the switch-on rate over the set of active inputs (Moebius / inclusion-exclusion),
    F(g sum_{j in A} W~_ij) = sum_j a_j F(g W~_ij) + sum_{j<k} a_j a_k D^i_jk + ...,
the first-order (branching) operator is B_ij(g) = f(g E_ij) for excitatory j and 0 for inhibitory j: a lone inhibitory spike
onto a silent neuron does nothing, so inhibition is absent at linear order. Here f(x) = (1 - exp(-tanh(x)_+ dt)) / dt is the
per-step switch-on probability per unit time of the synchronous simulation, and alpha_eff = (1 - e^{-dt}) / dt.
  branching threshold:      lambda_PF(B(g)) = alpha_eff
  pair-corrected threshold: lambda_max(P(g)) = alpha_eff, with the pair (dynamical-correlation) operator
      P_ij = f_ij (alpha + f_ji/2) / (alpha + (f_ij + f_ji)/2),   P_ii = - sum_j f_ij f_ji / (2 alpha + f_ij + f_ji)
Second order (mode ansatz a ~ m v, left vector u, <u,v> = 1): drift A(m) = (lam - alpha) m - [lam <u,v^2> + D(g,r)] m^2, where
    D(g, r) = sum_i u_i sum_{j<k} |D^i_jk| v_j v_k     (coincidences: excitation cancelled by inhibition, f concavity)
is the only place r enters. Effective number of units K_eff = 1 / ( <u^2,v> (<u,v^2> + D/lam) ). Extinction time of the
logistic birth-death process b(n) = lam n (1 - n/K_eff), d(n) = alpha n, from the top state, by the exact recursion
D_N = 1/mu_N, D_n = (1 + lambda_n D_{n+1}) / mu_n, tau = sum D_n (log space). Sustained onset: tau(g) = T.
Usage: theory_branch.py <kind> <r,r,...> [FMAX] [variants] [suffix]   -> results/pd/branch_<kind><suffix>.json
"""
import json, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.special import logsumexp

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
kind, RS = sys.argv[1], [float(x) for x in sys.argv[2].split(",")]
FMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5          # g grid: g_lin .. FMAX * g_lin
VARIANTS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["branching", "pair"]
SUFFIX = sys.argv[5] if len(sys.argv) > 5 else ""
TS = [3000.0, 8000.0, 30000.0]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:.0f}s]", *a, flush=True)
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
rng = np.random.default_rng(0)                       # same permutations as swc.py --null [--null_kind weights]
if kind == "null":
    post = post[rng.permutation(len(post))]
elif kind == "wnull":
    w = w[rng.permutation(len(w))]
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
dt = 0.2; a_eff = (1 - np.exp(-dt)) / dt
exc, inh = sign[pre] > 0, sign[pre] < 0
Ex = sp.csr_matrix((c * w[exc], (post[exc], pre[exc])), shape=(N, N)); Ex.sum_duplicates(); Ex.sort_indices()
Ih = sp.csr_matrix((c * w[inh], (post[inh], pre[inh])), shape=(N, N)); Ih.sum_duplicates(); Ih.sort_indices()
f = lambda x: (1 - np.exp(-np.tanh(np.maximum(x, 0)) * dt)) / dt
rows = np.repeat(np.arange(N), np.diff(Ex.indptr)); cols = Ex.indices
key = rows.astype(np.int64) * N + cols                # sorted (CSR row-major, sorted indices)
rkey = cols.astype(np.int64) * N + rows
pos = np.minimum(np.searchsorted(key, rkey), len(key) - 1); has = key[pos] == rkey
lamE = sla.eigs(Ex, k=1, which="LR", tol=1e-8)[0][0].real
log(kind, "lambda_PF(E) =", lamE, " reciprocal exc edges:", has.mean())


def op(g, pair):
    fij = f(g * Ex.data)
    if not pair:
        return sp.csr_matrix((fij, Ex.indices, Ex.indptr), shape=(N, N))
    fji = np.where(has, fij[pos], 0.0)
    val = fij * (a_eff + fji / 2) / (a_eff + (fij + fji) / 2)
    diag = np.bincount(rows, weights=fij * fji / (2 * a_eff + fij + fji), minlength=N)
    return (sp.csr_matrix((val, Ex.indices, Ex.indptr), shape=(N, N)) - sp.diags(diag)).tocsr()


def perron(M, v0=None):
    vals, vecs = sla.eigs(M, k=1, which="LR", v0=v0, tol=1e-9, maxiter=200000)
    return vals[0].real, np.abs(vecs[:, 0].real)


def coincidence(g, r, u, v):
    """D(g, r) = sum_i u_i sum_{j<k} |D^i_jk| v_j v_k over excitatory-inhibitory and excitatory-excitatory input pairs."""
    ou = np.argsort(-u); cu = np.cumsum(u[ou]) / u.sum(); I_set = ou[: min(np.searchsorted(cu, 0.99) + 1, 20000)]
    ov = np.argsort(-v); cv = np.cumsum(v[ov]) / v.sum(); vmask = np.zeros(N, bool); vmask[ov[: np.searchsorted(cv, 0.99) + 1]] = True
    D = 0.0
    for i in I_set:
        je = Ex.indices[Ex.indptr[i]:Ex.indptr[i + 1]]; xe = Ex.data[Ex.indptr[i]:Ex.indptr[i + 1]]
        m = vmask[je]; je, xe = je[m], xe[m]
        if len(je) == 0:
            continue
        ve = v[je]; fe = f(g * xe)
        ki = Ih.indices[Ih.indptr[i]:Ih.indptr[i + 1]]; yi = Ih.data[Ih.indptr[i]:Ih.indptr[i + 1]]
        m = vmask[ki]; ki, yi = ki[m], yi[m]
        t = 0.0
        if len(ki) and r > 0:
            dei = f(g * xe[:, None] - g * r * yi[None, :]) - fe[:, None]
            t -= (dei * ve[:, None] * v[ki][None, :]).sum()
        if len(je) > 1:
            dee = f(g * (xe[:, None] + xe[None, :])) - fe[:, None] - fe[None, :]
            np.fill_diagonal(dee, 0.0)
            t -= 0.5 * (dee * np.outer(ve, ve)).sum()
        D += u[i] * t
    return D


def log_tau(lam, K):
    n_top = int(np.floor(K))
    if n_top < 1 or lam <= 0:
        return -np.inf
    n = np.arange(1, n_top + 1, dtype=float)
    lb = lam * n * (1 - n / K); lb[-1] = 0.0                 # reflecting top state
    ld = a_eff * n
    logD = np.empty(n_top); logD[-1] = -np.log(ld[-1])
    for k in range(n_top - 2, -1, -1):
        logD[k] = np.logaddexp(0.0, np.log(lb[k]) + logD[k + 1]) - np.log(ld[k])
    return float(logsumexp(logD))


res = {"kind": kind, "lambda_PF_E": lamE, "g_QMF_linear": a_eff / lamE, "alpha_eff": a_eff, "T": TS, "variants": {}}
for name in VARIANTS:
    pair = name == "pair"
    lo, hi = a_eff / lamE * 0.999, a_eff / lamE * 6
    v0 = None
    for _ in range(16):
        mid = np.sqrt(lo * hi); lam, v0 = perron(op(mid, pair), v0)
        lo, hi = (mid, hi) if lam < a_eff else (lo, mid)
    g_lin = np.sqrt(lo * hi)
    log(kind, name, "g_lin =", g_lin, "(linear-synapse limit", a_eff / lamE, ")")
    rows_out = []
    v = v0
    for g in g_lin * np.logspace(0, np.log10(FMAX), 28 if FMAX <= 3.5 else 40)[1:]:
        M = op(g, pair)
        lam, v = perron(M, v)
        lamL, u = perron(M.T.tocsr(), v)
        u = u / (u @ v)
        base = (u ** 2) @ v
        row = {"g": float(g), "lam": float(lam), "R": float(lam / a_eff), "uv2": float(u @ v ** 2), "u2v": float(base)}
        for r in RS:
            D = coincidence(g, r, u, v)
            K = 1 / (base * (u @ v ** 2 + D / lam))
            row[f"D_r{r:g}"] = float(D); row[f"K_r{r:g}"] = float(K); row[f"logtau_r{r:g}"] = log_tau(lam, K)
        rows_out.append(row)
        log(kind, name, f"g {g:.4f} R {lam/a_eff:.3f} " + " ".join(f"r{r:g}:K{row[f'K_r{r:g}']:.0f},lnT{row[f'logtau_r{r:g}']:.1f}" for r in RS))
    onset = {}
    for r in RS:
        lt = np.array([x[f"logtau_r{r:g}"] for x in rows_out]); gg = np.array([x["g"] for x in rows_out])
        onset[f"{r:g}"] = {f"T{int(T)}": (float(gg[np.argmax(lt >= np.log(T))]) if (lt >= np.log(T)).any() else None) for T in TS}
    res["variants"][name] = {"g_lin": float(g_lin), "rows": rows_out, "g_on": onset}
    log(kind, name, "predicted sustained onsets:", onset)
    json.dump(res, open(f"{ROOT}/results/pd/branch_{kind}{SUFFIX}.json", "w"), indent=1)
log("saved")
