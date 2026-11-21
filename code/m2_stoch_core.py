"""m2_stoch_core.py -- stochastic / spectral theory of Model 2 (lane m2/theory-stochastic, 2026-11-09).

Library used by code/m2_stoch_run.py. Plan: results/m2/theory/stochastic_plan.md; derivations results/m2/theory/stochastic.md.

Model (results/m2/MODEL2.md; ald_sim.py md5 61580c76, spike mode): per step dt = 0.2, an active unit turns off with
p_off = 1 - e^{-dt}; a quiescent unit turns on with p = 1 - exp(-beta tanh(s)_+ dt), s_i = g sum_j c w_ij sigma_j a_j e_j + h_i.
Depressing units carry a resource x: every step x <- 1 + (x - 1) q (q = e^{-dt/TAU}), then an onset sets e = x and
x <- (1 - U) x. The output during an epoch is a e.

Per-unit mean variables (exact for one unit at fixed input p; QMF closure across units):
    rho = E[a], O = E[a e], R1 = E[a x], R0 = E[(1 - a) x]
    rho' = (1 - p_off) rho + p (1 - rho)
    O'   = (1 - p_off) O + p A0,          A0 = q R0 + (1 - q)(1 - rho),   A1 = q R1 + (1 - q) rho
    R1'  = (1 - p_off) A1 + p (1 - U) A0
    R0'  = p_off A1 + (1 - p) A0
"""
import json, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.special import logsumexp

ROOT = "/MALECNS_ROOT"
CACHE = "/MALECNS_CACHE"
DT = 0.2
P_OFF = 1 - np.exp(-DT)
A_EFF = P_OFF / DT
BETA = 1.0
H_ORN = 0.037926

DEP = {"M1": None, "M2": (0.25, 157.0, "exc"), "M2_KW09": (0.28, 240.0, "exc"), "M2_N15": (0.22, 89.0, "exc"),
       "M2_tau30": (0.25, 30.0, "exc"), "M2_allsyn": (0.25, 157.0, "all")}

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ single unit
def unit_stationary(p, U, q, xp=np):
    """Stationary (rho, O, R1, R0) of one unit with constant onset probability p per step (arrays; U, q broadcast)."""
    rho = p / (p + P_OFF)
    a11 = 1 - (1 - P_OFF) * q; a12 = -p * (1 - U) * q
    a21 = -P_OFF * q;          a22 = 1 - (1 - p) * q
    b1 = (1 - P_OFF) * (1 - q) * rho + p * (1 - U) * (1 - q) * (1 - rho)
    b2 = P_OFF * (1 - q) * rho + (1 - p) * (1 - q) * (1 - rho)
    det = a11 * a22 - a12 * a21
    nd = (U <= 0) | (q >= 1)                  # non-depressing unit: x = 1 always
    det = xp.where(nd, xp.ones_like(det) if xp is not np else np.ones_like(det), det)
    R1 = (b1 * a22 - a12 * b2) / det
    R0 = (a11 * b2 - a21 * b1) / det
    R1 = xp.where(nd, rho, R1); R0 = xp.where(nd, 1 - rho, R0)
    A0 = q * R0 + (1 - q) * (1 - rho)
    O = p * A0 / P_OFF
    O = xp.where(nd, rho, O)
    return rho, O, R1, R0


def eff_spike(rho, U, TAU):
    """Renewal formula for the mean efficacy of an isolated unit (ald_dep_theory.eff_spike; independent check)."""
    rho = np.clip(rho, 1e-14, 0.999)
    p = np.minimum(P_OFF * rho / (1 - rho), 1.0)
    q = np.exp(-DT / TAU)
    gp = lambda pp, qq: pp * qq / (1 - (1 - pp) * qq)
    M = gp(P_OFF, q) * gp(p, q)
    return (1 - M) / (1 - (1 - U) * M)


def kappa(U, TAU):
    """-de/drho at rho -> 0 (exact discrete-time renewal): e = 1 - kappa rho + O(rho^2)."""
    q = np.exp(-DT / TAU)
    GL = P_OFF * q / (1 - (1 - P_OFF) * q)
    return U * GL * q * P_OFF / (1 - q)


def f_branch(x, xp=np):
    """Per-unit-time onset rate of a silent unit with input x (the branching kernel of theory_branch.py)."""
    if xp is np:
        return (1 - np.exp(-np.tanh(np.clip(x, 0, None)) * DT)) / DT
    return (1 - xp.exp(-xp.tanh(xp.clamp(x, min=0)) * DT)) / DT


# ------------------------------------------------------------------------------------------------ network
class Net:
    """Connectome (or a --graph subgraph) on the GPU: excitatory / inhibitory operators (rows = post, cols = pre),
    their transposes, region masks."""

    def __init__(self, graph=None, dev=None, model="M2", null_seed=None):
        dev = dev or os.environ.get("M2_DEV", "cuda")
        import torch
        self.torch, self.dev = torch, dev
        sign_all = np.load(f"{CACHE}/sign.npy")
        if graph is None:
            pre = np.load(f"{CACHE}/pre.npy").astype(np.int64); post = np.load(f"{CACHE}/post.npy").astype(np.int64)
            w = np.load(f"{CACHE}/weight.npy").astype(np.float64); sign = sign_all; sub = np.arange(len(sign_all))
            N = len(sign)
            ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
            self.kind = "real"
        else:
            d = np.load(graph, allow_pickle=True)
            pre, post, w, sign = d["pre"].astype(np.int64), d["post"].astype(np.int64), d["w"].astype(np.float64), d["sign"]
            sub = d["sub"]; c = float(d["c"]); N = len(sign)
            self.kind = str(d["kind"]) if "kind" in d.files else os.path.basename(graph)
        self.N, self.c, self.sub, self.sign = N, c, sub, sign
        exc, inh = sign[pre] > 0, sign[pre] < 0
        self.n_exc_edges, self.n_inh_edges = int(exc.sum()), int(inh.sum())
        Ex = sp.csr_matrix((c * w[exc], (post[exc], pre[exc])), shape=(N, N)); Ex.sum_duplicates(); Ex.sort_indices()
        Ih = sp.csr_matrix((c * w[inh], (post[inh], pre[inh])), shape=(N, N)); Ih.sum_duplicates(); Ih.sort_indices()
        self.Ex_sp, self.Ih_sp = Ex, Ih
        T = lambda a, dt=torch.float64: torch.tensor(a, device=dev, dtype=dt)

        def pack(M):
            Mc = M.tocsr(); Mc.sort_indices()
            crow, col = T(Mc.indptr, torch.int64), T(Mc.indices, torch.int64)
            row = T(np.repeat(np.arange(N), np.diff(Mc.indptr)), torch.int64)
            return {"crow": crow, "col": col, "row": row, "val": T(Mc.data)}
        self.E, self.ET = pack(Ex), pack(Ex.T)
        self.I, self.IT = pack(Ih), pack(Ih.T)
        self.exc_node = sign > 0; self.inh_node = sign < 0
        # labels (global ids via sub)
        cls = np.load(f"{CACHE}/classes.npy", allow_pickle=True).astype(str)[sub]
        sup = np.load(f"{CACHE}/superclasses.npy", allow_pickle=True).astype(str)[sub]
        typ = np.load(f"{CACHE}/types.npy", allow_pickle=True).astype(str)[sub]
        side = np.load(f"{CACHE}/sides.npy", allow_pickle=True).astype(str)[sub] if os.path.exists(f"{CACHE}/sides.npy") else None
        mod = np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)["idx"]
        gmod = np.zeros(len(sign_all), bool); gmod[mod] = True
        upn = np.load(f"{ROOT}/results/aldef/drive/groups/uPN_ant.npz")
        upn = upn[upn.files[0]]
        gupn = np.zeros(len(sign_all), bool); gupn[upn] = True
        self.masks = {
            "ORN": cls == "olfactory", "nonORN": cls != "olfactory", "AL_module": gmod[sub], "uPN_ant": gupn[sub],
            "lLN1_bc": typ == "lLN1_bc",
            "CB": np.char.startswith(sup, "cb_"), "VNC": np.char.startswith(sup, "vnc_"),
            "OL": np.char.startswith(sup, "ol_") | np.char.startswith(sup, "visual_"),
            "DN": sup == "descending_neuron", "AN": sup == "ascending_neuron",
        }
        self.side = side
        self.set_model(model)

    def set_model(self, model):
        torch = self.torch
        self.model = model
        dep = DEP[model]
        N = self.N
        U = np.zeros(N); TAU = np.full(N, np.inf)
        if dep is not None:
            u, tau, grp = dep
            m = self.exc_node if grp == "exc" else (self.exc_node | self.inh_node)
            U[m] = u; TAU[m] = tau
        q = np.where(np.isfinite(TAU), np.exp(-DT / np.where(np.isfinite(TAU), TAU, 1.0)), 1.0)
        self.U_np, self.TAU_np, self.q_np = U, TAU, q
        self.dep_node = U > 0
        self.U = torch.tensor(U, device=self.dev, dtype=torch.float64)
        self.q = torch.tensor(q, device=self.dev, dtype=torch.float64)
        self.depm = torch.tensor(self.dep_node, device=self.dev)
        self.Upar = dep[0] if dep else 0.0
        self.TAUpar = dep[1] if dep else np.inf

    def mv(self, M, x, vals=None):
        torch = self.torch
        A = torch.sparse_csr_tensor(M["crow"], M["col"], M["val"] if vals is None else vals, (self.N, self.N))
        if x.dim() == 1:
            return torch.mv(A, x)
        return torch.sparse.mm(A, x)

    def tmask(self, name):
        return self.torch.tensor(self.masks[name], device=self.dev)

    def hvec(self, h_orn):
        h = np.zeros(self.N)
        h[self.masks["ORN"]] = h_orn
        return self.torch.tensor(h, device=self.dev, dtype=self.torch.float64)


# ------------------------------------------------------------------------------------------------ QMF + resources
def qmf_input(net, g, r, rho, O):
    out = net.torch.where(net.depm, O, rho)
    return g * (net.mv(net.E, out) - r * net.mv(net.I, out))


def p_of_s(s, torch):
    return 1 - torch.exp(-BETA * torch.tanh(torch.clamp(s, min=0)) * DT)


def pi_trig(x, torch):
    """Probability that ONE active input of strength x switches a silent unit on within a step."""
    return 1 - torch.exp(-torch.tanh(torch.clamp(x, min=0)) * DT)


def bqmf_p(net, g, rho, O, hlam):
    """First-order branching ('noisy-OR') closure, excitation only: each active excitatory input j independently
    triggers i with probability pi(g c w_ij e_j) per step, e_j = O_j / rho_j; the drive is one more independent cause
    (hlam = tanh(h) dt). p_i = 1 - exp(-hlam_i) prod_j (1 - rho_j pi_ij). Exact to first order in the activity; inhibition
    acts only through coincidences (second order) and is not included."""
    torch = net.torch
    e = torch.where(net.depm, O / torch.clamp(rho, min=1e-300), torch.ones_like(rho))
    e = torch.where(rho > 0, e, torch.ones_like(rho))
    M = net.E
    vals = torch.log1p(-torch.clamp(rho[M["col"]] * pi_trig(g * M["val"] * e[M["col"]], torch), max=1 - 1e-15))
    A = torch.sparse_csr_tensor(M["crow"], M["col"], vals, (net.N, net.N))
    lam = -torch.mv(A, torch.ones(net.N, device=net.dev, dtype=torch.float64)) + hlam
    return 1 - torch.exp(-lam)


def bqmf_solve(net, g, hlam, rho0=None, O0=None, iters=6000, tol=1e-9, eta=0.5):
    """Fixed point of the stationary map with the branching closure (bqmf_p)."""
    torch = net.torch
    N = net.N
    rho = torch.full((N,), 0.5, device=net.dev, dtype=torch.float64) if rho0 is None else rho0.clone()
    O = rho.clone() if O0 is None else O0.clone()
    res = np.inf
    for it in range(iters):
        p = bqmf_p(net, g, rho, O, hlam)
        rn, On, R1, R0 = unit_stationary(p, net.U, net.q, torch)
        res = float((rn - rho).abs().sum() / max(float(rn.sum()), 1e-300))
        rho = (1 - eta) * rho + eta * rn
        O = (1 - eta) * O + eta * On
        if res < tol or float(rho.sum()) < 1e-12:
            break
    p = bqmf_p(net, g, rho, O, hlam)
    rn, On, R1, R0 = unit_stationary(p, net.U, net.q, torch)
    return {"rho": rn, "O": On, "R1": R1, "R0": R0, "p": p, "it": it + 1, "res": res}


def qmf_solve(net, g, r, h, rho0=None, O0=None, iters=6000, tol=1e-9, eta=0.5):
    """Damped iteration of the stationary QMF map. Returns dict(rho, O, R1, R0, s, p, it, res)."""
    torch = net.torch
    N = net.N
    rho = torch.full((N,), 0.5, device=net.dev, dtype=torch.float64) if rho0 is None else rho0.clone()
    O = rho.clone() if O0 is None else O0.clone()
    res = np.inf
    for it in range(iters):
        s = qmf_input(net, g, r, rho, O) + h
        p = p_of_s(s, torch)
        rn, On, R1, R0 = unit_stationary(p, net.U, net.q, torch)
        dr = (rn - rho).abs().sum() / max(float(rn.sum()), 1e-300)
        rho = (1 - eta) * rho + eta * rn
        O = (1 - eta) * O + eta * On
        res = float(dr)
        if res < tol or float(rho.sum()) < 1e-12:
            break
    s = qmf_input(net, g, r, rho, O) + h
    p = p_of_s(s, torch)
    rn, On, R1, R0 = unit_stationary(p, net.U, net.q, torch)
    return {"rho": rn, "O": On, "R1": R1, "R0": R0, "s": s, "p": p, "it": it + 1, "res": res}


def qmf_dynamics(net, g, r, h, steps, rec_every=5, init="decay", closure="qmf"):
    """Deterministic mean map from a fully active start (decay: a = 1, e = 1, x = 1 - U) -> trajectory of the mean
    variables (records rho_tot(t), and the full state at the end)."""
    torch = net.torch
    N = net.N
    if init == "decay":
        rho = torch.ones(N, device=net.dev, dtype=torch.float64)
        O = rho.clone()
        R1 = torch.where(net.depm, 1 - net.U, torch.ones_like(rho))
        R0 = torch.zeros_like(rho)
    else:
        rho = torch.zeros(N, device=net.dev, dtype=torch.float64); O = rho.clone(); R1 = rho.clone(); R0 = torch.ones_like(rho)
    q, U = net.q, net.U
    rec_t, rec = [], []
    for st in range(1, steps + 1):
        if closure == "bqmf":
            p = bqmf_p(net, g, rho, O, h)
        else:
            s = qmf_input(net, g, r, rho, O) + h
            p = p_of_s(s, torch)
        A1 = q * R1 + (1 - q) * rho
        A0 = q * R0 + (1 - q) * (1 - rho)
        rho_n = (1 - P_OFF) * rho + p * (1 - rho)
        O_n = (1 - P_OFF) * O + p * A0
        R1_n = (1 - P_OFF) * A1 + p * (1 - U) * A0
        R0_n = P_OFF * A1 + (1 - p) * A0
        rho, O, R1, R0 = rho_n, torch.where(net.depm, O_n, rho_n), R1_n, R0_n
        if st % rec_every == 0:
            rec_t.append(st * DT); rec.append(rho)
    return rec_t, rec, {"rho": rho, "O": O, "R1": R1, "R0": R0}


# ------------------------------------------------------------------------------------------------ branching operator
def perron(net, g, e=None, transpose=False, x0=None, iters=30000, tol=1e-11):
    """Perron root and vector of B~ = f(g E diag(e)) (or its transpose) by shifted power iteration."""
    torch = net.torch
    M = net.ET if transpose else net.E
    if e is None:
        vals = f_branch(g * M["val"], torch)
    else:
        idx = M["row"] if transpose else M["col"]           # presynaptic index of each stored entry
        vals = f_branch(g * M["val"] * e[idx], torch)
    A = torch.sparse_csr_tensor(M["crow"], M["col"], vals, (net.N, net.N))
    x = torch.ones(net.N, device=net.dev, dtype=torch.float64) if x0 is None else x0.clone().abs() + 1e-30
    x = x / x.norm(); lam = 0.0; lam_new = 0.0
    for it in range(iters):
        y = torch.mv(A, x); lam_new = float(y.norm() / x.norm())
        if lam_new == 0:
            return 0.0, x, it
        x = x + y / lam_new; x = x / x.norm()
        if it > 50 and abs(lam_new - lam) < tol * lam_new:
            break
        lam = lam_new
    return lam_new, x, it


def renorm_mode(net, g, e, rho=None, v0=None, u0=None):
    """Renormalised (frozen-resource) branching mode at efficacies e: lambda~, v~, u~ (<u, v> = 1), lambda_rest, K_0,
    n (mode units) of the activity rho, PR / AL share of v~."""
    torch = net.torch
    lam, v, itv = perron(net, g, e, False, v0)
    _, u, itu = perron(net, g, e, True, u0 if u0 is not None else v)
    u = u / float((u * v).sum())
    B0 = torch.sparse_csr_tensor(net.E["crow"], net.E["col"], f_branch(g * net.E["val"], torch), (net.N, net.N))
    lam_rest = float((u * torch.mv(B0, v)).sum())
    u2v = float((u * u * v).sum()); uv2 = float((u * v * v).sum())
    out = {"lam": lam, "sigma": lam / A_EFF, "lam_rest": lam_rest, "X": lam / lam_rest if lam_rest > 0 else np.nan,
           "u2v": u2v, "uv2": uv2, "K0": 1 / (u2v * uv2), "it_v": itv, "it_u": itu,
           "Ks": float(v.sum()) / uv2 if uv2 > 0 else np.nan}
    vv = (v * v).cpu().numpy()
    out["PR_v"] = float(1 / (vv ** 2).sum() * (vv.sum() ** 2))
    out["AL_v"] = float(vv[net.masks["AL_module"]].sum() / vv.sum())
    if rho is not None:
        m = float((u * rho).sum())
        out["n_mode"] = m / u2v
    return out, v, u


# ------------------------------------------------------------------------------------------------ chains
def log_tau_chain(lam_n, K, n_top, alpha=A_EFF):
    """Exact mean extinction time (log) from the top state n_top of b(n) = lam_n[n-1] n (1 - n/K), d = alpha n."""
    if n_top < 1:
        return -np.inf
    n = np.arange(1, n_top + 1, dtype=float)
    lb = lam_n[:n_top] * n * (1 - n / K); lb[-1] = 0.0
    lb = np.where(lb > 0, lb, 0.0)
    ld = alpha * n
    logD = np.empty(n_top); logD[-1] = -np.log(ld[-1])
    for k in range(n_top - 2, -1, -1):
        logD[k] = (np.logaddexp(0.0, np.log(lb[k]) + logD[k + 1]) if lb[k] > 0 else 0.0) - np.log(ld[k])
    return float(logsumexp(logD))


def log_tau_smooth(lam_fun, K, alpha=A_EFF, cap=200000):
    """Top-state extinction time with the truncation level interpolated between floor(K) and floor(K) + 1 (as analysis2)."""
    K = float(K)
    if not np.isfinite(K) or K < 1:
        return -np.inf
    if K > cap:
        return np.inf
    k0 = int(np.floor(K)); f = K - k0
    n = np.arange(1, k0 + 2, dtype=float)
    lam_n = lam_fun(n)
    a = log_tau_chain(lam_n, K, k0, alpha)
    b = log_tau_chain(lam_n, K, k0 + 1, alpha) if f > 0 else a
    return float((1 - f) * a + f * b)


def _log_taus(lb, ld):
    """log tau_n = log sum_{j>=n} (1/d_j) prod_{l=n}^{j-1} b_l/d_l for n = 1..len (vectorised, log space)."""
    with np.errstate(divide="ignore"):
        lr = np.log(lb) - np.log(ld)
    L = np.concatenate([[0.0], np.cumsum(lr[:-1])])
    Aj = L - np.log(ld)
    Rn = np.logaddexp.accumulate(Aj[::-1])[::-1]
    return Rn - L


def log_tau_logistic_qs(lam_fun, K, n_start, alpha=A_EFF):
    """Mean extinction time (log) from state n_start of b(n) = lam(n) n (1 - n/K), d = alpha n; top = floor(K)
    (reflecting): T(m) = sum_{n<=m} tau_n."""
    K = float(K); k0 = int(np.floor(K))
    if k0 < 1:
        return -np.inf
    n = np.arange(1, k0 + 1, dtype=float)
    lb = lam_fun(n) * n * (1 - n / K); lb[-1] = 0.0; lb = np.where(lb > 0, lb, 0.0)
    ld = alpha * n
    lt = _log_taus(lb, ld)
    m = int(np.clip(round(n_start), 1, k0))
    return float(logsumexp(lt[:m]))


def tau_2d(lam_r, K, Ny, U, TAU, n_star, y_star, alpha=A_EFF, M=240, n_max=None, y_lo=None):
    """Mean extinction time of the piecewise-deterministic 2-D chain (n, y):
         n -> n + 1 at b = lam_r y n (1 - n/K),  n -> n - 1 at d = alpha n,
         dy/dt = (1 - y)/TAU - U y b / Ny   (resource recovery; depletion by births as a drift),
       from (n_star, y_star). Upwind finite differences in y (uniform grid on [y_lo, 1]), sparse direct solve.
       Returns (tau, info)."""
    if n_max is None:
        n_max = int(min(np.floor(K), max(3 * n_star + 40, n_star + 12 * np.sqrt(max(n_star, 1)) + 40)))
    n_max = max(int(n_max), 2)
    if y_lo is None:
        y_lo = max(1e-3, 0.15 * y_star)
    ys = np.linspace(y_lo, 1.0, M); dy = ys[1] - ys[0]
    nn = np.arange(1, n_max + 1, dtype=float)
    Nn = n_max
    idx = lambda n, j: (n - 1) * M + j       # n = 1..n_max
    b = lam_r * ys[None, :] * nn[:, None] * np.clip(1 - nn[:, None] / K, 0, None)
    b[-1, :] = 0.0
    d = alpha * nn[:, None] * np.ones((1, M))
    v = (1 - ys[None, :]) / TAU - U * ys[None, :] * b / Ny
    rows, cols, vals = [], [], []
    diag = -(b + d)
    I = np.arange(Nn * M).reshape(Nn, M)
    # birth
    m = b > 0
    rows.append(I[:-1][m[:-1]]); cols.append(I[1:][m[:-1]]); vals.append(b[:-1][m[:-1]])
    # death (n >= 2; n = 1 death goes to absorbing T = 0)
    rows.append(I[1:].ravel()); cols.append(I[:-1].ravel()); vals.append(d[1:].ravel())
    # drift (upwind)
    vp = np.where(v > 0, v, 0.0) / dy; vm = np.where(v < 0, -v, 0.0) / dy
    vp[:, -1] = 0.0; vm[:, 0] = 0.0           # no flux through the y boundaries
    rows.append(I[:, :-1].ravel()); cols.append(I[:, 1:].ravel()); vals.append(vp[:, :-1].ravel())
    rows.append(I[:, 1:].ravel()); cols.append(I[:, :-1].ravel()); vals.append(vm[:, 1:].ravel())
    diag = diag - vp - vm
    rows.append(I.ravel()); cols.append(I.ravel()); vals.append(diag.ravel())
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(Nn * M, Nn * M))
    Tsol = sla.spsolve(A.tocsc(), -np.ones(Nn * M))
    Tsol = Tsol.reshape(Nn, M)
    ns = int(np.clip(round(n_star), 1, Nn)); js = int(np.clip(round((y_star - y_lo) / dy), 0, M - 1))
    return float(Tsol[ns - 1, js]), {"n_max": Nn, "M": M, "y_lo": y_lo, "T_top_rest": float(Tsol[-1, -1]),
                                     "T_n1_y1": float(Tsol[0, -1]), "T_ns_y1": float(Tsol[ns - 1, -1])}


def lifetimes(par, U, TAU, alpha=A_EFF, do2d=True, cap2d=600000):
    """Frozen / adiabatic / 2-D lifetimes from the renormalised-mode parameters
    par = dict(lam (lambda~), lam_rest, X, n_star, K)."""
    lam, lam_r, X, ns, K = par["lam"], par["lam_rest"], par["X"], par["n_star"], par["K"]
    out = {}
    if not (ns > 0 and K > ns and lam > alpha):
        return {"lt_F": -np.inf, "lt_A": -np.inf, "lt_2D": -np.inf, "Ny": np.nan}
    if U > 0 and X < 1:
        Ny = U * TAU * alpha * X * ns / (1 - X)
    else:
        Ny = np.inf
    out["Ny"] = Ny
    out["lt_F"] = log_tau_logistic_qs(lambda n: lam * np.ones_like(n), K, ns, alpha)
    if np.isfinite(Ny):
        def yad(n):          # exact fast-resource manifold of the 2-D chain: (1 - y) = U TAU lam_r y^2 n (1 - n/K) / Ny
            cc = U * TAU * lam_r * n * np.clip(1 - n / K, 0, None) / Ny
            return np.where(cc > 1e-12, (-1 + np.sqrt(1 + 4 * cc)) / (2 * np.maximum(cc, 1e-12)), 1.0)
        out["lt_A"] = log_tau_logistic_qs(lambda n: lam_r * yad(n), K, ns, alpha)
    else:
        out["lt_A"] = out["lt_F"]
    if do2d and np.isfinite(Ny):
        # skip the 2-D solve where both brackets are far from the 200..3000 window or the system is too large
        n_max = int(min(np.floor(K), max(3 * ns + 40, ns + 12 * np.sqrt(ns) + 40)))
        if out["lt_A"] < np.log(20) or out["lt_F"] > np.log(1e6) or n_max * 240 > cap2d:
            out["lt_2D"] = np.nan; out["info2d"] = {"skipped": True, "n_max": n_max}
        else:
            t, info = tau_2d(lam_r, K, Ny, U, TAU, ns, X, alpha, M=240)
            lt1 = float(np.log(t)); info["lt_M240"] = lt1
            if n_max > 80 and n_max * 480 <= 2 * cap2d:
                t2, _ = tau_2d(lam_r, K, Ny, U, TAU, ns, X, alpha, M=480)
                lt2 = float(np.log(t2)); info["lt_M480"] = lt2
                out["lt_2D"] = 2 * lt2 - lt1          # Richardson (first-order upwind scheme)
            else:
                out["lt_2D"] = lt1
            out["info2d"] = info
    elif not np.isfinite(Ny):
        out["lt_2D"] = out["lt_F"]
    return out


def cross(g, lt, thr):
    """First upward crossing of ln(thr) by lt(g), interpolated in (ln g, ln tau); None if never."""
    g = np.asarray(g, float); lt = np.asarray(lt, float)
    ok = np.isfinite(lt) | (lt == np.inf)
    y = np.log(thr)
    hit = np.where(ok & (lt >= y))[0]
    if len(hit) == 0:
        return None
    k = hit[0]
    if k == 0 or not np.isfinite(lt[k - 1]):
        return float(g[k])
    a, b = lt[k - 1], (lt[k] if np.isfinite(lt[k]) else y + 50)
    f = (y - a) / (b - a)
    return float(np.exp(np.log(g[k - 1]) + f * np.log(g[k] / g[k - 1])))


# ------------------------------------------------------------------------------------------------ linear-noise approx.
class LNA:
    """Linearisation of the one-step mean map at a fixed point st (dict rho, O, R1, R0, s, p) of the QMF + resources.
    State z = (z_rho, z_O, z_1, z_0), each an [N, k] block (k observables at once); the O / R1 / R0 components of
    non-depressing units are identically 0."""

    def __init__(self, net, g, r, st, dtype=None):
        torch = net.torch
        self.net, self.g, self.r = net, g, r
        dt_ = dtype or torch.float64
        self.dt_ = dt_
        cv = lambda x: x.to(dt_)
        rho, O, R1, R0, s, p = (cv(st[k]) for k in ("rho", "O", "R1", "R0", "s", "p"))
        q, U = cv(net.q), cv(net.U)
        dep = net.depm.to(dt_)
        self.dep = dep[:, None]; self.ndep = (1 - dep)[:, None]
        self.rho, self.p = rho, p
        A0 = q * R0 + (1 - q) * (1 - rho)
        # local coefficients (columns broadcast)
        c = lambda x: x[:, None]
        self.L_rr = c(1 - P_OFF - p)
        self.L_Or, self.L_OO, self.L_O0 = c(-p * (1 - q)), (1 - P_OFF), c(p * q)
        self.L_1r, self.L_11, self.L_10 = c((1 - P_OFF) * (1 - q) - p * (1 - U) * (1 - q)), c((1 - P_OFF) * q), c(p * (1 - U) * q)
        self.L_0r, self.L_01, self.L_00 = c(P_OFF * (1 - q) - (1 - p) * (1 - q)), c(P_OFF * q), c((1 - p) * q)
        self.b_r, self.b_O, self.b_1, self.b_0 = c(1 - rho), c(A0), c((1 - U) * A0), c(-A0)
        sech2 = 1 - torch.tanh(torch.clamp(s, min=0)) ** 2
        self.pp = c(torch.where(s > 0, (1 - p) * BETA * DT * sech2, torch.zeros_like(s)))
        # sparse operators in the working dtype
        mk = lambda M: torch.sparse_csr_tensor(M["crow"], M["col"], M["val"].to(dt_), (net.N, net.N))
        self.E, self.I, self.ET, self.IT = mk(net.E), mk(net.I), mk(net.ET), mk(net.IT)
        # noise vectors d1 (active: stay - off), d0 (quiescent: on - stay), variances vA, vQ
        e = torch.where(rho > 0, O / torch.clamp(rho, min=1e-300), torch.ones_like(rho))
        x1 = torch.where(rho > 0, R1 / torch.clamp(rho, min=1e-300), torch.ones_like(rho))
        x0 = torch.where(rho < 1, R0 / torch.clamp(1 - rho, min=1e-300), torch.ones_like(rho))
        y1 = q * x1 + 1 - q; y0 = q * x0 + 1 - q
        self.vA = c(rho * P_OFF * (1 - P_OFF)); self.vQ = c((1 - rho) * p * (1 - p))
        self.d1 = [torch.ones_like(self.vA), c(e) * self.dep, c(y1) * self.dep, -c(y1) * self.dep]
        self.d0 = [torch.ones_like(self.vA), c(y0) * self.dep, c((1 - U) * y0) * self.dep, -c(y0) * self.dep]

    def _W(self, x):            # signed operator on an [N, k] block
        torch = self.net.torch
        return self.g * (torch.sparse.mm(self.E, x) - self.r * torch.sparse.mm(self.I, x))

    def _WT(self, x):
        torch = self.net.torch
        return self.g * (torch.sparse.mm(self.ET, x) - self.r * torch.sparse.mm(self.IT, x))

    def A(self, z):
        zr, zO, z1, z0 = z
        out = self.dep * zO + self.ndep * zr
        dp = self.pp * self._W(out)
        nr = self.L_rr * zr + self.b_r * dp
        nO = (self.L_Or * zr + self.L_OO * zO + self.L_O0 * z0 + self.b_O * dp) * self.dep
        n1 = (self.L_1r * zr + self.L_11 * z1 + self.L_10 * z0 + self.b_1 * dp) * self.dep
        n0 = (self.L_0r * zr + self.L_01 * z1 + self.L_00 * z0 + self.b_0 * dp) * self.dep
        return [nr, nO, n1, n0]

    def AT(self, w):
        wr, wO, w1, w0 = w
        wO, w1, w0 = wO * self.dep, w1 * self.dep, w0 * self.dep
        nr = self.L_rr * wr + self.L_Or * wO + self.L_1r * w1 + self.L_0r * w0
        nO = self.L_OO * wO
        n1 = self.L_11 * w1 + self.L_01 * w0
        n0 = self.L_O0 * wO + self.L_10 * w1 + self.L_00 * w0
        y = self._WT(self.pp * (self.b_r * wr + self.b_O * wO + self.b_1 * w1 + self.b_0 * w0))
        return [nr + self.ndep * y, (nO + self.dep * y) * self.dep, n1 * self.dep, n0 * self.dep]

    def Q(self, w):
        a = sum(di * wi for di, wi in zip(self.d1, w))
        b = sum(di * wi for di, wi in zip(self.d0, w))
        return [self.vA * a * d1 + self.vQ * b * d0 for d1, d0 in zip(self.d1, self.d0)]

    def wQw(self, w):           # per column
        a = sum(di * wi for di, wi in zip(self.d1, w))
        b = sum(di * wi for di, wi in zip(self.d0, w))
        return (self.vA * a * a + self.vQ * b * b).sum(0)

    @staticmethod
    def dot(a, b):
        return sum((x * y).sum(0) for x, y in zip(a, b))

    @staticmethod
    def add(a, b, s=1.0):
        return [x + s * y for x, y in zip(a, b)]

    def zero(self, k):
        torch = self.net.torch
        return [torch.zeros(self.net.N, k, device=self.net.dev, dtype=self.dt_) for _ in range(4)]

    def obs_activity(self, masks):
        """Observable vectors c = 1_X / n_X on the rho components (one column per mask)."""
        torch = self.net.torch
        k = len(masks)
        c = self.zero(k)
        for j, m in enumerate(masks):
            mt = torch.tensor(m, device=self.net.dev, dtype=self.dt_)
            c[0][:, j] = mt / mt.sum()
        return c

    def obs_onsets(self, mask):
        """Gradient of the expected onset count of the units in mask (per step) and the same-step state/onset noise
        cross-covariance q_on, and the Bernoulli variance of the count."""
        torch = self.net.torch
        mt = torch.tensor(mask, device=self.net.dev, dtype=self.dt_)[:, None]
        c = self.zero(1)
        c[0] = -self.p[:, None] * mt
        y = self._WT(self.b_r * self.pp * mt)      # (1 - rho) p' on the masked units, pulled back through the input
        c[0] = c[0] + self.ndep * y
        c[1] = c[1] + self.dep * y
        qon = [self.vQ * d * mt for d in self.d0]
        var_b = float((self.vQ * mt).sum())
        return c, qon, var_b

    def run(self, c, smax=40000, rtol=1e-10, ckpt=200, maxlag=2500, qon=None, log_every=0):
        """Var_j = sum_s w_s^T Q w_s, Cc = sum_s A^s Q w_s (checkpointed reverse Horner pass), ACF_j(k) = c_j^T A^k (C c)_j
        for k = 0..maxlag (steps), and, if qon is given, x_j(k) = c_j^T A^(k-1) qon (k >= 1)."""
        torch = self.net.torch
        k = c[0].shape[1]
        w = [x.clone() for x in c]
        var = torch.zeros(k, device=self.net.dev, dtype=self.dt_)
        cps = [];  s = 0; contrib_hist = []
        while s < smax:
            if s % ckpt == 0:
                cps.append([x.clone() for x in w])
            t = self.wQw(w); var = var + t
            contrib_hist.append(float(t.max()))
            s += 1
            if s > 200 and float((t / torch.clamp(var, min=1e-300)).max()) < rtol:
                break
            w = self.AT(w)
            if log_every and s % log_every == 0:
                log(f"  lna fwd s={s} rel={float((t / var).max()):.2e}")
        S = s
        # reverse Horner: acc = sum_{s<S} A^s Q w_s
        acc = self.zero(k)
        for bi in range(len(cps) - 1, -1, -1):
            s0 = bi * ckpt; s1 = min(S, s0 + ckpt)
            blk = [cps[bi]]
            for _ in range(s0 + 1, s1):
                blk.append(self.AT(blk[-1]))
            for j in range(s1 - 1, s0 - 1, -1):
                acc = self.add(self.Q(blk[j - s0]), self.A(acc))
            del blk
        Cc = acc
        acf = torch.zeros(maxlag + 1, k, device=self.net.dev, dtype=self.dt_)
        v = [x.clone() for x in Cc]
        for kk in range(maxlag + 1):
            acf[kk] = self.dot(c, v)
            v = self.A(v)
        xq = None
        if qon is not None:
            xq = torch.zeros(maxlag + 1, k, device=self.net.dev, dtype=self.dt_)
            v = [x.clone() for x in qon]
            for kk in range(1, maxlag + 1):
                xq[kk] = self.dot(c, v)
                v = self.A(v)
        return {"var": var.cpu().numpy(), "acf": acf.cpu().numpy(), "xq": None if xq is None else xq.cpu().numpy(),
                "S": S, "converged": S < smax}


def acf_metrics(C, dt=DT):
    """m2_obs rules on an exact (normalised) ACF C[0..]: tau_int, tau_e, trough/peak/period/coherence, OSC flags."""
    import sys
    sys.path.insert(0, f"{ROOT}/code")
    import m2_obs as MO
    c = np.asarray(C, float) / C[0]
    ti, te = MO.tau_ac(c, dt)
    out = {"tau_int": ti, "tau_e": te}
    lag_min = c_min = period = c_peak = np.nan
    gmin = np.nanmin(c[1:])
    if gmin < 0:
        d = np.diff(c)
        mins = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
        deep = mins[c[mins] <= 0.5 * gmin]
        if len(deep):
            k0 = int(deep[0]); lag_min, c_min = k0 * dt, c[k0]
            hi = min(3 * k0, len(c) - 1)
            if hi > k0:
                kp = k0 + 1 + int(np.argmax(c[k0 + 1:hi + 1]))
                period, c_peak = kp * dt, c[kp]
    coh = np.nan
    if np.isfinite(period) and c_peak > 0:
        kp = int(round(period / dt)); lo, hi = 2 * kp - kp // 4, min(2 * kp + kp // 4, len(c) - 1)
        if hi > lo:
            coh = float(np.max(c[lo:hi + 1]) / c_peak)
    out.update(lag_min=float(lag_min), c_min=float(c_min), period=float(period), c_peak=float(c_peak), coh_ratio=coh)
    out["osc"] = bool(np.isfinite(c_peak) and c_min <= MO.OSC_NEG and c_peak >= MO.OSC_PEAK)
    out["osc_strong"] = bool(out["osc"] and np.isfinite(coh) and coh >= MO.OSC_STRONG)
    return out


def mr_from_cov(cov_k, var0, kmax=20):
    """Multistep-regression m from exact lagged covariances (r_k = cov_k / var0, k = 1..kmax steps; fit r_k = b m^k)."""
    ks = np.arange(1, kmax + 1)
    r = np.asarray(cov_k[1:kmax + 1], float) / var0
    ok = r > 0
    if ok.sum() < 3:
        return np.nan, r
    slope = np.polyfit(ks[ok], np.log(r[ok]), 1)[0]
    return float(np.exp(slope)), r


def perron_masked(net, g, e, mask, transpose=False, x0=None, iters=30000, tol=1e-11):
    """Perron pair of D f(g E diag(e)) D (D = diag(mask)): the operator restricted to the unmasked neurons."""
    torch = net.torch
    M = net.ET if transpose else net.E
    idx = M["row"] if transpose else M["col"]
    em = e * mask
    vals = f_branch(g * M["val"] * em[idx], torch)
    A = torch.sparse_csr_tensor(M["crow"], M["col"], vals, (net.N, net.N))
    x = (torch.ones(net.N, device=net.dev, dtype=torch.float64) if x0 is None else x0.clone().abs() + 1e-30) * mask
    x = x / x.norm(); lam = 0.0; lam_new = 0.0
    for it in range(iters):
        y = torch.mv(A, x) * mask; lam_new = float(y.norm() / x.norm())
        if lam_new == 0:
            return 0.0, x, it
        x = x + y / lam_new; x = x / x.norm()
        if it > 50 and abs(lam_new - lam) < tol * lam_new:
            break
        lam = lam_new
    return lam_new, x, it


def peel_islands(net, g, e, rho, n_isl=6, mass=0.99):
    """Successive renormalised modes on disjoint supports ('islands'): mode k is the Perron mode of the frozen operator
    restricted to the neurons outside the 99%-mass supports of modes 0..k-1."""
    torch = net.torch
    mask = torch.ones(net.N, device=net.dev, dtype=torch.float64)
    B0vals = f_branch(g * net.E["val"], torch)
    B0 = torch.sparse_csr_tensor(net.E["crow"], net.E["col"], B0vals, (net.N, net.N))
    out = []
    sup = np.load(f"{CACHE}/superclasses.npy", allow_pickle=True).astype(str)[net.sub]
    for k in range(n_isl):
        lam, v, _ = perron_masked(net, g, e, mask)
        _, u, _ = perron_masked(net, g, e, mask, True, v)
        if lam <= 0 or float((u * v).sum()) <= 0:
            break
        u = u / float((u * v).sum())
        lam_rest = float((u * (torch.mv(B0, v * mask) * mask)).sum())
        u2v = float((u * u * v).sum()); uv2 = float((u * v * v).sum())
        vv = (v * v).cpu().numpy()
        o = np.argsort(-vv); cs = np.cumsum(vv[o]) / vv.sum(); nsup = int(np.searchsorted(cs, mass) + 1)
        supp = o[:nsup]
        rr = rho.cpu().numpy()
        row = {"k": k, "lam": lam, "sigma": lam / A_EFF, "lam_rest": lam_rest, "X": lam / lam_rest if lam_rest > 0 else np.nan,
               "K0": 1 / (u2v * uv2), "u2v": u2v, "PR_v": float(vv.sum() ** 2 / (vv ** 2).sum()),
               "AL_v": float(vv[net.masks["AL_module"]].sum() / vv.sum()), "n_support": nsup,
               "n_mode": float((u * rho).sum()) / u2v, "rho_support": float(rr[supp].sum()),
               "top_superclass": str(sup[o[0]]),
               "sup_share": {c: float(vv[np.char.startswith(sup, c)].sum() / vv.sum()) for c in ("cb_", "vnc_", "ol_", "visual_", "descending", "ascending")}}
        out.append(row)
        m = np.ones(net.N); m[supp] = 0.0
        mask = mask * torch.tensor(m, device=net.dev, dtype=torch.float64)
    return out
