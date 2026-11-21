"""E3 theory: the birth-death / branching onset theory (theory_branch.py, explainer theory.html P1-P2) extended with
short-term depression. CPU; inputs results/aldef/depression/theory_vecs.npz (ald_dep_theory_vec.py) and the dense theory rows
results/pd/branch_real_dense_part*.json (lam, K_r). Output results/aldef/depression/theory.json (+ functions imported by the
analysis).

DERIVATION (short)
 Base theory: with the mode ansatz a_i = m v_i (B(g) v = lam v, u^T B = lam u^T, <u, v> = 1), the projected drift is
     dm/dt = (lam - alpha) m - [lam <u, v^2> + D(g, r)] m^2,
 i.e. the logistic birth-death chain b(n) = lam n (1 - n/K), d(n) = alpha n in n = m / <u^2, v>, K = K_eff(g, r); tau(g) = exact
 mean extinction time from the top state.
 Depression multiplies every presynaptic output a_j by its efficacy e_j <= 1. The birth term becomes
     sum_i u_i sum_j B_ij a_j e_j = lam m sum_j u_j v_j e_j(rho_j)   (u^T B = lam u^T),   rho_j = m v_j,
 so the birth rate of the chain is scaled by the MODE-WEIGHTED EFFICACY
     X(n) = sum_j u_j v_j e_j(n <u^2, v> v_j)          (X(0) = 1 in spike mode, X(0) = phi < 1 in rate mode),
 and the saturation terms scale the same way to leading order:  b(n) = lam X(n) n (1 - n/K),  d(n) = alpha n.
 e_j(rho) = stationary efficacy of a unit with activity rho (rho clipped at the model maximum 0.5), from the exact single-unit
 renewal / Markov formulas (unit-tested in ald_dep_test.py):
     spike mode: e = (1 - M)/(1 - (1 - U) M),  M = E[q^L] E[q^Q], q = e^{-dt/TAU}, L ~ Geom(p_off), Q ~ Geom(p),
                 p = p_off rho/(1 - rho)   (Poisson limit: 1/(1 + U nu TAU), nu = alpha_eff rho)
     rate mode:  e = E[x | active] of the 2-state (a, x) chain (continuous depletion while active).
 Consequences: (1) linear onset: lam X(0) = alpha  ->  spike mode g_lin unchanged; rate mode lam(g) phi = alpha.
 (2) deterministic quasi-stationary level n*: lam X(n*) (1 - n*/K) = alpha. Because U TAU = 6-82 at literature values, X(n)
     falls steeply and n* is pinned close to the point where lam X = alpha, i.e. far below the undepressed K (1 - alpha/lam):
     the active state is driven back toward the critical point (self-organised near-criticality, Levina et al. 2007).
 (3) lifetime, two limits for the slow resource (TAU >> 1 = active-epoch time):
     FROZEN  : the resource is frozen at its quasi-stationary value X* = X(n*) during the fast escape to n = 0:
               b(n) = lam X* n (1 - n/K)                   (TAU -> infinity limit; expected to be the relevant one)
     ADIABATIC: the resource follows n instantly: b(n) = lam X(n) n (1 - n/K)   (TAU -> 0 limit; longer lifetimes)
     tau = exact mean extinction time of the chain from its top state (same recursion as theory_branch_gpu.py, with the
     truncation level smoothed in K as analysis2_phase.log_tau_smooth).
 K: K_eff from the dense rows (r = 0, 1), and K_cal = kappa_r K_eff with kappa_r fitted so that the BASE chain reproduces the
 simulated base onset (tau = 3000) of this experiment's own base runs (the base theory needs K ~2x K_eff at r = 1).
"""
import glob, json, sys
import numpy as np
from scipy.special import logsumexp

ROOT = "/MALECNS_ROOT"
C = "/MALECNS_CACHE"
DT = 0.2
P_OFF = 1 - np.exp(-DT)
A_EFF = P_OFF / DT
RHO_MAX = 0.5


def geom_pgf(p, q):
    return p * q / (1 - (1 - p) * q)


def eff_spike(rho, U, TAU):
    rho = np.clip(rho, 1e-12, RHO_MAX)
    p = np.minimum(P_OFF * rho / (1 - rho), 1.0)
    q = np.exp(-DT / TAU)
    M = geom_pgf(P_OFF, q) * geom_pgf(p, q)
    return (1 - M) / (1 - (1 - U) * M)


def eff_rate(rho, U, TAU):
    """E[x | active] of the exact 2-state chain (ald_test.py T3), vectorised over rho."""
    rho = np.clip(np.atleast_1d(np.asarray(rho, float)), 1e-12, RHO_MAX)
    pa = np.minimum(P_OFF * rho / (1 - rho), 1.0)
    k1 = 1 / TAU + U; xinf = (1 / TAU) / k1; ea, eq = np.exp(-k1 * DT), np.exp(-DT / TAU)
    c = {0: (1 - eq, eq), 1: (xinf * (1 - ea), ea)}
    out = np.empty_like(rho)
    for i, p in enumerate(pa):
        Ta = np.array([[1 - p, p], [P_OFF, 1 - P_OFF]]); pi = np.array([P_OFF, p]) / (p + P_OFF)
        Mm = np.array([[Ta[a, a1] * c[a][1] for a in (0, 1)] for a1 in (0, 1)])
        bm = np.array([sum(Ta[a, a1] * c[a][0] * pi[a] for a in (0, 1)) for a1 in (0, 1)])
        m = np.linalg.solve(np.eye(2) - Mm, bm)
        out[i] = m[1] / pi[1]
    return out


def phi_rate(U, TAU):
    k = 1 / TAU + U; xi = (1 / TAU) / k
    return xi + (1 - xi) * (1 - np.exp(-DT)) / (1 - np.exp(-(1 + k) * DT))


def log_tau(lam_n, K, n_top):
    """Exact mean extinction time (log) from the top state n_top of the chain b(n) = lam_n[n-1] n (1 - n/K), d = A_EFF n."""
    if n_top < 1:
        return -np.inf
    n = np.arange(1, n_top + 1, dtype=float)
    lb = lam_n[:n_top] * n * (1 - n / K); lb[-1] = 0.0
    lb = np.where(lb > 0, lb, 0.0)
    ld = A_EFF * n
    logD = np.empty(n_top); logD[-1] = -np.log(ld[-1])
    for k in range(n_top - 2, -1, -1):
        logD[k] = (np.logaddexp(0.0, np.log(lb[k]) + logD[k + 1]) if lb[k] > 0 else 0.0) - np.log(ld[k])
    return float(logsumexp(logD))


def log_tau_smooth(lam_fun, K):
    """lam_fun(n array) -> per-state birth multiplier lam*X(n); truncation level interpolated between floor(K), floor(K)+1."""
    k0 = int(np.floor(K)); f = K - k0
    n = np.arange(1, k0 + 2, dtype=float)
    lam_n = lam_fun(n)
    a = log_tau(lam_n, K, k0)
    b = log_tau(lam_n, K, k0 + 1) if f > 0 else a
    return float((1 - f) * a + f * b)


class Theory:
    def __init__(self):
        d = np.load(f"{ROOT}/results/aldef/depression/theory_vecs.npz")
        self.g, self.lam = d["g"], d["lam"]
        self.u2v = d["u2v"]
        self.idx = [d["idx"][i, :d["n"][i]] for i in range(len(self.g))]
        self.uv = [d["u"][i, :d["n"][i]] * d["v"][i, :d["n"][i]] for i in range(len(self.g))]
        self.v = [d["v"][i, :d["n"][i]] for i in range(len(self.g))]
        rows = []
        for p in sorted(glob.glob(f"{ROOT}/results/pd/branch_real_dense_part*.json")):
            rows += json.load(open(p))["rows"]
        rows.sort(key=lambda x: x["g"])
        assert np.allclose([x["g"] for x in rows], self.g)
        self.K = {r: np.array([x[f"K_r{r}"] for x in rows]) for r in (0, 1)}
        self.g_lin = 0.16315274988719286
        mod = np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)["idx"]
        self.mod = np.zeros(167216, bool); self.mod[mod] = True
        self._labels = None

    def labels(self):
        if self._labels is None:
            self._labels = {k: np.load(f"{C}/{f}.npy", allow_pickle=True).astype(str) for k, f in
                            (("class", "classes"), ("type", "types"))}
            self._labels["sign"] = np.load(f"{C}/sign.npy")
        return self._labels

    def dep_mask(self, group):
        """Boolean mask over all neurons of the depressing population (group = None -> all)."""
        if group is None:
            return np.ones(167216, bool)
        L = self.labels()
        al = np.isin(L["class"], ("olfactory", "hygrosensory", "thermosensory", "ALPN", "ALLN", "ALIN", "ALON")) | self.mod
        return {"sign:1": L["sign"] > 0, "class:olfactory": L["class"] == "olfactory", "class:ALPN": L["class"] == "ALPN",
                "class:ALLN": L["class"] == "ALLN", "type:lLN1_bc": L["type"] == "lLN1_bc", "al;module": al,
                "~(al;module)": ~al}[group]

    def X_fun(self, i, mode, U, TAU, group=None):
        """Mode-weighted efficacy X(n) at grid point i (n array)."""
        if mode is None:
            return lambda n: np.ones_like(np.asarray(n, float))
        uv, v = self.uv[i], self.v[i]
        dep = self.dep_mask(group)[self.idx[i]]
        ef = eff_spike if mode == "spike" else eff_rate
        rr = np.geomspace(1e-7, RHO_MAX, 400)           # tabulate e(rho) once
        er = ef(rr, U, TAU)
        e0 = 1.0 if mode == "spike" else phi_rate(U, TAU)

        def X(n):
            n = np.atleast_1d(np.asarray(n, float))
            rho = np.clip(n[:, None] * self.u2v[i] * v[None, :], 0, RHO_MAX)
            e = np.where(rho > 1e-7, np.interp(np.log(np.maximum(rho, 1e-7)), np.log(rr), er), e0)
            e = np.where(dep[None, :], e, 1.0)
            return (e * uv[None, :]).sum(1) / uv.sum()
        return X

    def point(self, i, r, mode=None, U=None, TAU=None, group=None, kappa=1.0):
        lam, K = self.lam[i], kappa * self.K[r][i]
        X = self.X_fun(i, mode, U, TAU, group)
        out = {"g": float(self.g[i]), "lam": float(lam), "K": float(K), "X0": float(X(np.array([1e-9]))[0])}
        out["lt_adiabatic"] = log_tau_smooth(lambda n: lam * X(n), K)
        # deterministic QS level: lam X(n) (1 - n/K) = A_EFF (continuous n in (0, K))
        nn = np.linspace(1e-6, K, 2000)
        F = lam * X(nn) * (1 - nn / K) - A_EFF
        if F[0] <= 0:
            out.update({"n_star": 0.0, "X_star": float(X(np.array([1e-9]))[0]), "lt_frozen": out["lt_adiabatic"],
                        "AL_rho_star": 0.0})
        else:
            k = int(np.argmax(F <= 0)) if (F <= 0).any() else len(nn) - 1
            ns = float(np.interp(0.0, [F[k], F[k - 1]], [nn[k], nn[k - 1]])) if k > 0 else float(nn[0])
            Xs = float(X(np.array([ns]))[0])
            out["n_star"], out["X_star"] = ns, Xs
            out["lt_frozen"] = log_tau_smooth(lambda n: lam * Xs * np.ones_like(n), K)
            rho = np.clip(ns * self.u2v[i] * self.v[i], 0, RHO_MAX)
            out["AL_rho_star"] = float(rho[self.mod[self.idx[i]]].sum() / self.mod.sum())
        return out

    def curve(self, r, mode=None, U=None, TAU=None, group=None, kappa=1.0):
        return [self.point(i, r, mode, U, TAU, group, kappa) for i in range(len(self.g))]


def cross(g, lt, thr):
    """First upward crossing of ln(thr) by lt(g), interpolated in (ln g, ln tau); None if never / below grid."""
    lt = np.asarray(lt, float); y = np.log(thr)
    hit = np.where(lt >= y)[0]
    if len(hit) == 0:
        return None
    k = hit[0]
    if k == 0:
        return float(g[0])
    f = (y - lt[k - 1]) / (lt[k] - lt[k - 1])
    return float(np.exp(np.log(g[k - 1]) + f * np.log(g[k] / g[k - 1])))


def fit_kappa(th, r, g_on_sim, T=3000.0):
    """kappa such that the BASE chain's tau = T crossing equals the simulated base onset (bisection in log kappa)."""
    lo, hi = 0.3, 10.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        c = th.curve(r, kappa=mid)
        gc = cross(th.g, [p["lt_adiabatic"] for p in c], T)
        if gc is None or gc > g_on_sim:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


if __name__ == "__main__":
    th = Theory()
    print("g_lin", th.g_lin, "lam[0]/alpha", th.lam[0] / A_EFF)
    for (U, TAU) in [(0.25, 157), (0.28, 240)]:
        X = th.X_fun(10, "spike", U, TAU)
        print("spike X(n)", U, TAU, [round(float(x), 3) for x in X(np.array([0.01, 1, 3, 10, 30]))])
