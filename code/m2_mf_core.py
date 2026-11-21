"""m2_mf_core.py -- quenched mean field (QMF) of Model 2 WITH per-spike Tsodyks-Markram resources (lane m2/theory,
meanfield part; 2026-11-09). Derivation: results/m2/theory/derivation.md. Plan: results/m2/theory/plan_meanfield.md.

The simulator (code/ald_sim.py, --std_mode spike) does, per synchronous step dt = 0.2 and with the OLD state:
    s_i = g sum_j W_ij a_j e_j + h_i ;  p_i = 1 - exp(-beta tanh(s_i)_+ dt) ;  active -> off w.p. P = 1 - e^{-dt},
    quiescent -> on w.p. p_i ;  every neuron: x <- 1 - q (1 - x), q = e^{-dt/TAU} ;  at an onset: e <- x, x <- (1-U) x.
Per neuron, for a GIVEN on-probability sequence p(t), the joint law of (a, e, x) has a CLOSED linear moment map in
    rho = E[a],  y = E[a e] (transmitted output),  m1 = E[a x],  m0 = E[(1 - a) x]:
    A  = q m1 + (1-q) rho                      (= E[a x~], x~ = recovered resource)
    Bq = q m0 + (1-q) (1 - rho)                (= E[(1-a) x~])
    rho' = (1-P) rho + p (1 - rho)
    y'   = (1-P) y   + p Bq
    m1'  = (1-P) A   + p (1-U) Bq
    m0'  = P A       + (1-p) Bq
This is exact for one unit with deterministic p (unit-tested against Monte Carlo of ald_sim's own update and against the
renewal formula). The ONLY approximation of the network theory is the QMF closure p_i = f(E[s_i]) with
E[s_i] = g sum_j W_ij y_j + h_i (independence + Jensen), as in M1's code/theory_qmf.py. Non-depressing neurons: U = 0,
q = 1 -> y = rho exactly (M1's QMF map is recovered bit-for-bit in exact arithmetic).

The operator is built with ald_sim.py's OWN functions (Labels/select/sigma_vector/node_scale/build_matrix), so W is the
simulator's operator (raw normalisation c = 1/745.91, released-transmitter signs, --edge_sign overrides, nulls with
--null_seed). Read-only use of ald_sim.py; nothing in it is changed.
"""
import sys, time
import numpy as np
import torch

ROOT = "/MALECNS_ROOT"
CODE = f"{ROOT}/code"
C = "/MALECNS_CACHE"
sys.path.insert(0, CODE)
import ald_sim as AS          # noqa: E402  (read-only: operator construction and selectors)
import m2_common as MC        # noqa: E402

DT, BETA = 0.2, 1.0
P_OFF = 1 - np.exp(-DT)
ALPHA = P_OFF / DT
EXC_EDGES = [("lLN1_bc", "ALPN", 0.0), ("lLN1_bc", "ALLN", 1.0)]          # M2s per-target signs

VARIANTS = {   # name: (depression group selector or None, U, TAU, edge overrides)
    "M1": (None, 0.0, np.inf, []),
    "M2": ("sign:1", 0.25, 157.0, []),
    "M2s": ("sign:1", 0.25, 157.0, EXC_EDGES),
    "M1s": (None, 0.0, np.inf, EXC_EDGES),
    "M2_allsyn": ("all", 0.25, 157.0, []),
    "M2_KW09": ("sign:1", 0.28, 240.0, []),
    "M2_N15": ("sign:1", 0.22, 89.0, []),
    "M2_tau30": ("sign:1", 0.25, 30.0, []),
}


# ---------------------------------------------------------------------------------------------- single unit (exact)
def geom_pgf(p, q):
    """E[q^L], L ~ Geometric(p) on {1, 2, ...}."""
    return p * q / (1 - (1 - p) * q)


def eff_exact(p, U, TAU, dt=DT):
    """Stationary mean efficacy per epoch of a unit with constant on-probability p (renewal, synchronous update)."""
    q = np.exp(-dt / TAU)
    M = geom_pgf(1 - np.exp(-dt), q) * geom_pgf(p, q)
    return (1 - M) / (1 - (1 - U) * M)


def p_of_rho(rho):
    return rho * P_OFF / (1 - rho)


def eff_poisson(rho, U, TAU):
    """Continuous-time Poisson limit x* = 1 / (1 + U TAU alpha_eff rho)."""
    return 1.0 / (1.0 + U * TAU * ALPHA * rho)


def moment_stationary(p, U, TAU, dt=DT):
    """Stationary (rho, y, m1, m0) of the per-neuron moment map with constant p (2x2 solve for the resources)."""
    q = np.exp(-dt / TAU); P = 1 - np.exp(-dt)
    rho = p / (p + P)
    # m1 = (1-P)(q m1 + (1-q) rho) + p (1-U)(q m0 + (1-q)(1-rho)); m0 = P (q m1 + (1-q) rho) + (1-p)(q m0 + (1-q)(1-rho))
    Am = np.array([[1 - (1 - P) * q, -p * (1 - U) * q], [-P * q, 1 - (1 - p) * q]])
    b = np.array([(1 - P) * (1 - q) * rho + p * (1 - U) * (1 - q) * (1 - rho), P * (1 - q) * rho + (1 - p) * (1 - q) * (1 - rho)])
    m1, m0 = np.linalg.solve(Am, b)
    Bq = q * m0 + (1 - q) * (1 - rho)
    y = p * Bq / P
    return rho, y, m1, m0


# ---------------------------------------------------------------------------------------------- model construction
class Model:
    """Operator, per-neuron parameters and region masks of one variant at one r (on the GPU)."""

    def __init__(self, variant="M2", r=1.0, null=None, null_seed=0, dev="cuda", dtype=torch.float32, closure="qmf"):
        t0 = time.time()
        assert closure in ("qmf", "cqmf")
        self.closure = closure
        grp, U0, TAU0, edges = VARIANTS[variant]
        self.variant, self.r, self.null, self.dev, self.dtype = variant, float(r), null, dev, dtype
        pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
        w = np.load(C + "/weight.npy").astype(np.float32); sign = np.load(C + "/sign.npy")
        N = len(sign); self.N = N
        if null is not None:                                   # exactly ald_sim.py --null [--null_kind] --null_seed S
            nrng = np.random.default_rng(null_seed)
            if null == "targets":
                post = post[nrng.permutation(len(post))]
            elif null == "weights":
                w = w[nrng.permutation(len(w))]
            else:
                raise ValueError(null)
        L = AS.Labels(N, sign=sign)
        self.L = L
        scale = AS.node_scale(post, w, w, N, "raw", "renorm")
        self.c = float(scale[0])
        sig = AS.sigma_vector(sign, self.r)
        edge_ov = []
        for ps, qs, v in edges:
            mp, mq = AS.select(ps, L), AS.select(qs, L)
            em = mp[pre] & mq[post]
            edge_ov.append((em, v if v >= 0 else v * self.r))
        self.n_edge_ov = [int(em.sum()) for em, _ in edge_ov]
        self.W, self.nnz = AS.build_matrix(pre, post, w, scale, sig, N, dev, edge_ov)   # CSR [post, pre] float32
        # excitatory part (positive coefficients) for the branching operator about silence
        crow, col, val = self.W.crow_indices(), self.W.col_indices(), self.W.values()
        pos = val > 0
        rows = torch.repeat_interleave(torch.arange(N, device=dev), crow[1:] - crow[:-1])
        self.E_row, self.E_col, self.E_val = rows[pos], col[pos], val[pos].double()
        if closure == "cqmf":          # split operator: excitatory / inhibitory magnitudes and their binary patterns
            neg = val < 0
            mk = lambda m, v: torch.sparse_coo_tensor(torch.stack([rows[m], col[m]]), v, (N, N)).coalesce().to_sparse_csr()
            self.WE = mk(pos, val[pos]); self.WI = mk(neg, -val[neg])
            self.AE = mk(pos, torch.ones_like(val[pos])); self.AI = mk(neg, torch.ones_like(val[neg]))
            self.has_inh = bool(neg.any())
        # depression parameters per neuron
        U = np.zeros(N); TAU = np.full(N, np.inf)
        if grp is not None:
            m = AS.select(grp, L); U[m] = U0; TAU[m] = TAU0
        self.dep_np = U > 0
        self.U0, self.TAU0 = U0, TAU0
        q = np.where(np.isfinite(TAU), np.exp(-DT / np.where(np.isfinite(TAU), TAU, 1.0)), 1.0)
        self.U = torch.tensor(U, device=dev, dtype=dtype)[:, None]
        self.q = torch.tensor(q, device=dev, dtype=dtype)[:, None]
        # regions
        reg = {"all": np.ones(N, bool)}
        reg["ORN"] = AS.select("olfactory", L)
        reg["nonORN"] = ~reg["ORN"]
        for k in ("uPN_ant", "ALsys"):
            reg[k] = AS.select(MC.REGIONS[k], L)
        reg["lLN1_bc"] = AS.select("type:lLN1_bc", L)
        reg["AL_module"] = L.module()
        reg["AL"] = L.al()
        reg["dep"] = self.dep_np.copy()
        self.reg_names = list(reg)
        self.reg_np = reg
        self.reg_size = np.array([reg[k].sum() for k in self.reg_names], float)
        self.R = torch.tensor(np.array([reg[k] for k in self.reg_names]), device=dev, dtype=torch.float64)   # [R, N]
        sc = L.get("superclass")
        self.sc_names, sc_inv = np.unique(sc, return_inverse=True)
        self.SC = torch.zeros(len(self.sc_names), N, device=dev, dtype=torch.float64)
        self.SC[torch.tensor(sc_inv, device=dev), torch.arange(N, device=dev)] = 1.0
        self.sc_size = np.bincount(sc_inv, minlength=len(self.sc_names)).astype(float)
        self.setup_s = time.time() - t0

    def h_vec(self, orn_h=0.0):
        hv = np.zeros(self.N, np.float32)
        hv[self.reg_np["ORN"]] = orn_h
        return torch.tensor(hv, device=self.dev, dtype=self.dtype)[:, None]

    # ---------------------------------------------------------------- states
    def init_state(self, kind, ncol, eps=1e-6):
        N, dev, dt_ = self.N, self.dev, self.dtype
        Z = {}
        if kind == "decay":                      # every unit active at t = 0 (an onset: e = 1, x = 1 - U)
            Z["rho"] = torch.ones(N, ncol, device=dev, dtype=dt_)
            Z["y"] = torch.ones(N, ncol, device=dev, dtype=dt_)
            Z["m1"] = (1 - self.U).expand(N, ncol).clone()
            Z["m0"] = torch.zeros(N, ncol, device=dev, dtype=dt_)
        elif kind in ("quiet", "eps"):           # all silent, resources full (eps: tiny uniform seed, rested)
            e = 0.0 if kind == "quiet" else eps
            Z["rho"] = torch.full((N, ncol), e, device=dev, dtype=dt_)
            Z["y"] = torch.full((N, ncol), e, device=dev, dtype=dt_)
            Z["m1"] = torch.full((N, ncol), e, device=dev, dtype=dt_)
            Z["m0"] = torch.full((N, ncol), 1 - e, device=dev, dtype=dt_)
        else:
            raise ValueError(kind)
        return Z

    # ---------------------------------------------------------------- the map
    def input(self, Z, gv, H):
        if self.closure == "qmf" or not self.has_inh:
            return torch.sparse.mm(self.W, Z["y"]) * gv + H
        # coincidence-corrected closure (CQMF): E[(S_E - S_I)_+] = mu_E - E[min(S_E, S_I)] and, for independent inputs,
        # E[min(S_E, S_I)] <= min(mu_E P(S_I > 0), mu_I P(S_E > 0)); the bound replaces the QMF's mu_I.
        y, rho = Z["y"], Z["rho"]
        muE = torch.sparse.mm(self.WE, y); muI = torch.sparse.mm(self.WI, y)
        lr = torch.log1p(-rho.clamp(max=1 - 1e-7))
        piE = -torch.expm1(torch.sparse.mm(self.AE, lr)); piI = -torch.expm1(torch.sparse.mm(self.AI, lr))
        # the positive part includes the external drive h >= 0, which inhibition can cancel too:
        # A = h + g mu_E, P(h + S_E > 0) = 1 where h > 0 (else pi_E);  s = A - min(A pi_I, g mu_I P(A > 0))
        # (fix 2026-11-09 18:25: the first version let inhibition cancel only g mu_E, shielding driven neurons)
        A = muE * gv + H
        piA = torch.where(H > 0, torch.ones_like(piE), piE)
        return A - torch.minimum(A * piI, muI * gv * piA)

    def step(self, Z, gv, H, keep=None):
        """One synchronous QMF step. gv [1, C]; H [N, 1] or [N, C]; keep [N, C] (0 = clamped off) or None."""
        s = self.input(Z, gv, H)
        p = -torch.expm1(-BETA * torch.tanh(s.clamp(min=0)) * DT)
        if keep is not None:
            p = p * keep
        rho, y, m1, m0 = Z["rho"], Z["y"], Z["m1"], Z["m0"]
        q, U = self.q, self.U
        A = q * m1 + (1 - q) * rho
        Bq = q * m0 + (1 - q) * (1 - rho)
        Z["rho"] = (1 - P_OFF) * rho + p * (1 - rho)
        Z["y"] = (1 - P_OFF) * y + p * Bq
        Z["m1"] = (1 - P_OFF) * A + p * (1 - U) * Bq
        Z["m0"] = P_OFF * A + (1 - p) * Bq
        return Z, s, p

    @staticmethod
    def clamp_off(Z, mask):
        """Force the masked neurons [N, C] (bool) silent now: active units become quiescent with their resource."""
        m = mask.to(Z["rho"].dtype)
        Z["m0"] = Z["m0"] + m * Z["m1"]
        for k in ("rho", "y", "m1"):
            Z[k] = Z[k] * (1 - m)
        return Z

    # ---------------------------------------------------------------- linearisation (JVP), exact for the map
    def jvp(self, Z, s, p, dZ, gv, keep=None):
        """J dZ at state Z (with its input s and p). Rectifier: derivative beta dt sech^2 e^{-beta tanh dt} for s > 0,
        the right derivative beta dt at s == 0 (two-sided, linear; conservative), 0 for s < 0."""
        th = torch.tanh(s.clamp(min=0))
        fp = torch.where(s >= 0, BETA * DT * (1 - th * th) * torch.exp(-BETA * th * DT), torch.zeros_like(s))
        if keep is not None:
            fp = fp * keep
        dp = fp * (torch.sparse.mm(self.W, dZ["y"]) * gv)
        rho, m1, m0 = Z["rho"], Z["m1"], Z["m0"]
        q, U = self.q, self.U
        Bq = q * m0 + (1 - q) * (1 - rho)
        dA = q * dZ["m1"] + (1 - q) * dZ["rho"]
        dB = q * dZ["m0"] - (1 - q) * dZ["rho"]
        out = {"rho": (1 - P_OFF - p) * dZ["rho"] + (1 - rho) * dp,
               "y": (1 - P_OFF) * dZ["y"] + p * dB + Bq * dp,
               "m1": (1 - P_OFF) * dA + (1 - U) * (p * dB + Bq * dp),
               "m0": P_OFF * dA + (1 - p) * dB - Bq * dp}
        return out

    def jvp_any(self, Z, s, p, dZ, gv, H, keep=None, eps=1e-6):
        if self.closure == "qmf" or not self.has_inh:
            return self.jvp(Z, s, p, dZ, gv, keep)
        sc = max(float(max(v.abs().max() for v in dZ.values())), 1e-300)
        e = eps / sc
        Zp, _, _ = self.step({k: Z[k] + e * dZ[k] for k in Z}, gv, H, keep)
        Zm, _, _ = self.step({k: Z[k] - e * dZ[k] for k in Z}, gv, H, keep)
        return {k: (Zp[k] - Zm[k]) / (2 * e) for k in Z}

    def to64(self):
        """float64 copies of the operators and parameters (for the linearisation)."""
        for k in ("W", "WE", "WI", "AE", "AI"):
            if hasattr(self, k):
                setattr(self, k, getattr(self, k).to(torch.float64))
        self.q, self.U = self.q.double(), self.U.double()
        self.dtype = torch.float64
        return self

    # ---------------------------------------------------------------- observables
    def counts(self, Z):
        """Expected number of active neurons per region [R, C] (float64)."""
        return self.R @ Z["rho"].double()

    def efficacy_next(self, Z):
        """Resource a new onset would get now: E[x~ | quiescent] (1 for non-depressing)."""
        rho = Z["rho"]
        Bq = self.q * Z["m0"] + (1 - self.q) * (1 - rho)
        return torch.where(rho < 1, Bq / (1 - rho).clamp(min=1e-12), torch.ones_like(rho))

    # ---------------------------------------------------------------- branching operator about silence
    def perron_branch(self, xcol, g, v0=None, iters=4000, tol=1e-8):
        """Perron root of B(g, x)_ij = f(g E_ij x_j) / dt (exact single-onset branching rates; x = resource at the next
        onset, frozen). Shifted power iteration (float64). Returns lam, sigma = lam / alpha_eff, right vector."""
        N = self.N
        vals = g * self.E_val * xcol[self.E_col].double()
        vals = -torch.expm1(-BETA * torch.tanh(vals) * DT) / DT
        B = torch.sparse_coo_tensor(torch.stack([self.E_row, self.E_col]), vals, (N, N)).coalesce().to_sparse_csr()
        x = torch.ones(N, 1, device=self.dev, dtype=torch.float64) if v0 is None else v0.clone()
        x /= x.norm(); lam = 0.0; lam_new = 0.0
        for it in range(iters):
            yv = torch.sparse.mm(B, x); lam_new = float(yv.norm() / x.norm())
            x = x + yv / max(lam_new, 1e-300); x /= x.norm()
            if it > 50 and abs(lam_new - lam) < tol * lam_new:
                break
            lam = lam_new
        return lam_new, lam_new / ALPHA, x, it


def matrix_pencil(x, M=None, Lfrac=1 / 3, rel=1e-6):
    """Damped complex exponentials x_k ~ sum_m c_m z_m^k (matrix-pencil method, Hua & Sarkar 1990).
    Returns z (per sample), amplitudes c, singular values. Model order M = #singular values > rel * max (cap 16)."""
    x = np.asarray(x, np.float64)
    K = len(x); Lp = max(2, int(K * Lfrac))
    Y = np.array([x[i:i + Lp + 1] for i in range(K - Lp)])
    U_, S, Vt = np.linalg.svd(Y, full_matrices=False)
    if M is None:
        M = int(min(16, max(1, (S > rel * S[0]).sum())))
    V = Vt[:M].conj().T
    V1, V2 = V[:-1], V[1:]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    Zm = np.vander(z, K, increasing=True).T
    c = np.linalg.lstsq(Zm, x.astype(complex), rcond=None)[0]
    return z, c, S
