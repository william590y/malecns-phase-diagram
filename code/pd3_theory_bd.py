"""pd3 theory audit, part 2: the one-mode birth-death reduction (theory.html section 5) as a finite-K mean-field directed
percolation (MF-DP) problem. Exact numerics (CPU, ~minutes), no fitting to simulations.

Chain (as theory_branch_gpu.log_tau): states n = 1..n_top, n_top = floor(K); birth b(n) = lam n (1 - n/K) (b(n_top) = 0),
death d(n) = alpha n, absorbing at 0. R = lam/alpha. Time in units of 1/alpha (multiply by 1/alpha_eff = 1.103 for model time).
 1. At R = 1: tau_top (from n_top), tau_1 (= mean avalanche duration from one seed), tau_QS (1/slowest decay rate),
    quasi-stationary mean n_QS and rho_QS = n_QS/K, for K = 8 .. 1e5 -> exponents and prefactors (MF-DP: tau ~ K^1/2,
    rho_QS ~ K^-1/2).
 2. Avalanches from one seed at R = 1: exact size distribution P(S) (jump-chain level iteration; S counts the seed),
    exact survival S(t) (symmetrised tridiagonal eigendecomposition); cutoffs S_c (P/P_inf = 1/2, P_inf = critical binary
    branching) and T_c (S(t)(1 + t) = 1/2) vs K -> exponents (SIS-type depletion: S_c ~ K, T_c ~ K^1/2).
 3. Monte Carlo (Gillespie) of the chain at R = 1 for K = 37, 75.8: <S>(T) and the mean skewness of avalanche profiles per
    duration bin (a 1-D birth-death chain is reversible, so the duration-conditioned mean profile is exactly time-symmetric).
 4. Off-critical scaling: tau_top / sqrt(K) vs x = (R - 1) sqrt(K) for several K (finite-size scaling collapse).
 5. Crossover predictions: g where tau_top = 200 and 3000 (model time) for the real module (lam(g) from the dense branch rows)
    at K = K_eff and K_needed, and for configuration nulls of size N with K(N) = kappa N.
Writes results/pd3/theory_bd.json.
"""
import json, glob, time
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import gammaln, logsumexp

ROOT = "/MALECNS_ROOT"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:.0f}s]", *a, flush=True)
dt = 0.2; a_eff = (1 - np.exp(-dt)) / dt


def rates(K, R):
    n_top = int(np.floor(K)); n = np.arange(1, n_top + 1, dtype=float)
    b = R * n * (1 - n / K); b[-1] = 0.0; d = n.copy()
    return n, b, d


def tau_top_and_1(K, R):
    n, b, d = rates(K, R)
    logD = np.empty(len(n)); logD[-1] = -np.log(d[-1])
    for k in range(len(n) - 2, -1, -1):
        logD[k] = np.logaddexp(0.0, np.log(b[k]) + logD[k + 1]) - np.log(d[k]) if b[k] > 0 else -np.log(d[k])
    return float(np.exp(logsumexp(logD))), float(np.exp(logD[0]))


def log_tau_top_vec(K, Rs):
    """ln tau_top for many R at once (same recursion as tau_top_and_1)."""
    n_top = int(np.floor(K)); Rs = np.asarray(Rs, float)
    logD = -np.log(n_top) * np.ones_like(Rs); acc = logD.copy()
    for nn in range(n_top - 1, 0, -1):
        lb = np.log(Rs * nn * (1 - nn / K))
        logD = np.logaddexp(0.0, lb + logD) - np.log(nn); acc = np.logaddexp(acc, logD)
    return acc


def sym_tridiag(K, R):
    n, b, d = rates(K, R)
    diag = -(b + d); off = np.sqrt(b[:-1] * d[1:])
    # log sqrt(pi): pi(n+1)/pi(n) = b(n)/d(n+1)
    lp = np.concatenate([[0.0], np.cumsum(np.log(np.maximum(b[:-1], 1e-300)) - np.log(d[1:]))])
    return n, diag, off, 0.5 * lp


def qsd(K, R):
    n, diag, off, lsp = sym_tridiag(K, R)
    m = len(n)
    if m == 1:
        return 1.0 / -diag[0], 1.0
    ev, vec = eigh_tridiagonal(diag, off, select="i", select_range=(m - 1, m - 1))
    q = np.abs(vec[:, 0]) * np.exp(lsp - lsp.max())        # right eigenvector of A = Pi^{1/2} (sym vec)
    q /= q.sum()
    return float(1 / -ev[0]), float(q @ n)


def survival(K, R, t):
    n, diag, off, lsp = sym_tridiag(K, R)
    ev, Q = eigh_tridiagonal(diag, off)
    # S(t) = 1^T Pi^{1/2} Q e^{ev t} Q^T Pi^{-1/2} e_1
    left = (np.exp(lsp - lsp[0]) @ Q)                       # 1^T Pi^{1/2} Q   (scaled by Pi^{1/2}(1)^{-1})
    right = Q[0, :]                                         # Q^T e_1 * Pi^{-1/2}(1)  (scale cancels)
    return np.array([float(left @ (np.exp(ev * tt) * right)) for tt in t])


def size_dist(K, R, s_max):
    n, b, d = rates(K, R); m = len(n)
    pb = b / (b + d); pd_ = 1 - pb
    Ld = np.concatenate([[0.0], np.cumsum(np.log(pd_))])     # Ld[k] = sum_{j<=k} log pd(j)
    T = np.zeros((m, m))                                    # T[new_state-1, state-1]
    for s in range(1, m + 1):                               # current state s (just after a birth)
        for mm in range(1, s + 1):                          # deaths down to mm, then birth -> mm + 1
            if pb[mm - 1] > 0 and mm + 1 <= m:
                T[mm, s - 1] = np.exp(Ld[s] - Ld[mm]) * pb[mm - 1]
    absorb = np.exp(Ld[1:])
    q = np.zeros(m); q[0] = 1.0; P = []
    for _ in range(s_max):
        P.append(float(absorb @ q)); q = T @ q
        if q.sum() < 1e-14:
            break
    return np.array(P)                                      # P[s-1] = P(S = s), S counts the seed


def P_inf(S):
    b = S - 1                                               # births; Catalan(b) / 2^(2b+1)
    return np.exp(gammaln(2 * b + 1) - gammaln(b + 1) - gammaln(b + 2) - (2 * b + 1) * np.log(2))


out = {"alpha_eff": a_eff}
# 1. R = 1 scaling
Ks = [8, 16, 32, 37, 54.7, 62.5, 64, 70.5, 75.8, 128, 256, 512, 1024, 2048, 4096, 16384, 65536, 100000]
crit = []
for K in Ks:
    tt, t1 = tau_top_and_1(K, 1.0); tq, nq = qsd(K, 1.0)
    crit.append({"K": K, "tau_top": tt, "tau_1": t1, "tau_QS": tq, "n_QS": nq, "rho_QS": nq / K})
big = [c for c in crit if c["K"] >= 1024]
fit = lambda key: np.polyfit(np.log([c["K"] for c in big]), np.log([c[key] for c in big]), 1).tolist()
out["R1"] = {"rows": crit, "fit_loglog_K>=1024": {k: fit(k) for k in ["tau_top", "tau_QS", "n_QS", "rho_QS", "tau_1"]},
             "sqrt(pi K/2) check": [c["tau_QS"] / np.sqrt(np.pi * c["K"] / 2) for c in crit],
             "nQS/sqrt(2K/pi) check": [c["n_QS"] / np.sqrt(2 * c["K"] / np.pi) for c in crit]}
log("R=1", out["R1"]["fit_loglog_K>=1024"])

# 2. avalanches at R = 1
av = []
tgrid = np.geomspace(0.01, 1e4, 400)
for K in [16, 32, 37, 54.7, 64, 75.8, 128, 256, 512, 1024, 2048]:
    row = {"K": K}
    if K <= 512:
        P = size_dist(K, 1.0, int(60 * K) + 200); S = np.arange(1, len(P) + 1)
        psi = P / P_inf(S); k = np.argmax(psi < 0.5)
        row.update({"S_c": int(S[k]), "P_S_total": float(P.sum()), "mean_S": float((S * P).sum()),
                    "P_S_at": {str(s): float(P[s - 1]) for s in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] if s <= len(P)},
                    "psi_at": {str(s): float(psi[s - 1]) for s in [1, 10, 100, 1000] if s <= len(P)}})
        if K in (37, 54.7, 75.8):
            row["P_S"] = P[:4000].tolist()
    Sv = survival(K, 1.0, tgrid); psiT = Sv * (1 + tgrid); kT = np.argmax(psiT < 0.5)
    row.update({"T_c": float(tgrid[kT]), "surv_at": {f"{t:g}": float(np.interp(t, tgrid, Sv)) for t in [1, 3, 10, 30, 100, 300]}})
    if K in (37, 54.7, 75.8):
        row["surv"] = Sv.tolist()
    av.append(row)
    log("aval K", K, "S_c", row.get("S_c"), "T_c", round(row["T_c"], 2))
fitS = np.polyfit(np.log([r["K"] for r in av if "S_c" in r and r["K"] >= 64]), np.log([r["S_c"] for r in av if "S_c" in r and r["K"] >= 64]), 1)
fitT = np.polyfit(np.log([r["K"] for r in av if r["K"] >= 64]), np.log([r["T_c"] for r in av if r["K"] >= 64]), 1)
out["aval_R1"] = {"rows": av, "tgrid": tgrid.tolist(), "fit_S_c": fitS.tolist(), "fit_T_c": fitT.tolist()}
log("S_c ~ K^%.3f, T_c ~ K^%.3f" % (fitS[0], fitT[0]))

# 3. Monte Carlo: <S>(T) and profile skewness
rng = np.random.default_rng(1)
mc = {}
for K in [37.0, 75.8]:
    nA = 400000
    n_t = np.ones(nA); t_t = np.zeros(nA); S_t = np.ones(nA)
    m1 = np.zeros(nA); m0 = np.zeros(nA); m2 = np.zeros(nA); m3 = np.zeros(nA)
    alive = np.arange(nA); top = np.floor(K)
    while len(alive):
        nn = n_t[alive]; b = np.where(nn < top, nn * (1 - nn / K), 0.0); d = nn; q = b + d
        h = rng.exponential(1 / q); t1 = t_t[alive]; t2 = t1 + h
        m0[alive] += nn * h; m1[alive] += nn * (t2 ** 2 - t1 ** 2) / 2; m2[alive] += nn * (t2 ** 3 - t1 ** 3) / 3
        m3[alive] += nn * (t2 ** 4 - t1 ** 4) / 4
        t_t[alive] = t2
        birth = rng.random(len(alive)) < b / q
        n_t[alive] = nn + np.where(birth, 1, -1); S_t[alive] += birth
        alive = alive[n_t[alive] > 0]
    T = t_t; mu = m1 / m0; var = m2 / m0 - mu ** 2; sk = (m3 / m0 - 3 * mu * var - mu ** 3) / np.maximum(var, 1e-300) ** 1.5
    bins = np.geomspace(0.3, T.max() * 1.01, 30); which = np.digitize(T, bins)
    rows = []
    for j in range(1, len(bins)):
        sel = which == j
        if sel.sum() >= 50:
            rows.append({"T_lo": float(bins[j - 1]), "T_hi": float(bins[j]), "n": int(sel.sum()), "mean_T": float(T[sel].mean()),
                         "mean_S": float(S_t[sel].mean()), "mean_skew": float(sk[sel].mean()),
                         "skew_sem": float(sk[sel].std() / np.sqrt(sel.sum())), "centroid_minus_half": float((mu[sel] / T[sel]).mean() - 0.5)})
    mc[f"{K:g}"] = {"n_aval": nA, "rows": rows, "mean_S": float(S_t.mean()), "mean_T": float(T.mean())}
    log("MC K", K, "mean S", S_t.mean(), "rows", len(rows))
out["mc_R1"] = mc

# 4. off-critical scaling collapse
xs = np.linspace(-3, 6, 37); coll = {}
for K in [37, 54.7, 75.8, 300, 1000, 5000]:
    coll[f"{K:g}"] = [float(np.log(tau_top_and_1(K, 1 + x / np.sqrt(K))[0] / np.sqrt(K))) for x in xs]
out["collapse"] = {"x": xs.tolist(), "ln_tau_over_sqrtK": coll}

# 5. crossover predictions


def rows_of(kind):
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/results/pd/branch_{kind}_dense_part*.json")):
        d = json.load(open(f)); rows += d["rows"]
    rows.sort(key=lambda r: r["g"]); return d["g_lin"], rows


def crossing(g_lin, rows, K, tau_model):
    """g where tau_top(R(g), K) [model time = tau/alpha_eff] reaches tau_model; R(g) interpolated in ln g,
    linearly from R = 1 at g_lin below the first row."""
    gs = np.array([g_lin] + [r["g"] for r in rows]); Rs = np.array([1.0] + [r["R"] for r in rows])
    fine = np.geomspace(g_lin * 1.0001, gs[-1], 3000); Rf = np.interp(np.log(fine), np.log(gs), Rs)
    lt = log_tau_top_vec(K, Rf) - np.log(a_eff)
    k = np.argmax(lt >= np.log(tau_model))
    return float(fine[k]) if lt[k] >= np.log(tau_model) else None


pred = {}
gl, rr = rows_of("real")
out["tau_at_glin_model_time"] = {f"{K:g}": tau_top_and_1(K, 1.0)[0] / a_eff for K in [37, 54.7, 58.5, 62.5, 70.5, 75.8]}
for lab, K in [("K_eff_r0_54.7", 54.7), ("K_eff_r0_glin_62.5", 62.5), ("K_nocoinc_70.5", 70.5), ("K_needed_r0_75.8", 75.8),
               ("K_eff_r1_16.9", 16.9), ("K_needed_r1_36.9", 36.9)]:
    pred[lab] = {"S_tau200": crossing(gl, rr, K, 200), "on_tau3000": crossing(gl, rr, K, 3000),
                 "on_tau30000": crossing(gl, rr, K, 30000)}
    log("real", lab, pred[lab])
out["real_crossings"] = pred
gln, rn = rows_of("null")
Nfull = 167216
kap = {"r0_theory": 5031.1 / Nfull, "r1_theory": 95.7 / Nfull, "r1_needed": 181.0 / Nfull}
nul = {}
for lab, kp in kap.items():
    nul[lab] = []
    for N in [500, 1000, 2000, 5000, 10000, 20000, 40000, 80000, 167216]:
        K = kp * N
        if K < 2:
            continue
        S_ = crossing(gln, rn, K, 200); on = crossing(gln, rn, K, 3000)
        nul[lab].append({"N": N, "K": K, "tau_glin_model": tau_top_and_1(K, 1.0)[0] / a_eff, "S_tau200": S_, "on_tau3000": on,
                         "on_over_glin": on / gln if on else None})
    log("null", lab, [(x["N"], round(x["K"], 1), x["on_over_glin"]) for x in nul[lab]])
out["null_family"] = {"g_lin_null": gln, "kappa": kap, "rows": nul,
                      "note": "R(g) of the full-size configuration null used for every N (assumes lam_PF and the mode shape are N-independent)"}
json.dump(out, open(f"{ROOT}/results/pd3/theory_bd.json", "w"), indent=1)
log("saved results/pd3/theory_bd.json")
