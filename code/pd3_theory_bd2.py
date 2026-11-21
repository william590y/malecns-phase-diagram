"""pd3 theory audit, part 2b (CPU, ~2 min): two more finite-K MF-DP checks of the logistic birth-death chain.
 1. Monte Carlo <S>(T) at R = 1 for K = 300 and 2000 (compare K = 37, 75.8 in theory_bd.json): does the local slope
    d ln<S>/d ln T approach the MF-DP value 2 only when the window 1 << T << T_c ~ 2 K^1/2 is wide?
 2. Quasi-stationary (QS) susceptibility chi = var(n)/<n> (= K var(rho)/<rho>) as a function of x = (R - 1) K^1/2:
    peak position x_peak and height vs K (MF-DP: chi_peak ~ K^1/2, R_peak - 1 ~ K^-1/2).
Writes results/pd3/theory_bd2.json.
"""
import json, time
import numpy as np
from scipy.linalg import eigh_tridiagonal

ROOT = "/MALECNS_ROOT"
t0 = time.time()


def rates(K, R):
    n_top = int(np.floor(K)); n = np.arange(1, n_top + 1, dtype=float)
    b = R * n * (1 - n / K); b[-1] = 0.0; d = n.copy()
    return n, b, d


def qsd_moments(K, R):
    n, b, d = rates(K, R); m = len(n)
    diag = -(b + d); off = np.sqrt(b[:-1] * d[1:])
    lsp = 0.5 * np.concatenate([[0.0], np.cumsum(np.log(np.maximum(b[:-1], 1e-300)) - np.log(d[1:]))])
    ev, vec = eigh_tridiagonal(diag, off, select="i", select_range=(m - 1, m - 1))
    q = np.abs(vec[:, 0]) * np.exp(lsp - lsp.max()); q /= q.sum()
    mu = q @ n; var = q @ n ** 2 - mu ** 2
    return float(mu), float(var), float(1 / -ev[0])


out = {"mc": {}, "chi": {}}
rng = np.random.default_rng(2)
for K, nA in [(300.0, 200000), (2000.0, 100000)]:
    n_t = np.ones(nA); t_t = np.zeros(nA); S_t = np.ones(nA); alive = np.arange(nA); top = np.floor(K)
    while len(alive):
        nn = n_t[alive]; b = np.where(nn < top, nn * (1 - nn / K), 0.0); q = b + nn
        t_t[alive] += rng.exponential(1 / q)
        birth = rng.random(len(alive)) < b / q
        n_t[alive] = nn + np.where(birth, 1, -1); S_t[alive] += birth
        alive = alive[n_t[alive] > 0]
    bins = np.geomspace(0.3, t_t.max() * 1.01, 40); wh = np.digitize(t_t, bins); rows = []
    for j in range(1, len(bins)):
        s = wh == j
        if s.sum() >= 50:
            rows.append({"mean_T": float(t_t[s].mean()), "mean_S": float(S_t[s].mean()), "n": int(s.sum())})
    T = np.array([r["mean_T"] for r in rows]); S = np.array([r["mean_S"] for r in rows])
    slopes = {f"{lo}-{hi}": float(np.polyfit(np.log(T[(T > lo) & (T < hi)]), np.log(S[(T > lo) & (T < hi)]), 1)[0])
              for lo, hi in [(1, 5), (2, 8), (5, 15), (10, 40), (20, 80), (40, 150)] if ((T > lo) & (T < hi)).sum() >= 3}
    out["mc"][f"{K:g}"] = {"rows": rows, "slopes": slopes}
    print(f"[{time.time()-t0:.0f}s] K={K:g} slopes {slopes}", flush=True)
xs = np.linspace(-2, 6, 81)
for K in [37, 54.7, 75.8, 150, 300, 1000, 3000]:
    chi = []; mus = []
    for x in xs:
        mu, var, tq = qsd_moments(K, 1 + x / np.sqrt(K)); chi.append(var / mu); mus.append(mu)
    k = int(np.argmax(chi))
    out["chi"][f"{K:g}"] = {"x": xs.tolist(), "chi": chi, "n_QS": mus, "x_peak": float(xs[k]), "chi_peak": float(chi[k]),
                            "R_peak": float(1 + xs[k] / np.sqrt(K)), "chi_at_R1": float(chi[int(np.argmin(np.abs(xs)))])}
    print(f"K={K:g} x_peak {xs[k]:.2f} R_peak {1+xs[k]/np.sqrt(K):.4f} chi_peak {chi[k]:.3f} chi(R=1) {out['chi'][f'{K:g}']['chi_at_R1']:.3f}", flush=True)
Kl = [37, 54.7, 75.8, 150, 300, 1000, 3000]
out["fit_chi_peak_vs_K"] = np.polyfit(np.log(Kl), np.log([out["chi"][f"{K:g}"]["chi_peak"] for K in Kl]), 1).tolist()
out["fit_chi_peak_vs_K_K>=300"] = np.polyfit(np.log(Kl[-3:]), np.log([out["chi"][f"{K:g}"]["chi_peak"] for K in Kl[-3:]]), 1).tolist()
json.dump(out, open(f"{ROOT}/results/pd3/theory_bd2.json", "w"), indent=1)
print("saved", out["fit_chi_peak_vs_K"], out["fit_chi_peak_vs_K_K>=300"], flush=True)
