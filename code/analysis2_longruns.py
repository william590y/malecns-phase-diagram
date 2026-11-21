"""Lifetime statistics of the slow-window runs at r = 1 (raw norm, h = 0, fully active start) vs the extinction theory.

Inputs (read-only):
  real: results/pd/long_real.npz (T 30000, B 32, g .26-.34), results/fine_decay_raw_r1.npz (T 8000, B 32, g .15-.60);
        T = 3000 sweeps used only for the T-drift test: results/pd/decay_r1_real.npz (B 16), results/decay_raw_r1.npz (B 16),
        results/pd2/onset_real_r1.npz (B 32; used if present).
  null (configuration null, seed 0 = the same rewired graph in every run and in theory_branch*.py):
        results/pd/long_null.npz (g .98-1.10), results/fine_decay_raw_r1_null.npz (g .60-1.40), T = 3000 sweeps
        results/pd/decay_r1_null.npz, results/decay_raw_r1_null.npz, results/pd2/onset_null_r1.npz (if present).
  theory: results/pd/branch_real_dense_part*.json (logtau_r1, lam), null: branch_null_dense_part*.json when all 8 parts
        exist, else branch_null.json (variants.branching).

Per replica: extinction time = first recorded time at which the replica is silent (t_up; the recording grid is
log-spaced, ~10% apart, so the true time lies in (t_lo, t_up], t_lo = last recorded time alive); censored at T if alive at
the end. Per g:
  - Kaplan-Meier survival (+ Greenwood s.e.);
  - censored-exponential MLE tau_sim = (sum of t_up over all replicas)/deaths, 95% CI from chi-square
    [2 TT / chi2_.975(2d+2), 2 TT / chi2_.025(2d)] (time-truncated convention) and a replica bootstrap; sensitivity:
    the exact interval-censored exponential MLE using (t_lo, t_up];
  - KM median; median / (tau ln 2) (= 1 for an exponential);
  - exponential test: KS distance between the empirical extinction CDF at the recording times (exact: a replica is silent
    at t_k iff it died by t_k) and the fitted exponential, p-value by parametric bootstrap that reproduces the recording
    grid, the censoring at T and the re-fit (Lilliefors-type); same for a shifted exponential (lag t0 = last time alive
    before the earliest death); coefficient of variation (uncensored g only) with its range under the exponential null;
  - mean density rho(t): power law vs exponential (and power law x exponential cutoff) least squares in log rho over
    t >= 20 while >= 25% of replicas survive; also the legacy window of code/analyze.py (t >= 5, t <= 0.9 * last death).
Theory: tau_theory(g) = exp(logtau_r1) interpolated linearly in log g; ratio tau_sim / tau_theory; the module size K that the
same birth-death recursion would need to reproduce tau_sim at the theory's lambda(g) (K_needed).
T-drift: coupling where half the replicas survive to T (pooled sweeps, isotonic in g, interpolated in log g, binomial
bootstrap CI; logistic check), plus a model-based value from a smooth censored-exponential fit ln tau(ln g) (validated
against the direct values at T = 3000, 8000); theory onsets where tau_theory = T and where tau_theory = T / ln 2.
Outputs: results/analysis2/longruns.{json,md}, figs/pd2/longruns_survival.png, figs/pd2/lifetime_sim_vs_theory.png.
"""
import glob, json, os
import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "/MALECNS_ROOT"
R = f"{ROOT}/results"; OUT = f"{R}/analysis2"; FIG = f"{ROOT}/figs/pd2"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
N = 167216
DT = 0.2; A_EFF = (1 - np.exp(-DT)) / DT
TS = [3000.0, 8000.0, 30000.0]
NBOOT = 2000
RHO_TMIN, RHO_SMIN = 20.0, 0.25
rng = np.random.default_rng(20261109)

DATA = {
    "real": [("long", "pd/long_real.npz"), ("fine", "fine_decay_raw_r1.npz"), ("stage1", "pd/decay_r1_real.npz"),
             ("stage0", "decay_raw_r1.npz"), ("pd2", "pd2/onset_real_r1.npz")],
    "null": [("long", "pd/long_null.npz"), ("fine", "fine_decay_raw_r1_null.npz"), ("stage1", "pd/decay_r1_null.npz"),
             ("stage0", "decay_raw_r1_null.npz"), ("pd2", "pd2/onset_null_r1.npz")],
}
LIFE_SETS = ("long", "fine")


def fnum(x):
    """JSON-safe float (inf/nan -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# ----------------------------------------------------------------------------------------------- data
def load_sets(kind):
    sets = {}
    for name, rel in DATA[kind]:
        p = f"{R}/{rel}"
        if not os.path.exists(p):
            continue
        d = np.load(p)
        meta = json.loads(str(d["meta"]))
        assert meta["exp"] == "decay" and meta["norm"] == "raw" and float(meta["r"]) == 1.0 and float(meta["h"]) == 0.0
        assert bool(meta["null"]) == (kind == "null") and meta.get("null_kind", "targets") == "targets"
        assert int(meta.get("seed", 0)) == 0 and float(meta["dt"]) == DT
        s = d["surv"] > 0
        assert (np.diff(s.astype(int), axis=1) <= 0).all(), "survival must be monotone (absorbing silence)"
        sets[name] = {"path": p, "t": d["t"].astype(float), "g": d["g"].astype(float), "surv": s,
                      "rho": d["rho"].astype(float), "T": float(meta["T"]), "B": int(meta["B"])}
    return sets


def extinction(t, s):
    """s [nt, B] bool alive -> t_up (first recorded silent time, or T if censored), t_lo, event."""
    alive_end = s[-1]
    k = np.argmax(~s, axis=0)
    t_up = np.where(alive_end, t[-1], t[k])
    t_lo = np.where(alive_end, t[-1], np.where(k > 0, t[np.maximum(k - 1, 0)], 0.0))
    return t_up, t_lo, ~alive_end


def kaplan_meier(times, event):
    ut = np.unique(times[event])
    S, var, out = 1.0, 0.0, []
    for u in ut:
        n = (times >= u).sum(); d = ((times == u) & event).sum()
        S *= 1 - d / n
        var += d / (n * (n - d)) if n > d else 0.0
        out.append((u, S, S * np.sqrt(var)))
    return np.array(out).reshape(-1, 3)


def exp_mle(times, event):
    d = int(event.sum()); TT = float(times.sum())
    if d == 0:
        return np.inf, 2 * TT / stats.chi2.ppf(0.95, 2), np.inf      # one-sided 95% lower bound
    return TT / d, 2 * TT / stats.chi2.ppf(0.975, 2 * d + 2), 2 * TT / stats.chi2.ppf(0.025, 2 * d)


def exp_interval_mle(t_lo, t_up, event):
    if event.sum() == 0:
        return np.inf
    a, b, c = t_lo[event], t_up[event], t_up[~event]

    def nll(lt):
        tau = np.exp(lt)
        return -(np.sum(-a / tau + np.log(-np.expm1(-(b - a) / tau))) - np.sum(c / tau))
    r = optimize.minimize_scalar(nll, bounds=(np.log(0.05), np.log(1e9)), method="bounded", options={"xatol": 1e-8})
    return float(np.exp(r.x))


def boot_tau(times, event, nb=NBOOT):
    n = len(times); idx = rng.integers(0, n, (nb, n))
    d = event[idx].sum(1); TT = times[idx].sum(1)
    tau = np.where(d > 0, TT / np.maximum(d, 1), np.inf)
    return np.percentile(tau, [2.5, 97.5])


def exp_gof(t, t_up, t_lo, event, shift):
    """KS distance between the exact empirical extinction CDF at the recording times and the fitted (shifted)
    exponential; parametric-bootstrap p-value reproducing grid, censoring and re-fit. Also CV range under H0."""
    T = t[-1]; n = len(t_up); d = int(event.sum())
    if d < 3:
        return None

    def fit(tu, tl, ev):            # works on [..., n] arrays
        dd = ev.sum(-1)
        if shift:
            t0 = np.where(ev, tl, np.inf).min(-1)
        else:
            t0 = np.zeros(tu.shape[:-1])
        tau = np.clip(tu - t0[..., None], 0, None).sum(-1) / np.maximum(dd, 1)
        return t0, tau

    def ks(tu, ev, t0, tau):
        ecdf = ((tu[..., None, :] <= t[:, None]) & ev[..., None, :]).mean(-1)            # [..., nt]
        F = np.where(t > t0[..., None], 1 - np.exp(-(t - t0[..., None]) / tau[..., None]), 0.0)
        return np.abs(ecdf - F).max(-1)

    t0, tau = fit(t_up[None], t_lo[None], event[None])
    D = float(ks(t_up[None], event[None], t0, tau)[0])
    X = t0[0] + rng.exponential(tau[0], (NBOOT, n))
    k = np.searchsorted(t, X, side="left")
    cens = k >= len(t)
    kk = np.minimum(k, len(t) - 1)
    tu_b = np.where(cens, T, t[kk]); tl_b = np.where(cens, T, np.where(kk > 0, t[np.maximum(kk - 1, 0)], 0.0)); ev_b = ~cens
    ok = ev_b.sum(1) >= 1
    t0b, taub = fit(tu_b, tl_b, ev_b)
    Db = ks(tu_b, ev_b, t0b, taub)[ok]
    res = {"t0": float(t0[0]), "tau": float(tau[0]), "KS_D": D, "KS_p": float((Db >= D).mean()), "n_boot": int(ok.sum())}
    if d == n:
        full = ev_b.all(1)
        x = tu_b[full] - (t0b[full][:, None] if shift else 0)
        cvb = x.std(1, ddof=1) / x.mean(1)
        res["CV_null_95"] = [float(np.percentile(cvb, 2.5)), float(np.percentile(cvb, 97.5))] if full.sum() > 100 else None
        res["CV_null_median"] = float(np.median(cvb)) if full.sum() > 100 else None
    return res


def rho_fits(t, rho, s):
    m = rho.mean(-1); S = s.mean(-1)
    out = {}
    if (~s[-1]).sum() < 3:                      # fewer than 3 extinctions by T: nothing decays, no decay law to fit
        return {"main": None, "legacy": None, "note": "fewer than 3 extinctions by T",
                "n_active_survivors_median": float(np.median(N * rho.sum(-1)[t >= RHO_TMIN] / np.maximum(s.sum(-1)[t >= RHO_TMIN], 1)))}
    life = t[np.argmax(m == 0)] if (m == 0).any() else np.inf
    for name, w in [("main", (t >= RHO_TMIN) & (S >= RHO_SMIN) & (m > 0)),
                    ("legacy", (t >= 5) & (t <= (0.9 * life if np.isfinite(life) else t[-1])) & (m > 0))]:
        if w.sum() < 6:
            out[name] = None; continue
        lt, lr, tt = np.log(t[w]), np.log(m[w]), t[w]
        sst = ((lr - lr.mean()) ** 2).sum()
        b, a = np.polyfit(lt, lr, 1); rss_p = ((lr - a - b * lt) ** 2).sum()
        k, c = np.polyfit(tt, lr, 1); rss_e = ((lr - c - k * tt) ** 2).sum()
        X = np.c_[np.ones_like(lt), lt, tt]; coef, *_ = np.linalg.lstsq(X, lr, rcond=None); rss_c = ((lr - X @ coef) ** 2).sum()
        nw = int(w.sum())
        aic = lambda rss, p: nw * np.log(max(rss, 1e-300) / nw) + 2 * p
        out[name] = {"t_range": [float(tt[0]), float(tt[-1])], "n_pts": nw,
                     "power_b": float(b), "power_a": float(a), "R2_power": float(1 - rss_p / sst),
                     "exp_tau": float(-1 / k) if k < 0 else None, "exp_c": float(c), "R2_exp": float(1 - rss_e / sst),
                     "cut_b": float(coef[1]), "cut_tau": float(-1 / coef[2]) if coef[2] < 0 else None,
                     "R2_cut": float(1 - rss_c / sst),
                     "AIC_power": float(aic(rss_p, 2)), "AIC_exp": float(aic(rss_e, 2)), "AIC_cut": float(aic(rss_c, 3)),
                     "better_2par": "exponential" if rss_e < rss_p else "power law"}
    w = (t >= RHO_TMIN) & (S >= RHO_SMIN)
    out["n_active_survivors_median"] = float(np.median(N * rho.sum(-1)[w] / s.sum(-1)[w])) if w.any() else None
    return out


# ----------------------------------------------------------------------------------------------- theory
def load_theory(kind):
    parts = sorted(glob.glob(f"{R}/pd/branch_{kind}_dense_part*.json"))
    if len(parts) == 8:
        rows, gl = [], None
        for p in parts:
            dd = json.load(open(p)); rows += dd["rows"]; gl = dd["g_lin"]
        src = f"results/pd/branch_{kind}_dense_part0-7.json"
    else:
        dd = json.load(open(f"{R}/pd/branch_{kind}.json"))
        rows, gl = dd["variants"]["branching"]["rows"], dd["variants"]["branching"]["g_lin"]
        src = f"results/pd/branch_{kind}.json (variants.branching; dense parts present: {len(parts)}/8)"
    rows = sorted(rows, key=lambda r: r["g"])
    th = {"src": src, "g_lin": float(gl), "g": np.array([r["g"] for r in rows]), "lam": np.array([r["lam"] for r in rows]),
          "logtau": np.array([r["logtau_r1"] for r in rows]), "K1": np.array([r["K_r1"] for r in rows]),
          "K0": np.array([r["K_r0"] for r in rows]), "logtau0": np.array([r["logtau_r0"] for r in rows])}
    # monotone stretch of logtau_r1 (the dense real table becomes noisy above g ~ 1.3, far outside the data)
    mono = np.r_[True, np.diff(th["logtau"]) > 0]
    th["n_mono"] = int(np.argmin(mono)) if not mono.all() else len(mono)
    th["dense_parts"] = parts
    return th


def interp_logg(th, key, g):
    gg = np.log(th["g"][:th["n_mono"]]); y = th[key][:th["n_mono"]]
    lg = np.log(np.atleast_1d(g))
    v = np.interp(lg, gg, y)
    v[(lg < gg[0]) | (lg > gg[-1])] = np.nan
    return v


def theory_onset(th, lnT):
    y = th["logtau"][:th["n_mono"]]; lg = np.log(th["g"][:th["n_mono"]])
    j = np.argmax(y >= lnT)
    if y[j] < lnT or j == 0:
        return np.nan
    return float(np.exp(lg[j - 1] + (lnT - y[j - 1]) * (lg[j] - lg[j - 1]) / (y[j] - y[j - 1])))


def log_tau(lam, K):
    """identical to theory_branch_gpu.log_tau (exact birth-death mean extinction time from the top state)"""
    n_top = int(np.floor(K))
    if n_top < 1 or lam <= 0:
        return -np.inf
    n = np.arange(1, n_top + 1, dtype=float)
    lb = lam * n * (1 - n / K); lb[-1] = 0.0
    ld = A_EFF * n
    logD = np.empty(n_top); logD[-1] = -np.log(ld[-1])
    for k in range(n_top - 2, -1, -1):
        logD[k] = np.logaddexp(0.0, np.log(lb[k]) + logD[k + 1]) - np.log(ld[k])
    return float(np.logaddexp.reduce(logD))


def K_needed(lam, ln_target, kmax=3000.0):
    if not np.isfinite(ln_target) or not np.isfinite(lam):
        return np.nan
    lo, hi = 1.0, kmax
    if log_tau(lam, lo) > ln_target:
        return np.nan
    if log_tau(lam, hi) < ln_target:
        return np.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if log_tau(lam, mid) < ln_target else (lo, mid)
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------------------------- T-drift helpers
def pava(y, w):
    y, w = list(map(float, y)), list(map(float, w)); blocks = [[y[i], w[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks.pop(i + 1)
            a[0] = (a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]); a[1] += b[1]; a[2] += b[2]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[b[0]] * b[2] for b in blocks])


def half_crossing(g, alive, n):
    """g where the isotonic (in g) survival fraction crosses 1/2, linear interpolation in log g.
    returns (value, flag) with flag 'ok', 'above_range' (never reaches 1/2: g_half > max g), 'below_range'."""
    p = pava(alive / n, n)
    if p[-1] < 0.5:
        return np.nan, "above_range"
    if p[0] >= 0.5:
        return np.nan, "below_range"
    j = int(np.argmax(p >= 0.5)); lg = np.log(g)
    return float(np.exp(lg[j - 1] + (0.5 - p[j - 1]) * (lg[j] - lg[j - 1]) / (p[j] - p[j - 1]))), "ok"


def logistic_half(g, alive, n):
    lg = np.log(g)

    def nll(th):
        m, ls = th; z = (lg - m) / np.exp(ls)
        return -np.sum(alive * -np.logaddexp(0, -z) + (n - alive) * -np.logaddexp(0, z))
    best = None
    for m0 in np.linspace(lg.min(), lg.max(), 7):
        r = optimize.minimize(nll, [m0, np.log(0.05)], method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def surv_counts(sets, T):
    """pooled per-g (alive at T, n) from every sweep with T_run >= T; survival read at the recording time nearest T."""
    rows = {}
    used = []
    for name, S in sets.items():
        if S["T"] < T * 0.999:
            continue
        k = int(np.argmin(np.abs(S["t"] - T)))
        used.append({"set": name, "t_read": float(S["t"][k])})
        for i, g in enumerate(S["g"]):
            key = round(float(g), 4)
            a, n = int(S["surv"][i, k].sum()), S["surv"].shape[2]
            if key in rows:
                rows[key][0] += a; rows[key][1] += n
            else:
                rows[key] = [a, n]
    g = np.array(sorted(rows)); a = np.array([rows[x][0] for x in g]); n = np.array([rows[x][1] for x in g])
    return g, a, n, used


def smooth_tau_fit(rep_g, rep_t, rep_e, deg):
    """censored-exponential ML with ln tau = poly(ln g); returns coefficients (highest power first)."""
    lg = np.log(rep_g)

    def nll(c):
        lt = np.polyval(c, lg)
        return np.sum(rep_e * lt + rep_t * np.exp(-lt))
    c0 = np.polyfit(lg, np.log(np.maximum(rep_t, 1.0)), deg)
    r = optimize.minimize(nll, c0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 20000})
    r = optimize.minimize(nll, r.x, method="BFGS")
    return r.x, float(r.fun)


def solve_g(c, ln_target, glo, ghi):
    f = lambda lg: np.polyval(c, lg) - ln_target
    a, b = np.log(glo), np.log(ghi)
    if f(a) * f(b) > 0:
        return np.nan
    return float(np.exp(optimize.brentq(f, a, b)))


# ----------------------------------------------------------------------------------------------- main
RES = {"inputs": {}, "settings": {"alpha_eff": A_EFF, "n_boot": NBOOT, "rho_fit_window": f"t >= {RHO_TMIN}, surviving fraction >= {RHO_SMIN}",
                                  "extinction_time": "first recorded silent time (upper bound); censored at T"}}
LIFE = {}; THEORY = {}; SETS = {}
for kind in ["real", "null"]:
    sets = load_sets(kind); SETS[kind] = sets
    th = load_theory(kind); THEORY[kind] = th
    RES["inputs"][kind] = {name: {"path": S["path"], "T": S["T"], "B": S["B"], "g": [float(x) for x in S["g"]]} for name, S in sets.items()}
    RES["inputs"][kind]["theory"] = th["src"]
    # sanity: the reimplemented recursion reproduces the stored logtau
    j = len(th["g"]) // 3
    chk = log_tau(th["lam"][j], th["K1"][j])
    RES["inputs"][kind]["theory_recursion_check"] = {"g": float(th["g"][j]), "stored_logtau_r1": float(th["logtau"][j]), "recomputed": chk}
    rows = []
    for name in LIFE_SETS:
        if name not in sets:
            continue
        S = sets[name]; t = S["t"]
        for i, g in enumerate(S["g"]):
            s = S["surv"][i]
            t_up, t_lo, ev = extinction(t, s)
            d = int(ev.sum()); n = len(ev)
            tau, lo, hi = exp_mle(t_up, ev)
            tau_lo_bound = exp_mle(t_lo, ev)[0]
            tau_int = exp_interval_mle(t_lo, t_up, ev)
            bci = boot_tau(t_up, ev) if d > 0 else [np.nan, np.nan]
            km = kaplan_meier(t_up, ev)
            med = float(km[np.argmax(km[:, 1] <= 0.5), 0]) if (km.size and (km[:, 1] <= 0.5).any()) else np.inf
            cv = float(t_up.std(ddof=1) / t_up.mean()) if d == n else np.nan
            if d == n:
                ib = rng.integers(0, n, (NBOOT, n)); x = t_up[ib]
                cv_ci = np.percentile(x.std(1, ddof=1) / x.mean(1), [2.5, 97.5])
            else:
                cv_ci = [np.nan, np.nan]
            g0 = exp_gof(t, t_up, t_lo, ev, shift=False)
            g1 = exp_gof(t, t_up, t_lo, ev, shift=True)
            cv_shift = float((t_up - g1["t0"]).std(ddof=1) / (t_up - g1["t0"]).mean()) if (d == n and g1) else np.nan
            tth = float(np.exp(interp_logg(th, "logtau", g)[0]))
            lam = float(interp_logg(th, "lam", g)[0])
            Kn = K_needed(lam, np.log(tau)) if np.isfinite(tau) and np.isfinite(lam) else np.nan
            row = {"set": name, "T": S["T"], "g": float(g), "n": n, "deaths": d, "censored": n - d,
                   "tau_sim": fnum(tau), "tau_ci95_chi2": [fnum(lo), fnum(hi)], "tau_ci95_boot": [fnum(bci[0]), fnum(bci[1])],
                   "tau_sim_lowerbound_times": fnum(tau_lo_bound), "tau_sim_interval_mle": fnum(tau_int),
                   "median_KM": fnum(med), "median_over_tau_ln2": fnum(med / (tau * np.log(2))) if np.isfinite(tau) and np.isfinite(med) else None,
                   "CV": fnum(cv), "CV_ci95_boot": [fnum(cv_ci[0]), fnum(cv_ci[1])], "CV_shifted": fnum(cv_shift),
                   "exp_test": g0, "shifted_exp_test": g1,
                   "min_death": fnum(t_up[ev].min()) if d else None, "max_death": fnum(t_up[ev].max()) if d else None,
                   "tau_theory": fnum(tth), "ratio_sim_theory": fnum(tau / tth) if np.isfinite(tth) and np.isfinite(tau) else None,
                   "ratio_ci95": [fnum(lo / tth), fnum(hi / tth)] if np.isfinite(tth) else [None, None],
                   "lam_theory": fnum(lam), "K_theory_r1": fnum(interp_logg(th, "K1", g)[0]), "K_theory_r0": fnum(interp_logg(th, "K0", g)[0]),
                   "K_needed": fnum(Kn),
                   "rho_decay": rho_fits(t, S["rho"][i], s),
                   "km": km.tolist()}
            rows.append(row)
    LIFE[kind] = rows

    # ---- theory/sim summary over the metastable range: theory defined, >= 5 deaths, tau_sim >= 200
    sel = [r for r in rows if r["ratio_sim_theory"] and r["deaths"] >= 5 and r["tau_sim"] >= 200]
    ratios = np.array([r["ratio_sim_theory"] for r in sel]); gs = np.array([r["g"] for r in sel])
    summ = {"selection": "theory defined, deaths >= 5, tau_sim >= 200", "n_g": len(sel), "g": gs.tolist()}
    if len(sel):
        summ.update({"ratio_geomean": float(np.exp(np.log(ratios).mean())), "ratio_median": float(np.median(ratios)),
                     "ratio_min": float(ratios.min()), "ratio_max": float(ratios.max())})
        if len(sel) >= 3:
            sl = np.polyfit(np.log(gs), np.log(ratios), 1)
            summ["dln_ratio_dln_g"] = float(sl[0])
        # equivalent coupling: the theory g that gives tau_sim, relative to the simulated g
        geq = []
        for r in sel:
            ge = theory_onset(th, np.log(r["tau_sim"]))
            geq.append(ge / r["g"] if np.isfinite(ge) else np.nan)
        summ["g_theory_equiv_over_g"] = [fnum(x) for x in geq]
        summ["g_theory_equiv_over_g_median"] = fnum(np.nanmedian(geq))
        Kn = np.array([r["K_needed"] for r in sel if r["K_needed"]])
        summ["K_needed_median"] = fnum(np.median(Kn)) if len(Kn) else None
        summ["K_theory_r1_median"] = fnum(np.median([r["K_theory_r1"] for r in sel]))
        summ["K_theory_r0_median"] = fnum(np.median([r["K_theory_r0"] for r in sel]))
    RES.setdefault("theory_vs_sim", {})[kind] = summ

    # ---- pooled exponential-shape summary over the slow window (independent runs -> Fisher's method)
    sw = [r for r in rows if r["tau_sim"] and r["tau_sim"] >= 200 and r["deaths"] >= 10 and r["exp_test"]]
    full = [r for r in sw if r["CV"] is not None]
    pool = {"selection": "tau_sim >= 200 and >= 10 deaths", "g": [r["g"] for r in sw], "sets": [r["set"] for r in sw]}
    for key in ["exp_test", "shifted_exp_test"]:
        ps = np.clip([r[key]["KS_p"] for r in sw], 1.0 / NBOOT, 1)
        pool[f"{key}_fisher_p"] = float(stats.chi2.sf(-2 * np.log(ps).sum(), 2 * len(ps))) if len(ps) else None
        pool[f"{key}_n_reject_05"] = int((ps < 0.05).sum())
    if full:
        pool["CV_g"] = [r["g"] for r in full]
        pool["CV_mean"] = float(np.mean([r["CV"] for r in full]))
        pool["CV_shifted_mean"] = float(np.mean([r["CV_shifted"] for r in full]))
        pool["CV_null_median_mean"] = float(np.mean([r["exp_test"]["CV_null_median"] for r in full]))
        pool["n_CV_below_1"] = int(sum(r["CV"] < 1 for r in full)); pool["n_CV"] = len(full)
        pool["n_CV_above_H0_97.5"] = int(sum(r["CV"] > r["exp_test"]["CV_null_95"][1] for r in full))
        pool["n_CV_below_H0_2.5"] = int(sum(r["CV"] < r["exp_test"]["CV_null_95"][0] for r in full))
    wr = [r for r in sw if r["rho_decay"]["main"] and r["rho_decay"]["legacy"]]
    pool["rho_n"] = len(wr)
    pool["rho_main_prefers_exp"] = int(sum(r["rho_decay"]["main"]["better_2par"] == "exponential" for r in wr))
    pool["rho_legacy_prefers_power"] = int(sum(r["rho_decay"]["legacy"]["better_2par"] == "power law" for r in wr))
    pool["rho_main_tau_exp_over_tau_sim"] = [fnum(r["rho_decay"]["main"]["exp_tau"] / r["tau_sim"]) if r["rho_decay"]["main"]["exp_tau"] else None for r in wr]
    pool["rho_main_cut_b"] = [r["rho_decay"]["main"]["cut_b"] for r in wr]
    pool["rho_legacy_power_b"] = [r["rho_decay"]["legacy"]["power_b"] for r in wr]
    RES.setdefault("exp_shape_pooled", {})[kind] = pool

    # ---- null dense-part cross-check against the table in use
    if kind == "null" and len(th["dense_parts"]) and len(th["dense_parts"]) < 8:
        cmp = []
        for p in th["dense_parts"]:
            for rr in json.load(open(p))["rows"]:
                v = interp_logg(th, "logtau", rr["g"])[0]
                if np.isfinite(v):
                    cmp.append({"g": rr["g"], "dense_logtau_r1": rr["logtau_r1"], "narrow_interp": float(v)})
        RES["inputs"][kind]["dense_vs_narrow"] = cmp

    # ---- T-drift
    drift = {"g_lin": th["g_lin"], "T": {}}
    # smooth ln tau(ln g) fit for the model-based half-survival coupling
    reps = []
    for name in LIFE_SETS:
        if name not in sets:
            continue
        S = sets[name]
        for i, g in enumerate(S["g"]):
            t_up, t_lo, ev = extinction(S["t"], S["surv"][i])
            reps.append((np.full(len(ev), g), t_up, ev.astype(float)))
    rg = np.concatenate([x[0] for x in reps]); rt = np.concatenate([x[1] for x in reps]); re_ = np.concatenate([x[2] for x in reps])
    # window rule (fixed before looking at the fit): g with per-g tau_sim >= 200, up to and including the first two g
    # (in increasing order) without any death
    per = sorted({(r["g"], r["tau_sim"] if r["tau_sim"] else np.inf) for r in rows})
    keep, nz = [], 0
    for g, tau in per:
        if tau >= 200:
            keep.append(g)
            if not np.isfinite(tau):
                nz += 1
                if nz == 2:
                    break
    win = np.isin(np.round(rg, 6), np.round(keep, 6))
    fits = {}
    for deg in (1, 2):
        c, f = smooth_tau_fit(rg[win], rt[win], re_[win], deg); fits[deg] = (c, f)
    lrt_p = float(stats.chi2.sf(2 * (fits[1][1] - fits[2][1]), 1))
    deg = 2 if lrt_p < 0.05 else 1
    c_best = fits[deg][0]
    glo, ghi = min(keep), max(keep)
    boot_c = []
    ug = np.unique(rg[win])
    for b in range(300):
        bi = np.concatenate([rng.choice(np.where(win & (np.round(rg, 6) == round(g, 6)))[0], (np.round(rg[win], 6) == round(g, 6)).sum()) for g in ug])
        cb, _ = smooth_tau_fit(rg[bi], rt[bi], re_[bi], deg); boot_c.append(cb)
    drift["smooth_fit"] = {"g_window": [float(glo), float(ghi)], "g_used": [float(x) for x in keep], "n_replicas": int(win.sum()),
                           "deg_selected": deg, "LRT_p_quadratic_vs_linear": lrt_p, "coef_lng_highest_first": c_best.tolist(),
                           "nll": {str(k): v[1] for k, v in fits.items()}}
    for T in TS:
        g, a, n, used = surv_counts(sets, T)
        gh, flag = half_crossing(g, a, n)
        bs = []
        for b in range(NBOOT):
            ab = rng.binomial(n, a / n)
            v, fl = half_crossing(g, ab, n)
            bs.append(v if fl == "ok" else np.nan)
        bs = np.array(bs)
        lg_half = logistic_half(g, a, n) if (a.sum() > 0 and (n - a).sum() > 0) else (np.nan, np.nan)
        model = solve_g(c_best, np.log(T / np.log(2)), glo, ghi)
        mb = np.array([solve_g(cb, np.log(T / np.log(2)), glo, ghi) for cb in boot_c])
        per_set = {}
        for name, S in sets.items():
            if S["T"] < T * 0.999:
                continue
            k = int(np.argmin(np.abs(S["t"] - T)))
            v, fl = half_crossing(S["g"], S["surv"][:, k].sum(-1), np.full(len(S["g"]), S["surv"].shape[2]))
            per_set[name] = {"g_half": fnum(v), "flag": fl}
        dirn = {"g_half": fnum(gh), "flag": flag, "ci95_boot": [fnum(np.nanpercentile(bs, 2.5)), fnum(np.nanpercentile(bs, 97.5))] if np.isfinite(bs).sum() > 100 else None,
                "boot_frac_defined": float(np.isfinite(bs).mean()),
                "g_half_logistic": fnum(lg_half[0]), "logistic_width_lng": fnum(lg_half[1]), "per_set": per_set,
                "sets_used": used, "max_g_with_data": float(g.max()),
                "max_g_with_zero_survivors": fnum(g[a == 0].max()) if (a == 0).any() else None,
                "min_g_with_all_surviving": fnum(g[a == n].min()) if (a == n).any() else None,
                "table": [[float(x), int(y), int(z)] for x, y, z in zip(g, a, n)]}
        drift["T"][str(int(T))] = {"sim_direct": dirn,
                                   "sim_model_exp": {"g_half": fnum(model), "ci95_boot": [fnum(np.nanpercentile(mb, 2.5)), fnum(np.nanpercentile(mb, 97.5))] if np.isfinite(mb).sum() > 50 else None},
                                   "theory_tau_eq_T": fnum(theory_onset(th, np.log(T))),
                                   "theory_tau_eq_T_over_ln2": fnum(theory_onset(th, np.log(T / np.log(2))))}
    # drift summaries
    def best_sim(Tk):
        e = drift["T"][Tk]
        return e["sim_direct"]["g_half"] if e["sim_direct"]["g_half"] else e["sim_model_exp"]["g_half"]
    sim_vals = {Tk: best_sim(Tk) for Tk in drift["T"]}
    th_vals = {Tk: drift["T"][Tk]["theory_tau_eq_T"] for Tk in drift["T"]}
    th2_vals = {Tk: drift["T"][Tk]["theory_tau_eq_T_over_ln2"] for Tk in drift["T"]}
    mdl_vals = {Tk: drift["T"][Tk]["sim_model_exp"]["g_half"] for Tk in drift["T"]}
    summ = {}
    for lab, vals in [("sim_best", sim_vals), ("sim_model_exp", mdl_vals), ("theory_tau_eq_T", th_vals), ("theory_tau_eq_T_over_ln2", th2_vals)]:
        if vals.get("3000") and vals.get("30000"):
            summ[lab] = {"ratio_g_30000_over_3000": vals["30000"] / vals["3000"],
                         "ratio_ln_g_over_glin_30000_over_3000": np.log(vals["30000"] / th["g_lin"]) / np.log(vals["3000"] / th["g_lin"])}
        if vals.get("3000") and vals.get("8000"):
            summ.setdefault(lab, {})["ratio_g_8000_over_3000"] = vals["8000"] / vals["3000"]
    summ["sqrt_lnT_law_prediction_ratio_ln_g_over_glin_30000_over_3000"] = float(np.sqrt(np.log(A_EFF * 30000) / np.log(A_EFF * 3000)))
    summ["sqrt_lnT_law_prediction_ratio_ln_g_over_glin_8000_over_3000"] = float(np.sqrt(np.log(A_EFF * 8000) / np.log(A_EFF * 3000)))
    drift["summary"] = summ
    RES.setdefault("T_drift", {})[kind] = drift

RES["lifetimes"] = {k: [{kk: vv for kk, vv in r.items() if kk != "km"} for r in v] for k, v in LIFE.items()}
RES["km_curves"] = {k: {f"{r['set']}_g{r['g']:g}": r["km"] for r in v} for k, v in LIFE.items()}
json.dump(RES, open(f"{OUT}/longruns.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("saved", f"{OUT}/longruns.json")

# ----------------------------------------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ORANGES = ["#fd9a4e", "#f3701b", "#d84801", "#9e3303"]
C_REAL, C_NULL = "#2a78d6", "#eb6834"


def by(kind, name):
    return [r for r in LIFE[kind] if r["set"] == name]


fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
for row, kind, cols in [(0, "real", BLUES), (1, "null", ORANGES)]:
    S = SETS[kind]["long"]; L = by(kind, "long")
    ax = axs[row, 0]
    for r, c in zip(L, cols):
        km = np.array(r["km"])
        tt = np.r_[0.2, np.repeat(km[:, 0], 2)]; ss = np.r_[1.0, 1.0, np.repeat(km[:-1, 1], 2), km[-1, 1]] if len(km) > 1 else None
        x = np.r_[0.2, km[:, 0]]; y = np.r_[1.0, km[:, 1]]
        ax.step(x, y, where="post", c=c, lw=2, label=f"g = {r['g']:.2f}  (τ = {r['tau_sim']:.0f})")
        se = np.r_[0.0, km[:, 2]]
        ax.fill_between(x, np.clip(y - 1.96 * se, 0, 1), np.clip(y + 1.96 * se, 0, 1), step="post", color=c, alpha=0.09, lw=0)
    ax.set_xscale("log"); ax.set_xlim(5, 3e4); ax.set_ylim(0, 1.02)
    ax.set_xlabel("time t (units of 1/α)"); ax.set_ylabel("fraction of replicas still active  S(t)")
    ax.set_title(f"{'MaleCNS' if kind == 'real' else 'configuration null'}: Kaplan–Meier survival, T = 30000, 32 replicas", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax = axs[row, 1]
    for r, c in zip(L, cols):
        km = np.array(r["km"]); x = np.r_[0, km[:, 0]] / r["tau_sim"]; y = np.r_[1.0, km[:, 1]]
        m = y > 0
        ax.step(x[m], y[m], where="post", c=c, lw=2, label=f"g = {r['g']:.2f}:  p = {r['exp_test']['KS_p']:.2f} ({r['shifted_exp_test']['KS_p']:.2f});  CV = {r['CV']:.2f}")
    xx = np.linspace(0, 5, 100)
    ax.plot(xx, np.exp(-xx), c="#0b0b0b", ls="--", lw=1.2, label="exponential  exp(−t/τ)")
    ax.set_yscale("log"); ax.set_xlim(0, 4.5); ax.set_ylim(1 / 40, 1.1)
    ax.set_xlabel("t / τ_sim  (τ_sim = censored-exponential MLE)"); ax.set_ylabel("S(t)  (log scale)")
    ax.set_title("rescaled survival: a straight line = memoryless (single metastable state)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", title="KS p: exponential (with lag); CV", title_fontsize=7.5)
    ax = axs[row, 2]
    t = S["t"]
    for i, (r, c) in enumerate(zip(L, cols)):
        m = S["rho"][i].mean(-1)
        ok = m > 0
        ax.loglog(t[ok], m[ok], c=c, lw=1.6, label=f"g = {r['g']:.2f}")
        f = r["rho_decay"]["main"]
        if f:
            tw = np.geomspace(*f["t_range"], 60)
            ax.loglog(tw, np.exp(f["exp_c"] - tw / f["exp_tau"]) if f["exp_tau"] else tw * np.nan, c=c, ls="--", lw=1)
            ax.loglog(tw, np.exp(f["power_a"]) * tw ** f["power_b"], c=c, ls=":", lw=1.3)
    ax.set_xlim(1, 3e4); ax.set_ylim(1e-7, 1.5)
    ax.set_xlabel("time t"); ax.set_ylabel("mean fraction of neurons active  ρ(t)")
    ax.set_title(f"mean density (all replicas); fits over t ≥ {RHO_TMIN:g} while ≥ 25% survive", fontsize=9)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], c="#52514e", ls="--", lw=1), Line2D([], [], c="#52514e", ls=":", lw=1.3)]
    l += ["exponential fit", "power-law fit"]
    ax.legend(h, l, fontsize=7.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/longruns_survival.png"); plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(13, 9.4))
ax = axs[0, 0]
for kind, c, mk in [("real", C_REAL, "o"), ("null", C_NULL, "s")]:
    th = THEORY[kind]
    gg = th["g"][:th["n_mono"]]; m = gg <= (0.75 if kind == "real" else 1.5)
    ax.plot(gg[m], np.exp(th["logtau"][:th["n_mono"]][m]), c=c, lw=2, label=f"{kind} theory, r = 1 (exp logtau_r1)")
    for name, fill in [("long", True), ("fine", False)]:
        rr = [r for r in by(kind, name) if r["tau_sim"]]
        g = np.array([r["g"] for r in rr]); tau = np.array([r["tau_sim"] for r in rr])
        lo = np.array([r["tau_ci95_chi2"][0] for r in rr]); hi = np.array([r["tau_ci95_chi2"][1] for r in rr])
        ax.errorbar(g, tau, yerr=[tau - lo, hi - tau], fmt=mk, ms=6, c=c, mfc=c if fill else "white", mew=1.4, lw=1, capsize=0,
                    label=f"{kind} sim, {'T = 30000' if name == 'long' else 'T = 8000'}")
        rc = [r for r in by(kind, name) if not r["tau_sim"]]
        if rc:
            ax.plot([r["g"] for r in rc], [r["tau_ci95_chi2"][0] for r in rc], "^", ms=6, c=c, mfc="white", mew=1.2,
                    label=f"{kind}: no deaths by T (95% lower bound)" if name == "fine" else None)
    ax.axvline(th["g_lin"], c=c, lw=0.8, ls="-.")
for T in TS:
    ax.axhline(T, c="#898781", lw=0.8, ls=":")
    ax.text(0.46, T * 1.1, f"T = {T:g}", fontsize=7, color="#52514e")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.15, 1.5); ax.set_ylim(3, 1e8)
ax.set_xlabel("coupling g  (dash-dot: g_lin)"); ax.set_ylabel("mean lifetime τ")
ax.set_title("mean lifetime from a fully active start, r = 1 (bars: 95% χ² CI)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)

ax = axs[0, 1]
for kind, c, mk in [("real", C_REAL, "o"), ("null", C_NULL, "s")]:
    for name, fill in [("long", True), ("fine", False)]:
        rr = [r for r in by(kind, name) if r["ratio_sim_theory"]]
        g = np.array([r["g"] for r in rr]); q = np.array([r["ratio_sim_theory"] for r in rr])
        lo = np.array([r["ratio_ci95"][0] for r in rr]); hi = np.array([r["ratio_ci95"][1] for r in rr])
        ax.errorbar(g, q, yerr=[q - lo, hi - q], fmt=mk, ms=6, c=c, mfc=c if fill else "white", mew=1.4, lw=1,
                    label=f"{kind} sim {'T = 30000' if name == 'long' else 'T = 8000'}")
ax.axhline(1, c="#0b0b0b", lw=1); ax.axhline(15, c="#898781", lw=1, ls="--")
ax.text(0.47, 16.5, "×15 (earlier estimate, STATE.md)", fontsize=7, color="#52514e")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.17, 1.5)
ax.set_xlabel("coupling g"); ax.set_ylabel("τ_sim / τ_theory")
ax.set_title("how much the theory underestimates the lifetime (only g where theory is defined)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper left")

ax = axs[1, 0]
Tx = np.array(TS)
Tl = np.geomspace(3000, 4e4, 40)
for kind, c, mk, dx in [("real", C_REAL, "o", 1.0), ("null", C_NULL, "s", 1.04)]:
    D = RES["T_drift"][kind]["T"]; gl = THEORY[kind]["g_lin"]
    g3 = D["3000"]["sim_direct"]["g_half"]
    for T in TS:
        e = D[str(int(T))]["sim_direct"]
        if e["g_half"]:
            ci = e["ci95_boot"]
            ax.errorbar([T * dx], [e["g_half"] / g3], yerr=[[(e["g_half"] - ci[0]) / g3], [(ci[1] - e["g_half"]) / g3]] if ci else None,
                        fmt=mk, c=c, ms=7, lw=1.2)
        elif e["max_g_with_zero_survivors"]:
            ax.plot([T * dx], [e["max_g_with_zero_survivors"] / g3], "^", c=c, ms=7, mfc="white", mew=1.3)
        m0 = D[str(int(T))]["sim_model_exp"]
        if m0["g_half"]:
            ci = m0["ci95_boot"]
            ax.errorbar([T * dx * 1.07], [m0["g_half"] / g3], yerr=[[(m0["g_half"] - ci[0]) / g3], [(ci[1] - m0["g_half"]) / g3]] if ci else None,
                        fmt=mk, c=c, mfc="white", mew=1.3, ms=7, lw=1)
    th3 = theory_onset(THEORY[kind], np.log(3000.0))
    ax.plot(Tl, [theory_onset(THEORY[kind], np.log(T)) / th3 for T in Tl], c=c, lw=2)
    # asymptotic law ln(g_on/g_lin) ~ sqrt(ln(alpha T)), anchored at the simulated T = 3000 onset
    ax.plot(Tl, gl * np.exp(np.log(g3 / gl) * np.sqrt(np.log(A_EFF * Tl) / np.log(A_EFF * 3000))) / g3, c=c, lw=1.2, ls=":")
ax.axhline(1, c="#c3c2b7", lw=0.8)
leg = [Line2D([], [], c=C_REAL, lw=5, label="MaleCNS"), Line2D([], [], c=C_NULL, lw=5, label="configuration null"),
       Line2D([], [], c="#52514e", marker="o", ls="none", label="sim: half the replicas alive at T (direct, 95% CI)"),
       Line2D([], [], c="#52514e", marker="o", mfc="white", ls="none", label="sim: from the smooth exponential τ(g) fit"),
       Line2D([], [], c="#52514e", marker="^", mfc="white", ls="none", label="sim T = 30000: no survivor at the largest g run (bound)"),
       Line2D([], [], c="#52514e", lw=2, label="theory: g where τ_theory = T (own T = 3000 value = 1)"),
       Line2D([], [], c="#52514e", lw=1.2, ls=":", label="√ln(αT) law, anchored at the sim T = 3000 onset")]
ax.legend(handles=leg, fontsize=7, frameon=False, loc="upper left")
ax.set_xscale("log"); ax.set_xlim(2500, 4.5e4); ax.set_ylim(0.93, 1.5)
ax.set_xticks([3000, 8000, 30000]); ax.set_xticklabels(["3000", "8000", "30000"])
ax.set_xlabel("observation time T"); ax.set_ylabel("apparent onset g_on(T) / g_on(3000)")
ax.set_title("T-drift of the apparent onset, r = 1 (absolute values in longruns.md)", fontsize=9)

ax = axs[1, 1]
for kind, c, mk in [("real", C_REAL, "o"), ("null", C_NULL, "s")]:
    th = THEORY[kind]; gg = th["g"][:th["n_mono"]]; m = gg <= (0.75 if kind == "real" else 1.5)
    ax.plot(gg[m], th["K1"][:th["n_mono"]][m], c=c, lw=2, label=f"{kind}: theory K_eff, r = 1")
    ax.plot(gg[m], th["K0"][:th["n_mono"]][m], c=c, lw=1.2, ls="--", label=f"{kind}: theory K_eff, r = 0")
    for name, fill in [("long", True), ("fine", False)]:
        rr = [r for r in by(kind, name) if r["K_needed"] and r["deaths"] >= 5 and r["tau_sim"] >= 200]
        ax.plot([r["g"] for r in rr], [r["K_needed"] for r in rr], mk, c=c, mfc=c if fill else "white", mew=1.4, ms=6,
                label=f"{kind}: K needed to match τ_sim ({'T = 30000' if name == 'long' else 'T = 8000'}; τ_sim ≥ 200, ≥ 5 deaths)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.17, 1.5)
ax.set_xlabel("coupling g"); ax.set_ylabel("effective module size K")
ax.set_title("module size the birth–death recursion needs to reproduce τ_sim (λ from theory)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{FIG}/lifetime_sim_vs_theory.png"); plt.close(fig)
print("figures saved")

# ----------------------------------------------------------------------------------------------- markdown
def f(x, p=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{p}g}"


md = ["# Slow-window lifetimes at r = 1: simulation vs extinction theory", "",
      "Generated by `code/analysis2_longruns.py`. Extinction time = first recorded silent time (log-spaced grid, ~10% "
      "resolution); censored at T. τ_sim = Σ t / deaths (censored-exponential MLE), 95% CI χ² (time-truncated). "
      "KS p: parametric bootstrap (grid, censoring and re-fit reproduced), exponential through the origin (p0) and with a lag t0 (p_shift). "
      "CV only when no replica is censored. τ_th = exp(logtau_r1) interpolated in log g. "
      f"Theory sources: real = {THEORY['real']['src']}; null = {THEORY['null']['src']}.", ""]
for kind in ["real", "null"]:
    md += [f"## {kind}: per-g lifetimes", "",
           "| set | T | g | deaths/n | τ_sim [95% CI] | τ interval-MLE | median | med/(τ ln2) | CV [boot CI] (H0 range) | KS p0 | lag t0 | KS p_shift | τ_th | τ_sim/τ_th | K_needed | ρ fit: b (R²) vs τ_exp (R²) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in LIFE[kind]:
        e0, e1, rd = r["exp_test"], r["shifted_exp_test"], r["rho_decay"]["main"]
        cvs = f"{f(r['CV'], 2)} [{f(r['CV_ci95_boot'][0], 2)}, {f(r['CV_ci95_boot'][1], 2)}]" if r["CV"] else "–"
        if r["CV"] and e0 and e0.get("CV_null_95"):
            cvs += f" ({e0['CV_null_95'][0]:.2f}–{e0['CV_null_95'][1]:.2f})"
        tau_s = f"{f(r['tau_sim'], 4)} [{f(r['tau_ci95_chi2'][0], 3)}, {f(r['tau_ci95_chi2'][1], 3)}]" if r["tau_sim"] else f"> {f(r['tau_ci95_chi2'][0], 3)}"
        rds = f"{rd['power_b']:.2f} ({rd['R2_power']:.3f}) vs {f(rd['exp_tau'], 3)} ({rd['R2_exp']:.3f})" if rd else "–"
        md.append(f"| {r['set']} | {r['T']:g} | {r['g']:.4g} | {r['deaths']}/{r['n']} | {tau_s} | {f(r['tau_sim_interval_mle'], 4)} | {f(r['median_KM'], 4)} | "
                  f"{f(r['median_over_tau_ln2'], 2)} | {cvs} | {f(e0['KS_p'], 2) if e0 else '–'} | {f(e1['t0'], 3) if e1 else '–'} | {f(e1['KS_p'], 2) if e1 else '–'} | "
                  f"{f(r['tau_theory'], 3)} | {f(r['ratio_sim_theory'], 3)} | {f(r['K_needed'], 3)} | {rds} |")
    s = RES["theory_vs_sim"][kind]
    md += ["", f"**Theory vs sim summary ({kind}; {s['selection']}; {s['n_g']} g values)**: "
               f"τ_sim/τ_th geometric mean {f(s.get('ratio_geomean'))}, median {f(s.get('ratio_median'))}, range {f(s.get('ratio_min'))}–{f(s.get('ratio_max'))}, "
               f"d ln(ratio)/d ln g = {f(s.get('dln_ratio_dln_g'))}; theory needs g × {f(s.get('g_theory_equiv_over_g_median'))} (median) to reach τ_sim; "
               f"K needed (median) {f(s.get('K_needed_median'))} vs theory K_eff(r=1) {f(s.get('K_theory_r1_median'))}, K_eff(r=0) {f(s.get('K_theory_r0_median'))}.", ""]
    P = RES["exp_shape_pooled"][kind]
    md += [f"**Lifetime shape, pooled ({kind}; {P['selection']}; g = {', '.join(f'{x:.4g}' for x in P['g'])})**: "
           f"Fisher-combined KS p, exponential through 0: {f(P['exp_test_fisher_p'])} ({P['exp_test_n_reject_05']} of {len(P['g'])} g reject at 0.05); "
           f"with lag: {f(P['shifted_exp_test_fisher_p'])} ({P['shifted_exp_test_n_reject_05']} reject). "
           + (f"CV (uncensored g {', '.join(f'{x:.4g}' for x in P['CV_g'])}): mean {P['CV_mean']:.3f} (lag-subtracted {P['CV_shifted_mean']:.3f}) "
              f"vs {P['CV_null_median_mean']:.3f} expected for an exponential with this n and grid; {P['n_CV_below_1']}/{P['n_CV']} below 1, "
              f"{P['n_CV_above_H0_97.5']} above and {P['n_CV_below_H0_2.5']} below the exponential 95% range." if P.get("CV_g") else ""), "",
           f"**Mean density ρ(t), same g set ({P['rho_n']} g)**: window t ≥ {RHO_TMIN:g} (after the transient from the fully active start): "
           f"exponential beats power law in {P['rho_main_prefers_exp']}/{P['rho_n']}; τ_exp/τ_sim = "
           f"{', '.join(f(x, 2) for x in P['rho_main_tau_exp_over_tau_sim'])}; power-law exponent of a power×exponential fit "
           f"{', '.join(f'{x:.2f}' for x in P['rho_main_cut_b'])}. Legacy window of code/analyze.py (t ≥ 5, includes the transient): "
           f"power law preferred in {P['rho_legacy_prefers_power']}/{P['rho_n']}, exponents {', '.join(f'{x:.2f}' for x in P['rho_legacy_power_b'])}.", ""]
    D = RES["T_drift"][kind]
    md += [f"## {kind}: T-drift of the apparent onset (g_lin = {D['g_lin']:.4f})", "",
           "| T | sim g_half (direct) [95% CI] | logistic | per sweep | sim g_half (exp τ(g) model) [CI] | theory τ=T | theory τ=T/ln2 |", "|---|---|---|---|---|---|---|"]
    for Tk, e in D["T"].items():
        d0 = e["sim_direct"]
        dv = f"{f(d0['g_half'], 4)} [{f(d0['ci95_boot'][0], 4)}, {f(d0['ci95_boot'][1], 4)}]" if d0["g_half"] else \
             (f"> {f(d0['max_g_with_zero_survivors'], 3)} (no survivor at any g ≤ {f(d0['max_g_with_data'], 3)})" if d0["flag"] == "above_range" else d0["flag"])
        ps = "; ".join(f"{k}: {f(v['g_half'], 4) if v['g_half'] else v['flag']}" for k, v in d0["per_set"].items())
        m0 = e["sim_model_exp"]
        mv = f"{f(m0['g_half'], 4)} [{f(m0['ci95_boot'][0], 4)}, {f(m0['ci95_boot'][1], 4)}]" if m0["g_half"] and m0["ci95_boot"] else f(m0["g_half"], 4)
        md.append(f"| {Tk} | {dv} | {f(d0['g_half_logistic'], 4)} | {ps} | {mv} | {f(e['theory_tau_eq_T'], 4)} | {f(e['theory_tau_eq_T_over_ln2'], 4)} |")
    sf = D["smooth_fit"]
    md += ["", f"Smooth fit: ln τ = poly{sf['deg_selected']}(ln g) over g ∈ [{sf['g_window'][0]:.3g}, {sf['g_window'][1]:.3g}] "
               f"({sf['n_replicas']} replicas; LRT p quadratic vs linear = {sf['LRT_p_quadratic_vs_linear']:.3g}).", "",
           "Drift summary: " + json.dumps({k: ({kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)) for k, v in D["summary"].items()}), ""]
open(f"{OUT}/longruns.md", "w").write("\n".join(md))
print("saved", f"{OUT}/longruns.md")
