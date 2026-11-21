"""m2_obs.py -- reference observables for the Model-2 phase diagrams (results/m2/MODEL2.md; 2026-11-09).

Every lane should compute its phase labels with these functions so that the criteria are identical across lanes.
Pure numpy (scipy only for the optional KS tables). Tested by code/m2_obs_test.py on synthetic data with known answers.

Lifetimes / boundaries (same rules as analysis2_phase.py, M1):
  extinction_times(t, surv)          per-replica first record time with no active unit (inf if alive at the last record)
  tau_mle(ext, T)                    censored-exponential MLE  sum(min(t_i, T)) / deaths  (inf with no deaths), + deaths
  isotonic(y, w)                     weighted pool-adjacent-violators (non-decreasing)
  first_crossing(g, y, level, w)     isotonic y in g, first crossing of `level`, linear in ln g (None if never)
Localization:
  participation_ratio(occ, mask)     PR = (sum o)^2 / sum o^2 over the masked neurons, and PR / n_mask
  region_shares(occ, masks)          share of the total occupancy in each region
Time series (rho(t) or a region's active fraction, per replica, columns = replicas):
  acf(x, maxlag)                     replica-averaged autocorrelation (each replica: own mean and variance), lag 0..maxlag
  tau_ac(c, dt)                      integrated autocorrelation time up to the first zero crossing, and the 1/e time
  osc_metrics(x, dt)                 oscillation measures + the OSC flag of MODEL2.md (ACF negative lobe + second peak)
  fluct(x, n)                        chi = n * <var_t x>_b  and  chi / <x>
Avalanches (from ald_sim.py --rec_counts):
  avalanches(cnt_act, cnt_on, thr)   maximal runs with active count > thr in each replica: size (onsets; area above thr
                                     for thr > 0), duration (steps), start; runs touching the window ends are dropped
  powerlaw_fit(x, xmin=None)         discrete power-law MLE with Clauset's KS scan over xmin
  mean_size_given_T(S, T)            <S>(T) and the fitted exponent gamma over a T range
  mr_branching(a, kmax)              multistep-regression branching ratio (Wilting & Priesemann 2018) for driven data
"""
import numpy as np


# ------------------------------------------------------------------------------------------------ lifetimes
def extinction_times(t, surv):
    """t[n_rec], surv[n_rec, B] (1 = some unit active) -> per-replica first record time with surv == 0 (inf if never)."""
    surv = np.asarray(surv)
    dead = surv <= 0
    first = np.where(dead.any(0), np.argmax(dead, 0), -1)
    return np.where(first >= 0, np.asarray(t)[np.maximum(first, 0)], np.inf)


def tau_mle(ext, T):
    """Censored-exponential MLE of the lifetime: (sum of min(t_i, T)) / deaths. Returns (tau, deaths)."""
    ext = np.asarray(ext, float)
    d = int(np.isfinite(ext).sum())
    E = float(np.minimum(ext, T).sum())
    return (E / d if d > 0 else np.inf), d


def isotonic(y, w=None):
    """Weighted pool-adjacent-violators: the non-decreasing sequence closest to y in weighted least squares."""
    y = np.asarray(y, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; cnt[-2] += cnt[-1]; vals[-2] = v
            vals.pop(); wts.pop(); cnt.pop()
    return np.repeat(vals, cnt)


def first_crossing(g, y, level, w=None, increasing=True):
    """First g where the isotonic fit of y crosses `level` (interpolated linearly in ln g). None if never crossed."""
    g = np.asarray(g, float); o = np.argsort(g); g, y = g[o], np.asarray(y, float)[o]
    w = None if w is None else np.asarray(w, float)[o]
    yi = isotonic(y if increasing else -y, w)
    lv = level if increasing else -level
    k = np.where(yi >= lv)[0]
    if len(k) == 0:
        return None
    k = k[0]
    if k == 0:
        return float(g[0])
    a, b = yi[k - 1], yi[k]
    f = 0.0 if b == a else (lv - a) / (b - a)
    return float(np.exp(np.log(g[k - 1]) + f * (np.log(g[k]) - np.log(g[k - 1]))))


# ------------------------------------------------------------------------------------------------ localization
def participation_ratio(occ, mask=None):
    """PR = (sum o)^2 / sum o^2 over masked neurons (float64). Returns (PR, PR / n_mask)."""
    o = np.asarray(occ, np.float64)
    if mask is not None:
        o = o[mask]
    s2 = (o * o).sum()
    pr = float(o.sum() ** 2 / s2) if s2 > 0 else 0.0
    return pr, pr / len(o)


def region_shares(occ, masks):
    """{name: share of the total occupancy} for a dict of boolean masks."""
    o = np.asarray(occ, np.float64); tot = o.sum()
    return {k: (float(o[m].sum() / tot) if tot > 0 else np.nan) for k, m in masks.items()}


# ------------------------------------------------------------------------------------------------ time series
def acf(x, maxlag):
    """x[T, B] -> replica-averaged autocorrelation C[0..maxlag] (each replica normalised by its own variance;
    replicas with zero variance are skipped). FFT-based, biased estimator (divides by T)."""
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    T = x.shape[0]
    xc = x - x.mean(0)
    v = (xc * xc).mean(0)
    ok = v > 0
    if not ok.any():
        return np.full(maxlag + 1, np.nan)
    n = 1 << int(np.ceil(np.log2(2 * T)))
    F = np.fft.rfft(xc[:, ok], n, axis=0)
    c = np.fft.irfft(F * np.conj(F), n, axis=0)[:maxlag + 1] / T
    return (c / v[ok]).mean(1)


def tau_ac(c, dt):
    """Integrated autocorrelation time (dt * (1/2 + sum_{k>=1} C_k) up to the first lag with C <= 0) and the first
    lag time with C <= 1/e (inf if never within the computed lags)."""
    c = np.asarray(c, float)
    z = np.where(c[1:] <= 0)[0]
    kz = (z[0] + 1) if len(z) else len(c)
    tint = dt * (0.5 + c[1:kz].sum())
    e = np.where(c <= np.exp(-1))[0]
    return float(tint), (float(e[0] * dt) if len(e) else np.inf)


# OSC flag thresholds (MODEL2.md, pre-registered): negative lobe <= -0.1 AND second peak >= 0.1 AND >= 4 periods fit;
# OSC-strong: the ACF peak at 2P is >= 0.7 of the peak at P (coherence time >= 2.8 periods; white measurement noise
# only lowers C at lag 0, so this ratio does not depend on it)
OSC_NEG, OSC_PEAK, OSC_MIN_PERIODS, OSC_STRONG = -0.10, 0.10, 4, 0.7


def osc_metrics(x, dt, maxlag=None):
    """Oscillation measures of x[T, B] (one region's active fraction, quasi-stationary window, alive replicas only).
    Returns dict: lag_min / c_min (first local minimum of the replica-averaged ACF that is at least half as deep as its
    global minimum), period / c_peak (maximum of the ACF over lags (lag_min, 3 lag_min]), f_peak (frequency of the largest periodogram peak above f = 2/T_window, replica-averaged Welch-free
    periodogram), peak_ratio (that peak / median power in [f/2, 2f]), cv_t (mean over replicas of std_t / mean_t),
    osc (bool: c_min <= OSC_NEG and c_peak >= OSC_PEAK and period <= T_window / OSC_MIN_PERIODS),
    coh_ratio = C(2P)/C(P) (max of C within 2P +- P/4), osc_strong (osc and coh_ratio >= OSC_STRONG: weakly damped,
    limit-cycle-like),
    tau_int, tau_e (autocorrelation times)."""
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    T = x.shape[0]
    maxlag = maxlag or T // 2
    c = acf(x, maxlag)
    out = {"n_rep": int(x.shape[1]), "T_window": T * dt}
    ti, te = tau_ac(c, dt)
    out.update(tau_int=ti, tau_e=te)
    m = x.mean(0)
    out["cv_t"] = float(np.mean(np.where(m > 0, x.std(0) / np.where(m > 0, m, 1), np.nan))) if (m > 0).any() else np.nan
    # trough and peak of the ACF, robust to small noise wiggles: the trough is the FIRST local minimum at least half as
    # deep as the global minimum over lags 1..maxlag; the peak is the maximum of C over (k_min, 3 k_min] (for an
    # oscillation of period P the first trough sits near P/2 and the next peak near P); period = that lag.
    lag_min = c_min = period = c_peak = np.nan
    gmin = np.nanmin(c[1:]) if len(c) > 2 else np.nan
    if np.isfinite(gmin) and gmin < 0:
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
    # periodogram (replica-averaged), largest peak above 2 cycles per window
    xc = x - x.mean(0)
    P = (np.abs(np.fft.rfft(xc, axis=0)) ** 2).mean(1)
    f = np.fft.rfftfreq(T, dt)
    sel = f >= 2.0 / (T * dt)
    if sel.sum() > 3 and P[sel].max() > 0:
        i = np.where(sel)[0][np.argmax(P[sel])]
        band = (f >= f[i] / 2) & (f <= 2 * f[i])
        out["f_peak"] = float(f[i]); out["peak_ratio"] = float(P[i] / np.median(P[band]))
    else:
        out["f_peak"] = np.nan; out["peak_ratio"] = np.nan
    osc = bool(np.isfinite(c_peak) and c_min <= OSC_NEG and c_peak >= OSC_PEAK and period <= T * dt / OSC_MIN_PERIODS)
    out["osc"] = osc
    out["osc_strong"] = bool(osc and np.isfinite(coh) and coh >= OSC_STRONG)
    return out


def fluct(x, n):
    """x[T, B] active fraction of a region with n units: chi = n * mean_b var_t(x_b); also chi / <x> (Fano-like)."""
    x = np.asarray(x, np.float64)
    chi = float(n * x.var(0).mean())
    mu = float(x.mean())
    return chi, (chi / mu if mu > 0 else np.nan)


# ------------------------------------------------------------------------------------------------ avalanches
def avalanches(cnt_act, cnt_on=None, thr=0):
    """Avalanches = maximal runs of consecutive steps with cnt_act > thr, per replica (columns of cnt_act[S, B]).
    size: sum of cnt_on over the run (thr == 0; the onsets, i.e. spikes, incl. the seed) or the area above threshold
    sum(cnt_act - thr) (thr > 0, Poil et al. 2012 convention); duration: run length in steps. Runs that touch the first
    or last step of the window are dropped (censored). Returns dict of arrays size, dur, start, rep."""
    a = np.asarray(cnt_act)
    if a.ndim == 1:
        a = a[:, None]
    on = None if cnt_on is None else (np.asarray(cnt_on) if np.asarray(cnt_on).ndim > 1 else np.asarray(cnt_on)[:, None])
    S, B = a.shape
    out = {"size": [], "dur": [], "start": [], "rep": []}
    for b in range(B):
        act = a[:, b] > thr
        if not act.any():
            continue
        x = np.r_[0, act.astype(np.int8), 0]
        dx = np.diff(x)
        st = np.where(dx == 1)[0]; en = np.where(dx == -1)[0]          # run = [st, en)
        keep = (st > 0) & (en < S)
        st, en = st[keep], en[keep]
        if len(st) == 0:
            continue
        if thr == 0 and on is not None:
            cs = np.r_[0, np.cumsum(on[:, b].astype(np.int64))]
        else:
            cs = np.r_[0, np.cumsum((a[:, b].astype(np.int64) - thr))]
        out["size"].append(cs[en] - cs[st]); out["dur"].append(en - st)
        out["start"].append(st); out["rep"].append(np.full(len(st), b))
    return {k: (np.concatenate(v) if v else np.zeros(0, np.int64)) for k, v in out.items()}


def _pl_alpha(x, xmin):
    """Discrete power-law MLE (Clauset, Shalizi & Newman 2009, eq. 3.7 approximation)."""
    y = x[x >= xmin]
    return 1.0 + len(y) / np.log(y / (xmin - 0.5)).sum(), len(y)


def powerlaw_fit(x, xmin=None, xmin_max=None, n_min=50):
    """Discrete power law P(x) ~ x^-alpha for x >= xmin. If xmin is None, scan integer xmin (log-spaced candidates up
    to xmin_max, default the 90th percentile) and keep the one minimising the KS distance (Clauset 2009).
    Returns dict alpha, xmin, n_tail, ks. The KS distance compares the empirical CCDF with the fitted continuous-
    approximation CCDF ((x - 0.5)/(xmin - 0.5))^(1 - alpha)."""
    x = np.asarray(x, float); x = x[x >= 1]
    if len(x) < n_min:
        return {"alpha": np.nan, "xmin": np.nan, "n_tail": int(len(x)), "ks": np.nan}
    cands = [xmin] if xmin is not None else np.unique(np.round(np.geomspace(1, xmin_max or np.percentile(x, 90), 30)))
    best = None
    for xm in cands:
        al, n = _pl_alpha(x, xm)
        if n < n_min or not np.isfinite(al):
            continue
        y = np.sort(x[x >= xm])
        emp = 1.0 - np.arange(len(y)) / len(y)
        fit = ((y - 0.5) / (xm - 0.5)) ** (1 - al)
        ks = float(np.abs(emp - fit).max())
        if best is None or ks < best["ks"]:
            best = {"alpha": float(al), "xmin": float(xm), "n_tail": int(n), "ks": ks}
    return best or {"alpha": np.nan, "xmin": np.nan, "n_tail": 0, "ks": np.nan}


def mean_size_given_T(S, T, tmin=2, tmax=None):
    """<S | T> for every duration and the least-squares slope gamma of ln<S> vs ln T over tmin <= T <= tmax
    (durations with >= 5 avalanches). Mean-field DP / critical branching: gamma = 2."""
    S, T = np.asarray(S, float), np.asarray(T, int)
    u = np.unique(T)
    ms = np.array([S[T == k].mean() for k in u]); n = np.array([(T == k).sum() for k in u])
    tmax = tmax or u.max()
    sel = (u >= tmin) & (u <= tmax) & (n >= 5)
    gam = float(np.polyfit(np.log(u[sel]), np.log(ms[sel]), 1)[0]) if sel.sum() >= 3 else np.nan
    return {"T": u, "meanS": ms, "n": n, "gamma": gam}


def mr_branching(a, kmax=20):
    """Multistep-regression estimate of the branching ratio m from activity a[S, B] (Wilting & Priesemann 2018,
    Nat Commun 9:2325): r_k = slope of a(t + k) on a(t) (pooled over replicas), fitted as r_k = b m^k for k = 1..kmax.
    Robust to subsampling and to a constant external drive. Returns dict m, r_k."""
    a = np.asarray(a, np.float64)
    if a.ndim == 1:
        a = a[:, None]
    ks = np.arange(1, kmax + 1)
    r = []
    for k in ks:
        x = a[:-k].ravel(); y = a[k:].ravel()
        vx = x.var()
        r.append(((x - x.mean()) * (y - y.mean())).mean() / vx if vx > 0 else np.nan)
    r = np.array(r)
    ok = r > 0
    if ok.sum() < 3:
        return {"m": np.nan, "r_k": r}
    slope = np.polyfit(ks[ok], np.log(r[ok]), 1)[0]
    return {"m": float(np.exp(slope)), "r_k": r}
