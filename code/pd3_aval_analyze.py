"""pd3 criticality C1 analysis (CPU): single-seed avalanches from pd3_avalanche.py vs the finite-K MF-DP chain.

Inputs: results/pd3/criticality/aval_<tag>.npz (production array pd3-criticality-aval); theory_bd.json / theory_bd2.json
(exact chain numbers quoted in the plan); results/pd/branch_real_dense_part*.json (R(g) of the real graph);
results/pd3/redraw_fss/fss_cnull_*.npz (R(x) of the nulls); results/pd/mode_real.json (net-input-closure mu(r)).

Conventions. Model time = steps x dt (dt = 0.2). The chain (logistic birth-death b(n) = R n (1 - n/K), d(n) = n, absorbing at
0) runs in units of 1/alpha; t_chain = alpha_eff t_model (alpha_eff = (1 - e^{-dt})/dt = 0.906). Only avalanches whose
observation window covers the cap are used for distributions (all of them when the run ended with every avalanche closed).

Outputs: results/pd3/criticality/aval_analysis.json; figures figs/pd3/aval_size_duration.png, aval_ST_slope.png,
aval_critical_point.png, aval_shapes.png, aval_examples.png (+ aval_recruitment.png).
"""
import glob, json, os, sys
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy import optimize, stats

ROOT = "/MALECNS_ROOT"
R_ = f"{ROOT}/results/pd3/criticality"; FIG = f"{ROOT}/figs/pd3"
DT = 0.2; A_EFF = (1 - np.exp(-DT)) / DT
G_LIN = 0.16315274988719286
AGES = np.array([5, 10, 20, 35, 50])
XS = [0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20, 1.35, 1.50]
rng = np.random.default_rng(12345)
SRC = os.environ.get("AVAL_DIR", R_)


# ------------------------------------------------------------------------------------------------------------ loading
def load(tag):
    p = f"{SRC}/aval_{tag}.npz"
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    m = json.loads(str(d["meta"]))
    cap = int(d["cap"]); gstep = int(d["gstep"])
    s0 = d["rec_s0"].astype(np.int64); se = d["rec_s_end"].astype(np.int64).copy(); st = d["rec_status"].astype(np.int64)
    fc = bool(d["final_complete"]) if "final_complete" in d.files else not np.any((st == 2) & (se < 0))
    se[se < 0] = (gstep + cap + 1) if fc else gstep
    win = se - s0
    hw = np.bincount(np.minimum(win, cap), minlength=cap + 1)
    den = hw[::-1].cumsum()[::-1].astype(np.float64)
    full = win >= cap
    o = {"tag": tag, "meta": m, "cap": cap, "x": float(d["x"]), "g": float(d["g"]), "g_lin": float(d["g_lin"]), "r": float(d["r"]),
         "T": d["rec_T"].astype(np.int64), "S": d["rec_S"].astype(np.int64), "status": st, "full": full, "win": win, "den": den,
         "sp4": d["rec_sp4"].astype(np.int64), "seed": d["rec_seed"], "skew_sp": d["rec_skew_sp"].astype(np.float64),
         "skew_act": d["rec_skew_act"].astype(np.float64), "maxact": d["rec_maxact"],
         "n_at": d["rec_n_at"].astype(np.float64) if "rec_n_at" in d.files else None,
         "agg_n": d["agg_n"], "agg_n2": d["agg_n2"], "agg_sp": d["agg_sp"], "agg_alive": d["agg_alive"],
         "prof_sum": d["prof_sum"], "prof_sq": d["prof_sq"], "prof_cnt": d["prof_cnt"], "prof_cnt_al": d["prof_cnt_al"],
         "prof_T": d["prof_T"], "prof_S": d["prof_S"], "bin_edges": d["bin_edges"], "ex_sum": d["ex_sum"], "ex_sq": d["ex_sq"],
         "ex_cnt": d["ex_cnt"], "tset": d["tset"], "final_complete": fc, "gstep": gstep,
         "samp_off": d["samp_off"], "samp_data": d["samp_data"], "samp_meta": json.loads(str(d["samp_meta"]))}
    return o


# ---------------------------------------------------------------------------------------------- exact chain numerics
def chain_rates(K, R):
    # state space truncated at 12 sqrt(K) + 60 (reflecting) for large K: at R ~ 1 the occupation beyond that is ~exp(-70);
    # exact (untruncated) for K <= 100
    n_top = min(int(np.floor(K)), int(12 * np.sqrt(K)) + 60); n = np.arange(1, n_top + 1, dtype=float)
    b = R * n * (1 - n / K); b[-1] = 0.0; d = n.copy()
    return n, b, d


def chain_eig(K, R):
    n, b, d = chain_rates(K, R)
    diag = -(b + d); off = np.sqrt(b[:-1] * d[1:])
    lsp = 0.5 * np.concatenate([[0.0], np.cumsum(np.log(np.maximum(b[:-1], 1e-300)) - np.log(d[1:]))])
    ev, Q = eigh_tridiagonal(diag, off)
    return n, ev, Q, lsp


def chain_surv_mean(K, R, t_model):
    """Survival P(T > t) and E[n(t)] (extinct = 0) from one seed; t in model time."""
    n, ev, Q, lsp = chain_eig(K, R)
    tc = A_EFF * np.asarray(t_model, float)
    left1 = np.exp(lsp - lsp[0]) @ Q; leftn = (n * np.exp(lsp - lsp[0])) @ Q; right = Q[0, :]
    E = np.exp(np.outer(tc, ev))
    return E @ (left1 * right), E @ (leftn * right)


def chain_size_dist(K, R, s_max):
    n, b, d = chain_rates(K, R); m = len(n)
    pb = b / (b + d); pd_ = 1 - pb
    Ld = np.concatenate([[0.0], np.cumsum(np.log(pd_))])
    T = np.zeros((m, m))
    for s in range(1, m + 1):
        for mm in range(1, s + 1):
            if pb[mm - 1] > 0 and mm + 1 <= m:
                T[mm, s - 1] = np.exp(Ld[s] - Ld[mm]) * pb[mm - 1]
    absorb = np.exp(Ld[1:]); q = np.zeros(m); q[0] = 1.0; P = []
    for _ in range(s_max):
        P.append(float(absorb @ q)); q = T @ q
        if q.sum() < 1e-14:
            break
    return np.array(P)


def chain_gamma(K, R):
    """Growth rate (per model time) of ln E[n] over the ages AGES (steps), same 5-point regression as the data."""
    _, En = chain_surv_mean(K, R, AGES * DT)
    return float(np.polyfit(AGES * DT, np.log(En), 1)[0])


def chain_Tc(K):
    """T_c in model time: (1 + alpha t) P(T > t) = 1/2 at R = 1 (theory_bd.json definition, chain time -> model time)."""
    t = np.geomspace(0.05, 2000, 2000); S, _ = chain_surv_mean(K, 1.0, t)
    k = np.argmax(S * (1 + A_EFF * t) < 0.5)
    return float(t[k])


# --------------------------------------------------------------------------------------------------- R(x) per graph
def R_real():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/results/pd/branch_real_dense_part*.json")):
        rows += json.load(open(f))["rows"]
    rows.sort(key=lambda r: r["g"])
    lg = np.log(np.r_[G_LIN, [r["g"] for r in rows]]); Rv = np.r_[1.0, [r["R"] for r in rows]]
    slope = (Rv[1] - Rv[0]) / (lg[1] - lg[0])

    def f(x):
        l = np.log(x * G_LIN)
        return float(np.interp(l, lg, Rv)) if l >= lg[0] else float(1.0 + slope * (l - lg[0]))
    return f


def R_from_npz(path):
    d = np.load(path, allow_pickle=True); xs = d["x"]; Rx = d["R_x"]
    lx = np.log(np.r_[1.0, xs]); Rv = np.r_[1.0, Rx]; o = np.argsort(lx); lx, Rv = lx[o], Rv[o]
    return lambda x: float(np.interp(np.log(x), lx, Rv))


def R_complete(K, g_lin):
    f = lambda y: (1 - np.exp(-np.tanh(y) * DT)) / DT
    return lambda x: float((K - 1) * f(x * g_lin / K) / A_EFF)


_VW = {}


def seed_weights(run):
    """Post-hoc mode weighting of the uniform seeds: w = v_seed, the right Perron vector of the AL module's branching operator
    B(g_lin,module) (an initial state ~ v is the pure leading mode, the chain's 'n = 1 in the mode'). Real and module graphs
    only (their seeds are the 366 excitatory module neurons); None otherwise."""
    if run["meta"]["graph"] not in ("real", "module"):
        return None
    if "v" not in _VW:
        sys.path.insert(0, f"{ROOT}/code")
        from pd3_common import exc_csr_torch, B_mats, perron_t
        mg = np.load(f"{ROOT}/results/module/module_graph.npz")
        pre = mg["pre"].astype(np.int64); post = mg["post"].astype(np.int64); w = mg["w"].astype(np.float64); s = mg["sign"].astype(np.int64)
        Ed = exc_csr_torch(pre, post, w, s[pre], float(mg["c"]), len(s), dev="cpu")
        _, v, _ = perron_t(B_mats(Ed, 0.16428095549634494)[0])
        lut = np.zeros(167216); lut[mg["idx"]] = v.abs().numpy().ravel()
        _VW["v"] = lut
    return _VW["v"][run["seed"].astype(np.int64)]


# --------------------------------------------------------------------------------------------------- per-run measures
def tbins_model(tmax, lo=0.2, ratio=1.25):
    e = [lo]
    while e[-1] < tmax:
        e.append(e[-1] * ratio)
    return np.array(e)


def survival_emp(run, t_model, w=None):
    f = run["full"]; T = run["T"][f]; st = run["status"][f]
    Tm = np.where(st == 1, np.inf, T * DT)
    if w is None:
        Ts = np.sort(Tm)
        return 1 - np.searchsorted(Ts, t_model, side="right") / len(Ts), len(Ts)
    ww = w[f]; o = np.argsort(Tm); cw = np.r_[0, np.cumsum(ww[o])] / ww.sum()
    k = np.searchsorted(Tm[o], t_model, side="right")
    return 1 - cw[k], float(ww.sum() ** 2 / (ww ** 2).sum())


def size_density(run):
    f = run["full"] & (run["status"] == 0); S = run["S"][f]; ntot = int(run["full"].sum())
    e = np.unique(np.round(np.geomspace(1, max(S.max(), 2) + 1, 60)).astype(int))
    c, _ = np.histogram(S, bins=e); w = np.diff(e)
    ctr = np.sqrt(e[:-1] * (e[1:] - 1).clip(min=e[:-1]))
    k = c > 0
    return ctr[k], c[k] / w[k] / ntot, c[k]


def dur_density(run, w=None):
    """Log-binned density of T in model time; bin edges on whole steps (bins narrower than one step would alias)."""
    f = run["full"] & (run["status"] == 0); Ts = run["T"][f]; ntot = float(run["full"].sum())
    ww = None if w is None else w[f] * (ntot / w[run["full"]].sum())
    es = np.unique(np.round(np.geomspace(1, Ts.max() + 2, 70)).astype(int)) - 0.5      # step edges
    c, _ = np.histogram(Ts, bins=es, weights=ww); e = es * DT; w_ = np.diff(e); ctr = np.sqrt(np.maximum(e[:-1], DT / 2) * e[1:])
    w = w_
    k = c > 0
    return ctr[k], c[k] / w[k] / ntot, c[k]


def fit_K_survival(run, t0=1.0, Kmin=4.0, Kmax=20000.0, w=None):
    """One-parameter chain fit of K: multinomial counts of T (model time) in log bins on [t0, t_hi] + tail, conditional on
    T > t0, against the chain at R(x) (here R = R_run). Returns K_fit, chi2/dof, and the compensated deviation curve."""
    f = run["full"]; T = run["T"][f]; st = run["status"][f]
    Tm = np.where(st == 1, np.inf, T * DT)
    ww = np.ones(len(Tm)) if w is None else w[f]
    keep = Tm > t0; Tm = Tm[keep]; ww = ww[keep]
    N = float(ww.sum() ** 2 / (ww ** 2).sum())                    # effective sample size (= count when unweighted)
    hi = np.quantile(Tm[np.isfinite(Tm)], 0.9995) if np.isfinite(Tm).any() else t0 * 10
    e = tbins_model(hi, lo=t0, ratio=1.25)
    c = np.r_[np.histogram(Tm, bins=e, weights=ww)[0], ww[Tm >= e[-1]].sum()] * (N / ww.sum())
    Rr = run.get("R_here", 1.0)

    def probs(K):
        S, _ = chain_surv_mean(K, Rr, e)
        S = np.maximum(S, 1e-300) / max(S[0], 1e-300)
        return np.r_[S[:-1] - S[1:], S[-1]]

    def nll(lK):
        p = np.maximum(probs(np.exp(lK)), 1e-300)
        return -float((c * np.log(p)).sum())
    grid = np.linspace(np.log(Kmin), np.log(Kmax), 60)
    vals = [nll(g) for g in grid]; k0 = int(np.argmin(vals))
    lo_, hi_ = grid[max(k0 - 1, 0)], grid[min(k0 + 1, len(grid) - 1)]
    res = optimize.minimize_scalar(nll, bounds=(lo_, hi_), method="bounded", options={"xatol": 1e-4})
    Kf = float(np.exp(res.x)); p = probs(Kf); Ep = N * p
    ok = Ep >= 5
    chi2 = float(((c[ok] - Ep[ok]) ** 2 / Ep[ok]).sum()); dof = int(ok.sum() - 2)
    # curvature SE of ln K
    h = 1e-3; d2 = (nll(res.x + h) - 2 * nll(res.x) + nll(res.x - h)) / h ** 2
    return {"K_fit": Kf, "lnK_se": float(1 / np.sqrt(d2)) if d2 > 0 else None, "chi2": chi2, "dof": dof,
            "chi2_dof": chi2 / max(dof, 1), "N_cond": N, "t0": t0, "bins": len(c)}


def compensated_dev(run, K, Rr=1.0, t_lo=1.0, t_hi=None, w=None):
    """max |P_emp(T > t) / P_chain(T > t) - 1| over t in [t_lo, t_hi] (model time; default t_hi = chain T_c(K))."""
    t_hi = chain_Tc(K) if t_hi is None else t_hi
    t = np.geomspace(t_lo, t_hi, 25)
    Pe, n = survival_emp(run, t, w); Pc, _ = chain_surv_mean(K, Rr, t)
    dev = Pe / Pc - 1
    return {"K": K, "t_hi": t_hi, "max_abs_dev": float(np.max(np.abs(dev))), "dev_at_t_hi": float(dev[-1]),
            "t": t.tolist(), "dev": dev.tolist()}


def mle_discrete_pl(x, a, b):
    """Discrete power-law MLE for x in [a, b] (truncated): P(x) ~ x^-alpha. Returns alpha, SE, n."""
    x = x[(x >= a) & (x <= b)]
    if len(x) < 50 or b <= a + 2:
        return None
    xs = np.arange(a, b + 1, dtype=float); slx = np.log(x).sum(); n = len(x)
    nll = lambda al: al * slx + n * np.log(np.sum(xs ** -al))
    res = optimize.minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
    h = 1e-4; d2 = (nll(res.x + h) - 2 * nll(res.x) + nll(res.x - h)) / h ** 2
    return {"alpha": float(res.x), "se": float(1 / np.sqrt(d2)) if d2 > 0 else None, "n": int(n), "window": [int(a), int(b)]}


def ST_relation(run, ratio=1.25, min_n=100, w=None):
    f = run["full"] & (run["status"] == 0); T = run["T"][f] * DT; S = run["S"][f].astype(float)
    ww = np.ones(len(T)) if w is None else w[f]
    e = tbins_model(T.max() + DT, lo=DT * 0.999, ratio=ratio)
    idx = np.digitize(T, e) - 1
    rows = []
    for k in range(len(e) - 1):
        m = idx == k
        if m.sum() >= min_n and ww[m].sum() > 0:
            wm = ww[m]; ne = wm.sum() ** 2 / (wm ** 2).sum(); mS = np.average(S[m], weights=wm)
            sd = np.sqrt(np.average((S[m] - mS) ** 2, weights=wm))
            rows.append((np.average(T[m], weights=wm), mS, sd / np.sqrt(ne), int(m.sum())))
    rows = np.array(rows)
    if len(rows) < 4:
        return {"rows": rows.tolist(), "slope": [], "peak": None}
    lT, lS = np.log(rows[:, 0]), np.log(rows[:, 1])
    sl = []
    for k in range(1, len(rows) - 1):
        sl.append((float(np.exp(lT[k])), float(np.polyfit(lT[k - 1:k + 2], lS[k - 1:k + 2], 1)[0])))
    sl = np.array(sl)
    # bootstrap the peak: resample bin means from normal(mean, se)
    pk = []
    for _ in range(300):
        lSb = np.log(np.maximum(rows[:, 1] + rng.normal(0, 1, len(rows)) * rows[:, 2], 1e-9))
        s_b = [np.polyfit(lT[k - 1:k + 2], lSb[k - 1:k + 2], 1)[0] for k in range(1, len(rows) - 1)]
        pk.append(max(s_b))
    kpk = int(np.argmax(sl[:, 1]))
    return {"rows": rows.tolist(), "slope": sl.tolist(), "peak": float(sl[kpk, 1]), "peak_T": float(sl[kpk, 0]),
            "peak_ci": [float(np.percentile(pk, 2.5)), float(np.percentile(pk, 97.5))],
            "global_fit_T1_10": (float(np.polyfit(lT[(rows[:, 0] >= 1) & (rows[:, 0] <= 10)], lS[(rows[:, 0] >= 1) & (rows[:, 0] <= 10)], 1)[0])
                                 if ((rows[:, 0] >= 1) & (rows[:, 0] <= 10)).sum() >= 3 else None)}


def gamma_run(run, nboot=2000, w=None):
    """gamma = slope of ln E[n(age)] vs model time over ages 5..50 steps. Point estimate and CLT bootstrap from the
    per-avalanche n_at (avalanches observed through age 50); plus the all-ages aggregate estimate."""
    a = np.arange(5, 51)
    En = run["agg_n"][a] / run["den"][a]
    g_agg = float(np.polyfit(a * DT, np.log(En), 1)[0])
    out = {"gamma_agg_all_ages": g_agg, "En_ages": dict(zip([int(x) for x in AGES], (run["agg_n"][AGES] / run["den"][AGES]).tolist()))}
    if run["n_at"] is not None:
        sel = run["win"] >= 50
        X = run["n_at"][sel]
        if w is None:
            N = len(X); mu = X.mean(0); C = np.cov(X.T) / N
        else:
            ww = w[sel]; ww = ww / ww.sum(); N = int(round(1 / (ww ** 2).sum()))
            mu = ww @ X; D = X - mu; C = (D * (ww ** 2)[:, None]).T @ D
        g5 = float(np.polyfit(AGES * DT, np.log(mu), 1)[0])
        draws = rng.multivariate_normal(mu, C, size=nboot)
        draws = np.maximum(draws, 1e-12)
        gb = np.polyfit(AGES * DT, np.log(draws.T), 1)[0]
        out.update({"gamma": g5, "gamma_se": float(gb.std()), "gamma_ci": [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))],
                    "n": int(N), "boot": gb})
    else:
        out.update({"gamma": g_agg, "gamma_se": None, "boot": None})
    return out


def skew_bins(run, edges_model, w=None):
    f = run["full"] & (run["status"] == 0); T = run["T"][f] * DT
    W = np.ones(len(T)) if w is None else w[f]
    rows = []
    for k in range(len(edges_model) - 1):
        m = (T >= edges_model[k]) & (T < edges_model[k + 1])
        out_ = {}
        for key in ("skew_sp", "skew_act"):
            sk = run[key][f][m]; wk = W[m]; ok = np.isfinite(sk) & (wk > 0); sk = sk[ok]; wk = wk[ok]
            if len(sk) < 30:
                break
            mu = np.average(sk, weights=wk); ne = wk.sum() ** 2 / (wk ** 2).sum()
            out_[key] = float(mu); out_[key + "_se"] = float(np.sqrt(np.average((sk - mu) ** 2, weights=wk) / ne))
        if len(out_) == 4:
            rows.append({"T_lo": float(edges_model[k]), "T_hi": float(edges_model[k + 1]), "T_mean": float(T[m].mean()),
                         "n": int(m.sum()), **out_})
    return rows


def recruitment(run, Tmin_steps=50, Tmax_steps=None):
    """Mean shares vs t/T over the log-duration bins with T in [Tmin, Tmax) steps (from the online profile sums)."""
    e = run["bin_edges"]; nb = len(e) - 1
    sel = [k for k in range(nb) if e[k] >= Tmin_steps and (Tmax_steps is None or e[k + 1] <= Tmax_steps * 1.0001)]
    if not sel:
        return None
    P = run["prof_sum"][sel].sum(0)                         # [40, 8]: total, L, R, other, seedlobe, otherlobe, major, minor
    cnt = float(run["prof_cnt"][sel].sum()); cal = float(run["prof_cnt_al"][sel].sum())
    tot = P[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        res = {"n": cnt, "n_al": cal, "t_over_T": ((np.arange(40) + 0.5) / 40).tolist(),
               "share_other": (P[:, 3] / tot).tolist(), "share_seedlobe": (P[:, 4] / (P[:, 4] + P[:, 5])).tolist(),
               "share_minority": (P[:, 7] / (P[:, 6] + P[:, 7])).tolist(), "mean_profile": (tot / max(cnt, 1)).tolist()}
    return res


def samples_minority(run, Tmin=50, nboot=1000):
    """Per sampled avalanche (reservoir, T >= 50 steps, extinct): minority-lobe share in the middle (t/T 0.4-0.6) and late
    (0.8-1.0) fifths of its duration, and the seed-lobe share; bootstrap over avalanches of the mean late - mid change."""
    off = run["samp_off"]; dat = run["samp_data"]; meta = run["samp_meta"]
    mids, lates, smid, slate = [], [], [], []
    for i, m in enumerate(meta):
        if m["status"] != 0 or m["T"] < Tmin or m["seed_chan"] > 1:
            continue
        x = dat[off[i]:off[i + 1]].astype(float); T = len(x)
        L, Rr = x[:, 0].sum(), x[:, 1].sum()
        mn = 1 if L >= Rr else 0                                 # minority lobe column
        sl = m["seed_chan"]
        a, b, c_, d_ = int(0.4 * T), int(0.6 * T), int(0.8 * T), T
        lr_mid = x[a:b, 0].sum() + x[a:b, 1].sum(); lr_late = x[c_:d_, 0].sum() + x[c_:d_, 1].sum()
        if lr_mid > 0 and lr_late > 0:
            mids.append(x[a:b, mn].sum() / lr_mid); lates.append(x[c_:d_, mn].sum() / lr_late)
            smid.append(x[a:b, sl].sum() / lr_mid); slate.append(x[c_:d_, sl].sum() / lr_late)
    if len(mids) < 30:
        return None
    mids, lates, smid, slate = map(np.array, (mids, lates, smid, slate))
    dlt = lates - mids; ds = slate - smid
    bi = rng.integers(0, len(dlt), (nboot, len(dlt)))
    return {"n": int(len(dlt)), "minority_mid": float(mids.mean()), "minority_late": float(lates.mean()),
            "delta_minority_late_minus_mid": float(dlt.mean()),
            "delta_ci": [float(np.percentile(dlt[bi].mean(1), 2.5)), float(np.percentile(dlt[bi].mean(1), 97.5))],
            "seedlobe_mid": float(smid.mean()), "seedlobe_late": float(slate.mean()),
            "delta_seedlobe": float(ds.mean()),
            "delta_seedlobe_ci": [float(np.percentile(ds[bi].mean(1), 2.5)), float(np.percentile(ds[bi].mean(1), 97.5))]}


# ================================================================================================================ main
if __name__ == "__main__":
    runs = {}
    for f in sorted(glob.glob(f"{SRC}/aval_*.npz")):
        if os.path.basename(f).startswith("aval_analysis"):
            continue
        tag = os.path.basename(f)[5:-4]
        runs[tag] = load(tag)
    print("loaded", len(runs), "runs:", " ".join(sorted(runs)), flush=True)
    Rreal = R_real()
    Rfun = {}
    for tag, run in runs.items():
        gph = run["meta"]["graph"]
        if gph == "real":
            Rfun[tag] = Rreal
        elif gph == "cnull":
            N = run["meta"]["args"]["N"]; s = run["meta"]["args"]["gseed"]
            p = f"{ROOT}/results/pd3/redraw_fss/fss_cnull_N{N}_s{s}_r0.npz"
            Rfun[tag] = R_from_npz(p) if os.path.exists(p) else (lambda x: x)
        elif gph == "complete":
            Rfun[tag] = R_complete(run["meta"]["args"]["K"], run["g_lin"])
        else:
            Rfun[tag] = lambda x: x                               # module: R(x) ~ x (checked in the QS analysis)
        run["R_here"] = Rfun[tag](run["x"])
    # reference K per graph (for chain overlays / exponent windows): K_needed of phase_complete.json (real), K_fit of
    # redraw_fss (nulls), realsub K_fit (module), unit count (complete)
    KN = {0: 75.8, 0.5: 48.6, 1: 36.6, 2: 26.7, 4: 19.5, 6: 16.2}
    fss = json.load(open(f"{ROOT}/results/pd3/redraw_fss/fss_analysis.json"))

    def Kref(run):
        gph = run["meta"]["graph"]; r = run["r"]
        if gph == "real":
            return KN.get(r, None)
        if gph == "module":
            return {0.0: 70.0, 1.0: 36.5}.get(r)
        if gph == "complete":
            return float(run["meta"]["args"]["K"])
        return run["meta"].get("K_nc_glin")
    out = {"runs": {}, "notes": []}
    for tag in sorted(runs):
        run = runs[tag]; m = run["meta"]
        nfull = int(run["full"].sum())
        row = {"graph": m["graph"], "r": run["r"], "x": run["x"], "g": run["g"], "g_lin": run["g_lin"], "R_at_x": run["R_here"],
               "n_records": int(len(run["T"])), "n_full_window": nfull, "final_complete": run["final_complete"],
               "status_counts": np.bincount(run["status"], minlength=3).tolist(), "gpu": m.get("gpu"), "wall_s": m.get("wall_s"),
               "ms_per_step": m.get("ms_per_step_mean"), "steps": m.get("steps"), "K_ref": Kref(run),
               "pool_size": m.get("pool_size"), "K_nc_glin": m.get("K_nc_glin"), "inv_ipr_glin": m.get("inv_ipr_glin")}
        f = run["full"]
        sp = run["sp4"][f].sum(0); row["spike_share_L_R_mid_rest"] = (sp / max(sp.sum(), 1)).tolist()
        row["mean_S"] = float(run["S"][f & (run["status"] == 0)].mean()); row["mean_T_model"] = float((run["T"][f & (run["status"] == 0)] * DT).mean())
        tt = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
        Pe, _ = survival_emp(run, tt); row["P_T_gt"] = dict(zip([str(int(t)) for t in tt], Pe.tolist()))
        row["gamma"] = {k: v for k, v in gamma_run(run).items() if k != "boot"}
        wv = seed_weights(run)
        if wv is not None:
            row["vw"] = {"gamma": {k: v for k, v in gamma_run(run, w=wv).items() if k != "boot"}, "ST": ST_relation(run, w=wv),
                         "P_T_gt": dict(zip(["1", "2", "5", "10", "20", "50"], survival_emp(run, np.array([1, 2, 5, 10, 20, 50.0]), wv)[0].tolist())),
                         "note": "post hoc: avalanches weighted by v_seed (module right Perron vector at g_lin)"}
        row["ST"] = ST_relation(run)
        row["skew"] = skew_bins(run, np.array([1, 2, 4, 8, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 500, 640, 800, 1000]))
        if abs(run["x"] - 1) < 1e-6 or m["graph"] in ("complete",):
            try:
                row["Kfit_T"] = fit_K_survival(run, t0=1.0)
                row["Kfit_T_t0_2"] = fit_K_survival(run, t0=2.0)
            except Exception as ex:                                # noqa
                row["Kfit_T"] = {"error": str(ex)}
            Kr = Kref(run)
            if Kr:
                row["chain_comp_Kref"] = compensated_dev(run, Kr, run["R_here"])
            if isinstance(row.get("Kfit_T"), dict) and "K_fit" in row["Kfit_T"]:
                row["chain_comp_Kfit"] = compensated_dev(run, row["Kfit_T"]["K_fit"], run["R_here"])
            for Kc in (37.0, 54.7, 75.8):
                row[f"chain_comp_K{Kc:g}"] = compensated_dev(run, Kc, 1.0)
            if wv is not None:
                row["vw"]["Kfit_T"] = fit_K_survival(run, t0=1.0, w=wv)
                row["vw"]["chain_comp_Kfit"] = compensated_dev(run, row["vw"]["Kfit_T"]["K_fit"], run["R_here"], w=wv)
                if Kref(run):
                    row["vw"]["chain_comp_Kref"] = compensated_dev(run, Kref(run), run["R_here"], w=wv)
                row["vw"]["skew"] = skew_bins(run, np.array([1, 2, 4, 8, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 500, 640, 800, 1000]), w=wv)
            # exponent MLEs (secondary): windows [2, S_c/3], [5 steps, T_c/3] with chain cutoffs at K_ref
            if Kr:
                Tc = chain_Tc(Kr)
                Sc = (1.94 if m["graph"] in ("real", "module") else 1.0) * 3.4 * Kr
                fe = f & (run["status"] == 0)
                row["mle_S"] = mle_discrete_pl(run["S"][fe].astype(float), 2, int(Sc / 3))
                row["mle_T_steps"] = mle_discrete_pl(run["T"][fe].astype(float), 5, int(Tc / 3 / DT))
                row["cutoffs_used"] = {"S_c": Sc, "T_c_model": Tc}
            row["recruit_50_250"] = recruitment(run, 50, 250)
            row["recruit_250up"] = recruitment(run, 250, None)
            row["samples_minority"] = samples_minority(run)
        out["runs"][tag] = row
        g_ = row["gamma"]
        print(f"{tag}: n {row['n_records']} full {nfull} capped {row['status_counts'][1]} | <S> {row['mean_S']:.2f} <T> {row['mean_T_model']:.2f} | "
              f"gamma {g_['gamma']:.4f} +- {g_.get('gamma_se') or float('nan'):.4f} | ST peak {row['ST'].get('peak')} "
              f"| Kfit {row.get('Kfit_T', {}).get('K_fit') if isinstance(row.get('Kfit_T'), dict) else None}", flush=True)
    # -------------------------------------------------------------------------------------- critical point x* (gamma = 0)
    gam = {}
    for tag, run in runs.items():
        gam[tag] = gamma_run(run)
        wv = seed_weights(run)
        if wv is not None and run["meta"]["graph"] == "real":
            gam[tag + "__vw"] = gamma_run(run, w=wv)
    crit = {}
    for fam, pref in [("real_r0", "real_r0_x"), ("real_r1", "real_r1_x"), ("real_r6", "real_r6_x"), ("real_r0_vw", "real_r0_x"),
                      ("real_r1_vw", "real_r1_x"), ("real_r6_vw", "real_r6_x"), ("complete37", "complete37_x"),
                      ("complete76", "complete76_x"), ("cnull_N167216_s0_r0", "cnull_N167216_s0_r0_x")]:
        tags = sorted([t for t in runs if t.startswith(pref)], key=lambda t: runs[t]["x"])
        if fam.endswith("_vw"):
            tags = [t + "__vw" for t in tags]
        if len(tags) < 2:
            continue
        xs = np.array([runs[t.replace("__vw", "")]["x"] for t in tags]); gs = np.array([gam[t]["gamma"] for t in tags])
        boots = [gam[t]["boot"] for t in tags]
        def zero(xs, gs):
            sgn = np.sign(gs); k = np.where((sgn[:-1] < 0) & (sgn[1:] >= 0))[0]
            if len(k) == 0:
                return None
            k = k[0]; return float(xs[k] + (0 - gs[k]) * (xs[k + 1] - xs[k]) / (gs[k + 1] - gs[k]))
        xstar = zero(xs, gs)
        xb = []
        if all(b is not None for b in boots):
            nb = min(len(b) for b in boots)
            for i in range(nb):
                z = zero(xs, np.array([b[i] for b in boots]))
                if z is not None:
                    xb.append(z)
        crit[fam] = {"x": xs.tolist(), "gamma": gs.tolist(), "gamma_se": [gam[t].get("gamma_se") for t in tags], "x_star": xstar,
                     "x_star_ci": [float(np.percentile(xb, 2.5)), float(np.percentile(xb, 97.5))] if len(xb) > 20 else None,
                     "n_boot_with_zero": len(xb),
                     "status": "ok" if xstar is not None else ("gamma>=0 at all x" if (gs >= 0).all() else ("gamma<0 at all x" if (gs < 0).all() else "no up-crossing"))}
        print("x*", fam, crit[fam]["x_star"], crit[fam]["x_star_ci"], crit[fam]["status"], flush=True)
    # chain predictions of gamma(x) and x* (continuous chain, same 5-age regression), with R(x) of the real graph
    chain_pred = {}
    mu_ = {row["r"]: row["mu"] for row in json.load(open(f"{ROOT}/results/pd/mode_real.json"))["rows"]}
    clos = lambda rr: (lambda x: Rreal(x) * mu_[rr] / mu_[0.0])
    for lab, K, Rf in [("K75.8_real", 75.8, Rreal), ("K36.6_real", 36.6, Rreal), ("K16.2_real", 16.2, Rreal),
                       ("K36.6_real_netinput_closure_r1", 36.6, clos(1.0)), ("K16.2_real_netinput_closure_r6", 16.2, clos(6.0)),
                       ("K37_complete", 37.0, None), ("K76_complete", 76.0, None), ("K300", 300.0, None), ("K2000", 2000.0, None)]:
        xg = np.linspace(0.85, 2.5, 331)
        Rg = np.array([Rf(x) for x in xg]) if Rf else xg
        gg = np.array([chain_gamma(K, R) for R in Rg])
        k = np.where((gg[:-1] < 0) & (gg[1:] >= 0))[0]
        xs_ = float(xg[k[0]] + (0 - gg[k[0]]) * (xg[k[0] + 1] - xg[k[0]]) / (gg[k[0] + 1] - gg[k[0]])) if len(k) else None
        chain_pred[lab] = {"x": xg.tolist(), "gamma": gg.tolist(), "x_star": xs_, "gamma_at_x1": float(chain_gamma(K, Rg[np.argmin(np.abs(xg - 1))]))}
        print("chain", lab, "x*", xs_, "gamma(1)", chain_pred[lab]["gamma_at_x1"], flush=True)
    # r-dependence at x = 1: data vs exact-linearisation chain (K_needed(r)) vs net-input closure (R_closure = mu(r)/mu(0))
    mu = {row["r"]: row["mu"] for row in json.load(open(f"{ROOT}/results/pd/mode_real.json"))["rows"]}
    rdep = []
    for r in [0, 0.5, 1, 2, 4, 6]:
        tag = f"real_r{r:g}_x1.00"
        if tag not in runs:
            continue
        K = KN[r]
        mu_r = mu.get(float(r)); Rcl = (mu_r / mu[0.0]) if mu_r else None
        rdep.append({"r": r, "gamma": gam[tag]["gamma"], "gamma_se": gam[tag].get("gamma_se"), "K_needed": K,
                     "gamma_vw": gam[tag + "__vw"]["gamma"], "gamma_vw_se": gam[tag + "__vw"].get("gamma_se"),
                     "chain_gamma_exact_linearisation": chain_gamma(K, 1.0),
                     "closure_R": Rcl, "chain_gamma_closure": chain_gamma(K, Rcl) if Rcl else None,
                     "En_ages": gam[tag]["En_ages"]})
    out["critical_point"] = {"families": crit, "chain_pred": {k: {kk: vv for kk, vv in v.items() if kk in ("x_star", "gamma_at_x1")} for k, v in chain_pred.items()},
                             "chain_curves": chain_pred, "r_dependence_x1": rdep,
                             "definition": "gamma = slope of ln E[n(age)] vs model time over ages 5, 10, 20, 35, 50 steps (n_at records, CLT bootstrap); x* = first up-crossing of 0, linear interpolation"}
    # chain reference numbers for the S(T) slope (theory_bd / bd2 MC rows re-analysed with the same 3-bin local slope)
    tb = json.load(open(f"{ROOT}/results/pd3/theory_bd.json")); tb2 = json.load(open(f"{ROOT}/results/pd3/theory_bd2.json"))
    chainST = {}
    for K, rows in [("37", tb["mc_R1"]["37"]["rows"]), ("75.8", tb["mc_R1"]["75.8"]["rows"]), ("300", tb2["mc"]["300"]["rows"]), ("2000", tb2["mc"]["2000"]["rows"])]:
        Tm = np.array([r_["mean_T"] for r_ in rows]); Sm = np.array([r_["mean_S"] for r_ in rows]); nn = np.array([r_["n"] for r_ in rows])
        k = nn >= 100; lT, lS = np.log(Tm[k]), np.log(Sm[k])
        sl = [np.polyfit(lT[i - 1:i + 2], lS[i - 1:i + 2], 1)[0] for i in range(1, len(lT) - 1)]
        chainST[K] = {"peak_local_slope_3bin": float(max(sl)), "note": "chain MC rows (bins of ratio ~1.2 in chain time) re-analysed"}
    out["chain_ST_peaks"] = chainST
    out["chain_ST_peaks_quoted"] = {"37": 1.55, "75.8": 1.66, "300": 1.79, "2000": 1.91, "source": "plan.md 0.2 / theory_notes.md 2.4(iii)"}
    out["chain_Tc_model"] = {str(K): chain_Tc(K) for K in (37.0, 54.7, 75.8)}
    json.dump(out, open(f"{R_}/aval_analysis.json", "w"), indent=1, default=lambda o: None if isinstance(o, np.ndarray) else str(o))
    np.savez(f"{R_}/aval_analysis_boot.npz", **{f"gamma_boot_{t}": gam[t]["boot"] for t in gam if gam[t]["boot"] is not None})
    print("saved", f"{R_}/aval_analysis.json", flush=True)
