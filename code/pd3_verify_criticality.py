"""Adversarial verification of program `criticality` (results/pd3/criticality.md). Independent re-implementation: no code is
imported from pd3_*.py. Reads the RAW outputs (results/pd3/criticality/aval_*.npz, qs_*.npz, field_*.npz, submodule_*.npz,
v0/), the connectome cache (read only), results/pd/branch_real_dense_part*.json (R(g) rows), results/pd/mode_real.json (mu(r)),
results/pd3/redraw_fss/fss_analysis.json (null tau_x1 / K, a verified upstream output).

Chain numerics are done differently from the program: dense generator of the logistic birth-death chain on the transient
states 1..floor(K) (b(n) = R n (1 - n/K), b(floor K) = 0, d(n) = n), scipy.linalg.expm propagators (no symmetrisation, no
tridiagonal eigen-solver); QSD by power iteration on the propagator; mean extinction times by a banded linear solve.
Bootstraps: i.i.d. bootstrap via random-permutation groups (not CLT draws).

Usage: pd3_verify_criticality.py --part {aval,chain,x1,minority,qs,field,c4,v0,all}
Output: results/pd3/verify_criticality.json (merged per part).
"""
import argparse, glob, json, os, time
import numpy as np
from scipy.linalg import expm, solve_banded
from scipy import optimize
from scipy.ndimage import gaussian_filter1d

ROOT = "/MALECNS_ROOT"
CR = f"{ROOT}/results/pd3/criticality"
OUT = f"{ROOT}/results/pd3/verify_criticality.json"
DT = 0.2
AE = (1.0 - np.exp(-DT)) / DT                       # alpha_eff: per-step death prob / dt
GL = 0.16315274988719286                            # g_lin real (phase_complete.json)
AGES = np.array([5, 10, 20, 35, 50])
KNEED = {0: 75.8, 0.5: 48.6, 1: 36.6, 2: 26.7, 4: 19.5, 6: 16.2}   # phase_complete.json (as quoted in criticality.md)
rng = np.random.default_rng(20261109)


def save(part, res):
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d[part] = res
    d["_meta"] = {"script": "code/pd3_verify_criticality.py", "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(d, open(OUT, "w"), indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))


# ---------------------------------------------------------------------------------------------------------- chain
def gen(K, R):
    m = int(np.floor(K)); n = np.arange(1, m + 1, dtype=float)
    b = R * n * (1 - n / K); b[-1] = 0.0; d = n.copy()
    A = np.diag(-(b + d)) + np.diag(b[:-1], 1) + np.diag(d[1:], -1)       # transient block, rows = from-state
    return n, A


def chain_surv_En(K, R, t_model):
    """P(T > t), E[n(t)] (extinct = 0) from n = 1, t in model time (t_chain = alpha_eff t_model)."""
    n, A = gen(K, R)
    out_s, out_n = [], []
    for t in np.atleast_1d(t_model):
        P = expm(A * AE * t)[0]
        out_s.append(P.sum()); out_n.append(P @ n)
    return np.array(out_s), np.array(out_n)


def chain_surv_grid(K, R, tmax_model, h_model=0.05):
    """Survival from n = 1 on a uniform model-time grid (repeated propagator)."""
    n, A = gen(K, R)
    M = expm(A * AE * h_model)
    nst = int(np.ceil(tmax_model / h_model)) + 1
    p = np.zeros(len(n)); p[0] = 1.0; S = np.empty(nst)
    for k in range(nst):
        S[k] = p.sum(); p = p @ M
    return np.arange(nst) * h_model, S


def chain_gamma(K, R):
    """Same 5-age regression as the data (ages 5, 10, 20, 35, 50 steps): one 5-step propagator, applied 10 times."""
    n, A = gen(K, R)
    M = expm(A * AE * 5 * DT)
    p = np.zeros(len(n)); p[0] = 1.0; En = {}
    for k in range(1, 11):
        p = p @ M
        En[5 * k] = p @ n
    return np.polyfit(AGES * DT, np.log([En[a] for a in AGES]), 1)[0]


def chain_qsd(K, R):
    n, A = gen(K, R)
    M = expm(A * 2.0)
    p = np.ones(len(n)) / len(n)
    for _ in range(4000):
        q = p @ M; s = q.sum(); q /= s
        if np.abs(q - p).max() < 1e-13:
            p = q; break
        p = q
    lam = -np.log(s) / 2.0                                                  # decay rate (chain time)
    return n, p, lam


def chain_tau_top(K, R=1.0):
    """Mean extinction time (model time) from n = floor(K): banded solve of A T = -1."""
    m = int(np.floor(K)); n = np.arange(1, m + 1, dtype=float)
    b = R * n * (1 - n / K); b[-1] = 0.0; d = n.copy()
    ab = np.zeros((3, m)); ab[0, 1:] = b[:-1]; ab[1] = -(b + d); ab[2, :-1] = d[1:]
    T = solve_banded((1, 1), ab, -np.ones(m))
    return T[-1] / AE


# ---------------------------------------------------------------------------------------------------------- R(x)
def R_real_fn():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/results/pd/branch_real_dense_part*.json")):
        rows += json.load(open(f))["rows"]
    g = np.array([r["g"] for r in rows]); R = np.array([r["R"] for r in rows]); o = np.argsort(g); g, R = g[o], R[o]
    lx = np.r_[0.0, np.log(g / GL)]; Rv = np.r_[1.0, R]
    s0 = (Rv[1] - Rv[0]) / (lx[1] - lx[0])
    return lambda x: np.where(np.log(x) < lx[1], 1.0 + s0 * np.log(x), np.interp(np.log(x), lx, Rv))


# ---------------------------------------------------------------------------------------------------------- aval loading
def load_run(tag, keys):
    d = np.load(f"{CR}/aval_{tag}.npz", allow_pickle=True)
    o = {k: d[k] for k in keys}
    o["x"] = float(d["x"]); o["g"] = float(d["g"]); o["g_lin"] = float(d["g_lin"]); o["final_complete"] = bool(d["final_complete"])
    o["meta"] = json.loads(str(d["meta"]))
    return o


def gamma_block(n_at, nblock=400, nboot=1000):
    """gamma (5-age regression of ln mean n_at) + bootstrap. Records are stored in closure order and, within a seeding
    wave, sorted by duration, so contiguous blocks are NOT exchangeable (a block bootstrap in record or s0 order inflates the
    SE ~10x: checked). Avalanches are i.i.d. (independent slots), so the records are randomly permuted before blocking
    (= an i.i.d. bootstrap with 400 groups)."""
    X = n_at.astype(np.float64)[rng.permutation(len(n_at))]; N = len(X)
    mu = X.mean(0); g0 = np.polyfit(AGES * DT, np.log(mu), 1)[0]
    edges = np.linspace(0, N, nblock + 1).astype(int)
    bs = np.add.reduceat(X, edges[:-1], axis=0); bc = np.diff(edges)
    idx = rng.integers(0, nblock, (nboot, nblock))
    mb = bs[idx].sum(1) / bc[idx].sum(1)[:, None]
    gb = np.polyfit(AGES * DT, np.log(np.maximum(mb, 1e-12)).T, 1)[0]
    # all ages 5..50 version is not available from n_at; done from agg_n separately
    return float(g0), gb


def upcross(xs, gs):
    k = np.where((gs[:-1] < 0) & (gs[1:] >= 0))[0]
    if len(k) == 0:
        return None
    k = k[0]
    return float(xs[k] + (0 - gs[k]) * (xs[k + 1] - xs[k]) / (gs[k + 1] - gs[k]))


def part_aval():
    fams = {"real_r0": "real_r0_x", "real_r1": "real_r1_x", "real_r6": "real_r6_x", "complete37": "complete37_x",
            "complete76": "complete76_x", "cnull_full_r0": "cnull_N167216_s0_r0_x"}
    res = {}
    allg = {}
    for fam, pref in fams.items():
        tags = sorted([os.path.basename(f)[5:-4] for f in glob.glob(f"{CR}/aval_{pref}*.npz")])
        rows = []
        for t in tags:
            r = load_run(t, ["rec_n_at", "agg_n", "rec_s0", "rec_s_end"])
            assert r["final_complete"]
            g0, gb = gamma_block(r["rec_n_at"])
            a = np.arange(5, 51)
            # all-age variant: agg_n / number of avalanches (every record observed to the cap when final_complete)
            Nrec = len(r["rec_n_at"]); En_all = r["agg_n"][a] / Nrec
            g_all = np.polyfit(a * DT, np.log(En_all), 1)[0]
            rows.append({"tag": t, "x": r["x"], "gamma": g0, "gamma_se_block": float(gb.std()), "gamma_all_ages": float(g_all), "boot": gb})
            allg[t] = g0
            print(f"  {t}: gamma {g0:.4f} +- {gb.std():.4f} (all ages {g_all:.4f})", flush=True)
        rows.sort(key=lambda q: q["x"])
        xs = np.array([q["x"] for q in rows]); gs = np.array([q["gamma"] for q in rows])
        xst = upcross(xs, gs)
        nb = min(len(q["boot"]) for q in rows)
        xb = [upcross(xs, np.array([q["boot"][i] for q in rows])) for i in range(nb)]
        xb = np.array([v for v in xb if v is not None])
        ga = np.array([q["gamma_all_ages"] for q in rows])
        res[fam] = {"x": xs.tolist(), "gamma": gs.tolist(), "gamma_se_block": [q["gamma_se_block"] for q in rows],
                    "gamma_all_ages": ga.tolist(), "x_star": xst,
                    "x_star_ci_block": [float(np.percentile(xb, 2.5)), float(np.percentile(xb, 97.5))] if len(xb) > 50 else None,
                    "x_star_all_ages": upcross(xs, ga)}
        print(fam, "x*", xst, res[fam]["x_star_ci_block"], "all-ages", res[fam]["x_star_all_ages"], flush=True)
    save("aval_xstar", res)
    return res


def part_chain():
    """Chain predictions of x* (exact linearisation vs closure, K_needed) + a 2-parameter (K, c) fit of the data gamma(x)
    with R = c R_exact(x): discriminates exact linearisation (c_r / c_0 = 1) from the net-input closure (c_r / c_0 = mu_r/mu_0)."""
    Rr = R_real_fn()
    mu = {row["r"]: row["mu"] for row in json.load(open(f"{ROOT}/results/pd/mode_real.json"))["rows"]}
    res = {"R_at": {str(x): float(Rr(np.array(x))) for x in [0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.3]}}
    xg = np.linspace(0.9, 2.4, 151)
    preds = {}
    for lab, K, c in [("r0_exact_K75.8", 75.8, 1.0), ("r1_exact_K36.6", 36.6, 1.0), ("r6_exact_K16.2", 16.2, 1.0),
                      ("r1_closure_K36.6", 36.6, mu[1.0] / mu[0.0]), ("r6_closure_K16.2", 16.2, mu[6.0] / mu[0.0])]:
        gg = np.array([chain_gamma(K, c * float(Rr(np.array(x)))) for x in xg])
        preds[lab] = {"x_star": upcross(xg, gg), "gamma_x1": float(chain_gamma(K, c * 1.0)), "c": c}
        print("chain", lab, preds[lab], flush=True)
    # complete-graph chain predictions with its own R(x) (synchronous: R = (K-1) f(x g_lin / K) / alpha_eff)
    res["chain_pred"] = preds
    # complete graphs: R(x) exact for the complete graph
    f = lambda y: (1 - np.exp(-np.tanh(y) * DT)) / DT
    av = json.load(open(OUT)).get("aval_xstar", {})
    for K in (37, 76):
        d = np.load(f"{CR}/aval_complete{K}_x1.00.npz", allow_pickle=True); gl = float(d["g_lin"])
        Rc = lambda x: (K - 1) * f(x * gl / K) / AE
        gg = np.array([chain_gamma(float(K), Rc(x)) for x in xg])
        preds[f"complete{K}_ownR"] = {"x_star": upcross(xg, gg), "R_at_x1": Rc(1.0)}
        print("chain complete", K, preds[f"complete{K}_ownR"], flush=True)
    # 2-parameter fit per r (uniform seeds; data from part_aval)
    fits = {}
    for fam, r in [("real_r0", 0.0), ("real_r1", 1.0), ("real_r6", 6.0)]:
        if fam not in av:
            continue
        xs = np.array(av[fam]["x"]); gs = np.array(av[fam]["gamma"]); se = np.maximum(np.array(av[fam]["gamma_se_block"]), 1e-3)
        Rx = np.array([float(Rr(np.array(x))) for x in xs])

        def loss_c(K, c):
            gm = np.array([chain_gamma(K, c * R) for R in Rx])
            return float((((gm - gs) / se) ** 2).sum())
        grid = []
        for K in np.exp(np.linspace(np.log(6), np.log(200), 16)):
            o = optimize.minimize_scalar(lambda c: loss_c(K, c), bounds=(0.6, 1.3), method="bounded", options={"xatol": 1e-4})
            grid.append((o.fun, K, o.x))
        grid.sort()
        class _B: pass
        best = _B(); best.fun, Kb, cb = grid[0]; best.x = [np.log(Kb), cb]
        res.setdefault("fit_grid", {})[fam] = [[float(K), float(c), float(f_)] for f_, K, c in sorted(grid, key=lambda q: q[1])]
        fits[fam] = {"K": float(np.exp(best.x[0])), "c": float(best.x[1]), "chi2": float(best.fun), "n_points": int(len(xs)),
                     "closure_ratio_mu_r_over_mu0": mu[r] / mu[0.0]}
        print("fit", fam, fits[fam], flush=True)
    if "real_r0" in fits:
        for fam in ("real_r1", "real_r6"):
            if fam in fits:
                fits[fam]["c_over_c0"] = fits[fam]["c"] / fits["real_r0"]["c"]
    res["fit_K_c"] = fits
    save("chain", res)
    return res


# ---------------------------------------------------------------------------------------------------------- x = 1 runs
def mle_pl(x, a, b):
    x = x[(x >= a) & (x <= b)].astype(float)
    if len(x) < 50:
        return None
    xs = np.arange(a, b + 1, dtype=float); slx = np.log(x).sum(); n = len(x)
    nll = lambda al: al * slx + n * np.log(np.sum(xs ** -al))
    o = optimize.minimize_scalar(nll, bounds=(0.3, 6), method="bounded", options={"xatol": 1e-6})
    return float(o.x)


def ST_peak(T, S, ratio=1.25, lo=0.2 * 0.999, npts=3, min_n=100):
    e = [lo]
    while e[-1] < T.max() + DT:
        e.append(e[-1] * ratio)
    e = np.array(e); k = np.digitize(T, e) - 1
    cnt = np.bincount(k, minlength=len(e)); sT = np.bincount(k, T, minlength=len(e)); sS = np.bincount(k, S, minlength=len(e))
    ok = cnt >= min_n
    mT = sT[ok] / cnt[ok]; mS = sS[ok] / cnt[ok]
    lT, lS = np.log(mT), np.log(mS); h = npts // 2
    sl = [(np.exp(lT[i]), np.polyfit(lT[i - h:i + h + 1], lS[i - h:i + h + 1], 1)[0]) for i in range(h, len(lT) - h)]
    sl = np.array(sl); i = int(np.argmax(sl[:, 1]))
    return float(sl[i, 1]), float(sl[i, 0])


def fit_K(Tm, Rr=1.0, t0=1.0, ratio=1.2, Kgrid=None):
    """Binned multinomial MLE of K for durations (model time) conditional on T > t0; my own bins and chain numerics."""
    Tm = Tm[Tm > t0]
    hi = np.quantile(Tm, 0.9995)
    e = [t0]
    while e[-1] < hi:
        e.append(e[-1] * ratio)
    e = np.array(e)
    c = np.r_[np.histogram(Tm, bins=e)[0], (Tm >= e[-1]).sum()]
    tg_max = e[-1] + 1

    def probs(K):
        tg, S = chain_surv_grid(K, Rr, tg_max, 0.02)
        Se = np.interp(e, tg, S); Se = Se / Se[0]
        return np.r_[Se[:-1] - Se[1:], Se[-1]]

    def nll(lK):
        p = np.maximum(probs(np.exp(lK)), 1e-300)
        return -float((c * np.log(p)).sum())
    Kg = np.exp(np.linspace(np.log(4), np.log(400), 35)) if Kgrid is None else Kgrid
    v = [nll(np.log(K)) for K in Kg]; k = int(np.argmin(v))
    o = optimize.minimize_scalar(nll, bounds=(np.log(Kg[max(k - 1, 0)]), np.log(Kg[min(k + 1, len(Kg) - 1)])), method="bounded",
                                 options={"xatol": 1e-4})
    return float(np.exp(o.x))


def comp_dev(Tm, K, Rr=1.0, t_lo=1.0, t_hi=None):
    tg, S = chain_surv_grid(K, Rr, 400, 0.02)
    if t_hi is None:                                                    # T_c: (1 + alpha_eff t) P(T > t) = 1/2 at R = 1
        _, S1 = chain_surv_grid(K, 1.0, 400, 0.02)
        t_hi = tg[np.argmax(S1 * (1 + AE * tg) < 0.5)]
    t = np.geomspace(t_lo, t_hi, 25)
    Ts = np.sort(Tm)
    Pe = 1 - np.searchsorted(Ts, t, side="right") / len(Ts)
    Pc = np.interp(t, tg, S)
    return float(np.max(np.abs(Pe / Pc - 1))), float(t_hi)


def part_x1():
    tags = ["real_r0_x1.00", "real_r1_x1.00", "real_r6_x1.00", "module_r0_x1.00", "module_r1_x1.00", "cnull_N167216_s0_r0_x1.00",
            "complete37_x1.00", "complete76_x1.00", "complete300_x1.00", "complete2000_x1.00"]
    Kref = {"real_r0_x1.00": 75.8, "real_r1_x1.00": 36.6, "real_r6_x1.00": 16.2, "module_r0_x1.00": 70.0, "module_r1_x1.00": 36.5,
            "complete37_x1.00": 37, "complete76_x1.00": 76, "complete300_x1.00": 300, "complete2000_x1.00": 2000}
    res = {}
    for t in tags:
        r = load_run(t, ["rec_T", "rec_S", "rec_status", "rec_skew_sp"])
        st = r["rec_status"]; ex = st == 0
        T = r["rec_T"][ex] * DT; S = r["rec_S"][ex].astype(float)
        Tall = np.where(st == 0, r["rec_T"] * DT, np.inf)
        row = {"n": int(len(st)), "n_extinct": int(ex.sum()), "mean_S": float(S.mean()), "P_T_gt_10": float((Tall > 10).mean())}
        pk, pkT = ST_peak(T, S); row["ST_peak_same_estimator"] = pk; row["ST_peak_T"] = pkT
        row["ST_peak_ratio1.15_3bin"] = ST_peak(T, S, ratio=1.15)[0]
        row["ST_peak_ratio1.4_3bin"] = ST_peak(T, S, ratio=1.4)[0]
        row["ST_peak_ratio1.25_5bin"] = ST_peak(T, S, ratio=1.25, npts=5)[0]
        # skewness in [8, 12) model time
        sk = r["rec_skew_sp"][ex].astype(float); m = (T >= 8) & (T < 12) & np.isfinite(sk)
        row["skew_T8_12"] = [float(sk[m].mean()), float(sk[m].std() / np.sqrt(m.sum())), int(m.sum())]
        m2 = (T >= 32) & (T < 40) & np.isfinite(sk)
        row["skew_T32_40"] = [float(sk[m2].mean()), float(sk[m2].std() / np.sqrt(max(m2.sum(), 1))), int(m2.sum())]
        # size MLE: their window [2, S_c/3] with S_c = 3.4 K (x1.94 for AL graphs), plus alternative windows
        if t in Kref and not t.startswith("cnull"):
            K = Kref[t]; Sc = (1.94 if t.startswith(("real", "module")) else 1.0) * 3.4 * K
            b = int(Sc / 3)
            row["mle_S_window"] = [2, b]
            row["mle_S"] = {f"[{a},{b}]": mle_pl(r["rec_S"][ex], a, b) for a in (2, 5, 10, 20, 40)}
            row["mle_S"].update({f"[2,{bb}]": mle_pl(r["rec_S"][ex], 2, bb) for bb in (30, 60)})
        if t.startswith(("real", "module", "complete37", "complete76")):
            Kf = fit_K(Tall[np.isfinite(Tall)] if np.isinf(Tall).sum() == 0 else Tall)
            row["K_fit_own"] = Kf
            row["comp_dev_at_Kfit"] = comp_dev(Tall, Kf)
            if t in Kref:
                row["comp_dev_at_Kref"] = comp_dev(Tall, Kref[t])
        res[t] = row
        print(t, json.dumps(row), flush=True)
    save("x1", res)
    return res


def part_minority():
    res = {}
    for t in ["real_r0_x1.00", "real_r1_x1.00", "real_r6_x1.00", "module_r0_x1.00", "module_r1_x1.00"]:
        d = np.load(f"{CR}/aval_{t}.npz", allow_pickle=True)
        off = d["samp_off"]; dat = d["samp_data"]; meta = json.loads(str(d["samp_meta"]))
        dl, info = [], []
        for i, m in enumerate(meta):
            if m["status"] != 0 or m["T"] < 50:
                continue
            x = dat[off[i]:off[i + 1]].astype(float)
            if x.shape[0] != m["T"]:
                info.append(("len_mismatch", i))
            L, R = x[:, 0].sum(), x[:, 1].sum()
            mn = 1 if L >= R else 0
            T_ = x.shape[0]
            mid = slice(int(0.4 * T_), int(0.6 * T_)); late = slice(int(0.8 * T_), T_)
            am = x[mid, 0].sum() + x[mid, 1].sum(); al = x[late, 0].sum() + x[late, 1].sum()
            if am > 0 and al > 0:
                dl.append(x[late, mn].sum() / al - x[mid, mn].sum() / am)
        dl = np.array(dl)
        bi = rng.integers(0, len(dl), (2000, len(dl)))
        bm = dl[bi].mean(1)
        res[t] = {"n": int(len(dl)), "delta_late_minus_mid": float(dl.mean()),
                  "ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], "len_mismatch": len(info)}
        print(t, res[t], flush=True)
    save("minority", res)
    return res


# ---------------------------------------------------------------------------------------------------------- QS
def qs_rows(files):
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True); B = json.loads(str(d["meta"]))["B"]
        for i in range(len(d["x"])):
            ns = float(d["nsamp"][i]); sn = d["s_n"][i]; sn2 = d["s_n2"][i]
            M = sn.sum(0) / (ns * B); V = sn2.sum(0) / (ns * B) - M ** 2
            rows.append({"x": float(d["x"][i]), "n_AL": M[1], "chi_AL": V[1] / M[1], "chi_tot": V[0] / M[0], "hist": d["hist_AL"][i],
                         "absorb": float(d["absorb"][i].sum()), "B": B})
    rows.sort(key=lambda q: q["x"])
    return rows


def parab_peak(x, y):
    k = int(np.argmax(y)); lx = np.log(x[k - 1:k + 2]); c = np.polyfit(lx, y[k - 1:k + 2], 2)
    return float(np.exp(-c[1] / (2 * c[0])))


def n_interior_modes(h, n0=3, sigma=1.5, rel=0.05):
    hs = gaussian_filter1d(np.asarray(h, float)[n0:], sigma)
    pk = [i for i in range(1, len(hs) - 1) if hs[i] > hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= rel * hs.max()]
    sig = 0
    for a, b in zip(pk[:-1], pk[1:]):
        if hs[a:b + 1].min() < 0.8 * min(hs[a], hs[b]):
            sig += 1
    return len(pk), sig


def part_qs():
    Rr = R_real_fn(); res = {}
    for key, files, r, K, gl in [("real_r0", sorted(glob.glob(f"{CR}/qs_real_r0_p*.npz")), 0, 75.8, GL),
                                 ("real_r1", sorted(glob.glob(f"{CR}/qs_real_r1_p*.npz")), 1, 36.9, GL),
                                 ("real_r6", sorted(glob.glob(f"{CR}/qs_real_r6_p*.npz")), 6, 16.2, GL)]:
        rows = qs_rows(files)
        xs = np.array([q["x"] for q in rows]); chi = np.array([q["chi_AL"] for q in rows]); nal = np.array([q["n_AL"] for q in rows])
        xp = parab_peak(xs, chi)
        Rx = Rr(xs)
        xk = (Rx - 1) * np.sqrt(K); w = (xk >= 2) & (xk <= 8)
        beta = float(np.polyfit(np.log(Rx[w] - 1), np.log(nal[w]), 1)[0])
        # chain QSD on a fine grid
        xf = np.exp(np.linspace(np.log(0.95), np.log(3.5), 120)); chis = []; mus = []
        for x in xf:
            n, p, _ = chain_qsd(K, float(Rr(np.array(x))))
            mu = p @ n; var = p @ n ** 2 - mu ** 2; chis.append(1.94 * var / mu); mus.append(1.94 * mu)
        chis = np.array(chis); kk = int(np.argmax(chis))
        beta_chain = float(np.polyfit(np.log(Rx[w] - 1), np.log(np.interp(np.log(xs[w]), np.log(xf), mus)), 1)[0])
        # alternative beta windows
        alt = {}
        for lo, hi in [(1, 4), (2, 8), (4, 16)]:
            ww = (xk >= lo) & (xk <= hi)
            if ww.sum() >= 3:
                alt[f"[{lo},{hi}]"] = [float(np.polyfit(np.log(Rx[ww] - 1), np.log(nal[ww]), 1)[0]), int(ww.sum())]
        modes = [n_interior_modes(q["hist"]) for q in rows]
        # chain QSD edge excess (chain units): does p(1) exceed the minimum over 1..mode?
        edge = []
        for x in (1.13, 1.33, 1.58):
            n, p, _ = chain_qsd(K, float(Rr(np.array(x)))); md = int(np.argmax(p))
            edge.append({"x": x, "mode_chain_units": int(n[md]), "p1_over_min_below_mode": float(p[0] / p[:max(md, 1)].min()) if md > 0 else 1.0})
        res[key] = {"chi_AL_peak_g": xp * gl, "chi_AL_peak_x": xp, "chi_AL_max": float(chi.max()), "chain_chi_peak_g": float(xf[kk] * gl),
                    "chain_chi_height_neurons": float(chis[kk]), "beta": beta, "beta_chain_same_window": beta_chain, "beta_window_n": int(w.sum()),
                    "beta_alt_windows": alt, "n_rows": len(rows), "max_interior_significant_pairs": int(max(m[1] for m in modes)),
                    "chain_qsd_edge": edge,
                    "nAL_over_chain_x0.95_to_3.1": [float(nal[0] / np.interp(np.log(xs[0]), np.log(xf), mus)),
                                                    float(nal[np.argmin(np.abs(xs - 3.12))] / np.interp(np.log(xs[np.argmin(np.abs(xs - 3.12))]), np.log(xf), mus))]}
        print(key, json.dumps(res[key]), flush=True)
    save("qs", res)
    return res


def part_field():
    res = {}
    for r in (0, 1):
        for x in ("0.90", "1.00", "1.10"):
            d = np.load(f"{CR}/field_real_r{r}_x{x}.npz", allow_pickle=True); B = json.loads(str(d["meta"]))["B"]
            h = d["h"]; ns = d["nsamp"].astype(float)
            rep = d["s_n"][:, :, 1] / ns[:, None]                       # [h, replica]
            m = rep.mean(1); se = rep.std(1, ddof=1) / np.sqrt(B)
            sl = np.diff(np.log(m)) / np.diff(np.log(h))
            sse = np.sqrt((se[1:] / m[1:]) ** 2 + (se[:-1] / m[:-1]) ** 2) / np.diff(np.log(h))
            nmid = np.sqrt(m[1:] * m[:-1])
            good = (sse < 0.25) & (nmid >= 1)
            res[f"r{r}_x{x}"] = {"n_AL": m.tolist(), "slopes_last3": sl[-3:].tolist(), "n_mid_last3": nmid[-3:].tolist(),
                                 "min_slope_good": float(sl[good].min()) if good.any() else None, "n_good": int(good.sum())}
            print(r, x, json.dumps(res[f"r{r}_x{x}"]), flush=True)
    save("field", res)
    return res


def part_c4():
    res = {}
    for r in (0, 1):
        d = np.load(f"{CR}/submodule_r{r}.npz", allow_pickle=True)
        meta = json.loads(str(d["meta"])); steps = int(round(meta["T"] / DT))
        e = d["ext_step"][:, 0, :]                                       # x = 1
        tau = []
        for i in range(e.shape[0]):
            dd = e[i] >= 0
            tau.append(np.where(dd, e[i], steps).sum() * DT / dd.sum())
        tau = np.array(tau); Kn = d["K_n"]
        p, cv = np.polyfit(np.log(Kn), np.log(tau), 1, cov=True)
        res[f"r{r}"] = {"n": d["n"].tolist(), "K_n": Kn.tolist(), "tau_x1": tau.tolist(), "deaths_x1": [int((e[i] >= 0).sum()) for i in range(len(e))],
                        "slope_vs_Kn": [float(p[0]), float(np.sqrt(cv[0, 0]))]}
        print("C4", r, json.dumps(res[f"r{r}"]), flush=True)
    # chain's own local exponent of tau_top(R = 1) over the K ranges spanned
    Ks = np.array([16.4, 25, 40, 67, 103, 300, 1000, 2900, 5631])
    tt = np.array([chain_tau_top(K) for K in Ks])
    res["chain_tau_top_R1"] = {"K": Ks.tolist(), "tau_model": tt.tolist(),
                               "slope_16_67": float(np.polyfit(np.log(Ks[:4]), np.log(tt[:4]), 1)[0]),
                               "slope_103_5631": float(np.polyfit(np.log(Ks[4:]), np.log(tt[4:]), 1)[0]),
                               "slope_16_5631": float(np.polyfit(np.log(Ks), np.log(tt), 1)[0])}
    print("chain tau_top", json.dumps(res["chain_tau_top_R1"]), flush=True)
    # per-family slopes with the nulls (fss_analysis.json, per_graph tau_x1, K_fit, diag.K_nc_glin)
    fss = json.load(open(f"{ROOT}/results/pd3/redraw_fss/fss_analysis.json"))["per_graph"]
    for r in (0, 1):
        pts = [(g_["tau_x1"], (g_["diag"] if isinstance(g_["diag"], dict) else json.loads(g_["diag"]))["K_nc_glin"], g_.get("K_fit"))
               for g_ in fss if g_["kind"] == "cnull" and abs(g_["r"] - r) < 1e-9 and g_.get("tau_x1")]
        tn = np.array([p_[0] for p_ in pts]); kn = np.array([p_[1] for p_ in pts]); kf = np.array([p_[2] for p_ in pts], float)
        sub = res[f"r{r}"]
        allK = np.r_[kn, sub["K_n"]]; allt = np.r_[tn, sub["tau_x1"]]
        res[f"r{r}"]["nulls_vs_Knc"] = float(np.polyfit(np.log(kn), np.log(tn), 1)[0])
        res[f"r{r}"]["nulls_vs_Kfit"] = float(np.polyfit(np.log(kf), np.log(tn), 1)[0])
        res[f"r{r}"]["merged_unfitted"] = float(np.polyfit(np.log(allK), np.log(allt), 1)[0])
        res[f"r{r}"]["null_Kfit_range"] = [float(kf.min()), float(kf.max())]
        res[f"r{r}"]["null_Knc_range"] = [float(kn.min()), float(kn.max())]
        # amplitude check: tau / chain_tau_top(K_fit) for nulls
        res[f"r{r}"]["null_tau_over_chain_at_Kfit"] = [float(np.min(tn / np.array([chain_tau_top(k) for k in kf]))),
                                                       float(np.max(tn / np.array([chain_tau_top(k) for k in kf])))]
        print("C4 merged", r, {k: res[f"r{r}"][k] for k in ("nulls_vs_Knc", "nulls_vs_Kfit", "merged_unfitted", "null_Kfit_range", "null_tau_over_chain_at_Kfit")}, flush=True)
    save("c4", res)
    return res


def part_v0():
    """Exact E[n(1 step)] from a uniform excitatory AL-module seed (r = 1; an excitatory seed only sends + input)."""
    C = "/MALECNS_CACHE"
    pre = np.load(f"{C}/pre.npy", mmap_mode="r"); post = np.load(f"{C}/post.npy", mmap_mode="r"); w = np.load(f"{C}/weight.npy", mmap_mode="r")
    sd = np.load(f"{CR}/seeds_module_exc366.npz"); seeds = sd["idx"]
    sign = np.load(f"{C}/sign.npy")
    pre_a = np.asarray(pre); m = np.isin(pre_a, seeds)
    P = pre_a[m].astype(np.int64); Q = np.asarray(post)[m].astype(np.int64); W = np.asarray(w)[m].astype(np.float64)
    key = P * 167216 + Q; uk, inv = np.unique(key, return_inverse=True); Ws = np.bincount(inv, W)
    up = uk // 167216; uq = uk % 167216
    c = 0.0013406441635011774                       # module_graph.npz c (= real graph normalisation, also in the aval meta)
    res = {"sign_of_seeds": np.unique(sign[seeds]).tolist(), "n_seeds": int(len(seeds))}
    for g in (0.20, 0.36):
        s = g * c * Ws; pon = 1 - np.exp(-np.tanh(s) * DT)
        # self-edges: a seed's own input does not matter (it is active); exclude q == p
        pon[uq == up] = 0.0
        per = np.bincount(np.searchsorted(seeds, up), pon, minlength=len(seeds))
        res[f"g{g:.2f}"] = float(np.exp(-DT) + per.mean())
        print("V0 exact E[n1]", g, res[f"g{g:.2f}"], flush=True)
    save("v0", res)
    return res


def ks_smin(S, smin_list, smax):
    """Clauset-style choice of S_min on a truncated window [S_min, smax]: discrete truncated power-law MLE and the KS distance
    between the empirical and fitted CDFs on the window; returns rows (S_min, alpha, KS, n)."""
    rows = []
    for a in smin_list:
        x = S[(S >= a) & (S <= smax)]
        if len(x) < 200:
            continue
        al = mle_pl(x, a, smax)
        xs = np.arange(a, smax + 1, dtype=float); pm = xs ** -al; pm /= pm.sum(); cdf_m = np.cumsum(pm)
        cnt = np.bincount((x - a).astype(int), minlength=len(xs))[:len(xs)]; cdf_e = np.cumsum(cnt) / cnt.sum()
        rows.append([int(a), float(al), float(np.abs(cdf_e - cdf_m).max()), int(len(x))])
    return rows


def part_extra():
    res = {}
    # (1) size exponent: S_min chosen by minimum KS on the same upper bound as the program (S_c/3)
    for t, smax in [("real_r0_x1.00", 166), ("real_r1_x1.00", 80), ("module_r0_x1.00", 153), ("complete76_x1.00", 86),
                    ("complete37_x1.00", 41), ("cnull_N167216_s0_r0_x1.00", 3000)]:
        r = load_run(t, ["rec_S", "rec_status"]); S = r["rec_S"][r["rec_status"] == 0]
        rows = ks_smin(S, [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50], smax)
        k = int(np.argmin([q[2] for q in rows]))
        # local slope of the log-binned density between S = 20 and smax (least squares on log-log)
        e = np.unique(np.round(np.geomspace(20, smax, 12)).astype(int)); c, _ = np.histogram(S, bins=e)
        dens = c / np.diff(e); ctr = np.sqrt(e[:-1] * e[1:]); ok = c > 0
        res[t] = {"rows_Smin_alpha_KS_n": rows, "best_by_KS": rows[k],
                  "lsq_slope_density_20_to_smax": float(np.polyfit(np.log(ctr[ok]), np.log(dens[ok]), 1)[0])}
        print(t, "best", rows[k], "lsq slope 20..smax", res[t]["lsq_slope_density_20_to_smax"], flush=True)
    # (2) chain beta slope in alternative windows (same x grid as the QS runs)
    Rr = R_real_fn(); qb = {}
    for key, pat, K in [("real_r0", "qs_real_r0_p*.npz", 75.8), ("real_r1", "qs_real_r1_p*.npz", 36.9), ("real_r6", "qs_real_r6_p*.npz", 16.2)]:
        rows = qs_rows(sorted(glob.glob(f"{CR}/{pat}")))
        xs = np.array([q["x"] for q in rows]); nal = np.array([q["n_AL"] for q in rows]); Rx = Rr(xs); xk = (Rx - 1) * np.sqrt(K)
        mu = []
        for R in Rx:
            n, p, _ = chain_qsd(K, float(R)); mu.append(1.94 * (p @ n))
        mu = np.array(mu); out = {}
        for lo, hi in [(1, 4), (2, 8), (4, 16), (8, 32)]:
            w = (xk >= lo) & (xk <= hi)
            if w.sum() >= 3:
                out[f"[{lo},{hi}]"] = {"data": float(np.polyfit(np.log(Rx[w] - 1), np.log(nal[w]), 1)[0]),
                                      "chain": float(np.polyfit(np.log(Rx[w] - 1), np.log(mu[w]), 1)[0]), "n": int(w.sum()),
                                      "R_minus_1": [float(Rx[w].min() - 1), float(Rx[w].max() - 1)]}
        qb[key] = out
        print(key, json.dumps(out), flush=True)
    res["beta_windows"] = qb
    save("extra", res)
    return res


def module_v():
    """Right Perron vector of the AL module's excitatory branching operator B = f(g E) at the module g_lin (dense, 526 x 526),
    f(y) = (1 - exp(-tanh(y)_+ dt)) / dt; E rows = post, cols = pre, excitatory presynaptic sign only, duplicates summed."""
    mg = np.load(f"{ROOT}/results/module/module_graph.npz")
    pre = mg["pre"].astype(int); post = mg["post"].astype(int); w = mg["w"].astype(float); sg = mg["sign"].astype(int); c = float(mg["c"])
    n = len(sg); E = np.zeros((n, n)); e = sg[pre] > 0
    np.add.at(E, (post[e], pre[e]), c * w[e])
    gl = 0.16428095549634494
    B = (1 - np.exp(-np.tanh(np.maximum(gl * E, 0)) * DT)) / DT
    ev, V = np.linalg.eig(B); k = int(np.argmax(ev.real))
    v = np.abs(V[:, k].real)
    lut = np.zeros(167216); lut[mg["idx"]] = v
    return lut, float(ev[k].real), float(np.sort(ev.real)[-2])


def fit_Kc(xs, gs, se, Rx):
    def loss_c(K, c):
        gm = np.array([chain_gamma(K, c * R) for R in Rx])
        return float((((gm - gs) / se) ** 2).sum())

    def best_c(K):
        o = optimize.minimize_scalar(lambda c: loss_c(K, c), bounds=(0.6, 1.3), method="bounded", options={"xatol": 1e-4})
        return o.fun, K, o.x
    coarse = sorted(best_c(K) for K in np.exp(np.linspace(np.log(8), np.log(200), 12)))
    K0 = coarse[0][1]
    fine = sorted(best_c(K) for K in K0 * np.exp(np.linspace(-0.3, 0.3, 13)))
    f_, K, c = fine[0]
    dof = max(len(xs) - 2, 1)
    # profile CI on c: chi2 <= min + chi2_min/dof (errors rescaled by sqrt(chi2/dof))
    ok = [q for q in fine if q[0] <= f_ * (1 + 1.0 / dof)]
    return {"K": float(K), "c": float(c), "chi2": float(f_), "chi2_dof": float(f_ / dof),
            "c_range_profile": [float(min(q[2] for q in ok)), float(max(q[2] for q in ok))]}


def part_vw():
    lut, lam, lam2 = module_v()
    res = {"lambda_PF_over_alpha_eff": lam / AE, "lambda2_over_lambda1": lam2 / lam}
    print("module v: lambda/alpha_eff", lam / AE, "lambda2/lambda1", lam2 / lam, flush=True)
    Rr = R_real_fn(); mu = {row["r"]: row["mu"] for row in json.load(open(f"{ROOT}/results/pd/mode_real.json"))["rows"]}
    fits = {}
    for fam, pref, r in [("real_r0", "real_r0_x", 0.0), ("real_r1", "real_r1_x", 1.0), ("real_r6", "real_r6_x", 6.0)]:
        tags = sorted([os.path.basename(f)[5:-4] for f in glob.glob(f"{CR}/aval_{pref}*.npz")])
        rows = []
        for t in tags:
            d = load_run(t, ["rec_n_at", "rec_seed"])
            X = d["rec_n_at"].astype(float); w = lut[d["rec_seed"].astype(int)]
            mu_u = X.mean(0); mu_w = (w[:, None] * X).sum(0) / w.sum()
            se_u = X.std(0) / np.sqrt(len(X)); ww = w / w.sum(); se_w = np.sqrt(((ww[:, None] * (X - mu_w)) ** 2).sum(0))
            gu = np.polyfit(AGES * DT, np.log(mu_u), 1)[0]; gw = np.polyfit(AGES * DT, np.log(mu_w), 1)[0]
            # delta-method SE of the slope (independent ages approximation is conservative enough for weighting)
            Xd = (AGES * DT - (AGES * DT).mean()) / ((AGES * DT - (AGES * DT).mean()) ** 2).sum()
            sgu = float(np.sqrt(((Xd * se_u / mu_u) ** 2).sum())); sgw = float(np.sqrt(((Xd * se_w / mu_w) ** 2).sum()))
            rows.append((d["x"], gu, sgu, gw, sgw))
        rows.sort()
        xs = np.array([q[0] for q in rows]); Rx = np.array([float(Rr(np.array(x))) for x in xs])
        for lab, gi, si in [("uniform", 1, 2), ("vw", 3, 4)]:
            gs = np.array([q[gi] for q in rows]); se = np.maximum(np.array([q[si] for q in rows]), 5e-4)
            f = fit_Kc(xs, gs, se, Rx); f["x_star"] = upcross(xs, gs); f["gamma"] = gs.tolist()
            fits[f"{fam}_{lab}"] = f
            print(fam, lab, json.dumps({k: v for k, v in f.items() if k != "gamma"}), flush=True)
    for lab in ("uniform", "vw"):
        for fam, r in (("real_r1", 1.0), ("real_r6", 6.0)):
            fits[f"{fam}_{lab}"]["c_over_c0"] = fits[f"{fam}_{lab}"]["c"] / fits[f"real_r0_{lab}"]["c"]
            fits[f"{fam}_{lab}"]["closure_mu_ratio"] = mu[r] / mu[0.0]
    res["fits"] = fits
    save("vw_and_Kc_fit", res)
    return res


def part_fig():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    V = json.load(open(OUT))
    OI = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "orange": "#E69F00", "sky": "#56B4E9", "pink": "#CC79A7", "black": "#000000", "grey": "#777777"}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    # (a) size exponent vs S_min
    a = ax[0]
    lab = {"real_r0_x1.00": ("real brain r=0 (upper bound 166)", OI["blue"], "o-"), "real_r1_x1.00": ("real brain r=1 (80)", OI["verm"], "s-"),
           "module_r0_x1.00": ("AL module alone r=0 (153)", OI["green"], "D--"), "cnull_N167216_s0_r0_x1.00": ("config. null r=0 (3000)", OI["black"], "x:"),
           "complete76_x1.00": ("complete graph K=76, one-mode control (86)", OI["grey"], "^-.")}
    for t, (lb, c, mk) in lab.items():
        rows = np.array(V["extra"][t]["rows_Smin_alpha_KS_n"]); b = V["extra"][t]["best_by_KS"]
        a.plot(rows[:, 0], rows[:, 1], mk, color=c, ms=4.5, lw=1.4, label=lb)
        a.plot([b[0]], [b[1]], "*", color=c, ms=15, mec="k", mew=0.6)
    a.axhline(1.5, color="k", lw=0.8, ls="--"); a.text(1.05, 1.507, "MF-DP 3/2", fontsize=9)
    a.axvline(2, color=OI["grey"], lw=0.8, ls=":"); a.text(2.1, 1.88, "program's S_min = 2", fontsize=8.5, color=OI["grey"])
    a.set_xscale("log"); a.set_xlabel("lower bound S_min of the fit window (spikes)"); a.set_ylabel("truncated discrete power-law MLE exponent of P(S)")
    a.set_title("(a) size exponent at x = 1 depends on S_min\n(star = S_min minimising the KS distance)", fontsize=10.5)
    a.legend(frameon=False, fontsize=8, loc="lower left"); a.grid(alpha=0.3); a.set_ylim(1.2, 1.95)
    # (b) lifetime vs structural K
    a = ax[1]
    fss = json.load(open(f"{ROOT}/results/pd3/redraw_fss/fss_analysis.json"))["per_graph"]
    Kc = np.geomspace(10, 8000, 60); tc = [chain_tau_top(K) for K in Kc]
    a.plot(Kc, tc, color=OI["grey"], lw=2, label="exact chain, R = 1, from n = K (local slope 0.56 at K 16-67, 0.51 at 100-5,600)")
    for r, c in ((0, OI["blue"]), (1, OI["verm"])):
        cc = V["c4"][f"r{r}"]
        a.plot(cc["K_n"], cc["tau_x1"], "o" if r == 0 else "s", color=c, ms=6, label=f"nested AL sub-modules r={r} (K_n): slope {cc['slope_vs_Kn'][0]:.2f}")
        pts = [(g_["tau_x1"], (g_["diag"] if isinstance(g_["diag"], dict) else json.loads(g_["diag"]))["K_nc_glin"]) for g_ in fss
               if g_["kind"] == "cnull" and abs(g_["r"] - r) < 1e-9 and g_.get("tau_x1")]
        a.plot([q[1] for q in pts], [q[0] for q in pts], "x" if r == 0 else "+", color=c, ms=7, mew=1.5,
               label=f"configuration nulls r={r} (K_nc): slope {cc['nulls_vs_Knc']:.2f}")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("structural module size K (K_n, K_nc; not fitted)")
    a.set_ylabel("mean lifetime at x = 1 from fully active (model time)")
    a.set_title("(b) lifetime vs unfitted K: within-family slopes 0.25-0.38\n(pre-registered Q2-c used K_n / K_nc)", fontsize=10.5)
    a.legend(frameon=False, fontsize=7.6, loc="upper left"); a.grid(alpha=0.3, which="both")
    # (c) effective branching scale vs r
    a = ax[2]; f = V["vw_and_Kc_fit"]["fits"]
    mu = {row["r"]: row["mu"] for row in json.load(open(f"{ROOT}/results/pd/mode_real.json"))["rows"]}
    rr = np.array(sorted(mu)); a.plot(rr, [mu[q] / mu[0.0] for q in rr], ":", color=OI["pink"], lw=2, label="net-input closure: mu(r) / mu(0)")
    a.axhline(1.0, color=OI["grey"], lw=2, ls="--", label="exact linearisation (r-independent g_lin)")
    for lab_, c, mk in (("uniform", OI["blue"], "o-"), ("vw", OI["orange"], "s-")):
        cs = [f[f"real_r{r}_{lab_}"]["c"] for r in (0, 1, 6)]
        Ks = [f[f"real_r{r}_{lab_}"]["K"] for r in (0, 1, 6)]
        a.plot([0, 1, 6], np.array(cs) / cs[0], mk, color=c, ms=7, lw=1.5,
               label=("uniform seeds" if lab_ == "uniform" else "seeds weighted by v (post hoc)") + f"; free K = {Ks[0]:.0f} / {Ks[1]:.0f} / {Ks[2]:.0f}")
    a.set_xlabel("inhibition scale r"); a.set_ylabel("best-fit branching scale c(r) / c(0)")
    a.set_title("(c) 2-parameter chain fit of gamma(x), R = c R_exact(x), K free\n(independent of K_needed)", fontsize=10.5)
    a.legend(frameon=False, fontsize=8, loc="lower left"); a.grid(alpha=0.3); a.set_ylim(0.8, 1.03)
    fig.suptitle("Verification of program criticality (code/pd3_verify_criticality.py): three methodological checks", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/figs/pd3/verify_criticality.png", dpi=130); print("saved figs/pd3/verify_criticality.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="all"); a = ap.parse_args()
    parts = {"aval": part_aval, "chain": part_chain, "x1": part_x1, "minority": part_minority, "qs": part_qs, "field": part_field,
             "c4": part_c4, "v0": part_v0, "extra": part_extra, "vw": part_vw, "fig": part_fig}
    for k, fn in parts.items():
        if a.part in ("all", k):
            t0 = time.time(); print(f"=== {k}", flush=True); fn(); print(f"=== {k} done {time.time() - t0:.0f} s", flush=True)
