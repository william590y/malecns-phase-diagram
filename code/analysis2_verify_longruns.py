"""Independent adversarial re-check of results/analysis2/longruns.{md,json} (written by analysis2_longruns.py).

Does NOT import or copy analysis2_longruns.py. Recomputes from the raw npz / json:
  A  per-g extinction times (first recorded silent time, censored at T), censored-exponential MLE tau, chi2 CI
  B  theory tau (logtau_r1 from branch_{real,null}_dense_part*.json), ratio tau_sim/tau_th, summary stats
  C  K needed to reproduce tau_sim with the theory's lambda(g) -- own birth-death formula (closed-form sum, not the
     downward recursion), plus: what a single constant K predicts at every g (is K "nearly constant" good enough?)
  D  exponential shape: CV and a parametric-bootstrap KS (Lilliefors-style; grid + censoring reproduced)
  E  rho(t) mean-density fits: exponential vs power law in two windows
  F  T-drift: g where half the replicas survive to T (direct, pooled logistic) and a censored-exp tau(g) model;
     theory g where tau_th = T; sensitivity of the T=30000 extrapolation to the tau(g) model form
  G  quasi-stationary number of active neurons among survivors
Writes results/analysis2/verify_longruns.json and prints a report.
"""
import glob, json
import numpy as np
from scipy import stats, optimize

ROOT = "/MALECNS_ROOT"
R = f"{ROOT}/results"
N = 167216
A_EFF = (1 - np.exp(-0.2)) / 0.2
rng = np.random.default_rng(12345)
OUT = {}

FILES = {
    "real": {"long": f"{R}/pd/long_real.npz", "fine": f"{R}/fine_decay_raw_r1.npz",
             "stage1": f"{R}/pd/decay_r1_real.npz", "stage0": f"{R}/decay_raw_r1.npz"},
    "null": {"long": f"{R}/pd/long_null.npz", "fine": f"{R}/fine_decay_raw_r1_null.npz",
             "stage1": f"{R}/pd/decay_r1_null.npz", "stage0": f"{R}/decay_raw_r1_null.npz"},
}


def load(path):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    assert meta["r"] == 1.0 and meta["norm"] == "raw" and meta["h"] == 0.0, (path, meta)
    return d["t"], d["g"], d["rho"], d["surv"], meta


def ext_times(t, surv_g):
    """surv_g [t, B] -> (times, dead flag). time = first grid time with replica silent; censored at t[-1]."""
    s = surv_g > 0
    dead = ~s[-1]
    first0 = np.argmax(~s, axis=0)
    tt = np.where(dead, t[first0], t[-1])
    # sanity: absorbing -> once silent stays silent
    for b in range(s.shape[1]):
        if dead[b]:
            assert not s[first0[b]:, b].any()
    return tt, dead


def tau_mle(tt, dead):
    d = dead.sum(); TT = tt.sum()
    if d == 0:
        return np.inf, (TT / stats.chi2.ppf(0.5, 2) * 2, np.inf), d
    lo = 2 * TT / stats.chi2.ppf(0.975, 2 * d + 2)
    hi = 2 * TT / stats.chi2.ppf(0.025, 2 * d)
    return TT / d, (lo, hi), d


def theory(kind):
    rows = []
    for f in sorted(glob.glob(f"{R}/pd/branch_{kind}_dense_part*.json")):
        J = json.load(open(f)); rows += J["rows"]; glin = J["g_lin"]
    rows.sort(key=lambda r: r["g"])
    g = np.array([r["g"] for r in rows])
    return dict(g=g, lam=np.array([r["lam"] for r in rows]), lt1=np.array([r["logtau_r1"] for r in rows]),
                K1=np.array([r["K_r1"] for r in rows]), K0=np.array([r["K_r0"] for r in rows]), g_lin=glin,
                nparts=len(glob.glob(f"{R}/pd/branch_{kind}_dense_part*.json")))


def interp_lg(th, key, g):
    lg = np.log(th["g"])
    x = np.log(g)
    if x < lg[0] or x > lg[-1]:
        return np.nan
    return float(np.interp(x, lg, th[key]))


def logtau_bd(lam, K, a=A_EFF):
    """Mean extinction time from the top state of b(n)=lam n(1-n/K) (b(n_top)=0), d(n)=a n, n_top=floor(K).
    Closed form: E[time n->n-1] = sum_{j>=n} (1/d_j) prod_{i=n}^{j-1} b_i/d_i ; tau = sum_n of that."""
    m = int(np.floor(K))
    if m < 1:
        return -np.inf
    n = np.arange(1, m + 1, dtype=float)
    b = lam * n * (1 - n / K); b[-1] = 0.0
    d = a * n
    with np.errstate(divide="ignore"):
        lr = np.log(b[:-1]) - np.log(d[:-1])          # log(b_i/d_i), i = 1..m-1
    C = np.concatenate([[0.0], np.cumsum(lr)])       # C[k] = sum_{i<k} lr_i  (k = 0..m-1)
    # log E_n = logsumexp_j>=n ( C[j]-C[n] - log d_j )
    tot = []
    for k in range(m):
        tot.append(np.logaddexp.reduce(C[k:] - C[k] - np.log(d[k:])))
    return float(np.logaddexp.reduce(np.array(tot)))


def K_needed(lam, logtau_target):
    lo, hi = 1.0, 2000.0
    if logtau_bd(lam, hi) < logtau_target:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if logtau_bd(lam, mid) < logtau_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------------------------- A, B, C
for kind in ["real", "null"]:
    th = theory(kind)
    o = {"theory_parts": th["nparts"], "g_lin": th["g_lin"], "rows": []}
    # check own recursion against a stored theory value
    j = len(th["g"]) // 3
    o["recursion_check"] = {"g": float(th["g"][j]), "stored": float(th["lt1"][j]),
                            "mine": logtau_bd(th["lam"][j], th["K1"][j])}
    for setname in ["long", "fine"]:
        t, gs, rho, surv, meta = load(FILES[kind][setname])
        T = meta["T"]
        for i, g in enumerate(gs):
            tt, dead = ext_times(t, surv[i])
            tau, ci, d = tau_mle(tt, dead)
            lt = interp_lg(th, "lt1", g)
            lam = interp_lg(th, "lam", g)
            row = {"set": setname, "T": T, "g": float(g), "deaths": int(d), "n": int(len(tt)), "tau": tau,
                   "ci": [float(ci[0]), float(ci[1])], "tau_th": float(np.exp(lt)) if np.isfinite(lt) else None}
            if np.isfinite(lt) and np.isfinite(tau):
                row["ratio"] = tau / np.exp(lt)
                row["K_needed"] = K_needed(lam, np.log(tau))
                row["K1_th"] = interp_lg(th, "K1", g); row["K0_th"] = interp_lg(th, "K0", g)
                row["lam"] = lam
            if d == len(tt):
                row["cv"] = float(tt.std(ddof=0) / tt.mean())
            # quasi-stationary active count among survivors (t >= 20, >= 25% alive)
            alive = surv[i].sum(1)
            w = (t >= 20) & (alive >= 0.25 * surv.shape[2])
            if w.any():
                row["n_active_surv_median"] = float(np.median(N * rho[i][w].sum(1) / alive[w]))
            o["rows"].append(row)
    sel = [r for r in o["rows"] if r.get("ratio") is not None and r["deaths"] >= 5 and r["tau"] >= 200]
    rat = np.array([r["ratio"] for r in sel]); gg = np.array([r["g"] for r in sel])
    Kn = np.array([r["K_needed"] for r in sel])
    o["summary"] = {"n": len(sel), "g": gg.tolist(), "sets": [r["set"] for r in sel],
                    "geomean": float(np.exp(np.log(rat).mean())), "median": float(np.median(rat)),
                    "min": float(rat.min()), "max": float(rat.max()),
                    "slope_dlnratio_dlng": float(np.polyfit(np.log(gg), np.log(rat), 1)[0]),
                    "K_needed_median": float(np.median(Kn)), "K_needed_range": [float(Kn.min()), float(Kn.max())],
                    "K1_th_median": float(np.median([r["K1_th"] for r in sel])),
                    "K0_th_median": float(np.median([r["K0_th"] for r in sel])),
                    "n_distinct_g": int(len(np.unique(np.round(gg, 6))))}
    # constant-K check: one K for all selected g -> predicted tau vs sim CI
    Kc = float(np.median(Kn))
    chk = []
    for r in sel:
        lp = logtau_bd(r["lam"], Kc)
        chk.append({"g": r["g"], "set": r["set"], "tau_sim": r["tau"], "ci": r["ci"], "tau_pred_constK": float(np.exp(lp)),
                    "inside_ci": bool(r["ci"][0] <= np.exp(lp) <= r["ci"][1])})
    o["constK_check"] = {"K": Kc, "n_inside_ci": int(sum(c["inside_ci"] for c in chk)), "n": len(chk), "rows": chk}
    OUT[kind] = o

# ------------------------------------------------------------------------------------------------------------- D
def boot_cv_ks(tt, dead, t_grid, T, nb=2000):
    """Parametric bootstrap under a pure exponential through 0 with the MLE tau, the recording grid (first grid time
    >= true death time) and censoring at T. Returns CV (if uncensored), H0 CV 2.5/50/97.5 %, and KS p (vs fitted exp CDF,
    statistic on uncensored part via max |ECDF - F| over observed death times; censored replicas set to T)."""
    n = len(tt)
    tau, _, d = tau_mle(tt, dead)

    def ks_stat(x, dd, tau_):
        xs = np.sort(x)
        F = 1 - np.exp(-xs / tau_)
        ecdf_hi = np.arange(1, n + 1) / n; ecdf_lo = np.arange(0, n) / n
        mask = xs < T          # only compare where uncensored
        return float(np.max(np.maximum(np.abs(ecdf_hi - F), np.abs(ecdf_lo - F))[mask])) if mask.any() else 0.0

    def to_grid(x):
        k = np.searchsorted(t_grid, x, side="left")
        out = np.where(k < len(t_grid), t_grid[np.minimum(k, len(t_grid) - 1)], T)
        return np.minimum(out, T), x <= T

    D0 = ks_stat(tt, dead, tau)
    Ds, cvs = [], []
    for _ in range(nb):
        x = rng.exponential(tau, n)
        xg, dd = to_grid(x)
        xg = np.where(dd, xg, T)
        tb, _, db = tau_mle(xg, dd)
        if db == 0:
            continue
        Ds.append(ks_stat(xg, dd, tb))
        if dd.all():
            cvs.append(xg.std() / xg.mean())
    Ds = np.array(Ds)
    res = {"ks_D": D0, "ks_p": float((Ds >= D0).mean())}
    if dead.all():
        res["cv"] = float(tt.std() / tt.mean())
        res["cv_H0"] = [float(np.percentile(cvs, q)) for q in (2.5, 50, 97.5)]
    return res


for kind in ["real", "null"]:
    shape = []
    for setname in ["long", "fine"]:
        t, gs, rho, surv, meta = load(FILES[kind][setname])
        for i, g in enumerate(gs):
            tt, dead = ext_times(t, surv[i])
            tau, _, d = tau_mle(tt, dead)
            if tau >= 200 and d >= 10:
                r = boot_cv_ks(tt, dead, t, meta["T"])
                r.update({"set": setname, "g": float(g), "deaths": int(d), "tau": tau})
                shape.append(r)
    ps = np.array([s["ks_p"] for s in shape])
    fisher = stats.combine_pvalues(np.clip(ps, 1e-4, 1), method="fisher")[1]
    cvs = [s for s in shape if "cv" in s]
    OUT[kind]["shape"] = {"rows": shape, "n": len(shape), "n_reject_05": int((ps < 0.05).sum()),
                          "fisher_p": float(fisher), "cv_mean": float(np.mean([s["cv"] for s in cvs])) if cvs else None,
                          "cv_H0_median_mean": float(np.mean([s["cv_H0"][1] for s in cvs])) if cvs else None,
                          "n_cv": len(cvs),
                          "n_cv_outside_H0": int(sum((s["cv"] < s["cv_H0"][0]) or (s["cv"] > s["cv_H0"][2]) for s in cvs))}

# ------------------------------------------------------------------------------------------------------------- E
def r2(y, yh):
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


for kind in ["real", "null"]:
    rows = []
    for s in OUT[kind]["shape"]["rows"]:
        t, gs, rho, surv, meta = load(FILES[kind][s["set"]])
        i = int(np.argmin(np.abs(gs - s["g"])))
        m = rho[i].mean(1); alive = surv[i].mean(1)
        # main window: t >= 20 while >= 25% replicas alive
        w = (t >= 20) & (alive >= 0.25) & (m > 0)
        lt, lr, tw = np.log(t[w]), np.log(m[w]), t[w]
        bp, ap = np.polyfit(lt, lr, 1); r2p = r2(lr, ap + bp * lt)
        ke, ce = np.polyfit(tw, lr, 1); r2e = r2(lr, ce + ke * tw)
        # power x exp: ln rho = c + b ln t - t/tau
        X = np.vstack([np.ones_like(tw), lt, tw]).T
        coef = np.linalg.lstsq(X, lr, rcond=None)[0]
        # legacy window (analyze.py definition): t >= 5, t <= 0.9 * time mean rho first hits 0, rho > 0
        life = t[np.argmax(m == 0)] if (m == 0).any() else np.inf
        end = 0.9 * life if np.isfinite(life) else t[-1]
        wl = (t >= 5) & (t <= end) & (m > 0)
        ltl, lrl = np.log(t[wl]), np.log(m[wl])
        bl, al = np.polyfit(ltl, lrl, 1); r2pl = r2(lrl, al + bl * ltl)
        kl, cl = np.polyfit(t[wl], lrl, 1); r2el = r2(lrl, cl + kl * t[wl])
        rows.append({"set": s["set"], "g": s["g"], "npts": int(w.sum()), "r2_pow": float(r2p), "r2_exp": float(r2e),
                     "prefers_exp": bool(r2e > r2p), "tau_exp": float(-1 / ke), "tau_exp_over_tau_sim": float(-1 / ke / s["tau"]),
                     "powexp_b": float(coef[1]), "legacy_b": float(bl), "legacy_prefers_pow": bool(r2pl > r2el)})
    OUT[kind]["rho_fits"] = {"rows": rows, "n_exp": sum(r["prefers_exp"] for r in rows),
                             "n_legacy_pow": sum(r["legacy_prefers_pow"] for r in rows), "n": len(rows)}

# ------------------------------------------------------------------------------------------------------------- F
def alive_at(t, surv_g, T):
    """alive at T: use first grid time >= T (alive there => alive at T, conservative) and last grid <= T."""
    k_after = np.searchsorted(t, T - 1e-9, side="left")
    k_before = np.searchsorted(t, T + 1e-9, side="right") - 1
    return surv_g[min(k_after, len(t) - 1)] > 0, surv_g[k_before] > 0, t[min(k_after, len(t) - 1)], t[k_before]


def logistic_half(gs, ys):
    x = np.log(np.asarray(gs)); y = np.asarray(ys, float)
    def nll(p):
        z = p[0] + p[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)
    p = optimize.minimize(nll, [0.0, 10.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000}).x
    return float(np.exp(-p[0] / p[1])), p


for kind in ["real", "null"]:
    th = theory(kind)
    drift = {}
    for T in [3000.0, 8000.0, 30000.0]:
        G, Y, Yb, per = [], [], [], {}
        for setname, path in FILES[kind].items():
            t, gs, rho, surv, meta = load(path)
            if meta["T"] < T - 1e-6:
                continue
            gl, yl, ybl = [], [], []
            for i, g in enumerate(gs):
                a_after, a_before, ta, tb = alive_at(t, surv[i], T)
                G += [g] * len(a_after); Y += list(a_after); Yb += list(a_before)
                gl.append(g); yl.append(a_after.mean()); ybl.append(a_before.mean())
            yl = np.array(yl); gl = np.array(gl)
            # interpolated crossing of fraction-alive = 0.5 (first crossing from below)
            cross = None
            for k in range(len(gl) - 1):
                if yl[k] < 0.5 <= yl[k + 1]:
                    f = (0.5 - yl[k]) / (yl[k + 1] - yl[k])
                    cross = float(np.exp(np.log(gl[k]) + f * (np.log(gl[k + 1]) - np.log(gl[k]))))
                    break
            per[setname] = {"cross_interp": cross, "grid_after": float(ta), "grid_before": float(tb),
                            "max_alive_frac": float(yl.max())}
        res = {"per_sweep": per}
        if len(set(Y)) > 1:
            res["logistic_after"] = logistic_half(G, Y)[0]
            res["logistic_before"] = logistic_half(G, Yb)[0]
            # bootstrap CI of the logistic half point (resample replicas within g)
            G_, Y_ = np.array(G), np.array(Y, float)
            bs = []
            for _ in range(300):
                idx = rng.integers(0, len(G_), len(G_))
                if len(set(Y_[idx])) > 1:
                    bs.append(logistic_half(G_[idx], Y_[idx])[0])
            res["logistic_after_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        # theory crossings: tau_th = T and tau_th = T/ln2
        for lab, target in [("theory_tau_eq_T", np.log(T)), ("theory_tau_eq_T_over_ln2", np.log(T / np.log(2)))]:
            lt = th["lt1"]; k = np.argmax(lt >= target)
            f = (target - lt[k - 1]) / (lt[k] - lt[k - 1])
            res[lab] = float(np.exp(np.log(th["g"][k - 1]) + f * (np.log(th["g"][k]) - np.log(th["g"][k - 1]))))
        drift[str(int(T))] = res
    OUT[kind]["T_drift"] = drift

    # censored-exp tau(g) model: ln tau = poly(ln g), fitted on long+fine, g window as stated by the analysed agent
    gwin = (0.285, 0.45) if kind == "real" else (1.02, 1.21)
    data = []
    for setname in ["long", "fine"]:
        t, gs, rho, surv, meta = load(FILES[kind][setname])
        for i, g in enumerate(gs):
            if gwin[0] - 1e-9 <= g <= gwin[1] + 1e-9:
                tt, dead = ext_times(t, surv[i])
                data.append((g, tt.sum(), dead.sum()))
    data = np.array(data)
    lx = np.log(data[:, 0]); TT = data[:, 1]; dd = data[:, 2]

    def fit(deg):
        def nll(p):
            ltau = np.polyval(p, lx)
            return np.sum(dd * ltau + TT * np.exp(-ltau))
        p0 = np.zeros(deg + 1); p0[-1] = np.log(TT.sum() / max(dd.sum(), 1))
        r = optimize.minimize(nll, p0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 200000})
        r = optimize.minimize(nll, r.x, method="BFGS")
        return r.x, r.fun

    model = {}
    for deg in [1, 2]:
        p, f = fit(deg)
        out = {"nll": float(f)}
        for T in [3000.0, 8000.0, 30000.0]:
            # half survival: exp(-T/tau) = 1/2  -> ln tau = ln(T/ln2)
            gg = np.exp(np.linspace(np.log(gwin[0]), np.log(gwin[1]) + 0.3, 20001))
            lt = np.polyval(p, np.log(gg))
            k = np.argmax(lt >= np.log(T / np.log(2)))
            out[str(int(T))] = float(gg[k]) if k > 0 else None
        model[f"poly{deg}"] = out
    model["LRT_p_quad_vs_lin"] = float(stats.chi2.sf(2 * (model["poly1"]["nll"] - model["poly2"]["nll"]), 1))
    # sensitivity: quadratic fit on the fine sweep only (no T=30000 data) and linear on g >= 0.33 / 1.08
    OUT[kind]["tau_model"] = model
    OUT[kind]["T_drift_ratios"] = {
        "theory_g_30000_over_3000": drift["30000"]["theory_tau_eq_T"] / drift["3000"]["theory_tau_eq_T"],
        "theory_lng_30000_over_3000": np.log(drift["30000"]["theory_tau_eq_T"] / th["g_lin"]) / np.log(drift["3000"]["theory_tau_eq_T"] / th["g_lin"]),
        "sim_model_g_30000_over_3000": model["poly2"]["30000"] / model["poly2"]["3000"],
        "sim_model_lng_30000_over_3000": np.log(model["poly2"]["30000"] / th["g_lin"]) / np.log(model["poly2"]["3000"] / th["g_lin"]),
        "sqrt_lnaT_law": float(np.sqrt(np.log(A_EFF * 30000) / np.log(A_EFF * 3000))),
    }
    # what would the theory onset drift be with K = K_needed median (theory lambda(g))?
    Kc = OUT[kind]["summary"]["K_needed_median"]
    gfine = np.exp(np.linspace(np.log(th["g"][0]), np.log(th["g"][-1]), 4000))
    lam_f = np.interp(np.log(gfine), np.log(th["g"]), th["lam"])
    ltK = np.array([logtau_bd(l, Kc) for l in lam_f])
    onK = {}
    for T in [3000.0, 8000.0, 30000.0]:
        k = np.argmax(ltK >= np.log(T)); onK[str(int(T))] = float(gfine[k])
    onK["lng_ratio_30000_over_3000"] = float(np.log(onK["30000"] / th["g_lin"]) / np.log(onK["3000"] / th["g_lin"]))
    onK["g_ratio_30000_over_3000"] = onK["30000"] / onK["3000"]
    OUT[kind]["theory_with_K_needed_onsets"] = onK

json.dump(OUT, open(f"{R}/analysis2/verify_longruns.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------------------------------------------------ print
for kind in ["real", "null"]:
    o = OUT[kind]
    print(f"===== {kind}  theory parts {o['theory_parts']}  g_lin {o['g_lin']:.4f}  recursion check {o['recursion_check']}")
    for r in o["rows"]:
        if r["set"] == "long" or (r.get("ratio") is not None and r["tau"] >= 100) or r["deaths"] < 32:
            print(f"  {r['set']:5s} g={r['g']:.4f} d={r['deaths']:2d}/{r['n']} tau={r['tau']:9.4g} CI=[{r['ci'][0]:.4g},{r['ci'][1]:.4g}]"
                  f" th={r['tau_th'] if r['tau_th'] is None else round(r['tau_th'],2)} ratio={r.get('ratio', float('nan')):.4g}"
                  f" Kneed={r.get('K_needed', float('nan')):.3g} K1={r.get('K1_th', float('nan')):.3g} cv={r.get('cv', float('nan')):.3g}"
                  f" nact={r.get('n_active_surv_median', float('nan')):.4g}")
    print("  summary", json.dumps(o["summary"]))
    print("  constK", o["constK_check"]["K"], o["constK_check"]["n_inside_ci"], "/", o["constK_check"]["n"])
    for c in o["constK_check"]["rows"]:
        print(f"     g={c['g']:.4f} {c['set']} sim={c['tau_sim']:.4g} [{c['ci'][0]:.4g},{c['ci'][1]:.4g}] predK={c['tau_pred_constK']:.4g} in={c['inside_ci']}")
    s = o["shape"]
    print("  shape: n", s["n"], "reject", s["n_reject_05"], "fisher", round(s["fisher_p"], 3), "cv_mean", s["cv_mean"],
          "H0 median mean", s["cv_H0_median_mean"], "n_cv", s["n_cv"], "outside", s["n_cv_outside_H0"])
    for r in s["rows"]:
        print(f"     {r['set']} g={r['g']:.4f} KS p={r['ks_p']:.3f} cv={r.get('cv', float('nan')):.3f} H0={r.get('cv_H0')}")
    rf = o["rho_fits"]
    print("  rho: exp preferred", rf["n_exp"], "/", rf["n"], " legacy power preferred", rf["n_legacy_pow"])
    for r in rf["rows"]:
        print(f"     {r['set']} g={r['g']:.4f} npts={r['npts']} R2pow={r['r2_pow']:.3f} R2exp={r['r2_exp']:.3f} tau_exp/tau_sim={r['tau_exp_over_tau_sim']:.2f}"
              f" powexp_b={r['powexp_b']:.3f} legacy_b={r['legacy_b']:.3f} legacy_pow={r['legacy_prefers_pow']}")
    print("  T drift:", json.dumps(o["T_drift"], indent=None))
    print("  tau model:", json.dumps(o["tau_model"]))
    print("  ratios:", json.dumps(o["T_drift_ratios"]))
    print("  theory onsets with K_needed:", json.dumps(o["theory_with_K_needed_onsets"]))
