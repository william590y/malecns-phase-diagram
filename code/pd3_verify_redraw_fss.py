"""Adversarial verification of program redraw_fss (results/pd3/redraw_fss.md). Independent re-implementation from RAW outputs:
does NOT import pd3_common / pd3_fss_analyze / pd3_redraw (own PAVA, crossing, chain, K fit, censored MLE, bootstrap).

Raw inputs:
  results/pd3/redraw_fss/fss_{cnull,realsub}_N*_s*_r*.npz and fss_fine_cnull_*.npz (ext_step, x, g, R_x, n_cnt/n_s1/n_s2, diag)
  results/analysis2/phase_complete.json (verified boundaries, real full-brain x_on), results/pd3/theory_bd.json,
  results/pd3/theory_nullK.json, results/pd3/module_theory/island_lifetimes.json, results/pd3/redraw/heatmap_*.npz
Output: results/pd3/redraw_fss/verify/verify_redraw_fss.json (+ stdout log).
"""
import glob, json, os, re, sys
import numpy as np
from scipy import stats
from scipy.optimize import minimize, curve_fit, brentq

ROOT = "/MALECNS_ROOT"
RES = f"{ROOT}/results"
D = f"{RES}/pd3/redraw_fss"
OUT = f"{D}/verify"
os.makedirs(OUT, exist_ok=True)
DTS = 0.2
AEFF = (1 - np.exp(-DTS)) / DTS
TMAX = 3000.0
rng = np.random.default_rng(424242)
V = {}


def P(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------- own primitives
def pava(y, w):
    """Weighted pool-adjacent-violators, increasing."""
    y = list(map(float, y)); w = list(map(float, w))
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


def cross_up(x, y, thr):
    """first i with y[i] >= thr; linear interpolation in (ln x, y). Returns (x*, i) or (None, status)."""
    idx = np.nonzero(y >= thr)[0]
    if len(idx) == 0:
        return None, "none"
    i = idx[0]
    if i == 0:
        return None, "below"
    if y[i] == y[i - 1]:
        return float(x[i]), i
    f = (thr - y[i - 1]) / (y[i] - y[i - 1])
    return float(np.exp(np.log(x[i - 1]) + f * (np.log(x[i]) - np.log(x[i - 1])))), i


def chain_lntau(lam, K, ntop):
    """ln mean extinction time from the top state ntop of the birth-death chain: birth lam n (1 - n/K) for n < ntop (0 at
    ntop), death AEFF n. tau = sum_k theta_k, theta_k = sum_{j>=k} (1/d_j) prod_{i=k}^{j-1} b_i/d_i."""
    if ntop < 1:
        return -np.inf
    n = np.arange(1, ntop + 1, dtype=float)
    b = lam * n * (1 - n / K); b[-1] = 0.0
    d = AEFF * n
    L = np.full(ntop, -np.inf)
    pos = b[:-1] > 0
    L[:-1][pos] = np.log(b[:-1][pos]) - np.log(d[:-1][pos])
    # C_j = sum_{i<j} L_i (j = 1..ntop), with -inf propagating
    C = np.concatenate([[0.0], np.cumsum(L[:-1])])
    a = C - np.log(d)                                          # term for j
    # logsumexp over j >= k, computed from the back
    rev = np.logaddexp.accumulate(a[::-1])[::-1]
    with np.errstate(invalid="ignore"):
        lth = rev - C
    lth = lth[np.isfinite(lth)]
    m = lth.max()
    return float(m + np.log(np.exp(lth - m).sum()))


def chain_smooth(lam, K):
    k0 = int(np.floor(K)); f = K - k0
    a = chain_lntau(lam, K, k0)
    return a if f == 0 else (1 - f) * a + f * chain_lntau(lam, K, k0 + 1)


def chain_check():
    """validate chain_lntau against a direct linear solve of the generator for small K."""
    out = []
    for lam, K in [(1.3 * AEFF, 12.0), (AEFF, 20.0), (0.8 * AEFF, 7.0), (2.0 * AEFF, 30.0)]:
        nt = int(np.floor(K)); n = np.arange(1, nt + 1, dtype=float)
        b = lam * n * (1 - n / K); b[-1] = 0; d = AEFF * n
        Q = np.zeros((nt, nt))
        for i in range(nt):
            Q[i, i] = -(b[i] + d[i])
            if i + 1 < nt:
                Q[i, i + 1] = b[i]
            if i - 1 >= 0:
                Q[i, i - 1] = d[i]
        T = np.linalg.solve(-Q, np.ones(nt))
        out.append(abs(np.log(T[-1]) - chain_lntau(lam, K, nt)))
    return max(out)


# ------------------------------------------------------------------------------------------------- load raw
def load_graph(kind, N, s, r):
    pts = []
    diag = None
    for pref in ["fss_", "fss_fine_"]:
        f = f"{D}/{pref}{kind}_N{N}_s{s}_r{r}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        if diag is None:
            diag = json.loads(str(z["diag"]))
        ext = z["ext_step"]; T = float(z["T"]); dt = float(z["dt"])
        assert abs(dt - DTS) < 1e-12 and T == TMAX
        for i in range(len(z["x"])):
            e = ext[i]
            died = e >= 0
            t = np.where(died, e * dt, T)
            pts.append(dict(x=float(z["x"][i]), g=float(z["g"][i]), R=float(z["R_x"][i]), died=died, t=t,
                            cnt=z["n_cnt"][i], s1=z["n_s1"][i], s2=z["n_s2"][i], src=pref))
    pts.sort(key=lambda p: p["x"])
    merged = []
    for p in pts:
        if merged and abs(p["x"] / merged[-1]["x"] - 1) < 1e-6:
            q = merged[-1]
            for k in ["died", "t", "cnt", "s1", "s2"]:
                q[k] = np.concatenate([q[k], p[k]])
        else:
            merged.append(dict(p))
    return merged, diag


def iso_lt(d, E):
    return pava(np.log(E / np.maximum(d, 0.5)), d + 0.5)


def analyse(pts, diag, boot=0):
    x = np.array([p["x"] for p in pts]); R = np.array([p["R"] for p in pts])
    d = np.array([p["died"].sum() for p in pts], float); E = np.array([p["t"].sum() for p in pts])
    B = np.array([len(p["died"]) for p in pts])
    lt = iso_lt(d, E)
    on, ion = cross_up(x, lt, np.log(3000.0)); S, iS = cross_up(x, lt, np.log(200.0))
    x100, _ = cross_up(x, lt, np.log(100.0)); x1e4, _ = cross_up(x, lt, np.log(1e4))
    res = dict(x_on=on, x_S=S, x100=x100, x1e4=x1e4, n=len(x),
               bracket_on=[float(x[ion - 1]), float(x[ion])] if on else None)
    # tau(x=1)
    tau = np.where(d > 0, E / np.maximum(d, 1), np.inf)
    k = int(np.searchsorted(x, 1.0))
    if d[k - 1] == B[k - 1] and d[k] == B[k]:
        f = (0 - np.log(x[k - 1])) / (np.log(x[k]) - np.log(x[k - 1]))
        res["tau_x1"] = float(np.exp(np.log(tau[k - 1]) + f * (np.log(tau[k]) - np.log(tau[k - 1]))))
    else:
        res["tau_x1"] = None
    # K fit (least squares in ln tau over 200 <= tau <= 3000, >= 3 deaths)
    sel = np.nonzero((d >= 3) & (tau >= 200) & (tau <= 3000))[0]
    res["K_n"] = int(len(sel))
    if len(sel):
        def sse(lk):
            K = np.exp(lk)
            return sum((np.log(tau[i]) - chain_smooth(R[i] * AEFF, K)) ** 2 for i in sel)
        grid = np.linspace(np.log(3), np.log(1e5), 120)
        v = np.array([sse(g_) for g_ in grid]); j = int(np.argmin(v))
        lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
        for _ in range(60):                                   # golden section
            m1 = lo + 0.382 * (hi - lo); m2 = lo + 0.618 * (hi - lo)
            if sse(m1) < sse(m2):
                hi = m2
            else:
                lo = m1
        res["K_fit"] = float(np.exp(0.5 * (lo + hi)))
        res["K_rms"] = float(np.sqrt(sse(0.5 * (lo + hi)) / len(sel)))
    else:
        res["K_fit"] = None
    # censored-exponential MLE slope over the tau in [100, 1e4] window (same window rule as the report)
    below = np.nonzero(lt < np.log(100.0))[0]; above = np.nonzero(lt >= np.log(1e4))[0]
    i0 = below[-1] if len(below) else 0; i1 = above[0] if len(above) else len(x) - 1
    w = np.arange(i0, i1 + 1)
    if (d[w] >= 3).sum() >= 2:
        u = np.log(x[w]) - np.log(x[w]).mean()
        nll = lambda th: float(np.sum(d[w] * (th[0] + th[1] * u) + E[w] * np.exp(-(th[0] + th[1] * u))))
        th0 = np.array([np.log(E[w].sum() / max(d[w].sum(), 1)), 20.0])
        o = minimize(nll, th0, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-10, maxiter=20000))
        o = minimize(nll, o.x, method="BFGS")
        res["s_fit"] = float(o.x[1]); res["s_fit_window_x"] = [float(x[i0]), float(x[i1])]
        res["s_fit_window"] = w.tolist()
        res["s_fit_lowest_x"] = float(x[i0])
    else:
        res["s_fit"] = None
    # chi_surv
    cnt = np.array([p["cnt"].sum() for p in pts]); s1 = np.array([p["s1"].sum() for p in pts])
    s2 = np.array([p["s2"].sum() for p in pts]); nrep = np.array([(p["cnt"] > 0).sum() for p in pts])
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / cnt; var = s2 / cnt - mu ** 2; chi = np.where((cnt >= 100) & (nrep >= 3), var / mu, np.nan)
    res["chi_win"] = None
    if on and S:
        ww = np.nonzero((x >= S) & (x <= 1.25 * on) & np.isfinite(chi))[0]
        if len(ww):
            jj = ww[np.argmax(chi[ww])]; res["chi_win"] = float(chi[jj]); res["chi_win_x_over_on"] = float(x[jj] / on)
    # bootstrap of x_on
    if boot:
        bs = []
        for b in range(boot):
            db = np.empty(len(x)); Eb = np.empty(len(x))
            for i, p in enumerate(pts):
                ii = rng.integers(0, len(p["died"]), len(p["died"]))
                db[i] = p["died"][ii].sum(); Eb[i] = p["t"][ii].sum()
            v, _ = cross_up(x, iso_lt(db, Eb), np.log(3000.0)); bs.append(np.nan if v is None else v)
        res["boot_on"] = np.array(bs)
    res.update(x=x, R=R, d=d, E=E, lt=lt, tau=tau)
    return res


def chain_secant(xs, Rs, K, xlo=None):
    """chain's own ln(100)/ln(x(1e4)/x(100)) with R(x) interpolated in ln x (grid of the graph)."""
    xf = np.exp(np.linspace(np.log(xs[0]), np.log(xs[-1]), 2500))
    Rf = np.exp(np.interp(np.log(xf), np.log(xs), np.log(Rs)))
    lt = np.array([chain_smooth(r_ * AEFF, K) for r_ in Rf])
    a, _ = cross_up(xf, lt, np.log(100.0)); b, _ = cross_up(xf, lt, np.log(1e4))
    return (np.log(100.0) / np.log(b / a)) if (a and b) else None, a, b


# ------------------------------------------------------------------------------------------------- main
P("chain self-check (max |d ln tau| vs direct generator solve):", chain_check())
V["chain_selfcheck_max_abs_dlntau"] = chain_check()

PJ = json.load(open(f"{RES}/analysis2/phase_complete.json"))
GL = PJ["g_lin"]["real"]
REALX = {0: PJ["sim"]["real"]["0"]["on"]["g"] / GL, 1: PJ["sim"]["real"]["1"]["on"]["g"] / GL}
keys = []
for f in sorted(glob.glob(f"{D}/fss_*_N*_s*_r*.npz")):
    m = re.match(r"fss_(cnull|realsub)_N(\d+)_s(\d+)_r(\d)\.npz", os.path.basename(f))
    if m:
        keys.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))))
nfine = len(glob.glob(f"{D}/fss_fine_cnull_*.npz"))
P("coarse files", len(keys), "fine files", nfine)
V["n_coarse_files"] = len(keys); V["n_fine_files"] = nfine
A = {}
for k in keys:
    pts, diag = load_graph(*k)
    nb = 400 if (k[0] == "cnull" and k[1] in (2500, 167216) and k[3] == 0) or (k == ("cnull", 167216, 0, 1)) else 0
    a = analyse(pts, diag, boot=nb); a["diag"] = diag; A[k] = a
    P(f"{k}: n {a['n']} x_S {a['x_S']} x_on {a['x_on']} tau1 {a['tau_x1']} K_fit {a['K_fit']} (n {a['K_n']}, rms {a.get('K_rms')}) "
      f"s_fit {a['s_fit']} chi_win {a['chi_win']} K_nc {diag['K_nc_glin']:.1f} g_lin {diag['g_lin']:.6f}")

# ---------------- FSS-1
NS = [2500, 5000, 10000, 20000, 40000, 80000, 167216]
fss1 = {}
for r in [0, 1]:
    means, sds = [], []
    for N in NS:
        v = [A[k]["x_on"] for k in A if k[0] == "cnull" and k[1] == N and k[3] == r]
        assert all(z is not None for z in v) and len(v) == 3
        means.append(float(np.mean(v))); sds.append(float(np.std(v, ddof=1)))
    sp = stats.spearmanr(NS, means).correlation
    fss1[r] = dict(N=NS, mean=means, sd=sds, spearman=float(sp), strictly_decreasing=bool(np.all(np.diff(means) < 0)))
    P("FSS1 r", r, fss1[r])
V["FSS1"] = fss1
# seed-mean replica bootstrap CIs at 2.5k and 167k, r = 0
for N in [2500, 167216]:
    bm = np.mean([A[k]["boot_on"] for k in A if k[0] == "cnull" and k[1] == N and k[3] == 0], axis=0)
    V[f"seedmean_ci_r0_N{N}"] = [float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5))]
    P("seed-mean CI r0", N, V[f"seedmean_ci_r0_N{N}"])

# ---------------- FSS-2
fss2 = {}
for r in [0, 1]:
    devs = {f"{k[1]}_s{k[2]}": A[k]["x_on"] / REALX[r] - 1 for k in A if k[0] == "realsub" and k[3] == r}
    gabs = [A[k]["x_on"] * A[k]["diag"]["g_lin"] for k in A if k[0] == "realsub" and k[3] == r]
    fss2[r] = dict(ref=REALX[r], devs=devs, max_abs=float(max(abs(v) for v in devs.values())), g_abs=[min(gabs), max(gabs)],
                   K_fit=[min(A[k]["K_fit"] for k in A if k[0] == "realsub" and k[3] == r), max(A[k]["K_fit"] for k in A if k[0] == "realsub" and k[3] == r)],
                   K_fit_npoints=[A[k]["K_n"] for k in A if k[0] == "realsub" and k[3] == r])
    P("FSS2 r", r, fss2[r])
V["FSS2"] = fss2
IL = json.load(open(f"{RES}/pd3/module_theory/island_lifetimes.json"))
V["AL_alone"] = {r: dict(x_on=IL[f"AL_r{r}"]["x_tau3000"], rel=IL[f"AL_r{r}"]["x_tau3000"] / REALX[r] - 1,
                         g_abs=IL[f"AL_r{r}"]["g_tau3000"], g_abs_rel=IL[f"AL_r{r}"]["g_tau3000"] / PJ["sim"]["real"][str(r)]["on"]["g"] - 1)
                 for r in [0, 1]}
P("AL alone", V["AL_alone"])

# ---------------- FSS-3
fss3 = {}
CH_x1 = {}
for r in [0, 1]:
    ks = [k for k in A if k[0] == "cnull" and k[3] == r]
    lN = np.log([k[1] for k in ks]); lK = np.log([A[k]["K_fit"] for k in ks]); lKn = np.log([A[k]["diag"]["K_nc_glin"] for k in ks])
    lt1 = np.log([A[k]["tau_x1"] for k in ks])
    a = stats.linregress(lN, lK); an = stats.linregress(lN, lKn)
    t_fit = stats.linregress(lK, lt1); t_nc = stats.linregress(lKn, lt1)
    # seed-mean regression (geometric means per N) and local slopes
    gm = [np.exp(np.mean([np.log(A[k]["K_fit"]) for k in ks if k[1] == N])) for N in NS]
    a_sm = stats.linregress(np.log(NS), np.log(gm))
    loc = [float(np.log(gm[i + 1] / gm[i]) / np.log(NS[i + 1] / NS[i])) for i in range(len(NS) - 1)]
    a_lo = stats.linregress(np.log(NS[:4]), np.log(gm[:4])); a_hi = stats.linregress(np.log(NS[3:]), np.log(gm[3:]))
    # K_fit / K_nc
    ratio = {N: [A[k]["K_fit"] / A[k]["diag"]["K_nc_glin"] for k in ks if k[1] == N] for N in NS}
    # Spearman s vs K_fit (cnull + realsub, as in the report)
    kk = [k for k in A if k[3] == r and A[k]["K_fit"] and A[k]["s_fit"]]
    sp_s = stats.spearmanr([A[k]["K_fit"] for k in kk], [A[k]["s_fit"] for k in kk]).correlation
    # chain exponent of tau(R=1) over the same K range
    Kr = np.array([A[k]["K_fit"] for k in ks])
    Kg = np.geomspace(Kr.min(), Kr.max(), 25)
    ch = np.array([chain_smooth(AEFF, K_) for K_ in Kg])
    ch_slope = stats.linregress(np.log(Kg), ch).slope
    # chain prediction at each graph's own K_fit: ratio tau_sim(x=1) / tau_chain(R=1, K_fit)
    rat = [float(A[k]["tau_x1"] / np.exp(chain_smooth(AEFF, A[k]["K_fit"]))) for k in ks]
    rat_rs = [float(A[k]["tau_x1"] / np.exp(chain_smooth(AEFF, A[k]["K_fit"]))) for k in A if k[0] == "realsub" and k[3] == r]
    fss3[r] = dict(a=float(a.slope), a_se=float(a.stderr), r2=float(a.rvalue ** 2), a_Knc=float(an.slope), a_Knc_se=float(an.stderr),
                   a_seedmean=float(a_sm.slope), a_seedmean_se=float(a_sm.stderr), local_slopes=loc,
                   a_N_le_20k=float(a_lo.slope), a_N_ge_20k=float(a_hi.slope),
                   tau1_vs_Kfit=float(t_fit.slope), tau1_vs_Kfit_se=float(t_fit.stderr), tau1_vs_Knc=float(t_nc.slope),
                   tau1_vs_Knc_se=float(t_nc.stderr), chain_slope_same_K_range=float(ch_slope),
                   tau1_over_chain_nulls=[min(rat), max(rat)], tau1_over_chain_realsub=[min(rat_rs), max(rat_rs)],
                   spearman_sfit_Kfit=float(sp_s), n_spearman=len(kk),
                   Kfit_over_Knc={str(N): [round(v, 3) for v in ratio[N]] for N in NS},
                   K_fit_npoints={str(N): [A[k]["K_n"] for k in ks if k[1] == N] for N in NS},
                   tau_x1_range=[float(np.exp(lt1.min())), float(np.exp(lt1.max()))])
    P("FSS3 r", r, json.dumps(fss3[r]))
V["FSS3"] = fss3
rms_all = [A[k].get("K_rms") for k in A if A[k]["K_fit"]]
V["K_fit_rms_max"] = float(max(rms_all))
V["K_fit_rms_max_npoints_ge3"] = float(max(A[k]["K_rms"] for k in A if A[k]["K_fit"] and A[k]["K_n"] >= 3))
P("max rms ln tau of K fits", V["K_fit_rms_max"], "(graphs with >= 3 points:", V["K_fit_rms_max_npoints_ge3"], ")")

# ---------------- steepness: data secant vs chain secant with each graph's own R(x) and K_fit
sec = []
for k in sorted(A):
    a = A[k]
    if not a["K_fit"]:
        continue
    s_data = (np.log(100.0) / np.log(a["x1e4"] / a["x100"])) if (a["x100"] and a["x1e4"]) else None
    s_ch, c100, c1e4 = chain_secant(a["x"], a["R"], a["K_fit"])
    # chain-implied s_fit: same censored-MLE slope fitted to the chain's ln tau at the same window points with the observed deaths
    s_ch_fit = None
    if a["s_fit"] is not None:
        w = np.array(a["s_fit_window"]); u = np.log(a["x"][w]) - np.log(a["x"][w]).mean()
        ltc = np.array([chain_smooth(R_ * AEFF, a["K_fit"]) for R_ in a["R"][w]])
        EE = a["E"][w]; dd = EE / np.exp(ltc)                    # expected deaths given the observed exposure
        nll = lambda th: float(np.sum(dd * (th[0] + th[1] * u) + EE * np.exp(-(th[0] + th[1] * u))))
        o = minimize(nll, np.array([np.log(EE.sum() / max(dd.sum(), 1)), 20.0]), method="Nelder-Mead",
                     options=dict(xatol=1e-9, fatol=1e-10, maxiter=20000))
        s_ch_fit = float(o.x[1])
    sec.append(dict(key=list(k), K_fit=a["K_fit"], s_fit=a["s_fit"], s_fit_lowest_x=a.get("s_fit_lowest_x"),
                    s_data_secant=s_data, s_chain_secant=s_ch, s_chain_fit_same_window=s_ch_fit,
                    x100_data=a["x100"], x100_chain=c100, tau_x1=a["tau_x1"]))
    P(f"steep {k}: K_fit {a['K_fit']:.1f} s_fit {a['s_fit']:.2f} (window from x {a.get('s_fit_lowest_x'):.3f}) chain-same-window "
      f"{s_ch_fit if s_ch_fit is None else round(s_ch_fit, 2)} | secant data {s_data if s_data is None else round(s_data, 2)} chain "
      f"{s_ch if s_ch is None else round(s_ch, 2)}")
V["steepness"] = sec
for r in [0, 1]:
    rr = [z for z in sec if z["key"][3] == r and z["s_chain_fit_same_window"]]
    q = np.array([z["s_fit"] / z["s_chain_fit_same_window"] for z in rr])
    rr2 = [z for z in sec if z["key"][3] == r and z["s_data_secant"] and z["s_chain_secant"]]
    q2 = np.array([z["s_data_secant"] / z["s_chain_secant"] for z in rr2])
    V[f"steep_ratio_r{r}"] = dict(sfit_over_chain_same_window=[float(q.min()), float(np.median(q)), float(q.max())], n=len(q),
                                  secant_ratio=[float(q2.min()), float(np.median(q2)), float(q2.max())] if len(q2) else None, n_secant=len(q2),
                                  n_without_chain_secant=sum(1 for z in sec if z["key"][3] == r and not z["s_chain_secant"]))
    P("steep ratio r", r, V[f"steep_ratio_r{r}"])
s167 = {r: [A[k]["s_fit"] for k in A if k[0] == "cnull" and k[1] == 167216 and k[3] == r] for r in [0, 1]}
s25 = {r: [A[k]["s_fit"] for k in A if k[0] == "cnull" and k[1] == 2500 and k[3] == r] for r in [0, 1]}
s80 = {r: [A[k]["s_fit"] for k in A if k[0] == "cnull" and k[1] == 80000 and k[3] == r] for r in [0, 1]}
s40 = {r: [A[k]["s_fit"] for k in A if k[0] == "cnull" and k[1] == 40000 and k[3] == r] for r in [0, 1]}
V["s_fit_by_N_r0"] = {"2500": s25[0], "40000": s40[0], "80000": s80[0], "167216": s167[0]}
P("s_fit r0 2.5k / 40k / 80k / 167k:", s25[0], s40[0], s80[0], s167[0])

# ---------------- chi_win
chi = {}
for kind in ["cnull", "realsub"]:
    for r in [0, 1]:
        ks = [k for k in A if k[0] == kind and k[3] == r and A[k]["chi_win"]]
        sl = stats.linregress(np.log([k[1] for k in ks]), np.log([A[k]["chi_win"] for k in ks]))
        chi[f"{kind}_r{r}"] = dict(slope=float(sl.slope), se=float(sl.stderr), n=len(ks),
                                   range=[min(A[k]["chi_win"] for k in ks), max(A[k]["chi_win"] for k in ks)],
                                   by_N={str(N): [round(A[k]["chi_win"], 2) for k in ks if k[1] == N] for N in sorted(set(k[1] for k in ks))})
        P("chi", kind, r, chi[f"{kind}_r{r}"])
V["chi_win"] = chi

# ---------------- onset extrapolation, with alternatives
ext = {}
for r in [0, 1]:
    N = np.array(NS, float); xm = np.array(fss1[r]["mean"]); se = np.maximum(np.array(fss1[r]["sd"]) / np.sqrt(3), 1e-3)
    f3 = lambda n, xi, Am, b: xi + Am * (n / 1e4) ** (-b)
    out = {}
    for lab, sel, sig in [("weighted_all", slice(None), se), ("unweighted_all", slice(None), None),
                          ("weighted_drop_2.5k", slice(1, None), se[1:]), ("weighted_drop_167k", slice(0, -1), se[:-1]),
                          ("weighted_N_ge_10k", slice(2, None), se[2:])]:
        try:
            p, c = curve_fit(f3, N[sel], xm[sel], p0=[1.0, 0.2, 0.4], sigma=sig, absolute_sigma=sig is not None, maxfev=50000)
            out[lab] = dict(x_inf=float(p[0]), x_inf_se=float(np.sqrt(c[0, 0])), b=float(p[2]), b_se=float(np.sqrt(c[2, 2])),
                            chi2=float(np.sum(((xm[sel] - f3(N[sel], *p)) / (sig if sig is not None else 1)) ** 2)))
        except Exception as e:
            out[lab] = str(e)
    # fixed x_inf = 1 (2 params) vs free (3 params): chi2 comparison, weighted
    f2 = lambda n, Am, b: 1 + Am * (n / 1e4) ** (-b)
    p2, c2 = curve_fit(f2, N, xm, p0=[0.2, 0.4], sigma=se, absolute_sigma=True, maxfev=50000)
    out["fixed_x_inf_1"] = dict(b=float(p2[1]), b_se=float(np.sqrt(c2[1, 1])), chi2=float(np.sum(((xm - f2(N, *p2)) / se) ** 2)))
    out["loglog_slope_xon_minus_1"] = float(np.polyfit(np.log(N), np.log(xm - 1), 1)[0])
    # K-based form: x_on - 1 = c K_fit^-1/2 ? (geometric-mean K per N)
    ext[r] = out
    P("extrapolation r", r, json.dumps(out))
V["onset_extrapolation"] = ext

# ---------------- full-null refinement
for r in [0, 1]:
    a = A[("cnull", 167216, 0, r)]; gl = a["diag"]["g_lin"]
    bo = a["boot_on"]
    V[f"full_null_s0_r{r}"] = dict(g_lin=gl, g_on=a["x_on"] * gl, bracket_g=[a["bracket_on"][0] * gl, a["bracket_on"][1] * gl],
                                   boot_ci_g=[float(np.nanpercentile(bo, 2.5) * gl), float(np.nanpercentile(bo, 97.5) * gl)],
                                   g_S=a["x_S"] * gl)
    P("full null s0 r", r, V[f"full_null_s0_r{r}"])
V["full_null_glin_vs_phase_complete"] = [A[("cnull", 167216, 0, 0)]["diag"]["g_lin"], PJ["g_lin"]["null"]]

# ---------------- validation vs theory_nullK (recomputed from diag vs the json itself)
TN = json.load(open(f"{RES}/pd3/theory_nullK.json"))["rows"]
dv = []
for row in TN:
    for r in [0, 1]:
        k = ("cnull", row["N"], row["seed"], r)
        if k in A:
            dg = A[k]["diag"]
            dv.append(max(abs(dg["lambda_PF_E"] / row["lambda_PF"] - 1), abs(dg["K_nc_E"] / row["K_nc"] - 1)))
V["validation_theory_nullK"] = dict(n=len(dv), max_rel_dev=float(max(dv)))
P("validation vs theory_nullK", V["validation_theory_nullK"])
# AL shares
V["al_share_realsub"] = [min(A[k]["diag"]["al_share_v2_glin"] for k in A if k[0] == "realsub"), max(A[k]["diag"]["al_share_v2_glin"] for k in A if k[0] == "realsub")]
V["al_share_cnull"] = [min(A[k]["diag"]["al_share_v2_glin"] for k in A if k[0] == "cnull"), max(A[k]["diag"]["al_share_v2_glin"] for k in A if k[0] == "cnull")]
V["realsub_glin"] = [min(A[k]["diag"]["g_lin"] for k in A if k[0] == "realsub"), max(A[k]["diag"]["g_lin"] for k in A if k[0] == "realsub")]
V["realsub_Knc"] = [min(A[k]["diag"]["K_nc_glin"] for k in A if k[0] == "realsub"), max(A[k]["diag"]["K_nc_glin"] for k in A if k[0] == "realsub")]
P("AL share realsub", V["al_share_realsub"], "cnull", V["al_share_cnull"], "realsub g_lin", V["realsub_glin"], "K_nc", V["realsub_Knc"])

# ---------------- kappa-N rule rows used in the text
TB = json.load(open(f"{RES}/pd3/theory_bd.json"))["null_family"]
V["kappaN_r1_needed"] = [(z["N"], z["K"], z["on_over_glin"]) for z in TB["rows"]["r1_needed"]]
P("kappa-N r1_needed", V["kappaN_r1_needed"])

# ---------------- Part A: crossings of the stored heat-map field vs the verified boundaries
pa = {}
for kind in ["real", "null", "wnull"]:
    H = np.load(f"{RES}/pd3/redraw/heatmap_{kind}.npz")
    g = H["g"]; LT = H["ln_tau_iso"]; ins = H["inside"]
    mx = {"S": 0.0, "on": 0.0}; nin = {"S": 0, "on": 0}; nrow = 0; nmulti = 0
    for i, r in enumerate(H["r"]):
        rk = f"{r:g}"
        sim = PJ["sim"][kind][rk]
        nrow += 1
        for lab, thr in [("S", 200.0), ("on", 3000.0)]:
            m = ins[i] & np.isfinite(LT[i])
            gg, yy = g[m], LT[i][m]
            v, _ = cross_up(gg, yy, np.log(thr))
            ref = sim[lab]["g"]
            if v is None or ref is None:
                continue
            up = np.sum((yy[1:] >= np.log(thr)) & (yy[:-1] < np.log(thr)))
            nmulti += int(up > 1)
            mx[lab] = max(mx[lab], abs(np.log(v / ref)))
            tot = sim[lab]["total"]
            nin[lab] += int(tot[0] <= v <= tot[1])
    pts = sum(PJ["sim"][kind][rk]["n_points"] for rk in PJ["sim"][kind])
    z0 = sum(1 for rk in PJ["sim"][kind] for p in PJ["sim"][kind][rk]["points"] if p["deaths"] == 0)
    pa[kind] = dict(max_abs_dlng=mx, n_inside_total=nin, n_rows=nrow, rows_multi_crossing=nmulti, n_points=pts, n_zero_death_points=z0,
                    step=float(np.log(g[1] / g[0])))
    P("Part A", kind, pa[kind])
V["partA"] = pa

# ---------------- per-graph records
V["per_graph"] = []
for k in sorted(A):
    a = A[k]
    V["per_graph"].append(dict(key=list(k), x_on=a["x_on"], x_S=a["x_S"], K_fit=a["K_fit"], K_fit_n=a["K_n"], K_nc=a["diag"]["K_nc_glin"],
                               tau_x1=a["tau_x1"], tau_chain_x1=float(np.exp(chain_smooth(AEFF, a["K_fit"]))), s_fit=a["s_fit"],
                               chi_win=a["chi_win"]))

json.dump(V, open(f"{OUT}/verify_redraw_fss.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
P("saved", f"{OUT}/verify_redraw_fss.json")

# ---------------- verification figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter
OI = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "orange": "#E69F00", "sky": "#56B4E9", "purple": "#CC79A7"}
fig, axs = plt.subplots(1, 3, figsize=(18, 5.8))
# A: onset extrapolation sensitivity (r = 0)
ax = axs[0]
Nn = np.array(NS, float); xm = np.array(fss1[0]["mean"])
for k in A:
    if k[0] == "cnull" and k[3] == 0:
        ax.plot(k[1], A[k]["x_on"], "o", color=OI["blue"], mfc="white", ms=5, zorder=3)
ax.plot(Nn, xm, "s", color=OI["blue"], ms=7, zorder=4, label="configuration null r = 0: seed mean (o = seeds)")
Nf = np.geomspace(2000, 1e8, 300)
f3 = lambda n, xi, Am, b: xi + Am * (n / 1e4) ** (-b)
for lab, col, ls in [("weighted_all", "k", "-"), ("unweighted_all", OI["verm"], "--"), ("weighted_N_ge_10k", OI["green"], "-."),
                     ("weighted_drop_167k", OI["purple"], ":")]:
    e = ext[0][lab]
    sel = {"weighted_all": slice(None), "unweighted_all": slice(None), "weighted_N_ge_10k": slice(2, None), "weighted_drop_167k": slice(0, -1)}[lab]
    sig = None if lab == "unweighted_all" else np.maximum(np.array(fss1[0]["sd"]) / np.sqrt(3), 1e-3)[sel]
    p_, _ = curve_fit(f3, Nn[sel], xm[sel], p0=[1.0, 0.2, 0.4], sigma=sig, absolute_sigma=sig is not None, maxfev=50000)
    ax.plot(Nf, f3(Nf, *p_), ls, color=col, lw=1.6,
            label=f"x_inf + A N^-b, {lab.replace('_', ' ')}: x_inf = {e['x_inf']:.3f} +/- {e['x_inf_se']:.3f}")
ax.axhline(1, color="#555555", lw=1)
ax.set_xscale("log"); ax.set_xlabel("network size N (neurons)", fontsize=10)
ax.set_ylabel("x_on = g_on / g_lin  (tau_sim = 3000)", fontsize=10)
ax.set_ylim(0.8, 1.4)
ax.set_title("A  Null onset extrapolation depends on weighting / N range", fontsize=10.5, loc="left")
ax.legend(fontsize=7.6, loc="upper right"); ax.grid(alpha=0.25, which="both", lw=0.5)
# B: steepness like-for-like
ax = axs[1]
for z in sec:
    k = z["key"]; col = OI["blue"] if k[3] == 0 else OI["verm"]; mk = "o" if k[0] == "cnull" else "s"
    if z["s_chain_fit_same_window"]:
        ax.plot(z["K_fit"], z["s_fit"] / z["s_chain_fit_same_window"], mk, color=col, mfc=col, ms=6)
    if z["s_data_secant"] and z["s_chain_secant"]:
        ax.plot(z["K_fit"], z["s_data_secant"] / z["s_chain_secant"], mk, color=col, mfc="white", ms=6, mew=1.3)
ax.axhline(1, color="k", lw=1.2)
ax.set_xscale("log"); ax.set_xlabel("K_fit (one-mode chain fit to the slow window)", fontsize=10)
ax.set_ylabel("measured steepness / chain steepness (same estimator)", fontsize=10)
ax.set_title("B  Steepness vs the chain with each graph's own R(x), K_fit", fontsize=10.5, loc="left")
ax.legend(handles=[Line2D([], [], color=OI["blue"], marker="o", ls="", label="r = 0"), Line2D([], [], color=OI["verm"], marker="o", ls="", label="r = 1"),
                   Line2D([], [], color="#777777", marker="o", ls="", label="filled: censored-MLE slope s_fit, chain fitted on the same points"),
                   Line2D([], [], color="#777777", marker="o", mfc="white", ls="", label="open: secant ln(100)/ln(x(1e4)/x(100)), data vs chain"),
                   Line2D([], [], color="#777777", marker="o", ls="", label="circle: configuration null; square: realsub")][:4] +
          [Line2D([], [], color="#777777", marker="s", ls="", label="square: real subsample (realsub); circle: null")], fontsize=7.6, loc="upper right")
ax.set_ylim(0.85, 1.5); ax.grid(alpha=0.25, which="both", lw=0.5)
# C: lifetime at g_lin over the chain's prediction at R = 1 with K_fit
ax = axs[2]
for z in V["per_graph"]:
    k = z["key"]; col = OI["blue"] if k[3] == 0 else OI["verm"]; mk = "o" if k[0] == "cnull" else "s"
    if z["tau_x1"]:
        ax.plot(z["K_fit"], z["tau_x1"] / z["tau_chain_x1"], mk, color=col, ms=6)
ax.axhline(1, color="k", lw=1.2)
ax.set_xscale("log"); ax.set_xlabel("K_fit (one-mode chain fit to the slow window)", fontsize=10)
ax.set_ylabel("tau_sim(x = 1) / chain tau(R = 1, K_fit)", fontsize=10)
ax.set_title("C  Out-of-sample lifetime at g_lin vs the chain", fontsize=10.5, loc="left")
ax.legend(handles=[Line2D([], [], color=OI["blue"], marker="o", ls="", label="r = 0 configuration null"),
                   Line2D([], [], color=OI["verm"], marker="o", ls="", label="r = 1 configuration null"),
                   Line2D([], [], color=OI["blue"], marker="s", ls="", label="r = 0 realsub"),
                   Line2D([], [], color=OI["verm"], marker="s", ls="", label="r = 1 realsub")], fontsize=8, loc="upper right")
ax.set_ylim(0.8, 2.3); ax.grid(alpha=0.25, which="both", lw=0.5)
fig.tight_layout()
os.makedirs(f"{ROOT}/figs/pd3", exist_ok=True)
fig.savefig(f"{ROOT}/figs/pd3/verify_redraw_fss.png", dpi=150); plt.close(fig)
P("saved figs/pd3/verify_redraw_fss.png")
