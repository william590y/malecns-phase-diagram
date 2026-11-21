"""Driven plane (external drive h > 0, r = 1, raw norm): simulation vs theory P5 (explainer/theory.html section 7).

INPUTS (read-only)
  simulation, fully active start, T = 1000, dt 0.2, h = drive:
    stage 1 : results/pd/drive_h{1e-5,1e-4,1e-3,1e-2,1e-1}_real.npz, results/pd/drive_h1e-3_null.npz (32 g, 0.1-30, B 16)
    pd2     : results/pd2/drive_h*_real.npz, drive_h1e-3_null.npz (40 g, 0.1-3, B 32) -- used whenever present
  theory  : results/pd/branch_real_dense_part*.json, branch_null_dense_part*.json (R, K_r*, logtau_r* per g)
            results/pd/qmf_drive_real.npz (QMF, same h, same grid, same start, same T)
            results/pd/flow_params.json (n_M = 526, the antennal-lobe module), results/pd/mode_{real,null}.json (1/IPR of
            the r = 1 rectified mode), results/analysis2/spreading.json (effective size of the module seed sets)
            results/analysis2/driven_linresp.json (code/analysis2_driven_linresp.py: exact linear response below g_lin)
  h = 0 reference (same graph, r = 1): results/analysis2/longruns.json (measured module lifetimes tau_sim(g), from
            results/fine_decay_raw_r1.npz and results/pd/long_real.npz), results/fine_decay_raw_r1{,_null}.npz (on-level)

MEASURES per (h, g); "late" = the recorded snapshots with t >= T/2 (10 log-spaced snapshots in [525, 1000]):
  rho      mean fraction active over replicas and late snapshots
  chi_rep  N * var over replicas (ddof 1) of each replica's late-time mean rho        (replica fluctuation)
  chi_time N * mean over replicas of the var over late snapshots (ddof 1)             (temporal fluctuation)
  F_rep, F_time = chi / rho: variance-to-mean ratio of the active-neuron count (1 = independent neurons). Used for the
           peak search because raw chi grows with the size of the active population once the bulk is recruited
           (g >~ 0.6), which hides the module peak; this choice was made after seeing the raw curves.
  drift    (mean of the last 5 late snapshots - mean of the first 5) / rho   (stationarity check)
  BC       Sarle's bimodality coefficient of the pooled late-snapshot active counts (> 0.555 suggests two modes)
  p_on     telegraph on-fraction: a late snapshot counts as "module on" when its active count exceeds n_off + A(g)/2,
           A(g) = module on-level (mean active count of surviving h = 0 replicas, t in [50, 1000], fine_decay_raw_r1;
           below g = 0.255 scaled as (R - 1), theory section 4), n_off = the h-specific mean count over g < g_lin
           (held fixed in g: the background's own growth with g is ignored, which biases p_on UP at larger h).
           Reported only where the separation S = (A/2)/sqrt(n_off + 1) >= 3.
  n_needed the module size n for which the telegraph model (measured tau, p_est = 1 - 1/R, on-start, actual snapshot
           times) reproduces the measured p_on (0.05 < p_on < 0.95), compared with the n choices below.
  Errors: replica bootstrap (2000 resamples) for every quantity and for the peak locations.
PEAK: the maximum of F_rep (and F_time, raw chi_rep) over the grid points with 0.1 <= g <= 1.0 (real; null 0.5-5.0, the
  same span relative to its g_lin; windows chosen to hold every theory prediction below), refined by a parabola in (log g, log F) through the maximum and its two neighbours.
  "resolved" = interior maximum and P_boot(F(peak) > F(window's last point)) >= 0.95. Width = full width at half
  height above the baseline (median F below g_lin), in decades of g, interpolated in log g. g_half = where p_on
  first crosses 1/2 (interpolated in log g).
THEORY P5: the module ignites at nu_on = beta h n p_est (beta = 1, p_est = 1 - 1/R) and dies at 1/tau; replica
  variance (n*)^2 p(1-p), p = nu tau/(1 + nu tau), peaks at nu_on tau = 1 (= p_on 1/2). The variance of a TIME AVERAGE
  over a window much longer than the switching time peaks instead at nu_on tau = 2 (x^2/(1+x)^3 is maximal at x = 2);
  both are given. tau(g) variants: theory r = 1 (logtau_r1), theory r = 0 (logtau_r0, an upper bracket), measured
  tau_sim(g) at h = 0 (semi-empirical: tests the P5 mechanism separately from the lifetime theory). n variants:
  526 (antennal-lobe module, primary), K_eff(g, r=1), 1/IPR of the r = 1 mode, effective seed-set size, 1000.
  R(g) and ln tau(g) for g_lin < g < 0.1751 (below the dense grid) come from R = alpha_eff-scaled linear interpolation
  to R(g_lin) = 1 and the same exact birth-death recursion as theory_branch_gpu.py with K from the first dense row.
  Telegraph model curves (p_on, chi_rep) start "on" at t = 0 (fully active start) and are evaluated at the actual late
  snapshot times: P_on(t) = p + (1 - p) exp(-t / tau_c), tau_c = tau / (1 + nu tau).
LINEAR RESPONSE below g_lin: sim excess rho/rho0 - 1 (rho0 = q0/(q0 + p_off), the exact g = 0 discrete-time level)
  vs the exact first-order solution, the B-only (h -> 0) operator and the leading-mode pole (driven_linresp.json).
OUTPUTS: results/analysis2/driven.json, results/analysis2/driven.md, figs/pd2/driven.png
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy import stats

ROOT = "/MALECNS_ROOT"
R_ = f"{ROOT}/results"; OUT = f"{R_}/analysis2"; FIG = f"{ROOT}/figs/pd2"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
N = 167216
DT = 0.2; P_OFF = 1 - np.exp(-DT); A_EFF = P_OFF / DT
NBOOT = 2000
WIN = {"real": (0.1, 1.0), "null": (0.5, 5.0)}   # null: same span relative to its g_lin (0.829 vs 0.163)
rng = np.random.default_rng(20261109)
HS = ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]


def rho0_of(h):
    q0 = 1 - np.exp(-np.tanh(h) * DT)
    return q0 / (q0 + P_OFF)


# ---------------------------------------------------------------- theory rows
def load_rows(kind):
    rows, meta = [], None
    for p in sorted(glob.glob(f"{R_}/pd/branch_{kind}_dense_part*.json")):
        j = json.load(open(p)); rows += j["rows"]; meta = j
    rows.sort(key=lambda x: x["g"])
    return rows, meta


def log_tau(lam, K):          # identical to theory_branch_gpu.py
    n_top = int(np.floor(K))
    if n_top < 1 or lam <= 0:
        return -np.inf
    n = np.arange(1, n_top + 1, dtype=float)
    lb = lam * n * (1 - n / K); lb[-1] = 0.0
    ld = A_EFF * n
    logD = np.empty(n_top); logD[-1] = -np.log(ld[-1])
    for k in range(n_top - 2, -1, -1):
        logD[k] = np.logaddexp(0.0, np.log(lb[k]) + logD[k + 1]) - np.log(ld[k])
    return float(logsumexp(logD))


class Theory:
    def __init__(self, kind):
        self.rows, meta = load_rows(kind)
        self.kind = kind
        self.g_lin = meta["g_lin"]
        self.g = np.array([x["g"] for x in self.rows])
        self.R = np.array([x["R"] for x in self.rows])
        self.K1 = np.array([x["K_r1"] for x in self.rows]); self.K0 = np.array([x["K_r0"] for x in self.rows])
        self.lt1 = np.array([x["logtau_r1"] for x in self.rows]); self.lt0 = np.array([x["logtau_r0"] for x in self.rows])
        # recursion check against the stored values (first three rows)
        self.check = [abs(log_tau(A_EFF * self.R[i], self.K1[i]) - self.lt1[i]) for i in range(3)]
        self.slope = self.R[0] / (self.g[0] / self.g_lin)        # R / (g / g_lin) at the first dense row

    def Rf(self, g):
        g = np.atleast_1d(g).astype(float)
        out = np.interp(np.log(g), np.log(self.g), self.R)
        lo = g < self.g[0]
        # R(g_lin) = 1 exactly; linear in g in between, matched to the first dense row
        out[lo] = 1 + (g[lo] - self.g_lin) / (self.g[0] - self.g_lin) * (self.R[0] - 1)
        out[g > self.g[-1]] = np.nan
        return out

    def ln_tau(self, g, r="1"):
        g = np.atleast_1d(g).astype(float)
        lt, K = (self.lt1, self.K1) if r == "1" else (self.lt0, self.K0)
        out = np.interp(np.log(g), np.log(self.g), lt)
        lo = (g < self.g[0]) & (g > self.g_lin)
        Rl = self.Rf(g[lo])
        out[lo] = [log_tau(A_EFF * x, K[0]) for x in Rl]
        out[g <= self.g_lin] = np.nan; out[g > self.g[-1]] = np.nan
        return out

    def Keff(self, g):
        return np.interp(np.log(np.atleast_1d(g)), np.log(self.g), self.K1)


class EmpTau:
    """measured h = 0 module lifetimes (censored-exponential MLE, deaths >= 5) + the smooth quadratic fit."""
    def __init__(self, kind):
        L = json.load(open(f"{R_}/analysis2/longruns.json"))
        acc = {}
        for r in L["lifetimes"][kind]:
            if r["set"] in ("fine", "long") and r["deaths"] >= 5 and r["tau_sim"]:
                k = round(r["g"], 4)
                tt, d = acc.get(k, (0.0, 0))
                acc[k] = (tt + r["tau_sim"] * r["deaths"], d + r["deaths"])      # pooled sum(t) / deaths
        gs = np.array(sorted(acc)); self.g = gs
        self.lt = np.log([acc[k][0] / acc[k][1] for k in gs])
        sf = L["T_drift"][kind]["smooth_fit"]
        self.fit_win = sf["g_window"]; self.coef = sf["coef_lng_highest_first"]
        self.table = [{"g": float(k), "tau": float(acc[k][0] / acc[k][1]), "deaths": int(acc[k][1])} for k in gs]

    def ln_tau(self, g):
        g = np.atleast_1d(g).astype(float)
        out = np.interp(np.log(g), np.log(self.g), self.lt, left=np.nan, right=np.nan)
        hi = (g > self.g[-1]) & (g <= self.fit_win[1])
        out[hi] = np.polyval(self.coef, np.log(g[hi]))
        return out


def solve_peak(h, n_fn, lt_fn, Rf, g_lin, x_target=1.0, gmax=3.0):
    gg = np.geomspace(g_lin * 1.0005, gmax, 4000)
    R = Rf(gg); pe = 1 - 1 / R
    n = n_fn(gg)
    y = np.log(h * n * pe) + lt_fn(gg) - np.log(x_target)
    ok = np.isfinite(y)
    if not ok.any():
        return None, "undefined"
    gg, y = gg[ok], y[ok]
    if y[0] >= 0:
        return float(gg[0]), "at_lower_edge"        # nu tau >= x already at the lowest defined g
    idx = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    if len(idx) == 0:
        return None, "not_reached"
    i = idx[0]
    gl = np.exp(np.interp(0, [y[i], y[i + 1]], np.log([gg[i], gg[i + 1]])))
    return float(gl), "ok"


# ---------------------------------------------------------------- simulation statistics
def load_A(path, g_query, th):
    """module on-level (active count of surviving h = 0 replicas, t in [50, 1000])"""
    d = np.load(path); t, g, rho, s = d["t"], d["g"], d["rho"], d["surv"]
    m = (t >= 50) & (t < 1000)
    A = []
    for i in range(len(g)):
        al = s[i][m] > 0
        A.append(rho[i][m][al].mean() * N if al.sum() >= 20 else np.nan)
    A = np.array(A); ok = np.isfinite(A)
    gA, AA = g[ok], A[ok]
    out = np.interp(np.log(g_query), np.log(gA), AA)
    lo = g_query < gA[0]
    R0 = th.Rf(gA[0])[0]
    out[lo] = AA[0] * np.clip(th.Rf(g_query[lo]) - 1, 0, None) / (R0 - 1)
    return out, {"g": gA.tolist(), "A": AA.tolist(), "source": os.path.relpath(path, ROOT)}


def classify(cnt, A, n_off):
    """cnt [Kt, B] active counts; ON = count above the fixed threshold n_off + A/2 (n_off = h-specific level below g_lin;
    an iterated off-level was tried first and ratcheted upward once nearly every snapshot was ON, so it was dropped)"""
    return cnt > n_off + A / 2


def run_stats(path, th, Afn, g_lin):
    d = np.load(path); t, g, rho = d["t"], d["g"], d["rho"]
    meta = json.loads(str(d["meta"])); h = meta["h"]
    late = t >= t[-1] / 2
    tl = t[late]; B = rho.shape[2]
    r0 = rho0_of(h)
    A = Afn(g)
    # h-specific off level: mean count over g < g_lin
    below = g < g_lin
    n_off0 = float(rho[below][:, late].mean() * N) if below.any() else r0 * N
    rows = []
    boot_idx = rng.integers(0, B, size=(NBOOT, B))
    curves_b = {"F_rep": [], "F_time": [], "chi_rep": [], "p_on": []}
    for i, gg in enumerate(g):
        x = rho[i][late]                        # [Kt, B]
        ta = x.mean(0)
        rb = x.mean()
        chi_rep = N * ta.var(ddof=1); chi_time = N * x.var(0, ddof=1).mean()
        # bootstrap
        xb = x[:, boot_idx]                     # [Kt, NBOOT, B]
        tab = xb.mean(0)                        # [NBOOT, B]
        rbb = tab.mean(1)
        chib = N * tab.var(1, ddof=1)
        chitb = N * xb.var(0, ddof=1).mean(1)
        cnt = x * N
        n_off = n_off0
        on = classify(cnt, A[i], n_off)
        S = (A[i] / 2) / np.sqrt(n_off + 1)
        pon = on.mean()
        ponb = on[:, boot_idx].mean((0, 2))
        cf = cnt.ravel(); nb = len(cf)
        sk = stats.skew(cf); ku = stats.kurtosis(cf)          # excess kurtosis
        bc = (sk ** 2 + 1) / (ku + 3 * (nb - 1) ** 2 / ((nb - 2) * (nb - 3))) if cf.std() > 0 else np.nan
        drift = (x[-5:].mean() - x[:5].mean()) / rb
        driftb = (xb[-5:].mean((0, 2)) - xb[:5].mean((0, 2))) / rbb
        q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
        rows.append({"g": float(gg), "rho": float(rb), "rho_se": float(ta.std(ddof=1) / np.sqrt(B)), "rho_ci": q(rbb),
                     "chi_rep": float(chi_rep), "chi_rep_ci": q(chib), "chi_time": float(chi_time), "chi_time_ci": q(chitb),
                     "F_rep": float(chi_rep / rb), "F_rep_ci": q(chib / rbb), "F_time": float(chi_time / rb),
                     "F_time_ci": q(chitb / rbb), "drift": float(drift), "drift_ci": q(driftb),
                     "excess_rel": float(rb / r0 - 1), "excess_se": float(ta.std(ddof=1) / np.sqrt(B) / r0),
                     "A_on_level": float(A[i]), "n_off": float(n_off), "S": float(S),
                     "p_on": float(pon) if S >= 3 else None, "p_on_ci": q(ponb) if S >= 3 else None,
                     "p_on_unreliable": float(pon), "bimodality_coef": float(bc)})
        curves_b["F_rep"].append(chib / rbb); curves_b["F_time"].append(chitb / rbb); curves_b["chi_rep"].append(chib)
        curves_b["p_on"].append(np.where(S >= 3, ponb, np.nan))
    curves_b = {k: np.array(v) for k, v in curves_b.items()}       # [G, NBOOT]
    return {"path": os.path.relpath(path, ROOT), "h": h, "B": int(B), "T": float(t[-1]), "late_t": tl.tolist(),
            "rho0": float(r0), "n_off_below_glin": n_off0, "rows": rows}, curves_b, g


def find_peak(g, F, Fb, g_lin, win):
    w = np.where((g >= win[0]) & (g <= win[1]))[0]
    gw, Fw = g[w], F[w]
    k = int(np.argmax(Fw)); last = len(w) - 1
    interior = 0 < k < last

    def refine(Fv, kk):
        if 0 < kk < len(Fv) - 1 and np.all(Fv[kk - 1:kk + 2] > 0):
            xs, ys = np.log(gw[kk - 1:kk + 2]), np.log(Fv[kk - 1:kk + 2])
            a, b, c = np.polyfit(xs, ys, 2)
            if a < 0:
                return float(np.exp(np.clip(-b / (2 * a), xs[0], xs[2])))
        return float(gw[kk])
    gp = refine(Fw, k)
    Fbw = Fb[w]                                  # [Gw, NBOOT]
    kb = np.argmax(Fbw, 0)
    gpb = np.array([refine(Fbw[:, j], kb[j]) for j in range(Fbw.shape[1])])
    p_gt_last = float((Fbw[k] > Fbw[last]).mean()) if interior else 0.0
    resolved = bool(interior and p_gt_last >= 0.95)
    base = float(np.median(F[g < g_lin])) if (g < g_lin).any() else 0.0
    half = base + (Fw[k] - base) / 2
    lg = np.log10(gw)

    def cross(side):
        idx = range(k - 1, -1, -1) if side < 0 else range(k + 1, len(Fw))
        prev = k
        for j in idx:
            if Fw[j] <= half:
                return float(np.interp(half, sorted([Fw[j], Fw[prev]]),
                                       [lg[j], lg[prev]] if Fw[j] < Fw[prev] else [lg[prev], lg[j]])), True
            prev = j
        return float(lg[0] if side < 0 else lg[-1]), False
    (lo, lo_ok), (hi, hi_ok) = cross(-1), cross(+1)
    # [verification 2026-11-09] the parabola CI ignores grid discretisation: report the grid cell of the argmax (geometric
    # midpoints to its neighbours), whether the peak is a single grid point (both neighbours below half height, so the
    # FWHM is only bounded by 2 grid spacings), and the largest F above the search window (is the peak a global maximum?)
    kf = int(w[k])
    g_cell = [float(np.sqrt(g[kf - 1] * g[kf])), float(np.sqrt(g[kf] * g[kf + 1]))] if 0 < kf < len(g) - 1 else None
    single = bool(interior and Fw[k - 1] < half and Fw[k + 1] < half)
    above = g > win[1]
    max_above = [float(F[above].max()), float(g[above][np.argmax(F[above])])] if above.any() else None
    return {"g_cell": g_cell, "single_point": single, "grid_spacing_decades": float(np.log10(g[1] / g[0])),
            "max_above_window": max_above,
            "g_peak_grid": float(gw[k]), "g_peak": gp, "g_peak_ci": [float(np.quantile(gpb, .025)), float(np.quantile(gpb, .975))],
            "F_peak": float(Fw[k]), "interior": bool(interior), "P_boot_peak_gt_window_end": p_gt_last,
            "resolved": resolved, "baseline": base, "fwhm_decades": float(hi - lo),
            "fwhm_bounds_reached": [lo_ok, hi_ok], "fwhm_g": [float(10 ** lo), float(10 ** hi)],
            "frac_boot_argmax_at_grid": {f"{gw[j]:.4g}": float((kb == j).mean()) for j in np.unique(kb)}}


def half_cross(g, p, pb):
    def one(pv):
        ok = np.isfinite(pv)
        gv, pv2 = g[ok], pv[ok]
        for j in range(1, len(gv)):
            if pv2[j - 1] < 0.5 <= pv2[j]:
                return float(np.exp(np.interp(0.5, [pv2[j - 1], pv2[j]], np.log([gv[j - 1], gv[j]]))))
        return np.nan
    g0 = one(p)
    gb = np.array([one(pb[:, j]) for j in range(pb.shape[1])])
    gb = gb[np.isfinite(gb)]
    ci = [float(np.quantile(gb, .025)), float(np.quantile(gb, .975))] if len(gb) > 20 else None
    return (None if not np.isfinite(g0) else g0), ci


def telegraph(g, h, n, lt, A, tl):
    """p_on and chi_rep of a telegraph module started on at t = 0, at the late snapshot times tl"""
    tl = np.asarray(tl)
    p_mod, chi_mod = [], []
    for gg, ltt, AA, Rr in zip(g, lt, A, TH["real"].Rf(g) if TELE_KIND == "real" else TH["null"].Rf(g)):
        if not np.isfinite(ltt) or not np.isfinite(Rr) or Rr <= 1:
            p_mod.append(np.nan); chi_mod.append(np.nan); continue
        tau = np.exp(ltt); nu = h * n * (1 - 1 / Rr)
        x = nu * tau; p = x / (1 + x); tc = tau / (1 + x)
        Pt = p + (1 - p) * np.exp(-tl / tc)
        T1, T2 = np.meshgrid(tl, tl, indexing="ij")
        tlo = np.minimum(T1, T2); dtt = np.abs(T1 - T2)
        Plo = p + (1 - p) * np.exp(-tlo / tc)
        cov = Plo * (p + (1 - p) * np.exp(-dtt / tc)) - np.outer(Pt, Pt)
        p_mod.append(float(Pt.mean())); chi_mod.append(float(AA ** 2 / N * cov.mean()))
    return np.array(p_mod), np.array(chi_mod)


# ================================================================ main
TH = {"real": Theory("real"), "null": Theory("null")}
EMP = {"real": EmpTau("real"), "null": EmpTau("null")}
TELE_KIND = "real"
fp = json.load(open(f"{R_}/pd/flow_params.json"))
mode = {k: {r["r"]: r for r in json.load(open(f"{R_}/pd/mode_{k}.json"))["rows"]} for k in ("real", "null")}
seeds = json.load(open(f"{R_}/analysis2/spreading.json"))["seed_sets"]
NCH = {
    "real": [("n=526 (AL module)", lambda g: np.full_like(g, fp["n_M"], dtype=float)),
             ("n=K_eff(g, r=1)", lambda g: TH["real"].Keff(g)),
             ("n=1/IPR r=1 mode (%.0f)" % mode["real"][1.0]["inv_ipr"], lambda g: np.full_like(g, mode["real"][1.0]["inv_ipr"])),
             ("n=seed-set eff. size (%.0f)" % seeds["real"]["eff_size"], lambda g: np.full_like(g, seeds["real"]["eff_size"])),
             ("n=1000 (top-1000 set)", lambda g: np.full_like(g, 1000.0))],
    "null": [("n=K_eff(g, r=1)", lambda g: TH["null"].Keff(g)),
             ("n=seed-set eff. size (%.0f)" % seeds["null"]["eff_size"], lambda g: np.full_like(g, seeds["null"]["eff_size"])),
             ("n=1000", lambda g: np.full_like(g, 1000.0)),
             ("n=1/IPR r=1 mode (%.0f)" % mode["null"][1.0]["inv_ipr"], lambda g: np.full_like(g, mode["null"][1.0]["inv_ipr"]))],
}
TAU = {k: [("theory r=1", lambda g, k=k: TH[k].ln_tau(g, "1")), ("theory r=0", lambda g, k=k: TH[k].ln_tau(g, "0")),
           ("measured tau_sim (h=0)", lambda g, k=k: EMP[k].ln_tau(g))] for k in ("real", "null")}

res = {"definitions": __doc__, "N": N, "alpha_eff": A_EFF, "window": WIN,
       "g_lin": {k: TH[k].g_lin for k in TH}, "recursion_check_abs_diff": {k: TH[k].check for k in TH},
       "emp_tau": {k: EMP[k].table for k in EMP}, "runs": {}, "peaks": {}, "theory": {}, "qmf": {}, "linresp": {}}

# ---- theory predictions
for kind in ("real", "null"):
    hs = HS if kind == "real" else ["1e-3"]
    th = TH[kind]
    tab = []
    for hstr in hs:
        h = float(hstr)
        for tname, ltf in TAU[kind]:
            for nname, nf in NCH[kind]:
                for xt in (1.0, 2.0):
                    gp, st = solve_peak(h, nf, ltf, th.Rf, th.g_lin, xt)
                    tab.append({"h": h, "tau": tname, "n": nname, "x_target": xt, "g_peak": gp, "status": st})
    res["theory"][kind] = tab

# ---- simulations
datasets = []
for hstr in HS:
    datasets.append(("stage1", "real", hstr, f"{R_}/pd/drive_h{hstr}_real.npz"))
    datasets.append(("pd2", "real", hstr, f"{R_}/pd2/drive_h{hstr}_real.npz"))
datasets += [("stage1", "null", "1e-3", f"{R_}/pd/drive_h1e-3_null.npz"), ("pd2", "null", "1e-3", f"{R_}/pd2/drive_h1e-3_null.npz")]
qmf = np.load(f"{R_}/pd/qmf_drive_real.npz")
A_info = {}
for kind in ("real", "null"):
    src = f"{R_}/fine_decay_raw_r1.npz" if kind == "real" else f"{R_}/fine_decay_raw_r1_null.npz"
    _, A_info[kind] = load_A(src, np.array([0.3]), TH[kind])
res["module_on_level_h0"] = A_info
curves = {}
for ds, kind, hstr, path in datasets:
    if not os.path.exists(path):
        res["runs"][f"{ds}_{kind}_h{hstr}"] = {"path": os.path.relpath(path, ROOT), "status": "missing (not finished)"}
        continue
    th = TH[kind]
    src = f"{R_}/fine_decay_raw_r1.npz" if kind == "real" else f"{R_}/fine_decay_raw_r1_null.npz"
    Afn = lambda gq, src=src, th=th: load_A(src, np.asarray(gq, float), th)[0]
    st, cb, g = run_stats(path, th, Afn, th.g_lin)
    key = f"{ds}_{kind}_h{hstr}"
    rows = st["rows"]
    F = np.array([r["F_rep"] for r in rows]); Ft = np.array([r["F_time"] for r in rows]); C = np.array([r["chi_rep"] for r in rows])
    pk = {"F_rep": find_peak(g, F, cb["F_rep"], th.g_lin, WIN[kind]), "F_time": find_peak(g, Ft, cb["F_time"], th.g_lin, WIN[kind]),
          "chi_rep_raw": find_peak(g, C, cb["chi_rep"], th.g_lin, WIN[kind])}
    pon = np.array([np.nan if r["p_on"] is None else r["p_on"] for r in rows])
    gh, ghci = half_cross(g, pon, cb["p_on"])
    pk["p_on_half"] = {"g_half": gh, "g_half_ci": ghci}
    # global maximum of raw chi_rep over the whole grid (transparency)
    pk["chi_rep_global_max_g"] = float(g[np.argmax(C)])
    # telegraph model (n = 526 / seed-set size for null) for the figure
    TELE_KIND = kind
    n_tel = fp["n_M"] if kind == "real" else seeds["null"]["eff_size"]
    Ag = Afn(g)
    tele = {}
    for tname, ltf in TAU[kind]:
        if tname == "theory r=0":
            continue
        pm, cm = telegraph(g, st["h"], n_tel, ltf(g), Ag, st["late_t"])
        tele[tname] = {"p_on": pm.tolist(), "chi_rep_module": cm.tolist(), "n": n_tel}
    st["telegraph_model"] = tele
    # how big is the predicted module (telegraph) contribution compared with the measured chi_rep, inside the window?
    wmask = (g >= WIN[kind][0]) & (g <= WIN[kind][1])
    amp = {}
    for tname, tv in tele.items():
        cm = np.array(tv["chi_rep_module"], dtype=float)
        okm = wmask & np.isfinite(cm)
        if okm.any():
            j = int(np.argmax(np.where(okm, cm, -np.inf)))
            amp[tname] = {"g_at_model_max": float(g[j]), "model_chi_max": float(cm[j]), "sim_chi_rep_there": float(C[j]),
                          "model_over_sim": float(cm[j] / C[j]),
                          "sim_chi_rep_median_below_glin": float(np.median(C[g < th.g_lin])) if (g < th.g_lin).any() else None}
    st["telegraph_amplitude"] = amp
    # module size needed by the telegraph model (measured tau) to reproduce the measured on-fraction
    ltm = dict(TAU[kind])["measured tau_sim (h=0)"](g)

    def n_need(i, ptarget):
        f = lambda ln: telegraph(g[i:i + 1], st["h"], np.exp(ln), ltm[i:i + 1], Ag[i:i + 1], st["late_t"])[0][0] - ptarget
        lo, hi = 0.0, np.log(1e7)
        if not (f(lo) < 0 < f(hi)):
            return None
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return float(np.exp((lo + hi) / 2))
    nn = []
    for i, r in enumerate(rows):
        if r["p_on"] is None or not (0.05 < r["p_on"] < 0.95) or r["g"] > 1.0 or not np.isfinite(ltm[i]):
            continue
        ci = r["p_on_ci"]
        nn.append({"g": r["g"], "p_on": r["p_on"], "tau_measured": float(np.exp(ltm[i])), "p_est": float(1 - 1 / TH[kind].Rf(g[i])[0]),
                   "n_needed": n_need(i, r["p_on"]),
                   "n_needed_range_from_p_on_ci": [n_need(i, max(ci[0], 1e-3)), n_need(i, min(ci[1], 0.999))]})
    st["n_needed"] = nn
    # QMF on the same grid (stage-1 grid identical; else log-log interpolation)
    if kind == "real":
        hi = int(np.argmin(np.abs(qmf["h"] - st["h"])))
        late = qmf["t"] >= qmf["t"][-1] / 2
        qr = qmf["rho"][0, hi][:, late].mean(1)
        qg = np.interp(np.log(g), np.log(qmf["g"]), np.log(np.maximum(qr, 1e-300)))
        for r, v in zip(rows, np.exp(qg)):
            r["rho_qmf"] = float(v); r["sim_over_qmf"] = float(r["rho"] / v)
    st["peaks"] = pk
    res["runs"][key] = st
    curves[key] = (g, st)

# ---- QMF summary
for key, (g, st) in curves.items():
    if "_real_" not in key:
        continue
    rr = st["rows"]; gl = TH["real"].g_lin
    seg = lambda lo, hi: [r["sim_over_qmf"] for r in rr if lo <= r["g"] < hi]
    res["qmf"][key] = {"median_sim_over_qmf": {"g<g_lin": float(np.median(seg(0, gl))), "g_lin..0.5": float(np.median(seg(gl, 0.5))),
                                               "0.5..3": float(np.median(seg(0.5, 3.01))), "3..30": float(np.median(seg(3.01, 31))) if seg(3.01, 31) else None},
                       "max_ratio": float(max(r["sim_over_qmf"] for r in rr)), "min_ratio": float(min(r["sim_over_qmf"] for r in rr)),
                       "g_at_min_ratio": float(rr[int(np.argmin([r["sim_over_qmf"] for r in rr]))]["g"])}
res["qmf"]["g_QMF_r1"] = mode["real"][1.0]["g_QMF"]

# ---- linear response
LR = None
if os.path.exists(f"{OUT}/driven_linresp.json"):
    LR = json.load(open(f"{OUT}/driven_linresp.json"))
    res["linresp"]["pole"] = LR["pole"]
    comp = []
    for key, (g, st) in curves.items():
        if "_real_" not in key or st["h"] > 0.011:
            continue
        for r in st["rows"]:
            if r["g"] >= TH["real"].g_lin:
                continue
            ex = [x for x in LR["rows"] if x["variant"] == "exact" and abs(x["h"] - st["h"]) < 1e-12 and abs(x["g"] - r["g"]) < 5e-4]
            bo = [x for x in LR["rows"] if x["variant"] == "B_only" and abs(x["g"] - r["g"]) < 5e-4]
            xg = r["g"] / TH["real"].g_lin
            comp.append({"run": key, "h": st["h"], "g": r["g"], "sim_excess": r["excess_rel"], "sim_se": r["excess_se"],
                         "exact": ex[0]["excess_rel"] if ex else None, "B_only": bo[0]["excess_rel"] if bo else None,
                         "pole_only": LR["pole"]["amp_sumv_sumu_over_N"] * xg / (1 - xg),
                         "z_exact": (r["excess_rel"] - ex[0]["excess_rel"]) / r["excess_se"] if ex else None,
                         "z_B_only": (r["excess_rel"] - bo[0]["excess_rel"]) / r["excess_se"] if bo else None})
    res["linresp"]["comparison"] = comp
    for hh in sorted({c["h"] for c in comp}):
        cs = [c for c in comp if c["h"] == hh and c["exact"] is not None]
        if cs:
            res["linresp"][f"chi2_h{hh:g}"] = {"n": len(cs), "chi2_exact": float(sum(c["z_exact"] ** 2 for c in cs)),
                                               "chi2_B_only": float(sum(c["z_B_only"] ** 2 for c in cs)),
                                               "chi2_zero_excess": float(sum((c["sim_excess"] / c["sim_se"]) ** 2 for c in cs))}
else:
    res["linresp"]["status"] = "results/analysis2/driven_linresp.json missing"

# ---- headline comparison table: per h, sim peaks vs theory
def th_pick(kind, h, tau, n, xt=1.0):
    for t in res["theory"][kind]:
        if abs(t["h"] - h) < 1e-12 and t["tau"] == tau and t["n"] == n and t["x_target"] == xt:
            return t["g_peak"], t["status"]
    return None, "missing"


head = []
for kind, hs in (("real", HS), ("null", ["1e-3"])):
    nprim = NCH[kind][0][0] if kind == "real" else NCH[kind][1][0]
    for hstr in hs:
        h = float(hstr)
        best = None
        for ds in ("pd2", "stage1"):
            k = f"{ds}_{kind}_h{hstr}"
            if k in curves:
                best = k; break
        if best is None:
            continue
        pk = res["runs"][best]["peaks"]
        row = {"kind": kind, "h": h, "run": best,
               "sim_F_rep_peak": pk["F_rep"]["g_peak"], "sim_F_rep_peak_ci": pk["F_rep"]["g_peak_ci"],
               "sim_F_rep_resolved": pk["F_rep"]["resolved"], "sim_F_rep_fwhm_decades": pk["F_rep"]["fwhm_decades"],
               "sim_F_rep_grid_cell": pk["F_rep"]["g_cell"], "sim_F_rep_single_point": pk["F_rep"]["single_point"],
               "sim_F_rep_peak_value": pk["F_rep"]["F_peak"], "sim_F_rep_max_above_window": pk["F_rep"]["max_above_window"],
               "grid_spacing_decades": pk["F_rep"]["grid_spacing_decades"],
               "sim_F_time_peak": pk["F_time"]["g_peak"], "sim_F_time_resolved": pk["F_time"]["resolved"],
               "sim_chi_rep_raw_peak": pk["chi_rep_raw"]["g_peak"], "sim_chi_rep_raw_resolved": pk["chi_rep_raw"]["resolved"],
               "sim_p_on_half": pk["p_on_half"]["g_half"], "sim_p_on_half_ci": pk["p_on_half"]["g_half_ci"]}
        for tname, _ in TAU[kind]:
            for xt in (1.0, 2.0):
                gp, stt = th_pick(kind, h, tname, nprim, xt)
                row[f"theory[{tname}; {nprim}; nu*tau={xt:g}]"] = gp if gp is not None else stt
        vals = [th_pick(kind, h, "measured tau_sim (h=0)", nn, 1.0)[0] for nn, _ in NCH[kind]]
        vals = [v for v in vals if v is not None]
        row["theory[measured tau; n range]"] = [min(vals), max(vals)] if vals else None
        vals = [th_pick(kind, h, "theory r=1", nn, 1.0)[0] for nn, _ in NCH[kind]]
        vals = [v for v in vals if v is not None]
        row["theory[theory r=1; n range]"] = [min(vals), max(vals)] if vals else None
        head.append(row)
res["headline"] = head
json.dump(res, open(f"{OUT}/driven.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else str(o))
print("wrote", f"{OUT}/driven.json")

# ================================================================ figure
H_COL = {"1e-5": "#86b6ef", "1e-4": "#5598e7", "1e-3": "#2a78d6", "1e-2": "#1c5cab", "1e-1": "#0d366b"}   # ordinal blue
H_MK = {"1e-5": "o", "1e-4": "s", "1e-3": "^", "1e-2": "D", "1e-1": "v"}
NULL_C = "#eb6834"; AQUA = "#1baf7a"; ORANGE = "#eb6834"; VIOLET = "#6250d6"; GREY = "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140,
                     "axes.titlesize": 10, "legend.fontsize": 7.5})
fig, axs = plt.subplots(3, 3, figsize=(16, 13.5))
gl = TH["real"].g_lin


def pick(kind, hstr):
    for ds in ("pd2", "stage1"):
        k = f"{ds}_{kind}_h{hstr}"
        if k in curves:
            return k
    return None


def vline(ax):
    ax.axvline(gl, color=GREY, ls=":", lw=1)


# (a) density
ax = axs[0, 0]
for hstr in HS:
    k = pick("real", hstr)
    if not k:
        continue
    g, st = curves[k]; rr = st["rows"]
    ax.loglog(g, [r["rho"] for r in rr], H_MK[hstr], ms=4, color=H_COL[hstr], mec="white", mew=0.4, label=f"sim h={hstr}")
    ax.loglog(g, [r["rho_qmf"] for r in rr], "-", lw=1.4, color=H_COL[hstr])
k = pick("null", "1e-3")
if k:
    g, st = curves[k]
    ax.loglog(g, [r["rho"] for r in st["rows"]], "x", ms=4, color=NULL_C, label="config. null h=1e-3")
ax.plot([], [], "-", color=GREY, label="QMF (lines)")
vline(ax); ax.set_xlabel("coupling g"); ax.set_ylabel("stationary fraction active ρ")
ax.set_title("(a) density: simulation (markers) vs QMF (lines)"); ax.legend(ncol=2)

# (b) Fano replica fluctuation
ax = axs[0, 1]
for hstr in HS:
    k = pick("real", hstr)
    if not k:
        continue
    g, st = curves[k]; rr = st["rows"]
    F = np.array([r["F_rep"] for r in rr]); ci = np.array([r["F_rep_ci"] for r in rr])
    ax.fill_between(g, ci[:, 0], ci[:, 1], color=H_COL[hstr], alpha=0.15, lw=0)
    ax.loglog(g, F, "-" + H_MK[hstr], ms=3.5, lw=1, color=H_COL[hstr], mec="white", mew=0.4, label=f"h={hstr}")
    pk = st["peaks"]["F_rep"]
    if pk["resolved"]:
        ax.plot(pk["g_peak"], pk["F_peak"] * 1.6, "v", ms=8, color=H_COL[hstr], mec="black", mew=0.6)
k = pick("null", "1e-3")
if k:
    g, st = curves[k]
    ax.loglog(g, [r["F_rep"] for r in st["rows"]], "--x", ms=3.5, lw=1, color=NULL_C, label="null h=1e-3")
ax.axvspan(*WIN["real"], color="#cfcfca", alpha=0.25, lw=0)
vline(ax); ax.set_xlabel("coupling g"); ax.set_ylabel("χ_rep / ρ  (= var/mean of active count)")
ax.set_title("(b) χ_rep/ρ (▼ = peak resolved in shaded window; band = 95% CI)"); ax.legend(ncol=2)

# (c) raw replica fluctuation + telegraph model
ax = axs[0, 2]
for hstr in HS:
    k = pick("real", hstr)
    if not k:
        continue
    g, st = curves[k]; rr = st["rows"]
    ax.loglog(g, [r["chi_rep"] for r in rr], "-" + H_MK[hstr], ms=3.5, lw=1, color=H_COL[hstr], mec="white", mew=0.4, label=f"h={hstr}")
    if hstr in ("1e-5", "1e-4", "1e-3"):
        cm = np.array(st["telegraph_model"]["measured tau_sim (h=0)"]["chi_rep_module"], dtype=float)
        ax.loglog(g, np.where(cm > 0, cm, np.nan), ":", lw=2, color=H_COL[hstr])
k = pick("null", "1e-3")
if k:
    g, st = curves[k]
    ax.loglog(g, [r["chi_rep"] for r in st["rows"]], "--x", ms=3.5, lw=1, color=NULL_C, label="null h=1e-3")
ax.plot([], [], ":", lw=2, color=GREY, label="telegraph model, measured τ, n=526")
vline(ax); ax.set_xlabel("coupling g"); ax.set_ylabel("χ_rep = N·var_replicas(time-avg ρ)")
ax.set_ylim(1e-7, 1); ax.set_title("(c) replica fluctuation, raw (dotted: telegraph-model module part)")
ax.legend(ncol=2, loc="lower right")

# (d) temporal fluctuation
ax = axs[1, 0]
for hstr in HS:
    k = pick("real", hstr)
    if not k:
        continue
    g, st = curves[k]; rr = st["rows"]
    ax.loglog(g, [r["chi_time"] for r in rr], "-" + H_MK[hstr], ms=3.5, lw=1, color=H_COL[hstr], mec="white", mew=0.4, label=f"h={hstr}")
k = pick("null", "1e-3")
if k:
    g, st = curves[k]
    ax.loglog(g, [r["chi_time"] for r in st["rows"]], "--x", ms=3.5, lw=1, color=NULL_C, label="null h=1e-3")
vline(ax); ax.set_xlabel("coupling g"); ax.set_ylabel("χ_time = N·mean_replicas var_t(ρ)")
ax.set_title("(d) temporal fluctuation (late half, 10 snapshots)"); ax.legend(ncol=2)

# (e) on-fraction vs telegraph model
ax = axs[1, 1]
for hstr in HS[:3]:
    k = pick("real", hstr)
    if not k:
        continue
    g, st = curves[k]; rr = st["rows"]
    ok = [i for i, r in enumerate(rr) if r["p_on"] is not None and r["g"] <= 1.0]
    if not ok:
        continue
    gg = g[ok]; pp = np.array([rr[i]["p_on"] for i in ok]); ci = np.array([rr[i]["p_on_ci"] for i in ok])
    ax.errorbar(gg, pp, yerr=[pp - ci[:, 0], ci[:, 1] - pp], fmt=H_MK[hstr], ms=5, color=H_COL[hstr], mec="black", mew=0.4,
                capsize=2, label=f"sim h={hstr}")
    if hstr in ("1e-5", "1e-4", "1e-3"):
        gm = g[g <= 1.0]
        for tname, ls in (("measured tau_sim (h=0)", "-"), ("theory r=1", "--")):
            pm = np.array(st["telegraph_model"][tname]["p_on"], dtype=float)[g <= 1.0]
            ax.plot(gm, pm, ls, lw=1.8, color=H_COL[hstr])
ax.plot([], [], "-", color=GREY, label="telegraph, measured τ(g)"); ax.plot([], [], "--", color=GREY, label="telegraph, theory τ (r=1)")
ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
vline(ax); ax.set_xscale("log"); ax.set_xlim(0.09, 1.05); ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("coupling g"); ax.set_ylabel("fraction of late snapshots with module ON")
ax.set_title("(e) module on-fraction, h ≤ 1e-3 (model n=526; ν_on·τ = 1 ⇔ p = ½)"); ax.legend(loc="center right")

# (f) g_peak vs h
ax = axs[1, 2]
hv = np.array([float(x) for x in HS])
hh = np.geomspace(1e-5, 1e-1, 60)
for tname, col, ls in (("theory r=1", ORANGE, "-"), ("theory r=0", ORANGE, ":"), ("measured tau_sim (h=0)", AQUA, "-")):
    for nname, nf in NCH["real"][:1]:
        ys = [solve_peak(x, nf, dict(TAU["real"])[tname], TH["real"].Rf, gl, 1.0)[0] for x in hh]
        ax.plot(hh, [np.nan if y is None else y for y in ys], ls, lw=1.8, color=col, label=f"P5: {tname}, n=526")
    lo, hi = [], []
    for x in hh:
        v = [solve_peak(x, nf, dict(TAU["real"])[tname], TH["real"].Rf, gl, 1.0)[0] for _, nf in NCH["real"]]
        v = [y for y in v if y is not None]
        lo.append(min(v) if v else np.nan); hi.append(max(v) if v else np.nan)
    if tname != "theory r=0":
        ax.fill_between(hh, lo, hi, color=col, alpha=0.13, lw=0)
for row in head:
    if row["kind"] != "real":
        continue
    h = row["h"]
    if row["sim_F_rep_resolved"]:
        ci = list(row["sim_F_rep_peak_ci"])
        if row["sim_F_rep_grid_cell"]:            # [verification] bar = bootstrap CI widened to the argmax's grid cell
            ci = [min(ci[0], row["sim_F_rep_grid_cell"][0]), max(ci[1], row["sim_F_rep_grid_cell"][1])]
        ax.errorbar(h, row["sim_F_rep_peak"], yerr=[[row["sim_F_rep_peak"] - ci[0]], [ci[1] - row["sim_F_rep_peak"]]],
                    fmt="o", ms=7, color="black", capsize=3)
    else:
        ax.plot(h, 0.125, "o", ms=7, mfc="white", mec="black")
    if row["sim_p_on_half"]:
        ci = row["sim_p_on_half_ci"] or [row["sim_p_on_half"]] * 2
        ax.errorbar(h * 1.15, row["sim_p_on_half"], yerr=[[row["sim_p_on_half"] - ci[0]], [ci[1] - row["sim_p_on_half"]]],
                    fmt="s", ms=6, color=VIOLET, capsize=3)
ax.plot([], [], "o", color="black", label="sim: peak of χ_rep/ρ in 0.1–1 (bar: grid cell ∪ boot. CI)")
ax.plot([], [], "o", mfc="white", mec="black", label="sim: no resolved peak (placeholder at 0.125)")
ax.plot([], [], "s", color=VIOLET, label="sim: on-fraction = ½")
ax.axhline(gl, color=GREY, ls=":", lw=1); ax.text(1.2e-5, gl * 1.03, "g_lin", color=GREY, fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.11, 0.8)
ax.set_xlabel("drive h"); ax.set_ylabel("g of the fluctuation peak")
ax.set_title("(f) real graph: peak g vs drive h (bands: n = K_eff…1000)"); ax.legend(loc="upper right", fontsize=7)

# (g) linear response below g_lin
ax = axs[2, 0]
if LR:
    for hstr in HS[:4]:
        k = pick("real", hstr)
        if not k:
            continue
        g, st = curves[k]; rr = [r for r in st["rows"] if r["g"] < gl]
        ax.errorbar([r["g"] / gl for r in rr], [r["excess_rel"] for r in rr], yerr=[2 * r["excess_se"] for r in rr],
                    fmt=H_MK[hstr], ms=5, color=H_COL[hstr], mec="black", mew=0.4, capsize=2, label=f"sim h={hstr} (±2 s.e.)")
        ex = sorted([x for x in LR["rows"] if x["variant"] == "exact" and abs(x["h"] - float(hstr)) < 1e-12], key=lambda x: x["g"])
        ax.plot([x["g"] / gl for x in ex], [x["excess_rel"] for x in ex], "-", lw=1.4, color=H_COL[hstr])
    bo = sorted([x for x in LR["rows"] if x["variant"] == "B_only"], key=lambda x: x["g"])
    ax.plot([x["g"] / gl for x in bo], [x["excess_rel"] for x in bo], "--", lw=1.6, color="black", label="B only (h→0), no inhibition")
    xs = np.linspace(0.45, 0.99, 100)
    ax.plot(xs, LR["pole"]["amp_sumv_sumu_over_N"] * xs / (1 - xs), ":", lw=1.6, color=VIOLET,
            label="leading-mode pole ∝ 1/(α−λ(g))")
    ax.plot([], [], "-", color=GREY, label="exact 1st-order response (per h)")
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xlabel("g / g_lin"); ax.set_ylabel("ρ/ρ₀ − 1  (excess over spontaneous level)")
ax.set_title("(g) linear response below threshold"); ax.legend(fontsize=7)

# (h) sim / QMF
ax = axs[2, 1]
for hstr in HS:
    k = pick("real", hstr)
    if not k:
        continue
    g, st = curves[k]; rr = st["rows"]
    ax.semilogx(g, [r["sim_over_qmf"] for r in rr], "-" + H_MK[hstr], ms=3.5, lw=1, color=H_COL[hstr], mec="white", mew=0.4, label=f"h={hstr}")
ax.axhline(1, color=GREY, lw=0.8); vline(ax)
ax.axvline(res["qmf"]["g_QMF_r1"], color=ORANGE, ls="--", lw=1); ax.text(res["qmf"]["g_QMF_r1"] * 1.04, 2.6, "g_QMF(r=1)", color=ORANGE, fontsize=8)
ax.set_yscale("log"); ax.set_xlabel("coupling g"); ax.set_ylabel("ρ_sim / ρ_QMF")
ax.set_title("(h) simulation over quenched mean field"); ax.legend(ncol=2)

# (i) null vs real at h = 1e-3, normalised fluctuation vs g/g_lin
ax = axs[2, 2]
for kind, col, mk in (("real", H_COL["1e-3"], "^"), ("null", NULL_C, "x")):
    k = pick(kind, "1e-3")
    if not k:
        continue
    g, st = curves[k]; rr = st["rows"]
    ax.loglog(g / TH[kind].g_lin, [r["F_rep"] for r in rr], "-" + mk, ms=4, lw=1, color=col, label=f"{kind} h=1e-3: χ_rep/ρ")
    ax.loglog(g / TH[kind].g_lin, [r["F_time"] for r in rr], "--", lw=1, color=col, label=f"{kind} h=1e-3: χ_time/ρ")
ax.axvline(1, color=GREY, ls=":", lw=1)
for kind, col, nm in (("real", H_COL["1e-3"], "n=526 (AL module)"), ("null", NULL_C, NCH["null"][1][0])):
    gp, _ = th_pick(kind, 1e-3, "measured tau_sim (h=0)", nm, 1.0)
    if gp:
        ax.axvline(gp / TH[kind].g_lin, color=col, ls="-.", lw=1.2)
ax.plot([], [], "-.", color=GREY, lw=1.2, label="P5 prediction (measured τ), per graph")
ax.set_xlabel("g / g_lin (real 0.163, null 0.829)"); ax.set_ylabel("fluctuation / ρ")
ax.set_title("(i) real vs configuration null at h = 1e-3"); ax.legend(loc="lower right")

src = ", ".join(sorted({curves[k][1]["path"].split("/")[1] for k in curves}))
fig.suptitle(f"Driven plane, r = 1: simulation vs theory P5 (fully active start, T = 1000, late half; data dirs: {src})", y=0.995)
fig.tight_layout()
fig.savefig(f"{FIG}/driven.png"); plt.close(fig)
print("wrote", f"{FIG}/driven.png")

# ================================================================ markdown report (every number from res)
def f3(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "–"
    return f"{x:.{nd}g}"


def ci(c, nd=3):
    return "–" if not c else f"[{f3(c[0], nd)}, {f3(c[1], nd)}]"


L = []
L.append("# Driven plane (r = 1): simulation vs theory P5\n")
L.append("Generated by `code/analysis2_driven.py` (linear response: `code/analysis2_driven_linresp.py`). Every number below is "
         "read from or computed from the files listed in the script docstring; full per-g rows are in `results/analysis2/driven.json`, "
         "figure `figs/pd2/driven.png`.\n")
L.append("## Data used\n")
for k, v in res["runs"].items():
    if "rows" in v:
        L.append(f"- `{v['path']}`: h = {v['h']:g}, B = {v['B']}, T = {v['T']:g}, {len(v['rows'])} g values, late snapshots "
                 f"t = {v['late_t'][0]:g}–{v['late_t'][-1]:g} ({len(v['late_t'])}); rho0 (g = 0 level) = {v['rho0']:.4g}")
    else:
        L.append(f"- `{v['path']}`: {v['status']}")
L.append(f"\ng_lin = {TH['real'].g_lin:.4f} (real), {TH['null'].g_lin:.4f} (configuration null); QMF onset g_QMF(r=1) = "
         f"{res['qmf']['g_QMF_r1']:.4f}. Recursion re-implementation reproduces the stored logtau to "
         f"{max(res['recursion_check_abs_diff']['real'] + res['recursion_check_abs_diff']['null']):.1e}.\n")

def nonover(a, b):
    return a and b and (a[1] < b[0] or b[1] < a[0])


# ---- summary (templated; every number from res / head)
def inside(x, c):
    return c is not None and x is not None and c[0] <= x <= c[1]
Lr = json.load(open(f"{R_}/analysis2/longruns.json"))["theory_vs_sim"]["real"]
L.append("## Summary\n")
for row in head:
    if row["kind"] != "real":
        continue
    pm1 = row["theory[measured tau_sim (h=0); n=526 (AL module); nu*tau=1]"]
    pm2 = row["theory[measured tau_sim (h=0); n=526 (AL module); nu*tau=2]"]
    pt1 = row["theory[theory r=1; n=526 (AL module); nu*tau=1]"]
    if row["sim_F_rep_resolved"]:
        cell = row["sim_F_rep_grid_cell"]; dg = row["grid_spacing_decades"]
        wtxt = (f"a single grid point (bootstrap CI {ci(row['sim_F_rep_peak_ci'])} is the 3-point parabola only; the grid cell is "
                f"{ci(cell)}, and the FWHM is unresolved, < {2 * dg:.2f} decades = 2 grid spacings)") if row["sim_F_rep_single_point"] \
            else f"{ci(row['sim_F_rep_peak_ci'])}, grid cell {ci(cell)}, FWHM {row['sim_F_rep_fwhm_decades']:.2f} decades"
        ma = row["sim_F_rep_max_above_window"]
        gtxt = (f" This is a LOCAL maximum only: χ_rep/ρ is larger above the search window ({ma[0]:.3g} at g = {ma[1]:.3g} vs "
                f"{row['sim_F_rep_peak_value']:.3g} at the peak)." if ma and ma[0] > row["sim_F_rep_peak_value"] else
                (f" It is the global maximum (largest value above the window {ma[0]:.3g} at g = {ma[1]:.3g})." if ma else ""))
        L.append(f"- h = {row['h']:g}: simulated χ_rep/ρ peak (value {row['sim_F_rep_peak_value']:.3g}) at g = {row['sim_F_rep_peak']:.3f}, {wtxt}.{gtxt} "
                 f"Module on-fraction ½ at g = {f3(row['sim_p_on_half'])} "
                 f"{ci(row['sim_p_on_half_ci'])} (interpolated between grid points). P5 with the MEASURED h = 0 lifetime and n = 526: {f3(pm1)} (ν·τ = 1; "
                 f"{'inside' if inside(pm1, cell) else 'outside'} the peak's grid cell, "
                 f"{'inside' if inside(pm1, row['sim_F_rep_peak_ci']) else 'outside'} the peak's parabola CI, "
                 f"{'inside' if inside(pm1, row['sim_p_on_half_ci']) else 'outside'} the on-fraction CI), {f3(pm2)} (ν·τ = 2). "
                 f"P5 with the THEORY lifetime (r = 1): {f3(pt1)}, i.e. {f3(pt1 / row['sim_F_rep_peak'], 3)}× the simulated peak.")
    else:
        L.append(f"- h = {row['h']:g}: NO resolved fluctuation peak in 0.1 ≤ g ≤ 1 (χ_rep/ρ maximum at the window edge or not "
                 f"significantly above it); P5 predicts {f3(pm1)} (measured τ) / {f3(pt1)} (theory τ), close to g_lin = {gl:.3f}.")
r1 = [r for r in head if r["kind"] == "real"][0]
hit = inside(r1["theory[measured tau_sim (h=0); n=526 (AL module); nu*tau=1]"], r1["sim_F_rep_peak_ci"])
_lt = json.load(open(f"{R_}/analysis2/longruns.json"))["lifetimes"]["real"]
_r30 = [x["tau_sim"] / x["tau_theory"] for x in _lt if abs(x["g"] - 0.3) < 1e-6 and x["tau_sim"] and x.get("tau_theory")]
hitc = inside(r1["theory[measured tau_sim (h=0); n=526 (AL module); nu*tau=1]"], r1.get("sim_F_rep_grid_cell"))
L.append(f"- The theory-lifetime column misses because the extinction theory underestimates the h = 0 module lifetime at r = 1. "
         f"The ratio τ_sim/τ_theory depends steeply on g: {min(_r30):.1f}–{max(_r30):.1f} at g = 0.30 (where the h = 1e-5 peak sits), "
         f"geometric mean {Lr['ratio_geomean']:.1f} (range {Lr['ratio_min']:.1f}–{Lr['ratio_max']:.0f}) over g = "
         f"{min(Lr['g']):.3g}–{max(Lr['g']):.3g} (τ_sim ≥ 200 only; results/analysis2/longruns.json). With the measured "
         f"lifetime the ignition/extinction balance {'lands inside' if hitc else 'misses'} the h = 1e-5 peak's grid cell "
         f"({'inside' if hit else 'outside'} its parabola CI).")
a, b = [r for r in head if r["kind"] == "real"][:2]
pa = a["theory[measured tau_sim (h=0); n=526 (AL module); nu*tau=1]"]; pb = b["theory[measured tau_sim (h=0); n=526 (AL module); nu*tau=1]"]
ta = a["theory[theory r=1; n=526 (AL module); nu*tau=1]"]; tb = b["theory[theory r=1; n=526 (AL module); nu*tau=1]"]
if a["sim_p_on_half"] and b["sim_p_on_half"]:
    L.append(f"- Downward shift: between h = 1e-5 and 1e-4 the on-fraction midpoint moves {a['sim_p_on_half']:.3f} → {b['sim_p_on_half']:.3f} "
             f"(Δln g = {np.log(b['sim_p_on_half'] / a['sim_p_on_half']):+.3f}; CIs {'do not overlap' if nonover(a['sim_p_on_half_ci'], b['sim_p_on_half_ci']) else 'overlap'}) "
             f"and the χ_rep/ρ peak {a['sim_F_rep_peak']:.3f} → {b['sim_F_rep_peak']:.3f} (CIs {'do not overlap' if nonover(a['sim_F_rep_peak_ci'], b['sim_F_rep_peak_ci']) else 'overlap'}). "
             f"Predicted Δln g: {np.log(pb / pa):+.3f} (measured τ, n = 526), {np.log(tb / ta):+.3f} (theory τ). "
             + ("So a downward shift is seen in the on-fraction midpoint" if nonover(a['sim_p_on_half_ci'], b['sim_p_on_half_ci'])
                else "So no significant shift is seen")
             + (", smaller than predicted" if abs(np.log(b['sim_p_on_half'] / a['sim_p_on_half'])) < abs(np.log(pb / pa)) else ", at least as large as predicted")
             + "; for h ≥ 1e-3 there is no peak to follow.")
amp3 = res["runs"][[r for r in head if r["kind"] == "real" and abs(r["h"] - 1e-3) < 1e-12][0]["run"]].get("telegraph_amplitude", {})
def bc_range(hh, lo, hi):
    k = [r["run"] for r in head if r["kind"] == "real" and abs(r["h"] - hh) < 1e-12][0]
    v = [x["bimodality_coef"] for x in res["runs"][k]["rows"] if lo < x["g"] <= hi and np.isfinite(x["bimodality_coef"])]
    return (min(v), max(v)) if v else (np.nan, np.nan)
if amp3:
    b3 = bc_range(1e-3, gl, 0.5); b5 = bc_range(1e-5, gl, 0.5); b4 = bc_range(1e-4, gl, 0.5)
    L.append("- Why no peak at h ≥ 1e-3: the telegraph model's module contribution is at most "
             + ", ".join(f"{a_['model_over_sim']:.2g}× ({t})" for t, a_ in amp3.items())
             + " of the measured χ_rep at h = 1e-3: in the model the module is ON (p near 1) wherever its on-level A(g) is large and "
               f"switches only close to g_lin, where A(g) → 0. Bimodality coefficient of the late counts for g_lin < g ≤ 0.5: h = 1e-3 {b3[0]:.2f}–{b3[1]:.2f}, "
               f"h = 1e-4 {b4[0]:.2f}–{b4[1]:.2f}, h = 1e-5 {b5[0]:.2f}–{b5[1]:.2f} (0.555 = uniform-distribution reference; "
               "higher suggests two modes, i.e. on/off switching).")
nl = [r for r in head if r["kind"] == "null"]
if nl:
    r_ = nl[0]
    L.append(f"- Configuration null, h = 1e-3: χ_rep/ρ maximum at g = {f3(r_['sim_F_rep_peak'])} {ci(r_['sim_F_rep_peak_ci'])} "
             f"({'resolved' if r_['sim_F_rep_resolved'] else 'not resolved'}), χ_time/ρ peak at {f3(r_['sim_F_time_peak'])} "
             f"({'resolved' if r_['sim_F_time_resolved'] else 'not resolved'}); P5 predicts "
             f"{f3(r_['theory[measured tau_sim (h=0); ' + NCH['null'][1][0] + '; nu*tau=1]'])} (measured τ) to "
             f"{f3(r_['theory[theory r=1; ' + NCH['null'][1][0] + '; nu*tau=1]'])} (theory τ), near its g_lin {TH['null'].g_lin:.3f}.")
L.append("")

L.append("## Fluctuation peak: simulation vs P5 (n = 526 unless stated; ν_on·τ = 1, in brackets ν_on·τ = 2)\n")
L.append("| graph | h | sim peak of χ_rep/ρ [95% CI] | resolved? | FWHM (decades) | sim p_on = ½ at g [CI] | P5 theory τ r=1 | P5 theory τ r=0 | P5 measured τ | P5 measured τ, n from K_eff to 1000 | P5 theory τ r=1, n range |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for row in head:
    nprim = "n=526 (AL module)" if row["kind"] == "real" else NCH["null"][1][0]
    gt = lambda tau, x: row.get(f"theory[{tau}; {nprim}; nu*tau={x}]")
    L.append(f"| {row['kind']} | {row['h']:g} | {f3(row['sim_F_rep_peak'])} {ci(row['sim_F_rep_peak_ci'])} | "
             f"{'yes' if row['sim_F_rep_resolved'] else 'NO'} | "
             f"{('< ' + format(2 * row['grid_spacing_decades'], '.2f') + ' (1 grid pt)') if (row['sim_F_rep_resolved'] and row['sim_F_rep_single_point']) else (f3(row['sim_F_rep_fwhm_decades'], 2) if row['sim_F_rep_resolved'] else '–')} | "
             f"{f3(row['sim_p_on_half'])} {ci(row['sim_p_on_half_ci'])} | "
             f"{f3(gt('theory r=1', 1))} ({f3(gt('theory r=1', 2))}) | {f3(gt('theory r=0', 1))} ({f3(gt('theory r=0', 2))}) | "
             f"{f3(gt('measured tau_sim (h=0)', 1))} ({f3(gt('measured tau_sim (h=0)', 2))}) | "
             f"{ci(row['theory[measured tau; n range]'])} | {ci(row['theory[theory r=1; n range]'])} |")
L.append("\nNull row: n = effective size of the null's module seed set; measured τ for the null from its own h = 0 runs.\n")
L.append("Other peak measures (same windows: real 0.1 ≤ g ≤ 1.0, null 0.5 ≤ g ≤ 5.0):\n")
for row in head:
    L.append(f"- {row['kind']} h = {row['h']:g}: χ_time/ρ peak at {f3(row['sim_F_time_peak'])} "
             f"({'resolved' if row['sim_F_time_resolved'] else 'not resolved'}); raw χ_rep peak at {f3(row['sim_chi_rep_raw_peak'])} "
             f"({'resolved' if row['sim_chi_rep_raw_resolved'] else 'not resolved'}); raw χ_rep global maximum over the whole grid at "
             f"g = {f3(res['runs'][row['run']]['peaks']['chi_rep_global_max_g'])}")

L.append("\nPredicted size of the module (telegraph) contribution, n = 526 (null: seed-set size), at the g where it is largest "
         "inside the window, vs the measured χ_rep at that g:\n")
for row in head:
    amp = res["runs"][row["run"]].get("telegraph_amplitude", {})
    parts = [f"{t}: model max {f3(a['model_chi_max'])} at g = {f3(a['g_at_model_max'])}, measured χ_rep there {f3(a['sim_chi_rep_there'])} "
             f"(ratio {f3(a['model_over_sim'], 2)})" for t, a in amp.items()]
    L.append(f"- {row['kind']} h = {row['h']:g}: " + "; ".join(parts))

# downward shift test
L.append("\n## Does the peak move down as h grows?\n")
r5 = [r for r in head if r["kind"] == "real"]
def nonover(a, b):
    return a and b and (a[1] < b[0] or b[1] < a[0])
for a, b in zip(r5[:-1], r5[1:]):
    msg = f"- h {a['h']:g} → {b['h']:g}: "
    if a["sim_F_rep_resolved"] and b["sim_F_rep_resolved"]:
        msg += (f"χ_rep/ρ peak {f3(a['sim_F_rep_peak'])} → {f3(b['sim_F_rep_peak'])} (CIs {ci(a['sim_F_rep_peak_ci'])}, "
                f"{ci(b['sim_F_rep_peak_ci'])}: {'separated' if nonover(a['sim_F_rep_peak_ci'], b['sim_F_rep_peak_ci']) else 'overlapping'}); ")
    else:
        msg += "χ_rep/ρ peak not resolved at " + ", ".join(f"h = {x['h']:g}" for x in (a, b) if not x["sim_F_rep_resolved"]) + "; "
    if a["sim_p_on_half"] and b["sim_p_on_half"]:
        msg += (f"p_on = ½ crossing {f3(a['sim_p_on_half'])} → {f3(b['sim_p_on_half'])} (CIs {ci(a['sim_p_on_half_ci'])}, "
                f"{ci(b['sim_p_on_half_ci'])}: {'separated' if nonover(a['sim_p_on_half_ci'], b['sim_p_on_half_ci']) else 'overlapping'})")
    else:
        msg += "p_on = ½ crossing not measurable at " + ", ".join(f"h = {x['h']:g}" for x in (a, b) if not x["sim_p_on_half"])
    L.append(msg)

L.append("\n## Telegraph check: module size the measured on-fraction needs (measured τ, p_est = 1 − 1/R)\n")
L.append("| run | g | p_on | τ_measured (h=0) | p_est | n needed [from p_on CI] |")
L.append("|---|---|---|---|---|---|")
for k, v in res["runs"].items():
    for r in v.get("n_needed", []):
        L.append(f"| {k} | {r['g']:.4g} | {r['p_on']:.3f} | {r['tau_measured']:.4g} | {r['p_est']:.3f} | {f3(r['n_needed'])} "
                 f"{ci(r['n_needed_range_from_p_on_ci'])} |")
L.append(f"\nn choices: 526 (AL module, flow_params n_M), K_eff(r=1) = {TH['real'].K1[0]:.1f} at g = {TH['real'].g[0]:.3f} falling to {TH['real'].Keff(0.5)[0]:.1f} at g = 0.5, "
         f"1/IPR of the r=1 mode {mode['real'][1.0]['inv_ipr']:.0f}, seed-set effective size {seeds['real']['eff_size']:.0f}, 1000.\n")

L.append("## Density: simulation vs QMF (median ρ_sim/ρ_QMF over g ranges)\n")
L.append("| run | g < g_lin | g_lin–0.5 | 0.5–3 | 3–30 | min ratio (at g) | max ratio |")
L.append("|---|---|---|---|---|---|---|")
for k, v in res["qmf"].items():
    if not isinstance(v, dict):
        continue
    m = v["median_sim_over_qmf"]
    L.append(f"| {k} | {f3(m['g<g_lin'])} | {f3(m['g_lin..0.5'])} | {f3(m['0.5..3'])} | {f3(m['3..30'])} | "
             f"{f3(v['min_ratio'])} ({f3(v['g_at_min_ratio'])}) | {f3(v['max_ratio'])} |")

L.append("\n## Linear response below g_lin (excess ρ/ρ0 − 1)\n")
if "comparison" in res["linresp"]:
    pole = res["linresp"]["pole"]
    L.append(f"Leading-mode pole amplitude (Σv)(Σu)/N = {pole['amp_sumv_sumu_over_N']:.4g} (λ at g_lin {pole['lam_at_g_lin']:.5f}; "
             f"1/IPR of v {pole['inv_ipr_v']:.0f}).\n")
    L.append("| run | g | g/g_lin | sim excess ± s.e. | exact 1st order | B only (no inhibition) | pole only |")
    L.append("|---|---|---|---|---|---|---|")
    for c in res["linresp"]["comparison"]:
        L.append(f"| {c['run']} | {c['g']:.4g} | {c['g'] / TH['real'].g_lin:.3f} | {c['sim_excess']:.4f} ± {c['sim_se']:.4f} | "
                 f"{f3(c['exact'], 4)} | {f3(c['B_only'], 4)} | {c['pole_only']:.4f} |")
    L.append("")
    for k, v in res["linresp"].items():
        if k.startswith("chi2_"):
            L.append(f"- {k[5:]}: χ² over {v['n']} points: exact {v['chi2_exact']:.1f}, B only {v['chi2_B_only']:.1f}, "
                     f"zero excess {v['chi2_zero_excess']:.1f}")
else:
    L.append(res["linresp"].get("status", ""))

L.append("\n## Per-h rows (g ≤ 1.1; all g in the json)\n")
for k, v in res["runs"].items():
    if "rows" not in v:
        continue
    L.append(f"### {k}\n")
    L.append("| g | ρ | χ_rep [CI] | χ_rep/ρ | χ_time | χ_time/ρ | p_on | drift | ρ_QMF | sim/QMF |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in v["rows"]:
        if r["g"] > 1.1:
            continue
        L.append(f"| {r['g']:.4g} | {r['rho']:.3e} | {r['chi_rep']:.3g} {ci(r['chi_rep_ci'])} | {r['F_rep']:.3g} | {r['chi_time']:.3g} | "
                 f"{r['F_time']:.3g} | {f3(r['p_on'])} | {r['drift']:+.3f} | {f3(r.get('rho_qmf'))} | {f3(r.get('sim_over_qmf'))} |")
    L.append("")
open(f"{OUT}/driven.md", "w").write("\n".join(L) + "\n")
print("wrote", f"{OUT}/driven.md")
