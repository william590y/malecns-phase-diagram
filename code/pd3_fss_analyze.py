"""pd3 redraw_fss part B analysis (CPU): finite-size scaling of the onset in size-N configuration nulls vs the real network.

Inputs: results/pd3/redraw_fss/fss_<kind>_N<N>_s<seed>_r<r>.npz (pd3_fss_sim.py); results/analysis2/phase_complete.json (real
full brain); results/pd3/theory_bd.json null_family (kappa-N rule); results/pd3/theory_nullK.json (K_nc table).
Per graph and r (x = g / the graph's own exact g_lin):
  tau_sim(x) = censored-exponential MLE from the exact extinction steps (censored at T = 3000).
  x_on (tau = 3000), x_S (tau = 200): ln tau_reg (E/max(d, 0.5)) made monotone by weighted isotonic regression (weights d + 0.5),
    first crossing interpolated in ln x (analysis2_phase.py rules); x_half: alive-at-T fraction (isotonic, weights B) crosses 1/2.
    2000-resample replica bootstrap (each point resampled independently) -> 95% CI; total = envelope of grid bracket and CI.
  steepness s = d ln tau / d ln g at tau = 1000: slope of the isotonic ln tau on the segment that crosses ln 1000; also the
    slow-window mean slope s_win = ln(3000/200) / ln(x_on/x_S).
  tau(x = 1): ln tau interpolated in ln x between the two grid points bracketing x = 1 (all replicas die there).
  K_fit: argmin_K sum (ln tau_sim - log_tau_smooth(R(x) alpha_eff, K))^2 over points with 200 <= tau_sim <= 3000 and >= 3 deaths
    (R(x) = lambda_PF(B(x g_lin))/alpha_eff measured on that graph); K_needed per point as well.
  chi_surv(x) = var(n)/<n> over survivor samples (every 5 steps in (T/2, T], pooled over replicas): peak x and height.
  chain overlay: x_on, x_S of the chain with the realisation's own R(x) and K_nc(g_lin) (meaningful at r = 0 only).
Decisions FSS-1/2/3 as pre-registered in results/pd3/plan.md section 2.2.
Outputs: results/pd3/redraw_fss/fss_analysis.json; figs/pd3/fss_onset_vs_N.png, fss_lifetime_curves.png, fss_steepness.png,
  lifetime_vs_K.png.
"""
import glob, json, os, re, sys
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd3_common import iso, first_cross, A_EFF, ROOT
from pd3_common import log_tau_smooth_fast as log_tau_smooth      # vectorised, identical to log_tau_smooth to 1e-15


def k_needed(lam, lt_target, kmax=50000.0):
    """Copy of pd3_common.k_needed (analysis2_phase.py) using the vectorised chain."""
    if not np.isfinite(lam) or lam <= A_EFF:
        return None
    lo, hi = 1.01, 4.0
    while log_tau_smooth(lam, hi) < lt_target:
        hi *= 2
        if hi > kmax:
            return None
    if log_tau_smooth(lam, lo) > lt_target:
        return None
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if log_tau_smooth(lam, mid) < lt_target else (lo, mid)
        if hi / lo < 1 + 1e-5:
            break
    return float(np.sqrt(lo * hi))

RES = f"{ROOT}/results"; DIR = f"{RES}/pd3/redraw_fss"; FIG = f"{ROOT}/figs/pd3"
T_REF = 3000.0; TAU_S = 200.0; NBOOT = int(os.environ.get("NBOOT", 2000)); DT = 0.2
rng = np.random.default_rng(20261110)
PJ = json.load(open(f"{RES}/analysis2/phase_complete.json"))
GLIN_REAL = PJ["g_lin"]["real"]
REAL_XON = {0: PJ["sim"]["real"]["0"]["on"]["g"] / GLIN_REAL, 1: PJ["sim"]["real"]["1"]["on"]["g"] / GLIN_REAL}
REAL_XS = {0: PJ["sim"]["real"]["0"]["S"]["g"] / GLIN_REAL, 1: PJ["sim"]["real"]["1"]["S"]["g"] / GLIN_REAL}
NF = json.load(open(f"{RES}/pd3/theory_bd.json"))["null_family"]
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7"}
SEED_MK = {0: "D", 1: "o", 2: "s", 3: "v"}
NS_NULL = [2500, 5000, 10000, 20000, 40000, 80000, 167216]
try:   # the 526-neuron AL module simulated alone (module_theory program, MT5): x_on relative to its own g_lin
    _il = json.load(open(f"{RES}/pd3/module_theory/island_lifetimes.json"))
    AL_ALONE = {0: _il["AL_r0"]["x_tau3000"], 1: _il["AL_r1"]["x_tau3000"]}
except Exception:
    AL_ALONE = None


def load(p):
    d = np.load(p)
    o = {k: d[k] for k in d.files if k not in ("diag", "meta")}
    o["diag"] = json.loads(str(d["diag"])); o["meta"] = json.loads(str(d["meta"]))
    ext = o["ext_step"]                                          # [G, B]
    o["died"] = ext >= 0
    o["times"] = np.where(o["died"], ext * DT, float(o["T"]))
    return o


def to_points(o, src):
    return [{"x": float(o["x"][i]), "g": float(o["g"][i]), "R": float(o["R_x"][i]), "died": o["died"][i].copy(),
             "times": o["times"][i].copy(), "cnt": o["n_cnt"][i].copy(), "s1": o["n_s1"][i].copy(), "s2": o["n_s2"][i].copy(),
             "src": src} for i in range(len(o["x"]))]


def merge_points(pts):
    """Sort by x; pool replicas of points at the same x (relative difference < 1e-6), as analysis2_phase.load_sim pools runs."""
    pts = sorted(pts, key=lambda p: p["x"]); out = []
    for p in pts:
        if out and abs(p["x"] / out[-1]["x"] - 1) < 1e-6:
            q = out[-1]
            for k in ["died", "times", "cnt", "s1", "s2"]:
                q[k] = np.r_[q[k], p[k]]
            q["src"] += "+" + p["src"]
        else:
            out.append(dict(p))
    return out


def crossings(x, d, E, alive, B, Elb=None):
    lt = iso(np.log(E / np.maximum(d, 0.5)), d + 0.5)
    on = first_cross(x, lt, np.log(T_REF)); S = first_cross(x, lt, np.log(TAU_S))
    half = first_cross(x, iso(alive, B), 0.5)
    m = first_cross(x, lt, np.log(1000.0))
    s1000, st = None, None
    if m["status"] == "ok":
        k = m["k"]
        if d[k] > 0 and d[k - 1] > 0:
            s1000 = float((lt[k] - lt[k - 1]) / np.log(x[k] / x[k - 1])); st = "resolved"
        elif d[k] == 0 and d[k - 1] > 0 and Elb is not None:
            # upper endpoint censored: slope to its one-sided 95% lower bound on tau (2E/chi2_.95(2)) -> lower bound on s
            s1000 = float((np.log(Elb[k]) - lt[k - 1]) / np.log(x[k] / x[k - 1])); st = "lower_bound"
        else:
            st = "unresolved"
    return {"on": on, "S": S, "half": half, "s1000": s1000, "s1000_status": st, "lt": lt}


def slope_mle(x, d, E, lt_iso, lo=100.0, hi=1e4):
    """Censored-exponential MLE of ln tau = a + b ln x over the window of grid points from the last point with isotonic
    ln tau < ln lo to the first with >= ln hi (inclusive): loglik = sum[-d (a + b u) - E exp(-(a + b u))], Newton.
    Returns (b, se_b, n_points, n_uncensored) or None if fewer than 2 points with >= 3 deaths or the fit diverges."""
    below = np.where(lt_iso < np.log(lo))[0]; above = np.where(lt_iso >= np.log(hi))[0]
    i0 = int(below[-1]) if len(below) else 0
    i1 = int(above[0]) if len(above) else len(x) - 1
    sel = np.arange(i0, i1 + 1)
    if (d[sel] >= 3).sum() < 2:
        return None
    u = np.log(x[sel]) - np.log(x[sel]).mean(); dd = d[sel]; EE = E[sel]
    m = dd > 0
    b, a = np.polyfit(u[m], np.log(EE[m] / dd[m]), 1) if m.sum() >= 2 else (10.0, np.log(EE.sum() / max(dd.sum(), 1)))
    th = np.array([a, b], float)
    for _ in range(100):
        eta = th[0] + th[1] * u; w = EE * np.exp(-eta)
        g = np.array([np.sum(w - dd), np.sum((w - dd) * u)])
        H = -np.array([[np.sum(w), np.sum(w * u)], [np.sum(w * u), np.sum(w * u * u)]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        th = th - step
        if not np.all(np.isfinite(th)) or abs(th[1]) > 1e4:
            return None
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = th[0] + th[1] * u; w = EE * np.exp(-eta)
    I = np.array([[np.sum(w), np.sum(w * u)], [np.sum(w * u), np.sum(w * u * u)]])
    try:
        cov = np.linalg.inv(I)
    except np.linalg.LinAlgError:
        return None
    return float(th[1]), float(np.sqrt(cov[1, 1])), int(len(sel)), int((dd > 0).sum())


def summarize(est, boot):
    v = np.asarray(boot, float); fin = v[np.isfinite(v)]
    out = {"x": est["g"], "status": est["status"], "bracket": est["bracket"], "boot_frac_defined": float(np.isfinite(v).mean())}
    out["boot_ci"] = [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) >= 20 else None
    if est["g"] is not None:
        los = [z for z in [est["bracket"][0], out["boot_ci"][0] if out["boot_ci"] else None] if z is not None]
        his = [z for z in [est["bracket"][1], out["boot_ci"][1] if out["boot_ci"] else None] if z is not None]
        out["total"] = [float(min(los)), float(max(his))]
    else:
        out["total"] = None
    return out


def analyse(P, diag):
    x = np.array([p["x"] for p in P]); G = len(x); Rx = np.array([p["R"] for p in P])
    Bv = np.array([len(p["died"]) for p in P])
    d = np.array([p["died"].sum() for p in P], float); E = np.array([p["times"].sum() for p in P])
    alive = 1 - d / Bv
    Elb = 2 * E / stats.chi2.ppf(0.95, 2)
    est = crossings(x, d, E, alive, Bv, Elb)
    # bootstrap: each point's replicas resampled independently (analysis2_phase.bootstrap)
    Db = np.zeros((NBOOT, G)); Eb = np.zeros((NBOOT, G))
    for j, p in enumerate(P):
        idx = rng.integers(0, Bv[j], (NBOOT, Bv[j]))
        Db[:, j] = p["died"][idx].sum(1); Eb[:, j] = p["times"][idx].sum(1)
    bs = {"on": [], "S": [], "half": [], "s1000": []}
    for b in range(NBOOT):
        c = crossings(x, Db[b], Eb[b], 1 - Db[b] / Bv, Bv, 2 * Eb[b] / stats.chi2.ppf(0.95, 2))
        for k in ["on", "S", "half"]:
            bs[k].append(c[k]["g"] if c[k]["g"] is not None else np.nan)
        bs["s1000"].append(c["s1000"] if c["s1000"] is not None else np.nan)
    res = {k: summarize(est[k], bs[k]) for k in ["on", "S", "half"]}
    res["boot_on_samples"] = np.array(bs["on"])
    sb = np.array(bs["s1000"]); fin = sb[np.isfinite(sb)]
    res["s1000"] = {"value": est["s1000"], "status": est["s1000_status"],
                    "boot_ci": [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) >= 20 else None}
    same_cell = (res["on"]["bracket"] == res["S"]["bracket"]) if (res["on"]["x"] and res["S"]["x"]) else None
    res["s_window"] = (float(np.log(T_REF / TAU_S) / np.log(res["on"]["x"] / res["S"]["x"]))
                       if (res["on"]["x"] and res["S"]["x"] and res["on"]["x"] > res["S"]["x"]) else None)
    res["s_window_same_cell"] = same_cell
    sm = slope_mle(x, d, E, est["lt"])
    res["s_fit"] = ({"value": sm[0], "se": sm[1], "n_points": sm[2], "n_uncensored": sm[3], "window_tau": [100.0, 1e4]}
                    if sm else None)
    tau = np.where(d > 0, E / np.maximum(d, 1), np.inf)
    res["points"] = [{"x": float(x[i]), "g": float(P[i]["g"]), "deaths": int(d[i]), "B": int(Bv[i]), "tau_sim": float(tau[i]),
                      "tau_reg": float(E[i] / max(d[i], 0.5)), "tau_lb95_if_censored": float(Elb[i]) if d[i] == 0 else None,
                      "R": float(Rx[i]), "alive_T": float(alive[i]), "src": P[i]["src"]} for i in range(G)]
    res["n_points"] = G; res["n_fine_points"] = int(sum("fine" in p["src"] for p in P))
    # tau(x = 1)
    k = int(np.searchsorted(x, 1.0))
    if 0 < k < G and d[k - 1] == Bv[k - 1] and d[k] == Bv[k]:
        f = np.log(1.0 / x[k - 1]) / np.log(x[k] / x[k - 1])
        res["tau_x1"] = float(np.exp(np.log(tau[k - 1]) + f * (np.log(tau[k]) - np.log(tau[k - 1]))))
        res["tau_x1_points"] = [[float(x[k - 1]), float(tau[k - 1])], [float(x[k]), float(tau[k])]]
    else:
        res["tau_x1"] = None
    # K_fit
    sel = [i for i in range(G) if d[i] >= 3 and TAU_S <= tau[i] <= T_REF]
    res["K_fit_n_points"] = len(sel)
    if sel:
        lam = Rx[sel] * A_EFF; lts = np.log(tau[sel])
        f = lambda lk: float(np.sum([(lts[j] - log_tau_smooth(lam[j], np.exp(lk))) ** 2 for j in range(len(sel))]))
        r_ = minimize_scalar(f, bounds=(np.log(2.0), np.log(2e5)), method="bounded", options={"xatol": 1e-5})
        res["K_fit"] = float(np.exp(r_.x)); res["K_fit_rms_dlntau"] = float(np.sqrt(r_.fun / len(sel)))
        res["K_needed_points"] = [{"x": float(x[i]), "tau": float(tau[i]), "K_needed": k_needed(Rx[i] * A_EFF, np.log(tau[i]))}
                                  for i in sel]
    else:
        res["K_fit"] = None
    # chi_surv
    cnt = np.array([p["cnt"].sum() for p in P]); s1 = np.array([p["s1"].sum() for p in P]); s2 = np.array([p["s2"].sum() for p in P])
    nrep = np.array([(p["cnt"] > 0).sum() for p in P])
    ok = (cnt >= 100) & (nrep >= 3)
    mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan); var = np.where(cnt > 0, s2 / np.maximum(cnt, 1) - mean ** 2, np.nan)
    chi = np.where(ok, var / mean, np.nan)
    res["chi_surv"] = chi.tolist(); res["n_surv_mean"] = mean.tolist()
    if np.isfinite(chi).any():
        j = int(np.nanargmax(chi))
        res["chi_peak"] = {"x": float(x[j]), "height": float(chi[j]), "interior": bool(np.isfinite(chi[:j]).any() and j < G - 1),
                           "n_valid": int(np.isfinite(chi).sum())}
    else:
        res["chi_peak"] = None
    # supplementary (not pre-registered): max of chi_surv inside the crossover window [x_S, 1.25 x_on]
    res["chi_win"] = None
    if res["S"]["x"] and res["on"]["x"]:
        w = np.where((x >= res["S"]["x"]) & (x <= 1.25 * res["on"]["x"]) & np.isfinite(chi))[0]
        if len(w):
            j = int(w[np.argmax(chi[w])])
            res["chi_win"] = {"x": float(x[j]), "height": float(chi[j]), "x_over_x_on": float(x[j] / res["on"]["x"]),
                              "n_surv_mean_at_peak": float(mean[j]), "window": "x_S <= x <= 1.25 x_on, >= 3 surviving replicas"}
    # chain overlay with the realisation's own R(x) and K_nc(g_lin)
    Kc = diag["K_nc_glin"]
    xf = np.exp(np.linspace(np.log(x[0]), np.log(x[-1]), 800))
    Rf = np.exp(np.interp(np.log(xf), np.log(x), np.log(Rx)))
    ltc = np.array([log_tau_smooth(R * A_EFF, Kc) if R > 1 else -10 for R in Rf])
    res["chain_Knc"] = {"K": Kc, "x_on": first_cross(xf, ltc, np.log(T_REF))["g"], "x_S": first_cross(xf, ltc, np.log(TAU_S))["g"],
                        "tau_x1": float(np.exp(log_tau_smooth(A_EFF, Kc)))}
    if res["K_fit"]:
        ltk = np.array([log_tau_smooth(R * A_EFF, res["K_fit"]) if R > 1 else -10 for R in Rf])
        c1 = first_cross(xf, ltk, np.log(1000.0))
        s_ch = None
        if c1["status"] == "ok":
            kk = c1["k"]; s_ch = float((ltk[kk] - ltk[kk - 1]) / np.log(xf[kk] / xf[kk - 1]))
        res["chain_Kfit"] = {"x_on": first_cross(xf, ltk, np.log(T_REF))["g"], "x_S": first_cross(xf, ltk, np.log(TAU_S))["g"],
                             "s1000": s_ch}
    res["curve_x"] = x.tolist(); res["curve_tau_reg"] = (E / np.maximum(d, 0.5)).tolist(); res["curve_lt_iso"] = est["lt"].tolist()
    res["curve_deaths"] = d.tolist()
    return res


def main():
    files = sorted(glob.glob(f"{DIR}/fss_*_N*_s*_r*.npz"))
    groups = {}
    for p in files:
        m = re.match(r"fss_(fine_)?(cnull|realsub)_N(\d+)_s(\d+)_r([\d.]+)\.npz", os.path.basename(p))
        if not m:
            continue
        key = (m.group(2), int(m.group(3)), int(m.group(4)), int(float(m.group(5))))
        groups.setdefault(key, {})["fine" if m.group(1) else "coarse"] = p
    R = {}
    for key, fs in sorted(groups.items()):
        if "coarse" not in fs:
            continue
        kind, N, seed, r = key
        o = load(fs["coarse"]); P = to_points(o, "coarse")
        files_used = [os.path.relpath(fs["coarse"], RES)]
        if "fine" in fs:
            of = load(fs["fine"])
            assert abs(of["diag"]["g_lin"] / o["diag"]["g_lin"] - 1) < 1e-9
            P += to_points(of, "fine"); files_used.append(os.path.relpath(fs["fine"], RES))
        P = merge_points(P)
        a = analyse(P, o["diag"])
        a.update({"kind": kind, "N": N, "seed": seed, "r": r, "diag": o["diag"], "gpu": o["meta"].get("gpu"),
                  "wall_s": o["meta"].get("wall_s"), "file": files_used})
        R[key] = a
        print(f"{kind:7s} N={N:6d} s{seed} r={r:g}: g_lin {o['diag']['g_lin']:.5f} K_nc {o['diag']['K_nc_glin']:.1f} | n {a['n_points']}"
              f" (fine {a['n_fine_points']}) | x_S {fmt(a['S']['x'], 4)} x_on {fmt(a['on']['x'], 4)} {a['on']['total']} | s1000 "
              f"{fmt(a['s1000']['value'])} ({a['s1000']['status']}) | tau(1) {fmt(a['tau_x1'])} | K_fit {fmt(a['K_fit'])} "
              f"(n {a['K_fit_n_points']}) | chain(Knc) x_on {fmt(a['chain_Knc']['x_on'], 4)}", flush=True)
    return R


# ------------------------------------------------------------------------------------------------------------ summary
def fmt(v, p=3):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}g}"


def summary(R):
    out = {"per_graph": [], "seed_means": {}, "decisions": {}, "inputs": {
        "sim_files": sorted(f for v in R.values() for f in v["file"]), "real_full_brain_x_on": REAL_XON, "real_full_brain_x_S": REAL_XS,
        "real_source": "results/analysis2/phase_complete.json sim.real (on.g, S.g) / g_lin 0.16315",
        "AL_alone_x_on": AL_ALONE, "AL_alone_source": "results/pd3/module_theory/island_lifetimes.json AL_r0/AL_r1 x_tau3000 (module_theory MT5)",
        "kappa_N_rule": "results/pd3/theory_bd.json null_family rows r0_theory (on_over_glin), r1_needed"}}
    for k, a in sorted(R.items()):
        e = {kk: a[kk] for kk in ["kind", "N", "seed", "r", "on", "S", "half", "s1000", "s_window", "tau_x1", "K_fit", "K_fit_n_points",
                                   "chi_peak", "chain_Knc", "file", "gpu", "wall_s"]}
        e["K_fit_rms_dlntau"] = a.get("K_fit_rms_dlntau"); e["chain_Kfit"] = a.get("chain_Kfit")
        e["K_needed_points"] = a.get("K_needed_points")
        e["diag"] = a["diag"]; e["points"] = a["points"]; e["chi_surv"] = a["chi_surv"]; e["n_surv_mean"] = a["n_surv_mean"]
        e["curve_lt_iso"] = a["curve_lt_iso"]; e["n_points"] = a["n_points"]; e["n_fine_points"] = a["n_fine_points"]
        e["s_window_same_cell"] = a["s_window_same_cell"]; e["s_fit"] = a["s_fit"]; e["chi_win"] = a["chi_win"]
        out["per_graph"].append(e)
    # seed means per (kind, N, r)
    for kind in ["cnull", "realsub"]:
        for r in [0, 1]:
            Ns = sorted(set(k[1] for k in R if k[0] == kind and k[3] == r))
            rows = []
            for N in Ns:
                ks = sorted(k for k in R if k[0] == kind and k[1] == N and k[3] == r)
                xs = [R[k]["on"]["x"] for k in ks]
                xmax = max(R[k]["curve_x"][-1] for k in ks)
                n_none = sum(1 for v in xs if v is None)
                fin = [v for v in xs if v is not None]
                bmat = np.array([R[k]["boot_on_samples"] for k in ks])          # [seeds, NBOOT]
                bmean = bmat.mean(0) if n_none == 0 else np.full(NBOOT, np.nan)
                fb = bmean[np.isfinite(bmean)]
                rows.append({"N": N, "seeds": [k[2] for k in ks], "x_on_per_seed": xs,
                             "x_on_mean": float(np.mean(fin)) if n_none == 0 else None,
                             "x_on_mean_lower_bound": (float(np.mean([v if v is not None else xmax for v in xs]))
                                                        if n_none else None),
                             "n_not_reached": n_none, "x_max": xmax,
                             "x_on_mean_ci": [float(np.percentile(fb, 2.5)), float(np.percentile(fb, 97.5))] if len(fb) > 20 else None,
                             "x_on_seed_sd": float(np.std(fin, ddof=1)) if len(fin) > 1 else None,
                             "x_S_mean": float(np.mean([R[k]["S"]["x"] for k in ks])) if all(R[k]["S"]["x"] for k in ks) else None,
                             "K_nc_glin": [R[k]["diag"]["K_nc_glin"] for k in ks],
                             "K_fit": [R[k]["K_fit"] for k in ks], "tau_x1": [R[k]["tau_x1"] for k in ks],
                             "s1000": [R[k]["s1000"]["value"] for k in ks], "s_window": [R[k]["s_window"] for k in ks],
                             "s_fit": [R[k]["s_fit"]["value"] if R[k]["s_fit"] else None for k in ks],
                             "g_lin": [R[k]["diag"]["g_lin"] for k in ks],
                             "chain_Knc_x_on": [R[k]["chain_Knc"]["x_on"] for k in ks],
                             "chi_peak_x": [R[k]["chi_peak"]["x"] if R[k]["chi_peak"] else None for k in ks],
                             "chi_peak_h": [R[k]["chi_peak"]["height"] if R[k]["chi_peak"] else None for k in ks]})
            out["seed_means"][f"{kind}_r{r}"] = rows
    sup = {}
    for kind in ["cnull", "realsub"]:
        for r in [0, 1]:
            ks = [k for k in R if k[0] == kind and k[3] == r and R[k]["chi_win"]]
            if len(ks) >= 3:
                sl = stats.linregress(np.log([k[1] for k in ks]), np.log([R[k]["chi_win"]["height"] for k in ks]))
                sup[f"{kind}_r{r}"] = {"loglog_slope_vs_N": float(sl.slope), "se": float(sl.stderr), "n": len(ks),
                                       "heights_by_N": {str(N): [round(R[k]["chi_win"]["height"], 2) for k in ks if k[1] == N]
                                                        for N in sorted(set(k[1] for k in ks))},
                                       "x_over_x_on": [round(R[k]["chi_win"]["x_over_x_on"], 3) for k in ks]}
    out["supplementary_chi_win"] = sup
    D = out["decisions"]
    # FSS-1
    for r in [0, 1]:
        rows = out["seed_means"].get(f"cnull_r{r}", [])
        Ns = [z["N"] for z in rows]
        xm = [z["x_on_mean"] if z["x_on_mean"] is not None else z["x_on_mean_lower_bound"] for z in rows]
        sp = stats.spearmanr(Ns, xm).correlation if len(rows) >= 3 else None
        mono = bool(all(np.diff(xm) < 0)) if len(xm) > 1 else None
        e = {"N": Ns, "x_on_seed_mean (lower bound where not reached)": xm, "spearman": sp, "strictly_decreasing": mono,
             "n_sizes": len(rows)}
        if r == 0:
            x167 = next((z["x_on_mean"] for z in rows if z["N"] == 167216), None)
            x10 = next((z["x_on_mean"] for z in rows if z["N"] == 10000), None)
            e.update({"x_on_167k": x167, "x_on_10k": x10,
                      "pass": bool(sp is not None and sp <= -0.9 and x167 is not None and x167 <= 1.10 and x10 is not None and x10 >= 1.15)})
        else:
            e["pass"] = bool(sp is not None and sp <= -0.9)
        D[f"FSS1_r{r}"] = e
    # FSS-2
    for r in [0, 1]:
        rows = out["seed_means"].get(f"realsub_r{r}", [])
        ref = REAL_XON[r]
        per = [{"N": z["N"], "x_on_per_seed": z["x_on_per_seed"],
                "rel_dev_per_seed": [(v / ref - 1) if v is not None else None for v in z["x_on_per_seed"]]} for z in rows]
        allv = [d_ for p in per for d_ in p["rel_dev_per_seed"]]
        D[f"FSS2_r{r}"] = {"reference_full_brain_x_on": ref, "rows": per,
                           "max_abs_rel_dev": float(max(abs(v) for v in allv if v is not None)) if any(v is not None for v in allv) else None,
                           "pass": bool(allv and all(v is not None and abs(v) <= 0.05 for v in allv))}
    # FSS-3
    for r in [0, 1]:
        ks = [k for k in R if k[0] == "cnull" and k[3] == r and R[k]["K_fit"]]
        if len(ks) >= 3:
            lN = np.log([k[1] for k in ks]); lK = np.log([R[k]["K_fit"] for k in ks])
            sl = stats.linregress(lN, lK)
            a_fit = {"a": float(sl.slope), "a_se": float(sl.stderr), "n": len(ks), "r2": float(sl.rvalue ** 2)}
        else:
            a_fit = None
        lNc = np.log([k[1] for k in R if k[0] == "cnull" and k[3] == r])
        lKc = np.log([R[k]["diag"]["K_nc_glin"] for k in R if k[0] == "cnull" and k[3] == r])
        knc = stats.linregress(lNc, lKc) if len(lNc) >= 3 else None
        # tau(x=1) vs K
        def tau_fit(keyK):
            kk = [k for k in R if k[3] == r and R[k]["tau_x1"] and (R[k]["K_fit"] if keyK == "K_fit" else True)]
            if len(kk) < 3:
                return None
            Kv = [R[k]["K_fit"] if keyK == "K_fit" else R[k]["diag"]["K_nc_glin"] for k in kk]
            sl = stats.linregress(np.log(Kv), np.log([R[k]["tau_x1"] for k in kk]))
            return {"slope": float(sl.slope), "se": float(sl.stderr), "n": len(kk), "r2": float(sl.rvalue ** 2),
                    "graphs": "cnull + realsub, r = %d" % r}
        def tau_fit_nulls(keyK):
            kk = [k for k in R if k[3] == r and k[0] == "cnull" and R[k]["tau_x1"] and (R[k]["K_fit"] if keyK == "K_fit" else True)]
            if len(kk) < 3:
                return None
            Kv = [R[k]["K_fit"] if keyK == "K_fit" else R[k]["diag"]["K_nc_glin"] for k in kk]
            sl = stats.linregress(np.log(Kv), np.log([R[k]["tau_x1"] for k in kk]))
            return {"slope": float(sl.slope), "se": float(sl.stderr), "n": len(kk), "r2": float(sl.rvalue ** 2)}
        ks2 = [k for k in R if k[3] == r and R[k]["K_fit"] and R[k]["s1000"]["value"]]
        ssp = stats.spearmanr([R[k]["K_fit"] for k in ks2], [R[k]["s1000"]["value"] for k in ks2]).correlation if len(ks2) >= 3 else None
        ks2r = [k for k in ks2 if R[k]["s1000"]["status"] == "resolved"]
        sspr = stats.spearmanr([R[k]["K_fit"] for k in ks2r], [R[k]["s1000"]["value"] for k in ks2r]).correlation if len(ks2r) >= 3 else None
        ks4 = [k for k in R if k[3] == r and R[k]["K_fit"] and R[k]["s_fit"]]
        ssf = stats.spearmanr([R[k]["K_fit"] for k in ks4], [R[k]["s_fit"]["value"] for k in ks4]).correlation if len(ks4) >= 3 else None
        ks4n = [k for k in ks4 if k[0] == "cnull"]
        sfl = stats.linregress(np.log([R[k]["K_fit"] for k in ks4n]), np.log([R[k]["s_fit"]["value"] for k in ks4n])) if len(ks4n) >= 3 else None
        ks3 = [k for k in R if k[3] == r and R[k]["K_fit"] and R[k]["s_window"]]
        ssw = stats.spearmanr([R[k]["K_fit"] for k in ks3], [R[k]["s_window"] for k in ks3]).correlation if len(ks3) >= 3 else None
        tf = tau_fit_nulls("K_fit"); tfn = tau_fit_nulls("K_nc")
        D[f"FSS3_r{r}"] = {"K_fit_vs_N_nulls": a_fit,
                           "K_nc_glin_vs_N_nulls": {"a": float(knc.slope), "se": float(knc.stderr)} if knc else None,
                           "tau_x1_vs_K_fit_nulls": tf, "tau_x1_vs_K_nc_nulls": tfn,
                           "tau_x1_vs_K_fit_all": tau_fit("K_fit"), "tau_x1_vs_K_nc_all": tau_fit("K_nc"),
                           "spearman_s1000_vs_K_fit (value or 95% lower bound)": ssp, "n_s1000": len(ks2),
                           "n_s1000_lower_bounds": len(ks2) - len(ks2r), "spearman_s1000_vs_K_fit_resolved_only": sspr, "n_resolved": len(ks2r), "spearman_s_window_vs_K_fit": ssw, "n_s_window": len(ks3),
                           "spearman_s_fit_vs_K_fit": ssf, "n_s_fit": len(ks4),
                           "s_fit_vs_K_fit_loglog_slope_nulls": ({"slope": float(sfl.slope), "se": float(sfl.stderr), "n": len(ks4n)} if sfl else None),
                           "pass_a": bool(a_fit and a_fit["a"] >= 0.7),
                           "pass_tau_sqrtK (K_fit, nulls)": bool(tf and 0.4 <= tf["slope"] <= 0.6),
                           "pass_s_increases (s_fit, primary)": bool(ssf is not None and ssf > 0),
                           "pass_s_increases (s1000, spec)": bool(ssp is not None and ssp > 0)}
    return out


# ------------------------------------------------------------------------------------------------------------ figures
def figures(R, S):
    from matplotlib.colors import ListedColormap
    cmapN = ListedColormap(plt.get_cmap("viridis")(np.linspace(0, 0.85, 256)))
    normN = LogNorm(2000, 2.2e5)
    # 1) onset vs N
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.8))
    for ax, r in zip(axs, [0, 1]):
        rows = S["seed_means"].get(f"cnull_r{r}", [])
        for k, a in sorted(R.items()):
            if k[0] != "cnull" or k[3] != r:
                continue
            xo = a["on"]["x"]
            if xo is not None:
                tot = a["on"]["total"]
                ax.errorbar(k[1] * (1 + 0.04 * (k[2] - 1.5)), xo - 1, yerr=[[xo - tot[0]], [tot[1] - xo]], fmt=SEED_MK[k[2]], color=OI["blue"],
                            mfc="white", ms=6.5, mew=1.4, capsize=2, lw=1, zorder=4)
            else:
                ax.annotate("", (k[1], a["curve_x"][-1] - 1 + 1.5), (k[1], a["curve_x"][-1] - 1), zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color=OI["blue"], lw=1.4))
        Ns = [z["N"] for z in rows]
        xm = [z["x_on_mean"] if z["x_on_mean"] is not None else np.nan for z in rows]
        ax.plot(Ns, np.array(xm) - 1, "-", color=OI["blue"], lw=2.2, zorder=3, label="configuration null, seed mean")
        if r == 0:
            # chain with each realisation's own K_nc and R(x): band = min-max over seeds
            lo = [min(c for c in z["chain_Knc_x_on"] if c) - 1 for z in rows]; hi = [max(c for c in z["chain_Knc_x_on"] if c) - 1 for z in rows]
            ax.fill_between(Ns, lo, hi, color=OI["orange"], alpha=0.30, lw=0, label="chain, own R(x) and K_nc (range over seeds)")
            kr = NF["rows"]["r0_theory"]
            ax.plot([z["N"] for z in kr if z["N"] >= 2000], [z["on_over_glin"] - 1 for z in kr if z["N"] >= 2000], ":", color=OI["verm"], lw=2,
                    label="κN rule (theory_bd null_family)")
        else:
            kr = [z for z in NF["rows"]["r1_needed"] if z["on_over_glin"]]
            ax.plot([z["N"] for z in kr], [z["on_over_glin"] - 1 for z in kr], ":", color=OI["verm"], lw=2,
                    label="κN rule, κ from K_needed (theory_bd)")
        ax.axhline(REAL_XON[r] - 1, color="k", lw=2, label=f"real full brain: x_on = {REAL_XON[r]:.2f}" if r == 0 else "real full brain")
        ax.text(1.9e5, (REAL_XON[r] - 1) * 1.04, f"real full brain {REAL_XON[r]:.2f}", ha="right", va="bottom", fontsize=9)
        for k, a in sorted(R.items()):
            if k[0] == "realsub" and k[3] == r and a["on"]["x"] is not None:
                xo = a["on"]["x"]; tot = a["on"]["total"]
                ax.errorbar(k[1] * (1 + 0.05 * (k[2] - 1.5)), xo - 1, yerr=[[xo - tot[0]], [tot[1] - xo]], fmt=SEED_MK[k[2]],
                            color=OI["verm"], ms=7, capsize=2, lw=1, zorder=5)
        if AL_ALONE:
            ax.plot(526, AL_ALONE[r] - 1, "*", color=OI["verm"], ms=13, mec="k", zorder=6)
        ax.fill_between([400, 3e5], (REAL_XON[r] * 0.95) - 1, (REAL_XON[r] * 1.05) - 1, color="k", alpha=0.08, lw=0,
                        label="±5% of the real x_on (FSS-2 band)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(380, 2.6e5)
        # verification 2026-11-09: explicit y tick labels (the log axis had only one labelled decade)
        _yt = [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0] if r == 0 else [0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5]
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(_yt))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("network size N (neurons)", fontsize=10.5); ax.set_ylabel("x_on − 1  (x = g/g_lin; onset at τ_sim = 3000)", fontsize=10.5)
        ax.set_title(f"{'AB'[r]}  r = {r}: onset distance from g_lin vs N", fontsize=11, loc="left")
        ax.grid(alpha=0.25, which="both", lw=0.5)
        if r == 0:
            hs0 = ax.get_legend_handles_labels()[0]
        else:
            hs0 += [ax.get_legend_handles_labels()[0][1]]
    hs = hs0 + [Line2D([], [], color=OI["blue"], marker=SEED_MK[s], mfc="white", ls="", label=f"null seed {s}") for s in [1, 2, 3]]
    hs += [Line2D([], [], color=OI["blue"], marker="D", mfc="white", ls="", label="null seed 0 (swc.py null; N = 167216 only)"),
           Line2D([], [], color=OI["verm"], marker="o", ls="", label="real subsample keeping the AL (realsub; o / s = seed 1 / 2)"),
           Line2D([], [], color=OI["verm"], marker="*", mec="k", ms=12, ls="", label="AL module alone, N = 526 (module_theory MT5)"),
           ] + ([Line2D([], [], color=OI["blue"], marker=r"$\uparrow$", ls="", ms=11, label="arrow: onset not reached by the largest x simulated")]
                if any(a["on"]["x"] is None for a in R.values()) else [])
    fig.legend(handles=hs, fontsize=8.6, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(f"{FIG}/fss_onset_vs_N.png", dpi=150); plt.close(fig)

    # 2) lifetime curves
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.6), sharey=True)
    for j, r in enumerate([0, 1]):
        for i, fam in enumerate(["cnull", "realsub"]):
            ax = axs[i, j]
            for k, a in sorted(R.items()):
                if k[0] != fam or k[3] != r:
                    continue
                x = np.array(a["curve_x"]); cens = np.array([p["deaths"] == 0 for p in a["points"]])
                t = np.array([p["tau_sim"] for p in a["points"]]); tlb = np.array([p["tau_lb95_if_censored"] or np.nan for p in a["points"]])
                col = cmapN(normN(k[1]))
                ax.plot(x[~cens], t[~cens], "-", color=col, lw=1.3, alpha=0.95, marker=SEED_MK[k[2]], ms=3, mfc="white", clip_on=True)
                ax.plot(x[cens], tlb[cens], "^", color=col, ms=3.5, alpha=0.8, mfc="none", mew=0.8)
            # real full brain
            P = PJ["sim"]["real"][str(r)]["points"]
            xr = np.array([p["g"] for p in P]) / GLIN_REAL; tr = np.array([float(p["tau_sim"]) if isinstance(p["tau_sim"], (int, float)) and np.isfinite(p["tau_sim"]) else np.nan for p in P])
            sel = (xr >= 0.85) & (xr <= (2.6 if r == 0 else 8.5))
            ax.plot(xr[sel], tr[sel], "k.", ms=4, label="real full brain (pooled sim, τ_sim)")
            ax.axhline(T_REF, color="#555555", lw=1, ls="--"); ax.axhline(TAU_S, color="#555555", lw=1, ls=":")
            ax.text(0.92, T_REF * 1.15, "τ = 3000", fontsize=8.5, color="#333333"); ax.text(0.92, TAU_S * 1.15, "τ = 200", fontsize=8.5, color="#333333")
            ax.axvline(1, color="k", lw=1.2)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5, 6e5)
            ax.set_xlim(0.88, 2.6 if r == 0 else 8.3)
            tk = [0.9, 1, 1.2, 1.5, 2, 2.5] if r == 0 else [0.9, 1, 1.5, 2, 3, 4, 6, 8]
            ax.set_xticks(tk); ax.set_xticklabels([f"{v:g}" for v in tk]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("x = g / g_lin (each graph's own exact g_lin)", fontsize=10)
            if j == 0:
                ax.set_ylabel("τ_sim (model time)", fontsize=10)
            nm = "configuration nulls" if fam == "cnull" else "real subsamples keeping the AL (realsub)"
            ax.set_title(f"{'ABCD'[2 * i + j]}  {nm}, r = {r}", fontsize=11, loc="left")
            ax.grid(alpha=0.25, which="both", lw=0.5)
            hl = ax.get_legend_handles_labels()[0] + [
                Line2D([], [], color="#555555", marker="^", mfc="none", ls="", label="no replica died: 95% lower bound on τ"),
                Line2D([], [], color="#555555", marker="o", mfc="white", ls="-", label="seeds: o / s / ▽ = 1 / 2 / 3, ◇ = 0")]
            ax.legend(handles=hl, fontsize=8, loc="lower right")
    sm = plt.cm.ScalarMappable(norm=normN, cmap=cmapN)
    cb = fig.colorbar(sm, ax=axs, shrink=0.6, pad=0.015); cb.set_label("network size N", fontsize=10)
    fig.savefig(f"{FIG}/fss_lifetime_curves.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 3) steepness vs K_fit (+ chi peak)
    fig, axs = plt.subplots(1, 3, figsize=(18.5, 6.0), gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(bottom=0.2)
    for ax, key, lab in [(axs[0], "s_fit", "s_fit = d ln τ / d ln g (censored MLE, τ ∈ [100, 10⁴])"), (axs[1], "s1000", "s1000 = finite difference of isotonic ln τ at τ = 1000")]:
        for k, a in sorted(R.items()):
            if not a["K_fit"]:
                continue
            v = a["s1000"]["value"] if key == "s1000" else (a["s_fit"]["value"] if a["s_fit"] else None)
            if key == "s_fit" and v is not None:
                col_ = cmapN(normN(k[1])) if k[0] == "cnull" else OI["verm"]
                ax.errorbar(a["K_fit"], v, yerr=1.96 * a["s_fit"]["se"], color=col_, lw=0.8, capsize=0, zorder=1)
            if v is None:
                continue
            col = cmapN(normN(k[1])) if k[0] == "cnull" else OI["verm"]
            ax.plot(a["K_fit"], v, SEED_MK[k[2]], color=col, mec=col, mew=1.6 if k[3] == 1 else 0.5,
                    mfc=col if k[3] == 0 else "white", ms=8)
            lb = key == "s1000" and a["s1000"]["status"] == "lower_bound"
            if lb:
                ax.annotate("", (a["K_fit"], v * 1.45), (a["K_fit"], v * 1.02), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("K_fit (one-mode chain fit to τ_sim)", fontsize=10)
        ax.set_ylabel(lab, fontsize=10); ax.grid(alpha=0.25, which="both", lw=0.5)
    # chain expectation: window slope ln(1e4/100)/ln(x(1e4)/x(100)) of the one-mode chain with a measured R(x), vs K
    for key_, lab_, ls_ in [(("cnull", 167216, 0, 0), "chain secant, null R(x) (167216 s0)", "-"), (("realsub", 80000, 1, 0), "chain secant, real R(x) (80k s1)", "--")]:
        if key_ not in R:
            continue
        a_ = R[key_]; xx = np.array([p_["x"] for p_ in a_["points"]]); RR = np.array([p_["R"] for p_ in a_["points"]])
        xf = np.exp(np.linspace(np.log(1.0001), np.log(xx[-1]), 1500)); Rf = np.exp(np.interp(np.log(xf), np.log(xx), np.log(RR)))
        Ks, ss = np.geomspace(10, 2e4, 40), []
        for K in Ks:
            lt = np.array([log_tau_smooth(R_ * A_EFF, K) if R_ > 1 else -10 for R_ in Rf])
            c1, c2 = first_cross(xf, lt, np.log(100.0))["g"], first_cross(xf, lt, np.log(1e4))["g"]
            ss.append(np.log(100.0) / np.log(c2 / c1) if (c1 and c2) else np.nan)
        axs[0].plot(Ks, ss, ls_, color="k", lw=1.4, label=lab_)
        S["chain_s_window_vs_K_" + key_[0]] = {"K": Ks.tolist(), "s": [float(v) for v in ss], "R_source": a_["file"][0]}
    axs[0].set_title("A  Crossover steepness s_fit vs fitted K", fontsize=11, loc="left")
    axs[1].set_title("B  Pre-registered finite difference s1000", fontsize=11, loc="left")
    hs = [Line2D([], [], color="#777777", marker="o", ls="", label="filled: r = 0"),
          Line2D([], [], color="#777777", marker="o", mfc="white", mew=1.6, ls="", label="open: r = 1"),
          Line2D([], [], color=OI["verm"], marker="o", ls="", label="realsub (vermillion)"),
          Line2D([], [], color=cmapN(normN(1e4)), marker="o", ls="", label="cnull (colour = N)")]
    if any(a["s1000"]["status"] == "lower_bound" for a in R.values()):
        hs.append(Line2D([], [], color="#777777", marker=r"$\uparrow$", ls="", ms=10, label="arrow: lower bound (transition inside one grid step)"))
    fig.legend(handles=hs, fontsize=8.5, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.45, -0.02))
    axs[0].legend(fontsize=7.5, loc="lower right", title="secant ln(100)/ln(x(1e4)/x(100)):\nnot the s_fit estimator; like-for-like\nin verify_redraw_fss.png B", title_fontsize=7.5)
    ax = axs[2]
    for k, a in sorted(R.items()):
        if a["chi_win"] is None:
            continue
        col = cmapN(normN(k[1])) if k[0] == "cnull" else OI["verm"]
        ax.plot(k[1] * (1 + 0.04 * (k[2] - 1.5)), a["chi_win"]["height"], SEED_MK[k[2]], color=col, mfc=col if k[3] == 0 else "white",
                mec=col, mew=1.6 if k[3] == 1 else 0.5, ms=8)
    if AL_ALONE:
        pass
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("network size N", fontsize=10)
    ax.set_ylabel("max χ_surv = var(n)/⟨n⟩ over x_S ≤ x ≤ 1.25 x_on", fontsize=10)
    ax.set_title("C  Survivor fluctuations at the crossover", fontsize=11, loc="left"); ax.grid(alpha=0.25, which="both", lw=0.5)
    sm = plt.cm.ScalarMappable(norm=normN, cmap=cmapN)
    cb = fig.colorbar(sm, ax=axs, shrink=0.8, pad=0.01); cb.set_label("network size N (nulls)", fontsize=10)
    fig.savefig(f"{FIG}/fss_steepness.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 4) lifetime at g_lin vs K
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    Kg = np.geomspace(5, 2e4, 200)
    chain = [np.exp(log_tau_smooth(A_EFF, K)) for K in Kg]
    sub = load_c4()
    for ax, keyK, lab in [(axs[0], "K_nc", "K_nc = 1/(⟨u²,v⟩⟨u,v²⟩) of B(g_lin) (no fitting)"), (axs[1], "K_fit", "K_fit (chain fit to the slow window)")]:
        ax.plot(Kg, chain, "-", color="k", lw=1.8, label="exact chain, R = 1 (τ ∝ K^0.5 at large K)")
        for k, a in sorted(R.items()):
            if not a["tau_x1"]:
                continue
            Kv = a["diag"]["K_nc_glin"] if keyK == "K_nc" else a["K_fit"]
            if not Kv:
                continue
            col = cmapN(normN(k[1])) if k[0] == "cnull" else OI["verm"]
            ax.plot(Kv, a["tau_x1"], SEED_MK[k[2]], color=col, mfc=col if k[3] == 0 else "white", mec=col, mew=1.6 if k[3] == 1 else 0.5, ms=8)
        if sub and keyK == "K_nc":
            ax.plot([z[0] for z in sub], [z[1] for z in sub], "*", color=OI["purple"], ms=11, mec="k", label="nested AL sub-modules (criticality C4)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(lab, fontsize=10)
        ax.grid(alpha=0.25, which="both", lw=0.5)
    axs[0].set_ylabel("τ_sim at x = 1 (g = g_lin), model time", fontsize=10)
    axs[0].set_title("A  Lifetime at the graph's own g_lin vs K_nc", fontsize=11, loc="left")
    axs[1].set_title("B  ... vs the fitted K (r = 1 and realsub lie 1.1-2.1x above)", fontsize=11, loc="left")
    hs = axs[0].get_legend_handles_labels()[0] + [
        Line2D([], [], color="#777777", marker="o", ls="", label="filled: r = 0; open: r = 1"),
        Line2D([], [], color=OI["verm"], marker="o", ls="", label="realsub (real wiring, AL kept)"),
        Line2D([], [], color=cmapN(normN(1e4)), marker="o", ls="", label="configuration nulls (colour = N)")]
    axs[0].legend(handles=hs, fontsize=8, loc="upper left")
    sm = plt.cm.ScalarMappable(norm=normN, cmap=cmapN)
    cb = fig.colorbar(sm, ax=axs, shrink=0.8, pad=0.01); cb.set_label("network size N (nulls)", fontsize=10)
    fig.savefig(f"{FIG}/lifetime_vs_K.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def load_c4():
    """Nested AL sub-module points (criticality program C4), if its output exists: list of (K_n, tau(x=1))."""
    cands = glob.glob(f"{RES}/pd3/criticality/*submodule*.json") + glob.glob(f"{RES}/pd3/*/submodule*.json")
    pts = []
    for p in cands:
        try:
            j = json.load(open(p))
        except Exception:
            continue
        rows = j.get("rows") or j.get("submodules") or []
        for z in rows:
            K = z.get("K_n") or z.get("K"); t = z.get("tau_x1") or z.get("tau_top_x1")
            if K and t:
                pts.append((float(K), float(t)))
    return pts


if __name__ == "__main__":
    R = main()
    S = summary(R)
    S["c4_points_used"] = load_c4()
    figures(R, S)
    json.dump(S, open(f"{DIR}/fss_analysis.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    for k, v in S["decisions"].items():
        print(k, json.dumps(v, default=float)[:1500], flush=True)
    print("figures saved", flush=True)
