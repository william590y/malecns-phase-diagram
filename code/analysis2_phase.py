"""Complete simulated-vs-theoretical (g, r) phase diagram of the stochastic Wilson-Cowan model on MaleCNS (raw weights, h = 0),
plus the driven (g, h) plane at r = 1.

PHASE CRITERIA (identical for simulation and theory; T = 3000, the stage-1 / pd2 run length):
  S starts   where the mean module lifetime tau reaches 200
  on (S->L)  where tau reaches T = 3000 (sustained onset); the half-survival point g_half (fraction of replicas alive at T
             crosses 1/2, i.e. tau = T/ln 2 for an exponential) is reported alongside
  D          where the participation ratio of the activity reaches N/4, among sustained points
SIMULATION (every (g, r) point of every decay run with norm raw, h = 0, dt 0.2, beta 1, fully active start, T >= 3000):
  per replica: extinction time = first recorded time with surv = 0; replicas alive at T_c are censored at T_c, where T_c =
  the last recorded time <= 3000 (T_c = 3000 for the T = 3000 runs; runs with T = 8000 / 30000 are truncated at T_c, so
  every point uses the same observation window). tau_sim = censored-exponential MLE = (sum of min(t_i, T_c)) / deaths, with
  the time-terminated chi-square 95% CI [2E/chi2_.975(2d+2), 2E/chi2_.025(2d)]; with 0 deaths tau_sim = inf and the one-sided
  95% lower bound 2E/chi2_.95(2) is given. alive_T = fraction of replicas alive at T_c.
  PR = (sum occ)^2 / (sum occ^2) / N, occ = per-neuron occupancy over the second half of the run (replica mean), only from
  T = 3000 runs (the occupancy window must be the same [1500, 3000]); no per-replica occupancy is stored, so PR has no
  replica bootstrap.
  Points at the same g (relative difference < 1e-4) from different runs are pooled replica by replica.
  BOUNDARIES: ln tau (0 deaths regularised as E/0.5 for this purpose only), alive_T and PR are each made monotone in g by
  weighted isotonic regression (weights deaths + 0.5, B, 1; physically all three rise with g and the fit only removes
  sampling reversals), then the first crossing is interpolated linearly in ln g (ln tau, alive_T, PR on the y axis).
  "Sustained" = points at or above the tau = T crossing. If the first sustained point already has PR >= N/4, D coincides
  with the onset (flag "at_onset").
  UNCERTAINTY: (i) grid bracket = the two grid points around the crossing; (ii) bootstrap over replicas (NBOOT resamples,
  each point resampled independently, whole procedure repeated) -> 2.5-97.5%; total = envelope of (i) and (ii).
  The raw (non-isotonic) first crossing is also stored as a check.
THEORY:
  S, on: crossings of logtau_r (natural log of the predicted mean extinction time, exact birth-death recursion of
  theory_branch_gpu.py) with ln 200 and ln 3000, first crossing on the merged dense rows (branch_<kind>_dense_part*.json),
  interpolated linearly in (ln g, ln tau). If the first dense row is already above the threshold (configuration null,
  small r), the crossing is found with the same recursion on inputs interpolated between g_lin (R = 1 exactly) and the first
  row (R linear in g, K and D from the first row): flag "extrap_inputs". A variant without the floor(K) truncation jitter
  of the recursion ("smoothK": ln tau interpolated linearly in K between floor(K) and floor(K)+1 states) is stored as a check.
  D: first g where the QMF fixed point is alive (rho_T > 1e-9) and its participation ratio >= N/4, interpolated in ln g
  (qmf_<kind>.npz, qmf_real_dense.npz, pd2/qmf_<kind>_r*.npz).
  tau_theory at a simulated g: logtau_r interpolated in ln g within the dense rows (extrapolated inputs below the first
  row, nan outside).
DRIVEN PLANE (r = 1, real graph): pd2/drive_h*_real.npz where present, else pd/drive_h*_real.npz. late = recorded t >=
  T/2; rho = mean over replicas and late snapshots; chi_rep = N var_replicas(late mean) (ddof 1); F_rep = chi_rep / rho
  (same definitions as analysis2_driven.py). Peak = argmax of F_rep in 0.1 <= g <= 1, replica bootstrap of the argmax;
  "resolved" = interior maximum and P_boot(F(peak) > F(window's last point)) >= 0.95 (analysis2_driven.py's criterion).
  Theory fluctuation peak (P5, theory page section 7): beta h n p_est tau(g) = 1 with n = 526 (antennal-lobe module),
  p_est = 1 - 1/R, i.e. h_peak(g) = 1 / (n p_est(g) tau(g)), for tau = theory r = 1 and for the measured h = 0 lifetime
  (results/analysis2/driven.json emp_tau, from results/fine_decay_raw_r1.npz and results/pd/long_real.npz).
OUTPUTS: results/analysis2/phase_complete.json, phase_complete_table.md (agreement tables; phase_complete.md is written
  by hand from these), flow_sim_boundaries.json; figs/pd2/phase_real.png, phase_nulls.png, phase_driven.png,
  tau_sim_vs_theory.png.
"""
import glob, json, os, re
import numpy as np
from scipy import stats
from scipy.optimize import isotonic_regression
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = "/MALECNS_ROOT"
RES = f"{ROOT}/results"; OUT = f"{RES}/analysis2"; FIG = f"{ROOT}/figs/pd2"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
N = 167216; T_REF = 3000.0; TAU_S = 200.0; PR_D = 0.25; NBOOT = 2000
KINDS = ["real", "null", "wnull"]
KIND_NAME = {"real": "MaleCNS", "null": "configuration null (wiring shuffled)", "wnull": "weight-shuffled null"}
rng = np.random.default_rng(20261109)
fmt = lambda x, p=3: "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}g}"


# ----------------------------------------------------------------------------------------------------------- simulation
def run_kind(m):
    if not m.get("null"):
        return "real"
    return "wnull" if m.get("null_kind") == "weights" else "null"


SIM_FILES = (sorted(glob.glob(f"{RES}/pd/decay_r*_*.npz")) + sorted(glob.glob(f"{RES}/pd2/onset_*_r*.npz")) +
             sorted(glob.glob(f"{RES}/pd2/deloc_*_r*.npz")) +
             [f"{RES}/{x}.npz" for x in ["decay_raw_r1", "decay_raw_r1_null", "fine_decay_raw_r1", "fine_decay_raw_r1_null",
                                         "upper_decay_raw_r1"]] + [f"{RES}/pd/long_real.npz", f"{RES}/pd/long_null.npz"])


def load_sim():
    pts = {}                                   # (kind, r) -> list of point dicts
    used = []
    for p in SIM_FILES:
        if not os.path.exists(p):
            continue
        d = np.load(p)
        m = json.loads(str(d["meta"]))
        ok = (m.get("exp") == "decay" and m.get("norm") == "raw" and float(m.get("h", 0)) == 0.0 and
              abs(float(m.get("dt", 0.2)) - 0.2) < 1e-9 and float(m.get("beta", 1)) == 1.0 and float(m["T"]) >= T_REF - 1e-6)
        if not ok:
            continue
        t, g, surv = d["t"], d["g"], d["surv"]
        win = t <= T_REF + 1e-6
        Tc = float(t[win].max())
        occ = d["occ"].astype(np.float64) if (abs(float(m["T"]) - T_REF) < 1e-6 and "occ" in d.files) else None
        kind, r = run_kind(m), float(m["r"])
        lab = os.path.relpath(p, RES)
        used.append({"file": lab, "kind": kind, "r": r, "T": float(m["T"]), "B": int(m["B"]), "n_g": int(len(g)),
                     "g_range": [float(g.min()), float(g.max())], "T_c": Tc})
        for i, gg in enumerate(g):
            s = surv[i][win]                          # [nt, B]
            dead = s == 0
            died = dead.any(0)
            ext = np.where(died, t[win][np.argmax(dead, 0)], Tc)
            pr = None
            if occ is not None:
                o = occ[i]
                pr = float(o.sum() ** 2 / (o ** 2).sum() / N) if o.sum() > 0 else 0.0
            pts.setdefault((kind, r), []).append({"g": float(gg), "times": ext.astype(float), "died": died.astype(bool),
                                                  "pr": pr, "B": int(s.shape[1]), "src": lab})
    merged = {}
    for key, lst in pts.items():
        lst.sort(key=lambda x: x["g"])
        out = []
        for x in lst:
            if out and abs(x["g"] / out[-1]["g"] - 1) < 1e-4:
                y = out[-1]
                y["times"] = np.r_[y["times"], x["times"]]; y["died"] = np.r_[y["died"], x["died"]]
                if x["pr"] is not None:
                    y["prs"].append((x["pr"], x["B"]))
                y["B"] += x["B"]; y["src"] += "+" + x["src"]
            else:
                out.append({"g": x["g"], "times": x["times"], "died": x["died"], "B": x["B"], "src": x["src"],
                            "prs": [(x["pr"], x["B"])] if x["pr"] is not None else []})
        for y in out:
            y["pr"] = (sum(a * b for a, b in y["prs"]) / sum(b for _, b in y["prs"])) if y["prs"] else None
            d_, E = int(y["died"].sum()), float(y["times"].sum())
            y["deaths"], y["E"] = d_, E
            y["tau"] = E / d_ if d_ > 0 else float("inf")
            y["tau_reg"] = E / max(d_, 0.5)
            if d_ > 0:
                y["tau_ci"] = [2 * E / stats.chi2.ppf(0.975, 2 * d_ + 2), 2 * E / stats.chi2.ppf(0.025, 2 * d_)]
            else:
                y["tau_ci"] = [2 * E / stats.chi2.ppf(0.95, 2), float("inf")]
            y["alive"] = float(1 - y["died"].mean())
        merged[key] = out
    return merged, used


def iso(y, w):
    return isotonic_regression(np.asarray(y, float), weights=np.asarray(w, float), increasing=True).x


def first_cross(x, y, thr, log_y=False):
    """First upward crossing of thr by y(x), x increasing; interpolated linearly in (ln x, y or ln y)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    yy = np.log(np.maximum(y, 1e-300)) if log_y else y
    tt = np.log(thr) if log_y else thr
    hit = np.where(yy >= tt)[0]
    if len(hit) == 0:
        return {"g": None, "bracket": [float(x[-1]), None], "status": "none"}
    k = int(hit[0])
    if k == 0:
        return {"g": None, "bracket": [None, float(x[0])], "status": "below_grid"}
    fr = (tt - yy[k - 1]) / (yy[k] - yy[k - 1]) if yy[k] > yy[k - 1] else 1.0
    g = float(np.exp(np.log(x[k - 1]) + fr * np.log(x[k] / x[k - 1])))
    return {"g": g, "bracket": [float(x[k - 1]), float(x[k])], "status": "ok", "k": k}


def sim_boundaries(P, d=None, tr=None, al=None):
    """Boundaries from one realisation of (deaths, exposure, alive) per point; P = point list (for g, B, PR)."""
    g = np.array([p["g"] for p in P])
    if d is None:
        d = np.array([p["deaths"] for p in P], float); E = np.array([p["E"] for p in P])
        tr = E / np.maximum(d, 0.5); al = np.array([p["alive"] for p in P])
    lt = iso(np.log(tr), d + 0.5)
    S = first_cross(g, lt, np.log(TAU_S))
    on = first_cross(g, lt, np.log(T_REF))
    half = first_cross(g, iso(al, [p["B"] for p in P]), 0.5)
    D = {"g": None, "bracket": [None, None], "status": "no_sustained_points"}
    if on["status"] in ("ok", "below_grid"):
        k0 = on.get("k", 0)
        sel = [j for j in range(k0, len(P)) if P[j]["pr"] is not None]
        if sel:
            gs = g[sel]; prs = iso([P[j]["pr"] for j in sel], np.ones(len(sel)))
            D = first_cross(gs, prs, PR_D)
            if D["status"] == "below_grid":         # already delocalized at the first sustained point
                D = {"g": on["g"], "bracket": on["bracket"], "status": "at_onset"}
            elif D["status"] == "none":
                D["status"] = "none"
    return {"S": S, "on": on, "half": half, "D": D}


def sim_raw_check(P):
    g = np.array([p["g"] for p in P]); tr = np.array([p["tau_reg"] for p in P])
    return {"S": first_cross(g, tr, TAU_S, log_y=True)["g"], "on": first_cross(g, tr, T_REF, log_y=True)["g"],
            "half": first_cross(g, [p["alive"] for p in P], 0.5)["g"]}


def bootstrap(P, nboot=NBOOT):
    nb = len(P)
    D = np.zeros((nboot, nb)); E = np.zeros((nboot, nb)); A = np.zeros((nboot, nb))
    for j, p in enumerate(P):
        idx = rng.integers(0, p["B"], (nboot, p["B"]))
        dd = p["died"][idx]; D[:, j] = dd.sum(1); E[:, j] = p["times"][idx].sum(1); A[:, j] = 1 - dd.mean(1)
    res = {k: [] for k in ["S", "on", "half", "D"]}
    for b in range(nboot):
        bb = sim_boundaries(P, D[b], E[b] / np.maximum(D[b], 0.5), A[b])
        for k in res:
            res[k].append(bb[k]["g"] if bb[k]["g"] is not None else np.nan)
    return {k: np.array(v) for k, v in res.items()}


def summarize(est, boot):
    out = {"g": est["g"], "status": est["status"], "bracket": est["bracket"]}
    v = boot[np.isfinite(boot)]
    out["boot_frac_defined"] = float(np.isfinite(boot).mean())
    if len(v) >= 20:
        lo, hi = np.percentile(v, [2.5, 97.5])
        out["boot_ci"] = [float(lo), float(hi)]
    else:
        out["boot_ci"] = None
    if est["g"] is not None:
        los = [x for x in [est["bracket"][0], out["boot_ci"][0] if out["boot_ci"] else None] if x is not None]
        his = [x for x in [est["bracket"][1], out["boot_ci"][1] if out["boot_ci"] else None] if x is not None]
        out["total"] = [float(min(los)) if los else None, float(max(his)) if his else None]
    else:
        out["total"] = None
    return out


# --------------------------------------------------------------------------------------------------------------- theory
A_EFF = (1 - np.exp(-0.2)) / 0.2


def log_tau(lam, K, n_top=None):
    """Exact mean extinction time from the top state of the logistic birth-death chain (theory_branch_gpu.py)."""
    n_top = int(np.floor(K)) if n_top is None else int(n_top)
    if n_top < 1 or lam <= 0:
        return -np.inf
    n = np.arange(1, n_top + 1, dtype=float)
    lb = lam * n * (1 - n / K); lb[-1] = 0.0
    if (lb[:-1] <= 0).any():                       # n_top above K: drop non-positive birth rates (smoothK variant only)
        lb = np.where(lb > 0, lb, 0.0)
    ld = A_EFF * n
    logD = np.empty(n_top); logD[-1] = -np.log(ld[-1])
    for k in range(n_top - 2, -1, -1):
        logD[k] = (np.logaddexp(0.0, np.log(lb[k]) + logD[k + 1]) if lb[k] > 0 else 0.0) - np.log(ld[k])
    return float(logsumexp(logD))


def log_tau_smooth(lam, K):
    """Same chain, truncation level interpolated: (1 - f) * [top state floor(K)] + f * [top state floor(K) + 1], f = K - floor(K),
    same lam and K in both (the extra top state has its birth rate set to 0 as in the original). Continuous in K."""
    k0 = int(np.floor(K)); f = K - k0
    a = log_tau(lam, K, k0)
    b = log_tau(lam, K, k0 + 1) if f > 0 else a
    return float((1 - f) * a + f * b)


def load_branch(kind):
    parts = sorted(glob.glob(f"{RES}/pd/branch_{kind}_dense_part*.json"))
    if not parts:
        return None
    js = [json.load(open(p)) for p in parts]
    rows = sorted([x for j in js for x in j["rows"]], key=lambda x: x["g"])
    th = {"n_parts": len(parts), "g_lin": js[0]["g_lin"], "alpha_eff": js[0]["alpha_eff"], "r": js[0]["r"],
          "G": np.array([x["g"] for x in rows]), "R": np.array([x["R"] for x in rows]),
          "lam": np.array([x["lam"] for x in rows]), "rows": rows}
    th["lt"] = {r: np.array([x.get(f"logtau_r{r:g}", np.nan) if x.get(f"logtau_r{r:g}") is not None else np.nan
                             for x in rows], float) for r in th["r"]}
    th["K"] = {r: np.array([x[f"K_r{r:g}"] for x in rows]) for r in th["r"]}
    # recursion check on 3 rows
    th["recursion_check"] = float(max(abs(log_tau(rows[i]["lam"], rows[i]["K_r1"]) - rows[i]["logtau_r1"])
                                      for i in [0, len(rows) // 2, len(rows) - 1]))
    th["lt_smooth"] = {r: np.array([log_tau_smooth(x["lam"], x[f"K_r{r:g}"]) for x in rows]) for r in th["r"]}
    th["KD0"] = np.array([1 / (x["u2v"] * x["uv2"]) for x in rows])
    th["lt_D0"] = np.array([log_tau(x["lam"], 1 / (x["u2v"] * x["uv2"])) for x in rows])
    return th


def theory_extrap(th, r, g):
    """ln tau for g_lin < g < G[0]: R linear in g from 1 at g_lin to R[0], K from the first row (same recursion)."""
    g0, R0, K0 = th["G"][0], th["R"][0], th["K"][r][0]
    R = 1 + (R0 - 1) * (g - th["g_lin"]) / (g0 - th["g_lin"])
    return log_tau(R * th["alpha_eff"], K0)


def theory_lt_at(th, r, g):
    G = th["G"]
    if g <= th["g_lin"] or g > G[-1] * (1 + 1e-9):
        return np.nan
    if g < G[0]:
        return theory_extrap(th, r, g)
    return float(np.interp(np.log(g), np.log(G), th["lt"][r]))


def theory_at(th, key, g, r=None):
    """lam or K_r (or K with D = 0: 1/(<u^2,v><u,v^2>)) at g, interpolated in ln g; first row below the grid."""
    G = th["G"]
    if g <= th["g_lin"] or g > G[-1] * (1 + 1e-9):
        return np.nan
    if key == "lam":
        y = th["lam"]
        if g < G[0]:
            return float((1 + (th["R"][0] - 1) * (g - th["g_lin"]) / (G[0] - th["g_lin"])) * th["alpha_eff"])
    elif key == "K":
        y = th["K"][r]
    else:
        y = np.array([1 / (x["u2v"] * x["uv2"]) for x in th["rows"]])
    return float(np.interp(np.log(g), np.log(G), y))


def k_needed(lam, lt_target, kmax=50000.0):
    """K for which the theory's own recursion (smooth in K) with the theory's lam(g) gives ln tau = lt_target."""
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


def theory_cross(th, r, thr, smooth=False):
    lt = th["lt_smooth"][r] if smooth else th["lt"][r]
    c = first_cross(th["G"], lt, np.log(thr))
    c.pop("k", None)
    if c["status"] == "below_grid":
        gg = np.geomspace(th["g_lin"] * (1 + 1e-6), th["G"][0], 400)
        lt2 = np.array([theory_extrap(th, r, x) for x in gg])
        c2 = first_cross(gg, lt2, np.log(thr)); c2.pop("k", None)
        c = {"g": c2["g"], "bracket": [float(th["g_lin"]), float(th["G"][0])], "status": "extrap_inputs"}
    return c


def load_qmf(kind):
    files = [f"{RES}/pd/qmf_{kind}.npz"] + ([f"{RES}/pd/qmf_real_dense.npz"] if kind == "real" else []) + \
        sorted(glob.glob(f"{RES}/pd2/qmf_{kind}_r*.npz"))
    per_r = {}
    for f in files:
        if not os.path.exists(f):
            continue
        q = np.load(f)
        for i, r in enumerate(q["r"]):
            per_r.setdefault(round(float(r), 4), []).append({"file": os.path.relpath(f, RES), "G": q["g"],
                                                             "PR": q["PR"][i, 0], "rho": q["rho_T"][i, 0],
                                                             "mu": float(q["mu"][i, 0])})
    out = {}
    for r, lst in sorted(per_r.items()):
        res = []
        for x in lst:
            G, PR, rho = x["G"], x["PR"], x["rho"]
            alive = rho > 1e-9
            k = [j for j in range(len(G)) if alive[j] and PR[j] >= PR_D]
            onset = float(G[np.argmax(alive)]) if alive.any() else None
            if not k:
                res.append({"g": None, "bracket": [float(G[-1]), None], "status": "none", "file": x["file"],
                            "qmf_onset": onset, "mu": x["mu"], "PR_max": float(PR[alive].max()) if alive.any() else 0.0})
                continue
            j = k[0]
            if j > 0 and alive[j - 1]:
                fr = (PR_D - PR[j - 1]) / (PR[j] - PR[j - 1])
                g = float(np.exp(np.log(G[j - 1]) + fr * np.log(G[j] / G[j - 1]))); st = "ok"
            else:
                g = float(G[j]); st = "jump_at_qmf_onset"
            res.append({"g": g, "bracket": [float(G[j - 1]) if j > 0 else None, float(G[j])], "status": st,
                        "file": x["file"], "qmf_onset": onset, "mu": x["mu"]})
        best = res[-1]
        best = dict(best); best["all_files"] = [{"file": z["file"], "g": z["g"]} for z in res]
        out[r] = best
    return out


# ------------------------------------------------------------------------------------------------------------- run all
sim, used = load_sim()
TH = {k: load_branch(k) for k in KINDS}
QMF = {k: load_qmf(k) for k in KINDS}
result = {"definitions": __doc__, "N": N, "T": T_REF, "tau_S": TAU_S, "PR_D": PR_D, "nboot": NBOOT,
          "inputs": {"simulation": used,
                     "theory_parts": {k: (TH[k]["n_parts"] if TH[k] else 0) for k in KINDS}},
          "g_lin": {k: (TH[k]["g_lin"] if TH[k] else None) for k in KINDS},
          "recursion_check_max_abs_diff": {k: (TH[k]["recursion_check"] if TH[k] else None) for k in KINDS},
          "theory": {}, "sim": {}, "agreement": {}}

for kind in KINDS:
    th = TH[kind]
    tb = {}
    rs_th = sorted(set(([float(r) for r in th["r"]] if th else []) + [float(r) for r in QMF[kind]]))
    for r in rs_th:
        e = {"r": r}
        if th and r in th["r"]:
            e["S"] = theory_cross(th, r, TAU_S); e["on"] = theory_cross(th, r, T_REF)
            e["S_smoothK"] = theory_cross(th, r, TAU_S, smooth=True)["g"]; e["on_smoothK"] = theory_cross(th, r, T_REF, smooth=True)["g"]
            lt = th["lt"][r]
            e["max_logtau"] = float(np.nanmax(lt))
            e["nonmonotone_steps"] = int((np.diff(lt) < -1e-9).sum())
        q = QMF[kind].get(round(r, 4))
        if q is not None:
            e["D"] = {k: q[k] for k in ["g", "bracket", "status", "qmf_onset", "mu", "all_files"] if k in q}
        tb[f"{r:g}"] = e
    result["theory"][kind] = tb
    if th:   # variant without any coincidence term (D = 0): r-independent, isolates what D does
        result["theory_no_coincidence"] = result.get("theory_no_coincidence", {})
        def cross_D0(thr):
            c = first_cross(th["G"], th["lt_D0"], np.log(thr))
            if c["status"] == "below_grid":      # same input extrapolation as theory_cross, with K = K_D0 of the first row
                gg = np.geomspace(th["g_lin"] * (1 + 1e-6), th["G"][0], 400)
                Rg = 1 + (th["R"][0] - 1) * (gg - th["g_lin"]) / (th["G"][0] - th["g_lin"])
                lt2 = np.array([log_tau(R_ * th["alpha_eff"], th["KD0"][0]) for R_ in Rg])
                c = {"g": first_cross(gg, lt2, np.log(thr))["g"], "status": "extrap_inputs"}
            return c
        cS = cross_D0(TAU_S); cO = cross_D0(T_REF)
        result["theory_no_coincidence"][kind] = {"S": cS["g"], "S_status": cS["status"], "on": cO["g"],
                                                 "on_status": cO["status"],
                                                 "K_D0_range": [float(th["KD0"].min()), float(th["KD0"].max())]}

    sb = {}
    for (k_, r), P in sorted(sim.items()):
        if k_ != kind:
            continue
        est = sim_boundaries(P)
        boot = bootstrap(P)
        e = {"r": r, "n_points": len(P), "sources": sorted(set(s for p in P for s in p["src"].split("+"))),
             "g_range": [P[0]["g"], P[-1]["g"]], "raw_first_crossing": sim_raw_check(P)}
        for key in ["S", "on", "half", "D"]:
            e[key] = summarize(est[key], boot[key])
        if e["D"]["status"] in ("none", "no_sustained_points"):
            e["D"]["note"] = "no sustained point with PR >= N/4 up to g = %.3g" % max(p["g"] for p in P if p["pr"] is not None)
        if e["D"]["status"] == "at_onset":
            e["D"]["boot_ci"] = e["on"]["boot_ci"]; e["D"]["total"] = e["on"]["total"]
        e["D"]["bootstrap_note"] = "PR has no per-replica data; bootstrap only moves the set of sustained points"
        pts = []
        for p in P:
            ltt = theory_lt_at(th, r, p["g"]) if (th and r in th["r"]) else np.nan
            q = {"g": p["g"], "B": p["B"], "deaths": p["deaths"], "tau_sim": p["tau"], "tau_ci": p["tau_ci"],
                 "alive_T": p["alive"], "PR": p["pr"], "ln_tau_theory": float(ltt) if np.isfinite(ltt) else None,
                 "tau_theory": float(np.exp(ltt)) if (np.isfinite(ltt) and ltt < 700) else None, "src": p["src"]}
            # module size that would reproduce tau_sim with the theory's lam(g) (slow window, >= 3 deaths only)
            if th and r in th["r"] and p["deaths"] >= 3 and np.isfinite(p["tau"]) and p["tau"] > TAU_S:
                q["K_needed"] = k_needed(theory_at(th, "lam", p["g"]), np.log(p["tau"]))
                q["K_theory"] = theory_at(th, "K", p["g"], r); q["K_theory_D0"] = theory_at(th, "KD0", p["g"])
            pts.append(q)
        e["points"] = pts
        sb[f"{r:g}"] = e
        print(f"{kind:5s} r={r:<6g} S {fmt(e['S']['g'])} {e['S']['total']}  on {fmt(e['on']['g'])} {e['on']['total']}  "
              f"half {fmt(e['half']['g'])}  D {fmt(e['D']['g'])} ({e['D']['status']})", flush=True)
    result["sim"][kind] = sb

    # agreement rows: every r with a simulation
    ag = []
    for rk, e in sb.items():
        t = result["theory"][kind].get(rk, {})
        row = {"r": e["r"]}
        for key, tkey in [("S", "S"), ("on", "on"), ("D", "D")]:
            ts = t.get(tkey, {}).get("g") if t.get(tkey) else None
            ss = e[key]["g"]
            row[key] = {"theory": ts, "theory_status": t.get(tkey, {}).get("status") if t.get(tkey) else None,
                        "sim": ss, "sim_total": e[key]["total"], "sim_boot_ci": e[key]["boot_ci"],
                        "sim_status": e[key]["status"],
                        "ratio_theory_over_sim": (ts / ss) if (ts and ss) else None}
            tot = e[key]["total"]
            row[key]["theory_inside_sim_total"] = bool(ts is not None and tot is not None and tot[0] is not None and
                                                       tot[1] is not None and tot[0] <= ts <= tot[1])
        row["half"] = e["half"]["g"]
        # lifetime ratio over the slow window (finite tau_sim > 200, theory defined)
        sel = [p for p in e["points"] if p["deaths"] >= 3 and np.isfinite(p["tau_sim"]) and p["tau_sim"] > TAU_S
               and p["ln_tau_theory"] is not None]
        lr = [np.log(p["tau_sim"]) - p["ln_tau_theory"] for p in sel]
        row["tau_ratio_sim_over_theory"] = {"n": len(lr), "geomean": float(np.exp(np.mean(lr))) if lr else None,
                                            "min": float(np.exp(min(lr))) if lr else None,
                                            "max": float(np.exp(max(lr))) if lr else None,
                                            "g_range": [sel[0]["g"], sel[-1]["g"]] if sel else None,
                                            "selection": "deaths >= 3, 200 < tau_sim (finite), theory defined"}
        kn = [p for p in sel if p.get("K_needed")]
        row["K_needed"] = {"n": len(kn), "median": float(np.median([p["K_needed"] for p in kn])) if kn else None,
                           "min": float(min(p["K_needed"] for p in kn)) if kn else None,
                           "max": float(max(p["K_needed"] for p in kn)) if kn else None,
                           "K_theory_median": float(np.median([p["K_theory"] for p in kn])) if kn else None,
                           "K_theory_D0_median": float(np.median([p["K_theory_D0"] for p in kn])) if kn else None}
        ag.append(row)
    result["agreement"][kind] = sorted(ag, key=lambda x: x["r"])


# ------------------------------------------------------------------------------------------------------------- driven
def driven():
    out = {"runs": {}}
    for h in ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "null_1e-3"]:
        hh, kk = (h.split("_")[1], "null") if h.startswith("null") else (h, "real")
        p2, p1 = f"{RES}/pd2/drive_h{hh}_{kk}.npz", f"{RES}/pd/drive_h{hh}_{kk}.npz"
        p = p2 if os.path.exists(p2) else p1
        wlo, whi = (0.1, 1.0) if kk == "real" else (0.5, 5.0)
        d = np.load(p); m = json.loads(str(d["meta"]))
        t, g, rho = d["t"], d["g"], d["rho"]
        late = t >= t[-1] / 2
        x = rho[:, late, :]                               # [g, tl, B]
        rm = x.mean(1)                                    # [g, B]
        rh = rm.mean(1); chi = N * rm.var(1, ddof=1); F = chi / rh
        win = np.where((g >= wlo - 1e-9) & (g <= whi + 1e-9))[0]
        k = int(win[np.argmax(F[win])])
        B = rm.shape[1]
        bk, above = [], []
        for _ in range(NBOOT):
            idx = rng.integers(0, B, B); rb = rm[:, idx]
            Fb = N * rb.var(1, ddof=1) / rb.mean(1)
            bk.append(g[win[np.argmax(Fb[win])]]); above.append(Fb[k] > Fb[win[-1]])
        bk = np.array(bk)
        interior = 0 < k - win[0] < len(win) - 1
        p_above = float(np.mean(above))
        out["runs"][h] = {"file": os.path.relpath(p, RES), "h": float(m["h"]), "B": int(m["B"]), "T": float(m["T"]),
                          "g": g.tolist(), "rho": rh.tolist(), "chi_rep": chi.tolist(), "F_rep": F.tolist(),
                          "peak_g": float(g[k]), "peak_F": float(F[k]), "peak_interior": bool(interior),
                          "p_boot_peak_above_window_end": p_above,
                          "peak_resolved": bool(interior and p_above >= 0.95),
                          "peak_cell": [float(g[k - 1]) if k > 0 else None, float(g[k + 1]) if k + 1 < len(g) else None],
                          "boot_frac_at_peak": float(np.mean(np.isclose(bk, g[k]))),
                          "boot_range_95": [float(np.percentile(bk, 2.5)), float(np.percentile(bk, 97.5))],
                          "window": [wlo, whi], "graph": kk,
                          "F_max_above_window": [float(F[g > whi].max()), float(g[g > whi][np.argmax(F[g > whi])])]
                          if (g > whi).any() else None}
    th = TH["real"]
    gg = np.geomspace(th["g_lin"] * 1.001, th["G"][-1], 600)
    R = np.array([np.interp(np.log(x), np.log(th["G"]), th["R"]) if x >= th["G"][0] else
                  1 + (th["R"][0] - 1) * (x - th["g_lin"]) / (th["G"][0] - th["g_lin"]) for x in gg])
    pe = 1 - 1 / R
    lt1 = np.array([theory_lt_at(th, 1.0, x) for x in gg])
    n = 526
    out["p5_theory_r1"] = {"g": gg.tolist(), "h": (1 / (n * pe * np.exp(lt1))).tolist(), "n": n}
    dj = json.load(open(f"{OUT}/driven.json"))
    emp = sorted(dj["emp_tau"]["real"], key=lambda z: z["g"])
    eg = np.array([z["g"] for z in emp]); et = np.array([z["tau"] for z in emp])
    sel = (gg >= eg.min()) & (gg <= eg.max())
    ltm = np.interp(np.log(gg[sel]), np.log(eg), np.log(et))
    out["p5_measured_tau"] = {"g": gg[sel].tolist(), "h": (1 / (n * pe[sel] * np.exp(ltm))).tolist(), "n": n,
                              "tau_source": "results/analysis2/driven.json emp_tau.real"}
    out["p5_points_driven_json"] = [{"h": z["h"], "tau": z["tau"], "g_peak": z["g_peak"]} for z in dj["theory"]["real"]
                                    if z["n"].startswith("n=526") and z["x_target"] == 1.0]
    return out


result["driven"] = driven()

# ------------------------------------------------------------------------------------------------------- flow page data
flow_rows = []
for rk, e in sorted(result["sim"]["real"].items(), key=lambda kv: float(kv[0])):
    def gv(key):
        return e[key]["g"] if e[key]["status"] in ("ok", "at_onset") else None
    flow_rows.append({"r": e["r"], "S": gv("S"), "on": gv("on"), "D": gv("D"),
                      "S_err": e["S"]["total"], "on_err": e["on"]["total"], "D_err": e["D"]["total"],
                      "half": gv("half"),
                      "D_not_reached_up_to": (max(p["g"] for p in e["points"] if p["PR"] is not None)
                                              if gv("D") is None else None)})
json.dump({"source": "code/analysis2_phase.py -> results/analysis2/phase_complete.json (sim.real)",
           "definitions": "S: tau_sim = 200; on: tau_sim = T = 3000; D: PR(occ) = N/4 among sustained points; "
                          "crossings interpolated in ln g after isotonic smoothing; *_err = envelope of grid bracket and "
                          "replica-bootstrap 95% CI; half = alive-at-T fraction 1/2; null = boundary not reached in the "
                          "simulated g range (D_not_reached_up_to = largest simulated g with a PR)", "T": T_REF, "graph": "real", "rows": flow_rows},
          open(f"{OUT}/flow_sim_boundaries.json", "w"), indent=1)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(result), open(f"{OUT}/phase_complete.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------------------- tables
L = ["# Agreement tables (generated by code/analysis2_phase.py)", ""]
for kind in KINDS:
    L.append(f"## {KIND_NAME[kind]} (g_lin = {fmt(result['g_lin'][kind], 4)})\n")
    L.append("| r | S theory | S sim [total] | ratio | on theory | on sim [total] | ratio | g_half sim | D theory (QMF) | "
             "D sim [total] | ratio | τ_sim/τ_th geomean (n, range) | K needed median (range) | K_eff theory (D = 0) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in result["agreement"][kind]:
        def cell(x, key):
            s = x["sim"]; tot = x["sim_total"]
            st = x["sim_status"]
            if s is None:
                b = result["sim"][kind][f"{row['r']:g}"][key]["bracket"]
                return {"none": f"> {fmt(b[0])}", "below_grid": f"< {fmt(b[1])}"}.get(st, st)
            tt = f" [{fmt(tot[0])}, {fmt(tot[1])}]" if tot and tot[0] is not None and tot[1] is not None else ""
            return f"{fmt(s)}{tt}" + (" (at onset)" if st == "at_onset" else "")
        def tcell(x):
            if x["theory"] is None:
                return "none" if x["theory_status"] in ("none", None) else x["theory_status"]
            return fmt(x["theory"]) + (" *" if x["theory_status"] == "extrap_inputs" else "")
        tr = row["tau_ratio_sim_over_theory"]
        trs = f"{fmt(tr['geomean'])} ({tr['n']}, {fmt(tr['min'])}–{fmt(tr['max'])})" if tr["n"] else "–"
        L.append(f"| {row['r']:g} | {tcell(row['S'])} | {cell(row['S'], 'S')} | {fmt(row['S']['ratio_theory_over_sim'])} | "
                 f"{tcell(row['on'])} | {cell(row['on'], 'on')} | {fmt(row['on']['ratio_theory_over_sim'])} | {fmt(row['half'])} | "
                 f"{tcell(row['D'])} | {cell(row['D'], 'D')} | {fmt(row['D']['ratio_theory_over_sim'])} | {trs} | "
                 + (f"{fmt(row['K_needed']['median'])} ({fmt(row['K_needed']['min'])}–{fmt(row['K_needed']['max'])}) | "
                    f"{fmt(row['K_needed']['K_theory_median'])} ({fmt(row['K_needed']['K_theory_D0_median'])}) |"
                    if row['K_needed']['n'] else "– | – |"))
    L.append("")
    if any(rk not in result["sim"][kind] for rk in result["theory"][kind]):
        L.append("Theory-only rows (r without a simulation): " + "; ".join(
        f"r={rk}: S {fmt(v.get('S', {}).get('g'))}, on {fmt(v.get('on', {}).get('g'))}, D {fmt(v.get('D', {}).get('g'))}"
            for rk, v in result["theory"][kind].items() if rk not in result["sim"][kind]))
    L.append("")
open(f"{OUT}/phase_complete_table.md", "w").write("\n".join(L))
print("wrote", f"{OUT}/phase_complete.json", f"{OUT}/phase_complete_table.md", f"{OUT}/flow_sim_boundaries.json")
