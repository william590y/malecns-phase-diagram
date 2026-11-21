"""Critic: independent recomputation of the (g, r) phase boundaries from the raw files, to check analysis2_phase.py.

Deliberately different methods from analysis2_phase.py wherever a choice exists:
SIMULATION
  - death time of a replica = geometric midpoint between the last record with activity and the first silent record
    (analysis2_phase.py uses the first silent record; this removes its <= 8-10% upward bias in tau);
    activity read from rho > 0 (not from surv).
  - same pooling rule (norm raw, h = 0, exp decay, dt 0.2, T >= 3000, truncated at the last record <= 3000).
  - boundary g(tau = thr) from a LOCAL censored-exponential likelihood fit  ln tau(g) = a + b ln g  over the pooled points
    within a +-factor window around the raw crossing (Newton on the convex negative log-likelihood; censored replicas
    enter through their exposure), instead of isotonic regression + linear interpolation. 95% CI: replica bootstrap
    (NB resamples, window fixed).
  - D: first g at or above the onset where the B-weighted PR of the occupancy (T = 3000 runs only) reaches N/4,
    interpolated linearly in (ln g, ln PR).
  - OPERATIONAL phase.py classification per single file (B fixed, not pooled): life = time the last replica died;
    S_op = first g with life >= 200 or 0 < surv_T < 1;  L_op = first g from which surv_T = 1 at every larger g;
    D_op = first g with surv_T = 1 and PR >= N/4. These depend on B (max of B lifetimes), unlike tau.
THEORY
  - extinction time from the closed-form sum  tau = sum_k sum_{j>=k} (1/d_j) prod_{i=k}^{j-1} (b_i/d_i)  in log space
    (not the downward recursion), checked against every stored logtau in every dense row.
  - crossings two ways: (i) output interpolation of ln tau in ln g between dense rows (as analysis2_phase.py);
    (ii) input interpolation: lam(g) and K_r(g) interpolated in ln g, ln tau evaluated on a fine grid (shows the
    floor(K) jitter and the dependence on the 7.3% / 5.3% row spacing).
  - the same crossings from the OLDER theory runs (branch_<kind>.json narrow 'branching' and 'pair' variants,
    branch_real_wide.json) to show how much the theory moves between runs / variants.
  - QMF D: per file and r, the first alive grid point with PR >= N/4 and the grid bracket (QMF grid = 32 log points in
    0.1-30, 20% per step), interpolated in (ln g, PR) and in (ln g, ln PR).
DRIVEN: argmax of chi_rep/rho in 0.1 <= g <= 1 recomputed from the raw drive files.
CLAIMS: numbers quoted in phase_complete.md re-derived from phase_complete.json.
Outputs: results/analysis2/critic.json, critic_run.log (stdout), figs/pd2/critic_boundaries.png.
"""
import glob, json, os, re, sys
import numpy as np
from scipy.special import logsumexp

ROOT = "/MALECNS_ROOT"
RES = f"{ROOT}/results"; OUT = f"{RES}/analysis2"; FIG = f"{ROOT}/figs/pd2"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
N = 167216; T = 3000.0; TAU_S = 200.0; PR_D = 0.25; NB = 400
A_EFF = (1 - np.exp(-0.2)) / 0.2
rng = np.random.default_rng(7)
KINDS = ["real", "null", "wnull"]
nz = lambda x: None if x is None or not np.isfinite(x) else float(x)


# =============================================================================================== simulation, own loader
def kind_of(m):
    if not m.get("null"):
        return "real"
    return "wnull" if m.get("null_kind") == "weights" else "null"


def load_points():
    files = sorted(set(glob.glob(f"{RES}/pd/decay_r*.npz") + glob.glob(f"{RES}/pd2/onset_*.npz") +
                       glob.glob(f"{RES}/pd2/deloc_*.npz") + glob.glob(f"{RES}/*decay_raw_r1*.npz") +
                       glob.glob(f"{RES}/pd/long_*.npz")))
    pts, used = {}, []
    for f in files:
        d = np.load(f); m = json.loads(str(d["meta"]))
        if not (m.get("exp") == "decay" and m.get("norm") == "raw" and float(m.get("h", 0)) == 0 and
                abs(float(m.get("dt", 0.2)) - 0.2) < 1e-9 and float(m.get("beta", 1)) == 1 and float(m["T"]) >= T - 1e-6):
            continue
        t = d["t"]; w = t <= T + 1e-6; tw = t[w]; Tc = float(tw[-1])
        rho = d["rho"][:, w, :]                                   # [g, t, B]
        occ = d["occ"].astype(np.float64) if (abs(float(m["T"]) - T) < 1e-6 and "occ" in d.files) else None
        kind, r = kind_of(m), float(m["r"])
        used.append({"file": os.path.relpath(f, RES), "kind": kind, "r": r, "B": int(m["B"]), "T": float(m["T"]), "Tc": Tc})
        for i, g in enumerate(d["g"]):
            act = rho[i] > 0                                      # [t, B]
            dead = ~act
            died = dead.any(0)
            k = np.argmax(dead, 0)                                # first silent record
            tm = np.where(k > 0, np.sqrt(tw[np.maximum(k - 1, 0)] * tw[k]), tw[k])   # geometric midpoint
            tfirst = tw[k]
            times = np.where(died, tm, Tc); times_first = np.where(died, tfirst, Tc)
            o = occ[i] if occ is not None else None
            pr = float(o.sum() ** 2 / (o ** 2).sum() / N) if (o is not None and o.sum() > 0) else (0.0 if o is not None else None)
            pts.setdefault((kind, r), []).append({"g": float(g), "times": times, "times_first": times_first,
                                                  "died": died, "B": act.shape[1], "pr": pr,
                                                  "src": os.path.relpath(f, RES)})
    pooled, dup = {}, []
    for key, lst in pts.items():
        lst.sort(key=lambda x: x["g"])
        out = []
        for x in lst:
            if out and abs(x["g"] / out[-1]["g"] - 1) < 1e-4:
                y = out[-1]
                # common-random-number check: identical replica outcomes in two sources at the same g?
                if x["B"] == y["Bs"][0] and np.array_equal(x["times_first"], y["tf_list"][0]):
                    dup.append({"key": [key[0], key[1]], "g": x["g"], "srcs": [y["src"], x["src"]]})
                y["times"] = np.r_[y["times"], x["times"]]; y["times_first"] = np.r_[y["times_first"], x["times_first"]]
                y["died"] = np.r_[y["died"], x["died"]]; y["Bs"].append(x["B"]); y["tf_list"].append(x["times_first"])
                if x["pr"] is not None:
                    y["prs"].append((x["pr"], x["B"]))
                y["src"] += "+" + x["src"]
            else:
                out.append({"g": x["g"], "times": x["times"], "times_first": x["times_first"], "died": x["died"],
                            "Bs": [x["B"]], "tf_list": [x["times_first"]], "src": x["src"],
                            "prs": [(x["pr"], x["B"])] if x["pr"] is not None else []})
        for y in out:
            y["B"] = int(sum(y["Bs"]))
            y["pr"] = sum(a * b for a, b in y["prs"]) / sum(b for _, b in y["prs"]) if y["prs"] else None
            y["d"] = int(y["died"].sum()); y["E"] = float(y["times"].sum()); y["E_first"] = float(y["times_first"].sum())
        pooled[key] = out
    return pooled, used, dup


def fit_loglin(x, d, E, a0=None):
    """Censored-exponential MLE of ln tau = a + b x. NLL = sum d (a + b x) + E exp(-a - b x) (convex). Damped Newton."""
    a = np.log(max(E.sum(), 1e-9) / max(d.sum(), 0.5)) if a0 is None else a0; b = 0.0
    th = np.array([a, b])
    X = np.c_[np.ones_like(x), x]
    def nll(p):
        z = X @ p
        return float((d * z + E * np.exp(-z)).sum())
    for _ in range(200):
        z = X @ th; e = E * np.exp(-z)
        gr = X.T @ (d - e); H = (X * e[:, None]).T @ X + 1e-12 * np.eye(2)
        step = np.linalg.solve(H, gr)
        f0, s = nll(th), 1.0
        while s > 1e-8 and nll(th - s * step) > f0 + 1e-12:
            s /= 2
        th = th - s * step
        if np.abs(s * step).max() < 1e-10:
            break
    return th


def sim_tau_boundary(P, thr, win=1.13, use_first=False):
    g = np.array([p["g"] for p in P]); d = np.array([p["d"] for p in P], float)
    E = np.array([p["E_first"] if use_first else p["E"] for p in P])
    treg = E / np.maximum(d, 0.5)
    hit = np.where(treg >= thr)[0]
    if len(hit) == 0:
        return {"g": None, "status": "not_reached", "gmax": float(g[-1])}
    k = int(hit[0])
    if k == 0:
        return {"g": None, "status": "below_grid", "gmin": float(g[0])}
    g0 = np.sqrt(g[k - 1] * g[k])
    f = win
    while True:
        sel = np.where((g >= g0 / f) & (g <= g0 * f))[0]
        if (len(sel) >= 4 and (g[sel] < g0).sum() >= 2 and (g[sel] > g0).sum() >= 2) or f > 1.6:
            break
        f *= 1.04
    x = np.log(g[sel] / g0)
    a, b = fit_loglin(x, d[sel], E[sel])
    gc = float(g0 * np.exp((np.log(thr) - a) / b)) if b > 0 else None
    # replica bootstrap, window fixed
    bs = []
    for _ in range(NB):
        dd, EE = [], []
        for j in sel:
            p = P[j]; idx = rng.integers(0, p["B"], p["B"])
            tt = (p["times_first"] if use_first else p["times"])[idx]
            dd.append(p["died"][idx].sum()); EE.append(tt.sum())
        dd, EE = np.array(dd, float), np.array(EE)
        if dd.sum() == 0:
            continue
        ab, bb = fit_loglin(x, dd, EE, a)
        if bb > 0:
            bs.append(g0 * np.exp((np.log(thr) - ab) / bb))
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) > 50 else None
    return {"g": gc, "ci": ci, "slope_dlntau_dlng": float(b), "n_points": int(len(sel)),
            "window": [float(g[sel].min()), float(g[sel].max())], "grid_bracket": [float(g[k - 1]), float(g[k])],
            "status": "ok"}


def sim_D(P, g_on):
    q = [(p["g"], p["pr"]) for p in P if p["pr"] is not None and (g_on is None or p["g"] >= g_on)]
    if g_on is None or not q:
        return {"g": None, "status": "no_sustained"}
    g = np.array([a for a, _ in q]); pr = np.array([b for _, b in q])
    hit = np.where(pr >= PR_D)[0]
    if len(hit) == 0:
        return {"g": None, "status": "not_reached", "gmax": float(g[-1]), "PR_max": float(pr.max())}
    k = int(hit[0])
    if k == 0:
        return {"g": float(g[0]), "status": "at_first_sustained_point", "PR_there": float(pr[0])}
    lp0, lp1 = np.log(max(pr[k - 1], 1e-12)), np.log(pr[k])
    fr = (np.log(PR_D) - lp0) / (lp1 - lp0)
    return {"g": float(np.exp(np.log(g[k - 1]) + fr * np.log(g[k] / g[k - 1]))), "status": "ok",
            "bracket": [float(g[k - 1]), float(g[k])], "PR_bracket": [float(pr[k - 1]), float(pr[k])]}


def operational(f):
    """phase.py classification on one file (not pooled)."""
    d = np.load(f); m = json.loads(str(d["meta"]))
    t, gg, rho, surv, occ = d["t"], d["g"], d["rho"], d["surv"], d["occ"].astype(np.float64)
    ph = []
    for i in range(len(gg)):
        mm = rho[i].mean(-1); alive = surv[i, -1].mean()
        life = float(t[np.argmax(mm == 0)]) if (mm == 0).any() else np.inf
        o = occ[i]; pr = float(o.sum() ** 2 / (o ** 2).sum() / N) if o.sum() > 0 else 0.0
        if alive == 0:
            p = "Q" if life < 200 else "S"
        elif alive < 1:
            p = "S"
        else:
            p = "D" if pr >= 0.25 else "L"
        ph.append(p)
    s = "".join(ph)
    S_op = next((float(gg[i]) for i, p in enumerate(ph) if p != "Q"), None)
    L_from = None
    for i in range(len(ph)):
        if all(p in "LD" for p in ph[i:]):
            L_from = float(gg[i]); break
    D_op = next((float(gg[i]) for i, p in enumerate(ph) if p == "D"), None)
    return {"file": os.path.relpath(f, RES), "B": int(m["B"]), "seq": s, "g": [float(x) for x in gg],
            "S_op": S_op, "L_op": L_from, "D_op": D_op,
            "S_op_prev": float(gg[[i for i, p in enumerate(ph) if p != "Q"][0] - 1]) if S_op and S_op > gg[0] else None,
            "L_op_prev": float(gg[list(gg).index(L_from) - 1]) if L_from and L_from > gg[0] else None}


# ======================================================================================================= theory, own code
def logtau_sum(lam, K):
    M = int(np.floor(K))
    if M < 1 or lam <= 0:
        return -np.inf
    n = np.arange(1, M + 1, dtype=float)
    b = lam * n * (1 - n / K); b[-1] = 0.0
    dd = A_EFF * n
    r = np.log(b[:-1]) - np.log(dd[:-1])                       # log(b_i/d_i), i = 1..M-1
    Cj = np.r_[0.0, np.cumsum(r)]                               # C_j = sum_{i<j} r_i, j = 1..M
    termj = Cj - np.log(dd)                                     # C_j - log d_j
    # log D_k = -C_k + logsumexp_{j >= k} termj  (reverse cumulative logsumexp)
    rev = np.logaddexp.accumulate(termj[::-1])[::-1]
    logD = -Cj + rev
    return float(logsumexp(logD))


def load_dense(kind):
    ps = sorted(glob.glob(f"{RES}/pd/branch_{kind}_dense_part*.json"))
    js = [json.load(open(p)) for p in ps]
    rows = sorted([x for j in js for x in j["rows"]], key=lambda x: x["g"])
    return {"n_parts": len(ps), "g_lin": js[0]["g_lin"], "r": js[0]["r"], "rows": rows,
            "G": np.array([x["g"] for x in rows])}


def cross_rows(G, lt, thr):
    lt = np.asarray(lt, float); y = np.log(thr)
    hit = np.where(lt >= y)[0]
    if len(hit) == 0:
        return {"g": None, "status": f"not_reached_by_{G[-1]:.3g}"}
    k = int(hit[0])
    if k == 0:
        return {"g": None, "status": f"below_first_row_{G[0]:.3g}"}
    fr = (y - lt[k - 1]) / (lt[k] - lt[k - 1])
    return {"g": float(np.exp(np.log(G[k - 1]) + fr * np.log(G[k] / G[k - 1]))), "status": "ok",
            "rows": [float(G[k - 1]), float(G[k])]}


def cross_inputs(G, lam, K, thr, g_lin):
    gf = np.geomspace(G[0], G[-1], 3000)
    L = np.interp(np.log(gf), np.log(G), lam); KK = np.interp(np.log(gf), np.log(G), K)
    lt = np.array([logtau_sum(a, b) for a, b in zip(L, KK)])
    return cross_rows(gf, lt, thr)


# ============================================================================================================ run it
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)


C = {"method": __doc__}
# ---------------- theory: recursion check, crossings
TH = {k: load_dense(k) for k in KINDS}
C["theory"] = {}
for kind in KINDS:
    th = TH[kind]; G = th["G"]
    maxdiff, nchk = 0.0, 0
    for x in th["rows"]:
        for r in th["r"]:
            s = x.get(f"logtau_r{r:g}")
            if s is None:
                continue
            v = logtau_sum(x["lam"], x[f"K_r{r:g}"]); nchk += 1
            if np.isfinite(v) and np.isfinite(s):
                maxdiff = max(maxdiff, abs(v - s))
    P(f"[theory {kind}] {th['n_parts']} parts, {len(G)} rows, g {G[0]:.4f}-{G[-1]:.4f}, step {np.median(G[1:]/G[:-1]):.4f}; "
      f"closed-form sum vs stored logtau over {nchk} (row, r): max |diff| = {maxdiff:.2e}")
    tk = {"n_rows": len(G), "g_range": [float(G[0]), float(G[-1])], "row_step": float(np.median(G[1:] / G[:-1])),
          "logtau_check_max_abs_diff": maxdiff, "n_checked": nchk, "per_r": {}}
    lam = np.array([x["lam"] for x in th["rows"]])
    for r in th["r"]:
        lt = np.array([x[f"logtau_r{r:g}"] if x.get(f"logtau_r{r:g}") is not None else np.nan for x in th["rows"]])
        K = np.array([x[f"K_r{r:g}"] for x in th["rows"]])
        e = {"S_out": cross_rows(G, lt, TAU_S), "on_out": cross_rows(G, lt, T),
             "S_in": cross_inputs(G, lam, K, TAU_S, th["g_lin"]), "on_in": cross_inputs(G, lam, K, T, th["g_lin"])}
        tk["per_r"][f"{r:g}"] = e
    C["theory"][kind] = tk

# older theory runs / variants
C["theory_versions"] = {}
for f, kind in [("branch_real.json", "real"), ("branch_real_wide.json", "real"), ("branch_null.json", "null"),
                ("branch_wnull.json", "wnull")]:
    j = json.load(open(f"{RES}/pd/{f}"))
    for vn, v in j["variants"].items():
        rows = sorted(v["rows"], key=lambda x: x["g"]); G = np.array([x["g"] for x in rows])
        rs = [k[8:] for k in rows[0] if k.startswith("logtau_r")]
        ent = {"g_lin": v.get("g_lin"), "g_range": [float(G[0]), float(G[-1])], "step": float(np.median(G[1:] / G[:-1])),
               "per_r": {}}
        for r in rs:
            lt = np.array([x[f"logtau_r{r}"] for x in rows], float)
            ent["per_r"][r] = {"S": cross_rows(G, lt, TAU_S), "on": cross_rows(G, lt, T)}
            # K at a common g, compared with the dense run (interpolated in ln g)
        Gd = TH[kind]["G"]
        common = [x for x in rows if Gd[0] <= x["g"] <= Gd[-1]]
        if common and vn == "branching":
            kr = []
            for x in common:
                for r in ["0", "1"]:
                    if f"K_r{r}" in x:
                        Kd = np.interp(np.log(x["g"]), np.log(Gd), [y[f"K_r{r}"] for y in TH[kind]["rows"]])
                        lamd = np.interp(np.log(x["g"]), np.log(Gd), [y["lam"] for y in TH[kind]["rows"]])
                        kr.append((x[f"K_r{r}"] / Kd, x["lam"] / lamd))
            kr = np.array(kr)
            ent["vs_dense_at_common_g"] = {"K_ratio_range": [float(kr[:, 0].min()), float(kr[:, 0].max())],
                                           "lam_ratio_range": [float(kr[:, 1].min()), float(kr[:, 1].max())]}
        C["theory_versions"][f"{f}:{vn}"] = ent
        P(f"[theory version {f}:{vn}] g_lin {v.get('g_lin'):.4f} rows {G[0]:.3f}-{G[-1]:.3f} step {ent['step']:.4f} "
          + " ".join(f"r{r}: S {nz(ent['per_r'][r]['S']['g']) and round(ent['per_r'][r]['S']['g'],3)} on "
                     f"{nz(ent['per_r'][r]['on']['g']) and round(ent['per_r'][r]['on']['g'],3)}" for r in rs)
          + (f" | vs dense at common g: K ratio {ent['vs_dense_at_common_g']['K_ratio_range']}, lam ratio "
             f"{ent['vs_dense_at_common_g']['lam_ratio_range']}" if "vs_dense_at_common_g" in ent else ""))

# QMF D with grid brackets
C["qmf"] = {}
for kind in KINDS:
    files = [f"{RES}/pd/qmf_{kind}.npz"] + ([f"{RES}/pd/qmf_real_dense.npz"] if kind == "real" else []) + \
        sorted(glob.glob(f"{RES}/pd2/qmf_{kind}_r*.npz"))
    per = {}
    for f in files:
        q = np.load(f); G = q["g"]
        for i, r in enumerate(q["r"]):
            pr, rho = q["PR"][i, 0], q["rho_T"][i, 0]
            alive = rho > 1e-9
            k = [j for j in range(len(G)) if alive[j] and pr[j] >= PR_D]
            if not k:
                e = {"g_lin": None, "g_log": None, "bracket": [float(G[-1]), None], "status": "none<=30",
                     "PR_max": float(pr[alive].max()) if alive.any() else 0.0}
            else:
                j = k[0]
                if j > 0 and alive[j - 1]:
                    fr = (PR_D - pr[j - 1]) / (pr[j] - pr[j - 1])
                    gl = float(np.exp(np.log(G[j - 1]) + fr * np.log(G[j] / G[j - 1])))
                    fr2 = (np.log(PR_D) - np.log(pr[j - 1])) / (np.log(pr[j]) - np.log(pr[j - 1]))
                    gg2 = float(np.exp(np.log(G[j - 1]) + fr2 * np.log(G[j] / G[j - 1])))
                    st = "ok"
                else:
                    gl = gg2 = float(G[j]); st = "jump_from_silent"
                e = {"g_lin": gl, "g_log": gg2, "bracket": [float(G[j - 1]), float(G[j])], "status": st,
                     "PR_bracket": [float(pr[j - 1]), float(pr[j])]}
            e["file"] = os.path.relpath(f, RES)
            per.setdefault(f"{float(r):g}", []).append(e)
    C["qmf"][kind] = per

# ---------------- simulation
pooled, used, dup = load_points()
C["sim_inputs"] = used
C["crn_identical_duplicates"] = dup
P(f"[sim] {len(used)} files used; pooled point sets: {len(pooled)}; identical-replica duplicates across sources: {len(dup)}")
C["sim"] = {}
for (kind, r), Pp in sorted(pooled.items()):
    S = sim_tau_boundary(Pp, TAU_S); on = sim_tau_boundary(Pp, T)
    on_first = sim_tau_boundary(Pp, T, use_first=True)
    D = sim_D(Pp, on["g"])
    C["sim"].setdefault(kind, {})[f"{r:g}"] = {"S": S, "on": on, "on_first_record_times": on_first, "D": D,
                                               "n_points": len(Pp)}


# MODEL-FREE DATA BRACKET: per pooled point the 95% chi-square CI of tau (midpoint death times); the crossing of thr must
# lie between the largest g whose CI is entirely below thr and the smallest g whose CI is entirely above thr.
from scipy import stats as _st
def data_bracket(Pp, thr):
    g = np.array([p["g"] for p in Pp]); lo_ci, hi_ci = [], []
    for p in Pp:
        dd, E = p["d"], p["E"]
        lo_ci.append(2 * E / _st.chi2.ppf(0.975, 2 * dd + 2))
        hi_ci.append(2 * E / _st.chi2.ppf(0.025, 2 * dd) if dd > 0 else np.inf)
    lo_ci, hi_ci = np.array(lo_ci), np.array(hi_ci)
    above = np.where(lo_ci > thr)[0]; below = np.where(hi_ci < thr)[0]
    if len(above) == 0:
        return {"lo": float(g[below].max()) if len(below) else None, "hi": None, "status": "not_reached"}
    hi = g[above].min()
    bl = below[g[below] < hi]
    lo = float(g[bl].max()) if len(bl) else None
    return {"lo": lo, "hi": float(hi), "status": "ok",
            "nonmonotone_below_points_above_hi": int((g[below] > hi).sum())}


for (kind, r), Pp in sorted(pooled.items()):
    e = C["sim"][kind][f"{r:g}"]
    e["S_data_bracket"] = data_bracket(Pp, TAU_S); e["on_data_bracket"] = data_bracket(Pp, T)

# operational, per file
C["operational"] = {}
for f in sorted(glob.glob(f"{RES}/pd2/onset_*.npz") + glob.glob(f"{RES}/pd/decay_r*.npz") + glob.glob(f"{RES}/pd2/deloc_*.npz")):
    o = operational(f)
    C["operational"][o["file"]] = o

# theory at the tau that the B-DEPENDENT operational criteria correspond to (independent exponential lifetimes tau):
#   S_op: P(max of B lifetimes >= 200) = 1/2  ->  tau = 200 / -ln(1 - 2^(-1/B))
#   L_op: P(all B alive at T) = 1/2           ->  tau = B T / ln 2
C["theory_operational"] = {}
for B in [16, 32]:
    tS = 200.0 / -np.log(1 - 2 ** (-1.0 / B)); tL = B * T / np.log(2)
    for kind in KINDS:
        th = TH[kind]; G = th["G"]
        for r in th["r"]:
            lt = np.array([x[f"logtau_r{r:g}"] if x.get(f"logtau_r{r:g}") is not None else np.nan for x in th["rows"]])
            C["theory_operational"].setdefault(f"B{B}", {"tau_S_equiv": tS, "tau_L_equiv": tL}).setdefault(kind, {})[f"{r:g}"] = {
                "S": cross_rows(G, lt, tS), "L": cross_rows(G, lt, tL)}

# common random numbers across r: every run used --seed 0, so runs with the same g list and B share their random stream.
# Spearman correlation of per-replica extinction times at the same g between neighbouring-r pd2 runs.
C["crn_correlation"] = []
for r1, r2 in [("0", "0.125"), ("1", "1.25"), ("4", "4.5")]:
    d1, d2 = np.load(f"{RES}/pd2/onset_real_r{r1}.npz"), np.load(f"{RES}/pd2/onset_real_r{r2}.npz")
    t = d1["t"]
    for i, g in enumerate(d1["g"]):
        a1 = d1["rho"][i] > 0; a2 = d2["rho"][i] > 0
        e1 = np.where((~a1).any(0), t[np.argmax(~a1, 0)], np.inf); e2 = np.where((~a2).any(0), t[np.argmax(~a2, 0)], np.inf)
        ok = np.isfinite(e1) & np.isfinite(e2)
        if ok.sum() >= 16:
            rho_s = _st.spearmanr(e1[ok], e2[ok]).statistic
            C["crn_correlation"].append({"r_pair": [float(r1), float(r2)], "g": float(g), "n": int(ok.sum()),
                                         "spearman": float(rho_s)})
    # control: same r, different g (adjacent grid points) -> different stream positions
cc = np.array([x["spearman"] for x in C["crn_correlation"]])
P(f"[CRN] Spearman of per-replica extinction times between neighbouring-r runs at the same g: n_g={len(cc)}, "
  f"median {np.median(cc):.2f}, range {cc.min():.2f}-{cc.max():.2f}" if len(cc) else "[CRN] no overlapping deaths")

# ---------------- driven
C["driven"] = {}
for f in sorted(glob.glob(f"{RES}/pd2/drive_h*_*.npz")):
    d = np.load(f); m = json.loads(str(d["meta"]))
    t, g, rho = d["t"], d["g"], d["rho"]
    rm = rho[:, t >= t[-1] / 2, :].mean(1)
    F = N * rm.var(1, ddof=1) / rm.mean(1)
    lo, hi = (0.1, 1.0) if not m.get("null") else (0.5, 5.0)
    w = np.where((g >= lo - 1e-9) & (g <= hi + 1e-9))[0]; k = w[np.argmax(F[w])]
    C["driven"][os.path.relpath(f, RES)] = {"h": float(m["h"]), "peak_g": float(g[k]), "peak_F": float(F[k]),
                                            "at_window_edge": bool(k in (w[0], w[-1]))}

# ---------------- claims check against phase_complete.json
J = json.load(open(f"{OUT}/phase_complete.json"))
cl = {}
ag = {k: {f"{x['r']:g}": x for x in J["agreement"][k]} for k in KINDS}
cl["real_on_ratio"] = {r: ag["real"][r]["on"]["ratio_theory_over_sim"] for r in ["0.125", "0.25", "0.5", "1", "2", "4", "6"]}
cl["real_S_ratio"] = {r: ag["real"][r]["S"]["ratio_theory_over_sim"] for r in ["0.125", "0.25", "0.5", "1", "2", "4", "6"]}
cl["rise_sim"] = ag["real"]["6"]["on"]["sim"] / ag["real"]["0"]["on"]["sim"]
cl["rise_th"] = ag["real"]["6"]["on"]["theory"] / ag["real"]["0"]["on"]["theory"]
cl["Kratio"] = {k: {r: (x["K_needed"]["median"] / x["K_needed"]["K_theory_median"]) if x["K_needed"]["n"] else None
                    for r, x in ag[k].items()} for k in KINDS}
lr = []
for kind in KINDS:
    for rk, e in J["sim"][kind].items():
        for p in e["points"]:
            if p["deaths"] >= 3 and p["tau_sim"] and p["tau_sim"] > 200 and p["ln_tau_theory"] is not None:
                lr.append((kind, float(rk), p["g"], np.log(p["tau_sim"]) - p["ln_tau_theory"]))
lr_real = np.array([x[3] for x in lr if x[0] == "real"])
cl["tau_ratio_real"] = {"n": int(len(lr_real)), "min": float(np.exp(lr_real.min())),
                        "frac_gt10": float((lr_real > np.log(10)).mean()), "frac_gt100": float((lr_real > np.log(100)).mean())}
for kind in ["null", "wnull"]:
    a = np.array([x[3] for x in lr if x[0] == kind])
    cl[f"tau_ratio_{kind}"] = {"n": int(len(a)), "min": float(np.exp(a.min())) if len(a) else None}
cl["half_over_on"] = {k: [J["sim"][k][r]["half"]["g"] / J["sim"][k][r]["on"]["g"] for r in J["sim"][k]
                          if J["sim"][k][r]["half"]["g"] and J["sim"][k][r]["on"]["g"]] for k in KINDS}
cl["half_over_on"] = {k: [float(min(v)), float(max(v))] for k, v in cl["half_over_on"].items()}
C["claims_check"] = cl

# ---------------- comparison table: assembler vs critic
rows = []
for kind in KINDS:
    for rk in sorted(C["sim"][kind], key=float):
        a = J["sim"][kind].get(rk); c = C["sim"][kind][rk]
        t_a = J["theory"][kind].get(rk, {}); t_c = C["theory"][kind]["per_r"].get(rk, {})
        qa = t_a.get("D", {}) if t_a else {}
        qc = C["qmf"][kind].get(rk, [])
        row = {"kind": kind, "r": float(rk)}
        for key in ["S", "on"]:
            row[f"{key}_sim_assembler"] = a[key]["g"] if a else None
            row[f"{key}_sim_assembler_total"] = a[key]["total"] if a else None
            row[f"{key}_sim_critic"] = c[key].get("g"); row[f"{key}_sim_critic_ci"] = c[key].get("ci")
            row[f"{key}_sim_critic_status"] = c[key]["status"]
            row[f"{key}_th_assembler"] = (t_a.get(key) or {}).get("g")
            row[f"{key}_th_critic_out"] = (t_c.get(f"{key}_out") or {}).get("g")
            row[f"{key}_th_critic_in"] = (t_c.get(f"{key}_in") or {}).get("g")
        row["on_sim_critic_firstrecord"] = c["on_first_record_times"].get("g")
        row["D_sim_assembler"] = a["D"]["g"] if a else None; row["D_sim_assembler_status"] = a["D"]["status"] if a else None
        row["D_sim_critic"] = c["D"].get("g"); row["D_sim_critic_status"] = c["D"]["status"]
        row["D_th_assembler"] = qa.get("g")
        row["D_th_critic"] = [{"file": e["file"], "g": e["g_lin"], "g_logPR": e["g_log"], "bracket": e["bracket"],
                               "status": e["status"]} for e in qc]
        rows.append(row)
C["comparison"] = rows


def f3(x):
    return "–" if x is None else f"{x:.3g}"


P("\n[comparison] kind r | S sim: assembler / critic [CI] | on sim: assembler / critic [CI] (first-record times) | "
  "S th: asm / out / in | on th: asm / out / in | D sim asm / critic | D QMF asm / critic brackets")
for row in rows:
    ci = lambda k: f"[{f3(row[k][0])},{f3(row[k][1])}]" if row[k] else "[–]"
    qb = "; ".join(f"{f3(e['g'])} ({e['bracket'][0]:.3g}-{f3(e['bracket'][1])},{e['file'].split('/')[-1]})" for e in row["D_th_critic"])
    P(f"{row['kind']:5s} {row['r']:<5g} | S {f3(row['S_sim_assembler'])} / {f3(row['S_sim_critic'])} {ci('S_sim_critic_ci')} | "
      f"on {f3(row['on_sim_assembler'])} / {f3(row['on_sim_critic'])} {ci('on_sim_critic_ci')} ({f3(row['on_sim_critic_firstrecord'])}) | "
      f"Sth {f3(row['S_th_assembler'])}/{f3(row['S_th_critic_out'])}/{f3(row['S_th_critic_in'])} | "
      f"onth {f3(row['on_th_assembler'])}/{f3(row['on_th_critic_out'])}/{f3(row['on_th_critic_in'])} | "
      f"D {f3(row['D_sim_assembler'])}({row['D_sim_assembler_status']}) / {f3(row['D_sim_critic'])}({row['D_sim_critic_status']}) | "
      f"QMF {f3(row['D_th_assembler'])} / {qb}")

P("\n[operational phase.py classification, per file]")
for k, o in C["operational"].items():
    P(f"{k:28s} B{o['B']:<3d} S_op {f3(o['S_op'])} (prev {f3(o['S_op_prev'])})  L_op {f3(o['L_op'])} (prev {f3(o['L_op_prev'])})  "
      f"D_op {f3(o['D_op'])}  {o['seq']}")
P("\n[driven peaks]", json.dumps(C["driven"]))
P("\n[claims]", json.dumps(cl, default=float))
if dup:
    P("[CRN duplicates]", json.dumps(dup[:10]))


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(C), open(f"{OUT}/critic.json", "w"), indent=1)
open(f"{OUT}/critic_run.log", "w").write("\n".join(log) + "\n")
print("wrote", f"{OUT}/critic.json")

# ---------------- summary printouts for the review (data brackets, operational criteria)
P("\n[data brackets vs assembler / theory] kind r | S: bracket, assembler, theory | on: bracket, assembler, theory")
for kind in KINDS:
    for rk in sorted(C["sim"][kind], key=float):
        e = C["sim"][kind][rk]; a = J["sim"][kind][rk]; t = J["theory"][kind].get(rk, {})
        def chk(key, bk):
            b = e[bk]; sv = a[key]["g"]; tv = (t.get(key) or {}).get("g")
            inside = lambda v: (v is not None and (b["lo"] is None or v >= b["lo"]) and (b["hi"] is None or v <= b["hi"]))
            return (f"[{f3(b['lo'])}, {f3(b['hi'])}] asm {f3(sv)} {'in' if inside(sv) else 'OUT'} th {f3(tv)} "
                    f"{'in' if inside(tv) else ('above' if (tv is not None and b['hi'] is not None and tv > b['hi']) else 'below/–')}")
        P(f"{kind:5s} {rk:5s} | S {chk('S', 'S_data_bracket')} | on {chk('on', 'on_data_bracket')}")
P("\n[operational criteria (B = 32 pd2 files) vs theory at the matched tau]")
TO = C["theory_operational"]["B32"]
P(f"tau equivalents for B = 32: S_op <-> tau {TO['tau_S_equiv']:.1f}, L_op <-> tau {TO['tau_L_equiv']:.3g}")
for kind in KINDS:
    for rk in sorted(TO[kind], key=float):
        f = f"pd2/onset_{kind}_r{rk}.npz"
        o = C["operational"].get(f)
        if not o:
            continue
        t = TO[kind][rk]
        P(f"{kind:5s} {rk:5s} S_op sim ({f3(o['S_op_prev'])}, {f3(o['S_op'])}]  theory {f3(t['S']['g'])} {t['S']['status'][:20]} | "
          f"L_op sim ({f3(o['L_op_prev'])}, {f3(o['L_op'])}]  theory {f3(t['L']['g'])} {t['L']['status'][:20]}")
json.dump(clean(C), open(f"{OUT}/critic.json", "w"), indent=1)
open(f"{OUT}/critic_run.log", "w").write("\n".join(log) + "\n")

# ---------------- write critic fields into phase_complete.json (adds keys only; no existing value is changed)
QSTEP = float(np.exp(np.log(300) / 31))                      # QMF grid: 32 log points in 0.1-30
for kind in KINDS:
    for rk, e in J["theory"][kind].items():
        if "D" in e and C["qmf"][kind].get(rk):
            q = C["qmf"][kind][rk][-1]                         # same file order as analysis2_phase.load_qmf (last wins)
            e["D"]["grid_bracket"] = q["bracket"]
            e["D"]["upper_bound_only"] = q["status"] == "jump_from_silent"
            e["D"]["g_interp_lnPR"] = q["g_log"]
            e["D"]["qmf_grid_step"] = QSTEP
    for row in J["agreement"][kind]:
        rk = f"{row['r']:g}"; cs = C["sim"][kind][rk]
        for key in ["S", "on"]:
            b = cs[f"{key}_data_bracket"]; tv = row[key]["theory"]
            row[key]["sim_data_bracket_95"] = [b["lo"], b["hi"]]
            row[key]["critic_local_fit"] = {"g": cs[key]["g"], "ci": cs[key].get("ci")}
            if tv is None:
                pos = None
            elif b["hi"] is not None and tv > b["hi"]:
                pos = "above"
            elif b["lo"] is not None and tv < b["lo"]:
                pos = "below"
            else:
                pos = "inside"
            row[key]["theory_vs_data_bracket"] = pos
        qd = J["theory"][kind].get(rk, {}).get("D")
        if qd and qd.get("grid_bracket"):
            row["D"]["theory_grid_bracket"] = qd["grid_bracket"]; row["D"]["theory_upper_bound_only"] = qd["upper_bound_only"]
        to = C["theory_operational"]["B32"][kind].get(rk)
        oo = C["operational"].get(f"pd2/onset_{kind}_r{rk}.npz")
        if to and oo:
            row["operational_B32"] = {"S_op_sim_cell": [oo["S_op_prev"], oo["S_op"]], "L_op_sim_cell": [oo["L_op_prev"], oo["L_op"]],
                                      "S_op_theory": to["S"]["g"], "L_op_theory": to["L"]["g"],
                                      "L_op_theory_status": to["L"]["status"],
                                      "tau_equiv": [C["theory_operational"]["B32"]["tau_S_equiv"],
                                                    C["theory_operational"]["B32"]["tau_L_equiv"]]}
J["critic_review"] = {
    "source": "results/analysis2/critic.json, critic_run.log (code/analysis2_critic.py)", "date": "2026-11-09",
    "added_keys": ["theory.<kind>.<r>.D.grid_bracket / upper_bound_only / g_interp_lnPR / qmf_grid_step",
                   "agreement.<kind>[].S|on.sim_data_bracket_95 / critic_local_fit / theory_vs_data_bracket",
                   "agreement.<kind>[].D.theory_grid_bracket / theory_upper_bound_only", "agreement.<kind>[].operational_B32"],
    "note": "added by the critic after analysis2_phase.py; rerunning analysis2_phase.py drops these keys, rerun "
            "analysis2_critic.py afterwards to restore them"}
json.dump(J, open(f"{OUT}/phase_complete.json", "w"), indent=1)
P("added critic keys to phase_complete.json")
