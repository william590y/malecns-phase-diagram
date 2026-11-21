"""m2_nature_fss.py -- lane m2/nature, finite-size scaling of the setting-(a) crossover (plan section 1) and the
oscillation finite-size check on the same runs (plan section 5).

  python m2_nature_fss.py cells     stage-1 crossings -> results/m2/nature/fss_cells.json (refine / extend lists)
  python m2_nature_fss.py analyze   everything -> results/m2/nature/fss.json

Estimators are the decay lane's (code/m2_decay_analyze.py, = analysis2): tau = censored-exponential MLE of the first
log-grid record with no active unit (m2_obs.extinction_times), T = 3000; isotonic ln(E / max(deaths, 0.5)) in g with
weights deaths + 0.5; first crossing of ln 200 (S) / ln 3000 (onset), linear in ln g; 2000-resample replica bootstrap.
Full-size points: the decay lane's rows M2_real_r*, M2_cnull_r* (null_seed 0), M2_cnull1_r1 (its sum files, read only).
"""
import glob, json, os, sys
import numpy as np
from scipy import stats
from scipy.optimize import isotonic_regression

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import m2_common as M
import m2_obs as O

ROOT = M.ROOT
OUT = f"{ROOT}/results/m2/nature"
SIM = f"{OUT}/sim/fss"
T_REF, DT = 3000.0, 0.2
NFULL = 167216
GC = M.G_COARSE
rng = np.random.default_rng(20261109)


def load_index():
    idx = {}
    for f in sorted(glob.glob(f"{OUT}/index_fss*.json")):
        for e in json.load(open(f)):
            idx[os.path.basename(e["out"])] = e
    return idx


def summarize(path, e):
    d = np.load(path)
    t = d["t"]; G = d["g"]
    names = [str(x) for x in d["reg_names"]]; sizes = d["reg_sizes"].astype(float)
    di = [names.index(n) for n in ("AL_module", "AL_rest", "CNS_rest")]
    N = float(sizes[di].sum())
    out = []
    for i, g in enumerate(G):
        ext = O.extinction_times(t, d["surv"][i])
        alive = ~np.isfinite(ext)
        p = {"g": float(g), "ext": ext, "B": int(len(ext)), "n_alive": int(alive.sum()), "N": N}
        if alive.any() and "win_mean" in d.files:
            wm = d["win_mean"][i, 0].astype(np.float64)
            p["qs"] = float((wm[di][:, alive] * sizes[di, None]).sum(0).mean() / N)
            p["qs_AL"] = float(wm[names.index("AL_module")][alive].mean())
        else:
            p["qs"] = np.nan; p["qs_AL"] = np.nan
        if "cnt_act" in d.files and alive.sum() >= 4:
            x = d["cnt_act"][i][:, alive].astype(np.float64) / N
            p["var_t"] = float(x.var(0).mean()); p["chi"] = N * p["var_t"]
            m = O.osc_metrics(x, DT)
            p["osc"] = {k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v)
                        for k, v in m.items()}
        out.append(p)
    return out


def load_curves():
    """{(family, N, gseed, r): [points pooled per g]} for this lane's runs."""
    idx = load_index()
    cur = {}
    for f in sorted(glob.glob(f"{SIM}/*.npz")):
        e = idx.get(os.path.basename(f))
        if e is None:
            continue
        try:
            pts = summarize(f, e)
        except Exception as ex:
            print("FAIL", f, repr(ex), flush=True); continue
        key = (e["family"], int(e["N"]), int(e["gseed"]), float(e["r"]))
        cur.setdefault(key, []).extend(pts)
    return {k: pool(v) for k, v in cur.items()}


def pool(pts):
    pts = sorted(pts, key=lambda p: p["g"])
    out = []
    for p in pts:
        if out and abs(p["g"] / out[-1]["g"] - 1) < 1e-4:
            q = out[-1]; q["ext"] = np.r_[q["ext"], p["ext"]]; q["B"] += p["B"]; q["n_alive"] += p["n_alive"]
        else:
            out.append(dict(p))
    for q in out:
        died = np.isfinite(q["ext"]); q["died"] = died
        q["times"] = np.where(died, q["ext"], T_REF); q["deaths"] = int(died.sum()); q["E"] = float(q["times"].sum())
        q["tau"] = q["E"] / q["deaths"] if q["deaths"] else np.inf
    return out


def decay_lane_rows():
    """Full-size rows of lane decay (read only): real graph and the 167k configuration nulls."""
    import m2_decay_analyze as DA
    rows = DA.load_rows(stages=("s1", "s2"))
    want = {}
    for r in (0, 1, 4):
        want[("real", NFULL, 0, float(r))] = f"M2_real_r{r:g}"
        want[("cnull", NFULL, 0, float(r))] = f"M2_cnull_r{r:g}"
    want[("cnull", NFULL, 1, 1.0)] = "M2_cnull1_r1"
    out = {}
    for key, rk in want.items():
        P = rows[rk]["points"]
        pts = []
        for p in P:
            q = {"g": p["g"], "B": p["B"], "died": p["died"], "times": p["times"], "deaths": p["deaths"], "E": p["E"],
                 "tau": p["tau"], "n_alive": p["n_alive"], "qs": p["qs_tot"], "N": float(NFULL), "src": rk}
            oc = [o.get("total") for o in p["osc_list"] if o and o.get("total")]
            if oc:
                q["osc"] = oc[0]
            # var_t of the survivors' active fraction from the raw run (lane decay's file, read only)
            try:
                src = p["src"].split("+")[0]
                d = np.load(f"{ROOT}/results/m2/decay/sim/{src}")
                i = int(np.argmin(np.abs(d["g"] - p["g"])))
                if "cnt_act" in d.files and abs(d["g"][i] / p["g"] - 1) < 1e-4:
                    ext = O.extinction_times(d["t"], d["surv"][i]); alive = ~np.isfinite(ext)
                    if alive.sum() >= 4:
                        x = d["cnt_act"][i][:, alive].astype(np.float64) / NFULL
                        q["var_t"] = float(x.var(0).mean()); q["chi"] = NFULL * q["var_t"]
            except Exception as ex:
                print("var_t fail", rk, p["g"], repr(ex))
            pts.append(q)
        out[key] = pts
    return out


def jsafe(o):
    """Recursively convert numpy types; non-finite floats -> None (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsafe(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


# ------------------------------------------------------------------------------------------------ crossings
def iso(y, w):
    return isotonic_regression(np.asarray(y, float), weights=np.asarray(w, float), increasing=True).x


def cross(g, lt, thr):
    hit = np.where(lt >= np.log(thr))[0]
    if len(hit) == 0:
        return None, "none", None
    k = int(hit[0])
    if k == 0:
        return None, "below_grid", None
    y0, y1 = lt[k - 1], lt[k]
    fr = (np.log(thr) - y0) / (y1 - y0) if y1 > y0 else 1.0
    return float(np.exp(np.log(g[k - 1]) + fr * np.log(g[k] / g[k - 1]))), "ok", (float(g[k - 1]), float(g[k]))


def curve_cross(P, d=None, E=None, taus=(200.0, 500.0, 3000.0)):
    g = np.array([p["g"] for p in P])
    if d is None:
        d = np.array([p["deaths"] for p in P], float); E = np.array([p["E"] for p in P])
    lt = iso(np.log(E / np.maximum(d, 0.5)), d + 0.5)
    return {tt: cross(g, lt, tt) for tt in taus}


def boot_W(P, nboot=2000):
    nb = len(P)
    D = np.zeros((nboot, nb)); E = np.zeros((nboot, nb))
    for j, p in enumerate(P):
        ix = rng.integers(0, p["B"], (nboot, p["B"]))
        D[:, j] = p["died"][ix].sum(1); E[:, j] = p["times"][ix].sum(1)
    S, on, W, W2 = [], [], [], []
    for b in range(nboot):
        c = curve_cross(P, D[b], E[b])
        s, o, s5 = c[200.0][0], c[3000.0][0], c[500.0][0]
        S.append(s if s else np.nan); on.append(o if o else np.nan)
        W.append(np.log(o / s) if (s and o) else np.nan); W2.append(np.log(o / s5) if (s5 and o) else np.nan)
    ci = lambda a: ([float(x) for x in np.nanpercentile(a, [2.5, 97.5])] if np.isfinite(a).sum() >= 20 else None)
    return {"S_ci": ci(np.array(S)), "on_ci": ci(np.array(on)), "W_ci": ci(np.array(W)), "W2_ci": ci(np.array(W2)),
            "W_se": float(np.nanstd(W)) if np.isfinite(W).sum() >= 20 else None}


# ------------------------------------------------------------------------------------------------ commands
def cmd_cells():
    cur = load_curves()
    cells = {}
    for key, P in sorted(cur.items()):
        fam, N, s, r = key
        c = curve_cross(P)
        g = np.array([p["g"] for p in P])
        ref, extra = [], []
        for tt in (200.0, 3000.0):
            v, st, br = c[tt]
            if st == "ok":
                if br not in ref and np.log(br[1] / br[0]) > 0.03:          # not yet refined (refined cells ~0.018)
                    ref.append(br)
            elif st == "none":                                   # extend upward by 4 coarse points
                extra += [float(x) for x in GC[GC > g.max() * 1.0001][:4]]
            elif st == "below_grid":
                extra += [float(x) for x in GC[GC < g.min() * 0.9999][-4:]]
        cells[f"{fam}|{N}|{s}|{r:g}"] = {"refine": ref, "extra": sorted(set(extra)),
                                         "S": c[200.0][0], "on": c[3000.0][0], "n": len(P)}
        print(key, "S", c[200.0][:2], "on", c[3000.0][:2], "extra", sorted(set(extra)))
    json.dump(cells, open(f"{OUT}/fss_cells.json", "w"), indent=1)


def spearman(x, y):
    if len(x) < 3:
        return None
    return float(stats.spearmanr(x, y)[0])


def cmd_analyze():
    cur = load_curves()
    cur.update(decay_lane_rows())
    gl = json.load(open(f"{OUT}/graphs/graphs.json"))["graphs"] if os.path.exists(f"{OUT}/graphs/graphs.json") else {}
    per = []
    for key, P in sorted(cur.items()):
        fam, N, s, r = key
        c = curve_cross(P)
        bt = boot_W(P)
        S, on, s5 = c[200.0][0], c[3000.0][0], c[500.0][0]
        gk = "real_full" if fam == "real" else f"{fam}_N{N}_s{s}"
        glin = gl.get(gk, {}).get("g_lin")
        # chi = N var_t of survivors; QS density; osc
        pts = [{"g": p["g"], "tau": (p["tau"] if np.isfinite(p["tau"]) else None), "deaths": p["deaths"], "B": p["B"],
                "n_alive": p["n_alive"], "qs": (None if not np.isfinite(p.get("qs", np.nan)) else p["qs"]),
                "chi": p.get("chi"), "var_t": p.get("var_t"),
                "osc": ({k: p["osc"].get(k) for k in ("osc", "osc_strong", "period", "c_min", "c_peak", "coh_ratio",
                                                       "cv_t", "tau_int", "peak_ratio")} if p.get("osc") else None)}
               for p in P]
        chis = [(p["chi"], p["g"]) for p in pts if p["chi"] is not None and 0.68 <= p["g"] <= 11.3]   # common g window
        above = [p for p in pts if on and p["g"] >= on and p["n_alive"] >= 4 and p["qs"] is not None]
        qs_on = above[0]["qs"] if above else None
        rec = {"family": fam, "N": N, "gseed": s, "r": r, "g_lin": glin, "n_points": len(P),
               "S": S, "on": on, "g_tau500": s5, "S_status": c[200.0][1], "on_status": c[3000.0][1],
               "W": (np.log(on / S) if (S and on) else None), "W2": (np.log(on / s5) if (s5 and on) else None),
               "x_on": (on / glin if (on and glin) else None), **bt,
               "qs_at_on": qs_on, "qs_at_on_g": (above[0]["g"] if above else None),
               "chi_max": (max(chis)[0] if chis else None), "chi_max_g": (max(chis)[1] if chis else None),
               "points": pts}
        per.append(rec)
        print(fam, N, s, r, "S", S, "on", on, "W", rec["W"], flush=True)
    # family summaries and decisions
    fams = {}
    for rec in per:
        fams.setdefault((rec["family"] if rec["family"] != "real" else "full", rec["r"]), []).append(rec)
    summ = {}
    for r in (0.0, 1.0, 4.0):
        full_real = [x for x in per if x["family"] == "real" and x["r"] == r]
        Wfull = full_real[0]["W"] if full_real else None
        for fam in ("cnull", "realsubu", "realsub", "realsubcn"):
            recs = [x for x in per if x["family"] == fam and x["r"] == r]
            if fam != "cnull":
                recs = recs + full_real
            if not recs:
                continue
            Ns = sorted({x["N"] for x in recs})
            rows = []
            for N in Ns:
                ws = [x["W"] for x in recs if x["N"] == N and x["W"] is not None]
                w2 = [x["W2"] for x in recs if x["N"] == N and x["W2"] is not None]
                ons = [x["on"] for x in recs if x["N"] == N and x["on"] is not None]
                Ss = [x["S"] for x in recs if x["N"] == N and x["S"] is not None]
                xo = [x["x_on"] for x in recs if x["N"] == N and x["x_on"] is not None]
                ch = [x["chi_max"] for x in recs if x["N"] == N and x["chi_max"] is not None]
                qo = [x["qs_at_on"] for x in recs if x["N"] == N and x["qs_at_on"] is not None]
                rows.append({"N": N, "n_seeds": len(ws), "W_mean": float(np.mean(ws)) if ws else None,
                             "W_sd": float(np.std(ws, ddof=1)) if len(ws) > 1 else None, "W": ws,
                             "W2_mean": float(np.mean(w2)) if w2 else None,
                             "on_mean": float(np.mean(ons)) if ons else None, "S_mean": float(np.mean(Ss)) if Ss else None,
                             "on": ons, "x_on_mean": float(np.mean(xo)) if xo else None,
                             "chi_max_mean": float(np.mean(ch)) if ch else None,
                             "qs_at_on_mean": float(np.mean(qo)) if qo else None})
            ok = [x for x in rows if x["W_mean"] is not None]
            Nv = np.array([x["N"] for x in ok], float); Wv = np.array([x["W_mean"] for x in ok])
            dec = {"spearman_W_N": spearman(Nv, Wv), "n_sizes": len(ok)}
            if fam == "cnull":
                dec["W_ratio_167k_5k"] = (float(Wv[-1] / Wv[0]) if len(ok) >= 2 and Nv[0] == 5000 and Nv[-1] == NFULL
                                          else None)
                dec["sharpening"] = bool(dec["spearman_W_N"] is not None and dec["spearman_W_N"] <= -0.9 and
                                         dec["W_ratio_167k_5k"] is not None and dec["W_ratio_167k_5k"] <= 0.5)
                dec["verdict"] = "sharpening" if dec["sharpening"] else "not sharpening (as pre-registered)"
            else:
                dev = [x["W_mean"] / Wfull - 1 for x in ok if Wfull]
                dec["max_abs_dev_from_full"] = float(max(abs(v) for v in dev)) if dev else None
                dec["W_ratio_full_20k"] = (float(Wv[-1] / Wv[0]) if len(ok) >= 2 and Nv[0] == 20000 else None)
                fk = (dec["max_abs_dev_from_full"] is not None and dec["max_abs_dev_from_full"] <= 0.2 and len(ok) >= 3)
                sh = (dec["spearman_W_N"] is not None and dec["spearman_W_N"] <= -0.999 and len(ok) == 4 and
                      dec["W_ratio_full_20k"] is not None and dec["W_ratio_full_20k"] <= 0.6)
                dec["verdict"] = ("insufficient sizes" if len(ok) < 3 else "finite-K-rounded crossover" if fk else
                                  "sharpening" if sh else "inconclusive / drifting")
            # chi growth exponent (secondary)
            ch = [(x["N"], x["chi_max_mean"]) for x in rows if x["chi_max_mean"]]
            if len(ch) >= 3:
                a = np.polyfit(np.log([c_[0] for c_ in ch]), np.log([c_[1] for c_ in ch]), 1)[0]
                dec["chi_max_growth_exponent"] = float(a)
            # onset vs N (secondary): fit on(N) = on_inf + A N^-b for cnull
            summ[f"{fam}|r{r:g}"] = {"family": fam, "r": r, "sizes": rows, "decision": dec, "W_full_real": Wfull}
            print(fam, r, dec, flush=True)
    res = {"definitions": __doc__, "per_graph": per, "families": summ}
    json.dump(jsafe(res), open(f"{OUT}/fss.json", "w"), indent=1)
    print("->", f"{OUT}/fss.json")


if __name__ == "__main__":
    {"cells": cmd_cells, "analyze": cmd_analyze}[sys.argv[1]]()
