"""m2_decay_analyze.py -- lane m2/decay analysis (results/m2/decay/plan.md; MODEL2.md section 5a).

  python m2_decay_analyze.py sum [--shard i n] [--force]   per-file summaries sim/<run>.npz -> sim/sum/<run>.npz
  python m2_decay_analyze.py cells                          stage-1 boundary brackets -> stage1_cells.json
  python m2_decay_analyze.py analyze                        everything -> boundaries.json, points.json, tables

Definitions are those of analysis2_phase.py (M1) so the comparison is like for like:
  tau = censored-exponential MLE (sum min(t_i, T) / deaths) of the first log-grid record with no active unit;
  isotonic ln tau in g (weights deaths + 0.5; zero deaths regularised as E/0.5), first crossing of ln 200 / ln 3000
  interpolated in ln g; D = isotonic PR/N among points at/above the onset crossing, first crossing of 1/4; uncertainty
  = envelope of the bracketing grid cell and a 2000-resample replica bootstrap.
Exact extinction times (lifetime shape) = max over the three disjoint default regions of reg_last, + dt.
"""
import argparse, glob, json, os, sys, time
import numpy as np
from scipy import stats
from scipy.optimize import isotonic_regression

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import m2_common as M
import m2_obs as O

ROOT = M.ROOT
OUT = f"{ROOT}/results/m2/decay"
SIMD = f"{OUT}/sim"
SUMD = f"{SIMD}/sum"
CACHE = "/MALECNS_CACHE"
N = 167216
DT = 0.2
T_REF, TAU_S, PR_D = 3000.0, 200.0, 0.25
NBOOT = 2000
DEFAULT_REG = ("AL_module", "AL_rest", "CNS_rest")
rng = np.random.default_rng(20261109)


# ------------------------------------------------------------------------------------------------ index and masks
def load_index():
    idx = {}
    for f in sorted(glob.glob(f"{SIMD}/index_*.json")):
        for e in json.load(open(f)):
            idx[os.path.basename(e["out"])] = e
    return idx


_MASKS = None


def masks():
    """Region masks (same selectors as the runs) and superclass labels."""
    global _MASKS
    if _MASKS is None:
        import ald_sim as A
        sign = np.load(f"{CACHE}/sign.npy")
        L = A.Labels(N, sign=sign)
        sel = {"AL_module": "module", "AL_rest": "al_rest", "CNS_rest": "cns_rest"}
        sel.update(M.REGIONS)
        mk = {k: A.select(v, L) for k, v in sel.items()}
        sup = np.load(f"{CACHE}/superclasses.npy", allow_pickle=True).astype(str)
        sup = np.where(sup == "", "(none)", sup)
        su, sinv = np.unique(sup, return_inverse=True)
        _MASKS = {"regions": mk, "sup_names": su, "sup_inv": sinv, "sup_size": np.bincount(sinv, minlength=len(su))}
    return _MASKS


# ------------------------------------------------------------------------------------------------ per-file summary
def summarize_file(path):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    T = float(meta["T"]); t = d["t"]; G = d["g"]
    names = [str(x) for x in d["reg_names"]]; sizes = d["reg_sizes"].astype(float)
    di = [names.index(n) for n in DEFAULT_REG]
    mk = masks()
    occ_all = d["occ"] if "occ" in d.files else None
    has_c = "cnt_act" in d.files; has_cr = "cnt_reg_act" in d.files
    out = {k: [] for k in ["g", "ext_grid", "ext_exact", "pr", "prn", "reg_share", "sup_share", "qs_reg", "qs_tot",
                           "n_alive", "rho_mean", "surv_mean", "reg_share_t", "n_act_t"]}
    osc = []
    for i, g in enumerate(G):
        surv = d["surv"][i]
        ext_grid = O.extinction_times(t, surv)
        last = d["reg_last"][i][di].max(0).astype(np.float64)
        alive = last >= T - DT / 2
        ext_exact = np.where(alive, np.inf, last + DT)
        o = occ_all[i].astype(np.float64) if occ_all is not None else None
        if o is not None and o.sum() > 0:
            pr, prn = O.participation_ratio(o)
            tot = o.sum()
            rs = np.array([o[mk["regions"][n]].sum() / tot for n in names])
            ss = np.bincount(mk["sup_inv"], weights=o, minlength=len(mk["sup_names"])) / tot
        else:
            pr, prn = 0.0, 0.0
            rs = np.full(len(names), np.nan); ss = np.full(len(mk["sup_names"]), np.nan)
        if "win_mean" in d.files and alive.any():
            wm = d["win_mean"][i, 0].astype(np.float64)                  # [R, B]
            qs = wm[:, alive].mean(1)
            qtot = float((wm[di][:, alive] * sizes[di, None]).sum(0).mean() / N)
        else:
            qs = np.full(len(names), np.nan); qtot = np.nan
        rr = d["reg_rho"][i].astype(np.float64)                          # [nt, R, B]
        cnt = rr * sizes[None, :, None]
        ctot = cnt[:, di].sum(1)                                          # [nt, B] total active
        n_act_t = ctot.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            share_t = cnt.sum(2) / n_act_t[:, None]
        oc = {}
        if has_c and alive.sum() >= 4:
            oc["total"] = O.osc_metrics(d["cnt_act"][i][:, alive] / N, DT)
            if has_cr:
                for rn in ("AL_module", "nonORN"):
                    j = names.index(rn)
                    x = d["cnt_reg_act"][i][:, j, alive] / sizes[j]
                    oc[rn] = O.osc_metrics(x, DT) if x.max() > 0 else None
        osc.append(oc)
        out["g"].append(float(g)); out["ext_grid"].append(ext_grid); out["ext_exact"].append(ext_exact)
        out["pr"].append(pr); out["prn"].append(prn); out["reg_share"].append(rs); out["sup_share"].append(ss)
        out["qs_reg"].append(qs); out["qs_tot"].append(qtot); out["n_alive"].append(int(alive.sum()))
        out["rho_mean"].append(d["rho"][i].mean(1)); out["surv_mean"].append(surv.mean(1))
        out["reg_share_t"].append(share_t); out["n_act_t"].append(n_act_t)
    res = {k: np.array(v) for k, v in out.items()}
    res.update(t=t, T=T, reg_names=np.array(names), reg_sizes=sizes, sup_names=mk["sup_names"],
               sup_size=mk["sup_size"], osc=json.dumps(osc, default=float), meta=json.dumps(meta),
               sec_per_step=d["sec_per_step"])
    return res


def cmd_sum(shard=None, force=False):
    os.makedirs(SUMD, exist_ok=True)
    files = sorted(glob.glob(f"{SIMD}/s*.npz"))
    if shard:
        i, n = shard
        files = files[i::n]
    done = 0
    for f in files:
        s = f"{SUMD}/{os.path.basename(f)}"
        if not force and os.path.exists(s) and os.path.getmtime(s) >= os.path.getmtime(f):
            continue
        try:
            r = summarize_file(f)
        except Exception as e:                       # partially written or corrupt file: report, keep going
            print("FAIL", f, repr(e), flush=True)
            continue
        tmp = s + ".tmp.npz"
        np.savez_compressed(tmp, **r); os.replace(tmp, s)
        done += 1
        print("ok", os.path.basename(f), flush=True)
    print("summarized", done)


# ------------------------------------------------------------------------------------------------ pooling per row
def row_key(model, graph, r):
    return f"{model}_{graph}_r{r:g}"


def load_rows(stages=("s1", "s2")):
    """{row_key: {"model", "graph", "r", "points": [pooled per g]}} from the summaries of T = 3000 runs."""
    idx = load_index()
    rows = {}
    for f in sorted(glob.glob(f"{SUMD}/s*.npz")):
        b = os.path.basename(f)
        e = idx.get(b)
        if e is None or e["stage"] not in stages:
            continue
        s = np.load(f, allow_pickle=True)
        if abs(float(s["T"]) - T_REF) > 1e-6:
            continue
        osc = json.loads(str(s["osc"]))
        key = row_key(e["model"], e["graph"], e["r"])
        R = rows.setdefault(key, {"model": e["model"], "graph": e["graph"], "r": float(e["r"]), "raw": []})
        for i, g in enumerate(s["g"]):
            eg = s["ext_grid"][i]
            died = np.isfinite(eg)
            R["raw"].append({"g": float(g), "times": np.where(died, eg, T_REF).astype(float), "died": died,
                             "ext_exact": s["ext_exact"][i], "pr": float(s["prn"][i]), "B": int(len(eg)),
                             "reg_share": s["reg_share"][i], "sup_share": s["sup_share"][i], "qs_reg": s["qs_reg"][i],
                             "qs_tot": float(s["qs_tot"][i]), "n_alive": int(s["n_alive"][i]), "osc": osc[i],
                             "rho_mean": s["rho_mean"][i], "t": s["t"], "reg_share_t": s["reg_share_t"][i],
                             "n_act_t": s["n_act_t"][i], "src": b, "stage": e["stage"]})
        R["reg_names"] = [str(x) for x in s["reg_names"]]; R["sup_names"] = [str(x) for x in s["sup_names"]]
        R["sup_size"] = s["sup_size"]; R["reg_sizes"] = s["reg_sizes"]
    for key, R in rows.items():
        R["raw"].sort(key=lambda x: x["g"])
        pts = []
        for x in R["raw"]:
            if pts and abs(x["g"] / pts[-1]["g"] - 1) < 1e-4:          # same g in two runs: pool replicas
                y = pts[-1]; Bx, By = x["B"], y["B"]
                for k in ("times", "died", "ext_exact"):
                    y[k] = np.r_[y[k], x[k]]
                y["pr"] = (y["pr"] * By + x["pr"] * Bx) / (Bx + By)
                y["B"] += Bx; y["src"] += "+" + x["src"]; y["n_alive"] += x["n_alive"]
                y["osc_list"].append(x["osc"])
            else:
                y = dict(x); y["osc_list"] = [x["osc"]]; pts.append(y)
        for y in pts:
            d_, E = int(y["died"].sum()), float(y["times"].sum())
            y["deaths"], y["E"] = d_, E
            y["tau"] = E / d_ if d_ > 0 else float("inf")
            y["tau_reg"] = E / max(d_, 0.5)
            y["tau_ci"] = ([2 * E / stats.chi2.ppf(0.975, 2 * d_ + 2), 2 * E / stats.chi2.ppf(0.025, 2 * d_)] if d_ > 0
                           else [2 * E / stats.chi2.ppf(0.95, 2), float("inf")])
            y["alive"] = float(1 - y["died"].mean())
        R["points"] = pts
        del R["raw"]
    return rows


# ------------------------------------------------------------------------------------------------ boundaries (= analysis2)
def iso(y, w):
    return isotonic_regression(np.asarray(y, float), weights=np.asarray(w, float), increasing=True).x


def first_cross(x, y, thr, log_y=False):
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
            if D["status"] == "below_grid":
                D = {"g": on["g"], "bracket": on["bracket"], "status": "at_onset"}
    return {"S": S, "on": on, "half": half, "D": D, "_lt": lt}


def g_at_tau(P, taus=(100, 200, 400, 800, 1600, 3000)):
    g = np.array([p["g"] for p in P]); d = np.array([p["deaths"] for p in P], float)
    E = np.array([p["E"] for p in P]); lt = iso(np.log(E / np.maximum(d, 0.5)), d + 0.5)
    return {str(tt): first_cross(g, lt, np.log(tt))["g"] for tt in taus}


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


def summarize_b(est, boot):
    out = {"g": est["g"], "status": est["status"], "bracket": est["bracket"]}
    v = boot[np.isfinite(boot)]
    out["boot_frac_defined"] = float(np.isfinite(boot).mean())
    out["boot_ci"] = [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) >= 20 else None
    if est["g"] is not None:
        los = [x for x in [est["bracket"][0], out["boot_ci"][0] if out["boot_ci"] else None] if x is not None]
        his = [x for x in [est["bracket"][1], out["boot_ci"][1] if out["boot_ci"] else None] if x is not None]
        out["total"] = [float(min(los)) if los else None, float(max(his)) if his else None]
    else:
        out["total"] = None
    return out


def cmd_cells():
    rows = load_rows(stages=("s1",))
    cells = {}
    for key, R in sorted(rows.items()):
        P = R["points"]
        if len(P) < 40:
            print("incomplete row", key, len(P)); continue
        est = sim_boundaries(P)
        c = {"model": R["model"], "graph": R["graph"], "r": R["r"], "n_points": len(P)}
        for b in ("S", "on", "D"):
            e = est[b]
            c[b] = e["bracket"] if e["status"] == "ok" else None
            c[b + "_status"] = e["status"]; c[b + "_g"] = e["g"]
        c["g_at_tau"] = g_at_tau(P)
        cells[key] = c
        print(key, {b: (round(est[b]["g"], 4) if est[b]["g"] else est[b]["status"]) for b in ("S", "on", "D")})
    json.dump(cells, open(f"{OUT}/stage1_cells.json", "w"), indent=1)
    print(len(cells), "rows ->", f"{OUT}/stage1_cells.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--shard", type=int, nargs=2, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "sum":
        cmd_sum(a.shard, a.force)
    elif a.cmd == "cells":
        cmd_cells()
    elif a.cmd == "analyze":
        import m2_decay_report
        m2_decay_report.main()
    else:
        raise SystemExit(a.cmd)
