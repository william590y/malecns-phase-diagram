"""m2_nature_aval.py -- lane m2/nature, avalanches and the Levina-Herrmann-Geisel (LHG 2007) test (plan section 4).

  python m2_nature_aval.py pulse [--shard i n]   per-point avalanche statistics of the pulse runs -> sim/aval/sum/*.json
  python m2_nature_aval.py lhg                   pool per (model, r, G, g) -> results/m2/nature/aval.json (+ LHG verdicts)
  python m2_nature_aval.py driven [--shard i n]  branching ratio / thresholded avalanches of the driven lane's raw runs
                                                 (read only) -> results/m2/nature/driven_branch/*.json
  python m2_nature_aval.py driven_merge          -> results/m2/nature/driven_branch.json
  python m2_nature_aval.py plcheck               pl_fit vs m2_obs.powerlaw_fit on Galton-Watson avalanches -> pl_check.json

Pulse avalanches: m2_obs.avalanches(cnt_act, cnt_on, thr = 0) per replica (size = onsets incl. the seed, duration in
steps of dt = 0.2; runs touching the run ends dropped). Fits: pl_fit below (m2_obs MLE + Clauset scan, tie-correct KS) on sizes and
durations; <S>(T) slope gamma (m2_obs.mean_size_given_T, T >= 2 steps); gamma_pred = (tau_T - 1)/(tau_S - 1).
Shape collapse: mean onset profile of avalanches with duration exactly T (T in 10, 14, 20, 28, 40, 56, 80 steps, >= 20
avalanches each), rescaled to t/T on 20 points and divided by T^(gamma - 1); gamma_collapse minimises the mean variance
across T relative to the mean squared profile (grid 1.0 .. 3.0, step 0.01).
Branching: m2_obs.mr_branching on the per-step onset counts (replicas pooled), m_step; m_unit = m_step^5 (5 steps = one
model time unit = the mean active epoch).
Critical-like (MODEL2 / plan): tau_S in [1.3, 1.7], KS <= 0.03, xmin <= 10, n_tail >= 1000, max size / xmin >= 100.
"""
import glob, json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import m2_common as M
import m2_obs as O
from m2_nature_fss import jsafe

OUT = f"{M.ROOT}/results/m2/nature"
SIM = f"{OUT}/sim/aval"
SUMD = f"{SIM}/sum"
DT = 0.2
T_SHAPES = (10, 14, 20, 28, 40, 56, 80)


def pl_fit(x, n_min=50):
    """Discrete power-law fit = m2_obs.powerlaw_fit's MLE (Clauset 2009 eq. 3.7) with the KS distance evaluated on the
    UNIQUE values (empirical P(X >= u) vs ((u - 0.5)/(xmin - 0.5))^(1 - alpha)). m2_obs evaluates the empirical CCDF at every
    sorted sample, so ties (most avalanche sizes are small integers) inflate its KS: on 1e5 critical Galton-Watson
    avalanches its KS is 0.37 / 0.09 / 0.048 at xmin 1 / 5 / 10 against 0.019 / 0.004 / 0.004 here (results/m2/nature/
    pl_check.json; plan amendment 2). xmin candidates: every integer 1..10, then 20 log-spaced integers up to the value that
    leaves n_min points in the tail; the minimum-KS candidate is kept (Clauset's scan)."""
    x = np.asarray(x, float); x = x[x >= 1]
    if len(x) < n_min:
        return {"alpha": np.nan, "xmin": np.nan, "n_tail": int(len(x)), "ks": np.nan}
    xs = np.sort(x)
    top = xs[-n_min]
    cands = list(range(1, 11)) + list(np.unique(np.round(np.geomspace(11, max(11, top), 20))))
    best = None
    for xm in cands:
        if xm > top:
            break
        y = xs[xs >= xm]
        if len(y) < n_min:
            break
        al = 1.0 + len(y) / np.log(y / (xm - 0.5)).sum()
        if not np.isfinite(al):
            continue
        u, cnt = np.unique(y, return_counts=True)
        emp = 1.0 - np.r_[0, np.cumsum(cnt)[:-1]] / len(y)
        ks = float(np.abs(emp - ((u - 0.5) / (xm - 0.5)) ** (1 - al)).max())
        if best is None or ks < best["ks"]:
            best = {"alpha": float(al), "xmin": float(xm), "n_tail": int(len(y)), "ks": ks}
    return best or {"alpha": np.nan, "xmin": np.nan, "n_tail": 0, "ks": np.nan}


def crit_like(f, smax):
    if f is None or not np.isfinite(f.get("alpha", np.nan)):
        return False
    return bool(1.3 <= f["alpha"] <= 1.7 and f["ks"] <= 0.03 and f["xmin"] <= 10 and f["n_tail"] >= 1000 and
                smax / f["xmin"] >= 100)


def shape_collapse(profiles):
    """profiles: {T: mean profile (length T)}; returns gamma_collapse and the error curve minimum."""
    if len(profiles) < 3:
        return None, None
    grid = np.linspace(0.025, 0.975, 20)
    Ts = sorted(profiles)
    best = (None, np.inf)
    for gam in np.arange(1.0, 3.0001, 0.01):
        Y = []
        for T in Ts:
            p = profiles[T]; tt = (np.arange(T) + 0.5) / T
            Y.append(np.interp(grid, tt, p) / T ** (gam - 1))
        Y = np.array(Y)
        err = float((Y.var(0)).mean() / (Y.mean(0) ** 2).mean())
        if err < best[1]:
            best = (float(gam), err)
    return best


def aval_stats(act, on, gap_note=None):
    """act, on: [S, B] per-step counts of one point. Returns the avalanche statistics dict (+ raw sizes for pooling)."""
    av = O.avalanches(act, on, thr=0)
    S_, T_ = av["size"], av["dur"]
    res = {"n_aval": int(len(S_)), "frac_active": float((act > 0).mean())}
    if len(S_) == 0:
        return res, av
    res.update({"mean_size": float(S_.mean()), "max_size": int(S_.max()), "mean_dur": float(T_.mean()),
                "max_dur": int(T_.max())})
    fs = pl_fit(S_); ft = pl_fit(T_)
    res["fit_S"] = fs; res["fit_T"] = ft
    ms = O.mean_size_given_T(S_, T_)
    res["gamma"] = ms["gamma"]
    if np.isfinite(fs.get("alpha", np.nan)) and np.isfinite(ft.get("alpha", np.nan)) and fs["alpha"] > 1:
        res["gamma_pred"] = float((ft["alpha"] - 1) / (fs["alpha"] - 1))
    res["crit_like"] = crit_like(fs, S_.max())
    # shape collapse: profiles of onsets for exact durations
    prof = {}
    for T in T_SHAPES:
        sel = np.where(T_ == T)[0]
        if len(sel) >= 20:
            P = np.zeros(T)
            for k in sel:
                b, st = av["rep"][k], av["start"][k]
                P += on[st:st + T, b]
            prof[T] = P / len(sel)
    gc, ge = shape_collapse(prof)
    res["gamma_collapse"] = gc; res["collapse_err"] = ge; res["collapse_T"] = sorted(prof)
    mr = O.mr_branching(on.astype(np.float64), kmax=20)
    res["m_step"] = mr["m"]; res["m_unit"] = (mr["m"] ** 5 if np.isfinite(mr["m"]) else None)
    # binned P(S), P(T) for figures (log bins)
    for nm, x in (("S", S_), ("T", T_)):
        bins = np.unique(np.round(np.geomspace(1, max(2, x.max() + 1), 40)))
        h, e = np.histogram(x, bins=bins)
        res[f"hist_{nm}"] = {"edges": e.tolist(), "dens": (h / np.diff(e) / len(x)).tolist()}
    res["meanS_T"] = {"T": ms["T"][:200].tolist(), "meanS": ms["meanS"][:200].tolist(), "n": ms["n"][:200].tolist()}
    return res, av


def cmd_pulse(shard=None):
    os.makedirs(SUMD, exist_ok=True)
    idx = {}
    for f in glob.glob(f"{OUT}/index_aval*.json"):
        for e in json.load(open(f)):
            idx[os.path.basename(e["out"])] = e
    files = sorted(glob.glob(f"{SIM}/*.npz"))
    if shard:
        files = files[shard[0]::shard[1]]
    for f in files:
        s = f"{SUMD}/{os.path.basename(f)[:-4]}.json"
        if os.path.exists(s) and os.path.getmtime(s) >= os.path.getmtime(f):
            continue
        e = idx.get(os.path.basename(f))
        if e is None:
            continue
        try:
            d = np.load(f)
            out = []
            for i, g in enumerate(d["g"]):
                act = d["cnt_act"][i]; on = d["cnt_on"][i]
                res, av = aval_stats(act, on)
                res.update({"g": float(g), "model": e["model"], "r": e["r"], "G": e["G"], "rep": e["rep"],
                            "src": os.path.basename(f)})
                # raw sizes / durations kept (compressed) for pooling across seeds
                np.savez_compressed(f"{SUMD}/{os.path.basename(f)[:-4]}_g{i}.npz", size=av["size"], dur=av["dur"])
                out.append(res)
            tmp = s + ".tmp"
            json.dump(jsafe(out), open(tmp, "w")); os.replace(tmp, s)
            print("ok", os.path.basename(f), flush=True)
        except Exception as ex:
            print("FAIL", f, repr(ex), flush=True)


def bump(h, n, smin=100, ratio=2.0, cmin=20):
    """POST HOC descriptor: does the log-binned P(S) have an interior local maximum at S >= smin, holding >= cmin
    avalanches, that is >= ratio x the lowest density between 10 and that maximum (a separate mode of large avalanches =
    population bursts)?"""
    if not h or not n:
        return False, None
    e = np.array(h["edges"]); dns = np.array(h["dens"]); c = np.sqrt(e[:-1] * e[1:])
    cnt = dns * np.diff(e) * n
    ok = dns > 0
    c, dns, cnt = c[ok], dns[ok], cnt[ok]
    for k in range(1, len(dns) - 1):
        if c[k] >= smin and cnt[k] >= cmin and dns[k] >= dns[k - 1] and dns[k] >= dns[k + 1]:
            lo = dns[(c >= 10) & (c < c[k])]
            if len(lo) and dns[k] >= ratio * lo.min():
                return True, float(c[k])
    return False, None


def exp_only(p):
    """POST HOC: the critical-like rule without the KS condition."""
    f = p.get("fit_S") or {}
    a = f.get("alpha")
    return bool(a is not None and np.isfinite(a) and 1.3 <= a <= 1.7 and f["xmin"] <= 10 and f["n_tail"] >= 1000 and
                (p.get("max_size") or 0) / f["xmin"] >= 100)


def span(gs_all, flags):
    """Longest run of consecutive grid points flagged; returns (span max/min of that run, all-points max/min)."""
    best, cur, bl = 0, [], []
    for g, fl in zip(gs_all, flags):
        if fl:
            cur.append(g)
            if len(cur) > best:
                best, bl = len(cur), list(cur)
        else:
            cur = []
    crit = [g for g, f in zip(gs_all, flags) if f]
    return (bl[-1] / bl[0] if bl else None), (max(crit) / min(crit) if crit else None), bl


def cmd_lhg():
    pts = {}
    for s in sorted(glob.glob(f"{SUMD}/*.json")):
        for p in json.load(open(s)):
            pts.setdefault((p["model"], float(p["r"]), int(p["G"]), round(p["g"], 5)), []).append(p)
    # pooled fits where several seeds exist
    pooled = []
    for key, ps in sorted(pts.items()):
        if len(ps) == 1:
            q = dict(ps[0]); q["n_files"] = 1
        else:
            Ss, Ts = [], []
            for p in ps:
                i = None
                d = np.load(f"{SIM}/{p['src']}")
                i = int(np.argmin(np.abs(d["g"] - p["g"])))
                z = np.load(f"{SUMD}/{p['src'][:-4]}_g{i}.npz"); Ss.append(z["size"]); Ts.append(z["dur"])
            S_, T_ = np.concatenate(Ss), np.concatenate(Ts)
            fs = pl_fit(S_); ft = pl_fit(T_)
            q = dict(ps[0]); q.update({"n_files": len(ps), "n_aval": int(len(S_)), "fit_S": fs, "fit_T": ft,
                                      "max_size": int(S_.max()), "crit_like": crit_like(fs, S_.max()),
                                      "gamma": O.mean_size_given_T(S_, T_)["gamma"],
                                      "m_unit": float(np.mean([p["m_unit"] for p in ps if p.get("m_unit")]))})
        pooled.append(q)
    # LHG per (r, G)
    G_AVAL = M.G_COARSE[(M.G_COARSE >= 0.1) & (M.G_COARSE <= 3.05)]
    cells = {}
    for r in M.R_CORE:
        for G in (0, 10, 100):
            c = {}
            for model in ("M1", "M2"):
                rows = sorted([p for p in pooled if p["model"] == model and p["r"] == r and p["G"] == G],
                              key=lambda p: p["g"])
                coarse = [p for p in rows if np.min(np.abs(G_AVAL / p["g"] - 1)) < 1e-4]
                gs = [p["g"] for p in coarse]; fl = [bool(p.get("crit_like")) for p in coarse]
                sp_run, sp_all, run = span(gs, fl)
                fe = [exp_only(p) for p in coarse]
                se_run, se_all, run_e = span(gs, fe)
                bm = [bump(p.get("hist_S"), p.get("n_aval")) for p in coarse]
                c[model] = {"g": gs, "crit_like": fl, "span_run": sp_run, "span_all": sp_all, "run": run,
                            "posthoc_exp_only": fe, "posthoc_exp_only_span_run": se_run, "posthoc_exp_only_run": run_e,
                            "posthoc_bump": [b[0] for b in bm], "posthoc_bump_S": [b[1] for b in bm],
                            "posthoc_first_bump_g": next((g for g, b in zip(gs, bm) if b[0]), None),
                            "tau_S": [p.get("fit_S", {}).get("alpha") for p in coarse],
                            "ks": [p.get("fit_S", {}).get("ks") for p in coarse],
                            "xmin": [p.get("fit_S", {}).get("xmin") for p in coarse],
                            "n_tail": [p.get("fit_S", {}).get("n_tail") for p in coarse],
                            "max_size": [p.get("max_size") for p in coarse],
                            "tau_T": [p.get("fit_T", {}).get("alpha") for p in coarse],
                            "gamma": [p.get("gamma") for p in coarse], "gamma_pred": [p.get("gamma_pred") for p in coarse],
                            "gamma_collapse": [p.get("gamma_collapse") for p in coarse],
                            "m_unit": [p.get("m_unit") for p in coarse], "mean_size": [p.get("mean_size") for p in coarse],
                            "frac_active": [p.get("frac_active") for p in coarse], "n_aval": [p.get("n_aval") for p in coarse]}
            s2 = c["M2"]["span_run"] or 0; s1 = c["M1"]["span_run"]
            any2 = any(c["M2"]["crit_like"]); any1 = any(c["M1"]["crit_like"])
            if not any1 and not any2:
                v = "inconclusive"
            elif s2 >= 1.5 and (s1 is None or s1 < 1.2):
                v = "supported"
            elif s2 < 1.2:
                v = "not supported"
            else:
                v = "mixed"
            c["verdict"] = v
            cells[f"r{r:g}|G{G}"] = c
            print(r, G, "M2 span", c["M2"]["span_run"], "M1 span", c["M1"]["span_run"], v, flush=True)
    from collections import Counter
    cnt = Counter(c["verdict"] for c in cells.values())
    overall = cnt.most_common(1)[0][0] if cnt else None
    ties = [k for k, v in cnt.items() if v == cnt[overall]] if cnt else []
    json.dump(jsafe({"definitions": __doc__, "points": pooled, "cells": cells, "verdict_counts": dict(cnt),
                     "overall": overall if len(ties) == 1 else "tie: " + "/".join(sorted(ties))}),
              open(f"{OUT}/aval.json", "w"), indent=1)
    print(dict(cnt), overall)


# ------------------------------------------------------------------------------------------------ driven lane data
DRV = f"{M.ROOT}/results/m2/driven"
DRV_OUT = f"{OUT}/driven_branch"


def thr_avalanches(a):
    """Per-replica thresholded avalanches (threshold = that replica's median count; size = area above it)."""
    S_, T_ = [], []
    for b in range(a.shape[1]):
        x = a[:, b].astype(np.int64)
        th = int(np.median(x))
        av = O.avalanches(x[:, None], None, thr=th)
        S_.append(av["size"]); T_.append(av["dur"])
    return np.concatenate(S_), np.concatenate(T_)


def cmd_driven(shard=None):
    os.makedirs(DRV_OUT, exist_ok=True)
    idx = []
    for f in ("index_A.json",):
        idx += json.load(open(f"{DRV}/{f}"))
    idx = [e for e in idx if e["set"] in ("A1", "A2", "A3", "A4")]
    if shard:
        idx = idx[shard[0]::shard[1]]
    for e in idx:
        f = e["out"]
        o = f"{DRV_OUT}/{os.path.basename(f)[:-4]}.json"
        if os.path.exists(o) or not os.path.exists(f):
            if not os.path.exists(f):
                print("missing", f)
            continue
        try:
            d = np.load(f)
            names = [str(x) for x in d["reg_names"]]; j = names.index("nonORN")
            res = []
            for i, g in enumerate(d["g"]):
                on = d["cnt_reg_on"][i][:, j, :].astype(np.float64)
                act = d["cnt_reg_act"][i][:, j, :]
                mr = O.mr_branching(on, kmax=20)
                mra = O.mr_branching(act.astype(np.float64), kmax=20)
                q = {"g": float(g), "set": e["set"], "model": e["model"], "graph": e["graph"], "r": e["r"],
                     "m_step": mr["m"], "m_unit": (mr["m"] ** 5 if np.isfinite(mr["m"]) else None),
                     "m_step_act": mra["m"], "mean_act": float(act.mean()), "mean_on": float(on.mean())}
                if act.mean() >= 1:
                    S_, T_ = thr_avalanches(act)
                    if len(S_) >= 50:
                        fs = pl_fit(S_); ft = pl_fit(T_)
                        q.update({"n_aval": int(len(S_)), "fit_S": fs, "fit_T": ft, "max_size": int(S_.max()),
                                  "gamma": O.mean_size_given_T(S_, T_)["gamma"], "crit_like": crit_like(fs, S_.max())})
                res.append(q)
            json.dump(jsafe(res), open(o + ".tmp", "w")); os.replace(o + ".tmp", o)
            print("ok", os.path.basename(f), flush=True)
        except Exception as ex:
            print("FAIL", f, repr(ex), flush=True)


def cmd_driven_merge():
    pts = []
    for f in sorted(glob.glob(f"{DRV_OUT}/*.json")):
        pts += json.load(open(f))
    rows = {}
    for p in pts:
        rows.setdefault(f"{p['model']}|{p['graph']}|r{p['r']:g}", []).append(p)
    summ = {}
    for k, ps in rows.items():
        ps.sort(key=lambda p: p["g"])
        gs = [p["g"] for p in ps]
        fl = [p["m_unit"] is not None and abs(1 - p["m_unit"]) < 0.01 for p in ps]
        fl_step = [p["m_step"] is not None and abs(1 - p["m_step"]) < 0.01 for p in ps]
        sp, spa, run = span(gs, fl)
        sps, spsa, runs = span(gs, fl_step)
        summ[k] = {"g": gs, "m_unit": [p["m_unit"] for p in ps], "m_step": [p["m_step"] for p in ps],
                   "marker_span_unit": sp, "marker_run_unit": run, "marker_span_step": sps, "marker_run_step": runs,
                   "marker_unit": bool(sp is not None and sp >= 1.5), "marker_step": bool(sps is not None and sps >= 1.5),
                   "crit_like_thr": [p.get("crit_like") for p in ps], "tau_S_thr": [p.get("fit_S", {}).get("alpha")
                                                                                    for p in ps]}
    json.dump(jsafe({"definitions": __doc__, "rows": summ, "points": pts}), open(f"{OUT}/driven_branch.json", "w"),
              indent=1)
    for k, v in sorted(summ.items()):
        print(k, "unit marker", v["marker_unit"], v["marker_span_unit"], "step marker", v["marker_step"], v["marker_span_step"])


def cmd_plcheck():
    rng = np.random.default_rng(1)
    out = {}
    for m in (1.0, 0.9, 0.8):
        S = []; T = []
        for _ in range(100000):
            z, s_, t = 1, 0, 0
            while z > 0 and t < 5000:
                s_ += z; t += 1; z = rng.poisson(m * z)
            if z == 0:
                S.append(s_); T.append(t)
        S = np.array(S)
        f_new = pl_fit(S); f_old = O.powerlaw_fit(S)
        f_old10 = min((O.powerlaw_fit(S, xmin=k) for k in range(1, 11)), key=lambda f: f["ks"])
        out[f"m{m:g}"] = {"pl_fit": f_new, "crit_like_pl_fit": crit_like(f_new, S.max()), "m2_obs_scan": f_old,
                          "crit_like_m2_obs_scan": crit_like(f_old, S.max()), "m2_obs_best_xmin_le_10": f_old10,
                          "n": int(len(S)), "max_size": int(S.max())}
        print(m, out[f"m{m:g}"])
    json.dump(jsafe(out), open(f"{OUT}/pl_check.json", "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    sh = (int(a[a.index("--shard") + 1]), int(a[a.index("--shard") + 2])) if "--shard" in a else None
    {"pulse": lambda: cmd_pulse(sh), "lhg": cmd_lhg, "driven": lambda: cmd_driven(sh),
     "driven_merge": cmd_driven_merge, "plcheck": cmd_plcheck}[a[0]]()
