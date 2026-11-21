"""m2_driven_analyze.py -- lane m2/driven analysis (results/m2/driven/plan.md; 2026-11-09).

Stage 1  extract:  every run file runs/<set>/<name>.npz -> points/<set>/<name>.npz (per-replica arrays per g) + .json
                   (per-point scalars, MODEL2 5b observables + phase label). Cached; redone when the run file is newer.
Stage 2  analyse:  curves (set, model, graph, ORN level, r, palp) sorted in g; lines with a 95 % replica bootstrap
                   (2000 resamples) and the bracketing grid cell; peak tests; field slope / static susceptibility from
                   the ORN levels; the "tracks an instability" test.
Outputs: results/m2/driven/points.csv, curves.json, summary.json, table.md
Usage:   python m2_driven_analyze.py [extract|analyse|all] [--jobs N]
"""
import csv, glob, json, os, sys, time
import numpy as np
from scipy.optimize import isotonic_regression
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_common as M
import m2_obs as O

ROOT = M.ROOT
OUT = f"{ROOT}/results/m2/driven"
PTS = f"{OUT}/points"
CACHE = "/MALECNS_CACHE"
N, N_NON, N_AL = 167216, 164577, 526
DT, T1, TEND, TPOST = 0.2, 1500.0, 4500.0, 3000.0
TAU_SIL, TAU_SS = 200.0, 3000.0
G_LO, G_HI, G_BROAD = 0.29, 1.45, (0.077, 1.67)
S1_THR, S2_MAX, S2_LEVEL = 0.10, 3, 3.5 / 64
NBOOT = int(os.environ.get("M2D_NBOOT", 2000))       # interim looks may lower it; final runs use 2000
STEP = float(M.G_COARSE[1] / M.G_COARSE[0])         # 1.178, one coarse grid step

_masks = {}


def masks():
    if not _masks:
        orn = np.zeros(N, bool); orn[np.load(f"{M.GRP}/ORN.npz")["idx"]] = True
        mod = np.zeros(N, bool); mod[np.load(f"{ROOT}/results/module/module_graph.npz")["idx"]] = True
        sc = np.load(f"{CACHE}/superclasses.npy", allow_pickle=True)
        names, codes = np.unique(sc, return_inverse=True)
        _masks.update(orn=orn, non=~orn, mod=mod, sc_names=names, sc_codes=codes,
                      sc_frac=np.bincount(codes[~orn], minlength=len(names)) / (~orn).sum())
    return _masks


def per_rep_tau_int(x):
    """Per-replica integrated autocorrelation time (own mean/variance; up to the first zero of its own ACF)."""
    x = np.asarray(x, np.float64); T = x.shape[0]
    xc = x - x.mean(0); v = (xc * xc).mean(0)
    n = 1 << int(np.ceil(np.log2(2 * T)))
    F = np.fft.rfft(xc, n, axis=0)
    c = np.fft.irfft(F * np.conj(F), n, axis=0)[:T // 2 + 1] / T
    out = np.full(x.shape[1], np.nan)
    for b in np.where(v > 0)[0]:
        out[b] = O.tau_ac(c[:, b] / v[b], DT)[0]
    return out


def win_index(wb, lo, hi):
    for k, (a, b) in enumerate(np.asarray(wb)):
        if abs(a - lo) < 1e-6 and abs(b - hi) < 1e-6:
            return k
    raise KeyError((lo, hi))


def extract_file(path, meta):
    """One run file -> list of per-point scalar dicts and per-replica arrays."""
    mk = masks()
    z = np.load(path, allow_pickle=True)
    side = json.load(open(path[:-4] + ".json"))
    names = [str(n) for n in z["reg_names"]]; ix = {n: i for i, n in enumerate(names)}
    iO, iN, iU, iM, iA = ix["ORN"], ix["nonORN"], ix["uPN_ant"], ix["AL_module"], ix["ALsys"]
    wb = z["win_bounds"]
    w0, w1, w2_ = win_index(wb, 500, 1500), win_index(wb, 1600, 1800), win_index(wb, 2500, 4500)
    wm, wq, last, occ = z["win_mean"], z["win_m2"], z["reg_last"], z["occ"]
    cnt = z["cnt_reg_act"]
    t = z["t"]
    resx = z["res_x"] if "res_x" in z.files else None
    resxa = z["res_x_act"] if "res_x_act" in z.files else None
    tsel = (t > 500) & (t <= 1500)
    pts, reps = [], []
    for gi, g in enumerate(z["g"]):
        g = float(g)
        r_orn, r_upn, r_non, r_al = (wm[gi, w0, i].astype(np.float64) for i in (iO, iU, iN, iM))
        v_non = np.maximum(wq[gi, w0, iN] - wm[gi, w0, iN] ** 2, 0).astype(np.float64)
        v_al = np.maximum(wq[gi, w0, iM] - wm[gi, w0, iM] ** 2, 0).astype(np.float64)
        upn_post, upn_late = wm[gi, w1, iU].astype(np.float64), wm[gi, w2_, iU].astype(np.float64)
        orn_post = float(max(wm[gi, w1, iO].max(), wm[gi, w2_, iO].max()))
        alive_alsys = last[gi, iA] >= TEND - 0.1
        lastn = last[gi, iN].astype(np.float64)
        ext = np.where(lastn >= TEND - 0.1, np.inf, np.maximum(lastn + DT - T1, 0.0))
        x_non = cnt[gi][:, iN, :].astype(np.float64) / N_NON
        x_al = cnt[gi][:, iM, :].astype(np.float64) / N_AL
        om = O.osc_metrics(x_non, DT)
        oma = O.osc_metrics(x_al, DT)
        ti_rep = per_rep_tau_int(x_non)
        h = x_non.shape[0] // 2
        m_all, m1, m2 = x_non.mean(), x_non[:h].mean(), x_non[h:].mean()
        exempt = m_all * N_NON < 1.0
        ns = bool((not exempt) and abs(m1 - m2) > 0.10 * m_all)
        o = occ[gi].astype(np.float64)
        pr, prn = O.participation_ratio(o, mk["non"])
        on = o[mk["non"]].sum()
        mod_share = float(o[mk["mod"] & mk["non"]].sum() / on) if on > 0 else np.nan
        sc = np.bincount(mk["sc_codes"][mk["non"]], weights=o[mk["non"]], minlength=len(mk["sc_names"]))
        top = []
        if on > 0:
            for k in np.argsort(-sc)[:5]:
                top.append([str(mk["sc_names"][k]) or "(none)", round(float(sc[k] / on), 4),
                            round(float(sc[k] / on / mk["sc_frac"][k]), 2) if mk["sc_frac"][k] > 0 else None])
        rho_orn, rho_upn, rho_non, rho_al = r_orn.mean(), r_upn.mean(), r_non.mean(), r_al.mean()
        G = rho_upn / rho_orn if rho_orn > 0 else np.nan
        s1res = upn_post.mean() / rho_upn if rho_upn > 0 else np.nan
        s1, s1_1 = bool(s1res <= S1_THR), bool(s1res <= 0.01)
        n_alive = int(alive_alsys.sum()); s2 = n_alive <= S2_MAX
        tau, d = O.tau_mle(ext, TPOST)
        E = float(np.minimum(ext, TPOST).sum())
        tau_reg = E / max(d, 0.5)
        palp = bool(meta.get("palp"))
        chi_non = N_NON * v_non.mean(); chi_al = N_AL * v_al.mean()
        p = {"set": meta["set"], "model": meta["model"], "graph": meta["graph"], "orn": meta["orn"], "r": meta["r"],
             "palp": palp, "g": g, "B": int(len(r_orn)), "file": os.path.relpath(path, ROOT), "seed": meta["seed"],
             "rho_ORN": rho_orn, "rho_uPN": rho_upn, "rho_nonORN": rho_non, "rho_AL": rho_al,
             "n_active_nonORN": rho_non * N_NON, "G": G,
             "chi_non": chi_non, "F_non": chi_non / rho_non if rho_non > 0 else np.nan,
             "chi_AL": chi_al, "F_AL": chi_al / rho_al if rho_al > 0 else np.nan,
             "tau_int": om["tau_int"], "tau_e": om["tau_e"], "tau_int_rep": float(np.nanmean(ti_rep)) if np.isfinite(ti_rep).any() else np.nan,
             "osc": om["osc"], "osc_strong": om["osc_strong"], "cv_t": om["cv_t"], "period": om["period"],
             "c_min": om["c_min"], "c_peak": om["c_peak"], "coh": om["coh_ratio"], "f_peak": om["f_peak"],
             "peak_ratio": om["peak_ratio"], "osc_AL": oma["osc"], "osc_strong_AL": oma["osc_strong"],
             "period_AL": oma["period"], "cv_t_AL": oma["cv_t"],
             "stat_m1": m1, "stat_m2": m2, "NS": ns, "NS_exempt": bool(exempt),
             "PR_non": pr, "PR_non_n": prn, "mod_share": mod_share, "top_superclasses": top,
             "S1_resid": s1res, "S1": s1, "S1_1pct": s1_1, "S2_alive": n_alive, "S2": s2,
             "tau_post": tau, "deaths_post": d, "tau_post_reg": tau_reg, "ORN_after_removal_max": orn_post,
             "S1p_resid": s1res, "S2p_resid": upn_late.mean() / rho_upn if rho_upn > 0 else np.nan}
        if resx is not None:
            for nm in ("ORN", "nonORN", "AL_module", "lLN1_bc", "uPN_ant"):
                p[f"x_{nm}"] = float(np.nanmean(resx[0, gi, tsel, ix[nm]])) if np.isfinite(resx[0, gi, tsel, ix[nm]]).any() else np.nan
                if resxa is not None:
                    v = resxa[0, gi, tsel, ix[nm]]
                    p[f"xact_{nm}"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        # classification (MODEL2 5b)
        if palp:
            p["removal_class"] = "palp-protocol"
            p["phase"] = "palp"
            p["palp_ok"] = bool(p["S1p_resid"] <= 0.1 and p["S2p_resid"] <= 0.1)
        else:
            cls = ("self-sustained" if tau >= TAU_SS else ("silent" if (tau < TAU_SIL and s1 and s2) else "transient"))
            p["removal_class"] = cls
            if cls == "self-sustained":
                base = "SS-" + ("D" if prn >= 0.25 else "L")
            elif cls == "transient":
                base = "DT"
            else:
                base = "DA" if G >= G_LO else "DQ"
            lab = base + ("-D" if (prn >= 0.25 and not base.startswith("SS")) else "") + ("-O" if om["osc"] else "") \
                + ("-NS" if ns else "")
            p["phase_base"], p["phase"] = base, lab
        p["allowed"] = bool(G_LO <= G <= G_HI and s1 and s2) if not palp else None
        p["allowed_broad"] = bool(G_BROAD[0] <= G <= G_BROAD[1] and s1 and s2) if not palp else None
        pts.append(p)
        reps.append({"rho_orn": r_orn, "rho_upn": r_upn, "rho_non": r_non, "rho_al": r_al, "v_non": v_non,
                     "v_al": v_al, "upn_post": upn_post, "upn_late": upn_late, "alive_alsys": alive_alsys,
                     "ext": ext, "tau_int_rep": ti_rep})
    return pts, reps, side


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def extract_one(args):
    path, meta = args
    stem = f"{PTS}/{meta['set']}/{meta['name']}"
    if os.path.exists(stem + ".json") and os.path.getmtime(stem + ".json") >= os.path.getmtime(path):
        return "cached"
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    pts, reps, side = extract_file(path, meta)
    arr = {f"{k}__{i}": v for i, rp in enumerate(reps) for k, v in rp.items()}
    np.savez(stem + ".npz", **arr)
    info = {"md5_sim": None, "gpu": side.get("gpu"), "ms_per_step": side.get("ms_per_step"), "wall_s": side.get("wall_s"),
            "points": [{k: _clean(v) for k, v in p.items()} for p in pts]}
    json.dump(info, open(stem + ".json", "w"), indent=0, default=_clean)
    return "done"


def load_index():
    meta = []
    for f in sorted(glob.glob(f"{OUT}/index_*.json")):
        for m in json.load(open(f)):
            m["wave"] = os.path.basename(f)[6:-5]
            meta.append(m)
    return meta


def extract_all(jobs=8):
    meta = load_index()
    todo = [(m["out"], m) for m in meta if os.path.exists(m["out"]) and os.path.exists(m["out"][:-4] + ".json")]
    missing = [m["out"] for m in meta if not (os.path.exists(m["out"]) and os.path.exists(m["out"][:-4] + ".json"))]
    t0 = time.time()
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            res = pool.map(extract_one, todo, chunksize=1)
    else:
        res = [extract_one(a) for a in todo]
    print(f"extract: {res.count('done')} new, {res.count('cached')} cached, {len(missing)} missing run files "
          f"({time.time() - t0:.0f} s)")
    return missing


# ------------------------------------------------------------------------------------------------ curves
def curve_key(p):
    return (p["set"], p["model"], p["graph"], p["orn"], p["r"], p["palp"])


def load_curves():
    """{key: {'g': [...], 'pts': [...], 'reps': {name: [n_g, B]}}}; a refinement point joins the curve of its parent
    (set C* lines carry the parent set name in 'set'). Duplicate g (|ratio - 1| < 1e-4): the first file wins."""
    meta = load_index()
    cur = {}
    for m in meta:
        stem = f"{PTS}/{m['set']}/{m['name']}"
        if not os.path.exists(stem + ".json"):
            continue
        info = json.load(open(stem + ".json"))
        arr = np.load(stem + ".npz")
        for i, p in enumerate(info["points"]):
            k = curve_key(p)
            c = cur.setdefault(k, {"g": [], "pts": [], "reps": []})
            if any(abs(p["g"] / g0 - 1) < 1e-4 for g0 in c["g"]):
                continue
            c["g"].append(p["g"]); c["pts"].append(p)
            c["reps"].append({kk.split("__")[0]: arr[kk] for kk in arr.files if kk.endswith(f"__{i}")})
    for k, c in cur.items():
        o = np.argsort(c["g"])
        c["g"] = np.array(c["g"])[o]; c["pts"] = [c["pts"][i] for i in o]
        reps = [c["reps"][i] for i in o]
        c["reps"] = {nm: np.stack([rp[nm] for rp in reps]) for nm in reps[0]}
    return cur


def _iso(y, w, increasing=True):
    return isotonic_regression(np.asarray(y, float), weights=np.asarray(w, float), increasing=increasing).x


def _cross_cell(g0, g1, y0, y1, lv):
    fr = 0.0 if y1 == y0 else (lv - y0) / (y1 - y0)
    fr = min(max(fr, 0.0), 1.0)
    return float(np.exp(np.log(g0) + fr * (np.log(g1) - np.log(g0))))


def first_up(g, y, lv):
    """First upward crossing of lv by the raw curve y(g): returns (g_cross, k) with k the first index >= lv (None if never;
    k = 0 -> below the grid)."""
    hit = np.where(np.asarray(y) >= lv)[0]
    if len(hit) == 0:
        return None, None
    k = int(hit[0])
    if k == 0:
        return float(g[0]), 0
    return _cross_cell(g[k - 1], g[k], y[k - 1], y[k], lv), k


def iso_cross(g, y, w, lv, increasing=True):
    yi = _iso(y if increasing else -np.asarray(y), w)
    return first_up(g, yi, lv if increasing else -lv)


def stats_from(reps, idx=None):
    """Per-g curve statistics from per-replica arrays (idx: [n_g, B] resample indices or None)."""
    def take(a):
        return a if idx is None else np.take_along_axis(a, idx, 1)
    ro, ru, rn, ra = take(reps["rho_orn"]), take(reps["rho_upn"]), take(reps["rho_non"]), take(reps["rho_al"])
    mo, mu, mn, ma = ro.mean(1), ru.mean(1), rn.mean(1), ra.mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = mu / mo
        s1 = take(reps["upn_post"]).mean(1) / mu
        s2p = take(reps["upn_late"]).mean(1) / mu
        F_al = N_AL * take(reps["v_al"]).mean(1) / ma
        F_non = N_NON * take(reps["v_non"]).mean(1) / mn
    chi_non = N_NON * take(reps["v_non"]).mean(1)
    alive = take(reps["alive_alsys"].astype(float)).mean(1)
    ext = take(reps["ext"])
    d = np.isfinite(ext).sum(1)
    E = np.minimum(ext, TPOST).sum(1)
    tau = np.where(d > 0, E / np.maximum(d, 1), np.inf)
    tau_reg = E / np.maximum(d, 0.5)
    ti = np.nanmean(take(reps["tau_int_rep"]), 1)
    return {"G": G, "s1": np.nan_to_num(s1, nan=0.0), "s2p": s2p, "alive": alive, "tau": tau, "tau_reg": tau_reg,
            "d": d, "F_AL": F_al, "F_non": F_non, "chi_non": chi_non, "tau_int": ti, "rho_non": mn, "rho_orn": mo}


def lines_from(g, st, prn):
    """Lines of one curve realisation (MODEL2 5b / plan section 3)."""
    out = {}
    G = np.nan_to_num(st["G"], nan=0.0)
    out["DQ_DA"] = first_up(g, G, G_LO)[0]
    silent = (st["tau"] < TAU_SIL) & (st["s1"] <= S1_THR) & (st["alive"] * 64 <= S2_MAX + 1e-9)
    allowed = (G >= G_LO) & (G <= G_HI) & (st["s1"] <= S1_THR) & (st["alive"] * 64 <= S2_MAX + 1e-9)
    # allowed region: first run of passing points and its top
    a = np.where(allowed)[0]
    out["allowed_lo"] = out["allowed_hi"] = out["allowed_hi_by"] = None
    if len(a):
        # top = first failure of an UPPER criterion (G > 1.45, S1, S2) after the first allowed point; a dip of G below
        # 0.29 just above the lower edge (noise where G ~ 0.29 over a range of g) does not end the region
        # (amendment 1 in plan.md, made after the interim look)
        upper_ok = (G <= G_HI) & (st["s1"] <= S1_THR) & (st["alive"] * 64 <= S2_MAX + 1e-9)
        k0 = a[0]; k1 = k0
        while k1 + 1 < len(g) and upper_ok[k1 + 1]:
            k1 += 1
        out["allowed_lo"] = out["DQ_DA"] if (out["DQ_DA"] is not None and out["DQ_DA"] <= g[k0] + 1e-12) else float(g[k0])
        if k1 + 1 < len(g):
            k = k1 + 1
            cands = []
            if G[k] > G_HI:
                cands.append((_cross_cell(g[k - 1], g[k], G[k - 1], G[k], G_HI), "G>1.45"))
            if st["s1"][k] > S1_THR:
                cands.append((_cross_cell(g[k - 1], g[k], st["s1"][k - 1], st["s1"][k], S1_THR), "S1"))
            if st["alive"][k] * 64 > S2_MAX:
                cands.append((_cross_cell(g[k - 1], g[k], st["alive"][k - 1], st["alive"][k], S2_LEVEL), "S2"))
            if cands:
                c = min(cands)
                out["allowed_hi"], out["allowed_hi_by"] = c[0], c[1]
            else:
                out["allowed_hi"], out["allowed_hi_by"] = float(g[k]), "?"
        else:
            out["allowed_hi"], out["allowed_hi_by"] = float(g[k1]), "grid_end"
        out["allowed_runs"] = int(np.sum(np.diff(np.r_[0, allowed.astype(int)]) == 1))
    # DA -> DT: first non-silent point after a silent one
    lt = np.log(np.maximum(st["tau_reg"], 0.1))
    out["silent_lost"] = out["silent_lost_by"] = None
    s = np.where(silent)[0]
    if len(s):
        ns_ = np.where(~silent[s[0]:])[0]
        if len(ns_):
            k = s[0] + ns_[0]
            cands = []
            if st["tau"][k] >= TAU_SIL:
                cands.append((_cross_cell(g[k - 1], g[k], lt[k - 1], lt[k], np.log(TAU_SIL)), "tau_post>=200"))
            if st["s1"][k] > S1_THR:
                cands.append((_cross_cell(g[k - 1], g[k], st["s1"][k - 1], st["s1"][k], S1_THR), "S1"))
            if st["alive"][k] * 64 > S2_MAX:
                cands.append((_cross_cell(g[k - 1], g[k], st["alive"][k - 1], st["alive"][k], S2_LEVEL), "S2"))
            if cands:
                c = min(cands); out["silent_lost"], out["silent_lost_by"] = c[0], c[1]
    w = st["d"] + 0.5
    out["tau200"] = iso_cross(g, lt, w, np.log(TAU_SIL))[0]
    out["DT_SS"] = iso_cross(g, lt, w, np.log(TAU_SS))[0]
    out["SS_D"] = iso_cross(g, prn, np.ones_like(prn), 0.25)[0] if prn is not None else None
    return out


def peak_info(g, y):
    """Grid argmax, parabola refinement in ln g through the 3 points around it, interior flag, beats both ends."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return None, None, False, False
    yy = np.where(ok, y, -np.inf)
    k = int(np.argmax(yy))
    interior = 0 < k < len(g) - 1 and ok[k - 1] and ok[k + 1]
    ends = [v for v in (y[0], y[-1]) if np.isfinite(v)]
    beats = bool(all(y[k] > e for e in ends))
    gp = float(g[k])
    if interior:
        x = np.log(g[k - 1:k + 2]); a, b, _ = np.polyfit(x, y[k - 1:k + 2], 2)
        if a < 0:
            gp = float(np.exp(min(max(-b / (2 * a), x[0]), x[2])))
    return gp, k, bool(interior), beats


def analyse_curve(key, c, rng):
    g = c["g"]; reps = c["reps"]; pts = c["pts"]
    prn = np.array([p["PR_non_n"] if p["PR_non_n"] is not None else np.nan for p in pts])
    st = stats_from(reps)
    res = {"key": dict(zip(("set", "model", "graph", "orn", "r", "palp"), key)), "g": g.tolist(), "n_g": len(g)}
    res["lines"] = lines_from(g, st, prn) if not key[5] else {}
    B = reps["rho_orn"].shape[1]
    boot = {k: [] for k in res["lines"] if not k.endswith("_by") and k != "allowed_runs"}
    peaks_b = {k: [] for k in ("F_AL", "F_non", "chi_non", "tau_int")}
    for _ in range(NBOOT):
        idx = rng.integers(0, B, size=(len(g), B))
        sb = stats_from(reps, idx)
        if not key[5]:
            lb = lines_from(g, sb, prn)
            for k in boot:
                boot[k].append(np.nan if lb.get(k) is None else lb[k])
        for k in peaks_b:
            peaks_b[k].append(peak_info(g, sb[k]))
    ci = {}
    for k, v in boot.items():
        v = np.array(v, float)
        f = np.isfinite(v).mean()
        ci[k] = {"lo": float(np.nanpercentile(v, 2.5)) if f > 0 else None,
                 "hi": float(np.nanpercentile(v, 97.5)) if f > 0 else None, "frac_defined": float(f)}
    res["lines_ci"] = ci
    # grid brackets for the lines
    br = {}
    for k, v in res["lines"].items():
        if isinstance(v, float) and k not in ("allowed_runs",):
            j = np.searchsorted(g, v)
            br[k] = [float(g[max(j - 1, 0)]), float(g[min(j, len(g) - 1)])]
    res["lines_bracket"] = br
    res["peaks"] = {}
    for k in peaks_b:
        gp, kk, inter, beats = peak_info(g, st[k])
        bb = peaks_b[k]
        frac = float(np.mean([b[2] and b[3] for b in bb if b[0] is not None])) if bb else 0.0
        locs = np.array([b[0] for b in bb if b[0] is not None and b[2]])
        res["peaks"][k] = {"g": gp, "k": kk, "interior": inter, "beats_ends": beats, "value": float(st[k][kk]) if kk is not None else None,
                           "frac_resolved": frac, "resolved": bool(frac >= 0.95),
                           "ci": [float(np.percentile(locs, 2.5)), float(np.percentile(locs, 97.5))] if len(locs) > 20 else None,
                           "phase_at_peak": pts[kk]["phase"] if kk is not None else None}
    # descriptive only (NOT pre-registered): local maxima of each peak observable on the point-estimate curve,
    # prominence >= 10 % of the value against the lower of the two neighbouring minima
    res["local_maxima"] = {}
    for k in peaks_b:
        y = np.asarray(st[k], float); lm = []
        for i in range(1, len(y) - 1):
            if np.isfinite(y[i - 1:i + 2]).all() and y[i] > y[i - 1] and y[i] >= y[i + 1]:
                # topographic prominence: base on each side = minimum between the peak and the nearest higher point
                hl = np.where(y[:i] > y[i])[0]; hr = np.where(y[i + 1:] > y[i])[0]
                lo_l = np.nanmin(y[(hl[-1] if len(hl) else 0):i])
                lo_r = np.nanmin(y[i + 1:(i + 1 + hr[0] + 1) if len(hr) else len(y)])
                prom = y[i] - max(lo_l, lo_r)
                if prom >= 0.10 * y[i]:
                    lm.append({"g": float(g[i]), "value": float(y[i]), "prominence": float(prom), "phase": pts[i]["phase"]})
        res["local_maxima"][k] = lm
    # point table extract (for figures)
    res["curve"] = {f: [p.get(f) for p in pts] for f in (
        "phase", "phase_base", "allowed", "allowed_broad", "G", "rho_ORN", "rho_nonORN", "rho_AL", "n_active_nonORN",
        "chi_non", "F_non", "chi_AL", "F_AL", "tau_int", "tau_int_rep", "tau_e", "osc", "osc_strong", "cv_t", "period",
        "NS", "NS_exempt", "PR_non_n", "mod_share", "S1_resid", "S1", "S1_1pct", "S2_alive", "S2", "tau_post",
        "deaths_post", "removal_class", "palp_ok", "S1p_resid", "S2p_resid", "osc_AL", "x_ORN", "x_nonORN",
        "x_AL_module", "xact_AL_module", "ORN_after_removal_max")}
    return res


# ------------------------------------------------------------------------------------------------ cross-curve
def field_scaling(curves, rng):
    """M2 real: static susceptibility chi_h (levels 0.02 / 0.07) and field slope s(g) over the 5 levels, per r."""
    out = {}
    lev_sets = {0.04: "A1", 0.01: "B1", 0.02: "B1", 0.07: "B1", 0.10: "B1"}
    for r in M.R_CORE:
        cs = {o: curves.get((s, "M2", "real", o, r, False)) for o, s in lev_sets.items()}
        if any(v is None for v in cs.values()):
            continue
        gs = sorted(set.intersection(*[set(np.round(c["g"], 5)) for c in cs.values()]))
        if len(gs) < 5:
            continue
        gs = np.array(gs)
        sel = {o: np.array([int(np.argmin(np.abs(c["g"] - x))) for x in gs]) for o, c in cs.items()}
        levs = sorted(cs)
        def curves_of(idx_fn):
            rn = np.array([[idx_fn(o, cs[o]["reps"]["rho_non"][sel[o]]) for o in levs]])[0]   # [lev, g]
            ro = np.array([[idx_fn(o, cs[o]["reps"]["rho_orn"][sel[o]]) for o in levs]])[0]
            return rn, ro
        def stat(rn, ro):
            lx = np.log(ro); ly = np.log(np.maximum(rn, 1e-12))
            xm = lx.mean(0); ym = ly.mean(0)
            s = ((lx - xm) * (ly - ym)).sum(0) / ((lx - xm) ** 2).sum(0)
            h = {o: M.h_of_rho(o) for o in levs}
            i2, i7 = levs.index(0.02), levs.index(0.07)
            chi_h = (rn[i7] - rn[i2]) / (h[0.07] - h[0.02])
            return s, chi_h
        rn, ro = curves_of(lambda o, a: a.mean(1))
        s, chi_h = stat(rn, ro)
        # amendment 4: slope against the TRANSMITTED ORN input rho_ORN * e_ORN (e = measured mean efficacy of active
        # ORN epochs, replica mean, treated as fixed in the bootstrap); ORN-synapse depression alone makes s ~ 0.4
        e = np.array([[cs[o]["pts"][i].get("xact_ORN") or np.nan for i in sel[o]] for o in levs])
        s_eff, _ = stat(rn, ro * e)
        sb, cb, seb = [], [], []
        for _ in range(NBOOT):
            def rs(o, a):
                idx = rng.integers(0, a.shape[1], size=a.shape)
                return np.take_along_axis(a, idx, 1).mean(1)
            # same resample for rho_non and rho_orn of one level (paired replicas)
            rnb, rob = [], []
            for o in levs:
                a1 = cs[o]["reps"]["rho_non"][sel[o]]; a2 = cs[o]["reps"]["rho_orn"][sel[o]]
                idx = rng.integers(0, a1.shape[1], size=a1.shape)
                rnb.append(np.take_along_axis(a1, idx, 1).mean(1)); rob.append(np.take_along_axis(a2, idx, 1).mean(1))
            s_, c_ = stat(np.array(rnb), np.array(rob))
            se_, _ = stat(np.array(rnb), np.array(rob) * e)
            sb.append(s_); cb.append(c_); seb.append(se_)
        sb, cb, seb = np.array(sb), np.array(cb), np.array(seb)
        # s = 0.5 line (isotonic decreasing; first crossing from above)
        def s_cross(sv):
            return iso_cross(gs, -sv, np.ones_like(sv), -0.5)[0]
        sc = s_cross(s); scb = np.array([np.nan if (x := s_cross(v)) is None else x for v in sb])
        sec = s_cross(s_eff); secb = np.array([np.nan if (x := s_cross(v)) is None else x for v in seb])
        pk = peak_info(gs, chi_h)
        pkb = [peak_info(gs, v) for v in cb]
        frac = float(np.mean([b[2] and b[3] for b in pkb if b[0] is not None]))
        locs = np.array([b[0] for b in pkb if b[0] is not None and b[2]])
        out[str(r)] = {"g": gs.tolist(), "s": s.tolist(), "s_lo": np.nanpercentile(sb, 2.5, 0).tolist(),
                       "s_hi": np.nanpercentile(sb, 97.5, 0).tolist(), "chi_h": chi_h.tolist(),
                       "chi_h_lo": np.percentile(cb, 2.5, 0).tolist(), "chi_h_hi": np.percentile(cb, 97.5, 0).tolist(),
                       "rho_non_by_level": {str(o): rn[i].tolist() for i, o in enumerate(levs)},
                       "rho_orn_by_level": {str(o): ro[i].tolist() for i, o in enumerate(levs)},
                       "s_half": sc, "s_half_ci": [float(np.nanpercentile(scb, 2.5)), float(np.nanpercentile(scb, 97.5))]
                       if np.isfinite(scb).any() else None,
                       "s_half_note": "s < 0.5 already at the lowest g (ORN-synapse depression compresses the input), "
                                      "so this crossing is at/below the grid start and not interpretable" if (s[0] < 0.5) else "",
                       "e_ORN_by_level": {str(o): e[i].tolist() for i, o in enumerate(levs)},
                       "s_eff": s_eff.tolist(), "s_eff_lo": np.nanpercentile(seb, 2.5, 0).tolist(),
                       "s_eff_hi": np.nanpercentile(seb, 97.5, 0).tolist(), "s_eff_half": sec,
                       "s_eff_half_ci": [float(np.nanpercentile(secb, 2.5)), float(np.nanpercentile(secb, 97.5))]
                       if np.isfinite(secb).any() else None,
                       "chi_h_peak": {"g": pk[0], "interior": pk[2], "beats_ends": pk[3], "frac_resolved": frac,
                                      "resolved": frac >= 0.95,
                                      "ci": [float(np.percentile(locs, 2.5)), float(np.percentile(locs, 97.5))] if len(locs) > 20 else None}}
    return out


def decay_lines():
    """Setting-(a) lines from lane m2/decay (results/m2/decay/boundaries.json 'rows', keys '<model>_<graph>_r<r>' with
    S / on / D dicts holding 'g'). Read as they are at analysis time (that lane may still be refining); md5 recorded."""
    f = f"{ROOT}/results/m2/decay/boundaries.json"
    if not os.path.exists(f):
        return {"file": None, "rows": {}}
    import hashlib
    raw = open(f, "rb").read()
    js = json.loads(raw)
    rows = dict(js.get("rows", {}))
    for k, v in (js.get("m1_reference") or {}).items():          # M1 reference rows: 'M1ref_<graph>_r<r>'
        if k.startswith("M1ref_"):
            gr = k.split("_")[1]
            rows.setdefault("M1_" + k[len("M1ref_"):], {"model": "M1", "graph": gr, "r": float(k.split("_r")[-1]), **v})
    for k, v in (js.get("variants") or {}).items():
        if isinstance(v, dict) and "on" in v:
            rows.setdefault(k, v)
    out = {}
    for k, v in rows.items():
        out[k] = {"model": v.get("model"), "graph": v.get("graph"), "r": v.get("r"),
                  **{b: (v.get(b) or {}).get("g") for b in ("S", "on", "D")},
                  **{b + "_ci": (v.get(b) or {}).get("boot_ci") for b in ("S", "on")},
                  "n_s2_points": v.get("n_s2_points")}
    return {"file": os.path.relpath(f, ROOT), "md5": hashlib.md5(raw).hexdigest()[:8],
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f))), "rows": out}


def track_test(peak_line, inst_line):
    """plan.md section 4: dict r -> g (peaks resolved only) vs dict r -> g."""
    rs = [r for r in peak_line if peak_line[r] is not None and inst_line.get(r) is not None]
    if len(rs) < 4:
        return {"verdict": "inconclusive", "n": len(rs), "reason": "fewer than 4 r with a resolved peak and a line"}
    gp = np.array([peak_line[r] for r in rs]); gl = np.array([inst_line[r] for r in rs])
    dist = np.abs(np.log(gp / gl))
    frac = float(np.mean(dist <= np.log(STEP) + 1e-9))
    varies = bool(np.ptp(np.log(gl)) > np.log(STEP))
    rho = float(spearmanr(gp, gl)[0]) if varies else None
    if frac >= 0.8 and (not varies or (rho is not None and rho >= 0.8)):
        v = "tracks"
    elif frac < 0.5:
        v = "does not track"
    else:
        v = "partial"
    return {"verdict": v, "n": len(rs), "frac_within_one_step": frac, "line_varies": varies, "spearman": rho,
            "r": rs, "g_peak": gp.tolist(), "g_line": gl.tolist(), "ratio": (gp / gl).tolist()}


def _curve_job(a):
    k, c = a
    seed = [20261109] + [int(x) for x in np.frombuffer("|".join(map(str, k)).encode(), np.uint8)]
    return "|".join(map(str, k)), analyse_curve(k, c, np.random.default_rng(seed))


def analyse(jobs=8):
    rng = np.random.default_rng(20261109)
    curves = load_curves()
    print(f"{len(curves)} curves")
    keys = sorted(curves, key=lambda k: (k[0], k[1], k[2], k[3], k[4], k[5]))
    t0 = time.time()
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            out = pool.map(_curve_job, [(k, curves[k]) for k in keys], chunksize=1)
    else:
        out = [_curve_job((k, curves[k])) for k in keys]
    res = dict(out)
    print(f"curves analysed in {time.time() - t0:.0f} s")
    fs = field_scaling(curves, rng)
    # tracking tests per set (primary: F_AL; also F_non, chi_non, tau_int; and the field slope line)
    tracks = {}
    groups = {}
    dec = decay_lines()
    for name, c in res.items():
        k = c["key"]
        if k["palp"]:
            continue
        groups.setdefault((k["set"], k["model"], k["graph"], k["orn"]), []).append(c)
    for gk, cs in groups.items():
        tk = "|".join(map(str, gk)); tracks[tk] = {}
        lines = {"g_lin": {c["key"]["r"]: M.G_LIN for c in cs},
                 "DT_SS": {c["key"]["r"]: c["lines"].get("DT_SS") for c in cs}}
        def dk(c, b):
            v = dec["rows"].get(f"{gk[1]}_{gk[2]}_r{c['key']['r']:g}")
            return v.get(b) if v else None
        if any(dk(c, "on") is not None for c in cs):
            lines["decay_on"] = {c["key"]["r"]: dk(c, "on") for c in cs}
            lines["decay_S"] = {c["key"]["r"]: dk(c, "S") for c in cs}
        for pk in ("F_AL", "F_non", "chi_non", "tau_int"):
            pl = {c["key"]["r"]: (c["peaks"][pk]["g"] if c["peaks"][pk]["resolved"] else None) for c in cs}
            tracks[tk][pk] = {"n_resolved": sum(v is not None for v in pl.values()), "n_r": len(cs),
                              **{f"vs_{ln}": track_test(pl, lv) for ln, lv in lines.items()}}
        if gk == ("A1", "M2", "real", 0.04):
            sl = {float(r): v["s_half"] for r, v in fs.items()}
            ch = {float(r): (v["chi_h_peak"]["g"] if v["chi_h_peak"]["resolved"] else None) for r, v in fs.items()}
            tracks[tk]["s_half"] = {f"vs_{ln}": track_test(sl, lv) for ln, lv in lines.items()}
            se = {float(r): v["s_eff_half"] for r, v in fs.items()}
            tracks[tk]["s_eff_half (amendment 4)"] = {f"vs_{ln}": track_test(se, lv) for ln, lv in lines.items()}
            tracks[tk]["chi_h"] = {f"vs_{ln}": track_test(ch, lv) for ln, lv in lines.items()}
    summ = {"n_curves": len(res), "field_scaling": fs, "tracking": tracks, "decay_lane": dec,
            "g_lin": M.G_LIN, "grid_step": STEP}
    json.dump(res, open(f"{OUT}/curves.json", "w"), default=_clean)
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1, default=_clean)
    # flat point table
    rows = []
    for name, c in res.items():
        for i, g in enumerate(c["g"]):
            row = {**c["key"], "g": g}
            row.update({f: c["curve"][f][i] for f in c["curve"]})
            rows.append(row)
    with open(f"{OUT}/points.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        for r_ in rows:
            wr.writerow({k: _clean(v) for k, v in r_.items()})
    print(f"wrote curves.json ({len(res)} curves), summary.json, points.csv ({len(rows)} rows)")
    return res, summ


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
    if what in ("extract", "all"):
        extract_all(jobs)
    if what in ("analyse", "all"):
        analyse(jobs)
