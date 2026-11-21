"""m2_decay_report.py -- lane m2/decay: boundaries, M1 comparison, oscillation, where, lifetime shape, variants.

Called by `python m2_decay_analyze.py analyze`. Reads the per-file summaries (sim/sum/), the indices and the M1
reference results/analysis2/phase_complete.json. Writes results/m2/decay/{boundaries.json, points.json, osc.json,
where.json, lifetimes.json, tables.md}. Definitions: m2_decay_analyze.py docstring and plan.md.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

import m2_decay_analyze as A
import m2_common as M

OUT = A.OUT
N = A.N
T_REF = A.T_REF
rng = np.random.default_rng(7)
G_LIN = {"real": 0.1632, "cnull": 0.8287, "wnull": 0.663, "cnull1": None, "wnull1": None}
M1_GRAPH = {"real": "real", "cnull": "null", "wnull": "wnull"}


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else ("inf" if x > 0 else ("-inf" if x < 0 else None))
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def phase_of(p):
    tau = p["tau_reg"]
    if tau < 200:
        return "Q"
    if tau < 3000:
        return "S"
    return "D" if (p["pr"] is not None and p["pr"] >= 0.25) else "L"


# ------------------------------------------------------------------------------------------------ lifetime tests
def ks_shift_exp(times, died, T, nboot=500, shift=True):
    """KS distance between the empirical extinction CDF (exact times, censored at T) and a fitted (shifted)
    exponential; parametric-bootstrap p reproducing censoring and the re-fit. CV range under H0 if uncensored."""
    times = np.asarray(times, float); died = np.asarray(died, bool); n = len(times); d = int(died.sum())
    if d < 3:
        return None

    def fit(tu, ev):
        t0 = np.where(ev, tu, np.inf).min(-1) if shift else np.zeros(tu.shape[:-1])
        t0 = np.where(np.isfinite(t0), t0, 0.0)
        tau = np.clip(np.minimum(tu, T) - t0[..., None], 0, None).sum(-1) / np.maximum(ev.sum(-1), 1)
        return t0, tau

    def ks(tu, ev, t0, tau):
        # evaluate at every death time (both sides of the step)
        tt = np.sort(np.where(ev, tu, np.inf), -1)
        k = np.arange(1, n + 1)
        F = lambda x: np.where(x > t0[..., None], 1 - np.exp(-(x - t0[..., None]) / tau[..., None]), 0.0)
        Ft = F(np.where(np.isfinite(tt), tt, T))
        fin = np.isfinite(tt)
        D1 = np.where(fin, np.abs(k / n - Ft), 0).max(-1)
        D2 = np.where(fin, np.abs((k - 1) / n - Ft), 0).max(-1)
        return np.maximum(D1, D2)

    t0, tau = fit(times[None], died[None])
    D = float(ks(times[None], died[None], t0, tau)[0])
    X = t0[0] + rng.exponential(tau[0], (nboot, n))
    ev = X < T
    tu = np.where(ev, X, T)
    ok = ev.sum(1) >= 2
    t0b, taub = fit(tu, ev)
    Db = ks(tu, ev, t0b, taub)[ok]
    res = {"t0": float(t0[0]), "tau": float(tau[0]), "KS_D": D, "KS_p": float((Db >= D).mean()), "n": n, "deaths": d}
    if d == n:
        x = times - (t0[0] if shift else 0)
        res["CV"] = float(x.std(ddof=1) / x.mean())
        full = ev.all(1)
        if full.sum() > 100:
            xb = tu[full] - (t0b[full][:, None] if shift else 0)
            cvb = xb.std(1, ddof=1) / xb.mean(1)
            res["CV_null_95"] = [float(np.percentile(cvb, 2.5)), float(np.percentile(cvb, 97.5))]
    return res


def late_conditional(ext_exact, T, tc, nboot=500):
    """POST HOC (plan section 8): residual lifetimes of replicas alive at tc, exponential through the origin."""
    e = np.asarray(ext_exact, float)
    alive = e > tc
    n0 = len(e); n = int(alive.sum())
    if n < 8:
        return {"tc": tc, "n_alive_tc": n, "early_frac": float(1 - n / n0)}
    res_t = np.minimum(e[alive], T) - tc
    died = np.isfinite(e[alive])
    r = ks_shift_exp(res_t, died, T - tc, nboot=nboot, shift=False)
    out = {"tc": tc, "n_alive_tc": n, "early_frac": float(1 - n / n0), "deaths_after": int(died.sum()),
           "tau_late": float(res_t.sum() / died.sum()) if died.sum() else float("inf")}
    if r:
        out.update(KS_p=r["KS_p"], KS_D=r["KS_D"], CV=r.get("CV"), CV_null_95=r.get("CV_null_95"))
    return out


def hazard(ext_exact, T, nbins=14, tmin=10.0):
    """POST HOC: empirical hazard (deaths / exposure) in log-spaced time bins, with Garwood 95% intervals."""
    e = np.asarray(ext_exact, float)
    edges = np.geomspace(tmin, T, nbins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        expo = np.clip(np.minimum(e, b) - a, 0, None).sum()
        d = int(((e > a) & (e <= b)).sum())
        if expo <= 0:
            continue
        lo = stats.chi2.ppf(0.025, 2 * d) / 2 / expo if d > 0 else 0.0
        hi = stats.chi2.ppf(0.975, 2 * d + 2) / 2 / expo
        out.append([float(np.sqrt(a * b)), d / expo, lo, hi, d, float(expo)])
    return out


def late_hazard(ext_exact, T, t_from=100.0):
    e = np.asarray(ext_exact, float)
    alive_at = e > t_from
    if alive_at.sum() == 0:
        return None
    x = np.minimum(e[alive_at], T) - t_from
    d = int(np.isfinite(e[alive_at]).sum())
    return {"tau_late": float(x.sum() / d) if d else float("inf"), "n_at_100": int(alive_at.sum()), "deaths": d}


def kaplan_meier(times, died):
    o = np.argsort(times); t = times[o]; ev = died[o]
    n = len(t); S = 1.0; out = [(0.0, 1.0)]
    for i in range(n):
        if ev[i]:
            S *= 1 - 1 / (n - i)
            out.append((t[i], S))
    return np.array(out)


def rho_fit(t, m, nal, t0, nmin=5):
    w = (t >= t0) & (nal >= nmin) & (m > 0)
    if w.sum() < 6:
        return None
    tt, lr = t[w], np.log(m[w]); lt = np.log(tt)
    sst = ((lr - lr.mean()) ** 2).sum()
    b, a = np.polyfit(lt, lr, 1); rp = ((lr - a - b * lt) ** 2).sum()
    k, c = np.polyfit(tt, lr, 1); re_ = ((lr - c - k * tt) ** 2).sum()
    mid = np.sqrt(tt[0] * tt[-1])
    h1, h2 = tt <= mid, tt > mid
    b1 = np.polyfit(lt[h1], lr[h1], 1)[0] if h1.sum() >= 3 else np.nan
    b2 = np.polyfit(lt[h2], lr[h2], 1)[0] if h2.sum() >= 3 else np.nan
    span = float(tt[-1] / tt[0])
    stable = bool(np.isfinite(b1) and np.isfinite(b2) and abs(b2 - b1) < 0.3 * abs(b))
    return {"t_range": [float(tt[0]), float(tt[-1])], "decades": float(np.log10(span)), "n_pts": int(w.sum()),
            "power_b": float(b), "R2_power": float(1 - rp / sst), "exp_tau": float(-1 / k) if k < 0 else None,
            "R2_exp": float(1 - re_ / sst), "b_half1": float(b1), "b_half2": float(b2), "half_stable": stable,
            "power_ok": bool(span >= 10 and (1 - rp / sst) >= (1 - re_ / sst) and stable)}


# ------------------------------------------------------------------------------------------------ M1 reference
def m1_reference():
    pc = json.load(open(f"{M.ROOT}/results/analysis2/phase_complete.json"))
    ref = {}
    for gk, gm in (("real", "real"), ("cnull", "null"), ("wnull", "wnull")):
        for rk, e in pc["sim"][gm].items():
            pts = [{"g": p["g"], "tau": (float("inf") if p["tau_sim"] in ("inf", None) else p["tau_sim"]),
                    "deaths": p["deaths"], "B": p["B"], "PR": p["PR"], "alive": p["alive_T"]} for p in e["points"]]
            for p in pts:
                E = p["tau"] * p["deaths"] if p["deaths"] > 0 else None
                p["tau_reg"] = p["tau"] if p["deaths"] > 0 else None
            ref[f"M1ref_{gk}_r{float(rk):g}"] = {"model": "M1ref", "graph": gk, "r": float(rk),
                                                 "S": e["S"], "on": e["on"], "D": e["D"], "half": e["half"],
                                                 "points": pts}
    return ref


def m1_point_phase(p):
    """tile phase of an analysis2 point (tau regularised as E/0.5 when there are no deaths, as for M2)."""
    if p["deaths"] > 0:
        tau = p["tau"]
    else:
        tau = float("inf")
    if tau < 200:
        return "Q"
    if tau < 3000:
        return "S"
    pr = p["PR"]
    return "D" if (pr is not None and pr >= 0.25) else "L"


# ------------------------------------------------------------------------------------------------ main
def main():
    rows = A.load_rows(("s1", "s2"))
    m1 = m1_reference()
    bnd, pts_out, osc_out, where, screen = {}, {}, [], {}, {}
    for key, R in sorted(rows.items(), key=lambda kv: (kv[1]["model"], kv[1]["graph"], kv[1]["r"])):
        P = R["points"]
        est = A.sim_boundaries(P)
        boot = A.bootstrap(P)
        e = {"model": R["model"], "graph": R["graph"], "r": R["r"], "n_points": len(P),
             "n_s2_points": sum(1 for p in P if "s2_" in p["src"]), "g_range": [P[0]["g"], P[-1]["g"]],
             "g_lin": G_LIN.get(R["graph"])}
        for b in ("S", "on", "half", "D"):
            e[b] = A.summarize_b(est[b], boot[b])
        if e["D"]["status"] == "at_onset":
            e["D"]["boot_ci"] = e["on"]["boot_ci"]; e["D"]["total"] = e["on"]["total"]
        if e["S"]["g"] and e["on"]["g"]:
            e["window_on_over_S"] = e["on"]["g"] / e["S"]["g"]
        if e["on"]["g"] and e["D"]["g"]:
            e["D_over_on"] = e["D"]["g"] / e["on"]["g"]
        # per-point table
        pl = []
        for p in P:
            ph = phase_of(p)
            lh = A_late = late_hazard(p["ext_exact"], T_REF)
            oc = [o for o in p["osc_list"] if o]
            osc_any = {k: any((o.get(k) or {}).get("osc", False) for o in oc) for k in ("total", "AL_module", "nonORN")}
            osc_strong = {k: any((o.get(k) or {}).get("osc_strong", False) for o in oc) for k in ("total", "AL_module", "nonORN")}
            pl.append({"g": p["g"], "B": p["B"], "deaths": p["deaths"], "tau": p["tau"], "tau_ci": p["tau_ci"],
                       "alive_T": p["alive"], "PRN": p["pr"], "phase": ph, "qs_tot": p["qs_tot"],
                       "AL_share": float(p["reg_share"][R["reg_names"].index("AL_module")]),
                       "late": lh, "osc": osc_any, "osc_strong": osc_strong, "src": p["src"]})
            for o in oc:
                for sig, v in o.items():
                    if v is None:
                        continue
                    osc_out.append({"row": key, "g": p["g"], "signal": sig, "n_rep": v["n_rep"], "osc": v["osc"],
                                    "osc_strong": v["osc_strong"], "c_min": v["c_min"], "lag_min": v["lag_min"],
                                    "c_peak": v["c_peak"], "period": v["period"], "coh_ratio": v["coh_ratio"],
                                    "cv_t": v["cv_t"], "tau_int": v["tau_int"], "f_peak": v["f_peak"],
                                    "peak_ratio": v["peak_ratio"], "phase": ph})
        pts_out[key] = pl
        # where activity lives: first sustained point and first D point
        wr = {}
        sus = [p for p in P if p["tau_reg"] >= T_REF]
        for lab, cand in (("on", sus), ("D", [p for p in sus if p["pr"] >= 0.25])):
            if not cand:
                continue
            p = cand[0]
            ss = p["sup_share"]; sz = R["sup_size"] / N
            top = np.argsort(-np.nan_to_num(ss))[:5]
            wr[lab] = {"g": p["g"], "tau": p["tau"], "alive_T": p["alive"], "PRN": p["pr"], "PR": p["pr"] * N,
                       "region_share": dict(zip(R["reg_names"], p["reg_share"])),
                       "qs_region": dict(zip(R["reg_names"], p["qs_reg"])), "qs_tot": p["qs_tot"],
                       "top_superclasses": [{"superclass": R["sup_names"][i], "share": float(ss[i]),
                                             "enrichment": float(ss[i] / sz[i])} for i in top]}
        # S-region cross-check from the record-time region shares (points with 150 <= tau <= 260)
        nearS = [p for p in P if 150 <= p["tau_reg"] <= 260]
        if nearS:
            p = min(nearS, key=lambda p: abs(np.log(p["tau_reg"] / 200)))
            t = p["t"]; w = (t >= 100) & (t <= 300) & (p["n_act_t"] > 0)
            if w.any():
                sh = np.nansum(p["reg_share_t"][w] * p["n_act_t"][w, None], 0) / p["n_act_t"][w].sum()
                wr["S_records"] = {"g": p["g"], "tau": p["tau"], "region_share_t100_300": dict(zip(R["reg_names"], sh))}
        where[key] = wr
        # lifetime screen (exact times; tau >= 200 and >= 10 deaths)
        sc = []
        for p in P:
            if p["tau_reg"] >= 200 and p["deaths"] >= 10:
                died = np.isfinite(p["ext_exact"])
                r_ = ks_shift_exp(np.where(died, p["ext_exact"], T_REF), died, T_REF, nboot=500)
                r0 = ks_shift_exp(np.where(died, p["ext_exact"], T_REF), died, T_REF, nboot=500, shift=False)
                if r_:
                    lc = late_conditional(p["ext_exact"], T_REF, 200.0)
                    sc.append({"g": p["g"], "tau": p["tau"], **{k: r_[k] for k in ("t0", "KS_D", "KS_p", "n", "deaths")},
                               "KS_p0": r0["KS_p"] if r0 else None, "CV": r_.get("CV"), "CV_null_95": r_.get("CV_null_95"),
                               "late200": lc})
        if sc:
            ps = np.clip([s["KS_p"] for s in sc], 1e-4, 1)
            fis = stats.combine_pvalues(ps, method="fisher")
            lp = np.clip([x["late200"]["KS_p"] for x in sc if x["late200"].get("KS_p") is not None], 1e-4, 1)
            screen[key] = {"points": sc, "n": len(sc), "n_reject_05": int((ps < 0.05).sum()),
                           "fisher_p": float(fis.pvalue),
                           "late200_n": int(len(lp)), "late200_n_reject_05": int((lp < 0.05).sum()),
                           "late200_fisher_p": float(stats.combine_pvalues(lp, method="fisher").pvalue) if len(lp) else None,
                           "n_cv_above": sum(1 for s in sc if s["CV"] is not None and s["CV_null_95"] and s["CV"] > s["CV_null_95"][1]),
                           "n_cv_below": sum(1 for s in sc if s["CV"] is not None and s["CV_null_95"] and s["CV"] < s["CV_null_95"][0]),
                           "n_cv": sum(1 for s in sc if s["CV"] is not None and s["CV_null_95"])}
        bnd[key] = e
        print(key, {b: (round(e[b]["g"], 4) if e[b]["g"] else e[b]["status"]) for b in ("S", "on", "D")}, flush=True)

    # M1 comparison (real 23 r; nulls where M1 has the r)
    comp = {}
    for key, e in bnd.items():
        if e["model"] not in ("M2", "M2s") or e["graph"] not in ("real", "cnull", "wnull"):
            continue
        mk = f"M1ref_{e['graph']}_r{e['r']:g}"
        if mk not in m1:
            continue
        r1 = m1[mk]
        c = {}
        for b in ("S", "on", "D"):
            a, bb = e[b]["g"], r1[b]["g"]
            c[b] = {"M2": a, "M1": bb, "ratio": (a / bb) if (a and bb) else None,
                    "M2_total": e[b]["total"], "M1_total": r1[b]["total"], "M1_status": r1[b]["status"],
                    "M2_status": e[b]["status"]}
        comp[key] = c
    # M1 pipeline check (my M1 reruns vs analysis2): pre-registered on the stage-1 (coarse) points only, and also on
    # the pooled stage 1 + 2 points (refined)
    pipe = {}
    rows1 = A.load_rows(("s1",))
    b1 = {}
    for r in (0, 1, 4):
        k = f"M1_real_r{r:g}"
        if k in rows1:
            P1 = rows1[k]["points"]; est1 = A.sim_boundaries(P1); bo1 = A.bootstrap(P1)
            b1[k + "_coarse"] = {b: A.summarize_b(est1[b], bo1[b]) for b in ("S", "on", "D")}
    for k0, src in [(k, b1[k]) for k in b1] + [(f"M1_real_r{r:g}_pooled", bnd[f"M1_real_r{r:g}"]) for r in (0, 1, 4)
                                               if f"M1_real_r{r:g}" in bnd]:
        k = k0
        if True:
            r1 = m1["M1ref_real_r" + k.split("_real_r")[1].split("_")[0]]
            pipe[k] = {}
            for b in ("S", "on", "D"):
                mine, a2 = src[b]["g"], r1[b]["g"]
                tot = src[b]["total"]
                if mine is None and a2 is None:
                    ok = "both not reached in the simulated range"
                else:
                    ok = bool(tot and a2 and tot[0] <= a2 <= tot[1])
                pipe[k][b] = {"mine": mine, "mine_total": tot, "a2": a2, "a2_total": r1[b]["total"], "a2_inside_mine": ok,
                              "intervals_overlap": bool(tot and r1[b]["total"] and tot[0] <= r1[b]["total"][1]
                                                        and r1[b]["total"][0] <= tot[1])}
    # variants relative to M2
    var = {}
    for key, e in bnd.items():
        if e["model"].startswith("M2_") or e["model"] in ("M1s", "M1", "M2s"):
            base = bnd.get(f"M2_real_r{e['r']:g}")
            if base is None:
                continue
            var[key] = {b: {"g": e[b]["g"], "total": e[b]["total"], "status": e[b]["status"],
                            "ratio_to_M2": (e[b]["g"] / base[b]["g"]) if (e[b]["g"] and base[b]["g"]) else None}
                        for b in ("S", "on", "D")}
            w = where.get(key, {}).get("on")
            var[key]["AL_share_first_sustained"] = w["region_share"]["AL_module"] if w else None
            var[key]["PRN_first_sustained"] = w["PRN"] if w else None
            var[key]["window_on_over_S"] = e.get("window_on_over_S")
    # stage 3b where-at-S
    whereS = where_at_S()
    for k, v in whereS.items():
        where.setdefault(k, {})["S"] = v
    # long runs
    lt = long_runs()

    os.makedirs(OUT, exist_ok=True)
    json.dump(clean({"rows": bnd, "m1_reference": {k: {b: v[b] for b in ("S", "on", "D", "half")} | {"graph": v["graph"], "r": v["r"]}
                                                   for k, v in m1.items()},
                     "m1_comparison": comp, "m1_pipeline_check": pipe, "variants": var,
                     "definitions": A.__doc__}), open(f"{OUT}/boundaries.json", "w"), indent=1)
    json.dump(clean(pts_out), open(f"{OUT}/points.json", "w"), indent=0)
    json.dump(clean({"rows": osc_out}), open(f"{OUT}/osc.json", "w"), indent=0)
    json.dump(clean(where), open(f"{OUT}/where.json", "w"), indent=1)
    json.dump(clean({"screen": screen, "long": lt}), open(f"{OUT}/lifetimes.json", "w"), indent=1)
    tables(bnd, comp, pipe, var, where, osc_out, screen, lt, pts_out)


def where_at_S():
    idx = A.load_index()
    out = {}
    for f in sorted(glob.glob(f"{A.SUMD}/s3b_*.npz")):
        e = idx.get(os.path.basename(f))
        if e is None:
            continue
        s = np.load(f, allow_pickle=True)
        names = [str(x) for x in s["reg_names"]]; ss = s["sup_share"][0]; sz = s["sup_size"] / N
        top = np.argsort(-np.nan_to_num(ss))[:5]
        key = A.row_key(e["model"], e["graph"], e["r"])
        died = np.isfinite(s["ext_grid"][0])
        out[key] = {"g": float(s["g"][0]), "occ_win": "50:400", "PRN": float(s["prn"][0]), "PR": float(s["pr"][0]),
                    "tau_this_run": float(np.where(died, s["ext_grid"][0], T_REF).sum() / max(died.sum(), 0.5)),
                    "region_share": dict(zip(names, s["reg_share"][0])),
                    "top_superclasses": [{"superclass": str(s["sup_names"][i]), "share": float(ss[i]),
                                          "enrichment": float(ss[i] / sz[i])} for i in top]}
    return out


def long_runs():
    idx = A.load_index()
    groups = {}
    for f in sorted(glob.glob(f"{A.SUMD}/s3a_*.npz")):
        e = idx.get(os.path.basename(f))
        if e is None:
            continue
        s = np.load(f, allow_pickle=True)
        key = (A.row_key(e["model"], e["graph"], e["r"]), round(float(s["g"][0]), 5))
        G = groups.setdefault(key, {"ext": [], "rho": [], "surv": [], "t": s["t"], "T": float(s["T"]), "B": []})
        G["ext"].append(s["ext_exact"][0]); G["rho"].append(s["rho_mean"][0]); G["surv"].append(s["surv_mean"][0])
        G["B"].append(len(s["ext_exact"][0]))
    res = {}
    for (rk, g), G in sorted(groups.items()):
        ext = np.concatenate(G["ext"]); died = np.isfinite(ext); T = G["T"]
        Bs = np.array(G["B"], float)
        m = np.average(np.array(G["rho"]), axis=0, weights=Bs)
        nal = (np.array(G["surv"]) * Bs[:, None]).sum(0)
        times = np.where(died, ext, T)
        ksr = ks_shift_exp(times, died, T, nboot=2000)
        ks0 = ks_shift_exp(times, died, T, nboot=2000, shift=False)
        d = int(died.sum()); tau = float(times.sum() / d) if d else float("inf")
        f50 = rho_fit(G["t"], m, nal, 50.0); f200 = rho_fit(G["t"], m, nal, 200.0)
        broad = bool(ksr and ksr["KS_p"] < 0.01 and ksr.get("CV") is not None and ksr.get("CV_null_95")
                     and ksr["CV"] > ksr["CV_null_95"][1])
        expo = bool(ksr and ksr["KS_p"] >= 0.05 and (ksr.get("CV") is None or not ksr.get("CV_null_95")
                                                       or ksr["CV_null_95"][0] <= ksr["CV"] <= ksr["CV_null_95"][1]))
        sig = bool(broad and f50 and f200 and f50["power_ok"] and f200["power_ok"])
        km = kaplan_meier(times, died)
        post = {"late200": late_conditional(ext, T, 200.0, nboot=2000), "late500": late_conditional(ext, T, 500.0, nboot=2000),
                "hazard": hazard(ext, T)}
        res.setdefault(rk, []).append({"post_hoc": post, "g": g, "n": int(len(ext)), "deaths": d, "tau": tau, "ks_shift": ksr, "ks_origin": ks0,
                                       "rho_fit_t50": f50, "rho_fit_t200": f200, "broad": broad, "exp_ok": expo,
                                       "signature": sig, "km": km[:: max(1, len(km) // 300)].tolist(),
                                       "t": G["t"].tolist(), "rho_mean": m.tolist(), "n_alive_t": nal.tolist()})
    verdict = {}
    for rk, L in res.items():
        L.sort(key=lambda x: x["g"])
        sig = [x["signature"] for x in L]
        run = best = 0
        for s_ in sig:
            run = run + 1 if s_ else 0; best = max(best, run)
        slow = [x for x in L if x["tau"] >= 200]
        n_exp = sum(1 for x in slow if x["exp_ok"])
        grif = False
        if best >= 3:
            # exponents of the longest run must change monotonically by >= 20 %
            for i in range(len(L) - 2):
                seg = L[i:i + 3]
                if all(x["signature"] for x in seg):
                    b = [x["rho_fit_t200"]["power_b"] for x in seg]
                    if (np.all(np.diff(b) > 0) or np.all(np.diff(b) < 0)) and abs(b[-1] - b[0]) >= 0.2 * abs(b[0]):
                        grif = True
        if grif:
            v = "Griffiths-like window"
        elif not any(sig) and slow and n_exp >= (2 / 3) * len(slow):
            v = "exponential (no Griffiths window)"
        else:
            v = "inconclusive"
        verdict[rk] = {"verdict": v, "longest_signature_run": best, "n_slow": len(slow), "n_exp_ok_slow": n_exp,
                       "n_broad": sum(x["broad"] for x in L), "n_power_ok_both": sum(1 for x in L if x["rho_fit_t50"]
                       and x["rho_fit_t200"] and x["rho_fit_t50"]["power_ok"] and x["rho_fit_t200"]["power_ok"])}
    return {"per_g": res, "verdict": verdict}


def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "∞"
    return f"{x:.{nd}g}"


def ftot(t):
    return f"[{fmt(t[0])}, {fmt(t[1])}]" if t else ""


def tables(bnd, comp, pipe, var, where, osc_out, screen, lt, pts):
    L = ["# m2/decay generated tables (code/m2_decay_report.py)", ""]
    L += ["## Boundaries, every row", "",
          "| row | n pts | S [total] | onset [total] | g_half | D [total] (status) | on/S | D/on |", "|---|---|---|---|---|---|---|---|"]
    for k, e in sorted(bnd.items(), key=lambda kv: (kv[1]["model"], kv[1]["graph"], kv[1]["r"])):
        L.append(f"| {k} | {e['n_points']} | {fmt(e['S']['g'])} {ftot(e['S']['total'])} | {fmt(e['on']['g'])} "
                 f"{ftot(e['on']['total'])} | {fmt(e['half']['g'])} | {fmt(e['D']['g'])} {ftot(e['D']['total'])} "
                 f"({e['D']['status']}) | {fmt(e.get('window_on_over_S'))} | {fmt(e.get('D_over_on'))} |")
    L += ["", "## M2 / M2s vs M1 (analysis2)", "", "| row | S M1 → M2 (×) | onset M1 → M2 (×) | D M1 → M2 (×) |", "|---|---|---|---|"]
    for k, c in sorted(comp.items(), key=lambda kv: (bnd[kv[0]]["model"], bnd[kv[0]]["graph"], bnd[kv[0]]["r"])):
        L.append(f"| {k} | " + " | ".join(f"{fmt(c[b]['M1'])} → {fmt(c[b]['M2'])} ({fmt(c[b]['ratio'])})" +
                                          ("" if c[b]["M2_status"] == "ok" else f" [{c[b]['M2_status']}]")
                                          for b in ("S", "on", "D")) + " |")
    L += ["", "## M1 pipeline check (my code on my M1 reruns vs analysis2)", ""]
    for k, v in pipe.items():
        def _w(x):
            return x if isinstance(x, str) else ("inside" if x else "OUTSIDE")
        L.append(f"- {k}: " + "; ".join(f"{b}: mine {fmt(v[b]['mine'])} {ftot(v[b]['mine_total'])} vs a2 {fmt(v[b]['a2'])} "
                                         f"{ftot(v[b]['a2_total'])} (a2 value {_w(v[b]['a2_inside_mine'])} my interval; "
                                         f"intervals overlap: {v[b]['intervals_overlap']})" for b in ("S", "on", "D")))
    L += ["", "## Variants (ratio to M2 at the same r)", "",
          "| row | S (×M2) | onset (×M2) | D (×M2) | on/S | AL share first sustained | PR/N first sustained |", "|---|---|---|---|---|---|---|"]
    for k, v in sorted(var.items()):
        L.append(f"| {k} | " + " | ".join(f"{fmt(v[b]['g'])} ({fmt(v[b]['ratio_to_M2'])})" for b in ("S", "on", "D")) +
                 f" | {fmt(v['window_on_over_S'])} | {fmt(v['AL_share_first_sustained'])} | {fmt(v['PRN_first_sustained'])} |")
    L += ["", "## Where activity lives", "",
          "| row | boundary | g | PR/N | AL module | AL_rest | CNS_rest | lLN1_bc | top superclasses (share, enrichment) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, w in sorted(where.items()):
        for b in ("S", "on", "D"):
            if b not in w:
                continue
            x = w[b]; rs = x["region_share"]
            L.append(f"| {k} | {b} | {fmt(x['g'])} | {fmt(x['PRN'])} | {fmt(rs['AL_module'])} | {fmt(rs['AL_rest'])} | "
                     f"{fmt(rs['CNS_rest'])} | {fmt(rs['lLN1_bc'])} | " +
                     ", ".join(f"{s['superclass']} ({s['share']:.2f}, ×{s['enrichment']:.1f})" for s in x["top_superclasses"]) + " |")
    L += ["", "## Oscillation flags", ""]
    n_tot = len(osc_out); n_osc = sum(o["osc"] for o in osc_out); n_str = sum(o["osc_strong"] for o in osc_out)
    L.append(f"Evaluated (point, signal) pairs: {n_tot}; OSC: {n_osc}; OSC-strong: {n_str}.")
    cm = [o["c_min"] for o in osc_out if o["c_min"] is not None and np.isfinite(o["c_min"])]
    if cm:
        L.append(f"Deepest ACF trough over all evaluated pairs: {min(cm):.3f}; troughs <= -0.10: {sum(1 for c in cm if c <= -0.1)}.")
    L += ["", "| row | g | signal | n_rep | c_min (lag) | c_peak (period) | C(2P)/C(P) | CV_t | OSC | strong | phase |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in osc_out:
        if o["osc"] or (o["c_min"] is not None and np.isfinite(o["c_min"]) and o["c_min"] <= -0.1):
            L.append(f"| {o['row']} | {fmt(o['g'])} | {o['signal']} | {o['n_rep']} | {fmt(o['c_min'])} ({fmt(o['lag_min'])}) | "
                     f"{fmt(o['c_peak'])} ({fmt(o['period'])}) | {fmt(o['coh_ratio'])} | {fmt(o['cv_t'])} | {o['osc']} | {o['osc_strong']} | {o['phase']} |")
    L += ["", "## Lifetime screen (stages 1-2, B = 64, exact times; shifted-exponential KS, 500 bootstrap)", "",
          "Pre-registered shifted-exponential test, and (post hoc, plan section 8) the residual lifetimes of replicas alive at t = 200.", "",
          "| row | n points | KS rejections p<0.05 | Fisher p | CV above / below / tested | post hoc t>200: n, rejections, Fisher p |", "|---|---|---|---|---|---|"]
    for k, s in sorted(screen.items()):
        L.append(f"| {k} | {s['n']} | {s['n_reject_05']} | {fmt(s['fisher_p'])} | {s['n_cv_above']} / {s['n_cv_below']} / {s['n_cv']} | "
                 f"{s['late200_n']}, {s['late200_n_reject_05']}, {fmt(s['late200_fisher_p'])} |")
    L += ["", "## Long runs (T = 30000, B = 128): lifetime shape and rho(t)", ""]
    for rk, v in lt["verdict"].items():
        L.append(f"**{rk}: {v['verdict']}** (longest signature run {v['longest_signature_run']}, slow points {v['n_slow']}, "
                 f"exponential-compatible {v['n_exp_ok_slow']}, broad {v['n_broad']}, power-law-ok both windows {v['n_power_ok_both']})")
        L += ["", "| g | deaths/n | tau | t0 | KS p (shift) | KS p (origin) | CV [H0 95%] | rho t>=50: R2 pow / exp, b (halves) | rho t>=200: R2 pow / exp, b (halves) | broad | signature |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        L2 = ["", "Post hoc (plan section 8): residual lifetimes after t_c, exponential through the origin.", "",
              "| g | t_c | early fraction | n alive at t_c | τ_late | KS p | CV [H0 95%] | hazard (bin centre: rate ×1e3) |", "|---|---|---|---|---|---|---|---|"]
        for x in lt["per_g"][rk]:
            for lc in (x["post_hoc"]["late200"], x["post_hoc"]["late500"]):
                hz = ", ".join(f"{h[0]:.0f}: {1e3 * h[1]:.2g}" for h in x["post_hoc"]["hazard"] if h[4] >= 3) if lc["tc"] == 200 else ""
                L2.append(f"| {fmt(x['g'], 4)} | {lc['tc']:g} | {fmt(lc['early_frac'])} | {lc['n_alive_tc']} | {fmt(lc.get('tau_late'))} | "
                          f"{fmt(lc.get('KS_p'))} | {fmt(lc.get('CV'))} {ftot(lc.get('CV_null_95'))} | {hz} |")
        for x in lt["per_g"][rk]:
            k1, k0 = x["ks_shift"] or {}, x["ks_origin"] or {}
            def ff(f):
                if not f:
                    return "–"
                return (f"{f['R2_power']:.3f} / {f['R2_exp']:.3f}, {f['power_b']:.2f} ({f['b_half1']:.2f}, {f['b_half2']:.2f}); "
                        f"{f['decades']:.1f} dec")
            L.append(f"| {fmt(x['g'], 4)} | {x['deaths']}/{x['n']} | {fmt(x['tau'])} | {fmt(k1.get('t0'))} | {fmt(k1.get('KS_p'))} | "
                     f"{fmt(k0.get('KS_p'))} | {fmt(k1.get('CV'))} {ftot(k1.get('CV_null_95'))} | {ff(x['rho_fit_t50'])} | "
                     f"{ff(x['rho_fit_t200'])} | {x['broad']} | {x['signature']} |")
        L += L2
        L.append("")
    open(f"{OUT}/tables.md", "w").write("\n".join(L) + "\n")
    print("tables ->", f"{OUT}/tables.md")
