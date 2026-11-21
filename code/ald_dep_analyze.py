"""E3 analysis: lifetimes, Q/S/L boundaries, quasi-stationary AL activity, AL leadership, resources; theory comparison.
Inputs: results/aldef/depression/sim/*.npz, sim2/*.npz (ald_sim.py outputs; arms in index_stage*.json), theory via
code/ald_dep_theory.py. Outputs: results/aldef/depression/{points.json, summary.json, table.md, theory_curves.json}.

Definitions (same as code/analysis2_phase.py, plan.md):
 extinction time of a replica = first recorded time with no active neuron (records: 120-point log grid + every 100 units);
 censored at T_c = 3000. tau = censored-exponential MLE sum(min(t_i, T_c)) / deaths, 95% chi-square CI; tau_reg = E/max(d, .5).
 tau_late = same MLE from t0 = 100 on, among replicas alive at t0 (removes the transient of the fully active start).
 Boundaries: ln tau_reg made monotone in g by weighted isotonic regression (weights deaths + 0.5), first crossing of
 ln 200 (S) and ln 3000 (on) interpolated in ln g; g_half: alive fraction at T crosses 1/2. Uncertainty: grid bracket and
 replica bootstrap (NBOOT) -> 2.5-97.5 %.
 QS amplitude: per-replica mean active fraction of each region over 1500 < t <= 3000 (win_mean), averaged over replicas
 alive at T. AL share = sum of occupancy (second half, replica mean) over the AL module / total. PR = (sum occ)^2/sum occ^2.
 last region: for replicas that died, the region (AL_module, AL_rest, CNS_rest) whose last active time is latest.
 efficacy: res_x_act of the AL module region (mean efficacy of active depressing neurons; spike mode: epoch efficacy e,
 rate mode: resource x of active neurons), mean over records in 1500 < t <= 3000 (NaN-mean).
"""
import glob, json, os, sys
import numpy as np
from scipy import stats
from scipy.optimize import isotonic_regression

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/aldef/depression"
sys.path.insert(0, f"{ROOT}/code")
T_REF, TAU_S, T0_LATE, NBOOT = 3000.0, 200.0, 100.0, 1000
G_LIN = 0.16315274988719286
N = 167216
rng = np.random.default_rng(20261109)
P_OFF = 1 - np.exp(-0.2)


def load_index():
    idx = {}
    for p in sorted(glob.glob(f"{OUT}/index_*.json")):
        for e in json.load(open(p)):
            idx.setdefault(e["arm"], {k: v for k, v in e.items() if k not in ("task", "out", "seed")})
            idx[e["arm"]].setdefault("files", []).append(e["out"])
    return idx


def load_arm(files):
    pts = []
    for f in files:
        if not os.path.exists(f):
            continue
        d = np.load(f)
        t, g, surv = d["t"], d["g"], d["surv"]
        names = [str(x) for x in d["reg_names"]]
        win = t <= T_REF + 1e-6
        Tc = float(t[win].max())
        mod = np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)["idx"]
        for i, gg in enumerate(g):
            s = surv[i][win]
            dead = s == 0
            died = dead.any(0)
            ext = np.where(died, t[win][np.argmax(dead, 0)], Tc)
            p = {"g": float(gg), "times": ext.astype(float), "died": died, "B": int(s.shape[1]), "src": [os.path.basename(f)]}
            if "win_mean" in d.files:
                p["win"] = d["win_mean"][i, 0]                 # [R, B]
            p["names"] = names
            if "reg_last" in d.files:
                p["last"] = d["reg_last"][i]                   # [R, B]
            o = d["occ"][i].astype(np.float64)
            p["occ_sum"], p["occ_mod"], p["occ_sq"] = float(o.sum()), float(o[mod].sum()), float((o ** 2).sum())
            p["occ_w"] = int(s.shape[1])
            if "res_x_act" in d.files:
                sel = (t > 1500) & (t <= 3000)
                xa = d["res_x_act"][0, i][sel][:, 0]; xx = d["res_x"][0, i][sel][:, 0]
                p["xact_AL"] = float(np.nanmean(xa)) if np.isfinite(xa).any() else np.nan
                p["x_AL"] = float(np.nanmean(xx)) if np.isfinite(xx).any() else np.nan
            if "res_theta" in d.files:
                sel = (t > 1500) & (t <= 3000)
                th = d["res_theta"][0, i][sel][:, 0]
                p["theta_AL"] = float(np.nanmean(th)) if np.isfinite(th).any() else np.nan
            pts.append(p)
    pts.sort(key=lambda x: x["g"])
    merged = []
    for p in pts:
        if merged and abs(p["g"] / merged[-1]["g"] - 1) < 1e-4:
            q = merged[-1]
            for k in ("times", "died"):
                q[k] = np.r_[q[k], p[k]]
            for k in ("win", "last"):
                if k in q and k in p:
                    q[k] = np.concatenate([q[k], p[k]], axis=1)
            w0, w1 = q["occ_w"], p["occ_w"]
            for k in ("occ_sum", "occ_mod", "occ_sq"):
                q[k] = (q[k] * w0 + p[k] * w1) / (w0 + w1)
            for k in ("xact_AL", "x_AL", "theta_AL"):
                if k in q and k in p:
                    q[k] = float(np.nanmean([q[k], p[k]]))
            q["occ_w"] += w1; q["B"] += p["B"]; q["src"] += p["src"]
        else:
            merged.append(p)
    for p in merged:
        d_, E = int(p["died"].sum()), float(p["times"].sum())
        p["deaths"], p["E"] = d_, E
        p["tau"] = E / d_ if d_ > 0 else float("inf")
        p["tau_reg"] = E / max(d_, 0.5)
        p["tau_ci"] = ([2 * E / stats.chi2.ppf(0.975, 2 * d_ + 2), 2 * E / stats.chi2.ppf(0.025, 2 * d_)] if d_ > 0
                       else [2 * E / stats.chi2.ppf(0.95, 2), float("inf")])
        p["alive"] = float(1 - p["died"].mean())
        al0 = p["times"] > T0_LATE
        dl = int((p["died"] & al0).sum()); El = float((p["times"][al0] - T0_LATE).sum())
        p["n_alive_t0"], p["deaths_late"] = int(al0.sum()), dl
        p["tau_late"] = (El / dl if dl > 0 else float("inf")) if al0.any() else 0.0
        alive = ~p["died"]
        if "win" in p:
            nm = p["names"]
            p["qs"] = {n: (float(p["win"][k, alive].mean()) if alive.any() else None) for k, n in enumerate(nm)}
            p["qs_all"] = {n: float(p["win"][k].mean()) for k, n in enumerate(nm)}
        if "last" in p and p["died"].any():
            L3 = p["last"][:3][:, p["died"]]                   # AL_module, AL_rest, CNS_rest
            p["last_AL_frac"] = float((L3[0] >= L3.max(0)).mean())
        p["AL_share"] = p["occ_mod"] / p["occ_sum"] if p["occ_sum"] > 0 else None
        p["PR"] = p["occ_sum"] ** 2 / p["occ_sq"] / N if p["occ_sq"] > 0 else None
    return merged


def iso(y, w):
    return isotonic_regression(np.asarray(y, float), weights=np.asarray(w, float), increasing=True).x


def first_cross(x, y, thr):
    x = np.asarray(x, float); y = np.asarray(y, float)
    hit = np.where(y >= thr)[0]
    if len(hit) == 0:
        return {"g": None, "bracket": [float(x[-1]), None], "status": "none"}
    k = int(hit[0])
    if k == 0:
        return {"g": None, "bracket": [None, float(x[0])], "status": "below_grid"}
    fr = (thr - y[k - 1]) / (y[k] - y[k - 1]) if y[k] > y[k - 1] else 1.0
    return {"g": float(np.exp(np.log(x[k - 1]) + fr * np.log(x[k] / x[k - 1]))), "bracket": [float(x[k - 1]), float(x[k])],
            "status": "ok"}


def boundaries(P, d=None, tr=None, al=None):
    g = np.array([p["g"] for p in P])
    if d is None:
        d = np.array([p["deaths"] for p in P], float); tr = np.array([p["tau_reg"] for p in P])
        al = np.array([p["alive"] for p in P])
    lt = iso(np.log(tr), d + 0.5)
    return {"S": first_cross(g, lt, np.log(TAU_S)), "on": first_cross(g, lt, np.log(T_REF)),
            "half": first_cross(g, iso(al, [p["B"] for p in P]), 0.5)}


def boot(P):
    nb = len(P)
    D = np.zeros((NBOOT, nb)); E = np.zeros((NBOOT, nb)); A = np.zeros((NBOOT, nb))
    for j, p in enumerate(P):
        ii = rng.integers(0, p["B"], (NBOOT, p["B"]))
        dd = p["died"][ii]; D[:, j] = dd.sum(1); E[:, j] = p["times"][ii].sum(1); A[:, j] = 1 - dd.mean(1)
    res = {k: [] for k in ("S", "on", "half")}
    for b in range(NBOOT):
        bb = boundaries(P, D[b], E[b] / np.maximum(D[b], 0.5), A[b])
        for k in res:
            res[k].append(bb[k]["g"] if bb[k]["g"] is not None else np.nan)
    return {k: np.array(v) for k, v in res.items()}


def summarize(est, bt):
    out = {"g": est["g"], "status": est["status"], "bracket": est["bracket"]}
    v = bt[np.isfinite(bt)]
    out["boot_frac_defined"] = float(np.isfinite(bt).mean())
    out["boot_ci"] = [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) >= 20 else None
    if est["g"] is not None:
        los = [x for x in [est["bracket"][0], out["boot_ci"][0] if out["boot_ci"] else None] if x is not None]
        his = [x for x in [est["bracket"][1], out["boot_ci"][1] if out["boot_ci"] else None] if x is not None]
        out["total"] = [float(min(los)), float(max(his))]
    return out


def pon(h):
    return 1 - np.exp(-np.tanh(h) * 0.2)


def driven_arm(f):
    d = np.load(f)
    m = json.loads(str(d["meta"]))
    h = float(m["h"]); rho_iso = pon(h) / (pon(h) + P_OFF)
    W = d["win_mean"][:, 0]                                   # [g, R, B]
    nm = [str(x) for x in d["reg_names"]]
    out = {"h": h, "rho_iso": rho_iso, "g": d["g"].tolist(), "regions": nm}
    for k, n in enumerate(nm):
        x = W[:, k, :]
        out[n] = {"mean": x.mean(1).tolist(), "se": (x.std(1, ddof=1) / np.sqrt(x.shape[1])).tolist(),
                  "gain": (x.mean(1) / rho_iso).tolist()}
    return out


def removal_arm(files):
    """E1 protocol with depression: per g the driven gain G = rho_uPN_ant / rho_ORN (500-1500), residual uPN_ant activity
    100-300 and 1000-3000 units after all ORNs are clamped (relative to driven), fraction of replicas with any AL-system
    activity at the end (reg_last of ALsys = last step). E1 criteria (results/aldef/drive/plan.md): G in [0.29, 1.45],
    S1 resid100-300 <= 0.1, S2 p_alive_end <= 0.05."""
    rows = []
    for f in files:
        if not os.path.exists(f):
            continue
        d = np.load(f)
        nm = [str(x) for x in d["reg_names"]]; k = {n: i for i, n in enumerate(nm)}
        W = d["win_mean"]                                        # [g, W, R, B]
        last = d["reg_last"]; Tend = float(d["t"][-1])
        for i, g in enumerate(d["g"]):
            orn, pn = W[i, 0, k["ORN"]], W[i, 0, k["uPN_ant"]]
            G = float(pn.mean() / orn.mean()) if orn.mean() > 0 else None
            bt = rng.integers(0, len(pn), (500, len(pn)))
            Gb = pn[bt].mean(1) / np.maximum(orn[bt].mean(1), 1e-12)
            r1 = float(W[i, 1, k["uPN_ant"]].mean() / pn.mean()) if pn.mean() > 0 else None
            r2 = float(W[i, 2, k["uPN_ant"]].mean() / pn.mean()) if pn.mean() > 0 else None
            pa = float((last[i, k["ALsys"]] >= Tend - 0.2 - 1e-6).mean())
            ok = (G is not None and 0.29 <= G <= 1.45 and r1 is not None and r1 <= 0.1 and pa <= 0.05)
            rows.append({"g": float(g), "rho_ORN": float(orn.mean()), "rho_uPN_ant": float(pn.mean()), "G": G,
                         "G_ci": [float(x) for x in np.percentile(Gb, [2.5, 97.5])], "AL_module": float(W[i, 0, k["AL_module"]].mean()),
                         "resid_100_300": r1, "resid_1000_3000": r2, "p_alive_end": pa, "allowed": bool(ok)})
    rows.sort(key=lambda x: x["g"])
    gs = [x["g"] for x in rows]
    allowed = [x["g"] for x in rows if x["allowed"]]
    out = {"rows": rows, "allowed_g": allowed}
    for key, thr in (("G", 0.29), ("G", 1.45)):
        y = [x[key] if x[key] is not None else 0 for x in rows]
        out[f"g_at_{key}_{thr}"] = first_cross(gs, y, thr)["g"]
    y = [x["p_alive_end"] for x in rows]
    out["g_at_p_alive_0.05"] = first_cross(gs, y, 0.05 + 1e-9)["g"]
    y = [x["resid_100_300"] or 0 for x in rows]
    out["g_at_resid_0.1"] = first_cross(gs, y, 0.1)["g"]
    ups = [out[k] for k in ("g_at_G_1.45", "g_at_p_alive_0.05", "g_at_resid_0.1") if out[k] is not None]
    lo_, hi_ = out["g_at_G_0.29"], (min(ups) if ups else None)
    out["allowed_interval"] = [lo_, hi_]
    out["allowed_ratio_hi_lo"] = (hi_ / lo_) if (lo_ and hi_) else None
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def main(with_theory=True):
    idx = load_index()
    summ, points, drv = {}, {}, {}
    for arm, a in idx.items():
        if a["kind"] == "removal":
            rm = removal_arm(a["files"])
            if rm["rows"]:
                drv[arm] = {**{k: v for k, v in a.items() if k != "files"}, **rm}
                print(f"{arm:22s} allowed g {rm['allowed_g']}  G=0.29 at {rm['g_at_G_0.29']} G=1.45 at {rm['g_at_G_1.45']} "
                      f"p_alive>0.05 at {rm['g_at_p_alive_0.05']} resid>0.1 at {rm['g_at_resid_0.1']}", flush=True)
            continue
        if a["kind"] == "driven":
            fs = [f for f in a["files"] if os.path.exists(f)]
            if fs:
                drv[arm] = {**{k: v for k, v in a.items() if k != "files"}, **driven_arm(fs[0])}
            continue
        P = load_arm(a["files"])
        if not P:
            continue
        est = boundaries(P); bt = boot(P)
        s = {k: v for k, v in a.items() if k != "files"}
        s["n_points"] = len(P); s["g_range"] = [P[0]["g"], P[-1]["g"]]; s["B_min"] = min(p["B"] for p in P)
        s["n_files"] = sum(os.path.exists(f) for f in a["files"]); s["n_files_expected"] = len(a["files"])
        for k in ("S", "on", "half"):
            s[k] = summarize(est[k], bt[k])
        sus = [p for p in P if p["alive"] >= 0.5]
        s["first_sustained_g"] = sus[0]["g"] if sus else None
        if sus:
            p0 = sus[0]
            s["at_first_sustained"] = {"g": p0["g"], "AL_share": p0["AL_share"], "PR": p0["PR"], "qs": p0.get("qs"),
                                       "xact_AL": p0.get("xact_AL")}
        summ[arm] = s
        points[arm] = [{k: p.get(k) for k in ("g", "B", "deaths", "tau", "tau_ci", "tau_reg", "alive", "tau_late",
                                               "n_alive_t0", "deaths_late", "qs", "qs_all", "AL_share", "PR",
                                               "last_AL_frac", "xact_AL", "x_AL", "theta_AL", "src")} for p in P]
        print(f"{arm:22s} n={len(P):2d} S={s['S']['g']} on={s['on']['g']} half={s['half']['g']} first_sus={s['first_sustained_g']}",
              flush=True)
    for arm, s in summ.items():
        b = summ.get(f"base_r{s['r']}")
        if b and s["on"]["g"] and b["on"]["g"]:
            s["on_ratio_vs_base"] = s["on"]["g"] / b["on"]["g"]
        if b and s["S"]["g"] and b["S"]["g"]:
            s["S_ratio_vs_base"] = s["S"]["g"] / b["S"]["g"]
    th_out = {}
    if with_theory:
        th_out = theory_compare(summ)
    meso = {"arms": {}}
    for f in sorted(glob.glob(f"{OUT}/meso_part*.json")):
        m = json.load(open(f)); meso["arms"].update(m["arms"]); meso.update({k: m[k] for k in ("S_TOP", "B", "T")})
        meso["kappa"] = m["spec"]["kappa"]
    if meso["arms"]:
        json.dump(meso, open(f"{OUT}/meso.json", "w"), indent=1)
        mb = {r: meso["arms"].get(f"base_r{r}", {}).get("on") for r in (0, 1)}
        for arm, m in meso["arms"].items():
            if arm in summ:
                summ[arm]["meso"] = {"S": m["S"], "on": m["on"],
                                     "on_ratio_vs_meso_base": (m["on"] / mb[summ[arm]["r"]]) if (m["on"] and mb.get(summ[arm]["r"])) else None}
    json.dump(clean(points), open(f"{OUT}/points.json", "w"))
    json.dump(clean({"arms": summ, "driven": drv, "theory": th_out}), open(f"{OUT}/summary.json", "w"), indent=1)
    write_table(summ, th_out)


_TH = None


def _theory_job(job):
    """One theory curve (run in a worker process)."""
    import ald_dep_theory as TH
    global _TH
    if _TH is None:
        _TH = TH.Theory()
    th = _TH
    arm, lab, r, mode, U, TAU, grp, kap = job
    c = th.curve(r, mode, U, TAU, grp, kap)
    e = {"lt_adiabatic": [p["lt_adiabatic"] for p in c], "lt_frozen": [p["lt_frozen"] for p in c],
         "X_star": [p["X_star"] for p in c], "AL_rho_star": [p["AL_rho_star"] for p in c],
         "n_star": [p["n_star"] for p in c], "X0": c[0]["X0"]}
    for v in ("adiabatic", "frozen"):
        e[f"S_{v}"] = TH.cross(th.g, e[f"lt_{v}"], TAU_S)
        e[f"on_{v}"] = TH.cross(th.g, e[f"lt_{v}"], T_REF)
    return arm, lab, list(job[2:]), e


def theory_compare(summ):
    import ald_dep_theory as TH
    from multiprocessing import Pool
    th = TH.Theory()
    out = {"kappa": {}, "curves": {}, "g": th.g.tolist(), "g_lin": th.g_lin}
    for r in (0, 1):
        b = summ.get(f"base_r{r}")
        if b and b["on"]["g"]:
            out["kappa"][r] = TH.fit_kappa(th, r, b["on"]["g"])
    cpath = f"{OUT}/theory_cache.json"
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    jobs = []
    for arm, s in summ.items():
        if s["kind"] not in ("base", "std_spike", "std_rate", "std_group"):
            continue
        r = s["r"]
        mode = None if s["kind"] == "base" else ("rate" if s["kind"] == "std_rate" else "spike")
        grp = s.get("group") if s["kind"] == "std_group" else None
        for lab, kap in (("raw", 1.0), ("cal", out["kappa"].get(r))):
            if kap is not None:
                jobs.append((arm, lab, r, mode, s.get("U"), s.get("TAU"), grp, kap))
    todo = [j for j in jobs if cache.get(f"{j[0]}|{j[1]}", {}).get("params") != list(j[2:])]
    if todo:
        with Pool(min(16, len(todo))) as pool:
            for arm, lab, params, e in pool.imap_unordered(_theory_job, todo):
                cache[f"{arm}|{lab}"] = {"params": params, "e": e}
        json.dump(clean(cache), open(cpath, "w"))
    for j in jobs:
        out["curves"].setdefault(j[0], {})[j[1]] = cache[f"{j[0]}|{j[1]}"]["e"]
    for arm, res in out["curves"].items():
        print("theory", arm, {lab: (round(e["S_frozen"] or -1, 3), round(e["on_frozen"] or -1, 3),
                                    round(e["S_adiabatic"] or -1, 3), round(e["on_adiabatic"] or -1, 3)) for lab, e in res.items()},
              flush=True)
    return out


def fmt(x, p=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}g}"


def write_table(summ, th):
    L = ["# E3 depression / adaptation / refractoriness: boundaries (generated by code/ald_dep_analyze.py)", "",
         "S: tau = 200; on: tau = 3000 (T); half: half of the replicas alive at T. [total] = envelope of grid bracket and "
         "replica-bootstrap 95% range. Theory: depression-extended birth-death chain (frozen / adiabatic resource), "
         "K calibrated on this experiment's base onset (kappa), r as in the arm. '> 4.9' = no crossing on the theory grid.", "",
         "| arm | kind | params | S [total] | on [total] | on / base | g_half | first g with half alive | AL share there | "
         "theory on frozen (cal) | theory on adiabatic (cal) | theory on / base (frozen, cal) | meso on | meso on / meso base |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, s in summ.items():
        par = ", ".join(f"{k}={s[k]}" for k in ("U", "TAU", "D", "TAUA", "nref", "group") if k in s)
        tot = lambda k: f"{fmt(s[k]['g'])} [{fmt((s[k].get('total') or [None, None])[0])}, {fmt((s[k].get('total') or [None, None])[1])}]" \
            if s[k]["g"] else ("none ≤ " + fmt(s["g_range"][1]) if s[k]["status"] == "none" else s[k]["status"])
        t = th.get("curves", {}).get(arm, {}).get("cal", {})
        tb = th.get("curves", {}).get(f"base_r{s['r']}", {}).get("cal", {})
        tr = (t.get("on_frozen") / tb.get("on_frozen")) if (t.get("on_frozen") and tb.get("on_frozen")) else None
        a0 = s.get("at_first_sustained", {})
        L.append(f"| {arm} | {s['kind']} | {par} | {tot('S')} | {tot('on')} | {fmt(s.get('on_ratio_vs_base'))} | "
                 f"{fmt(s['half']['g'])} | {fmt(s.get('first_sustained_g'))} | {fmt(a0.get('AL_share'))} | "
                 f"{fmt(t.get('on_frozen')) if t else '–'}{'' if (not t or t.get('on_frozen')) else ' (> 4.9)'} | "
                 f"{fmt(t.get('on_adiabatic')) if t else '–'} | {fmt(tr)} | {fmt(s.get('meso', {}).get('on'))} | "
                 f"{fmt(s.get('meso', {}).get('on_ratio_vs_meso_base'))} |")
    L += ["", f"kappa (K_cal / K_eff): {th.get('kappa')}"]
    open(f"{OUT}/table.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(with_theory="--no_theory" not in sys.argv)
