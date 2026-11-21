"""Lane m2/verify: independent recomputation of key Model-2 numbers from RAW outputs (plan results/m2/verify/plan.md, A).
Imports no project module (numpy / scipy only); reads raw npz and the lanes' JSON only to obtain the lane values to
compare against.  Run: python m2_verify.py [items...]  -> results/m2/verify/verify.json (merged per item)
"""
import glob, json, os, sys, time
import numpy as np

ROOT = "/MALECNS_ROOT"
M2 = f"{ROOT}/results/m2"
OUT = f"{M2}/verify"
DT = 0.2
N_ALL = 167216
rng = np.random.default_rng(20261109)


# ----------------------------------------------------------------------------------------------- generic estimators
def pava(y, w):
    """isotonic (non-decreasing) regression, pool-adjacent-violators"""
    y = list(map(float, y)); w = list(map(float, w))
    blocks = [[y[i], w[i], 1] for i in range(len(y))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            y2, w2, n2 = out.pop(); y1, w1, n1 = out.pop()
            out.append([(y1 * w1 + y2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    res = []
    for v, _, n in out:
        res += [v] * n
    return np.array(res)


def first_cross(g, y, thr):
    """first upward crossing of thr by y(g), linear in ln g; None if never; ('below', g0) if already above at g[0]"""
    g = np.asarray(g, float); y = np.asarray(y, float)
    idx = np.where(y >= thr)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(g[0])
    y0, y1 = y[i - 1], y[i]
    f = (thr - y0) / (y1 - y0) if y1 != y0 else 0.0
    return float(np.exp(np.log(g[i - 1]) + f * (np.log(g[i]) - np.log(g[i - 1]))))


def ext_times_grid(t, rho_gtb):
    """first record time with rho == 0 per replica (censored at T = t[-1])"""
    T = t[-1]
    dead = rho_gtb <= 0  # [t, B]
    any_dead = dead.any(axis=0)
    first = np.argmax(dead, axis=0)
    et = np.where(any_dead, t[first], np.inf)
    return et, T


def tau_mle(et, T):
    d = int(np.isfinite(et).sum() if np.isinf(et).any() else (et < T).sum())
    d = int((et < T).sum())
    E = float(np.minimum(et, T).sum())
    return (E / d if d > 0 else np.inf), d, E


def pr(o):
    o = np.asarray(o, np.float64)
    s2 = (o ** 2).sum()
    return float(o.sum() ** 2 / s2) if s2 > 0 else 0.0


# ----------------------------------------------------------------------------------------------- decay rows
_MODULE = None


def module_idx():
    global _MODULE
    if _MODULE is None:
        _MODULE = np.load(f"{ROOT}/results/module/module_graph.npz")["idx"].astype(np.int64)
    return _MODULE


def decay_files(row, sim=f"{M2}/decay/sim"):
    fs = sorted(glob.glob(f"{sim}/s1_{row}_p*.npz")) + sorted(glob.glob(f"{sim}/s2_{row}_*.npz"))
    return fs


def decay_points(files, want_occ=True):
    """pool replicas per g over files; returns sorted list of dicts g, tau, deaths, E, B, et, occ (replica-weighted)"""
    P = {}
    for f in files:
        d = np.load(f)
        t = d["t"]; g = d["g"]; rho = d["rho"]
        occ = d["occ"] if want_occ else None
        for gi, gv in enumerate(g):
            et, T = ext_times_grid(t, rho[gi])
            key = round(float(gv), 6)
            p = P.setdefault(key, {"g": float(gv), "et": [], "occ": None, "B": 0, "T": T, "files": []})
            assert abs(p["T"] - T) < 1e-6
            p["et"].append(et)
            if want_occ:
                o = occ[gi].astype(np.float64) * len(et)
                p["occ"] = o if p["occ"] is None else p["occ"] + o
            p["B"] += len(et); p["files"].append(os.path.basename(f))
    pts = []
    for k in sorted(P):
        p = P[k]
        et = np.concatenate(p["et"])
        tau, dd, E = tau_mle(et, p["T"])
        p.update(et=et, tau=tau, deaths=dd, E=E)
        if want_occ:
            p["occ"] = p["occ"] / p["B"]
        pts.append(p)
    return pts


def decay_lines(pts, N=N_ALL):
    g = np.array([p["g"] for p in pts])
    E = np.array([p["E"] for p in pts]); d = np.array([p["deaths"] for p in pts])
    y = np.log(E / np.maximum(d, 0.5))
    iso = pava(y, d + 0.5)
    S = first_cross(g, iso, np.log(200.0)); on = first_cross(g, iso, np.log(3000.0))
    alive_half = None
    res = {"S": S, "on": on, "n_points": len(g)}
    if pts[0].get("occ") is not None and on is not None:
        sel = [i for i in range(len(g)) if g[i] >= on]
        prn = np.array([pr(pts[i]["occ"]) / N for i in sel])
        isop = pava(prn, np.ones(len(prn)))
        D = first_cross(g[sel], isop, 0.25)
        res["D"] = D
        res["D_at_onset"] = bool(len(prn) and isop[0] >= 0.25)
        # first sustained grid point (point estimate tau >= 3000)
        fs = [i for i in range(len(g)) if pts[i]["tau"] >= 3000]
        if fs:
            i = fs[0]; o = pts[i]["occ"]
            res["first_sustained"] = {"g": float(g[i]), "PR": pr(o), "PR_N": pr(o) / N,
                                      "AL_share": float(o[module_idx()].sum() / o.sum())}
    return res


def lane_decay(row):
    b = json.load(open(f"{M2}/decay/boundaries.json"))["rows"][row]
    return {k: (b[k]["g"], b[k].get("total")) for k in ("S", "on", "D") if k in b}


def passes(mine, lane, total=None, tol=0.02):
    if mine is None or lane is None:
        return mine is None and lane is None
    if abs(mine / lane - 1) <= tol:
        return True
    if total and total[0] is not None and total[1] is not None and total[0] <= mine <= total[1]:
        return True
    return False


# ----------------------------------------------------------------------------------------------- items
def item_decay():
    out = {}
    for row in ("M2_real_r0", "M2_real_r1", "M2_real_r6", "M2_cnull_r0", "M2_cnull_r6", "M2_real_r0.25", "M2_real_r4"):
        pts = decay_points(decay_files(row))
        L = decay_lines(pts)
        lane = lane_decay(row)
        o = {"mine": L, "lane": {k: v[0] for k, v in lane.items()}, "lane_total": {k: v[1] for k, v in lane.items()},
             "n_files": len(decay_files(row)), "B_per_point": sorted(set(p["B"] for p in pts))}
        o["pass"] = {k: passes(L.get(k), lane[k][0], lane[k][1]) for k in lane if k in ("S", "on", "D")}
        if row.startswith("M2_cnull"):
            o["pass"]["D"] = None  # D at onset for the null
        out[row] = o
        print(row, {k: (round(L[k], 4) if isinstance(L.get(k), float) else L.get(k)) for k in ("S", "on", "D")},
              o["lane"], o["pass"], L.get("first_sustained"))
    out["ratio_real_null"] = {r: out[f"M2_real_r{r}"]["mine"]["on"] / out[f"M2_cnull_r{r}"]["mine"]["on"] for r in ("0", "6")}
    print("real/null", out["ratio_real_null"])
    return out


def driven_files(sett, model_graph, r, palp=False):
    tag = f"{model_graph}_o0.04_r{r}" + ("_palp" if palp else "")
    fs = sorted(glob.glob(f"{M2}/driven/runs/{sett}/{tag}_c*.npz")) + sorted(glob.glob(f"{M2}/driven/runs/{sett}/{tag}_ref*_c*.npz"))
    return fs


def driven_points(files):
    P = {}
    for f in files:
        d = np.load(f)
        names = list(d["reg_names"]); ix = {n: i for i, n in enumerate(names)}
        wm = d["win_mean"]; w2 = d["win_m2"]; rl = d["reg_last"]; g = d["g"]
        for gi, gv in enumerate(g):
            key = round(float(gv), 6)
            p = P.setdefault(key, {"g": float(gv), "orn": [], "upn": [], "upn_post": [], "upn_late": [], "non": [],
                                   "var_non": [], "rl_non": [], "rl_als": [], "files": []})
            p["orn"].append(wm[gi, 0, ix["ORN"]]); p["upn"].append(wm[gi, 0, ix["uPN_ant"]])
            p["upn_post"].append(wm[gi, 1, ix["uPN_ant"]]); p["upn_late"].append(wm[gi, 2, ix["uPN_ant"]])
            p["non"].append(wm[gi, 0, ix["nonORN"]])
            p["var_non"].append(w2[gi, 0, ix["nonORN"]] - wm[gi, 0, ix["nonORN"]].astype(np.float64) ** 2)
            p["rl_non"].append(rl[gi, ix["nonORN"]]); p["rl_als"].append(rl[gi, ix["ALsys"]])
            p["files"].append(os.path.basename(f))
    pts = []
    for k in sorted(P):
        p = P[k]
        for q in ("orn", "upn", "upn_post", "upn_late", "non", "var_non", "rl_non", "rl_als"):
            p[q] = np.concatenate(p[q]).astype(np.float64)
        B = len(p["orn"])
        G = p["upn"].mean() / p["orn"].mean()
        s1 = p["upn_post"].mean() / p["upn"].mean()
        s2p = p["upn_late"].mean() / p["upn"].mean()
        alive_als = int((p["rl_als"] >= 4500 - DT - 1e-6).sum())
        ext = p["rl_non"] + DT - 1500.0
        cens = p["rl_non"] >= 4500 - DT - 1e-6
        ext = np.where(cens, np.inf, ext)
        tau, dd, E = tau_mle(ext, 3000.0)
        chi = 164577 * p["var_non"].mean()
        pts.append({"g": p["g"], "B": B, "G": G, "S1": s1, "S2p": s2p, "alive_ALsys": alive_als, "tau_post": tau,
                    "deaths": dd, "E": E, "rho_non": p["non"].mean(), "rho_orn": p["orn"].mean(),
                    "F_non": chi / p["non"].mean() if p["non"].mean() > 0 else np.nan, "files": sorted(set(p["files"]))})
    return pts


def driven_lines(pts):
    g = np.array([p["g"] for p in pts]); G = np.array([p["G"] for p in pts])
    S1 = np.array([p["S1"] for p in pts]); a2 = np.array([p["alive_ALsys"] / p["B"] for p in pts])
    # DQ->DA: first upward crossing of G = 0.29 on the raw curve
    lo = first_cross(g, G, 0.29)
    allowed = (G >= 0.29) & (G <= 1.45) & (S1 <= 0.1) & (a2 <= 3.5 / 64)
    top = None; top_by = None
    if allowed.any():
        i0 = np.where(allowed)[0][0]
        for i in range(i0 + 1, len(g)):
            fails = {"G": G[i] > 1.45, "S1": S1[i] > 0.1, "S2": a2[i] > 3.5 / 64}
            if any(fails.values()):
                cands = []
                if fails["S1"]:
                    cands.append(("S1", first_cross(g[i - 1:i + 1], S1[i - 1:i + 1], 0.1)))
                if fails["G"]:
                    cands.append(("G", first_cross(g[i - 1:i + 1], G[i - 1:i + 1], 1.45)))
                if fails["S2"]:
                    cands.append(("S2", first_cross(g[i - 1:i + 1], a2[i - 1:i + 1], 3.5 / 64)))
                top_by, top = min(cands, key=lambda c: c[1])
                break
    E = np.array([p["E"] for p in pts]); d = np.array([p["deaths"] for p in pts])
    iso = pava(np.log(E / np.maximum(d, 0.5)), d + 0.5)
    ss = first_cross(g, iso, np.log(3000.0))
    k = int(np.nanargmax([p["F_non"] for p in pts]))
    return {"DQ_DA": lo, "allowed_top": top, "allowed_top_by": top_by, "DT_SS": ss,
            "F_non_peak": {"g": float(g[k]), "value": float(pts[k]["F_non"])}}


def item_driven():
    cur = json.load(open(f"{M2}/driven/curves.json"))
    out = {}
    dec = json.load(open(f"{M2}/decay/boundaries.json"))["rows"]
    ratios = []
    for r in ("0", "0.125", "0.25", "0.375", "0.5", "0.625", "0.75", "0.875", "1", "1.25", "1.5", "1.75", "2", "2.25",
              "2.5", "2.75", "3", "3.5", "4", "4.5", "5", "5.5", "6"):
        pts = driven_points(driven_files("A1", "M2_real", r))
        L = driven_lines(pts)
        lane = cur[f"A1|M2|real|0.04|{r}|False"]["lines"]
        o = {"mine": L, "lane": {k: lane.get(k) for k in ("DQ_DA", "allowed_hi", "allowed_hi_by", "DT_SS")}}
        o["pass"] = {"DQ_DA": passes(L["DQ_DA"], lane.get("DQ_DA")), "allowed_top": passes(L["allowed_top"], lane.get("allowed_hi")),
                     "DT_SS": passes(L["DT_SS"], lane.get("DT_SS"))}
        on_dec = decay_lines(decay_points(decay_files(f"M2_real_r{r}"), want_occ=False))["on"]
        o["DT_SS_over_decay_on_mine"] = L["DT_SS"] / on_dec if (L["DT_SS"] and on_dec) else None
        ratios.append(o["DT_SS_over_decay_on_mine"])
        out[r] = o
        print("driven r", r, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in L.items() if k != "F_non_peak"},
              L["F_non_peak"], o["lane"], o["pass"], "DT/on", o["DT_SS_over_decay_on_mine"])
    out["DT_SS_over_decay_on_range"] = [min(x for x in ratios if x), max(x for x in ratios if x)]
    # palp-intact
    pal = {}
    for model in ("M2_real", "M2s_real"):
        sett = "B4"
        for r in ("0", "1"):
            pts = driven_points(driven_files(sett, model, r, palp=True))
            rows = [{"g": p["g"], "G": round(p["G"], 4), "S1p": round(p["S1"], 4), "S2p": round(p["S2p"], 4), "B": p["B"],
                     "ok": bool(p["S1"] <= 0.1 and p["S2p"] <= 0.1)} for p in pts]
            ok = [x["g"] for x in rows if x["ok"]]
            first_fail = next((x["g"] for x in rows if not x["ok"]), None)
            gcross = first_cross([x["g"] for x in rows], [max(x["S1p"], x["S2p"]) for x in rows], 0.1)
            gG = first_cross([x["g"] for x in rows], [x["G"] for x in rows], 0.29)
            pal[f"{model}|r{r}"] = {"rows": rows, "largest_pass_before_fail": max([g for g in ok if g < (first_fail or 1e9)], default=None),
                                    "first_fail": first_fail, "palp_10pct_crossing_interp": gcross,
                                    "G_029_crossing_same_runs": gG}
            print("palp", model, r, pal[f"{model}|r{r}"]["largest_pass_before_fail"], first_fail, "cross", gcross, "Gedge", gG)
    out["palp"] = pal
    return out


def item_driven_extra():
    """M1 (A2), nulls (A3) and M2s (A4) driven lines at core r; is there any allowed point in the nulls?"""
    out = {}
    for sett, mg in (("A2", "M1_real"), ("A3", "M2_cnull"), ("A3", "M2_wnull"), ("A4", "M2s_real")):
        for r in ("0", "1", "6"):
            pts = driven_points(driven_files(sett, mg, r))
            if not pts:
                continue
            L = driven_lines(pts)
            n_allowed = sum(1 for p in pts if 0.29 <= p["G"] <= 1.45 and p["S1"] <= 0.1 and p["alive_ALsys"] / p["B"] <= 3.5 / 64)
            g = np.array([p["g"] for p in pts])
            s1lost = first_cross(g, np.array([p["S1"] for p in pts]), 0.1)
            out[f"{mg}|r{r}"] = {k: L[k] for k in ("DQ_DA", "allowed_top", "DT_SS")}
            out[f"{mg}|r{r}"].update(n_allowed_points=n_allowed, S1_lost=s1lost)
            print("drvx", mg, r, out[f"{mg}|r{r}"])
    return out


def item_decay_exact():
    """estimator sensitivity: onset/S from EXACT extinction times (max over the 3 default regions of reg_last + dt)
    instead of MODEL2's first log-grid record with no activity"""
    out = {}
    for row in ("M2_real_r0", "M2_real_r1", "M2_real_r6", "M2_cnull_r0"):
        P = {}
        for fpath in decay_files(row):
            d = np.load(fpath)
            for gi, gv in enumerate(d["g"]):
                rl = d["reg_last"][gi, :3, :].max(0)
                et = np.where(rl >= 3000 - DT - 1e-6, np.inf, rl + DT)
                P.setdefault(round(float(gv), 6), []).append(et)
        g = np.array(sorted(P)); E = []; dd = []
        for k in g:
            et = np.concatenate(P[k]); t_, d_, E_ = tau_mle(et, 3000.0); E.append(E_); dd.append(d_)
        E = np.array(E); dd = np.array(dd)
        iso = pava(np.log(E / np.maximum(dd, 0.5)), dd + 0.5)
        S = first_cross(g, iso, np.log(200)); on = first_cross(g, iso, np.log(3000))
        grid = decay_lines(decay_points(decay_files(row), want_occ=False))
        out[row] = {"S_exact": S, "on_exact": on, "S_grid": grid["S"], "on_grid": grid["on"],
                    "on_ratio_exact_grid": on / grid["on"], "S_ratio_exact_grid": S / grid["S"]}
        print("exact", row, out[row])
    return out


def item_misc():
    """ORN-level 0.10 allowed region (r 1, 4), variant onsets (tau30 / KW09 / N15 at r 0, ratio to M2), M2s onset r 0,
    allowed-top / decay-S and DT->SS / allowed-top over the 23 r (from this lane's own recomputed lines)"""
    out = {}
    for lev in ("0.01", "0.10"):
        for r in ("0", "1", "4"):
            fs = sorted(glob.glob(f"{M2}/driven/runs/B1/M2_real_o{lev}_r{r}_c*.npz")) + \
                 sorted(glob.glob(f"{M2}/driven/runs/B1/M2_real_o{lev}_r{r}_ref*_c*.npz"))
            pts = driven_points(fs)
            n_al = sum(1 for p in pts if 0.29 <= p["G"] <= 1.45 and p["S1"] <= 0.1 and p["alive_ALsys"] / p["B"] <= 3.5 / 64)
            L = driven_lines(pts)
            out[f"orn{lev}|r{r}"] = {"n_allowed_points": n_al, "DQ_DA": L["DQ_DA"], "allowed_top": L["allowed_top"],
                                     "DT_SS": L["DT_SS"], "maxG_with_S1": max([p["G"] for p in pts if p["S1"] <= 0.1], default=None)}
            print("orn", lev, r, out[f"orn{lev}|r{r}"])
    on_m2 = decay_lines(decay_points(decay_files("M2_real_r0"), want_occ=False))["on"]
    for v in ("M2_tau30_real_r0", "M2_KW09_real_r0", "M2_N15_real_r0", "M2s_real_r0", "M2_allsyn_real_r0"):
        L = decay_lines(decay_points(decay_files(v)))
        out[v] = {"on": L["on"], "ratio_to_M2": L["on"] / on_m2, "first_sustained": L.get("first_sustained")}
        print("variant", v, out[v])
    R = json.load(open(f"{OUT}/verify.json"))
    top_S, ss_top = [], []
    for r, o in R["driven"].items():
        if not isinstance(o, dict) or "mine" not in o:
            continue
        L = decay_lines(decay_points(decay_files(f"M2_real_r{r}"), want_occ=False))
        top_S.append(o["mine"]["allowed_top"] / L["S"]); ss_top.append(o["mine"]["DT_SS"] / o["mine"]["allowed_top"])
    out["allowed_top_over_decay_S"] = [min(top_S), max(top_S)]
    out["DT_SS_over_allowed_top"] = [min(ss_top), max(ss_top)]
    print("ratios", out["allowed_top_over_decay_S"], out["DT_SS_over_allowed_top"])
    return out


def item_fss():
    """FSS W = ln(on/S) per graph; family means; ratio W(167k)/W(5k) (cnull) and W(full)/W(20k) (realsubu)"""
    sim = f"{M2}/nature/sim/fss"
    out = {}
    for fam, sizes in (("cnull", (5000, 10000, 20000, 40000, 80000, 167216)), ("realsubu", (20000, 40000, 80000))):
        for r in ("0", "1", "4"):
            fr = {}
            for N in sizes:
                Ws = []
                for s in (1, 2, 3):
                    fs = sorted(glob.glob(f"{sim}/fss_{fam}_N{N}_s{s}_r{r}_*.npz"))
                    if not fs:
                        continue
                    L = decay_lines(decay_points(fs, want_occ=False), N=N)
                    if L["S"] and L["on"]:
                        Ws.append({"seed": s, "S": L["S"], "on": L["on"], "W": float(np.log(L["on"] / L["S"]))})
                if fam == "cnull" and N == 167216:  # decay-lane null rows are the remaining seeds
                    extra = [f"M2_cnull_r{r}"] + (["M2_cnull1_r1"] if r == "1" else [])
                    for row in extra:
                        L = decay_lines(decay_points(decay_files(row), want_occ=False))
                        Ws.append({"seed": row, "S": L["S"], "on": L["on"], "W": float(np.log(L["on"] / L["S"]))})
                fr[N] = Ws
            if fam == "realsubu":
                L = decay_lines(decay_points(decay_files(f"M2_real_r{r}"), want_occ=False))
                fr["full"] = [{"seed": "real", "S": L["S"], "on": L["on"], "W": float(np.log(L["on"] / L["S"]))}]
            means = {str(N): (float(np.mean([w["W"] for w in v])) if v else None, len(v)) for N, v in fr.items()}
            o = {"per_graph": {str(k): v for k, v in fr.items()}, "W_mean": means}
            if fam == "cnull":
                ws = [m[0] for m in means.values()]
                o["ratio_167k_5k"] = ws[-1] / ws[0] if ws[0] else None
                # seed jackknife of the ratio (drop one seed at each end)
                a = [w["W"] for w in fr[5000]]; b = [w["W"] for w in fr[167216]]
                jk = [np.mean([x for j, x in enumerate(b) if j != jb]) / np.mean([x for j, x in enumerate(a) if j != ja])
                      for ja in range(len(a)) for jb in range(len(b))]
                o["ratio_jackknife_range"] = [float(min(jk)), float(max(jk))]
                # bootstrap over seeds
                bs = [np.mean(rng.choice(b, len(b))) / np.mean(rng.choice(a, len(a))) for _ in range(4000)]
                o["ratio_seed_boot_95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                o["frac_boot_le_0.5"] = float(np.mean(np.array(bs) <= 0.5))
            else:
                o["ratio_full_20k"] = means["full"][0] / means["20000"][0]
            out[f"{fam}|r{r}"] = o
            print(fam, r, {k: (round(v[0], 4) if v[0] else None, v[1]) for k, v in means.items()},
                  o.get("ratio_167k_5k") or o.get("ratio_full_20k"), o.get("ratio_seed_boot_95"))
    return out


def item_hyst():
    sim = f"{M2}/nature/sim/hyst"
    out = {}
    dl = np.log(2.0 / 0.5) / 16
    for proto, model, r in (("ramp_h0_down", "M2", "0"), ("ramp_h0_down", "M2", "1"), ("ramp_driven", "M2", "1"),
                            ("ramp_driven", "M2", "4")):
        A = {}
        for D in (50, 200, 800, 3200):
            f = f"{sim}/hyst_{proto}_{model}_r{r}_D{D}_k0.npz"
            if not os.path.exists(f):
                continue
            d = np.load(f)
            if proto == "ramp_driven":
                ix = list(d["reg_names"]).index("nonORN"); x = d["seg_reg"][:, ix, :]
            else:
                x = d["seg_rho"]
            n = x.shape[0]; m = (n - 1) // 2
            first = x[:m].mean(1); second = x[m + 1:][::-1].mean(1)
            assert np.allclose(d["seg_g"][:m], d["seg_g"][m + 1:][::-1])
            A[D] = float(np.abs(first - second).sum() * dl)
        o = {"A": A}
        if 50 in A and 800 in A:
            o["A50_over_A800"] = A[50] / A[800]
        out[f"{proto}|{model}|r{r}"] = o
        print("hyst", proto, model, r, {k: f"{v:.3g}" for k, v in A.items()}, o.get("A50_over_A800"))
    lane = json.load(open(f"{M2}/nature/hyst.json"))["verdicts"]
    out["lane"] = {k: lane[k]["A"] for k in lane if k.split("|")[1] == "M2" and k.split("|")[2] in ("r0", "r1", "r4")}
    return out


def mr_m(a, kmax=20):
    """own multistep regression (Wilting & Priesemann 2018): r_k regression slopes, ln r_k = ln b + k ln m"""
    a = np.asarray(a, np.float64)
    ks = np.arange(1, kmax + 1); r = []
    for k in ks:
        x = a[:-k].ravel(); y = a[k:].ravel()
        r.append(np.cov(x, y, bias=True)[0, 1] / x.var())
    r = np.array(r); ok = r > 0
    sl = np.polyfit(ks[ok], np.log(r[ok]), 1)[0]
    return float(np.exp(sl))


def item_branch():
    out = {}
    lane = json.load(open(f"{M2}/nature/driven_branch.json"))["points"]
    best = max((q for q in lane if q["model"] == "M2" and q["graph"] == "real" and q["set"] == "A1"), key=lambda q: q["m_step"])
    r = best["r"]; gv = best["g"]
    for f in driven_files("A1", "M2_real", f"{r:g}"):
        d = np.load(f)
        gi = np.where(np.abs(d["g"] - gv) < 1e-6)[0]
        if len(gi):
            ix = list(d["reg_names"]).index("nonORN")
            a = d["cnt_reg_on"][gi[0], :, ix, :]
            m = mr_m(a)
            out = {"r": r, "g": gv, "file": os.path.basename(f), "m_step_mine": m, "m_unit_mine": m ** 5,
                   "m_step_lane": best["m_step"], "m_unit_lane": best["m_unit"],
                   "tau_ac_units_from_m_step": float(-DT / np.log(m)),
                   "n_points_m_step_ge_0.985": sum(1 for q in lane if q["model"] == "M2" and q["graph"] == "real" and q["m_step"] >= 0.985),
                   "pulse_max_m_unit_lane": None}
            break
    a = json.load(open(f"{M2}/nature/aval.json"))["points"]
    m2p = [q for q in a if q.get("model") == "M2" and q.get("m_unit") is not None]
    if m2p:
        q = max(m2p, key=lambda q: q["m_unit"])
        out["pulse_max_m_unit_lane"] = q["m_unit"]; out["pulse_max_m_step_equiv"] = q["m_unit"] ** 0.2
        out["pulse_max_at"] = {k: q.get(k) for k in ("r", "G", "g")}
    print("branch", out)
    return out


def item_lifetimes():
    out = {}
    for tag, row, gi in (("real_r1_g1.57", "M2_real_r1", 5), ("real_r0", "M2_real_r0", 5), ("real_r4", "M2_real_r4", 5)):
        fs = sorted(glob.glob(f"{M2}/decay/sim/s3a_{row}_g{gi}_p*.npz"))
        if not fs:
            continue
        ets, g = [], None
        rows_grid = []
        for f in fs:
            d = np.load(f); g = float(d["g"][0])
            rl = d["reg_last"][0, :3, :].max(0)  # default disjoint regions
            T = float(d["t"][-1])
            et = np.where(rl >= T - DT - 1e-6, np.inf, rl + DT)
            ets.append(et)
        et = np.concatenate(ets); T = 30000.0
        n = len(et)
        tau30k, d30k, _ = tau_mle(et, T)
        tau3k, d3k, _ = tau_mle(np.where(et > 3000, np.inf, et), 3000.0)
        tc = 200.0
        al = et[et > tc]; early = float(np.mean(et <= tc))
        res = np.minimum(al, T) - tc; dd = int(np.isfinite(al).sum() - (al >= T).sum())
        dd = int((al < T).sum())
        tau_late = float(res.sum() / dd) if dd else np.inf
        # KS of residual lifetimes against an exponential through the origin (uncensored part; censoring is rare)
        obs = np.sort(res[al < T])
        def ksD(x, sc):
            F = 1 - np.exp(-x / sc); nn = len(x)
            return float(max((np.arange(1, nn + 1) / nn - F).max(), (F - np.arange(0, nn) / nn).max()))
        D0 = ksD(obs, obs.mean())
        bs = []
        for _ in range(2000):
            x = np.sort(rng.exponential(obs.mean(), len(obs)))
            bs.append(ksD(x, x.mean()))
        p = float(np.mean(np.array(bs) >= D0))
        out[tag] = {"g": g, "n": n, "tau_T30000": tau30k, "tau_T3000_truncated": tau3k, "deaths_T3000": d3k,
                    "early_frac_t200": early, "tau_late_200": tau_late, "n_alive_200": int(len(al)),
                    "KS_resid_D": D0, "KS_resid_p": p, "files": [os.path.basename(f) for f in fs]}
        print("life", tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[tag].items() if k != "files"})
    # finite-T: onset from long runs (real r 1, 6 g): tau with T = 3000 vs 30000, crossing of 3000
    for row in ("M2_real_r0", "M2_real_r1", "M2_real_r4"):
        gs, t3, t30 = [], [], []
        for gi in range(6):
            fs = sorted(glob.glob(f"{M2}/decay/sim/s3a_{row}_g{gi}_p*.npz"))
            if not fs:
                continue
            ets = []
            for f in fs:
                d = np.load(f); gv = float(d["g"][0])
                rl = d["reg_last"][0, :3, :].max(0)
                ets.append(np.where(rl >= 30000 - DT - 1e-6, np.inf, rl + DT))
            et = np.concatenate(ets)
            gs.append(gv)
            a, d1, E1 = tau_mle(np.where(et > 3000, np.inf, et), 3000.0)
            b, d2, E2 = tau_mle(et, 30000.0)
            t3.append((E1, d1)); t30.append((E2, d2))
        gs = np.array(gs)
        def cross(tt):
            E = np.array([x[0] for x in tt]); d = np.array([x[1] for x in tt])
            return first_cross(gs, pava(np.log(E / np.maximum(d, 0.5)), d + 0.5), np.log(3000))
        out[f"finiteT|{row}"] = {"g": gs.tolist(), "tau_T3000": [x[0] / max(x[1], 0.5) for x in t3],
                                 "tau_T30000": [x[0] / max(x[1], 0.5) for x in t30],
                                 "on_T3000_exact": cross(t3), "on_T30000_exact": cross(t30)}
        print("finiteT", row, out[f"finiteT|{row}"]["on_T3000_exact"], out[f"finiteT|{row}"]["on_T30000_exact"])
    return out


def acf_metrics(x, dt=DT):
    """x [S, B]: replica-averaged autocorrelation (mean of per-replica autocovariances / mean variance); MODEL2 5a rule"""
    x = x - x.mean(0, keepdims=True)
    S = x.shape[0]
    nf = 1 << (2 * S - 1).bit_length()
    F = np.fft.rfft(x, nf, axis=0)
    ac = np.fft.irfft(F * np.conj(F), nf, axis=0)[:S].mean(1) / np.arange(S, 0, -1)
    C = ac / ac[0]
    half = S // 2
    c = C[:half]
    gmin = c[1:].min()
    # first local minimum at least half as deep as the global minimum
    lm = None
    for k in range(1, half - 1):
        if c[k] <= c[k - 1] and c[k] <= c[k + 1] and c[k] <= 0.5 * gmin:
            lm = k; break
    if lm is None:
        return {"osc": False}
    hi = min(3 * lm, half - 1)
    kp = lm + 1 + int(np.argmax(c[lm + 1:hi + 1]))
    P = kp * dt
    res = {"c_min": float(c[lm]), "lag_min": lm * dt, "c_peak": float(c[kp]), "period": P}
    res["osc"] = bool(c[lm] <= -0.10 and c[kp] >= 0.10 and P <= (S * dt) / 4)
    k2 = 2 * kp
    res["coh"] = float(C[k2] / C[kp]) if k2 < S else None
    return res


def item_osc():
    lane = [q for q in json.load(open(f"{M2}/decay/osc.json"))["rows"] if q["row"] == "M2_real_r0" and q["signal"] == "total"]
    want = {round(q["g"], 5): q for q in lane}
    out = {}
    for f in decay_files("M2_real_r0"):
        d = np.load(f)
        if "cnt_act" not in d.files:
            continue
        for gi, gv in enumerate(d["g"]):
            k = round(float(gv), 5)
            if k not in want or k < 1.0 or k > 1.4:
                continue
            rl = d["reg_last"][gi, :3, :].max(0)
            alive = rl >= 3000 - DT - 1e-6
            if alive.sum() < 4:
                continue
            x = d["cnt_act"][gi][:, alive].astype(np.float64) / N_ALL
            m = acf_metrics(x)
            q = want[k]
            out[str(k)] = {"mine": m, "n_alive": int(alive.sum()),
                           "lane": {kk: q[kk] for kk in ("osc", "c_min", "c_peak", "period", "coh_ratio", "n_rep")}}
            print("osc", k, int(alive.sum()), m, out[str(k)]["lane"])
    out["flag_rows_real_M2"] = sorted(set(q["row"] for q in json.load(open(f"{M2}/decay/osc.json"))["rows"]
                                          if q["osc"] and q["row"].startswith("M2_real")))
    return out


def item_iso():
    out = {}
    lane = json.load(open(f"{M2}/modules/iso.json"))
    for mod in ("AL", "CB", "VNC", "OL"):
        f = f"{M2}/modules/sim/ISO/M2_r0_{mod}.npz"
        if not os.path.exists(f):
            continue
        L = decay_lines(decay_points([f], want_occ=False))
        out[mod] = {"on": L["on"], "S": L["S"]}
        print("iso", mod, L["on"], L["S"])
    out["lane_snippet"] = str(lane)[:300]
    return out


def item_glin():
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigs
    C = "/MALECNS_CACHE"
    pre = np.load(f"{C}/pre.npy").astype(np.int64); post = np.load(f"{C}/post.npy").astype(np.int64)
    w = np.load(f"{C}/weight.npy").astype(np.float64); sign = np.load(f"{C}/sign.npy")
    N = int(max(pre.max(), post.max()) + 1)
    instr = np.bincount(post, weights=w, minlength=N)
    c = 1.0 / instr[instr > 0].mean()
    exc = sign[pre] > 0
    pdt = 1 - np.exp(-DT)
    def rho_B(g):
        val = (1 - np.exp(-np.tanh(g * c * w[exc]) * DT)) / pdt  # expected onsets per epoch of j
        B = sp.csr_matrix((val, (post[exc], pre[exc])), shape=(N, N))
        return float(np.real(eigs(B, k=1, which="LR", return_eigenvectors=False, tol=1e-9, maxiter=5000)[0]))
    lam = None
    E = sp.csr_matrix((c * w[exc], (post[exc], pre[exc])), shape=(N, N))
    lam = float(np.real(eigs(E, k=1, which="LR", return_eigenvectors=False, tol=1e-9, maxiter=5000)[0]))
    a_eff = pdt / DT
    g0 = a_eff / lam
    # exact (nonlinear f) root of rho(B(g)) = 1 by secant
    ga, gb = g0 * 0.99, g0 * 1.01
    fa, fb = rho_B(ga) - 1, rho_B(gb) - 1
    for _ in range(8):
        gc = gb - fb * (gb - ga) / (fb - fa)
        ga, fa = gb, fb; gb, fb = gc, rho_B(gc) - 1
        if abs(fb) < 1e-10:
            break
    out = {"c": c, "one_over_c": 1 / c, "lambda_PF_cWexc": lam, "alpha_eff": a_eff, "g_lin_linear": g0, "g_lin_exact": gb,
           "residual": fb, "lane": 0.16315}
    print("glin", out)
    return out


def item_theory(dec):
    """theory/sim ratios computed from the theory files and MY simulated lines"""
    mf = json.load(open(f"{M2}/theory/meanfield_predictions.json"))["P3_decay_M2"]
    st = json.load(open(f"{M2}/theory/stochastic_onset.json"))["models"]["M2"]["boundaries"]
    out = {}
    sim_on = {r: dec[f"M2_real_r{r}"]["mine"]["on"] for r in ("0", "6")}
    sim_on["1"] = dec["M2_real_r1"]["mine"]["on"]
    out["MF_equal_size_on_r0"] = mf["0"]["primary"]["on_equal_size"] / sim_on["0"]
    out["stoch_single_on_r0"] = st["single_PF_r0_lt_2D"]["on"] / sim_on["0"]
    out["trend_sim"] = sim_on["6"] / sim_on["0"]
    out["trend_MF"] = mf["6"]["primary"]["on_equal_size"] / mf["0"]["primary"]["on_equal_size"]
    out["trend_stoch"] = st["single_CAL_r6_lt_2D"]["on"] / st["single_PF_r0_lt_2D"]["on"]
    # MF D vs simulated D at all 23 r (sim D recomputed)
    Drat = {}
    for r in mf:
        Dm = mf[r]["primary"].get("D")
        pts = decay_points(decay_files(f"M2_real_r{r}"))
        Ds = decay_lines(pts).get("D")
        Drat[r] = {"MF": Dm, "sim": Ds, "ratio": (Dm / Ds if (Dm and Ds) else None),
                   "agree": (abs(Dm / Ds - 1) <= 0.10 if (Dm and Ds) else False)}
    out["MF_D"] = Drat
    out["MF_D_agree_r"] = [r for r, v in Drat.items() if v["agree"]]
    print("theory", {k: v for k, v in out.items() if k != "MF_D"})
    print("MF D agree at r", out["MF_D_agree_r"], {r: (round(v["ratio"], 3) if v["ratio"] else None) for r, v in Drat.items()})
    return out


def item_spike():
    """ORN active-epoch efficacy at g = 0.05 (driven A1, r 0) vs the exact renewal formula at the measured ORN rate"""
    f = driven_files("A1", "M2_real", "0")[0]
    d = np.load(f); names = list(d["reg_names"]); io = names.index("ORN")
    gi = int(np.argmin(np.abs(d["g"] - 0.05)))
    t = d["t"]; sel = (t > 500) & (t <= 1500)
    xa = float(np.nanmean(d["res_x_act"][0, gi, sel, io]))
    rho = float(d["win_mean"][gi, 0, io].mean())
    U, TAU = 0.25, 157.0
    poff = 1 - np.exp(-DT)
    # measured rho -> per-step on probability q: rho = E[L]/(E[L]+E[Q]) with E[L] = 1/poff, E[Q] = 1/q
    q = 1.0 / ((1 - rho) / rho / poff)  # E[Q] = E[L](1-rho)/rho
    s = np.exp(-DT / TAU)
    gen = lambda p: p * s / (1 - (1 - p) * s)
    M = gen(poff) * gen(q)
    e_exact = (1 - M) / (1 - (1 - U) * M)
    e_pois = 1 / (1 + U * (poff / DT) * rho * TAU)
    h = 0.037926; q_h = 1 - np.exp(-np.tanh(h) * DT); rho_iso = q_h / (q_h + poff - q_h * poff)
    out = {"g": float(d["g"][gi]), "rho_ORN_measured": rho, "x_act_measured": xa, "e_renewal_exact": float(e_exact),
           "e_poisson": float(e_pois), "rel_err": xa / e_exact - 1, "rho_isolated_from_h": float(rho_iso),
           "file": os.path.basename(f)}
    print("spike", out)
    return out


ITEMS = ["decay", "decay_exact", "driven", "driven_extra", "fss", "hyst", "branch", "lifetimes", "osc", "iso", "glin", "theory", "spike", "misc"]

if __name__ == "__main__":
    want = sys.argv[1:] or ITEMS
    path = f"{OUT}/verify.json"
    R = json.load(open(path)) if os.path.exists(path) else {}
    for it in want:
        t0 = time.time()
        if it == "theory":
            if "decay" not in R:
                R["decay"] = item_decay()
            R["theory"] = item_theory(R["decay"])
        else:
            R[it] = globals()[f"item_{it}"]()
        print(f"[{it}] {time.time() - t0:.0f} s", flush=True)
        json.dump(R, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
