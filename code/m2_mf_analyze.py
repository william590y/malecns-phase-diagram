"""m2_mf_analyze.py -- analysis of the M2 mean-field runs (lane m2/theory, meanfield). CPU.
Criteria: results/m2/theory/plan_meanfield.md (pre-registered). Inputs: results/m2/theory/runs/*.npz (code/m2_mf_run.py),
meanfield_glin.json, meanfield_chain.json, the M1 references named in the plan.
Outputs: results/m2/theory/meanfield_points.json (per run and g), meanfield_lines.json (lines per variant/closure/r),
meanfield_validation.json (V1/V2), meanfield_analytic.json (single-unit and slow-fast numbers), and figures
figs/m2/meanfield_*.png. The predictions summary (meanfield_predictions.json) is written by m2_mf_predict.py.
"""
import glob, json, os, sys
import numpy as np

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
import m2_obs as OBS            # noqa: E402
import m2_common as MC          # noqa: E402
import m2_mf_core as M          # noqa: E402

TH = f"{ROOT}/results/m2/theory"
N, N_NON = 167216, 164577
NC_I = 1                         # index of n_c = 1 in N_CUT (0.1, 1, 10)
OSC_REL, OSC_HALF, OSC_PMAX_DECAY, OSC_PMAX_DRV = 1e-3, 0.5, 333.0, 333.0


def fl(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


# ------------------------------------------------------------------------------------------------ trajectory osc rule
def osc_rule(x, pmax):
    """Deterministic oscillation flag on a per-unit trajectory segment x (pre-registered): rel. peak-to-peak > 1e-3,
    amplitude in the second half >= 0.5 x the first half, dominant FFT period <= pmax with >= 3 cycles."""
    x = np.asarray(x, float)
    m = np.abs(x.mean())
    if not np.all(np.isfinite(x)) or m <= 0:
        return {"osc": False, "rel_p2p": None, "period": None, "amp_ratio": None}
    h = len(x) // 2
    a1, a2 = np.ptp(x[:h]), np.ptp(x[h:])
    rel = np.ptp(x) / m
    f = np.abs(np.fft.rfft(x - x.mean())); f[0] = 0
    k = int(f.argmax()); per = len(x) / k if k > 0 else np.inf
    osc = bool(rel > OSC_REL and a2 >= OSC_HALF * a1 and per <= pmax and len(x) / per >= 3)
    return {"osc": osc, "rel_p2p": float(rel), "period": float(per), "amp_ratio": float(a2 / a1) if a1 > 0 else None}


# ------------------------------------------------------------------------------------------------ ring-down modes
def modes(x, t0=5, rel=1e-3, cap=12):
    """Energy-ranked damped modes of a ring-down series (1-unit samples) by the matrix pencil."""
    x = np.asarray(x, float)[t0:]
    if not np.all(np.isfinite(x)) or np.abs(x).max() == 0 or len(x) < 30:
        return None
    x = x / np.abs(x).max()
    K = len(x); Lp = K // 3
    Y = np.array([x[i:i + Lp + 1] for i in range(K - Lp)])
    S = np.linalg.svd(Y, compute_uv=False)
    Mo = int(min(cap, max(1, (S > rel * S[0]).sum())))
    z, c, _ = M.matrix_pencil(x, M=Mo)
    k = np.arange(K)
    with np.errstate(over="ignore", invalid="ignore"):
        E = np.array([np.sum(np.abs(cc * zz ** k) ** 2) for cc, zz in zip(c, z)])
        fit = np.real((c[None, :] * z[None, :] ** k[:, None]).sum(1))
    err = float(np.linalg.norm(fit - x) / np.linalg.norm(x))
    E = np.where(np.isfinite(E), E, 0)
    o = np.argsort(-E)
    out = []
    seen = set()
    for i in o:
        P = 2 * np.pi / abs(np.angle(z[i])) if abs(np.angle(z[i])) > 1e-9 else np.inf
        key = (round(abs(z[i]), 6), round(P, 3))
        if key in seen:
            continue
        seen.add(key)
        share = E[i] / E.sum() * (2 if np.isfinite(P) else 1)          # a conjugate pair counts twice
        out.append({"absz": float(abs(z[i])), "P": float(P), "share": float(min(share, 1.0)),
                    "tau": float(-1 / np.log(abs(z[i]))) if abs(z[i]) < 1 else np.inf})
    return {"modes": out, "fit_err": err, "order": Mo}


def lead_modes(md):
    """leading oscillatory pair (P <= 375, share >= 5%) and slowest real mode (share >= 5%); max |z| overall."""
    if md is None:
        return {}
    osc = [m for m in md["modes"] if np.isfinite(m["P"]) and m["P"] <= 375 and m["share"] >= 0.05]
    real = [m for m in md["modes"] if (not np.isfinite(m["P"]) or m["P"] > 375) and m["share"] >= 0.05]
    res = {"max_absz": max(m["absz"] for m in md["modes"] if m["share"] >= 0.05) if any(m["share"] >= 0.05 for m in md["modes"]) else None,
           "fit_err": md["fit_err"]}
    if osc:
        m = max(osc, key=lambda m: m["share"])
        res.update({"osc_P": m["P"], "osc_absz": m["absz"], "osc_share": m["share"],
                    "osc_DP": float(m["absz"] ** m["P"]) if m["absz"] < 1 else float("inf")})
    if real:
        m = max(real, key=lambda m: m["absz"])
        res.update({"slow_tau": m["tau"], "slow_absz": m["absz"], "slow_share": m["share"]})
    return res


# ------------------------------------------------------------------------------------------------ per run
def analyse_run(path):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    R = [str(x) for x in d["reg_names"]]; ri = {k: i for i, k in enumerate(R)}
    G, ics, col_ic, col_g = d["G"], [str(x) for x in d["ics"]], d["col_ic"], d["col_g"]
    rec = d["rec"]                                   # [T_units, R, C] expected counts
    nC = len(col_g); nG = len(G)
    pts = []
    for j in range(nC):
        ic = ics[col_ic[j]]
        p = {"g": float(col_g[j]), "ic": ic}
        if meta["proto"] == "decay":
            tau = {str(nc): fl(d["cut_all"][k, j]) for k, nc in enumerate(d["n_cut"])}
            p["tau_cut"] = tau
            p["tau_MF"] = tau["1.0"]                 # None = never below 1 (sustained)
            p["nmin"] = float(d["nmin"][j])
            wq = d["win_qs"][:, j]
            p["rho_qs"] = float(wq[ri["all"]]); p["n_qs"] = float(wq[ri["all"]] * N)
            p["rho_qs_reg"] = {k: float(wq[ri[k]]) for k in R}
            occ = d["occ_reg"][:, j]
            p["PR_N"] = float(d["PR_all"][j] / N); p["PR"] = float(d["PR_all"][j])
            p["share"] = {k: (float(occ[ri[k]] / occ[ri["all"]]) if occ[ri["all"]] > 0 else None) for k in R if k != "all"}
            p["eff_act"] = {k: fl(d["eff_act_reg"][ri[k], j]) for k in ("all", "AL_module", "nonORN")}
            p["xnext"] = {k: float(d["xnext_reg"][ri[k], j]) for k in ("all", "AL_module", "dep")}
            seg = rec[2000:3000, ri["all"], j]
            p["osc"] = osc_rule(seg, OSC_PMAX_DECAY)
            last = rec[2500:3000, ri["all"], j]
            p["stationary"] = bool(np.ptp(last) / max(abs(last.mean()), 1e-300) < 1e-3)
            p["n_T"] = float(rec[-1, ri["all"], j])
        else:
            wd = d["win_d"][:, j]
            p["rho_ORN"] = float(wd[ri["ORN"]]); p["rho_drv"] = float(wd[ri["nonORN"]])
            p["rho_uPN"] = float(wd[ri["uPN_ant"]]); p["rho_AL_module"] = float(wd[ri["AL_module"]])
            p["G"] = float(wd[ri["uPN_ant"]] / wd[ri["ORN"]]) if wd[ri["ORN"]] > 0 else None
            p["PR_nonORN_frac"] = float(d["PR_nonORN"][j] / N_NON)
            m1, m2 = d["win_d1"][ri["nonORN"], j], d["win_d2"][ri["nonORN"], j]
            p["NS"] = bool(abs(m1 - m2) > 0.1 * max((m1 + m2) / 2, 1e-300))
            rm = d["win_rm"][ri["uPN_ant"], j]
            p["S1"] = bool(rm <= 0.1 * wd[ri["uPN_ant"]])
            tp = fl(d["cut_nonORN"][NC_I, j]); p["tau_post"] = tp
            nal_T = float(rec[-1, ri["ALsys"], j])
            p["n_ALsys_T"] = nal_T
            p["n_nonORN_T"] = float(rec[-1, ri["nonORN"], j])
            p["S2"] = bool((tp is not None and tp < 3000) or nal_T < 0.05)
            if tp is not None and tp < 200 and p["S1"] and p["S2"]:
                rc = "silent"
            elif tp is None or tp >= 3000:
                rc = "self-sustained"
            else:
                rc = "transient"
            p["removal"] = rc
            seg = rec[500:1500, ri["nonORN"], j]
            p["osc"] = osc_rule(seg, OSC_PMAX_DRV)
            if rc == "self-sustained":
                ph = "SS-D" if p["PR_nonORN_frac"] >= 0.25 else "SS-L"
            elif rc == "transient":
                ph = "DT"
            else:
                ph = "DA" if (p["G"] or 0) >= 0.29 else "DQ"
            mods = ("-O" if p["osc"]["osc"] else "") + ("-D" if (p["PR_nonORN_frac"] >= 0.25 and not ph.startswith("SS")) else "") + ("-NS" if p["NS"] else "")
            p["phase"] = ph + mods
            p["allowed"] = bool(p["G"] is not None and 0.29 <= p["G"] <= 1.45 and p["S1"] and p["S2"])
            p["n_drv"] = float(rec[1000:1500, ri["nonORN"], j].mean())
            p["xnext"] = {k: float(d["xnext_reg"][ri[k], j]) for k in ("ORN", "AL_module", "dep")}
            p["tau_post_cut"] = {str(nc): fl(d["cut_nonORN"][k, j]) for k, nc in enumerate(d["n_cut"])}
        pts.append(p)
    # ring-down + sigma for the protocol-IC columns
    if "sigma_x" in d.files:
        main_cols = np.where(col_ic == 0)[0]
        reg = ri["all"] if meta["proto"] == "decay" else ri["nonORN"]
        for jj, j in enumerate(main_cols):
            p = pts[j]
            p["sigma_x"] = float(d["sigma_x"][jj]); p["sigma_1"] = float(d["sigma_1"][jj])
            p["ring_resid"] = float(d["ring_resid"][jj])
            lm = {}
            for kn in ("demog", "unif"):
                tr, ls = d[f"ring_{kn}"], d[f"ring_{kn}_logscale"]
                x = tr[:, reg, jj] * np.exp(ls[:, jj] - ls[0, jj])
                lm[kn] = lead_modes(modes(x))
                xa = tr[:, ri["AL_module"], jj] * np.exp(ls[:, jj] - ls[0, jj])
                lm[kn + "_AL"] = lead_modes(modes(xa))
            p["ring"] = lm
            a_, b_ = lm["demog"].get("osc_P"), lm["unif"].get("osc_P")
            p["ring_robust_P"] = bool(a_ and b_ and abs(a_ - b_) / max(a_, b_) <= 0.1)
    return meta, pts


# ------------------------------------------------------------------------------------------------ lines
def first_cross(g, y, level, increasing=True, iso=True):
    g = np.asarray(g, float); y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() == 0:
        return None
    if iso:
        return OBS.first_crossing(g[ok], y[ok], level, increasing=increasing)
    g, y = g[ok], y[ok]
    hit = np.where(y >= level)[0] if increasing else np.where(y <= level)[0]
    if len(hit) == 0:
        return None
    k = hit[0]
    if k == 0:
        return float(g[0])
    f = (level - y[k - 1]) / (y[k] - y[k - 1])
    return float(np.exp(np.log(g[k - 1]) + f * np.log(g[k] / g[k - 1])))


def intervals(g, flags):
    """[g_first, g_last] of contiguous True runs, with the bracketing grid neighbours."""
    out, cur = [], None
    for i, f in enumerate(flags):
        if f and cur is None:
            cur = i
        if (not f or i == len(flags) - 1) and cur is not None:
            e = i if f else i - 1
            out.append({"from": float(g[cur]), "to": float(g[e]), "below": float(g[cur - 1]) if cur > 0 else None,
                        "above": float(g[e + 1]) if e + 1 < len(g) else None})
            cur = None
    return out


def lines_decay(pts):
    P = [p for p in pts if p["ic"] == "decay"]
    P.sort(key=lambda p: p["g"])
    g = np.array([p["g"] for p in P])
    out = {}
    for nc in ("0.1", "1.0", "10.0"):
        lt = np.array([np.log(p["tau_cut"][nc]) if p["tau_cut"][nc] is not None else np.log(1e7) for p in P])
        out[f"S_nc{nc}"] = first_cross(g, lt, np.log(200.0)); out[f"on_nc{nc}"] = first_cross(g, lt, np.log(3000.0))
    out["S"], out["on"] = out["S_nc1.0"], out["on_nc1.0"]
    sus = np.array([p["tau_MF"] is None for p in P])
    pr = np.array([p["PR_N"] if s else 0.0 for p, s in zip(P, sus)])
    out["D"] = first_cross(g, pr, 0.25)
    osc = np.array([p["osc"]["osc"] and s for p, s in zip(P, sus)])
    out["hopf_first_g"] = float(g[osc][0]) if osc.any() else None
    out["hopf_bracket"] = [float(g[np.argmax(osc) - 1]), float(g[np.argmax(osc)])] if osc.any() and np.argmax(osc) > 0 else None
    out["osc_intervals"] = intervals(g, osc)
    out["osc_periods"] = {f"{p['g']:g}": p["osc"]["period"] for p, o in zip(P, osc) if o}
    # bistability: decay vs eps IC
    E = {p["g"]: p for p in pts if p["ic"] == "eps"}
    bi = []
    for p in P:
        q = E.get(p["g"])
        if q is None:
            bi.append(False); continue
        na, nb = p["n_T"], q["n_T"]
        bi.append(bool(p["stationary"] and q["stationary"] and max(na, nb) >= 1 and abs(na - nb) / max(na, nb) > 0.1))
    out["bistable_intervals"] = intervals(g, np.array(bi)) if E else None
    out["n_bistable"] = int(np.sum(bi)) if E else None
    out["nonstationary_g"] = [float(p["g"]) for p, s in zip(P, sus) if s and not p["stationary"]]
    return out


def lines_driven(pts):
    P = [p for p in pts if p["ic"] == "quiet"]
    P.sort(key=lambda p: p["g"])
    g = np.array([p["g"] for p in P])
    out = {}
    Gv = np.array([p["G"] if p["G"] is not None else np.nan for p in P])
    out["DQ_DA_G029"] = first_cross(g, Gv, 0.29, iso=False)
    out["G_145"] = first_cross(g, Gv, 1.45, iso=False)
    out["G_max"] = float(np.nanmax(Gv)); out["g_at_G_max"] = float(g[np.nanargmax(Gv)])
    rc = [p["removal"] for p in P]
    ns = [i for i, x in enumerate(rc) if x != "silent"]
    out["DA_DT_first_nonsilent_g"] = float(g[ns[0]]) if ns else None
    out["DA_DT_bracket"] = [float(g[ns[0] - 1]), float(g[ns[0]])] if ns and ns[0] > 0 else None
    ss = [i for i, x in enumerate(rc) if x == "self-sustained"]
    out["DT_SS_first_g"] = float(g[ss[0]]) if ss else None
    out["DT_SS_bracket"] = [float(g[ss[0] - 1]), float(g[ss[0]])] if ss and ss[0] > 0 else None
    lt = np.array([np.log(p["tau_post"]) if p["tau_post"] is not None else np.log(1e7) for p in P])
    out["tau_post_200"] = first_cross(g, lt, np.log(200.0)); out["tau_post_3000"] = first_cross(g, lt, np.log(3000.0))
    al = np.array([p["allowed"] for p in P])
    out["allowed_intervals"] = intervals(g, al)
    osc = np.array([p["osc"]["osc"] for p in P])
    out["osc_intervals"] = intervals(g, osc)
    out["phases"] = {f"{p['g']:g}": p["phase"] for p in P}
    out["rho_ORN"] = {f"{p['g']:g}": p["rho_ORN"] for p in P}
    out["G"] = {f"{p['g']:g}": p["G"] for p in P}
    D = {p["g"]: p for p in pts if p["ic"] == "decay"}
    if D:
        bi = []
        for p in P:
            q = D[p["g"]]
            na, nb = p["n_drv"], q["n_drv"]
            bi.append(bool(not p["NS"] and not q["NS"] and max(na, nb) >= 1 and abs(na - nb) / max(na, nb) > 0.1))
        out["bistable_intervals"] = intervals(g, np.array(bi)); out["n_bistable"] = int(np.sum(bi))
    return out


# ------------------------------------------------------------------------------------------------ analytic numbers
def analytic():
    out = {"alpha_eff": M.ALPHA, "variants": {}}
    for name, (U, TAU) in {"M2": (0.25, 157), "M2_KW09": (0.28, 240), "M2_N15": (0.22, 89), "M2_tau30": (0.25, 30)}.items():
        rho_dep = 1 / (U * TAU * M.ALPHA)
        e_tab = {f"{r:g}": float(M.eff_exact(M.p_of_rho(r), U, TAU)) for r in (0.001, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.5)}
        phi_max = 0.5 * M.eff_exact(M.P_OFF, U, TAU)
        coh_max = float(np.exp(-2 * np.pi / np.sqrt(M.ALPHA * TAU)))
        # slow-fast reduction (c = 0): period and per-period damping vs sigma0
        sf = {}
        for s0 in (1.2, 1.5, 2.0, 3.0, 5.0, 10.0):
            w0 = M.ALPHA * (s0 - 1) / TAU; gam = s0 / (2 * TAU)
            if w0 > gam ** 2:
                w = np.sqrt(w0 - gam ** 2); Pp = 2 * np.pi / w
                sf[f"{s0:g}"] = {"period": float(Pp), "damping_per_period": float(np.exp(-gam * Pp)), "n_star_over_rho_dep": s0 - 1}
        out["variants"][name] = {"U": U, "TAU": TAU, "rho_dep": rho_dep, "eff_exact": e_tab, "phi_max": float(phi_max),
                                 "eff_poisson_at_0.04": float(M.eff_poisson(0.04, U, TAU)),
                                 "coherence_bound_exp(-2pi/sqrt(alpha TAU))": coh_max, "slow_fast": sf}
    out["TAU_needed_for_osc_strong_0.7"] = float(4 * np.pi ** 2 / (M.ALPHA * np.log(1 / 0.7) ** 2))
    # crash after the fully active start (one-mode estimate, derivation section 6)
    crash = {}
    for xc in (0.05, 0.1, 0.2):
        row = {}
        for s0 in (1.5, 2, 3, 4, 5, 7, 10):
            if s0 * xc < 1:
                Bv = 1 - s0 * xc - (s0 - 1) * np.log(s0 * (1 - xc) / (s0 - 1))
                tm = 157 * np.log(s0 * (1 - xc) / (s0 - 1))
                row[f"{s0:g}"] = {"alphaTAU_B": float(M.ALPHA * 157 * Bv), "t_min": float(tm)}
        crash[f"xc{xc:g}"] = row
    out["crash_M2_lnN0_over_nmin"] = crash
    return out


if __name__ == "__main__":
    runs = sorted(glob.glob(f"{TH}/runs/*.npz"))
    runs = [p for p in runs if not os.path.basename(p).startswith("smoke")]
    allpts, lines = {}, {}
    for p in runs:
        name = os.path.basename(p)[:-4]
        try:
            meta, pts = analyse_run(p)
        except Exception as e:                                   # incomplete file etc.
            print("skip", name, e); continue
        allpts[name] = {"meta": meta, "points": pts}
        lines[name] = lines_decay(pts) if meta["proto"] == "decay" else lines_driven(pts)
    json.dump(allpts, open(f"{TH}/meanfield_points.json", "w"), default=float)
    json.dump(lines, open(f"{TH}/meanfield_lines.json", "w"), indent=1, default=float)
    json.dump(analytic(), open(f"{TH}/meanfield_analytic.json", "w"), indent=1, default=float)
    print(len(allpts), "runs analysed")
