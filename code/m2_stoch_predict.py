"""m2_stoch_predict.py -- assemble the prediction files of lane m2/theory-stochastic from its own raw outputs
(results/m2/theory/stochastic/**). CPU only. Reads NO experiment-lane file.

Writes results/m2/theory/stochastic_{onset,modes,driven,qs,aval,fss}.json and prints a summary.
Lifetime variants for setting (a) (section 4 of stochastic.md):
  single  : renormalised Perron mode of the frozen operator at the bQMF state (pre-registered primary; chains from the GPU run)
  pooled  : (added after seeing my own output: the renormalised Perron mode is localised while the activity is broad)
            the whole active state as PR identical units with mean activity n_tot/PR and efficacy e_mean_act;
            sigma_p = 1/(1 - n_tot/PR), lam_rest = alpha sigma_p / X_p; frozen / adiabatic / 2-D (when small enough) and
            the Gaussian 2-D estimate.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/MALECNS_ROOT/code")
import m2_stoch_core as S

R = f"{S.ROOT}/results/m2/theory/stochastic"
OUT = f"{S.ROOT}/results/m2/theory"
R_CORE = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0)
A = S.A_EFF


def J(p):
    return json.load(open(p))


def gauss2d(n, sig, y, U, TAU, Ny, K):
    """Gaussian (LNA) lifetime of the 2-D chain about (n*, y*): ln tau ~ n*^2/(2 Var n) + ln prefactor (OU first passage)."""
    J11 = -A * (sig - 1)
    J12 = A * n / y
    J21 = -U * y * A * (1 - 2 * n / K) / ((1 - n / K) * Ny)
    J22 = -1 / TAU - 2 * U * A * n / Ny
    tr = J11 + J22; det = J11 * J22 - J12 * J21
    D = A * n
    var = D * (J22 ** 2 + det) / (abs(tr) * det)
    a2 = n * n / var
    kap = abs(tr) / 2
    lt = a2 / 2 + np.log(np.sqrt(2 * np.pi) / (np.sqrt(a2) * kap)) if a2 >= 4 else np.nan   # valid only for a >= 2 sd
    om2 = det - tr ** 2 / 4
    P = 2 * np.pi / np.sqrt(om2) if om2 > 0 else np.nan
    return {"lt_G": float(lt), "var_n": float(var), "fano_n": float(var / n), "period": float(P),
            "coh": float(np.exp(-kap * P)) if np.isfinite(P) else np.nan, "tr": float(tr), "det": float(det)}


def pooled(row, U, TAU, c=1.0, do2d=True):
    n = c * row["rho_tot"]; PR = c * row["PR"]
    if not (n > 0 and PR > n):
        return None
    X = row.get("e_mean_act", 1.0)
    sig = 1 / (1 - n / PR)
    out = {"n_star": n, "K": PR, "sigma": sig, "X": X}
    if U > 0 and X < 1:
        lam_r = A * sig / X
        Ny = U * TAU * A * X * n / (1 - X)
        par = {"lam": A * sig, "lam_rest": lam_r, "X": X, "n_star": n, "K": PR}
        big = n > 350
        lt = S.lifetimes(par, U, TAU, do2d=do2d and not big, cap2d=900000)
        out.update({k: lt[k] for k in ("lt_F", "lt_A", "lt_2D", "Ny") if k in lt})
        out.update(gauss2d(n, sig, X, U, TAU, Ny, PR))
    else:
        out["lt_F"] = S.log_tau_logistic_qs(lambda m: A * sig * np.ones_like(m), PR, n) if PR < 3e5 else np.inf
        out["lt_A"] = out["lt_2D"] = out["lt_F"]
    return out


def crossing_table(g, series):
    out = {}
    for k, y in series.items():
        y = np.array([(-np.inf if (v is None or (isinstance(v, float) and np.isnan(v))) else v) for v in y], float)
        out[k] = {"S": S.cross(g, y, 200.0), "on": S.cross(g, y, 3000.0)}
    return out


def at_g(rows, g0, key):
    gs = np.array([r["g"] for r in rows]); vals = np.array([r.get(key, np.nan) for r in rows], float)
    o = np.argsort(gs)
    if g0 is None or not np.isfinite(g0):
        return None
    return float(np.exp(np.interp(np.log(g0), np.log(gs[o]), np.log(np.maximum(vals[o], 1e-300))))) if np.all(vals[o] > 0) else float(np.interp(g0, gs[o], vals[o]))


def decay_summary(path, U, TAU, with_pooled=True, rlist=R_CORE):
    d = J(path)
    rows = [r for r in d["rows"] if not r.get("silent")]
    rows.sort(key=lambda r: r["g"])
    g = np.array([r["g"] for r in rows])
    res = {"file": os.path.relpath(path, S.ROOT), "model": d["model"], "closure": d["closure"], "N": d["N"],
           "g_range": [float(g.min()), float(g.max())], "n_g": len(g), "c_r": d["c_r"]}
    ser = {}
    # single renormalised mode: recomputed here from the saved mode parameters (same chains as the GPU run, Richardson-
    # extrapolated 2-D solver)
    for r in rows:
        r["_single"] = {}
        sig = r.get("mode_sigma", 0); ns = r.get("n_star", 0)
        if not (sig > 1 and ns > 0):
            continue
        Kp = ns / (1 - 1 / sig)
        for rr in rlist:
            cr = d["c_r"][str(rr)] if isinstance(list(d["c_r"].keys())[0], str) else d["c_r"][rr]
            for name, fac in (("PF", 1.0), ("CAL", cr)):
                if name == "PF" and rr != rlist[0]:
                    continue
                par = {"lam": r["mode_lam"], "lam_rest": r["mode_lam_rest"], "X": r["mode_X"], "n_star": fac * ns, "K": fac * Kp}
                r["_single"][f"{name}_r{rr:g}"] = S.lifetimes(par, U, TAU, do2d=True, cap2d=900000)
    for vk in sorted({k for r in rows for k in r["_single"]}):
        for lt in ("lt_F", "lt_A", "lt_2D"):
            ser[f"single_{vk}_{lt}"] = [r["_single"].get(vk, {}).get(lt, -np.inf) for r in rows]
    pooled_rows = []
    if with_pooled:
        for r in rows:
            pr = {}
            for rr in rlist:
                c = 1.0 if rr == 0 and False else (d["c_r"][str(rr)] if isinstance(list(d["c_r"].keys())[0], str) else d["c_r"][rr])
                for name, fac in (("PF", 1.0), ("CAL", c)):
                    if name == "PF" and rr != rlist[0]:
                        continue
                    want2d = (name == "PF") or (rr in (1.0, 4.0) and len(rlist) > 3)
                    pp = pooled(r, U, TAU, fac, do2d=want2d)
                    pr[f"{name}_r{rr:g}"] = pp
            pooled_rows.append(pr)
            for vk, pp in pr.items():
                for lt in ("lt_F", "lt_A", "lt_2D", "lt_G"):
                    ser.setdefault(f"pooled_{vk}_{lt}", []).append(None if pp is None else pp.get(lt, None))
    res["cross"] = crossing_table(g, ser)
    res["profile"] = [{k: r.get(k) for k in ("g", "rho_tot", "rho_mean", "PR", "share_AL_module", "share_ORN", "share_CB",
                                             "share_VNC", "share_OL", "share_DN", "e_mean_act", "e_AL", "mode_sigma",
                                             "mode_X", "mode_K0", "mode_PR_v", "mode_AL_v", "n_star", "K_pf")}
                      for r in rows]
    tr = [{"g": r["g"], **{k: r["transient"][k] for k in ("n_min", "t_min", "n_end", "n_min_over_nstar")}}
          for r in rows if "transient" in r]
    res["transient"] = tr
    res["_rows"] = rows; res["_pooled"] = pooled_rows
    return res


def peel_summary(path, U=0.25, TAU=157.0, rlist=(0.0, 1.0, 4.0, 6.0)):
    """Islands (successive renormalised modes on disjoint supports): per-island 2-D chain; the longest-lived island bounds
    the network lifetime from below (islands that do not interact), so its tau = 3000 crossing is an UPPER bound on the onset."""
    d = J(path)
    rows = sorted([r for r in d["rows"] if "islands" in r], key=lambda r: r["g"])
    g = np.array([r["g"] for r in rows])
    ser = {}; isl_tab = []
    for r in rows:
        best = {}
        for isl in r["islands"]:
            sig, ns = isl["sigma"], isl["n_mode"]
            rec = {"k": isl["k"], "sigma": sig, "K0": isl["K0"], "n_mode": ns, "AL_v": isl["AL_v"], "PR_v": isl["PR_v"],
                   "top": isl["top_superclass"], "X": isl["X"], "lam_rest_over_alpha": isl["lam_rest"] / A}
            if sig > 1 and ns > 0:
                Kp = ns / (1 - 1 / sig)
                for rr in rlist:
                    cr = d["c_r"][str(rr)] if isinstance(list(d["c_r"].keys())[0], str) else d["c_r"][rr]
                    fac = 1.0 if rr == 0 else cr
                    nm = "PF_r0" if rr == 0 else f"CAL_r{rr:g}"
                    lt = S.lifetimes({"lam": isl["lam"], "lam_rest": isl["lam_rest"], "X": isl["X"], "n_star": fac * ns,
                                      "K": fac * Kp}, U, TAU, do2d=True, cap2d=900000)
                    v = lt.get("lt_2D")
                    if v is None or not np.isfinite(v):
                        v = lt.get("lt_F", -np.inf) if lt.get("lt_F", -np.inf) > np.log(1e6) else (-np.inf if lt.get("lt_A", -np.inf) < np.log(20) else np.nan)
                    rec[f"lt2D_{nm}"] = v
                    if v is not None and np.isfinite(v):
                        best[nm] = max(best.get(nm, -np.inf), v)
            isl_tab.append({"g": r["g"], **rec})
        for nm in ("PF_r0",) + tuple(f"CAL_r{x:g}" for x in rlist if x != 0):
            ser.setdefault(f"islands_max_{nm}", []).append(best.get(nm, -np.inf))
    return {"file": os.path.relpath(path, S.ROOT), "cross": crossing_table(g, ser), "islands": isl_tab}


def driven_summary(model, orn):
    ps = sorted(glob.glob(f"{R}/driven/{model}_r0_orn{orn:g}_part*.json"))
    rows = []
    for p in ps:
        rows += J(p)["up"]
    rows.sort(key=lambda r: r["g"])
    if not rows:
        return None
    keep = []
    for r in rows:
        a = r.get("acf_nonORN", {}); b = r.get("acf_AL", {})
        keep.append({"g": r["g"], "rho_nonORN": r["rho_nonORN"], "rho_ORN": r["rho_ORN"], "rho_uPN": r["rho_uPN_ant"],
                     "rho_AL": r["rho_AL_module"], "G_gain": r["G_gain"], "PR": r.get("PR"),
                     "share_AL": r.get("share_AL_module"),
                     "chi_nonORN": r.get("chi_nonORN"), "fano_nonORN": r.get("fano_nonORN"),
                     "chi_AL": r.get("chi_AL"), "fano_AL": r.get("fano_AL"),
                     "tau_int": a.get("tau_int"), "tau_e": a.get("tau_e"), "period": a.get("period"),
                     "c_min": a.get("c_min"), "c_peak": a.get("c_peak"), "coh": a.get("coh_ratio"),
                     "osc": a.get("osc"), "osc_strong": a.get("osc_strong"),
                     "tau_int_AL": b.get("tau_int"), "period_AL": b.get("period"), "osc_AL": b.get("osc"),
                     "coh_AL": b.get("coh_ratio"),
                     "m_step": r.get("mr_m_step"), "m_unit": r.get("mr_m_unit"), "lna_converged": r.get("lna_conv"),
                     "lna_S": r.get("lna_S")})
    g = np.array([k["g"] for k in keep])

    def argmax_interior(key):
        y = np.array([np.nan if k[key] is None else k[key] for k in keep], float)
        if not np.isfinite(y).any():
            return None
        i = int(np.nanargmax(y))
        return {"g": float(g[i]), "value": float(y[i]), "interior": bool(0 < i < len(y) - 1)}
    summ = {"n_points": len(keep), "g_range": [float(g.min()), float(g.max())],
            "peak_fano_AL": argmax_interior("fano_AL"), "peak_chi_nonORN": argmax_interior("chi_nonORN"),
            "peak_tau_int": argmax_interior("tau_int"), "peak_fano_nonORN": argmax_interior("fano_nonORN"),
            "osc_g": [k["g"] for k in keep if k["osc"]], "osc_strong_g": [k["g"] for k in keep if k["osc_strong"]],
            "osc_periods": [k["period"] for k in keep if k["osc"]],
            "m_unit_max": float(np.nanmax([k["m_unit"] for k in keep if k["m_unit"] is not None])),
            "m_step_max": float(np.nanmax([k["m_step"] for k in keep if k["m_step"] is not None])),
            "g_m_step_within_0.01": [k["g"] for k in keep if k["m_step"] is not None and abs(1 - k["m_step"]) < 0.01],
            "g_m_unit_within_0.01": [k["g"] for k in keep if k["m_unit"] is not None and abs(1 - k["m_unit"]) < 0.01],
            "gain_window_G_0.29_1.45": [k["g"] for k in keep if 0.29 <= k["G_gain"] <= 1.45]}
    Gg = np.array([k["G_gain"] for k in keep])
    def gcross(level):
        i = np.where(Gg >= level)[0]
        if len(i) == 0 or i[0] == 0:
            return None
        i = i[0]; f_ = (np.log(level) - np.log(Gg[i - 1])) / (np.log(Gg[i]) - np.log(Gg[i - 1]))
        return float(np.exp(np.log(g[i - 1]) + f_ * np.log(g[i] / g[i - 1])))
    summ["g_at_G_0.29"] = gcross(0.29); summ["g_at_G_1.45"] = gcross(1.45)
    return {"model": model, "orn": orn, "files": [os.path.relpath(p, S.ROOT) for p in ps], "rows": keep, "summary": summ}


def lnaqs_summary(model):
    ps = sorted(glob.glob(f"{R}/lnaqs/{model}_r0_*.json"))
    rows = []
    for p in ps:
        rows += J(p)["rows"]
    rows = [r for r in rows if "var_all" in r]
    rows.sort(key=lambda r: r["g"])
    out = []
    for r in rows:
        a = r["acf_all"]; n_tot = r["rho_tot"]; N = n_tot / r["rho_mean"]
        varc = r["var_all"] * N * N
        kap = (-np.log(a["coh_ratio"]) / a["period"]) if (a.get("osc") and a.get("coh_ratio") and a["coh_ratio"] > 0) else \
            (1 / a["tau_e"] if np.isfinite(a.get("tau_e", np.inf)) and a["tau_e"] > 0 else 1 / max(a["tau_int"], 1e-3))
        a2 = n_tot ** 2 / varc
        lt = a2 / 2 + np.log(np.sqrt(2 * np.pi) / (np.sqrt(a2) * kap)) if a2 >= 4 else np.nan   # valid only for a >= 2 sd
        out.append({"g": r["g"], "rho_tot": n_tot, "PR": r.get("PR"), "share_AL": r.get("share_AL_module"),
                    "cv_all": r["cv_all"], "fano_count": varc / n_tot, "lt_LNA_gauss": float(lt),
                    "period": a["period"], "coh": a["coh_ratio"], "osc": a["osc"], "osc_strong": a["osc_strong"],
                    "c_min": a["c_min"], "c_peak": a["c_peak"], "tau_int": a["tau_int"],
                    "period_AL": r["acf_AL"]["period"], "osc_AL": r["acf_AL"]["osc"], "coh_AL": r["acf_AL"]["coh_ratio"],
                    "osc_nonORN": r["acf_nonORN"]["osc"], "period_nonORN": r["acf_nonORN"]["period"],
                    "lna_converged": r.get("lna_conv")})
    res = {"model": model, "files": [os.path.relpath(p, S.ROOT) for p in ps], "rows": out}
    ok = [x for x in out if np.isfinite(x["lt_LNA_gauss"])]
    if len(ok) >= 2:
        g = np.array([x["g"] for x in ok]); y = np.array([x["lt_LNA_gauss"] for x in ok])
        res["cross_LNA_gauss"] = {"S": S.cross(g, y, 200.0), "on": S.cross(g, y, 3000.0),
                                  "valid_from_g": float(g.min()),
                                  "note": "Gaussian estimate valid only where the QS mean is >= 2 SD above 0 (g >= valid_from_g); "
                                          "a crossing at valid_from_g means 'at or below'"}
        # prefactor sensitivity: kappa = 1/tau_int instead
        y2 = np.array([x["cv_all"] ** -2 / 2 + np.log(np.sqrt(2 * np.pi) * x["cv_all"] * x["tau_int"]) for x in ok])
        res["cross_LNA_gauss_kappa_tauint"] = {"S": S.cross(g, y2, 200.0), "on": S.cross(g, y2, 3000.0)}
    return res


def pulse_summary(path, U):
    d = J(path)
    out = {}
    for gap in sorted({r["gap"] for r in d["rows"]}):
        rows = sorted([r for r in d["rows"] if r["gap"] == gap], key=lambda r: r["g"])
        crit = [r["g"] for r in rows if min(r["S_eps"], r["S_dep"]) >= 1000 and r["eps"] > 0]
        span = (max(crit) / min(crit)) if crit else 0.0
        burst = [r["g"] for r in rows if r["eps"] <= 0]
        out[f"gap{gap:g}"] = {"rows": [{k: r.get(k) for k in ("g", "m", "eps", "S_eps", "S_dep", "K_s", "K0", "mean_size",
                                                               "T_av", "share_AL_F", "x_AL", "x_min", "PR_v", "AL_v", "method")}
                                       for r in rows],
                              "critical_like_g": crit, "critical_like_span": span,
                              "no_subcritical_quasistatic_state_g": burst}
    return {"file": os.path.relpath(path, S.ROOT), "model": d["model"], "by_gap": out}


def main():
    out = {"g_lin": 0.16315274988719286, "kappa_M2": S.kappa(0.25, 157.0), "K_shrink_factor_M2": 1 + S.kappa(0.25, 157.0)}
    models = {"M2": (0.25, 157.0), "M2_KW09": (0.28, 240.0), "M2_N15": (0.22, 89.0), "M2_tau30": (0.25, 30.0), "M1": (0.0, np.inf)}
    dec = {}
    for m, (U, TAU) in models.items():
        p = f"{R}/decay/bqmf_{m}.json"
        if os.path.exists(p):
            dec[m] = decay_summary(p, U, TAU)
    for m in ("M2", "M1"):
        p = f"{R}/decay/qmf_{m}_r0.json"
        if os.path.exists(p):
            dec[f"{m}_qmf_r0"] = decay_summary(p, *models[m], rlist=(0.0,))
    for p in sorted(glob.glob(f"{R}/decay/bqmf_M*_*N*_s*.json")) + sorted(glob.glob(f"{R}/decay/bqmf_M*_mod_*.json")):
        nm = os.path.basename(p)[5:-5]
        m = nm.split("_")[0]
        dec[nm] = decay_summary(p, *models[m], rlist=(0.0, 1.0, 4.0))
    out["decay"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in dec.items()}
    out["driven"] = {}
    for m, o in (("M2", 0.04), ("M2", 0.01), ("M2", 0.02), ("M2", 0.07), ("M2", 0.1), ("M1", 0.04), ("M2_tau30", 0.04)):
        ds = driven_summary(m, o)
        if ds:
            out["driven"][f"{m}_orn{o:g}"] = ds
    out["lnaqs"] = {m: lnaqs_summary(m) for m in ("M2", "M1", "M2_tau30", "M2_KW09")}
    pp = f"{R}/decay/bqmf_M2_peel.json"
    out["peel"] = peel_summary(pp) if os.path.exists(pp) else None
    out["pulse"] = {}
    for p in sorted(glob.glob(f"{R}/pulse/*.json")):
        if "smoke" in p:
            continue
        nm = os.path.basename(p)[:-5]
        out["pulse"][nm] = pulse_summary(p, 0.25 if nm.startswith("M2") else 0.0)
    return out, dec
def onset_record(dsum, variant_keys):
    """S / on for the named crossing keys + profile at the predicted onset."""
    rec = {}
    for vk in variant_keys:
        c = dsum["cross"].get(vk)
        if c is None:
            continue
        rec[vk] = dict(c)
        on = c.get("on")
        if on:
            prof = dsum["profile"]
            gs = np.array([p["g"] for p in prof])
            def itp(key):
                y = np.array([p.get(key) if p.get(key) is not None else np.nan for p in prof], float)
                ok = np.isfinite(y)
                return float(np.interp(np.log(on), np.log(gs[ok]), y[ok])) if ok.sum() > 1 else None
            rec[vk]["at_onset"] = {k: itp(k) for k in ("rho_tot", "PR", "share_AL_module", "share_CB", "share_VNC", "share_OL",
                                                       "share_DN", "e_mean_act", "e_AL", "mode_sigma", "mode_X", "mode_K0",
                                                       "mode_AL_v", "n_star")}
            if c.get("S"):
                rec[vk]["on_over_S"] = on / c["S"]
                rec[vk]["W"] = float(np.log(on / c["S"]))
    return rec


def write_predictions():
    out, dec = main()
    md = {}
    keys_main = ["single_PF_r0_lt_2D", "single_PF_r0_lt_F", "single_PF_r0_lt_A"] + \
        [f"single_CAL_r{r:g}_lt_{x}" for r in R_CORE for x in ("2D", "F", "A")] + \
        ["pooled_PF_r0_lt_2D", "pooled_PF_r0_lt_F", "pooled_PF_r0_lt_A", "pooled_PF_r0_lt_G"] + \
        [f"pooled_CAL_r{r:g}_lt_{x}" for r in R_CORE for x in ("2D", "F", "A", "G")]
    onset = {"g_lin": out["g_lin"], "kappa_M2": out["kappa_M2"], "K_shrink_factor_M2": out["K_shrink_factor_M2"],
             "notes": "single = renormalised Perron mode of the frozen operator at the bQMF state (pre-registered primary, 2D chain); "
                      "pooled = whole active state as PR identical units (post-hoc variant, upper bound on lifetime). "
                      "PF = no calibration (inhibition neglected, r-independent); CAL = n*, K x c_r (M1 K_needed/K_noD).",
             "models": {}}
    for m, dsum in out["decay"].items():
        if any(x in m for x in ("_cnull_", "_usub_", "_realsub_", "_mod_")):
            continue
        onset["models"][m] = {"file": dsum["file"], "closure": dsum["closure"], "c_r": dsum["c_r"],
                              "boundaries": onset_record(dsum, keys_main), "transient": dsum["transient"],
                              "profile": dsum["profile"]}
    onset["islands_M2"] = out.get("peel")
    json.dump(onset, open(f"{OUT}/stochastic_onset.json", "w"), indent=1, default=float)
    # FSS + modules
    fss = {"notes": "decay theory (bQMF profile, single renormalised mode 2D chain and pooled) on my own FSS / module graphs "
                    "(results/m2/theory/stochastic/graphs; c fixed at full-brain value except cnull, which renormalises c).",
           "graphs": {}}
    keys_g = ["single_PF_r0_lt_2D", "single_CAL_r1_lt_2D", "single_CAL_r4_lt_2D", "pooled_PF_r0_lt_2D", "pooled_CAL_r1_lt_2D",
              "pooled_CAL_r4_lt_2D", "single_PF_r0_lt_F", "pooled_PF_r0_lt_F", "pooled_PF_r0_lt_A", "single_PF_r0_lt_A"]
    for m, dsum in out["decay"].items():
        if any(x in m for x in ("_cnull_", "_usub_", "_realsub_", "_mod_")):
            fss["graphs"][m] = {"file": dsum["file"], "N": dsum["N"], "boundaries": onset_record(dsum, keys_g)}
    json.dump(fss, open(f"{OUT}/stochastic_fss.json", "w"), indent=1, default=float)
    json.dump({"notes": "LNA around the driven QMF fixed point (r = 0 closure; first-order theory is r-independent).",
               **out["driven"]}, open(f"{OUT}/stochastic_driven.json", "w"), indent=1, default=float)
    json.dump({"notes": "LNA at the h = 0 QMF active state (r = 0): oscillation (quasi-cycle) and coherent-total lifetime.",
               **out["lnaqs"]}, open(f"{OUT}/stochastic_qs.json", "w"), indent=1, default=float)
    json.dump({"notes": "pulse protocol: quasi-static self-organised resources; critical-like iff min(S_eps, S_dep) >= 1000.",
               **out["pulse"]}, open(f"{OUT}/stochastic_aval.json", "w"), indent=1, default=float)
    mp = f"{R}/modes/modes.json"
    if os.path.exists(mp):
        mo = J(mp)
        json.dump({"notes": "top real modes of the excitatory operator (island approximation |v|, |u|; purity < 0.9 = mixed-sign "
                            "mode, approximation poor). Single-mode onsets: M1 logistic (PF K0 / CAL1 c_r=1 x K0) and M2 frozen / "
                            "adiabatic / 2-D chains.", "kappa": mo["kappa"], "c_r1": mo["c_r1"],
                   "modes": mo["modes"]}, open(f"{OUT}/stochastic_modes.json", "w"), indent=1, default=float)
    ms = sorted(glob.glob(f"{R}/multistart/*.json"))
    if ms:
        json.dump({"notes": "bQMF fixed points from 6 starts (all 0.5, AL-L, AL-R, CB, VNC, 1% random); max relative L1 distance",
                   **{os.path.basename(p)[:-5]: J(p) for p in ms}}, open(f"{OUT}/stochastic_multistart.json", "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    out = write_predictions()
    for k, v in out["decay"].items():
        print("==", k)
        for ck, cv in v["cross"].items():
            if ("2D" in ck) and ("r0_" in ck or "r1_" in ck or "r6_" in ck):
                print(f"   {ck:40s} S {cv['S']}  on {cv['on']}")
