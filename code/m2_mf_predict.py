"""m2_mf_predict.py -- assemble the mean-field predictions of Model 2 (lane m2/theory, meanfield) from the analysed runs.
Inputs: results/m2/theory/meanfield_{points,lines,glin,chain,analytic}.json (m2_mf_analyze.py, m2_mf_glin.py,
m2_mf_chain.py) and the M1 references named in plan_meanfield.md. No experiment-lane output is read.
Outputs: meanfield_validation.json (V1, V2, A1), meanfield_predictions.json, figs/m2/meanfield_{phase,driven,modes}.png.
"""
import json, sys, time
import numpy as np

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
import m2_common as MC          # noqa: E402
import m2_obs as OBS            # noqa: E402

TH = f"{ROOT}/results/m2/theory"
L = json.load(open(f"{TH}/meanfield_lines.json"))
P = json.load(open(f"{TH}/meanfield_points.json"))
GL = json.load(open(f"{TH}/meanfield_glin.json"))
CH = json.load(open(f"{TH}/meanfield_chain.json"))
AN = json.load(open(f"{TH}/meanfield_analytic.json"))
SIM1 = {row["r"]: row for row in json.load(open(f"{ROOT}/results/analysis2/flow_sim_boundaries.json"))["rows"]}
RS = MC.R_ALL; RC = MC.R_CORE
rk = lambda r: f"r{r:g}"


def name(proto, var, cl, r, extra=""):
    if cl == "cqmf" and r == 0 and f"{proto}_{var}_cqmf_r0{extra}" not in L:
        cl = "qmf"                                   # identical closures at r = 0
    return f"{proto}_{var}_{cl}_r{r:g}{extra}"


def get(proto, var, cl, r, extra=""):
    return L.get(name(proto, var, cl, r, extra))


def pts(proto, var, cl, r, ic, extra=""):
    d = P.get(name(proto, var, cl, r, extra))
    return None if d is None else sorted([p for p in d["points"] if p["ic"] == ic], key=lambda p: p["g"])


def interp_ln(g, y, gq):
    g = np.asarray(g, float); y = np.asarray(y, float)
    return float(np.exp(np.interp(np.log(gq), np.log(g), np.log(np.maximum(y, 1e-300)))))


# ------------------------------------------------------------------------------------------------ V1 / V2 / A1
val = {"V1": {}, "V2": {}, "A1": {}}
for cl in ("qmf", "cqmf"):
    errs, wrong, rows = [], 0, {}
    for r in RC:
        ln = get("decay", "M1", cl, r)
        if ln is None:
            continue
        Dm, Ds = ln["D"], SIM1[r]["D"]
        rows[f"{r:g}"] = {"D_MF": Dm, "D_sim": Ds, "on_MF_cutoff": ln["on"], "on_sim": SIM1[r]["on"]}
        if r in (0.25, 0.5, 1):
            errs.append(abs(np.log((Dm if Dm is not None else 30.0) / Ds)))
        if r in (2, 4, 6):
            wrong += int(Dm is not None)              # sim: none up to 30
    val["V1"][cl] = {"rows": rows, "mean_abs_ln_err": float(np.mean(errs)) if errs else None, "n_wrong_calls": wrong,
                     "score": (float(np.mean(errs)) + 0.5 * wrong) if errs else None}
sc = {cl: val["V1"][cl]["score"] for cl in ("qmf", "cqmf")}
primary = min((c for c in sc if sc[c] is not None), key=lambda c: sc[c]) if any(v is not None for v in sc.values()) else "cqmf"
val["V1"]["primary"] = primary
alt = "qmf" if primary == "cqmf" else "cqmf"
for cl in ("qmf", "cqmf"):
    rows = {}
    for r in RC:
        ln = get("driven", "M1", cl, r)
        if ln is None:
            continue
        rows[f"{r:g}"] = {"G029": ln["DQ_DA_G029"], "allowed": ln["allowed_intervals"], "DA_DT": ln["DA_DT_bracket"]}
    val["V2"][cl] = rows
val["V2"]["E1_reference"] = "aldef REPORT: allowed g r=0 0.090-0.231, r=1 0.100-0.289 (ORN 0.01-0.10); ORN 0.04 r=1 0.114-0.278"
for cl in ("qmf", "cqmf"):
    ncal, nscal = {}, {}
    for r in RC:
        pp = pts("decay", "M1", cl, r, "decay")
        if pp is None:
            continue
        g = [p["g"] for p in pp]; n = [max(p["n_qs"], 1e-300) for p in pp]
        ncal[f"{r:g}"] = interp_ln(g, n, SIM1[r]["on"]); nscal[f"{r:g}"] = interp_ln(g, n, SIM1[r]["S"])
    v = np.array(list(ncal.values())) if ncal else np.array([np.nan])
    ok = bool(len(ncal) >= 5 and np.nanmax(v) / np.nanmin(v) <= 3)
    val["A1"][cl] = {"n_cal_by_r": ncal, "n_cal_S_by_r": nscal, "max_over_min": float(np.nanmax(v) / np.nanmin(v)),
                     "valid": ok, "n_cal": float(np.exp(np.mean(np.log(v)))) if ok else None,
                     "n_cal_S": float(np.exp(np.mean(np.log(list(nscal.values()))))) if ok else None}
json.dump(val, open(f"{TH}/meanfield_validation.json", "w"), indent=1, default=float)
print("V1 scores", sc, "primary", primary, "| A1 valid", {c: val["A1"][c]["valid"] for c in ("qmf", "cqmf")},
      {c: round(val["A1"][c]["max_over_min"], 2) for c in ("qmf", "cqmf")})


# ------------------------------------------------------------------------------------------------ helpers
def equal_size_on(var, cl, r, proto="decay", extra=""):
    a = val["A1"][cl]
    if not a["valid"]:
        return None, None
    pp = pts(proto, var, cl, r, "decay", extra)
    if pp is None:
        return None, None
    g = np.array([p["g"] for p in pp]); n = np.array([p["n_qs"] if p["tau_MF"] is None else 0.0 for p in pp])
    return (OBS.first_crossing(g, np.log(np.maximum(n, 1e-30)), np.log(a["n_cal"])),
            OBS.first_crossing(g, np.log(np.maximum(n, 1e-30)), np.log(a["n_cal_S"])))


def decay_block(var, cl, r, extra=""):
    ln = get("decay", var, cl, r, extra)
    if ln is None:
        return None
    pp = pts("decay", var, cl, r, "decay", extra)
    out = {k: ln.get(k) for k in ("S", "on", "D", "S_nc0.1", "on_nc0.1", "S_nc10.0", "on_nc10.0", "hopf_first_g",
                                    "hopf_bracket", "osc_intervals", "bistable_intervals", "osc_periods")}
    eon, eS = equal_size_on(var, cl, r, extra=extra)
    out["on_equal_size"], out["S_equal_size"] = eon, eS
    # state at reference g values
    ref = {}
    for gq in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
        p = min(pp, key=lambda p: abs(np.log(p["g"] / gq)))
        ref[f"{p['g']:g}"] = {"sustained_MF": p["tau_MF"] is None, "n_qs": p["n_qs"], "rho_qs": p["rho_qs"],
                              "PR_N": p["PR_N"], "AL_module_share": p["share"].get("AL_module"),
                              "eff_act_all": p["eff_act"]["all"], "nmin": p["nmin"], "tau_MF": p["tau_MF"],
                              "sigma_x": p.get("sigma_x"), "sigma_1": p.get("sigma_1"),
                              "ring_demog": p.get("ring", {}).get("demog"), "ring_robust_P": p.get("ring_robust_P"),
                              "osc": p["osc"]["osc"], "osc_period": p["osc"]["period"] if p["osc"]["osc"] else None}
    out["at_g"] = ref
    return out


def driven_block(var, cl, r, extra=""):
    ln = get("driven", var, cl, r, extra)
    if ln is None:
        return None
    pp = pts("driven", var, cl, r, "quiet", extra)
    out = {k: ln.get(k) for k in ("DQ_DA_G029", "G_145", "G_max", "g_at_G_max", "DA_DT_bracket", "DT_SS_bracket",
                                    "tau_post_200", "tau_post_3000", "allowed_intervals", "osc_intervals",
                                    "bistable_intervals")}
    a = val["A1"][cl]
    out["SS_equal_size"] = None
    if a["valid"]:
        g = np.array([p["g"] for p in pp]); n = np.array([p.get("n_nonORN_T", 0.0) for p in pp])
        out["SS_equal_size"] = OBS.first_crossing(g, np.log(np.maximum(n, 1e-30)), np.log(a["n_cal"]))
    # derived (labelled secondary): the gain window alone, and the gain window below the equal-size removal line
    gg = np.array([p["g"] for p in pp]); Gv = np.array([p["G"] if p["G"] is not None else np.nan for p in pp])
    win = (Gv >= 0.29) & (Gv <= 1.45)
    import m2_mf_analyze as AZ
    out["G_window_intervals"] = AZ.intervals(gg, win)
    sse = out.get("SS_equal_size")
    out["allowed_A1_intervals"] = AZ.intervals(gg, win & (gg < (sse if sse is not None else np.inf)))
    ref = {}
    for gq in (0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        p = min(pp, key=lambda p: abs(np.log(p["g"] / gq)))
        ref[f"{p['g']:g}"] = {k: p.get(k) for k in ("phase", "G", "rho_ORN", "rho_drv", "rho_uPN", "rho_AL_module",
                                                    "PR_nonORN_frac", "tau_post", "S1", "S2", "allowed", "sigma_x")}
        if "ring" in p:
            ref[f"{p['g']:g}"]["ring_demog_nonORN"] = p["ring"]["demog"]
            ref[f"{p['g']:g}"]["ring_demog_AL"] = p["ring"]["demog_AL"]
    out["at_g"] = ref
    return out


# ------------------------------------------------------------------------------------------------ assemble
pred = {"written": time.strftime("%Y-%m-%d %H:%M"), "lane": "m2/theory meanfield",
        "experiment_outputs_read": "none (results/m2/{decay,driven,nature,modules} and figs/m2 not opened)",
        "primary_closure": primary, "closure_scores_V1": sc,
        "files": {"derivation": "results/m2/theory/derivation.md", "plan": "results/m2/theory/plan_meanfield.md",
                  "lines": "meanfield_lines.json", "points": "meanfield_points.json", "validation": "meanfield_validation.json",
                  "glin": "meanfield_glin.json", "chain": "meanfield_chain.json", "analytic": "meanfield_analytic.json"}}
pred["P1_silent_state"] = {
    "g_lin": {k: v["g_lin"] for k, v in GL["glin"].items()},
    "g_lin_note": "exact linearisation about RESTED silence; spike-mode depression leaves it unchanged (M2 = M1 = M2_allsyn = KW09 = N15 = tau30); M2s/M1s operator differs",
    "g_lin_uniform_x_check_sigma(2 g_lin, x=0.5)": GL["glin"]["real"].get("sigma_at_2glin_x0.5"),
    "g_QMF_silent_edge_by_r": {k: {r: v["g_QMF"] for r, v in d.items()} for k, d in GL["mu"].items()}}
pred["P2_single_unit"] = AN
dec = {}
for r in RS:
    dec[f"{r:g}"] = {"primary": decay_block("M2", primary, r), "alternate": decay_block("M2", alt, r),
                     "chain_AL_mode": {"frozen_kcal": CH["lines"].get(f"M2|r{r:g}|kcal"), "frozen_k1": CH["lines"].get(f"M2|r{r:g}|k1"),
                                       "kappa": CH["kappa"].get(f"{r:g}")}}
pred["P3_decay_M2"] = dec
drv = {}
for r in RS:
    drv[f"{r:g}"] = {"primary": driven_block("M2", primary, r), "alternate": driven_block("M2", alt, r)}
pred["P6_driven_M2_ORN0.04"] = drv
lev = {}
for orn in (0.01, 0.02, 0.07, 0.1):
    for r in RC:
        for cl in (primary, alt):
            b = driven_block("M2", cl, r, f"_orn{orn:g}")
            if b is not None:
                lev[f"orn{orn:g}|r{r:g}|{cl}"] = {k: b[k] for k in ("DQ_DA_G029", "G_max", "allowed_intervals", "DA_DT_bracket", "DT_SS_bracket", "osc_intervals", "SS_equal_size", "G_window_intervals", "allowed_A1_intervals")}
pred["P6b_driven_ORN_levels"] = lev
var = {}
for v in ("M2s", "M1s", "M2_allsyn", "M2_KW09", "M2_N15", "M2_tau30", "M1"):
    for r in RC:
        for cl in (primary, alt):
            b = decay_block(v, cl, r)
            if b is not None:
                var[f"decay|{v}|r{r:g}|{cl}"] = {k: b[k] for k in ("S", "on", "D", "hopf_first_g", "on_equal_size", "S_equal_size")}
            b = driven_block(v, cl, r)
            if b is not None:
                var[f"driven|{v}|r{r:g}|{cl}"] = {k: b[k] for k in ("DQ_DA_G029", "G_max", "allowed_intervals", "DA_DT_bracket", "DT_SS_bracket", "osc_intervals", "SS_equal_size", "G_window_intervals", "allowed_A1_intervals")}
        for nl in ("targets", "weights"):
            for cl in (primary, alt):
                b = decay_block("M2", cl, r, f"_null{nl}") if v == "M2s" else None
                if b is not None:
                    var[f"decay|M2_null{nl}|r{r:g}|{cl}"] = {k: b[k] for k in ("S", "on", "D", "hopf_first_g", "on_equal_size")}
    if v in ("M2_KW09", "M2_N15", "M2_tau30", "M2_allsyn"):
        for r in RC:
            c = CH["lines"].get(f"{v}|r{r:g}|kcal")
            if c:
                var[f"chain|{v}|r{r:g}"] = c
pred["P8_variants_nulls"] = var
json.dump(pred, open(f"{TH}/meanfield_predictions.json", "w"), indent=1, default=float)
print("saved predictions; primary", primary)

# ------------------------------------------------------------------------------------------------ oscillations, sigma_x
def mode_summary(proto, var, cl, r, ic, key):
    pp = pts(proto, var, cl, r, ic)
    if pp is None:
        return None
    sus = [p for p in pp if (p.get("tau_MF") is None if proto == "decay" else True)]
    good = [p for p in sus if "ring" in p and p.get("ring_robust_P") and not p["osc"]["osc"] and p.get("ring_resid", 1) < 1e-6
            and p["ring"][key].get("osc_absz", 2) < 1]
    osc = [p["g"] for p in sus if p["osc"]["osc"]]
    out = {"n_points_ringed": sum("ring" in p for p in sus), "n_robust_stationary": len(good),
           "mf_oscillating_g": osc,
           "mf_osc_periods": {f"{p['g']:g}": p["osc"]["period"] for p in sus if p["osc"]["osc"]},
           "mf_osc_rel_p2p": {f"{p['g']:g}": p["osc"]["rel_p2p"] for p in sus if p["osc"]["osc"]}}
    if good:
        Pv = np.array([p["ring"][key]["osc_P"] for p in good]); Dv = np.array([p["ring"][key]["osc_DP"] for p in good])
        out.update({"damped_mode_period_median": float(np.median(Pv)), "damped_mode_period_range": [float(Pv.min()), float(Pv.max())],
                    "damping_per_period_max": float(Dv.max()), "damping_per_period_median": float(np.median(Dv)),
                    "g_of_max_DP": float(good[int(np.argmax(Dv))]["g"]), "n_DP_ge_0.7": int((Dv >= 0.7).sum()),
                    "by_g": {f"{p['g']:g}": [p["ring"][key]["osc_P"], p["ring"][key]["osc_DP"]] for p in good}})
    sx = [(p["g"], p["sigma_x"]) for p in sus if "sigma_x" in p and (proto != "decay" or p["tau_MF"] is None)]
    if sx:
        g_, s_ = np.array(sx).T
        for lim in (1.1, 1.25):
            m = (s_ >= 0.98) & (s_ <= lim)
            out[f"sigma_x_within_1_{lim}_g"] = [float(g_[m].min()), float(g_[m].max())] if m.any() else None
        out["sigma_x_by_g"] = {f"{a:g}": float(b) for a, b in sx}
    return out


osc = {}
for r in RC:
    osc[f"decay|r{r:g}"] = mode_summary("decay", "M2", primary, r, "decay", "demog")
    osc[f"driven|r{r:g}"] = mode_summary("driven", "M2", primary, r, "quiet", "demog")
    osc[f"decay|M1|r{r:g}"] = mode_summary("decay", "M1", primary, r, "decay", "demog")
osc["first_mf_oscillation_g_by_r_decay"] = {f"{r:g}": (get("decay", "M2", primary, r) or {}).get("hopf_first_g") for r in RS}
osc["mf_osc_intervals_by_r_decay"] = {f"{r:g}": (get("decay", "M2", primary, r) or {}).get("osc_intervals") for r in RS}
osc["mf_osc_intervals_by_r_driven"] = {f"{r:g}": (get("driven", "M2", primary, r) or {}).get("osc_intervals") for r in RS}
pred["P4_oscillations"] = osc
json.dump(pred, open(f"{TH}/meanfield_predictions.json", "w"), indent=1, default=float)
print("added P4")
