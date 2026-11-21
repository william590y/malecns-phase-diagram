"""pd3 criticality: assemble results/pd3/criticality.json and evaluate the pre-registered Q2 criteria (plan.md 4.6) from the
analysis outputs. Every number is copied from (source given in "sources"):
  results/pd3/criticality/aval_analysis.json (C1), qs_analysis.json (C2/C3/C4), v0_validation.json (V0), c0_drive.json (C0),
  results/pd3/redraw_fss/fss_analysis.json (FSS nulls, via qs_analysis C4_merged), results/pd3/module_theory.md (mean-field
  hysteresis statement, quoted).
Post-hoc analyses (not pre-registered) are kept under keys named post_hoc_*.
"""
import json
import numpy as np

ROOT = "/MALECNS_ROOT"
R_ = f"{ROOT}/results/pd3/criticality"
A = json.load(open(f"{R_}/aval_analysis.json")); Q = json.load(open(f"{R_}/qs_analysis.json"))
V = json.load(open(f"{R_}/v0_validation.json")); C0 = json.load(open(f"{R_}/c0_drive.json"))
RU = A["runs"]; CP = A["critical_point"]
out = {"sources": {"C1": "results/pd3/criticality/aval_analysis.json", "C2_C3_C4": "results/pd3/criticality/qs_analysis.json",
                   "V0": "results/pd3/criticality/v0_validation.json", "C0": "results/pd3/criticality/c0_drive.json",
                   "jobs": "results/pd3/criticality/JOBS.md"}}

# ------------------------------------------------------------------------------------------------------------------ V0
out["V0"] = {k: {kk: V[k][kk] for kk in ("g", "reference", "swc_B", "P_max_abs_z", "P_n_beyond_2SE", "En_max_abs_z", "En_n_beyond_2SE",
                                         "KS_D", "KS_p", "pass_P_2SE_all_t", "pass_En_2SE_all_t", "pass_KS", "n_compared_t", "timing", "P_at_100")}
             for k in V if isinstance(V[k], dict) and "KS_p" in V[k]}
out["V0"]["exact_En1"] = V.get("exact_En1")

# ------------------------------------------------------------------------------------------------------------------ C0
k3 = [i for i, g in enumerate(C0["g"]) if 0.14 < g < 0.19]
out["C0"] = {"g": [C0["g"][i] for i in k3], "x": [C0["x"][i] for i in k3], "h": [C0["rows"][h]["h"] for h in C0["rows"]],
             "n_AL": [[C0["rows"][h]["n_AL"][i] for h in C0["rows"]] for i in k3], "h_mid": C0["h_mid"],
             "slope_nAL": [[C0["slope_nAL"][j][i] for j in range(len(C0["h_mid"]))] for i in k3]}

# ------------------------------------------------------------------------------------------------------------------ C1
def runrow(t, keys):
    r = RU.get(t)
    return None if r is None else {k: r.get(k) for k in keys}


c1 = {"x1": {}}
for t in ["real_r0_x1.00", "real_r1_x1.00", "real_r0.5_x1.00", "real_r2_x1.00", "real_r4_x1.00", "real_r6_x1.00", "module_r0_x1.00",
          "module_r1_x1.00", "cnull_N167216_s0_r0_x1.00", "cnull_N167216_s0_r1_x1.00", "cnull_N5000_s1_r0_x1.00", "cnull_N20000_s1_r0_x1.00",
          "cnull_N80000_s1_r0_x1.00", "complete37_x1.00", "complete76_x1.00", "complete300_x1.00", "complete2000_x1.00"]:
    r = RU.get(t)
    if r is None:
        continue
    row = {"n_records": r["n_records"], "mean_S": r["mean_S"], "mean_T_model": r["mean_T_model"], "P_T_gt": r["P_T_gt"],
           "spike_share_L_R_mid_rest": r["spike_share_L_R_mid_rest"], "gamma": r["gamma"].get("gamma"), "gamma_se": r["gamma"].get("gamma_se"),
           "ST_peak": r["ST"].get("peak"), "ST_peak_T": r["ST"].get("peak_T"), "ST_peak_ci": r["ST"].get("peak_ci"),
           "K_ref": r.get("K_ref"), "K_nc_glin": r.get("K_nc_glin"),
           "Kfit_T": {k: r.get("Kfit_T", {}).get(k) for k in ("K_fit", "lnK_se", "chi2_dof", "N_cond")} if isinstance(r.get("Kfit_T"), dict) else None,
           "comp_dev_Kref": {k: r["chain_comp_Kref"][k] for k in ("K", "t_hi", "max_abs_dev", "dev_at_t_hi")} if r.get("chain_comp_Kref") else None,
           "comp_dev_Kfit": {k: r["chain_comp_Kfit"][k] for k in ("K", "t_hi", "max_abs_dev", "dev_at_t_hi")} if r.get("chain_comp_Kfit") else None,
           "mle_S": r.get("mle_S"), "mle_T_steps": r.get("mle_T_steps"), "cutoffs_used": r.get("cutoffs_used"),
           "skew_by_T": [{k: q[k] for k in ("T_mean", "n", "skew_sp", "skew_sp_se", "skew_act", "skew_act_se")} for q in r.get("skew", [])],
           "samples_minority": r.get("samples_minority")}
    rc = r.get("recruit_50_250")
    if rc and rc.get("n_al"):
        sm = np.array(rc["share_minority"], float); ss = np.array(rc["share_seedlobe"], float); so = np.array(rc["share_other"], float)
        row["recruit_T10_50"] = {"n": rc["n"], "minority_share_early_mid_late": [float(np.nanmean(sm[2:8])), float(np.nanmean(sm[16:24])), float(np.nanmean(sm[32:40]))],
                                 "seedlobe_share_early_mid_late": [float(np.nanmean(ss[2:8])), float(np.nanmean(ss[16:24])), float(np.nanmean(ss[32:40]))],
                                 "rest_share_early_mid_late": [float(np.nanmean(so[2:8])), float(np.nanmean(so[16:24])), float(np.nanmean(so[32:40]))],
                                 "windows_t_over_T": [[0.05, 0.2], [0.4, 0.6], [0.8, 1.0]]}
    if r.get("vw"):
        vw = r["vw"]
        row["post_hoc_seeds_v_weighted"] = {"gamma": vw["gamma"].get("gamma"), "gamma_se": vw["gamma"].get("gamma_se"),
                                            "ST_peak": vw["ST"].get("peak"), "ST_peak_ci": vw["ST"].get("peak_ci"), "P_T_gt": vw.get("P_T_gt"),
                                            "Kfit_T": {k: vw.get("Kfit_T", {}).get(k) for k in ("K_fit", "lnK_se", "chi2_dof")} if vw.get("Kfit_T") else None,
                                            "comp_dev_Kfit": {k: vw["chain_comp_Kfit"][k] for k in ("K", "max_abs_dev", "dev_at_t_hi")} if vw.get("chain_comp_Kfit") else None,
                                            "comp_dev_Kref": {k: vw["chain_comp_Kref"][k] for k in ("K", "max_abs_dev", "dev_at_t_hi")} if vw.get("chain_comp_Kref") else None,
                                            "skew_by_T": [{k: q[k] for k in ("T_mean", "skew_sp", "skew_sp_se")} for q in vw.get("skew", [])]}
    c1["x1"][t] = row
c1["critical_point"] = {"families": {k: {kk: v[kk] for kk in ("x", "gamma", "gamma_se", "x_star", "x_star_ci", "status")} for k, v in CP["families"].items()},
                        "chain_x_star": CP["chain_pred"], "r_dependence_x1": CP["r_dependence_x1"], "definition": CP["definition"]}
c1["chain_ST_peaks_quoted"] = A["chain_ST_peaks_quoted"]; c1["chain_ST_peaks_reanalysed"] = A["chain_ST_peaks"]; c1["chain_Tc_model"] = A["chain_Tc_model"]
out["C1"] = c1

# ------------------------------------------------------------------------------------------------------------- C2-C4
c2 = {}
for k, v in Q.get("C2", {}).items():
    c2[k] = {"K_chain": v["K"], "beta": v["beta"], "beta_chain_same_window": v["beta_chain_same_window"], "beta_window_points": v["beta_window_points"],
             "chi_AL_peak": v["chi_AL_peak"], "chi_tot_peak": v["chi_tot_peak"], "chain_chi_peak": v["chain_chi_peak"], "all_unimodal": v["all_unimodal"],
             "rows": [{kk: q[kk] for kk in ("x", "g", "R", "n_AL", "n_AL_se", "n_tot", "chain_nQS_neurons", "chain_ratio_nAL", "MF_amp_nAL", "chi_AL",
                                             "chi_AL_ci", "chi_tot", "chi_tot_ci", "chain_chi_neurons", "tau_QS", "chain_tauQS_model", "PR", "AL_share")}
                      | {"unimodal_nAL_ge_1": q["unimodality"]["unimodal"], "n_peaks": q["unimodality"]["n_peaks"], "P_nAL0": q["unimodality"].get("P_nAL0")} for q in v["rows"]]}
out["C2"] = c2
out["C3"] = {k: {kk: v[kk] for kk in ("h", "n_AL", "n_AL_se", "slope", "slope_se", "n_mid", "min_slope")} | {"chain_min_slope": v["chain"]["min_slope"], "chain_K": v["chain"]["K"]}
             for k, v in Q.get("C3", {}).items()}
out["C4"] = {k: {"slope_tau_vs_Kn": v["slope_tau_vs_Kn"], "slope_se": v["slope_se"], "slope_tau_vs_Kfit": v["slope_tau_vs_Kfit"], "slope_Kfit_se": v["slope_Kfit_se"],
                 "rows": [{kk: q[kk] for kk in ("n", "g_lin", "K_n", "K_fit", "tau", "deaths", "tau_x1", "pred_tau_x1_from_K_n", "mod_overlap")} for q in v["rows"]]}
             for k, v in Q.get("C4", {}).items()}
out["C4_merged_with_FSS_nulls"] = {k: {kk: v.get(kk) for kk in ("n_null", "n_sub", "slope_all_Kunfitted", "slope_all_Kfit", "slope_nulls_Knc")}
                                   for k, v in Q.get("C4_merged", {}).items()}

# ---------------------------------------------------------------------------------------------------- Q2 criteria
crit = {}
fam = CP["families"]
xs = {r: fam.get(f"real_r{r}", {}).get("x_star") for r in (0, 1, 6)}
crit["Q2a_xstar_in_0.97_1.03"] = {"x_star": xs, "x_star_ci": {r: fam.get(f"real_r{r}", {}).get("x_star_ci") for r in (0, 1, 6)},
                                  "r6_status": fam.get("real_r6", {}).get("status"),
                                  "pass": all(v is not None and 0.97 <= v <= 1.03 for v in xs.values())}
pk = {}
for k, gp in (("real_r0", 0.195), ("real_r1", 0.209)):
    if k in Q.get("C2", {}):
        g = Q["C2"][k]["chi_AL_peak"]["g"]; pk[k] = {"g_peak_chi_AL": g, "predicted": gp, "rel_dev": g / gp - 1, "where": Q["C2"][k]["chi_AL_peak"]["where"],
                                                       "g_peak_chi_tot": Q["C2"][k]["chi_tot_peak"]["g"]}
crit["Q2a_chi_peak_within_10pct"] = {"peaks": pk, "pass": bool(pk) and all(abs(v["rel_dev"]) <= 0.10 for v in pk.values())}
b = {}
for t, Kn in (("real_r0_x1.00", 75.8), ("real_r1_x1.00", 36.6)):
    r = RU.get(t, {})
    kf = r.get("Kfit_T", {}) if isinstance(r.get("Kfit_T"), dict) else {}
    cd = r.get("chain_comp_Kfit", {}) or {}
    b[t] = {"K_fit": kf.get("K_fit"), "chi2_dof": kf.get("chi2_dof"), "max_comp_dev_upto_Tc": cd.get("max_abs_dev"), "K_needed": Kn,
            "K_ratio": (kf.get("K_fit") / Kn) if kf.get("K_fit") else None,
            "pass_shape": bool(cd and (cd["max_abs_dev"] <= 0.10 or (kf.get("chi2_dof") or 99) <= 2)),
            "pass_K": bool(kf.get("K_fit") and 1 / 1.5 <= kf["K_fit"] / Kn <= 1.5)}
st = {t: RU[t]["ST"]["peak"] for t in ("real_r0_x1.00", "real_r1_x1.00") if t in RU}
crit["Q2b_finite_K_MFDP"] = {"P_T_chain_fit": b, "ST_peak_real": st, "ST_peak_in_1.5_1.75": all(1.5 <= v <= 1.75 for v in st.values()),
                             "pass": all(v["pass_shape"] and v["pass_K"] for v in b.values()) and all(1.5 <= v <= 1.75 for v in st.values())}
m = Q.get("C4_merged", {})
crit["Q2c_tau_vs_K"] = {"submodules_only_Kn": {k: [v["slope_tau_vs_Kn"], v["slope_se"]] for k, v in Q.get("C4", {}).items()},
                        "submodules_only_Kfit": {k: [v["slope_tau_vs_Kfit"], v["slope_Kfit_se"]] for k, v in Q.get("C4", {}).items()},
                        "submodules_plus_nulls_Kunfitted": {k: v.get("slope_all_Kunfitted") for k, v in m.items()},
                        "submodules_plus_nulls_Kfit": {k: v.get("slope_all_Kfit") for k, v in m.items()}}
crit["Q2c_tau_vs_K"]["pass_with_Kfit"] = all(v and 0.4 <= v[0] <= 0.6 for v in crit["Q2c_tau_vs_K"]["submodules_plus_nulls_Kfit"].values())
crit["Q2c_tau_vs_K"]["pass_with_unfitted_K"] = all(v and 0.4 <= v[0] <= 0.6 for v in crit["Q2c_tau_vs_K"]["submodules_plus_nulls_Kunfitted"].values())
d = {}
for k in ("real_r0", "real_r1", "real_r6"):
    if k in Q.get("C2", {}):
        rows = Q["C2"][k]["rows"]
        rat = [q["chain_ratio_nAL"] for q in rows if q["R"] > 1.0]
        d[k] = {"ratio_nAL_over_chain_R_gt_1": [min(rat), max(rat)] if rat else None, "within_20pct_all": bool(rat) and all(0.8 <= x <= 1.2 for x in rat),
                "frac_within_20pct": float(np.mean([0.8 <= x <= 1.2 for x in rat])) if rat else None}
def _min_ok(d, xmin_n=1.0):
    sl = np.array(d["slope"], float); se = np.array(d["slope_se"], float); nm = np.array(d["n_mid"], float)
    ok = np.isfinite(sl) & (se < 0.25) & (nm >= xmin_n)
    return float(sl[ok].min()) if ok.any() else None
fs = {k: _min_ok(Q["C3"][k]) for k in Q.get("C3", {})}          # well-measured slopes only (SE < 0.25, n_AL >= 1)
crit["Q2d_exponents"] = {"nQS_vs_chain": d, "field_min_slope": fs,
                         "field_min_slope_definition": "minimum local slope over intervals with slope SE < 0.25 and n_AL(mid) >= 1",
                         "field_pass": (all(fs.get(f"r{r}_x1.00", 9) <= 0.65 for r in (0, 1)) and all(fs.get(f"r{r}_x0.90", 0) >= 0.85 for r in (0, 1)))
                         if all(f"r{r}_x{x}" in Q.get("C3", {}) for r in (0, 1) for x in ("0.90", "1.00")) else None}
crit["Q2d_exponents"]["nQS_pass"] = bool(d) and all(v["within_20pct_all"] for v in d.values())
crit["Q2e_not_first_order"] = {"QS_histograms_unimodal_nAL_ge_1": {k: v["all_unimodal"] for k, v in Q.get("C2", {}).items()},
                               "note_nAL0": "unimodality tested on n_AL >= 1; the n_AL = 0 bin holds samples with only non-AL activity (QS conditioning on total activity), reported as P_nAL0 per x",
                               "mean_field_hysteresis_at_onset": "none (module_theory.md Q2 row: 'At the AL onset the neuron-level mean field is continuous at every r: one attractor, up- and down-sweeps identical'; multistability only at g >= 0.33)",
                               "stochastic_hysteresis": "not tested by this program (multistability program M2, in progress at the time of writing)"}
out["Q2_criteria"] = crit
json.dump(out, open(f"{ROOT}/results/pd3/criticality.json", "w"), indent=1, default=lambda o: None if o is None else float(o))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("pass", "pass_with_Kfit", "pass_with_unfitted_K", "field_pass", "nQS_pass", "x_star")} for k, v in crit.items()}, indent=1, default=str))
