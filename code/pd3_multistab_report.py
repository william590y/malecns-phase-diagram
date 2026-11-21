"""Collect the multistability program's numbers into results/pd3/multistability.json (every entry carries its source file).
Inputs: results/pd3/multistab/{validation.json, analysis_m1.json, analysis_m2a.json, analysis_m2b.json, analysis_m2c.json,
analysis_m3.json, analysis_m3b.json} (written by pd3_multistab_validate.py / pd3_multistab_analyze.py).
"""
import json, os
import numpy as np

ROOT = "/MALECNS_ROOT"
D_ = f"{ROOT}/results/pd3/multistab"
S = lambda f: f"results/pd3/multistab/{f}"


def J(f):
    return json.load(open(f"{D_}/{f}"))


def main():
    out = {"program": "multistability (plan.md section 5)", "date": "2026-11-09", "workflow": "wf_98fb990d-4fb",
           "code": ["code/pd3_multistab.py", "code/pd3_multistab_args.py", "code/pd3_multistab_validate.py", "code/pd3_multistab_analyze.py",
                    "code/pd3_multistab_figs.py", "code/pd3_multistab_report.py"],
           "jobs": "results/pd3/multistability/JOBS.md",
           "figures": ["figs/pd3/multistab_AL_symmetry.png", "figs/pd3/multistab_hysteresis.png", "figs/pd3/multistab_ic_attractors.png",
                       "figs/pd3/multistab_qmf_cells.png", "figs/pd3/multistab_m2b.png"]}
    # validation
    v = J("validation.json")
    out["validation"] = {"source": S("validation.json"), "pass": v["pass"],
                         "runs": {k: {"gpu": r["gpu"], "rows": [{kk: row[kk] for kk in ("g", "B", "deaths", "tau_snapped", "tau_ci_snapped", "inside_pooled_ci")
                                                                 if kk in row} | ({"pooled_tau": row["pooled"]["tau_sim"], "pooled_ci": row["pooled"]["tau_ci"],
                                                                                   "lr_p": row["lr_p_equal_rate"]} if "pooled" in row else {"envelope": row["envelope"]})
                                                                for row in r["rows"]]} for k, r in v["runs"].items()}}
    # M1
    m1 = J("analysis_m1.json")
    pts = []
    for r, vv in m1["per_r"].items():
        for row in vv["rows"]:
            pts.append({"r": float(r), "g": row["g"], "x_gon": row["x_gon"], "closure_predicts_B1": row["closure_predicts_B1"],
                        "n_surv": {s: row["starts"][s]["n_surv"] for s in row["starts"]},
                        "t_sym_median": {s: row["starts"][s]["t_sym"]["median"] for s in ("AL-L", "AL-R")},
                        "t_sym_q75": {s: row["starts"][s]["t_sym"]["q75"] for s in ("AL-L", "AL-R")},
                        "n_AL_died_before_sym": {s: row["starts"][s]["t_sym"]["n_AL_died_first"] for s in ("AL-L", "AL-R")},
                        "late_A_mean": {s: row["starts"][s]["late_A_mean"] for s in row["starts"]},
                        "frac_Lonly_keep_A_gt_0.6": row["starts"]["AL-L"]["frac_surv_keeping_start_lobe"],
                        "frac_Ronly_keep_A_lt_-0.6": row["starts"]["AL-R"]["frac_surv_keeping_start_lobe"],
                        "ks_L_vs_R_p": row["ks_L_vs_R"]["p"] if row["ks_L_vs_R"] else None,
                        "all_start_bimodal": row["bimodality_all"]["bimodal"], "all_start_dBIC": row["bimodality_all"].get("dBIC"),
                        "criteria": row["criteria"], "broken_symmetry": row["broken_symmetry"]})
    tsm = [p["t_sym_median"][s] for p in pts for s in ("AL-L", "AL-R") if p["t_sym_median"][s] is not None]
    late = [(abs(x), p["r"], p["g"], s) for p in pts for s, x in p["late_A_mean"].items() if x is not None and p["n_surv"][s] >= 10]
    allv = [abs(a) for r, vv in m1["per_r"].items() for row in vv["rows"] for s in row["late_A_values"] for a in row["late_A_values"][s]]
    ksp = [p["ks_L_vs_R_p"] for p in pts if p["ks_L_vs_R_p"] is not None]
    out["M1"] = {"source": S("analysis_m1.json"), "n_points": len(pts), "points": pts,
                 "summary": {"any_broken_symmetry": any(p["broken_symmetry"] for p in pts),
                             "t_sym_median_range": [min(tsm), max(tsm)], "max_frac_Lonly_keep": max((p["frac_Lonly_keep_A_gt_0.6"] or 0) for p in pts),
                             "max_frac_Ronly_keep": max((p["frac_Ronly_keep_A_lt_-0.6"] or 0) for p in pts),
                             "min_ks_p_L_vs_R": min(ksp), "n_ks_tests": len(ksp), "n_bimodal_all": sum(p["all_start_bimodal"] for p in pts),
                             "max_abs_late_A_mean_start_with_ge10_surv": max(late)[0], "max_abs_late_A_any_replica": max(allv),
                             "points_without_survivors": [(p["r"], p["g"]) for p in pts if sum(p["n_surv"].values()) == 0],
                             "n_closure_B1_points": sum(p["closure_predicts_B1"] for p in pts),
                             "n_closure_B1_points_with_ge10_survivors_per_one_lobe_start": sum(1 for p in pts if p["closure_predicts_B1"]
                                                                                              and min(p["n_surv"]["AL-L"], p["n_surv"]["AL-R"]) >= 10)}}
    # M2a
    m2 = J("analysis_m2a.json")
    out["M2a"] = {"source": S("analysis_m2a.json"), "decisions": m2["decisions"], "any_candidate_spec_rho": m2["any_candidate_spec"],
                  "any_candidate_noise_corrected_rho": m2["any_candidate_noise_corrected"], "any_candidate_spec_PR": m2["any_PR_candidate_spec"],
                  "any_candidate_noise_corrected_PR": m2["any_PR_candidate_noise_corrected"],
                  "areas": {k: {"A_rho": r["area_all"]["A_rho"], "A_rho_se": r["area_all"]["A_rho_se"], "A_rho_floor": r["area_all"]["A_rho_noise_floor"],
                                "A_rho_onset": r["area_onset_window"]["A_rho"], "A_rho_onset_se": r["area_onset_window"]["A_rho_se"],
                                "A_PR": r["area_all"]["A_PR"], "A_PR_se": r["area_all"]["A_PR_se"], "A_PR_floor": r["area_all"]["A_PR_noise_floor"],
                                "max_dlnrho": max(r["area_all"]["d_ln_rho"]), "g_at_max_dlnrho": r["g"][int(np.argmax(r["area_all"]["d_ln_rho"]))],
                                "gpu": r["gpu"], "seconds": r["seconds"]} for k, r in m2["runs"].items()},
                  "h0_down": {r: {kk: vv for kk, vv in x.items() if kk not in ("g_ext_values", "rho_down", "g")} for r, x in m2["h0"].items()}}
    # M2b
    if os.path.exists(f"{D_}/analysis_m2b.json"):
        b = J("analysis_m2b.json")
        out["M2b"] = {"source": S("analysis_m2b.json"), "tasks": b, "any_coexistence": any(x["coexistence"] for x in b.values()),
                      "min_ks_p_rho": min(x["ks_p_rho"] for x in b.values()), "min_ks_p_PR": min(x["ks_p_PR"] for x in b.values())}
    # M2c
    c = J("analysis_m2c.json")
    out["M2c"] = {"source": S("analysis_m2c.json"),
                  "per_g": {g: {gn: {"A_rho": x[gn]["area"]["A_rho"], "A_rho_se": x[gn]["area"]["A_rho_se"], "A_rho_floor": x[gn]["area"]["A_rho_noise_floor"],
                                     "A_PR": x[gn]["area"]["A_PR"], "A_PR_se": x[gn]["area"]["A_PR_se"], "A_PR_floor": x[gn]["area"]["A_PR_noise_floor"],
                                     "candidate_spec_D200": x[gn]["candidate_spec_D200"], "candidate_noise_corrected_D200": x[gn]["candidate_noise_corrected_D200"]}
                                for gn in ("fwd", "rev")} | {"onelobe_dwell_audit": {k: v for k, v in x["onelobe_dwell_audit"].items() if k != "dwells"}}
                            for g, x in c.items()}}
    # M3
    m3 = J("analysis_m3.json")
    out["M3"] = {"source": S("analysis_m3.json"), "any_multistable": m3["any_multistable"], "any_distinct_pair_spec": m3["any_distinct_pair_spec"],
                 "cells": {k: {"r": x["r"], "g": x["g"], "multistable": x["multistable"], "n_held_nonsilent_clusters": x["n_held_nonsilent_clusters"],
                               "max_rel_l1_between_start_means": x["max_rel_l1_between_start_means"],
                               "max_spread": max((s["spread"] for s in x["starts"].values() if s["spread"] is not None), default=None),
                               "alive_T": {s: v["alive_T"] for s, v in x["starts"].items()}} for k, x in m3["cells"].items()}}
    # M3b
    if os.path.exists(f"{D_}/analysis_m3b.json"):
        q = J("analysis_m3b.json")
        cells = {}
        for k, x in q["cells"].items():
            mx = []
            for pl, pr in x["pairs"].items():
                for kind in ("mt3", "fp"):
                    a, b_ = pr[kind]["z_late_median_a"], pr[kind]["z_late_median_b"]
                    if a is not None and b_ is not None:
                        mx.append(abs(a - b_))
            cells[k] = {"r": x["r"], "g": x["g"], "attractors": x["attractors"], "n_region": x["n_region"], "alive": x["alive"],
                        "coexistence": x["coexistence"], "max_abs_median_z_difference": max(mx) if mx else None,
                        "min_ks_p": min((pr[kind]["ks_p"] for pr in x["pairs"].values() for kind in ("mt3", "fp") if pr[kind]["ks_p"] is not None), default=None),
                        "all_dead": all(a == 0 for a in x["alive"].values())}
        out["M3b"] = {"source": S("analysis_m3b.json"), "n_cells": len(cells), "any_coexistence": q["any_coexistence"],
                      "n_cells_all_dead": sum(c_["all_dead"] for c_ in cells.values()),
                      "max_abs_median_z_difference": max((c_["max_abs_median_z_difference"] for c_ in cells.values() if c_["max_abs_median_z_difference"] is not None), default=None),
                      "cells": cells}
    # normalisation constant actually used by swc.py / pd3_multistab.py (computed from the cache)
    post = np.load("/MALECNS_CACHE/post.npy"); w = np.load("/MALECNS_CACHE/weight.npy")
    ins = np.bincount(post, weights=w.astype(np.float64), minlength=167216)
    out["notes"] = {"c_swc": {"mean_in_strength_over_nonzero": float(ins[ins > 0].mean()), "n_nonzero": int((ins > 0).sum()),
                              "total_over_N": float(ins.sum() / len(ins)), "source": "computed from the cache (post.npy, weight.npy)",
                              "comment": "swc.py and pd3_multistab.py use 1/mean over nonzero (745.91); docs quote 742.7 = total/N"}}
    out["answers"] = {
        "Q1_broken_symmetry": "none: 0/48 M1 points meet any of the 3 pre-registered criteria; one-lobe starts re-symmetrise in median 1.0-7.0 model time",
        "Q1_first_order_lines": "none: loop areas shrink with dwell to the noise floor (r = 0, h = 1e-5 still shrinking at D = 800); no coexistence "
                                "at any of 14 M2b loop centres; one non-silent attractor in all 23 surviving M3 cells; no coexistence in 40 QMF multistable cells (M3b)",
        "Q2e_onset_first_order": "no: no hysteresis or coexistence at the onset; h = 0 down-sweeps go extinct where tau = D (lifetime crossover)",
        "strength": "strong within B = 32-64, D <= 800, T <= 5000; cannot exclude sub-noise-floor hysteresis or metastability living > 5000"}
    json.dump(out, open(f"{ROOT}/results/pd3/multistability.json", "w"), indent=1)
    print("wrote results/pd3/multistability.json")


if __name__ == "__main__":
    main()
