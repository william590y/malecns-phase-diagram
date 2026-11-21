"""Assemble results/pd3/redraw_fss.json (program redraw_fss summary) from the two analysis outputs:
results/pd3/redraw/heatmap_meta.json (part A, pd3_redraw.py) and results/pd3/redraw_fss/fss_analysis.json (part B,
pd3_fss_analyze.py). No new computation except tabulation; every number keeps a pointer to its source file.
"""
import json, os
import numpy as np

ROOT = "/MALECNS_ROOT"
R = f"{ROOT}/results"
H = json.load(open(f"{R}/pd3/redraw/heatmap_meta.json"))
F = json.load(open(f"{R}/pd3/redraw_fss/fss_analysis.json"))
PJ = json.load(open(f"{R}/analysis2/phase_complete.json"))

out = {"program": "redraw_fss (results/pd3/plan.md section 2)", "date": "2026-11-09",
       "sources": {"A": "results/pd3/redraw/heatmap_meta.json (code/pd3_redraw.py)",
                   "B": "results/pd3/redraw_fss/fss_analysis.json (code/pd3_fss_analyze.py)",
                   "jobs": "results/pd3/redraw_fss/JOBS.md"}}
# ---------------- part A
cc = H["consistency_check"]
out["A_heatmap"] = {
    "grid": H["grid"], "n_rows": {k: len(H["rows"][k]) for k in H["rows"]},
    "n_simulation_files": len(H["inputs"]["simulation_files"]),
    "consistency_max_abs_dlng": {k: {"S_tau200": cc[k]["S"]["max_abs_dlng"], "on_tau3000": cc[k]["on"]["max_abs_dlng"],
                                     "n_rows_compared": cc[k]["S"]["n_compared"],
                                     "n_inside_total_band": {b: cc[k][b]["n_inside_total_band"] for b in ["S", "on"]},
                                     "rows_with_more_than_one_crossing": sum(z["n_crossings_on_row"] > 1 for z in cc[k]["S"]["rows"] + cc[k]["on"]["rows"])}
                                 for k in cc},
    "consistency_threshold_one_grid_step": H["grid"]["step_ln_g"],
    "consistency_pass": all(cc[k][b]["max_abs_dlng"] <= H["grid"]["step_ln_g"] for k in cc for b in ["S", "on"]),
    "sustained_cells": {k: H[f"sustained_cells_{k}"] for k in ["real", "null", "wnull"]},
    "chain_T_drift_r1": {kk: H["chain_drift_r1"][kk] for kk in ["ours_at_3000", "ours_at_30000", "theory_bd_on_tau3000", "theory_bd_on_tau30000", "K"]},
    "longruns_g_half_r1": H["longruns_g_half_r1"], "g_lin": H["g_lin"]}
# ---------------- part B
pg = F["per_graph"]
tab = []
for e in pg:
    tab.append({"kind": e["kind"], "N": e["N"], "seed": e["seed"], "r": e["r"], "g_lin": e["diag"]["g_lin"],
                "lambda_PF_E": e["diag"]["lambda_PF_E"], "K_nc_E": e["diag"]["K_nc_E"], "K_nc_glin": e["diag"]["K_nc_glin"],
                "inv_ipr_glin": e["diag"]["inv_ipr_glin"], "al_share_v2_glin": e["diag"]["al_share_v2_glin"],
                "n_points": e["n_points"], "n_fine_points": e["n_fine_points"],
                "x_S": e["S"]["x"], "x_S_total": e["S"]["total"], "x_on": e["on"]["x"], "x_on_total": e["on"]["total"],
                "x_half": e["half"]["x"], "s_fit": e["s_fit"], "s1000": e["s1000"], "tau_x1": e["tau_x1"],
                "K_fit": e["K_fit"], "K_fit_n_points": e["K_fit_n_points"], "K_fit_rms_dlntau": e["K_fit_rms_dlntau"],
                "chain_Knc_x_on": e["chain_Knc"]["x_on"], "chain_Knc_tau_x1": e["chain_Knc"]["tau_x1"],
                "chain_Kfit": e.get("chain_Kfit"), "chi_peak": e["chi_peak"], "validation": e["diag"]["validation"],
                "gpu": e["gpu"], "files": e["file"]})
out["B_fss"] = {"per_graph": tab, "seed_means": F["seed_means"], "decisions": F["decisions"], "inputs": F["inputs"],
                "chain_s_window_vs_K": {k: v for k, v in F.items() if k.startswith("chain_s_window")},
                "c4_points_used": F.get("c4_points_used")}
# supplementary (not pre-registered): extrapolation of the null onset, x_on(N) = x_inf + A N^-b, on the seed means
from scipy.optimize import curve_fit
ext = {}
for r in [0, 1]:
    rows = F["seed_means"][f"cnull_r{r}"]
    N = np.array([z["N"] for z in rows], float); xm = np.array([z["x_on_mean"] for z in rows])
    se = np.array([z["x_on_seed_sd"] / np.sqrt(len(z["seeds"])) for z in rows])
    se = np.maximum(se, 1e-3)                                    # floor: 0.1% in x
    f3 = lambda n, xi, A, b: xi + A * (n / 1e4) ** (-b)
    try:
        p3, c3 = curve_fit(f3, N, xm, p0=[1.0, xm[2] - 1, 0.4], sigma=se, absolute_sigma=True, maxfev=20000)
        e3 = {"x_inf": float(p3[0]), "x_inf_se": float(np.sqrt(c3[0, 0])), "A_at_1e4": float(p3[1]), "b": float(p3[2]),
              "b_se": float(np.sqrt(c3[2, 2])), "chi2": float(np.sum(((xm - f3(N, *p3)) / se) ** 2)), "dof": len(N) - 3}
    except Exception as ex:
        e3 = {"error": str(ex)}
    sl = np.polyfit(np.log(N), np.log(xm - 1), 1)
    ext[f"r{r}"] = {"three_param_fit": e3, "loglog_slope_x_on_minus_1_vs_N (x_inf = 1)": float(sl[0]),
                    "sigma": "between-seed SE of the mean (floor 1e-3)", "N": N.tolist(), "x_on_mean": xm.tolist()}
out["B_fss"]["supplementary_onset_extrapolation"] = ext
out["B_fss"]["supplementary_chi_win"] = F.get("supplementary_chi_win")
val = [t["validation"] for t in tab if t["validation"]]
out["B_fss"]["validation_summary"] = {
    "n_graphs_validated": len(val),
    "max_rel_dev_lambda": max(v["rel_dev_lambda"] for v in val), "max_rel_dev_K": max(v["rel_dev_K"] for v in val),
    "max_rel_dev_c": max(v["rel_dev_c"] for v in val), "all_edge_counts_equal": all(v["edges_equal"] for v in val),
    "full_null_seed0_g_lin": next(t["g_lin"] for t in tab if t["kind"] == "cnull" and t["N"] == 167216 and t["seed"] == 0),
    "phase_complete_g_lin_null": PJ["g_lin"]["null"]}
json.dump(out, open(f"{R}/pd3/redraw_fss.json", "w"), indent=1, default=float)
print("saved results/pd3/redraw_fss.json", out["B_fss"]["validation_summary"])
