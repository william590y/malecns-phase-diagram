"""pd3 module_theory: collect every number used in results/pd3/module_theory.md into results/pd3/module_theory.json (CPU).
Each block names the file it was read from ('source'); nothing is computed here except medians / counts of those files' values.
"""
import json, os
import numpy as np

ROOT = "/MALECNS_ROOT"; R = f"{ROOT}/results/pd3"
J = lambda p: json.load(open(p))
out = {"program": "module_theory (plan.md section 3)", "date": "2026-11-09", "jobs": "results/pd3/module_theory/JOBS.md"}

# ---------- MT1 ----------
p0 = np.load(f"{R}/modules/partition_P0.npz", allow_pickle=True); m0 = json.loads(str(p0["meta_json"]))
tab = J(f"{R}/modules/modules_table.json"); lr = J(f"{R}/modules/leiden_runs.json")
mt1 = {"source": ["results/pd3/modules/partition_P0.npz (meta_json)", "results/pd3/modules/modules_table.json",
                  "results/pd3/modules/leiden_runs.json", "results/pd3/modules/quotient_<P>.npz", "results/pd3/modules/deadzone_<P>.json"],
       "P0_partition": {"n_modules": m0["n_modules"], "sizes": dict(zip([str(x) for x in p0["names"]], p0["size"].tolist())),
                        "skipped_AL_modes_first6": m0["skipped_AL_modes_first6"], "n_modes_processed": m0["n_modes_processed"],
                        "nonAL_modes": m0["nonAL_modes_k>=3"],
                        "share_nonAL_modes_support_gt50pct_inside_mode2_support": m0["share_nonAL_modes_with_support_gt50pct_inside_mode2_support"],
                        "median_share_inside_mode2_support": m0["median_share_inside_mode2_support"]},
       "P1_leiden": {"choice": lr["choice"], "per_gamma": lr["per_gamma"],
                     "chosen_run": [r for r in lr["runs"] if r["gamma"] == lr["choice"]["gamma"] and r["seed"] == lr["choice"]["seed"]][0]},
       "P2": {"n_groups": lr["P2_n_groups"], "nmi_P2_P0": lr["P2_nmi_vs_P0"], "nmi_P1_P2": lr["P1_nmi_vs_P2"]}}
for P in ["P0", "P1", "P2"]:
    rows = tab[P]["rows"]
    q = np.load(f"{R}/modules/quotient_{P}.npz", allow_pickle=True)
    off = ~np.eye(len(q["CV_E"]), dtype=bool)
    gl = np.array([r["g_lin"] for r in rows[:-1]])
    dz = J(f"{R}/modules/deadzone_{P}.json")
    pairs = {(p["k"], p["l"]): p for p in dz["pairs"]}
    mutual = sorted({tuple(sorted(k)) for k, p in pairs.items() if p["candidate_meaningful"] and (k[1], k[0]) in pairs
                     and pairs[(k[1], k[0])]["candidate_meaningful"]})
    mt1[P] = {"rows": [{k: r[k] for k in ["name", "size", "Lambda", "g_weak", "g_lin", "K", "r0_PW", "r0_UN", "superclass_mix"]}
                       for r in rows],
              "n_top30_g_lin_below": {str(x): int((gl < x).sum()) for x in [0.3, 0.5, 1, 1.47, 2, 3, 30]},
              "equitability_CV_median": {"E_self": float(np.nanmedian(np.diag(q["CV_E"]))), "E_cross": float(np.nanmedian(q["CV_E"][off])),
                                         "I_self": float(np.nanmedian(np.diag(q["CV_I"]))), "I_cross": float(np.nanmedian(q["CV_I"][off]))},
              "deadzone_summary": {k: v for k, v in dz["summary"].items() if not k.startswith("candidates_plain")},
              "deadzone_mutual_meaningful_pairs": mutual}
    if P == "P0":
        al = {f"{a}<-{b}": {k: pairs[(a, b)][k] for k in ["share_all", "share_u", "share_recv", "r1_PW", "r0_PW_k", "r1_UN", "r0_UN_k"]}
              for a, b in [("AL-L", "AL-R"), ("AL-R", "AL-L"), ("AL-L", "rest"), ("rest", "AL-L")]}
        mt1[P]["AL_pairs"] = al
        n = [str(x) for x in q["names"]]; iL, iR = n.index("AL-L"), n.index("AL-R")
        mt1[P]["AL_PW"] = {"E": q["E_PW"][np.ix_([iL, iR], [iL, iR])].tolist(), "I": q["I_PW"][np.ix_([iL, iR], [iL, iR])].tolist(),
                           "CV_E_LL": float(q["CV_E"][iL, iL]), "CV_E_LR": float(q["CV_E"][iL, iR])}
out["MT1"] = mt1

# ---------- MT2 ----------
mt2 = {"source": ["results/pd3/modules/coarse_mf_<P>.json", "results/pd3/modules/coarse_mf_validation.json",
                  "results/pd3/modules/coarse_post_<P>.{json,npz} (corrected census, use these)"],
       "validation": J(f"{R}/modules/coarse_mf_validation.json")}
for P in ["P0", "P1"]:
    f = f"{R}/modules/coarse_mf_{P}.json"
    if os.path.exists(f):
        js = J(f); d = np.load(f"{R}/modules/coarse_mf_{P}.npz", allow_pickle=True)
        m_up = (d["up"] > 1e-3).sum(2)
        mt2[P] = {k: js[k] for k in ["n_starts_per_pass", "g_SL_edge", "max_n_attractors", "n_cells_multistable", "n_cells",
                                     "n_cells_with_unconverged", "g_range_cells_with_unconverged", "dtcheck_r1"] if k in js}
        mt2[P]["n_first_order_sweep_boundaries"] = len(js["first_order_boundaries"])
        mt2[P]["first_order_boundaries"] = js["first_order_boundaries"][:80]
        mt2[P]["segments_r"] = {r: js["segments"][r] for r in ["0", "1", "2", "2.5", "4", "6"] if r in js["segments"]}
        mt2[P]["multistable_r_range"] = [float(d["r"][np.where((d["n_attr"] > 1).any(1))[0]].min()),
                                         float(d["r"][np.where((d["n_attr"] > 1).any(1))[0]].max())] if (d["n_attr"] > 1).any() else None
        mt2[P]["multistable_g_range"] = [float(d["g"][np.where((d["n_attr"] > 1).any(0))[0]].min()),
                                         float(d["g"][np.where((d["n_attr"] > 1).any(0))[0]].max())] if (d["n_attr"] > 1).any() else None
        mt2[P]["n_active_modules_up_at_g30_by_r"] = {f"{r:g}": int(m_up[i, -1]) for i, r in enumerate(d["r"])}
        mt2[P]["multistable_examples"] = js["multistable_cells_examples"][:20]
        cp = J(f"{R}/modules/coarse_post_{P}.json")
        mt2[P]["corrected"] = {k: cp[k] for k in ["n_SL_entries_dropped_above_edge", "n_cells_multistable_corrected",
                                                  "max_n_attractors_corrected", "n_cells_no_stable_attractor",
                                                  "n_cells_no_stable_attractor_and_nonstationary", "multistable_g_range",
                                                  "n_hysteresis_intervals", "SL_coexisting_with_active_cells",
                                                  "first_nonstationary_g_by_r", "n_steep_jumps_without_hysteresis"]}
        mt2[P]["corrected"]["hysteresis_intervals"] = [{k: h[k] for k in ["r", "g_lo", "g_hi", "n_points", "max_Linf"]} for h in cp["hysteresis_intervals"]]
        mt2[P]["corrected"]["segments_r"] = {r: cp["segments_corrected"][r][:10] for r in ["0", "1", "2", "2.5", "4", "6"]}
        pz = np.load(f"{R}/modules/coarse_post_{P}.npz", allow_pickle=True); labs = pz["labels_corr"]; nat = pz["n_attr_corr"]
        mt2[P]["corrected"]["max_attractors_by_r"] = {f"{x:g}": int(nat[i].max()) for i, x in enumerate(d["r"])}
        bs = [(float(d["r"][ri]), float(d["g"][gi])) for ri in range(len(d["r"])) for gi in range(len(d["g"]))
              if any("AL-L" in x and "AL-R" not in x for x in labs[ri, gi].split("|"))
              and any("AL-R" in x and "AL-L" not in x for x in labs[ri, gi].split("|"))]
        mt2[P]["corrected"]["AL_lobe_broken_cells"] = len(bs)
        mt2[P]["corrected"]["AL_lobe_broken_first_g_by_r"] = {f"{x:g}": min(g_ for r_, g_ in bs if r_ == x) for x in sorted(set(r_ for r_, _ in bs))}
out["MT2"] = mt2

# ---------- MT3 ----------
qa = J(f"{R}/module_theory/qmf_analysis.json")
mt3 = {"source": ["results/pd3/module_theory/qmf_r<r>.npz", "results/pd3/module_theory/qmf_analysis.json",
                  "results/pd3/module_theory/qmf_multi.json"], "summary": qa["summary"], "per_r": {}}
for r, a in qa["per_r"].items():
    mt3["per_r"][r] = {k: a[k] for k in ["MT3_multi", "max_n_fixed_point_attractors", "g_multistable_fixed_points", "MT3_first_order",
                                         "MT3_first_order_fixed_points_only", "hysteresis_points_by_type", "max_rel_up_down_diff",
                                         "g_at_max_rel_diff", "oscillation_range_all_active_pass1", "n_g_oscillating_pass1_all_active",
                                         "pass3_dt0.05_oscillating_g", "oscillating_composition_at_max", "SL_edge_up_sweep",
                                         "D_up_PR_quarter", "D_down_PR_quarter", "D_pass1_all_active_PR_quarter", "g_B1_like_AL_L_start"]}
    mt3["per_r"][r]["hysteresis_runs"] = [{k: h[k] for k in ["g_lo", "g_hi", "n_points", "max_rel_diff", "n_type_A_both_fixed_points",
                                                             "n_type_B_fixed_point_vs_nonstationary", "n_type_C_both_nonstationary"]}
                                          for h in a["hysteresis_runs"]]
    mt3["per_r"][r]["max_neurons_oscillating_all_active"] = int(max(a["n_neurons_oscillating_all_active"]))
if os.path.exists(f"{R}/module_theory/qmf_multi.json"):
    mm = J(f"{R}/module_theory/qmf_multi.json")["per_r"]
    mt3["multi_diag"] = {r: {"n_g": v["n_g"], "median_n_diff_0.05": v["median_n_diff_0.05"], "max_n_diff_0.05": v["max_n_diff_0.05"],
                             "max_mean_check": v["max_mean_check"],
                             "max_n_components": int(max(p["n_components_0.05"] for p in v["pairs"])) if v["pairs"] else None,
                             "pairs": [{k: p[k] for k in ["g", "A", "B", "n_diff_0.05", "n_components_0.05", "largest_component_0.05",
                                                         "by_module", "by_superclass", "by_sign"]} for p in v["pairs"]]}
                         for r, v in mm.items()}
    # lobe asymmetry A = (X_L - X_R)/(X_L + X_R) of the coexisting fixed points (pass-1 module means of the representatives)
    asym = {}
    for r, v in mm.items():
        d = np.load(f"{R}/module_theory/qmf_r{r}.npz", allow_pickle=True); st = [str(x) for x in d["starts"]]
        mods = [str(x) for x in d["modules"]]; iL, iR = mods.index("AL-L"), mods.index("AL-R"); g1 = d["g1"]
        rows_ = {}
        for p_ in v["pairs"]:
            gi = int(np.argmin(np.abs(g1 - p_["g"])))
            for s_ in (p_["A"], p_["B"]):
                XL, XR = d["p1_X"][st.index(s_), gi, iL], d["p1_X"][st.index(s_), gi, iR]
                rows_.setdefault(f"{p_['g']:.4f}", {})[s_] = {"X_AL_L": float(XL), "X_AL_R": float(XR), "A": float((XL - XR) / (XL + XR))}
        Avals = [x["A"] for gg in rows_.values() for x in gg.values()]
        asym[r] = {"per_g": rows_, "A_min": float(min(Avals)), "A_max": float(max(Avals)),
                   "max_A_spread_within_g": float(max(max(x["A"] for x in gg.values()) - min(x["A"] for x in gg.values()) for gg in rows_.values()))}
    mt3["AL_lobe_asymmetry_of_coexisting_states"] = asym
out["MT3"] = mt3

# ---------- MT4 ----------
occ = J(f"{R}/modules/module_occ_P0.json")
key = ["AL-L", "AL-R", "isl1", "isl2", "isl3", "isl5", "rest"]
mt4 = {"source": "results/pd3/modules/module_occ_P0.json", "files_used": len(occ["files_used"]), "sim": {}, "qmf": {}}
for r, s in occ["sim"].items():
    g = np.array(s["g"]); X = np.array(s["X_among_survivors"]); ok = np.array(s["surv2"]) >= 0.5
    isl = [k for k in occ["modules"] if k.startswith("isl")]
    mt4["sim"][r] = {"cross_1pct": {k: s["cross_1pct_survivors"][k] for k in key},
                     "cross_10pct": {k: s["cross_10pct_survivors"][k] for k in key},
                     "median_island_cross_1pct": float(np.nanmedian([s["cross_1pct_survivors"][k] or np.nan for k in isl])),
                     "n_islands_reaching_1pct": int(sum(1 for k in isl if s["cross_1pct_survivors"][k])),
                     "max_n_modules_ge_1pct": int((X[ok] >= 0.01).sum(1).max()) if ok.any() else 0}
for r, s in occ["qmf"].items():
    isl = [k for k in occ["modules"] if k.startswith("isl")]
    mt4["qmf"][r] = {"cross_1pct": {k: s["cross_1pct"][k] for k in key},
                     "median_island_cross_1pct": float(np.nanmedian([s["cross_1pct"][k] or np.nan for k in isl]))}
out["MT4"] = mt4

# ---------- MT5 ----------
il = J(f"{R}/module_theory/island_lifetimes.json")
out["MT5"] = {"source": "results/pd3/module_theory/island_lifetimes.json",
              "rows": {k: {kk: v[kk] for kk in ["n", "Lambda", "g_lin", "tau_x1", "x_tau200", "x_tau3000", "g_tau200", "g_tau3000"]}
                       for k, v in il.items()}}
json.dump(out, open(f"{R}/module_theory.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("saved results/pd3/module_theory.json")
