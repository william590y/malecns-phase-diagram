"""Collects the AL data-grounding results into results/al_lit/data_grounding.json (the numbers quoted in
data_grounding.md), plus two small extra checks: the Perron roots of the left and right AL separately (why the two
leading modes mix sides) and the bilateral-PN coupling between them."""
import json, glob
import numpy as np
from al_common import load, labels, exc_operator, pf, OUT, ROOT, ALPHA_EFF

G = OUT
mod = json.load(open(f"{G}/module.json")); E = json.load(open(f"{G}/edges.json"))
tail = json.load(open(f"{G}/tail20.json")); rec = json.load(open(f"{G}/recruit.json"))
peel = json.load(open(f"{G}/spec_peel.json"))
spec = {f.split("spec_")[1][:-5]: json.load(open(f)) for f in glob.glob(f"{G}/spec_*.json") if "peel" not in f}

# side-restricted AL Perron roots and the cross-side coupling
d = load(); L = labels()
al = L["reg"] == "AL"
side = {}
for s in ["L", "R"]:
    M, c = exc_operator(d, keep_node=al & (L["sides"] == s))
    side[s] = float(pf(M, k=3)[0].real[0])
pre, post, w, sg = d["pre"], d["post"], d["weight"], d["sign"]
m = (sg[pre] > 0) & al[pre] & al[post] & np.isin(L["sides"][pre], ["L", "R"]) & np.isin(L["sides"][post], ["L", "R"]) \
    & (L["sides"][pre] != L["sides"][post])
cross_types = {}
for t, ww in zip(L["ty"][pre[m]], w[m]):
    cross_types[t] = cross_types.get(t, 0) + float(ww)
cross_types = dict(sorted(cross_types.items(), key=lambda kv: -kv[1])[:8])
upn = L["cat"] == "AL uPN"
idx = np.load(f"{ROOT}/results/module/module_graph.npz")["idx"]

out = {
    "operator": "directed excitatory (ACh presyn), M[post,pre] = c*w, c = 1/742.7 (r = 0 simulation operator); "
                "lambda in these units; naive switch-on g ~ alpha_eff/lambda, alpha_eff = 0.906",
    "Q1_module": {
        "n": mod["n_module"], "by_category": mod["by_category"], "by_nt": mod["by_nt"], "by_side": mod["by_side"],
        "by_superclass": mod["by_superclass"], "top_types": mod["by_type_top25"],
        "hemilineage_top": dict(list(mod["by_hemilineage"].items())[:6]),
        "tracing_status": mod["by_status"],
        "uPN_glomeruli_in_module": len(mod["uPN_glomeruli_in_module"]), "uPN_glomeruli_cns": int(len(set(L["glom"][upn]))),
        "uPN_glomeruli_by_weight": mod["uPN_glomeruli_by_right1"],
        "ORN_total": mod["ORN_total"], "ORN_in_module": mod["ORN_in_module"], "ORN_glomeruli": mod["ORN_n_glomeruli"],
        "ORN_share_left1": mod["ORN_left1_total"], "ORN_share_right1": mod["ORN_right1_total"],
        "ORN_glomeruli_by_left1": mod["ORN_glomeruli_by_left1_top15"],
        "HRN_TRN_in_module": mod["HRN_TRN_in_module"], "LN_types": mod["LN_types"][:16],
        "ALLN_consensus_nt_cns": mod["ALLN_consensus_nt_cns"], "ALLN_ground_truth_cns": mod["ALLN_ground_truth_cns"],
        "module_exc_input_sources": mod["module_exc_input_sources"],
        "module_exc_output_targets": mod["module_exc_output_targets"],
        "module_input_by_sign": mod["module_input_by_sign"], "module_internal_by_sign": mod["module_internal_by_sign"],
        "mode_localization": {"right1": spec["full"]["modes"][0], "right2": spec["full"]["modes"][1],
                              "left1": spec["full"]["left_modes"][0]},
        "AL_side_perron": side, "AL_cross_side_exc_synapses_by_pre_type": cross_types,
    },
    "Q2_why": {
        "variants": {k: {"lam": v["lam"][:4], "lead_region": max(v["modes"][0]["region"], key=v["modes"][0]["region"].get),
                         "lead_share": max(v["modes"][0]["region"].values()), "inv_ipr": v["modes"][0]["inv_ipr"],
                         "top_cats": dict(list(v["modes"][0]["cat"].items())[:3]),
                         "best_nonAL": v.get("best_nonAL"),
                         "ratio_to_best_nonAL": (v["lam"][0] / v["best_nonAL"]["lam"]) if v.get("best_nonAL") and v["best_nonAL"]["j"] > 0 else None}
                     for k, v in spec.items()},
        "region_edge_stats": {k: v["stats"] for k, v in E["regions"].items()},
        "region_perron": {k: {kk: v[kk]["lam"] for kk in ["perron", "perron_cap50", "perron_binary"] if kk in v}
                          for k, v in E["regions"].items() if "perron" in v},
        "region_perron_top_types": {k: v["perron"].get("top_types") for k, v in E["regions"].items() if "perron" in v},
        "lambda_decomposition": E["decomposition"],
        "al_subgraphs": E["al_subgraphs"], "strongest": E["strong"],
    },
    "Q3_order": {"tail20": [{k: m[k] for k in ["k", "lam", "lam_im", "g_on_naive", "inv_ipr", "lead_region", "region", "types", "side"]}
                            for m in tail["modes"]],
                 "first_appearance": tail["first_appearance"],
                 "peel": [{"lam": s["lam"][0], "region": s["modes"][0]["region"], "types": s["modes"][0]["types"],
                           "removed": s["removed"]} for s in peel["steps"]],
                 "sim_switch_on_r0": rec["0.0"]["first_g_mode_ge_0.05"], "sim_switch_on_r1": rec["1.0"]["first_g_mode_ge_0.05"],
                 "sim_activity_share_r0": rec["0.0"]["activity_share"], "sim_activity_share_r1": rec["1.0"]["activity_share"]},
    "figures": sorted(glob.glob(f"{ROOT}/figs/al/*.png")),
    "code": sorted(glob.glob(f"{ROOT}/code/al_*")),
}
json.dump(out, open(f"{ROOT}/results/al_lit/data_grounding.json", "w"), indent=1, default=str)
print("side Perron", side, "cross-side", cross_types)
print("uPN glomeruli module/cns", out["Q1_module"]["uPN_glomeruli_in_module"], out["Q1_module"]["uPN_glomeruli_cns"])
