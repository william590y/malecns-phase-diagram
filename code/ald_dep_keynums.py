"""E3 key numbers -> results/aldef/depression/key_numbers.json (every number quoted in the E3 summary comes from here or from
summary.json / points.json / meso.json / validation/*.json, all written by code/ald_dep_*.py)."""
import json
import numpy as np

OUT = "/MALECNS_ROOT/results/aldef/depression"
S = json.load(open(f"{OUT}/summary.json")); P = json.load(open(f"{OUT}/points.json"))
A, D, TH = S["arms"], S["driven"], S["theory"]
M = json.load(open(f"{OUT}/meso.json"))
G_LIN = 0.16315274988719286
K = {"validation": {"unit_tests_e3": json.load(open(f"{OUT}/validation/unit_tests_e3.json"))["all_pass"],
                    "validate_e3": json.load(open(f"{OUT}/validation/validate_e3.json"))["all_pass"]}}


def b(arm, k):
    x = A[arm][k]
    return {"g": x["g"], "total": x.get("total"), "status": x["status"]}


K["boundaries"] = {arm: {"S": b(arm, "S"), "on": b(arm, "on"), "half": A[arm]["half"]["g"],
                         "on_ratio_vs_base": A[arm].get("on_ratio_vs_base"), "S_ratio_vs_base": A[arm].get("S_ratio_vs_base"),
                         "window_on_over_S": (A[arm]["on"]["g"] / A[arm]["S"]["g"]) if (A[arm]["on"]["g"] and A[arm]["S"]["g"]) else None}
                   for arm in A}


def at_first_sustained(arm):
    pts = [p for p in P[arm] if p["alive"] >= 0.5]
    if not pts:
        return None
    p = pts[0]; q = p["qs"]
    return {"g": p["g"], "alive": p["alive"], "AL_share": p["AL_share"], "PR_over_N": p["PR"], "AL_module_qs": q["AL_module"],
            "lLN1_bc_qs": q.get("lLN1_bc"), "CNS_rest_qs": q["CNS_rest"], "AL_over_CNS_rate": (q["AL_module"] / q["CNS_rest"]) if q["CNS_rest"] else None,
            "eff_active_AL": p.get("xact_AL")}


K["at_first_sustained"] = {arm: at_first_sustained(arm) for arm in A}


def point_at(arm, g):
    for p in P[arm]:
        if abs(p["g"] / g - 1) < 2e-3:
            return p
    return None


ref = {}
for g in (0.378, 0.6, 0.9609, 1.948, 3.119):
    row = {}
    for arm in A:
        p = point_at(arm, g)
        if p:
            row[arm] = {"tau": p["tau"], "tau_ci": p["tau_ci"], "alive": p["alive"],
                        "AL_module_qs": (p.get("qs") or {}).get("AL_module"), "AL_share": p["AL_share"]}
    ref[str(g)] = row
K["at_reference_g"] = ref

# driven linear response: inverse-variance mean ratio (arm / base) of AL-module activity for g <= 0.155 (< g_lin)
lin = {}
for h in ("1e-5", "1e-4"):
    bb = D.get(f"drv_base_h{h}_r1")
    for arm in ("sp_NW16", "sp_KW09", "rt_NW16"):
        d = D.get(f"drv_{arm}_h{h}_r1")
        if not (bb and d):
            continue
        g = np.array(d["g"]); x, sx = np.array(d["AL_module"]["mean"]), np.array(d["AL_module"]["se"])
        y, sy = np.array(bb["AL_module"]["mean"]), np.array(bb["AL_module"]["se"])
        r = x / y; se = r * np.sqrt((sx / x) ** 2 + (sy / y) ** 2)
        out = {}
        for lab, sel in (("g<=0.155", g <= 0.156), ("0.17<=g<=0.20", (g >= 0.17) & (g <= 0.2)), ("g>=0.27", g >= 0.27)):
            w = 1 / se[sel] ** 2
            out[lab] = {"ratio": float((r[sel] * w).sum() / w.sum()), "se": float(1 / np.sqrt(w.sum())), "n_g": int(sel.sum())}
        out["gain_at_g0.30"] = {"arm": float(np.interp(0.30, g, d["AL_module"]["gain"])),
                                "base": float(np.interp(0.30, g, bb["AL_module"]["gain"]))}
        lin[f"{arm}_h{h}"] = out
import sys
sys.path.insert(0, "/MALECNS_ROOT/code")
import ald_dep_theory as T
lin["phi_rate_mode_NW16(U .29, TAU 157)"] = float(T.phi_rate(0.29, 157))
lin["phi_rate_mode_KW09(U .34, TAU 240)"] = float(T.phi_rate(0.34, 240))
K["driven_linear_response"] = lin

K["removal_protocol"] = {arm: {"allowed_interval": D[arm]["allowed_interval"], "allowed_ratio_hi_lo": D[arm]["allowed_ratio_hi_lo"],
                               "allowed_grid_g": D[arm]["allowed_g"], "g_at_G_0.29": D[arm]["g_at_G_0.29"],
                               "g_at_G_1.45": D[arm]["g_at_G_1.45"], "g_at_p_alive_0.05": D[arm]["g_at_p_alive_0.05"],
                               "g_at_resid_0.1": D[arm]["g_at_resid_0.1"],
                               "G_at_g0.06": D[arm]["rows"][0]["G"],
                               "resid_100_300_at_g2.6": D[arm]["rows"][-1]["resid_100_300"]}
                         for arm in D if arm.startswith("rm_")}

th = {"kappa": TH["kappa"]}
for arm, c in TH["curves"].items():
    cal = c.get("cal", {}); raw = c.get("raw", {})
    th[arm] = {"on_frozen_cal": cal.get("on_frozen"), "on_adiabatic_cal": cal.get("on_adiabatic"),
               "S_frozen_cal": cal.get("S_frozen"), "S_adiabatic_cal": cal.get("S_adiabatic"),
               "on_frozen_raw": raw.get("on_frozen"), "on_adiabatic_raw": raw.get("on_adiabatic"),
               "X0": cal.get("X0"), "meso_on": M["arms"].get(arm, {}).get("on"), "meso_S": M["arms"].get(arm, {}).get("S"),
               "sim_on": A[arm]["on"]["g"], "sim_S": A[arm]["S"]["g"]}
for k, src in (("iso_base_ALonly_r1", "base_r1"), ("iso_sp_NW16_ALonly_r1", "sp_NW16_r1"), ("iso_sp_KW09_ALonly_r1", "sp_KW09_r1")):
    th[f"{k} (theory of {src})"] = {"sim_on_AL_alone": A[k]["on"]["g"], "sim_S_AL_alone": A[k]["S"]["g"],
                                     "on_frozen_cal": th[src]["on_frozen_cal"], "on_adiabatic_cal": th[src]["on_adiabatic_cal"],
                                     "meso_on": th[src]["meso_on"], "S_frozen_cal": th[src]["S_frozen_cal"], "meso_S": th[src]["meso_S"]}
K["theory_vs_sim"] = th
K["meso_base_check"] = {r: [{"g": x["g"], "ln_tau_meso": float(np.log(x["tau_reg"])), "ln_tau_exact": x.get("lt_exact")}
                            for x in M["arms"][f"base_r{r}"]["rows"][:8]] for r in (0, 1)}
K["isolation"] = {arm: {"S": A[arm]["S"]["g"], "on": A[arm]["on"]["g"], "on_total": A[arm]["on"].get("total")}
                  for arm in A if arm.startswith("iso_")}
json.dump(K, open(f"{OUT}/key_numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: K[k] for k in ("driven_linear_response", "removal_protocol", "isolation")}, indent=1, default=float)[:6000])
for k, v in K["theory_vs_sim"].items():
    if k != "kappa":
        print(k, {a: (round(b_, 3) if isinstance(b_, float) else b_) for a, b_ in v.items()})
