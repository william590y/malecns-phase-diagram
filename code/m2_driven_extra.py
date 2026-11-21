"""m2_driven_extra.py -- extra numbers for REPORT.md of lane m2/driven (reads curves.json, points/*.json, run sidecars).
Writes results/m2/driven/extra_numbers.json."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_common as M
RES = f"{M.ROOT}/results/m2/driven"
C = json.load(open(f"{RES}/curves.json"))
P = {}
for f in glob.glob(f"{RES}/points/*/*.json"):
    for p in json.load(open(f))["points"]:
        P[(p["set"], p["model"], p["graph"], p["orn"], p["r"], p["palp"], round(p["g"], 5))] = p
out = {"where": {}, "peak_heights": {}, "gain_at_top": {}, "cost": {}}
def near(c, g0):
    g = np.array(c["g"]); return float(g[int(np.argmin(np.abs(np.log(g / g0))))])
for key in ("A1|M2|real|0.04", "A2|M1|real|0.04", "A4|M2s|real|0.04"):
    for r in (0, 1, 4):
        c = C.get(f"{key}|{r}|False")
        if c is None:
            continue
        s, m, gr, o = key.split("|")
        L = c["lines"]
        pts = {}
        for lab, g0 in (("mid_allowed", np.sqrt(L["allowed_lo"] * L["allowed_hi"])), ("allowed_top", L["allowed_hi"]),
                        ("DT_SS", L["DT_SS"]), ("x2_DT_SS", 2 * L["DT_SS"])):
            g = near(c, g0)
            p = P[(s, m, gr, float(o), r, False, round(g, 5))]
            pts[lab] = {"g": g, "phase": p["phase"], "G": p["G"], "rho_ORN": p["rho_ORN"], "rho_nonORN": p["rho_nonORN"],
                        "n_active_nonORN": p["n_active_nonORN"], "PR_non_n": p["PR_non_n"], "AL_module_share": p["mod_share"],
                        "top_superclasses[name, share, enrichment]": p["top_superclasses"][:3],
                        "x_AL_module": p.get("x_AL_module"), "xact_ORN": p.get("xact_ORN"), "tau_int": p["tau_int"],
                        "F_AL": p["F_AL"], "F_non": p["F_non"]}
        out["where"][f"{key}|r{r}"] = pts
for k, c in C.items():
    kk = c["key"]
    if kk["palp"]:
        continue
    cv = c["curve"]; L = c["lines"]
    rec = {}
    for ob in ("F_AL", "F_non", "tau_int"):
        y = np.array([np.nan if v is None else v for v in cv[ob]], float)
        rec[ob] = {"max": float(np.nanmax(y)), "g_at_max": float(c["g"][int(np.nanargmax(y))]), "at_g_min": float(y[0]),
                   "at_DT_SS": float(y[int(np.argmin(np.abs(np.log(np.array(c["g"]) / L["DT_SS"]))))]) if L.get("DT_SS") else None,
                   "local_maxima": c.get("local_maxima", {}).get(ob)}
    out["peak_heights"][k] = rec
    if L.get("allowed_hi"):
        g = np.array(c["g"]); a = np.array([bool(x) for x in cv["allowed"]])
        if a.any():
            Gs = np.array(cv["G"], float)[a]
            out["gain_at_top"][k] = {"G_max_in_allowed": float(Gs.max()), "G_at_last_allowed": float(Gs[-1]),
                                     "rho_ORN_range_in_allowed": [float(np.min(np.array(cv["rho_ORN"], float)[a])),
                                                                  float(np.max(np.array(cv["rho_ORN"], float)[a]))]}
# GPU cost from the sidecars
tot = 0.0; n = 0; gpus = {}
for f in glob.glob(f"{RES}/runs/*/*.json"):
    s = json.load(open(f)); tot += s.get("wall_s", 0); n += 1
    gpus[s.get("gpu")] = gpus.get(s.get("gpu"), 0) + 1
out["cost"] = {"runs": n, "gpu_hours_completed_runs": tot / 3600, "points": sum(len(c["g"]) for c in C.values()),
               "gpus": gpus, "note": "wall time of completed runs only; preempted partial runs not counted"}
json.dump(out, open(f"{RES}/extra_numbers.json", "w"), indent=1, default=float)
print(json.dumps(out["cost"], indent=1))
for k, v in out["where"].items():
    print(k)
    for lab, p in v.items():
        print("  ", lab, round(p["g"], 3), p["phase"], "G", round(p["G"], 3), "n_act", round(p["n_active_nonORN"]), "PRn", round(p["PR_non_n"], 3),
              "ALshare", round(p["AL_module_share"], 3), p["top_superclasses[name, share, enrichment]"][:2], "xAL", p["x_AL_module"] and round(p["x_AL_module"], 3))
for k in ("A1|M2|real|0.04|0|False", "A1|M2|real|0.04|1|False", "A1|M2|real|0.04|4|False", "A2|M1|real|0.04|1|False", "A3|M2|cnull|0.04|1|False"):
    print(k, {ob: (round(v["max"], 2), round(v["g_at_max"], 3), round(v["at_g_min"], 2), v["at_DT_SS"] and round(v["at_DT_SS"], 2)) for ob, v in out["peak_heights"][k].items()})
    print("   localmax F_AL", [(round(x["g"], 3), round(x["value"], 2), x["phase"]) for x in out["peak_heights"][k]["F_AL"]["local_maxima"] or []])
for k in ("A1|M2|real|0.04|0|False", "A1|M2|real|0.04|1|False", "A1|M2|real|0.04|6|False", "A2|M1|real|0.04|1|False"):
    print(k, out["gain_at_top"].get(k))
