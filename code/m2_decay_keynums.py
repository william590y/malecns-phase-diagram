"""m2_decay_keynums.py -- collect the numbers quoted in results/m2/decay/REPORT.md into key_numbers.json (lane m2/decay).
Reads boundaries.json, points.json, where.json, osc.json, lifetimes.json, osccheck.json (all written by this lane) and
sacct for the GPU time of the jobs listed in JOBS.md."""
import json, os, re, subprocess
import numpy as np

OUT = "/MALECNS_ROOT/results/m2/decay"


def jl(n):
    p = f"{OUT}/{n}"
    return json.load(open(p)) if os.path.exists(p) else {}


def num(x):
    if x is None:
        return np.nan
    if isinstance(x, str):
        return np.inf if x == "inf" else np.nan
    return float(x)


def rng_(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(min(v)), float(max(v))] if v else None


def main():
    B = jl("boundaries.json"); W = jl("where.json"); OS = jl("osc.json"); LT = jl("lifetimes.json"); OC = jl("osccheck.json")
    rows = B["rows"]; comp = B["m1_comparison"]
    K = {}
    # main diagram
    real = sorted([e for e in rows.values() if e["model"] == "M2" and e["graph"] == "real"], key=lambda e: e["r"])
    tab = []
    for e in real:
        c = comp.get(f"M2_real_r{e['r']:g}", {})
        tab.append({"r": e["r"], "S": e["S"]["g"], "S_total": e["S"]["total"], "on": e["on"]["g"], "on_total": e["on"]["total"],
                    "half": e["half"]["g"], "D": e["D"]["g"], "D_total": e["D"]["total"], "D_status": e["D"]["status"],
                    "on_over_S": e.get("window_on_over_S"), "D_over_on": e.get("D_over_on"),
                    "M1_S": c.get("S", {}).get("M1"), "M1_on": c.get("on", {}).get("M1"), "M1_D": c.get("D", {}).get("M1"),
                    "ratio_S": c.get("S", {}).get("ratio"), "ratio_on": c.get("on", {}).get("ratio"),
                    "ratio_D": c.get("D", {}).get("ratio"), "n_points": e["n_points"]})
    K["M2_real_table"] = tab
    K["M2_real_ranges"] = {k: rng_([num(x[k]) for x in tab]) for k in ("S", "on", "D", "on_over_S", "D_over_on", "ratio_S",
                                                                       "ratio_on", "ratio_D")}
    K["M2_real_D_status"] = {f"{x['r']:g}": x["D_status"] for x in tab}
    m1w = [num(B["m1_reference"][f"M1ref_real_r{x['r']:g}"]["on"]["g"]) / num(B["m1_reference"][f"M1ref_real_r{x['r']:g}"]["S"]["g"])
           for x in tab if f"M1ref_real_r{x['r']:g}" in B["m1_reference"]]
    K["M1_real_on_over_S_range"] = rng_(m1w)
    # nulls
    nl = {}
    for e in sorted(rows.values(), key=lambda e: (e["graph"], e["r"])):
        if e["model"] == "M2" and e["graph"] != "real":
            m1r = B["m1_reference"].get(f"M1ref_{e['graph']}_r{e['r']:g}")
            nl[f"{e['graph']}_r{e['r']:g}"] = {"S": e["S"]["g"], "on": e["on"]["g"], "on_total": e["on"]["total"],
                                               "D": e["D"]["g"], "D_status": e["D"]["status"],
                                               "M1_on": m1r["on"]["g"] if m1r else None, "M1_D": m1r["D"]["g"] if m1r else None,
                                               "M1_D_status": m1r["D"]["status"] if m1r else None,
                                               "real_M2_on_same_r": rows.get(f"M2_real_r{e['r']:g}", {}).get("on", {}).get("g")}
    K["nulls"] = nl
    K["variants"] = B["variants"]
    K["m1_pipeline_check"] = B["m1_pipeline_check"]
    # where
    wh = {}
    for k, w in W.items():
        e = rows.get(k)
        if not e:
            continue
        wh[k] = {b: {"g": w[b]["g"], "PRN": w[b]["PRN"], "PR": w[b].get("PR"), "AL_module": w[b]["region_share"]["AL_module"],
                     "lLN1_bc": w[b]["region_share"].get("lLN1_bc"), "ORN": w[b]["region_share"].get("ORN"),
                     "top": [(s["superclass"], round(s["share"], 3), round(s["enrichment"], 2)) for s in w[b]["top_superclasses"]]}
                 for b in ("S", "on", "D") if b in w}
    K["where"] = wh
    real_on = [wh[k]["on"] for k in wh if k.startswith("M2_real_r") and "on" in wh[k]]
    K["where_M2_real_first_sustained_ranges"] = {"PRN": rng_([x["PRN"] for x in real_on]),
                                                 "AL_module": rng_([x["AL_module"] for x in real_on]),
                                                 "PR_neurons": rng_([x["PRN"] * 167216 for x in real_on])}
    real_D = [wh[k]["D"] for k in wh if k.startswith("M2_real_r") and "D" in wh[k]]
    K["where_M2_real_first_D_ranges"] = {"AL_module": rng_([x["AL_module"] for x in real_D])}
    # oscillation
    o = OS.get("rows", [])
    K["osc"] = {"n_pairs": len(o), "n_osc": sum(x["osc"] for x in o), "n_strong": sum(x["osc_strong"] for x in o),
                "rows_with_osc": sorted({x["row"] for x in o if x["osc"]}),
                "osc_points": [{k: x[k] for k in ("row", "g", "signal", "n_rep", "c_min", "c_peak", "period", "coh_ratio", "cv_t",
                                                  "osc_strong", "phase")} for x in o if x["osc"]],
                "deepest_trough": min((num(x["c_min"]) for x in o if np.isfinite(num(x["c_min"]))), default=None),
                "n_trough_le_m01": sum(1 for x in o if num(x["c_min"]) <= -0.1)}
    by_model = {}
    for x in o:
        m = x["row"].split("_")[0] if not x["row"].startswith("M2_") else "_".join(x["row"].split("_")[:2])
        by_model.setdefault(x["row"].rsplit("_r", 1)[0], []).append(x["osc"])
    K["osc_by_rowfamily"] = {k: [int(sum(v)), len(v)] for k, v in by_model.items()}
    K["osccheck"] = {k: {"median_slope": v.get("median_slope"), "verdict": v.get("verdict"), "N": v.get("N"),
                         "osc_points": v.get("osc_points")} for k, v in OC.get("families", {}).items()}
    # lifetimes
    K["lifetime_verdicts"] = LT.get("long", {}).get("verdict")
    lg = LT.get("long", {}).get("per_g", {})
    K["lifetime_long"] = {rk: [{"g": x["g"], "tau": x["tau"], "KS_p_shift": (x["ks_shift"] or {}).get("KS_p"),
                                "CV": (x["ks_shift"] or {}).get("CV"), "broad": x["broad"], "signature": x["signature"],
                                "early200": x["post_hoc"]["late200"]["early_frac"],
                                "late200_KS_p": x["post_hoc"]["late200"].get("KS_p"),
                                "tau_late200": x["post_hoc"]["late200"].get("tau_late")} for x in L] for rk, L in lg.items()}
    sc = LT.get("screen", {})
    K["lifetime_screen_totals"] = {
        "rows": len(sc), "points": sum(v["n"] for v in sc.values()),
        "rejections_shift": sum(v["n_reject_05"] for v in sc.values()),
        "late200_points": sum(v["late200_n"] for v in sc.values()),
        "late200_rejections": sum(v["late200_n_reject_05"] for v in sc.values())}
    # GPU time
    jobs = re.findall(r"^\| (\d+) \| (m2-decay[^ ]*) ", open(f"{OUT}/JOBS.md").read(), re.M)
    tot = 0.0; per = {}
    for j, name in jobs:
        q = subprocess.run(["sacct", "-j", j, "-X", "-n", "-S", "2026-11-09T11:00", "-o", "elapsedraw,alloctres%60"],
                           capture_output=True, text=True).stdout.splitlines()
        s = 0.0
        for line in q:
            parts = line.split()
            if parts and parts[0].isdigit() and "gres/gpu" in line:
                s += int(parts[0]) / 3600
        per[f"{j} {name}"] = round(s, 2); tot += s
    K["gpu_hours"] = {"total": round(tot, 1), "per_job": per}
    json.dump(K, open(f"{OUT}/key_numbers.json", "w"), indent=1, default=float)
    print(json.dumps({k: K[k] for k in ("M2_real_ranges", "M1_real_on_over_S_range", "where_M2_real_first_sustained_ranges",
                                        "where_M2_real_first_D_ranges", "lifetime_verdicts", "lifetime_screen_totals", "gpu_hours")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
