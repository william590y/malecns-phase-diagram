"""m2_nature_summary.py -- lane m2/nature: collect the key numbers and apply the verdict rules of plan section 7
(+ amendment 1) -> results/m2/nature/summary.json. Reads only this lane's JSON outputs (fss, dfss, hyst, aval,
driven_branch, osc), each named next to its numbers.
"""
import json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
from m2_nature_fss import jsafe

OUT = "/MALECNS_ROOT/results/m2/nature"
J = lambda n: json.load(open(f"{OUT}/{n}.json")) if os.path.exists(f"{OUT}/{n}.json") else None


def main():
    fss, dfss, hyst, aval, drv, osc = (J(n) for n in ("fss", "dfss", "hyst", "aval", "driven_branch", "osc"))
    S = {"sources": {n: f"results/m2/nature/{n}.json" for n in ("fss", "dfss", "hyst", "aval", "driven_branch", "osc")}}
    # ---------------- FSS
    T1 = {}
    for r in (0, 1, 4):
        row = {}
        for fam in ("cnull", "realsubu", "realsubcn", "realsub"):
            f = fss["families"].get(f"{fam}|r{r:g}") if fss else None
            if f:
                row[fam] = {"verdict": f["decision"]["verdict"], "decision": f["decision"],
                            "N": [x["N"] for x in f["sizes"]], "W": [x["W_mean"] for x in f["sizes"]],
                            "W_sd": [x["W_sd"] for x in f["sizes"]], "onset": [x["on_mean"] for x in f["sizes"]],
                            "S": [x["S_mean"] for x in f["sizes"]], "x_on": [x["x_on_mean"] for x in f["sizes"]],
                            "qs_at_on": [x.get("qs_at_on_mean") for x in f["sizes"]],
                            "chi_max": [x.get("chi_max_mean") for x in f["sizes"]]}
        # hysteresis at this r (M2, both protocols; pre-registered runs)
        hv = {}
        for proto in ("ramp_h0_down", "ramp_driven"):
            for model in ("M2", "M1"):
                v = hyst["verdicts"].get(f"{proto}|{model}|r{r:g}") if hyst else None
                if v:
                    hv[f"{proto}|{model}"] = {k: v.get(k) for k in ("verdict", "A", "p", "p99", "ratio_800_200",
                                                                     "ratio_3200_800", "bimodal_support",
                                                                     "bimodal_n_by_D", "posthoc_all_runs")}
        # POST HOC descriptive log-log slopes vs N (not pre-registered)
        desc = {}
        for fam in ("cnull", "realsubu"):
            f = row.get(fam)
            if not f:
                continue
            N = np.array(f["N"], float)
            def sl(y, sub=0.0):
                y = np.array([np.nan if v is None else v for v in y], float) - sub
                ok = np.isfinite(y) & (y > 0)
                return float(np.polyfit(np.log(N[ok]), np.log(y[ok]), 1)[0]) if ok.sum() >= 3 else None
            desc[fam] = {"slope_lnW_lnN": sl(f["W"]), "slope_ln_qs_at_on_lnN": sl(f["qs_at_on"]),
                         "slope_ln_xon_minus_1_lnN": sl(f["x_on"], 1.0) if fam == "cnull" else None,
                         "slope_lnW_lnN_N_ge_40k": (float(np.polyfit(np.log(N[N >= 40000]), np.log(np.array(f["W"], float)[N >= 40000]), 1)[0])
                                                    if (N >= 40000).sum() >= 3 else None)}
        row["posthoc_loglog_slopes"] = desc
        row["hysteresis"] = hv
        # plan section 7 + amendment 1
        fo = any(v["verdict"].startswith("first-order") for k, v in hv.items() if k.endswith("|M2")) or \
            any(v.get("bimodal_support") for k, v in hv.items() if k == "ramp_driven|M2")
        cn = row.get("cnull", {}).get("verdict") == "sharpening"
        ru = row.get("realsubu", {}).get("verdict")
        if fo:
            vv = "first-order"
        elif cn and ru == "sharpening":
            vv = "continuous transition, finite-size rounded"
        elif ru == "finite-K-rounded crossover":
            vv = "finite-K-rounded crossover"
        else:
            vv = f"inconclusive (cnull: {row.get('cnull', {}).get('verdict')}; realsubu: {ru})"
        row["verdict_T1_M2"] = vv
        T1[f"r{r:g}"] = row
    S["T1_onset"] = T1
    # hysteresis, all R_CORE
    if hyst:
        S["hysteresis_all"] = {k: {kk: v.get(kk) for kk in ("verdict", "ratio_800_200", "ratio_3200_800", "A800_above_floor",
                                                         "posthoc_all_runs")} for k, v in hyst["verdicts"].items()}
        S["hysteresis_counts"] = {}
        for k, v in hyst["verdicts"].items():
            proto, model, r = k.split("|")
            S["hysteresis_counts"].setdefault(f"{proto}|{model}", {}).setdefault(v["verdict"], 0)
            S["hysteresis_counts"][f"{proto}|{model}"][v["verdict"]] += 1
        S["hysteresis_missing"] = hyst["missing"]
    # driven FSS
    if dfss:
        S["driven_fss"] = {k: {"growth_exponent": v["growth_exponent"], "verdict_if_peak_tracks": v["verdict_if_peak_tracks"],
                               "sizes": v["sizes"]} for k, v in dfss["families"].items()}
    # avalanches
    if aval:
        cells = aval["cells"]
        S["avalanches"] = {"LHG_overall": aval["overall"], "LHG_counts": aval["verdict_counts"],
                           "n_critical_like": {m: int(sum(sum(c[m]["crit_like"]) for c in cells.values())) for m in ("M1", "M2")},
                           "posthoc_exp_only_span": {k: {m: c[m]["posthoc_exp_only_span_run"] for m in ("M1", "M2")}
                                                     for k, c in cells.items()},
                           "posthoc_first_bump_g": {k: {m: c[m]["posthoc_first_bump_g"] for m in ("M1", "M2")}
                                                    for k, c in cells.items()}}
        # exponent ranges where tau_S in [1.3, 1.7] and n_tail >= 1000 (gap 0)
        rng = {}
        for m in ("M1", "M2"):
            ks, gam, gp, mu = [], [], [], []
            for k, c in cells.items():
                if not k.endswith("|G0"):
                    continue
                x = c[m]
                for i, fl in enumerate(x["posthoc_exp_only"]):
                    if fl:
                        ks.append(x["ks"][i]); gam.append(x["gamma"][i]); gp.append(x["gamma_pred"][i])
                        mu.append(x["m_unit"][i])
            f = lambda a: [float(np.nanmin(np.array(a, float))), float(np.nanmax(np.array(a, float)))] if a else None
            rng[m] = {"n_points": len(ks), "KS_range": f(ks), "gamma_range": f(gam), "gamma_pred_range": f(gp),
                      "m_unit_range": f(mu)}
        S["avalanches"]["exp_only_points_gap0"] = rng
        allm = {m: max((v for c in cells.values() for v in c[m]["m_unit"] if v is not None), default=None)
                for m in ("M1", "M2")}
        S["avalanches"]["max_m_unit_pulse"] = allm
    if drv:
        S["driven_branching"] = {k: {"max_m_unit": (float(np.nanmax(np.array([np.nan if x is None else x for x in v["m_unit"]], float)))
                                                    if any(x is not None for x in v["m_unit"]) else None),
                                     "marker_unit": v["marker_unit"], "marker_step": v["marker_step"],
                                     "marker_span_step": v["marker_span_step"]}
                                 for k, v in drv["rows"].items()}
    if osc:
        S["oscillations"] = {"rows_with_OSC": {k: {"g_osc": v["g_osc"], "n_strong": v["n_strong"], "g_strong": v["g_strong"],
                                                   "g_H_candidate": v["g_H_candidate"]}
                                               for k, v in osc["rows"].items() if v["n_osc"]},
                             "rows_without_OSC": [k for k, v in osc["rows"].items() if not v["n_osc"]],
                             "hopf_candidates": osc["rows_with_hopf_candidate"],
                             "fss": {k: {"median_slope": v["median_slope"], "verdict": v["verdict"],
                                         "osc_by_N": v["osc_by_N"]} for k, v in osc["fss"].items()},
                             "decay_lane_osccheck": osc["decay_lane_osccheck"],
                             "driven": {k: {kk: v[kk] for kk in ("n", "osc", "strong")} for k, v in osc["driven"].items()}}
    json.dump(jsafe(S), open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(jsafe({k: S["T1_onset"][k]["verdict_T1_M2"] for k in S["T1_onset"]}), indent=1))


if __name__ == "__main__":
    main()
