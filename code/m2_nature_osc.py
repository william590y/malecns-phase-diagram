"""m2_nature_osc.py -- lane m2/nature, oscillations: is there a Hopf-like onset? (results/m2/nature/plan.md section 5)

  python m2_nature_osc.py     -> results/m2/nature/osc.json

Inputs (read only): lane decay results/m2/decay/osc.json (per-point m2_obs.osc_metrics of the total active fraction of
the replicas alive at T, window 1500-3000) and points.json (rho_QS), osccheck.json (r 0 finite-size check); lane driven
points.csv (OSC flags in the driven window); this lane's fss.json (per-point var_t, rho_QS and OSC on every FSS graph).
Per r (M2 real): OSC / OSC-strong points, period, coherence C(2P)/C(P), relative amplitude CV_t and absolute amplitude
std = CV_t * rho_QS, against g.
Hopf-like onset (pre-registered): OSC-strong on >= 3 consecutive grid points from g_H upward AND var_t not falling with N
(slope >= -0.3 at matched rho_QS). Quasi-cycle: OSC not strong and slope <= -0.7.
Finite-size slope at matched rho_QS (decay-lane rule): within each size, ln var_t interpolated in ln rho_QS (survivor
points, >= 4 alive replicas); at common rho_QS values the slope of ln var_t vs ln N; median over rho_QS values.
"""
import csv, json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import m2_common as M
from m2_nature_fss import jsafe

OUT = f"{M.ROOT}/results/m2/nature"
DEC = f"{M.ROOT}/results/m2/decay"


def matched_slope(curves):
    """curves: {N: [(rho, var), ...]} -> list of (rho, slope) and the median slope."""
    Ns = sorted(curves)
    if len(Ns) < 3:
        return [], None
    lo = max(min(r for r, v in curves[N]) for N in Ns); hi = min(max(r for r, v in curves[N]) for N in Ns)
    if not (hi > lo):
        return [], None
    out = []
    for rq in np.geomspace(lo, hi, 6):
        lv = []
        for N in Ns:
            c = sorted(curves[N]); rr = np.log([x[0] for x in c]); vv = np.log([x[1] for x in c])
            lv.append(np.interp(np.log(rq), rr, vv))
        out.append((float(rq), float(np.polyfit(np.log(Ns), lv, 1)[0])))
    return out, float(np.median([s for _, s in out]))


def main():
    osc = json.load(open(f"{DEC}/osc.json"))["rows"]
    pts = json.load(open(f"{DEC}/points.json"))
    rows = {}
    for r in M.R_ALL:
        key = f"M2_real_r{r:g}"
        P = {round(p["g"], 5): p for p in pts.get(key, [])}
        o = sorted([x for x in osc if x["row"] == key and x["signal"] == "total"], key=lambda x: x["g"])
        L = []
        for x in o:
            q = P.get(round(x["g"], 5), {})
            rho = q.get("qs_tot")
            L.append({"g": x["g"], "n_rep": x["n_rep"], "osc": x["osc"], "osc_strong": x["osc_strong"],
                      "period": x["period"], "c_min": x["c_min"], "c_peak": x["c_peak"], "coh": x["coh_ratio"],
                      "cv_t": x["cv_t"], "peak_ratio": x.get("peak_ratio"), "rho_qs": rho,
                      "amp": (x["cv_t"] * rho if (rho is not None and x["cv_t"] is not None) else None),
                      "tau_int": x["tau_int"]})
        st = [p["osc_strong"] for p in L]
        gH = None
        for k in range(len(L) - 2):
            if all(st[k:k + 3]):
                gH = L[k]["g"]; break
        rows[key] = {"r": r, "points": L, "n_osc": int(sum(p["osc"] for p in L)), "n_strong": int(sum(st)),
                     "g_osc": [p["g"] for p in L if p["osc"]], "g_strong": [p["g"] for p in L if p["osc_strong"]],
                     "g_H_candidate": gH}
        print(key, "n points", len(L), "OSC", rows[key]["n_osc"], "strong", rows[key]["n_strong"], "gH", gH)
    # finite-size slope from this lane's FSS runs (r 0, 1, 4) per family
    fss = json.load(open(f"{OUT}/fss.json")) if os.path.exists(f"{OUT}/fss.json") else None
    fsz = {}
    if fss:
        for r in (0.0, 1.0, 4.0):
            for fam in ("cnull", "realsubu", "realsubcn", "realsub"):
                curves = {}
                oscN = {}
                for rec in fss["per_graph"]:
                    if rec["r"] != r or (rec["family"] != fam and not (fam != "cnull" and rec["family"] == "real")):
                        continue
                    for p in rec["points"]:
                        if p.get("var_t") and p.get("qs") and p["n_alive"] >= 4:
                            curves.setdefault(rec["N"], []).append((p["qs"], p["var_t"]))
                        if p.get("osc"):
                            oscN.setdefault(rec["N"], []).append((p["g"], p["osc"]["osc"], p["osc"]["osc_strong"],
                                                                  p["osc"]["coh_ratio"], p["osc"]["period"]))
                sl, med = matched_slope(curves)
                verdict = (None if med is None else "quasi-cycle (var ~ 1/N)" if med <= -0.7 else
                           "macroscopic (var not falling)" if med >= -0.3 else "inconclusive")
                fsz[f"{fam}|r{r:g}"] = {"slopes": sl, "median_slope": med, "verdict": verdict,
                                        "osc_by_N": {str(N): {"n_points": len(v), "n_osc": sum(x[1] for x in v),
                                                              "n_strong": sum(x[2] for x in v),
                                                              "g_osc": [x[0] for x in v if x[1]]}
                                                     for N, v in sorted(oscN.items())}}
                print("FSS", fam, r, med, verdict)
    oc = json.load(open(f"{DEC}/osccheck.json"))
    dec_fss = {k: {"median_slope": v["median_slope"], "verdict": v["verdict"]} for k, v in oc["families"].items()}
    # driven window OSC (lane driven)
    drv = {}
    with open(f"{M.ROOT}/results/m2/driven/points.csv") as fh:
        for row in csv.DictReader(fh):
            k = f"{row['set']}|{row['model']}|{row['graph']}|o{row['orn']}"
            d = drv.setdefault(k, {"n": 0, "osc": 0, "strong": 0, "g_osc": []})
            d["n"] += 1
            if row["osc"] in ("True", "1", "true"):
                d["osc"] += 1; d["g_osc"].append((float(row["r"]), float(row["g"])))
            if row["osc_strong"] in ("True", "1", "true"):
                d["strong"] += 1
    # overall verdict
    any_gH = {k: v["g_H_candidate"] for k, v in rows.items() if v["g_H_candidate"] is not None}
    json.dump(jsafe({"definitions": __doc__, "rows": rows, "fss": fsz, "decay_lane_osccheck": dec_fss,
                     "driven": drv, "rows_with_hopf_candidate": any_gH}), open(f"{OUT}/osc.json", "w"), indent=1)
    print("hopf candidates", any_gH)


if __name__ == "__main__":
    main()
