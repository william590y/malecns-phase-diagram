"""Lane m2/verify: analysis of the new-seed GPU reruns (plan results/m2/verify/plan.md, section B).
Uses the estimators of code/m2_verify.py (own code). Each line is estimated from the RERUN DATA ALONE and compared with
the lane value / total interval: replicates if inside the lane's total interval or within 3%.
-> results/m2/verify/reruns.json
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_verify as V  # noqa: E402

SIM = f"{V.OUT}/sim"
rng = np.random.default_rng(7)


def f(tid):
    return f"{SIM}/{tid}.npz"


def have(*tids):
    return all(os.path.exists(f(t)) for t in tids)


def rep(mine, lane, total):
    if mine is None or lane is None:
        return None
    ok = abs(mine / lane - 1) <= 0.03 or (total is not None and total[0] <= mine <= total[1])
    return {"mine": mine, "lane": lane, "lane_total": total, "rel": mine / lane - 1, "replicates": bool(ok)}


def main():
    dec = json.load(open(f"{V.M2}/decay/boundaries.json"))["rows"]
    cur = json.load(open(f"{V.M2}/driven/curves.json"))
    R = {}
    for tids, row, keys in ((("V1_dec_r0_S", "V2_dec_r0_on"), "M2_real_r0", ("S", "on")),
                            (("V3_dec_r1_S", "V4_dec_r1_on"), "M2_real_r1", ("S", "on")),
                            (("V5_dec_r6_on",), "M2_real_r6", ("on",)),
                            (("V7_cnull0_r0_on",), "M2_cnull_r0", ("on",))):
        if not have(*tids):
            continue
        pts = V.decay_points([f(t) for t in tids], want_occ=False)
        L = V.decay_lines(pts)
        R[row] = {k: rep(L[k], dec[row][k]["g"], dec[row][k]["total"]) for k in keys}
        R[row]["points"] = [{"g": p["g"], "tau": p["tau"], "deaths": p["deaths"], "B": p["B"]} for p in pts]
        print(row, {k: R[row][k] for k in keys})
    if have("V6_dec_r0_D"):
        pts = V.decay_points([f("V6_dec_r0_D")])
        g = np.array([p["g"] for p in pts]); prn = np.array([V.pr(p["occ"]) / V.N_ALL for p in pts])
        D = V.first_cross(g, V.pava(prn, np.ones(len(prn))), 0.25)
        R["M2_real_r0_D"] = {"D": rep(D, dec["M2_real_r0"]["D"]["g"], dec["M2_real_r0"]["D"]["total"]),
                             "PR_N": dict(zip(map(float, g), map(float, prn))), "tau": [p["tau"] for p in pts]}
        print("D", R["M2_real_r0_D"])
    if have("V8_cnull2_r0_on"):
        pts = V.decay_points([f("V8_cnull2_r0_on")], want_occ=False)
        L = V.decay_lines(pts)
        on_real = dec["M2_real_r0"]["on"]["g"]
        R["M2_cnull_nullseed2_r0"] = {"on": L["on"], "on_nullseed0_lane": dec["M2_cnull_r0"]["on"]["g"],
                                      "real_over_null_nullseed2": on_real / L["on"] if L["on"] else None,
                                      "points": [{"g": p["g"], "tau": p["tau"], "deaths": p["deaths"]} for p in pts]}
        print("cnull seed2", R["M2_cnull_nullseed2_r0"])
    # driven
    for tids, r in ((("V9_drv_r0_lo_top", "V10_drv_r0_ss"), "0"), (("V11_drv_r1_lo_top",), "1")):
        if not have(*tids):
            continue
        pts = V.driven_points([f(t) for t in tids])
        L = V.driven_lines(pts)
        lane = cur[f"A1|M2|real|0.04|{r}|False"]
        ci = lane["lines_ci"]
        def tot(k):
            c = ci.get(k) or {}
            br = lane["lines_bracket"].get(k)
            lo = min([x for x in (c.get("lo"), br[0] if br else None) if x is not None], default=None)
            hi = max([x for x in (c.get("hi"), br[1] if br else None) if x is not None], default=None)
            return [lo, hi] if lo is not None else None
        R[f"driven_r{r}"] = {"DQ_DA": rep(L["DQ_DA"], lane["lines"]["DQ_DA"], tot("DQ_DA")),
                             "allowed_top": rep(L["allowed_top"], lane["lines"]["allowed_hi"], tot("allowed_hi")),
                             "allowed_top_by": L["allowed_top_by"],
                             "points": [{k: p[k] for k in ("g", "G", "S1", "alive_ALsys", "tau_post", "B")} for p in pts]}
        if r == "0":
            R["driven_r0"]["DT_SS"] = rep(L["DT_SS"], lane["lines"]["DT_SS"], tot("DT_SS"))
        print("driven", r, {k: v for k, v in R[f"driven_r{r}"].items() if k != "points"})
    # palp-intact, two seeds pooled, replica bootstrap of the residual ratios
    if have("V12_palp_r0_a", "V13_palp_r0_b"):
        P = {}
        for t in ("V12_palp_r0_a", "V13_palp_r0_b"):
            d = np.load(f(t)); ix = {n: i for i, n in enumerate(d["reg_names"])}
            for gi, gv in enumerate(d["g"]):
                q = P.setdefault(round(float(gv), 6), {"orn": [], "upn": [], "s1": [], "s2": []})
                q["orn"].append(d["win_mean"][gi, 0, ix["ORN"]]); q["upn"].append(d["win_mean"][gi, 0, ix["uPN_ant"]])
                q["s1"].append(d["win_mean"][gi, 1, ix["uPN_ant"]]); q["s2"].append(d["win_mean"][gi, 2, ix["uPN_ant"]])
        rows = []
        for gv in sorted(P):
            q = {k: np.concatenate(v).astype(np.float64) for k, v in P[gv].items()}
            B = len(q["orn"])
            G = q["upn"].mean() / q["orn"].mean(); s1 = q["s1"].mean() / q["upn"].mean(); s2 = q["s2"].mean() / q["upn"].mean()
            bs = []
            for _ in range(2000):
                i = rng.integers(0, B, B)
                bs.append((q["upn"][i].mean() / q["orn"][i].mean(), q["s1"][i].mean() / q["upn"][i].mean(),
                           q["s2"][i].mean() / q["upn"][i].mean()))
            bs = np.array(bs)
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            rows.append({"g": gv, "B": B, "G": G, "G_ci": [lo[0], hi[0]], "S1p": s1, "S1p_ci": [lo[1], hi[1]],
                         "S2p": s2, "S2p_ci": [lo[2], hi[2]]})
            print("palp", gv, B, "G %.3f [%.3f,%.3f]  S1p %.3f [%.3f,%.3f]  S2p %.3f [%.3f,%.3f]" %
                  (G, lo[0], hi[0], s1, lo[1], hi[1], s2, lo[2], hi[2]))
        refute = [r_ for r_ in rows if r_["G"] >= 0.29 and r_["S1p_ci"][1] <= 0.10 and r_["S2p_ci"][1] <= 0.10]
        confirm = all(max(r_["S1p_ci"][0], r_["S2p_ci"][0]) > 0.10 for r_ in rows if r_["G"] >= 0.29)
        n_gain = sum(r_["G"] >= 0.29 for r_ in rows)
        verdict = "REFUTED" if refute else ("CONFIRMED" if (confirm and n_gain) else "MARGINAL")
        gedge = V.first_cross([r_["g"] for r_ in rows], [r_["G"] for r_ in rows], 0.29)
        g10 = V.first_cross([r_["g"] for r_ in rows], [max(r_["S1p"], r_["S2p"]) for r_ in rows], 0.10)
        R["palp_r0"] = {"rows": rows, "verdict": verdict, "gain_edge_rerun": gedge, "palp_10pct_crossing_rerun": g10,
                        "n_points_with_G_ge_0.29": int(n_gain)}
        print("palp verdict", verdict, "gain edge", gedge, "palp 10% crossing", g10)
    iso_lane = {"CB": 1.42, "AL": 2.19}
    for t, mod in (("V14_iso_CB_r0", "CB"), ("V15_iso_AL_r0", "AL")):
        if not have(t):
            continue
        pts = V.decay_points([f(t)], want_occ=False)
        L = V.decay_lines(pts)
        R[f"iso_{mod}_r0"] = {"on": rep(L["on"], iso_lane[mod], None),
                              "points": [{"g": p["g"], "tau": p["tau"], "deaths": p["deaths"]} for p in pts]}
        print("iso", mod, R[f"iso_{mod}_r0"]["on"], [(round(p["g"], 3), round(p["tau"])) for p in pts])
    json.dump(R, open(f"{V.OUT}/reruns.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
