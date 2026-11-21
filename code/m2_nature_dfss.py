"""m2_nature_dfss.py -- lane m2/nature, driven finite-size check (results/m2/nature/plan.md section 2).

  python m2_nature_dfss.py      -> results/m2/nature/dfss.json

Per point (protocol driven, M2, ORN 0.04, B 64): rho_nonORN = replica mean of win_mean[nonORN] over (500, 1500];
chi_nonORN = n_nonORN * mean_b(win_m2 - win_mean^2) (MODEL2 5b); chi_AL likewise on the AL module where the graph holds
>= 100 AL-module neurons; tau_post = censored MLE of t = reg_last[nonORN] + dt - 1500 (replicas active at 4500 censored
at 3000), DT->SS = isotonic first crossing of tau_post = 3000 (decay-lane crossing rule).
Peak per curve = maximum of chi over g (interior only; else flagged 'edge'); family value per N = mean over seeds.
Full-size reference points: results/m2/driven/points.csv (set A1 = real graph, A3 = configuration null, M2, ORN 0.04).
Growth exponent a = slope of ln chi_max vs ln N over the sizes with an interior peak.
"""
import csv, glob, json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import m2_common as M
import m2_obs as O
from m2_nature_fss import jsafe, iso, cross

OUT = f"{M.ROOT}/results/m2/nature"
SIM = f"{OUT}/sim/dfss"
DT = 0.2
NFULL = 167216


def summarize(path):
    d = np.load(path)
    names = [str(x) for x in d["reg_names"]]; sizes = d["reg_sizes"].astype(float)
    jn, ja = names.index("nonORN"), names.index("AL_module")
    out = []
    for i, g in enumerate(d["g"]):
        wm = d["win_mean"][i, 0].astype(np.float64); w2 = d["win_m2"][i, 0].astype(np.float64)
        var = np.maximum(w2 - wm ** 2, 0)
        last = d["reg_last"][i][jn].astype(np.float64)
        alive = last >= 4500 - DT / 2
        t = np.where(alive, np.inf, np.maximum(last + DT - 1500, 0))
        tau, dd = O.tau_mle(t, 3000)
        p = {"g": float(g), "rho_non": float(wm[jn].mean()), "chi_non": float(sizes[jn] * var[jn].mean()),
             "n_non": float(sizes[jn]), "n_AL": float(sizes[ja]), "tau_post": tau, "deaths_post": dd,
             "E_post": float(np.minimum(t, 3000).sum()), "B": int(len(t))}
        p["F_non"] = p["chi_non"] / p["rho_non"] if p["rho_non"] > 0 else None
        if sizes[ja] >= 100:
            p["chi_AL"] = float(sizes[ja] * var[ja].mean()); p["rho_AL"] = float(wm[ja].mean())
        out.append(p)
    return out


def peak(gs, ys):
    ys = np.array([np.nan if y is None else y for y in ys], float)
    if not np.isfinite(ys).any():
        return None, None, "none"
    k = int(np.nanargmax(ys))
    return float(ys[k]), float(gs[k]), ("interior" if 0 < k < len(ys) - 1 else "edge")


def dtss(P):
    g = np.array([p["g"] for p in P]); d = np.array([p["deaths_post"] for p in P], float)
    E = np.array([p["E_post"] for p in P])
    lt = iso(np.log(E / np.maximum(d, 0.5)), d + 0.5)
    return cross(g, lt, 3000.0)[0]


def main():
    idx = {}
    for f in glob.glob(f"{OUT}/index_dfss*.json"):
        for e in json.load(open(f)):
            idx[os.path.basename(e["out"])] = e
    cur = {}
    for f in sorted(glob.glob(f"{SIM}/*.npz")):
        e = idx.get(os.path.basename(f))
        if e is None:
            continue
        try:
            cur.setdefault((e["family"], int(e["N"]), int(e["gseed"]), float(e["r"])), []).extend(summarize(f))
        except Exception as ex:
            print("FAIL", f, repr(ex))
    # full-size reference from lane driven
    with open(f"{M.ROOT}/results/m2/driven/points.csv") as fh:
        for row in csv.DictReader(fh):
            if row["model"] != "M2" or abs(float(row["orn"]) - 0.04) > 1e-9 or row["set"] not in ("A1", "A3"):
                continue
            if row["graph"] not in ("real", "cnull") or float(row["r"]) not in (0.0, 1.0):
                continue
            fam = "full_real" if row["graph"] == "real" else "cnull"
            f_ = lambda k: (float(row[k]) if row[k] not in ("", "nan", "None") else None)
            p = {"g": float(row["g"]), "chi_non": f_("chi_non"), "rho_non": f_("rho_nonORN"), "F_non": f_("F_non"),
                 "chi_AL": f_("chi_AL"), "tau_post": f_("tau_post"), "deaths_post": f_("deaths_post"), "B": 64}
            cur.setdefault((fam, NFULL, 0, float(row["r"])), []).append(p)
    per = []
    for key, P in sorted(cur.items()):
        P.sort(key=lambda p: p["g"])
        gs = np.array([p["g"] for p in P])
        G_D = gs[(gs >= 0.42 - 1e-6) & (gs <= 11.22)]
        P2 = [p for p in P if 0.42 - 1e-6 <= p["g"] <= 11.22]       # same g window as the subgraph runs
        cmax, cg, cst = peak([p["g"] for p in P2], [p["chi_non"] for p in P2])
        amax, ag, ast = peak([p["g"] for p in P2], [p.get("chi_AL") for p in P2])
        rec = {"family": key[0], "N": key[1], "gseed": key[2], "r": key[3], "chi_non_max": cmax, "chi_non_g": cg,
               "chi_non_peak": cst, "chi_AL_max": amax, "chi_AL_g": ag, "chi_AL_peak": ast,
               "DT_SS": (dtss(P2) if all("E_post" in p for p in P2) else None), "points": P2}
        per.append(rec)
        print(key, "chi_non max", cmax, "at", cg, cst, "| AL", amax, ag, ast, "| DT->SS", rec["DT_SS"], flush=True)
    fams = {}
    for r in (0.0, 1.0):
        for fam in ("cnull", "realsubu", "realsubcn"):
            recs = [x for x in per if x["family"] == fam and x["r"] == r]
            if fam != "cnull":
                recs += [x for x in per if x["family"] == "full_real" and x["r"] == r]
            Ns = sorted({x["N"] for x in recs})
            rows = []
            for N in Ns:
                xs = [x for x in recs if x["N"] == N]
                ints = [x for x in xs if x["chi_non_peak"] == "interior"]
                rows.append({"N": N, "n": len(xs), "n_interior": len(ints),
                             "chi_non_max": float(np.mean([x["chi_non_max"] for x in xs])) if xs else None,
                             "chi_non_g": float(np.mean([x["chi_non_g"] for x in xs])) if xs else None,
                             "chi_AL_max": (float(np.mean([x["chi_AL_max"] for x in xs if x["chi_AL_max"]]))
                                            if any(x["chi_AL_max"] for x in xs) else None),
                             "DT_SS": (float(np.mean([x["DT_SS"] for x in xs if x["DT_SS"]]))
                                       if any(x["DT_SS"] for x in xs) else None)})
            ok = [x for x in rows if x["n_interior"] > 0 and x["chi_non_max"]]
            a = float(np.polyfit(np.log([x["N"] for x in ok]), np.log([x["chi_non_max"] for x in ok]), 1)[0]) \
                if len(ok) >= 3 else None
            verdict = (None if a is None else "divergence-like (transition)" if a >= 0.5 else
                       "no divergence (crossover)" if a <= 0.2 else "inconclusive")
            fams[f"{fam}|r{r:g}"] = {"sizes": rows, "growth_exponent": a, "verdict_if_peak_tracks": verdict}
            print(fam, r, "a =", a, verdict)
    json.dump(jsafe({"definitions": __doc__, "per_graph": per, "families": fams}), open(f"{OUT}/dfss.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
