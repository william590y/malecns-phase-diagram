"""m2_nature_hyst.py -- lane m2/nature, hysteresis loops (results/m2/nature/plan.md section 3).

  python m2_nature_hyst.py      -> results/m2/nature/hyst.json

Per run (protocol ramp_driven or ramp_h0_down, model, r, dwell D, replicate k): the 17-point g list is visited once in
each direction (33 dwells). Signal x per (dwell, replica): ramp_driven -> nonORN active fraction (seg_reg 'nonORN');
ramp_h0_down -> total active fraction (seg_rho); both averaged over the last half of each dwell.
Loop area A = sum over the 16 g visited twice of |mean_b x_first - mean_b x_second| * dln g (dln g = ln(2.0/0.5)/16).
Normalised area A / sum_g (mean of both branches) dln g. Permutation floor: 2000 draws, each flipping the up/down labels
independently per (g, replica) (equivalently: random sign per paired difference); p = P(A_perm >= A); floor = 99th pct.
Verdicts per (protocol, model, r): first-order if A(800) > floor(800) AND A(800)/A(200) >= 0.7 AND (if run)
A(3200)/A(800) >= 0.7; finite-rate if A(800)/A(200) < 0.5 or A(3200)/A(800) < 0.5; else inconclusive.
Bimodality per (g, branch): bimodality coefficient (bias-corrected skewness / excess kurtosis) > 0.555 AND the Otsu split
leaves >= 10 % of replicas on each side. ramp_driven: >= 2 adjacent bimodal g on one branch at D = 800 counts as
first-order support; ramp_h0_down bimodality (alive vs extinct replicas) is reported only.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import m2_common as M
from m2_nature_fss import jsafe

OUT = f"{M.ROOT}/results/m2/nature"
SIM = f"{OUT}/sim/hyst"
NPERM = 2000
rng = np.random.default_rng(31337)


def otsu_ok(v, frac=0.10):
    v = np.sort(np.asarray(v, float)); n = len(v)
    if n < 4 or v[-1] - v[0] <= 0:
        return False, None
    best, kb = -1, None
    for k in range(1, n):
        a, b = v[:k], v[k:]
        bv = k * (n - k) * (a.mean() - b.mean()) ** 2
        if bv > best:
            best, kb = bv, k
    return (min(kb, n - kb) >= frac * n), float(kb / n)


def bimodal(v):
    v = np.asarray(v, float); n = len(v)
    if n < 4 or np.std(v) == 0:
        return {"bc": None, "bimodal": False}
    sk = stats.skew(v, bias=False); ku = stats.kurtosis(v, bias=False)
    bc = (sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    ok, f = otsu_ok(v)
    return {"bc": float(bc), "otsu_low_frac": f, "bimodal": bool(bc > 0.555 and ok)}


def analyse_run(path, e):
    d = np.load(path)
    sg = d["seg_g"]; n = len(sg); half = n // 2              # 33 dwells: 0..16 first pass, 17..32 second pass
    if e["proto"] == "ramp_driven":
        names = [str(x) for x in d["reg_names"]]
        x = d["seg_reg"][:, names.index("nonORN"), :].astype(np.float64)       # [n_seg, B]
    else:
        x = d["seg_rho"].astype(np.float64)
    B = x.shape[1]
    i1 = np.arange(half)                                      # first-pass index of g[i], i = 0..15
    i2 = n - 1 - i1                                           # second-pass index of the same g
    assert np.allclose(sg[i1], sg[i2])
    g = sg[i1]
    dlng = float(np.log(sg[1] / sg[0])) if sg[1] > sg[0] else float(np.log(sg[0] / sg[1]))
    m1, m2 = x[i1].mean(1), x[i2].mean(1)
    A = float(np.abs(m1 - m2).sum() * dlng)
    norm = float(((m1 + m2) / 2).sum() * dlng)
    diff = x[i1] - x[i2]                                      # [16, B]
    s = rng.choice([-1.0, 1.0], size=(NPERM, len(i1), B))
    Ap = np.abs((s * diff[None]).mean(2)).sum(1) * dlng
    # bimodality on every visited g, both passes (turning point once)
    bim = []
    for j in range(n):
        b = bimodal(x[j]); b.update({"seg": j, "g": float(sg[j]), "pass": 1 if j < half else (0 if j == half else 2)})
        bim.append(b)
    # longest run of adjacent bimodal g on one pass
    def runlen(flags):
        best = cur = 0
        for f in flags:
            cur = cur + 1 if f else 0; best = max(best, cur)
        return best
    p1 = [b["bimodal"] for b in bim[:half + 1]]; p2 = [b["bimodal"] for b in bim[half:]]
    xres = None
    if "seg_x" in d.files:
        names = [str(x_) for x_ in d["reg_names"]]
        xr = d["seg_x"]
        xres = {nm: [None if not np.isfinite(v) else float(v) for v in xr[:, k]] for k, nm in enumerate(names)
                if nm in ("AL_module", "nonORN", "CNS_rest")}
    return {"name": e["name"], "posthoc": bool(e.get("posthoc", False)), "proto": e["proto"], "model": e["model"], "r": e["r"], "D": e["D"], "rep": e["rep"],
            "g": g.tolist(), "g_on": e["g_on"], "x_first": m1.tolist(), "x_second": m2.tolist(),
            "sd_first": x[i1].std(1).tolist(), "sd_second": x[i2].std(1).tolist(),
            "x_turn": float(x[half].mean()), "A": A, "A_norm": (A / norm if norm > 0 else None),
            "perm_p": float((Ap >= A).mean()), "perm_p99": float(np.percentile(Ap, 99)),
            "perm_mean": float(Ap.mean()), "B": int(B), "dlng": dlng,
            "alive_end_first": float(d["seg_alive"][half - 1].mean()) if "seg_alive" in d.files else None,
            "bimodal_n": int(sum(b["bimodal"] for b in bim)), "bimodal_maxrun": int(max(runlen(p1), runlen(p2))),
            "bimodal_cells": [b for b in bim if b["bimodal"]], "resource": xres, "seg_g": sg.tolist()}


def main():
    idx = {}
    for fi in sorted(glob.glob(f"{OUT}/index_hyst*.json")):
        for e in json.load(open(fi)):
            idx[os.path.basename(e["out"])] = e
    runs = []
    for f in sorted(glob.glob(f"{SIM}/*.npz")):
        e = idx.get(os.path.basename(f))
        if e is None:
            continue
        try:
            runs.append(analyse_run(f, e))
        except Exception as ex:
            print("FAIL", f, repr(ex), flush=True)
    missing = sorted(set(idx) - {os.path.basename(f) for f in glob.glob(f"{SIM}/*.npz")})
    ver = {}

    def verdict(rr):
        byD = {}
        for x in rr:
            byD.setdefault(x["D"], []).append(x)
        A = {D: float(np.mean([x["A"] for x in v])) for D, v in byD.items()}
        k0 = {D: [x for x in v if x["rep"] == 0][0] for D, v in byD.items() if any(x["rep"] == 0 for x in v)}
        v = {"A": A, "A_norm": {D: float(np.mean([x["A_norm"] for x in vv if x["A_norm"] is not None]))
                                 for D, vv in byD.items()},
             "p": {D: k0[D]["perm_p"] for D in k0}, "p99": {D: k0[D]["perm_p99"] for D in k0},
             "p_reps": {D: [x["perm_p"] for x in vv] for D, vv in byD.items()},
             "A_reps": {D: [x["A"] for x in vv] for D, vv in byD.items()},
             "p99_reps": {D: [x["perm_p99"] for x in vv] for D, vv in byD.items()}}
        r82 = A[800] / A[200] if (800 in A and 200 in A and A[200] > 0) else None
        r38 = A[3200] / A[800] if (3200 in A and 800 in A and A[800] > 0) else None
        above = (800 in k0) and (k0[800]["A"] > k0[800]["perm_p99"])
        v.update({"ratio_800_200": r82, "ratio_3200_800": r38, "A800_above_floor": bool(above),
                  "ratio_200_50": (A[200] / A[50] if (200 in A and 50 in A and A[50] > 0) else None),
                  "ratio_800_50": (A[800] / A[50] if (800 in A and 50 in A and A[50] > 0) else None)})
        fo = above and r82 is not None and r82 >= 0.7 and (r38 is None or r38 >= 0.7)
        fr = (r82 is not None and r82 < 0.5) or (r38 is not None and r38 < 0.5)
        v["verdict"] = "first-order (hysteresis)" if fo else ("finite-rate" if fr else "inconclusive")
        if rr[0]["proto"] == "ramp_driven" and 800 in k0:
            v["bimodal_support"] = bool(k0[800]["bimodal_maxrun"] >= 2)
        v["bimodal_n_by_D"] = {D: k0[D]["bimodal_n"] for D in k0}
        return v

    for proto in ("ramp_driven", "ramp_h0_down"):
        for model in ("M1", "M2"):
            for r in M.R_CORE:
                rr = [x for x in runs if x["proto"] == proto and x["model"] == model and x["r"] == r]
                pre = [x for x in rr if not x.get("posthoc")]
                if not pre:
                    continue
                v = verdict(pre)                                   # pre-registered runs only
                if len(rr) > len(pre):
                    vp = verdict(rr)
                    v["posthoc_all_runs"] = {k: vp[k] for k in ("A", "A_reps", "p_reps", "p99_reps", "ratio_800_200",
                                                                "ratio_3200_800", "verdict")}
                ver[f"{proto}|{model}|r{r:g}"] = v
                print(proto, model, r, {D: round(a, 6) for D, a in sorted(v["A"].items())}, v["verdict"],
                      "| posthoc:", v.get("posthoc_all_runs", {}).get("verdict"), flush=True)
    json.dump(jsafe({"definitions": __doc__, "runs": runs, "verdicts": ver, "missing": missing}),
              open(f"{OUT}/hyst.json", "w"), indent=1)
    print("missing", len(missing))


if __name__ == "__main__":
    main()
