"""m2_driven_refine.py -- stage-C refinement cells of lane m2/driven (plan.md section 3; 2026-11-09).
Usage: python m2_driven_refine.py NAME   -> results/m2/driven/refine_NAME.json (spec for m2_driven_args.py C:<file>)
Targets: A1 (M2 real ORN 0.04) at R_CORE; A2 (M1) and A4 (M2s) at r = 0, 1. Only curves whose coarse grid is complete
(40 points) and that have not been refined in an earlier C wave. Cells: DQ->DA, allowed top, DT->SS (bracketing coarse
cells) and the chi_AL/rho peak (the two cells around the grid argmax) if resolved; 8 log points per cell
(m2_common.refine); points already present (|ratio - 1| < 1e-3) are skipped.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_common as M

RES = f"{M.ROOT}/results/m2/driven"
C = json.load(open(f"{RES}/curves.json"))
done = set()
for f in glob.glob(f"{RES}/index_C*.json"):
    for m in json.load(open(f)):
        done.add((m["set"], m["model"], m["graph"], m["orn"], m["r"]))
targets = [("A1", "M2", "real", 0.04, r) for r in M.R_CORE] + \
          [(s, m, "real", 0.04, r) for s, m in (("A2", "M1"), ("A4", "M2s")) for r in (0, 1)]
spec, log = [], []
G40 = M.G_COARSE
for t in targets:
    k = "|".join(map(str, t)) + "|False"
    c = C.get(k)
    if c is None or c["n_g"] < 40 or t in done:
        log.append([k, "skip", None if c is None else c["n_g"], t in done]); continue
    g = np.array(c["g"])
    cells = []
    for ln in ("DQ_DA", "allowed_hi", "DT_SS"):
        b = c["lines_bracket"].get(ln)
        if b and b[0] < b[1]:
            cells.append((ln, b[0], b[1]))
    p = c["peaks"]["F_AL"]
    if p["resolved"] and p["k"] is not None and 0 < p["k"] < len(g) - 1:
        cells.append(("F_AL_peak", float(g[p["k"] - 1]), float(g[p["k"] + 1])))
    pts = []
    for ln, lo, hi in cells:
        n = 8
        for x in M.refine(lo, hi, n):
            if not any(abs(x / y - 1) < 1e-3 for y in list(g) + pts):
                pts.append(float(x))
    if pts:
        spec.append({"set": t[0], "model": t[1], "graph": t[2], "orn": t[3], "r": t[4], "g": sorted(pts), "per": 8,
                     "tag": f"_ref{sys.argv[1]}", "cells": cells})
    log.append([k, "cells", cells, len(pts)])
out = f"{RES}/refine_{sys.argv[1]}.json"
json.dump(spec, open(out, "w"), indent=1)
print(json.dumps(log, indent=0)[:3000]); print(len(spec), "curves,", sum(len(s["g"]) for s in spec), "points ->", out)
