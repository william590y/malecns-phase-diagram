"""E3 stage 2 spec: extra g points where a stage-1 boundary (S or on) falls in a grid cell wider than x1.15 (3 log-spaced
points inside the bracket), and an extension g = 10.5, 13.8, 18.2, 24 for arms with no sustained point up to g = 8.
Reads results/aldef/depression/summary.json -> stage2_spec.json (arm -> g list) for `ald_dep_args.py stage2`."""
import json
import numpy as np

OUT = "/MALECNS_ROOT/results/aldef/depression"
A = json.load(open(f"{OUT}/summary.json"))["arms"]
spec = {}
for arm, s in A.items():
    gl = []
    for key in ("S", "on"):
        b = s[key]
        if b["status"] == "ok":
            lo, hi = b["bracket"]
            if hi / lo > 1.15:
                gl += list(np.geomspace(lo, hi, 5)[1:-1])
        elif b["status"] == "none" and key == "on":
            gl += [10.5, 13.8, 18.2, 24.0]
    gl = sorted(set(round(float(x), 4) for x in gl))
    if gl:
        spec[arm] = gl
json.dump(spec, open(f"{OUT}/stage2_spec.json", "w"), indent=1)
print(sum(len(v) for v in spec.values()), "points in", len(spec), "arms")
for k, v in spec.items():
    print(k, v)
