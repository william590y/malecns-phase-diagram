"""Write args lines for the fine-grid follow-up (pd3_fss_fine.py) of finished coarse FSS runs.
Selection: configuration nulls with N >= 20000, both r. Fine grid = 16 ln-spaced x in [lower grid bracket of x_S (tau = 200),
upper grid bracket of x_on (tau = 3000)] of the coarse run (isotonic + interpolation rules of pd3_fss_analyze.crossings).
Usage: pd3_fss_fine_args.py <out_args_file> [<already_submitted_args_file> ...]  (appends only new lines; prints them)
"""
import glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd3_fss_analyze import load, crossings, DIR

out = sys.argv[1]
seen = set()
for f in sys.argv[1:]:
    if os.path.exists(f):
        seen |= {" ".join(l.split()[1:5]) for l in open(f) if l.strip()}
new = []
for p in sorted(glob.glob(f"{DIR}/fss_cnull_N*_s*_r*.npz")):
    m = re.match(r"fss_cnull_N(\d+)_s(\d+)_r([\d.]+)\.npz", os.path.basename(p))
    N, seed, r = int(m.group(1)), int(m.group(2)), float(m.group(3))
    key = f"cnull {N} {seed} {r:g}"
    if N < 20000 or key in seen:
        continue
    o = load(p); x = o["x"]; B = o["died"].shape[1]
    c = crossings(x, o["died"].sum(1).astype(float), o["times"].sum(1), 1 - o["died"].mean(1), np.full(len(x), B))
    if c["S"]["status"] != "ok" or c["on"]["status"] != "ok":
        print("skip (no S/on crossing on the coarse grid):", key); continue
    lo, hi = c["S"]["bracket"][0], c["on"]["bracket"][1]
    new.append(f"pd3_fss_fine.py {key} --xlo {lo:.6f} --xhi {hi:.6f} --nx 16")
with open(out, "a") as fh:
    for l in new:
        fh.write(l + "\n")
print("\n".join(new))
