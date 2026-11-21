"""Write the FSS graphs of plan section 9 (lane m2/decay) with m2_common.write_graph (CPU)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_common as M
D = f"{M.ROOT}/results/m2/decay/graphs"
os.makedirs(D, exist_ok=True)
info = {}
for kind, Ns in (("cnull", (20000, 40000, 80000)), ("realsub", (40000, 80000))):
    for N in Ns:
        p = f"{D}/{kind}_N{N}_s1.npz"
        if not os.path.exists(p):
            info[p] = M.write_graph(kind, N, 1, p)
            print(p, info[p], flush=True)
json.dump(info, open(f"{D}/graphs.json", "w"), indent=1)
