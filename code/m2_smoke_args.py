"""Smoke tests of every Model-2 protocol at full production length (cost per run for the lanes; 2026-11-09).
Writes results/m2/sim/smoke/args.txt + index.json (one ald_sim.py line per array task) and the FSS graph file.
Every task: r = 1, B = 64 (ramps 32), production T. Analysis: code/m2_smoke_check.py.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_common as M

OUT = f"{M.ROOT}/results/m2/sim/smoke"
os.makedirs(OUT, exist_ok=True)
GDIR = f"{M.ROOT}/results/m2/sim/graphs"
os.makedirs(GDIR, exist_ok=True)

tasks = {}
def add(tag, model, proto, g, B=64, extra="", **pk):
    tasks[tag] = M.args_line(model, proto, g, 1, f"{OUT}/{tag}.npz", 100 + len(tasks), B=B, extra=extra, **pk)

add("decay_M2", "M2", "decay_h0", [0.8, 1.6, 3.2])
add("decay_M2_spmm", "M2", "decay_h0", [1.6, 3.2], extra="--engine spmm")
add("decay_M2_countsreg", "M2", "decay_h0", [3.2], counts_reg=True)
add("decay_M1", "M1", "decay_h0", [0.3, 0.6])
add("driven_M2", "M2", "driven", [0.5, 1.5, 3.0], counts_reg=True)
add("driven_M2_spmm", "M2", "driven", [1.5], extra="--engine spmm")
add("driven_M1", "M1", "driven", [0.2, 0.4])
add("aval_M2_gap0", "M2", "aval_pulse", [1.0, 1.5])
add("aval_M2_gap10", "M2", "aval_pulse", [1.5], gap=10.0)
add("aval_M1_gap0", "M1", "aval_pulse", [0.17, 0.25])
add("ramp_driven_M2", "M2", "ramp_driven", list(np.round(np.geomspace(0.3, 6.0, 12), 4)), B=32, dwell=100.0)
add("ramp_h0_M2", "M2", "ramp_h0_down", list(np.round(np.geomspace(6.0, 0.3, 12), 4)), B=32, dwell=100.0, gap=10.0)
add("decay_M2s", "M2s", "decay_h0", [0.8, 1.6, 3.2])
add("driven_M2s", "M2s", "driven", [0.5, 1.5])
add("null_targets_M2", "M2", "decay_h0", [1.6, 3.2], extra="--null --null_seed 0")
add("null_weights_M2", "M2", "decay_h0", [1.6, 3.2], extra="--null --null_kind weights --null_seed 0")
for v in ("M2_allsyn", "M2_KW09", "M2_N15", "M2_tau30"):
    add(f"decay_{v}", v, "decay_h0", [1.6])
# FSS: a 20,000-neuron configuration null (pd3 construction) with M2 dynamics; default regions only
gfile = f"{GDIR}/cnull_N20000_s1.npz"
if not os.path.exists(gfile):
    info = M.write_graph("cnull", 20000, 1, gfile)
    json.dump(info, open(gfile[:-4] + ".json", "w"))
tasks["fss_cnull20k_M2"] = (f"--graph {gfile} --exp decay --norm raw --T 3000 --reg_last --win 1500:3000 --engine push "
                            f"--push_max 0.012 {M.MODEL['M2']} --r 1 --g 1.6 3.2 6.4 --B 64 --seed 999 "
                            f"--rec_counts --counts_win 1500:3000 --out {OUT}/fss_cnull20k_M2.npz")
names = list(tasks)
open(f"{OUT}/args.txt", "w").write("\n".join(tasks[k] for k in names) + "\n")
json.dump({"tasks": names, "lines": tasks}, open(f"{OUT}/index.json", "w"), indent=1)
print(len(names), "tasks")
