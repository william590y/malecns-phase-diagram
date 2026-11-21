"""m2_modules_rerun.py -- lane m2/modules: lines of a set whose output is missing or unreadable and whose array task is no
longer queued or running (preempted / failed) -> sim/args_<SET>_rerun<k>.txt (same lines, same seeds).
Usage: python m2_modules_rerun.py SET JOBID [JOBID ...]   (prints the file name and the number of lines; nothing if none)
"""
import glob, json, os, subprocess, sys
import numpy as np

OUT = "/MALECNS_ROOT/results/m2/modules/sim"
SET, jobs = sys.argv[1], sys.argv[2:]
lines = open(f"{OUT}/args_{SET}.txt").read().strip().split("\n")
idx = json.load(open(f"{OUT}/index_{SET}.json"))
q = subprocess.run(["squeue", "-u", __import__("os").environ["USER"], "-r", "-h", "-o", "%i"], capture_output=True, text=True).stdout.split()
active = set()
for j in jobs:
    for x in q:
        if x.startswith(j + "_"):
            active.add(int(x.split("_")[1]))


def ok(p):
    if not os.path.exists(p):
        return False
    try:
        z = np.load(p, allow_pickle=True); _ = z["meta"]; _ = z["surv"]
        return True
    except Exception:
        return False


todo = [i for i, e in enumerate(idx) if i not in active and not ok(e["out"])]
if todo:
    k = len(glob.glob(f"{OUT}/args_{SET}_rerun*.txt")) + 1
    fn = f"{OUT}/args_{SET}_rerun{k}.txt"
    open(fn, "w").write("\n".join(lines[i] for i in todo) + "\n")
    print(fn, len(todo), todo)
