"""m2_mf_args.py -- argument files for the mean-field production arrays (lane m2/theory, meanfield).
Writes results/m2/theory/args/{main,var}.txt (one python call per line) and args/index.json."""
import json, sys
sys.path.insert(0, "/MALECNS_ROOT/code")
import m2_common as MC
R = "results/m2/theory/runs"
main, var, idx = [], [], []


def add(lst, name, extra):
    lst.append(f"code/m2_mf_run.py {extra} --out {R}/{name}.npz")
    idx.append({"name": name, "args": extra})


for proto in ("decay", "driven"):
    for cl in ("qmf", "cqmf"):
        for r in MC.R_ALL:
            if cl == "cqmf" and r == 0:
                continue                          # identical to qmf at r = 0
            ring = " --ring --ring_T 1000" if r in MC.R_CORE else ""
            ics = "decay,eps" if proto == "decay" else "quiet,decay"
            add(main, f"{proto}_M2_{cl}_r{r:g}", f"--variant M2 --r {r:g} --proto {proto} --closure {cl} --ics {ics}{ring}")
        for r in MC.R_CORE:
            if cl == "cqmf" and r == 0:
                continue
            ics = "decay,eps" if proto == "decay" else "quiet,decay"
            add(main, f"{proto}_M1_{cl}_r{r:g}", f"--variant M1 --r {r:g} --proto {proto} --closure {cl} --ics {ics} --ring --ring_T 1000")
for v in ("M2s", "M1s", "M2_allsyn", "M2_KW09", "M2_N15", "M2_tau30"):
    for proto in ("decay", "driven"):
        for cl in ("qmf", "cqmf"):
            for r in MC.R_CORE:
                if cl == "cqmf" and r == 0 and v not in ("M2s", "M1s"):
                    continue                      # M2s/M1s: the lLN1_bc->ALLN override keeps an edge set where cqmf = qmf too, but cheap
                ics = "decay" if proto == "decay" else "quiet"
                add(var, f"{proto}_{v}_{cl}_r{r:g}", f"--variant {v} --r {r:g} --proto {proto} --closure {cl} --ics {ics}")
for orn in (0.01, 0.02, 0.07, 0.10):
    for cl in ("qmf", "cqmf"):
        for r in MC.R_CORE:
            if cl == "cqmf" and r == 0:
                continue
            add(var, f"driven_M2_{cl}_r{r:g}_orn{orn:g}", f"--variant M2 --r {r:g} --proto driven --orn {orn:g} --closure {cl} --ics quiet")
for null in ("targets", "weights"):
    for cl in ("qmf", "cqmf"):
        for r in MC.R_CORE:
            if cl == "cqmf" and r == 0:
                continue
            add(var, f"decay_M2_{cl}_r{r:g}_null{null}", f"--variant M2 --r {r:g} --proto decay --null {null} --null_seed 0 --closure {cl} --ics decay")
# longest first within each file: driven before decay, cqmf before qmf
key = lambda s: (0 if "--proto driven" in s else 1, 0 if "cqmf" in s else 1)
main.sort(key=key); var.sort(key=key)
open("results/m2/theory/args/main.txt", "w").write("\n".join(main) + "\n")
open("results/m2/theory/args/var.txt", "w").write("\n".join(var + ["code/m2_mf_glin.py"]) + "\n")
json.dump(idx, open("results/m2/theory/args/index.json", "w"), indent=0)
print(len(main), "main lines;", len(var) + 1, "var lines")
