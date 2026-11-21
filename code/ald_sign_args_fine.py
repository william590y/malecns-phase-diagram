"""E2 (transmitter signs): refinement sweeps around the sustained onset of each deterministic condition.
Grid: 20 log-spaced points from 0.7 to 1.25 x the coarse onset (ratio 1.031), B = 64, T = 3000, push engine
(--push_max 0.008, the measured B = 64 crossover). Coarse onsets from results/aldef/signs/summary.json (coarse sweeps);
baseline from the earlier pd2 analysis (results/analysis2/flow_sim_boundaries.json: r0 0.266, r1 0.344, r3 0.456).
-> results/aldef/signs/args_fine.txt, args_fine_index.json ; outputs results/aldef/signs/sim_fine/<cond>_r<r>.npz
"""
import json
import numpy as np

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/aldef/signs"
S = json.load(open(f"{OUT}/summary.json"))
fb = {int(x["r"]): x["on"] for x in json.load(open(f"{ROOT}/results/analysis2/flow_sim_boundaries.json"))["rows"]
      if x["r"] in (0.0, 1.0, 3.0)}
det = {"base": "", "lLN1bc_inh": "--sign_override \"lLN1_bc=-1\"", "lLN1bc_zero": "--sign_override \"lLN1_bc=0\"",
       "lLN1bc_lLN2_inh": "--sign_override \"lLN1_bc;type:lLN2.*&sign:+1=-1\""}
lines, index = [], []
for name, extra in det.items():
    for r in (0, 1, 3):
        if name == "lLN1bc_inh" and r == 0:
            continue
        on = fb[r] if name == "base" else S["det"][f"{name}_r{r}"]["bounds"]["on"]
        G = [float(f"{x:.4g}") for x in on * np.exp(np.linspace(np.log(0.7), np.log(1.25), 20))]
        out = f"{OUT}/sim_fine/{name}_r{r}.npz"
        lines.append(f"--exp decay --norm raw --T 3000 --engine push --push_max 0.008 --region \"VNC=superclass:vnc.*\" "
                     f"--r {r} --B 64 --seed {2000 + len(lines)} {extra} --g {' '.join(f'{x:g}' for x in G)} --out {out}"
                     .replace("  ", " "))
        index.append({"task": len(lines) - 1, "cond": name, "r": r, "on_coarse": on, "g": G, "out": out})
open(f"{OUT}/args_fine.txt", "w").write("\n".join(lines) + "\n")
json.dump(index, open(f"{OUT}/args_fine_index.json", "w"), indent=1)
print(len(lines), "tasks")
