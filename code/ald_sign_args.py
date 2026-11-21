"""E2 (transmitter signs): write the ald_sim.py argument files for the coarse (g, r) sweeps.

Conditions (r = 0, 1, 3; raw norm, h = 0, fully active start, T = 3000 as in the stage-1/pd2 runs):
  base            consensus signs (rerun here with region traces, same grid)
  lLN1bc_inh  (i) lLN1_bc -> -r          (identical to (ii) at r = 0, so r = 1, 3 only)
  lLN1bc_zero (ii) lLN1_bc -> 0
  lLN1bc_lLN2_inh (iii) lLN1_bc and the cholinergic lLN2 types (lLN2T_a/b/c/e, lLN2X04/05/11/12; 40 neurons) -> -r
  ens_type    (iv) type-level sign ensemble (Dale: one draw per cell type), --ens_gt ignore, --ens_seed 1, 24 draws
  ens_neuron  (iv') neuron-level sign ensemble, --ens_gt ignore, --ens_seed 2, 16 draws
g grid: 42 log-spaced points 0.12 .. 30 (ratio 1.144). Deterministic: B = 32; ensemble: B = 16 per draw, one task per
(draw, r). Engine: --engine push with --push_max 0.012 (B 32) / 0.02 (B 16), the measured spmm/push crossovers (event-driven; validated in results/aldef/signs/validation/validate_push.json).
Usage: python ald_sign_args.py -> results/aldef/signs/args_sim.txt (one line per array task) + args_sim_index.json
"""
import json
import numpy as np

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/aldef/signs"
G = [float(f"{x:.4g}") for x in 0.12 * 250 ** (np.arange(42) / 41)]
gs = " ".join(f"{x:g}" for x in G)
common = f"--exp decay --norm raw --T 3000 --engine push --region \"VNC=superclass:vnc.*\" --g {gs}"
det = {"base": "", "lLN1bc_inh": "--sign_override \"lLN1_bc=-1\"", "lLN1bc_zero": "--sign_override \"lLN1_bc=0\"",
       "lLN1bc_lLN2_inh": "--sign_override \"lLN1_bc;type:lLN2.*&sign:+1=-1\""}
lines, index = [], []
for name, extra in det.items():
    for r in (0, 1, 3):
        if name == "lLN1bc_inh" and r == 0:
            continue
        seed = 1000 + len(lines)
        out = f"{OUT}/sim/{name}_r{r}.npz"
        lines.append(f"{common} --r {r} --B 32 --push_max 0.012 --seed {seed} {extra} --out {out}".replace("  ", " "))
        index.append({"task": len(lines) - 1, "cond": name, "r": r, "draw": None, "out": out})
for name, unit, seed_e, K in (("ens_type", "type", 1, 24), ("ens_neuron", "neuron", 2, 16)):
    for r in (0, 1, 3):
        for d in range(K):
            seed = 1000 + len(lines)
            out = f"{OUT}/sim/{name}_d{d:02d}_r{r}.npz"
            lines.append(f"{common} --r {r} --B 16 --push_max 0.02 --seed {seed} --sign_ensemble {K} --ens_unit {unit} --ens_gt ignore "
                         f"--ens_seed {seed_e} --ens_only {d} --out {out}")
            index.append({"task": len(lines) - 1, "cond": name, "r": r, "draw": d, "out": out})
open(f"{OUT}/args_sim.txt", "w").write("\n".join(lines) + "\n")
json.dump({"g": G, "tasks": index}, open(f"{OUT}/args_sim_index.json", "w"), indent=1)
print(len(lines), "tasks;", len(G), "g points", G[0], G[-1])
