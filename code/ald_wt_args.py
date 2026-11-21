"""E4 (weight transforms in simulation): argument files for ald_sim.py.

Transforms: linear, cap:50, cap:20, cap:10, sqrt, log1p, binary (ald_sim --wtransform, --wnorm renorm = equal mean
in-strength 1). r = 0 and 1. Fully active start (--exp decay), h = 0, --norm raw, T = 3000 (the stage-1/pd2/E2 protocol),
--engine push. Regions (reg_rho traces + per-replica last-active time --reg_last): the default AL_module / AL_rest /
CNS_rest plus MB, MB_KC, VNC, OL, CX, LH, CB_other, AN_DN, lLN1_bc (results/aldef/weights/regions/*.npz).
Under --norm raw the global c only rescales g, so one sweep per transform gives every unit system:
  g (equal mean in-strength), g * lam1(T)/lam1(linear) (equal leading eigenvalue), g / g_lin(T).
COARSE (mode 'coarse'): 42 log-spaced g from 0.72 to 180 x g_lin(T) (ratio 1.144, as E2's 0.12..30 grid for linear);
  B = 32, push_max 0.012; split into two tasks per (T, r) with interleaved g (even / odd grid index) to halve wall time.
FINE (mode 'fine'): 20 log-spaced g from 0.7 to 1.25 x the coarse onset (tau = 3000 crossing, from summary.json of the
  coarse analysis), B = 64, push_max 0.008.
Usage: python ald_wt_args.py coarse | fine   -> results/aldef/weights/args_<mode>.txt + args_<mode>_index.json
"""
import json, sys
import numpy as np

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/aldef/weights"
TRANSFORMS = ["linear", "cap:50", "cap:20", "cap:10", "sqrt", "log1p", "binary"]
REGS = ["MB", "MB_KC", "VNC", "OL", "CX", "LH", "CB_other", "AN_DN", "lLN1_bc"]
reg_args = " ".join(f"--region \"{k}=idx:{OUT}/regions/{k}.npz\"" for k in REGS)
common = f"--exp decay --norm raw --T 3000 --engine push --wtransform {{tr}} --wnorm renorm --reg_last {reg_args}"
nm = lambda tr: tr.replace(":", "")
mode = sys.argv[1]
lines, index = [], []
if mode == "coarse":
    for tr in TRANSFORMS:
        spec = json.load(open(f"{OUT}/spec/{nm(tr)}.json"))
        G = np.array([float(f"{x:.4g}") for x in spec["g_lin"] * 0.72 * 250 ** (np.arange(42) / 41)])
        for r in (0, 1):
            for part in (0, 1):
                gs = G[part::2]
                out = f"{OUT}/sim/{nm(tr)}_r{r}_p{part}.npz"
                lines.append(f"{common.format(tr=tr)} --r {r} --B 32 --push_max 0.012 --seed {4000 + len(lines)} "
                             f"--g {' '.join(f'{x:g}' for x in gs)} --out {out}")
                index.append({"task": len(lines) - 1, "transform": tr, "r": r, "part": part, "g": gs.tolist(),
                              "g_lin": spec["g_lin"], "out": out})
elif mode == "fine":
    # 'fine' also carries the two causal AL tests (same array):
    #   noAL   = the 526-neuron AL module clamped off (--clamp_off module): does anything else ignite at the same g?
    #   ALonly = everything except the module clamped off (--clamp_off "~module"): can the AL module alone sustain there?
    #   grids: 18 log-spaced g from 0.85 to 4 x the coarse onset, B = 32, push_max 0.012.
    S = json.load(open(f"{OUT}/summary_coarse.json"))
    for tr in TRANSFORMS:
        for r in (0, 1):
            on = S["runs"][f"{nm(tr)}_r{r}"]["bounds"]["on"]
            if on is None:
                print("no coarse onset for", tr, r); continue
            G = [float(f"{x:.4g}") for x in on * np.exp(np.linspace(np.log(0.7), np.log(1.25), 20))]
            out = f"{OUT}/sim_fine/{nm(tr)}_r{r}.npz"
            lines.append(f"{common.format(tr=tr)} --r {r} --B 64 --push_max 0.008 --seed {5000 + len(lines)} "
                         f"--g {' '.join(f'{x:g}' for x in G)} --out {out}")
            index.append({"task": len(lines) - 1, "kind": "fine", "transform": tr, "r": r, "on_coarse": on, "g": G, "out": out})
            G2 = [float(f"{x:.4g}") for x in on * np.exp(np.linspace(np.log(0.85), np.log(4.0), 18))]
            for kind, clamp in (("noAL", "module"), ("ALonly", "~module")):
                out = f"{OUT}/sim_{kind}/{nm(tr)}_r{r}.npz"
                lines.append(f"{common.format(tr=tr)} --r {r} --B 32 --push_max 0.012 --seed {5000 + len(lines)} "
                             f"--clamp_off \"{clamp}\" --g {' '.join(f'{x:g}' for x in G2)} --out {out}")
                index.append({"task": len(lines) - 1, "kind": kind, "transform": tr, "r": r, "on_coarse": on, "g": G2,
                              "out": out})
open(f"{OUT}/args_{mode}.txt", "w").write("\n".join(lines) + "\n")
json.dump(index, open(f"{OUT}/args_{mode}_index.json", "w"), indent=1)
print(len(lines), "tasks")
