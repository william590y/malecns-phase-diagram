"""m2_modules_args.py -- lane m2/modules: ald_sim.py argument lines for every run set (plan.md sections 4-6).

Sets (one line = one array task; index written next to the args file):
  LRa   Part B (a): h = 0, starts all/AL_L/AL_R/hemiL/hemiR, B 64, T 2000, g = {1,1.25,1.6,2.5,4,8} x onset(row)
        M2 R_CORE; M2s and M2_tau30 (AL_L, AL_R) at r 0,1,4; M1 (5 starts) at r 0,1,4 with pd3's M1 onsets.
  ICa   Part C (a): h = 0, module / random starts, B 32, T 2000, same g; M2 r 0,1,2,4; M1 r 0,1 (g x 1.25, 2.5, 8).
  DRb   Parts B+C (b): ORN 0.04 on throughout, T 3000, g = {.5,.8,.95,1.05,1.25,1.6,2.5,4} x DT->SS(row);
        M2 r 0,1,4: quiet/all/AL_L/AL_R/hemiL/hemiR (B 64) + nonAL/CB/VNC/OL/isl1/rand10 (B 32);
        M2 quiet/all at r 0.25, 0.5, 2, 6; M1 quiet/all/AL_L/AL_R at r 0, 1.
  ISO   Part D (ii): module subgraphs alone, h = 0, decay start, B 64, T 3000, G_COARSE; M1 and M2; r 0, 1, 4.
  SMOKE short versions of one line of each kind.
Usage: python m2_modules_args.py SET   ->  results/m2/modules/sim/args_<SET>.txt + index_<SET>.json
"""
import json, os, sys
import numpy as np

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
import m2_common as M

OUT = f"{ROOT}/results/m2/modules"
REG = f"{OUT}/groups/regions.npz"
RJ = json.load(open(f"{OUT}/groups/regions.json"))
ENG = "--engine push --push_max 0.012"
REGFLAGS = M.region_flags() + " " + " ".join(f'--region "{r["name"]}=idx:{REG}:{r["key"]}"' for r in RJ["regions"])
XA = (1.0, 1.25, 1.6, 2.5, 4.0, 8.0)
XB = (0.5, 0.8, 0.95, 1.05, 1.25, 1.6, 2.5, 4.0)
GON_M1 = {0: 0.266, 0.5: 0.306, 1: 0.344, 2: 0.402, 3: 0.456, 4: 0.508, 6: 0.579}   # pd3 spec values (pd3 M1 used these)
BND = json.load(open(f"{ROOT}/results/m2/decay/boundaries.json"))["rows"]
CUR = json.load(open(f"{ROOT}/results/m2/driven/curves.json"))
H04 = M.H_ORN[0.04]


def onset(model, r):
    if model == "M1":
        return GON_M1[r]
    return float(BND[f"{model}_real_r{r:g}"]["on"]["g"])


def dtss(model, r):
    s = "A2" if model == "M1" else "A1"
    return float(CUR[f"{s}|{model}|real|0.04|{r:g}|False"]["lines"]["DT_SS"])


def init_flag(start, setting):
    if start == "all":
        return "--exp decay" if setting == "a" else "--exp quiet --init all"
    if start == "quiet":
        return "--exp quiet"
    if start.startswith("rand"):
        return f"--exp quiet --init all --init_frac {int(start[4:]) / 100:g}"
    return f'--exp quiet --init "idx:{REG}:S_{start}"'


def line(setting, model, r, start, gs, B, seed, out, T=None):
    gtxt = " ".join(f"{g:.5g}" for g in gs)
    if setting == "a":
        T = T or 2000
        wins = " ".join(f"--win {a}:{a + 500}" for a in range(500, int(T), 500))
        base = f"--norm raw --T {T:g} --reg_last {wins} --rec_lin 5 {ENG} {REGFLAGS}"
    else:
        T = T or 3000
        wins = " ".join(f"--win {a}:{a + 500}" for a in range(500, int(T), 500))
        base = f"--norm raw --T {T:g} --drive olfactory={H04} --reg_last {wins} --rec_lin 5 {ENG} {REGFLAGS}"
    return (f"{init_flag(start, setting)} {base} {M.MODEL[model]} --r {r:g} --g {gtxt} --B {B} --seed {seed} "
            f"--out {out}").replace("  ", " ")


def build(SET):
    L = []; idx = []
    os.makedirs(f"{OUT}/sim/{SET}", exist_ok=True)

    def add(setting, model, r, start, gs, B, T=None, tag=""):
        seed = 61000 + 1000 * ["LRa", "ICa", "DRb", "ISO", "SMOKE", "DR2"].index(SET) + len(L)
        name = f"{model}_r{r:g}_{start}{tag}"
        out = f"{OUT}/sim/{SET}/{name}.npz"
        L.append(line(setting, model, r, start, gs, B, seed, out, T))
        idx.append({"set": SET, "setting": setting, "model": model, "r": r, "start": start, "g": [float(f"{g:.5g}") for g in gs],
                    "B": B, "T": T or (2000 if setting == "a" else 3000), "seed": seed, "out": out, "name": name})

    if SET == "LRa":
        for r in M.R_CORE:
            for s in ("all", "AL_L", "AL_R", "hemiL", "hemiR"):
                add("a", "M2", r, s, [onset("M2", r) * x for x in XA], 64)
        for mod in ("M2s", "M2_tau30"):
            for r in (0, 1, 4):
                for s in ("AL_L", "AL_R"):
                    add("a", mod, r, s, [onset(mod, r) * x for x in XA], 64)
        for r in (0, 1, 4):
            for s in ("all", "AL_L", "AL_R", "hemiL", "hemiR"):
                add("a", "M1", r, s, [round(GON_M1[r] * x, 4) for x in XA], 64)
    elif SET == "ICa":
        for r in (0, 1, 2, 4):
            for s in ("AL", "nonAL", "CB", "VNC", "OL", "isl1", "isl3", "isl5", "rand10", "rand1"):
                add("a", "M2", r, s, [onset("M2", r) * x for x in XA], 32)
        for r in (0, 1):
            for s in ("AL", "nonAL", "CB", "VNC", "OL", "isl1", "rand10"):
                add("a", "M1", r, s, [round(GON_M1[r] * x, 4) for x in (1.25, 2.5, 8.0)], 32)
    elif SET == "DRb":
        for r in (0, 1, 4):
            for s in ("quiet", "all", "AL_L", "AL_R", "hemiL", "hemiR"):
                add("b", "M2", r, s, [dtss("M2", r) * x for x in XB], 64)
            for s in ("nonAL", "CB", "VNC", "OL", "isl1", "rand10"):
                add("b", "M2", r, s, [dtss("M2", r) * x for x in XB], 32)
        for r in (0.25, 0.5, 2, 6):
            for s in ("quiet", "all"):
                add("b", "M2", r, s, [dtss("M2", r) * x for x in XB], 64)
        for r in (0, 1):
            for s in ("quiet", "all", "AL_L", "AL_R"):
                add("b", "M1", r, s, [dtss("M1", r) * x for x in XB], 64)
    elif SET == "ISO":
        for mod in ("AL", "AL_L", "AL_R", "isl1", "isl3", "isl5", "CB", "VNC", "OL"):
            for model in ("M1", "M2"):
                for r in (0, 1, 4):
                    seed = 61000 + 3000 + len(L); name = f"{model}_r{r:g}_{mod}"
                    out = f"{OUT}/sim/ISO/{name}.npz"
                    gs = M.G_COARSE
                    L.append(f"--exp decay --norm raw --T 3000 --no_default_regions {ENG} --graph {OUT}/graphs/{mod}.npz "
                             f"{M.MODEL[model]} --r {r:g} --g {' '.join(f'{g:g}' for g in gs)} --B 64 --seed {seed} "
                             f"--out {out}".replace("  ", " "))
                    idx.append({"set": SET, "model": model, "r": r, "module": mod, "g": [float(g) for g in gs], "B": 64,
                                "T": 3000, "seed": seed, "out": out, "name": name})
    elif SET == "DR2":                       # stage 2 of plan section 5: every (b) quiet-vs-all candidate, T = 10000, B 64
        ms = json.load(open(f"{OUT}/multistab.json"))["cells"]
        for key, v in ms.items():
            st, model, r = key.split("|")
            if st != "b":
                continue
            for rw in v["rows"]:
                if rw.get("quiet_vs_all", {}).get("candidate"):
                    for s in ("quiet", "all"):
                        add("b", model, float(r), s, [rw["g"]], 64, T=10000, tag=f"_g{rw['g']:.4g}")
    elif SET == "SMOKE":
        add("a", "M2", 1, "AL_L", [onset("M2", 1) * 1.6], 64, T=1000)
        add("b", "M2", 1, "hemiL", [dtss("M2", 1)], 64, T=1000)
        add("a", "M2", 1, "rand10", [onset("M2", 1) * 2.5], 32, T=200)
        seed = 65999; out = f"{OUT}/sim/SMOKE/iso_M2_r1_AL.npz"
        L.append(f"--exp decay --norm raw --T 200 --no_default_regions {ENG} --graph {OUT}/graphs/AL.npz {M.MODEL['M2']} "
                 f"--r 1 --g 2 4 --B 64 --seed {seed} --out {out}")
        idx.append({"set": SET, "model": "M2", "r": 1, "module": "AL", "g": [2, 4], "B": 64, "T": 200, "seed": seed, "out": out,
                    "name": "iso_M2_r1_AL"})
    else:
        raise ValueError(SET)
    open(f"{OUT}/sim/args_{SET}.txt", "w").write("\n".join(L) + "\n")
    json.dump(idx, open(f"{OUT}/sim/index_{SET}.json", "w"), indent=1)
    print(SET, len(L), "lines")


if __name__ == "__main__":
    os.makedirs(f"{OUT}/sim", exist_ok=True)
    for s in sys.argv[1:]:
        build(s)
