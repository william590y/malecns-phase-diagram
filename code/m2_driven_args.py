"""m2_driven_args.py -- task lines for lane m2/driven (results/m2/driven/plan.md; 2026-11-09).

Usage: python m2_driven_args.py WAVE      WAVE in {A, B, C:<json of extra point sets>}
Writes results/m2/driven/args_<WAVE>.txt (one ald_sim.py line per array task) and index_<WAVE>.json (per line: set,
model, graph, ORN level, r, g list, seed, out). Lines come from m2_common.args_line (protocol 'driven', counts_reg on).
Each task runs ~8 g values, interleaved over the grid so that tasks have similar cost.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_common as M

OUT = f"{M.ROOT}/results/m2/driven"
RUNS = f"{OUT}/runs"
G40 = [float(x) for x in M.G_COARSE]
G34 = [g for g in G40 if g <= 11.3]                       # B sets stop at 11.21 (plan.md section 1)
GPALP = [g for g in G40 if 0.09 <= g <= 3.1]              # B4: 0.096 .. 3.02 (21 points)
R_SUB = (0, 1, 4)
GRAPH_FLAGS = {"real": "", "cnull": "--null --null_seed 0", "wnull": "--null --null_kind weights --null_seed 0"}
PALP_SEL = '"class:olfactory&~type:ORN_(VA4|VC1|VC2|VA7l|VM7d|VM7v)"'
SEED0 = {"A": 20000, "B": 30000, "C": 40000}


def chunks(gs, per=8):
    k = max(1, int(round(len(gs) / per)))
    return [gs[i::k] for i in range(k)]


def build(wave):
    lines, meta = [], []
    c_off = 1000 * len(glob.glob(f"{OUT}/index_C*.json")) if wave.startswith("C") else 0   # distinct seeds per C wave

    def add(set_, model, graph, orn, r, gs, palp=False, per=8, tag=""):
        for ci, gl in enumerate(chunks(gs, per)):
            gl = sorted(gl)
            name = f"{model}_{graph}_o{orn:.2f}_r{r:g}" + ("_palp" if palp else "") + tag + f"_c{ci}"
            assert not any(m["name"] == name and m["set"] == set_ for m in meta), name
            out = f"{RUNS}/{set_}/{name}.npz"
            seed = SEED0[wave[0]] + c_off + len(lines)
            line = M.args_line(model, "driven", gl, r, out, seed, B=64, extra=GRAPH_FLAGS[graph], orn=orn,
                               counts=True, counts_reg=True)
            if palp:
                assert "--sw_clamp_off olfactory" in line
                line = line.replace("--sw_clamp_off olfactory", f"--sw_clamp_off {PALP_SEL}")
            lines.append(line)
            meta.append({"set": set_, "model": model, "graph": graph, "orn": orn, "r": r, "g": gl, "palp": palp,
                         "seed": seed, "out": out, "name": name})

    if wave == "A":
        # A1 first (main diagram), then the R_CORE sets interleaved by r
        for r in M.R_ALL:
            add("A1", "M2", "real", 0.04, r, G40)
        for r in M.R_CORE:
            add("A2", "M1", "real", 0.04, r, G40)
            add("A3", "M2", "cnull", 0.04, r, G40)
            add("A3", "M2", "wnull", 0.04, r, G40)
            add("A4", "M2s", "real", 0.04, r, G40)
    elif wave == "B":
        for r in M.R_CORE:
            for orn in (0.01, 0.02, 0.07, 0.10):
                add("B1", "M2", "real", orn, r, G34)
        for r in R_SUB:
            for v in ("M2_allsyn", "M2_KW09", "M2_N15", "M2_tau30"):
                add("B2", v, "real", 0.04, r, G34)
            add("B2", "M2_tau30", "real", 0.10, r, G34)
            add("B3", "M1s", "real", 0.04, r, G34)
        for r in (0, 1):
            for m in ("M2", "M2s"):
                add("B4", m, "real", 0.04, r, GPALP, palp=True, per=7)
    elif wave.startswith("C"):
        # C:<file.json> with a list of {set, model, graph, orn, r, g:[...], palp}
        spec = json.load(open(wave.split(":", 1)[1]))
        for s in spec:
            add(s["set"], s["model"], s["graph"], s["orn"], s["r"], [float(x) for x in s["g"]], s.get("palp", False),
                per=s.get("per", 8), tag=s["tag"])
        # refinement names must not collide with any earlier wave's files
        old_names = {m["out"] for f in glob.glob(f"{OUT}/index_*.json") for m in json.load(open(f))}
        assert not any(m["out"] in old_names for m in meta), "refinement output would overwrite an existing run"
        wave = "C" + os.path.basename(wave.split(":", 1)[1]).replace(".json", "").replace("refine", "")
    else:
        raise SystemExit("wave A, B or C:<json>")
    for m in meta:
        os.makedirs(os.path.dirname(m["out"]), exist_ok=True)
    open(f"{OUT}/args_{wave}.txt", "w").write("\n".join(lines) + "\n")
    json.dump(meta, open(f"{OUT}/index_{wave}.json", "w"), indent=0)
    pts = sum(len(m["g"]) for m in meta)
    print(f"wave {wave}: {len(lines)} tasks, {pts} points -> {OUT}/args_{wave}.txt")
    return wave, len(lines)


if __name__ == "__main__":
    build(sys.argv[1])
