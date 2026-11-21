"""m2_decay_args.py -- argument lines for lane m2/decay (results/m2/decay/plan.md).

  python m2_decay_args.py stage1            -> results/m2/decay/sim/args_s1.txt + index_s1.json
  python m2_decay_args.py stage2 B          -> args_s2B.txt + index_s2B.json for rows of stage1_cells.json not yet
                                               in an earlier s2 batch (batches let stage 2 start while stage 1 runs)
  python m2_decay_args.py stage3a B         -> args_s3aB.txt + index_s3aB.json (same batching)
  python m2_decay_args.py stage3b           -> args_s3b.txt + index_s3b.json (needs boundaries.json after stage 2)
Each line is one ald_sim.py run built with m2_common.args_line (protocol decay_h0).
"""
import glob, json, os, sys, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_common as M

OUT = f"{M.ROOT}/results/m2/decay"
SIMD = f"{OUT}/sim"
GRAPH = {"real": "", "cnull": "--null --null_seed 0", "wnull": "--null --null_kind weights --null_seed 0",
         "cnull1": "--null --null_seed 1", "wnull1": "--null --null_kind weights --null_seed 1"}
R_VAR = (0, 1, 4)


def rows_stage1():
    rows = []
    for r in M.R_ALL:
        rows.append(("M2", "real", r))
    for r in M.R_CORE:
        rows.append(("M2s", "real", r))
    for gr in ("cnull", "wnull"):
        for r in M.R_CORE:
            rows.append(("M2", gr, r))
    for gr in ("cnull1", "wnull1"):
        rows.append(("M2", gr, 1))
    for v in ("M2_allsyn", "M2_KW09", "M2_N15", "M2_tau30"):
        for r in R_VAR:
            rows.append((v, "real", r))
    for v in ("M1s", "M1"):
        for r in R_VAR:
            rows.append((v, "real", r))
    return rows


def row_key(model, graph, r):
    return f"{model}_{graph}_r{r:g}"


def counts_reg(model, graph, r):
    return graph == "real" and ((model in ("M2", "M2s") and r in M.R_CORE) or model == "M1")


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:7], 16)


def line(model, graph, r, g, out, seed, B=64, extra="", counts=True, creg=False):
    ex = (GRAPH[graph] + " " + extra).strip()
    return M.args_line(model, "decay_h0", g, r, out, seed, B=B, extra=ex, counts=counts, counts_reg=creg)


def write(stage, lines, index):
    os.makedirs(SIMD, exist_ok=True)
    if os.path.exists(f"{SIMD}/args_{stage}.txt"):
        raise SystemExit(f"args_{stage}.txt exists (batches are never overwritten)")
    with open(f"{SIMD}/args_{stage}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    json.dump(index, open(f"{SIMD}/index_{stage}.json", "w"), indent=1)
    print(stage, len(lines), "tasks")


def stage1():
    lines, index = [], []
    G = M.G_COARSE
    for (model, graph, r) in rows_stage1():
        for p in range(4):
            gs = G[p::4]
            key = row_key(model, graph, r)
            out = f"{SIMD}/s1_{key}_p{p}.npz"
            sd = seed_of("s1", key, p)
            lines.append(line(model, graph, r, gs, out, sd, creg=counts_reg(model, graph, r)))
            index.append({"task": len(lines) - 1, "stage": "s1", "model": model, "graph": graph, "r": r, "part": p,
                          "g": gs.tolist(), "seed": sd, "out": out})
    write("s1", lines, index)


def submitted(stage):
    """row keys already in an earlier batch of this stage (index_<stage>*.json)."""
    done = set()
    for f in glob.glob(f"{SIMD}/index_{stage}*.json"):
        for e in json.load(open(f)):
            done.add(row_key(e["model"], e["graph"], e["r"]))
    return done


def stage2(batch):
    cells = json.load(open(f"{OUT}/stage1_cells.json"))
    done = submitted("s2")
    lines, index = [], []
    for key, c in cells.items():
        if key in done:
            continue
        model, graph, r = c["model"], c["graph"], c["r"]
        for b in ("S", "on", "D"):
            br = c.get(b)
            if not br or br[0] is None or br[1] is None:
                continue
            gs = M.refine(br[0], br[1], 8)
            out = f"{SIMD}/s2_{key}_{b}.npz"
            sd = seed_of("s2", key, b)
            lines.append(line(model, graph, r, gs, out, sd, creg=counts_reg(model, graph, r)))
            index.append({"task": len(lines) - 1, "stage": "s2", "model": model, "graph": graph, "r": r, "boundary": b,
                          "g": gs.tolist(), "seed": sd, "out": out})
    write(f"s2{batch}", lines, index)


def stage3a(batch):
    """long runs (T = 30000, 2 x 64 replicas) at the stage-1 isotonic tau = 100 ... 3000 of four rows."""
    cells = json.load(open(f"{OUT}/stage1_cells.json"))
    done = submitted("s3a")
    lines, index = [], []
    for key in ("M2_real_r0", "M2_real_r1", "M2_real_r4", "M2_cnull_r1"):
        if key in done or key not in cells:
            continue
        c = cells[key]
        gl = [c["g_at_tau"].get(str(tt)) for tt in (100, 200, 400, 800, 1600, 3000)]
        gl = [round(x, 5) for x in gl if x is not None]
        for gi, g in enumerate(gl):
            for p in range(2):
                out = f"{SIMD}/s3a_{key}_g{gi}_p{p}.npz"
                sd = seed_of("s3a", key, gi, p)
                f = M.args_line(c["model"], "decay_h0", g, c["r"], out, sd, B=64,
                                extra=(GRAPH[c["graph"]] + " --T 30000").strip(), counts=False)
                f = f.replace("--T 3000 ", "", 1)      # protocol's T replaced by the extra --T 30000
                assert "--T 30000" in f and "--T 3000 " not in f
                lines.append(f)
                index.append({"task": len(lines) - 1, "stage": "s3a", "model": c["model"], "graph": c["graph"],
                              "r": c["r"], "g": [g], "g_index": gi, "part": p, "seed": sd, "out": out, "T": 30000})
    write(f"s3a{batch}", lines, index)


def stage3b():
    bd = json.load(open(f"{OUT}/boundaries.json"))
    lines, index = [], []
    for key, e in bd["rows"].items():
        if e["model"] not in ("M2", "M2s") or e["graph"] not in ("real", "cnull", "wnull") or e["r"] not in M.R_CORE:
            continue
        gS = e["S"]["g"]
        if gS is None:
            continue
        out = f"{SIMD}/s3b_{key}_S.npz"
        sd = seed_of("s3b", key)
        lines.append(line(e["model"], e["graph"], e["r"], [round(gS, 5)], out, sd, counts=False,
                          extra="--occ_win 50:400"))
        index.append({"task": len(lines) - 1, "stage": "s3b", "model": e["model"], "graph": e["graph"], "r": e["r"],
                      "g": [round(gS, 5)], "seed": sd, "out": out, "occ_win": "50:400"})
    write("s3b", lines, index)


def stage_osc():
    """plan section 9: OSC finite-size check on small graphs (M2, r = 0, default regions only)."""
    D = f"{OUT}/graphs"
    G = [x for x in M.G_COARSE if x >= 0.8]
    lines, index = [], []
    for kind, Ns in (("cnull", (20000, 40000, 80000)), ("realsub", (40000, 80000))):
        for N in Ns:
            gf = f"{D}/{kind}_N{N}_s1.npz"
            for p in range(2):
                gs = G[p::2]
                out = f"{SIMD}/x9_{kind}{N // 1000}k_r0_p{p}.npz"
                sd = seed_of("x9", kind, N, p)
                gl = " ".join(f"{x:g}" for x in gs)
                lines.append(f"--graph {gf} --exp decay --norm raw --T 3000 --reg_last --win 1500:3000 --engine push "
                             f"--push_max 0.012 --rec_counts --counts_win 1500:3000 {M.MODEL['M2']} --r 0 --g {gl} --B 64 "
                             f"--seed {sd} --out {out}")
                index.append({"task": len(lines) - 1, "stage": "x9", "model": "M2", "graph": f"{kind}{N // 1000}k",
                              "N": N, "r": 0, "part": p, "g": [float(x) for x in gs], "seed": sd, "out": out})
    os.makedirs(SIMD, exist_ok=True)
    open(f"{SIMD}/args_x9.txt", "w").write("\n".join(lines) + "\n")
    json.dump(index, open(f"{SIMD}/osccheck_index.json", "w"), indent=1)
    print("x9", len(lines), "tasks")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("stage2", "stage3a"):
        {"stage2": stage2, "stage3a": stage3a}[cmd](sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        {"stage1": stage1, "stage3b": stage3b, "osc": stage_osc}[cmd]()
