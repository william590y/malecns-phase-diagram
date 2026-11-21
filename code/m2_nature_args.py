"""m2_nature_args.py -- task lines for lane m2/nature (results/m2/nature/plan.md). One line = one ald_sim.py run.

  python m2_nature_args.py fss1     FSS stage 1 (decay_h0 on every FSS graph + the 167k nulls not run by lane decay)
  python m2_nature_args.py fss2     FSS stage 2: refine(lo, hi, 8) inside the tau = 200 / 3000 cells (reads stage-1 sums)
  python m2_nature_args.py fssx     FSS stage 1b: extend curves that have not reached tau = 3000 (4 more coarse points)
  python m2_nature_args.py dfss     driven FSS (plan section 2)
  python m2_nature_args.py hyst     hysteresis ramps (plan section 3)
  python m2_nature_args.py aval     pulse avalanches (plan section 4)
Writes results/m2/nature/args_<name>.txt and index_<name>.json (one dict per line: part, model, family, N, seed, r, g,
out, ...). Outputs go to results/m2/nature/sim/<part>/.
"""
import json, os, re, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import m2_common as M
import m2_nature_graphs as NG

OUT = f"{M.ROOT}/results/m2/nature"
SIM = f"{OUT}/sim"
BND = json.load(open(f"{M.ROOT}/results/m2/decay/boundaries.json"))
GC = M.G_COARSE


def g_bound(model, r, key="on"):
    """Full-graph boundary from the decay lane (rows; M1 falls back to m1_reference = analysis2)."""
    k = f"{model}_real_r{r:g}"
    if k in BND["rows"]:
        return float(BND["rows"][k][key]["g"])
    if model == "M1":
        return float(BND["m1_reference"][f"M1ref_real_r{r:g}"][key]["g"])
    raise KeyError(k)


def strip_regions(flags, keep=()):
    """Drop the extra --region flags (they can select nothing on small graphs), keeping those named in keep."""
    out = re.sub(r'--region "([^"=]+)=[^"]*"', lambda m: m.group(0) if m.group(1) in keep else "", flags)
    return re.sub(r"\s+", " ", out).strip()


def seed_of(name):
    """Deterministic per-line seed (stable across reruns)."""
    import zlib
    return zlib.crc32(name.encode()) % 2_000_000_000


def line_graph(proto, graph_path, g, r, out, seed, B=64, keep=(), **pk):
    fl = strip_regions(M.protocol(proto, **pk), keep)
    gs = " ".join(f"{x:g}" for x in np.atleast_1d(g))
    return (f"--graph {graph_path} {fl} {M.MODEL['M2']} --r {r:g} --g {gs} --B {B} --seed {seed} --out {out}")


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/args_{name}.txt", "w") as f:
        for r in rows:
            f.write(r["line"] + "\n")
    json.dump([{k: v for k, v in r.items() if k != "line"} for r in rows], open(f"{OUT}/index_{name}.json", "w"),
              indent=0)
    print(f"{name}: {len(rows)} lines -> {OUT}/args_{name}.txt")


# ---------------------------------------------------------------------------------------------------------- FSS (a)
FSS_R = {"cnull": (0, 1, 4), "realsubcn": (0, 1, 4), "realsub": (0, 1), "realsubu": (0, 1, 4)}
G1_SMALL = GC[(GC >= 0.68) & (GC <= 11.3)]          # 18 points
G1_FULL = GC[(GC >= 0.95) & (GC <= 3.02)]           # 8 points


def fss_graphs():
    out = []
    for kind, N, s in NG.all_graphs():
        for r in FSS_R[kind]:
            out.append((kind, N, s, r))
    for s in (1, 2):                                    # 167k nulls not run by lane decay (s0 r0/1/4, s1 r1 exist)
        for r in (0, 1, 4):
            if s == 1 and r == 1:
                continue
            out.append(("cnull", NG.NFULL, s, r))
    return out


def fss_line(kind, N, s, r, g, tag):
    name = f"fss_{kind}_N{N}_s{s}_r{r:g}_{tag}"
    out = f"{SIM}/fss/{name}.npz"
    seed = seed_of(name)
    if N == NG.NFULL:
        line = M.args_line("M2", "decay_h0", g, r, out, seed, B=64, extra=f"--null --null_seed {s}")
    else:
        line = line_graph("decay_h0", NG.path(kind, N, s), g, r, out, seed)
    return {"part": "fss", "stage": tag, "model": "M2", "family": kind, "N": N, "gseed": s, "r": r,
            "g": [float(x) for x in np.atleast_1d(g)], "seed": seed, "name": name, "out": out, "line": line}


def fss1(only=None):
    rows = []
    for kind, N, s, r in fss_graphs():
        if only and kind != only:
            continue
        if N == NG.NFULL:
            rows.append(fss_line(kind, N, s, r, G1_FULL[:4], "c0"))
            rows.append(fss_line(kind, N, s, r, G1_FULL[4:], "c1"))
        else:
            rows.append(fss_line(kind, N, s, r, G1_SMALL, "c0"))
    write("fss1" + (only or ""), rows)


def fss_refine(stage_name, cells_file):
    """cells_file: results/m2/nature/fss_cells.json from m2_nature_fss.py cells: {"family|N|s|r": {"refine": [[lo, hi],
    ...], "extra": [g, ...]}} -> refine(lo, hi, 8) points of every cell + the extra (extension) points."""
    cells = json.load(open(cells_file))
    rows = []
    for key, c in cells.items():
        kind, N, s, r = key.split("|"); N = int(N); s = int(s); r = float(r)
        gs = {float(x) for lo, hi in c["refine"] for x in (M.refine(lo, hi, 8) if hi / lo > 1.0001 else [])}
        gs |= {float(x) for x in c["extra"]}
        gs = sorted(gs)
        if not gs:
            continue
        if N == NG.NFULL or len(gs) > 18:
            h = (len(gs) + 1) // 2
            for j, part in enumerate((gs[:h], gs[h:])):
                if part:
                    rows.append(fss_line(kind, N, s, r, part, f"{stage_name}{j}"))
        else:
            rows.append(fss_line(kind, N, s, r, gs, stage_name))
    write(stage_name, rows)


# ---------------------------------------------------------------------------------------------------------- driven FSS
DFSS_G = GC[(GC >= 0.42) & (GC <= 11.3)]            # 20 points


def dfss(only=None):
    rows = []
    graphs = [("cnull", N, s) for N in (10000, 20000, 40000, 80000) for s in (1, 2)]
    graphs += [(k, N, s) for k in ("realsubcn", "realsubu") for N in NG.REAL_N for s in (1, 2)]
    if only:
        graphs = [x for x in graphs if x[0] == only]
    for kind, N, s in graphs:
        for r in (0, 1):
            for j, gpart in enumerate((DFSS_G[:10], DFSS_G[10:])):
                name = f"dfss_{kind}_N{N}_s{s}_r{r:g}_c{j}"
                out = f"{SIM}/dfss/{name}.npz"; seed = seed_of(name)
                line = line_graph("driven", NG.path(kind, N, s), gpart, r, out, seed, keep=("ORN", "nonORN"),
                                  counts=False)
                rows.append({"part": "dfss", "model": "M2", "family": kind, "N": N, "gseed": s, "r": r,
                             "g": [float(x) for x in gpart], "seed": seed, "name": name, "out": out, "line": line})
    write("dfss" + (only or ""), rows)


# ---------------------------------------------------------------------------------------------------------- hysteresis
X_RAMP = np.geomspace(0.5, 2.0, 17)


def hyst():
    rows = []
    for proto in ("ramp_driven", "ramp_h0_down"):
        for model in ("M1", "M2"):
            for r in M.R_CORE:
                gon = g_bound(model, r)
                g = np.round(X_RAMP * gon, 5)
                if proto == "ramp_h0_down":
                    g = g[::-1]
                Ds = [50, 200, 800]
                if model == "M2" and r in (0, 1, 4):
                    Ds.append(3200)
                for D in Ds:
                    reps = (0, 1) if (model == "M2" and r in (0, 1, 4) and D == 800) else (0,)
                    for rep in reps:
                        name = f"hyst_{proto}_{model}_r{r:g}_D{D}_k{rep}"
                        out = f"{SIM}/hyst/{name}.npz"; seed = seed_of(name)
                        pk = {"dwell": D}
                        if proto == "ramp_h0_down":
                            pk["gap"] = 10.0
                        line = M.args_line(model, proto, g, r, out, seed, B=32, **pk)
                        rows.append({"part": "hyst", "proto": proto, "model": model, "r": r, "D": D, "rep": rep,
                                     "g": [float(x) for x in g], "g_on": gon, "seed": seed, "name": name, "out": out,
                                     "line": line})
    # longest first (better packing)
    rows.sort(key=lambda d: -d["D"])
    write("hyst", rows)


# ---------------------------------------------------------------------------------------------------------- avalanches
G_AVAL = GC[(GC >= 0.1) & (GC <= 3.05)]             # 21 points 0.1135 .. 3.019


def aval():
    rows = []
    for model in ("M1", "M2"):
        for r in M.R_CORE:
            gon = g_bound(model, r)
            gs = G_AVAL[G_AVAL <= 1.25 * gon]
            for G in (0, 10, 100):
                # chunks of <= 5 g (the dense points near the onset cost the most)
                for j in range(0, len(gs), 5):
                    gpart = gs[j:j + 5]
                    name = f"aval_{model}_r{r:g}_G{G}_c{j // 5}_k0"
                    out = f"{SIM}/aval/{name}.npz"; seed = seed_of(name)
                    line = M.args_line(model, "aval_pulse", gpart, r, out, seed, B=64, gap=float(G))
                    rows.append({"part": "aval", "model": model, "r": r, "G": G, "rep": 0,
                                 "g": [float(x) for x in gpart], "g_on": gon, "seed": seed, "name": name, "out": out,
                                 "line": line})
    write("aval", rows)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fss1":
        fss1(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "dfss":
        dfss(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd in ("fss2", "fssx", "fss3"):
        fss_refine(cmd, sys.argv[2])
    else:
        {"dfss": dfss, "hyst": hyst, "aval": aval}[cmd]()
