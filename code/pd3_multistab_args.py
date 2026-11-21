"""Write the args files of the pd3 multistability arrays (one pd3_multistab.py call per line) and list missing outputs.

Usage: python pd3_multistab_args.py            -> results/pd3/multistability/args_{validate,m1,m2a,m2c,m3,m3b}.txt
       python pd3_multistab_args.py --missing ARGSFILE   -> comma-separated 0-based indices whose output file is missing
Grid sources (plan.md section 5 / task spec):
  M1  g_on(r) = simulated tau = 3000 onsets of results/analysis2/flow_sim_boundaries.json (rounded to 3 decimals as in the spec);
      closure B1 windows r = 4: g 0.29 / 0.32 / 0.40, r = 6: 0.43 / 0.58 / 0.70 (theory_module2.json coarse_2module).
  M2a r in {0, .5, 1, 2, 4, 6} x h in {1e-5, 1e-4} x D in {50, 200, 800}; + h = 0 down-only at D = 200.
  M2c g in {0.6, 2, 8}, h = 1e-5, D = 200.
  M3  r in {0, 1, 2, 4} x g in {0.4, 0.7, 1.2, 2.5, 6, 15}.
  M3b every (r, g) cell with >= 2 QMF fixed-point attractors (results/pd3/module_theory/qmf_multi.json), 3 cells per task.
"""
import argparse, json, os, re
import numpy as np

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/pd3/multistab"
AD = f"{ROOT}/results/pd3/multistability"
S = "pd3_multistab.py"


def lines():
    fb = json.load(open(f"{ROOT}/results/analysis2/flow_sim_boundaries.json"))
    gon_file = {round(row["r"], 3): row["on"] for row in fb["rows"]}
    R1 = [0, 0.5, 1, 2, 3, 4, 6]
    GON = {0: 0.266, 0.5: 0.306, 1: 0.344, 2: 0.402, 3: 0.456, 4: 0.508, 6: 0.579}     # spec values
    for r in R1:
        assert abs(gon_file[r] - GON[r]) < 6e-4, (r, gon_file[r], GON[r])
    EXTRA = {4: [0.29, 0.32, 0.40], 6: [0.43, 0.58, 0.70]}
    L = {}
    L["validate"] = [(f"{S} validate --r 1 --seed 11", f"{OUT}/validate_r1.npz"),
                     (f"{S} validate --r 1 --seed 12 --force_split", f"{OUT}/validate_r1_split.npz")]
    L["m1"] = []
    for i, r in enumerate(R1):
        G = sorted(set([round(GON[r] * x, 4) for x in (1.0, 1.25, 1.6, 2.5, 4, 8)] + EXTRA.get(r, [])))
        L["m1"].append((f"{S} alsym --r {r:g} --g {' '.join(f'{g:g}' for g in G)} --seed {100 + i}", f"{OUT}/al_sym_r{r:g}.npz"))
    L["m2a"] = []
    k = 0
    for r in [0, 0.5, 1, 2, 4, 6]:
        for h in [1e-5, 1e-4]:
            for D in [50, 200, 800]:
                L["m2a"].append((f"{S} sweep --r {r:g} --h {h:g} --D {D:g} --seed {1000 + k}", f"{OUT}/sweep_r{r:g}_h{h:g}_D{D:g}.npz")); k += 1
    for r in [0, 0.5, 1, 2, 4, 6]:
        L["m2a"].append((f"{S} sweep --r {r:g} --h 0 --D 200 --down_only --seed {1000 + k}", f"{OUT}/sweep_r{r:g}_h0_D200_down.npz")); k += 1
    L["m2c"] = [(f"{S} rsweep --g {g:g} --h 1e-5 --D 200 --seed {2000 + i}", f"{OUT}/rsweep_g{g:g}.npz") for i, g in enumerate([0.6, 2, 8])]
    L["m3"] = []
    k = 0
    for r in [0, 1, 2, 4]:
        for g in [0.4, 0.7, 1.2, 2.5, 6, 15]:
            L["m3"].append((f"{S} ic --r {r:g} --g {g:g} --seed {3000 + k}", f"{OUT}/ic_r{r:g}_g{g:g}.npz")); k += 1
    qm = json.load(open(f"{ROOT}/results/pd3/module_theory/qmf_multi.json"))["per_r"]
    cells = []
    for r, v in qm.items():
        gs = sorted(set(round(p["g"], 4) for p in v["pairs"]))
        cells += [(float(r), g) for g in gs]
    L["m3b"] = []
    by_r = {}
    for r, g in cells:
        by_r.setdefault(r, []).append(g)
    k = 0
    for r, gs in by_r.items():
        for j in range(0, len(gs), 3):
            ch = gs[j:j + 3]
            outs = ";".join(f"{OUT}/icq_r{r:g}_g{g:.4f}.npz" for g in ch)
            L["m3b"].append((f"{S} icq --cells {','.join(f'{r:g}:{g:.4f}' for g in ch)} --seed {4000 + k}", outs)); k += 1
    return L


def m2b_lines():
    """M2b (plan 5.3): one task per distinct loop centre of every pre-registered (spec) hysteresis candidate in
    results/pd3/multistab/analysis_m2a.json, plus CONTROLS (added): the largest remaining D = 800 loop of each h (whole sweep), which is
    not a candidate, to show what a lag loop does under the same test."""
    d = json.load(open(f"{OUT}/analysis_m2a.json"))["decisions"]
    todo = {}
    for key, v in d.items():
        if v["candidate_spec"]:
            todo.setdefault((v["file800"], v["loop_centre_seg_up"]), []).append(key)
        if v.get("PR_candidate_spec"):
            todo.setdefault((v["file800"], v["PR_loop_centre_seg_up"]), []).append(key + "_PR")
    for h in ("1e-05", "0.0001"):
        best = max((v for k, v in d.items() if k.endswith(f"_h{h}_area_all") or f"_h{h}_" in k and k.endswith("area_all")),
                   key=lambda v: v["loop_centre_dlnrho"])
        todo.setdefault((best["file800"], best["loop_centre_seg_up"]), []).append("control_largest_loop_h" + h)
    out = []
    for i, ((f, k), why) in enumerate(sorted(todo.items())):
        sw = f"{ROOT}/{f}"
        m = json.loads(str(np.load(sw, allow_pickle=True)["meta"]))
        g = m["g_grid"][k]
        out.append((f"{S} m2b --sweep {sw} --seg {k} --T 5000 --seed {5000 + i}  # {','.join(why)}",
                    f"{OUT}/m2b_r{m['r']:g}_h{m['h']:g}_D{m['D']:g}_g{g:.4f}.npz"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", default=None)
    ap.add_argument("--m2b", action="store_true")
    a = ap.parse_args()
    if a.m2b:
        ls = m2b_lines()
        with open(f"{AD}/args_m2b.txt", "w") as f:
            for cmd, _ in ls:
                f.write(cmd + "\n")
        with open(f"{AD}/outputs_m2b.txt", "w") as f:
            for _, o in ls:
                f.write(o + "\n")
        print("m2b", len(ls)); print("\n".join(c for c, _ in ls))
        return
    L = lines()
    if a.missing:
        name = re.sub(r"^args_|\.txt$", "", os.path.basename(a.missing))
        if name == "m2b":
            outs = [l.strip() for l in open(f"{AD}/outputs_m2b.txt")]
            print(",".join(str(i) for i, o in enumerate(outs) if not os.path.exists(o))); return
        miss = [str(i) for i, (_, o) in enumerate(L[name]) if not all(os.path.exists(x) for x in o.split(";"))]
        print(",".join(miss))
        return
    os.makedirs(AD, exist_ok=True)
    for name, ls in L.items():
        with open(f"{AD}/args_{name}.txt", "w") as f:
            for cmd, _ in ls:
                f.write(cmd + "\n")
        with open(f"{AD}/outputs_{name}.txt", "w") as f:
            for _, o in ls:
                f.write(o + "\n")
        print(name, len(ls))


if __name__ == "__main__":
    main()
