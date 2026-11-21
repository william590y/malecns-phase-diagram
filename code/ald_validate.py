"""Validation of ald_sim.py against swc.py (run on ONE GPU, inside a Slurm job; 2026-11-09).

1. Exact reproduction: swc.py and ald_sim.py run with identical arguments and seeds; rho, surv, occ compared bit for bit.
   Cases: decay raw (dying + surviving g), spread from module seeds, in-norm weight-shuffled null, driven h > 0, and the
   GENERAL code path of ald_sim (per-neuron drive vector of zeros + depression with U = 0 + regions), which must also be
   identical. swc.py is also run twice to check that it is itself deterministic on this GPU.
2. Performance: full CNS, B = 64, T = 3000 (15,000 steps), g = 0.35 raw r = 1 (sustained, no early stop):
   swc.py vs ald_sim.py (all options off) vs ald_sim.py with every new feature on. Loop time is taken from each program's
   own log timestamps (graph built -> g line), so graph loading is excluded for both.
Usage: python ald_validate.py [--skip_perf] -> results/aldef/validation/validate.json (+ the npz/log files)
"""
import json, os, re, subprocess, sys, time
import numpy as np

ROOT = "/MALECNS_ROOT"
PY = "/MALECNS_PYBIN/python"
OUT = f"{ROOT}/results/aldef/validation"
os.makedirs(OUT, exist_ok=True)
res = {"cases": {}, "perf": {}}


def run(prog, args, tag):
    out = f"{OUT}/{tag}.npz"
    cmd = [PY, f"{ROOT}/code/{prog}"] + args + ["--out", out]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    open(f"{OUT}/{tag}.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"{tag} failed:\n{p.stderr[-3000:]}")
    return out, p.stdout, time.time() - t0


def loop_seconds(log):
    """Seconds between the 'graph on GPU' line and the last 'g=' line (both programs print [elapsed s])."""
    ts = [(float(m.group(1)), line) for line in log.splitlines() if (m := re.match(r"\[(\d+\.\d+)s\]", line))]
    t_graph = [t for t, l in ts if "graph on GPU" in l][0]
    t_g = [t for t, l in ts if " g=" in l][-1]
    return t_g - t_graph


def compare(fa, fb):
    A, B = np.load(fa), np.load(fb)
    d = {}
    for k in ("t", "g", "rho", "surv", "occ"):
        x, y = A[k], B[k]
        d[k] = {"shape_equal": x.shape == y.shape,
                "bit_identical": bool(x.shape == y.shape and np.array_equal(x, y)),
                "max_abs_diff": float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.shape == y.shape else None}
    d["all_identical"] = all(v["bit_identical"] for v in d.values() if isinstance(v, dict))
    d["rho_T_mean_swc"] = A["rho"][:, -1].mean(1).tolist(); d["rho_T_mean_ald"] = B["rho"][:, -1].mean(1).tolist()
    return d


cases = {
    "decay_raw_r1": "--exp decay --norm raw --r 1 --g 0.25 0.35 --T 300 --B 16 --seed 3",
    "spread_raw_module": f"--exp spread --norm raw --r 1 --g 0.45 --T 200 --B 64 --seed 5 --seed_set {ROOT}/results/seeds_module_real.npz",
    "decay_in_wnull": "--exp decay --norm in --r 0.5 --g 2.0 --T 100 --B 8 --null --null_kind weights --seed 1",
    "decay_raw_h1e-3": "--exp decay --norm raw --r 1 --h 1e-3 --g 0.2 --T 100 --B 8 --seed 2",
}
for name, args in cases.items():
    fs, _, _ = run("swc.py", args.split(), f"swc_{name}")
    fa, _, _ = run("ald_sim.py", args.split(), f"ald_{name}")
    res["cases"][name] = {"args": args, **compare(fs, fa)}
    print(name, "identical" if res["cases"][name]["all_identical"] else "DIFFERENT", flush=True)
# swc.py determinism + the general (non-plain) code path of ald_sim with trivial settings
a0 = cases["decay_raw_r1"]
fs2, _, _ = run("swc.py", a0.split(), "swc_decay_raw_r1_repeat")
res["cases"]["swc_self_repeat"] = {"args": a0, **compare(f"{OUT}/swc_decay_raw_r1.npz", fs2)}
gen = a0.split() + ["--drive", "all=0", "--std", "0", "inf", "--region", "ORN=olfactory"]
fg, _, _ = run("ald_sim.py", gen, "ald_decay_raw_r1_generalpath")
res["cases"]["general_path_trivial"] = {"args": " ".join(gen), **compare(f"{OUT}/swc_decay_raw_r1.npz", fg)}
print("general path", "identical" if res["cases"]["general_path_trivial"]["all_identical"] else "DIFFERENT", flush=True)

# statistical comparison (only meaningful if a case is not bit-identical): KS test of rho(T) across replicas
from scipy.stats import ks_2samp
for name, d in res["cases"].items():
    if not d["all_identical"]:
        tag = {"swc_self_repeat": "swc_decay_raw_r1_repeat", "general_path_trivial": "ald_decay_raw_r1_generalpath"}.get(name, f"ald_{name}")
        A = np.load(f"{OUT}/swc_{name if name in cases else 'decay_raw_r1'}.npz"); Bz = np.load(f"{OUT}/{tag}.npz")
        d["ks_rhoT_p"] = [float(ks_2samp(A["rho"][i, -1], Bz["rho"][i, -1]).pvalue) for i in range(len(A["g"]))]

if "--skip_perf" not in sys.argv:
    perf = "--exp decay --norm raw --r 1 --g 0.35 --T 3000 --B 64 --seed 11".split()
    full = perf + ["--drive", "olfactory=0.02", "--clamp_off", "type:ORN_DA1", "--std", "0.3", "20",
                   "--std_group", "olfactory=0.5,10", "--adapt", "0.02", "50", "--refractory", "1",
                   "--region", "ORN=olfactory", "--sign_override", "lLN1_bc=1"]
    for tag, prog, args in [("perf_swc", "swc.py", perf), ("perf_ald_off", "ald_sim.py", perf),
                            ("perf_ald_full", "ald_sim.py", full)]:
        f, log, wall = run(prog, args, tag)
        ls = loop_seconds(log)
        res["perf"][tag] = {"args": " ".join(args), "loop_s": ls, "ms_per_step": 1e3 * ls / 15000, "wall_s": wall,
                            "rho_T_mean": float(np.load(f)["rho"][0, -1].mean())}
        print(tag, res["perf"][tag], flush=True)
    b = res["perf"]["perf_swc"]["loop_s"]
    res["perf"]["ratio_off_vs_swc"] = res["perf"]["perf_ald_off"]["loop_s"] / b
    res["perf"]["ratio_full_vs_swc"] = res["perf"]["perf_ald_full"]["loop_s"] / b
    res["perf"]["identical_off_vs_swc"] = compare(f"{OUT}/perf_swc.npz", f"{OUT}/perf_ald_off.npz")["all_identical"]

try:
    import torch
    res["gpu"] = torch.cuda.get_device_name(0)
except Exception:
    res["gpu"] = None
res["host"] = os.uname().nodename
json.dump(res, open(f"{OUT}/validate{'_skipperf' if '--skip_perf' in sys.argv else ''}.json", "w"), indent=1)
print(json.dumps({k: v.get("all_identical") for k, v in res["cases"].items()}), json.dumps(res["perf"]), flush=True)
