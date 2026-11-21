"""Adversarial verification of the AL-defect study: independent simulator checks (2026-11-09, verify lane).

Run one task per Slurm array index (code/ald_verify_sim.sbatch). Everything is written to results/aldef/verify/sim/.
NEW seeds (never used by the E1-E4 lanes), the CURRENT code/ald_sim.py (md5 recorded) and the untouched code/swc.py.

task 0  bit-identity: swc.py vs ald_sim.py, all options off, 3 cases with new seeds (decay r=1 dying+surviving g,
        decay r=0, h>0). Writes sim/bitid.json.
task 1  push-vs-spmm statistics (spmm engine): base r=1, g 0.32 0.34 0.36, B 128, T 3000.
task 2  same with --engine push (production setting of E1-E4).
task 3  replication of E3 sp_NW16_r1 (per-spike depression U .25 TAU 157) at g 1.29 1.368 1.451 1.539, B 64.
task 4  replication of E2 lLN1_bc sign 0 at r=1, g 0.94 0.98 1.02 1.06, B 64.
task 5  replication of E3 iso_sp_NW16_ALonly (AL alone + depression) at g 2.75 3.0 3.25, B 64.
task 6  replication of E3 iso_sp_NW16_noAL (CNS without AL + depression) at g 1.35 1.45 1.55, B 64.
"""
import hashlib, json, os, subprocess, sys
import numpy as np

ROOT = "/MALECNS_ROOT"
PY = "/MALECNS_PYBIN/python"
OUT = f"{ROOT}/results/aldef/verify/sim"
os.makedirs(OUT, exist_ok=True)
COMMON = "--exp decay --norm raw --T 3000"
REG = "--reg_last --rec_lin 100 --win 1500:3000"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def run(prog, args, tag):
    out = f"{OUT}/{tag}.npz"
    cmd = f"{PY} {ROOT}/code/{prog} {args} --out {out}"
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    open(f"{OUT}/{tag}.log", "w").write(cmd + "\n" + p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"{tag} failed:\n{p.stderr[-3000:]}")
    return out


task = int(sys.argv[1])
meta = {"task": task, "md5_ald_sim": md5(f"{ROOT}/code/ald_sim.py"), "md5_swc": md5(f"{ROOT}/code/swc.py")}
if task == 0:
    cases = {"decay_r1": "--exp decay --norm raw --r 1 --g 0.25 0.36 --T 300 --B 16 --seed 7101",
             "decay_r0": "--exp decay --norm raw --r 0 --g 0.28 --T 400 --B 16 --seed 7102",
             "decay_r1_h": "--exp decay --norm raw --r 1 --h 1e-3 --g 0.2 --T 150 --B 8 --seed 7103"}
    res = {}
    for name, a in cases.items():
        fa = run("swc.py", a, f"bitid_{name}_swc"); fb = run("ald_sim.py", a, f"bitid_{name}_ald")
        A, B = np.load(fa), np.load(fb)
        res[name] = {"args": a}
        for k in ("t", "g", "rho", "surv", "occ"):
            res[name][k] = bool(A[k].shape == B[k].shape and np.array_equal(A[k], B[k]))
        res[name]["all_identical"] = all(res[name][k] for k in ("t", "g", "rho", "surv", "occ"))
        res[name]["surv_T_mean"] = B["surv"][:, -1].mean(1).tolist()
    json.dump({**meta, "cases": res, "all_pass": all(v["all_identical"] for v in res.values())},
              open(f"{OUT}/bitid.json", "w"), indent=1)
elif task in (1, 2):
    eng = "spmm" if task == 1 else "push --push_max 0.012"
    run("ald_sim.py", f"{COMMON} --r 1 --g 0.32 0.34 0.36 --B 128 --seed {7200 + task} --engine {eng}", f"eng_{'spmm' if task == 1 else 'push'}")
elif task == 3:
    run("ald_sim.py", f"{COMMON} --B 64 --engine push --push_max 0.012 {REG} --r 1 --g 1.29 1.368 1.451 1.539 --seed 7301 "
        "--std 0.25 157 --std_mode spike", "rep_sp_NW16_r1")
elif task == 4:
    run("ald_sim.py", f'{COMMON} --B 64 --engine push --push_max 0.012 --r 1 --g 0.94 0.98 1.02 1.06 --seed 7401 '
        '--sign_override "lLN1_bc=0"', "rep_lLN1bc_zero_r1")
elif task == 5:
    run("ald_sim.py", f'{COMMON} --B 64 --engine push --push_max 0.012 {REG} --r 1 --g 2.75 3.0 3.25 --seed 7501 '
        '--std 0.25 157 --std_mode spike --clamp_off "cns_rest"', "rep_iso_NW16_ALonly_r1")
elif task == 6:
    run("ald_sim.py", f'{COMMON} --B 64 --engine push --push_max 0.012 {REG} --r 1 --g 1.35 1.45 1.55 --seed 7601 '
        '--std 0.25 157 --std_mode spike --clamp_off "al;module"', "rep_iso_NW16_noAL_r1")
json.dump(meta, open(f"{OUT}/task{task}_meta.json", "w"))
print("done", meta)
