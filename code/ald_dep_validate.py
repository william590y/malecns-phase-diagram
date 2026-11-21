"""E3 full-CNS validation of the patched ald_sim.py (adds --std_mode spike; 2026-11-09). One GPU, inside a Slurm job.
Usage: python ald_dep_validate.py <new ald_sim file> <old ald_sim file> -> results/aldef/depression/validation/validate_e3.json

Bit-identity (rho, surv, occ, and reg_rho / res_x where both have them) for:
  V1 swc.py vs NEW ald_sim, all options off (decay raw r=1, g 0.25 0.35)       -- the original contract
  V2 OLD vs NEW ald_sim, rate-mode STD + adaptation + refractory + push engine  -- the default path of the STD code is unchanged
  V3 NEW spike mode with U = 0 vs NEW options off (general path)                  -- spike code draws no random numbers
  V4 NEW spike mode, U = 0.5, TAU = 1e-6 (instant recovery: every epoch undepressed) vs NEW options off
  V5 NEW spike mode, U = 0.25, TAU = 157 runs (sanity: finite, resources in [0, 1], efficacy <= 1)
"""
import json, os, subprocess, sys
import numpy as np

ROOT = "/MALECNS_ROOT"
PY = "/MALECNS_PYBIN/python"
OUT = f"{ROOT}/results/aldef/depression/validation"
NEW, OLD = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
res = {"new": NEW, "old": OLD, "cases": {}}


def run(prog, args, tag):
    out = f"{OUT}/{tag}.npz"
    cmd = [PY, prog] + args.split() + ["--out", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    open(f"{OUT}/{tag}.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"{tag} failed:\n{p.stderr[-3000:]}")
    return out


def compare(fa, fb, keys=("t", "g", "rho", "surv", "occ", "reg_rho", "res_x", "res_x_act", "res_theta")):
    A, B = np.load(fa), np.load(fb)
    d = {}
    for k in keys:
        if k not in A.files or k not in B.files:
            continue
        x, y = A[k], B[k]
        same = x.shape == y.shape and np.array_equal(x, y, equal_nan=True)
        d[k] = {"bit_identical": bool(same),
                "max_abs_diff": float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.shape == y.shape else None}
    d["all_identical"] = all(v["bit_identical"] for v in d.values())
    return d


base = "--exp decay --norm raw --r 1 --g 0.25 0.35 --T 300 --B 16 --seed 3"
a = run(f"{ROOT}/code/swc.py", base, "v1_swc")
b = run(NEW, base, "v1_new")
res["cases"]["V1_swc_vs_new_off"] = compare(a, b, ("t", "g", "rho", "surv", "occ"))

feat = ("--exp decay --norm raw --r 1 --g 0.3 0.6 --T 300 --B 16 --seed 4 --engine push --std 0.25 89 "
        "--std_group sign:-1=0,1 --adapt_group ALLN=0.02,50 --refractory 3")
a = run(OLD, feat, "v2_old_rate")
b = run(NEW, feat, "v2_new_rate")
res["cases"]["V2_old_vs_new_rate_mode_features"] = compare(a, b)

off = "--exp decay --norm raw --r 1 --g 0.3 0.45 --T 300 --B 16 --seed 6"
a = run(NEW, off + " --region X=module", "v3_new_off")
b = run(NEW, off + " --region X=module --std 0 89 --std_mode spike", "v3_new_spike_U0")
res["cases"]["V3_spike_U0_vs_off"] = compare(a, b, ("t", "g", "rho", "surv", "occ", "reg_rho"))
b = run(NEW, off + " --region X=module --std 0.5 1e-6 --std_mode spike", "v4_new_spike_instant_recovery")
res["cases"]["V4_spike_instant_recovery_vs_off"] = compare(a, b, ("t", "g", "rho", "surv", "occ", "reg_rho"))

b = run(NEW, "--exp decay --norm raw --r 1 --g 0.3 1.0 --T 300 --B 16 --seed 8 --engine push --std 0.25 157 --std_mode spike",
        "v5_new_spike_lit")
d = np.load(b)
x, e = d["res_x"], d["res_x_act"]
res["cases"]["V5_spike_sanity"] = {"finite_rho": bool(np.isfinite(d["rho"]).all()),
                                   "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
                                   "e_range": [float(np.nanmin(e)), float(np.nanmax(e))],
                                   "ok": bool(np.nanmin(x) >= 0 and np.nanmax(x) <= 1 and np.nanmax(e) <= 1 + 1e-6),
                                   "surv_T": d["surv"][:, -1].mean(1).tolist(),
                                   "AL_module_T": d["reg_rho"][:, -1, 0, :].mean(1).tolist()}
res["all_pass"] = (all(c["all_identical"] for k, c in res["cases"].items() if k != "V5_spike_sanity")
                   and res["cases"]["V5_spike_sanity"]["ok"])
json.dump(res, open(f"{OUT}/validate_e3.json", "w"), indent=1)
print(json.dumps(res, indent=1))
