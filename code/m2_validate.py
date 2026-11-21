"""Real-cache validation of the Model-2 version of ald_sim.py (one GPU, inside a Slurm job; 2026-11-09).
Usage: python m2_validate.py <new ald_sim file> <old ald_sim file> [<previous M2 ald_sim file>]
       -> results/m2/sim/validation/validate_m2.json

 V1 swc.py vs NEW ald_sim, identical arguments and seeds, rho/surv/occ bit for bit: the 4 cases of ald_validate.py
    (decay raw dying + surviving g, spread from module seeds, in-norm weight-shuffled null, h > 0) + the general code
    path with trivial settings.
 V2 OLD ald_sim (md5 92c2e0d3, the version every aldef lane used) vs NEW, every npz key bit for bit (spmm engine), on
    option sets of E1 (drive, switch, windows, reg_last, rec_lin, spike depression groups), E3 (decay + spike
    depression + regions + window), E2 (sign override + weight transform), sign ensembles, adaptation + refractory +
    clamps.
 V4 (with a third file) the previous M2 version vs NEW, all keys bit for bit, on lines using --edge_sign, --null_seed,
    --rec_counts/--counts_reg, --pulse, --ramp/--ramp_loop with depression and drive.
 V3 NEW options that must not change anything when trivial (bit for bit): identity --edge_sign (V = existing sign);
    --rec_counts (and counts at record steps = rho * N); --null --null_seed 0 --seed 0 vs --null --seed 0;
    single-dwell --ramp vs plain; --pulse on a run that never goes silent.
"""
import json, os, subprocess, sys, time
import numpy as np

ROOT = "/MALECNS_ROOT"
PY = "/MALECNS_PYBIN/python"
NEW, OLD = sys.argv[1], sys.argv[2]
OUT = f"{ROOT}/results/m2/sim/validation"
RUN = f"{OUT}/runs"
os.makedirs(RUN, exist_ok=True)
res = {"new": NEW, "old": OLD, "V1": {}, "V2": {}, "V3": {}}
SKIP = {"sec_per_step", "meta"}


def run(prog, args, tag):
    out = f"{RUN}/{tag}.npz"
    cmd = [PY, prog] + args + ["--out", out]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    open(f"{RUN}/{tag}.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"{tag} failed:\n{p.stderr[-3000:]}")
    return out, time.time() - t0


def compare(fa, fb, keys=None):
    A, B = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
    ks = keys or sorted((set(A.files) | set(B.files)) - SKIP)
    d = {}
    for k in ks:
        if k not in A.files or k not in B.files:
            d[k] = "missing in " + ("old/swc" if k not in A.files else "new")
            continue
        x, y = A[k], B[k]
        d[k] = bool(x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))
    return all(v is True for v in d.values()), d


SWC = f"{ROOT}/code/swc.py"
cases = {
    "decay_raw_r1": "--exp decay --norm raw --r 1 --g 0.25 0.35 --T 300 --B 16 --seed 3",
    "spread_raw_module": f"--exp spread --norm raw --r 1 --g 0.45 --T 200 --B 64 --seed 5 --seed_set {ROOT}/results/seeds_module_real.npz",
    "decay_in_wnull": "--exp decay --norm in --r 0.5 --g 2.0 --T 100 --B 8 --null --null_kind weights --seed 1",
    "decay_raw_h1e-3": "--exp decay --norm raw --r 1 --h 1e-3 --g 0.2 --T 100 --B 8 --seed 2",
}
for name, a in cases.items():
    fs, _ = run(SWC, a.split(), f"V1_swc_{name}")
    fn, _ = run(NEW, a.split(), f"V1_new_{name}")
    ok, d = compare(fs, fn, ["t", "g", "rho", "surv", "occ"])
    res["V1"][name] = {"args": a, "identical": ok, "keys": d}
    print("V1", name, ok, flush=True)
gen = cases["decay_raw_r1"].split() + ["--drive", "all=0", "--std", "0", "inf", "--region", "ORN=olfactory"]
fn, _ = run(NEW, gen, "V1_new_generalpath")
ok, d = compare(f"{RUN}/V1_swc_decay_raw_r1.npz", fn, ["t", "g", "rho", "surv", "occ"])
res["V1"]["general_path_trivial"] = {"args": " ".join(gen), "identical": ok, "keys": d}
print("V1 general", ok, flush=True)

G = f"{ROOT}/results/aldef/drive/groups"
v2 = {
    "E1_drive_switch_spike_group": (f"--exp quiet --norm raw --T 60 --B 16 --switch_at 30 --win 10:30 --win 30:60 --reg_last "
                                    f"--rec_lin 5 --drive olfactory=0.037926 --sw_clamp_off olfactory --r 1 --g 0.3 1.2 "
                                    f"--region ORN=olfactory --region uPN_ant={G}/uPN_ant.npz --std_group sign:1=0.25,157 "
                                    f"--std_mode spike --occ_win 10:30 --seed 4"),
    "E3_decay_spike_all": ("--exp decay --norm raw --T 60 --B 16 --r 1 --g 1.2 2.0 --std 0.25 157 --std_mode spike "
                           "--region PN=class:ALPN --win 30:60 --reg_last --seed 5"),
    "E3_rate_adapt_ref_clamp": ("--exp quiet --norm raw --T 40 --B 8 --r 1 --g 0.5 --std 0.29 157 --adapt 0.02 50 "
                                "--refractory 2 --clamp_off type:ORN_DA1 --drive olfactory=0.05 --clamp_on type:lLN1_bc --seed 6"),
    "E2_sign_wtransform": "--exp decay --norm raw --T 40 --B 8 --r 1 --g 0.5 --sign_override lLN1_bc=0 --wtransform sqrt --seed 7",
    "ensemble_type": "--exp decay --norm raw --T 20 --B 8 --r 1 --g 0.4 --sign_ensemble 2 --ens_unit type --seed 8",
}
for name, a in v2.items():
    fo, to = run(OLD, a.split(), f"V2_old_{name}")
    fn, tn = run(NEW, a.split(), f"V2_new_{name}")
    ok, d = compare(fo, fn)
    res["V2"][name] = {"args": a, "identical": ok, "keys": d, "wall_old": to, "wall_new": tn}
    print("V2", name, ok, flush=True)

base = "--exp decay --norm raw --r 1 --g 0.35 --T 100 --B 16 --seed 9".split()
f0, _ = run(NEW, base, "V3_base")
f1, _ = run(NEW, base + ["--edge_sign", "lLN1_bc->ALPN=1"], "V3_edge_identity")
ok, d = compare(f0, f1, ["t", "g", "rho", "surv", "occ", "reg_rho"])
res["V3"]["edge_sign_identity"] = {"identical": ok, "keys": d}
f2, _ = run(NEW, base + ["--rec_counts"], "V3_counts")
ok, d = compare(f0, f2, ["t", "g", "rho", "surv", "occ", "reg_rho"])
Z = np.load(f2)
steps = np.round(Z["t"] / 0.2).astype(int)
n_from_rho = np.round(Z["rho"][0].astype(np.float64) * 167216).astype(np.int64)
ok_c = bool(np.array_equal(n_from_rho, Z["cnt_act"][0][steps - 1].astype(np.int64)))
res["V3"]["rec_counts"] = {"identical": ok, "keys": d, "counts_equal_rhoN_at_records": ok_c}
nb = "--exp decay --norm raw --r 1 --g 1.1 --T 50 --B 8 --null --seed 0".split()
f3, _ = run(NEW, nb, "V3_null_s0")
f4, _ = run(NEW, nb + ["--null_seed", "0"], "V3_null_ns0")
ok, d = compare(f3, f4, ["t", "g", "rho", "surv", "occ", "reg_rho"])
j3, j4 = json.load(open(f3[:-4] + ".json")), json.load(open(f4[:-4] + ".json"))
res["V3"]["null_seed0_equals_seed0"] = {"identical": ok, "keys": d, "md5": [j3["null_edges_md5"], j4["null_edges_md5"]]}
f5, _ = run(NEW, base + ["--ramp", "100"], "V3_ramp_single")
ok, d = compare(f0, f5, ["t", "g", "rho", "surv", "occ", "reg_rho"])
res["V3"]["single_dwell_ramp"] = {"identical": ok, "keys": d}
pb = "--exp decay --norm raw --r 1 --g 0.6 --T 100 --B 16 --seed 10".split()
f6, _ = run(NEW, pb, "V3_nopulse")
f7, _ = run(NEW, pb + ["--pulse", "olfactory"], "V3_pulse")
ok, d = compare(f6, f7, ["t", "g", "rho", "surv", "occ", "reg_rho"])
res["V3"]["pulse_never_silent"] = {"identical": ok, "keys": d, "min_surv": float(np.load(f6)["surv"].min())}
for k, v in res["V3"].items():
    print("V3", k, v["identical"], flush=True)

# V4 (optional third argument): the previous M2 version vs NEW on lines that use the M2 options, every key bit for bit
if len(sys.argv) > 3:
    PREV = sys.argv[3]
    res["prev_m2"] = PREV; res["V4"] = {}
    v4 = {
        "decay_M2s_counts_nullseed": ("--exp decay --norm raw --T 40 --B 8 --r 1 --g 1.6 --std_group sign:1=0.25,157 --std_mode spike "
                                      "--edge_sign lLN1_bc->ALPN=0 --edge_sign lLN1_bc->ALLN=1 --rec_counts --counts_reg --seed 12"),
        "null_seed_counts": "--exp decay --norm raw --T 20 --B 8 --r 1 --g 1.2 --null --null_seed 3 --rec_counts --seed 13",
        "pulse_counts_spike": ("--exp quiet --norm raw --T 60 --B 8 --r 1 --g 0.5 --pulse olfactory --pulse_gap 2 --rec_counts "
                               "--std_group sign:1=0.25,157 --std_mode spike --seed 14"),
        "ramp_loop_driven": ("--exp quiet --norm raw --r 1 --g 0.4 0.8 1.6 --ramp 10 --ramp_loop --drive olfactory=0.037926 "
                             "--std_group sign:1=0.25,157 --std_mode spike --B 8 --seed 15"),
    }
    for name, a_ in v4.items():
        fo, _ = run(PREV, a_.split(), f"V4_prev_{name}")
        fn, _ = run(NEW, a_.split(), f"V4_new_{name}")
        ok, d = compare(fo, fn)
        res["V4"][name] = {"args": a_, "identical": ok, "keys": d}
        print("V4", name, ok, flush=True)

ok_all = (all(v["identical"] for v in res["V1"].values()) and all(v["identical"] for v in res["V2"].values())
          and all(v["identical"] for v in res.get("V4", {}).values())
          and all(v["identical"] for v in res["V3"].values()) and res["V3"]["rec_counts"]["counts_equal_rhoN_at_records"]
          and res["V3"]["pulse_never_silent"]["min_surv"] == 1.0)
try:
    import torch
    res["gpu"] = torch.cuda.get_device_name(0)
except Exception:
    res["gpu"] = None
res["host"] = os.uname().nodename
res["all_pass"] = bool(ok_all)
json.dump(res, open(f"{OUT}/validate_m2.json", "w"), indent=1)
print("ALL PASS" if ok_all else "SOME FAILED", flush=True)
