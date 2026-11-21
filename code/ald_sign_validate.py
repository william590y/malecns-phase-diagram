"""E2 (transmitter signs) -- validation of the opt-in --engine push added to ald_sim.py (run on ONE GPU in a Slurm job).

1. Default engine still bit-identical to swc.py (2 cases, same as ald_validate.py's first two).
2. Push matvec vs torch.sparse.mm on the REAL operator (raw norm, r = 1, baseline signs and lLN1_bc -> -1) for random binary
   and real-valued Y at active fractions 1e-4 .. 0.2: max |diff| relative to max |W Y|; per-call time of both.
3. Same seed, short runs (T = 300, B = 16): spmm vs push trajectories (fraction of recorded (t, replica) rho values equal).
4. Independent-seed long runs (T = 3000, B = 64), spmm vs push: survival at T, censored-exponential lifetime, late mean
   rho, and occupancy profile (cosine similarity + PR); r = 1 baseline at g 0.30 / 0.34 / 0.40, r = 0 at g 2.0
   (dense, fallback path), lLN1_bc -> 0 at r = 1, g 0.9.
5. Speed: ms/step of both engines at a sustained point for B = 16 / 64.
Usage: python ald_sign_validate.py -> results/aldef/signs/validation/validate_push.json (+ npz/log files)
"""
import json, os, re, subprocess, sys, time
import numpy as np

ROOT = "/MALECNS_ROOT"
PY = "/MALECNS_PYBIN/python"
OUT = f"{ROOT}/results/aldef/signs/validation"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{ROOT}/code")
res = {"bitexact": {}, "matvec": [], "same_seed": {}, "stat": {}, "speed": {}}


def run(prog, args, tag):
    out = f"{OUT}/{tag}.npz"
    cmd = [PY, f"{ROOT}/code/{prog}"] + args + ["--out", out]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    open(f"{OUT}/{tag}.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"{tag} failed:\n{p.stderr[-3000:]}")
    return out, p.stdout, time.time() - t0


def ms_per_step(log):
    return [float(x) for x in re.findall(r"\| ([\d.]+) ms/step", log)]


# 1. bit-identity of the default path
cases = {"decay_raw_r1": "--exp decay --norm raw --r 1 --g 0.25 0.35 --T 300 --B 16 --seed 3",
         "spread_raw_module": f"--exp spread --norm raw --r 1 --g 0.45 --T 200 --B 64 --seed 5 "
                              f"--seed_set {ROOT}/results/seeds_module_real.npz"}
for name, args in cases.items():
    fs, _, _ = run("swc.py", args.split(), f"swc_{name}")
    fa, _, _ = run("ald_sim.py", args.split(), f"ald_{name}")
    A, Bz = np.load(fs), np.load(fa)
    res["bitexact"][name] = {k: bool(np.array_equal(A[k], Bz[k])) for k in ("t", "g", "rho", "surv", "occ")}
    print("bitexact", name, res["bitexact"][name], flush=True)

# 2. matvec on the real operator
import torch
import ald_sim as S
C = S.C
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float32); sign = np.load(C + "/sign.npy"); N = len(sign)
L = S.Labels(N, sign=sign)
scale = S.node_scale(post, w, w, N, "raw", "renorm")
for label, ov in [("baseline_r1", []), ("lLN1bc_inh_r1", [(S.select("lLN1_bc", L), -1.0)])]:
    sig = S.sigma_vector(sign, 1.0, ov)
    Wm, nnz = S.build_matrix(pre, post, w, scale, sig, N, "cuda")
    pm = S.PushMV(Wm, N, push_max=1e9)
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    for B in (16, 64):
        for frac in (1e-4, 1e-3, 1e-2, 0.05, 0.2):
            if B == 64 and frac > 0.1:
                continue
            X = (torch.rand(N, B, device="cuda", generator=g) < frac).float()
            Yr = X * torch.rand(N, B, device="cuda", generator=g)
            for kind, Y in (("binary", X), ("real", Yr)):
                a = torch.sparse.mm(Wm, Y); b = pm(Wm, Y, kind == "binary")
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(5):
                    a = torch.sparse.mm(Wm, Y)
                torch.cuda.synchronize(); t1 = time.time()
                for _ in range(5):
                    b = pm(Wm, Y, kind == "binary")
                torch.cuda.synchronize(); t2 = time.time()
                row = {"op": label, "B": B, "frac": frac, "Y": kind,
                       "max_abs_diff_rel": float((a - b).abs().max() / a.abs().max().clamp(min=1e-30)),
                       "ms_spmm": 1e3 * (t1 - t0) / 5, "ms_push": 1e3 * (t2 - t1) / 5}
                res["matvec"].append(row); print(row, flush=True)
    del Wm, pm
    torch.cuda.empty_cache()

# 3. same seed, short
base = "--exp decay --norm raw --r 1 --g 0.30 0.45 --T 300 --B 16 --seed 0".split()
fa, _, _ = run("ald_sim.py", base, "short_spmm")
fb, _, _ = run("ald_sim.py", base + ["--engine", "push"], "short_push")
A, Bz = np.load(fa), np.load(fb)
eq = A["rho"] == Bz["rho"]
res["same_seed"] = {"frac_rho_equal": float(eq.mean()), "frac_rho_equal_first_half": float(eq[:, : eq.shape[1] // 2].mean()),
                    "surv_T_spmm": A["surv"][:, -1].mean(1).tolist(), "surv_T_push": Bz["surv"][:, -1].mean(1).tolist(),
                    "rho_T_spmm": A["rho"][:, -1].mean(1).tolist(), "rho_T_push": Bz["rho"][:, -1].mean(1).tolist()}
print("same_seed", res["same_seed"], flush=True)


# 4. independent-seed long runs
def summ(f):
    d = np.load(f); t = d["t"]; out = []
    for i, gg in enumerate(d["g"]):
        s = d["surv"][i]; dead = s == 0; died = dead.any(0)
        ext = np.where(died, t[np.argmax(dead, 0)], t[-1])
        late = t >= t[-1] / 2
        o = d["occ"][i].astype(np.float64)
        out.append({"g": float(gg), "alive_T": float(1 - died.mean()), "deaths": int(died.sum()),
                    "tau_mle": float(ext.sum() / died.sum()) if died.any() else float("inf"),
                    "rho_late": float(d["rho"][i][late].mean()), "rho_late_rep_sd": float(d["rho"][i][late].mean(0).std()),
                    "occ": o})
    return out


stat_cases = {"r1_base": "--exp decay --norm raw --r 1 --g 0.30 0.34 0.40 --T 3000 --B 64",
              "r0_dense": "--exp decay --norm raw --r 0 --g 2.0 --T 3000 --B 64",
              "r1_lLN1bc0": "--exp decay --norm raw --r 1 --g 0.9 --T 3000 --B 64 --sign_override lLN1_bc=0"}
for name, args in stat_cases.items():
    fa, la, _ = run("ald_sim.py", args.split() + ["--seed", "101"], f"stat_{name}_spmm")
    fb, lb, _ = run("ald_sim.py", args.split() + ["--seed", "202", "--engine", "push"], f"stat_{name}_push")
    sa, sb = summ(fa), summ(fb)
    rows = []
    for x, y in zip(sa, sb):
        ox, oy = x.pop("occ"), y.pop("occ")
        cos = float(ox @ oy / np.sqrt((ox @ ox) * (oy @ oy))) if ox.sum() > 0 and oy.sum() > 0 else None
        pr = lambda o: float(o.sum() ** 2 / (o ** 2).sum()) if o.sum() > 0 else 0.0
        rows.append({"g": x["g"], "spmm": x, "push": y, "occ_cosine": cos, "PR_spmm": pr(ox), "PR_push": pr(oy)})
    side = json.load(open(fb.replace(".npz", ".json")))
    res["stat"][name] = {"args": args, "rows": rows, "ms_spmm": ms_per_step(la), "ms_push": ms_per_step(lb),
                         "push_engine_steps": side.get("engine_steps")}
    print("stat", name, json.dumps(res["stat"][name])[:1500], flush=True)

res["gpu"] = torch.cuda.get_device_name(0); res["host"] = os.uname().nodename
json.dump(res, open(f"{OUT}/validate_push.json", "w"), indent=1)
print("done", flush=True)
