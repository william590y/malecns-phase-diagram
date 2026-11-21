"""Lane m2/verify: argument lines for the new-seed GPU reruns (plan results/m2/verify/plan.md section B).
Protocol strings are copied from the lanes' own argument files (decay s1, driven A1/B4, modules ISO) with only g, B,
seed and out changed (counts recording dropped: test U3 shows it does not change the trajectories).
Writes results/m2/verify/sim/args_V.txt and index_V.json.
"""
import json, os, re

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/m2/verify/sim"
os.makedirs(OUT, exist_ok=True)


def first_line(path, pat):
    for ln in open(path):
        if pat in ln:
            return ln.strip()
    raise KeyError(pat)


def strip(line):
    """remove g list, B, seed, out and the counts flags; return the protocol part"""
    line = re.sub(r" --g( [0-9][0-9.e+-]*)+", "", line)
    for k in ("--B", "--seed", "--out", "--counts_win"):
        line = re.sub(rf" {k} \S+", "", line)
    line = line.replace(" --rec_counts", "").replace(" --counts_reg", "")
    return line


dec = strip(first_line(f"{ROOT}/results/m2/decay/sim/args_s1.txt", "s1_M2_real_r0_p0.npz"))
dec_null = strip(first_line(f"{ROOT}/results/m2/decay/sim/args_s1.txt", "s1_M2_cnull_r0_p0.npz"))
drv = strip(first_line(f"{ROOT}/results/m2/driven/args_A.txt", "A1/M2_real_o0.04_r0_c0.npz"))
palp = strip(first_line(f"{ROOT}/results/m2/driven/args_B.txt", "B4/M2_real_o0.04_r0_palp_c0.npz"))
iso_al = strip(first_line(f"{ROOT}/results/m2/modules/sim/args_ISO.txt", "graphs/AL.npz --std_group"))
iso_cb = strip(first_line(f"{ROOT}/results/m2/modules/sim/args_ISO.txt", "graphs/CB.npz --std_group"))
for s in (dec, dec_null, drv, palp, iso_al, iso_cb):
    assert " --r 0" in s, s


def setr(s, r):
    return s.replace(" --r 0", f" --r {r}")


TASKS = [  # id, base, r, g list, B
    ("V1_dec_r0_S", setr(dec, 0), [0.93, 0.96, 0.99, 1.02, 1.06, 1.10], 128),
    ("V2_dec_r0_on", setr(dec, 0), [1.12, 1.15, 1.18, 1.21, 1.25, 1.29], 128),
    ("V3_dec_r1_S", setr(dec, 1), [1.13, 1.17, 1.21, 1.25, 1.30, 1.36], 128),
    ("V4_dec_r1_on", setr(dec, 1), [1.47, 1.52, 1.57, 1.62, 1.67, 1.73], 128),
    ("V5_dec_r6_on", setr(dec, 6), [1.85, 1.93, 2.02, 2.10, 2.19], 128),
    ("V6_dec_r0_D", setr(dec, 0), [5.6, 5.95, 6.3, 6.65, 7.0], 64),
    ("V7_cnull0_r0_on", setr(dec_null, 0), [1.50, 1.55, 1.61, 1.67, 1.73], 64),
    ("V8_cnull2_r0_on", setr(dec_null, 0).replace("--null_seed 0", "--null_seed 2"),
     [1.40, 1.48, 1.56, 1.64, 1.73, 1.85], 64),
    ("V9_drv_r0_lo_top", setr(drv, 0), [0.20, 0.208, 0.216, 0.224, 0.233, 0.95, 0.968, 0.99], 64),
    ("V10_drv_r0_ss", setr(drv, 0), [1.15, 1.18, 1.21, 1.24, 1.28], 64),
    ("V11_drv_r1_lo_top", setr(drv, 1), [0.30, 0.317, 0.335, 1.18, 1.23, 1.28], 64),
    ("V12_palp_r0_a", setr(palp, 0), [0.19, 0.20, 0.208, 0.216, 0.224, 0.233, 0.243], 128),
    ("V13_palp_r0_b", setr(palp, 0), [0.19, 0.20, 0.208, 0.216, 0.224, 0.233, 0.243], 128),
    ("V14_iso_CB_r0", setr(iso_cb, 0), [1.20, 1.27, 1.34, 1.42, 1.50, 1.58], 64),
    ("V15_iso_AL_r0", setr(iso_al, 0), [1.85, 1.97, 2.09, 2.19, 2.32, 2.46], 64),
]
assert "--null_seed 2" in TASKS[7][1]
lines, index = [], []
for k, (tid, base, g, B) in enumerate(TASKS):
    seed = 7_700_001 + 1009 * k
    out = f"{OUT}/{tid}.npz"
    lines.append(f"{base} --g {' '.join(map(str, g))} --B {B} --seed {seed} --out {out}")
    index.append({"task": k, "id": tid, "g": g, "B": B, "seed": seed, "out": out})
open(f"{OUT}/args_V.txt", "w").write("\n".join(lines) + "\n")
json.dump(index, open(f"{OUT}/index_V.json", "w"), indent=1)
print(len(lines), "tasks")
for ln in lines:
    print(ln[:160], "...", ln[-120:])
