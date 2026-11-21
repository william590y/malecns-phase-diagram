"""E1 (ORN drive / antenna removal): neuron groups, drive levels and the Slurm argument files (2026-11-09).

Writes
  results/aldef/drive/groups/*.npz      neuron index sets used as regions (idx), + groups.json (sizes, definitions)
  results/aldef/drive/levels.json       ORN drive levels: isolated-unit active fraction rho_ORN -> h, and the two readings
                                         of each level (ORN rate at tau_u = 10 ms; tau_u if ORNs fire at 3.44 Hz)
  results/aldef/drive/args_main.txt     one ald_sim.py argument line per array task
Protocol of every task: quiet start; phase 1 (0 < t <= 1500) all 2,639 ORNs driven at h_ORN, every other neuron h = 0;
phase 2 (1500 < t <= 4500) "antenna removal":
  proto all : every ORN clamped to 0 (full deafferentation; the network becomes absorbing),
  proto ant : only the 2,454 antennal ORNs clamped to 0; the 185 maxillary-palp ORNs keep their drive (Olsen et al. 2007
              removed antennae with palps intact).
Grid: r in {0, 1}; 5 ORN levels; 24 g values in 4 chunks of 6 -> 2 x 5 x 2 x 4 = 80 tasks, B = 64 replicas each.
"""
import json, os, re
import numpy as np

ROOT = "/MALECNS_ROOT"
C = "/MALECNS_CACHE"
OUT = f"{ROOT}/results/aldef/drive"
GRP = f"{OUT}/groups"
os.makedirs(GRP, exist_ok=True)
DT, BETA = 0.2, 1.0
P_OFF = 1 - np.exp(-DT)
PALP = "VA4|VC1|VC2|VA7l|VM7d|VM7v"           # maxillary-palp glomeruli (same set as the simulator report's selector)

ty = np.load(f"{C}/types.npy", allow_pickle=True).astype(str)
cl = np.load(f"{C}/classes.npy", allow_pickle=True).astype(str)
sg = np.load(f"{C}/sign.npy")
N = len(sg)
mod = np.zeros(N, bool); mod[np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)["idx"]] = True
AL_CLASSES = ("olfactory", "hygrosensory", "thermosensory", "ALPN", "ALLN", "ALIN", "ALON")
al = np.isin(cl, AL_CLASSES)
orn = cl == "olfactory"
orn_palp = orn & np.array([re.fullmatch(f"ORN_({PALP})", t) is not None for t in ty])
orn_ant = orn & ~orn_palp
rx = re.compile(r"(DA|DC|DL|DM|DP|VA|VC|VL|VM|D|V)[0-9]*[a-z]*_[a-z0-9]*PN")   # uniglomerular olfactory PN types (no VP)
upn = (cl == "ALPN") & (sg > 0) & np.array([rx.fullmatch(t) is not None for t in ty])
upn_palp = upn & np.array([re.match(f"({PALP})_", t) is not None for t in ty])
upn_ant = upn & ~upn_palp
groups = {
    "ORN": (orn, "class olfactory (all 2,639 ORNs, drive target)"),
    "ORN_ant": (orn_ant, "antennal ORNs (ORN types not of the 6 palp glomeruli)"),
    "ORN_palp": (orn_palp, f"maxillary-palp ORNs (ORN_({PALP}))"),
    "PN_all": (cl == "ALPN", "class ALPN (incl. 217 GABAergic and multiglomerular PNs)"),
    "uPN": (upn, "excitatory (ACh) uniglomerular olfactory PNs (type <glomerulus>_<lineage>PN, no VP glomeruli)"),
    "uPN_ant": (upn_ant, "uPN of antennal glomeruli (the PNs recorded by Olsen 2007 / Kazama & Wilson)"),
    "uPN_palp": (upn_palp, "uPN of the 6 palp glomeruli"),
    "uPN_mod": (upn & mod, "uPN inside the 526-neuron AL module"),
    "uPN_nonmod": (upn & ~mod, "uPN outside the AL module"),
    "LN_exc": ((cl == "ALLN") & (sg > 0), "AL local neurons with consensus ACh"),
    "LN_inh": ((cl == "ALLN") & (sg < 0), "AL local neurons with consensus GABA/Glu"),
    "lLN1_bc": (ty == "lLN1_bc", "the 30 cholinergic lLN1_bc"),
    "ALsys": ((al & ~orn) | mod, "AL without ORNs, union the AL module (persistence region)"),
}
info = {}
for k, (m, d) in groups.items():
    np.savez(f"{GRP}/{k}.npz", idx=np.where(m)[0])
    info[k] = {"n": int(m.sum()), "def": d, "in_module": int((m & mod).sum())}
json.dump(info, open(f"{GRP}/groups.json", "w"), indent=1)
print(json.dumps({k: v["n"] for k, v in info.items()}))

# ORN drive levels: isolated-unit stationary activity rho = p_on / (p_on + p_off)
levels = {}
for rho in (0.01, 0.02, 0.04, 0.07, 0.10):
    p_on = rho * P_OFF / (1 - rho)
    h = float(np.arctanh(-np.log(1 - p_on) / (BETA * DT)))
    pa = 1 - np.exp(-BETA * np.tanh(h) * DT)
    levels[f"{rho:.2f}"] = {"rho_iso": rho, "h": h, "check_rho": pa / (pa + P_OFF),
                            "ORN_Hz_at_tau10ms": 0.9063462346100909 * rho / 0.010,
                            "tau_u_ms_if_ORN_3.44Hz": 1e3 * 0.9063462346100909 * rho / 3.44}
json.dump(levels, open(f"{OUT}/levels.json", "w"), indent=1)

G = [0.0, 0.04, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38,
     0.42, 0.46, 0.50, 0.60, 0.80, 1.00]
chunks = [G[i::4] for i in range(4)]            # interleaved so every chunk has a similar mix of fast and slow g values
regs = " ".join(f"--region {k}={GRP}/{k}.npz" for k in groups)
wins = " ".join(f"--win {w}" for w in ("500:1500", "1500:1505", "1505:1520", "1520:1550", "1550:1600", "1600:1800",
                                       "1800:2500", "2500:4500"))
common = (f"--exp quiet --norm raw --T 4500 --B 64 --switch_at 1500 --rec_lin 10 {wins} --reg_last "
          f"--occ_win 500:1500 --engine push {regs}")
lines, task = [], 0
for proto in ("all", "ant"):
    for r in (0, 1):
        for lk, lv in levels.items():
            for ci, gs in enumerate(chunks):
                clamp = "olfactory" if proto == "all" else f"{GRP}/ORN_ant.npz"
                tag = f"{proto}_r{r}_o{lk}_c{ci}"
                lines.append(f"{common} --r {r} --g {' '.join(str(g) for g in gs)} --drive olfactory={lv['h']:.6f} "
                             f"--sw_clamp_off {clamp} --seed {1000 + task} --out {OUT}/runs/{tag}.npz")
                task += 1
os.makedirs(f"{OUT}/runs", exist_ok=True)
open(f"{OUT}/args_main.txt", "w").write("\n".join(lines) + "\n")
print(len(lines), "tasks;", lines[0][:300])
