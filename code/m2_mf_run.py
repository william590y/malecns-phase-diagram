"""m2_mf_run.py -- GPU runner of the M2 quenched mean field with resources (code/m2_mf_core.py) on the experiment grids.

Protocols (the deterministic counterparts of MODEL2.md section 4, same windows):
  decay   h = 0. Columns: IC 'decay' (every unit active at t = 0, e = 1, x = 1 - U; the protocol) and IC 'eps' (silent,
          rested, uniform seed 1e-6: the low branch, for bistability), both x every g. T = 3000.
  driven  ORN drive h_ORN (all 2,639 ORNs), everyone else h = 0. Columns: IC 'quiet' (protocol), IC 'decay' + drive (high
          branch, bistability), IC 'quiet' with h_ORN x 1.05 (static susceptibility). At T1 = 1500 every ORN is clamped
          off (p = 0, rho -> 0) until T = 4500.
Per column it records the expected number of active neurons n_X(t) = sum_{i in X} rho_i(t) per region every model unit,
the first time n_X < n_c (cutoff extinction; n_c in N_CUT) after the reference time (0 for decay, T1 for driven), window
means (decay 1500-3000; driven 500-1500, 500-1000, 1000-1500, 1600-1800, 2500-4500), the time-averaged occupancy over the
window (PR over all N / over nonORN; region and superclass shares), the mean efficacy of active depressing neurons, an
oscillation diagnostic of the deterministic trajectory, and (with --ring) at the decay end state / at the driven state at
T1: (i) a ring-down of the exact linearised map (JVP, float64) after a demographic-noise-shaped kick, recorded for
RING units (population sums), for the matrix-pencil mode analysis; (ii) the renormalised branching ratio
sigma_x = lambda_PF(f(g E diag(x_next)))/alpha_eff with x_next = the resource a new onset would get (frozen).
Usage: m2_mf_run.py --variant M2 --r 1 --proto decay|driven [--orn 0.04] [--null targets --null_seed 0] [--ring]
       [--save_state] --out results/m2/theory/runs/<name>.npz
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/MALECNS_ROOT/code")
import m2_mf_core as M          # noqa: E402
import m2_common as MC          # noqa: E402

N_CUT = (0.1, 1.0, 10.0)
ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="M2")
ap.add_argument("--r", type=float, required=True)
ap.add_argument("--proto", choices=["decay", "driven"], required=True)
ap.add_argument("--orn", type=float, default=0.04)
ap.add_argument("--null", default=None, choices=[None, "targets", "weights"])
ap.add_argument("--null_seed", type=int, default=0)
ap.add_argument("--g", type=float, nargs="*", default=None)
ap.add_argument("--ring", action="store_true")
ap.add_argument("--ring_T", type=float, default=1500.0)
ap.add_argument("--save_state", action="store_true")
ap.add_argument("--T", type=float, default=None)
ap.add_argument("--closure", default="qmf", choices=["qmf", "cqmf"])
ap.add_argument("--ics", default=None, help="comma list overriding the default initial conditions")
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.time()
log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
dev = "cuda"
G = np.array(a.g if a.g else MC.G_COARSE, float)
nG = len(G)
mdl = M.Model(a.variant, a.r, null=a.null, null_seed=a.null_seed, closure=a.closure)
log(f"model {a.variant} r={a.r} null={a.null} nnz={mdl.nnz} c=1/{1/mdl.c:.2f} setup {mdl.setup_s:.0f}s GPU "
    f"{torch.cuda.get_device_name()}")
N = mdl.N
RN = mdl.reg_names; ri = {k: i for i, k in enumerate(RN)}
DT = M.DT
if a.proto == "decay":
    ics = ["decay", "eps"]; T = a.T or MC.T_DECAY; T1 = None
    wins = {"qs": (1500.0, 3000.0)}; occ_win = (1500.0, 3000.0); ref_t = 0.0
    h_orn = 0.0
else:
    ics = ["quiet", "decay", "quiet_x1.05"]; T = a.T or (MC.T1_DRIVE + MC.T_POST); T1 = MC.T1_DRIVE
    wins = {"d": (500.0, 1500.0), "d1": (500.0, 1000.0), "d2": (1000.0, 1500.0), "rm": (1600.0, 1800.0),
            "post": (2500.0, 4500.0)}
    occ_win = (500.0, 1500.0); ref_t = T1
    h_orn = MC.h_of_rho(a.orn)
if a.ics:
    ics = a.ics.split(",")
nC = nG * len(ics)
col_ic = np.repeat(np.arange(len(ics)), nG); col_g = np.tile(G, len(ics))
gv = torch.tensor(col_g[None, :], device=dev, dtype=torch.float32)
# initial state per column block
Z = {k: torch.empty(N, nC, device=dev, dtype=torch.float32) for k in ("rho", "y", "m1", "m0")}
for b, ic in enumerate(ics):
    z = mdl.init_state("decay" if ic == "decay" else ("eps" if ic == "eps" else "quiet"), nG)
    for k in Z:
        Z[k][:, b * nG:(b + 1) * nG] = z[k]
# drive per column
Hc = torch.zeros(N, nC, device=dev, dtype=torch.float32)
if a.proto == "driven":
    orn_t = torch.tensor(mdl.reg_np["ORN"], device=dev)
    for b, ic in enumerate(ics):
        Hc[orn_t, b * nG:(b + 1) * nG] = h_orn * (1.05 if ic == "quiet_x1.05" else 1.0)
keep = None
steps = int(round(T / DT)); s1 = None if T1 is None else int(round(T1 / DT))
per_unit = int(round(1 / DT))
rec = torch.zeros(int(round(T)), len(RN), nC, device=dev, dtype=torch.float32)       # per model unit, all regions
cut_idx = [ri["all"], ri["nonORN"], ri["ALsys"]]
rec_step = torch.zeros(steps, 3, nC, device=dev, dtype=torch.float32)                 # per step: all, nonORN, ALsys
wsum = {k: torch.zeros(len(RN), nC, device=dev, dtype=torch.float64) for k in wins}
wsq = {k: torch.zeros(len(RN), nC, device=dev, dtype=torch.float64) for k in wins}
wn = {k: 0 for k in wins}
occ = torch.zeros(N, nC, device=dev, dtype=torch.float64); nocc = 0
eff_w = torch.zeros(len(RN), nC, device=dev, dtype=torch.float64); rho_w = torch.zeros_like(eff_w)
Rdep = mdl.R * torch.tensor(mdl.dep_np, device=dev, dtype=torch.float64)[None, :]
rsz = torch.tensor(mdl.reg_size, device=dev)[:, None]
Z1 = None
for s in range(1, steps + 1):
    if s1 is not None and s == s1 + 1:                      # removal: every ORN clamped off from now on
        Z1 = {k: v.clone() for k, v in Z.items()}           # driven state at T1 (before the clamp)
        km = torch.zeros(N, nC, device=dev, dtype=torch.bool); km[orn_t] = True
        Z = M.Model.clamp_off(Z, km)
        keep = (~km).float(); Hc = Hc * keep
    Z, s_, p_ = mdl.step(Z, gv, Hc, keep)
    t = s * DT
    cnt = mdl.counts(Z)                                     # [R, C] float64
    rec_step[s - 1] = cnt[cut_idx].float()
    if s % per_unit == 0:
        rec[s // per_unit - 1] = cnt.float()
    fr = cnt / rsz
    for k, (w0, w1) in wins.items():
        if w0 < t <= w1:
            wsum[k] += fr; wsq[k] += fr * fr; wn[k] += 1
    if occ_win[0] < t <= occ_win[1]:
        occ += Z["rho"].double(); nocc += 1
        eff_w += Rdep @ Z["y"].double(); rho_w += Rdep @ Z["rho"].double()
    if s % 2500 == 0:
        log(f"step {s}/{steps} t={t:.0f} n_all(cols)= " + " ".join(f"{v:.3g}" for v in cnt[0, ::max(1, nC // 8)].cpu().numpy()))
rec = rec.cpu().numpy(); rec_step = rec_step.cpu().numpy()
tt = np.arange(1, steps + 1) * DT
after = tt > ref_t
first_cut = {}
for X, j in (("all", 0), ("nonORN", 1), ("ALsys", 2)):
    fc = np.full((len(N_CUT), nC), np.inf)
    for jj, nc in enumerate(N_CUT):
        below = (rec_step[:, j, :] < nc) & after[:, None]
        anyb = below.any(0)
        fc[jj, anyb] = tt[np.argmax(below[:, anyb], 0)] - ref_t
    first_cut[X] = fc
nmin = np.where(after[:, None], rec_step[:, 0, :], np.inf).min(0)
log("main loop done")
out = {"G": G, "ics": np.array(ics), "col_ic": col_ic, "col_g": col_g, "reg_names": np.array(RN), "reg_size": mdl.reg_size,
       "rec": rec, "rec_step_all": rec_step[:, 0, :] if a.proto == "decay" else rec_step[::5, :, :], "n_cut": np.array(N_CUT), "nmin": nmin}
for X in first_cut:
    out[f"cut_{X}"] = first_cut[X]
for k in wins:
    m_ = (wsum[k] / max(wn[k], 1)); out[f"win_{k}"] = m_.cpu().numpy()
    out[f"win2_{k}"] = (wsq[k] / max(wn[k], 1)).cpu().numpy()
occ = occ / max(nocc, 1)
o2 = (occ * occ)
nonorn = torch.tensor(mdl.reg_np["nonORN"], device=dev)
out["PR_all"] = ((occ.sum(0) ** 2) / o2.sum(0).clamp(min=1e-300)).cpu().numpy()
out["PR_nonORN"] = ((occ[nonorn].sum(0) ** 2) / o2[nonorn].sum(0).clamp(min=1e-300)).cpu().numpy()
out["occ_reg"] = (mdl.R @ occ).cpu().numpy()                  # [R, C] summed occupancy per region
out["occ_sc"] = (mdl.SC @ occ).cpu().numpy(); out["sc_names"] = mdl.sc_names; out["sc_size"] = mdl.sc_size
out["eff_act_reg"] = (eff_w / rho_w.clamp(min=1e-300)).cpu().numpy()   # mean efficacy of active depressing neurons
# state at the analysis point: decay -> end state; driven -> state at T1
Zs = Z if a.proto == "decay" else Z1
xn = mdl.efficacy_next(Zs)
out["xnext_reg"] = ((mdl.R @ xn.double()) / torch.tensor(mdl.reg_size, device=dev)[:, None]).cpu().numpy()
out["rho_state_reg"] = (mdl.counts(Zs)).cpu().numpy()
if a.save_state:
    out["state_rho"] = Zs["rho"].half().cpu().numpy(); out["state_xnext"] = xn.half().cpu().numpy()
# ---------------------------------------------------------------- ring-down (linearised, float64) + sigma_x
if a.ring:
    main_cols = np.where(col_ic == 0)[0]                    # protocol IC columns only
    cols_t = torch.tensor(main_cols, device=dev)
    Zr = {k: v[:, cols_t].double() for k, v in Zs.items()}
    Hr = torch.tensor(np.where(mdl.reg_np["ORN"], h_orn, 0.0), device=dev, dtype=torch.float64)[:, None]
    mdl.to64()
    gr = torch.tensor(G[None, :], device=dev, dtype=torch.float64)
    s_ = mdl.input(Zr, gr, Hr)
    p_ = -torch.expm1(-torch.tanh(s_.clamp(min=0)) * DT)
    # one exact step: residual (stationarity of the analysed state)
    Zn, _, _ = mdl.step({k: v.clone() for k, v in Zr.items()}, gr, Hr)
    out["ring_resid"] = (Zn["rho"] - Zr["rho"]).abs().amax(0).cpu().numpy()
    del Zn
    gen = torch.Generator(device=dev); gen.manual_seed(20261109)
    rho = Zr["rho"]
    sd = torch.sqrt((rho * (1 - rho)).clamp(min=0))
    xi = torch.randn(N, len(main_cols), device=dev, dtype=torch.float64, generator=gen)
    dr = xi * sd
    ebar = torch.where(rho > 0, Zr["y"] / rho.clamp(min=1e-300), torch.ones_like(rho))
    xa = torch.where(rho > 0, Zr["m1"] / rho.clamp(min=1e-300), torch.ones_like(rho))
    kicks = {"demog": {"rho": dr, "y": ebar * dr, "m1": xa * dr, "m0": -xa * dr},
             "unif": {"rho": rho * 1e-3, "y": Zr["y"] * 1e-3, "m1": Zr["m1"] * 1e-3, "m0": -Zr["m1"] * 1e-3}}
    nR = int(round(a.ring_T))
    RR = mdl.R
    for kn, dZ in kicks.items():
        tr = torch.zeros(nR + 1, len(RN), len(main_cols), device=dev, dtype=torch.float64)
        lsc = torch.zeros(nR + 1, len(main_cols), device=dev, dtype=torch.float64)    # cumulative log scale
        nrm0 = torch.sqrt(sum((v * v).sum(0) for v in dZ.values())).clamp(min=1e-300)
        dZ = {k: v / nrm0 for k, v in dZ.items()}; cum = torch.log(nrm0)
        tr[0] = RR @ dZ["rho"]; lsc[0] = cum
        for st in range(1, nR * per_unit + 1):
            dZ = mdl.jvp_any(Zr, s_, p_, dZ, gr, Hr)
            if st % per_unit == 0:
                nrm = torch.sqrt(sum((v * v).sum(0) for v in dZ.values())).clamp(min=1e-300)
                dZ = {k: v / nrm for k, v in dZ.items()}; cum = cum + torch.log(nrm)
                tr[st // per_unit] = RR @ dZ["rho"]; lsc[st // per_unit] = cum
        out[f"ring_{kn}"] = tr.cpu().numpy(); out[f"ring_{kn}_logscale"] = lsc.cpu().numpy()
        log(f"ring-down {kn} done")
    # renormalised branching ratio at the frozen resources of the analysed state, and undepressed (x = 1)
    sig_x = np.full(len(main_cols), np.nan); sig_1 = np.full(len(main_cols), np.nan); its = np.zeros(len(main_cols))
    v = None; v1 = None
    one = torch.ones(N, device=dev, dtype=torch.float64)
    xnr = xn[:, cols_t].double()
    for j in range(len(main_cols)):
        lam, sg, v, it = mdl.perron_branch(xnr[:, j], float(G[j]), v0=v, iters=3000, tol=1e-7)
        sig_x[j] = sg; its[j] = it
        lam1, sg1, v1, _ = mdl.perron_branch(one, float(G[j]), v0=v1, iters=3000, tol=1e-7)
        sig_1[j] = sg1
    out["sigma_x"] = sig_x; out["sigma_1"] = sig_1; out["sigma_iters"] = its
    log("sigma_x: " + " ".join(f"{G[j]:.3g}:{sig_x[j]:.3f}" for j in range(0, len(G), 5)))
out["meta"] = json.dumps({"variant": a.variant, "r": a.r, "proto": a.proto, "closure": a.closure, "ics": ics, "orn": a.orn, "h_orn": h_orn, "null": a.null,
                          "null_seed": a.null_seed, "T": T, "T1": T1, "nnz": mdl.nnz, "c": mdl.c, "U": mdl.U0,
                          "TAU": mdl.TAU0, "n_dep": int(mdl.dep_np.sum()), "edge_ov": mdl.n_edge_ov,
                          "gpu": torch.cuda.get_device_name(), "seconds": time.time() - t0})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
tmp = a.out + ".tmp.npz"
np.savez_compressed(tmp, **out)
os.replace(tmp, a.out)
log("saved", a.out)
