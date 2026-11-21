"""pd3 criticality C2/C3: quasi-stationary (QS) order parameter and susceptibility, and field scaling (GPU, dense).

Dynamics: swc.py's dense synchronous update (--norm raw), identical rules: an active unit turns off with probability
1 - e^{-dt}; a quiescent unit turns on with probability 1 - exp(-tanh(g sum_j c w_ij sigma_j a_j + h)_+ dt); dt = 0.2;
sigma = +1 exc, -r inh, 0 modulatory. Graph: --graph real (full brain) or module (526-neuron AL module alone,
results/module/module_graph.npz, internal synapses, full-brain c). x = g / (the graph's own g_lin): real 0.16315275
(phase_complete.json), module 0.16428096 (pd3_common.g_lin_exact on the module, rtol 1e-7; computed 2026-11-09).

--mode qs (C2; de Oliveira & Dickman 2005 QS method; h = 0):
  B replicas, each with M stored configurations (uint8 [N, B, M]). All slots start fully active; during the first 400 steps
  every 25 steps a live replica writes slot (step/25 - 1) (fill), and during the rest of the burn-in it keeps writing slot
  (step/25 - 1) mod M (round-robin refresh; added after the first submission showed rare resets to stale fully-active-derived
  configurations, which inflated var(n_tot) by up to 10^5x in single replicas); during the measurement every 25 steps each
  live replica overwrites a random slot with probability 0.5. A replica found absorbed (no active unit) after a step is reset to a random stored
  configuration of its own and the event is counted. Discard t < --burn (model time), measure --meas model time.
  Every 5 steps: n_tot, n_AL, n_L, n_R per replica -> per-replica sums of n, n^2 (tot and AL, L, R), histograms of n_AL
  (0..526) and of the L/R asymmetry; occupancy occ_i (mean activity over replicas and measurement samples) -> PR;
  absorptions per replica during the measurement. chi_QS = N var(rho)/<rho> = var(n_tot)/<n_tot>, chi_AL = var(n_AL)/<n_AL>,
  tau_QS = B * measured time / absorptions.
--mode field (C3): drive h > 0 (no reactivation, h makes silence non-absorbing). For each h in --hs, start from silence,
  burn-in --burn, measure --meas (model time), the same observables.
Output: results/pd3/criticality/qs_<tag>.npz (or field_<tag>.npz) with one row per x (or h); per-x checkpoint.
Usage: pd3_qs.py --mode qs --graph real --r 0 --xs 0.95 1.03 ... --tag real_r0_p0
       pd3_qs.py --mode field --graph real --r 1 --xs 1.0 --hs 1e-7 3e-7 ... --B 64 --burn 500 --meas 2000 --tag real_r1_x1.00
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd3_common import DT, ROOT, load_cache, c_of

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["qs", "field"], required=True)
ap.add_argument("--graph", choices=["real", "module"], required=True)
ap.add_argument("--r", type=float, required=True)
ap.add_argument("--xs", type=float, nargs="+", required=True)
ap.add_argument("--hs", type=float, nargs="*", default=[0.0])
ap.add_argument("--B", type=int, default=256); ap.add_argument("--M", type=int, default=16)
ap.add_argument("--burn", type=float, default=1000.0); ap.add_argument("--meas", type=float, default=4000.0)
ap.add_argument("--rng", type=int, default=0)
ap.add_argument("--tag", required=True)
ap.add_argument("--outdir", default=f"{ROOT}/results/pd3/criticality")
a = ap.parse_args()
dev = "cuda"; t0 = time.time()
log = lambda *x: print(f"[{time.time()-t0:.1f}s]", *x, flush=True)
pref = "qs" if a.mode == "qs" else "field"
out_path = f"{a.outdir}/{pref}_{a.tag}.npz"; part_path = f"{a.outdir}/partial/{pref}_{a.tag}.partial.npz"
os.makedirs(f"{a.outdir}/partial", exist_ok=True)
if os.path.exists(out_path):
    log("exists:", out_path); sys.exit(0)
gpu = torch.cuda.get_device_name(0); log("GPU", gpu, vars(a))
G_LIN = {"real": 0.16315274988719286, "module": 0.16428095549634494}[a.graph]

mg = np.load(f"{ROOT}/results/module/module_graph.npz")
if a.graph == "module":
    N = len(mg["idx"]); pre = mg["pre"].astype(np.int64); post = mg["post"].astype(np.int64); w = mg["w"].astype(np.float64)
    esign = mg["sign"].astype(np.int64)[pre]; c = float(mg["c"]); sides = mg["sides"]
    alL = np.where(sides == "L")[0]; alR = np.where(sides == "R")[0]; alI = np.arange(N)
else:
    pre, post, w, sign = load_cache(); N = len(sign); esign = sign[pre]; c = c_of(post, w, N)
    alI = mg["idx"].astype(np.int64); sides = mg["sides"]
    alL = alI[sides == "L"]; alR = alI[sides == "R"]
coef = np.where(esign > 0, 1.0, np.where(esign < 0, -a.r, 0.0)) * w * c
keep = coef != 0
idx = torch.tensor(np.vstack([post[keep], pre[keep]]), device=dev)
W = torch.sparse_coo_tensor(idx, torch.tensor(coef[keep].astype(np.float32), device=dev), (N, N)).coalesce().to_sparse_csr()
del idx, pre, post, w, coef, keep
log(f"graph {a.graph}: N={N}, c={c:.6g}, r={a.r}, g_lin={G_LIN}")
mAL = torch.zeros(N, device=dev); mAL[torch.tensor(alI, device=dev)] = 1
mL = torch.zeros(N, device=dev); mL[torch.tensor(alL, device=dev)] = 1
mR = torch.zeros(N, device=dev); mR[torch.tensor(alR, device=dev)] = 1
MASKS = torch.stack([torch.ones(N, device=dev), mAL, mL, mR], 0)        # [4, N] -> counts tot, AL, L, R
NAL = len(alI)
p_off = 1 - np.exp(-DT)
B, M = a.B, a.M
burn_steps = int(round(a.burn / DT)); meas_steps = int(round(a.meas / DT)); total = burn_steps + meas_steps
gen = torch.Generator(device=dev)

jobs = [(x, h) for x in a.xs for h in (a.hs if a.mode == "field" else [0.0])]
KEYS = ["x", "g", "h", "s_n", "s_n2", "nsamp", "absorb", "hist_AL", "hist_asym", "occ_PR", "occ_AL_share", "sec",
        "n_series"]
done = {k: [] for k in KEYS}
if os.path.exists(part_path):
    P = np.load(part_path)
    for k in KEYS:
        done[k] = list(P[k])
    log(f"resume: {len(done['x'])} of {len(jobs)} done")
NSER = 2000                                    # stored time series (replica 0..3, n_AL) for autocorrelation, every 5 steps
for ji in range(len(done["x"]), len(jobs)):
    x, h = jobs[ji]; g = x * G_LIN; tj = time.time()
    gen.manual_seed(1_000_003 * a.rng + 10007 * ji + int(1e6 * x) % 99991)
    if a.mode == "qs":
        X = torch.ones(N, B, device=dev)
        store = torch.ones(N, B, M, dtype=torch.uint8, device=dev)
    else:
        X = torch.zeros(N, B, device=dev)
    s_n = torch.zeros(B, 4, dtype=torch.float64, device=dev); s_n2 = torch.zeros_like(s_n)
    nsamp = 0; absorb = torch.zeros(B, dtype=torch.float64, device=dev)
    hist_AL = torch.zeros(NAL + 1, dtype=torch.float64, device=dev); hist_asym = torch.zeros(41, dtype=torch.float64, device=dev)
    occ = torch.zeros(N, dtype=torch.float64, device=dev); ser = []
    ar = torch.arange(B, device=dev)
    for s in range(1, total + 1):
        inp = g * torch.sparse.mm(W, X) + h
        if h > 0:   # field mode: -expm1 keeps the tiny drive rates (1 - exp(-x) rounds to 0 in float32 for x < ~3e-8)
            p_on = -torch.expm1(-torch.tanh(inp.clamp(min=0)) * DT)
        else:       # identical to swc.py
            p_on = 1 - torch.exp(-torch.tanh(inp.clamp(min=0)) * DT)
        u = torch.rand(X.shape, generator=gen, device=dev)
        X = torch.where(X > 0, (u >= p_off).float(), (u < p_on).float())
        if a.mode == "qs":
            n_act = X.sum(0)
            dead = torch.nonzero(n_act == 0).ravel()
            if dead.numel():
                slot = torch.randint(M, (dead.numel(),), generator=gen, device=dev)
                X[:, dead] = store[:, dead, slot].float()
                if s > burn_steps:
                    absorb[dead] += 1
            if s % 25 == 0:
                live = n_act > 0
                if s <= burn_steps:
                    # fill (s <= 400) and then round-robin refresh during the whole burn-in: every slot is rewritten every
                    # 400 steps, so no configuration from the fully active start survives into the measurement
                    li = torch.nonzero(live).ravel()
                    store[:, li, (s // 25 - 1) % M] = X[:, li].to(torch.uint8)
                else:
                    wr = live & (torch.rand(B, generator=gen, device=dev) < 0.5)
                    li = torch.nonzero(wr).ravel()
                    if li.numel():
                        sl = torch.randint(M, (li.numel(),), generator=gen, device=dev)
                        store[:, li, sl] = X[:, li].to(torch.uint8)
        if s > burn_steps and s % 5 == 0:
            cnt = (MASKS @ X).T.double()                                   # [B, 4]
            s_n += cnt; s_n2 += cnt * cnt; nsamp += 1
            hist_AL += torch.bincount(cnt[:, 1].long().clamp(max=NAL), minlength=NAL + 1).double()
            lr = cnt[:, 2] + cnt[:, 3]
            asy = torch.where(lr > 0, (cnt[:, 2] - cnt[:, 3]) / lr.clamp(min=1), torch.zeros_like(lr))
            hist_asym += torch.bincount(torch.clamp(((asy + 1) * 20).round().long(), 0, 40), minlength=41).double()
            occ += X.double().mean(1)
            if len(ser) < NSER:
                ser.append(cnt[:4, 1].cpu().numpy())
    occ /= max(nsamp, 1)
    pr = float(occ.sum() ** 2 / (occ ** 2).sum()) if float(occ.sum()) > 0 else 0.0
    alsh = float(occ[torch.tensor(alI, device=dev)].sum() / occ.sum()) if float(occ.sum()) > 0 else 0.0
    sn = s_n.cpu().numpy(); sn2 = s_n2.cpu().numpy()
    vals = {"x": x, "g": g, "h": h, "s_n": sn, "s_n2": sn2, "nsamp": nsamp, "absorb": absorb.cpu().numpy(),
            "hist_AL": hist_AL.cpu().numpy(), "hist_asym": hist_asym.cpu().numpy(), "occ_PR": pr, "occ_AL_share": alsh,
            "sec": time.time() - tj, "n_series": np.array(ser + [np.full(4, np.nan)] * (NSER - len(ser)))}
    for k in KEYS:
        done[k].append(vals[k])
    m = sn.sum(0) / (nsamp * B); v = sn2.sum(0) / (nsamp * B) - m ** 2
    tau = (B * a.meas / float(absorb.sum())) if float(absorb.sum()) > 0 else float("inf")
    log(f"x={x:.4f} g={g:.5f} h={h:g}: <n_tot> {m[0]:.4g} <n_AL> {m[1]:.4g} chi_tot {v[0]/max(m[0],1e-12):.3g} "
        f"chi_AL {v[1]/max(m[1],1e-12):.3g} tau_QS {tau:.4g} PR {pr:.4g} AL share {alsh:.3f} ({time.time()-tj:.0f}s)")
    tmp = part_path + ".tmp.npz"
    np.savez(tmp, **{k: np.array(v_) for k, v_ in done.items()}); os.replace(tmp, part_path)

out = {k: np.array(v) for k, v in done.items()}
out["meta"] = json.dumps({**vars(a), "gpu": gpu, "N": N, "c": c, "g_lin": G_LIN, "NAL": NAL, "nL": len(alL), "nR": len(alR),
                          "wall_s": time.time() - t0, "burn_steps": burn_steps, "meas_steps": meas_steps,
                          "count_order": ["tot", "AL", "L", "R"], "hist_asym_bins": "round((A+1)*20), A=(nL-nR)/(nL+nR)"})
tmp = out_path + ".tmp.npz"
np.savez_compressed(tmp, **out); os.replace(tmp, out_path)
if os.path.exists(part_path):
    os.remove(part_path)
log("saved", out_path)
