"""pd3 redraw_fss part B: finite-size-scaling decay runs on size-N graphs (GPU). One call = one (graph, r) with all g.

Usage: pd3_fss_sim.py <kind> <N> <seed> <r> [--B 64] [--T 3000]
  kind in {cnull, realsub} (graph built by pd3_fss_graph.build, deterministic in (kind, N, seed)).
Dynamics: adapted copy of swc.py --exp decay --norm raw (identical update rule): binary units, synchronous dt = 0.2; an active
  unit turns off with probability 1 - e^{-dt}; a quiescent unit turns on with probability 1 - exp(-tanh(s)_+ dt),
  s = g sum_j c w_ij sigma_j a_j (h = 0), sigma = +1 exc, -r inh, 0 modulatory (per-edge presynaptic sign), c = the graph's c.
  Fully active start, B replicas, T = 3000 (15,000 steps).
g grid relative to the graph's own exact g_lin: r = 0: x = g/g_lin at 30 ln-spaced points in [0.90, 2.50];
  r = 1 (and any r > 0): 36 points in [0.90, 8.0].
Recorded per g: ext_step[B] (int32, the exact step at which the replica first had no active unit; -1 = alive at T);
  rho/surv on swc.py's log time grid; occ[N] (float16; per-neuron activity averaged over replicas and the second half);
  per replica, among samples with n > 0 taken every 5 steps over (T/2, T]: count, sum n, sum n^2 (float64).
Per-graph diagnostics at task start (GPU, float64): lambda_PF(E) and its left/right Perron vectors (weak-synapse K_nc, 1/IPR,
  the pd3_theory_nullK.py definitions, for validation against results/pd3/theory_nullK.json); exact g_lin (bisection, rtol 1e-6);
  u, v of B(g_lin) -> K_nc(g_lin), 1/IPR(g_lin), AL share of v^2; R(x) = lambda_PF(B(x g_lin)) / alpha_eff on the grid.
Output (atomic): results/pd3/redraw_fss/fss_<kind>_N<N>_s<seed>_r<r>.npz; a per-g checkpoint (partial file) lets a
  resubmitted task resume.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd3_common import (A_EFF, DT, ROOT, exc_csr_torch, B_mats, perron_t, k_nc, g_lin_exact, load_cache)
from pd3_fss_graph import build

ap = argparse.ArgumentParser()
ap.add_argument("kind"); ap.add_argument("N", type=int); ap.add_argument("seed", type=int); ap.add_argument("r", type=float)
ap.add_argument("--B", type=int, default=64); ap.add_argument("--T", type=float, default=3000.0)
ap.add_argument("--outdir", default=f"{ROOT}/results/pd3/redraw_fss")
ap.add_argument("--ng", type=int, default=0, help="smoke test only: simulate the first ng grid points")
a = ap.parse_args()
dev = "cuda"; t0 = time.time()
log = lambda *x: print(f"[{time.time()-t0:.1f}s]", *x, flush=True)
tag = f"fss_{a.kind}_N{a.N}_s{a.seed}_r{a.r:g}"
out_path = f"{a.outdir}/{tag}.npz"; part_path = f"{a.outdir}/partial/{tag}.partial.npz"
os.makedirs(f"{a.outdir}/partial", exist_ok=True)
if os.path.exists(out_path):
    log("exists, nothing to do:", out_path); sys.exit(0)
gpu = torch.cuda.get_device_name(0)
log("GPU", gpu, "task", tag)

G = build(a.kind, a.N, a.seed, load_cache())
N = G["N"]
log(f"graph built: N={N} edges={G['n_edges']} c={G['c']:.6g} (1/c = {1/G['c']:.2f}) AL in graph: {len(G['al_local'])}")

# ------------------------------------------------------------------------------------------------ spectral diagnostics
Ed = exc_csr_torch(G["pre"], G["post"], G["w"], G["esign"], G["c"], N, dev)
E, ET = B_mats(Ed, None)
lamE, vE, itE = perron_t(E, iters=20000, tol=1e-10)
lamET, uE, itET = perron_t(ET, iters=20000, tol=1e-10)
K_E, ipr_E = k_nc(uE, vE)
log(f"lambda_PF(E) {lamE:.10g} (left {lamET:.10g}; iters {itE}/{itET}); weak g {A_EFF/lamE:.6g}; K_nc(E) {K_E:.6g}; 1/IPR {ipr_E:.4g}")
g_lin, v = g_lin_exact(Ed, lamE, rtol=1e-6, v0=vE)
B, BT = B_mats(Ed, g_lin)
lam_l, v, itv = perron_t(B, v)
_, u, itu = perron_t(BT, v)
K_g, ipr_g = k_nc(u, v)
vv = (v.abs().double().ravel() ** 2); vv = (vv / vv.sum()).cpu().numpy()
al_share = float(vv[G["al_local"]].sum()) if len(G["al_local"]) else 0.0
log(f"g_lin {g_lin:.7g} (R at g_lin {lam_l/A_EFF:.7f}); K_nc(g_lin) {K_g:.6g}; 1/IPR {ipr_g:.4g}; AL share of v^2 {al_share:.3f}")

if a.r == 0:
    xg = np.exp(np.linspace(np.log(0.90), np.log(2.50), 30))
else:
    xg = np.exp(np.linspace(np.log(0.90), np.log(8.0), 36))
gg = xg * g_lin
if a.ng:
    xg, gg = xg[:a.ng], gg[:a.ng]
Rx = []
vw = v
for g in gg:
    lam, vw, _ = perron_t(B_mats(Ed, g)[0], vw)
    Rx.append(lam / A_EFF)
Rx = np.array(Rx)
log("R(x):", " ".join(f"{x:.3f}:{R:.3f}" for x, R in zip(xg[::5], Rx[::5])))

# validation against pd3_theory_nullK.json (same construction, same seeds)
val = None
ref = json.load(open(f"{ROOT}/results/pd3/theory_nullK.json"))["rows"]
for row in ref:
    if a.kind == "cnull" and row["N"] == a.N and row["seed"] == a.seed:
        val = {"ref_lambda_PF": row["lambda_PF"], "ref_K_nc": row["K_nc"], "ref_inv_ipr": row["inv_ipr"], "ref_c": row["c"],
               "ref_n_edges": row["n_edges"],
               "rel_dev_lambda": abs(lamE / row["lambda_PF"] - 1), "rel_dev_K": abs(K_E / row["K_nc"] - 1),
               "rel_dev_inv_ipr": abs(ipr_E / row["inv_ipr"] - 1), "rel_dev_c": abs(G["c"] / row["c"] - 1),
               "edges_equal": G["n_edges"] == row["n_edges"]}
        log("VALIDATION vs theory_nullK:", val)
diag = {"kind": a.kind, "N": N, "seed": a.seed, "n_edges": G["n_edges"], "c": G["c"], "lambda_PF_E": lamE,
        "lambda_PF_E_left": lamET, "g_weak": A_EFF / lamE, "K_nc_E": K_E, "inv_ipr_E": ipr_E, "g_lin": g_lin,
        "R_at_g_lin": lam_l / A_EFF, "K_nc_glin": K_g, "inv_ipr_glin": ipr_g, "al_share_v2_glin": al_share,
        "n_al_in_graph": int(len(G["al_local"])), "perron_iters": [itE, itET, itv, itu], "validation": val}
del E, ET, B, BT, Ed
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------------------------------------- simulation
coef = np.where(G["esign"] > 0, 1.0, np.where(G["esign"] < 0, -a.r, 0.0)) * G["w"] * G["c"]
keep = coef != 0
idx = torch.tensor(np.vstack([G["post"][keep], G["pre"][keep]]), device=dev)
Wm = torch.sparse_coo_tensor(idx, torch.tensor(coef[keep].astype(np.float32), device=dev), (N, N)).coalesce().to_sparse_csr()
del idx
steps = int(round(a.T / DT))
rec = np.unique(np.round(np.logspace(0, np.log10(steps), 120)).astype(int))
p_off = 1 - np.exp(-1.0 * DT)
torch.manual_seed(1_000_003 * a.seed + 7 * a.N + int(round(100 * a.r)) + (0 if a.kind == "cnull" else 55_555))

keys = ["ext_step", "rho", "surv", "occ", "n_cnt", "n_s1", "n_s2", "sec", "steps_run"]
done = {k: [] for k in keys}
if os.path.exists(part_path):
    P = np.load(part_path)
    if np.allclose(P["g"], gg[:len(P["g"])], rtol=1e-9):
        for k in keys:
            done[k] = list(P[k])
        log(f"resuming from checkpoint: {len(done['sec'])} of {len(gg)} g done")
for gi in range(len(done["sec"]), len(gg)):
    g = float(gg[gi]); tg = time.time()
    X = torch.ones(N, a.B, device=dev)
    ext = torch.full((a.B,), -1, dtype=torch.int32, device=dev)
    cnt = torch.zeros(a.B, dtype=torch.float64, device=dev); s1 = torch.zeros_like(cnt); s2 = torch.zeros_like(cnt)
    occ = torch.zeros(N, device=dev); n_occ = 0
    rho, surv, k = [], [], 0
    s_last = steps
    for s in range(1, steps + 1):
        inp = g * torch.sparse.mm(Wm, X)
        p_on = 1 - torch.exp(-torch.tanh(inp.clamp(min=0)) * DT)
        uu = torch.rand_like(X)
        X = torch.where(X > 0, (uu >= p_off).float(), (uu < p_on).float())
        n_act = X.sum(0)
        dead_now = (n_act == 0) & (ext < 0)
        ext = torch.where(dead_now, torch.full_like(ext, s), ext)
        if s > steps // 2:
            occ += X.mean(1); n_occ += 1
            if s % 5 == 0:
                al = (n_act > 0).double(); nd = n_act.double()
                cnt += al; s1 += nd * al; s2 += nd * nd * al
        if s == rec[k]:
            rho.append((n_act / N).cpu().numpy()); surv.append((n_act > 0).float().cpu().numpy())
            k += 1
            if k == len(rec):
                break
            if bool((ext >= 0).all()):          # all replicas absorbed (h = 0): pad and stop
                pad = len(rec) - k
                rho += [np.zeros(a.B, np.float32)] * pad; surv += [np.zeros(a.B, np.float32)] * pad
                s_last = s
                break
    done["ext_step"].append(ext.cpu().numpy().astype(np.int32))
    done["rho"].append(np.array(rho, np.float32)); done["surv"].append(np.array(surv, np.uint8))
    done["occ"].append((occ / max(n_occ, 1)).cpu().numpy().astype(np.float16))
    done["n_cnt"].append(cnt.cpu().numpy()); done["n_s1"].append(s1.cpu().numpy()); done["n_s2"].append(s2.cpu().numpy())
    done["sec"].append(time.time() - tg); done["steps_run"].append(s_last)
    e = done["ext_step"][-1]
    log(f"g={g:.5g} x={xg[gi]:.3f} deaths {int((e >= 0).sum())}/{a.B} rho(T)={done['rho'][-1][-1].mean():.4g} "
        f"steps {s_last} ({done['sec'][-1]:.1f}s, {1e3*done['sec'][-1]/s_last:.2f} ms/step)")
    tmp = part_path + ".tmp.npz"
    np.savez(tmp, g=gg[:gi + 1], **{k: np.array(v) for k, v in done.items()})
    os.replace(tmp, part_path)

out = {"x": xg, "g": gg, "R_x": Rx, "t": rec * DT, "rec_steps": rec, "dt": DT, "T": a.T, "B": a.B, "r": a.r,
       "diag": json.dumps(diag), "meta": json.dumps({**vars(a), "gpu": gpu, "wall_s": time.time() - t0})}
out.update({k: np.array(v) for k, v in done.items()})
tmp = out_path + ".tmp.npz"
np.savez_compressed(tmp, **out)
os.replace(tmp, out_path)
if os.path.exists(part_path):
    os.remove(part_path)
log("saved", out_path)
