"""pd3 criticality C4: lifetime at g_lin against module size K, on nested AL sub-modules (GPU, dense, swc.py rules).

Sub-modules: the top-n neurons ranked by P[:,0] + P[:,1] (|v|^2 of the two leading excitatory modes,
results/pd/tail_real_r0_dir.npz), n in --ns; induced internal subgraph (edges with both ends inside, presynaptic sign != 0),
full-brain c = 1/(mean in-strength over neurons with in-strength > 0) = the c of module_graph.npz (module_sim convention).
Per sub-module (GPU, float64): exact g_lin,n (pd3_common.g_lin_exact, rtol 1e-7), the Perron vectors of B(g_lin,n) ->
K_n = 1/(<u^2,v><u,v^2>) and 1/IPR; R(x) = lambda_PF(B(x g_lin,n))/alpha_eff at the simulated x.
Dynamics: swc.py's synchronous update (dt 0.2, h = 0), fully active start, B replicas, T model time; the exact extinction step
per replica is stored (int32, -1 = alive at T); also rho on swc.py's log grid (replica mean).
Usage: pd3_submodule.py --r 0 [--ns 50 75 ...] [--xs 1.0 1.2 1.5 2.0] [--B 4096] [--T 3000] --tag r0
Output: results/pd3/criticality/submodule_<tag>.npz (+ per-sub-module checkpoint).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd3_common import A_EFF, DT, ROOT, load_cache, c_of, exc_csr_torch, B_mats, perron_t, k_nc, g_lin_exact

ap = argparse.ArgumentParser()
ap.add_argument("--r", type=float, required=True)
ap.add_argument("--ns", type=int, nargs="+", default=[50, 75, 100, 150, 200, 300, 400, 526, 800, 1200])
ap.add_argument("--xs", type=float, nargs="+", default=[1.0, 1.2, 1.5, 2.0])
ap.add_argument("--B", type=int, default=4096); ap.add_argument("--T", type=float, default=3000.0)
ap.add_argument("--rng", type=int, default=0)
ap.add_argument("--tag", required=True)
ap.add_argument("--outdir", default=f"{ROOT}/results/pd3/criticality")
a = ap.parse_args()
dev = "cuda"; t0 = time.time()
log = lambda *x: print(f"[{time.time()-t0:.1f}s]", *x, flush=True)
out_path = f"{a.outdir}/submodule_{a.tag}.npz"; part_path = f"{a.outdir}/partial/submodule_{a.tag}.partial.npz"
os.makedirs(f"{a.outdir}/partial", exist_ok=True)
if os.path.exists(out_path):
    log("exists:", out_path); sys.exit(0)
gpu = torch.cuda.get_device_name(0); log("GPU", gpu, vars(a))
pre, post, w, sign = load_cache(); NF = len(sign); c = c_of(post, w, NF)
P = np.load(f"{ROOT}/results/pd/tail_real_r0_dir.npz")["P"]
score = P[:, 0].astype(np.float64) + P[:, 1].astype(np.float64)
order = np.argsort(-score, kind="stable")
mod = np.load(f"{ROOT}/results/module/module_graph.npz")["idx"]
log(f"c = {c:.6g}; ranked by P0+P1; top-526 overlap with the 526-module: {len(np.intersect1d(order[:526], mod))}")
steps = int(round(a.T / DT)); rec = np.unique(np.round(np.logspace(0, np.log10(steps), 100)).astype(int))
p_off = 1 - np.exp(-DT)
KEYS = ["n", "g_lin", "K_n", "inv_ipr", "lam_E", "R_x", "n_edges", "n_exc", "n_inh", "ext_step", "rho", "sec", "mod_overlap"]
done = {k: [] for k in KEYS}
if os.path.exists(part_path):
    Pp = np.load(part_path)
    for k in KEYS:
        done[k] = list(Pp[k])
    log(f"resume: {len(done['n'])} sub-modules done")
gen = torch.Generator(device=dev)
for ni in range(len(done["n"]), len(a.ns)):
    n = a.ns[ni]; tn = time.time()
    S = np.sort(order[:n]); loc = -np.ones(NF, np.int64); loc[S] = np.arange(n)
    e = (loc[pre] >= 0) & (loc[post] >= 0) & (sign[pre] != 0)
    pi, po, ww, es = loc[pre[e]], loc[post[e]], w[e], sign[pre[e]].astype(np.int64)
    Ed = exc_csr_torch(pi, po, ww, es, c, n, dev)
    lamE, vE, _ = perron_t(B_mats(Ed, None)[0])
    gl, v = g_lin_exact(Ed, lamE, rtol=1e-7, v0=vE)
    lam, v, _ = perron_t(B_mats(Ed, gl)[0], v); _, u, _ = perron_t(B_mats(Ed, gl)[1], v)
    K, ipr = k_nc(u, v)
    Rx = []
    for x in a.xs:
        lx, _, _ = perron_t(B_mats(Ed, x * gl)[0], v); Rx.append(lx / A_EFF)
    del Ed
    coef = np.where(es > 0, 1.0, -a.r) * ww * c
    kp = coef != 0
    W = torch.sparse_coo_tensor(torch.tensor(np.vstack([po[kp], pi[kp]]), device=dev),
                                torch.tensor(coef[kp].astype(np.float32), device=dev), (n, n)).coalesce().to_sparse_csr()
    ext_all, rho_all = [], []
    for xi, x in enumerate(a.xs):
        g = x * gl
        gen.manual_seed(1_000_003 * a.rng + 1009 * n + 17 * xi + int(round(100 * a.r)))
        X = torch.ones(n, a.B, device=dev); ext = torch.full((a.B,), -1, dtype=torch.int32, device=dev)
        rho, k = [], 0
        for s in range(1, steps + 1):
            inp = g * torch.sparse.mm(W, X)
            p_on = 1 - torch.exp(-torch.tanh(inp.clamp(min=0)) * DT)
            uu = torch.rand(X.shape, generator=gen, device=dev)
            X = torch.where(X > 0, (uu >= p_off).float(), (uu < p_on).float())
            na = X.sum(0)
            ext = torch.where((na == 0) & (ext < 0), torch.full_like(ext, s), ext)
            if s == rec[k]:
                rho.append(float(na.mean()) / n); k += 1
                if k == len(rec):
                    break
                if bool((ext >= 0).all()):
                    rho += [0.0] * (len(rec) - k); break
        ext_all.append(ext.cpu().numpy()); rho_all.append(np.array(rho))
        ee = ext_all[-1]; d = ee >= 0
        tau = (np.where(d, ee, steps).sum() * DT / max(d.sum(), 1))
        log(f"n={n} x={x} g={g:.5f} R={Rx[xi]:.4f}: deaths {d.sum()}/{a.B} tau(censored MLE) {tau:.4g}")
    vals = {"n": n, "g_lin": gl, "K_n": K, "inv_ipr": ipr, "lam_E": lamE, "R_x": np.array(Rx), "n_edges": int(kp.sum()),
            "n_exc": int((sign[S] > 0).sum()), "n_inh": int((sign[S] < 0).sum()), "ext_step": np.array(ext_all),
            "rho": np.array(rho_all), "sec": time.time() - tn, "mod_overlap": len(np.intersect1d(S, mod))}
    for k_ in KEYS:
        done[k_].append(vals[k_])
    log(f"n={n}: g_lin {gl:.6g} K_n {K:.4g} 1/IPR {ipr:.4g} lam_E {lamE:.4g} edges {vals['n_edges']} ({vals['sec']:.0f}s)")
    tmp = part_path + ".tmp.npz"; np.savez(tmp, **{k_: np.array(v_) for k_, v_ in done.items()}); os.replace(tmp, part_path)
out = {k_: np.array(v_) for k_, v_ in done.items()}
out.update({"xs": np.array(a.xs), "rec_t": rec * DT, "meta": json.dumps({**vars(a), "gpu": gpu, "c": c, "wall_s": time.time() - t0})})
tmp = out_path + ".tmp.npz"; np.savez_compressed(tmp, **out); os.replace(tmp, out_path)
if os.path.exists(part_path):
    os.remove(part_path)
log("saved", out_path)
