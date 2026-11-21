"""Stage 2: stochastic Wilson-Cowan dynamics on the MaleCNS v1.0 neuron graph (GPU, torch sparse).

Model: stochastic Wilson-Cowan (Benayoun, Cowan, van Drongelen & Wallace 2010, PLoS Comput Biol 6:e1000846), one binary
unit per neuron. An active unit decays at rate alpha (=1, the time unit). A quiescent unit activates at rate f(s_i), with
   s_i = g * ( sum_j We_ij a_j  -  r * sum_j Wi_ij a_j ) + h ,   f(s) = beta * tanh(s) for s > 0, else 0.
We/Wi = synapse counts from excitatory (ACh) / inhibitory (GABA, Glu, His) presynaptic neurons, normalised either per
neuron by total in-strength ("in", so every neuron's maximal drive is O(1)) or by the global mean in-strength ("raw",
which keeps the hub heterogeneity). Modulatory/unclear presynaptic neurons (sign 0) do not contribute.
With h = 0 the all-quiescent state is absorbing, so contact-process-style spreading analysis applies: in a Griffiths phase
the density decays as a power law t^-a(g), with an exponent that varies continuously over an extended g interval below the
global critical point (Moretti & Munoz 2013).

Nulls (--null): --null_kind targets (directed configuration model) or weights (same wiring, synapse counts shuffled).
Experiments (--exp):
  decay  : start fully active, h=0; record rho(t) (fraction active) on a log time grid, B replicas.
  spread : start from one active seed neuron per replica (B random seeds, or drawn from --seed_set with probabilities p,
           e.g. |v_i|^2 of the leading eigenvector); record N(t) and survival P(t).
Usage: python swc.py --exp decay --norm in --g 1.0 1.2 ... --r 1.0 --T 2000 --dt 0.2 --B 32 --out results/xxx.npz
"""
import argparse, json, time
import numpy as np
import torch

C = "/MALECNS_CACHE"
ap = argparse.ArgumentParser()
ap.add_argument("--exp", choices=["decay", "spread"], required=True)
ap.add_argument("--norm", choices=["in", "raw"], default="in")
ap.add_argument("--g", type=float, nargs="+", required=True)
ap.add_argument("--r", type=float, default=1.0)
ap.add_argument("--h", type=float, default=0.0)
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--T", type=float, default=2000.0)
ap.add_argument("--dt", type=float, default=0.2)
ap.add_argument("--B", type=int, default=32)
ap.add_argument("--null", action="store_true", help="null graph, see --null_kind")
ap.add_argument("--null_kind", choices=["targets", "weights"], default="targets",
                help="targets: directed configuration model (post permuted; keeps in/out degrees and the weight multiset); "
                     "weights: keep the wiring, permute synapse counts across edges (destroys which connections are strong)")
ap.add_argument("--seed_set", default=None, help="spread only: npz with idx (candidate seed neurons) and p (probabilities)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"
t0 = time.time()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)

pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float32); sign = np.load(C + "/sign.npy")
N = len(sign)
if a.null and a.null_kind == "targets":
    post = post[rng.permutation(len(post))]
elif a.null:
    w = w[rng.permutation(len(w))]
ins = np.bincount(post, weights=w, minlength=N).astype(np.float32)
scale = np.where(ins > 0, 1.0 / np.maximum(ins, 1e-9), 0.0) if a.norm == "in" else np.full(N, 1.0 / ins[ins > 0].mean())
coef = np.where(sign[pre] > 0, 1.0, np.where(sign[pre] < 0, -a.r, 0.0)).astype(np.float32) * w * scale[post].astype(np.float32)
keep = coef != 0
idx = torch.tensor(np.vstack([post[keep], pre[keep]]), device=dev)
Wm = torch.sparse_coo_tensor(idx, torch.tensor(coef[keep], device=dev), (N, N)).coalesce().to_sparse_csr()
del idx
print(f"[{time.time()-t0:.1f}s] graph on GPU: N={N}, nnz={int(keep.sum())}, norm={a.norm}, r={a.r}, null={a.null}"
      f"{' (' + a.null_kind + ')' if a.null else ''}", flush=True)

steps = int(round(a.T / a.dt))
rec = np.unique(np.round(np.logspace(0, np.log10(steps), 120)).astype(int))
p_off = 1 - np.exp(-1.0 * a.dt)
out = {"t": rec * a.dt, "g": np.array(a.g)}
res_rho, res_surv, res_occ = [], [], []
for g in a.g:
    if a.exp == "decay":
        X = torch.ones(N, a.B, device=dev)
    else:
        X = torch.zeros(N, a.B, device=dev)
        if a.seed_set:
            ss = np.load(a.seed_set)
            seeds = rng.choice(ss["idx"], a.B, replace=True, p=ss["p"])
        else:
            seeds = rng.choice(N, a.B, replace=False)
        X[torch.tensor(seeds, device=dev), torch.arange(a.B, device=dev)] = 1
    rho, surv, k = [], [], 0
    occ = torch.zeros(N, device=dev); n_occ = 0     # per-neuron activity averaged over replicas, second half of the run
    for s in range(1, steps + 1):
        inp = g * torch.sparse.mm(Wm, X) + a.h
        p_on = 1 - torch.exp(-a.beta * torch.tanh(inp.clamp(min=0)) * a.dt)
        u = torch.rand_like(X)
        X = torch.where(X > 0, (u >= p_off).float(), (u < p_on).float())
        if s > steps // 2:
            occ += X.mean(1); n_occ += 1
        if s == rec[k]:
            n_act = X.sum(0)
            rho.append((n_act / N).cpu().numpy()); surv.append((n_act > 0).float().cpu().numpy())
            k += 1
            if k == len(rec):
                break
            if a.h == 0 and n_act.sum() == 0:      # all replicas absorbed: pad and stop
                pad = len(rec) - k
                rho += [np.zeros(a.B)] * pad; surv += [np.zeros(a.B)] * pad
                break
    res_rho.append(np.array(rho)); res_surv.append(np.array(surv))
    res_occ.append((occ / max(n_occ, 1)).cpu().numpy().astype(np.float16))
    r_end = res_rho[-1][-1].mean()
    print(f"[{time.time()-t0:.1f}s] g={g:.4f} rho(T)={r_end:.4g} surv(T)={res_surv[-1][-1].mean():.3f}", flush=True)
out["rho"] = np.array(res_rho); out["surv"] = np.array(res_surv); out["occ"] = np.array(res_occ)
out["meta"] = json.dumps(vars(a))
np.savez_compressed(a.out, **out)
print("saved", a.out, flush=True)
