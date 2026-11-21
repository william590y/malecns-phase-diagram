"""E4 (weight transforms): leading EXCITATORY eigenmodes of the whole-CNS operator for each synapse-count transform, in
the units the simulation uses (ald_sim.py --wtransform T --wnorm renorm, --norm raw).

Operator: M_T[post, pre] = c_T * T(w) for presynaptic neurons with model sign +1 (ACh), where
  c_T = 1 / mean_i sum_j T(w_ij)   (all edges, neurons with input; = ald_sim.node_scale with wnorm renorm)
so the mean in-strength is 1 for every transform ("equal mean in-strength" units). r does not enter M (inhibition acts
only beyond linear order). Also reported: lam_clin = eigenvalue with the LINEAR c (1/742.7), the convention of
results/al_lit/grounding/spec_*.json (al_spectra.py), as a cross-check.
For each transform: top-K eigenvalues (real part), localization of each mode (1/IPR, n50/n90, region / category / type
composition via al_common.describe_mode), share of |v|^2 in the 526-neuron AL module and in AL classes, first mode carried
by the AL (AL share >= 0.5) and first mode not carried by it; naive threshold g_lin_naive = alpha_eff / lam_1; exact g_lin
of the tanh-saturated branching operator B(g) = (1 - exp(-tanh(g M) dt))/dt (entrywise, bisection to
lambda_PF(B) = alpha_eff, as theory_branch_gpu.py / ald_sign_spec.py); left Perron vector; |v|^2 of the top 3 right modes.
ARPACK (scipy eigs) with the mat-vec done on the GPU in float64.
Usage: ald_wt_spec.py <transform> [<transform> ...]   ->  results/aldef/weights/spec/<name>.json + <name>_P.npz
"""
import json, os, sys, time
import numpy as np
import scipy.sparse.linalg as sla
import torch

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
from al_common import load, labels, describe_mode, ALPHA_EFF          # noqa: E402
import ald_sim as S                                                   # noqa: E402

OUT = f"{ROOT}/results/aldef/weights/spec"
os.makedirs(OUT, exist_ok=True)
DT, K_MODES = 0.2, 12
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:.0f}s] {m}", flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

d = load()
L = labels()
N = L["N"]
pre, post, w, sign = d["pre"], d["post"], d["weight"].astype(np.float32), d["sign"]
SL = S.Labels(N, sign=sign)
module = SL.module()
isAL = L["reg"] == "AL"
exc = sign[pre] > 0
c_lin = float(S.node_scale(post, w, w, N, "raw", "renorm")[0])


def name_of(spec):
    return spec.replace(":", "")


def gpu_op(vals, transpose=False):
    """LinearOperator x -> M x (or M^T x) with M[post, pre] = vals on the excitatory edges, on the GPU in float64."""
    r, c = (pre[exc], post[exc]) if transpose else (post[exc], pre[exc])
    idx = torch.tensor(np.vstack([r, c]), device=dev)
    A = torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float64, device=dev), (N, N)).coalesce().to_sparse_csr()

    def mv(x):
        xt = torch.from_numpy(np.ascontiguousarray(np.asarray(x, np.float64).ravel())).to(dev)
        return (A @ xt[:, None])[:, 0].cpu().numpy()
    return sla.LinearOperator((N, N), matvec=mv, dtype=np.float64), A


def eigs(op, k, v0=None, tol=1e-9):
    vals, vecs = sla.eigs(op, k=k, which="LR", tol=tol, maxiter=200000, ncv=max(2 * k + 1, 32), v0=v0)
    o = np.argsort(-vals.real)
    vals, vecs = vals[o], vecs[:, o]
    P = np.abs(vecs) ** 2
    P /= P.sum(0, keepdims=True)
    return vals, vecs, P


def g_lin_exact(base, lam1):
    """Bisection on lambda_PF((1 - exp(-tanh(g M) dt))/dt) = alpha_eff (entrywise on the nonzeros of M)."""
    lo, hi = ALPHA_EFF / lam1 * 0.999, ALPHA_EFF / lam1 * 3
    v0 = None
    for _ in range(24):
        mid = np.sqrt(lo * hi)
        op, A = gpu_op((1 - np.exp(-np.tanh(mid * base) * DT)) / DT)
        val, vec = sla.eigs(op, k=1, which="LR", tol=1e-11, maxiter=200000, ncv=32, v0=v0)
        v0 = np.abs(vec[:, 0].real) + 1e-12
        del A
        lo, hi = (mid, hi) if val[0].real < ALPHA_EFF else (lo, mid)
    return float(np.sqrt(lo * hi))


for spec in sys.argv[1:]:
    nm = name_of(spec)
    wT = S.wtransform(w, spec)
    c_T = float(S.node_scale(post, w, wT, N, "raw", "renorm")[0])
    base = (wT[exc].astype(np.float64) * c_T)
    op, A = gpu_op(base)
    vals, vecs, P = eigs(op, K_MODES)
    del A
    modes = []
    for j in range(K_MODES):
        m = describe_mode(P[:, j], L)
        m["module_share"] = float(P[module, j].sum()); m["AL_class_share"] = float(P[isAL, j].sum())
        m["lam"] = float(vals[j].real); m["lam_im"] = float(vals[j].imag)
        modes.append(m)
    lam1 = float(vals[0].real)
    res = {"transform": spec, "name": nm, "c_T": c_T, "c_lin": c_lin, "c_ratio_T_over_lin": c_T / c_lin,
           "lam": vals.real.tolist(), "lam_clin": (vals.real * c_lin / c_T).tolist(), "alpha_eff": ALPHA_EFF,
           "g_lin_naive": ALPHA_EFF / lam1, "modes": modes,
           "max_edge_input_per_g": float(base.max()), "edge_input_q99_per_g": float(np.quantile(base, 0.99))}
    res["first_AL_mode"] = next(({"j": j, "lam": m["lam"], "module_share": m["module_share"]}
                                 for j, m in enumerate(modes) if m["AL_class_share"] >= 0.5), None)
    res["first_nonAL_mode"] = next(({"j": j, "lam": m["lam"], "lead_region": max(m["region"], key=m["region"].get),
                                     "types": list(m["types"])[:6], "inv_ipr": m["inv_ipr"]}
                                    for j, m in enumerate(modes) if m["AL_class_share"] < 0.5), None)
    if res["first_AL_mode"] and res["first_nonAL_mode"]:
        res["AL_over_nonAL"] = res["first_AL_mode"]["lam"] / res["first_nonAL_mode"]["lam"]
    log(f"{spec}: c_T {c_T:.4g} (lin {c_lin:.4g}) lam {np.round(vals.real[:6], 4)} clin {np.round(res['lam_clin'][:3], 4)} "
        f"mode0 1/IPR {modes[0]['inv_ipr']:.0f} module {modes[0]['module_share']:.3f} AL {modes[0]['AL_class_share']:.3f} "
        f"region {list(modes[0]['region'].items())[:3]}")
    res["g_lin"] = g_lin_exact(base, lam1)
    log(f"{spec}: g_lin {res['g_lin']:.5f} (naive {res['g_lin_naive']:.5f})")
    opT, AT = gpu_op(base, transpose=True)
    lv, lvec, LP = eigs(opT, 2)
    del AT
    res["left_lam"] = lv.real.tolist()
    res["left_mode0"] = describe_mode(LP[:, 0], L)
    res["left_mode0"]["module_share"] = float(LP[module, 0].sum()); res["left_mode0"]["AL_class_share"] = float(LP[isAL, 0].sum())
    json.dump(res, open(f"{OUT}/{nm}.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/{nm}_P.npz", P=P[:, :3].astype(np.float32), Pleft=LP[:, :1].astype(np.float32),
                        lam=vals.real[:3])
    log(f"saved {nm}")
