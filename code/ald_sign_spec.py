"""E2 (transmitter signs): leading EXCITATORY eigenmodes of the whole-CNS operator under sign changes (CPU, ARPACK).

Operator (same as al_common.exc_operator and the simulation at linear order): M[post, pre] = c * w for presynaptic neurons
with model sign +1 (after the change); c = 1/742.7 = 1/(mean in-strength over ALL edges with input), unchanged by signs
(ald_sim's node_scale does not depend on signs). r does not enter M (inhibition acts only beyond linear order), so
conditions (i) lLN1_bc -> -r and (ii) lLN1_bc -> 0 share one operator ("lLN1bc_off").
For each operator: top-K eigenvalues (real part), localization of the top modes (1/IPR, n50/n90, region / category / type
composition via al_common.describe_mode), share of |v|^2 in the 526-neuron AL module and in AL classes, the first mode
among the top K carried by the AL (AL share >= 0.5) and the first mode NOT carried by it, and the naive linear threshold
g_lin_naive = alpha_eff / lambda_1. For deterministic conditions also: the exact g_lin of the tanh-saturated branching
operator B(g) = (1 - exp(-tanh(g M) dt))/dt (entrywise; bisection to lambda_PF(B) = alpha_eff, as theory_branch_gpu.py),
the left Perron vector, and |v|^2 of the top 2 right modes (npz).
Ensemble draws are generated EXACTLY as ald_sim.py does (sign_probs + draw_signs with default_rng([ens_seed, 1])), so draw d
here is draw d of an ald_sim run with the same --sign_ensemble settings and --ens_seed (draws are prefix-stable in K).

Usage: ald_sign_spec.py det                         -> results/aldef/signs/spec/det.json + det_P.npz
       ald_sign_spec.py ens <unit> <gt> <seed> <K> <d0> <d1> [<nsave>]
                                                    -> spec/ens_<unit>_<gt>_s<seed>_<d0>-<d1>.json (+ _P.npz: leading |v|^2
                                                       of draws < nsave)
"""
import json, os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
from al_common import load, labels, describe_mode, ALPHA_EFF          # noqa: E402
import ald_sim as S                                                   # noqa: E402

OUT = f"{ROOT}/results/aldef/signs/spec"
os.makedirs(OUT, exist_ok=True)
DT = 0.2
K_MODES = 8
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:.0f}s] {m}", flush=True)

d = load()
L = labels()
N = L["N"]
pre, post, w, sign0 = d["pre"], d["post"], d["weight"], d["sign"]
ins = np.bincount(post, weights=w, minlength=N)
c = 1.0 / ins[ins > 0].mean()
SL = S.Labels(N, sign=sign0)
module = SL.module()
isAL = L["reg"] == "AL"
types = SL.get("type")


def operator(sgn):
    m = sgn[pre] > 0
    M = sp.csr_matrix((w[m] * c, (post[m], pre[m])), shape=(N, N))
    M.sum_duplicates()
    return M


def eigs(M, k=K_MODES, left=False):
    A = M.T.tocsr() if left else M
    vals, vecs = sla.eigs(A, k=k, which="LR", tol=1e-8, maxiter=200000, ncv=max(2 * k + 1, 24))
    o = np.argsort(-vals.real)
    vals, vecs = vals[o], vecs[:, o]
    P = np.abs(vecs) ** 2
    P /= P.sum(0, keepdims=True)
    return vals, vecs, P


def summarize(name, M, k=K_MODES):
    vals, vecs, P = eigs(M, k)
    modes = []
    for j in range(k):
        m = describe_mode(P[:, j], L)
        m["module_share"] = float(P[module, j].sum())
        m["AL_class_share"] = float(P[isAL, j].sum())
        m["lam"] = float(vals[j].real); m["lam_im"] = float(vals[j].imag)
        modes.append(m)
    res = {"name": name, "lam": vals.real.tolist(), "nnz": int(M.nnz), "g_lin_naive": float(ALPHA_EFF / vals[0].real),
           "modes": modes}
    res["first_AL_mode"] = next(({"j": j, "lam": m["lam"], "module_share": m["module_share"]}
                                 for j, m in enumerate(modes) if m["AL_class_share"] >= 0.5), None)
    res["first_nonAL_mode"] = next(({"j": j, "lam": m["lam"], "lead_region": max(m["region"], key=m["region"].get),
                                     "types": list(m["types"])[:5]} for j, m in enumerate(modes)
                                    if m["AL_class_share"] < 0.5), None)
    return res, vals, vecs, P


def g_lin_exact(M, lam1):
    """Bisection on lambda_PF((1 - exp(-tanh(g M) dt))/dt) = alpha_eff (entrywise on the nonzeros)."""
    base = M.data.copy()
    lo, hi = ALPHA_EFF / lam1 * 0.999, ALPHA_EFF / lam1 * 3
    v0 = None
    for _ in range(22):
        mid = np.sqrt(lo * hi)
        B = M.copy(); B.data = (1 - np.exp(-np.tanh(mid * base) * DT)) / DT
        val, vec = sla.eigs(B, k=1, which="LR", tol=1e-10, maxiter=200000, ncv=24, v0=v0)
        v0 = np.abs(vec[:, 0].real) + 1e-12
        lo, hi = (mid, hi) if val[0].real < ALPHA_EFF else (lo, mid)
    return float(np.sqrt(lo * hi))


def cond_signs():
    s = np.sign(sign0).astype(np.float64)
    lln1 = S.select("lLN1_bc", SL)
    lln2 = S.select("type:lLN2.*&sign:+1", SL)
    allLN = S.select("class:ALLN&sign:+1", SL)
    out = {"base": s.copy()}
    x = s.copy(); x[lln1] = -1; out["lLN1bc_off"] = x
    x = s.copy(); x[lln1 | lln2] = -1; out["lLN1bc_lLN2ach_off"] = x
    x = s.copy(); x[allLN] = -1; out["allALLN_ach_off"] = x
    info = {"lLN1_bc": int(lln1.sum()), "lLN2_ach": int(lln2.sum()), "lLN2_ach_types": sorted(set(types[lln2])),
            "ALLN_ach": int(allLN.sum())}
    return out, info


if sys.argv[1] == "det":
    conds, info = cond_signs()
    res = {"c": c, "alpha_eff": ALPHA_EFF, "selection": info, "conds": {}}
    Pout = {}
    for name, sg in conds.items():
        M = operator(sg)
        r, vals, vecs, P = summarize(name, M)
        r["g_lin"] = g_lin_exact(M, vals[0].real)
        lv, lvec, LP = eigs(M, k=2, left=True)
        r["left_mode0"] = describe_mode(LP[:, 0], L)
        r["left_mode0"]["module_share"] = float(LP[module, 0].sum())
        res["conds"][name] = r
        Pout[f"{name}_P"] = P[:, :2].astype(np.float32)
        Pout[f"{name}_Pleft"] = LP[:, :1].astype(np.float32)
        m0 = r["modes"][0]
        log(f"{name}: lam {np.round(r['lam'][:4], 3)} g_lin {r['g_lin']:.4f} (naive {r['g_lin_naive']:.4f}) 1/IPR "
            f"{m0['inv_ipr']:.0f} module {m0['module_share']:.3f} region {list(m0['region'].items())[:3]} "
            f"types {list(m0['types'])[:4]}")
    json.dump(res, open(f"{OUT}/det.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/det_P.npz", **Pout)
    log("saved det")
else:
    _, unit, gt, seed, K, d0, d1 = sys.argv[1:8]
    seed, K, d0, d1 = int(seed), int(K), int(d0), int(d1)
    nsave = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    tab = S.nt_table()
    Pr, q = S.sign_probs(tab, "body", gt, 0.5)
    draws = S.draw_signs(Pr, K, np.random.default_rng([seed, 1]), unit, types)
    lln1 = types == "lLN1_bc"
    tag = f"ens_{unit}_{gt}_s{seed}_{d0}-{d1}"
    rows, Psave = [], {}
    for dd in range(d0, min(d1, K)):
        sg = draws[dd].astype(np.float64)
        M = operator(sg)
        r, vals, vecs, P = summarize(f"draw{dd}", M, k=6)
        m0 = r["modes"][0]
        row = {"draw": dd, "lam": r["lam"], "g_lin_naive": r["g_lin_naive"], "nnz": r["nnz"],
               "n_changed": int((sg != np.sign(sign0)).sum()), "n_changed_module": int((sg[module] != np.sign(sign0[module])).sum()),
               "lLN1_bc_signs": {str(int(k)): int(v) for k, v in zip(*np.unique(sg[lln1], return_counts=True))},
               "n_exc": int((sg > 0).sum()), "n_inh": int((sg < 0).sum()),
               "mode0": {k: m0[k] for k in ("inv_ipr", "n50", "n90", "n99", "region", "cat", "types", "side",
                                            "module_share", "AL_class_share")},
               "first_AL_mode": r["first_AL_mode"], "first_nonAL_mode": r["first_nonAL_mode"],
               "modes_AL_share": [m["AL_class_share"] for m in r["modes"]],
               "modes_region": [max(m["region"], key=m["region"].get) for m in r["modes"]]}
        rows.append(row)
        if dd < nsave:
            Psave[f"draw{dd}"] = P[:, 0].astype(np.float32)
        log(f"{tag} draw {dd}: lLN1_bc {row['lLN1_bc_signs']} lam {np.round(r['lam'][:3], 3)} module {m0['module_share']:.3f} "
            f"region {list(m0['region'].items())[:2]}")
    json.dump({"unit": unit, "gt": gt, "seed": seed, "K": K, "rows": rows,
               "P_lLN1_bc": Pr[lln1].mean(0).tolist()}, open(f"{OUT}/{tag}.json", "w"), indent=1)
    if Psave:
        np.savez_compressed(f"{OUT}/{tag}_P.npz", **Psave)
    log(f"saved {tag}")
