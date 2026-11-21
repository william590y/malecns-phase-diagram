"""m2_stoch_modes.py -- per-mode depression renormalisation (lane m2/theory-stochastic, method M-A). CPU (scipy eigs).

Top-K right and left eigenpairs of the excitatory operator E = c w (ACh presynaptic, raw normalisation), matched by
eigenvalue. For each real mode: |v|, |u| (island approximation; sign purity reported), K_0 = 1/(<u^2,v><u,v^2>) with
<u, v> = 1, AL share, top superclass, PR. Per mode and g: lambda_k(g) = Rayleigh quotient of f(gE) on (|u|, |v|) and the
single-mode chains:
  M1  : logistic chain, K = K_0 (PF) and K = c_r K_0 (CAL, M1's K_needed / K_noD at r = 1 and r = 0);
  M2  : QS level n* from lambda_k X(n) (1 - n/K) = alpha with the exact per-neuron efficacy X(n); frozen / adiabatic /
        2-D lifetimes (m2_stoch_core.lifetimes).
Output results/m2/theory/stochastic/modes/modes.json (+ npz with the vectors' summary).
"""
import json, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
sys.path.insert(0, "/MALECNS_ROOT/code")
import m2_stoch_core as S
from m2_stoch_run import m1_factors

K = int(sys.argv[1]) if len(sys.argv) > 1 else 80
OUT = f"{S.ROOT}/results/m2/theory/stochastic/modes"
import os
os.makedirs(OUT, exist_ok=True)
C = S.CACHE
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
exc = sign[pre] > 0
E = sp.csr_matrix((c * w[exc], (post[exc], pre[exc])), shape=(N, N)); E.sum_duplicates()
sup = np.load(C + "/superclasses.npy", allow_pickle=True).astype(str)
typ = np.load(C + "/types.npy", allow_pickle=True).astype(str)
mod = np.load(f"{S.ROOT}/results/module/module_graph.npz", allow_pickle=True)["idx"]
ALm = np.zeros(N, bool); ALm[mod] = True
S.log("eigs right")
lr, vr = sla.eigs(E, k=K, which="LR", ncv=max(2 * K + 20, 120), tol=1e-9, maxiter=200000)
S.log("eigs left")
ll, vl = sla.eigs(E.T.tocsr(), k=K, which="LR", ncv=max(2 * K + 20, 120), tol=1e-9, maxiter=200000)
S.log("done eigs")
o = np.argsort(-lr.real); lr, vr = lr[o], vr[:, o]
o = np.argsort(-ll.real); ll, vl = ll[o], vl[:, o]
U, TAU = 0.25, 157.0
Ecoo = E.tocoo(); Er, Ec, Ed = Ecoo.row, Ecoo.col, Ecoo.data
rows = m1_factors()
cr1 = rows[1.0]["K_needed"] / rows[1.0]["K_noD"]; cr0 = rows[0.0]["K_needed"] / rows[0.0]["K_noD"]
G = np.geomspace(0.15, 40, 56)
modes = []
used = set()
for i in range(K):
    lam = lr[i]
    if abs(lam.imag) > 1e-6 * abs(lam):
        modes.append({"k": i, "lam": float(lam.real), "lam_im": float(lam.imag), "complex": True}); continue
    j = int(np.argmin(np.abs(ll - lam) + np.array([1e9 if x in used else 0 for x in range(K)])))
    used.add(j)
    v = vr[:, i].real; u = vl[:, j].real
    purity_v = abs(v.sum()) / np.abs(v).sum(); purity_u = abs(u.sum()) / np.abs(u).sum()
    v = np.abs(v); u = np.abs(u)
    v /= np.linalg.norm(v); u /= (u * v).sum()
    u2v = (u * u * v).sum(); uv2 = (u * v * v).sum(); K0 = 1 / (u2v * uv2)
    vv = v * v
    row = {"k": i, "lam": float(lam.real), "lam_left_match": float(ll[j].real), "complex": False,
           "purity_v": float(purity_v), "purity_u": float(purity_u), "K0": float(K0),
           "PR_v": float(1 / (vv ** 2).sum()), "AL_v": float(vv[ALm].sum()), "g_lin_naive": float(S.A_EFF / lam.real),
           "top_superclass": str(sup[np.argmax(vv)]), "top_type": str(typ[np.argmax(vv)]),
           "kappa_dep_K": float(K0 / (1 + S.kappa(U, TAU)))}
    # Rayleigh lambda_k(g) with the saturating kernel
    m = (u[Er] > 1e-9 * u.max()) & (v[Ec] > 1e-9 * v.max())
    rr, cc, ee = Er[m], Ec[m], Ed[m]
    lamg = np.array([(u[rr] * S.f_branch(g * ee) * v[cc]).sum() for g in G])
    row["g_lin_k"] = float(np.interp(S.A_EFF, lamg, G)) if lamg[-1] > S.A_EFF else None
    dep = sign > 0
    uvw = u * v
    rr_ = np.geomspace(1e-8, 0.5, 600); er = S.eff_spike(rr_, U, TAU)

    sel = np.where(uvw > 1e-10 * uvw.max())[0]
    vs, uvs, deps = v[sel], uvw[sel], dep[sel]
    uvsum = uvw.sum()

    def Xn(n):
        n = np.atleast_1d(n)
        out = np.empty(len(n))
        for a0 in range(0, len(n), 64):
            rho = np.clip(n[a0:a0 + 64, None] * u2v * vs[None, :], 0, 0.5)
            e = np.where(rho > 1e-8, np.interp(np.log(np.maximum(rho, 1e-8)), np.log(rr_), er), 1.0)
            e = np.where(deps[None, :], e, 1.0)
            out[a0:a0 + 64] = ((e * uvs[None, :]).sum(1) + (uvsum - uvs.sum())) / uvsum
        return out
    out = {}
    for name, Kx in (("PF", K0), ("CAL1", cr1 * K0)):
        ltM1, ltF, ltA, lt2 = [], [], [], []
        for lg in lamg:
            ltM1.append(S.log_tau_smooth(lambda n: lg * np.ones_like(n), Kx) if lg > 0 else -np.inf)
            if lg <= S.A_EFF:
                ltF.append(-np.inf); ltA.append(-np.inf); lt2.append(-np.inf); continue
            Ff = lambda n: lg * Xn(np.array([n]))[0] * (1 - n / Kx) - S.A_EFF
            lo, hi = 1e-6, Kx * (1 - 1e-9)
            for _ in range(60):                   # F is decreasing in n: bisection for the QS level
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if Ff(mid) > 0 else (lo, mid)
            ns = 0.5 * (lo + hi)
            X = float(Xn(np.array([ns]))[0])
            lt = S.lifetimes({"lam": lg * X, "lam_rest": lg, "X": X, "n_star": ns, "K": Kx}, U, TAU, do2d=True, cap2d=400000)
            ltF.append(lt["lt_F"]); ltA.append(lt["lt_A"]); lt2.append(lt["lt_2D"])
        for nm, arr in (("M1", ltM1), ("M2F", ltF), ("M2A", ltA), ("M2_2D", lt2)):
            arr = np.where(np.isnan(arr), -np.inf, np.array(arr, float))
            out[f"{name}_{nm}_S"] = S.cross(G, arr, 200.0); out[f"{name}_{nm}_on"] = S.cross(G, arr, 3000.0)
    row.update(out)
    modes.append(row)
    S.log(f"mode {i} lam {lam.real:.3f} K0 {K0:.1f} PR {row['PR_v']:.0f} AL {row['AL_v']:.2f} {row['top_superclass']} "
          f"pur {purity_v:.2f} | M1 on {out['PF_M1_on']} | M2 2D on {out['PF_M2_2D_on']} F {out['PF_M2F_on']} A {out['PF_M2A_on']}")
json.dump({"K": K, "c_r1": cr1, "c_r0": cr0, "kappa": S.kappa(U, TAU), "G": G.tolist(), "modes": modes},
          open(f"{OUT}/modes.json", "w"), default=float)
S.log("saved")
