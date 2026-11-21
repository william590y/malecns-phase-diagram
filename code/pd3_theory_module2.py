"""pd3 theory audit, part 1b (CPU, ~1 min): de Candia's coarse-grained 2-module mean field for the AL pair, done properly,
and why the neuron-level mean field has no broken-symmetry state.
 1. Perron-weighted couplings with v^k normalised to MEAN 1 over its side (so X_k = 1 means the side's Perron pattern at
    mean activity 1, the analogue of de Candia's 'all neurons of module k active'); u^k with <u^k, v^k> = 1.
 2. The 2-variable map X_k <- X_k e^{-dt} + (1 - X_k)(1 - exp(-tanh(g sum_l W_kl X_l)_+ dt)), W = E_PW - r I_PW,
    on a (g, r) grid. Attractors from 4 starts, each with a ONE-NEURON perturbation (1/n_k) of every side that starts silent
    (so the invariant-subspace artefact of rectified maps is removed, but the dead zone, if wider than one neuron, survives):
    (1, eps), (eps, 1), (1, 1), (eps, eps). States: SL / B1L / B1R / SH. Also de Candia's analytic lines:
    SL edge g = a_eff / max eig_+(W) (rectified 2x2 Perron root), w1 sign change r1*, B1_k existence g (W_kk) > a_eff.
 3. Neuron-level dead zone: for every neuron i of side k with v^k_i > 0, the sign of its net input from the other side l in
    l's Perron pattern, sum_j (E_ij - r I_ij) v^l_j; share of such neurons (and u^k-weighted share) with net-POSITIVE input,
    i.e. neurons that the other side keeps seeding whatever r says on average.
Writes results/pd3/theory_module2.json.
"""
import json, time
import numpy as np
import scipy.sparse as sp

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
t0 = time.time()
dt = 0.2; a_eff = (1 - np.exp(-dt)) / dt
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
sides = np.load(C + "/sides.npy", allow_pickle=True)
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
mg = np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True); idx = mg["idx"]; n = len(idx)
loc = -np.ones(N, np.int64); loc[idx] = np.arange(n)
m = (loc[pre] >= 0) & (loc[post] >= 0)
pi, po, ww, sg = loc[pre[m]], loc[post[m]], w[m], sign[pre[m]]
EM = sp.csr_matrix((c * ww * (sg > 0), (po, pi)), shape=(n, n)).toarray()
IM = sp.csr_matrix((c * ww * (sg < 0), (po, pi)), shape=(n, n)).toarray()
sd = sides[idx]; S = {"L": sd == "L", "R": sd == "R"}; keys = ["L", "R"]


def perron(A):
    ev, V = np.linalg.eig(A); k = np.argmax(ev.real); v = np.abs(V[:, k].real)
    evl, U = np.linalg.eig(A.T); kl = np.argmax(evl.real); u = np.abs(U[:, kl].real)
    return float(ev[k].real), v, u


V, U = {}, {}
for k in keys:
    lam, v, u = perron(EM[np.ix_(S[k], S[k])])
    v = v / v.mean(); u = u / (u @ v)
    V[k] = np.zeros(n); V[k][S[k]] = v; U[k] = np.zeros(n); U[k][S[k]] = u
PWE = np.array([[U[a] @ EM @ V[b] for b in keys] for a in keys]); PWI = np.array([[U[a] @ IM @ V[b] for b in keys] for a in keys])
out = {"PW_E_mean1": PWE.tolist(), "PW_I_mean1": PWI.tolist(), "r1_star": [PWE[0, 1] / PWI[0, 1], PWE[1, 0] / PWI[1, 0]],
       "r_where_L_overtakes_R_netinput": float((PWE[1, 1] - PWE[0, 0]) / (PWI[1, 1] - PWI[0, 0]))}

# 2. coarse-grained map
G = np.geomspace(0.15, 3.0, 61); RS = np.round(np.arange(0, 6.01, 0.25), 3)
eps = {"L": 1 / S["L"].sum(), "R": 1 / S["R"].sum()}
starts = {"L_on": [1, eps["R"]], "R_on": [eps["L"], 1], "both": [1, 1], "seed": [eps["L"], eps["R"]]}
decay = np.exp(-dt)
phase = {}
for r in RS:
    W2 = PWE - r * PWI
    finals = {}
    for nm, x0 in starts.items():
        X = np.repeat(np.array(x0, float)[:, None], len(G), 1)
        for _ in range(30000):
            X = X * decay + (1 - X) * (1 - np.exp(-np.tanh(np.clip((W2 @ X) * G, 0, None)) * dt))
        finals[nm] = X
    lab = []
    for j in range(len(G)):
        s = set()
        for X in finals.values():
            xl, xr = X[:, j]
            s.add("SL" if max(xl, xr) < 1e-6 else "SH" if min(xl, xr) > 1e-3 else ("B1L" if xl > xr else "B1R"))
        lab.append("+".join(sorted(s)))
    # analytic: rectified 2x2 Perron root (max over supports {L},{R},{L,R} consistent with rectification)
    cand = [W2[0, 0], W2[1, 1]]
    ev, vec = np.linalg.eig(W2); kk = np.argmax(ev.real); vv = vec[:, kk].real
    if (vv > 0).all() or (vv < 0).all():
        cand.append(ev[kk].real)
    mu = max(cand)
    seg = []; prev = None
    for j, l in enumerate(lab):
        if l != prev:
            seg.append([float(G[j]), l]); prev = l
    phase[f"{r:g}"] = {"segments": seg, "labels": lab, "g_SL_edge": a_eff / mu, "w1_LR": float(W2[0, 1]), "w1_RL": float(W2[1, 0]),
                       "g_B1L_exists": a_eff / W2[0, 0] if W2[0, 0] > 0 else None,
                       "g_B1R_exists": a_eff / W2[1, 1] if W2[1, 1] > 0 else None}
    print(f"[{time.time()-t0:.0f}s] r={r:g} SL-edge {a_eff/mu:.4f} w1 {W2[0,1]:.3f},{W2[1,0]:.3f} :: " +
          "  ".join(f"{a:.3f}:{b}" for a, b in seg), flush=True)
out["coarse_2module"] = {"g": G.tolist(), "r": RS.tolist(), "eps": eps, "phase": phase}

# 3. neuron-level dead zone
dz = {}
for r in [0, 1, 2, 2.5, 3, 4, 6]:
    row = {}
    for k, l in [("R", "L"), ("L", "R")]:
        tgt = S[k] & (V[k] > 0)
        net = (EM - r * IM)[tgt] @ V[l]
        row[f"{k}<-{l}"] = {"n_target": int(tgt.sum()), "share_net_positive": float((net > 0).mean()),
                            "u_weighted_share_net_positive": float(U[k][tgt] @ (net > 0) / U[k][tgt].sum()),
                            "share_net_negative": float((net < 0).mean()), "share_zero": float((net == 0).mean())}
    dz[f"{r:g}"] = row
    print(r, row, flush=True)
out["neuron_dead_zone"] = dz
json.dump(out, open(f"{ROOT}/results/pd3/theory_module2.json", "w"), indent=1)
print("saved", flush=True)
