"""pd3 theory audit, part 1: de Candia-style module quantities for the MaleCNS connectome (CPU, ~minutes).

Computes, from the cache (read-only) and existing results (read-only):
 A. The antennal-lobe (AL) module (526 neurons = union of the 99% supports of the two leading excitatory modes, as in
    module_sim.py) split by soma side (L / R / midline). Per-side Perron roots, two-mode Galerkin couplings
    (Perron-weighted: M_kl = <u^k, E_kl v^l>, <u^k, v^k> = 1) and uniform block couplings (mean input to a neuron of k from a
    fully active l), excitatory and inhibitory, and the inhibition levels r* where de Candia's imbalances change sign:
    w0_k(g, r) = g (E_kk - r I_kk), w1_kl(g, r) = g (E_kl - r I_kl)  ->  r0* = E_kk / I_kk, r1* = E_kl / I_kl.
 B. The AL vs the rest of the brain (uniform and Perron-weighted couplings in both directions, r1*).
 C. Two-mode amplitude (normal-form) equations of the exact process for the L and R modes, with the factorised pair closure of
    theory.html section 4 (coincidence terms incl. inhibition), and the competition ratio
    Gamma = a_LR a_RL / (a_LL a_RR) (> 1: winner-take-all, i.e. a true broken-symmetry B1 state; < 1: coexistence).
    Also the module's single-mode K_eff (check against the whole-brain dense rows).
 D. Neuron-level QMF on the module alone from three initial conditions (L only, R only, both active) on a (g, r) grid:
    which de Candia states (SL / B1(L) / B1(R) / SH) are attractors, and is there coexistence (hysteresis)?
    Also the 2-variable coarse-grained (Perron-weighted) mean field on the same grid.
 E. The spectral islands (top-400 tail) as a heterogeneous multi-module system: weak-synapse thresholds g_k = a_eff/lambda_k.
Writes results/pd3/theory_module.json.
"""
import json, time
import numpy as np
import scipy.sparse as sp

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:.0f}s]", *a, flush=True)
dt = 0.2; a_eff = (1 - np.exp(-dt)) / dt
fdt = lambda x: (1 - np.exp(-np.tanh(np.clip(x, 0, None)) * dt)) / dt     # discrete-time rate, as theory_branch_gpu.py

pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
sides = np.load(C + "/sides.npy", allow_pickle=True)
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
exc, inh = sign[pre] > 0, sign[pre] < 0
E = sp.csr_matrix((c * w[exc], (post[exc], pre[exc])), shape=(N, N)); E.sum_duplicates()
I = sp.csr_matrix((c * w[inh], (post[inh], pre[inh])), shape=(N, N)); I.sum_duplicates()
tail = np.load(f"{ROOT}/results/pd/tail_real_r0_dir.npz", allow_pickle=True)
Mm = np.zeros(N, bool)
for k in range(2):
    p = tail["P"][:, k].astype(float); o = np.argsort(-p); Mm[o[: np.searchsorted(np.cumsum(p[o]) / p.sum(), 0.99) + 1]] = True
mg = np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)
# the saved module (module_sim.py) is the definition used everywhere; recomputing the 99% supports can swap one neuron at a
# float16 tie on the boundary (2 neurons with identical |v|^2), so take the saved index set
n_diff = int(len(set(np.where(Mm)[0]) ^ set(mg["idx"])))
Mm = np.zeros(N, bool); Mm[mg["idx"]] = True
idx = np.where(Mm)[0]; n = len(idx)
EM = E[idx][:, idx].toarray(); IM = I[idx][:, idx].toarray()
sd = sides[idx]; L = sd == "L"; R = sd == "R"
out = {"c": c, "alpha_eff": a_eff, "n_module": int(n), "recomputed_support_symdiff": n_diff, "n_L": int(L.sum()), "n_R": int(R.sum()), "n_mid": int((sd == "M").sum()),
       "module_signs": {str(k): int(v) for k, v in zip(*np.unique(sign[idx], return_counts=True))}}
for k in range(2):
    p = tail["P"][:, k].astype(float)
    out[f"mode{k}"] = {"lambda": float(tail["lam"][k]), "v2_share_L": float(p[sides == "L"].sum() / p.sum()),
                       "v2_share_R": float(p[sides == "R"].sum() / p.sum())}
log("module", out["n_module"], "L", out["n_L"], "R", out["n_R"], out["mode0"], out["mode1"])


def perron(A):
    ev, V = np.linalg.eig(A); k = np.argmax(ev.real); v = np.abs(V[:, k].real)
    evl, U = np.linalg.eig(A.T); kl = np.argmax(evl.real); u = np.abs(U[:, kl].real)
    return float(ev[k].real), v, u / (u @ v)


# ---------------- A. AL pair L/R ----------------
evM = np.sort(np.linalg.eigvals(EM).real)[::-1]
out["module_E_top_eigs"] = evM[:6].tolist()
S = {"L": L, "R": R}
Vfull, Ufull, Lam = {}, {}, {}
for k, m in S.items():
    lam, v, u = perron(EM[np.ix_(m, m)]); V = np.zeros(n); V[m] = v; U = np.zeros(n); U[m] = u
    Vfull[k], Ufull[k], Lam[k] = V, U, lam
keys = ["L", "R"]
PW_E = [[float(Ufull[a] @ EM @ Vfull[b]) for b in keys] for a in keys]
PW_I = [[float(Ufull[a] @ IM @ Vfull[b]) for b in keys] for a in keys]
UN_E = [[float(EM[np.ix_(S[a], S[b])].sum(1).mean()) for b in keys] for a in keys]
UN_I = [[float(IM[np.ix_(S[a], S[b])].sum(1).mean()) for b in keys] for a in keys]
ev2 = np.sort(np.linalg.eigvals(np.array(PW_E)).real)[::-1]
A = {"Lambda_L": Lam["L"], "Lambda_R": Lam["R"], "PW_E": PW_E, "PW_I": PW_I, "UN_E": UN_E, "UN_I": UN_I,
     "PW_E_eigs": ev2.tolist(),
     "coupling_over_half_detuning": float(np.sqrt(PW_E[0][1] * PW_E[1][0]) / (abs(Lam["R"] - Lam["L"]) / 2)),
     "r0_star_PW": {k: PW_E[i][i] / PW_I[i][i] for i, k in enumerate(keys)},
     "r1_star_PW": {"L<-R": PW_E[0][1] / PW_I[0][1], "R<-L": PW_E[1][0] / PW_I[1][0]},
     "r0_star_UN": {k: UN_E[i][i] / UN_I[i][i] for i, k in enumerate(keys)},
     "r1_star_UN": {"L<-R": UN_E[0][1] / UN_I[0][1], "R<-L": UN_E[1][0] / UN_I[1][0]},
     "g_threshold_PW_r0": {k: a_eff / Lam[k] for k in keys}, "g_threshold_UN_r0": {k: a_eff / UN_E[i][i] for i, k in enumerate(keys)},
     "in_strength_share_from_other_side": {
         "exc_L": float(EM[np.ix_(L, R)].sum() / EM[L].sum()), "exc_R": float(EM[np.ix_(R, L)].sum() / EM[R].sum()),
         "inh_L": float(IM[np.ix_(L, R)].sum() / IM[L].sum()), "inh_R": float(IM[np.ix_(R, L)].sum() / IM[R].sum())}}
out["A_AL_pair"] = A
log("A", json.dumps(A)[:900])

# ---------------- B. AL vs rest of brain ----------------
Bm = ~Mm
e_BM = np.asarray(E[Bm][:, Mm].sum(1)).ravel(); i_BM = np.asarray(I[Bm][:, Mm].sum(1)).ravel()
e_MB = np.asarray(E[Mm][:, Bm].sum(1)).ravel(); i_MB = np.asarray(I[Mm][:, Bm].sum(1)).ravel()
lamM, vM, uM = perron(EM)
recv = (e_BM + i_BM) > 0
Bq = {"uniform_exc_into_rest_from_full_AL": float(e_BM.mean()), "uniform_inh_into_rest_from_full_AL": float(i_BM.mean()),
      "n_rest_receiving_AL": int(recv.sum()),
      "exc_into_receivers": float(e_BM[recv].mean()), "inh_into_receivers": float(i_BM[recv].mean()),
      "r1_star_rest<-AL_uniform": float(e_BM.sum() / i_BM.sum()),
      "frac_receivers_net_inhibited_at_r": {f"{r:g}": float(((e_BM - r * i_BM)[recv] < 0).mean()) for r in [0.5, 1, 2, 4, 6]},
      "PW_exc_into_AL_from_full_rest": float(vM @ e_MB / vM.sum()), "PW_inh_into_AL_from_full_rest": float(vM @ i_MB / vM.sum()),
      "r1_star_AL<-rest_PW": float((vM @ e_MB) / (vM @ i_MB)), "module_Lambda": lamM}
out["B_AL_rest"] = Bq
log("B", Bq)

# ---------------- C. two-mode normal form with coincidences ----------------
# in-edge lists within the module (signed weights at r=1 for inhibitory: -I; scaled by r later)
Ein = [np.nonzero(EM[i])[0] for i in range(n)]; Iin = [np.nonzero(IM[i])[0] for i in range(n)]


def normal_form(g, rs):
    Bg = fdt(g * EM)                                   # branching operator on the module (exc only)
    res = []
    vec = {}
    for k, m in S.items():
        lam, v, u = perron(Bg[np.ix_(m, m)]); V = np.zeros(n); V[m] = v; U = np.zeros(n); U[m] = u
        s = V.sum(); V /= s; U *= s                     # normalise sum v = 1 (A_k = active-neuron count / ... scale-free)
        vec[k] = (lam, V, U)
    lam0, v0, u0 = perron(Bg)                           # single leading mode of the whole module
    for r in rs:
        # coincidence bilinear forms C^k(x, y) for x, y in {vL, vR}, and for the single mode
        Ck = {(k, a, b): 0.0 for k in keys for a in keys for b in keys}; D1 = 0.0
        for i in range(n):
            je, ji = Ein[i], Iin[i]
            if len(je) == 0:
                continue
            xe = EM[i, je]; fe = fdt(g * xe)
            # exc-exc pairs (concavity), exc-inh pairs (coincident inhibition)
            dee = np.abs(fdt(g * (xe[:, None] + xe[None, :])) - fe[:, None] - fe[None, :]); np.fill_diagonal(dee, 0.0)
            if len(ji) and r > 0:
                yi = IM[i, ji]; dei = np.abs(fdt(g * xe[:, None] - g * r * yi[None, :]) - fe[:, None])
            else:
                dei = None
            # sum_{pairs} |Delta| (x_j y_m + x_m y_j)/2 ; ee: 0.5 * sum over ordered pairs; ei: each unordered pair once
            def form(x, y):
                t = 0.5 * 0.5 * (x[je] @ dee @ y[je] + y[je] @ dee @ x[je])
                if dei is not None:
                    t += 0.5 * (x[je] @ dei @ y[ji] + y[je] @ dei @ x[ji])
                return t
            fab = {(a, b): form(vec[a][1], vec[b][1]) for a in keys for b in keys}
            for k in keys:
                uk = vec[k][2][i]
                if uk == 0:
                    continue
                for a in keys:
                    for b in keys:
                        Ck[(k, a, b)] += uk * fab[(a, b)]
            if u0[i] > 0:
                D1 += u0[i] * form(v0, v0)
        # (1 - rho_i) term: -rho_i (B rho)_i projected: c^k_{ab} += <u^k, v^a * (B v^b)> (v^a nonzero only on its side)
        coef = {}
        for k in keys:
            for a in keys:
                for b in keys:
                    coef[(k, a, b)] = float(vec[k][2] @ (vec[a][1] * (Bg @ vec[b][1]))) + Ck[(k, a, b)]
        lin = [[float(vec[k][2] @ Bg @ vec[l][1]) for l in keys] for k in keys]
        aLL, aRR = coef[("L", "L", "L")], coef[("R", "R", "R")]
        aLR = coef[("L", "L", "R")] + coef[("L", "R", "L")]; aRL = coef[("R", "R", "L")] + coef[("R", "L", "R")]
        K_single = 1 / (float((u0 ** 2) @ v0) * (float(u0 @ (v0 ** 2)) + D1 / lam0))
        K_nc = 1 / (float((u0 ** 2) @ v0) * float(u0 @ (v0 ** 2)))
        res.append({"g": g, "r": r, "lam_module": lam0, "R_module": lam0 / a_eff, "lin_2x2": lin,
                    "a_LL": aLL, "a_RR": aRR, "a_LR": aLR, "a_RL": aRL, "Gamma": aLR * aRL / (aLL * aRR),
                    "cross_over_self_L": aLR / aLL, "cross_over_self_R": aRL / aRR,
                    "D_single": D1, "K_eff_module": K_single, "K_nocoinc_module": K_nc,
                    "uv2": float(u0 @ (v0 ** 2)), "u2v": float((u0 ** 2) @ v0),
                    "sbar_sum_v": float((u0 ** 2) @ v0) * float(v0.sum()), "sum_v": float(v0.sum()),
                    "PR_v": float(v0.sum() ** 2 / (v0 ** 2).sum())})
    return res


g_lin = 0.16315274988719286   # results/pd/branch_real_dense_part*.json
rows = []
for g in [g_lin * 1.0001, 0.2679, 0.35, 0.5, 0.8]:
    rows += normal_form(g, [0.0, 0.5, 1.0, 2.0, 4.0, 6.0])
    log("C g", g, " ".join(f"r{x['r']:g}:G{x['Gamma']:.3f},K{x['K_eff_module']:.1f}" for x in rows[-6:]))
out["C_normal_form"] = rows

# ---------------- D. neuron-level QMF on the module from asymmetric starts ----------------
G = np.geomspace(0.15, 3.0, 41); RS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
inits = {"L_only": L.astype(float), "R_only": R.astype(float), "both": np.ones(n)}
decay = np.exp(-dt); D = {"g": G.tolist(), "r": RS, "XL": {}, "XR": {}}
for r in RS:
    W = EM - r * IM
    X = np.concatenate([np.repeat(x[:, None], len(G), 1) for x in inits.values()], 1)
    gv = np.tile(G, len(inits))
    for st in range(15000):
        p_on = 1 - np.exp(-np.tanh(np.clip((W @ X) * gv, 0, None)) * dt)
        Xn = X * decay + (1 - X) * p_on
        if st % 500 == 0 and np.abs(Xn - X).max() < 1e-12:
            X = Xn; break
        X = Xn
    for j, nm in enumerate(inits):
        Xi = X[:, j * len(G):(j + 1) * len(G)]
        D["XL"][f"{r:g}_{nm}"] = Xi[L].mean(0).tolist(); D["XR"][f"{r:g}_{nm}"] = Xi[R].mean(0).tolist()
    log("D r", r, "steps", st)
out["D_module_qmf"] = D

# coarse-grained 2-variable mean field (Perron-weighted couplings, one population per side, same discrete map)
PWE, PWI = np.array(PW_E), np.array(PW_I)
D2 = {"g": G.tolist(), "r": RS, "X": {}}
for r in RS:
    W2 = PWE - r * PWI
    for nm, x0 in {"L_only": [1.0, 0.0], "R_only": [0.0, 1.0], "both": [1.0, 1.0]}.items():
        X = np.repeat(np.array(x0)[:, None], len(G), 1)
        for st in range(15000):
            X = X * decay + (1 - X) * (1 - np.exp(-np.tanh(np.clip((W2 @ X) * G, 0, None)) * dt))
        D2["X"][f"{r:g}_{nm}"] = X.T.tolist()
out["D_2module_mf"] = D2

# ---------------- E. spectral islands ----------------
lam = tail["lam"]; ipr = tail["ipr"]
out["E_islands"] = {"lambda": lam[:30].tolist(), "inv_ipr": (1 / ipr[:30]).tolist(), "AL_share": tail["AL"][:30].tolist(),
                    "g_k_weak_synapse": (a_eff / lam[:30]).tolist(),
                    "n_islands_with_g_k_below": {str(x): int((a_eff / lam < x).sum()) for x in [0.3, 0.5, 1.0, 1.47, 2.0, 3.0]},
                    "lambda_400": float(lam[-1])}
json.dump(out, open(f"{ROOT}/results/pd3/theory_module.json", "w"), indent=1)
log("saved results/pd3/theory_module.json")
