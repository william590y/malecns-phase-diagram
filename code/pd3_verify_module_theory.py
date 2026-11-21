"""pd3 adversarial verification of program module_theory (results/pd3/module_theory.md). CPU only; does NOT import any
module_theory code. Recomputes headline numbers from the raw inputs/outputs with independent implementations:

 V1  nesting statistic of the top-400 excitatory modes (tail_real_r0_dir.npz P): share of non-AL modes (k >= 3) whose 90%
     |v|^2 support lies > 50% inside mode 2's support; mode-2 support composition; max 1/IPR.
 V2  isolated thresholds g_lin,k and K_k of AL-L, AL-R and the 526 AL module, from the connectome (dense eig + bisection).
 V3  P0 grouping (top 30 by Perron root of the excitatory block + rest) and the median equitability CVs.
 V4  closure Perron-weighted AL pair (E^PW, I^PW) from the connectome; closure SL edge g = alpha_eff / mu_+(E^PW - r I^PW)
     (r = 0: eigvals; r > 0: multi-start rectified iteration) + a direct check by iterating the closure map from 1e-9.
 V5  QMF silent-edge brackets from qmf_r<r>.npz up_mean, vs alpha_eff / mu(r) of the earlier theory_qmf run (pd/qmf_real.npz).
 V6  QMF fixed-point census (own clustering; sensitivity to the convergence / stability thresholds and to size-weighting).
 V7  QMF hysteresis by type (own re-implementation of the A/B/C classification) + edge continuity below g = 0.5.
 V8  QMF non-stationarity: first g at dt 0.2 / 0.05 (with amplitudes), oscillating-neuron counts and optic-lobe share.
 V9  MT5 lifetimes: censored-exponential MLE, x at tau = 200 / 3000, from island_<m>_r<r>.npz ext_step.
 V10 MT4 island 1% crossings among survivors at r = 0, 1, 4, 6 from the raw occ/surv of the T = 3000 runs.
 V11 closure census recount (multistable cells, attractors, B1L|B1R cells, no-stable-attractor cells, hysteresis intervals)
     incl. a recount with the QMF's 10% relative-L1 distinctness criterion.
 V12 Leiden P1: module counts, largest, singletons, NMI(P1, P0) (own implementation).
 V13 coexisting QMF states (qmf_multi_vectors.npz): AL lobe asymmetry and neuron-level size of the difference; connected
     components of the differing neurons (own graph).
 V14 QMF D (PR = N/4 on the up-sweep) at r = 0 and the simulated boundaries quoted from phase_complete.json.
Output: results/pd3/verify_module_theory.json (+ printout).
"""
import glob, json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

ROOT = "/MALECNS_ROOT"; C = "/MALECNS_CACHE"; R3 = f"{ROOT}/results/pd3"
t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
dt = 0.2; dec = np.exp(-dt); a_eff = (1 - dec) / dt
OUT = {}


def J(p):
    return json.load(open(p))


def save():
    json.dump(OUT, open(f"{R3}/verify_module_theory.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


sup = np.load(C + "/superclasses.npy", allow_pickle=True); sides = np.load(C + "/sides.npy", allow_pickle=True)
sign = np.load(C + "/sign.npy"); N = len(sign); typ = np.load(C + "/types.npy", allow_pickle=True)
p0 = np.load(f"{R3}/modules/partition_P0.npz", allow_pickle=True)
memb0 = p0["membership"].astype(np.int64); names0 = [str(x) for x in p0["names"]]

# ---------------- V1 ----------------
tail = np.load(f"{ROOT}/results/pd/tail_real_r0_dir.npz", allow_pickle=True)
P = tail["P"]; K = P.shape[1]
al526 = np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)["idx"]
ALm = np.zeros(N, bool); ALm[al526] = True
supp, alsh = [], np.zeros(K)
for k in range(K):
    p = P[:, k].astype(np.float64); p = p / p.sum()
    alsh[k] = p[ALm].sum()
    o = np.argsort(-p); cs = np.cumsum(p[o]); n90 = int(np.argmax(cs >= 0.9)) + 1
    supp.append(np.sort(o[:n90]))
S2 = set(supp[2].tolist())
nonal = [k for k in range(3, K) if alsh[k] < 0.5]
share_in2 = np.array([len(S2.intersection(supp[k].tolist())) / len(supp[k]) for k in nonal])
# a stricter look: how many non-AL modes have > 50% of their support inside the union of the supports of EARLIER modes
cum = set(supp[2].tolist()); inside_earlier = []
for k in nonal:
    s = supp[k].tolist(); inside_earlier.append(len(cum.intersection(s)) / len(s)); cum.update(s)
u2, c2 = np.unique(sup[supp[2]], return_counts=True)
OUT["V1"] = {"n_AL_modes": int((alsh >= 0.5).sum()), "AL_modes": [int(k) for k in np.where(alsh >= 0.5)[0]],
             "n_nonAL_modes_k_ge_3": len(nonal), "share_support_gt50pct_in_mode2": float((share_in2 > 0.5).mean()),
             "median_share_in_mode2": float(np.median(share_in2)), "mode2_support_size": len(supp[2]),
             "mode2_superclass": {str(a): round(int(b) / len(supp[2]), 3) for a, b in sorted(zip(u2, c2), key=lambda x: -x[1])[:4]},
             "mode2_support_in_AL526": int(ALm[supp[2]].sum()),
             "share_gt50pct_inside_union_of_earlier_modes": float((np.array(inside_earlier) > 0.5).mean()),
             "max_inv_ipr_top400": float((1 / tail["ipr"]).max()), "median_support90_size": float(np.median([len(s) for s in supp]))}
log("V1", OUT["V1"]); save()

# ---------------- connectome ----------------
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64)
ins = np.bincount(post, weights=w, minlength=N); c = 1.0 / ins[ins > 0].mean()
ex, ih = sign[pre] > 0, sign[pre] < 0
E = sp.csr_matrix((c * w[ex], (post[ex], pre[ex])), shape=(N, N)); E.sum_duplicates()
I = sp.csr_matrix((c * w[ih], (post[ih], pre[ih])), shape=(N, N)); I.sum_duplicates()
log(f"connectome: c = {c:.6g} (1/c = {1/c:.2f}), E nnz {E.nnz}, I nnz {I.nnz}")
fdt = lambda x: (1 - np.exp(-np.tanh(np.clip(x, 0, None)) * dt)) / dt


def perron_dense(A):
    ev, V = np.linalg.eig(A); k = np.argmax(ev.real); v = np.abs(V[:, k].real)
    evl, U = np.linalg.eig(A.T); kl = np.argmax(evl.real); u = np.abs(U[:, kl].real)
    return float(ev[k].real), v, u


def glin_dense(Ab):
    """exact branching threshold: lambda_PF(f(g A)) = alpha_eff, bisection in ln g (own implementation)"""
    lam = lambda g: np.linalg.eigvals(fdt(g * Ab)).real.max()
    lo, hi = np.log(0.01), np.log(100.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lam(np.exp(mid)) > a_eff:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12:
            break
    g = float(np.exp(0.5 * (lo + hi)))
    _, v, u = perron_dense(fdt(g * Ab)); u = u / (u @ v)
    return g, float(1 / (((u ** 2) @ v) * (u @ (v ** 2))))


# ---------------- V2 ----------------
sd = sides[al526]
idxL, idxR = al526[sd == "L"], al526[sd == "R"]
v2 = {"n_L": len(idxL), "n_R": len(idxR), "n_mid": int(len(al526) - len(idxL) - len(idxR))}
for nm, ids in [("AL-L", idxL), ("AL-R", idxR), ("AL526", np.sort(al526))]:
    Ab = E[ids][:, ids].toarray()
    gl, Kk = glin_dense(Ab)
    v2[nm] = {"Lambda": float(np.linalg.eigvals(Ab).real.max()), "g_lin": gl, "K": Kk}
OUT["V2"] = v2; log("V2", v2); save()

# ---------------- V3 + V4 (AL pair quotient) ----------------
cand = [k for k, n in enumerate(names0) if n != "rest"]
Lam = {}
for k in cand:
    ids = np.where(memb0 == k)[0]
    Lam[k] = float(np.linalg.eigvals(E[ids][:, ids].toarray()).real.max()) if len(ids) >= 2 else 0.0
top = sorted(cand, key=lambda k: -Lam[k])[:30]
grp = np.full(N, 30, np.int64)
for i, k in enumerate(top):
    grp[memb0 == k] = i
q0 = np.load(f"{R3}/modules/quotient_P0.npz", allow_pickle=True)
G = 31
OH = sp.csr_matrix((np.ones(N), (np.arange(N), grp)), shape=(N, G))
EO = (E @ OH).toarray(); IO = (I @ OH).toarray()
CVE = np.full((G, G), np.nan); CVI = np.full((G, G), np.nan)
for k in range(G):
    sel = grp == k
    for l in range(G):
        for A_, CV_ in [(EO, CVE), (IO, CVI)]:
            x = A_[sel, l]; m = x.mean()
            if m > 0:
                CV_[k, l] = x.std() / m
off = ~np.eye(G, dtype=bool)
v3 = {"group_identical_to_quotient_P0": bool((grp == q0["group"].astype(np.int64)).all()),
      "excluded_from_top30": [names0[k] for k in cand if k not in top],
      "rest_size": int((grp == 30).sum()),
      "median_CV": {"E_self": float(np.nanmedian(np.diag(CVE))), "E_cross": float(np.nanmedian(CVE[off])),
                    "I_self": float(np.nanmedian(np.diag(CVI))), "I_cross": float(np.nanmedian(CVI[off]))},
      "share_cross_pairs_with_zero_input_fraction_gt_half": None}
# how much of the cross CV comes from neurons receiving nothing: median fraction of target neurons with zero input
zf = []
for k in range(G):
    sel = grp == k
    for l in range(G):
        if k != l and EO[sel, l].mean() > 0:
            zf.append(float((EO[sel, l] == 0).mean()))
v3["median_fraction_of_targets_with_zero_E_input_cross"] = float(np.median(zf))
# CV among receivers only (neurons with nonzero input) as a sensitivity check
cvr = []
for k in range(G):
    sel = grp == k
    for l in range(G):
        if k != l:
            x = EO[sel, l]; x = x[x > 0]
            if len(x) >= 2:
                cvr.append(x.std() / x.mean())
v3["median_CV_E_cross_receivers_only"] = float(np.median(cvr))
# Perron-weighted AL pair, from the connectome (own Perron vectors of E_kk, v mean 1, <u, v> = 1)
iL, iR = [names0[k] for k in top].index("AL-L"), [names0[k] for k in top].index("AL-R")
gL, gR = np.where(grp == iL)[0], np.where(grp == iR)[0]
PV = {}
for nm, ids in [("L", gL), ("R", gR)]:
    lam, v, u = perron_dense(E[ids][:, ids].toarray()); v = v / v.mean(); u = u / (u @ v); PV[nm] = (ids, v, u)
EPW = np.zeros((2, 2)); IPW = np.zeros((2, 2))
for a, ka in enumerate("LR"):
    for b, kb in enumerate("LR"):
        ia, va, ua = PV[ka]; ib, vb, ub = PV[kb]
        EPW[a, b] = ua @ (E[ia][:, ib] @ vb); IPW[a, b] = ua @ (I[ia][:, ib] @ vb)
v3["AL_PW_from_connectome"] = {"E": EPW.tolist(), "I": IPW.tolist()}
v3["AL_PW_quotient_file"] = {"E": q0["E_PW"][np.ix_([iL, iR], [iL, iR])].tolist(), "I": q0["I_PW"][np.ix_([iL, iR], [iL, iR])].tolist()}
v3["max_abs_diff_AL_PW"] = float(max(np.abs(EPW - q0["E_PW"][np.ix_([iL, iR], [iL, iR])]).max(),
                                     np.abs(IPW - q0["I_PW"][np.ix_([iL, iR], [iL, iR])]).max()))
OUT["V3"] = v3; log("V3", v3); save()


# ---------------- V4: closure SL edge ----------------
def cone_radius(W, nstart=64, iters=6000, seed=1):
    """max over random positive starts of the growth rate of x -> [W x]_+ (damped normalised iteration)"""
    rng = np.random.default_rng(seed); best = 0.0
    for s in range(nstart):
        x = rng.random(W.shape[0]) + 1e-3; x /= x.sum(); m = 0
        for it in range(iters):
            y = np.clip(W @ x, 0, None); m = y.sum() / x.sum()
            if m <= 0:
                break
            x = 0.5 * x + 0.5 * y / m; x /= x.sum()
        best = max(best, m)
    return best


def closure_grows(Ew, Iw, r, g, steps=40000):
    """iterate the closure map from X = 1e-9 everywhere; True if activity grows by > 100x"""
    X = np.full(Ew.shape[0], 1e-9); W = Ew - r * Iw
    for _ in range(steps):
        s = g * (W @ X); X = X * dec + (1 - X) * (-np.expm1(-np.tanh(np.clip(s, 0, None)) * dt))
        if X.max() > 1e-7 or X.max() < 1e-30:
            break
    return bool(X.max() > 1e-7)


v4 = {}
for P_ in ["P0", "P1"]:
    q = np.load(f"{R3}/modules/quotient_{P_}.npz", allow_pickle=True); Ew, Iw = q["E_PW"], q["I_PW"]
    cp = J(f"{R3}/modules/coarse_post_{P_}.json")["g_SL_edge"]
    rows = {}
    for r in [0.0, 1.0, 2.5, 4.0, 6.0]:
        mu = float(np.linalg.eigvals(Ew).real.max()) if r == 0 else cone_radius(Ew - r * Iw)
        ge = a_eff / mu if mu > 0 else np.inf
        rows[f"{r:g}"] = {"mu_plus": mu, "g_edge_mine": ge, "g_edge_file": cp[f"{r:g}"],
                          "grows_at_0.99_edge": closure_grows(Ew, Iw, r, 0.99 * ge) if np.isfinite(ge) else None,
                          "grows_at_1.01_edge": closure_grows(Ew, Iw, r, 1.01 * ge) if np.isfinite(ge) else None}
    v4[P_] = rows
OUT["V4"] = v4; log("V4", json.dumps(v4)[:800]); save()

# ---------------- V5, V6, V7, V8, V14: QMF ----------------
mu_old = np.load(f"{ROOT}/results/pd/qmf_real.npz", allow_pickle=True)
mu_r = dict(zip([float(x) for x in mu_old["r"]], [float(x) for x in mu_old["mu"][:, 0]]))
sizes0 = np.bincount(memb0, minlength=len(names0)).astype(float)
qfiles = sorted([f for f in glob.glob(f"{R3}/module_theory/qmf_r*.npz") if "smoke" not in f],
                key=lambda f: float(f.split("qmf_r")[1][:-4]))


def rl1(a, b):
    return np.abs(a - b).sum() / max(np.abs(a).sum(), np.abs(b).sum(), 1e-300)


def n_clusters(vecs, thr):
    """single-linkage clusters (edge if relative L1 <= thr): an order-independent count"""
    n = len(vecs)
    if n == 0:
        return 0
    A = np.array([[rl1(vecs[i], vecs[j]) <= thr for j in range(n)] for i in range(n)])
    return connected_components(sp.csr_matrix(A), directed=False)[0]


def first_cross(g, y, thr):
    i = np.where(y >= thr)[0]
    if not len(i) or i[0] == 0:
        return None
    i = i[0]
    return float(np.exp(np.log(g[i - 1]) + (thr - y[i - 1]) / (y[i] - y[i - 1]) * (np.log(g[i]) - np.log(g[i - 1]))))


qa = J(f"{R3}/module_theory/qmf_analysis.json")["per_r"]
v5, v6, v7, v8 = {}, {}, {}, {}
for f in qfiles:
    d = np.load(f, allow_pickle=True); r = float(json.loads(str(d["meta"]))["r"]); rk = f"{r:g}"
    g1, g2 = d["g1"], d["g2"]
    # V5 silent edge
    up = d["up_mean"]; i = int(np.argmax(up > 1e-8))
    v5[rk] = {"bracket": [float(g2[i - 1]), float(g2[i])], "pred_alpha_over_mu_theory_qmf": a_eff / mu_r[r] if r in mu_r else None,
              "pred_inside": bool(g2[i - 1] <= a_eff / mu_r[r] <= g2[i]) if r in mu_r else None,
              "down_leaves_silence_bracket": [float(g2[int(np.argmax(d["dn_mean"] > 1e-8)) - 1]), float(g2[int(np.argmax(d["dn_mean"] > 1e-8))])]}
    # V6 census
    res_, rho, X, mean = d["p1_resid"], d["p1_rhoJ"], d["p1_X"].copy(), d["p1_mean"]
    X[mean < 1e-8] = 0
    variants = {"as_published": dict(rt=1e-7, dist=1e-6, rmax=1.0, w=None, thr=0.1),
                "resid<1e-9": dict(rt=1e-9, dist=1e-6, rmax=1.0, w=None, thr=0.1),
                "rhoJ<0.995": dict(rt=1e-7, dist=1e-6, rmax=0.995, w=None, thr=0.1),
                "size_weighted_L1": dict(rt=1e-7, dist=1e-6, rmax=1.0, w=sizes0, thr=0.1),
                "threshold_1pct": dict(rt=1e-7, dist=1e-6, rmax=1.0, w=None, thr=0.01),
                "threshold_1pct_resid<1e-9_rhoJ<0.995": dict(rt=1e-9, dist=1e-6, rmax=0.995, w=None, thr=0.01)}
    v6[rk] = {}
    for vn, pa in variants.items():
        conv = (res_ < pa["rt"]) & (res_ / np.maximum(1 - rho, 1e-6) < pa["dist"]) & (rho < pa["rmax"])
        glist, mx = [], 1
        for gi in range(len(g1)):
            vecs = [X[s, gi] * (pa["w"] if pa["w"] is not None else 1) for s in range(X.shape[0]) if conv[s, gi]]
            nc = n_clusters(vecs, pa["thr"])
            if nc >= 2:
                glist.append(round(float(g1[gi]), 4)); mx = max(mx, nc)
        v6[rk][vn] = {"n_g": len(glist), "g_min": min(glist) if glist else None, "g_max": max(glist) if glist else None, "max_count": int(mx)}
    v6[rk]["published_g"] = [round(x, 4) for x in qa[rk]["g_multistable_fixed_points"]]
    v6[rk]["published_max"] = qa[rk]["max_n_fixed_point_attractors"]
    # min rho(J) gap and residual of the states counted at multistable g (as published criterion)
    conv = (res_ < 1e-7) & (res_ / np.maximum(1 - rho, 1e-6) < 1e-6) & (rho < 1)
    mg = [gi for gi in range(len(g1)) if any(abs(g1[gi] - x) < 1e-9 for x in qa[rk]["g_multistable_fixed_points"])]
    if mg:
        rr = np.concatenate([rho[conv[:, gi], gi] for gi in mg]); ee = np.concatenate([res_[conv[:, gi], gi] for gi in mg])
        v6[rk]["counted_states_rhoJ_max"] = float(rr.max()); v6[rk]["counted_states_resid_max"] = float(ee.max())
        v6[rk]["counted_states_frac_resid_lt_1e-9"] = float((ee < 1e-9).mean())
    # V7 hysteresis types (own implementation)
    upm, dnm = d["up_tavg_mean"], d["dn_tavg_mean"]; ur, dr = d["up_resid"], d["dn_resid"]; uj, dj = d["up_rhoJ"], d["dn_rhoJ"]
    rel = np.abs(upm - dnm) / np.maximum(np.maximum(upm, dnm), 1e-300)
    okc = (((ur < 1e-7) | (ur > 1e-5)) & ((dr < 1e-7) | (dr > 1e-5)) & (np.abs(uj - 1) > 0.01) & (np.abs(dj - 1) > 0.01)
           & (np.maximum(upm, dnm) >= 1e-6))
    big = (rel > 0.05) & okc
    run = np.zeros(len(g2), bool)
    for k in range(len(g2)):
        if big[k] and ((k > 0 and big[k - 1]) or (k + 1 < len(g2) and big[k + 1])):
            run[k] = True
    A_ = run & (ur < 1e-7) & (dr < 1e-7) & (uj < 1) & (dj < 1)
    B_ = run & ((ur < 1e-7) != (dr < 1e-7))
    C_ = run & (ur > 1e-5) & (dr > 1e-5)
    low = g2 < 0.5
    rawbig = (rel > 0.05) & (np.maximum(upm, dnm) >= 1e-6) & low
    v7[rk] = {"A": int(A_.sum()), "B": int(B_.sum()), "C": int(C_.sum()),
              "A_g": [round(float(x), 3) for x in g2[A_]], "B_g": [round(float(x), 3) for x in g2[B_]],
              "A_max_rel_diff": float(rel[A_].max()) if A_.any() else None,
              "A_rhoJ_max": float(max(uj[A_].max(), dj[A_].max())) if A_.any() else None,
              "published": qa[rk]["hysteresis_points_by_type"],
              "max_rel_updown_g_lt_0.5_all_points": float(rel[low & (np.maximum(upm, dnm) >= 1e-6)].max()),
              "g_lt_0.5_points_rel_gt_5pct": [round(float(x), 4) for x in g2[rawbig]],
              "rhoJ_at_those": [round(float(max(uj[k], dj[k])), 4) for k in np.where(rawbig)[0]],
              "max_rel_updown_r0_all": float(rel[np.maximum(upm, dnm) >= 1e-6].max()) if r == 0 else None}
    # V8 non-stationarity
    oscA = (d["p1_resid"][0] > 1e-5) & (d["p1_period"][0] <= 512)
    p3 = d["p3_resid"] > 2.5e-6
    std = d["all_active_std"].astype(np.float32); nos = (std > 1e-3).sum(1); gi = int(np.argmax(nos))
    sel = std[gi] > 1e-3
    ol = int(np.isin(sup[sel], ["ol_intrinsic"]).sum())
    v8[rk] = {"first_g_dt0.2": float(g1[oscA][0]) if oscA.any() else None, "n_g_dt0.2": int(oscA.sum()),
              "first_g_dt0.05": float(g1[p3][0]) if p3.any() else None, "n_g_dt0.05": int(p3.sum()),
              "dt0.05_amp_at_first": float(d["p3_amp"][p3][0]) if p3.any() else None,
              "dt0.05_min_amp_over_flagged": float(d["p3_amp"][p3].min()) if p3.any() else None,
              "dt0.2_amp_at_first": float(d["p1_amp"][0][oscA][0]) if oscA.any() else None,
              "max_n_oscillating_neurons": int(nos.max()), "at_g": float(g1[gi]), "n_ol_intrinsic": ol,
              "ol_share": ol / max(int(sel.sum()), 1)}
    if r == 0:
        v14 = {"QMF_D_up_PR_quarter_mine": first_cross(g2, d["up_pr"], 0.25),
               "r0_n_fixed_point_attractors_max": int(max(n_clusters([X[s, gi] for s in range(X.shape[0]) if conv[s, gi]], 0.1)
                                                          for gi in range(len(g1))))}
    log(f"QMF r={rk}: edge {v5[rk]['bracket']} pred {v5[rk]['pred_alpha_over_mu_theory_qmf']}; census {v6[rk]['as_published']} "
        f"(published {v6[rk]['published_max']}, n_g {len(v6[rk]['published_g'])}); hyst A/B/C {v7[rk]['A']}/{v7[rk]['B']}/{v7[rk]['C']} "
        f"pub {v7[rk]['published']}; osc {v8[rk]['first_g_dt0.2']} / {v8[rk]['first_g_dt0.05']} n {v8[rk]['max_n_oscillating_neurons']} OL {ol}")
OUT["V5"], OUT["V6"], OUT["V7"], OUT["V8"] = v5, v6, v7, v8
pc = J(f"{ROOT}/results/analysis2/phase_complete.json")
v14["sim_D"] = {r: pc["sim"]["real"][r]["D"]["g"] for r in ["0", "0.25", "0.5", "1"]}
v14["sim_on"] = {r: pc["sim"]["real"][r]["on"]["g"] for r in ["0", "0.5", "1", "1.5", "2", "3", "4", "6"] if r in pc["sim"]["real"]}
v14["sim_S"] = {r: pc["sim"]["real"][r]["S"]["g"] for r in ["0", "1", "4", "6"]}
v14["theory_D"] = {r: pc["theory"]["real"][r]["D"]["g"] for r in ["0", "0.25", "0.5"]}
OUT["V14"] = v14; log("V14", v14); save()

# ---------------- V9: MT5 ----------------
il = J(f"{R3}/module_theory/island_lifetimes.json")
v9 = {}
for f in sorted(glob.glob(f"{R3}/module_theory/island_*_r*.npz")):
    d = np.load(f, allow_pickle=True); m = json.loads(str(d["meta"]))
    T = float(d["T"]); st = int(round(T / float(d["dt"])))
    ext = d["ext_step"]; tt = np.where(ext >= 0, ext, st) * float(d["dt"]); deaths = (ext >= 0).sum(1)
    tau = np.where(deaths > 0, tt.sum(1) / np.maximum(deaths, 1), np.inf)
    x = d["x"]; lt = np.log(np.where(np.isfinite(tau), tau, 1e12))

    def xcross(th):
        i = np.where(lt >= np.log(th))[0]
        if not len(i):
            return None
        i = i[0]
        if i == 0:
            return float(x[0])
        return float(np.exp(np.log(x[i - 1]) + (np.log(th) - lt[i - 1]) / (lt[i] - lt[i - 1]) * (np.log(x[i]) - np.log(x[i - 1]))))
    key = f"{m['module']}_r{m['r']:g}"
    x3 = xcross(3000)
    v9[key] = {"n": int(d["n"]), "g_lin": float(d["g_lin"]), "tau_x1": float(tau[0]), "x_tau200": xcross(200), "x_tau3000": x3,
               "g_tau3000": x3 * float(d["g_lin"]) if x3 else None, "max_tau_finite": float(tau[np.isfinite(tau)].max()),
               "n_g_all_alive": int((deaths == 0).sum())}
    pub = il.get(key) or il.get(m["module"] + f"_r{m['r']:g}")
    v9[key]["published"] = {k: pub[k] for k in ["tau_x1", "x_tau200", "x_tau3000", "g_tau3000"]} if pub else None
OUT["V9"] = v9; log("V9", json.dumps(v9)[:1500]); save()

# ---------------- V10: MT4 from raw occ ----------------
OH0 = sp.csr_matrix((np.ones(N), (memb0, np.arange(N))), shape=(len(names0), N)); sz0 = np.asarray(OH0.sum(1)).ravel()
tab0 = {r_["name"]: r_["g_lin"] for r_ in J(f"{R3}/modules/modules_table.json")["P0"]["rows"]}
v10 = {}
for rr in ["0", "1", "4", "6"]:
    pts = []
    for f in (glob.glob(f"{ROOT}/results/pd/decay_r{rr}_real.npz") + glob.glob(f"{ROOT}/results/pd2/onset_real_r{rr}.npz")
              + glob.glob(f"{ROOT}/results/pd2/deloc_real_r{rr}.npz")):
        d = np.load(f, allow_pickle=True); m = json.loads(str(d["meta"]))
        if m["exp"] != "decay" or m["norm"] != "raw" or float(m["h"]) != 0 or abs(float(m["T"]) - 3000) > 1e-6 or m.get("null"):
            continue
        Xm = (OH0 @ d["occ"].astype(np.float64).T).T / sz0
        s2 = d["surv"][:, d["t"] >= 1500, :].mean((1, 2))
        for i, g in enumerate(d["g"]):
            pts.append((float(g), Xm[i], float(s2[i]), int(m["B"])))
    pts.sort(key=lambda x: x[0]); pool = []
    for g, Xm, s2, B in pts:
        if pool and abs(g / pool[-1][0] - 1) < 1e-4:
            g0, X0, s0, B0 = pool[-1]; pool[-1] = (g0, (X0 * B0 + Xm * B) / (B0 + B), (s0 * B0 + s2 * B) / (B0 + B), B0 + B)
        else:
            pool.append((g, Xm, s2, B))
    gg = np.array([p[0] for p in pool]); XX = np.array([p[1] for p in pool]); S2 = np.array([p[2] for p in pool])
    ok = S2 >= 0.5; gs, Xs = gg[ok], XX[ok] / S2[ok, None]
    cr = {}
    for k, nm in enumerate(names0):
        i = np.where(Xs[:, k] >= 0.01)[0]
        if not len(i):
            cr[nm] = None; continue
        i = i[0]
        cr[nm] = float(gs[0]) if i == 0 else float(np.exp(np.log(gs[i - 1]) + (np.log(0.01) - np.log(max(Xs[i - 1, k], 1e-12))) /
                                                           (np.log(Xs[i, k]) - np.log(max(Xs[i - 1, k], 1e-12))) * (np.log(gs[i]) - np.log(gs[i - 1]))))
    isl = [n for n in names0 if n.startswith("isl")]
    vals = [cr[n] for n in isl if cr[n] is not None]
    below = [n for n in isl if n in tab0 and cr[n] is not None and cr[n] < tab0[n]]
    above = [(n, round(cr[n], 3), round(tab0[n], 3)) for n in isl if n in tab0 and cr[n] is not None and cr[n] >= tab0[n]]
    ncount = (Xs >= 0.01).sum(1)
    v10[rr] = {"first_sustained_g": float(gs[0]) if len(gs) else None, "n_islands_reaching_1pct": len(vals),
               "island_cross_range": [min(vals), max(vals)] if vals else None, "island_cross_median": float(np.median(vals)) if vals else None,
               "key": {n: cr[n] for n in ["isl1", "isl2", "isl3", "isl5", "rest"]},
               "n_top30_islands_crossing_below_own_glin": len(below), "islands_crossing_at_or_above_own_glin": above,
               "n_modules_ge_1pct_vs_g": [[round(float(a), 3), int(b)] for a, b in zip(gs, ncount)][:40]}
OUT["V10"] = v10; log("V10", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'n_modules_ge_1pct_vs_g'} for k, v in v10.items()})); save()

# ---------------- V11: closure census recount ----------------
v11 = {}
for P_ in ["P0", "P1"]:
    d = np.load(f"{R3}/modules/coarse_mf_{P_}.npz", allow_pickle=True)
    g, rv, nA, AX, rhoA = d["g"], d["r"], d["n_attr"], d["attr_X"], d["attr_rhoJ"]
    nm = [str(x) for x in d["names"]]; gSL = d["g_SL_edge"]; nunc = d["n_unconverged"]
    iLq, iRq = (nm.index("AL-L"), nm.index("AL-R")) if "AL-L" in nm else (None, None)
    ncorr = np.zeros(nA.shape, int); n10 = np.zeros(nA.shape, int); nlab = np.zeros(nA.shape, int); broken = 0; brk_first = {}
    nzero_all_unconv = 0; nzero = 0; zero_unc = []
    for ri in range(len(rv)):
        for gi in range(len(g)):
            st = []
            for j in range(nA[ri, gi]):
                x = AX[ri, gi, j]
                if np.nanmax(x) < 1e-12:
                    if g[gi] < gSL[ri]:
                        st.append(np.zeros_like(x))
                else:
                    st.append(x)
            ncorr[ri, gi] = len(st)
            n10[ri, gi] = n_clusters(st, 0.1) if st else 0
            nlab[ri, gi] = len({tuple((x > 1e-3).tolist()) for x in st})
            if iLq is not None:
                hasL = any(x[iLq] > 1e-3 and x[iRq] <= 1e-3 for x in st); hasR = any(x[iRq] > 1e-3 and x[iLq] <= 1e-3 for x in st)
                if hasL and hasR:
                    broken += 1; brk_first.setdefault(f"{rv[ri]:g}", float(g[gi]))
            if not st:
                nzero += 1; zero_unc.append(int(nunc[ri, gi]))
                if nunc[ri, gi] == 2 * int(d["n_starts"]):
                    nzero_all_unconv += 1
    # hysteresis recount (both branches settled; L_inf > 1e-3; >= 2 consecutive)
    hyst = []; hyst_rel = 0
    for ri in range(len(rv)):
        up, dn = d["up"][ri], d["dn"][ri]; okk = (d["up_resid"][ri] < 1e-10) & (d["dn_resid"][ri] < 1e-10)
        h = (np.abs(up - dn).max(1) > 1e-3) & okk
        hr = (np.abs(up.sum(1) - dn.sum(1)) / np.maximum(np.maximum(up.sum(1), dn.sum(1)), 1e-300) > 0.05) & okk
        k = 0
        while k < len(g):
            if h[k]:
                j = k
                while j + 1 < len(g) and h[j + 1]:
                    j += 1
                if j > k:
                    hyst.append((float(rv[ri]), float(g[k]), float(g[j])))
                k = j + 1
            else:
                k += 1
        k = 0
        while k < len(g):
            if hr[k]:
                j = k
                while j + 1 < len(g) and hr[j + 1]:
                    j += 1
                if j > k:
                    hyst_rel += 1
                k = j + 1
            else:
                k += 1
    # first g with ANY unconverged start vs first g with NO stable attractor, per r
    first_unc = {f"{rv[ri]:g}": (float(g[np.where(nunc[ri] > 0)[0][0]]) if (nunc[ri] > 0).any() else None) for ri in range(len(rv))}
    first_none = {f"{rv[ri]:g}": (float(g[np.where(ncorr[ri] == 0)[0][0]]) if (ncorr[ri] == 0).any() else None) for ri in range(len(rv))}
    up0 = d["up"][0]; nact0 = (up0 > 1e-3).sum(1)
    v11[P_] = {"multistable_cells": int((ncorr >= 2).sum()), "max_attractors": int(ncorr.max()),
               "max_attractors_by_r": {f"{rv[ri]:g}": int(ncorr[ri].max()) for ri in range(len(rv))},
               "multistable_cells_10pct_relL1": int((n10 >= 2).sum()), "max_attractors_10pct_relL1": int(n10.max()),
               "multistable_cells_distinct_active_sets": int((nlab >= 2).sum()), "max_distinct_active_sets": int(nlab.max()),
               "AL_lobe_broken_cells": broken, "AL_lobe_broken_first_g": brk_first,
               "cells_no_stable_attractor": nzero, "of_which_all_starts_unconverged": nzero_all_unconv,
               "unconverged_starts_in_those_cells_min_median_max": [min(zero_unc), float(np.median(zero_unc)), max(zero_unc)] if zero_unc else None,
               "n_starts_total": 2 * int(d["n_starts"]),
               "first_g_any_unconverged_by_r": first_unc, "first_g_no_stable_attractor_by_r": first_none,
               "hysteresis_intervals_Linf1e-3": len(hyst), "hysteresis_r_values": sorted({h_[0] for h_ in hyst}),
               "hysteresis_intervals_rel5pct_total": hyst_rel,
               "r0_up_first_g_n_active": {str(n): float(g[np.argmax(nact0 >= n)]) for n in [2, 3, 6, 11, 31] if (nact0 >= n).any()}}
    log(f"V11 {P_}", {k: v for k, v in v11[P_].items() if not isinstance(v, dict)})
    # V11b: the same census with the SL edge computed directly (pd3_verify_module_theory_sledge.py), if available
    try:
        sl = J(f"{R3}/verify_module_theory_sledge.json")[P_]
        gtrue = np.array([sl[f"{x:g}"][0] if sl[f"{x:g}"][0] is not None else np.inf for x in rv])
        nt = np.zeros(nA.shape, int); changed = 0
        for ri in range(len(rv)):
            for gi in range(len(g)):
                k_ = 0; kf = 0
                for j in range(nA[ri, gi]):
                    x = AX[ri, gi, j]
                    if np.nanmax(x) < 1e-12:
                        k_ += g[gi] < gtrue[ri]; kf += g[gi] < gSL[ri]
                    else:
                        k_ += 1; kf += 1
                nt[ri, gi] = k_; changed += k_ != kf
        v11[P_]["direct_SL_edge"] = {"g_SL_edge_direct": {f"{x:g}": (float(y) if np.isfinite(y) else None) for x, y in zip(rv, gtrue)},
                                     "max_rel_diff_vs_file_where_both_finite": float(np.nanmax(np.where(np.isfinite(gtrue) & (gSL < 1e10),
                                                                                                         np.abs(gtrue / gSL - 1), np.nan))),
                                     "multistable_cells": int((nt >= 2).sum()), "cells_no_stable_attractor": int((nt == 0).sum()),
                                     "cells_with_changed_attractor_count": int(changed),
                                     "first_g_no_stable_attractor_by_r": {f"{rv[ri]:g}": (float(g[np.where(nt[ri] == 0)[0][0]])
                                                                                       if (nt[ri] == 0).any() else None) for ri in range(len(rv))}}
        log(f"V11b {P_} direct SL edge:", {k: v for k, v in v11[P_]["direct_SL_edge"].items() if not isinstance(v, dict)})
    except FileNotFoundError:
        log("V11b skipped: run pd3_verify_module_theory_sledge.py first")
OUT["V11"] = v11; save()

# ---------------- V12: Leiden ----------------
p1 = np.load(f"{R3}/modules/partition_P1.npz", allow_pickle=True)
m1 = p1["membership"].astype(np.int64); s1 = np.bincount(m1)


def nmi(a, b):
    n = len(a); _, a = np.unique(a, return_inverse=True); _, b = np.unique(b, return_inverse=True)
    Cm = sp.csr_matrix((np.ones(n), (a, b))).tocoo(); pij = Cm.data / n
    pi = np.bincount(a) / n; pj = np.bincount(b) / n
    Iab = (pij * np.log(pij / (pi[Cm.row] * pj[Cm.col]))).sum()
    Ha = -(pi * np.log(pi)).sum(); Hb = -(pj * np.log(pj)).sum()
    return float(2 * Iab / (Ha + Hb))


lr = J(f"{R3}/modules/leiden_runs.json")
OUT["V12"] = {"n_modules": int((s1 > 0).sum()), "n_ge50": int((s1 >= 50).sum()), "largest": int(s1.max()),
              "n_singletons": int((s1 == 1).sum()), "nmi_P1_P0_arithmetic": nmi(m1, memb0),
              "published_nmi_vs_P0": [r_["nmi_vs_P0"] for r_ in lr["runs"] if r_["gamma"] == 4.0 and r_["seed"] == 1]}
log("V12", OUT["V12"]); save()

# ---------------- V13: coexisting QMF states ----------------
vz = np.load(f"{R3}/module_theory/qmf_multi_vectors.npz")
nz = sign[pre] != 0
Au = sp.csr_matrix((np.ones(int(nz.sum()), np.int8), (post[nz], pre[nz])), shape=(N, N)); Au = ((Au + Au.T) > 0).tocsr()
isL, isR = memb0 == names0.index("AL-L"), memb0 == names0.index("AL-R")
groups = {}
for k in vz.files:
    rr, gg, st = k.split("_", 2); groups.setdefault((rr, gg), {})[st] = vz[k].astype(np.float64)
v13 = {"pairs": 0, "one_component": 0, "rel_L1_neuron_level": [], "per_r": {}}
for (rr, gg), sts in groups.items():
    names_ = sorted(sts)
    for i in range(len(names_)):
        for j in range(i + 1, len(names_)):
            a, b = sts[names_[i]], sts[names_[j]]
            dif = np.abs(a - b) > 0.05; nd = int(dif.sum())
            ncomp = connected_components(Au[dif][:, dif], directed=False)[0] if nd else 0
            v13["pairs"] += 1; v13["one_component"] += int(ncomp == 1)
            relL1 = float(np.abs(a - b).sum() / max(a.sum(), b.sum())); v13["rel_L1_neuron_level"].append(relL1)
            e = v13["per_r"].setdefault(rr, {"n_diff": [], "rel_L1": [], "sum_abs_diff_neurons": [], "AL_A": []})
            e["n_diff"].append(nd); e["rel_L1"].append(relL1); e["sum_abs_diff_neurons"].append(float(np.abs(a - b).sum()))
    for st_, v in sts.items():
        XL, XR = v[isL].mean(), v[isR].mean()
        v13["per_r"][rr]["AL_A"].append((gg, st_, round(float(XL), 4), round(float(XR), 4), round(float((XL - XR) / (XL + XR)), 4)))
for rr, e in v13["per_r"].items():
    e["n_diff_range"] = [min(e["n_diff"]), max(e["n_diff"])]; e["rel_L1_range"] = [min(e["rel_L1"]), max(e["rel_L1"])]
    e["sum_abs_diff_range"] = [min(e["sum_abs_diff_neurons"]), max(e["sum_abs_diff_neurons"])]
    As = [x[4] for x in e["AL_A"]]; e["A_range"] = [min(As), max(As)]
    for kk in ["n_diff", "rel_L1", "sum_abs_diff_neurons"]:
        e.pop(kk)
v13["rel_L1_neuron_level_median"] = float(np.median(v13["rel_L1_neuron_level"]))
v13["frac_pairs_rel_L1_neuron_lt_10pct"] = float((np.array(v13["rel_L1_neuron_level"]) < 0.1).mean())
v13.pop("rel_L1_neuron_level")
OUT["V13"] = v13; log("V13", json.dumps({k: v for k, v in v13.items() if k != "per_r"}),
                      {rr: (e["n_diff_range"], e["rel_L1_range"], e["sum_abs_diff_range"], e["A_range"]) for rr, e in v13["per_r"].items()})
save()
log("saved results/pd3/verify_module_theory.json")

# ---------------- write the verified / corrected numbers into module_theory.json ----------------
mt = J(f"{R3}/module_theory.json")
ver = {"by": "code/pd3_verify_module_theory.py (+ pd3_verify_module_theory_sledge.py), 2026-11-09; full output results/pd3/verify_module_theory.json",
       "note": "added by the verifier; rerunning pd3_module_theory_summary.py drops this block, rerun pd3_verify_module_theory.py afterwards",
       "reproduced_exactly": ["MT1 nesting 0.866 / median 0.625", "AL-L / AL-R g_lin, K", "P0 equitability CV medians", "AL Perron-weighted pair",
                              "P0 closure SL edge (all 25 r)", "QMF silent-edge brackets", "QMF hysteresis A/B/C counts", "QMF non-stationary first g and neuron counts",
                              "MT5 tau / x at tau 200, 3000", "MT4 r = 0 island 1% crossings", "P0 closure multistable cells / max / B1L|B1R cells / hysteresis intervals",
                              "Leiden counts and NMI", "MT3b 78/80 one-component pairs, n_diff ranges, AL lobe asymmetry"]}
if "direct_SL_edge" in OUT["V11"].get("P1", {}):
    ver["MT2_P1_g_SL_edge_direct"] = OUT["V11"]["P1"]["direct_SL_edge"]["g_SL_edge_direct"]
    ver["MT2_P1_cells_no_stable_attractor_with_direct_edge"] = OUT["V11"]["P1"]["direct_SL_edge"]["cells_no_stable_attractor"]
    ver["MT2_P1_first_g_no_stable_attractor_with_direct_edge"] = OUT["V11"]["P1"]["direct_SL_edge"]["first_g_no_stable_attractor_by_r"]
    mt["MT2"]["P1"]["g_SL_edge_CORRECTED_direct"] = ver["MT2_P1_g_SL_edge_direct"]
ver["MT2_P0_no_stable_attractor"] = {k: OUT["V11"]["P0"][k] for k in ["cells_no_stable_attractor", "of_which_all_starts_unconverged",
                                                                      "unconverged_starts_in_those_cells_min_median_max", "first_g_no_stable_attractor_by_r",
                                                                      "first_g_any_unconverged_by_r"]}
ver["MT2_closure_with_QMF_10pct_criterion"] = {P_: {"multistable_cells": OUT["V11"][P_]["multistable_cells_10pct_relL1"],
                                                     "max_attractors": OUT["V11"][P_]["max_attractors_10pct_relL1"]} for P_ in ["P0", "P1"]}
ver["MT3_multistability_threshold_sensitivity"] = {r: {k: OUT["V6"][r][k] for k in ["as_published", "threshold_1pct",
                                                                                   "threshold_1pct_resid<1e-9_rhoJ<0.995", "resid<1e-9"]}
                                                    for r in OUT["V6"]}
ver["MT3_counted_states_resid"] = {r: {k: OUT["V6"][r][k] for k in ["counted_states_resid_max", "counted_states_frac_resid_lt_1e-9",
                                                                    "counted_states_rhoJ_max"] if k in OUT["V6"][r]} for r in OUT["V6"]}
ver["MT3_oscillating_neurons_optic_lobe_share"] = {r: round(OUT["V8"][r]["ol_share"], 3) for r in OUT["V8"] if OUT["V8"][r]["max_n_oscillating_neurons"]}
ver["MT3b_neuron_level_difference"] = {r: {k: e[k] for k in ["n_diff_range", "rel_L1_range", "sum_abs_diff_range", "A_range"]}
                                       for r, e in OUT["V13"]["per_r"].items()}
ver["MT4_islands_reaching_1pct_at_or_above_own_g_lin"] = {r: OUT["V10"][r]["islands_crossing_at_or_above_own_glin"] for r in OUT["V10"]}
ver["sim_boundaries_phase_complete"] = {"on": OUT["V14"]["sim_on"], "S": OUT["V14"]["sim_S"]}
mt["verification"] = ver
json.dump(mt, open(f"{R3}/module_theory.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("wrote the verification block into results/pd3/module_theory.json")
