"""pd3 module_theory MT1 (items 1-5): per-module thresholds, quotient matrices, equitability, crossover r*, dead-zone screen.

For partition P (P0 / P1 / P2: results/pd3/modules/partition_<P>.npz) take the top M = 30 modules by the Perron root Lambda_k
of the excitatory block E_kk (P0: among its 33 non-rest modules), and lump everything else into 'rest' (index M). Then:
 1. Lambda_k, weak-synapse threshold alpha_eff / Lambda_k, the exact branching threshold g_lin,k of the block alone
    (lambda_PF(B_kk(g)) = alpha_eff with B_ij(g) = f(g E_ij), f(x) = (1 - exp(-tanh(x) dt))/dt; brentq in ln g, rel. tol 1e-7;
    inf if the block can never become supercritical because f <= alpha_eff), and K_k = 1/(<u^2, v><u, v^2>) (<u, v> = 1) of
    B_kk(g_lin,k).
 2. Quotient matrices: uniform E^UN_kl = (1/n_k) sum_{i in k, j in l} E_ij (I^UN the same), Perron-weighted
    E^PW_kl = <u^k, E_kl v^l>, I^PW_kl = <u^k, I_kl v^l>, with v^k, u^k the right/left Perron vectors of E_kk, v^k mean 1 over
    module k and <u^k, v^k> = 1 (theory_notes.md 1.2).
 3. Equitability error (Schaub et al. 2016): CV over i in k of sum_{j in l} E_ij (and of I), for every (k, l).
 4. r1*(k <- l) = E_kl / I_kl, r0*(k) = E_kk / I_kk, under both closures.
 5. Dead-zone screen (generalises neuron_dead_zone of pd3_theory_module2.py): for every ordered pair l -> k (k != l, any
    coupling) and r in {0, 0.5, 1, 2, 4, 6}, with y = sum_{j in l} (E_ij - r I_ij) v^l_j: the share of k's neurons with y > 0
    ('all'), the share among k's receivers (neurons with any input from l, 'recv') and the u^k-weighted share ('u').
    de Candia candidate pair: r1*_PW(k <- l) < r0*_PW(k) and a net-positive share < 5% at some r <= 6. Reported for the
    plain share (the plan's wording, which sparse pairs pass trivially), the strict version (u-weighted AND receiver
    shares both < 5% at the same r) and the 'meaningful' version (added after the first P0 run, where most strict
    candidates had a Perron-pattern drive of exactly 0 because v^l vanishes on parts of a reducible block): u-weighted share
    and share among receivers of l's Perron PATTERN both < 5%, coupling (E_PW_kl + I_PW_kl) >= 1e-3 Lambda_k, and both
    modules able to switch on alone below g = 30 (g_lin,k, g_lin,l < 30).
Outputs: results/pd3/modules/quotient_<P>.npz, deadzone_<P>.json, module rows merged into modules_table.json.
Usage: python pd3_module_stats.py P0|P1|P2
"""
import json, os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import brentq

C = "/MALECNS_CACHE"; ROOT = "/MALECNS_ROOT"
P = sys.argv[1]; M = 30
t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
dt = 0.2; a_eff = (1 - np.exp(-dt)) / dt
fdt = lambda x: (1 - np.exp(-np.tanh(np.clip(x, 0, None)) * dt)) / dt
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
sup = np.load(C + "/superclasses.npy", allow_pickle=True); sides = np.load(C + "/sides.npy", allow_pickle=True)
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
exc, inh = sign[pre] > 0, sign[pre] < 0
E = sp.csr_matrix((c * w[exc], (post[exc], pre[exc])), shape=(N, N)); E.sum_duplicates()
I = sp.csr_matrix((c * w[inh], (post[inh], pre[inh])), shape=(N, N)); I.sum_duplicates()
del pre, post, w
part = np.load(f"{ROOT}/results/pd3/modules/partition_{P}.npz", allow_pickle=True)
memb = part["membership"].astype(np.int64); names = [str(x) for x in part["names"]]; nmod = len(names)
log(f"{P}: {nmod} modules; E nnz {E.nnz}, I nnz {I.nnz}")


def perron(A, need_vec=True):
    """Perron root and (right v, left u) of a nonnegative (sparse or dense) square matrix"""
    n = A.shape[0]
    if n == 0 or (sp.issparse(A) and A.nnz == 0) or (not sp.issparse(A) and not A.any()):
        return 0.0, np.ones(n), np.ones(n)
    if n <= 400:
        D = A.toarray() if sp.issparse(A) else A
        if not need_vec:
            return float(np.linalg.eigvals(D).real.max()), None, None
        ev, V = np.linalg.eig(D); k = np.argmax(ev.real); v = np.abs(V[:, k].real)
        evl, U = np.linalg.eig(D.T); kl = np.argmax(evl.real); u = np.abs(U[:, kl].real)
        return float(ev[k].real), v, u
    A = sp.csr_matrix(A)
    def top(B):
        try:
            ev, V = sla.eigs(B, k=3, which="LR", tol=1e-9, maxiter=20000, v0=np.ones(n), ncv=min(n - 1, 40))
        except sla.ArpackNoConvergence as e:
            ev, V = e.eigenvalues, e.eigenvectors
        k = np.argmax(ev.real); return float(ev[k].real), np.abs(V[:, k].real)
    lam, v = top(A)
    if not need_vec:
        return lam, None, None
    _, u = top(A.T.tocsr())
    return lam, v, u


def block(A, idx):
    return A[idx][:, idx].tocsr()


def glin_K(Ekk, Lam):
    if Lam <= 0:
        return np.inf, np.nan
    def Bf(g):
        B = Ekk.copy().tocsr(); B.data = fdt(g * B.data); return B
    h = lambda lg: np.log(max(perron(Bf(np.exp(lg)), need_vec=False)[0], 1e-300)) - np.log(a_eff)
    lo = np.log(a_eff / Lam) - 1e-9
    if h(np.log(1e4)) < 0:
        return np.inf, np.nan
    hi = lo + 0.1
    while h(hi) < 0:
        hi += 0.5
    lg = brentq(h, lo, hi, xtol=1e-9, rtol=1e-7)
    g = float(np.exp(lg)); _, v, u = perron(Bf(g)); u = u / (u @ v)
    return g, float(1 / (((u ** 2) @ v) * (u @ (v ** 2))))


# ---------- 1. rank modules by Lambda_k ----------
cand = [k for k in range(nmod) if names[k] != "rest"]
size = np.bincount(memb, minlength=nmod)
members = [np.where(memb == k)[0] for k in range(nmod)]
Lam_all = np.zeros(nmod)
for k in cand:
    if size[k] >= 2:
        Lam_all[k] = perron(block(E, members[k]), need_vec=False)[0]
top = sorted(cand, key=lambda k: -Lam_all[k])[:M]
log("top modules:", [(names[k], int(size[k]), round(Lam_all[k], 3)) for k in top])
grp = np.full(N, M, np.int64)
for i, k in enumerate(top):
    grp[members[k]] = i
gnames = [names[k] for k in top] + ["rest"]
G = M + 1
gidx = [np.where(grp == i)[0] for i in range(G)]
n_g = np.array([len(x) for x in gidx])
rows = []
V = np.zeros((N, G)); U = np.zeros((N, G))
for i in range(G):
    Ekk = block(E, gidx[i])
    lam, v, u = perron(Ekk)
    v = v / v.mean(); u = u / (u @ v)
    V[gidx[i], i] = v; U[gidx[i], i] = u
    gl, K = glin_K(Ekk, lam)
    scm = {}
    for s_, cnt in zip(*np.unique(sup[gidx[i]], return_counts=True)):
        scm[str(s_) or "?"] = round(cnt / n_g[i], 4)
    scm = dict(sorted(scm.items(), key=lambda x: -x[1])[:4])
    sdm = {str(s_) or "?": round(cnt / n_g[i], 4) for s_, cnt in zip(*np.unique(sides[gidx[i]], return_counts=True))}
    rows.append({"partition": P, "idx": i, "name": gnames[i], "size": int(n_g[i]), "Lambda": lam,
                 "g_weak": a_eff / lam if lam > 0 else None, "g_lin": gl, "K": K,
                 "n_exc": int((sign[gidx[i]] > 0).sum()), "n_inh": int((sign[gidx[i]] < 0).sum()),
                 "superclass_mix": scm, "side_mix": sdm,
                 "PR_v": float(v.sum() ** 2 / (v ** 2).sum()), "frac_v_pos": float((v > 1e-12 * v.max()).mean())})
    log(f"{gnames[i]:8s} n={n_g[i]:6d} Lambda={lam:.4f} g_weak={a_eff/max(lam,1e-300):.4f} g_lin={gl:.4f} K={K:.1f}")

# ---------- 2-4. quotient matrices, equitability, r* ----------
OH = sp.csr_matrix((np.ones(N), (np.arange(N), grp)), shape=(N, G))
EO, IO = (E @ OH).toarray(), (I @ OH).toarray()             # input to each neuron from each group (all of it active)
EV, IV = E @ V, I @ V                                       # input from each group's Perron pattern
EUN = np.array([EO[gidx[k]].mean(0) for k in range(G)]); IUN = np.array([IO[gidx[k]].mean(0) for k in range(G)])
EPW = U.T @ EV; IPW = U.T @ IV


def cv(x):
    m = x.mean(); return float(x.std() / m) if m > 0 else np.nan


CVE = np.array([[cv(EO[gidx[k], l]) for l in range(G)] for k in range(G)])
CVI = np.array([[cv(IO[gidx[k], l]) for l in range(G)] for k in range(G)])
with np.errstate(divide="ignore", invalid="ignore"):
    r1_PW, r1_UN = EPW / IPW, EUN / IUN
r0_PW, r0_UN = np.diag(r1_PW).copy(), np.diag(r1_UN).copy()
for i in range(G):
    rows[i].update({"r0_PW": float(r0_PW[i]), "r0_UN": float(r0_UN[i]), "E_UN_kk": float(EUN[i, i]),
                    "I_UN_kk": float(IUN[i, i]), "E_PW_kk": float(EPW[i, i]), "I_PW_kk": float(IPW[i, i]),
                    "equit_CV_exc_self": float(CVE[i, i]), "equit_CV_inh_self": float(CVI[i, i])})

# ---------- 5. dead-zone screen ----------
RS = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
dz = []
for l in range(G):
    for k in range(G):
        if k == l:
            continue
        e, ii = EV[gidx[k], l], IV[gidx[k], l]
        recv = (EO[gidx[k], l] + IO[gidx[k], l]) > 0
        if not recv.any():
            continue
        uk = U[gidx[k], k]
        precv = (e + ii) > 1e-6                              # receivers of l's Perron PATTERN (v^l is 0 on parts of a block)
        row = {"k": gnames[k], "l": gnames[l], "ki": k, "li": l, "n_k": int(n_g[k]), "n_recv": int(recv.sum()),
               "n_pattern_recv": int(precv.sum()), "coupling_rel": float((EPW[k, l] + IPW[k, l]) / max(EPW[k, k], 1e-300)),
               "g_lin_k": rows[k]["g_lin"], "g_lin_l": rows[l]["g_lin"], "share_precv": {},
               "E_PW": float(EPW[k, l]), "I_PW": float(IPW[k, l]), "E_UN": float(EUN[k, l]), "I_UN": float(IUN[k, l]),
               "r1_PW": float(r1_PW[k, l]), "r0_PW_k": float(r0_PW[k]), "r1_UN": float(r1_UN[k, l]), "r0_UN_k": float(r0_UN[k]),
               "share_all": {}, "share_recv": {}, "share_u": {}}
        for r in RS:
            pos = (e - r * ii) > 0
            row["share_all"][f"{r:g}"] = float(pos.mean())
            row["share_recv"][f"{r:g}"] = float(pos[recv].mean())
            row["share_precv"][f"{r:g}"] = float(pos[precv].mean()) if precv.any() else np.nan
            row["share_u"][f"{r:g}"] = float(uk @ pos / uk.sum()) if uk.sum() > 0 else np.nan
        window = row["r1_PW"] < row["r0_PW_k"]
        rr = [r for r in RS if 0 < r <= 6]
        row["candidate_plain"] = bool(window and any(row["share_all"][f"{r:g}"] < 0.05 for r in rr))
        row["candidate_strict"] = bool(window and any(row["share_u"][f"{r:g}"] < 0.05 and row["share_recv"][f"{r:g}"] < 0.05
                                                      for r in rr))
        row["candidate_meaningful"] = bool(window and row["coupling_rel"] >= 1e-3 and row["n_pattern_recv"] > 0 and
                                           max(row["g_lin_k"], row["g_lin_l"]) < 30 and
                                           any(row["share_u"][f"{r:g}"] < 0.05 and row["share_precv"][f"{r:g}"] < 0.05 for r in rr))
        row["min_share_u_r<=6"] = float(min(row["share_u"][f"{r:g}"] for r in rr))
        row["min_share_recv_r<=6"] = float(min(row["share_recv"][f"{r:g}"] for r in rr))
        dz.append(row)
n_cand_plain = sum(r["candidate_plain"] for r in dz); n_cand_strict = sum(r["candidate_strict"] for r in dz)
n_cand_mean = sum(r["candidate_meaningful"] for r in dz)
log(f"dead-zone pairs {len(dz)}: candidates plain {n_cand_plain}, strict {n_cand_strict}")
summary = {"partition": P, "M": M, "n_pairs_coupled": len(dz), "n_candidate_plain": n_cand_plain,
           "n_candidate_strict": n_cand_strict, "n_candidate_meaningful": n_cand_mean,
           "candidates_meaningful": [(r["k"], r["l"], r["r1_PW"], r["r0_PW_k"], r["coupling_rel"], r["min_share_u_r<=6"],
                                      r["n_pattern_recv"], r["g_lin_k"], r["g_lin_l"]) for r in dz if r["candidate_meaningful"]],
           "n_pairs_r1_lt_r0_PW": int(sum(r["r1_PW"] < r["r0_PW_k"] for r in dz)),
           "candidates_strict": [(r["k"], r["l"], r["r1_PW"], r["r0_PW_k"], r["min_share_u_r<=6"], r["min_share_recv_r<=6"])
                                 for r in dz if r["candidate_strict"]],
           "candidates_plain_top20": sorted([(r["k"], r["l"], r["n_recv"], r["n_k"], r["r1_PW"], r["r0_PW_k"],
                                              r["min_share_u_r<=6"]) for r in dz if r["candidate_plain"]],
                                            key=lambda x: -abs(x[2]))[:20]}
json.dump({"summary": summary, "pairs": dz}, open(f"{ROOT}/results/pd3/modules/deadzone_{P}.json", "w"), indent=1)
np.savez_compressed(f"{ROOT}/results/pd3/modules/quotient_{P}.npz", names=np.array(gnames), n=n_g, top_ids=np.array(top),
                    E_UN=EUN, I_UN=IUN, E_PW=EPW, I_PW=IPW, CV_E=CVE, CV_I=CVI, r1_PW=r1_PW, r1_UN=r1_UN,
                    Lambda=np.array([r["Lambda"] for r in rows]), g_lin=np.array([r["g_lin"] for r in rows]),
                    K=np.array([r["K"] for r in rows]), group=grp.astype(np.int16), alpha_eff=a_eff, c=c)
tp = f"{ROOT}/results/pd3/modules/modules_table.json"
tab = json.load(open(tp)) if os.path.exists(tp) else {}
tab[P] = {"rows": rows, "deadzone_summary": summary, "Lambda_all_candidates": {names[k]: float(Lam_all[k]) for k in cand
                                                                              if Lam_all[k] > 0 and size[k] >= 2} if P != "P1" else
          {"n_candidates": len(cand), "top50": sorted([(names[k], int(size[k]), float(Lam_all[k])) for k in cand], key=lambda x: -x[2])[:50]}}
tmp = tp + f".{P}.tmp"; json.dump(tab, open(tmp, "w"), indent=1); os.replace(tmp, tp)
log("saved quotient/deadzone/modules_table for", P)
