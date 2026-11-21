"""Independent fact-check of the AL data-grounding summary (results/al_lit/data_grounding.md), 2026-11-09.

Written from scratch (does NOT import al_common.py or reuse any saved eigenvectors): rebuilds the r = 0 excitatory
operator M[post, pre] = c * w (presynaptic consensus-ACh neurons only; c = 1 / mean in-strength over neurons with
input, all edges) from the raw cache arrays, re-solves the spectrum, and recomputes a set of headline numbers:

  V1  c, lambda_1, lambda_2 of the whole-CNS operator (claimed 1/742.7, 5.574, 5.389)
  V2  module = union of 99% supports of right modes 1 and 2 (claimed n 526, 0 of 2,639 ORNs, NT ACh 366 / GABA 96 /
      unclear 32 / Glu 25, sides R 275 / L 248, 30 lLN1_bc carry 51% of mode 1, ORNs 3.3% of left mode 1, ORNs 39%
      of the module's excitatory input), plus overlap with the saved module_graph.npz
  V3  lLN1_bc clique (claimed 15 + 15, 420/420 same-side ordered pairs, median 204 syn, no cross-side edges,
      subgraph lambda 3.92, uniform-clique value 14 * 207 / 742.7)
  V4  exact decomposition lambda = sum u_i M_ij v_j / (u.v) (claimed LN>LN 64.7%, lLN1_bc>lLN1_bc 42.2%, PN>LN 16.8%,
      LN>PN 16.5%, all ORN edges < 1%)
  V5  161 of the 200 strongest excitatory 2-cycles (sqrt(w_ij w_ji), both neurons ACh) are AL<>AL
  V6  region Perron roots (claimed CX 1.28, MB 0.47 / binary 0.279, LH 0.49, VNC 1.70) and AL-core connection stats
      (mean 12.0 syn/conn, 4.9% conns >= 50 syn holding 46%, weighted reciprocity 0.51; MB 0.53)
  V7  whole-CNS perturbations (claimed cap50 3.339, binary 0.280 led by MB with AL 0.249, LN non-excitatory 1.879)
  V8  lLN1_bc ground_truth ACh 30/30, mean predicted_nt_confidence ~0.75; 'Prelim Roughly traced' 228 of 526 module
      neurons, all 30 lLN1_bc, 80% of mode 1

  V9  (`--sim`, run separately, cheap) simulated switch-on from the raw occupancy arrays of the existing sweeps
      (claimed: AL mode supports reach mean occupancy 0.05 at g 0.26 (r 0) / 0.339 (r 1); motor tier ~0.50 at r 0;
      abdominal VNC 0.63 / 2.28; activity share at g 0.30, r 0: AL 52%, MB 19%, LH 12%). Mode supports come from the
      saved tail_real_r0_dir.npz (their eigenvectors), whose eigenvalues the main run checks against its own solve.

Read only on the cache; writes results/al_lit/verify3.json (main) and verify3_sim.json (--sim).
"""
import json, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla

CACHE = "/MALECNS_CACHE"
RAW = "/MALECNS_RAW"
ROOT = "/MALECNS_ROOT"
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
R = {}

if "--sim" in sys.argv:
    cl = np.load(f"{CACHE}/classes.npy", allow_pickle=True).astype(str)
    ty = np.load(f"{CACHE}/types.npy", allow_pickle=True).astype(str)
    sup = np.load(f"{CACHE}/superclasses.npy", allow_pickle=True).astype(str)
    AL = np.isin(cl, ["olfactory", "hygrosensory", "thermosensory", "ALPN", "ALLN", "ALIN", "ALON"])
    MB = (np.isin(cl, ["Kenyon_Cell", "MBON", "DAN"]) | np.isin(ty, ["APL", "DPM"])) & ~AL
    LH = (sup == "cb_intrinsic") & (cl == "") & np.char.startswith(ty, "LH") & ~MB
    t = np.load(f"{ROOT}/results/pd/tail_real_r0_dir.npz", allow_pickle=True)
    sups = {}
    for k in [0, 1, 2, 4, 5]:                       # modes 1, 2, 3 (VNC leg), 5 (abdominal), 6
        p = t["P"][:, k].astype(np.float64); p /= p.sum(); o = np.argsort(-p)
        sups[f"mode{k + 1}"] = o[: np.searchsorted(np.cumsum(p[o]), 0.9) + 1]
    out = {"tail_lam_top6": np.round(t["lam"][:6], 4).tolist()}
    for r, files in {0: ["pd/decay_r0_real.npz", "pd2/onset_real_r0.npz", "pd2/deloc_real_r0.npz"],
                     1: ["pd/decay_r1_real.npz", "pd2/onset_real_r1.npz", "pd2/deloc_real_r1.npz"]}.items():
        G, O = [], []
        for f in files:
            z = np.load(f"{ROOT}/results/{f}", allow_pickle=True)
            G += list(np.round(z["g"], 4)); O += list(z["occ"].astype(np.float32))
        G = np.array(G); O = np.array(O)
        ug = np.unique(G)
        occ = np.array([O[G == g].mean(0) for g in ug])
        res = {}
        for k, s in sups.items():
            y = occ[:, s].mean(1)
            hit = np.where(y >= 0.05)[0]
            res[k] = float(ug[hit[0]]) if len(hit) else None
        i = np.argmin(np.abs(ug - 0.30)); a = occ[i]
        res["g_share"] = float(ug[i])
        res["share_AL_MB_LH"] = [round(float(a[m].sum() / a.sum()), 3) for m in (AL, MB, LH)]
        out[f"r{r}"] = res
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{ROOT}/results/al_lit/verify3_sim.json", "w"), indent=1)
    sys.exit(0)

pre = np.load(f"{CACHE}/pre.npy").astype(np.int64)
post = np.load(f"{CACHE}/post.npy").astype(np.int64)
w = np.load(f"{CACHE}/weight.npy").astype(np.float64)
nt = np.load(f"{CACHE}/nt.npy", allow_pickle=True).astype(str)
cl = np.load(f"{CACHE}/classes.npy", allow_pickle=True).astype(str)
sup = np.load(f"{CACHE}/superclasses.npy", allow_pickle=True).astype(str)
ty = np.load(f"{CACHE}/types.npy", allow_pickle=True).astype(str)
side = np.load(f"{CACHE}/sides.npy", allow_pickle=True).astype(str)
bid = np.load(f"{CACHE}/body_ids.npy")
N = len(nt)
# own sign from the consensus NT string (not the cached sign array)
ach = nt == "acetylcholine"
exc = ach[pre]
log("loaded", N, len(pre), "edges; duplicate (pre,post) pairs:",
    int(len(pre) - len(np.unique(pre * N + post))))

# own labels, following the rules written in data_grounding.md
AL = np.isin(cl, ["olfactory", "hygrosensory", "thermosensory", "ALPN", "ALLN", "ALIN", "ALON"])
ORN = cl == "olfactory"
HRNTRN = np.isin(cl, ["hygrosensory", "thermosensory"])
PN = cl == "ALPN"
LN = cl == "ALLN"
BC = ty == "lLN1_bc"
MB = (np.isin(cl, ["Kenyon_Cell", "MBON", "DAN"]) | np.isin(ty, ["APL", "DPM"])) & ~AL
CX = (cl == "CX") & ~AL & ~MB
LH = (sup == "cb_intrinsic") & (cl == "") & np.char.startswith(ty, "LH") & ~MB
VNC = np.char.startswith(sup, "vnc")

# ---- V1: operator and spectrum --------------------------------------------------------------------------------
ins = np.bincount(post, weights=w, minlength=N)
c = 1.0 / ins[ins > 0].mean()
R["V1"] = {"mean_in_strength": float(1 / c)}


def op(edge_mask, weights=None):
    ww = w[edge_mask] if weights is None else weights[edge_mask]
    return sp.csr_matrix((ww * c, (post[edge_mask], pre[edge_mask])), shape=(N, N))


def top(M, k=6):
    vals, vecs = sla.eigs(M, k=k, which="LR", tol=1e-9, maxiter=500000, ncv=40)
    o = np.argsort(-vals.real)
    return vals[o], vecs[:, o]


M = op(exc)
vals, V = top(M)
valsL, U = top(M.T.tocsr(), k=4)
R["V1"]["lambda_right_top6"] = [str(np.round(x, 4)) for x in vals]
R["V1"]["lambda_left_top4"] = [str(np.round(x, 4)) for x in valsL]
log("V1", R["V1"])

v1 = np.abs(V[:, 0].real); v2 = np.abs(V[:, 1].real); u1 = np.abs(U[:, 0].real)
P1 = v1 ** 2 / (v1 ** 2).sum(); P2 = v2 ** 2 / (v2 ** 2).sum(); PL = u1 ** 2 / (u1 ** 2).sum()


def supp(p, q=0.99):
    o = np.argsort(-p)
    k = np.searchsorted(np.cumsum(p[o]), q) + 1
    return set(o[:k].tolist())


mod_idx = np.array(sorted(supp(P1) | supp(P2)))
inmod = np.zeros(N, bool); inmod[mod_idx] = True
saved = np.load(f"{ROOT}/results/module/module_graph.npz")["idx"]
uc = lambda m: {str(k): int(v) for k, v in zip(*np.unique(m, return_counts=True))}
em = exc & inmod[post]
R["V2"] = {
    "n_module": int(inmod.sum()), "overlap_with_saved_module": int(np.isin(saved, mod_idx).sum()),
    "n_saved": int(len(saved)), "n_AL_in_module": int((AL & inmod).sum()),
    "ORN_total": int(ORN.sum()), "ORN_in_module": int((ORN & inmod).sum()),
    "HRN_TRN_in_module": int((HRNTRN & inmod).sum()), "PN_in_module": int((PN & inmod).sum()),
    "LN_in_module": int((LN & inmod).sum()),
    "lLN2_in_module_of_cns": [int((LN & np.char.startswith(ty, "lLN2") & inmod).sum()),
                              int((LN & np.char.startswith(ty, "lLN2")).sum())],
    "lLN1_in_module": int((LN & np.char.startswith(ty, "lLN1") & inmod).sum()),
    "nt_counts": uc(nt[inmod]), "side_counts": uc(side[inmod]),
    "lLN1_bc_n": int(BC.sum()), "lLN1_bc_share_right1": float(P1[BC].sum()),
    "lLN1_bc_share_left1": float(PL[BC].sum()),
    "lLN2_share_right1": float(P1[LN & np.char.startswith(ty, "lLN2")].sum()),
    "ORN_share_left1": float(PL[ORN].sum()), "ORN_DP1m_share_left1": float(PL[ty == "ORN_DP1m"].sum()),
    "ORN_share_of_module_exc_input": float(w[em & ORN[pre]].sum() / w[em].sum()),
    "mode1_side_R_L": [float(P1[side == "R"].sum()), float(P1[side == "L"].sum())],
    "mode2_side_R_L": [float(P2[side == "R"].sum()), float(P2[side == "L"].sum())],
}
log("V2", R["V2"])

# ---- V3: lLN1_bc clique ---------------------------------------------------------------------------------------
bc = np.where(BC)[0]
loc = -np.ones(N, np.int64); loc[bc] = np.arange(len(bc))
mb = BC[pre] & BC[post] & (pre != post)
W = np.zeros((len(bc), len(bc)))
np.add.at(W, (loc[post[mb]], loc[pre[mb]]), w[mb])
same = (side[bc][:, None] == side[bc][None, :]) & ~np.eye(len(bc), dtype=bool)
cross = side[bc][:, None] != side[bc][None, :]
ev = np.linalg.eigvals(W * c)
R["V3"] = {"sides": uc(side[bc]), "nt": uc(nt[bc]), "same_side_ordered_pairs": int(same.sum()),
           "same_side_connected": int((W[same] > 0).sum()), "cross_side_edges": int((W[cross] > 0).sum()),
           "median_syn_same_side": float(np.median(W[same])), "mean_syn_same_side": float(W[same].mean()),
           "autapses_lLN1_bc": int((BC[pre] & (pre == post)).sum()),
           "lambda_clique": float(ev.real.max()),
           "lambda_clique_with_autapses": float(np.linalg.eigvals(op(exc & BC[pre] & BC[post]).toarray()[np.ix_(bc, bc)]).real.max()),
           "uniform_clique_14w_c": float(14 * W[same].mean() * c)}
log("V3", R["V3"])

# ---- V4: exact decomposition --------------------------------------------------------------------------------
u = U[:, 0].real; v = V[:, 0].real
if u.sum() < 0: u = -u
if v.sum() < 0: v = -v
contrib = np.where(exc, u[post] * w * c * v[pre], 0.0) / (u @ v)
lam_chk = contrib.sum()
share = lambda m: float(contrib[m].sum() / lam_chk)
ORNany = ORN[pre] | ORN[post]
R["V4"] = {"lambda_check": float(lam_chk), "LN>LN": share(LN[pre] & LN[post]),
           "lLN1_bc>lLN1_bc": share(BC[pre] & BC[post]), "PN>LN": share(PN[pre] & LN[post]),
           "LN>PN": share(LN[pre] & PN[post]), "any edge touching an ORN": share(ORNany),
           "ORN>LN_exc_syn": float(w[exc & ORN[pre] & LN[post]].sum()),
           "ORN>PN_exc_syn": float(w[exc & ORN[pre] & PN[post]].sum())}
log("V4", R["V4"])

# ---- V5: strongest excitatory 2-cycles ------------------------------------------------------------------------
key = pre * N + post
o = np.argsort(key); ks = key[o]
rk = post * N + pre
p_ = np.searchsorted(ks, rk); p_[p_ >= len(ks)] = 0
hasrev = ks[p_] == rk
wrev = np.where(hasrev, w[o][p_], 0.0)
both = exc & ach[post] & hasrev & (pre < post)
bi = np.where(both)[0]
s2 = np.sqrt(w[bi] * wrev[bi])
t200 = bi[np.argsort(-s2)[:200]]
R["V5"] = {"n_exc_2cycles": int(len(bi)), "top200_AL<>AL": int((AL[pre[t200]] & AL[post[t200]]).sum()),
           "rank200_strength_sqrt": float(np.sort(s2)[::-1][199])}
log("V5", R["V5"])

# ---- V6: region Perron roots and connection stats -------------------------------------------------------------


def sub_lambda(nodes, tf=None):
    m = exc & nodes[pre] & nodes[post]
    idx = np.where(nodes)[0]
    lo = -np.ones(N, np.int64); lo[idx] = np.arange(len(idx))
    ww = w[m] if tf is None else tf(w[m])
    A = sp.csr_matrix((ww * c, (lo[post[m]], lo[pre[m]])), shape=(len(idx), len(idx)))
    if len(idx) < 3000:
        return float(np.linalg.eigvals(A.toarray()).real.max())
    return float(sla.eigs(A, k=3, which="LR", tol=1e-9, maxiter=500000, ncv=30)[0].real.max())


def stats(nodes):
    m = nodes[pre] & nodes[post]
    ww = w[m]
    return {"n": int(nodes.sum()), "syn_per_conn_mean": float(ww.mean()), "median": float(np.median(ww)),
            "frac_conn_ge50": float((ww >= 50).mean()), "frac_syn_in_ge50": float(ww[ww >= 50].sum() / ww.sum()),
            "weighted_reciprocity_sum_min_over_sum": float(np.minimum(ww, wrev[m]).sum() / ww.sum())}


ALcore = AL & ~ORN & ~HRNTRN
R["V6"] = {"perron": {"AL": sub_lambda(AL), "CX": sub_lambda(CX), "MB": sub_lambda(MB),
                      "MB_binary": sub_lambda(MB, np.ones_like), "LH": sub_lambda(LH), "VNC": sub_lambda(VNC),
                      "AL_binary": sub_lambda(AL, np.ones_like)},
           "stats": {"AL_core": stats(ALcore), "MB": stats(MB), "CX": stats(CX), "VNC": stats(VNC), "LH": stats(LH)}}
log("V6", R["V6"])

# ---- V7: whole-CNS perturbations --------------------------------------------------------------------------------


def lead(M, k=8):
    vals, vecs = top(M, k=k)
    out = []
    for j in range(k):
        p = np.abs(vecs[:, j]) ** 2; p /= p.sum()
        out.append({"lam": str(np.round(vals[j], 4)), "AL": round(float(p[AL].sum()), 3),
                    "MB": round(float(p[MB].sum()), 3), "VNC": round(float(p[VNC].sum()), 3)})
    return out


R["V7"] = {"cap50": lead(op(exc, np.minimum(w, 50.0)), k=4),
           "binary": lead(op(exc, np.ones_like(w)), k=8),
           "LN_nonexc": lead(op(exc & ~LN[pre]), k=8)}
log("V7", json.dumps(R["V7"], indent=0))

# ---- V8: raw-annotation checks -----------------------------------------------------------------------------------
ann = pd.read_feather(f"{RAW}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                      columns=["bodyId", "statusLabel"]).drop_duplicates("bodyId").set_index("bodyId").reindex(bid)
ntt = pd.read_feather(f"{RAW}/body-neurotransmitters-male-cns-v1.0.feather",
                      columns=["body", "ground_truth", "predicted_nt", "predicted_nt_confidence", "consensus_nt",
                               "celltype_predicted_nt"]).drop_duplicates("body").set_index("body").reindex(bid)
st = ann["statusLabel"].astype(str).values
prelim = st == "Prelim Roughly traced"
R["V8"] = {"lLN1_bc_ground_truth": pd.Series(ntt["ground_truth"].values[BC].astype(str)).value_counts().to_dict(),
           "lLN1_bc_predicted_nt": pd.Series(ntt["predicted_nt"].values[BC].astype(str)).value_counts().to_dict(),
           "lLN1_bc_celltype_predicted_nt": pd.Series(ntt["celltype_predicted_nt"].values[BC].astype(str)).value_counts().to_dict(),
           "lLN1_bc_mean_pred_conf": float(np.nanmean(ntt["predicted_nt_confidence"].values[BC].astype(float))),
           "prelim_in_module": int((prelim & inmod).sum()), "prelim_lLN1_bc": int((prelim & BC).sum()),
           "prelim_share_right1": float(P1[prelim].sum()), "prelim_frac_cns": float(prelim.mean())}
log("V8", R["V8"])

json.dump(R, open(f"{ROOT}/results/al_lit/verify3.json", "w"), indent=1, default=str)
log("saved results/al_lit/verify3.json")
