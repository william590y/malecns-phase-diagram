"""Parameters of the two-variable caricature used for the flow-field animation (explainer/flow.html).
x = activity fraction of the antennal-lobe module M (union of the supports of the two leading excitatory modes),
y = activity fraction of the rest of the brain B.
  dx/dt = alpha[(R_M - 1) x - R_M (1 + delta(r)) x^2] + g e_MB y (1 - x),   R_M = g lambda_M / alpha
  dy/dt = alpha[(R_B - 1) y - R_B (1 + delta(r)) y^2] + g e_BM x (1 - y),   R_B = g lambda_B(r) / alpha
lambda_M = Perron root of E restricted to M; e_BM = mean excitatory input to a neuron of B from a fully active M;
e_MB = Perron-weighted mean input to M from a fully active B; delta(r) = K_eff(0)/K_eff(r) - 1 from theory_branch (coincident
inhibition); lambda_B(r) = alpha_eff / g_D(r), with g_D(r) the QMF line where the activity's participation ratio reaches N/4
(mean field is the right description of dense activity, theory page section 3). Noise on x from K_eff(r).
Writes results/pd/flow_params.json.
"""
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

C = "/MALECNS_CACHE"; R = "/MALECNS_ROOT/results/pd"
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
exc = sign[pre] > 0
E = sp.csr_matrix((c * w[exc], (post[exc], pre[exc])), shape=(N, N))
t = np.load(f"{R}/tail_real_r0_dir.npz", allow_pickle=True)
M = np.zeros(N, bool)
for k in range(2):
    p = t["P"][:, k].astype(np.float64); o = np.argsort(-p); M[o[: np.searchsorted(np.cumsum(p[o]) / p.sum(), 0.99) + 1]] = True
B = ~M
EMM = E[M][:, M]; lamM, vM = sla.eigs(EMM, k=1, which="LR"); lamM = lamM[0].real; vM = np.abs(vM[:, 0].real)
inB = np.asarray(E[B][:, M].sum(1)).ravel()                  # input to each B neuron from all of M active
inM = np.asarray(E[M][:, B].sum(1)).ravel()                  # input to each M neuron from all of B active
a_eff = (1 - np.exp(-0.2)) / 0.2
q = np.load(f"{R}/qmf_real.npz"); G = q["g"]; rs = q["r"]; PR = q["PR"][:, 0, :]
gD = {}
for i, r in enumerate(rs):
    k = np.argmax(PR[i] >= 0.25) if (PR[i] >= 0.25).any() else None
    gD[f"{r:g}"] = float(G[k]) if k is not None else None
b = json.load(open(f"{R}/branch_real.json"))["variants"]["branching"]
rows = b["rows"]; j = int(np.argmin([abs(x["R"] - 2) for x in rows]))
delta = {r: rows[j]["K_r0"] / rows[j][f"K_r{r}"] - 1 for r in ["0", "0.25", "0.5", "1", "1.5", "2", "3", "4", "6"]}
Keff = {r: rows[j][f"K_r{r}"] for r in delta}
out = {"n_M": int(M.sum()), "lambda_M": float(lamM), "e_BM": float(inB.mean()), "e_BM_p99": float(np.percentile(inB, 99)),
       "e_MB": float((vM * inM).sum() / vM.sum()), "g_lin": b["g_lin"], "alpha_eff": a_eff, "g_D_qmf": gD,
       "delta": delta, "K_eff": Keff, "g_on_theory": b["g_on"]}
json.dump(out, open(f"{R}/flow_params.json", "w"), indent=1)
print(json.dumps(out, indent=1))
