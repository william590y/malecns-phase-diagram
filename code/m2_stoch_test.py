"""Tests of m2_stoch_core (CPU torch, small graphs): single-unit formulas, QMF fixed point vs exact simulation, LNA
variance / ACF / onset-count covariances vs exact stochastic simulation of the same synchronous model (ald_sim.py rules,
spike-mode depression), chain solvers (logistic recursion vs 2-D solver in the frozen limit).
Writes results/m2/theory/stochastic/test/tests.json."""
import json, sys, time
import numpy as np
sys.path.insert(0, "/MALECNS_ROOT/code")
import m2_stoch_core as S
import torch

OUT = "/MALECNS_ROOT/results/m2/theory/stochastic/test"
res = {}
rng = np.random.default_rng(1)

# ---------------------------------------------------------------- T1 single unit: exact stationary O/rho = renewal formula
q = np.exp(-S.DT / 157.0)
rr = np.array([1e-4, 1e-2, 0.04, 0.2])
p = S.P_OFF * rr / (1 - rr)
_, O, _, _ = S.unit_stationary(p, 0.25, q)
res["T1_unit_vs_renewal_maxdiff"] = float(np.abs(O / rr - S.eff_spike(rr, 0.25, 157.0)).max())

# ---------------------------------------------------------------- small graph
N = 400
pre = rng.integers(0, N, 40 * N); post = rng.integers(0, N, 40 * N)
w = rng.integers(1, 6, len(pre)).astype(float)
sign = np.where(rng.random(N) < 0.75, 1, -1).astype(np.int8)
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
gfile = f"{OUT}/small.npz"
np.savez(gfile, pre=pre, post=post, w=w.astype(np.float32), sign=sign, sub=np.arange(N), c=c, kind="test")
net = S.Net(gfile, dev="cpu", model="M2")
net.set_model("M2")
# make the depression faster so a short simulation samples many resource cycles
TAU = 20.0
net.TAU_np[:] = np.where(net.dep_node, TAU, np.inf)
qq = np.where(net.dep_node, np.exp(-S.DT / TAU), 1.0)
net.q = torch.tensor(qq, dtype=torch.float64)
g, r = 0.9, 0.5
hv = torch.full((N,), 0.08, dtype=torch.float64)
st = S.qmf_solve(net, g, r, hv, iters=20000, tol=1e-13)
res["qmf_it"] = st["it"]; res["qmf_res"] = st["res"]

# exact simulation (ald_sim rules), numpy
Ex = net.Ex_sp.toarray(); Ih = net.Ih_sp.toarray()
import scipy.sparse as sps
W = sps.csr_matrix(g * (Ex - r * Ih))
B = 600; T_burn = 1500; T_rec = 12000
U = net.U_np; qv = qq; dep = net.dep_node
X = (rng.random((N, B)) < 0.1).astype(float); ef = np.ones((N, B)); xr = np.ones((N, B))
h = 0.08
tot = []; onsN = []
for t in range(T_burn + T_rec):
    Y = X * np.where(dep[:, None], ef, 1.0)
    s = W @ Y + h
    pon = 1 - np.exp(-np.tanh(np.clip(s, 0, None)) * S.DT)
    u = rng.random((N, B))
    Xn = np.where(X > 0, (u >= S.P_OFF).astype(float), (u < pon).astype(float))
    ons = (X == 0) & (Xn > 0)
    xr = 1 + (xr - 1) * qv[:, None]
    ef = np.where(ons, xr, ef)
    xr = np.where(ons & dep[:, None], xr * (1 - U[:, None]), xr)
    X = Xn
    if t >= T_burn:
        tot.append(X.mean(0)); onsN.append(ons.sum(0))
tot = np.array(tot); onsN = np.array(onsN, float)
res["sim_mean_rho"] = float(tot.mean()); res["qmf_mean_rho"] = float(st["rho"].mean())
res["sim_var_rho"] = float(tot.var(0).mean())
lna = S.LNA(net, g, r, st)
cvec = lna.obs_activity([np.ones(N, bool)])
out = lna.run(cvec, maxlag=200, ckpt=100)
res["lna_var_rho"] = float(out["var"][0]); res["lna_S"] = out["S"]
# ACF comparison at a few lags
xc = tot - tot.mean(0)
def acf_sim(x, ks):
    return [float((x[k:] * x[:len(x) - k]).mean() / (x * x).mean()) for k in ks]
ks = [1, 5, 20, 50, 100]
res["acf_sim"] = acf_sim(xc, ks)
res["acf_lna"] = [float(out["acf"][k, 0] / out["acf"][0, 0]) for k in ks]
# onset counts
con, qon, varb = lna.obs_onsets(np.ones(N, bool))
o2 = lna.run(con, maxlag=40, ckpt=100, qon=qon)
var_on = float(o2["acf"][0, 0]) + varb
cov_on = np.r_[var_on, o2["acf"][1:, 0] + o2["xq"][1:, 0]]
oc = onsN - onsN.mean(0)
res["on_mean_sim"] = float(onsN.mean()); res["on_mean_qmf"] = float(((1 - st["rho"]) * st["p"]).sum())
res["on_var_sim"] = float(oc.var(0).mean()); res["on_var_lna"] = var_on
res["on_cov_sim"] = [float((oc[k:] * oc[:len(oc) - k]).mean()) for k in (1, 3, 10, 20)]
res["on_cov_lna"] = [float(cov_on[k]) for k in (1, 3, 10, 20)]
m_l, _ = S.mr_from_cov(cov_on, var_on)
import m2_obs as MO
res["mr_sim"] = MO.mr_branching(onsN, 20)["m"]; res["mr_lna"] = m_l

# ---------------------------------------------------------------- chains: 2-D solver in the frozen limit (TAU huge, Ny huge)
lam, K = 1.3 * S.A_EFF, 30.0
lt1 = S.log_tau_logistic_qs(lambda n: lam * np.ones_like(n), K, K * (1 - S.A_EFF / lam))
t2, _ = S.tau_2d(lam, K, 1e12, 0.25, 1e12, K * (1 - S.A_EFF / lam), 0.999999, M=40)
res["chain_logistic_lt"] = lt1; res["chain_2d_frozen_lt"] = float(np.log(t2))
# adiabatic limit: TAU small with U*TAU/Ny fixed -> compare with 1-D chain b = lam y_ad(n) n (1 - n/K)
lam_r, K, ns, X = 3.0 * S.A_EFF, 60.0, 12.0, 0.45
Ny = 0.25 * 157 * S.A_EFF * X * ns / (1 - X)
lt_ad = S.log_tau_logistic_qs(lambda n: lam_r / (1 + 0.25 * 157 * S.A_EFF * n / Ny), K, ns)
ts = []
for TAUx in (157.0, 20.0, 2.0, 0.2, 0.02):
    Nyx = 0.25 * TAUx * S.A_EFF * X * ns / (1 - X)
    tt, _ = S.tau_2d(lam_r, K, Nyx, 0.25, TAUx, ns, X, M=300)
    ts.append(float(np.log(tt)))
res["chain_adiabatic_lt"] = lt_ad; res["chain_2d_TAU_157_20_2_0.2_0.02_lt"] = ts
json.dump(res, open(f"{OUT}/tests.json", "w"), indent=1)
print(json.dumps(res, indent=1))
