"""Unit tests for ald_sim.py on small graphs with exact / analytic expectations (2026-11-09).
Run on a GPU node (falls back to CPU): python ald_test.py [--quick] -> results/aldef/validation/unit_tests.json

 T1 drive only        isolated units with per-population drive h: rho = p_on/(p_on + p_off) (exact 2-state chain)
 T2 feed-forward      3 driven A units -> 1 B unit (motifs): rho_B from the exact 8-state joint Markov chain (n_A, b)
 T3 depression        driven A units with STD: E[x] and E[x | active] from the exact Markov-modulated linear recursion;
                      continuum x* = 1/(1 + U rho tau) reported for comparison (approximate: x and a are correlated)
 T4 adaptation        driven units with adaptation: E[theta] = D tau_a rho (exact identity); mean-field rho reported
 T5 refractory        rho = (1/p_off) / (1/p_off + n_ref + 1/p_on) (exact renewal formula)
 T6 clamp_on + STD    A clamped active, depressing, drives B: rho_B(t) from the exact deterministic-x recursion
 T7 real-cache logic  selectors, sign overrides, weight transforms (+renorm), NT sign probabilities and draws
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ald_sim as S

OUT = f"{S.ROOT}/results/aldef/validation"
dev = "cuda" if torch.cuda.is_available() else "cpu"
QUICK = "--quick" in sys.argv
DT, BETA = 0.2, 1.0
P_OFF = 1 - np.exp(-DT)
results = {}


def pon(s):
    return 1 - np.exp(-BETA * np.tanh(np.maximum(s, 0)) * DT)


def csr(rows, cols, vals, N):
    if len(rows) == 0:
        rows, cols, vals = [0], [0], [0.0]
    i = torch.tensor(np.vstack([rows, cols]), device=dev)
    return torch.sparse_coo_tensor(i, torch.tensor(np.asarray(vals, np.float32), device=dev), (N, N)).coalesce().to_sparse_csr()


def regions_of(masks, names):
    return {"names": names, "Rm": torch.tensor(np.array(masks), dtype=torch.float32, device=dev)}


def second_half(res, key, g=0):
    t = res["t"]; T = t[-1]
    return res[key][g][t > T / 2]


def report(name, ok, **kw):
    results[name] = {"pass": bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v)
                                          for k, v in kw.items()}}
    print(f"{'PASS' if ok else 'FAIL'} {name}: " + ", ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def chain_tau(lam):
    return (1 + lam) / (1 - lam)


T_LONG = 600.0 if QUICK else 2000.0
BB = 256 if QUICK else 512
t0 = time.time()

# ---------------------------------------------------------------------------------------------------- T1 drive only
hs = [0.01, 0.1, 1.0]; n = 300; N = n * len(hs)
hv = np.repeat(hs, n).astype(np.float64)
masks = [np.repeat(np.arange(len(hs)) == k, n) for k in range(len(hs))]
P = S.make_params(N, DT, dev, h=0.0, hv=hv)
res = S.simulate(csr([], [], [], N), N, [1.0], P, exp="quiet", T=T_LONG, dt=DT, B=BB, beta=BETA,
                 rng=np.random.default_rng(0), dev=dev, regions=regions_of(masks, [f"h{h}" for h in hs]))
occ = res["occ"][0].astype(np.float64)
for k, h in enumerate(hs):
    pa = pon(h); rho = pa / (pa + P_OFF)
    meas = occ[masks[k]].mean()
    nsamp = n * BB * (int(round(T_LONG / DT)) // 2) / chain_tau(1 - pa - P_OFF)
    se = np.sqrt(rho * (1 - rho) / nsamp)
    tol = 5 * se + 1e-3 * rho
    report(f"T1_drive_h{h}", abs(meas - rho) < tol, expected=rho, measured=meas, se=se, z=(meas - rho) / se)

# ---------------------------------------------------------------------------------------------- T5 refractory
for nref in [1, 3]:
    h = 0.5; pa = pon(h)
    P = S.make_params(n, DT, dev, h=h, nref=nref)
    res = S.simulate(csr([], [], [], n), n, [1.0], P, exp="quiet", T=T_LONG, dt=DT, B=BB, beta=BETA,
                     rng=np.random.default_rng(1), dev=dev)
    meas = res["occ"][0].astype(np.float64).mean()
    rho = (1 / P_OFF) / (1 / P_OFF + nref + 1 / pa)
    rho0 = pa / (pa + P_OFF)
    se = np.sqrt(rho * (1 - rho) / (n * BB * (int(round(T_LONG / DT)) // 2) / 15))
    report(f"T5_refractory_n{nref}", abs(meas - rho) < 5 * se + 1e-3 * rho, expected=rho, measured=meas,
           no_refractory=rho0, se=se)

# -------------------------------------------------------------------------------------------- T2 feed-forward
k_in, nm, hA, wAB = 3, 500, 0.3, 0.5
N = nm * (k_in + 1)
A = np.arange(nm * k_in); Bn = nm * k_in + np.arange(nm)
rows = np.repeat(Bn, k_in); cols = A
hv = np.zeros(N); hv[A] = hA
P = S.make_params(N, DT, dev, h=0.0, hv=hv)
mA = np.zeros(N, bool); mA[A] = True
res = S.simulate(csr(rows, cols, np.full(len(rows), wAB), N), N, [1.0], P, exp="quiet", T=T_LONG, dt=DT, B=BB,
                 beta=BETA, rng=np.random.default_rng(2), dev=dev, regions=regions_of([mA, ~mA], ["A", "B"]))
occ = res["occ"][0].astype(np.float64)
pa = pon(hA)
from scipy.stats import binom
TA = np.zeros((k_in + 1, k_in + 1))           # n -> n' for k_in independent 2-state units
for n0 in range(k_in + 1):
    for stay in range(n0 + 1):                 # active units that stay active
        for new in range(k_in - n0 + 1):       # inactive units that switch on
            TA[n0, stay + new] += binom.pmf(stay, n0, 1 - P_OFF) * binom.pmf(new, k_in - n0, pa)
M = np.zeros((2 * (k_in + 1), 2 * (k_in + 1)))
for n0 in range(k_in + 1):
    pb = pon(1.0 * wAB * n0)
    for b in (0, 1):
        tb = {1: (P_OFF, 1 - P_OFF), 0: (1 - pb, pb)}[b]
        for n1 in range(k_in + 1):
            for b1 in (0, 1):
                M[2 * n0 + b, 2 * n1 + b1] = TA[n0, n1] * tb[b1]
ev, evec = np.linalg.eig(M.T)
i0 = np.argmin(abs(ev - 1)); pi = np.real(evec[:, i0]); pi /= pi.sum()
rhoB = pi[1::2].sum()
lam2 = sorted(abs(ev))[-2]
meas = occ[Bn].mean()
se = np.sqrt(rhoB * (1 - rhoB) * chain_tau(lam2) / (nm * BB * (int(round(T_LONG / DT)) // 2)))
mf_pon =sum((pi[2 * n0] + pi[2 * n0 + 1]) * pon(wAB * n0) for n0 in range(k_in + 1))
naive = mf_pon / (mf_pon + P_OFF)
report("T2_feedforward_rhoB", abs(meas - rhoB) < 5 * se + 1e-3 * rhoB, expected_exact=rhoB, measured=meas, se=se,
       rhoA_meas=occ[A].mean(), rhoA_exact=pa / (pa + P_OFF), naive_mean_pon=naive)

# -------------------------------------------------------------------------------------------- T3 depression
nA, hA, U, tau = 1000, 0.3, 0.5, 10.0
pa = pon(hA)
P = S.make_params(nA, DT, dev, h=hA, std_spec=(np.full(nA, U), np.full(nA, tau)))
res = S.simulate(csr([], [], [], nA), nA, [1.0], P, exp="quiet", T=T_LONG, dt=DT, B=BB, beta=BETA,
                 rng=np.random.default_rng(3), dev=dev, regions=regions_of([np.ones(nA, bool)], ["A"]))
x_meas = second_half(res, "res_x")[:, 0].mean()
xa_meas = second_half(res, "res_x_act")[:, 0].mean()
rho_meas = res["occ"][0].astype(np.float64).mean()
k1 = 1 / tau + U; xinf = (1 / tau) / k1; ea, eq = np.exp(-k1 * DT), np.exp(-DT / tau)
cA = {0: (1.0 * (1 - eq), eq), 1: (xinf * (1 - ea), ea)}     # x' = c + e x
Ta = np.array([[1 - pa, pa], [P_OFF, 1 - P_OFF]])            # a -> a'
piA = np.array([P_OFF, pa]) / (pa + P_OFF)
# fixed point of m_{a'} = sum_a Ta[a, a'] (c_a pi_a + e_a m_a)
Mm = np.array([[Ta[a, a1] * cA[a][1] for a in (0, 1)] for a1 in (0, 1)])
bm = np.array([sum(Ta[a, a1] * cA[a][0] * piA[a] for a in (0, 1)) for a1 in (0, 1)])
m = np.linalg.solve(np.eye(2) - Mm, bm)
x_ex, xa_ex = m.sum(), m[1] / piA[1]
x_cont = 1 / (1 + U * piA[1] * tau)
report("T3_depression_mean_x", abs(x_meas - x_ex) < 2e-3, expected_exact=x_ex, measured=x_meas,
       continuum_1_over_1_plus_U_rho_tau=x_cont, rho_meas=rho_meas, rho_exact=piA[1])
report("T3_depression_x_given_active", abs(xa_meas - xa_ex) < 3e-3, expected_exact=xa_ex, measured=xa_meas)
# U = 0 group leaves x at 1 (per-group parameters)
P = S.make_params(nA, DT, dev, h=hA, std_spec=(np.r_[np.full(nA // 2, U), np.zeros(nA - nA // 2)],
                                               np.r_[np.full(nA // 2, tau), np.full(nA - nA // 2, np.inf)]))
mh = np.arange(nA) < nA // 2
res = S.simulate(csr([], [], [], nA), nA, [1.0], P, exp="quiet", T=T_LONG / 4, dt=DT, B=64, beta=BETA,
                 rng=np.random.default_rng(4), dev=dev, regions=regions_of([mh, ~mh], ["dep", "nodep"]))
r = second_half(res, "res_x")
report("T3_group_params", abs(r[:, 0].mean() - x_ex) < 5e-3 and np.all(np.isnan(r[:, 1])),
       dep_x=r[:, 0].mean(), nodep_x="NaN (no depressing neurons in region) as documented")

# -------------------------------------------------------------------------------------------- T4 adaptation
nA, h, D, ta = 1000, 0.5, 0.05, 20.0
P = S.make_params(nA, DT, dev, h=h, adapt_spec=(np.full(nA, D), np.full(nA, ta)))
res = S.simulate(csr([], [], [], nA), nA, [1.0], P, exp="quiet", T=T_LONG, dt=DT, B=BB, beta=BETA,
                 rng=np.random.default_rng(5), dev=dev, regions=regions_of([np.ones(nA, bool)], ["A"]))
th_meas = second_half(res, "res_theta")[:, 0].mean()
rho_meas = res["occ"][0].astype(np.float64).mean()
lo, hi = 0.0, 1.0
for _ in range(60):                      # mean-field self-consistency rho = p_on(h - D ta rho)/(p_on + p_off)
    mid = (lo + hi) / 2; pp = pon(h - D * ta * mid)
    lo, hi = (mid, hi) if pp / (pp + P_OFF) > mid else (lo, mid)
report("T4_adaptation_theta_identity", abs(th_meas - D * ta * rho_meas) < 3e-3 * D * ta * rho_meas + 2e-4,
       theta_meas=th_meas, D_tau_rho=D * ta * rho_meas, rho_meas=rho_meas, rho_meanfield=lo,
       rho_no_adapt=pon(h) / (pon(h) + P_OFF))
report("T4_adaptation_meanfield", abs(rho_meas - lo) < 0.02 * lo, rho_meas=rho_meas, rho_meanfield=lo,
       note="approximate (theta fluctuations neglected); 2% tolerance")

# ------------------------------------------------------------------------------------- T6 clamp_on + depression
nm, wAB, g, U, tau = 2000, 1.2, 1.0, 0.3, 5.0
N = 2 * nm; A = np.arange(nm); Bn = nm + A
on = np.zeros(N, bool); on[A] = True
Uv = np.zeros(N); Uv[A] = U; Tv = np.full(N, np.inf); Tv[A] = tau
P = S.make_params(N, DT, dev, h=0.0, std_spec=(Uv, Tv), on=on)
T6 = 40.0
res = S.simulate(csr(Bn, A, np.full(nm, wAB), N), N, [g], P, exp="decay", T=T6, dt=DT, B=256, beta=BETA,
                 rng=np.random.default_rng(6), dev=dev, regions=regions_of([on, ~on], ["A", "B"]))
steps = int(round(T6 / DT)); recs = np.unique(np.round(np.logspace(0, np.log10(steps), 120)).astype(int))
k1 = 1 / tau + U; xinf = (1 / tau) / k1; ea = np.exp(-k1 * DT)
x, rB, traj = 1.0, 1.0, {}
for s in range(1, steps + 1):
    rB = rB * (1 - P_OFF) + (1 - rB) * pon(g * wAB * x)      # step s uses x before its update
    x = xinf + (x - xinf) * ea
    traj[s] = rB
exact = np.array([traj[s] for s in recs])
measB = res["reg_rho"][0][:, 1, :].mean(1)
se = np.sqrt(exact * (1 - exact) / (nm * 256))
z = (measB - exact) / se
xA = res["res_x"][0][:, 0]
x_exact = np.array([xinf + (1 - xinf) * ea ** s for s in recs])
report("T6_clamp_on_std_trajectory", np.abs(z).max() < 5 and np.allclose(xA, x_exact, atol=1e-5),
       max_abs_z=np.abs(z).max(), rhoB_final_exact=exact[-1], rhoB_final_meas=measB[-1],
       x_final=xA[-1], x_final_exact=x_exact[-1], A_always_active=float(res["reg_rho"][0][:, 0, :].min()))

# ------------------------------------------------------------------------------- T7 real-cache logic (no dynamics)
sign = np.load(S.C + "/sign.npy"); Nc = len(sign)
L = S.Labels(Nc, sign=sign)
cnt = {s: int(S.select(s, L).sum()) for s in ["lLN1_bc", "olfactory", "module", "al", "al_rest", "cns_rest",
                                              "type:lLN2.*", "ALLN&sign:+1", "class:olfactory&~type:ORN_(VA4|VC1|VC2|VA7l|VM7d|VM7v)"]}
part = S.select("module", L).astype(int) + S.select("al_rest", L) + S.select("cns_rest", L)
report("T7_selectors", cnt["lLN1_bc"] == 30 and cnt["olfactory"] == 2639 and cnt["module"] == 526 and np.all(part == 1),
       **{f"n[{k}]": v for k, v in cnt.items()})
try:
    S.select("lLN1_bcX", L); bad = False
except ValueError:
    bad = True
report("T7_selector_empty_is_error", bad)
m = S.select("lLN1_bc", L)
sig = S.sigma_vector(sign, 2.0, [(m, -1.0)])
sig0 = S.sigma_vector(sign, 2.0, [(m, 0.0)])
base = S.sigma_vector(sign, 2.0)
report("T7_sign_override", np.all(sig[m] == -2.0) and np.all(sig0[m] == 0) and np.all(sig[~m] == base[~m])
       and np.all(base[m] == 1.0), lLN1_bc_consensus=float(base[m].mean()), override_m1_r2=float(sig[m].mean()))
pre = np.load(S.C + "/pre.npy").astype(np.int64); post = np.load(S.C + "/post.npy").astype(np.int64)
w = np.load(S.C + "/weight.npy").astype(np.float32)
ins_lin = np.bincount(post, weights=w, minlength=Nc); has = ins_lin > 0
tr = {}
for spec in ["linear", "cap:20", "sqrt", "log1p", "binary"]:
    wT = S.wtransform(w, spec)
    for wn in ["renorm", "none"]:
        sc = S.node_scale(post, w, wT, Nc, "raw", wn)
        tr[f"{spec}/{wn}"] = float((np.bincount(post, weights=wT, minlength=Nc) * sc)[has].mean())
ok = all(abs(tr[f"{s}/renorm"] - 1) < 1e-5 for s in ["linear", "cap:20", "sqrt", "log1p", "binary"])
ok &= float(S.wtransform(np.array([0.5, 3, 50], np.float32), "cap:20").max()) == 20.0
report("T7_wtransform_renorm_mean_instrength", ok, **{f"mean_in[{k}]": v for k, v in tr.items()})
tab = S.nt_table()
Pr, q = S.sign_probs(tab, "body", "ignore", 0.5)
P1, _ = S.sign_probs({**tab, "pconf": np.ones(Nc), "ctconf": np.ones(Nc)}, "body", "ignore", 1.0)
mode_ok = np.all(np.array([1, -1, 0])[P1.argmax(1)] == np.sign(sign))
mt = L.get("type") == "lLN1_bc"
draws = S.draw_signs(Pr, 20, np.random.default_rng(0))
freq = np.array([(draws == v).mean(0) for v in (1, -1, 0)]).T
dev_freq = np.abs(freq.mean(0) - Pr.mean(0)).max()
td = S.draw_signs(Pr, 50, np.random.default_rng(1), "type", L.get("type"))
same_type = all(len(np.unique(td[k][mt])) == 1 for k in range(50))
Pg, _ = S.sign_probs(tab, "body", "fix", 0.5)
gt_has = tab["gt"] != ""
report("T7_nt_sign_probs", np.allclose(Pr.sum(1), 1) and mode_ok and dev_freq < 2e-3 and same_type
       and np.all(Pg[gt_has].max(1) == 1),
       rows_sum_to_1=bool(np.allclose(Pr.sum(1), 1)), q1_reproduces_consensus=bool(mode_ok),
       lLN1_bc_P_mean=Pr[mt].mean(0).round(4).tolist(), lLN1_bc_q_mean=float(q[mt].mean()),
       all_P_mean=Pr.mean(0).round(4).tolist(), draw_freq_max_dev=dev_freq,
       type_unit_lLN1_bc_flip_frac=float((td[:, mt][:, 0] != 1).mean()),
       neuron_unit_lLN1_bc_flip_frac=float((draws[:, mt] != 1).mean()), gt_fix_one_hot=bool(np.all(Pg[gt_has].max(1) == 1)),
       n_with_ground_truth=int(gt_has.sum()))

results["_meta"] = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
                    "quick": QUICK, "B": BB, "T": T_LONG, "wall_s": time.time() - t0,
                    "n_pass": sum(v["pass"] for k, v in results.items() if not k.startswith("_")),
                    "n_tests": sum(1 for k in results if not k.startswith("_"))}
os.makedirs(OUT, exist_ok=True)
json.dump(results, open(f"{OUT}/unit_tests{'_quick' if QUICK else ''}.json", "w"), indent=1, default=str)
print(json.dumps(results["_meta"]), flush=True)
