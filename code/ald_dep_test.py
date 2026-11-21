"""E3 unit tests for the --std_mode spike option of ald_sim.py (small graphs, exact expectations; 2026-11-09).
Usage: python ald_dep_test.py <path to ald_sim module file> -> results/aldef/depression/validation/unit_tests_e3.json

 S1 renewal    isolated driven units, spike-mode depression: E[e | active] = (1 - M)/(1 - (1 - U) M), with
               M = E[q^L] E[q^Q], q = e^{-dt/TAU}, L ~ Geom(p_off) and Q ~ Geom(p_on) on {1, 2, ...} (exact; the efficacy
               of an epoch is the resource at its onset and the epoch length is independent of it); activity rho is
               unchanged by depression (isolated unit, no input): rho = p_on/(p_on + p_off).
 S2 mean x     same run: E[x] over all steps from the exact Markov-modulated linear recursion (state, x).
 S3 rate mode  the old rate-mode T3 numbers (ald_test.py) are reproduced by the patched code (same expectations).
 S4 chain      A (driven, depressing, spike mode) -> B (one input, weight w): with U = 1 and TAU = inf only A's FIRST
               epoch transmits; B can switch on only while A is in that first epoch. P(B ever active) is exact:
               1 - E[(1 - p_B)^L], L ~ Geom(p_off) (A's first epoch length), p_B = 1 - exp(-tanh(g w) dt).
"""
import importlib.util, json, os, sys
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("ald_sim_x", sys.argv[1])
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
OUT = "/MALECNS_ROOT/results/aldef/depression/validation"
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
DT, BETA = 0.2, 1.0
P_OFF = 1 - np.exp(-DT)
res_all = {"module_file": sys.argv[1]}


def pon(s):
    return 1 - np.exp(-BETA * np.tanh(np.maximum(s, 0)) * DT)


def csr(rows, cols, vals, N):
    if len(rows) == 0:
        rows, cols, vals = [0], [0], [0.0]
    i = torch.tensor(np.vstack([rows, cols]), device=dev)
    return torch.sparse_coo_tensor(i, torch.tensor(np.asarray(vals, np.float32), device=dev), (N, N)).coalesce().to_sparse_csr()


def regions_of(masks, names):
    return {"names": names, "Rm": torch.tensor(np.array(masks), dtype=torch.float32, device=dev)}


def report(name, ok, **kw):
    res_all[name] = {"pass": bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v)
                                          for k, v in kw.items()}}
    print(f"{'PASS' if ok else 'FAIL'} {name}: " + ", ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def geom_pgf(p, q):
    """E[q^L], L ~ Geom(p) on {1, 2, ...}."""
    return p * q / (1 - (1 - p) * q)


T_LONG, BB = 2000.0, 256
# ---------------------------------------------------------------------------------------------- S1/S2 renewal
for (hA, U, tau) in [(0.3, 0.25, 30.0), (0.05, 0.5, 100.0), (1.0, 0.28, 240.0)]:
    nA = 1000
    pa = pon(hA)
    P = S.make_params(nA, DT, dev, h=hA, std_spec=(np.full(nA, U), np.full(nA, tau)), std_mode="spike")
    r = S.simulate(csr([], [], [], nA), nA, [1.0], P, exp="quiet", T=T_LONG, dt=DT, B=BB, beta=BETA,
                   rng=np.random.default_rng(11), dev=dev, regions=regions_of([np.ones(nA, bool)], ["A"]))
    t = r["t"]; late = t > t[-1] / 2
    e_meas = np.nanmean(r["res_x_act"][0][late][:, 0])
    x_meas = np.nanmean(r["res_x"][0][late][:, 0])
    rho_meas = r["occ"][0].astype(np.float64).mean()
    q = np.exp(-DT / tau)
    M = geom_pgf(P_OFF, q) * geom_pgf(pa, q)
    e_ex = (1 - M) / (1 - (1 - U) * M)
    # exact E[x] over all steps: state chain (a, x) with x' = c_a + e_a x for the a -> a' transition, depletion at onsets
    # step order in the code: x <- 1 + (x - 1) q (every step), then if onset (a = 0 -> a' = 1): x <- (1 - U) x
    Ta = np.array([[1 - pa, pa], [P_OFF, 1 - P_OFF]])
    piA = np.array([P_OFF, pa]) / (pa + P_OFF)
    lin = {(0, 0): (1 - q, q), (0, 1): ((1 - U) * (1 - q), (1 - U) * q), (1, 0): (1 - q, q), (1, 1): (1 - q, q)}
    Mm = np.array([[Ta[a, a1] * lin[(a, a1)][1] for a in (0, 1)] for a1 in (0, 1)])
    bm = np.array([sum(Ta[a, a1] * lin[(a, a1)][0] * piA[a] for a in (0, 1)) for a1 in (0, 1)])
    m = np.linalg.solve(np.eye(2) - Mm, bm)
    x_ex = m.sum()
    rho_ex = pa / (pa + P_OFF)
    tag = f"h{hA}_U{U}_tau{tau:g}"
    report(f"S1_spike_efficacy_{tag}", abs(e_meas - e_ex) < 3e-3, expected_exact=e_ex, measured=e_meas,
           poisson_meanfield=1 / (1 + U * (1 - np.exp(-DT)) / DT * rho_ex * tau))
    report(f"S1_spike_rho_unchanged_{tag}", abs(rho_meas - rho_ex) < 3e-3 * max(rho_ex, 0.01) + 5e-4,
           expected_exact=rho_ex, measured=rho_meas)
    report(f"S2_spike_mean_x_{tag}", abs(x_meas - x_ex) < 3e-3, expected_exact=x_ex, measured=x_meas)

# ---------------------------------------------------------------------------------------------- S3 rate mode unchanged
nA, hA, U, tau = 1000, 0.3, 0.5, 10.0
pa = pon(hA)
P = S.make_params(nA, DT, dev, h=hA, std_spec=(np.full(nA, U), np.full(nA, tau)))
r = S.simulate(csr([], [], [], nA), nA, [1.0], P, exp="quiet", T=T_LONG, dt=DT, B=BB, beta=BETA,
               rng=np.random.default_rng(3), dev=dev, regions=regions_of([np.ones(nA, bool)], ["A"]))
t = r["t"]; late = t > t[-1] / 2
x_meas = r["res_x"][0][late][:, 0].mean(); xa_meas = r["res_x_act"][0][late][:, 0].mean()
k1 = 1 / tau + U; xinf = (1 / tau) / k1; ea, eq = np.exp(-k1 * DT), np.exp(-DT / tau)
cA = {0: (1.0 * (1 - eq), eq), 1: (xinf * (1 - ea), ea)}
Ta = np.array([[1 - pa, pa], [P_OFF, 1 - P_OFF]]); piA = np.array([P_OFF, pa]) / (pa + P_OFF)
Mm = np.array([[Ta[a, a1] * cA[a][1] for a in (0, 1)] for a1 in (0, 1)])
bm = np.array([sum(Ta[a, a1] * cA[a][0] * piA[a] for a in (0, 1)) for a1 in (0, 1)])
m = np.linalg.solve(np.eye(2) - Mm, bm)
report("S3_rate_mode_mean_x", abs(x_meas - m.sum()) < 2e-3, expected_exact=m.sum(), measured=x_meas)
report("S3_rate_mode_x_given_active", abs(xa_meas - m[1] / piA[1]) < 3e-3, expected_exact=m[1] / piA[1], measured=xa_meas)
# first-epoch factor phi of the rate mode (documented in the docstring): Monte Carlo over epochs vs formula
for (U, tau) in [(0.25, 89.0), (0.34, 240.0)]:
    k = 1 / tau + U; xi = (1 / tau) / k
    phi = xi + (1 - xi) * (1 - np.exp(-DT)) / (1 - np.exp(-(1 + k) * DT))
    rng = np.random.default_rng(5); L = rng.geometric(P_OFF, 400000)
    n = np.arange(L.max()); xs = xi + (1 - xi) * np.exp(-k * n * DT); cs = np.cumsum(xs)
    mc = cs[L - 1].sum() / L.sum()
    report(f"S3_rate_first_epoch_phi_U{U}_tau{tau:g}", abs(mc - phi) < 2e-3, formula=phi, monte_carlo=mc)

# ---------------------------------------------------------------------------------------------- S4 chain: first epoch only
nA = 4000; hA = 0.02; g = 1.0; w = 0.4
N = 2 * nA
rows = np.arange(nA, 2 * nA); cols = np.arange(nA)                    # A_i -> B_i, weight w
hv = np.r_[np.full(nA, hA), np.zeros(nA)]
P = S.make_params(N, DT, dev, h=0.0, hv=hv, std_spec=(np.r_[np.ones(nA), np.zeros(nA)], np.r_[np.full(nA, np.inf), np.full(nA, np.inf)]),
                  std_mode="spike")
Bm = np.r_[np.zeros(nA, bool), np.ones(nA, bool)]
r = S.simulate(csr(rows, cols, np.full(nA, w), N), N, [g], P, exp="quiet", T=1500.0, dt=DT, B=64, beta=BETA,
               rng=np.random.default_rng(7), dev=dev, regions=regions_of([~Bm, Bm], ["A", "B"]), reg_last=True)
lastB = r["reg_last"][0][1]                                           # per replica: last time any B unit was active
# per-unit: need P(B_i ever active). With 64 replicas x nA units, use the region time-course: after A's first epochs are all
# over, B never switches on again; fraction of B ever active is not stored per unit, so use the plateau of the
# cumulative count instead: count B activations through occupancy? -> compare the steady LATE B activity with 0 and the
# early B occupancy integral with the exact expectation below.
pB = pon(g * w)
EL = 1 / P_OFF
# expected total B active time per B unit = P(B on during A's first epoch) * E[B epoch duration in steps] * DT, with the
# number of B onsets during A's first epoch: B can switch on, off and on again while A is still in its first epoch.
# exact: B onsets while A active form a 2-state chain; expected number of B activation steps over the whole run =
# sum over steps of P(B active). Simulate the exact 4-state (A_state, first_epoch_flag, B) chain numerically instead:
# states: A in {pre (never fired), first (in first epoch), post (after first epoch, any state; output 0)}; B in {0, 1}.
pA = pon(hA)
steps = int(round(1500.0 / DT))
pr = np.zeros((3, 2)); pr[0, 0] = 1.0
occB = np.zeros(steps)
for s in range(steps):
    nw = np.zeros((3, 2))
    for a in range(3):
        for b in range(2):
            p = pr[a, b]
            if p == 0:
                continue
            # B update uses A's output during this step (A in 'first' -> input g w)
            pb_on = pB if a == 1 else 0.0
            b_next = [(1, 1 - P_OFF), (0, P_OFF)] if b == 1 else [(1, pb_on), (0, 1 - pb_on)]
            a_next = [(1, pA), (0, 1 - pA)] if a == 0 else ([(1, 1 - P_OFF), (2, P_OFF)] if a == 1 else [(2, 1.0)])
            for a1, pa1 in a_next:
                for b1, pb1 in b_next:
                    nw[a1, b1] += p * pa1 * pb1
    pr = nw
    occB[s] = pr[:, 1].sum()
t = r["t"]; idx = np.round(t / DT).astype(int) - 1
sim_B = r["reg_rho"][0][:, 1, :].mean(1)
ex_B = occB[idx]
sel = (t > 5) & (t < 600)
se = np.sqrt(np.maximum(ex_B[sel], 1e-9) / (nA * 64))
z = (sim_B[sel] - ex_B[sel]) / se
report("S4_spike_U1_first_epoch_only_trajectory", np.abs(z).max() < 5.0, max_abs_z=float(np.abs(z).max()),
       n_times=int(sel.sum()), B_late_sim=float(sim_B[t > 1000].mean()), B_late_exact=float(occB[-1]),
       B_peak_exact=float(occB.max()))

res_all["all_pass"] = all(v["pass"] for k, v in res_all.items() if isinstance(v, dict))
json.dump(res_all, open(f"{OUT}/unit_tests_e3.json", "w"), indent=1)
print("ALL PASS" if res_all["all_pass"] else "SOME FAILED")
