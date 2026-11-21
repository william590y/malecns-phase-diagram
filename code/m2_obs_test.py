"""Tests of code/m2_obs.py on synthetic data with known answers (CPU; 2026-11-09).
Usage: python m2_obs_test.py -> results/m2/sim/validation/obs_tests.json
 O1 tau_mle      exponential lifetimes (tau 500) censored at T 3000: MLE within 3 SE
 O2 crossing     isotonic + first crossing in ln g reproduces an exact crossing; PAV on a known violator
 O3 PR           uniform -> n; one-hot -> 1; two-level known value
 O4 acf/osc      AR(2) oscillator (period 50 steps, r 0.97): osc True but not strong (C(2P)/C(P) = 0.22), period within
                 5%; undamped noisy sine (period 40): osc and strong; white noise: no osc; AR(1) phi 0.95: osc False,
                 integrated time (1 + phi) / (2 (1 - phi)) = 19.5 steps within 10%
 O5 avalanches   exact extraction on a hand-made count array (sizes, durations, censored runs dropped, threshold area)
 O6 critical GW  Poisson(1) Galton-Watson avalanches written as per-step counts: size exponent 1.5, duration exponent 2,
                 gamma 2 (within 0.1; MF-DP values); subcritical m = 0.8 fails the power-law range test (KS larger)
 O7 MR estimator driven branching A(t+1) ~ Poisson(m A(t) + h), m 0.9 / 0.98, also 10% binomial subsampling: m within 0.01
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_obs as O

OUT = "/MALECNS_ROOT/results/m2/sim/validation"
os.makedirs(OUT, exist_ok=True)
R = {}
rng = np.random.default_rng(12345)


def rep(name, ok, **kw):
    R[name] = {"pass": bool(ok), **{k: (float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v)
                                    for k, v in kw.items()}}
    print(("PASS " if ok else "FAIL ") + name, kw, flush=True)


# O1
tau, T, n = 500.0, 3000.0, 4000
ext = rng.exponential(tau, n); ext[ext > T] = np.inf
est, d = O.tau_mle(ext, T)
se = est / np.sqrt(d)
rep("O1_tau_mle", abs(est - tau) < 3 * se, est=est, true=tau, se=se, deaths=d)

# O2
g = np.geomspace(0.1, 10, 21); y = np.log(g) * 2 + 1          # crossing of y = 0 at g = e^-0.5
gc = O.first_crossing(g, y, 0.0)
iso = O.isotonic([1, 3, 2, 4], [1, 1, 1, 1])
rep("O2_crossing", abs(np.log(gc) + 0.5) < 1e-9 and np.allclose(iso, [1, 2.5, 2.5, 4]), g_cross=gc,
    iso=iso.tolist())

# O3
pr1 = O.participation_ratio(np.ones(1000))[0]; pr2 = O.participation_ratio(np.r_[1.0, np.zeros(99)])[0]
pr3 = O.participation_ratio(np.r_[np.full(10, 2.0), np.full(90, 1.0)])[0]    # (110)^2 / (40 + 90) = 93.08
rep("O3_PR", abs(pr1 - 1000) < 1e-9 and abs(pr2 - 1) < 1e-12 and abs(pr3 - 110 ** 2 / 130) < 1e-9, pr=[pr1, pr2, pr3])

# O4
def ar2(T, B, period, r):
    a1, a2 = 2 * r * np.cos(2 * np.pi / period), -r * r
    x = np.zeros((T + 500, B)); e = rng.normal(size=(T + 500, B))
    for t in range(2, T + 500):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + e[t]
    return x[500:]
x = 10 + ar2(8000, 16, 50, 0.97)
m = O.osc_metrics(x, 1.0)
rep("O4_osc_AR2", m["osc"] and (not m["osc_strong"]) and abs(m["period"] - 50) / 50 < 0.05 and abs(1 / m["f_peak"] - 50) / 50 < 0.1,
    **{k: m[k] for k in ("period", "c_peak", "c_min", "coh_ratio", "f_peak", "peak_ratio", "osc", "osc_strong")})
# undamped oscillation (random phase per replica) buried in white noise: OSC and OSC-strong, period 40
tt = np.arange(6000)[:, None]
x = 1 + 0.3 * np.sin(2 * np.pi * tt / 40 + rng.uniform(0, 2 * np.pi, 8)) + 0.3 * rng.normal(size=(6000, 8))
m = O.osc_metrics(x, 1.0)
rep("O4_osc_sustained_noisy", m["osc"] and m["osc_strong"] and abs(m["period"] - 40) <= 1,
    **{k: m[k] for k in ("period", "c_peak", "c_min", "coh_ratio", "osc", "osc_strong")})
m = O.osc_metrics(1 + rng.normal(size=(6000, 8)), 1.0)
rep("O4_white_noise_no_osc", not m["osc"], **{k: m[k] for k in ("c_min", "c_peak", "osc")})
phi = 0.95
x = np.zeros((20000, 16)); e = rng.normal(size=x.shape)
for t in range(1, len(x)):
    x[t] = phi * x[t - 1] + e[t]
m = O.osc_metrics(10 + x, 1.0)
true_int = (1 + phi) / (2 * (1 - phi))
rep("O4_noosc_AR1", (not m["osc"]) and abs(m["tau_int"] - true_int) / true_int < 0.1, tau_int=m["tau_int"],
    true=true_int, tau_e=m["tau_e"], true_e=-1 / np.log(phi), osc=m["osc"], c_min=m["c_min"], c_peak=m["c_peak"])

# O5
a = np.array([[0, 1, 1, 0, 2, 3, 0, 0, 1],
              [1, 1, 0, 0, 4, 0, 1, 1, 0]]).T            # [S, B]
on = np.array([[0, 1, 0, 0, 2, 1, 0, 0, 1],
               [1, 0, 0, 0, 4, 0, 1, 0, 0]]).T
av = O.avalanches(a, on)
# replica 0: runs [1,3) size 1 dur 2; [4,6) size 3 dur 2; [8,9) touches the end -> dropped
# replica 1: [0,2) touches the start -> dropped; [4,5) size 4 dur 1; [6,8) size 1 dur 2
ok = av["size"].tolist() == [1, 3, 4, 1] and av["dur"].tolist() == [2, 2, 1, 2] and av["rep"].tolist() == [0, 0, 1, 1]
avt = O.avalanches(a, None, thr=1)                        # runs with a > 1: rep0 [4,6) area 1+2 = 3; rep1 [4,5) area 3
ok2 = avt["size"].tolist() == [3, 3] and avt["dur"].tolist() == [2, 1]
rep("O5_avalanche_extraction", ok and ok2, sizes=av["size"].tolist(), durs=av["dur"].tolist(),
    thr_sizes=avt["size"].tolist())

# O6 critical Galton-Watson avalanches as per-step counts
def gw_series(m, n_av, cap=5000):
    chunks_a, chunks_on = [], []
    for _ in range(n_av):
        z, gen = 1, []
        while z > 0 and len(gen) < cap:
            gen.append(z); z = rng.poisson(m * z)
        if z > 0:                     # capped: drop (keeps the sample uncensored)
            continue
        gen = np.array(gen)
        chunks_a.append(np.r_[gen, 0]); chunks_on.append(np.r_[gen, 0])
    a = np.r_[0, np.concatenate(chunks_a)]; o = np.r_[0, np.concatenate(chunks_on)]
    return a[:, None], o[:, None]
a, o = gw_series(1.0, 200000)
av = O.avalanches(a, o)
fs = O.powerlaw_fit(av["size"], xmin_max=200)
ft = O.powerlaw_fit(av["dur"], xmin_max=50)
ms = O.mean_size_given_T(av["size"], av["dur"], tmin=5, tmax=200)
rep("O6_critical_GW_exponents", abs(fs["alpha"] - 1.5) < 0.1 and abs(ft["alpha"] - 2.0) < 0.15 and abs(ms["gamma"] - 2) < 0.15,
    alpha_S=fs["alpha"], xmin_S=fs["xmin"], ks_S=fs["ks"], alpha_T=ft["alpha"], xmin_T=ft["xmin"], gamma=ms["gamma"],
    n_av=len(av["size"]))
a8, o8 = gw_series(0.8, 200000)
av8 = O.avalanches(a8, o8)
f8 = O.powerlaw_fit(av8["size"], xmin=fs["xmin"], n_min=20)
rep("O6_subcritical_not_powerlaw", (not np.isfinite(f8["ks"])) or f8["ks"] > 2 * fs["ks"] or f8["n_tail"] < 0.05 * len(av8["size"]),
    ks_sub=f8["ks"], ks_crit=fs["ks"], n_tail_sub=f8["n_tail"], max_size_sub=int(av8["size"].max()))

# O7 MR estimator
res = {}
ok = True
for mtrue in (0.9, 0.98):
    h = 10.0
    A = np.zeros((40000, 8)); A[0] = h / (1 - mtrue)
    for t in range(1, len(A)):
        A[t] = rng.poisson(mtrue * A[t - 1] + h)
    est = O.mr_branching(A[1000:], 20)["m"]
    sub = rng.binomial(A[1000:].astype(int), 0.1)
    est_s = O.mr_branching(sub, 20)["m"]
    naive = float((A[1001:] * A[1000:-1]).sum() / (A[1000:-1] ** 2).sum())
    res[str(mtrue)] = {"mr": est, "mr_subsampled_10pct": est_s, "naive_ratio": naive}
    ok &= abs(est - mtrue) < 0.01 and abs(est_s - mtrue) < 0.01
rep("O7_MR_branching", ok, **{f"m{k}": v for k, v in res.items()})

R["_meta"] = {"n_pass": sum(v["pass"] for k, v in R.items() if not k.startswith("_")),
              "n_tests": sum(1 for k in R if not k.startswith("_"))}
json.dump(R, open(f"{OUT}/obs_tests.json", "w"), indent=1, default=str)
print(json.dumps(R["_meta"]))
