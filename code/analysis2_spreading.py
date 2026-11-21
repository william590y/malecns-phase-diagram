"""Single-seed spreading at r = 1 (raw norm, h = 0): simulation vs the branching / extinction theory.

Data (all r = 1, raw weights, B = 512 replicas, one active seed neuron per replica):
  random seeds  results/spread_raw_r1.npz       real,  g .30 .34 .38 .45 .60          T 4000
                results/spread_raw_r1_null.npz  null,  g .85 .95 1.05 1.15 1.25       T 4000
  module seeds  results/pd/mspread_real_{a,b}.npz  real, g .20 .24 .28 .32 | .36 .42 .50 .60   T 3000
                results/pd/mspread_null.npz        null, g .95 1.05 1.11 1.15 1.25            T 3000
  Module seeds are drawn with replacement, prob. prop. to |v0_i|^2 over the top-1000 neurons of each graph's leading
  eigenvector of the symmetrized weighted excitatory matrix (results/seeds_module_{real,null}.npz).

Seed reconstruction. swc.py draws seeds with np.random.default_rng(seed) (after the null permutation of the targets, which
consumes the generator first), one rng.choice per g in order. We replay that sequence to recover every replica's seed neuron.
Check: a non-excitatory seed (sign <= 0) can never switch anyone on (rectified tanh), so its replica must have N(t) <= 1 at
all recorded times; the replay is accepted only if that holds for every such replica.

Per (graph, seed type, g): survival P(t), mean active count over all runs N_all(t) and over survivors N_surv(t), P(T) with a
Wilson 95% interval, and fits of P(t). Fit definitions (fixed before looking at the fit results):
  decaying range  t in [1, t_hi]: t_hi = last recorded time with >= 10 surviving replicas and P(t) > 1.05 P(T_end)
                  (excludes the plateau of established runs; P is monotone so the range is contiguous)
  late range      same, starting at t = 30 (after the establishment transient)
  power law       ln P = a - delta ln t ;  exponential  ln P = a - t / tau ;  OLS on the recorded (log-spaced) times;
                  R^2 computed on ln P for both, so the two are directly comparable.
  exponent drift  power law refit on each half (in ln t) of the range; a clean power law has delta_1 ~ delta_2, an
                  exponential has delta growing with t.
  Griffiths test  (pre-specified) a Griffiths signature needs >= 3 consecutive g with a late range spanning >= 1 decade,
                  R^2_pow >= R^2_exp, |delta_2 - delta_1| < 0.3 delta, and delta varying with g.
  late hazard     tau_sim = exposure / deaths over t in [30, T] (constant-hazard MLE; exposure = trapezoid of the survivor
                  count), Garwood 95% interval on the death count.

Theory (explainer/theory.html, sections 3-5): dense branch rows at r = 1 (results/pd/branch_<graph>_dense_part*.json;
the null falls back on the narrow branch_null.json "branching" rows where dense parts are still missing), interpolated in ln g:
  R(g) = lam / alpha_eff,  p_est = 1 - 1/R  (single-type branching),  tau(g) = exp(logtau_r1)  (mean extinction time from the
  quasi-stationary top state of the logistic birth-death chain with K_eff),  P_theory(T) = p_est * exp(-T / tau).
  Diagnostics: tau at r = 0 (no inhibitory coincidence term; the small excitatory-excitatory term D_r0 remains), and K_needed = the K_eff that makes the theory's extinction time equal
  tau_sim (exact birth-death recursion, same code as theory_branch_gpu.py).
Random seeds: a random seed lands in the 1000-neuron seed set with probability f_hit = 1000/N; the direct-hit prediction is
  P_rand(3000) ~ f_hit * P_mod(3000) at matched g (P_mod interpolated linearly in g where the g grids differ).
Outputs: results/analysis2/spreading.{json,md}, figs/pd2/spreading.png.
"""
import glob, json, os
import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2, poisson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "/MALECNS_ROOT"
R_, C = f"{ROOT}/results", "/MALECNS_CACHE"
OUT, FIG = f"{R_}/analysis2", f"{ROOT}/figs/pd2"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
N = 167216
DT = 0.2
A_EFF = (1 - np.exp(-DT)) / DT
NMIN, PLATEAU, T_LO, T_LATE = 10, 1.05, 1.0, 30.0
T_CMP = 3000.0

sign = np.load(f"{C}/sign.npy"); assert len(sign) == N
sup = np.load(f"{C}/superclasses.npy", allow_pickle=True); types = np.load(f"{C}/types.npy", allow_pickle=True)
n_edges = len(np.load(f"{C}/pre.npy", mmap_mode="r"))

DATA = [
    dict(key="real_random", graph="real", seeds="random", files=["spread_raw_r1.npz"]),
    dict(key="null_random", graph="null", seeds="random", files=["spread_raw_r1_null.npz"]),
    dict(key="real_module", graph="real", seeds="module", files=["pd/mspread_real_a.npz", "pd/mspread_real_b.npz"]),
    dict(key="null_module", graph="null", seeds="module", files=["pd/mspread_null.npz"]),
]
VEC = {"real": "sym_exc_weighted_vecs_top10.npy", "null": "null_sym_exc_weighted_vecs_top10.npy"}
SEEDSET = {"real": "seeds_module_real.npz", "null": "seeds_module_null.npz"}


# ---------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return [np.nan, np.nan]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def linfit(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef; sst = ((y - y.mean()) ** 2).sum()
    return coef, float(1 - (res ** 2).sum() / sst) if sst > 0 else np.nan, float(np.sqrt((res ** 2).mean()))


def fit_range(t, P, nal, p_end, t_lo):
    ok = (t >= t_lo) & (nal >= NMIN) & (P > PLATEAU * p_end) & (P > 0)
    idx = np.where(ok)[0]
    if len(idx) < 5:
        return None
    idx = idx[: np.argmax(np.diff(np.r_[idx, -9]) != 1) + 1]          # contiguous block from t_lo
    if len(idx) < 5:
        return None
    tt, yy = t[idx], np.log(P[idx])
    (a_p, b_p), r2_p, rm_p = linfit(np.log(tt), yy)
    (a_e, b_e), r2_e, rm_e = linfit(tt, yy)
    half = np.log(tt) <= 0.5 * (np.log(tt[0]) + np.log(tt[-1]))
    d1 = -linfit(np.log(tt[half]), yy[half])[0][1] if half.sum() >= 3 else np.nan
    d2 = -linfit(np.log(tt[~half]), yy[~half])[0][1] if (~half).sum() >= 3 else np.nan
    return {"t_lo": float(tt[0]), "t_hi": float(tt[-1]), "n_pts": int(len(tt)), "decades": float(np.log10(tt[-1] / tt[0])),
            "pow": {"delta": float(-b_p), "lnA": float(a_p), "R2": r2_p, "rmse": rm_p},
            "exp": {"tau": float(-1 / b_e) if b_e < 0 else float("inf"), "P0": float(np.exp(a_e)), "R2": r2_e, "rmse": rm_e},
            "delta_half1": float(d1), "delta_half2": float(d2), "better": "power" if r2_p >= r2_e else "exponential"}


def late_hazard(t, nal, t0=T_LATE):
    m = t >= t0
    tt, nn = t[m], nal[m].astype(float)
    if len(tt) < 2 or nn[0] == 0:
        return None
    k = int(round(nn[0] - nn[-1])); expo = float(np.trapezoid(nn, tt))
    lo = chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return {"t0": float(tt[0]), "n0": int(nn[0]), "deaths": k, "exposure": expo,
            "tau": expo / k if k > 0 else float("inf"), "tau_ci": [expo / hi, expo / lo if lo > 0 else float("inf")]}


def log_tau(lam, K):            # identical to theory_branch_gpu.log_tau (exact birth-death recursion from the top state)
    n_top = int(np.floor(K))
    if n_top < 1 or lam <= 0:
        return -np.inf
    n = np.arange(1, n_top + 1, dtype=float)
    lb = lam * n * (1 - n / K); lb[-1] = 0.0
    ld = A_EFF * n
    logD = np.empty(n_top); logD[-1] = -np.log(ld[-1])
    for k in range(n_top - 2, -1, -1):
        logD[k] = np.logaddexp(0.0, np.log(lb[k]) + logD[k + 1]) - np.log(ld[k])
    return float(logsumexp(logD))


def k_needed(lam, tau):
    if not np.isfinite(tau) or tau <= 0:
        return None
    target = np.log(tau); lo, hi = 2.0, 3000.0
    if log_tau(lam, lo) > target or log_tau(lam, hi) < target:
        return None
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if log_tau(lam, mid) < target else (lo, mid)
    return float(np.sqrt(lo * hi))


# ---------------------------------------------------------------- theory tables
def load_theory(graph):
    rows = []
    for f in sorted(glob.glob(f"{R_}/pd/branch_{graph}_dense_part*.json")):
        d = json.load(open(f)); rows += d["rows"]; g_lin = d["g_lin"]
    src = [f"results/pd/branch_{graph}_dense_part*.json ({len(rows)} rows)"]
    # [verifier 2026-11-09] fill from the old narrow run ONLY while dense parts are missing; with all 8 parts present the old
    # rule (fill any old g not within 1% of a dense g) interleaved 18 rows from a second source into a complete dense grid.
    if graph == "null" and len(glob.glob(f"{R_}/pd/branch_null_dense_part*.json")) < 8:
        old = json.load(open(f"{R_}/pd/branch_null.json"))["variants"]["branching"]
        have = np.array([r["g"] for r in rows])
        extra = [r for r in old["rows"] if len(have) == 0 or np.min(np.abs(np.log(have / r["g"]))) > 0.01]
        rows += extra; src.append(f"results/pd/branch_null.json branching ({len(extra)} rows filling gaps)")
        if not glob.glob(f"{R_}/pd/branch_null_dense_part*.json"):
            g_lin = old["g_lin"]
    rows.sort(key=lambda r: r["g"])
    g = np.array([r["g"] for r in rows])
    th = {"g_lin": g_lin, "g": g, "R": np.array([r["R"] for r in rows]), "lt1": np.array([r["logtau_r1"] for r in rows]),
          "lt0": np.array([r["logtau_r0"] for r in rows]), "K1": np.array([r["K_r1"] for r in rows]),
          "K0": np.array([r["K_r0"] for r in rows]), "src": src}
    return th


def th_at(th, g):
    lg, x = np.log(g), np.log(th["g"])
    # R: include the threshold point (g_lin, R = 1) so that g between g_lin and the first row is interpolated, not extrapolated
    xr, yr = np.r_[np.log(th["g_lin"]), x], np.r_[1.0, th["R"]]
    R = float(np.interp(lg, xr, yr)) if lg >= xr[0] else float("nan")
    extrap = bool(lg < x[0] or lg > x[-1])
    def lin(y):
        if lg < x[0] or lg > x[-1]:     # [verifier] no extrapolation outside the theory rows (null r=0 extrapolated to 3.5 < tau_r1)
            return float("nan")
        return float(np.interp(lg, x, y))
    lt1, lt0, K1, K0 = lin(th["lt1"]), lin(th["lt0"]), lin(th["K1"]), lin(th["K0"])
    p_est = 1 - 1 / R if R > 1 else 0.0
    return {"R": R, "p_est": p_est, "logtau_r1": lt1, "tau_r1": float(np.exp(lt1)), "tau_r0": float(np.exp(lt0)),
            "K_r1": K1, "K_r0": K0, "g_over_glin": float(g / th["g_lin"]), "extrapolated_logtau": extrap}


TH = {k: load_theory(k) for k in ["real", "null"]}

# ---------------------------------------------------------------- simulations
vecs, rank, inset, tomod = {}, {}, {}, {}
pre = np.load(f"{C}/pre.npy"); post = np.load(f"{C}/post.npy"); w = np.load(f"{C}/weight.npy")
exc_e = sign[pre] > 0
for gr in ["real", "null"]:
    V = np.load(f"{R_}/{VEC[gr]}")[:, 0]; o = np.argsort(-V ** 2); rk = np.empty(N, int); rk[o] = np.arange(N)
    ss = np.load(f"{R_}/{SEEDSET[gr]}"); m = np.zeros(N, bool); m[ss["idx"]] = True
    assert set(ss["idx"]) == set(o[:1000])
    rank[gr], inset[gr] = rk, m
    pp = post if gr == "real" else post[np.random.default_rng(0).permutation(len(post))]
    sel = exc_e & m[pp]
    tomod[gr] = np.bincount(pre[sel], weights=w[sel], minlength=N)          # exc synapses from each neuron into the set
    outs = np.bincount(pre[exc_e], weights=w[exc_e], minlength=N)            # exc out-strength (same in the target-permuted null)
    p = ss["p"]
    vecs[gr] = {"n_set": int(len(ss["idx"])), "eff_size": float(1 / (p ** 2).sum()),
                "v0sq_mass_in_set": float((V[o[:1000]] ** 2).sum()), "inv_IPR": float(1 / (V ** 4).sum() * (V ** 2).sum() ** 2),
                "p_mass_nonexc": float(p[sign[ss["idx"]] <= 0].sum()),
                "frac_exc_neurons_projecting_into_set": float((tomod[gr][sign > 0] > 0).mean())}
del pre, post, w

curves, recon = {}, {}
for ds in DATA:
    gr = ds["graph"]
    for f in ds["files"]:
        d = np.load(f"{R_}/{f}"); meta = json.loads(str(d["meta"]))
        rng = np.random.default_rng(meta["seed"])
        if meta["null"]:
            assert meta.get("null_kind", "targets") == "targets"
            rng.permutation(n_edges)
        ss = np.load(f"{R_}/{SEEDSET[gr]}") if ds["seeds"] == "module" else None
        t = d["t"]; T = float(meta["T"]); B = d["rho"].shape[-1]
        for i, g in enumerate(d["g"]):
            seeds = rng.choice(N, B, replace=False) if ss is None else rng.choice(ss["idx"], B, replace=True, p=ss["p"])
            nA = np.round(d["rho"][i] * N); surv = d["surv"][i] > 0
            nonexc = sign[seeds] <= 0
            viol = int((nA[:, nonexc].max(0) > 1).sum())
            recon[f"{ds['key']}_g{g:g}"] = {"nonexc_seeds": int(nonexc.sum()), "violations": viol,
                                            "exc_seeds_with_N_gt_1": int((nA[:, ~nonexc].max(0) > 1).sum())}
            assert viol == 0, f"seed replay failed for {f} g={g}"
            nal = surv.sum(1); P = nal / B
            Nall = nA.mean(1); Nsurv = np.where(nal > 0, (nA * surv).sum(1) / np.maximum(nal, 1), np.nan)
            k_end = int(nal[-1])
            i3 = np.searchsorted(t, T_CMP, side="right") - 1
            alive_end = surv[-1]
            sv = [{"neuron": int(s), "v0_rank": int(rank[gr][s]), "in_set": bool(inset[gr][s]), "superclass": str(sup[s]),
                   "type": str(types[s]), "exc_syn_into_set": float(tomod[gr][s])} for s in seeds[alive_end]] if ds["seeds"] == "random" else None
            ex = ~nonexc
            i100 = np.searchsorted(t, 100.0)
            a100 = surv[i100]; o = outs[seeds]; q = np.quantile(o[ex], [1 / 3, 2 / 3])
            terc = [ex & (o <= q[0]), ex & (o > q[0]) & (o <= q[1]), ex & (o > q[1])]
            est = {"t": float(t[i100]), "P_all": float(a100.mean()), "P_exc_seeds": float(a100[ex].mean()),
                   "P_exc_ci": wilson(int(a100[ex].sum()), int(ex.sum())),
                   "by_exc_outstrength_tercile": [{"median_out_syn": float(np.median(o[b])), "n": int(b.sum()),
                                                   "P": float(a100[b].mean())} for b in terc]}
            c = {"graph": gr, "seeds": ds["seeds"], "file": f"results/{f}", "g": float(g), "T": T, "B": int(B),
                 "t": t.tolist(), "P": P.tolist(), "n_alive": nal.astype(int).tolist(), "N_all": Nall.tolist(),
                 "N_surv": [None if np.isnan(x) else float(x) for x in Nsurv],
                 "P_T": float(P[-1]), "k_T": k_end, "P_T_ci": wilson(k_end, B),
                 "t_cmp": float(t[i3]), "P_cmp": float(P[i3]), "k_cmp": int(nal[i3]),
                 "last_death_t": float(t[np.argmax(nal == 0)]) if k_end == 0 else None,
                 "N_surv_T": float(Nsurv[-1]) if k_end > 0 else None,
                 "establishment_P100": est, "frac_exc_seeds": float(ex.mean()), "P_T_given_exc_seed": float(alive_end[ex].mean()),
                 "seeds_in_set": int(inset[gr][seeds].sum()), "P_T_given_in_set": float(alive_end[inset[gr][seeds]].mean()) if inset[gr][seeds].any() else None,
                 "survivor_seeds": sv,
                 "fit_decay": fit_range(t, P, nal, P[-1], T_LO), "fit_late": fit_range(t, P, nal, P[-1], T_LATE),
                 "late_hazard": late_hazard(t, nal)}
            c["theory"] = th_at(TH[gr], float(g))
            th = c["theory"]
            th["P_T_theory"] = th["p_est"] * float(np.exp(-T / th["tau_r1"]))
            th["P_T_theory_r0tau"] = th["p_est"] * float(np.exp(-T / th["tau_r0"]))
            th["p_est_x_frac_exc"] = th["p_est"] * c["frac_exc_seeds"]
            lh = c["late_hazard"]
            if lh and np.isfinite(lh["tau"]):
                th["K_needed_for_tau_sim"] = k_needed(th["R"] * A_EFF, lh["tau"])
                th["tau_sim_over_theory_r1"] = lh["tau"] / th["tau_r1"]
            curves[f"{ds['key']}_g{g:g}"] = c

# ---------------------------------------------------------------- random vs module (direct-hit prediction)
cmp = []
for gr in ["real", "null"]:
    mods = sorted([c for c in curves.values() if c["graph"] == gr and c["seeds"] == "module"], key=lambda c: c["g"])
    gm = np.array([c["g"] for c in mods]); Pm = np.array([c["P_cmp"] for c in mods])
    f_hit = vecs[gr]["n_set"] / N
    for c in sorted([c for c in curves.values() if c["graph"] == gr and c["seeds"] == "random"], key=lambda c: c["g"]):
        g = c["g"]
        if g < gm[0] or g > gm[-1]:
            cmp.append({"graph": gr, "g": g, "P_rand_3000": c["P_cmp"], "k_rand_3000": c["k_cmp"], "P_mod_3000": None,
                        "note": "no module-seed run brackets this g"}); continue
        exact = bool(np.any(np.isclose(gm, g)))
        pm = float(np.interp(g, gm, Pm))
        lam_exp = c["B"] * f_hit * pm
        k = c["k_cmp"]
        ci = wilson(k, c["B"])
        cmp.append({"graph": gr, "g": g, "P_rand_3000": c["P_cmp"], "k_rand_3000": k, "P_rand_ci": ci,
                    "P_mod_3000": pm, "P_mod_source": "exact g" if exact else "linear interpolation in g",
                    "ratio_rand_over_mod": c["P_cmp"] / pm if pm > 0 else None,
                    "f_hit": f_hit, "pred_P_rand_direct_hit": f_hit * pm, "expected_survivors_direct_hit": lam_exp,
                    "p_obs_ge": float(poisson.sf(k - 1, lam_exp)) if k > 0 else 1.0, "p_obs_le": float(poisson.cdf(k, lam_exp)),
                    "catchment_N_eff": N * c["P_cmp"] / pm if pm > 0 else None,
                    "catchment_ci": [N * ci[0] / pm, N * ci[1] / pm] if pm > 0 else None,
                    "theory_direct_hit_P_T": f_hit * c["theory"]["P_T_theory"],
                    "seeds_in_set": c["seeds_in_set"], "P_T_given_in_set": c["P_T_given_in_set"],
                    "n_survivors_T": c["k_T"], "survivors_in_set": sum(s["in_set"] for s in c["survivor_seeds"]),
                    "survivors_projecting_into_set": sum((s["exc_syn_into_set"] > 0) for s in c["survivor_seeds"])})

# ---------------------------------------------------------------- Griffiths test (pre-specified rule)
gtest = {}
for key in ["real_module", "null_module", "real_random", "null_random"]:
    cs = sorted([c for k, c in curves.items() if k.startswith(key)], key=lambda c: c["g"])
    ok = []
    for c in cs:
        fl = c["fit_late"]
        good = bool(fl and fl["decades"] >= 1 and fl["pow"]["R2"] >= fl["exp"]["R2"] and
                    abs(fl["delta_half2"] - fl["delta_half1"]) < 0.3 * fl["pow"]["delta"])
        ok.append(good)
    run, best = 0, 0
    for o in ok:
        run = run + 1 if o else 0; best = max(best, run)
    gtest[key] = {"g": [c["g"] for c in cs], "passes": ok, "longest_run": best, "griffiths_signature": best >= 3}

res = {"definitions": __doc__, "N": N, "alpha_eff": A_EFF, "seed_sets": vecs, "seed_replay_check": recon,
       "theory_sources": {k: v["src"] for k, v in TH.items()}, "g_lin": {k: v["g_lin"] for k, v in TH.items()},
       "curves": curves, "random_vs_module": cmp, "griffiths_test": gtest}
def clean(x):                   # strict JSON: non-finite floats -> "inf" / "-inf" / null
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if np.isfinite(x) else ("inf" if x > 0 else "-inf" if x < 0 else None)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


json.dump(clean(res), open(f"{OUT}/spreading.json", "w"), indent=1, allow_nan=False)

# ---------------------------------------------------------------- markdown
def fm(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "inf" if isinstance(x, float) and x == np.inf else "-"
    if x == 0:
        return "0"
    return f"{x:.{nd}g}"


L = ["# Single-seed spreading at r = 1: simulation vs theory", "",
     "Generated by `code/analysis2_spreading.py`; all numbers below come from the npz/json files named in the script docstring.",
     f"g_lin: real {TH['real']['g_lin']:.4f}, null {TH['null']['g_lin']:.4f}. Theory sources: " +
     "; ".join(f"{k}: {', '.join(v['src'])}" for k, v in TH.items()) + ".", "",
     "Seed replay check (non-excitatory seeds must never switch anyone on): " +
     f"{sum(v['nonexc_seeds'] for v in recon.values())} non-excitatory seed replicas, "
     f"{sum(v['violations'] for v in recon.values())} violations, across {len(recon)} runs.", "",
     "Seed sets: " + "; ".join(f"{gr}: 1000 neurons, effective size 1/sum p^2 = {v['eff_size']:.0f}, |v0|^2 mass in set "
                               f"{v['v0sq_mass_in_set']:.3f}, 1/IPR of v0 = {v['inv_IPR']:.0f}, seed probability on "
                               f"non-excitatory neurons {v['p_mass_nonexc']:.3f}" for gr, v in vecs.items()), ""]
def _c(key, g):
    return curves[f"{key}_g{g:g}"]


H = ["## Headline (computed)", ""]
for key in ["real_module", "null_module"]:
    for c in sorted([c for k, c in curves.items() if k.startswith(key)], key=lambda c: c["g"]):
        lh = c["late_hazard"]; th = c["theory"]
        H.append(f"- {key} g {c['g']:g}: P(T) sim {c['P_T']:.4f} vs theory p_est e^(-T/tau) {th['P_T_theory']:.3g}; "
                 f"P(100) exc seeds {c['establishment_P100']['P_exc_seeds']:.3f} vs p_est {th['p_est']:.3f}; "
                 f"late tau sim {fm(lh['tau']) if lh else '-'} vs theory r=1 {th['tau_r1']:.3g} (r=0 {th['tau_r0']:.3g})")
H.append("")
L += H
for key, title in [("real_module", "Real graph, module seeds"), ("null_module", "Configuration null, module seeds"),
                   ("real_random", "Real graph, random seeds"), ("null_random", "Configuration null, random seeds")]:
    cs = sorted([c for k, c in curves.items() if k.startswith(key)], key=lambda c: c["g"])
    L += [f"## {title} (T = {cs[0]['T']:g}, B = {cs[0]['B']})", "",
          "| g | g/g_lin | R | P(10) | P(100) | P(T) [95% CI] | N_surv(T) | N_all(100) | last death | late tau_sim [CI] | "
          "p_est | tau_th r=1 | tau_th r=0 | P_th(T) | K_needed |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cs:
        t = np.array(c["t"]); P = np.array(c["P"]); th = c["theory"]; lh = c["late_hazard"]
        i10, i100 = np.searchsorted(t, 10), np.searchsorted(t, 100)
        tau_s = f"{fm(lh['tau'])} [{fm(lh['tau_ci'][0])}, {fm(lh['tau_ci'][1])}] ({lh['deaths']}/{lh['n0']})" if lh else "-"
        L.append(f"| {c['g']:g} | {th['g_over_glin']:.2f} | {th['R']:.3f} | {P[i10]:.3f} | {P[i100]:.3f} | "
                 f"{c['P_T']:.4f} [{c['P_T_ci'][0]:.4f}, {c['P_T_ci'][1]:.4f}] | {fm(c['N_surv_T'])} | {c['N_all'][i100]:.3g} | "
                 f"{fm(c['last_death_t'])} | {tau_s} | {th['p_est']:.3f} | {th['tau_r1']:.3g} | {th['tau_r0']:.3g} | "
                 f"{th['P_T_theory']:.3g} | {fm(th.get('K_needed_for_tau_sim'))} |")
    L += ["", "Fits of P(t) (ln P vs ln t = power law, ln P vs t = exponential; R^2 on ln P):", "",
          "| g | range | t range | decades | n | delta (pow) | R2 pow | tau (exp) | P0 (exp) | R2 exp | delta 1st/2nd half | better |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cs:
        for nm in ["fit_decay", "fit_late"]:
            fr = c[nm]
            if not fr:
                L.append(f"| {c['g']:g} | {nm[4:]} | (fewer than 5 points) | | | | | | | | | |"); continue
            L.append(f"| {c['g']:g} | {nm[4:]} | {fr['t_lo']:.3g}-{fr['t_hi']:.4g} | {fr['decades']:.2f} | {fr['n_pts']} | "
                     f"{fr['pow']['delta']:.3f} | {fr['pow']['R2']:.4f} | {fm(fr['exp']['tau'])} | {fr['exp']['P0']:.3f} | "
                     f"{fr['exp']['R2']:.4f} | {fr['delta_half1']:.2f} / {fr['delta_half2']:.2f} | {fr['better']} |")
    gt = gtest[key]
    L += ["", f"Pre-specified Griffiths test: passes per g {dict(zip(gt['g'], gt['passes']))}; longest run {gt['longest_run']}; "
              f"signature {'YES' if gt['griffiths_signature'] else 'NO'}.", ""]
    if cs[0]["seeds"] == "random":
        for c in cs:
            if c["survivor_seeds"]:
                L.append(f"- g {c['g']:g}: {c['k_T']} survivors at T; seeds in set: {sum(s['in_set'] for s in c['survivor_seeds'])}; "
                         "survivor seeds (v0 rank, type, exc synapses into set): " +
                         ", ".join(f"({s['v0_rank']}, {s['type']}, {s['exc_syn_into_set']:.0f})" for s in c["survivor_seeds"][:30]))
        L.append("")
L += ["## Establishment: P(t = 100) vs single-type p_est = 1 - 1/R (module seeds)", "",
      "Non-excitatory seeds can never switch anyone on, so the excitatory-seed column is the like-for-like comparison. "
      "Terciles split the excitatory seeds by their excitatory out-strength (synapses).", "",
      "| graph | g | R | p_est | P(100) all seeds | P(100) exc seeds [95% CI] | ratio exc/p_est | terciles low/mid/high (median out-syn) |",
      "|---|---|---|---|---|---|---|---|"]
for key in ["real_module", "null_module"]:
    for c in sorted([c for k, c in curves.items() if k.startswith(key)], key=lambda c: c["g"]):
        e = c["establishment_P100"]; th = c["theory"]
        L.append(f"| {c['graph']} | {c['g']:g} | {th['R']:.3f} | {th['p_est']:.3f} | {e['P_all']:.3f} | {e['P_exc_seeds']:.3f} "
                 f"[{e['P_exc_ci'][0]:.3f}, {e['P_exc_ci'][1]:.3f}] | {e['P_exc_seeds'] / th['p_est']:.2f} | " +
                 " / ".join(f"{x['P']:.3f} ({x['median_out_syn']:.0f})" for x in e["by_exc_outstrength_tercile"]) + " |")
L += ["", "Fraction of excitatory neurons with at least one excitatory synapse into the 1000-neuron seed set: " +
      ", ".join(f"{gr} {v['frac_exc_neurons_projecting_into_set']:.3f}" for gr, v in vecs.items()), ""]
L += ["## Random vs module seeds at matched g (t = 3000 for both)", "",
      "| graph | g | P_rand(3000) (k/512) | P_mod(3000) | source | ratio | f_hit*P_mod | expected survivors | P(X>=k) | P(X<=k) | "
      "catchment N*P_rand/P_mod [CI] | seeds in set (P(T) among them) | survivors in set / projecting into set |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for x in cmp:
    if x["P_mod_3000"] is None:
        L.append(f"| {x['graph']} | {x['g']:g} | {x['P_rand_3000']:.4f} ({x['k_rand_3000']}) | - | {x['note']} | | | | | | | | |"); continue
    L.append(f"| {x['graph']} | {x['g']:g} | {x['P_rand_3000']:.4f} ({x['k_rand_3000']}) | {x['P_mod_3000']:.4f} | {x['P_mod_source']} | "
             f"{fm(x['ratio_rand_over_mod'])} | {x['pred_P_rand_direct_hit']:.4g} | {x['expected_survivors_direct_hit']:.2f} | "
             f"{x['p_obs_ge']:.3g} | {x['p_obs_le']:.3g} | {fm(x['catchment_N_eff'], 4)} [{fm((x['catchment_ci'] or [None, None])[0], 3)}, {fm((x['catchment_ci'] or [None, None])[1], 3)}] | "
             f"{x['seeds_in_set']} ({fm(x['P_T_given_in_set'])}) | {x['survivors_in_set']} / {x['survivors_projecting_into_set']} of {x['n_survivors_T']} |")
open(f"{OUT}/spreading.md", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6})
BLUE, ORANGE = "#2a78d6", "#eb6834"
ramp = {"real": plt.cm.Blues, "null": plt.cm.Oranges}
FLOOR = 1e-3
np.seterr(over="ignore")


def cols(gr, n):
    return [ramp[gr](x) for x in (np.linspace(0.42, 0.98, n) if n > 1 else [0.8])]


def get(key):
    return sorted([c for k, c in curves.items() if k.startswith(key)], key=lambda c: c["g"])


fig, axs = plt.subplots(3, 3, figsize=(16.5, 14.5))


def p_panel(ax, key, title, theory=True):
    cs = get(key); gr = cs[0]["graph"]; cl = cols(gr, len(cs))
    for c, col in zip(cs, cl):
        t = np.array(c["t"]); P = np.array(c["P"]); m = P > 0
        ax.loglog(t[m], P[m], "-", c=col, lw=2, label=f"g = {c['g']:g}")
        if c["k_T"] == 0:
            j = np.where(m)[0][-1]; ax.plot(t[j], P[j], "x", c=col, ms=7, mew=2)
        if theory:
            th = c["theory"]; tt = np.logspace(0.3, np.log10(c["T"]), 200)
            y = th["p_est"] * np.exp(-tt / th["tau_r1"]); mm = y >= FLOOR
            if mm.any():
                ax.plot(tt[mm], y[mm], "--", c=col, lw=1.4)
            yT = th["P_T_theory"]
            ax.plot(c["T"] * 1.25, max(yT, FLOOR), "D" if yT >= FLOOR else "v", mfc="white", mec=col, mew=1.8, ms=8, clip_on=False)
    ax.set_ylim(FLOOR * 0.8, 1.15); ax.set_xlim(0.18, 6000)
    ax.set_xlabel("time t (units of 1/alpha)"); ax.set_ylabel("survival probability P(t)"); ax.set_title(title, loc="left")
    ax.legend(loc="lower left", fontsize=7.5, ncol=2)


p_panel(axs[0, 0], "real_module", "(a) real connectome, module seeds (r = 1)")
p_panel(axs[0, 1], "null_module", "(b) configuration null, module seeds (r = 1)")

ax = axs[0, 2]
for key in ["real_random", "null_random"]:
    cs = get(key); gr = cs[0]["graph"]; cl = cols(gr, len(cs))
    for c, col in zip(cs, cl):
        t = np.array(c["t"]); P = np.array(c["P"]); m = P > 0
        ax.loglog(t[m], P[m], "-", c=col, lw=2, label=f"{gr} g = {c['g']:g}")
        if c["k_T"] == 0:
            j = np.where(m)[0][-1]; ax.plot(t[j], P[j], "x", c=col, ms=7, mew=2)
        x = next((x for x in cmp if x["graph"] == gr and np.isclose(x["g"], c["g"]) and x["P_mod_3000"] is not None), None)
        if x and x["pred_P_rand_direct_hit"] >= FLOOR:
            ax.plot(T_CMP, max(x["pred_P_rand_direct_hit"], FLOOR), "*", c=col, mec="k", mew=0.6, ms=12, zorder=5)
ax.axhline(1 / 512, c="0.55", lw=0.8, ls=":")
ax.text(300, 1 / 512 * 0.8, "1 of 512 replicas", fontsize=7, color="0.4", ha="center", va="top")
ax.set_ylim(FLOOR * 0.8, 1.15); ax.set_xlim(0.18, 6000)
ax.set_xlabel("time t"); ax.set_ylabel("survival probability P(t)")
ax.set_title("(c) random seeds (one uniform neuron), real and null", loc="left")
ax.legend(loc="lower left", fontsize=7, ncol=2)


def n_panel(ax, key, title, show_all=True):
    cs = get(key); gr = cs[0]["graph"]; cl = cols(gr, len(cs))
    for c, col in zip(cs, cl):
        t = np.array(c["t"]); ns = np.array([np.nan if x is None else x for x in c["N_surv"]]); na = np.array(c["N_all"])
        ax.loglog(t, ns, "-", c=col, lw=2, label=f"g = {c['g']:g}")
        if show_all:
            m = na > 0; ax.loglog(t[m], na[m], ":", c=col, lw=1.4)
    ax.set_xlabel("time t"); ax.set_ylabel("mean number of active neurons"); ax.set_title(title, loc="left")
    ax.set_xlim(0.18, 6000)
    l1 = ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    if show_all:
        ax.add_artist(l1)
        ax.legend([Line2D([], [], c="0.3", lw=2), Line2D([], [], c="0.3", ls=":", lw=1.4)],
                  ["over surviving replicas", "over all replicas"], loc="lower right", fontsize=7.5)


n_panel(axs[1, 0], "real_module", "(d) N(t), real, module seeds")
n_panel(axs[1, 1], "null_module", "(e) N(t), null, module seeds")
ax = axs[1, 2]
for key in ["real_random", "null_random"]:
    cs = get(key); gr = cs[0]["graph"]; cl = cols(gr, len(cs))
    for c, col in zip(cs, cl):
        t = np.array(c["t"]); ns = np.array([np.nan if x is None else x for x in c["N_surv"]])
        ax.loglog(t, ns, "-", c=col, lw=2, label=f"{gr} g = {c['g']:g}")
ax.set_xlabel("time t"); ax.set_ylabel("mean active neurons over survivors"); ax.set_xlim(0.18, 6000)
ax.set_title("(f) N(t) over survivors, random seeds", loc="left"); ax.legend(fontsize=7, ncol=2, loc="upper left")

# (g) P(3000) vs g/g_lin
ax = axs[2, 0]
for gr, col, mk in [("real", BLUE, "o"), ("null", ORANGE, "s")]:
    th = TH[gr]; gg = np.exp(np.linspace(np.log(th["g_lin"] * 1.001), np.log(th["g_lin"] * 4.2), 300))
    ths = [th_at(th, g) for g in gg]
    pe = np.array([x["p_est"] for x in ths]); pt = np.array([x["p_est"] * np.exp(-T_CMP / x["tau_r1"]) for x in ths])
    xg = gg / th["g_lin"]
    ax.plot(xg, pe, ":", c=col, lw=1.6)
    m = pt >= FLOOR; ax.plot(xg[m], pt[m], "--", c=col, lw=1.6)
    for key, filled in [(f"{gr}_module", True), (f"{gr}_random", False)]:
        cs = get(key)
        x = np.array([c["theory"]["g_over_glin"] for c in cs]); y = np.array([c["P_cmp"] for c in cs])
        ci = np.array([wilson(c["k_cmp"], c["B"]) for c in cs])
        pos = y > 0
        ax.errorbar(x[pos], y[pos], yerr=[y[pos] - ci[pos, 0], ci[pos, 1] - y[pos]], fmt=mk, c=col, mfc=col if filled else "white",
                    mec=col, mew=1.6, ms=7, capsize=2, lw=1)
        ax.plot(x[~pos], np.full((~pos).sum(), FLOOR), "v", c=col, mfc=col if filled else "white", mec=col, mew=1.6, ms=7)
        if filled:
            ax.plot(x, [c["establishment_P100"]["P_exc_seeds"] for c in cs], "P", c=col, mec="k", mew=0.5, ms=8, zorder=6)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(4e-5, 1.3)
ax.set_xlabel("g / g_lin  (g_lin: real 0.163, null 0.829)"); ax.set_ylabel("P(t = 3000)")
ax.set_title("(g) survival to t = 3000 and establishment vs theory", loc="left")
h = [Line2D([], [], marker="o", ls="", c=BLUE, ms=7), Line2D([], [], marker="s", ls="", c=ORANGE, ms=7),
     Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", mew=1.6, ms=7), Line2D([], [], marker="v", ls="", c="0.3", ms=7),
     Line2D([], [], c="0.3", ls=":", lw=1.6), Line2D([], [], c="0.3", ls="--", lw=1.6),
     Line2D([], [], marker="P", ls="", c="0.3", ms=8)]
ax.legend(h, ["real (blue)", "null (orange)", "open = random seeds, filled = module", "simulated P = 0 (at floor)",
              "theory p_est = 1 - 1/R (real, null nearly coincide)", "theory p_est exp(-3000/tau), r = 1",
              "+ module seeds, P(t=100) excitatory seeds only"], fontsize=6.8, loc="lower right", ncol=2)

# (h) lifetimes
ax = axs[2, 1]
for gr, col, mk in [("real", BLUE, "o"), ("null", ORANGE, "s")]:
    th = TH[gr]; gg = np.exp(np.linspace(np.log(th["g_lin"] * 1.02), np.log(th["g_lin"] * 4.2), 300))
    ths = [th_at(th, g) for g in gg]; xg = gg / th["g_lin"]
    ax.plot(xg, [x["tau_r1"] for x in ths], "--", c=col, lw=1.6)
    ax.plot(xg, [x["tau_r0"] for x in ths], "-.", c=col, lw=1.2)
    for key, filled in [(f"{gr}_module", True), (f"{gr}_random", False)]:
        for c in get(key):
            lh = c["late_hazard"]
            if not lh or lh["n0"] < NMIN:
                continue
            x = c["theory"]["g_over_glin"]
            if np.isfinite(lh["tau"]):
                lo, hi = lh["tau_ci"]
                ax.errorbar([x], [lh["tau"]], yerr=[[lh["tau"] - lo], [(hi if np.isfinite(hi) else lh["tau"]) - lh["tau"]]], fmt=mk,
                            c=col, mfc=col if filled else "white", mec=col, mew=1.6, ms=7, capsize=2, lw=1)
            else:
                ax.plot([x], [lh["tau_ci"][0]], "^", c=col, mfc=col if filled else "white", mec=col, mew=1.6, ms=8)
ax.axhline(3000, c="0.5", lw=0.8, ls=":"); ax.text(3.0, 2500, "T = 3000", fontsize=7, color="0.4", va="top")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3, 3e7)
ax.set_xlabel("g / g_lin"); ax.set_ylabel("lifetime (units of 1/alpha)")
ax.set_title("(h) late-time lifetime: simulation vs theory", loc="left")
h = [Line2D([], [], marker="o", ls="", c=BLUE, ms=7), Line2D([], [], marker="s", ls="", c=ORANGE, ms=7),
     Line2D([], [], marker="^", ls="", c="0.3", ms=7), Line2D([], [], c="0.3", ls="--", lw=1.6), Line2D([], [], c="0.3", ls="-.", lw=1.2)]
ax.legend(h, ["real, sim tau (t >= 30 hazard, 95% CI)", "null, sim tau", "no deaths: lower bound",
              "theory tau, r = 1", "theory tau, r = 0 (no inhibitory coincidence term)"], fontsize=7, loc="upper left")

# (i) fitted exponents
ax = axs[2, 2]
for gr, col in [("real", BLUE), ("null", ORANGE)]:
    for key, mk in [(f"{gr}_module", "o"), (f"{gr}_random", "s")]:
        for c in get(key):
            x = c["theory"]["g_over_glin"]
            for nm, sz, dx in [("fit_decay", 7, 1.0), ("fit_late", 9, 1.015)]:
                fr = c[nm]
                if not fr:
                    continue
                m = mk if nm == "fit_decay" else "^"
                ax.plot(x * dx, fr["pow"]["delta"], m, c=col, mfc=col if fr["better"] == "power" else "white", mec=col, mew=1.6, ms=sz)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("g / g_lin"); ax.set_ylabel("power-law exponent delta  (P ~ t^-delta)")
ax.set_title("(i) power-law fits of P(t): exponent and winner", loc="left")
h = [Line2D([], [], marker="o", ls="", c="0.3", ms=7), Line2D([], [], marker="s", ls="", c="0.3", ms=7),
     Line2D([], [], marker="^", ls="", c="0.3", ms=8), Line2D([], [], marker="o", ls="", c="0.3", mfc="0.3", ms=7),
     Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", mew=1.6, ms=7)]
ax.legend(h, ["module seeds, decaying range t >= 1", "random seeds, decaying range", "module/random, late range t >= 30",
              "filled: power law has higher R^2", "open: exponential has higher R^2"], fontsize=7, loc="lower left")
ax.text(0.98, 0.02, "blue = real, orange = null", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="0.3")

fig.suptitle("Single-seed spreading, stochastic Wilson-Cowan on MaleCNS, r = 1, h = 0 (512 replicas per point)", fontsize=12, y=0.995)
h = [Line2D([], [], c="0.3", lw=2), Line2D([], [], c="0.3", ls="--", lw=1.4),
     Line2D([], [], marker="D", ls="", mfc="white", mec="0.3", mew=1.8, ms=7),
     Line2D([], [], marker="v", ls="", mfc="white", mec="0.3", mew=1.8, ms=7),
     Line2D([], [], marker="x", ls="", c="0.3", mew=2, ms=7),
     Line2D([], [], marker="*", ls="", c="0.5", mec="k", mew=0.6, ms=11)]
fig.legend(h, ["simulation (512 replicas)", "theory p_est exp(-t/tau), r = 1 (panels a, b)", "theory P(T), drawn just right of T",
               f"theory P(T) below {FLOOR:g}", "last replica died", "direct-hit prediction (1000/N) x P_module(3000) (panel c)"],
           loc="upper center", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.975))
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(f"{FIG}/spreading.png"); plt.close(fig)
print("wrote", f"{OUT}/spreading.json", f"{OUT}/spreading.md", f"{FIG}/spreading.png")
