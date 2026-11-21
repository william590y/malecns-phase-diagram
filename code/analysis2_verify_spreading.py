"""Independent verification of results/analysis2/spreading.{md,json} (written by code/analysis2_spreading.py).

Written from scratch; does not import or copy analysis2_spreading.py. Reads only raw files:
  results/spread_raw_r1.npz, results/spread_raw_r1_null.npz, results/pd/mspread_real_{a,b}.npz, results/pd/mspread_null.npz,
  results/pd/long_real.npz (cross-check of lifetimes), results/seeds_module_{real,null}.npz,
  results/{,null_}sym_exc_weighted_vecs_top10.npy, results/pd/branch_{real,null}_dense_part*.json (ALL 8 parts each),
  the connectome cache (sign, pre, post, weight, types).
Checks (each printed and saved to results/analysis2/verify_spreading.json):
  V1 seed replay (own implementation of swc.py's generator sequence) + a permutation test of the check's power
  V2 P(T) per run (raw surv)
  V3 theory at r = 1 / r = 0 from the merged dense parts, two interpolation schemes (linear in ln g and in g) -> sensitivity
  V4 P(100) for excitatory seeds and by excitatory out-strength tercile (tercile split by rank, not quantile)
  V5 late hazard tau_sim: Riemann (left/right) exposure bounds, sub-window hazards [30,300] vs [300,T]
  V6 K_eff that reproduces tau_sim: own closed-form birth-death absorption time (double sum), validated on the theory rows
  V7 late-range fits at g = 0.32 and the Griffiths rule for real module seeds
  V8 random vs module: direct-hit expectation, Poisson tail, survivor seeds in/out of the set
  V9 seed-set |v0|^2 mass and 1/IPR
"""
import glob, json
import numpy as np
from scipy.stats import poisson

ROOT = "/MALECNS_ROOT"
RES, C = f"{ROOT}/results", "/MALECNS_CACHE"
N, DT = 167216, 0.2
AEFF = (1 - np.exp(-DT)) / DT
out = {}

sign = np.load(f"{C}/sign.npy")
types = np.load(f"{C}/types.npy", allow_pickle=True)
pre = np.load(f"{C}/pre.npy").astype(np.int64)
post = np.load(f"{C}/post.npy").astype(np.int64)
w = np.load(f"{C}/weight.npy").astype(np.float64)
E = len(pre)
excE = sign[pre] > 0
exc_out = np.zeros(N); np.add.at(exc_out, pre[excE], w[excE])        # excitatory out-strength (unchanged by target shuffle)


# ------------------------------------------------------------------ theory
def theory_rows(kind):
    rows, glin = [], None
    files = sorted(glob.glob(f"{RES}/pd/branch_{kind}_dense_part*.json"))
    for f in files:
        d = json.load(open(f)); rows += d["rows"]; glin = d["g_lin"]
    rows = sorted(rows, key=lambda r: r["g"])
    return glin, rows, len(files)


TH = {}
for kind in ["real", "null"]:
    glin, rows, nf = theory_rows(kind)
    TH[kind] = dict(glin=glin, nfiles=nf, g=np.array([r["g"] for r in rows]),
                    R=np.array([r["R"] for r in rows]), lam=np.array([r["lam"] for r in rows]),
                    lt1=np.array([r["logtau_r1"] for r in rows]), lt0=np.array([r["logtau_r0"] for r in rows]),
                    K1=np.array([r["K_r1"] for r in rows]), K0=np.array([r["K_r0"] for r in rows]),
                    D0=np.array([r["D_r0"] for r in rows]), D1=np.array([r["D_r1"] for r in rows]),
                    uv2=np.array([r["uv2"] for r in rows]), u2v=np.array([r["u2v"] for r in rows]))
    print(f"theory {kind}: {nf} part files, {len(rows)} rows, g_lin {glin:.4f}, g {rows[0]['g']:.4f}..{rows[-1]['g']:.3f}")


def th(kind, g, scheme="lng"):
    t = TH[kind]
    x = np.log(t["g"]) if scheme == "lng" else t["g"]
    xg = np.log(g) if scheme == "lng" else g
    xR = np.r_[np.log(t["glin"]) if scheme == "lng" else t["glin"], x]
    R = float(np.interp(xg, xR, np.r_[1.0, t["R"]]))
    inside = x[0] <= xg <= x[-1]
    f = lambda y: float(np.interp(xg, x, y)) if inside else float("nan")
    return dict(R=R, p_est=max(0.0, 1 - 1 / R), lt1=f(t["lt1"]), lt0=f(t["lt0"]), K1=f(t["K1"]), K0=f(t["K0"]), inside=inside)


# exc-exc coincidence term at r = 0: is it zero? (claim: "r = 0, which has no coincidence term")
for kind in ["real", "null"]:
    t = TH[kind]; K_noD = 1 / (t["u2v"] * t["uv2"])
    m = t["g"] < 0.7 if kind == "real" else t["g"] < 1.4
    out[f"D_r0_{kind}"] = dict(D_r0_range=[float(t["D0"][m].min()), float(t["D0"][m].max())],
                               D_r1_range=[float(t["D1"][m].min()), float(t["D1"][m].max())],
                               K_r0_range=[float(t["K0"][m].min()), float(t["K0"][m].max())],
                               K_noD_range=[float(K_noD[m].min()), float(K_noD[m].max())])
    print(f"V3a {kind}: D_r0 {t['D0'][m].min():.4g}-{t['D0'][m].max():.4g} (nonzero), D_r1 {t['D1'][m].min():.3g}-{t['D1'][m].max():.3g}; "
          f"K_r0 {t['K0'][m].min():.1f}-{t['K0'][m].max():.1f} vs K with D=0 {K_noD[m].min():.1f}-{K_noD[m].max():.1f}")


# ------------------------------------------------------------------ birth-death absorption time (independent form)
def log_tau_bd(lam, K):
    """Mean absorption time at 0 from the top state M = floor(K) of b(n) = lam n (1 - n/K) (b(M) = 0), d(n) = AEFF n.
    T_M = sum_{k<=j} (1/d_j) prod_{i=k}^{j-1} b_i/d_i  (double sum, evaluated in log space)."""
    M = int(np.floor(K))
    if M < 1:
        return -np.inf
    n = np.arange(1, M + 1, dtype=float)
    b = lam * n * (1 - n / K); d = AEFF * n
    lr = np.log(np.maximum(b[:-1], 1e-300)) - np.log(d[:-1])          # ln(b_i/d_i), i = 1..M-1
    S = np.r_[0.0, np.cumsum(lr)]                                      # S_j = sum_{i<j} ln(b_i/d_i)
    A = -np.log(d)[None, :] + S[None, :] - S[:, None]                  # [k, j]
    A[np.tril_indices(M, -1)] = -np.inf
    mx = A.max()
    return float(mx + np.log(np.exp(A - mx).sum()))


chk = []
for kind in ["real", "null"]:
    t = TH[kind]
    for i in [0, 5, 10, 20]:
        mine = log_tau_bd(t["lam"][i], t["K1"][i])
        chk.append(abs(mine - t["lt1"][i]))
out["V6_bd_validation_max_abs_diff_logtau"] = float(max(chk))
print(f"V6a own birth-death formula vs theory logtau_r1 rows: max |diff| = {max(chk):.2e}")


def K_for_tau(lam, tau):
    lo, hi = 1.5, 5000.0
    if log_tau_bd(lam, hi) < np.log(tau) or log_tau_bd(lam, lo) > np.log(tau):
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if log_tau_bd(lam, mid) < np.log(tau):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ seed sets
SS = {gr: np.load(f"{RES}/seeds_module_{gr}.npz") for gr in ["real", "null"]}
VEC = {"real": np.load(f"{RES}/sym_exc_weighted_vecs_top10.npy")[:, 0].astype(np.float64),
       "null": np.load(f"{RES}/null_sym_exc_weighted_vecs_top10.npy")[:, 0].astype(np.float64)}
inset, rank = {}, {}
for gr in ["real", "null"]:
    v = VEC[gr]; o = np.argsort(-v * v); rk = np.empty(N, int); rk[o] = np.arange(N); rank[gr] = rk
    m = np.zeros(N, bool); m[SS[gr]["idx"]] = True; inset[gr] = m
    v2 = v * v / (v * v).sum()
    out[f"V9_{gr}"] = dict(mass_top1000=float(v2[o[:1000]].sum()), inv_ipr=float(1 / (v2 ** 2).sum()),
                           set_equals_top1000=bool(set(SS[gr]["idx"].tolist()) == set(o[:1000].tolist())),
                           p_nonexc=float(SS[gr]["p"][sign[SS[gr]["idx"]] <= 0].sum()))
    print(f"V9 {gr}: |v0|^2 mass in top-1000 {out[f'V9_{gr}']['mass_top1000']:.3f}, 1/IPR {out[f'V9_{gr}']['inv_ipr']:.0f}, "
          f"set==top1000 {out[f'V9_{gr}']['set_equals_top1000']}, p on non-exc {out[f'V9_{gr}']['p_nonexc']:.3f}")

# ------------------------------------------------------------------ simulations
RUNS = [("real_random", "real", None, ["spread_raw_r1.npz"]), ("null_random", "null", None, ["spread_raw_r1_null.npz"]),
        ("real_module", "real", "real", ["pd/mspread_real_a.npz", "pd/mspread_real_b.npz"]),
        ("null_module", "null", "null", ["pd/mspread_null.npz"])]
runs = {}
tot_nonexc = tot_viol = 0
perm_viol = []
prng = np.random.default_rng(12345)
for key, gr, ssk, files in RUNS:
    for f in files:
        d = np.load(f"{RES}/{f}"); meta = json.loads(str(d["meta"]))
        gen = np.random.default_rng(meta["seed"])
        if meta["null"]:
            _ = gen.permutation(E)                                     # swc.py: post = post[rng.permutation(len(post))]
        for i, g in enumerate(d["g"]):
            B = d["rho"].shape[2]
            if ssk is None:
                seeds = gen.choice(N, B, replace=False)
            else:
                seeds = gen.choice(SS[ssk]["idx"], B, replace=True, p=SS[ssk]["p"])
            nact = np.rint(d["rho"][i] * N)                            # [t, B]
            surv = d["surv"][i] > 0.5
            ne = sign[seeds] <= 0
            spread = nact.max(0) > 1                                   # replica ever had > 1 active neuron
            viol = int((spread & ne).sum())
            tot_nonexc += int(ne.sum()); tot_viol += viol
            pv = [int((spread & (sign[prng.permutation(seeds)] <= 0)).sum()) for _ in range(20)]
            perm_viol.append(np.mean(pv))
            runs[f"{key}_{g:g}"] = dict(key=key, gr=gr, g=float(g), t=d["t"], surv=surv, nact=nact, seeds=seeds,
                                        T=float(meta["T"]), B=B, file=f)
out["V1"] = dict(nonexc_seed_replicas=tot_nonexc, violations=tot_viol, runs=len(runs),
                 expected_violations_if_replay_wrong=float(np.sum(perm_viol)))
print(f"V1 replay: {tot_nonexc} non-exc seed replicas, {tot_viol} violations, {len(runs)} runs; "
      f"a wrong (shuffled) seed assignment would give ~{np.sum(perm_viol):.0f} violations in total")


def hazard(t, n, t0, t1):
    m = (t >= t0) & (t <= t1); tt, nn = t[m], n[m].astype(float)
    k = nn[0] - nn[-1]
    ex_trap = np.trapezoid(nn, tt)
    ex_lo = np.sum(nn[1:] * np.diff(tt)); ex_hi = np.sum(nn[:-1] * np.diff(tt))
    return dict(t0=float(tt[0]), t1=float(tt[-1]), n0=int(nn[0]), deaths=int(k),
                tau=ex_trap / k if k > 0 else float("inf"), tau_lo=ex_lo / k if k > 0 else float("inf"),
                tau_hi=ex_hi / k if k > 0 else float("inf"))


tab = {}
for name, r in runs.items():
    t, surv, nact = r["t"], r["surv"], r["nact"]
    nal = surv.sum(1); P = nal / r["B"]
    i100 = int(np.argmax(t >= 100)); i3000 = int(np.where(t <= 3000 + 1e-9)[0][-1])
    ex = sign[r["seeds"]] > 0
    o = exc_out[r["seeds"]]
    exi = np.where(ex)[0]; order = exi[np.argsort(o[exi], kind="stable")]
    terc = np.array_split(order, 3)
    T = r["T"]
    th1 = th(r["gr"], r["g"], "lng"); th2 = th(r["gr"], r["g"], "lin")
    tau1, tau1b = np.exp(th1["lt1"]), np.exp(th2["lt1"])
    hz = hazard(t, nal, 30, T)
    hz_a = hazard(t, nal, 30, 300); hz_b = hazard(t, nal, 300, T)
    Kneed = K_for_tau(th1["R"] * AEFF, hz["tau"]) if np.isfinite(hz["tau"]) and hz["deaths"] > 0 else None
    alive_T = surv[-1]
    rec = dict(g=r["g"], T=T, P_T=float(P[-1]), k_T=int(nal[-1]), P_3000=float(P[i3000]), k_3000=int(nal[i3000]),
               t100=float(t[i100]), P100_all=float(P[i100]), P100_exc=float(surv[i100, ex].mean()),
               P100_terc=[float(surv[i100, b].mean()) for b in terc], n_alive_30=int(nal[np.argmax(t >= 30)]),
               Nsurv_T=float(nact[-1, alive_T].mean()) if alive_T.any() else None,
               R=th1["R"], p_est=th1["p_est"], tau_th_r1=float(tau1), tau_th_r1_lin=float(tau1b),
               tau_th_r0=float(np.exp(th1["lt0"])), K1=th1["K1"], K0=th1["K0"], th_inside=th1["inside"],
               P_th_T=float(th1["p_est"] * np.exp(-T / tau1)), P_th_T_lin=float(th2["p_est"] * np.exp(-T / tau1b)),
               hz=hz, hz_30_300=hz_a, hz_300_T=hz_b, K_needed=Kneed,
               seeds_in_set=int(inset[r["gr"]][r["seeds"]].sum()),
               survivors=[(int(rank[r["gr"]][s]), str(types[s]), bool(inset[r["gr"]][s])) for s in r["seeds"][alive_T]]
               if r["key"].endswith("random") else None)
    tab[name] = rec

print("\nV2/V3/V5/V6 per run: P(T) | theory P_th(T) [ln g interp | g interp] | P100 exc vs p_est | tau_sim [left,right Riemann] "
      "(deaths/n30) | tau 30-300 / 300-T | tau_th r1 [lin-g], r0 | K_needed (K1, K0)")
for name, x in tab.items():
    hz = x["hz"]
    print(f"{name:18s} P(T) {x['P_T']:.4f} ({x['k_T']}) | Pth {x['P_th_T']:.3g} [{x['P_th_T_lin']:.3g}] | "
          f"P100exc {x['P100_exc']:.3f} vs {x['p_est']:.3f} terc {['%.3f' % v for v in x['P100_terc']]} | "
          f"tau {hz['tau']:.4g} [{hz['tau_lo']:.4g},{hz['tau_hi']:.4g}] ({hz['deaths']}/{hz['n0']}) | "
          f"{x['hz_30_300']['tau']:.4g} ({x['hz_30_300']['deaths']}) / {x['hz_300_T']['tau']:.4g} ({x['hz_300_T']['deaths']}) | "
          f"th {x['tau_th_r1']:.4g} [{x['tau_th_r1_lin']:.4g}], r0 {x['tau_th_r0']:.4g} | "
          f"K {x['K_needed'] if x['K_needed'] is None else round(x['K_needed'], 1)} ({x['K1']:.1f}, {x['K0']:.1f}) | Nsurv {x['Nsurv_T']}")


# ------------------------------------------------------------------ V7 fits (late range) and Griffiths rule
def late_fit(name, t_lo=30.0):
    r = runs[name]; t = r["t"]; nal = r["surv"].sum(1); P = nal / r["B"]
    ok = (t >= t_lo) & (nal >= 10) & (P > 1.05 * P[-1]) & (P > 0)
    j = np.where(ok)[0]
    if len(j) == 0:
        return None
    stop = j[0]
    while stop + 1 < len(t) and ok[stop + 1]:
        stop += 1
    j = np.arange(j[0], stop + 1)
    if len(j) < 5:
        return None
    lt, lP, tt = np.log(t[j]), np.log(P[j]), t[j]
    def r2(x, y):
        c = np.polyfit(x, y, 1); res = y - np.polyval(c, x); return c, 1 - res.var() / y.var()
    cp, r2p = r2(lt, lP); ce, r2e = r2(tt, lP)
    mid = 0.5 * (lt[0] + lt[-1]); h = lt <= mid
    d1 = -np.polyfit(lt[h], lP[h], 1)[0]; d2 = -np.polyfit(lt[~h], lP[~h], 1)[0]
    return dict(t_lo=float(tt[0]), t_hi=float(tt[-1]), decades=float(np.log10(tt[-1] / tt[0])), n=len(j),
                delta=float(-cp[0]), R2_pow=float(r2p), tau_exp=float(-1 / ce[0]), R2_exp=float(r2e), d1=float(d1), d2=float(d2))


fits = {}
for name in sorted(runs):
    fits[name] = late_fit(name)
out["V7_fits_late"] = fits
for name in ["real_module_0.24", "real_module_0.28", "real_module_0.32", "real_module_0.36", "null_module_1.11", "null_module_1.15"]:
    f = fits[name]
    print(f"V7 {name}: t {f['t_lo']:.4g}-{f['t_hi']:.4g} ({f['decades']:.2f} dec, n {f['n']}): pow delta {f['delta']:.3f} R2 {f['R2_pow']:.4f}"
          f" | exp tau {f['tau_exp']:.4g} R2 {f['R2_exp']:.4f} | halves {f['d1']:.2f}/{f['d2']:.2f}")
gt = {}
for key in ["real_module", "null_module", "real_random", "null_random"]:
    names = sorted([n for n in runs if runs[n]["key"] == key], key=lambda n: runs[n]["g"])
    passes = []
    for n in names:
        f = fits[n]
        passes.append(bool(f and f["decades"] >= 1 and f["R2_pow"] >= f["R2_exp"] and abs(f["d2"] - f["d1"]) < 0.3 * f["delta"]))
    gt[key] = passes
out["V7_griffiths_passes"] = gt
print("V7 Griffiths rule passes:", {k: f"{sum(v)}/{len(v)}" for k, v in gt.items()})

# ------------------------------------------------------------------ V8 random vs module
cmpv = {}
for gr, gs in [("real", [0.45, 0.6]), ("null", [1.15, 1.25])]:
    mods = sorted([(tab[n]["g"], tab[n]["P_3000"]) for n in tab if n.startswith(f"{gr}_module")])
    gm, pm = np.array([m[0] for m in mods]), np.array([m[1] for m in mods])
    for g in gs:
        x = tab[f"{gr}_random_{g:g}"]
        Pm = float(np.interp(g, gm, pm)); lam = 512 * 1000 / N * Pm
        surv_in = sum(s[2] for s in x["survivors"])
        cmpv[f"{gr}_{g:g}"] = dict(k_3000=x["k_3000"], k_T=x["k_T"], P_mod=Pm, expected=lam,
                                   p_ge=float(poisson.sf(x["k_3000"] - 1, lam)), ratio=x["P_3000"] / Pm,
                                   catchment=N * x["P_3000"] / Pm, seeds_in_set=x["seeds_in_set"],
                                   survivors_T=len(x["survivors"]), survivors_in_set=int(surv_in),
                                   max_rank=int(max(s[0] for s in x["survivors"])) if x["survivors"] else None,
                                   Nsurv_rand=x["Nsurv_T"])
        c = cmpv[f"{gr}_{g:g}"]
        print(f"V8 {gr} g {g}: k(3000) {c['k_3000']} (k(T) {c['k_T']}), P_mod {Pm:.4f}, expected {lam:.2f}, P(X>=k) {c['p_ge']:.3g}, "
              f"ratio {c['ratio']:.4f}, catchment {c['catchment']:.0f}, survivors at T {c['survivors_T']} of which in set "
              f"{c['survivors_in_set']}, max rank {c['max_rank']}, N_surv(T) {c['Nsurv_rand']}")
        if gr == "real":
            print("     survivor seeds (rank, type, in set):", x["survivors"])
out["V8"] = cmpv
print("V8 N_surv(T) module: real 0.6", tab["real_module_0.6"]["Nsurv_T"], " null 1.15", tab["null_module_1.15"]["Nsurv_T"])

# ------------------------------------------------------------------ theory crossing tau_r1 = 3000 (real) and sim crossing
gg = np.exp(np.linspace(np.log(0.3), np.log(0.9), 2000))
tt = np.array([np.exp(th("real", g)["lt1"]) for g in gg])
out["theory_real_r1_tau_eq_3000_g"] = float(gg[np.argmax(tt >= 3000)])
print(f"theory real r=1 tau = 3000 at g = {out['theory_real_r1_tau_eq_3000_g']:.3f}")

# ------------------------------------------------------------------ cross-check lifetimes with long decay runs (T 30000, B 32)
d = np.load(f"{RES}/pd/long_real.npz")
lr = {}
for i, g in enumerate(d["g"]):
    s = d["surv"][i] > 0.5; t = d["t"]
    alive = s.any(1)
    # extinction time per replica = first recorded time it is silent (censored at T)
    ext = np.array([t[np.argmax(~s[:, b])] if (~s[:, b]).any() else np.nan for b in range(s.shape[1])])
    lr[f"{g:g}"] = dict(deaths=int(np.isfinite(ext).sum()), mean_ext=float(np.nanmean(ext)))
out["long_real_mean_extinction"] = lr
print("long_real mean extinction time from fully active (cross-check of tau_sim):", lr)

def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else str(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


out["per_run"] = tab
json.dump(clean(out), open(f"{RES}/analysis2/verify_spreading.json", "w"), indent=1)
print("wrote", f"{RES}/analysis2/verify_spreading.json")

# ------------------------------------------------------------------ V10 synapses into the seed set; shared seeds between files
for gr in ["real", "null"]:
    pp = post if gr == "real" else post[np.random.default_rng(0).permutation(E)]
    sel = excE & inset[gr][pp]
    into = np.zeros(N); np.add.at(into, pre[sel], w[sel])
    frac = float((into[sign > 0] > 0).mean())
    out[f"V10_{gr}_frac_exc_projecting_into_set"] = frac
    print(f"V10 {gr}: fraction of excitatory neurons with >= 1 exc synapse into the seed set {frac:.3f}")
    if gr == "real":
        for nm in ["real_random_0.45", "real_random_0.6"]:
            r = runs[nm]
            for s in r["seeds"][r["surv"][-1]]:
                print(f"     {nm}: seed rank {rank['real'][s]} {types[s]} exc synapses into set {into[s]:.0f}")
same = {f"{a}/{b}": bool(np.array_equal(runs[f'real_module_{a}']['seeds'], runs[f'real_module_{b}']['seeds']))
        for a, b in [("0.2", "0.36"), ("0.24", "0.42"), ("0.28", "0.5"), ("0.32", "0.6"), ("0.2", "0.24")]}
out["V10_identical_seed_draws_between_mspread_a_and_b"] = same
print("V10 identical module-seed draws (mspread_real_a vs _b, both seed 0):", same)
json.dump(clean(out), open(f"{RES}/analysis2/verify_spreading.json", "w"), indent=1)
