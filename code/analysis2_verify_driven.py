"""Adversarial verification of results/analysis2/driven.{md,json} (code/analysis2_driven.py), written independently.

Recomputes the headline numbers from the RAW files (no code shared with analysis2_driven.py):
  sim      results/pd/drive_h{1e-5..1e-1}_real.npz, results/pd/drive_h1e-3_null.npz   (rho[g,t,B], occ[g,N]; r = 1, raw norm)
  h = 0    results/fine_decay_raw_r1{,_null}.npz, results/pd/long_{real,null}.npz       (surv -> lifetimes; rho -> on-level)
  theory   results/pd/branch_{real,null}_dense_part*.json (R, logtau_r1), results/pd/qmf_drive_real.npz (r = [1])
           results/pd/flow_params.json (n_M), results/analysis2/spreading.json (null seed-set size, only for the null n)
  linresp  results/analysis2/driven_linresp.json (theory columns; sim side recomputed here). Optional --solve: an
           independent fixed-point solve of the first-order response at a few (g, h) on the real graph (CPU).
  module   results/sym_exc_weighted_vecs_top10.npy (top-1000 neurons of the leading excitatory eigenvector)
Output: results/analysis2/verify_driven.json (+ printed log).
"""
import glob, json, sys
import numpy as np
from scipy import stats

ROOT = "/MALECNS_ROOT"
R = f"{ROOT}/results"
N = 167216
DT = 0.2
POFF = 1 - np.exp(-DT)
AEFF = POFF / DT
HS = ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]
rs = np.random.default_rng(7)
NB = 2000
out = {}


def say(*a):
    print(*a, flush=True)


def load_drive(path):
    d = np.load(path)
    t, g, rho = d["t"], d["g"], d["rho"].astype(np.float64)
    m = json.loads(str(d["meta"]))
    late = t >= m["T"] / 2
    return t, g, rho, late, m, d


# ------------------------------------------------------------------ A. fluctuation statistics from raw rho
def fluct(x):
    """x [Kt, B] late fractions -> rho, chi_rep, chi_time"""
    per = x.mean(0)
    rho = per.mean()
    return rho, N * np.var(per, ddof=1), N * np.mean(np.var(x, axis=0, ddof=1))


agent = json.load(open(f"{R}/analysis2/driven.json"))
runs = {f"real_{h}": f"{R}/pd/drive_h{h}_real.npz" for h in HS}
runs["null_1e-3"] = f"{R}/pd/drive_h1e-3_null.npz"
S = {}
maxdiff = {}
for key, p in runs.items():
    t, g, rho, late, m, d = load_drive(p)
    B = rho.shape[2]
    st = np.array([fluct(rho[i][late]) for i in range(len(g))])          # [G, 3]
    # bootstrap over replicas
    bi = rs.integers(0, B, (NB, B))
    Fb = np.empty((len(g), NB)); Ftb = np.empty((len(g), NB))
    for i in range(len(g)):
        x = rho[i][late][:, bi]                                          # [Kt, NB, B]
        per = x.mean(0)
        rb = per.mean(1)
        Fb[i] = N * per.var(1, ddof=1) / rb
        Ftb[i] = N * x.var(0, ddof=1).mean(1) / rb
    S[key] = dict(t=t, g=g, rho=rho, late=late, h=m["h"], B=B, st=st, Fb=Fb, Ftb=Ftb, occ=d["occ"].astype(np.float64))
    # compare with the agent's rows
    ak = f"stage1_{key.split('_')[0]}_h{key.split('_')[1]}"
    rows = agent["runs"][ak]["rows"]
    a = np.array([[r["rho"], r["chi_rep"], r["chi_time"]] for r in rows])
    maxdiff[key] = float(np.max(np.abs(a - st) / np.abs(st)))
out["A_recompute_rows_max_rel_diff"] = maxdiff
say("A. max relative diff of (rho, chi_rep, chi_time) vs agent rows:", maxdiff)
dlog = float(np.log10(S["real_1e-5"]["g"][1] / S["real_1e-5"]["g"][0]))
out["grid_spacing_decades"] = dlog
say(f"   g grid spacing = {dlog:.4f} decades (factor {10**dlog:.3f})")

# ------------------------------------------------------------------ B. peaks of chi_rep/rho and chi_time/rho
def peak_info(key, lo, hi, which="F_rep"):
    s = S[key]; g = s["g"]
    F = s["st"][:, 1] / s["st"][:, 0] if which == "F_rep" else s["st"][:, 2] / s["st"][:, 0]
    Fb = s["Fb"] if which == "F_rep" else s["Ftb"]
    w = np.where((g >= lo) & (g <= hi))[0]
    k = w[np.argmax(F[w])]
    kb = w[np.argmax(Fb[w], 0)]
    frac = {f"{g[j]:.4g}": float((kb == j).mean()) for j in np.unique(kb)}
    cell = [float(np.sqrt(g[k - 1] * g[k])), float(np.sqrt(g[k] * g[k + 1]))] if 0 < k < len(g) - 1 else None
    outside = np.where(g > hi)[0]
    res = {"argmax_grid_g": float(g[k]), "F_at_argmax": float(F[k]), "neighbours": [float(F[k - 1]), float(F[k + 1])],
           "boot_argmax_fraction": frac, "grid_cell_bracket": cell,
           "argmax_is_window_edge": bool(k in (w[0], w[-1])),
           "P_boot_F_argmax_gt_F_window_end": float((Fb[k] > Fb[w[-1]]).mean()),
           "max_F_outside_window_above": float(F[outside].max()) if len(outside) else None,
           "g_of_max_outside": float(g[outside][np.argmax(F[outside])]) if len(outside) else None,
           "global_argmax_g": float(g[np.argmax(F)]), "global_max_F": float(F.max())}
    # single-point peak? both neighbours below half of (peak - baseline)
    base = float(np.median(F[g < 0.1632]))
    half = base + (F[k] - base) / 2
    res["single_point_peak"] = bool(F[k - 1] < half and F[k + 1] < half) if 0 < k < len(g) - 1 else None
    res["fwhm_upper_bound_decades"] = 2 * dlog if res["single_point_peak"] else None
    return res


out["B_peaks"] = {}
for key in [f"real_{h}" for h in HS]:
    out["B_peaks"][key] = {"F_rep": peak_info(key, 0.1, 1.0), "F_time": peak_info(key, 0.1, 1.0, "F_time")}
out["B_peaks"]["null_1e-3"] = {"F_rep": peak_info("null_1e-3", 0.5, 5.0), "F_time": peak_info("null_1e-3", 0.5, 5.0, "F_time")}
for k, v in out["B_peaks"].items():
    f = v["F_rep"]; ft = v["F_time"]
    say(f"B. {k}: F_rep argmax g={f['argmax_grid_g']:.4g} F={f['F_at_argmax']:.3g} nb={np.round(f['neighbours'],3)} "
        f"boot={f['boot_argmax_fraction']} edge={f['argmax_is_window_edge']} P>end={f['P_boot_F_argmax_gt_F_window_end']:.3f} "
        f"maxF(g>win)={f['max_F_outside_window_above']} at {f['g_of_max_outside']} single={f['single_point_peak']} | "
        f"F_time argmax {ft['argmax_grid_g']:.4g} F={ft['F_at_argmax']:.3g} edge={ft['argmax_is_window_edge']} P>end={ft['P_boot_F_argmax_gt_F_window_end']:.3f}")

# ------------------------------------------------------------------ C. counts, on-fraction (two thresholds), module share
th_rows = []
for p in sorted(glob.glob(f"{R}/pd/branch_real_dense_part*.json")):
    j = json.load(open(p)); th_rows += j["rows"]; G_LIN = j["g_lin"]
th_rows.sort(key=lambda x: x["g"])
TG = np.array([x["g"] for x in th_rows]); TR = np.array([x["R"] for x in th_rows]); TLT1 = np.array([x["logtau_r1"] for x in th_rows])


def R_of(g, TG=TG, TR=TR, glin=None):
    glin = G_LIN if glin is None else glin
    gx = np.r_[glin, TG]; rx = np.r_[1.0, TR]
    return np.interp(np.log(g), np.log(gx), rx)


# on-level A(g) at h = 0: mean active count of replicas alive, t in [50, 1000], fine sweep (my own read)
fd = np.load(f"{R}/fine_decay_raw_r1.npz")
fm = (fd["t"] >= 50) & (fd["t"] < 1000)
A_g, A_v = [], []
for i, gg in enumerate(fd["g"]):
    al = fd["surv"][i][fm] > 0
    if al.sum() >= 20:
        A_g.append(gg); A_v.append(fd["rho"][i][fm][al].mean() * N)
A_g = np.array(A_g); A_v = np.array(A_v)
out["C_on_level_h0"] = {"g": A_g.tolist(), "A": A_v.tolist()}


def A_of(g):
    g = np.atleast_1d(g)
    a = np.interp(np.log(g), np.log(A_g), A_v)
    lo = g < A_g[0]
    a[lo] = A_v[0] * np.clip(R_of(g[lo]) - 1, 0, None) / (R_of(A_g[0]) - 1)
    return a


V = np.load(f"{R}/sym_exc_weighted_vecs_top10.npy")
top = np.argsort(-V[:, 0] ** 2)
mod1000 = np.zeros(N, bool); mod1000[top[:1000]] = True
mod100 = np.zeros(N, bool); mod100[top[:100]] = True


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def half_cross(g, p, mode="loglin"):
    for j in range(1, len(g)):
        if np.isfinite(p[j - 1]) and np.isfinite(p[j]) and p[j - 1] < 0.5 <= p[j]:
            if mode == "loglin":
                return float(np.exp(np.interp(0.5, [p[j - 1], p[j]], np.log([g[j - 1], g[j]]))))
            y0, y1 = logit(p[j - 1]), logit(p[j])
            return float(np.exp(np.log(g[j - 1]) + (0 - y0) / (y1 - y0) * np.log(g[j] / g[j - 1])))
    return None


out["C_on"] = {}
for h in ["1e-5", "1e-4", "1e-3"]:
    s = S[f"real_{h}"]; g = s["g"]; late = s["late"]
    cnt = s["rho"][:, late, :] * N                                     # [G, Kt, B]
    noff = float(cnt[g < G_LIN].mean())
    A = A_of(g)
    thrA = noff + A / 2                                                # agent's definition, re-implemented
    thrN = noff + 6 * np.sqrt(noff + 1)                                # noise-only threshold, independent of A(g)
    res = {"n_off": noff, "rows": []}
    pA = np.full(len(g), np.nan); pN = np.full(len(g), np.nan)
    pAb = np.full((len(g), NB), np.nan); pNb = np.full((len(g), NB), np.nan)
    bi = rs.integers(0, s["B"], (NB, s["B"]))
    for i in range(len(g)):
        if g[i] > 1.0 or g[i] < G_LIN:
            continue
        sep = (A[i] / 2) / np.sqrt(noff + 1)
        onA = cnt[i] > thrA[i]; onN = cnt[i] > thrN
        if sep >= 3:
            pA[i] = onA.mean(); pAb[i] = onA[:, bi].mean((0, 2))
        pN[i] = onN.mean(); pNb[i] = onN[:, bi].mean((0, 2))
        c = cnt[i].ravel()
        on_c = c[c > thrN]; off_c = c[c <= thrN]
        occ = s["occ"][i]; r0 = np.mean(s["occ"][g < G_LIN], 0)             # per-neuron baseline from g < g_lin
        ex = occ - r0
        res["rows"].append({"g": float(g[i]), "A_h0": float(A[i]), "sep": float(sep), "p_on_A": float(pA[i]),
                            "p_on_noise6": float(pN[i]), "mean_on_count": float(on_c.mean()) if len(on_c) else None,
                            "mean_off_count": float(off_c.mean()) if len(off_c) else None,
                            "count_q05_q50_q95": np.quantile(c, [.05, .5, .95]).tolist(),
                            "module1000_share_of_excess_occ": float(ex[mod1000].sum() / ex.sum()) if ex.sum() > 0 else None,
                            "module100_share_of_excess_occ": float(ex[mod100].sum() / ex.sum()) if ex.sum() > 0 else None,
                            "excess_active_neurons": float(ex.sum())})
    for nm, pv, pb in (("A", pA, pAb), ("noise6", pN, pNb)):
        ok = np.isfinite(pv)
        gh = half_cross(g[ok], pv[ok]); ghl = half_cross(g[ok], pv[ok], "logit")
        bs = [half_cross(g[ok], pb[ok, j]) for j in range(NB)]
        bs = np.array([b for b in bs if b is not None])
        res[f"g_half_{nm}"] = gh; res[f"g_half_{nm}_logit_interp"] = ghl
        res[f"g_half_{nm}_boot_ci"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))] if len(bs) > 20 else None
        res[f"p_on_boot_{nm}_at_0.3016"] = [float(np.nanquantile(pb[g.round(4) == 0.3016][0], q)) for q in (.025, .5, .975)]
    out["C_on"][h] = res
    say(f"C. h={h}: n_off={noff:.2f}; g_half A-thr {res['g_half_A']} (logit {res['g_half_A_logit_interp']}) ci {res['g_half_A_boot_ci']}; "
        f"noise-thr {res['g_half_noise6']} (logit {res['g_half_noise6_logit_interp']}) ci {res['g_half_noise6_boot_ci']}")
    for r in res["rows"]:
        if r["g"] <= 0.45:
            say(f"     g={r['g']:.4f} A={r['A_h0']:.1f} pA={r['p_on_A']:.3f} pN={r['p_on_noise6']:.3f} on={r['mean_on_count']} off={r['mean_off_count']} "
                f"q={np.round(r['count_q05_q50_q95'],1)} mod1000 share={r['module1000_share_of_excess_occ']} mod100={r['module100_share_of_excess_occ']} exc={r['excess_active_neurons']:.1f}")
# direct same-g test: p_on at g = 0.3016, h = 1e-4 vs 1e-5 (bootstrap, independent replicas)
pa = out["C_on"]["1e-4"]["p_on_boot_noise6_at_0.3016"]; pb_ = out["C_on"]["1e-5"]["p_on_boot_noise6_at_0.3016"]
say(f"   p_on(noise thr) at g=0.3016: h=1e-4 {pa}, h=1e-5 {pb_}")

# ------------------------------------------------------------------ D. measured h = 0 lifetimes from raw survival arrays
def lifetimes(paths):
    acc = {}
    for p in paths:
        d = np.load(p); t, g, sv = d["t"], d["g"], d["surv"]; T = json.loads(str(d["meta"]))["T"]
        for i, gg in enumerate(g):
            dead = sv[i] == 0                                          # [t, B]
            died = dead.any(0)
            td = np.where(died, t[np.argmax(dead, 0)], T)              # first recorded silent time (upper bound)
            k = round(float(gg), 4)
            e, n = acc.get(k, (0.0, 0))
            acc[k] = (e + td.sum(), n + int(died.sum()))
    gs = np.array(sorted(acc))
    return gs, np.array([acc[k][0] for k in gs]), np.array([acc[k][1] for k in gs])


TAU = {}
for kind, ps in (("real", [f"{R}/fine_decay_raw_r1.npz", f"{R}/pd/long_real.npz"]),
                 ("null", [f"{R}/fine_decay_raw_r1_null.npz", f"{R}/pd/long_null.npz"])):
    gs, ex, nd = lifetimes(ps)
    ok = nd >= 5
    TAU[kind] = (gs[ok], ex[ok] / nd[ok])
    out.setdefault("D_tau_measured", {})[kind] = [{"g": float(a), "tau": float(b), "deaths": int(c)} for a, b, c in zip(gs, ex / np.maximum(nd, 1), nd)]
say("D. measured tau (pooled exposure/deaths, deaths>=5), real:",
    [(round(a, 3), round(b, 1)) for a, b in zip(*TAU["real"]) if 0.24 <= a <= 0.38])
# geometric-mean ratio vs theory (longruns' own selection: tau_sim >= 200, deaths >= 5, theory defined)
L = json.load(open(f"{R}/analysis2/longruns.json"))
rat = [x["tau_sim"] / x["tau_theory"] for x in L["lifetimes"]["real"] if x["tau_sim"] and x.get("tau_theory") and x["deaths"] >= 5 and x["tau_sim"] >= 200]
rat_all = [(x["g"], x["tau_sim"] / x["tau_theory"]) for x in L["lifetimes"]["real"] if x["tau_sim"] and x.get("tau_theory") and x["deaths"] >= 5]
out["D_ratio_sim_over_theory"] = {"geomean_tau_ge_200": float(np.exp(np.mean(np.log(rat)))), "min": float(min(rat)), "max": float(max(rat)),
                                  "all_g_with_theory": rat_all}
say(f"   tau_sim/tau_theory geomean (tau>=200) {out['D_ratio_sim_over_theory']['geomean_tau_ge_200']:.1f}, range {min(rat):.1f}-{max(rat):.1f}; "
    f"all g: {[(round(a,3), round(b,1)) for a,b in rat_all]}")

# ------------------------------------------------------------------ E. P5 roots: h n p_est tau = x
def ln_tau_meas(g, kind="real"):
    gs, tau = TAU[kind]
    return np.interp(np.log(g), np.log(gs), np.log(tau), left=np.nan, right=np.nan)


def ln_tau_th(g, TG=TG, TLT=TLT1):
    return np.interp(np.log(g), np.log(TG), TLT, left=np.nan, right=np.nan)


def p5_root(h, n, lnt, Rf, glin, x=1.0, gmax=3.0):
    gg = np.geomspace(glin * 1.001, gmax, 20000)
    y = np.log(h * n * (1 - 1 / Rf(gg))) + lnt(gg) - np.log(x)
    ok = np.isfinite(y)
    gg, y = gg[ok], y[ok]
    if y[0] >= 0:
        return ("<=", float(gg[0]))
    j = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    return float(np.exp(np.interp(0, y[j[0]:j[0] + 2], np.log(gg[j[0]:j[0] + 2])))) if len(j) else None


fp = json.load(open(f"{R}/pd/flow_params.json")); nM = fp["n_M"]
out["E_P5"] = {}
for h in ["1e-5", "1e-4", "1e-3"]:
    hv = float(h)
    e = {f"meas_x{x:g}": p5_root(hv, nM, ln_tau_meas, R_of, G_LIN, x) for x in (1, 2)}
    e.update({f"theory_r1_x{x:g}": p5_root(hv, nM, ln_tau_th, R_of, G_LIN, x) for x in (1, 2)})
    out["E_P5"][h] = e
    say(f"E. P5 real h={h} n={nM}: measured tau {e['meas_x1']} (x2 {e['meas_x2']}); theory tau r=1 {e['theory_r1_x1']} (x2 {e['theory_r1_x2']})")
ths = {}
for h in ["1e-5", "1e-4"]:
    ths[h] = out["E_P5"][h]
dm = np.log(ths["1e-4"]["meas_x1"] / ths["1e-5"]["meas_x1"]); dt_ = np.log(ths["1e-4"]["theory_r1_x1"] / ths["1e-5"]["theory_r1_x1"])
out["E_P5"]["dln_pred_meas"] = float(dm); out["E_P5"]["dln_pred_theory"] = float(dt_)
ghA5 = out["C_on"]["1e-5"]["g_half_A"]; ghA4 = out["C_on"]["1e-4"]["g_half_A"]
out["E_P5"]["dln_sim_A"] = float(np.log(ghA4 / ghA5))
out["E_P5"]["dln_sim_A_logit"] = float(np.log(out["C_on"]["1e-4"]["g_half_A_logit_interp"] / out["C_on"]["1e-5"]["g_half_A_logit_interp"]))
gN5, gN4 = out["C_on"]["1e-5"]["g_half_noise6"], out["C_on"]["1e-4"]["g_half_noise6"]
out["E_P5"]["dln_sim_noise6"] = float(np.log(gN4 / gN5)) if gN5 and gN4 else None
say(f"   predicted dln g (1e-5 -> 1e-4): measured {dm:.3f}, theory {dt_:.3f}; sim (A-thr) {out['E_P5']['dln_sim_A']:.3f}, "
    f"logit-interp {out['E_P5']['dln_sim_A_logit']:.3f}, noise-thr {out['E_P5']['dln_sim_noise6']}")
# null
nrows = []
for p in sorted(glob.glob(f"{R}/pd/branch_null_dense_part*.json")):
    j = json.load(open(p)); nrows += j["rows"]; G_LIN_N = j["g_lin"]
nrows.sort(key=lambda x: x["g"])
NG = np.array([x["g"] for x in nrows]); NR = np.array([x["R"] for x in nrows]); NLT = np.array([x["logtau_r1"] for x in nrows])
n_null = json.load(open(f"{R}/analysis2/spreading.json"))["seed_sets"]["null"]["eff_size"]
RfN = lambda g: R_of(g, NG, NR, G_LIN_N)
out["E_P5"]["null_1e-3"] = {"n": n_null, "meas": p5_root(1e-3, n_null, lambda g: ln_tau_meas(g, "null"), RfN, G_LIN_N),
                            "theory_r1": p5_root(1e-3, n_null, lambda g: ln_tau_th(g, NG, NLT), RfN, G_LIN_N)}
say(f"   null h=1e-3 n={n_null:.1f}: {out['E_P5']['null_1e-3']}")

# ------------------------------------------------------------------ F. telegraph module variance at h = 1e-5, g = 0.3016 (Monte Carlo)
def telegraph_mc(g, h, n, tl, nrep=400000):
    tau = float(np.exp(ln_tau_meas(np.array([g]))[0])); nu = h * n * (1 - 1 / R_of(np.array([g]))[0])
    k_on, k_off = nu, 1 / tau
    p = k_on / (k_on + k_off); lam = k_on + k_off
    s = np.ones(nrep, bool); tprev = 0.0; acc = np.zeros(nrep)
    for tt in tl:
        e = np.exp(-lam * (tt - tprev))
        p_stay_on = p + (1 - p) * e; p_turn_on = p * (1 - e)
        u = rs.random(nrep)
        s = np.where(s, u < p_stay_on, u < p_turn_on)
        acc += s; tprev = tt
    m = acc / len(tl)
    return {"tau": tau, "nu": nu, "nu_tau": nu * tau, "p_stat": p, "p_on_window": float(m.mean()), "var_timeavg": float(m.var())}


s5 = S["real_1e-5"]; tl = s5["t"][s5["late"]]
i30 = int(np.argmin(np.abs(s5["g"] - 0.3016)))
tm = telegraph_mc(s5["g"][i30], 1e-5, nM, tl)
A30 = float(A_of(np.array([s5["g"][i30]]))[0])
chi_mod = A30 ** 2 / N * tm["var_timeavg"]
xb = s5["rho"][i30][s5["late"]][:, rs.integers(0, s5["B"], (NB, s5["B"]))].mean(0)       # [NB, B]
chib = N * xb.var(1, ddof=1)
chi_ci = np.quantile(chib, [.025, .975])
out["F_telegraph_h1e-5_g0.3016"] = dict(tm, A=A30, chi_model=chi_mod, chi_sim=float(s5["st"][i30, 1]), ratio=float(chi_mod / s5["st"][i30, 1]), chi_sim_ci=chi_ci.tolist(), ratio_range=[float(chi_mod / chi_ci[1]), float(chi_mod / chi_ci[0])])
say(f"F. telegraph MC h=1e-5 g={s5['g'][i30]:.4f}: {out['F_telegraph_h1e-5_g0.3016']}")

# ------------------------------------------------------------------ G. QMF ratios
q = np.load(f"{R}/pd/qmf_drive_real.npz")
assert np.allclose(q["r"], [1.0])
ql = q["t"] >= q["t"][-1] / 2
out["G_qmf"] = {}
for ih, h in enumerate(HS):
    s = S[f"real_{h}"]; g = s["g"]
    assert np.allclose(q["g"], g, rtol=1e-3) and np.isclose(q["h"][ih], float(h))
    qr = q["rho"][0, ih][:, ql].mean(1)
    ratio = s["st"][:, 0] / qr
    seg = lambda a, b: ratio[(g >= a) & (g < b)]
    out["G_qmf"][h] = {"med_below": float(np.median(seg(0, G_LIN))), "max_abs_dev_below": float(np.max(np.abs(seg(0, G_LIN) - 1))),
                       "med_glin_0.5": float(np.median(seg(G_LIN, 0.5))), "med_0.5_3": float(np.median(seg(0.5, 3.01))),
                       "range_0.5_3": [float(seg(0.5, 3.01).min()), float(seg(0.5, 3.01).max())],
                       "max_abs_dev_g_le_3": float(np.max(np.abs(ratio[g <= 3.01] - 1))),
                       "min": float(ratio.min()), "g_min": float(g[np.argmin(ratio)])}
    say(f"G. h={h}: {out['G_qmf'][h]}")

# ------------------------------------------------------------------ H. linear response: sim side + chi2
LR = json.load(open(f"{R}/analysis2/driven_linresp.json"))
out["H_linresp"] = {}
for h in HS[:4]:
    s = S[f"real_{h}"]; g = s["g"]; hv = float(h)
    q0 = 1 - np.exp(-np.tanh(hv) * DT); r0 = q0 / (q0 + POFF)
    chi = {"exact": 0.0, "B_only": 0.0, "zero": 0.0}; pts = []
    for i in np.where(g < G_LIN)[0]:
        per = s["rho"][i][s["late"]].mean(0)
        ex = per.mean() / r0 - 1; se = per.std(ddof=1) / np.sqrt(s["B"]) / r0
        e = [x for x in LR["rows"] if x["variant"] == "exact" and np.isclose(x["h"], hv) and abs(x["g"] - g[i]) < 5e-4][0]["excess_rel"]
        b = [x for x in LR["rows"] if x["variant"] == "B_only" and abs(x["g"] - g[i]) < 5e-4][0]["excess_rel"]
        chi["exact"] += ((ex - e) / se) ** 2; chi["B_only"] += ((ex - b) / se) ** 2; chi["zero"] += (ex / se) ** 2
        pole = LR["pole"]["amp_sumv_sumu_over_N"] * (g[i] / G_LIN) / (1 - g[i] / G_LIN)
        pts.append({"g": float(g[i]), "sim": float(ex), "se": float(se), "exact": e, "B_only": b, "pole": float(pole),
                    "pole_over_B_only": float(pole / b), "pole_over_sim": float(pole / ex)})
    chi = {k: float(v) for k, v in chi.items()}
    chi["p_exact_chi2_3dof"] = float(stats.chi2.sf(chi["exact"], 3))
    out["H_linresp"][h] = {"chi2": chi, "points": pts}
    say(f"H. h={h}: chi2 {chi}; pole/B_only {[round(p['pole_over_B_only'],3) for p in pts]}")

# ------------------------------------------------------------------ I. bimodality coefficient of pooled late counts, g_lin < g <= 0.5
def bc(c, bias):
    n = len(c); sk = stats.skew(c, bias=bias); ku = stats.kurtosis(c, bias=bias)
    return (sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


out["I_bimodality"] = {}
for h in ["1e-5", "1e-4", "1e-3"]:
    s = S[f"real_{h}"]; g = s["g"]
    sel = np.where((g > G_LIN) & (g <= 0.5))[0]
    v_b = [bc(s["rho"][i][s["late"]].ravel() * N, True) for i in sel]
    v_u = [bc(s["rho"][i][s["late"]].ravel() * N, False) for i in sel]
    out["I_bimodality"][h] = {"g": g[sel].tolist(), "BC_biased": v_b, "BC_unbiased": v_u}
    say(f"I. h={h}: g {np.round(g[sel],3)} BC biased {np.round(v_b,2)} unbiased {np.round(v_u,2)}")

# ------------------------------------------------------------------ J. drift
dr = []
for key, s in S.items():
    for i in range(len(s["g"])):
        x = s["rho"][i][s["late"]]
        dr.append((key, float(s["g"][i]), float((x[-5:].mean() - x[:5].mean()) / x.mean())))
out["J_max_abs_drift"] = max(dr, key=lambda z: abs(z[2]))
say("J. max |drift| over all runs/g:", out["J_max_abs_drift"])

# ------------------------------------------------------------------ K. optional independent first-order solve on the graph
# ------------------------------------------------------------------ L. module size n the telegraph model needs (both thresholds)
def p_window(n, g, h, tl):
    tau = float(np.exp(ln_tau_meas(np.array([g]))[0])); pe = 1 - 1 / R_of(np.array([g]))[0]
    x = h * n * pe * tau; p = x / (1 + x); tc = tau / (1 + x)
    return float(np.mean(p + (1 - p) * np.exp(-np.asarray(tl) / tc)))


def n_for(pt, g, h, tl):
    lo, hi = np.log(1e-2), np.log(1e7)
    if not (p_window(np.exp(lo), g, h, tl) < pt < p_window(np.exp(hi), g, h, tl)):
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_window(np.exp(mid), g, h, tl) < pt else (lo, mid)
    return float(np.exp(mid))


out["L_n_needed"] = []
for h in ["1e-5", "1e-4"]:
    s = S[f"real_{h}"]; tl = s["t"][s["late"]]
    for r in out["C_on"][h]["rows"]:
        if r["g"] < 0.24 or r["g"] > 0.31:
            continue
        i = int(np.argmin(np.abs(s["g"] - r["g"])))
        for nm, key in (("A", "p_on_A"), ("noise6", "p_on_noise6")):
            pv = r[key]
            if not np.isfinite(pv) or not (0.02 < pv < 0.98):
                continue
            # replica-bootstrap CI of p_on with this threshold
            cnt = s["rho"][i][s["late"]] * N
            thr = (out["C_on"][h]["n_off"] + r["A_h0"] / 2) if nm == "A" else (out["C_on"][h]["n_off"] + 6 * np.sqrt(out["C_on"][h]["n_off"] + 1))
            on = cnt > thr
            bi = rs.integers(0, s["B"], (NB, s["B"]))
            pb = on[:, bi].mean((0, 2))
            ci = np.quantile(pb, [.025, .975])
            out["L_n_needed"].append({"h": h, "g": r["g"], "threshold": nm, "p_on": pv, "p_ci": ci.tolist(),
                                      "n": n_for(pv, r["g"], float(h), tl),
                                      "n_ci": [n_for(max(ci[0], 1e-3), r["g"], float(h), tl), n_for(min(ci[1], 0.999), r["g"], float(h), tl)]})
for x in out["L_n_needed"]:
    say(f"L. h={x['h']} g={x['g']:.4f} thr={x['threshold']}: p_on {x['p_on']:.3f} CI {np.round(x['p_ci'],3)} -> n {x['n']} CI {x['n_ci']}")

# ------------------------------------------------------------------ M. telegraph amplitude at h = 1e-3 (agent: 0.17 measured tau at g 0.251, 0.34 theory tau at g 0.362)
s3 = S["real_1e-3"]; tl3 = s3["t"][s3["late"]]
M = {}
for nm, gq, lnt in (("measured", 0.2509, ln_tau_meas), ("theory_r1", 0.3625, ln_tau_th)):
    i = int(np.argmin(np.abs(s3["g"] - gq))); gg = s3["g"][i]
    tau = float(np.exp(lnt(np.array([gg]))[0])); nu = 1e-3 * nM * (1 - 1 / R_of(np.array([gg]))[0])
    k = nu + 1 / tau; p = nu / k
    T1, T2 = np.meshgrid(tl3, tl3, indexing="ij")
    Pt = p + (1 - p) * np.exp(-k * tl3)
    Pmin = p + (1 - p) * np.exp(-k * np.minimum(T1, T2))
    cov = Pmin * (p + (1 - p) * np.exp(-k * np.abs(T1 - T2))) - np.outer(Pt, Pt)
    Ai = float(A_of(np.array([gg]))[0])
    chi_m = Ai ** 2 / N * cov.mean()
    M[nm] = {"g": float(gg), "tau": tau, "nu_tau": nu * tau, "A": Ai, "chi_model": float(chi_m), "chi_sim": float(s3["st"][i, 1]),
             "ratio": float(chi_m / s3["st"][i, 1])}
out["M_telegraph_h1e-3"] = M
say("M.", M)

if "--solve" in sys.argv:
    C = "/MALECNS_CACHE"
    import scipy.sparse as sp
    pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
    w = np.load(C + "/weight.npy").astype(np.float64); sg = np.load(C + "/sign.npy")
    ins = np.bincount(post, weights=w, minlength=N); c = 1.0 / ins[ins > 0].mean()
    sv = np.where(sg[pre] > 0, 1.0, np.where(sg[pre] < 0, -1.0, 0.0)) * c * w        # r = 1
    nz = sv != 0
    pre, post, sv = pre[nz], post[nz], sv[nz]
    F = lambda x: np.tanh(np.maximum(x, 0))
    res = []
    for g, h, var in [(0.14448218, 1e-2, "exact"), (0.14448218, 1e-2, "B_only"), (0.14448218, 1e-3, "exact"), (0.1, 1e-3, "exact")]:
        q0 = 1 - np.exp(-F(h) * DT)
        if var == "exact":                                   # P(on | one active input j) - P(on | none)
            a = (1 - np.exp(-F(h + g * sv) * DT)) - q0
        else:
            a = np.where(sv > 0, 1 - np.exp(-F(g * sv) * DT), 0.0)
            q0 = 1 - np.exp(-F(1e-5) * DT)
        M = sp.coo_matrix((a, (post, pre)), shape=(N, N)).tocsr()
        r0 = q0 / (q0 + POFF); x = np.full(N, r0)
        for it in range(5000):
            Q = np.maximum(q0 + M @ x, 0); xn = Q / (Q + POFF)
            if np.max(np.abs(xn - x)) < 1e-8 * r0:
                x = xn; break
            x = xn
        res.append({"g": g, "h": h, "variant": var, "excess_rel": float(x.mean() / r0 - 1), "iters": it + 1})
        say("K.", res[-1])
    out["K_solve"] = res

json.dump(out, open(f"{R}/analysis2/verify_driven.json", "w"), indent=1, default=float)
say("wrote", f"{R}/analysis2/verify_driven.json")
