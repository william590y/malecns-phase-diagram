"""Adversarial verification of the pd3 'multistability' program (results/pd3/multistability.md).

Independent re-computation from the RAW outputs in results/pd3/multistab/*.npz (does NOT import pd3_multistab_* code; only numpy,
scipy, sklearn).  Writes results/pd3/verify_multistability.json.

Checks
  V0  normalisation constant c of swc.py --norm raw (from the cache)
  V1  validation: censored-exponential lifetime at r = 1, g = 0.30 (exact and snapped to swc.py's log record grid)
  V2  M1: t_sym medians, late asymmetry, the three broken-symmetry criteria, KS L vs R, dead points, closure-window points
  V3  M2a: loop areas A_rho (deterministic), own replica bootstrap SE, own random-half noise floor, own label-PERMUTATION test of
      A_rho (whole-column swap of up/down replicas), A(800)/A(200), 3-point log-log slope; PR areas; h = 0 down-sweep extinction g
  V4  M2b: KS p of late rho / PR_rep from up- vs down-branch end-states; first-10-model-time active-count KS
  V5  M2c: A_rho in r, AL fraction at g = 0.6, one-lobe dwell audit
  V6  M3: survival per start, max relative L1 between start means, single-cluster check (distance to the grand mean)
  V7  M3b: projections z, median differences, KS p-values, extinct cells, early-vs-late drift
"""
import glob, json, os
import numpy as np
from scipy import stats

ROOT = "/MALECNS_ROOT"
D = f"{ROOT}/results/pd3/multistab"
CACHE = "/MALECNS_CACHE"
DT = 0.2
OUT = f"{ROOT}/results/pd3/verify_multistability.json"
rng = np.random.default_rng(20261109)
res = {}


def meta(z):
    return json.loads(str(z["meta"]))


def fl(x):
    if isinstance(x, dict):
        return {str(k): fl(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [fl(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return fl(x.tolist())
    return x


# ------------------------------------------------------------------------------------------------------------------ V0
ins_post = np.load(CACHE + "/post.npy"); ins_w = np.load(CACHE + "/weight.npy").astype(np.float64)
N = len(np.load(CACHE + "/sign.npy"))
ins = np.bincount(ins_post, weights=ins_w, minlength=N)
res["V0_norm"] = {"N": N, "n_nonzero_in": int((ins > 0).sum()), "mean_in_nonzero": float(ins[ins > 0].mean()),
                  "total_over_N": float(ins.sum() / N)}
del ins_post, ins_w
print("V0", res["V0_norm"], flush=True)

# ------------------------------------------------------------------------------------------------------------------ V1
z = np.load(f"{D}/validate_r1.npz", allow_pickle=True); m = meta(z)
G, B, T = m["g"], m["B"], m["T"]; steps = int(round(T / DT))
rec = np.unique(np.round(np.logspace(0, np.log10(steps), 120)).astype(int))
ext = z["ext_step"]
v1 = []
for i, g in enumerate(G):
    e = ext[i * B:(i + 1) * B]; died = e >= 0; d = int(died.sum())
    te = np.where(died, e * DT, T)
    ts = np.where(died, rec[np.clip(np.searchsorted(rec, np.maximum(e, 1)), 0, len(rec) - 1)] * DT, T)
    E_e, E_s = te.sum(), ts.sum()
    ci = [2 * E_s / stats.chi2.ppf(0.975, 2 * d + 2), 2 * E_s / stats.chi2.ppf(0.025, 2 * d)] if d else [None, None]
    v1.append({"g": g, "deaths": d, "tau_exact": E_e / d if d else None, "tau_snapped": E_s / d if d else None, "ci_snapped": ci})
res["V1_validation"] = v1
pc = json.load(open(f"{ROOT}/results/analysis2/phase_complete.json"))["sim"]["real"]["1"]["points"]
p30 = [p for p in pc if abs(p["g"] - 0.30) < 1e-6]
res["V1_pooled_g0.30"] = {k: p30[0][k] for k in ("g", "B", "deaths", "tau_sim", "tau_ci")} if p30 else None
print("V1", v1[0], res["V1_pooled_g0.30"], flush=True)

# ------------------------------------------------------------------------------------------------------------------ V2
SZ_L, SZ_R = 248, 275
GON = {0: 0.266, 0.5: 0.306, 1: 0.344, 2: 0.402, 3: 0.456, 4: 0.508, 6: 0.579}
B1_FROM = {3: 0.2471, 4: 0.2871, 6: 0.4280}            # theory_module2.json coarse_2module (checked below)
tm2 = json.load(open(f"{ROOT}/results/pd3/theory_module2.json"))
res["V2_theory_module2_keys"] = list(tm2)[:20]


def asym(nl, nr):
    fL, fR = nl / SZ_L, nr / SZ_R
    with np.errstate(all="ignore"):
        return np.where(fL + fR > 0, (fL - fR) / (fL + fR), np.nan)


def gmm_dbic(x):
    from sklearn.mixture import GaussianMixture
    x = np.asarray(x)[:, None]
    if len(x) < 10:
        return None
    g1 = GaussianMixture(1, random_state=1).fit(x); g2 = GaussianMixture(2, n_init=8, random_state=1).fit(x)
    mu = g2.means_.ravel(); sd = np.sqrt(g2.covariances_.ravel())
    return {"dBIC": float(g1.bic(x) - g2.bic(x)), "wmin": float(g2.weights_.min()), "sep_over_maxsd": float(abs(mu[0] - mu[1]) / sd.max())}


m1 = []
for f in sorted(glob.glob(f"{D}/al_sym_r*.npz")):
    z = np.load(f, allow_pickle=True); mm = meta(z); r = mm["r"]; Gs = mm["g"]; Bm = mm["B"]; starts = mm["starts"]
    modc = z["modc"]; fine = z["fine"]; ex = z["ext_step"]; exal = z["ext_al_step"]
    for gi, g in enumerate(Gs):
        A = asym(modc[gi, :, 0, :].astype(float), modc[gi, :, 1, :].astype(float))      # every model time
        Af = asym(fine[gi, :, 0, :].astype(float), fine[gi, :, 1, :].astype(float))     # every step, t <= 100
        tc = np.arange(A.shape[0]) * 1.0; tf = np.arange(Af.shape[0]) * DT
        late = (tc >= 500) & (tc <= 1000)
        row = {"r": r, "g": g, "x_gon": g / GON[r], "closure_window": bool(r in B1_FROM and B1_FROM[r] <= g <= 3.0), "starts": {}}
        lateA = {}
        for si, s in enumerate(starts):
            cols = np.arange(si * Bm, (si + 1) * Bm)
            al = A[late][:, cols]; ok = np.isfinite(al)
            surv = (ex[gi, cols] < 0) & (ok.mean(0) >= 0.5)
            with np.errstate(all="ignore"):
                mA = np.nansum(np.where(ok, al, 0), 0) / ok.sum(0)
            lateA[s] = mA[surv]
            d = {"n_surv": int(surv.sum()), "all_dead_before_500": bool(np.all((ex[gi, cols] >= 0) & (ex[gi, cols] * DT < 500))),
                 "late_mean_A": float(mA[surv].mean()) if surv.any() else None,
                 "max_abs_replica_A": float(np.abs(mA[surv]).max()) if surv.any() else None}
            if s in ("AL-L", "AL-R"):
                ts = np.full(Bm, np.nan); died_first = 0; cens = 0
                for j, c in enumerate(cols):
                    h = np.where(np.abs(Af[:, c]) < 0.3)[0]
                    if len(h):
                        ts[j] = tf[h[0]]; continue
                    h = np.where((np.abs(A[:, c]) < 0.3) & (tc > tf[-1]))[0]
                    eal = exal[gi, c] * DT if exal[gi, c] >= 0 else np.inf
                    if len(h) and tc[h[0]] < eal:
                        ts[j] = tc[h[0]]
                    elif np.isfinite(eal):
                        died_first += 1
                    else:
                        cens += 1
                okk = np.isfinite(ts)
                # coarse-only variant (modc every model time, no fine record) as a robustness check
                ts2 = np.array([tc[np.where(np.abs(A[:, c]) < 0.3)[0][0]] if (np.abs(A[:, c]) < 0.3).any() else np.nan for c in cols])
                d.update(t_sym_median=float(np.median(ts[okk])) if okk.any() else None,
                         t_sym_q75=float(np.percentile(ts[okk], 75)) if okk.any() else None,
                         t_sym_max=float(ts[okk].max()) if okk.any() else None, n_died_first=died_first, n_censored=cens,
                         t_sym_median_coarse=float(np.nanmedian(ts2)) if np.isfinite(ts2).any() else None,
                         frac_keep=float(((mA[surv] > 0.6) if s == "AL-L" else (mA[surv] < -0.6)).mean()) if surv.any() else None)
            row["starts"][s] = d
        L, R = lateA["AL-L"], lateA["AL-R"]
        row["ks_LR_p"] = float(stats.ks_2samp(L, R).pvalue) if len(L) >= 2 and len(R) >= 2 else None
        row["c1"] = bool((row["starts"]["AL-L"]["frac_keep"] or 0) >= 0.25)
        row["c2"] = bool(row["ks_LR_p"] is not None and row["ks_LR_p"] < 1e-3)
        bm = gmm_dbic(lateA["all"])
        row["gmm_all"] = bm
        row["c3"] = bool(bm is not None and bm["dBIC"] > 10 and bm["wmin"] >= 0.1 and bm["sep_over_maxsd"] > 2)
        m1.append(row)
    del modc, fine
    print("V2 done r", r, flush=True)
tmed = [row["starts"][s]["t_sym_median"] for row in m1 for s in ("AL-L", "AL-R") if row["starts"][s]["t_sym_median"] is not None]
lam = [row["starts"][s]["late_mean_A"] for row in m1 for s in row["starts"] if row["starts"][s]["n_surv"] >= 10]
kp = [(row["ks_LR_p"], row["r"], row["g"]) for row in m1 if row["ks_LR_p"] is not None]
win = [row for row in m1 if row["closure_window"]]
res["V2_M1"] = {
    "n_points": len(m1), "n_c1": sum(r_["c1"] for r_ in m1), "n_c2": sum(r_["c2"] for r_ in m1), "n_c3": sum(r_["c3"] for r_ in m1),
    "t_sym_median_range": [min(tmed), max(tmed)],
    "t_sym_q75_max": max(row["starts"][s]["t_sym_q75"] or 0 for row in m1 for s in ("AL-L", "AL-R")),
    "t_sym_max": max(row["starts"][s]["t_sym_max"] or 0 for row in m1 for s in ("AL-L", "AL-R")),
    "n_died_first": sum(row["starts"][s]["n_died_first"] for row in m1 for s in ("AL-L", "AL-R")),
    "n_censored": sum(row["starts"][s]["n_censored"] for row in m1 for s in ("AL-L", "AL-R")),
    "coarse_t_sym_median_range": [min(x for x in (row["starts"][s]["t_sym_median_coarse"] for row in m1 for s in ("AL-L", "AL-R")) if x is not None),
                                  max(x for x in (row["starts"][s]["t_sym_median_coarse"] for row in m1 for s in ("AL-L", "AL-R")) if x is not None)],
    "late_mean_A_range_ge10": [min(lam), max(lam)],
    "max_abs_replica_A": max(row["starts"][s]["max_abs_replica_A"] or 0 for row in m1 for s in row["starts"]),
    "n_ks_tests": len(kp), "min_ks_p": min(kp), "max_frac_keep": max((row["starts"][s]["frac_keep"] or 0) for row in m1 for s in ("AL-L", "AL-R")),
    "max_dBIC_all": max(((row["gmm_all"] or {}).get("dBIC", -1e9), row["r"], row["g"]) for row in m1),
    "n_closure_window_points": len(win),
    "window_all_dead_before_500": [(row["r"], row["g"]) for row in win if all(row["starts"][s]["all_dead_before_500"] for s in row["starts"])],
    "window_points_ge10_surv_one_lobe": sum(1 for row in win if min(row["starts"]["AL-L"]["n_surv"], row["starts"]["AL-R"]["n_surv"]) >= 10),
    "window_ks_range_ge10": [min(row["ks_LR_p"] for row in win if min(row["starts"]["AL-L"]["n_surv"], row["starts"]["AL-R"]["n_surv"]) >= 10),
                             max(row["ks_LR_p"] for row in win if min(row["starts"]["AL-L"]["n_surv"], row["starts"]["AL-R"]["n_surv"]) >= 10)],
    "r4_g0.40_surv": {s: row["starts"][s]["n_surv"] for row in m1 if row["r"] == 4 and abs(row["g"] - 0.40) < 1e-9 for s in row["starts"]},
    "late_mean_A_by_start_mean": {s: float(np.mean([row["starts"][s]["late_mean_A"] for row in m1 if row["starts"][s]["n_surv"] >= 10])) for s in ("AL-L", "AL-R", "AL", "all")},
    "rows": [{k: v for k, v in row.items()} for row in m1]}
print("V2", {k: v for k, v in res["V2_M1"].items() if k != "rows"}, flush=True)

# ------------------------------------------------------------------------------------------------------------------ V3
def trap(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def area(u, d, x):          # u, d: [K, n] per-replica rho
    return trap(np.abs(np.log(np.maximum(u.mean(1), 1e-12)) - np.log(np.maximum(d.mean(1), 1e-12))), x)


m2a = {}
h0 = {}
hm = np.load(f"{ROOT}/results/pd3/redraw/heatmap_real.npz")
for f in sorted(glob.glob(f"{D}/sweep_r*_h*_D*.npz")):
    z = np.load(f, allow_pickle=True); mm = meta(z); r, h, Dw = mm["r"], mm["h"], mm["D"]
    Gg = np.array(mm["g_grid"]); K = len(Gg); Bs = mm["B"]; rho = z["rho"].astype(np.float64)
    if mm.get("down_only"):
        st = int(round(Dw / DT)); e = z["ext_step"]
        seg = (e - 1) // st
        gext = np.where(e >= 0, Gg[K - 1 - np.clip(seg, 0, K - 1)], np.nan)
        i = int(np.argmin(np.abs(hm["r"] - r))); lt = hm["ln_tau_iso"][i]; gg = hm["g"]
        k = np.where(lt >= np.log(Dw))[0][0]
        fr = (np.log(Dw) - lt[k - 1]) / (lt[k] - lt[k - 1]); gx = float(np.exp(np.log(gg[k - 1]) + fr * np.log(gg[k] / gg[k - 1])))
        # own prediction: survival through each dwell (descending g) = exp(-D / tau(g))
        tau = np.exp(np.interp(np.log(Gg[::-1]), np.log(gg), lt))
        S = np.exp(-np.cumsum(Dw / tau)); pdie = np.r_[1, S[:-1]] - S
        cdf = np.cumsum(pdie); pmed = float(Gg[::-1][np.searchsorted(cdf, 0.5)])
        # exact discrete goodness of fit: chi-square-free -> compare observed counts per grid g with multinomial expectation
        h0[f"{r:g}"] = {"n_died": int(np.isfinite(gext).sum()), "median": float(np.nanmedian(gext)),
                        "iqr": [float(np.nanpercentile(gext, 25)), float(np.nanpercentile(gext, 75))], "g_tau200": gx, "pred_median": pmed,
                        "obs_counts": {f"{x:.4f}": int((gext == x).sum()) for x in np.unique(gext[np.isfinite(gext)])},
                        "pred_probs": {f"{Gg[::-1][j]:.4f}": float(pdie[j]) for j in range(K) if pdie[j] > 0.01}}
        continue
    up = rho[:, :Bs]; dn = rho[::-1, Bs:]
    x = np.log(Gg)
    A = area(up, dn, x)
    bs = np.array([area(up[:, rng.integers(0, Bs, Bs)], dn[:, rng.integers(0, Bs, Bs)], x) for _ in range(1000)])
    fls = []
    for arr in (up, dn):
        for _ in range(100):
            p = rng.permutation(Bs)
            fls.append(area(arr[:, p[:Bs // 2]], arr[:, p[Bs // 2:]], x) / np.sqrt(2))
    allc = np.concatenate([up, dn], 1)
    perm = np.array([(lambda p: area(allc[:, p[:Bs]], allc[:, p[Bs:]], x))(rng.permutation(2 * Bs)) for _ in range(2000)])
    won = Gg <= 2.5 * GON[r]
    dl = np.abs(np.log(np.maximum(up.mean(1), 1e-12)) - np.log(np.maximum(dn.mean(1), 1e-12)))
    prm = z["PR_mean"]; A_pr = trap(np.abs(prm[:, 0] - prm[::-1, 1]), x)
    m2a[f"r{r:g}_h{h:g}_D{Dw:g}"] = {"r": r, "h": h, "D": Dw, "A_rho": A, "se_boot": float(bs.std()), "floor_halves": float(np.mean(fls)),
                                     "perm_mean": float(perm.mean()), "perm_p": float((1 + (perm >= A).sum()) / (1 + len(perm))),
                                     "A_onset_window": area(up[won], dn[won], x[won]), "A_above_window": area(up[~won], dn[~won], x[~won]),
                                     "max_dlnrho": float(dl.max()), "g_max_dlnrho": float(Gg[int(np.argmax(dl))]), "A_PR": A_pr}
    print("V3", f, {k: round(v, 4) if isinstance(v, float) else v for k, v in m2a[f"r{r:g}_h{h:g}_D{Dw:g}"].items()}, flush=True)
dec = {}
for key, v in m2a.items():
    if v["D"] != 800:
        continue
    k2 = key.replace("D800", "D200"); k5 = key.replace("D800", "D50")
    As = [m2a[k5]["A_rho"], m2a[k2]["A_rho"], v["A_rho"]]
    dec[key] = {"A50": As[0], "A200": As[1], "A800": As[2], "ratio_800_200": As[2] / As[1], "ratio_50_800": As[0] / As[2],
                "slope3": float(np.polyfit(np.log([50, 200, 800]), np.log(As), 1)[0]),
                "slope_50_200": float(np.log(As[1] / As[0]) / np.log(4)), "slope_200_800": float(np.log(As[2] / As[1]) / np.log(4)),
                "z800_over_floor": (v["A_rho"] - v["floor_halves"]) / v["se_boot"],
                "z50_over_floor": (m2a[k5]["A_rho"] - m2a[k5]["floor_halves"]) / m2a[k5]["se_boot"],
                "perm_p800": v["perm_p"], "perm_p200": m2a[k2]["perm_p"], "perm_p50": m2a[k5]["perm_p"],
                "spec_candidate": bool(v["A_rho"] > 3 * v["se_boot"] and As[2] / As[1] > 0.6),
                "A_PR_800": v["A_PR"]}
res["V3_M2a"] = {"runs": m2a, "decisions_D800": dec, "h0": h0}
print("V3 dec", json.dumps(fl(dec), indent=0)[:4000], flush=True)
print("V3 h0", json.dumps(fl(h0))[:3000], flush=True)

# ------------------------------------------------------------------------------------------------------------------ V4
m2b = {}
for f in sorted(glob.glob(f"{D}/m2b_*.npz")):
    z = np.load(f, allow_pickle=True); mm = meta(z); Bb = mm["B"]
    rho = z["rho"][0]; pr = z["PR_rep"][0]
    tot = z["modc"].sum(1).astype(float); t = np.arange(tot.shape[0]) * float(z["rec_dt"])
    first = tot[(t > 0) & (t <= 10)].mean(0)
    m2b[os.path.basename(f)] = {"g": mm["g"], "r": mm["r"], "h": mm["h"], "ks_rho": float(stats.ks_2samp(rho[:Bb], rho[Bb:]).pvalue),
                                "ks_PR": float(stats.ks_2samp(pr[:Bb], pr[Bb:]).pvalue), "ks_first10": float(stats.ks_2samp(first[:Bb], first[Bb:]).pvalue),
                                "mw_first10": float(stats.mannwhitneyu(first[:Bb], first[Bb:]).pvalue),
                                "n_first10_up": float(first[:Bb].mean()), "n_first10_dn": float(first[Bb:].mean()),
                                "rho_up": float(rho[:Bb].mean()), "rho_dn": float(rho[Bb:].mean()),
                                "late_rho_bimodal_dBIC_pooled": (gmm_dbic(np.log(np.maximum(rho, 1e-12))) or {}).get("dBIC")}
res["V4_M2b"] = {"n": len(m2b), "min_ks_rho": min((v["ks_rho"], k) for k, v in m2b.items()), "min_ks_PR": min((v["ks_PR"], k) for k, v in m2b.items()),
                 "min_ks_first10": min((v["ks_first10"], k) for k, v in m2b.items()), "tasks": m2b}
print("V4", json.dumps(fl({k: v for k, v in res["V4_M2b"].items() if k != "tasks"})), flush=True)
for k, v in m2b.items():
    print("   ", k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}, flush=True)

# ------------------------------------------------------------------------------------------------------------------ V5
m2c = {}
for f in sorted(glob.glob(f"{D}/rsweep_g*.npz")):
    z = np.load(f, allow_pickle=True); mm = meta(z); Bs = mm["B"]; R = np.array(mm["r_grid"]); K = len(R)
    rho = z["rho"].astype(np.float64); Xk = z["Xk"]
    o = {}
    for nm, cols, ia, idn in (("fwd", slice(0, Bs), np.arange(K), np.arange(2 * K - 1, K - 1, -1)),
                              ("rev", slice(Bs, 2 * Bs), np.arange(K, 2 * K), np.arange(K - 1, -1, -1))):
        u, d = rho[ia][:, cols], rho[idn][:, cols]
        A = trap(np.abs(np.log(u.mean(1)) - np.log(d.mean(1))), R)
        bs = []
        for _ in range(1000):
            j = rng.integers(0, Bs, Bs); bs.append(trap(np.abs(np.log(u[:, j].mean(1)) - np.log(d[:, j].mean(1))), R))
        o[nm] = {"A": A, "se": float(np.std(bs))}
    XL, XR = Xk[:, 0, :], Xk[:, 1, :]
    XAL = (XL * 248 + XR * 275) / 523
    with np.errstate(all="ignore"):
        Aa = (XL - XR) / (XL + XR)
    o["XAL_fwd_r0"] = float(XAL[0, :Bs].mean()); o["XAL_fwd_r6"] = float(XAL[K - 1, :Bs].mean())
    o["n_onelobe_dwells"] = int((np.abs(Aa) > 0.6).sum()); o["n_dwells"] = int(Aa.size)
    m2c[f"{mm['g']:g}"] = o
res["V5_M2c"] = m2c
print("V5", json.dumps(fl(m2c)), flush=True)

# ------------------------------------------------------------------------------------------------------------------ V6
m3 = {}
for f in sorted(glob.glob(f"{D}/ic_r*_g*.npz")):
    z = np.load(f, allow_pickle=True); mm = meta(z); r, g, Bs, starts = mm["r"], mm["g"], mm["B"], mm["starts"]
    V = z["Xk"][0].T.astype(np.float64); alive = z["ext_step"] < 0
    lab = np.repeat(np.arange(len(starts)), Bs)
    means = {s: V[(lab == i) & alive].mean(0) for i, s in enumerate(starts) if ((lab == i) & alive).any()}
    rl = lambda a, b: float(np.abs(a - b).sum() / max(np.abs(a).sum(), np.abs(b).sum()))
    ks = list(means)
    mx = max([rl(means[a], means[b]) for i, a in enumerate(ks) for b in ks[i + 1:]], default=0.0)
    if alive.any():
        gm = V[alive].mean(0); dist = np.array([rl(v, gm) for v in V[alive]])
        frac_within = float((dist < 0.2).mean()); maxd = float(dist.max())
    else:
        frac_within = None; maxd = None
    m3[f"r{r:g}_g{g:g}"] = {"alive": {s: float(alive[lab == i].mean()) for i, s in enumerate(starts)}, "max_relL1_means": mx,
                            "n_alive": int(alive.sum()), "frac_surv_within_0.2_of_grand_mean": frac_within, "max_dist_to_grand_mean": maxd,
                            "X_AL_isl_survivors": {s: float(((V[(lab == i) & alive][:, :3] * [248, 275, 3]).sum(1) / 526).mean())
                                                   for i, s in enumerate(starts) if s.startswith("isl") and ((lab == i) & alive).any()}}
isl_le07 = {k: {s: v["alive"][s] for s in v["alive"] if s.startswith("isl")} for k, v in m3.items() if float(k.split("_g")[1]) <= 0.7}
isl_ge12 = min(v["alive"][s] for k, v in m3.items() if float(k.split("_g")[1]) >= 1.2 for s in v["alive"] if s.startswith("isl"))
res["V6_M3"] = {"cells": m3, "max_relL1_means": max((v["max_relL1_means"], k) for k, v in m3.items()),
                "n_cells_all_dead": [k for k, v in m3.items() if v["n_alive"] == 0],
                "min_frac_within": min(v["frac_surv_within_0.2_of_grand_mean"] for v in m3.values() if v["frac_surv_within_0.2_of_grand_mean"] is not None),
                "max_dist_to_grand_mean": max(v["max_dist_to_grand_mean"] for v in m3.values() if v["max_dist_to_grand_mean"] is not None),
                "island_alive_g_le_0.7": isl_le07, "min_island_alive_g_ge_1.2": isl_ge12}
print("V6", json.dumps(fl({k: v for k, v in res["V6_M3"].items() if k != "cells"})), flush=True)

# ------------------------------------------------------------------------------------------------------------------ V7
m3b = {}
allp = []; zmed = []; drift = []; dmed = []
for f in sorted(glob.glob(f"{D}/icq_r*_g*.npz")):
    z = np.load(f, allow_pickle=True); mm = meta(z)
    starts, attr, pairs, Bm, Bf = mm["starts"], mm["attractors"], mm["pairs"], mm["B_mt3"], mm["B_fp"]
    sizes = [Bm] * len(attr) + [Bf] * len(attr); off = np.r_[0, np.cumsum(sizes)]
    grp = {s: np.arange(off[i], off[i + 1]) for i, s in enumerate(starts)}
    alive = z["ext_step"] < 0; reg = z["reg_occ"].astype(np.float64); fpr = z["fp_reg"].astype(np.float64)
    cell = {"alive": {s: float(alive[c].mean()) for s, c in grp.items()}, "pairs": {}}
    for pl in pairs:
        a, b = pl.split("->"); da = fpr[attr.index(b)] - fpr[attr.index(a)]
        zl = (reg[1].T - fpr[attr.index(a)]) @ da / (da @ da); ze = (reg[0].T - fpr[attr.index(a)]) @ da / (da @ da)
        for kind in ("mt3", "fp"):
            ca, cb = grp[f"{kind}:{a}"], grp[f"{kind}:{b}"]
            za, zb = zl[ca][alive[ca]], zl[cb][alive[cb]]
            if len(za) and len(zb):
                dmed.append((abs(np.median(za) - np.median(zb)), mm["r"], mm["g"], pl, kind))
            for zz in (za, zb):
                if len(zz):
                    zmed.append(float(np.median(zz)))
            for cc in (ca, cb):
                if alive[cc].any():
                    drift.append(abs(float(np.median(zl[cc][alive[cc]]) - np.median(ze[cc][alive[cc]]))))
            if len(za) >= 2 and len(zb) >= 2:
                p = float(stats.ks_2samp(za, zb).pvalue)
                allp.append((p, mm["r"], round(mm["g"], 4), pl, kind, len(za), len(zb), float(np.median(za)), float(np.median(zb))))
    cell["all_dead"] = bool(not alive.any())
    m3b[os.path.basename(f)] = cell
res["V7_M3b"] = {"n_cells": len(m3b), "n_all_dead": sum(v["all_dead"] for v in m3b.values()),
                 "all_dead_cells": [k for k, v in m3b.items() if v["all_dead"]], "n_ks": len(allp), "min_ks": min(allp),
                 "max_abs_median_z_diff": max(dmed), "z_median_range": [min(zmed), max(zmed)], "max_early_late_drift": max(drift),
                 "r6_g0.58_alive": [v["alive"] for k, v in m3b.items() if k.startswith("icq_r6_g0.58")]}
print("V7", json.dumps(fl(res["V7_M3b"])), flush=True)

# ------------------------------------------------------------------------------------------------------- follow-up checks
# F1: M1 closure-window points with no survivors: when did the last replica die?
f1 = {}
for r, g in ((4, 0.29), (4, 0.32), (4, 0.40), (6, 0.43)):
    z = np.load(f"{D}/al_sym_r{r}.npz", allow_pickle=True); Gs = meta(z)["g"]; e = z["ext_step"][Gs.index(g)]
    t = np.where(e >= 0, e * DT, np.inf)
    f1[f"r{r}_g{g}"] = {"n_died_by_T1000": int((e >= 0).sum()), "last_ext_t": float(t[np.isfinite(t)].max()),
                        "n_alive_past_t500": int((t >= 500).sum())}
res["F1_M1_window_extinction"] = f1
print("F1", f1, flush=True)

# F2: h = 0 down-sweep vs the tau(g) prediction: exact (Monte-Carlo multinomial) G-test on the discrete grid, and the systematic
#     shift of mean ln g_ext (observed - predicted) with its one-sided MC p, Stouffer combination and sign test over the 6 r
f2 = {}; zs = []; signs = []
for r in (0, 0.5, 1, 2, 4, 6):
    z = np.load(f"{D}/sweep_r{r:g}_h0_D200_down.npz", allow_pickle=True); mm = meta(z); Gg = np.array(mm["g_grid"]); K = len(Gg)
    st = int(round(mm["D"] / DT)); e = z["ext_step"]
    i = int(np.argmin(np.abs(hm["r"] - r))); lt = hm["ln_tau_iso"][i]
    tau = np.exp(np.interp(np.log(Gg[::-1]), np.log(hm["g"]), lt)); S = np.exp(-np.cumsum(mm["D"] / tau))
    p = np.r_[np.r_[1, S[:-1]] - S, S[-1]]                       # die in dwell j (descending g) ... or survive all
    obs = np.bincount(np.where(e >= 0, (e - 1) // st, K), minlength=K + 1); n = int(obs.sum()); ex = n * p

    def G2(o):
        k = o > 0
        return 2 * float(np.sum(o[k] * np.log(o[k] / ex[k])))
    sims = rng.multinomial(n, p / p.sum(), size=20000)
    g2s = np.array([G2(s) for s in sims]); g2o = G2(obs)
    lg = np.r_[np.log(Gg[::-1]), np.nan]
    mo = float(np.sum(obs[:K] * lg[:K]) / n); ms = sims[:, :K] @ lg[:K] / n
    p_one = float(((ms <= mo).sum() + 1) / (len(ms) + 1))        # one-sided: observed extinction at LOWER g than predicted
    zs.append(stats.norm.isf(p_one)); signs.append(mo < ms.mean())
    f2[f"{r:g}"] = {"G2": g2o, "G2_mc_p": float(((g2s >= g2o).sum() + 1) / (len(g2s) + 1)), "mean_ln_gext_obs": mo,
                    "mean_ln_gext_pred": float(ms.mean()), "shift_ln": mo - float(ms.mean()), "shift_in_grid_steps": (mo - float(ms.mean())) / float(np.log(Gg[1] / Gg[0])),
                    "p_one_sided_lower": p_one}
f2["stouffer_z"] = float(np.sum(zs) / np.sqrt(len(zs))); f2["stouffer_p_one_sided"] = float(stats.norm.sf(f2["stouffer_z"]))
f2["n_obs_below_pred"] = int(np.sum(signs)); f2["sign_test_p_two_sided"] = float(stats.binomtest(int(np.sum(signs)), len(signs), 0.5).pvalue)
res["F2_h0_extinction_vs_tau"] = f2
print("F2", json.dumps(fl(f2)), flush=True)

# F3: where are the M2b loop centres relative to the simulated S (tau = 200) and on (tau = 3000) lines?
pcr = json.load(open(f"{ROOT}/results/analysis2/phase_complete.json"))["sim"]["real"]
f3 = {}
for k, v in m2b.items():
    key = f"{v['r']:g}"
    if key in pcr:
        S_, on_ = pcr[key]["S"]["g"], pcr[key]["on"]["g"]
    else:                                                          # r = 5.75: linear interpolation between r = 5.5 and 6
        a, b = pcr["5.5"], pcr["6"]; w_ = (v["r"] - 5.5) / 0.5
        S_ = (1 - w_) * a["S"]["g"] + w_ * b["S"]["g"]; on_ = (1 - w_) * a["on"]["g"] + w_ * b["on"]["g"]
    pos = "below S (quiescent / drive-dominated)" if v["g"] < S_ else ("between S and on (onset)" if v["g"] <= on_ * 1.25 else "above 1.25 x on (active)")
    f3[k] = {"g": v["g"], "r": v["r"], "h": v["h"], "S": S_, "on": on_, "g_over_on": v["g"] / on_, "position": pos,
             "mean_active_first10": [v["n_first10_up"], v["n_first10_dn"]]}
res["F3_M2b_centre_positions"] = f3
res["F3_counts"] = {p_: sum(1 for x in f3.values() if x["position"] == p_) for p_ in set(x["position"] for x in f3.values())}
print("F3", json.dumps(fl(res["F3_counts"])), flush=True)

# F4: total in-script GPU time over all 132 outputs; M3b region sizes
secs = 0.0; nf = 0; nreg = []
for f in sorted(glob.glob(f"{D}/*.npz")):
    mm = meta(np.load(f, allow_pickle=True)); nf += 1
    secs += float(mm.get("seconds", 0.0))
    if "n_region" in mm:
        nreg.append(mm["n_region"])
res["F4_gpu"] = {"n_npz": nf, "sum_seconds_h": secs / 3600, "M3b_n_region_range": [min(nreg), max(nreg)]}
print("F4", res["F4_gpu"], flush=True)

# F5: M2b late window (t > 2500): share of records with a completely silent AL per replica, and mean active count per replica
f5 = {}
for f in sorted(glob.glob(f"{D}/m2b_*.npz")):
    z = np.load(f, allow_pickle=True); Bb = meta(z)["B"]; mc = z["modc"]; t = np.arange(mc.shape[0]) * float(z["rec_dt"])
    late = t > 2500; al = mc[late][:, :3, :].sum(1); fs = (al == 0).mean(0)
    f5[os.path.basename(f)] = {"AL_silent_frac_range": [float(fs.min()), float(fs.max())], "AL_silent_frac_median": float(np.median(fs)),
                               "ks_up_down_AL_silent_frac": float(stats.ks_2samp(fs[:Bb], fs[Bb:]).pvalue),
                               "late_mean_active_range": [float(mc[late].sum(1).mean(0).min()), float(mc[late].sum(1).mean(0).max())]}
res["F5_M2b_late_AL_silence"] = f5
for k, v in f5.items():
    print("F5", k, v, flush=True)

json.dump(fl(res), open(OUT, "w"))
print("wrote", OUT)
