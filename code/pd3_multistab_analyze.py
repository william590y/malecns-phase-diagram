"""Analysis of the pd3 multistability program (plan.md section 5) -> results/pd3/multistab/summary.json (+ per-part json).

Usage: python pd3_multistab_analyze.py [m1] [m2a] [m2c] [m3] [m3b] [m2b]   (default: all parts whose files exist)

M1 (al_sym_r*.npz; AL-L / AL-R / AL / all starts, B = 64 each, T = 1000, h = 0):
  f_L = n_L / 248, f_R = n_R / 275, A = (f_L - f_R) / (f_L + f_R) (undefined when both lobes are silent).
  t_sym: first time |A| < 0.3 (every-step record for t <= 100, every model time after); 'AL died first' if the AL went extinct
  before; censored at 1000. Late window t in [500, 1000] (every model time): per-replica mean of A over the samples with an active
  AL; survivor = network alive at T and AL active in >= 50% of the window samples.
  Broken symmetry at (g, r) (pre-registered): (c1) >= 25% of surviving AL-L-start replicas have late mean A > 0.6, (c2) AL-L vs
  AL-R start late distributions differ (two-sample KS p < 0.001), (c3) the all-active start's late A is bimodal: a 2-component
  Gaussian mixture beats 1 component by BIC > 10, both weights >= 0.1 and |mu1 - mu2| > 2 max(sd1, sd2) (Sarle's coefficient also
  reported). All three are required.
M2a (sweep_r*_h*_D*.npz): branch observables per g over the last half of each dwell; loop areas
  A_rho = int |ln rho_up - ln rho_down| d ln g, A_PR = int |PR_up - PR_down| d ln g (PR = PR of the replica-mean occupancy / N,
  the phase_complete definition), trapezoid on the 40 ln-g points; replica bootstrap with the run's stored resample weights (up and
  down groups resampled independently); NOISE FLOOR (added): the same area between two random halves of one branch, scaled by
  1/sqrt(2) to 32-vs-32 replicas, averaged over 100 splits and both branches (what the area is when up and down do not differ).
  Hysteresis candidate (pre-registered): A(D = 800) > 3 SE and A(800) / A(200) > 0.6 (shrinks by < 40%). Reported with and without
  the noise floor (A - floor > 3 SE).
  h = 0 down-sweeps: per-replica extinction g; compared with the tau = D (= 200) crossing of the lifetime map (redraw heat map
  results/pd3/redraw/heatmap_real.npz ln_tau_iso, row r) and with the extinction distribution predicted from that tau(g) curve
  (survival through dwell j = exp(-D / tau(g_j)), every dwell independent).
M2c (rsweep_g*.npz): the same loop measures in r (d r instead of d ln g) for the fwd (0 -> 6 -> 0) and rev (6 -> 0 -> 6) groups; up
  and down branches are the same replicas, resampled jointly.
M3 (ic_r*_g*.npz; 13 starts x 32, T = 2000, h = 0): per-replica late (t in [1000, 2000]) 34-module activity vector X_k (fraction
  active); relative L1 = |a - b|_1 / max(|a|_1, |b|_1) (MT3's definition, pd3_qmf_analyze.py). Survivors = alive at T.
  Start-level: start means, spreads (median rel L1 of a start's survivors to their mean); a pair of starts is a distinct-attractor
  pair if rel L1(means) > 0.2 and both spreads < 0.05 (spec). Replica-level: average-linkage clustering of all survivors with
  cut 0.2; multistability (pre-registered) = >= 2 non-silent clusters each holding >= 50% of the replicas (of all B) of some start.
M3b (icq_r*_g*.npz; QMF multistable cells): per pair of QMF attractors (a, b), projection z = (x - rho_a) . d / |d|^2 of the
  late per-replica region occupancy (t in [3000, 5000]); z = 0 at the QMF attractor a, 1 at b. Coexistence in the stochastic model
  = survivors started at a (MT3 start or Bernoulli(rho_a)) and at b have late z distributions that differ (KS p < 0.001) AND
  medians that differ by > 0.5; otherwise the noise merges them (or kills them).
M2b (m2b_*.npz): up-end-state vs down-end-state replicas at a loop centre, T = 5000; per-replica late rho and PR (t in [2500, 5000]);
  coexistence = KS p < 0.001 on rho or PR.
"""
import glob as _glob, json, os, re, sys
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = "/MALECNS_ROOT"
D_ = f"{ROOT}/results/pd3/multistab"
DT = 0.2
SIZES = {"AL-L": 248, "AL-R": 275, "AL-mid": 3}
GON = {0: 0.266, 0.5: 0.306, 1: 0.344, 2: 0.402, 3: 0.456, 4: 0.508, 6: 0.579}      # flow_sim_boundaries.json (spec values)
CLOSURE_B1 = {2.5: 0.2236, 3: 0.2471, 4: 0.2871, 6: 0.4280}                         # theory_module2.json coarse_2module: B1L+B1R from
CLOSURE_SH = {2.5: 0.2351, 3: 0.2598, 4: 0.3172, 6: 0.5775}                         # ... and B1 + SH coexistence from (grid end 3.0)


class glob:                                                  # never pick up in-progress atomic-write temporaries
    @staticmethod
    def glob(p):
        return [f for f in _glob.glob(p) if ".tmp" not in os.path.basename(f)]


def jl(x):
    """json-safe"""
    if isinstance(x, dict):
        return {str(k): jl(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jl(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return jl(x.tolist())
    return x


def meta_of(z):
    return json.loads(str(z["meta"]))


# --------------------------------------------------------------------------------------------------------------- M1
def bimodal(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    res = {"n": int(len(x))}
    if len(x) < 10 or np.std(x) < 1e-9:
        res.update(bimodal=False, reason="n < 10 or no spread"); return res
    n = len(x); s = stats.skew(x); k = stats.kurtosis(x)
    res["sarle_bc"] = float((s ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))
    from sklearn.mixture import GaussianMixture
    X = x[:, None]
    g1 = GaussianMixture(1, random_state=0).fit(X); g2 = GaussianMixture(2, n_init=5, random_state=0).fit(X)
    dbic = g1.bic(X) - g2.bic(X)
    mu = g2.means_.ravel(); sd = np.sqrt(g2.covariances_.ravel()); wt = g2.weights_
    sep = abs(mu[0] - mu[1]) > 2 * max(sd)
    res.update(dBIC=float(dbic), mu=mu.tolist(), sd=sd.tolist(), w=wt.tolist(),
               bimodal=bool(dbic > 10 and wt.min() >= 0.1 and sep))
    return res


def analyze_m1():
    out = {}
    for f in sorted(glob.glob(f"{D_}/al_sym_r*.npz")):
        z = np.load(f, allow_pickle=True); m = meta_of(z)
        r = m["r"]; G = list(m["g"]); B = m["B"]; starts = m["starts"]
        modc = z["modc"]; fine = z["fine"]; ext = z["ext_step"]; ext_al = z["ext_al_step"]
        rows = []
        for gi, g in enumerate(G):
            nL = modc[gi, :, 0, :].astype(float); nR = modc[gi, :, 1, :].astype(float)
            fL, fR = nL / SIZES["AL-L"], nR / SIZES["AL-R"]; den = fL + fR
            with np.errstate(invalid="ignore", divide="ignore"):
                A = np.where(den > 0, (fL - fR) / den, np.nan)                       # [1001, 256]
                fl2 = fine[gi, :, 0, :] / SIZES["AL-L"]; fr2 = fine[gi, :, 1, :] / SIZES["AL-R"]
                Af = np.where(fl2 + fr2 > 0, (fl2 - fr2) / (fl2 + fr2), np.nan)       # [501, 256]
            tc = np.arange(A.shape[0]) * float(z["rec_dt"]); tf = np.arange(Af.shape[0]) * DT
            row = {"g": g, "x_gon": g / GON[r], "starts": {}}
            late = (tc >= 500) & (tc <= 1000)
            lateA = {}
            for si, s in enumerate(starts):
                cols = np.arange(si * B, (si + 1) * B)
                a_late = A[late][:, cols]; valid = np.isfinite(a_late)
                surv = (ext[gi, cols] < 0) & (valid.mean(0) >= 0.5)
                mA = np.where(valid.sum(0) > 0, np.nansum(a_late, 0) / np.maximum(valid.sum(0), 1), np.nan)
                lateA[s] = mA[surv]
                st = {"n": B, "alive_T": float((ext[gi, cols] < 0).mean()), "AL_alive_T": float((ext_al[gi, cols] < 0).mean()),
                      "n_surv": int(surv.sum()), "late_A_mean": float(np.mean(mA[surv])) if surv.any() else None,
                      "late_A_sd": float(np.std(mA[surv])) if surv.sum() > 1 else None,
                      "late_A_median": float(np.median(mA[surv])) if surv.any() else None,
                      "late_fL": float(np.nanmean(fL[late][:, cols][:, surv])) if surv.any() else None,
                      "late_fR": float(np.nanmean(fR[late][:, cols][:, surv])) if surv.any() else None}
                if s in ("AL-L", "AL-R"):
                    tsym = np.full(B, np.nan); status = []
                    for j, c in enumerate(cols):
                        hit = np.where(np.abs(Af[:, c]) < 0.3)[0]
                        if len(hit):
                            tsym[j] = tf[hit[0]]; status.append("sym"); continue
                        hit = np.where((np.abs(A[:, c]) < 0.3) & (tc > tf[-1]))[0]
                        e_al = ext_al[gi, c] * DT if ext_al[gi, c] >= 0 else np.inf
                        if len(hit) and tc[hit[0]] < e_al:
                            tsym[j] = tc[hit[0]]; status.append("sym")
                        elif np.isfinite(e_al):
                            status.append("AL_died_first")
                        else:
                            status.append("censored")
                    status = np.array(status); ok = status == "sym"
                    st["t_sym"] = {"n_sym": int(ok.sum()), "n_AL_died_first": int((status == "AL_died_first").sum()),
                                   "n_censored": int((status == "censored").sum()),
                                   "median": float(np.median(tsym[ok])) if ok.any() else None,
                                   "q25": float(np.percentile(tsym[ok], 25)) if ok.any() else None,
                                   "q75": float(np.percentile(tsym[ok], 75)) if ok.any() else None,
                                   "max": float(np.max(tsym[ok])) if ok.any() else None,
                                   "frac_sym_by_10": float((ok & (tsym <= 10)).mean()), "frac_sym_by_50": float((ok & (tsym <= 50)).mean()),
                                   "values": tsym.tolist()}
                    keep = (mA[surv] > 0.6) if s == "AL-L" else (mA[surv] < -0.6)
                    st["frac_surv_keeping_start_lobe"] = float(keep.mean()) if surv.any() else None
                row["starts"][s] = st
            # median A(t) traces for the figure (L / R starts), log-spaced times
            tt = np.unique(np.r_[tf[1:], tc[tc > tf[-1]]])
            row["trace_t"] = tt.tolist()
            for s in ("AL-L", "AL-R"):
                si = starts.index(s); cols = np.arange(si * B, (si + 1) * B)
                Aall = np.vstack([Af[1:, cols], A[tc > tf[-1]][:, cols]])
                row[f"trace_{s}_n_AL_active"] = np.isfinite(Aall).sum(1).tolist()
                with np.errstate(all="ignore"):
                    row[f"trace_{s}_median"] = np.nanmedian(Aall, 1).tolist()
                    row[f"trace_{s}_q25"] = np.nanpercentile(Aall, 25, 1).tolist()
                    row[f"trace_{s}_q75"] = np.nanpercentile(Aall, 75, 1).tolist()
            L, R_ = lateA["AL-L"], lateA["AL-R"]
            ks = stats.ks_2samp(L, R_) if len(L) >= 2 and len(R_) >= 2 else None
            row["ks_L_vs_R"] = {"stat": float(ks.statistic), "p": float(ks.pvalue)} if ks else None
            row["ks_pairs"] = {}
            for i1 in range(4):
                for i2 in range(i1 + 1, 4):
                    a1, a2 = lateA[starts[i1]], lateA[starts[i2]]
                    if len(a1) >= 2 and len(a2) >= 2:
                        k2 = stats.ks_2samp(a1, a2); row["ks_pairs"][f"{starts[i1]}|{starts[i2]}"] = [float(k2.statistic), float(k2.pvalue)]
            row["bimodality_all"] = bimodal(lateA["all"])
            row["late_A_values"] = {s: lateA[s].tolist() for s in starts}
            fl = row["starts"]["AL-L"]["frac_surv_keeping_start_lobe"]
            c1 = fl is not None and fl >= 0.25
            c2 = ks is not None and ks.pvalue < 1e-3
            c3 = row["bimodality_all"]["bimodal"]
            row["criteria"] = {"c1_Lonly_keep_ge_25pct": bool(c1), "c2_KS_L_vs_R_p_lt_1e-3": bool(c2), "c3_all_bimodal": bool(c3)}
            row["broken_symmetry"] = bool(c1 and c2 and c3)
            row["closure_predicts_B1"] = bool(r in CLOSURE_B1 and CLOSURE_B1[r] <= g <= 3.0)       # closure grid ends at g = 3.0
            rows.append(row)
            print(f"M1 r={r:g} g={g:.4g} (x{g / GON[r]:.2f}) surv L/R/AL/all "
                  + "/".join(str(row['starts'][s]['n_surv']) for s in starts)
                  + f" tsym med L {row['starts']['AL-L']['t_sym']['median']} R {row['starts']['AL-R']['t_sym']['median']}"
                  + f" lateA L {row['starts']['AL-L']['late_A_mean']} R {row['starts']['AL-R']['late_A_mean']} all {row['starts']['all']['late_A_mean']}"
                  + f" keepL {fl} KS p {row['ks_L_vs_R']['p'] if ks else None} bimodal {c3} -> broken {row['broken_symmetry']}", flush=True)
        out[f"{r:g}"] = {"file": f.replace(ROOT + "/", ""), "gpu": m.get("gpu"), "seconds": m.get("seconds"), "rows": rows}
    summ = {"any_broken_symmetry": any(rw["broken_symmetry"] for v in out.values() for rw in v["rows"]),
            "n_points": sum(len(v["rows"]) for v in out.values()),
            "max_frac_Lonly_keep": max((rw["starts"]["AL-L"]["frac_surv_keeping_start_lobe"] or 0) for v in out.values() for rw in v["rows"]),
            "max_t_sym_median": max((rw["starts"][s]["t_sym"]["median"] or 0) for v in out.values() for rw in v["rows"] for s in ("AL-L", "AL-R")),
            "n_c1": sum(rw["criteria"]["c1_Lonly_keep_ge_25pct"] for v in out.values() for rw in v["rows"]),
            "n_c2": sum(rw["criteria"]["c2_KS_L_vs_R_p_lt_1e-3"] for v in out.values() for rw in v["rows"]),
            "n_c3": sum(rw["criteria"]["c3_all_bimodal"] for v in out.values() for rw in v["rows"])}
    return {"per_r": out, "summary": summ}


# --------------------------------------------------------------------------------------------------------------- M2
def trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def loop_areas(rho_up, rho_dn, pr_up_b, pr_dn_b, pr_up, pr_dn, x, wb_up, wb_dn, rng, joint=False, n_split=100):
    """rho_*: [K, Bg] per replica on the common x grid (branch-aligned); pr_*_b: [K, nb] bootstrap PR of the replica mean;
    wb_*: [Bg, nb] multinomial counts; joint: same replicas on both branches (resample jointly with wb_up)."""
    eps = 1e-12
    ru, rd = rho_up.mean(1), rho_dn.mean(1)
    d_rho = np.abs(np.log(np.maximum(ru, eps)) - np.log(np.maximum(rd, eps)))
    A_rho = trapz(d_rho, x); A_pr = trapz(np.abs(pr_up - pr_dn), x)
    Bg = rho_up.shape[1]
    wu = wb_up / Bg; wd = (wb_up if joint else wb_dn) / Bg
    bu = rho_up @ wu; bd = rho_dn @ wd                                     # [K, nb]
    Ab = np.array([trapz(np.abs(np.log(np.maximum(bu[:, i], eps)) - np.log(np.maximum(bd[:, i], eps))), x) for i in range(bu.shape[1])])
    Abp = np.array([trapz(np.abs(pr_up_b[:, i] - pr_dn_b[:, i]), x) for i in range(pr_up_b.shape[1])]) if pr_up_b is not None else None
    # noise floor: area between random halves of one branch, scaled to the full-branch comparison
    fl = []
    for arr in (rho_up, rho_dn):
        for _ in range(n_split // 2):
            p = rng.permutation(Bg); h1, h2 = arr[:, p[:Bg // 2]].mean(1), arr[:, p[Bg // 2:]].mean(1)
            fl.append(trapz(np.abs(np.log(np.maximum(h1, eps)) - np.log(np.maximum(h2, eps))), x) / np.sqrt(2))
    # PR noise floor (added): E|X| of X ~ N(0, sd_up^2 + sd_down^2) per g from the bootstrap SDs of the replica-mean PR, integrated
    fl_pr = trapz(np.sqrt(2 / np.pi) * np.sqrt(pr_up_b.std(1) ** 2 + pr_dn_b.std(1) ** 2), x) if pr_up_b is not None else None
    return {"A_rho": A_rho, "A_rho_se": float(Ab.std()), "A_rho_boot_ci": [float(np.percentile(Ab, 2.5)), float(np.percentile(Ab, 97.5))],
            "A_rho_noise_floor": float(np.mean(fl)), "A_PR": A_pr, "A_PR_se": float(Abp.std()) if Abp is not None else None,
            "A_PR_noise_floor": fl_pr, "d_ln_rho": d_rho.tolist(), "d_PR": np.abs(pr_up - pr_dn).tolist()}


def tau_curve(r):
    """ln tau_sim(g) of the redraw heat map (isotonic, interpolated in ln g; results/pd3/redraw/heatmap_real.npz)."""
    h = np.load(f"{ROOT}/results/pd3/redraw/heatmap_real.npz")
    i = int(np.argmin(np.abs(h["r"] - r))); assert abs(h["r"][i] - r) < 1e-9
    return h["g"], h["ln_tau_iso"][i]


def analyze_m2a():
    out = {"runs": {}, "h0": {}}
    rng = np.random.default_rng(5)
    for f in sorted(glob.glob(f"{D_}/sweep_r*_h*_D*.npz")):
        z = np.load(f, allow_pickle=True); m = meta_of(z)
        r, h, D = m["r"], m["h"], m["D"]; G = np.array(m["g_grid"]); K = len(G); B = m["B"]
        rho = z["rho"]; Xk = z["Xk"]; PRm = z["PR_mean"]; PRb = z["PR_boot"]; wb = z["wb"]; PRr = z["PR_rep"]
        if m.get("down_only"):
            # h = 0 control: extinction g per replica
            steps = int(round(D / DT)); e = z["ext_step"]
            seg = np.where(e >= 0, (e - 1) // steps, -1)
            gext = np.where(seg >= 0, G[K - 1 - np.clip(seg, 0, K - 1)], np.nan)
            gg, lt = tau_curve(r)
            ltau = np.interp(np.log(G[::-1]), np.log(gg), lt)               # dwell order (descending g)
            haz = D / np.exp(ltau)
            S_ = np.exp(-np.cumsum(haz))                                      # survival after dwell j
            pdie = np.r_[1.0, S_[:-1]] - S_
            cdf = np.cumsum(pdie)
            pred_med = float(G[::-1][np.searchsorted(cdf, 0.5)]) if cdf[-1] >= 0.5 else None
            # tau = D crossing of the heat-map row (first upward crossing in g)
            above = np.where(lt >= np.log(D))[0]
            g_cross = None
            if len(above) and above[0] > 0:
                k = above[0]; fr = (np.log(D) - lt[k - 1]) / (lt[k] - lt[k - 1])
                g_cross = float(np.exp(np.log(gg[k - 1]) + fr * (np.log(gg[k]) - np.log(gg[k - 1]))))
            died = np.isfinite(gext)
            # predicted vs observed extinction-segment distributions (chi-square-free comparison: KS on samples from the prediction)
            samp = rng.choice(G[::-1], size=20000, p=pdie / pdie.sum()) if pdie.sum() > 0 else np.array([])
            ksp = float(stats.ks_2samp(gext[died], samp).pvalue) if died.sum() >= 2 and len(samp) else None
            out["h0"][f"{r:g}"] = {"file": f.replace(ROOT + "/", ""), "B": B, "n_died": int(died.sum()),
                                   "g_ext_median": float(np.median(gext[died])) if died.any() else None,
                                   "g_ext_q25_q75": [float(np.percentile(gext[died], 25)), float(np.percentile(gext[died], 75))] if died.any() else None,
                                   "g_ext_values": gext.tolist(), "g_tau_eq_D_heatmap": g_cross, "pred_median_from_tau_curve": pred_med,
                                   "pred_prob_die_by_end": float(pdie.sum()), "ks_p_obs_vs_pred": ksp,
                                   "rho_down": rho.mean(1)[::-1].tolist(), "g": G.tolist()}
            print(f"M2a h=0 r={r:g}: died {died.sum()}/{B}, g_ext median {out['h0'][f'{r:g}']['g_ext_median']} vs tau=D crossing {g_cross} "
                  f"(pred median {pred_med}, KS p {ksp})", flush=True)
            continue
        up = slice(0, B); dn = slice(B, 2 * B)
        # align: up branch at G[k] in segment k; down branch at G[k] in segment K-1-k
        rho_up = rho[:, up]; rho_dn = rho[::-1, dn]
        pr_up, pr_dn = PRm[:, 0], PRm[::-1, 1]; pb_up, pb_dn = PRb[:, 0, :], PRb[::-1, 1, :]
        nal = lambda X: X[:, 0, :] * 248 + X[:, 1, :] * 275 + X[:, 2, :] * 3
        nal_up = nal(Xk[:, :, up]); nal_dn = nal(Xk[::-1, :, dn])
        xg = np.log(G)
        la = loop_areas(rho_up, rho_dn, pb_up, pb_dn, pr_up, pr_dn, xg, wb[0][:B], wb[1][:B], rng)
        # windows: onset (g <= 2.5 g_on) and above
        won = G <= 2.5 * GON[r]
        la_on = loop_areas(rho_up[won], rho_dn[won], pb_up[won], pb_dn[won], pr_up[won], pr_dn[won], xg[won], wb[0][:B], wb[1][:B], rng)
        la_hi = loop_areas(rho_up[~won], rho_dn[~won], pb_up[~won], pb_dn[~won], pr_up[~won], pr_dn[~won], xg[~won], wb[0][:B], wb[1][:B], rng)
        # per-g significance of the up/down difference (Mann-Whitney on per-replica ln rho)
        pg = []
        for k in range(K):
            a1, a2 = np.log(np.maximum(rho_up[k], 1e-12)), np.log(np.maximum(rho_dn[k], 1e-12))
            pg.append(float(stats.mannwhitneyu(a1, a2).pvalue) if np.std(np.r_[a1, a2]) > 0 else 1.0)
        key = f"r{r:g}_h{h:g}_D{D:g}"
        out["runs"][key] = {"file": f.replace(ROOT + "/", ""), "r": r, "h": h, "D": D, "B": B, "gpu": m.get("gpu"), "seconds": m.get("seconds"),
                            "g": G.tolist(), "rho_up": rho_up.mean(1).tolist(), "rho_down": rho_dn.mean(1).tolist(),
                            "rho_up_q": np.percentile(rho_up, [25, 75], 1).tolist(), "rho_down_q": np.percentile(rho_dn, [25, 75], 1).tolist(),
                            "PR_up": pr_up.tolist(), "PR_down": pr_dn.tolist(), "PRrep_up": PRr[:, up].mean(1).tolist(),
                            "PRrep_down": PRr[::-1, dn].mean(1).tolist(), "nAL_up": nal_up.mean(1).tolist(), "nAL_down": nal_dn.mean(1).tolist(),
                            "area_all": la, "area_onset_window": {k: v for k, v in la_on.items() if k != "d_ln_rho"},
                            "area_above_window": {k: v for k, v in la_hi.items() if k != "d_ln_rho"}, "onset_window_g_max": 2.5 * GON[r],
                            "p_updown_per_g": pg}
        print(f"M2a {key}: A_rho {la['A_rho']:.3f} +- {la['A_rho_se']:.3f} (floor {la['A_rho_noise_floor']:.3f}); onset-window "
              f"{la_on['A_rho']:.3f} +- {la_on['A_rho_se']:.3f} (floor {la_on['A_rho_noise_floor']:.3f}); A_PR {la['A_PR']:.4f} +- {la['A_PR_se']:.4f}; "
              f"max |dln rho| {max(la['d_ln_rho']):.2f} at g {G[int(np.argmax(la['d_ln_rho']))]:.3g}", flush=True)
    # hysteresis-candidate decision per (r, h) and window
    dec = {}
    for key, v in out["runs"].items():
        r, h, D = v["r"], v["h"], v["D"]
        if D != 800:
            continue
        k200 = f"r{r:g}_h{h:g}_D200"; k50 = f"r{r:g}_h{h:g}_D50"
        if k200 not in out["runs"]:
            continue
        for win in ("area_all", "area_onset_window", "area_above_window"):
            a8 = v[win]; a2 = out["runs"][k200][win]; a5 = out["runs"].get(k50, {}).get(win)
            ratio = a8["A_rho"] / a2["A_rho"] if a2["A_rho"] > 0 else None
            spec = bool(a8["A_rho"] > 3 * a8["A_rho_se"] and ratio is not None and ratio > 0.6)
            corr = bool(a8["A_rho"] - a8["A_rho_noise_floor"] > 3 * a8["A_rho_se"] and ratio is not None and ratio > 0.6)
            Ds = [50, 200, 800]; As = [a5["A_rho"] if a5 else np.nan, a2["A_rho"], a8["A_rho"]]
            ok = np.isfinite(As) & (np.array(As) > 0)
            slope = float(np.polyfit(np.log(np.array(Ds)[ok]), np.log(np.array(As)[ok]), 1)[0]) if ok.sum() >= 2 else None
            # loop centre: the g (inside the window) with the largest |ln rho_up - ln rho_down| at D = 800
            Gv = np.array(v["g"]); dl = np.array(v["area_all"]["d_ln_rho"])
            wmask = np.ones(len(Gv), bool) if win == "area_all" else (Gv <= v["onset_window_g_max"]) if win == "area_onset_window" else (Gv > v["onset_window_g_max"])
            kc = int(np.where(wmask)[0][np.argmax(dl[wmask])])
            dp = np.array(v["area_all"]["d_PR"]); kp = int(np.where(wmask)[0][np.argmax(dp[wmask])])
            rp = a8["A_PR"] / a2["A_PR"] if a2["A_PR"] > 0 else None
            spec_pr = bool(a8["A_PR"] > 3 * a8["A_PR_se"] and rp is not None and rp > 0.6)
            corr_pr = bool(a8["A_PR"] - a8["A_PR_noise_floor"] > 3 * a8["A_PR_se"] and rp is not None and rp > 0.6)
            dec[f"r{r:g}_h{h:g}_{win}"] = {"loop_centre_g": float(Gv[kc]), "loop_centre_seg_up": kc, "loop_centre_dlnrho": float(dl[kc]),
                                           "PR_A800": a8["A_PR"], "PR_SE800": a8["A_PR_se"], "PR_floor800": a8["A_PR_noise_floor"],
                                           "PR_A200": a2["A_PR"], "PR_A50": a5["A_PR"] if a5 else None, "PR_ratio_800_200": rp,
                                           "PR_candidate_spec": spec_pr, "PR_candidate_noise_corrected": corr_pr,
                                           "PR_loop_centre_g": float(Gv[kp]), "PR_loop_centre_seg_up": kp, "PR_loop_centre_dPR": float(dp[kp]),
                                           "loop_centre_p_updown": v["p_updown_per_g"][kc], "file800": v["file"],
                                           "A800": a8["A_rho"], "SE800": a8["A_rho_se"], "floor800": a8["A_rho_noise_floor"],
                                           "A200": a2["A_rho"], "A50": a5["A_rho"] if a5 else None, "ratio_800_200": ratio,
                                           "loglog_slope_A_vs_D": slope, "candidate_spec": spec, "candidate_noise_corrected": corr,
                                           "A800_PR": a8["A_PR"], "SE800_PR": a8["A_PR_se"]}
    out["decisions"] = dec
    out["any_candidate_spec"] = any(v["candidate_spec"] for v in dec.values())
    out["any_candidate_noise_corrected"] = any(v["candidate_noise_corrected"] for v in dec.values())
    out["any_PR_candidate_spec"] = any(v["PR_candidate_spec"] for v in dec.values())
    out["any_PR_candidate_noise_corrected"] = any(v["PR_candidate_noise_corrected"] for v in dec.values())
    return out


def analyze_m2c():
    out = {}
    rng = np.random.default_rng(6)
    for f in sorted(glob.glob(f"{D_}/rsweep_g*.npz")):
        z = np.load(f, allow_pickle=True); m = meta_of(z)
        g = m["g"]; B = m["B"]; R = np.array(m["r_grid"]); K = len(R)
        rho = z["rho"]; PRm = z["PR_mean"]; PRb = z["PR_boot"]; wb = z["wb"]; Xk = z["Xk"]
        res = {"file": f.replace(ROOT + "/", ""), "g": g, "r": R.tolist(), "gpu": m.get("gpu")}
        for gi, (gname, cols) in enumerate([("fwd", slice(0, B)), ("rev", slice(B, 2 * B))]):
            if gname == "fwd":      # segments 0..K-1: r ascending (R[k]); K..2K-1: r descending (R[K-1-(k-K)])
                ia, idn = np.arange(K), np.arange(2 * K - 1, K - 1, -1)
            else:                   # segments 0..K-1: r descending (R[K-1-k]); K..2K-1: r ascending (R[k-K])
                ia, idn = np.arange(K, 2 * K), np.arange(K - 1, -1, -1)
            ru, rd = rho[ia][:, cols], rho[idn][:, cols]
            la = loop_areas(ru, rd, PRb[ia, gi, :], PRb[idn, gi, :], PRm[ia, gi], PRm[idn, gi], R, wb[gi][:B], wb[gi][:B], rng, joint=True)
            XL = Xk[:, 0, cols]; XR = Xk[:, 1, cols]
            with np.errstate(all="ignore"):
                Aal = np.where(XL + XR > 0, (XL - XR) / (XL + XR), np.nan)
            res[gname] = {"rho_r_up": ru.mean(1).tolist(), "rho_r_down": rd.mean(1).tolist(), "PR_r_up": PRm[ia, gi].tolist(),
                          "PR_r_down": PRm[idn, gi].tolist(), "area": la, "candidate_spec_D200": bool(la["A_rho"] > 3 * la["A_rho_se"]),
                          "candidate_noise_corrected_D200": bool(la["A_rho"] - la["A_rho_noise_floor"] > 3 * la["A_rho_se"]),
                          "AL_asym_up": np.nanmean(Aal[ia], 1).tolist(), "AL_asym_down": np.nanmean(Aal[idn], 1).tolist(),
                          "AL_asym_absmax_replica": float(np.nanmax(np.abs(Aal))),
                          # per dwell: share of replicas with an active AL whose dwell-mean |A| > 0.6, and the mean AL fraction
                          "frac_onelobe_up": [float(np.mean(np.abs(a[np.isfinite(a)]) > 0.6)) if np.isfinite(a).any() else None for a in Aal[ia]],
                          "frac_onelobe_down": [float(np.mean(np.abs(a[np.isfinite(a)]) > 0.6)) if np.isfinite(a).any() else None for a in Aal[idn]],
                          "XAL_up": ((XL * 248 + XR * 275)[ia] / 523).mean(1).tolist(), "XAL_down": ((XL * 248 + XR * 275)[idn] / 523).mean(1).tolist(),
                          "AL_dead_frac_up": (~np.isfinite(Aal[ia])).mean(1).tolist(), "AL_dead_frac_down": (~np.isfinite(Aal[idn])).mean(1).tolist()}
            print(f"M2c g={g:g} {gname}: A_rho {la['A_rho']:.3f} +- {la['A_rho_se']:.3f} (floor {la['A_rho_noise_floor']:.3f}) "
                  f"A_PR {la['A_PR']:.4f}; max |dln rho| {max(la['d_ln_rho']):.2f} at r {R[int(np.argmax(la['d_ln_rho']))]:g}; "
                  f"AL asym range {np.nanmin(Aal):.2f}..{np.nanmax(Aal):.2f}", flush=True)
        # audit of every replica-dwell with |A_dwell| > 0.6: how much of that dwell was the AL completely silent?
        modc = z["modc"]; rps = int(round(m["D"] / DT)) // 5
        XLa, XRa = Xk[:, 0, :], Xk[:, 1, :]
        with np.errstate(all="ignore"):
            Aa = (XLa - XRa) / (XLa + XRa)
        aud = []
        for seg, col in np.argwhere(np.abs(Aa) > 0.6):
            al = modc[seg * rps + 1:(seg + 1) * rps + 1, 0, col] + modc[seg * rps + 1:(seg + 1) * rps + 1, 1, col]
            aud.append({"seg": int(seg), "col": int(col), "r": float((z["r_fwd"] if col < B else z["r_rev"])[seg]), "A": float(Aa[seg, col]),
                        "frac_records_AL_silent": float((al == 0).mean()), "mean_AL_count": float(al.mean())})
        res["onelobe_dwell_audit"] = {"n": len(aud), "n_replica_dwells": int(Aa.size),
                                      "min_frac_records_AL_silent": min((x["frac_records_AL_silent"] for x in aud), default=None),
                                      "max_mean_AL_count": max((x["mean_AL_count"] for x in aud), default=None), "dwells": aud}
        print(f"M2c g={g:g}: one-lobe dwells {len(aud)} of {Aa.size}; min share of records with a silent AL {res['onelobe_dwell_audit']['min_frac_records_AL_silent']}, "
              f"max mean AL count {res['onelobe_dwell_audit']['max_mean_AL_count']}", flush=True)
        out[f"{g:g}"] = res
    return out


# --------------------------------------------------------------------------------------------------------------- M3
def rel_l1(a, b):
    return float(np.abs(a - b).sum() / max(np.abs(a).sum(), np.abs(b).sum(), 1e-300))


def rel_l1_matrix(V):
    S = np.abs(V).sum(1)
    D = np.abs(V[:, None, :] - V[None, :, :]).sum(2)
    return D / np.maximum(np.maximum(S[:, None], S[None, :]), 1e-300)


def cluster_states(V, labels, B_per, starts, thr=0.2):
    """V [n_surv, 34]; labels: start index per row. Returns cluster census."""
    if len(V) == 0:
        return {"n_clusters": 0, "clusters": []}
    if len(V) == 1:
        cl = np.array([1])
    else:
        Dm = rel_l1_matrix(V); np.fill_diagonal(Dm, 0)
        cl = fcluster(linkage(squareform(Dm, checks=False), "average"), t=thr, criterion="distance")
    res = []
    for c in np.unique(cl):
        idx = np.where(cl == c)[0]; cen = V[idx].mean(0)
        spread = float(np.median([rel_l1(V[i], cen) for i in idx]))
        comp = {starts[s]: int((labels[idx] == s).sum()) for s in np.unique(labels[idx])}
        held = [s for s, n in comp.items() if n >= 0.5 * B_per]
        res.append({"size": int(len(idx)), "spread": spread, "mean_rho_weighted": None, "composition": comp, "held_by": held,
                    "centroid": cen.tolist()})
    res.sort(key=lambda x: -x["size"])
    return {"n_clusters": len(res), "clusters": res}


def analyze_m3():
    out = {}
    for f in sorted(glob.glob(f"{D_}/ic_r*_g*.npz")):
        z = np.load(f, allow_pickle=True); m = meta_of(z)
        r, g, B, starts = m["r"], m["g"], m["B"], m["starts"]
        Xk = z["Xk"][0]                       # [34, B*13]
        ext = z["ext_step"]; size = np.array([int(x) for x in np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz")["size"]])
        V = Xk.T                               # [cols, 34]
        alive = ext < 0
        lab = np.repeat(np.arange(len(starts)), B)
        st = {}
        means = {}
        for si, s in enumerate(starts):
            c = lab == si; a = c & alive
            if a.any():
                mu = V[a].mean(0); means[s] = mu
                spread = float(np.median([rel_l1(v, mu) for v in V[a]]))
            else:
                mu = None; spread = None
            st[s] = {"alive_T": float(alive[c].mean()), "n_surv": int(a.sum()), "spread": spread,
                     "rho_mean": float((V[a] * size).sum(1).mean() / size.sum()) if a.any() else 0.0,
                     "X_AL": float(((V[a][:, :3] * size[:3]).sum(1) / size[:3].sum()).mean()) if a.any() else 0.0,
                     "mean_vec": mu.tolist() if mu is not None else None}
        pairs = []
        ss = list(means)
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                d = rel_l1(means[ss[i]], means[ss[j]])
                if d > 0.2:
                    pairs.append({"a": ss[i], "b": ss[j], "rel_l1": d, "spreads": [st[ss[i]]["spread"], st[ss[j]]["spread"]],
                                  "distinct_spec": bool(st[ss[i]]["spread"] < 0.05 and st[ss[j]]["spread"] < 0.05)})
        cen = cluster_states(V[alive], lab[alive], B, starts)
        nonsilent_held = [c for c in cen["clusters"] if c["held_by"]]
        multi = len(nonsilent_held) >= 2
        maxd = max([rel_l1(means[a], means[b]) for a in ss for b in ss if a < b], default=0.0)
        out[f"r{r:g}_g{g:g}"] = {"file": f.replace(ROOT + "/", ""), "r": r, "g": g, "starts": st, "distinct_pairs_gt_0.2": pairs,
                                 "n_distinct_pairs_spec": int(sum(p["distinct_spec"] for p in pairs)), "max_rel_l1_between_start_means": maxd,
                                 "clusters": cen, "n_held_nonsilent_clusters": len(nonsilent_held), "multistable": bool(multi),
                                 "dead_starts": [s for s in starts if st[s]["alive_T"] < 0.5]}
        print(f"M3 r={r:g} g={g:g}: alive " + " ".join(f"{s}:{st[s]['alive_T']:.2f}" for s in starts)
              + f" | clusters {cen['n_clusters']} (held non-silent {len(nonsilent_held)}) max relL1(start means) {maxd:.3f} "
                f"spreads {np.nanmax([st[s]['spread'] or np.nan for s in starts]):.3f} -> multistable {multi}", flush=True)
    return {"cells": out, "any_multistable": any(v["multistable"] for v in out.values()),
            "any_distinct_pair_spec": any(v["n_distinct_pairs_spec"] > 0 for v in out.values())}


def analyze_m3b():
    out = {}
    for f in sorted(glob.glob(f"{D_}/icq_r*_g*.npz")):
        z = np.load(f, allow_pickle=True); m = meta_of(z)
        r, g = m["r"], m["g"]; starts = m["starts"]; attr = m["attractors"]; pairs = m["pairs"]
        Bm, Bf = m["B_mt3"], m["B_fp"]
        ext = z["ext_step"]; reg = z["reg_occ"].astype(np.float64)            # [2, n_reg, cols]
        fpr = z["fp_reg"].astype(np.float64)                                    # [n_attr, n_reg]
        sizes = [Bm] * len(attr) + [Bf] * len(attr)
        off = np.r_[0, np.cumsum(sizes)]
        grp = {s: np.arange(off[i], off[i + 1]) for i, s in enumerate(starts)}
        alive = ext < 0
        cell = {"file": f.replace(ROOT + "/", ""), "r": r, "g": g, "attractors": attr, "n_region": m["n_region"],
                "alive": {s: float(alive[c].mean()) for s, c in grp.items()}, "pairs": {}}
        coexist = False
        for pi, pl in enumerate(pairs):
            a, b = pl.split("->")
            ia, ib = attr.index(a), attr.index(b)
            d = fpr[ib] - fpr[ia]; dd = d @ d
            zl = (reg[1].T - fpr[ia]) @ d / dd; ze = (reg[0].T - fpr[ia]) @ d / dd      # [cols]
            res = {}
            for kind in ("mt3", "fp"):
                ca, cb = grp[f"{kind}:{a}"], grp[f"{kind}:{b}"]
                za, zb = zl[ca][alive[ca]], zl[cb][alive[cb]]
                k = stats.ks_2samp(za, zb) if len(za) >= 2 and len(zb) >= 2 else None
                dist = bool(k is not None and k.pvalue < 1e-3 and abs(np.median(zb) - np.median(za)) > 0.5)
                res[kind] = {"n_a": int(len(za)), "n_b": int(len(zb)), "z_late_median_a": float(np.median(za)) if len(za) else None,
                             "z_late_median_b": float(np.median(zb)) if len(zb) else None,
                             "z_early_median_a": float(np.median(ze[ca][alive[ca]])) if len(za) else None,
                             "z_early_median_b": float(np.median(ze[cb][alive[cb]])) if len(zb) else None,
                             "ks_p": float(k.pvalue) if k else None, "distinct": dist,
                             "z_a": za.tolist(), "z_b": zb.tolist()}
                coexist |= dist
            cell["pairs"][pl] = res
        cell["coexistence"] = bool(coexist)
        Xk = z["Xk"][0]
        cell["late_X_AL"] = {s: float(np.mean((Xk[0, c] * 248 + Xk[1, c] * 275 + Xk[2, c] * 3)[alive[c]]) / 526) if alive[c].any() else 0.0
                             for s, c in grp.items()}
        cell["qmf_X_AL"] = {s: float((z["fp_Xk"][i, 0] * 248 + z["fp_Xk"][i, 1] * 275 + z["fp_Xk"][i, 2] * 3) / 526) for i, s in enumerate(attr)}
        out[f"r{r:g}_g{g:.4f}"] = cell
        print(f"M3b r={r:g} g={g:.4g}: alive " + " ".join(f"{s}:{v:.2f}" for s, v in cell['alive'].items())
              + " | " + "; ".join(f"{pl}: mt3 z {cell['pairs'][pl]['mt3']['z_late_median_a']}/{cell['pairs'][pl]['mt3']['z_late_median_b']} "
                                  f"p {cell['pairs'][pl]['mt3']['ks_p']}, fp z {cell['pairs'][pl]['fp']['z_late_median_a']}/"
                                  f"{cell['pairs'][pl]['fp']['z_late_median_b']} p {cell['pairs'][pl]['fp']['ks_p']}" for pl in pairs)
              + f" -> coexistence {coexist}", flush=True)
    return {"cells": out, "any_coexistence": any(v["coexistence"] for v in out.values()), "n_cells": len(out)}


def analyze_m2b():
    out = {}
    for f in sorted(glob.glob(f"{D_}/m2b_*.npz")):
        z = np.load(f, allow_pickle=True); m = meta_of(z); B = m["B"]
        rho = z["rho"][0]; pr = z["PR_rep"][0]
        ku = stats.ks_2samp(rho[:B], rho[B:]); kp = stats.ks_2samp(pr[:B], pr[B:])
        # per-replica mean active count in time windows (modc every model time), up-start vs down-start replicas
        tot = z["modc"].sum(1).astype(float)                         # [n_rec, 2B]
        t = np.arange(tot.shape[0]) * float(z["rec_dt"])
        wins = {}
        for a0, a1 in [(0, 10), (10, 100), (100, 500), (500, 2500), (2500, 5000)]:
            sel = (t > a0) & (t <= a1)
            x = tot[sel].mean(0)
            k = stats.ks_2samp(x[:B], x[B:])
            wins[f"{a0}-{a1}"] = {"n_from_up": float(x[:B].mean()), "n_from_down": float(x[B:].mean()), "ks_p": float(k.pvalue)}
        out[os.path.basename(f)] = {"r": m["r"], "h": m["h"], "D": m["D"], "g": m["g"], "B_each": B, "rho_from_up": float(rho[:B].mean()),
                                    "rho_from_down": float(rho[B:].mean()), "PR_rep_from_up": float(pr[:B].mean()), "PR_rep_from_down": float(pr[B:].mean()),
                                    "ks_p_rho": float(ku.pvalue), "ks_p_PR": float(kp.pvalue), "windows_active_count": wins,
                                    "late_window": "t in [2500, 5000] (last half)", "coexistence": bool(ku.pvalue < 1e-3 or kp.pvalue < 1e-3)}
        print("M2b", os.path.basename(f), out[os.path.basename(f)], flush=True)
    return out


def main():
    parts = sys.argv[1:] or ["m1", "m2a", "m2c", "m3", "m3b", "m2b"]
    fn = {"m1": analyze_m1, "m2a": analyze_m2a, "m2c": analyze_m2c, "m3": analyze_m3, "m3b": analyze_m3b, "m2b": analyze_m2b}
    sp = f"{D_}/summary.json"
    summ = json.load(open(sp)) if os.path.exists(sp) else {}
    for p in parts:
        res = fn[p]()
        json.dump(jl(res), open(f"{D_}/analysis_{p}.json", "w"))
        # compact copy in summary.json (no per-replica value lists)
        def strip(x):
            if isinstance(x, dict):
                return {k: strip(v) for k, v in x.items() if k not in ("values", "late_A_values", "z_a", "z_b", "g_ext_values", "d_ln_rho", "d_PR")
                        and not k.startswith("trace_")}
            if isinstance(x, list):
                return [strip(v) for v in x]
            return x
        summ[p] = strip(jl(res))
        tmp = sp + ".tmp"; json.dump(summ, open(tmp, "w"), indent=1); os.replace(tmp, sp)
    print("wrote", sp)


if __name__ == "__main__":
    main()
