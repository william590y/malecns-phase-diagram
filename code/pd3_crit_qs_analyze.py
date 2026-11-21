"""pd3 criticality C2 / C3 / C4 analysis (CPU).

C2 (qs_real_r{0,1,6}_p*.npz, qs_module_r{0,1}.npz): quasi-stationary <n_AL>, <n_tot>, chi_AL = var(n_AL)/<n_AL>,
   chi_tot = var(n_tot)/<n_tot> (= N var(rho)/<rho>), tau_QS, PR, AL share, n_AL histograms, all against x = g/g_lin and
   R(x) (real: dense branch rows; module: computed here, pd3_common power iteration on the 526-neuron module, CPU).
   Chain comparison WITHOUT refitting: exact QS distribution of the logistic chain (b = R n (1 - n/K), d = n) at
   K_needed = 75.8 (r 0) / 36.9 (r 1) / 16.2 (r 6); chain units -> AL neurons x 1.94 (<u^2,v> sum v of the module leading
   mode, theory_notes 2.2); mean-field amplitudes 115 / 35 / 18.5 x (x - 1) (theory_notes 2.1, T10).
   Order-parameter exponent: slope of ln n_AL vs ln(R - 1) over (R - 1) sqrt(K) in [2, 8], and the chain's slope over the
   same window. chi peak (parabola in ln x through the 3 highest points) vs the chain-predicted g (0.195 / 0.209).
   Unimodality: local maxima of the n_AL histogram (Gaussian smoothing sigma 2 neurons) with height >= 5% of the maximum
   and a separating dip below 80% of the lower peak.
C3 (field_real_r{0,1}_x{0.90,1.00,1.10}.npz): <n_AL>(h), local slopes d ln n_AL / d ln h; replica SE; the exact driven
   chain (b = R n (1 - n/K) + (h/alpha_eff)(K - n), d = n, stationary distribution) at K_needed, slope plotted against n.
C4 (submodule_r{0,1}.npz): tau_top(x) (censored MLE), tau(x=1) vs K_n, K_fit per sub-module (chain fit of ln tau over
   x in {1.2, 1.5, 2.0}), log-log slopes; merged with the FSS nulls (results/pd3/redraw_fss/fss_analysis.json tau_x1, K).
Outputs: results/pd3/criticality/qs_analysis.json; figs/pd3/qs_order_parameter.png, qs_susceptibility.png,
field_scaling.png, lifetime_vs_K.png (C4 + FSS nulls; the redraw_fss version is kept as is).
"""
import glob, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh_tridiagonal

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
from pd3_common import A_EFF, DT, log_tau_smooth_fast
R_ = f"{ROOT}/results/pd3/criticality"
G_LIN = {"real": 0.16315274988719286, "module": 0.16428095549634494}
KN = {0: 75.8, 1: 36.9, 6: 16.2}
AMP = {0: 115.0, 1: 35.0, 6: 18.5}
UNIT = 1.94
rng = np.random.default_rng(7)


def R_real():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/results/pd/branch_real_dense_part*.json")):
        rows += json.load(open(f))["rows"]
    rows.sort(key=lambda r: r["g"])
    lg = np.log(np.r_[G_LIN["real"], [r["g"] for r in rows]]); Rv = np.r_[1.0, [r["R"] for r in rows]]
    slope = (Rv[1] - Rv[0]) / (lg[1] - lg[0])
    return lambda x: np.where(np.log(x * G_LIN["real"]) >= lg[0], np.interp(np.log(x * G_LIN["real"]), lg, Rv),
                              1.0 + slope * (np.log(x * G_LIN["real"]) - lg[0]))


def R_module(xs):
    import torch
    from pd3_common import exc_csr_torch, B_mats, perron_t
    mg = np.load(f"{ROOT}/results/module/module_graph.npz")
    pre = mg["pre"].astype(np.int64); post = mg["post"].astype(np.int64); w = mg["w"].astype(np.float64); s = mg["sign"].astype(np.int64)
    Ed = exc_csr_torch(pre, post, w, s[pre], float(mg["c"]), len(s), dev="cpu")
    out = []; v = None
    for x in xs:
        lam, v, _ = perron_t(B_mats(Ed, x * G_LIN["module"])[0], v); out.append(lam / A_EFF)
    return np.array(out)


def chain_qs(K, R):
    n_top = int(np.floor(K)); n = np.arange(1, n_top + 1, dtype=float)
    b = R * n * (1 - n / K); b[-1] = 0.0; d = n.copy(); m = len(n)
    diag = -(b + d); off = np.sqrt(b[:-1] * d[1:])
    lsp = 0.5 * np.concatenate([[0.0], np.cumsum(np.log(np.maximum(b[:-1], 1e-300)) - np.log(d[1:]))])
    ev, vec = eigh_tridiagonal(diag, off, select="i", select_range=(m - 1, m - 1))
    q = np.abs(vec[:, 0]) * np.exp(lsp - lsp.max()); q /= q.sum()
    mu = q @ n; var = q @ n ** 2 - mu ** 2
    tq = float(1 / -ev[0] / A_EFF) if ev[0] < -1e-12 else None          # None: beyond double precision (> ~1e12)
    return float(mu), float(var), (tq if tq is None or tq < 1e10 else None)


def driven_chain(K, R, h):
    """Stationary distribution of b(n) = R n (1 - n/K) + h' (K - n), d(n) = n on n = 0..floor(K) (h' = h / alpha_eff)."""
    n_top = int(np.floor(K)); n = np.arange(0, n_top + 1, dtype=float); hp = h / A_EFF
    b = R * n * (1 - n / K) + hp * (K - n); b[-1] = 0.0; d = n
    lp = np.concatenate([[0.0], np.cumsum(np.log(np.maximum(b[:-1], 1e-300)) - np.log(d[1:]))])
    p = np.exp(lp - lp.max()); p /= p.sum()
    return float(p @ n)


def unimodal(h_full, n0=3, sigma=1.0):
    """First-order test = two INTERIOR maxima. Tested on n_AL >= n0 (Gaussian smoothing sigma): (i) the QS state is conditioned on
    TOTAL activity > 0, so samples with only non-AL activity pile up at n_AL = 0; (ii) at every x (also in the module alone) the
    histogram has an edge excess at n_AL = 1 (~1.5x the n = 2-3 level; the exact chain QSD has none); both are absorbing-boundary
    features of the extinction channel, reported separately as P_nAL0 and edge_ratio_n1."""
    hf = np.asarray(h_full, float); h = hf[n0:]
    hs = gaussian_filter1d(h, sigma)
    pk = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] >= 0.05 * hs.max()]
    if len(hs) > 1 and hs[0] > hs[1] and hs[0] >= 0.05 * hs.max():
        pk = [0] + pk
    sig = []
    for a, b in zip(pk[:-1], pk[1:]):
        if hs[a:b + 1].min() < 0.8 * min(hs[a], hs[b]):
            sig.append((a, b))
    mode = int(np.argmax(hf[1:])) + 1
    mn = float(hf[1:max(mode, 2)].min()) if mode > 1 else float(hf[1])
    return {"n_peaks": len(pk), "peaks": [p_ + n0 for p_ in pk], "significant_pairs": [[a + n0, b + n0] for a, b in sig],
            "unimodal": len(sig) == 0, "P_nAL0": float(hf[0] / max(hf.sum(), 1e-300)), "edge_ratio_n1": float(hf[1] / mn) if mn > 0 else None,
            "tested_on": f"n_AL >= {n0}, smoothing sigma {sigma}"}


def summarize_qs(files, kind):
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["meta"])); B = meta["B"]
        for i in range(len(d["x"])):
            sn = d["s_n"][i]; sn2 = d["s_n2"][i]; ns = float(d["nsamp"][i])
            M = sn.sum(0) / (ns * B); V = sn2.sum(0) / (ns * B) - M ** 2
            # replica bootstrap of means and chi
            bi = rng.integers(0, B, (500, B))
            mb = sn[bi].sum(1) / (ns * B); vb = sn2[bi].sum(1) / (ns * B) - mb ** 2
            chib = vb / np.maximum(mb, 1e-12)
            rep_var = sn2[:, 0] / ns - (sn[:, 0] / ns) ** 2
            ab = float(d["absorb"][i].sum())
            rows.append({"x": float(d["x"][i]), "g": float(d["g"][i]), "n_tot": float(M[0]), "n_AL": float(M[1]), "n_L": float(M[2]),
                         "n_R": float(M[3]), "n_AL_se": float(mb[:, 1].std()), "n_tot_se": float(mb[:, 0].std()),
                         "chi_tot": float(V[0] / M[0]), "chi_tot_ci": np.percentile(chib[:, 0], [2.5, 97.5]).tolist(),
                         "chi_AL": float(V[1] / M[1]), "chi_AL_ci": np.percentile(chib[:, 1], [2.5, 97.5]).tolist(),
                         "chi_tot_replica_median_based": float(np.median(rep_var) / max(M[0], 1e-12)),
                         "max_replica_var_over_median": float(rep_var.max() / max(np.median(rep_var), 1e-12)),
                         "tau_QS": float(B * meta["meas"] / ab) if ab > 0 else None, "absorptions": ab,
                         "PR": float(d["occ_PR"][i]), "AL_share": float(d["occ_AL_share"][i]),
                         "hist_AL": d["hist_AL"][i].tolist(), "unimodality": unimodal(d["hist_AL"][i]),
                         "asym_hist": d["hist_asym"][i].tolist()})
    rows.sort(key=lambda r: r["x"])
    return rows


def peak_parabola(x, y):
    k = int(np.argmax(y))
    if k == 0 or k == len(y) - 1:
        return float(x[k]), float(y[k]), "edge"
    lx = np.log(x[k - 1:k + 2]); c = np.polyfit(lx, y[k - 1:k + 2], 2)
    xp = -c[1] / (2 * c[0]); return float(np.exp(xp)), float(np.polyval(c, xp)), "interior"


if __name__ == "__main__":
    out = {}
    Rr = R_real()
    # ------------------------------------------------------------------------------------------------------------ C2
    qs = {}
    for r in [0, 1, 6]:
        fs = sorted(glob.glob(f"{R_}/qs_real_r{r}_p*.npz"))
        if fs:
            qs[f"real_r{r}"] = (summarize_qs(fs, "real"), r, "real")
    for r in [0, 1]:
        fs = glob.glob(f"{R_}/qs_module_r{r}.npz")
        if fs:
            qs[f"module_r{r}"] = (summarize_qs(fs, "module"), r, "module")
    c2 = {}
    for key, (rows, r, kind) in qs.items():
        xs = np.array([q["x"] for q in rows])
        Rx = Rr(xs) if kind == "real" else R_module(xs)
        K = KN[r]
        chain = [chain_qs(K, R) for R in Rx]
        for q, R, (mu, var, tq) in zip(rows, Rx, chain):
            q["R"] = float(R); q["chain_nQS_neurons"] = UNIT * mu; q["chain_chi_neurons"] = UNIT * var / mu; q["chain_tauQS_model"] = tq
            q["chain_ratio_nAL"] = q["n_AL"] / (UNIT * mu)
            q["MF_amp_nAL"] = AMP[r] * (q["x"] - 1) if q["x"] > 1 else 0.0
        # order-parameter exponent over (R - 1) sqrt(K) in [2, 8]
        xk = (Rx - 1) * np.sqrt(K); w = (xk >= 2) & (xk <= 8)
        beta = None; beta_chain = None
        if w.sum() >= 3:
            nal = np.array([q["n_AL"] for q in rows])
            beta = float(np.polyfit(np.log(Rx[w] - 1), np.log(nal[w]), 1)[0])
            beta_chain = float(np.polyfit(np.log(Rx[w] - 1), np.log([UNIT * c[0] for c, ww in zip(chain, w) if ww]), 1)[0])
        chis = np.array([q["chi_AL"] for q in rows]); xpk, chpk, where = peak_parabola(xs, chis)
        chit = np.array([q["chi_tot"] for q in rows]); xpt, chpt, wheret = peak_parabola(xs, chit)
        # chain-predicted chi peak (same K, same R(x)), neuron units
        xf = np.exp(np.linspace(np.log(xs[0]), np.log(xs[-1]), 400)); Rf = Rr(xf) if kind == "real" else np.interp(xf, xs, Rx)
        cf = [chain_qs(K, R_) for R_ in Rf]; chif = np.array([UNIT * v / m for m, v, _ in cf]); kf = int(np.argmax(chif))
        c2[key] = {"rows": rows, "K": K, "g_lin": G_LIN[kind], "beta_window_points": int(w.sum()), "beta": beta, "beta_chain_same_window": beta_chain,
                   "chi_AL_peak": {"x": xpk, "g": xpk * G_LIN[kind], "height": chpk, "where": where},
                   "chi_tot_peak": {"x": xpt, "g": xpt * G_LIN[kind], "height": chpt, "where": wheret},
                   "chain_chi_peak": {"x": float(xf[kf]), "g": float(xf[kf] * G_LIN[kind]), "height_neurons": float(chif[kf]),
                                      "height_chain_units": float(chif[kf] / UNIT)},
                   "all_unimodal": all(q["unimodality"]["unimodal"] for q in rows),
                   "ratio_nAL_to_chain": {"x": xs.tolist(), "ratio": [q["chain_ratio_nAL"] for q in rows]}}
        print(f"C2 {key}: chi_AL peak x {xpk:.3f} (g {xpk*G_LIN[kind]:.4f}, {where}) height {chpk:.2f}; chain peak g {xf[kf]*G_LIN[kind]:.4f} "
              f"height {chif[kf]:.2f}; chi_tot peak g {xpt*G_LIN[kind]:.4f}; beta {beta} (chain {beta_chain}); unimodal {c2[key]['all_unimodal']}", flush=True)
        for q in rows[::3]:
            print(f"   x {q['x']:.3f} R {q['R']:.3f} n_AL {q['n_AL']:.2f}+-{q['n_AL_se']:.2f} chain {q['chain_nQS_neurons']:.2f} "
                  f"(ratio {q['chain_ratio_nAL']:.2f}) chi_AL {q['chi_AL']:.2f} chi_tot {q['chi_tot']:.1f} tau_QS {q['tau_QS']} "
                  f"chain tau {q['chain_tauQS_model']} PR {q['PR']:.0f} P(nAL=0) {q['unimodality']['P_nAL0']:.4f}", flush=True)
    out["C2"] = c2
    # ------------------------------------------------------------------------------------------------------------ C3
    c3 = {}
    for r in [0, 1]:
        for x in [0.90, 1.00, 1.10]:
            f = f"{R_}/field_real_r{r}_x{x:.2f}.npz"
            if not os.path.exists(f):
                continue
            d = np.load(f, allow_pickle=True); meta = json.loads(str(d["meta"])); B = meta["B"]
            hs = d["h"]; nal = []; se = []; ntot = []
            for i in range(len(hs)):
                sn = d["s_n"][i]; ns = float(d["nsamp"][i]); rep = sn[:, 1] / ns
                nal.append(rep.mean()); se.append(rep.std(ddof=1) / np.sqrt(B)); ntot.append((sn[:, 0] / ns).mean())
            nal = np.array(nal); se = np.array(se)
            sl = np.diff(np.log(nal)) / np.diff(np.log(hs))
            sl_se = np.sqrt((se[1:] / nal[1:]) ** 2 + (se[:-1] / nal[:-1]) ** 2) / np.diff(np.log(hs))
            K = KN[r]; Rx = float(Rr(np.array([x]))[0])
            hc = np.geomspace(1e-8, 1e-1, 200); nc = np.array([UNIT * driven_chain(K, Rx, h) for h in hc])
            slc = np.diff(np.log(nc)) / np.diff(np.log(hc))
            c3[f"r{r}_x{x:.2f}"] = {"h": hs.tolist(), "n_AL": nal.tolist(), "n_AL_se": se.tolist(), "n_tot": ntot, "slope": sl.tolist(),
                                     "slope_se": sl_se.tolist(), "h_mid": np.sqrt(hs[1:] * hs[:-1]).tolist(),
                                     "n_mid": np.sqrt(nal[1:] * nal[:-1]).tolist(), "min_slope": float(sl.min()),
                                     "chain": {"K": K, "R": Rx, "h": hc.tolist(), "n": nc.tolist(), "slope": slc.tolist(),
                                               "min_slope": float(slc.min())}}
            print(f"C3 r{r} x{x}: n_AL {np.round(nal, 4).tolist()} slopes {np.round(sl, 3).tolist()} (chain min slope {slc.min():.3f})", flush=True)
    out["C3"] = c3
    # ------------------------------------------------------------------------------------------------------------ C4
    c4 = {}
    for r in [0, 1]:
        f = f"{R_}/submodule_r{r}.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True); xs = d["xs"]; meta = json.loads(str(d["meta"])); steps = int(round(meta["T"] / DT))
        rows = []
        for i, n in enumerate(d["n"]):
            taus, dths = [], []
            for j in range(len(xs)):
                e = d["ext_step"][i, j]; dd = e >= 0
                taus.append(float(np.where(dd, e, steps).sum() * DT / max(dd.sum(), 1)) if dd.sum() else float("inf")); dths.append(int(dd.sum()))
            # K_fit: chain ln tau_top(R(x) alpha_eff, K) vs measured, x in {1.2, 1.5, 2.0} with >= 3 deaths
            Rx = d["R_x"][i]; use = [j for j in range(len(xs)) if xs[j] > 1.01 and dths[j] >= 3 and np.isfinite(taus[j])]
            Kf = None
            if use:
                Kg = np.exp(np.linspace(np.log(3), np.log(3000), 400))
                err = [np.sum([(log_tau_smooth_fast(Rx[j] * A_EFF, K) - np.log(taus[j])) ** 2 for j in use]) for K in Kg]
                Kf = float(Kg[int(np.argmin(err))])
            rows.append({"n": int(n), "g_lin": float(d["g_lin"][i]), "K_n": float(d["K_n"][i]), "inv_ipr": float(d["inv_ipr"][i]),
                         "tau": dict(zip([f"{x:g}" for x in xs], taus)), "deaths": dict(zip([f"{x:g}" for x in xs], dths)),
                         "tau_x1": taus[0], "K_fit": Kf, "R_x": Rx.tolist(), "mod_overlap": int(d["mod_overlap"][i]),
                         "pred_tau_x1_from_K_n": float(np.exp(log_tau_smooth_fast(Rx[0] * A_EFF, float(d["K_n"][i]))))})
        K_ = np.array([q["K_n"] for q in rows]); t1 = np.array([q["tau_x1"] for q in rows])
        p, cov = np.polyfit(np.log(K_), np.log(t1), 1, cov=True)
        Kf_ = np.array([q["K_fit"] or np.nan for q in rows]); okf = np.isfinite(Kf_)
        pf = np.polyfit(np.log(Kf_[okf]), np.log(t1[okf]), 1, cov=True) if okf.sum() >= 3 else None
        c4[f"r{r}"] = {"rows": rows, "slope_tau_vs_Kn": float(p[0]), "slope_se": float(np.sqrt(cov[0, 0])),
                       "slope_tau_vs_Kfit": float(pf[0][0]) if pf else None, "slope_Kfit_se": float(np.sqrt(pf[1][0, 0])) if pf else None}
        print(f"C4 r{r}: tau(x=1) vs K_n slope {p[0]:.3f} +- {np.sqrt(cov[0,0]):.3f}; vs K_fit {c4[f'r{r}']['slope_tau_vs_Kfit']}", flush=True)
        for q in rows:
            print(f"   n {q['n']} K_n {q['K_n']:.1f} K_fit {q['K_fit']} tau {[round(v, 1) for v in q['tau'].values()]} pred(K_n) {q['pred_tau_x1_from_K_n']:.1f}", flush=True)
    # merge with the FSS nulls (redraw_fss), r = 0 and 1
    fss = json.load(open(f"{ROOT}/results/pd3/redraw_fss/fss_analysis.json"))["per_graph"]
    merged = {}
    for r in [0, 1]:
        pts = []
        for gr in fss:
            if gr["kind"] == "cnull" and abs(gr["r"] - r) < 1e-9 and gr.get("tau_x1"):
                dg = gr["diag"] if isinstance(gr["diag"], dict) else json.loads(gr["diag"])
                pts.append({"N": gr["N"], "seed": gr["seed"], "tau_x1": gr["tau_x1"], "K_nc": dg["K_nc_glin"], "K_fit": gr.get("K_fit")})
        sub = c4.get(f"r{r}", {}).get("rows", [])
        Kn = np.array([q["K_nc"] for q in pts] + [q["K_n"] for q in sub]); tt = np.array([q["tau_x1"] for q in pts] + [q["tau_x1"] for q in sub])
        Kfit = np.array([q["K_fit"] or np.nan for q in pts] + [q["K_fit"] or np.nan for q in sub])
        res = {"n_null": len(pts), "n_sub": len(sub), "nulls": pts}
        if len(Kn) >= 3:
            pa, ca = np.polyfit(np.log(Kn), np.log(tt), 1, cov=True); res["slope_all_Kunfitted"] = [float(pa[0]), float(np.sqrt(ca[0, 0]))]
            ok = np.isfinite(Kfit)
            if ok.sum() >= 3:
                pb, cb = np.polyfit(np.log(Kfit[ok]), np.log(tt[ok]), 1, cov=True); res["slope_all_Kfit"] = [float(pb[0]), float(np.sqrt(cb[0, 0]))]
            pn = [q for q in pts]
            if len(pn) >= 3:
                pc, cc = np.polyfit(np.log([q["K_nc"] for q in pn]), np.log([q["tau_x1"] for q in pn]), 1, cov=True)
                res["slope_nulls_Knc"] = [float(pc[0]), float(np.sqrt(cc[0, 0]))]
        merged[f"r{r}"] = res
        print(f"C4+FSS r{r}: {res.get('slope_all_Kunfitted')} (K unfitted: K_n / K_nc), {res.get('slope_all_Kfit')} (K_fit); nulls K_nc {res.get('slope_nulls_Knc')}", flush=True)
    out["C4"] = c4; out["C4_merged"] = merged
    out["prediction_tau_x1"] = {"law": "1.90 K^0.503 / alpha_eff (model time)", "K37": 12.0, "K54.7": 14.8, "K75.8": 17.7,
                                "source": "theory_bd.json R1 fit (K >= 1024) / plan 4.5"}
    json.dump(out, open(f"{R_}/qs_analysis.json", "w"), indent=1, default=float)
    print("saved", f"{R_}/qs_analysis.json")
