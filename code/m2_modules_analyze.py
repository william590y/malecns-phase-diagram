"""m2_modules_analyze.py -- lane m2/modules Parts B, C, D: L/R symmetry, multistability, modules ON alone (CPU).

plan.md sections 4-6 (criteria pre-registered there). Inputs: results/m2/modules/sim/<SET>/*.npz (index_<SET>.json).
Usage: python m2_modules_analyze.py [lr] [ms] [iso]   -> results/m2/modules/{lr,multistab,iso}.json
  lr   Part B: for the pairs AL (P0_AL-L / P0_AL-R) and HEM (HEM_L / HEM_R): A(t) = (f_L - f_R)/(f_L + f_R) on the record grid;
       t_sym (first |A| < 0.3 after a one-sided start), late mean A per survivor, c1-c3 (pd3, unchanged), c4 + L/R correlation.
  ms   Part C + D(i): per cell (model, setting, r, g) the late P0 34-vector per replica (win_mean over the late windows),
       average-linkage clustering (relative L1, cut 0.2), multistable = >= 2 clusters each holding >= 50% of some start's B;
       secondary P1 / P2 vectors; (b) quiet vs all KS on late nonORN activity; D(i) confinement / t_out / module-alone state.
  iso  Part D(ii): isolated modules, tau MLE per g, isotonic ln tau crossings of ln 200 / ln 3000 (decay-lane rules).
"""
import glob, json, os, sys
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
import m2_obs as O

OUT = f"{ROOT}/results/m2/modules"
DT = 0.2
RJ = json.load(open(f"{OUT}/groups/regions.json"))
START_REGION = {"AL_L": "P0_AL-L", "AL_R": "P0_AL-R", "AL": "S_AL", "nonAL": "S_nonAL", "CB": "S_CB", "VNC": "S_VNC",
                "OL": "S_OL", "isl1": "P0_isl1", "isl3": "P0_isl3", "isl5": "P0_isl5", "hemiL": "HEM_L", "hemiR": "HEM_R"}
PAIRS = {"AL": ("P0_AL-L", "P0_AL-R"), "HEM": ("HEM_L", "HEM_R")}
ONE_SIDED = {"AL": ("AL_L", "AL_R"), "HEM": ("hemiL", "hemiR")}


def jl(x):
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


def load_index(sets):
    idx = []
    for s in sets:
        p = f"{OUT}/sim/index_{s}.json"
        if os.path.exists(p):
            idx += json.load(open(p))
    return idx


_CACHE = {}


def read(path, keys):
    """loads (and caches) every requested array of a file once; the cache holds the files of the current cell group"""
    c = _CACHE.setdefault(path, {})
    miss = [k for k in list(keys) + ["meta"] if k not in c]
    if miss:
        d = np.load(path, allow_pickle=True)
        for k in miss:
            if k in d.files:
                c[k] = d[k]
    return {k: c[k] for k in keys if k in c}, json.loads(str(c["meta"]))


def clear_cache():
    _CACHE.clear()


def bimodal(x):
    """pd3_multistab_analyze.bimodal (unchanged rule)."""
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
    res.update(dBIC=float(dbic), mu=mu.tolist(), sd=sd.tolist(), w=wt.tolist(), bimodal=bool(dbic > 10 and wt.min() >= 0.1 and sep))
    return res


def late_bounds(setting, T):
    return (T / 2, T)                                    # (a) [1000, 2000]; (b) [1500, 3000] -> use [2000, 3000] below


def late_window(setting, T):
    return (1000.0, 2000.0) if setting == "a" else (T - 1000.0, T)


# ----------------------------------------------------------------------------------------------------------------- B
def lr_cell(files, setting, gi):
    """files: {start: path}; one g index. Returns per pair the stats and criteria."""
    res = {}
    cache = {}
    for s, p in files.items():
        z, m = read(p, ["t", "g", "surv", "reg_rho", "reg_names", "reg_sizes"])
        names = [str(x) for x in z["reg_names"]]
        cache[s] = (z, m, names)
    z0, m0, names0 = next(iter(cache.values()))
    T = float(m0["T"]); w0, w1 = late_window(setting, T)
    for pair, (nL, nR) in PAIRS.items():
        pr = {"starts": {}}
        lateA = {}
        for s, (z, m, names) in cache.items():
            t = z["t"]; iL, iR = names.index(nL), names.index(nR)
            fL = z["reg_rho"][gi, :, iL, :].astype(np.float64); fR = z["reg_rho"][gi, :, iR, :].astype(np.float64)
            den = fL + fR
            with np.errstate(invalid="ignore", divide="ignore"):
                A = np.where(den > 0, (fL - fR) / den, np.nan)                     # [nt, B]
            alive_T = z["surv"][gi, -1, :] > 0
            late = (t > w0) & (t <= w1)
            al = A[late]; valid = np.isfinite(al)
            surv = (alive_T if setting == "a" else np.ones_like(alive_T)) & (valid.mean(0) >= 0.5)
            with np.errstate(invalid="ignore"):
                mA = np.where(valid.sum(0) > 0, np.nansum(al, 0) / np.maximum(valid.sum(0), 1), np.nan)
            lateA[s] = mA[surv]
            # L/R correlation over the late records and the fraction of records with |A| > 0.6 (c4)
            cors, fr6 = [], []
            for b in np.where(surv)[0]:
                x, y = fL[late, b], fR[late, b]
                if x.std() > 0 and y.std() > 0:
                    cors.append(np.corrcoef(x, y)[0, 1])
                v = al[:, b][np.isfinite(al[:, b])]
                fr6.append(float((np.abs(v) > 0.6).mean()) if len(v) else np.nan)
            st = {"B": int(len(alive_T)), "alive_T": float(alive_T.mean()), "n_surv": int(surv.sum()),
                  "late_fL": float(fL[late][:, surv].mean()) if surv.any() else None,
                  "late_fR": float(fR[late][:, surv].mean()) if surv.any() else None,
                  "late_A_mean": float(np.mean(mA[surv])) if surv.any() else None,
                  "late_A_sd": float(np.std(mA[surv])) if surv.sum() > 1 else None,
                  "late_A_absmax": float(np.max(np.abs(mA[surv]))) if surv.any() else None,
                  "corr_LR_median": float(np.median(cors)) if cors else None, "n_corr": len(cors),
                  "frac_rec_absA_gt_0.6_median": float(np.nanmedian(fr6)) if fr6 else None,
                  "frac_surv_with_25pct_rec_absA_gt_0.6": float(np.mean(np.array(fr6) >= 0.25)) if fr6 else None}
            if s in ONE_SIDED[pair]:
                tsym = np.full(len(alive_T), np.nan); status = []
                for b in range(len(alive_T)):
                    hit = np.where(np.abs(A[:, b]) < 0.3)[0]
                    dead = np.where(~np.isfinite(A[:, b]))[0]
                    if len(hit) and (not len(dead) or hit[0] < dead[0]):
                        tsym[b] = t[hit[0]]; status.append("sym")
                    elif len(dead):
                        status.append("pair_died_first")
                    else:
                        status.append("censored")
                status = np.array(status); ok = status == "sym"
                st["t_sym"] = {"n_sym": int(ok.sum()), "n_died_first": int((status == "pair_died_first").sum()),
                               "n_censored": int((status == "censored").sum()),
                               "median": float(np.median(tsym[ok])) if ok.any() else None,
                               "q25": float(np.percentile(tsym[ok], 25)) if ok.any() else None,
                               "q75": float(np.percentile(tsym[ok], 75)) if ok.any() else None,
                               "max": float(np.max(tsym[ok])) if ok.any() else None}
                keep = (mA[surv] > 0.6) if s == ONE_SIDED[pair][0] else (mA[surv] < -0.6)
                st["frac_surv_keeping_start_side"] = float(keep.mean()) if surv.any() else None
            pr["starts"][s] = st
        sL, sR = ONE_SIDED[pair]
        sym_start = "all"
        if sL in lateA and sR in lateA:
            L, R_ = lateA[sL], lateA[sR]
            ks = stats.ks_2samp(L, R_) if len(L) >= 2 and len(R_) >= 2 else None
            pr["ks_L_vs_R"] = {"stat": float(ks.statistic), "p": float(ks.pvalue)} if ks else None
            fl = pr["starts"][sL].get("frac_surv_keeping_start_side")
        else:
            ks, fl = None, None
            pr["ks_L_vs_R"] = None
        pr["bimodality_all"] = bimodal(lateA[sym_start]) if sym_start in lateA else None
        c1 = fl is not None and fl >= 0.25
        c2 = ks is not None and ks.pvalue < 1e-3
        c3 = bool(pr["bimodality_all"] and pr["bimodality_all"]["bimodal"])
        sa = pr["starts"].get(sym_start, {})
        c4 = bool(sa.get("frac_surv_with_25pct_rec_absA_gt_0.6") is not None and sa["frac_surv_with_25pct_rec_absA_gt_0.6"] >= 0.25
                  and sa.get("corr_LR_median") is not None and sa["corr_LR_median"] < -0.2)
        pr["criteria"] = {"c1": bool(c1), "c2": bool(c2), "c3": c3, "c4_switching": c4}
        pr["broken_symmetry"] = bool(c1 and c2 and c3)
        pr["testable"] = bool(sL in lateA and sR in lateA and len(lateA[sL]) >= 10 and len(lateA[sR]) >= 10)
        res[pair] = pr
    return res


def analyze_lr(only=None, tag=""):
    idx = load_index(["LRa", "DRb"])
    if only:
        idx = [e for e in idx if e["setting"] == only]
    groups = {}
    for e in idx:
        if not os.path.exists(e["out"]):
            continue
        groups.setdefault((e["setting"], e["model"], e["r"]), {})[e["start"]] = e
    out = {}
    for (setting, model, r), st in sorted(groups.items()):
        if not any(s in st for s in ("AL_L", "AL_R", "hemiL", "hemiR")):
            continue
        G = st[next(iter(st))]["g"]
        rows = []
        files = {s: e["out"] for s, e in st.items() if s in ("all", "quiet", "AL_L", "AL_R", "hemiL", "hemiR")}
        for gi, g in enumerate(G):
            c = lr_cell(files, setting, gi)
            c["g"] = g
            rows.append(c)
            msg = " | ".join(f"{p}: n_surv L/R/all " + "/".join(str(c[p]['starts'].get(s, {}).get('n_surv')) for s in
                                                               (*ONE_SIDED[p], 'all'))
                             + f" tsym med {c[p]['starts'].get(ONE_SIDED[p][0], {}).get('t_sym', {}).get('median')}"
                             + f" lateA all {c[p]['starts'].get('all', {}).get('late_A_mean')}"
                             + f" crit {c[p]['criteria']} corr {c[p]['starts'].get('all', {}).get('corr_LR_median')}"
                             for p in PAIRS)
            print(f"LR {setting} {model} r={r:g} g={g:.4g}: {msg}", flush=True)
        out[f"{setting}|{model}|{r:g}"] = {"setting": setting, "model": model, "r": r, "starts": sorted(files), "rows": rows}
        clear_cache()
    summ = {}
    for key, v in out.items():
        for p in PAIRS:
            rs = [rw[p] for rw in v["rows"]]
            summ[f"{key}|{p}"] = {"n_points": len(rs), "n_testable": sum(x["testable"] for x in rs),
                                  "n_broken": sum(x["broken_symmetry"] for x in rs),
                                  "n_c1": sum(x["criteria"]["c1"] for x in rs), "n_c2": sum(x["criteria"]["c2"] for x in rs),
                                  "n_c3": sum(x["criteria"]["c3"] for x in rs), "n_c4": sum(x["criteria"]["c4_switching"] for x in rs)}
    json.dump(jl({"cells": out, "summary": summ}), open(f"{OUT}/lr{tag}.json", "w"))
    print(json.dumps(summ, indent=0))


# ----------------------------------------------------------------------------------------------------------------- C, D(i)
def rel_l1(a, b):
    return float(np.abs(a - b).sum() / max(np.abs(a).sum(), np.abs(b).sum(), 1e-300))


def rel_l1_matrix(V):
    S = np.abs(V).sum(1)
    D = np.abs(V[:, None, :] - V[None, :, :]).sum(2)
    return D / np.maximum(np.maximum(S[:, None], S[None, :]), 1e-300)


def cluster_states(V, lab, Bs, starts, thr=0.2):
    """pd3 cluster_states with per-start B (held = >= 50% of that start's replicas)."""
    if len(V) == 0:
        return {"n_clusters": 0, "clusters": []}
    cl = np.array([1]) if len(V) == 1 else fcluster(linkage(squareform(np.nan_to_num(rel_l1_matrix(V)), checks=False), "average"),
                                                     t=thr, criterion="distance")
    res = []
    for c in np.unique(cl):
        ii = np.where(cl == c)[0]; cen = V[ii].mean(0)
        comp = {starts[s]: int((lab[ii] == s).sum()) for s in np.unique(lab[ii])}
        held = [s for s, n in comp.items() if n >= 0.5 * Bs[s]]
        res.append({"size": int(len(ii)), "spread": float(np.median([rel_l1(V[i], cen) for i in ii])), "composition": comp,
                    "held_by": held, "centroid_sum": float(cen.sum())})
    res.sort(key=lambda x: -x["size"])
    return {"n_clusters": len(res), "clusters": res}


def ms_cell(files, setting, gi_map, sizes_ref):
    """files: {start: path}; gi_map: {start: g index in that start's file} (same g for all)"""
    starts = sorted(files)
    vecs = {P: [] for P in ("P0", "P1", "P2")}; labs = []; Bs = {}; per = {}; nonorn = {}
    for si, s in enumerate(starts):
        gi = gi_map[s]
        z, m = read(files[s], ["t", "surv", "reg_rho", "reg_names", "reg_sizes", "win_mean", "win_bounds"])
        names = [str(x) for x in z["reg_names"]]; sizes = z["reg_sizes"].astype(np.float64)
        T = float(m["T"]); w0, w1 = late_window(setting, T)
        wb = z["win_bounds"]; wsel = [k for k in range(len(wb)) if wb[k][0] >= w0 - 1e-6 and wb[k][1] <= w1 + 1e-6]
        wm = z["win_mean"][gi][wsel].astype(np.float64).mean(0)                     # [R, B]
        B = wm.shape[1]; Bs[s] = B
        alive = z["surv"][gi, -1, :] > 0
        keep = alive if setting == "a" else np.ones(B, bool)
        ip = {P: [k for k, n in enumerate(names) if n.startswith(P + "_")] for P in ("P0", "P1", "P2")}
        for P in vecs:
            vecs[P].append(wm[ip[P]][:, keep].T)
        labs.append(np.full(keep.sum(), si))
        tot = (wm[ip["P0"]] * sizes[ip["P0"], None]).sum(0)                          # active count per replica (late)
        st = {"B": B, "alive_T": float(alive.mean()), "n_keep": int(keep.sum()),
              "rho_late_mean": float(tot[keep].mean() / sizes[ip["P0"]].sum()) if keep.any() else 0.0,
              "nonORN_late_median": float(np.median(wm[names.index("nonORN")])) if "nonORN" in names else None,
              "X_AL_late": float(wm[names.index("S_AL")][keep].mean()) if keep.any() else 0.0}
        if "nonORN" in names:
            nonorn[s] = {k: wm_k for k, wm_k in [("late", wm[names.index("nonORN")])]}
            # gap-persistence windows (b): [1500, 2000] and [2500, 3000]
            for lab_, (a0, a1) in (("w1500", (1500, 2000)), ("w2500", (2500, 3000)), ("w500", (500, 1000))):
                kk = [k for k in range(len(wb)) if abs(wb[k][0] - a0) < 1e-6 and abs(wb[k][1] - a1) < 1e-6]
                if kk:
                    nonorn[s][lab_] = z["win_mean"][gi][kk[0], names.index("nonORN")].astype(np.float64)
        # D(i) confinement and t_out
        if s in START_REGION and keep.any():
            j = names.index(START_REGION[s])
            inside = wm[j] * sizes[j]
            with np.errstate(invalid="ignore", divide="ignore"):
                C = np.where(tot > 0, inside / tot, np.nan)
            st["confinement_median"] = float(np.nanmedian(C[keep])) if np.isfinite(C[keep]).any() else None
            st["confinement_q"] = [float(x) for x in np.nanpercentile(C[keep], [25, 75])] if np.isfinite(C[keep]).any() else None
            rr = z["reg_rho"][gi].astype(np.float64)                                     # [nt, R, B]
            cin = rr[:, j, :] * sizes[j]; call = (rr[:, ip["P0"], :] * sizes[ip["P0"]][None, :, None]).sum(1)
            t = z["t"]; tout = []
            for b in range(B):
                hit = np.where(call[:, b] - cin[:, b] > cin[:, b])[0]
                tout.append(t[hit[0]] if len(hit) else np.inf)
            tout = np.array(tout)
            st["t_out_median"] = float(np.median(tout)) if np.isfinite(tout).any() else None
            st["frac_t_out_finite"] = float(np.isfinite(tout).mean())
            st["start_size"] = int(sizes[j])
        per[s] = st
    lab = np.concatenate(labs) if labs else np.array([])
    res = {"starts": per}
    for P in ("P0", "P1", "P2"):
        V = np.concatenate(vecs[P]) if vecs[P] else np.zeros((0, 1))
        cen = cluster_states(V, lab, Bs, starts)
        held = [c for c in cen["clusters"] if c["held_by"] and c["centroid_sum"] > 0]
        means = {s: V[lab == i].mean(0) for i, s in enumerate(starts) if (lab == i).any()}
        maxd = max([rel_l1(means[a], means[b]) for a in means for b in means if a < b], default=0.0)
        res[P] = {"clusters": cen, "n_held_nonsilent_clusters": len(held), "multistable": bool(len(held) >= 2),
                  "max_rel_l1_start_means": maxd,
                  "rel_l1_to_all": {s: rel_l1(means[s], means["all"]) for s in means if "all" in means}}
    # D(i) module-alone state (pre-registered) on P0
    for s, st in per.items():
        if s in START_REGION and s not in ("all", "quiet"):
            d_all = res["P0"]["rel_l1_to_all"].get(s)
            st["module_alone_state"] = bool(setting == "a" and st["alive_T"] >= 0.25 and (st.get("confinement_median") or 0) >= 0.8
                                            and d_all is not None and d_all > 0.2)
    # (b) quiet vs all coexistence candidate
    if setting == "b" and "quiet" in nonorn and "all" in nonorn:
        q, a = nonorn["quiet"], nonorn["all"]
        ks = stats.ks_2samp(a["late"], q["late"])
        mq, ma = np.median(q["late"]), np.median(a["late"])
        ratio = ma / mq if mq > 0 else np.inf
        g15 = abs(np.median(a.get("w1500", np.nan)) - np.median(q.get("w1500", np.nan)))
        g25 = abs(np.median(a.get("w2500", np.nan)) - np.median(q.get("w2500", np.nan)))
        res["quiet_vs_all"] = {"ks_p": float(ks.pvalue), "ks_stat": float(ks.statistic), "median_quiet": float(mq),
                               "median_all": float(ma), "ratio": float(ratio), "gap_1500_2000": float(g15), "gap_2500_3000": float(g25),
                               "gap_500_1000": float(abs(np.median(a.get("w500", np.nan)) - np.median(q.get("w500", np.nan)))),
                               "candidate": bool(ks.pvalue < 1e-3 and (ratio >= 1.5 or ratio <= 1 / 1.5) and g25 >= 0.5 * g15)}
    return res


def analyze_ms(only=None, tag=""):
    idx = load_index(["LRa", "ICa", "DRb"])
    if only:
        idx = [e for e in idx if e["setting"] == only]
    groups = {}
    for e in idx:
        if os.path.exists(e["out"]):
            groups.setdefault((e["setting"], e["model"], e["r"]), {})[e["start"]] = e
    out = {}
    for (setting, model, r), st in sorted(groups.items()):
        if len(st) < 2:
            continue
        G = sorted({round(g, 6) for e in st.values() for g in e["g"]})
        rows = []
        for g in G:
            fl = {s: e["out"] for s, e in st.items() if any(abs(x / g - 1) < 1e-6 for x in e["g"])}
            gim = {s: [i for i, x in enumerate(st[s]["g"]) if abs(x / g - 1) < 1e-6][0] for s in fl}
            if len(fl) < 2:
                continue
            c = ms_cell(fl, setting, gim, None); c["g"] = g
            rows.append(c)
            alone = [s for s, v in c["starts"].items() if v.get("module_alone_state")]
            qa = c.get("quiet_vs_all", {})
            print(f"MS {setting} {model} r={r:g} g={g:.4g}: alive " + " ".join(f"{s}:{v['alive_T']:.2f}" for s, v in c["starts"].items())
                  + f" | P0 clusters {c['P0']['clusters']['n_clusters']} held {c['P0']['n_held_nonsilent_clusters']}"
                  + f" maxL1 {c['P0']['max_rel_l1_start_means']:.3f} multi {c['P0']['multistable']}/{c['P1']['multistable']}/{c['P2']['multistable']}"
                  + f" alone {alone} q/a {qa.get('ratio')} p {qa.get('ks_p')} cand {qa.get('candidate')}", flush=True)
        out[f"{setting}|{model}|{r:g}"] = {"setting": setting, "model": model, "r": r, "starts": sorted(st), "rows": rows}
        clear_cache()
    summ = {k: {"n_cells": len(v["rows"]), "n_multistable_P0": sum(rw["P0"]["multistable"] for rw in v["rows"]),
                "n_multistable_P1": sum(rw["P1"]["multistable"] for rw in v["rows"]),
                "n_multistable_P2": sum(rw["P2"]["multistable"] for rw in v["rows"]),
                "max_rel_l1_P0": max(rw["P0"]["max_rel_l1_start_means"] for rw in v["rows"]),
                "module_alone": sorted({(s, rw["g"]) for rw in v["rows"] for s, x in rw["starts"].items() if x.get("module_alone_state")}),
                "quiet_all_candidates": [rw["g"] for rw in v["rows"] if rw.get("quiet_vs_all", {}).get("candidate")]}
            for k, v in out.items()}
    json.dump(jl({"cells": out, "summary": summ}), open(f"{OUT}/multistab{tag}.json", "w"))
    print(json.dumps(jl(summ), indent=0))


# ----------------------------------------------------------------------------------------------------------------- D(ii)
def analyze_iso():
    idx = load_index(["ISO"])
    full = json.load(open(f"{ROOT}/results/m2/decay/boundaries.json"))["rows"]
    out = {}
    for e in idx:
        if not os.path.exists(e["out"]):
            continue
        z, m = read(e["out"], ["t", "g", "surv", "occ"])
        T = float(m["T"]); t = z["t"]; G = np.asarray(z["g"], float)
        taus, deaths, E, alive = [], [], [], []
        for i in range(len(G)):
            ext = O.extinction_times(t, z["surv"][i])
            tau, d = O.tau_mle(ext, T)
            taus.append(tau); deaths.append(d); E.append(float(np.minimum(ext, T).sum())); alive.append(float(np.isinf(ext).mean()))
        d = np.array(deaths, float); E = np.array(E)
        lt = O.isotonic(np.log(E / np.maximum(d, 0.5)), d + 0.5)
        cr = {}
        for nm, thr in (("S", 200.0), ("on", 3000.0)):
            hit = np.where(lt >= np.log(thr))[0]
            if not len(hit):
                cr[nm] = {"g": None, "status": "none", "bracket": [float(G[-1]), None]}
            elif hit[0] == 0:
                cr[nm] = {"g": None, "status": "below_grid", "bracket": [None, float(G[0])]}
            else:
                k = hit[0]; fr = (np.log(thr) - lt[k - 1]) / (lt[k] - lt[k - 1]) if lt[k] > lt[k - 1] else 1.0
                cr[nm] = {"g": float(np.exp(np.log(G[k - 1]) + fr * np.log(G[k] / G[k - 1]))), "status": "ok",
                          "bracket": [float(G[k - 1]), float(G[k])]}
        fk = f"{'M1' if e['model'] == 'M1' else e['model']}_real_r{e['r']:g}"
        fb = full.get(fk, {}).get("on", {}).get("g")
        PR = []
        for i in range(len(G)):
            o = z["occ"][i].astype(np.float64)
            PR.append(O.participation_ratio(o)[1] if o.sum() > 0 else 0.0)
        out[e["name"]] = {"model": e["model"], "r": e["r"], "module": e["module"], "N": int(z["occ"].shape[1]), "g": G.tolist(),
                          "tau": taus, "deaths": deaths, "alive_T": alive, "PR_over_N": PR, "S": cr["S"], "on": cr["on"],
                          "full_brain_on": fb, "on_over_full": (cr["on"]["g"] / fb) if (cr["on"]["g"] and fb) else None}
        print(f"ISO {e['name']}: S {cr['S']['g']} on {cr['on']['g']} ({cr['on']['status']}) full-brain on {fb}", flush=True)
    json.dump(jl(out), open(f"{OUT}/iso.json", "w"))


if __name__ == "__main__":
    parts = sys.argv[1:] or ["lr", "ms", "iso"]
    if "lr" in parts:
        analyze_lr()
    if "lr_a" in parts:
        analyze_lr("a", "_a")
    if "ms" in parts:
        analyze_ms()
    if "ms_a" in parts:
        analyze_ms("a", "_a")
    if "iso" in parts:
        analyze_iso()
