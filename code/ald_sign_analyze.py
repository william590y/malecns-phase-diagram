"""E2 (transmitter signs): analysis of the coarse (g, r) sweeps + spectra -> results/aldef/signs/summary.json, summary.md
tables, figs/aldef/signs_*.png.

PHASE CRITERIA (same as analysis2_phase.py, T = 3000):
  per replica extinction time = first recorded time with surv = 0 (censored at T); tau = censored-exponential MLE
  (sum of min(t_i, T)) / deaths; tau_reg = E / max(deaths, 0.5) for the crossing only. alive = fraction alive at T.
  PR = (sum occ)^2 / sum occ^2 / N (occ = replica-mean occupancy over [T/2, T]).
  ln tau_reg (weights deaths + 0.5), alive (weight B), PR (weight 1) made monotone increasing in g by weighted isotonic
  regression; first crossings interpolated linearly in ln g:  S = tau 200,  on = tau 3000,  half = alive 1/2,
  D = PR N/4 among sustained points (g >= on; if the first sustained grid point already has PR >= 1/4, D = that point,
  flag at_onset). Uncertainty for deterministic runs: envelope of the grid bracket and a replica bootstrap (NBOOT).
POOLING: for the deterministic conditions the coarse sweep (B = 32, 42 g in 0.12..30) and the refinement sweep
  (sim_fine/, B = 64, 20 g in 0.7..1.25 x the coarse onset) are pooled point by point (analysis2_phase.py convention);
  'coarse_only' keeps the coarse-sweep boundaries as a check.
IGNITION SITE: the occupancy vector at the first sustained grid point (first g >= on; from the refinement sweep when present) -- and, where it is earlier, at the
  first grid point with any occupancy (first_active) -- described by its composition over coarse regions / AL categories /
  cell types (al_common.labels), its share in the 526-neuron AL module, PR in neurons, n90 (neurons holding 90% of the
  occupancy), and its cosine similarity with |v_1| (the leading right eigenvector of the condition's excitatory operator,
  from ald_sign_spec.py) -- 1 = activity has exactly the shape of the leading mode.
"""
import glob, json, os, sys
import numpy as np
from scipy.optimize import isotonic_regression
from scipy.stats import spearmanr

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
from al_common import labels, composition          # noqa: E402

RES = f"{ROOT}/results/aldef/signs"; SIM = f"{RES}/sim"; SPEC = f"{RES}/spec"; FIG = f"{ROOT}/figs/aldef"
os.makedirs(FIG, exist_ok=True)
T_REF, TAU_S, PR_D, NBOOT = 3000.0, 200.0, 0.25, 400
rng = np.random.default_rng(20261109)
L = labels(); N = L["N"]
module = np.zeros(N, bool); module[np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)["idx"]] = True
isAL = L["reg"] == "AL"
# per-neuron mean occupancy (fraction of time active, replica mean incl. dead replicas) of AL cell classes and big regions
CATMASK = {c: L["cat"] == c for c in ("AL ORN", "AL uPN", "AL mPN", "AL lLN1", "AL lLN2", "AL v/il3LN", "AL LN other")}
CATMASK.update({"lLN1_bc": L["ty"] == "lLN1_bc", "VNC": L["reg"] == "VNC", "MB KC": L["cat"] == "MB KC",
                "OL intrinsic": L["reg"] == "OL intrinsic", "CB other": L["reg"] == "CB other"})
DET = ["base", "lLN1bc_inh", "lLN1bc_zero", "lLN1bc_lLN2_inh"]
DET_LABEL = {"base": "baseline (release signs)", "lLN1bc_inh": "(i) lLN1_bc inhibitory", "lLN1bc_zero": "(ii) lLN1_bc sign 0",
             "lLN1bc_lLN2_inh": "(iii) lLN1_bc + ACh lLN2 inhibitory"}
SPEC_OF = {"base": "base", "lLN1bc_inh": "lLN1bc_off", "lLN1bc_zero": "lLN1bc_off", "lLN1bc_lLN2_inh": "lLN1bc_lLN2ach_off"}


# ------------------------------------------------------------------------------------------------- per-point statistics
def load_points(path):
    """Per-g points of one run: g, per-replica extinction times (censored at T) and death flags, occupancy."""
    d = np.load(path)
    t, g, surv = d["t"], d["g"], d["surv"]
    win = t <= T_REF + 1e-6; Tc = float(t[win].max())
    dead = surv[:, win, :] == 0                                     # [g, t, B]
    died = dead.any(1)
    ext = np.where(died, t[win][np.argmax(dead, 1)], Tc)            # [g, B]
    occ = (d["occ_ens"][0] if "occ_ens" in d.files else d["occ"]).astype(np.float64)
    lab = os.path.relpath(path, RES)
    return [{"g": float(g[i]), "ext": ext[i].astype(float), "died": died[i].astype(bool), "occ": occ[i], "B": int(died.shape[1]),
             "src": lab} for i in range(len(g))]


def pool(paths):
    """Merge the points of several runs (same g within 1e-4 relative: replicas pooled, occupancy B-weighted)."""
    pts = sorted([p for f in paths for p in load_points(f)], key=lambda x: x["g"])
    out = []
    for x in pts:
        if out and abs(x["g"] / out[-1]["g"] - 1) < 1e-4:
            y = out[-1]
            y["occ"] = (y["occ"] * y["B"] + x["occ"] * x["B"]) / (y["B"] + x["B"])
            y["ext"] = np.r_[y["ext"], x["ext"]]; y["died"] = np.r_[y["died"], x["died"]]; y["B"] += x["B"]
            y["src"] += "+" + x["src"]
        else:
            out.append(dict(x))
    return out


def point_arrays(pts):
    dd = np.array([p["died"].sum() for p in pts], float); E = np.array([p["ext"].sum() for p in pts])
    Bv = np.array([p["B"] for p in pts], float)
    return dd, E, E / np.maximum(dd, 0.5), 1 - dd / Bv, Bv


def pr_of(occ):
    s = occ.sum(1); q = (occ ** 2).sum(1)
    return np.where(s > 0, s ** 2 / np.where(q > 0, q, 1) / N, 0.0)


def first_cross(g, y, thr):
    """First crossing of increasing y (isotonic-fitted) with thr, interpolated in ln g. None if never; g[0] if already."""
    idx = np.where(y >= thr)[0]
    if len(idx) == 0:
        return None
    k = idx[0]
    if k == 0:
        return float(g[0])
    x0, x1, y0, y1 = np.log(g[k - 1]), np.log(g[k]), y[k - 1], y[k]
    f = 0.5 if y1 == y0 else (thr - y0) / (y1 - y0)
    return float(np.exp(x0 + f * (x1 - x0)))


def boundaries(g, dd, tau_reg, alive, Bv, pr):
    lt = isotonic_regression(np.log(tau_reg), weights=dd + 0.5, increasing=True).x
    al = isotonic_regression(alive, weights=Bv, increasing=True).x
    S = first_cross(g, lt, np.log(TAU_S)); on = first_cross(g, lt, np.log(T_REF)); half = first_cross(g, al, 0.5)
    D, at_onset = None, False
    if on is not None:
        ks = np.where(g >= on * (1 - 1e-9))[0]
        if len(ks):
            p = isotonic_regression(pr[ks], increasing=True).x
            if p[0] >= PR_D:
                D, at_onset = float(g[ks[0]]), True
            else:
                D = first_cross(g[ks], p, PR_D)
    return {"S": S, "on": on, "half": half, "D": D, "D_at_onset": at_onset}


def bracket(g, x):
    if x is None:
        return None
    k = np.searchsorted(g, x)
    return [float(g[max(k - 1, 0)]), float(g[min(k, len(g) - 1)])]


def describe_occ(o, v1=None):
    s = o.sum()
    if s <= 0:
        return None
    p = o / s
    srt = np.sort(p)[::-1]; cs = np.cumsum(srt)
    out = {"PR_neurons": float(1 / (p ** 2).sum()), "PR_frac": float(1 / (p ** 2).sum() / N),
           "n90": int(np.searchsorted(cs, 0.9) + 1), "module_share": float(p[module].sum()), "AL_share": float(p[isAL].sum()),
           "region": composition(p, L["reg"], 6), "cat": composition(p, L["cat"], 6), "types": composition(p, L["ty"], 8),
           "mean_occ_module": float(o[module].mean()), "mean_occ": float(o.mean()),
           "mean_occ_by_cat": {c: float(o[m].mean()) for c, m in CATMASK.items()}}
    if v1 is not None:
        a = np.sqrt(np.maximum(v1, 0))
        out["cos_v1"] = float(o @ a / np.sqrt((o @ o) * (a @ a)))
    return out


def analyze_run(paths, v1=None, boot=False, ignition_from=None):
    """paths: one npz or a list (pooled). ignition_from: optional list of npz whose points are used for the ignition-site
    description (e.g. the fine sweep, closer to the onset); default = all pooled points."""
    paths = [paths] if isinstance(paths, str) else list(paths)
    pts = pool(paths)
    g = np.array([p["g"] for p in pts]); occ = np.array([p["occ"] for p in pts])
    pr = pr_of(occ)
    dd, E, tau_reg, alive, Bv = point_arrays(pts)
    b = boundaries(g, dd, tau_reg, alive, Bv, pr)
    pl = [{"g": float(g[i]), "B": int(Bv[i]), "deaths": int(dd[i]), "tau": float(E[i] / dd[i]) if dd[i] else None,
           "alive": float(alive[i]), "PR_frac": float(pr[i]),
           "occ_module_share": float(occ[i][module].sum() / occ[i].sum()) if occ[i].sum() > 0 else None,
           "src": pts[i]["src"]} for i in range(len(g))]
    res = {"files": [os.path.relpath(p, RES) for p in paths], "bounds": b, "points": pl}
    for k in ("S", "on", "half", "D"):
        res["bounds"][k + "_grid"] = bracket(g, b[k])
    if boot and b["on"] is not None:
        bs = {k: [] for k in ("S", "on", "half")}
        for _ in range(NBOOT):
            dd2, E2 = np.empty(len(pts)), np.empty(len(pts))
            for i, p in enumerate(pts):
                ii = rng.integers(0, p["B"], p["B"])
                dd2[i] = p["died"][ii].sum(); E2[i] = p["ext"][ii].sum()
            bb = boundaries(g, dd2, E2 / np.maximum(dd2, 0.5), 1 - dd2 / Bv, Bv, pr)
            for k in bs:
                if bb[k] is not None:
                    bs[k].append(bb[k])
        for k, v in bs.items():
            if len(v) > 0.9 * NBOOT:
                lo, hi = np.percentile(v, [2.5, 97.5])
                gb = res["bounds"][k + "_grid"] or [lo, hi]
                res["bounds"][k + "_ci"] = [float(min(lo, gb[0])), float(max(hi, gb[1]))]
                res["bounds"][k + "_boot95"] = [float(lo), float(hi)]
    # ignition site
    if ignition_from:
        ipts = pool(ignition_from)
        gi = np.array([p["g"] for p in ipts]); occi = np.array([p["occ"] for p in ipts])
    else:
        gi, occi = g, occ
    if b["on"] is not None:
        ks = np.where(gi >= b["on"] * (1 - 1e-9))[0]
        if len(ks):
            k = ks[0]
            res["ignition"] = {"g": float(gi[k]), **(describe_occ(occi[k], v1) or {})}
        ka95 = np.where((g >= b["on"] * (1 - 1e-9)) & (alive >= 0.95))[0]
        if len(ka95):      # first point above the onset with >= 95% of replicas alive (occupancy not diluted by deaths)
            res["first_all_alive"] = {"g": float(g[ka95[0]]), "alive": float(alive[ka95[0]]),
                                      **(describe_occ(occ[ka95[0]], v1) or {})}
        k2 = np.searchsorted(g, 2 * b["on"])
        if k2 < len(g):
            res["at_2x_onset"] = {"g": float(g[k2]), **(describe_occ(occ[k2], v1) or {})}
    ka = np.where(occi.sum(1) > 0)[0]
    if len(ka):
        res["first_active"] = {"g": float(gi[ka[0]]), **describe_occ(occi[ka[0]], v1)}
    res["pr_curve"] = pr.tolist(); res["alive_curve"] = alive.tolist(); res["g"] = g.tolist()
    res["tau_reg_curve"] = tau_reg.tolist()
    return res


# --------------------------------------------------------------------------------------------------------- spectra
spec_det = json.load(open(f"{SPEC}/det.json")) if os.path.exists(f"{SPEC}/det.json") else None
Pdet = np.load(f"{SPEC}/det_P.npz") if os.path.exists(f"{SPEC}/det_P.npz") else None
ens_spec = {}
for f in sorted(glob.glob(f"{SPEC}/ens_*.json")):
    j = json.load(open(f))
    key = f"{j['unit']}_{j['gt']}"
    ens_spec.setdefault(key, {"rows": [], "P_lLN1_bc": j["P_lLN1_bc"], "seed": j["seed"], "K": j["K"]})
    ens_spec[key]["rows"] += j["rows"]
Pens = {}
for f in glob.glob(f"{SPEC}/ens_*_P.npz"):
    key = "_".join(os.path.basename(f).split("_")[1:3])
    z = np.load(f)
    for k in z.files:
        Pens[(key, int(k[4:]))] = z[k].astype(np.float64)

summary = {"criteria": __doc__, "det": {}, "ens": {}, "spec": {}}
if spec_det:
    for name, c in spec_det["conds"].items():
        m0 = c["modes"][0]
        summary["spec"][name] = {"lam": c["lam"], "g_lin": c["g_lin"], "g_lin_naive": c["g_lin_naive"],
                                 "mode0": {k: m0[k] for k in ("inv_ipr", "n50", "n90", "module_share", "AL_class_share",
                                                              "region", "cat", "types", "side")},
                                 "first_AL_mode": c["first_AL_mode"], "first_nonAL_mode": c["first_nonAL_mode"],
                                 "modes_region": [max(m["region"], key=m["region"].get) for m in c["modes"]],
                                 "modes_AL_share": [m["AL_class_share"] for m in c["modes"]],
                                 "modes_module_share": [m["module_share"] for m in c["modes"]],
                                 "modes_inv_ipr": [m["inv_ipr"] for m in c["modes"]],
                                 "modes_types": [list(m["types"])[:4] for m in c["modes"]],
                                 "left_mode0_region": c["left_mode0"]["region"], "left_mode0_types": c["left_mode0"]["types"]}
    summary["spec_selection"] = spec_det["selection"]
for key, e in ens_spec.items():
    rows = sorted(e["rows"], key=lambda r: r["draw"])
    lam = np.array([r["lam"][0] for r in rows])
    st = [r["lLN1_bc_signs"] for r in rows]
    n_exc = np.array([s.get("1", 0) for s in st])
    alm = np.array([r["mode0"]["AL_class_share"] >= 0.5 for r in rows])
    summary["spec"][f"ens_{key}"] = {
        "K": len(rows), "P_lLN1_bc(+1,-1,0)": e["P_lLN1_bc"], "lam1_median": float(np.median(lam)),
        "lam1_range": [float(lam.min()), float(lam.max())], "frac_mode0_in_AL": float(alm.mean()),
        "frac_lLN1bc_all_exc": float((n_exc == 30).mean()), "lLN1bc_exc_count_mean": float(n_exc.mean()),
        "lam1_median_when_lLN1bc_exc": float(np.median(lam[n_exc == 30])) if (n_exc == 30).any() else None,
        "lam1_median_when_lLN1bc_not_exc": float(np.median(lam[n_exc == 0])) if (n_exc == 0).any() else None,
        "frac_mode0_in_AL_when_lLN1bc_not_exc": float(alm[n_exc == 0].mean()) if (n_exc == 0).any() else None,
        "mode0_region_counts": {k: int(v) for k, v in zip(*np.unique([max(r["mode0"]["region"], key=r["mode0"]["region"].get)
                                                                      for r in rows], return_counts=True))},
        "spearman_lam1_vs_lLN1bc_exc": float(spearmanr(lam, n_exc).correlation) if n_exc.std() > 0 else None}

# ----------------------------------------------------------------------------------------------------- deterministic
for name in DET:
    for r in (0, 1, 3):
        p = f"{SIM}/{name}_r{r}.npz"
        if not os.path.exists(p):
            continue
        pf = f"{RES}/sim_fine/{name}_r{r}.npz"
        paths = [p] + ([pf] if os.path.exists(pf) else [])
        v1 = Pdet[f"{SPEC_OF[name]}_P"][:, 0].astype(np.float64) if Pdet is not None else None
        res = analyze_run(paths, v1, boot=True, ignition_from=[pf] if os.path.exists(pf) else None)
        res["coarse_only"] = analyze_run(p, v1, boot=False)["bounds"]
        summary["det"][f"{name}_r{r}"] = res
        b = res["bounds"]
        ig = res.get("ignition", {})
        print(f"{name:16s} r={r}: S {b['S']} on {b['on']} half {b['half']} D {b['D']} | ignition g {ig.get('g')} "
              f"module {ig.get('module_share')} PR {ig.get('PR_neurons')} region {list(ig.get('region', {}).items())[:2]}",
              flush=True)

# ---------------------------------------------------------------------------------------------------------- ensembles
ENS = {"ens_type": "type_ignore", "ens_neuron": "neuron_ignore"}
for cname, skey in ENS.items():
    rows_spec = {r["draw"]: r for r in ens_spec.get(skey, {"rows": []})["rows"]}
    for r in (0, 1, 3):
        runs = []
        for p in sorted(glob.glob(f"{SIM}/{cname}_d*_r{r}.npz")):
            dr = int(os.path.basename(p).split("_d")[1].split("_")[0])
            v1 = Pens.get((skey, dr))
            res = analyze_run(p, v1, boot=False)
            sp = rows_spec.get(dr, {})
            res["draw"] = dr
            res["lam1"] = sp.get("lam", [None])[0]; res["g_lin_naive"] = sp.get("g_lin_naive")
            res["lLN1_bc_signs"] = sp.get("lLN1_bc_signs"); res["spec_mode0_region"] = (
                max(sp["mode0"]["region"], key=sp["mode0"]["region"].get) if sp else None)
            res["spec_mode0_AL_share"] = sp.get("mode0", {}).get("AL_class_share")
            runs.append(res)
        if not runs:
            continue
        on = np.array([x["bounds"]["on"] if x["bounds"]["on"] is not None else np.nan for x in runs])
        exc = np.array([x["lLN1_bc_signs"].get("1", 0) if x["lLN1_bc_signs"] else -1 for x in runs])
        ig_mod = np.array([x.get("ignition", {}).get("module_share", np.nan) for x in runs])
        ig_reg = [max(x["ignition"]["region"], key=x["ignition"]["region"].get) if x.get("ignition", {}).get("region") else None
                  for x in runs]
        Dv = np.array([x["bounds"]["D"] if x["bounds"]["D"] is not None else np.nan for x in runs])
        glin = np.array([x["g_lin_naive"] if x["g_lin_naive"] is not None else np.nan for x in runs])
        ok = np.isfinite(on) & np.isfinite(glin)
        summary["ens"][f"{cname}_r{r}"] = {
            "n_draws": len(runs), "n_sustained_by_30": int(np.isfinite(on).sum()),
            "on_median": float(np.nanmedian(on)) if np.isfinite(on).any() else None,
            "on_range": [float(np.nanmin(on)), float(np.nanmax(on))] if np.isfinite(on).any() else None,
            "on_median_lLN1bc_exc": float(np.nanmedian(on[exc == 30])) if np.isfinite(on[exc == 30]).any() else None,
            "on_median_lLN1bc_not_exc": float(np.nanmedian(on[exc == 0])) if np.isfinite(on[exc == 0]).any() else None,
            "n_lLN1bc_all_exc": int((exc == 30).sum()), "n_lLN1bc_none_exc": int((exc == 0).sum()),
            "frac_ignition_module_major": float(np.nanmean(ig_mod > 0.5)) if np.isfinite(ig_mod).any() else None,
            "ignition_module_share_median": float(np.nanmedian(ig_mod)) if np.isfinite(ig_mod).any() else None,
            "ignition_module_share_range": [float(np.nanmin(ig_mod)), float(np.nanmax(ig_mod))] if np.isfinite(ig_mod).any() else None,
            "frac_ignition_module_ge_0.2": float(np.mean(ig_mod[np.isfinite(ig_mod)] >= 0.2)) if np.isfinite(ig_mod).any() else None,
            "ignition_module_share_median_lLN1bc_exc": float(np.nanmedian(ig_mod[exc == 30])) if np.isfinite(ig_mod[exc == 30]).any() else None,
            "ignition_module_share_median_lLN1bc_not_exc": float(np.nanmedian(ig_mod[exc == 0])) if np.isfinite(ig_mod[exc == 0]).any() else None,
            "lLN1bc_exc_counts": [int(x) for x in exc],
            "on_per_draw": [None if not np.isfinite(x) else float(x) for x in on],
            "ignition_module_share_per_draw": [None if not np.isfinite(x) else float(x) for x in ig_mod],
            "D_per_draw": [None if not np.isfinite(x) else float(x) for x in Dv],
            "spearman_on_vs_n_exc_lLN1bc": float(spearmanr(on[np.isfinite(on)], exc[np.isfinite(on)]).correlation)
                if np.isfinite(on).sum() > 3 and exc[np.isfinite(on)].std() > 0 else None,
            "ignition_region_counts": {k: ig_reg.count(k) for k in sorted(set(x for x in ig_reg if x))},
            "D_median": float(np.nanmedian(Dv)) if np.isfinite(Dv).any() else None,
            "spearman_on_vs_glin_naive": float(spearmanr(on[ok], glin[ok]).correlation) if ok.sum() > 3 else None,
            "runs": runs}
        s = summary["ens"][f"{cname}_r{r}"]
        print(f"   module share at ignition: median {s['ignition_module_share_median']}, exc {s['ignition_module_share_median_lLN1bc_exc']} "
              f"not {s['ignition_module_share_median_lLN1bc_not_exc']}, frac >= 0.2 {s['frac_ignition_module_ge_0.2']}, D median {s['D_median']}")
        print(f"{cname} r={r}: {s['n_draws']} draws, on median {s['on_median']} range {s['on_range']} | lLN1bc exc "
              f"{s['on_median_lLN1bc_exc']} vs not {s['on_median_lLN1bc_not_exc']} | ignition regions "
              f"{s['ignition_region_counts']} | rho(on, glin) {s['spearman_on_vs_glin_naive']}", flush=True)

# how many excitatory modes are linearly supercritical (lambda > alpha_eff / g) at each sustained onset: top-400 modes of
# the BASELINE operator (results/pd/tail_real_r0_dir.npz, spectrum_tail.py); AL modes = AL share >= 0.5. Removing the
# excitatory output of lLN1_bc / ACh lLN2 removes AL modes and leaves the non-AL top-8 unchanged (spec/det.json), so the
# non-AL count is the relevant one when the AL clique is not excitatory.
tail = np.load(f"{ROOT}/results/pd/tail_real_r0_dir.npz")
AEFF = (1 - np.exp(-0.2)) / 0.2
summary["supercritical_modes_at_onset"] = {}
for k, v in summary["det"].items():
    on = v["bounds"]["on"]
    if on is None:
        continue
    th = AEFF / on
    summary["supercritical_modes_at_onset"][k] = {
        "on": on, "lambda_threshold": float(th), "n_nonAL_modes": int(((tail["lam"] > th) & (tail["AL"] < 0.5)).sum()),
        "n_AL_modes": int(((tail["lam"] > th) & (tail["AL"] >= 0.5)).sum())}

# baseline of the earlier study (pd2 pooled runs) for comparison
fb = json.load(open(f"{ROOT}/results/analysis2/flow_sim_boundaries.json"))
summary["pd2_baseline"] = {str(int(x["r"])): {k: x[k] for k in ("S", "on", "D", "half", "D_not_reached_up_to")}
                           for x in fb["rows"] if x["r"] in (0.0, 1.0, 3.0)}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


json.dump(clean(summary), open(f"{RES}/summary.json", "w"), indent=1)
print("saved", f"{RES}/summary.json")
