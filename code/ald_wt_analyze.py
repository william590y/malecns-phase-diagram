"""E4 (weight transforms in simulation): analysis of the coarse + fine sweeps -> results/aldef/weights/summary.json
(and summary_coarse.json from the coarse sweeps alone, used to place the fine sweeps), table.md.

PHASE CRITERIA (identical to analysis2_phase.py / ald_sign_analyze.py, T = 3000):
  per replica extinction time = first recorded time with surv = 0 (censored at T); tau = censored-exponential MLE
  (sum of min(t_i, T)) / deaths; tau_reg = E / max(deaths, 0.5) for the crossing. alive = fraction alive at T.
  PR = (sum occ)^2 / sum occ^2 / N (occ = replica-mean occupancy over [T/2, T]).
  ln tau_reg (weights deaths + 0.5), alive (weight B), PR (weight 1) made monotone in g by weighted isotonic regression;
  first crossings interpolated linearly in ln g: S = tau 200 (start of the slow window), on = tau 3000 (sustained onset),
  half = alive 1/2, D = PR N/4 among sustained points (if the first sustained point already has PR >= 1/4, D = that point,
  flag at_onset). CIs: envelope of the grid bracket and a 400-sample replica bootstrap.
  Phases: Q (tau < 200), S (200 <= tau < 3000), L (sustained, PR < N/4), D (sustained, PR >= N/4).
POOLING: coarse (two interleaved halves, B = 32) + fine (B = 64, 0.7-1.25 x coarse onset) pooled point by point.
UNITS (one sweep per transform; under --norm raw the global scale only rescales g):
  g      = equal mean in-strength (--wnorm renorm: mean_i sum_j c_T T(w_ij) = 1 for every transform);
  g_lam  = g * lam1(T) / lam1(linear): equal LEADING EIGENVALUE of the excitatory operator (linear g unchanged);
  g/g_lin(T): distance above the exact linear instability of silence (tanh branching operator, ald_wt_spec.py).
IGNITION SITE: occupancy at the first sustained grid point (first g >= on; fine sweep when present) and at the first
  grid point with any second-half occupancy (first_active): composition by coarse region / AL category / cell type,
  AL-module share, PR in neurons, n90, cos with |v_1| of the transform's own leading right eigenvector.
LATE ACTIVITY in the slow window (S <= g < on, grid points from 0.9 S): time-weighted composition of the active neurons
  over record times 100 <= t <= T of live replicas (reg_rho x region size, weights = time between records), over the
  near-disjoint regions AL_module, AL_rest, MB, VNC, OL, CX, LH, CB_other, AN_DN (other = remainder; the module shares 30
  LH and 4 MB neurons, so shares can double count slightly). This is what keeps the long transients alive below the onset.
  (A 'last active neuron' measure was tried and dropped: the last neuron is usually a downstream echo of the dying driver.)
CAUSAL AL TESTS (sim_noAL: --clamp_off module; sim_ALonly: --clamp_off "~module"; 18 g from 0.85 to 4 x the coarse onset,
  B = 32): same criteria; on_noAL / on_intact > 1 means the AL module was needed for the first sustained state at that g;
  on_ALonly / on_intact ~ 1 means the module alone can sustain there.
Usage: python ald_wt_analyze.py [coarse]
"""
import glob, json, os, sys
import numpy as np
from scipy.optimize import isotonic_regression

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
from al_common import labels, composition          # noqa: E402

RES = f"{ROOT}/results/aldef/weights"; SIM = f"{RES}/sim"; FINE = f"{RES}/sim_fine"; SPEC = f"{RES}/spec"
T_REF, TAU_S, PR_D, NBOOT = 3000.0, 200.0, 0.25, 400
TRANSFORMS = ["linear", "cap50", "cap20", "cap10", "sqrt", "log1p", "binary"]
COARSE_ONLY = len(sys.argv) > 1 and sys.argv[1] == "coarse"
rng = np.random.default_rng(20261109)
L = labels(); N = L["N"]
module = np.zeros(N, bool); module[np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)["idx"]] = True
isAL = L["reg"] == "AL"
CATMASK = {c: L["cat"] == c for c in ("AL ORN", "AL uPN", "AL mPN", "AL lLN1", "AL lLN2", "AL v/il3LN", "AL LN other")}
CATMASK.update({"lLN1_bc": L["ty"] == "lLN1_bc", "VNC": L["reg"] == "VNC", "MB KC": L["cat"] == "MB KC",
                "OL intrinsic": L["reg"] == "OL intrinsic", "CB other": L["reg"] == "CB other"})
CARRIER_REGS = ["AL_module", "AL_rest", "MB", "VNC", "OL", "CX", "LH", "CB_other", "AN_DN"]
spec = {n: json.load(open(f"{SPEC}/{n}.json")) for n in TRANSFORMS}
Pv = {n: np.load(f"{SPEC}/{n}_P.npz")["P"][:, 0].astype(np.float64) for n in TRANSFORMS}
LAM_LIN = spec["linear"]["lam"][0]


def load_points(path):
    d = np.load(path)
    t, g, surv = d["t"], d["g"], d["surv"]
    win = t <= T_REF + 1e-6; Tc = float(t[win].max())
    dead = surv[:, win, :] == 0
    died = dead.any(1)
    ext = np.where(died, t[win][np.argmax(dead, 1)], Tc)
    occ = d["occ"].astype(np.float64)
    names = [str(x) for x in d["reg_names"]]
    last = d["reg_last"] if "reg_last" in d.files else None            # [g, R, B]
    regT = d["reg_rho"][:, -1] if "reg_rho" in d.files else None        # [g, R, B] final active fraction
    late = None
    if "reg_rho" in d.files:            # time-weighted active counts per region over 100 <= t <= T while the replica is alive
        tw = np.diff(np.r_[0.0, t]); m = (t >= 100) & (t <= T_REF + 1e-6)
        cnt = d["reg_rho"] * d["reg_sizes"][None, None, :, None]           # [g, t, R, B]
        al = (surv > 0)[:, :, None, :]
        late = (cnt * al * (tw * m)[None, :, None, None]).sum(1)            # [g, R, B]
        late_tot = (d["rho"] * N * (surv > 0) * (tw * m)[None, :, None]).sum(1)   # [g, B]
    lab = os.path.relpath(path, RES)
    return [{"g": float(g[i]), "ext": ext[i].astype(float), "died": died[i].astype(bool), "occ": occ[i], "B": int(died.shape[1]),
             "last": None if last is None else last[i].astype(float), "regT": None if regT is None else regT[i].astype(float),
             "late": None if late is None else late[i].astype(float), "late_tot": None if late is None else late_tot[i].astype(float),
             "names": names, "src": lab} for i in range(len(g))]


def pool(paths):
    pts = sorted([p for f in paths for p in load_points(f)], key=lambda x: x["g"])
    out = []
    for x in pts:
        if out and abs(x["g"] / out[-1]["g"] - 1) < 1e-4:
            y = out[-1]
            y["occ"] = (y["occ"] * y["B"] + x["occ"] * x["B"]) / (y["B"] + x["B"])
            y["ext"] = np.r_[y["ext"], x["ext"]]; y["died"] = np.r_[y["died"], x["died"]]; y["B"] += x["B"]
            y["last"] = np.concatenate([y["last"], x["last"]], 1); y["regT"] = np.concatenate([y["regT"], x["regT"]], 1)
            y["late"] = np.concatenate([y["late"], x["late"]], 1); y["late_tot"] = np.r_[y["late_tot"], x["late_tot"]]
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
           "MB_share": float(p[L["reg"] == "MB"].sum()),
           "region": composition(p, L["reg"], 8), "cat": composition(p, L["cat"], 10), "types": composition(p, L["ty"], 10),
           "mean_occ_module": float(o[module].mean()), "mean_occ": float(o.mean()),
           "mean_occ_by_cat": {c: float(o[m].mean()) for c, m in CATMASK.items()}}
    if v1 is not None:
        a = np.sqrt(np.maximum(v1, 0))
        out["cos_v1"] = float(o @ a / np.sqrt((o @ o) * (a @ a)))
    return out


def late_composition(pts):
    """Time-weighted composition of the activity of live replicas at 100 <= t <= T, pooled over the points pts."""
    if not pts or pts[0]["late"] is None:
        return None
    nm = pts[0]["names"]
    tot = sum(float(p["late_tot"].sum()) for p in pts)
    if tot <= 0:
        return None
    out = {k: sum(float(p["late"][nm.index(k)].sum()) for p in pts) / tot for k in CARRIER_REGS if k in nm}
    out["other"] = max(0.0, 1 - sum(out.values()))
    return {"share": dict(sorted(out.items(), key=lambda kv: -kv[1])), "active_neuron_time": tot,
            "g": [p["g"] for p in pts], "n_replicas": int(sum(p["B"] for p in pts))}


def region_lifetimes(p):
    """Mean over replicas of each region's last-active time (censored at T) and the fraction of replicas in which the
    region is active at the last record."""
    if p["last"] is None:
        return None
    return {n: {"mean_last": float(np.minimum(p["last"][i], T_REF).mean()), "active_at_T": float((p["regT"][i] > 0).mean())}
            for i, n in enumerate(p["names"])}


def analyze(paths, name, boot=True, ignition_from=None):
    pts = pool(paths)
    g = np.array([p["g"] for p in pts]); occ = np.array([p["occ"] for p in pts])
    pr = pr_of(occ)
    dd, E, tau_reg, alive, Bv = point_arrays(pts)
    b = boundaries(g, dd, tau_reg, alive, Bv, pr)
    v1 = Pv[name]
    pl = [{"g": float(g[i]), "B": int(Bv[i]), "deaths": int(dd[i]), "tau": float(E[i] / dd[i]) if dd[i] else None,
           "alive": float(alive[i]), "PR_frac": float(pr[i]),
           "occ_module_share": float(occ[i][module].sum() / occ[i].sum()) if occ[i].sum() > 0 else None,
           "occ_MB_share": float(occ[i][L["reg"] == "MB"].sum() / occ[i].sum()) if occ[i].sum() > 0 else None,
           "late": late_composition([pts[i]]), "region_life": region_lifetimes(pts[i]), "src": pts[i]["src"]}
          for i in range(len(g))]
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
    if ignition_from:
        ipts = pool(ignition_from)
        gi = np.array([p["g"] for p in ipts]); occi = np.array([p["occ"] for p in ipts])
    else:
        gi, occi = g, occ
    if b["on"] is not None:
        ks = np.where(gi >= b["on"] * (1 - 1e-9))[0]
        if len(ks):
            res["ignition"] = {"g": float(gi[ks[0]]), **(describe_occ(occi[ks[0]], v1) or {})}
        ka95 = np.where((g >= b["on"] * (1 - 1e-9)) & (alive >= 0.95))[0]
        if len(ka95):
            res["first_all_alive"] = {"g": float(g[ka95[0]]), "alive": float(alive[ka95[0]]),
                                      **(describe_occ(occ[ka95[0]], v1) or {})}
        k2 = np.searchsorted(g, 2 * b["on"])
        if k2 < len(g):
            res["at_2x_onset"] = {"g": float(g[k2]), **(describe_occ(occ[k2], v1) or {})}
    ka = np.where(occi.sum(1) > 0)[0]
    if len(ka):
        res["first_active"] = {"g": float(gi[ka[0]]), **describe_occ(occi[ka[0]], v1)}
    # what carries the long transients below the onset (slow window)
    if b["S"] is not None and b["on"] is not None:
        sel = [p for p in pts if b["S"] * 0.9 <= p["g"] < b["on"]]
        lc = late_composition(sel)
        if lc:
            res["late_slow_window"] = lc
    res["g"] = g.tolist(); res["pr_curve"] = pr.tolist(); res["alive_curve"] = alive.tolist()
    res["tau_reg_curve"] = tau_reg.tolist(); res["deaths"] = dd.tolist(); res["B"] = Bv.tolist()
    return res


def units(x, name):
    if x is None:
        return None
    s = spec[name]
    return {"g": x, "g_lam": x * s["lam"][0] / LAM_LIN, "g_over_glin": x / s["g_lin"]}


summary = {"criteria": __doc__, "spec": {}, "runs": {}}
for n in TRANSFORMS:
    s = spec[n]; m0 = s["modes"][0]
    summary["spec"][n] = {"transform": s["transform"], "c_T": s["c_T"], "c_ratio_T_over_lin": s["c_ratio_T_over_lin"],
                          "lam1": s["lam"][0], "lam": s["lam"][:6], "lam1_clin": s["lam_clin"][0], "g_lin": s["g_lin"],
                          "g_lin_naive": s["g_lin_naive"], "mode0_inv_ipr": m0["inv_ipr"],
                          "mode0_module_share": m0["module_share"], "mode0_AL_share": m0["AL_class_share"],
                          "mode0_region": m0["region"], "mode0_types": dict(list(m0["types"].items())[:5]),
                          "first_AL_mode": s["first_AL_mode"], "first_nonAL_mode": s["first_nonAL_mode"],
                          "AL_over_nonAL": s.get("AL_over_nonAL"),
                          "max_edge_input_per_g": s["max_edge_input_per_g"]}
for n in TRANSFORMS:
    for r in (0, 1):
        paths = sorted(glob.glob(f"{SIM}/{n}_r{r}_p*.npz"))
        if not paths:
            continue
        pf = f"{FINE}/{n}_r{r}.npz"
        fine = os.path.exists(pf) and not COARSE_ONLY
        res = analyze(paths + ([pf] if fine else []), n, boot=True, ignition_from=[pf] if fine else None)
        res["n_files"] = len(paths) + int(fine); res["fine"] = fine
        if fine:
            res["coarse_only"] = analyze(paths, n, boot=False)["bounds"]
        b = res["bounds"]
        res["units"] = {k: units(b[k], n) for k in ("S", "on", "half", "D")}
        res["window_on_over_S"] = b["on"] / b["S"] if (b["on"] and b["S"]) else None
        res["D_over_on"] = b["D"] / b["on"] if (b["D"] and b["on"]) else None
        res["D_not_reached_up_to"] = None if b["D"] is not None else res["g"][-1]
        summary["runs"][f"{n}_r{r}"] = res
        ig = res.get("ignition", {})
        u = res["units"]["on"] or {}
        print(f"{n:7s} r={r}: S {b['S']} on {b['on']} (on/g_lin {u.get('g_over_glin')}, g_lam {u.get('g_lam')}) D {b['D']} | "
              f"ign g {ig.get('g')} module {ig.get('module_share')} MB {ig.get('MB_share')} PR {ig.get('PR_neurons')} "
              f"cos {ig.get('cos_v1')} region {[(k, round(v, 3)) for k, v in list(ig.get('region', {}).items())[:3]]} | late "
              f"{ {k: round(v, 3) for k, v in list(res.get('late_slow_window', {}).get('share', {}).items())[:4]} }", flush=True)


summary["causal"] = {}
for kind in ("noAL", "ALonly"):
    for n in TRANSFORMS:
        for r in (0, 1):
            p = f"{RES}/sim_{kind}/{n}_r{r}.npz"
            key = f"{n}_r{r}"
            if COARSE_ONLY or not os.path.exists(p) or key not in summary["runs"]:
                continue
            res = analyze([p], n, boot=True)
            on0 = summary["runs"][key]["bounds"]["on"]
            b = res["bounds"]
            res["on_intact"] = on0
            res["on_ratio"] = b["on"] / on0 if (b["on"] and on0) else None
            res["S_ratio"] = b["S"] / summary["runs"][key]["bounds"]["S"] if (b["S"] and summary["runs"][key]["bounds"]["S"]) else None
            res["not_sustained_up_to"] = None if b["on"] is not None else res["g"][-1]
            # survival at the intact onset and at 1.2 x (nearest grid point at or above)
            g = np.array(res["g"]); al = np.array(res["alive_curve"]); tr_ = np.array(res["tau_reg_curve"])
            for f in (1.0, 1.2, 1.5):
                k = int(np.searchsorted(g, f * on0 * (1 - 1e-9))) if on0 else len(g)
                if k < len(g):
                    res[f"at_{f}x_on"] = {"g": float(g[k]), "alive": float(al[k]), "tau_reg": float(tr_[k])}
            summary["causal"][f"{kind}_{key}"] = res
            ig = res.get("ignition", {})
            print(f"{kind:6s} {n:7s} r={r}: on {b['on']} (intact {on0}; ratio {res['on_ratio']}) S {b['S']} | "
                  f"alive at 1.0/1.2/1.5x on {[res.get(f'at_{f}x_on', {}).get('alive') for f in (1.0, 1.2, 1.5)]} | ign "
                  f"{[(k, round(v, 3)) for k, v in list(ig.get('region', {}).items())[:3]]} PR {ig.get('PR_neurons')}", flush=True)

# ignition class (a convention on the two causal ratios; stated with the thresholds):
#   'AL alone'      : AL module alone sustains within 10% of the intact onset AND removing it raises the onset by >= 25%
#   'AL not needed' : removing the AL module moves the onset by < 5%
#   'cooperative'   : everything else (neither the AL alone nor the rest alone sustains near the intact onset)
summary["class_rule"] = "AL alone: ALonly/intact <= 1.10 and noAL/intact >= 1.25; AL not needed: noAL/intact < 1.05; else cooperative"
for key in summary["runs"]:
    cn, ca = summary["causal"].get(f"noAL_{key}"), summary["causal"].get(f"ALonly_{key}")
    if not cn or not ca:
        continue
    rn = cn["on_ratio"] if cn["on_ratio"] is not None else np.inf
    ra = ca["on_ratio"] if ca["on_ratio"] is not None else np.inf
    summary["runs"][key]["ignition_class"] = ("AL not needed" if rn < 1.05 else
                                              "AL alone" if (ra <= 1.10 and rn >= 1.25) else "cooperative")
    summary["runs"][key]["causal_ratios"] = {"noAL": cn["on_ratio"], "ALonly": ca["on_ratio"]}
    print(key, summary["runs"][key]["ignition_class"], round(rn, 3), round(ra, 3))


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


outp = f"{RES}/summary_coarse.json" if COARSE_ONLY else f"{RES}/summary.json"
json.dump(clean(summary), open(outp, "w"), indent=1)
print("saved", outp)


# ------------------------------------------------------------------------------------------------------------ table
def f3(v, d=3):
    return "-" if v is None else f"{v:.{d}g}"


def top(dct, k=3):
    return ", ".join(f"{a or '(untyped)'} {b:.2f}" for a, b in list(dct.items())[:k]) if dct else "-"


if not COARSE_ONLY:
    lines = ["# E4 weight transforms: table generated by code/ald_wt_analyze.py from summary.json", "",
             "Units: g = equal mean in-strength (--wnorm renorm); g_lam = g lam1(T)/lam1(linear) (equal leading eigenvalue).",
             "S = tau 200, on = tau 3000 (sustained onset), D = PR N/4; 95% CI = bootstrap + grid bracket. Ignition = occupancy",
             "at the first sustained g (fine sweep). late = time-weighted activity of live replicas at t >= 100 in the slow window.",
             "noAL / ALonly = onset with the AL module clamped off / with everything except the module clamped off (ratio to intact).",
             "class rule: " + summary["class_rule"], "",
             "## Spectra (excitatory operator, mean in-strength 1)", "",
             "| transform | lam1 | lam1 at linear c | best non-AL mode (region) | AL margin | mode-0 1/IPR | mode-0 AL share | g_lin |",
             "|---|---|---|---|---|---|---|---|"]
    for n in TRANSFORMS:
        s = summary["spec"][n]; na = s["first_nonAL_mode"]
        lines.append(f"| {s['transform']} | {s['lam1']:.3f} | {s['lam1_clin']:.3f} | {na['lam']:.3f} ({na['lead_region']}, mode {na['j']}) | "
                     f"{f3(s['AL_over_nonAL'])} | {s['mode0_inv_ipr']:.0f} | {s['mode0_AL_share']:.2f} | {s['g_lin']:.4f} |")
    lines += ["", "## Simulation boundaries", "",
              "| transform | r | S | on [95% CI] | on/g_lin | on (g_lam) | D | D/on | noAL on ratio | ALonly on ratio |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for n in TRANSFORMS:
        for r in (0, 1):
            R = summary["runs"].get(f"{n}_r{r}")
            if R is None:
                continue
            b = R["bounds"]; u = R["units"]["on"] or {}
            ci = b.get("on_ci")
            cn = summary["causal"].get(f"noAL_{n}_r{r}", {}); ca = summary["causal"].get(f"ALonly_{n}_r{r}", {})
            rn = f3(cn.get("on_ratio")) if cn.get("on_ratio") else (f"> {cn['not_sustained_up_to'] / b['on']:.2f}" if cn.get("not_sustained_up_to") else "-")
            ra = f3(ca.get("on_ratio")) if ca.get("on_ratio") else (f"> {ca['not_sustained_up_to'] / b['on']:.2f}" if ca.get("not_sustained_up_to") else "-")
            Dtxt = f3(b["D"]) if b["D"] is not None else f"> {R['D_not_reached_up_to']:.3g}"
            lines.append(f"| {spec[n]['transform']} | {r} | {f3(b['S'])} | {f3(b['on'])} [{f3(ci[0]) if ci else '-'}, {f3(ci[1]) if ci else '-'}] | "
                         f"{f3(u.get('g_over_glin'))} | {f3(u.get('g_lam'))} | {Dtxt} | {f3(R['D_over_on'])} | {rn} | {ra} |")
    lines += ["", "## Where the first sustained activity sits", "",
              "| transform | r | class | g | top regions (occupancy share) | AL-module share | MB share | PR (neurons) | cos with own v1 | late activity in slow window |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for n in TRANSFORMS:
        for r in (0, 1):
            R = summary["runs"].get(f"{n}_r{r}")
            if R is None or "ignition" not in R:
                continue
            ig = R["ignition"]; lt = R.get("late_slow_window", {}).get("share")
            lines.append(f"| {spec[n]['transform']} | {r} | {R.get('ignition_class', '-')} | {ig['g']:.4g} | {top(ig['region'])} | {ig['module_share']:.2f} | "
                         f"{ig['MB_share']:.2f} | {ig['PR_neurons']:.0f} | {f3(ig.get('cos_v1'), 2)} | {top(lt)} |")
    lines += ["", "## Top cell types at the first sustained point", ""]
    for n in TRANSFORMS:
        for r in (0, 1):
            R = summary["runs"].get(f"{n}_r{r}")
            if R is None or "ignition" not in R:
                continue
            lines.append(f"- {spec[n]['transform']} r={r}: {top(R['ignition']['types'], 8)}")
    open(f"{RES}/table.md", "w").write("\n".join(lines) + "\n")
    print("saved", f"{RES}/table.md")
