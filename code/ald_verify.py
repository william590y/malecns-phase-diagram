"""Adversarial verification of the AL-defect synthesis (2026-11-09, verify lane).

Recomputes the load-bearing numbers of the synthesis from the RAW simulator outputs with an independent implementation
(own censored-MLE lifetimes, own crossing rule = running maximum instead of isotonic regression, own pooling), plus
robustness checks with two other onset definitions:
   on_tau  : mean lifetime (censored exponential MLE, T_c = 3000) crosses 3000      (the lanes' definition)
   on_half : fraction of replicas alive at T crosses 1/2                            (no lifetime model)
   on_late : lifetime MLE from t0 = 100 on, among replicas alive at t0, crosses 3000 (drops the start transient)
Crossings are interpolated linearly in ln g between the bracketing grid points of a running-maximum curve.
Also: E1 allowed range from raw drive runs, palp leak, E3 removal-protocol grid brackets, E2 type-ensemble counts from the
drawn sign vectors, PR / module share at ignition, D interpolation, mode counts above threshold, replication runs
(results/aldef/verify/sim/, code/ald_verify_simcheck.py). Writes results/aldef/verify/verify.json and prints a table.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

ROOT = "/MALECNS_ROOT"
R = f"{ROOT}/results/aldef"
OUT = f"{R}/verify"
os.makedirs(OUT, exist_ok=True)
TC, T0 = 3000.0, 100.0
N = 167216
AEFF = (1 - np.exp(-0.2)) / 0.2
MOD = np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)
MODIDX = MOD["idx"]; modmask = np.zeros(N, bool); modmask[MODIDX] = True
LLN1 = MODIDX[np.array([str(t) == "lLN1_bc" for t in MOD["types"]])]
res = {}


def load(path):
    d = np.load(path)
    t, g, surv = d["t"], d["g"], d["surv"]
    w = t <= TC + 1e-6
    tw = t[w]; s = surv[:, w, :]
    dead = s == 0
    died = dead.any(1)
    ext = np.where(died, tw[np.argmax(dead, 1)], tw.max())
    out = []
    for i in range(len(g)):
        out.append({"g": float(g[i]), "ext": ext[i].astype(float), "died": died[i], "B": int(s.shape[2]),
                    "occ": d["occ"][i].astype(np.float64)})
    return out


def pool(paths):
    pts = sorted([p for f in paths for p in load(f)], key=lambda x: x["g"])
    m = []
    for p in pts:
        if m and abs(p["g"] / m[-1]["g"] - 1) < 1e-4:
            q = m[-1]
            q["occ"] = (q["occ"] * q["B"] + p["occ"] * p["B"]) / (q["B"] + p["B"])
            q["ext"] = np.r_[q["ext"], p["ext"]]; q["died"] = np.r_[q["died"], p["died"]]; q["B"] += p["B"]
        else:
            m.append(dict(p))
    return m


def cross(g, y, thr):
    y = np.maximum.accumulate(np.asarray(y, float))
    k = np.where(y >= thr)[0]
    if len(k) == 0:
        return None
    k = k[0]
    if k == 0:
        return float(g[0])
    f = (thr - y[k - 1]) / (y[k] - y[k - 1]) if y[k] > y[k - 1] else 0.5
    return float(np.exp(np.log(g[k - 1]) + f * (np.log(g[k]) - np.log(g[k - 1]))))


def onsets(pts):
    g = np.array([p["g"] for p in pts])
    tau = np.array([p["ext"].sum() / max(p["died"].sum(), 0.5) for p in pts])
    alive = np.array([1 - p["died"].mean() for p in pts])
    late = []
    for p in pts:
        a = p["ext"] > T0
        late.append(((p["ext"][a] - T0).sum() / max((p["died"] & a).sum(), 0.5)) if a.any() else 0.0)
    late = np.array(late)
    o = {"on_tau": cross(g, np.log(tau), np.log(TC)), "S_tau": cross(g, np.log(tau), np.log(200.0)),
         "on_half": cross(g, alive, 0.5), "on_late": cross(g, np.log(np.maximum(late, 1e-9)), np.log(TC))}
    return o, g, tau, alive


def describe(occ):
    s = occ.sum()
    if s <= 0:
        return None
    p = occ / s
    return {"PR_neurons": float(1 / (p ** 2).sum()), "module_share": float(p[modmask].sum())}


def first_at_or_above(pts, gx):
    for p in pts:
        if p["g"] >= gx * (1 - 1e-9):
            return p
    return None


def e3_files(arm):
    fs = []
    for f in sorted(glob.glob(f"{R}/depression/index_*.json")):
        fs += [e["out"] for e in json.load(open(f)) if e["arm"] == arm]
    return [f for f in fs if os.path.exists(f)]


# ------------------------------------------------------------------------------------------------ E2 sign conditions
claims = {"base_r0": 0.2633, "base_r1": 0.3476, "base_r3": 0.4579, "lLN1bc_zero_r0": 0.4483, "lLN1bc_zero_r1": 1.0035,
          "lLN1bc_inh_r1": 1.0361, "lLN1bc_lLN2_inh_r0": 0.5662, "lLN1bc_lLN2_inh_r1": 1.0416}
e2 = {}
for k, c in claims.items():
    paths = [f"{R}/signs/sim/{k}.npz"] + ([f"{R}/signs/sim_fine/{k}.npz"] if os.path.exists(f"{R}/signs/sim_fine/{k}.npz") else [])
    pts = pool(paths)
    o, g, tau, alive = onsets(pts)
    ig = first_at_or_above(pts, o["on_tau"])
    e2[k] = {**o, "claimed_on": c, "rel_diff": o["on_tau"] / c - 1, "ignition_g": ig["g"], **(describe(ig["occ"]) or {})}
res["E2"] = e2

# ------------------------------------------------------------------------------------------------ E3 depression arms
claims3 = {"base_r1": 0.3462, "base_r0": 0.2618, "sp_NW16_r1": 1.3838, "sp_KW09_r1": 1.6627, "sp_N15_r1": 1.1397,
           "sp_NW16_r0": 1.2057, "spg_lLN1bc_r1": 0.9442, "spg_AL_r1": 1.0114, "spg_notAL_r1": 0.3421, "spg_exc_r1": 1.5688,
           "iso_base_ALonly_r1": 0.3368, "iso_base_noAL_r1": 1.0415, "iso_sp_NW16_ALonly_r1": 3.0042,
           "iso_sp_NW16_noAL_r1": 1.4815, "iso_sp_KW09_ALonly_r1": 4.5284, "iso_sp_KW09_noAL_r1": 1.7359,
           "sp_U25_tau3_r1": 0.396, "sp_U25_tau10_r1": 0.550, "sp_U25_tau30_r1": 0.872, "sp_slow_r1": 0.8078}
e3 = {}; P3 = {}
for k, c in claims3.items():
    pts = pool(e3_files(k)); P3[k] = pts
    o, g, tau, alive = onsets(pts)
    fs = next((p for p in pts if 1 - p["died"].mean() >= 0.5), None)
    d = {**o, "claimed_on": c, "rel_diff": o["on_tau"] / c - 1}
    if fs is not None:
        d.update({"first_half_alive_g": fs["g"], **(describe(fs["occ"]) or {})})
    sel = (g >= 0.163) & (g <= 1.0)
    d["tau_range_g0.163_to_1"] = [float(tau[sel].min()), float(tau[sel].max())] if sel.any() else None
    d["tau_at_g_le_0.16"] = float(tau[g <= 0.165].max()) if (g <= 0.165).any() else None
    e3[k] = d
res["E3"] = e3
b1, b0 = e3["base_r1"], e3["base_r0"]
rat = {}
for defn in ("on_tau", "on_half", "on_late"):
    rat[defn] = {"NW16/base_r1": e3["sp_NW16_r1"][defn] / b1[defn], "KW09/base_r1": e3["sp_KW09_r1"][defn] / b1[defn],
                 "N15/base_r1": e3["sp_N15_r1"][defn] / b1[defn], "NW16/base_r0": e3["sp_NW16_r0"][defn] / b0[defn],
                 "lLN1bc_dep/base_r1": e3["spg_lLN1bc_r1"][defn] / b1[defn], "AL_dep/base_r1": e3["spg_AL_r1"][defn] / b1[defn],
                 "sign0/base_r1(E2)": e2["lLN1bc_zero_r1"][defn] / e2["base_r1"][defn],
                 "iii/base_r1(E2)": e2["lLN1bc_lLN2_inh_r1"][defn] / e2["base_r1"][defn],
                 "noAL/base_r1": e3["iso_base_noAL_r1"][defn] / b1[defn],
                 "ALonly_NW16/noAL_NW16": e3["iso_sp_NW16_ALonly_r1"][defn] / e3["iso_sp_NW16_noAL_r1"][defn],
                 "tau30/base_r1": e3["sp_U25_tau30_r1"][defn] / b1[defn]}
res["ratios_by_definition"] = rat

# D (PR = N/4) for sp_NW16_r0 and base_r0 (sustained points only), PR at g = 8 for r = 1
def D_of(pts, on):
    gs, prs = [], []
    for p in pts:
        if p["g"] >= on and p["died"].mean() < 0.1:
            dsc = describe(p["occ"]); gs.append(p["g"]); prs.append(dsc["PR_neurons"] / N if dsc else 0)
    return cross(np.array(gs), np.array(prs), 0.25), list(zip(gs, prs))
Dn, curve = D_of(P3["sp_NW16_r0"], e3["sp_NW16_r0"]["on_tau"])
Db, _ = D_of(P3["base_r0"], e3["base_r0"]["on_tau"])
pr8 = {k: describe(next(p for p in P3[k] if abs(p["g"] - 8) < 1e-6)["occ"])["PR_neurons"] / N for k in ("base_r1", "sp_NW16_r1")}
res["D"] = {"sp_NW16_r0": Dn, "base_r0": Db, "ratio": Dn / Db if Dn and Db else None, "PR_over_N_at_g8_r1": pr8,
            "sp_NW16_r0_PR_curve_tail": curve[-4:]}

# ------------------------------------------------------------------------------------------------ E1 raw drive runs
def e1_rows(proto, r, lev):
    rows = []
    for f in sorted(glob.glob(f"{R}/drive/runs/{proto}_r{r}_o{lev}_c*.npz")):
        d = np.load(f); nm = list(d["reg_names"]); ix = {n: i for i, n in enumerate(nm)}
        wm, wb, last = d["win_mean"], d["win_bounds"], d["reg_last"]
        w100 = [w for w in range(len(wb)) if abs(wb[w, 0] - 1600) < 1e-6 and abs(wb[w, 1] - 1800) < 1e-6][0]
        for gi, g in enumerate(d["g"]):
            orn, pn = wm[gi, 0, ix["ORN"]].mean(), wm[gi, 0, ix["uPN_ant"]].mean()
            rows.append({"g": float(g), "G": float(pn / orn), "resid100": float(wm[gi, w100, ix["uPN_ant"]].mean() / max(pn, 1e-12)),
                         "p_alive": float((last[gi, ix["ALsys"]] >= 4500 - 1e-3).mean())})
    return sorted(rows, key=lambda x: x["g"])


def lin_cross(xs, ys, thr):
    for a, b, ya, yb in zip(xs, xs[1:], ys, ys[1:]):
        if ya < thr <= yb:
            return a + (thr - ya) / (yb - ya) * (b - a)
    return None
e1 = {}
for r in (0, 1):
    for lev in ("0.01", "0.04", "0.10"):
        rw = [x for x in e1_rows("all", r, lev) if x["g"] > 0]
        gs = [x["g"] for x in rw]
        lo = lin_cross(gs, [x["G"] for x in rw], 0.29)
        ups = [lin_cross(gs, [x["G"] for x in rw], 1.45), lin_cross(gs, [x["resid100"] for x in rw], 0.1),
               lin_cross(gs, [x["p_alive"] for x in rw], 0.05 + 1e-9)]
        e1[f"r{r}_o{lev}"] = {"lower": lo, "upper": min(u for u in ups if u is not None), "upper_parts_G_resid_palive": ups}
    ant = {x["g"]: x["resid100"] for x in e1_rows("ant", 0, "0.04")}
    e1[f"palp_leak_r{r}_o0.04"] = {g: ant.get(g) for g in (0.1, 0.12, 0.14, 0.16, 0.18)} if r == 0 else None
leaks = []
for lev in ("0.01", "0.02", "0.04", "0.07", "0.10"):
    a = {x["g"]: x["resid100"] for x in e1_rows("ant", 0, lev)}
    leaks += [a[0.1], a[0.16]]
e1["palp_leak_r0_g0.10_0.16_all_levels"] = [min(leaks), max(leaks)]
res["E1"] = e1

# ------------------------------------------------------------------------------------------------ E3 removal grid brackets
s3 = json.load(open(f"{R}/depression/summary.json"))
rm = {}
for arm, on in (("rm_sp_NW16_r1", e3["sp_NW16_r1"]["on_tau"]), ("rm_sp_KW09_r1", e3["sp_KW09_r1"]["on_tau"]),
                ("rm_spg_exc_r1", e3["spg_exc_r1"]["on_tau"]), ("rm_base_r1", b1["on_tau"])):
    rows = s3["driven"][arm]["rows"]
    ok = [x["g"] for x in rows if x["allowed"]]
    above = [x["g"] for x in rows if x["g"] > max(ok)]
    up_interp = s3["driven"][arm]["allowed_interval"][1]
    rm[arm] = {"last_allowed_grid_g": max(ok), "first_failing_grid_g": min(above), "interp_upper": up_interp,
               "onset": on, "upper_over_onset_interp": up_interp / on,
               "upper_over_onset_bracket": [max(ok) / on, min(above) / on]}
res["E3_removal_brackets"] = rm

# driven-window (500-1500) fluctuations of the AL module under ORN drive, with and without depression: is the depressed,
# linearly unstable (g > g_lin) but data-allowed regime bursty?  chi = n_module * mean_b var_t(rho_mod) / mean rho_mod
# (= 1 for independent Poisson-like units; E1's chi_time), and CV_t of uPN_ant activity (mean over replicas).
flu = {}
for arm in ("rm_base_r1", "rm_sp_NW16_r1", "rm_sp_KW09_r1"):
    rows_ = []
    for f in sorted(glob.glob(f"{R}/depression/rm/{arm}_t*.npz")):
        d = np.load(f); nm = [str(x) for x in d["reg_names"]]; ix = {n: i for i, n in enumerate(nm)}
        wm, w2, sz = d["win_mean"], d["win_m2"], d["reg_sizes"]
        for gi, g in enumerate(d["g"]):
            m, m2 = wm[gi, 0, ix["AL_module"]].astype(float), w2[gi, 0, ix["AL_module"]].astype(float)
            pm, p2 = wm[gi, 0, ix["uPN_ant"]].astype(float), w2[gi, 0, ix["uPN_ant"]].astype(float)
            chi = float(sz[ix["AL_module"]] * np.mean(np.maximum(m2 - m ** 2, 0)) / max(m.mean(), 1e-12))
            cv = float(np.mean(np.sqrt(np.maximum(p2 - pm ** 2, 0)) / np.maximum(pm, 1e-12)))
            rows_.append((float(g), chi, cv))
    flu[arm] = sorted(rows_)
res["E3_removal_driven_fluct"] = flu

# ------------------------------------------------------------------------------------------------ E4 causal ratios
e4 = {}
for tr in ("linear", "binary"):
    for r in (0, 1):
        intact = pool(sorted(glob.glob(f"{R}/weights/sim/{tr}_r{r}_p*.npz")) + [f"{R}/weights/sim_fine/{tr}_r{r}.npz"])
        oi = onsets(intact)[0]["on_tau"]
        on_noal = onsets(pool([f"{R}/weights/sim_noAL/{tr}_r{r}.npz"]))[0]["on_tau"]
        on_alo = onsets(pool([f"{R}/weights/sim_ALonly/{tr}_r{r}.npz"]))[0]["on_tau"]
        e4[f"{tr}_r{r}"] = {"intact": oi, "noAL_ratio": on_noal / oi if on_noal else None, "ALonly_ratio": on_alo / oi if on_alo else None}
res["E4"] = e4

# ------------------------------------------------------------------------------------------------ E2 type ensemble
ens = {}
for r in (0, 1, 3):
    n_exc = n_top_mod = 0; draws = []
    for f in sorted(glob.glob(f"{R}/signs/sim/ens_type_d*_r{r}.npz")):
        d = np.load(f)
        s = d["signs_ens"][0]
        allx = bool((s[LLN1] == 1).all())
        draws.append(allx)
    ens[f"r{r}"] = {"n_draws": len(draws), "n_all30_lLN1bc_exc": int(sum(draws))}
res["E2_ens_counts"] = ens
res["n_lLN1bc"] = int(len(LLN1))

# ------------------------------------------------------------------------------------------------ spectra consistency
sp = json.load(open(f"{R}/signs/summary.json"))["spec"]
res["spectra"] = {k: {"lam1": v["lam"][0], "g_lin": v["g_lin"], "g_lin_naive": v["g_lin_naive"], "aeff_over_lam1": AEFF / v["lam"][0]}
                  for k, v in sp.items() if "lam" in v}
res["spectra_ens_type_ignore"] = {k: sp["ens_type_ignore"][k] for k in ("K", "P_lLN1_bc(+1,-1,0)", "frac_lLN1bc_all_exc", "frac_mode0_in_AL")}
tail = np.load(f"{ROOT}/results/pd/tail_real_r0_dir.npz")
lam, ALs = tail["lam"], tail["AL"]
res["modes_past_naive_threshold"] = {f"g={gx}": {"n": int((lam >= AEFF / gx).sum()), "n_nonAL(AL<0.5)": int(((lam >= AEFF / gx) & (ALs < 0.5)).sum())}
                                     for gx in (0.3476, 1.0, 1.3838, 1.6627)}

# ------------------------------------------------------------------------------------------------ replications (verify/sim)
V = f"{OUT}/sim"
rep = {}
if os.path.exists(f"{V}/bitid.json"):
    rep["bitid"] = json.load(open(f"{V}/bitid.json"))["all_pass"]
def tau_ci(p):
    d, E = int(p["died"].sum()), float(p["ext"].sum())
    lo = 2 * E / stats.chi2.ppf(0.975, 2 * d + 2); hi = 2 * E / stats.chi2.ppf(0.025, 2 * d) if d > 0 else float("inf")
    return [E / max(d, 0.5), lo, hi, d, p["B"]]
for tag in ("eng_spmm", "eng_push", "rep_sp_NW16_r1", "rep_lLN1bc_zero_r1", "rep_iso_NW16_ALonly_r1", "rep_iso_NW16_noAL_r1"):
    f = f"{V}/{tag}.npz"
    if os.path.exists(f):
        pts = load(f)
        rep[tag] = {"points": [{"g": p["g"], "alive": float(1 - p["died"].mean()), "tau_mle_lo_hi_d_B": tau_ci(p)} for p in pts],
                    "onsets": onsets(pts)[0]}
ref = {"rep_sp_NW16_r1": "sp_NW16_r1", "rep_iso_NW16_ALonly_r1": "iso_sp_NW16_ALonly_r1", "rep_iso_NW16_noAL_r1": "iso_sp_NW16_noAL_r1"}
for tag, arm in ref.items():
    if tag in rep:
        rep[tag]["lane_alive_at_g"] = {p["g"]: float(1 - p["died"].mean()) for p in P3[arm]
                                       if any(abs(p["g"] - q["g"]) < 1e-3 for q in rep[tag]["points"])}
        rep[tag]["lane_on"] = e3[arm]["on_tau"]
if "rep_lLN1bc_zero_r1" in rep:
    rep["rep_lLN1bc_zero_r1"]["lane_on"] = e2["lLN1bc_zero_r1"]["on_tau"]
res["replications"] = rep


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o
json.dump(clean(res), open(f"{OUT}/verify.json", "w"), indent=1)
print(json.dumps(clean(res), indent=1)[:20000])
