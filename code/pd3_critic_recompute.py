#!/usr/bin/env python
"""pd3 completeness critic (workflow wf_98fb990d-4fb, 2026-11-09): independent recomputation of key REPORT numbers
from RAW outputs. Imports no project code (own PAVA, crossing rule, censored-exponential MLE, Perron root).

Recomputed (REPORT section in brackets):
  K1  g_lin = 0.1632 from the connectome cache: lambda_PF(B(g)) = alpha_eff, B_ij = f(g E_ij)           [3.2, 4 row 4]
  K2  tau = 200 / tau = 3000 crossings at r = 0 and r = 1 from the raw pooled decay runs                [3.1, 3.2]
  K3  D (PR = N/4 among sustained points) at r = 0 from the raw occupancies                             [3.1]
  K4  FSS: null x_on seed means at N = 2.5k and 167k (r = 0, 1), real-subsample spread vs the full brain  [6]
  K5  M1: late lobe asymmetry range over starts with >= 10 survivors, criterion c1, 48 points            [3.1, 5]
  K6  island-1 1% crossing among survivors (r = 0, 1, 6) from the raw occupancies                       [3.1]
Writes results/pd3/critic_recompute.json.  CPU only.
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np

ROOT = "/MALECNS_ROOT"
CACHE = "/MALECNS_CACHE"
OUT = {}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def pava(y, w):
    """weighted isotonic (non-decreasing) regression, own implementation"""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            ww = wts[-2] + wts[-1]; cc = cnt[-2] + cnt[-1]
            vals.pop(); wts.pop(); cnt.pop()
            vals[-1], wts[-1], cnt[-1] = v, ww, cc
    out = []
    for v, c in zip(vals, cnt):
        out += [v] * c
    return np.array(out)


def first_cross(x, y, thr):
    """first crossing of y >= thr, linear interpolation in (ln x, y)"""
    i = np.where(y >= thr)[0]
    if not len(i):
        return None
    i = i[0]
    if i == 0:
        return float(x[0])
    lx0, lx1 = np.log(x[i - 1]), np.log(x[i])
    return float(np.exp(lx0 + (thr - y[i - 1]) / (y[i] - y[i - 1]) * (lx1 - lx0)))


def tau_cross(g, E, d, thr):
    lt = np.log(E / np.maximum(d, 0.5))
    iso = pava(lt, d + 0.5)
    return first_cross(g, iso, np.log(thr)), iso


# ------------------------------------------------------------------------------------------------ K2, K3: raw decay runs
pc = json.load(open(f"{ROOT}/results/analysis2/phase_complete.json"))
GLIN_PC = pc["g_lin"]["real"]
N = 167216


def pooled_points(r):
    pts = {}
    for inp in pc["inputs"]["simulation"]:
        if inp["kind"] != "real" or abs(inp["r"] - r) > 1e-9:
            continue
        d = np.load(f"{ROOT}/results/{inp['file']}", allow_pickle=True)
        t = d["t"]; Tc = t[t <= 3000.0 + 1e-9].max(); surv = d["surv"]          # [g, t, B]
        keep = t <= Tc + 1e-9
        for i, g in enumerate(d["g"]):
            s = surv[i][keep]                                                      # [t, B]
            dead = (s == 0)
            first = np.where(dead.any(0), dead.argmax(0), -1)
            ext = np.where(first >= 0, t[keep][np.maximum(first, 0)], Tc)
            E = float(ext.sum()); dd = int((first >= 0).sum())
            occ = d["occ"][i].astype(np.float64) if (abs(inp["T"] - 3000.0) < 1e-6 and "occ" in d.files) else None
            key = None
            for k in pts:
                if abs(g / k - 1) < 1e-4:
                    key = k
            if key is None:
                pts[float(g)] = {"E": 0.0, "d": 0, "B": 0, "occ": None, "Bocc": 0}
                key = float(g)
            p = pts[key]; p["E"] += E; p["d"] += dd; p["B"] += s.shape[1]
            if occ is not None:
                p["occ"] = occ * s.shape[1] if p["occ"] is None else p["occ"] + occ * s.shape[1]
                p["Bocc"] += s.shape[1]
    gs = np.array(sorted(pts))
    return gs, [pts[g] for g in gs]


for r in (0.0, 1.0):
    gs, P = pooled_points(r)
    E = np.array([p["E"] for p in P]); dd = np.array([p["d"] for p in P], float)
    S, _ = tau_cross(gs, E, dd, 200.0)
    ON, iso = tau_cross(gs, E, dd, 3000.0)
    rec = {"n_points": int(len(gs)), "S_tau200": S, "on_tau3000": ON,
           "published_S": pc["sim"]["real"][f"{r:g}"]["S"]["g"], "published_on": pc["sim"]["real"][f"{r:g}"]["on"]["g"]}
    if r == 0.0:
        sus = iso >= np.log(3000.0)
        gD, PR = [], []
        for g, p, s in zip(gs, P, sus):
            if s and p["occ"] is not None:
                o = p["occ"] / p["Bocc"]
                PR.append((o.sum() ** 2 / max((o ** 2).sum(), 1e-300)) / N); gD.append(g)
        gD = np.array(gD); PRi = pava(np.array(PR), np.ones(len(PR)))
        rec["D_PR_N4"] = first_cross(gD, PRi, 0.25)
        rec["published_D"] = pc["sim"]["real"]["0"]["D"]["g"]
    OUT[f"K2K3_r{r:g}"] = rec
    log("r", r, rec)

# ------------------------------------------------------------------------------------------------ K4: FSS raw ext_step
fss = {}
for f in sorted(glob.glob(f"{ROOT}/results/pd3/redraw_fss/fss_*.npz")):
    m = re.search(r"fss_(?:fine_)?(cnull|realsub)_N(\d+)_s(\d+)_r(\d+)\.npz", f)
    kind, n, s, r = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    d = np.load(f, allow_pickle=True)
    T, dt = float(d["T"]), float(d["dt"]); e = d["ext_step"]
    tt = np.where(e >= 0, e * dt, T); E = tt.sum(1); dd = (e >= 0).sum(1)
    k = (kind, n, s, r)
    fss.setdefault(k, {"x": [], "E": [], "d": []})
    fss[k]["x"] += list(d["x"]); fss[k]["E"] += list(E); fss[k]["d"] += list(dd)
xon = {}
for k, v in fss.items():
    o = np.argsort(v["x"]); x = np.array(v["x"])[o]; E = np.array(v["E"], float)[o]; dd = np.array(v["d"], float)[o]
    # pool exact duplicates of x (fine + coarse on the same graph)
    xu, inv = np.unique(np.round(x, 9), return_inverse=True)
    Eu = np.bincount(inv, E); du = np.bincount(inv, dd)
    xon[k], _ = tau_cross(xu, Eu, du, 3000.0)
res4 = {}
for kind in ("cnull", "realsub"):
    for r in (0, 1):
        byN = {}
        for (kk, n, s, rr), v in xon.items():
            if kk == kind and rr == r and v is not None:
                byN.setdefault(n, []).append(v)
        res4[f"{kind}_r{r}"] = {str(n): {"seed_mean": float(np.mean(v)), "per_seed": [round(x_, 4) for x_ in v]}
                                for n, v in sorted(byN.items())}
full = {0: pc["sim"]["real"]["0"]["on"]["g"] / GLIN_PC, 1: pc["sim"]["real"]["1"]["on"]["g"] / GLIN_PC}
for r in (0, 1):
    allv = [x_ for n, v in res4[f"realsub_r{r}"].items() for x_ in v["per_seed"]]
    res4[f"realsub_r{r}_max_abs_rel_dev_from_full"] = float(max(abs(a / full[r] - 1) for a in allv))
OUT["K4_fss"] = res4
log("K4", json.dumps({k: (v if not isinstance(v, dict) else {n: round(w['seed_mean'], 4) for n, w in v.items()})
                      for k, v in res4.items()}))

# ------------------------------------------------------------------------------------------------ K5: M1 late asymmetry
late_means, c1_max, npts, tsym_meds = [], 0.0, 0, []
for f in sorted(glob.glob(f"{ROOT}/results/pd3/multistab/al_sym_r*.npz")):
    d = np.load(f, allow_pickle=True); meta = json.loads(str(d["meta"]))
    B = int(meta["B"]); starts = meta["starts"]; modc = d["modc"]; ext = d["ext_step"]   # [G, rec, 34, 4B], [G, 4B]
    nrec = modc.shape[1]; tt = np.arange(nrec) * float(d["rec_dt"])
    win = (tt >= 500) & (tt <= 1000)
    for gi in range(modc.shape[0]):
        npts += 1
        nL = modc[gi][:, 0, :].astype(float); nR = modc[gi][:, 1, :].astype(float)   # [rec, 4B]
        fL, fR = nL / 248.0, nR / 275.0; tot = fL + fR
        A = np.where(tot > 0, (fL - fR) / np.where(tot > 0, tot, 1), np.nan)
        alive_T = ext[gi] < 0
        alact = (tot[win] > 0).mean(0) >= 0.5
        surv = alive_T & alact
        for si, st in enumerate(starts):
            cols = np.arange(si * B, (si + 1) * B)
            sv = cols[surv[cols]]
            if len(sv) >= 10:
                lm = np.nanmean(A[win][:, sv], 0)
                late_means.append(float(np.mean(lm)))
                if st == "AL-L":
                    c1_max = max(c1_max, float((lm > 0.6).mean()))
res5 = {"n_points": npts, "late_mean_A_range_starts_ge10_survivors": [min(late_means), max(late_means)],
        "c1_max_fraction_ALL_start_keep_A_gt_0.6": c1_max}
OUT["K5_M1"] = res5
log("K5", res5)

# ------------------------------------------------------------------------------------------------ K6: island-1 1% crossing
pP0 = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz", allow_pickle=True)
names = list(pP0["names"]); memb = pP0["membership"]
isl1 = memb == names.index("isl1")
pts6 = {}
files = sorted(glob.glob(f"{ROOT}/results/pd/decay_r*_real.npz")) + sorted(glob.glob(f"{ROOT}/results/pd2/onset_real_r*.npz")) + \
    sorted(glob.glob(f"{ROOT}/results/pd2/deloc_real_r*.npz"))
for f in files:
    d = np.load(f, allow_pickle=True); m = json.loads(str(d["meta"]))
    if not (m.get("exp") == "decay" and m.get("norm") == "raw" and float(m["h"]) == 0 and abs(float(m["T"]) - 3000) < 1e-6
            and not m.get("null") and "occ" in d.files):
        continue
    r = float(m["r"])
    if r not in (0.0, 1.0, 6.0):
        continue
    half = d["t"] >= 1500; B = int(m["B"])
    for i, g in enumerate(d["g"]):
        X = float(d["occ"][i][isl1].astype(np.float64).mean()); s2 = float(d["surv"][i][half].mean())
        L = pts6.setdefault(r, [])
        hit = [j for j, q in enumerate(L) if abs(q[0] / g - 1) < 1e-4]
        if hit:
            j = hit[0]; g0, X0, s0, B0 = L[j]; L[j] = (g0, (X0 * B0 + X * B) / (B0 + B), (s0 * B0 + s2 * B) / (B0 + B), B0 + B)
        else:
            L.append((float(g), X, s2, B))
res6 = {}
for r, L in sorted(pts6.items()):
    L.sort(); g = np.array([q[0] for q in L]); X = np.array([q[1] for q in L]); s2 = np.array([q[2] for q in L])
    mk = s2 >= 0.5; g, Xs = g[mk], X[mk] / s2[mk]
    i = np.where(Xs >= 0.01)[0]
    if len(i) and i[0] > 0:
        j = i[0]; a, b = np.log(max(Xs[j - 1], 1e-12)), np.log(Xs[j])
        gc = float(np.exp(np.log(g[j - 1]) + (np.log(0.01) - a) / (b - a) * (np.log(g[j]) - np.log(g[j - 1]))))
    else:
        gc = float(g[i[0]]) if len(i) else None
    res6[f"{r:g}"] = gc
OUT["K6_isl1_1pct"] = res6
log("K6", res6)

# ------------------------------------------------------------------------------------------------ K1: g_lin from the cache
if "--no-glin" not in sys.argv:
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    from scipy.optimize import brentq
    pre = np.load(f"{CACHE}/pre.npy").astype(np.int64); post = np.load(f"{CACHE}/post.npy").astype(np.int64)
    w = np.load(f"{CACHE}/weight.npy").astype(np.float64); sign = np.load(f"{CACHE}/sign.npy")
    assert len(sign) == N
    ins = np.bincount(post, weights=w, minlength=N); c = 1.0 / ins[ins > 0].mean()
    ex = sign[pre] > 0
    E = sp.coo_matrix((c * w[ex], (post[ex], pre[ex])), shape=(N, N)).tocsr(); E.sum_duplicates()
    dt = 0.2; a_eff = (1 - np.exp(-dt)) / dt
    data0 = E.data.copy()

    def lam(g):
        B = sp.csr_matrix(((1 - np.exp(-np.tanh(g * data0) * dt)) / dt, E.indices, E.indptr), shape=(N, N))
        return float(sla.eigs(B, k=1, which="LR", tol=1e-10, maxiter=20000)[0][0].real)

    lamE = float(sla.eigs(E, k=1, which="LR", tol=1e-10)[0][0].real)
    glo = a_eff / lamE
    log("1/c =", 1 / c, "lambda_PF(E) =", lamE, "a_eff =", a_eff, "g_weak =", glo)
    gl = brentq(lambda g: lam(g) - a_eff, glo, 1.5 * glo, xtol=1e-7)
    OUT["K1_glin"] = {"g_lin": gl, "inv_c": 1 / c, "lambda_PF_E": lamE, "alpha_eff": a_eff, "published": GLIN_PC}
    log("K1", OUT["K1_glin"])

with open(f"{ROOT}/results/pd3/critic_recompute.json", "w") as fh:
    json.dump({"generated_by": "code/pd3_critic_recompute.py", "date": "2026-11-09", "results": OUT}, fh, indent=1)
log("wrote results/pd3/critic_recompute.json")
