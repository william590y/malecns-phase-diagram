"""pd3 module_theory MT4: module occupancy map from the existing simulations (+ the MT3 QMF), CPU.

Data: occ[g, N] (per-neuron activity averaged over replicas in the second half, dead replicas count as 0) of every real-graph
T = 3000 decay run with norm raw, h = 0 (results/pd/decay_r*_real.npz, results/pd2/onset_real_r*.npz, deloc_real_r*.npz).
Per module k of partition P (P0: the 34 modules; P1: the top-30 + rest groups of quotient_P1.npz):
  X_k^sim(g, r) = mean occ over module k. Duplicate g (within 1e-4 relative) across files are pooled weighted by B.
  Also surv2 = fraction of replicas alive, averaged over the recorded times in [T/2, T], and X_k / surv2 ('among survivors',
  approximate because occ is a time average).
QMF: X_k^QMF from results/pd3/module_theory/qmf_r<r>.npz (all-active start of pass 1; time average if the column oscillates).
Activation point of module k (coupled system): g at the maximum of d ln X_k / d ln g (finite differences between neighbouring
  pooled points, only where X_k >= 1e-5 at the upper point); also the maximum of d ln(X_k / X_AL) / d ln g above the AL
  onset ('relative activation', the module's own amplification beyond the AL drive). Because the replica-mean occupancy jumps
  at the sustained onset for EVERY module (survival goes 0 -> 1), the same two quantities are also computed among survivors
  (X_k / surv2, only g with surv2 >= 0.5): 'activation_survivors'. Isolated g_lin,k (MT1) is the reference.
Map: number of modules with X_k >= 1% and >= 10% per (g, r), with identities; per module the first g where X_k >= 1% / 10%
  (sim: among survivors, g with surv2 >= 0.5; QMF: all-active start) = the imperfect, coupled activation point.
Output: results/pd3/modules/module_occ_<P>.json
Usage: python pd3_module_occ.py P0|P1
"""
import glob, json, os, sys
import numpy as np

ROOT = "/MALECNS_ROOT"; P = sys.argv[1]
if P == "P0":
    part = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz", allow_pickle=True)
    grp = part["membership"].astype(int); names = [str(x) for x in part["names"]]
else:
    q = np.load(f"{ROOT}/results/pd3/modules/quotient_{P}.npz", allow_pickle=True)
    grp = q["group"].astype(int); names = [str(x) for x in q["names"]]
M = len(names); N = len(grp)
OH = np.zeros((M, N)); OH[grp, np.arange(N)] = 1; size = OH.sum(1)
tab = json.load(open(f"{ROOT}/results/pd3/modules/modules_table.json"))
glin = {}
if P in tab:
    glin = {r_["name"]: r_["g_lin"] for r_ in tab[P]["rows"]}
AL_ids = [k for k, n in enumerate(names) if n.startswith("AL")]
files = sorted(glob.glob(f"{ROOT}/results/pd/decay_r*_real.npz")) + sorted(glob.glob(f"{ROOT}/results/pd2/onset_real_r*.npz")) + \
    sorted(glob.glob(f"{ROOT}/results/pd2/deloc_real_r*.npz"))
pts = {}; used = []
for f in files:
    d = np.load(f, allow_pickle=True); m = json.loads(str(d["meta"]))
    if not (m["exp"] == "decay" and m["norm"] == "raw" and float(m["h"]) == 0 and abs(float(m["T"]) - 3000) < 1e-6
            and not m.get("null") and "occ" in d.files):
        continue
    r = float(m["r"]); t = d["t"]; half = t >= 1500
    X = (OH @ d["occ"].astype(np.float64).T / size[:, None]).T             # [g, M]
    s2 = d["surv"][:, half, :].mean((1, 2))
    used.append({"file": os.path.relpath(f, ROOT), "r": r, "B": int(m["B"]), "n_g": int(len(d["g"]))})
    for i, g in enumerate(d["g"]):
        pts.setdefault(r, []).append((float(g), X[i], float(s2[i]), int(m["B"])))
sim = {}
for r, lst in sorted(pts.items()):
    lst.sort(key=lambda x: x[0]); out = []
    for g, X, s2, B in lst:
        if out and abs(g / out[-1][0] - 1) < 1e-4:
            g0, X0, s0, B0 = out[-1]; out[-1] = (g0, (X0 * B0 + X * B) / (B0 + B), (s0 * B0 + s2 * B) / (B0 + B), B0 + B)
        else:
            out.append((g, X, s2, B))
    sim[r] = {"g": np.array([o[0] for o in out]), "X": np.array([o[1] for o in out]), "surv2": np.array([o[2] for o in out])}


def activation(g, X, floor=1e-5, mask=None):
    """per module: g at max d lnX/d ln g, and at max d ln(X/X_AL)/d ln g above the AL onset (optionally on a subset of g)"""
    if mask is not None:
        g, X = g[mask], X[mask]
    if len(g) < 3:
        return []
    lg = np.log(g); res = []
    XAL = (X[:, AL_ids] * size[AL_ids]).sum(1) / size[AL_ids].sum()
    on = np.argmax(XAL >= 1e-3) if (XAL >= 1e-3).any() else len(g)
    for k in range(M):
        x = np.maximum(X[:, k], 1e-12); sl = np.diff(np.log(x)) / np.diff(lg); ok = X[1:, k] >= floor
        ga = float(np.sqrt(g[:-1] * g[1:])[ok][np.argmax(sl[ok])]) if ok.any() else None
        smax = float(sl[ok].max()) if ok.any() else None
        rel = np.log(x / np.maximum(XAL, 1e-12)); slr = np.diff(rel) / np.diff(lg); okr = ok & (np.arange(1, len(g)) > on)
        gr = float(np.sqrt(g[:-1] * g[1:])[okr][np.argmax(slr[okr])]) if okr.any() else None
        res.append({"module": names[k], "g_act": ga, "max_slope": smax, "g_act_rel": gr,
                    "max_slope_rel": float(slr[okr].max()) if okr.any() else None, "g_lin_isolated": glin.get(names[k])})
    return res


def crossings(g, X, thr, mask=None):
    """per module: first g where X_k >= thr (linear interpolation in ln g / ln X), None if never"""
    if mask is not None:
        g, X = g[mask], X[mask]
    out = {}
    for k in range(M):
        x = X[:, k]; i = np.where(x >= thr)[0]
        if not len(i):
            out[names[k]] = None; continue
        i = i[0]
        if i == 0:
            out[names[k]] = float(g[0]); continue
        a_, b_ = np.log(max(x[i - 1], 1e-12)), np.log(x[i])
        out[names[k]] = float(np.exp(np.log(g[i - 1]) + (np.log(thr) - a_) / (b_ - a_) * (np.log(g[i]) - np.log(g[i - 1]))))
    return out


def counts(X):
    c1 = (X >= 0.01).sum(1); c10 = (X >= 0.1).sum(1)
    ids1 = [[names[k] for k in np.argsort(-x) if x[k] >= 0.01] for x in X]
    return c1.tolist(), c10.tolist(), ids1


outj = {"partition": P, "modules": names, "sizes": size.tolist(), "files_used": used, "sim": {}, "qmf": {}}
for r, d in sim.items():
    c1, c10, ids = counts(d["X"])
    outj["sim"][f"{r:g}"] = {"g": d["g"].tolist(), "X": d["X"].tolist(), "surv2": d["surv2"].tolist(),
                             "X_among_survivors": (d["X"] / np.maximum(d["surv2"], 1e-12)[:, None]).tolist(),
                             "n_ge_1pct": c1, "n_ge_10pct": c10, "ids_ge_1pct": ids, "activation": activation(d["g"], d["X"]),
                             "activation_survivors": activation(d["g"], d["X"] / np.maximum(d["surv2"], 1e-12)[:, None],
                                                                mask=d["surv2"] >= 0.5),
                             "cross_1pct_survivors": crossings(d["g"], d["X"] / np.maximum(d["surv2"], 1e-12)[:, None], 0.01,
                                                               mask=d["surv2"] >= 0.5),
                             "cross_10pct_survivors": crossings(d["g"], d["X"] / np.maximum(d["surv2"], 1e-12)[:, None], 0.1,
                                                                mask=d["surv2"] >= 0.5)}
for f in sorted(glob.glob(f"{ROOT}/results/pd3/module_theory/qmf_r*.npz")):
    if "smoke" in f:
        continue
    d = np.load(f, allow_pickle=True); m = json.loads(str(d["meta"])); r = float(m["r"])
    X34 = d["p1_tavg_X"][0]                                                   # all-active start, [48, 34] (P0 modules)
    if P == "P0":
        X = X34
    else:                                                                     # P1 groups from the saved all-active vectors
        Xn = d["all_active_final"].astype(np.float64); X = (OH @ Xn.T / size[:, None]).T
    c1, c10, ids = counts(X)
    outj["qmf"][f"{r:g}"] = {"g": d["g1"].tolist(), "X": X.tolist(), "n_ge_1pct": c1, "n_ge_10pct": c10, "ids_ge_1pct": ids,
                             "oscillating": (d["p1_resid"][0] > 1e-5).tolist(), "activation": activation(d["g1"], X),
                             "cross_1pct": crossings(d["g1"], X, 0.01), "cross_10pct": crossings(d["g1"], X, 0.1)}
json.dump(outj, open(f"{ROOT}/results/pd3/modules/module_occ_{P}.json", "w"))
print(f"saved module_occ_{P}.json: sim r = {sorted(sim)}; qmf r = {sorted(outj['qmf'])}")
for r in [0.0, 1.0, 4.0]:
    if r in sim:
        a_ = {x["module"]: x for x in outj["sim"][f"{r:g}"]["activation"]}
        print(f"r={r:g} sim activation (g_act / g_act_rel / isolated g_lin):",
              {k: (a_[k]["g_act"] and round(a_[k]["g_act"], 3), a_[k]["g_act_rel"] and round(a_[k]["g_act_rel"], 3),
                   a_[k]["g_lin_isolated"] and round(a_[k]["g_lin_isolated"], 3)) for k in names[:8] + ["rest"]})
