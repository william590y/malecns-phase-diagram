"""m2_modules_phase.py -- lane m2/modules Part A: module phase diagram from EXISTING occupancies (CPU, no simulation).

plan.md section 3. Data (read-only):
  (a) h = 0, fully active start, T = 3000, occ = replica-mean occupancy over t > 1500 (dead replicas count 0):
      decay lane  results/m2/decay/sim/s1_*, s2_* (index_s1 / index_s2*.json; real graph; models M2, M2s, M1, M1s, variants)
      pd3 M1      results/pd/decay_r*_real.npz, results/pd2/onset_real_r*.npz, deloc_real_r*.npz (pd3_module_occ.py's filter)
  (b) ORN 0.04 driven, occ over the driven window 500-1500 (all replicas): driven lane runs A1 (M2), A2 (M1), A4 (M2s)
Per module k of partition P (P0 34, P1 50 + small, P2 >= 50 groups + small; results/m2/modules/groups):
  X_k = mean occupancy over k; duplicate g pooled weighted by B. (a): among survivors X_k / surv2 (surv2 = fraction of replicas
  alive averaged over the records with t >= 1500), only where surv2 >= 0.5 (pd3 MT4). (b): all replicas.
  on = X_k >= theta (0.01 primary = pd3, 0.001 secondary). Label: SL (a: surv2 < 0.5, or no module on), SH (all M on), B_m.
  Crossing of module k: first g with X_k >= theta (interpolated in ln g / ln X, pd3's rule), over the usable g of the row.
  Mean-level L/R tilt: A = (X_L - X_R)/(X_L + X_R) for AL-L/AL-R and every P2 L/R pair with X_L + X_R >= theta.
Output: results/m2/modules/phase_A.json
"""
import glob, json, os
import numpy as np
import scipy.sparse as sp

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/m2/modules"
RJ = json.load(open(f"{OUT}/groups/regions.json")); RZ = np.load(f"{OUT}/groups/regions.npz")
N = 167216
THETAS = (0.01, 0.001)
PARTS = {}
for P in ("P0", "P1", "P2"):
    regs = [r for r in RJ["regions"] if r["partition"] == P]
    rows, cols = [], []
    for i, r in enumerate(regs):
        idx = RZ[r["key"]]; rows.append(np.full(len(idx), i)); cols.append(idx)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    assert len(cols) == N and len(np.unique(cols)) == N, P            # each partition covers every neuron exactly once
    OH = sp.csr_matrix((np.ones(N), (rows, cols)), shape=(len(regs), N))
    PARTS[P] = {"names": [r["name"][3:] for r in regs], "OH": OH, "size": np.asarray(OH.sum(1)).ravel()}
# L/R pairs: P0 AL-L / AL-R, and P2 X_L / X_R
p2n = PARTS["P2"]["names"]
LR_P2 = [(n[:-2], p2n.index(n), p2n.index(n[:-2] + "_R")) for n in p2n if n.endswith("_L") and n[:-2] + "_R" in p2n]


def load_occ(f, need_surv):
    d = np.load(f, allow_pickle=True)
    m = json.loads(str(d["meta"]))
    occ = d["occ"].astype(np.float64)                                        # [g, N]
    s2 = None
    if need_surv:
        t = d["t"]; half = t >= float(m["T"]) / 2
        s2 = d["surv"][:, half, :].mean((1, 2))
    return m, np.asarray(d["g"], float), occ, s2


def module_X(occ):
    return {P: (D["OH"] @ occ.T / D["size"][:, None]).T for P, D in PARTS.items()}   # [g, M]


def add_points(store, key, gs, X, s2, B):
    for i, g in enumerate(gs):
        store.setdefault(key, []).append((float(g), {P: X[P][i] for P in X}, None if s2 is None else float(s2[i]), int(B)))


def pool(lst):
    lst = sorted(lst, key=lambda x: x[0]); out = []
    for g, X, s2, B in lst:
        if out and abs(g / out[-1][0] - 1) < 1e-4:
            g0, X0, s0, B0 = out[-1]
            out[-1] = (g0, {P: (X0[P] * B0 + X[P] * B) / (B0 + B) for P in X},
                       None if s2 is None else (s0 * B0 + s2 * B) / (B0 + B), B0 + B)
        else:
            out.append((g, X, s2, B))
    return out


def crossings(g, X, thr, names):
    out = {}
    for k, nm in enumerate(names):
        x = X[:, k]; i = np.where(x >= thr)[0]
        if not len(i):
            out[nm] = None; continue
        i = i[0]
        if i == 0:
            out[nm] = float(g[0]); continue
        a_, b_ = np.log(max(x[i - 1], 1e-12)), np.log(x[i])
        out[nm] = float(np.exp(np.log(g[i - 1]) + (np.log(thr) - a_) / (b_ - a_) * (np.log(g[i]) - np.log(g[i - 1]))))
    return out


def summarize(lst, setting):
    pts = pool(lst)
    g = np.array([p[0] for p in pts])
    s2 = np.array([p[2] if p[2] is not None else 1.0 for p in pts])
    res = {"g": g.tolist(), "surv2": s2.tolist() if setting == "a" else None, "B": [p[3] for p in pts], "parts": {}}
    usable = s2 >= 0.5 if setting == "a" else np.ones(len(g), bool)
    for P, D in PARTS.items():
        X = np.array([p[1][P] for p in pts])                                  # [g, M]
        Xs = X / np.maximum(s2, 1e-12)[:, None] if setting == "a" else X
        M = X.shape[1]; pr = {"names": D["names"], "sizes": D["size"].tolist(), "X": Xs.tolist(), "theta": {}}
        for th in THETAS:
            on = Xs >= th
            n_on = np.where(usable, on.sum(1), 0)
            lab = ["SL" if (not usable[i] or n_on[i] == 0) else ("SH" if n_on[i] == M else f"B{int(n_on[i])}")
                   for i in range(len(g))]
            ids = [[D["names"][k] for k in np.argsort(-Xs[i]) if on[i, k]] if usable[i] else [] for i in range(len(g))]
            cr = crossings(g[usable], Xs[usable], th, D["names"]) if usable.sum() >= 2 else {}
            order = sorted([(v, k) for k, v in cr.items() if v is not None])
            first_on = int(np.argmax(n_on > 0)) if (n_on > 0).any() else None
            pr["theta"][f"{th:g}"] = {"n_on": n_on.tolist(), "label": lab, "ids_on": ids, "crossing": cr,
                                     "activation_order": [k for _, k in order],
                                     "first_on_g": float(g[first_on]) if first_on is not None else None,
                                     "first_on_ids": ids[first_on] if first_on is not None else None,
                                     "first_SH_g": float(g[np.argmax(n_on == M)]) if (n_on == M).any() else None,
                                     "n_never_on": int(sum(v is None for v in cr.values()))}
        res["parts"][P] = pr
    # mean-level L/R tilt
    X0 = np.array([p[1]["P0"] for p in pts]); X2 = np.array([p[1]["P2"] for p in pts])
    iL, iR = PARTS["P0"]["names"].index("AL-L"), PARTS["P0"]["names"].index("AL-R")
    tilt = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        tilt["AL"] = np.where(X0[:, iL] + X0[:, iR] > 0, (X0[:, iL] - X0[:, iR]) / (X0[:, iL] + X0[:, iR]), np.nan)
        for nm, a, b in LR_P2:
            tilt[nm] = np.where(X2[:, a] + X2[:, b] > 0, (X2[:, a] - X2[:, b]) / (X2[:, a] + X2[:, b]), np.nan)
    res["LR_tilt"] = {k: [None if not np.isfinite(x) else float(x) for x in v] for k, v in tilt.items()}
    big = {}
    for k, v in tilt.items():
        a = X0[:, [iL, iR]].sum(1) if k == "AL" else X2[:, [p2n.index(k + "_L"), p2n.index(k + "_R")]].sum(1)
        a = a / (np.maximum(s2, 1e-12) if setting == "a" else 1)
        ok = usable & (a >= 0.01) & np.isfinite(v)
        big[k] = {"n_points": int(ok.sum()), "max_abs": float(np.nanmax(np.abs(v[ok]))) if ok.any() else None,
                  "g_at_max": float(g[ok][np.nanargmax(np.abs(v[ok]))]) if ok.any() else None}
    res["LR_tilt_max_where_pair_ge_1pct"] = big
    return res


def main():
    store = {}; used = []
    # (a) decay lane
    idx = []
    for f in sorted(glob.glob(f"{ROOT}/results/m2/decay/sim/index_s1.json") + glob.glob(f"{ROOT}/results/m2/decay/sim/index_s2*.json")):
        idx += json.load(open(f))
    for e in idx:
        if e["graph"] != "real" or not os.path.exists(e["out"]):
            continue
        m, gs, occ, s2 = load_occ(e["out"], True)
        assert abs(float(m["T"]) - 3000) < 1e-6 and m["exp"] == "decay" and float(m.get("h", 0)) == 0
        add_points(store, ("a", e["model"] + "_lane", float(e["r"])), gs, module_X(occ), s2, m["B"])
        used.append(os.path.relpath(e["out"], ROOT))
    # (a) pd3 M1 (pd3_module_occ.py's file set and filter)
    files = sorted(glob.glob(f"{ROOT}/results/pd/decay_r*_real.npz")) + sorted(glob.glob(f"{ROOT}/results/pd2/onset_real_r*.npz")) + \
        sorted(glob.glob(f"{ROOT}/results/pd2/deloc_real_r*.npz"))
    for f in files:
        d = np.load(f, allow_pickle=True); m = json.loads(str(d["meta"]))
        if not (m["exp"] == "decay" and m["norm"] == "raw" and float(m["h"]) == 0 and abs(float(m["T"]) - 3000) < 1e-6
                and not m.get("null") and "occ" in d.files):
            continue
        m, gs, occ, s2 = load_occ(f, True)
        add_points(store, ("a", "M1_pd3", float(m["r"])), gs, module_X(occ), s2, m["B"])
        used.append(os.path.relpath(f, ROOT))
    # (b) driven lane, ORN 0.04, palp False
    for sdir, model in (("A1", "M2"), ("A2", "M1"), ("A4", "M2s")):
        for f in sorted(glob.glob(f"{ROOT}/results/m2/driven/runs/{sdir}/*.npz")):
            try:
                d = np.load(f, allow_pickle=True); m = json.loads(str(d["meta"]))
            except Exception as ex:                                            # noqa
                print("skip unreadable", f, ex); continue
            if not (m["exp"] == "quiet" and "olfactory=0.037926" in m["drive"] and m["sw_clamp_off"] == ["olfactory"]
                    and m.get("occ_win") == "500:1500" and not m.get("null") and m.get("graph") is None):
                continue
            occ = d["occ"].astype(np.float64); gs = np.asarray(d["g"], float)
            add_points(store, ("b", model, float(m["r"])), gs, module_X(occ), None, m["B"])
            used.append(os.path.relpath(f, ROOT))
    out = {"partitions": {P: {"names": D["names"], "sizes": D["size"].tolist()} for P, D in PARTS.items()},
           "thetas": list(THETAS), "rows": {}, "n_files": len(used)}
    for (st, model, r), lst in sorted(store.items()):
        k = f"{st}|{model}|{r:g}"
        out["rows"][k] = summarize(lst, st)
        p = out["rows"][k]["parts"]["P0"]["theta"]["0.01"]
        print(k, "n_g", len(lst), "first on", p["first_on_g"], p["first_on_ids"][:6] if p["first_on_ids"] else None,
              "first SH", p["first_SH_g"], "max n_on", max(p["n_on"]), flush=True)
    json.dump(out, open(f"{OUT}/phase_A.json", "w"))
    json.dump(used, open(f"{OUT}/phase_A_files.json", "w"), indent=0)
    print("saved phase_A.json from", len(used), "files")


if __name__ == "__main__":
    main()
