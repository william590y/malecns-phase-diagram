"""m2_mf_chain.py -- the exact branching linearisation (theory_branch_gpu.py, M1 theory P1-P2) extended with per-spike
depression resources, for all 23 r values (lane m2/theory, meanfield). CPU.

Inputs (M1 theory, read-only): results/pd/branch_real_dense_part*.json (48 g from g_lin to 30 g_lin: lambda(g) of the
branching operator B(g) = f(g E)/dt, K_r(g) = K_eff with the coincidence term, for 23 r) and the Perron vectors at the same g
(results/aldef/depression/theory_vecs.npz). Functions reused from code/ald_dep_theory.py (aldef E3 theory; read-only):
eff_spike (exact renewal efficacy), log_tau_smooth (exact birth-death extinction time), cross.

Chain (derivation.md section 7): mode ansatz a_i = m v_i, n = m / <u^2, v>; birth b(n) = lambda X n (1 - n/K), death alpha n,
with the mode-weighted efficacy X(n) = sum_j u_j v_j e_j(rho_j = n <u^2,v> v_j) / sum_j u_j v_j (e = 1 for non-depressing j).
  frozen    : X fixed at X* = X(n*), n* the deterministic level lambda X(n*) (1 - n*/K) = alpha (slow resource, TAU >> 1)
  adiabatic : X = X(n) (resource follows n instantly)
Lifetime tau = exact mean extinction time from the top state (fully active start). S: tau = 200, on: tau = 3000 (ln g interp).
K: kappa = 1 (K_eff) and kappa_r fitted per r so that the UNDEPRESSED chain reproduces the M1 simulated onset
(results/analysis2/flow_sim_boundaries.json) -- the aldef procedure (code/ald_dep_theory.py fit_kappa), now for every r.
Variants: M2 (exc outputs, 0.25/157), M2_KW09 (0.28/240), M2_N15 (0.22/89), M2_tau30 (0.25/30), M2_allsyn (all outputs).
M2s needs another operator (no rows): not computed here.
Output: results/m2/theory/meanfield_chain.json
"""
import glob, json, sys
import numpy as np

ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
import ald_dep_theory as TH       # noqa: E402  (read-only reuse of the aldef E3 chain functions)

C = "/MALECNS_CACHE"
A_EFF = TH.A_EFF
rows = []
for p in sorted(glob.glob(f"{ROOT}/results/pd/branch_real_dense_part*.json")):
    rows += json.load(open(p))["rows"]
rows.sort(key=lambda x: x["g"])
RS = json.load(open(f"{ROOT}/results/pd/branch_real_dense_part0.json"))["r"]
G = np.array([x["g"] for x in rows]); LAM = np.array([x["lam"] for x in rows])
K = {r: np.array([x[f"K_r{r:g}"] for x in rows]) for r in RS}
V = np.load(f"{ROOT}/results/aldef/depression/theory_vecs.npz")
assert np.allclose(V["g"], G)
u2v = V["u2v"]
idx = [V["idx"][i, :V["n"][i]] for i in range(len(G))]
uv = [V["u"][i, :V["n"][i]] * V["v"][i, :V["n"][i]] for i in range(len(G))]
vv = [V["v"][i, :V["n"][i]] for i in range(len(G))]
sign = np.load(C + "/sign.npy")
DEPS = {"M2": ("exc", 0.25, 157.0), "M2_KW09": ("exc", 0.28, 240.0), "M2_N15": ("exc", 0.22, 89.0),
        "M2_tau30": ("exc", 0.25, 30.0), "M2_allsyn": ("all", 0.25, 157.0)}
sim = {row["r"]: row for row in json.load(open(f"{ROOT}/results/analysis2/flow_sim_boundaries.json"))["rows"]}


NTAB = np.r_[0.0, np.geomspace(1e-6, 1000.0, 1500)]       # n grid for the tabulated X(n) (K_max ~ 62 x kappa)


def X_fun(i, grp, U, TAU):
    """X(n) tabulated once per (variant, g) on NTAB (exact evaluation at the nodes), linear interpolation between."""
    dep = (sign[idx[i]] > 0) if grp == "exc" else np.ones(len(idx[i]), bool)
    rr = np.geomspace(1e-7, TH.RHO_MAX, 400); er = TH.eff_spike(rr, U, TAU)
    w = uv[i]; v = vv[i]
    tab = np.empty(len(NTAB))
    for a in range(0, len(NTAB), 100):
        n = NTAB[a:a + 100]
        rho = np.clip(n[:, None] * u2v[i] * v[None, :], 0, TH.RHO_MAX)
        e = np.where(rho > 1e-7, np.interp(np.log(np.maximum(rho, 1e-7)), np.log(rr), er), 1.0)
        e = np.where(dep[None, :], e, 1.0)
        tab[a:a + 100] = (e * w[None, :]).sum(1) / w.sum()

    def X(n):
        return np.interp(np.atleast_1d(np.asarray(n, float)), NTAB, tab)
    return X


def point(i, Kv, X=None):
    lam = LAM[i]
    if X is None:
        lt = TH.log_tau_smooth(lambda n: lam * np.ones_like(n), Kv)
        return {"lt_plain": lt}
    out = {"X0": float(X(np.array([1e-9]))[0])}
    out["lt_adiabatic"] = TH.log_tau_smooth(lambda n: lam * X(n), Kv)
    nn = np.linspace(1e-6, Kv, 2000)
    F = lam * X(nn) * (1 - nn / Kv) - A_EFF
    if F[0] <= 0:
        out.update({"n_star": 0.0, "X_star": out["X0"], "lt_frozen": out["lt_adiabatic"]})
    else:
        k = int(np.argmax(F <= 0)) if (F <= 0).any() else len(nn) - 1
        ns = float(np.interp(0.0, [F[k], F[k - 1]], [nn[k], nn[k - 1]])) if k > 0 else float(nn[0])
        Xs = float(X(np.array([ns]))[0])
        out.update({"n_star": ns, "X_star": Xs, "sigma_frozen": float(lam * Xs / A_EFF),
                    "lt_frozen": TH.log_tau_smooth(lambda n: lam * Xs * np.ones_like(n), Kv)})
    return out


def fit_kappa(r, g_on, T=3000.0):
    lo, hi = 0.3, 10.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lt = [point(i, mid * K[r][i])["lt_plain"] for i in range(len(G))]
        gc = TH.cross(G, lt, T)
        if gc is None or gc > g_on:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


res = {"g": G.tolist(), "lam_over_alpha": (LAM / A_EFF).tolist(), "r": RS, "kappa": {}, "lines": {}, "curves": {}}
Xcache = {}
for r in RS:
    kap = fit_kappa(r, sim[r]["on"])
    res["kappa"][f"{r:g}"] = kap
    # check: undepressed chain lines with this kappa (on reproduces the M1 onset by construction; S is a test)
    ltb = [point(i, kap * K[r][i])["lt_plain"] for i in range(len(G))]
    res["lines"][f"M1|r{r:g}|kcal"] = {"S": TH.cross(G, ltb, 200.0), "on": TH.cross(G, ltb, 3000.0),
                                       "sim_S": sim[r]["S"], "sim_on": sim[r]["on"]}
    for name, (grp, U, TAU) in DEPS.items():
        for kname, kk in (("k1", 1.0), ("kcal", kap)):
            pts = []
            for i in range(len(G)):
                key = (name, i)
                if key not in Xcache:
                    Xcache[key] = X_fun(i, grp, U, TAU)
                pts.append(point(i, kk * K[r][i], Xcache[key]))
            lf = [p["lt_frozen"] for p in pts]; la = [p["lt_adiabatic"] for p in pts]
            res["lines"][f"{name}|r{r:g}|{kname}"] = {
                "S_frozen": TH.cross(G, lf, 200.0), "on_frozen": TH.cross(G, lf, 3000.0),
                "S_adiabatic": TH.cross(G, la, 200.0), "on_adiabatic": TH.cross(G, la, 3000.0)}
            if r in (0, 1, 4) or name == "M2":
                res["curves"][f"{name}|r{r:g}|{kname}"] = {k: [p.get(k) for p in pts] for k in
                                                           ("lt_frozen", "lt_adiabatic", "n_star", "X_star", "sigma_frozen")}
    print(f"r={r:g} kappa={kap:.3f}", {k: v for k, v in res["lines"][f"M2|r{r:g}|kcal"].items()}, flush=True)
res["note"] = ("lines are None when the crossing is not reached within the rows' g range [g_lin, 30 g_lin] = "
               f"[{G[0]:.3f}, {G[-1]:.3f}]; S/on from the fully active top state")
json.dump(res, open(f"{ROOT}/results/m2/theory/meanfield_chain.json", "w"), indent=1, default=float)
print("saved")
