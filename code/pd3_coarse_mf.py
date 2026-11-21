"""pd3 module_theory MT2: de Candia's closure on connectome modules (GPU if available, else CPU; torch float64).

Model (theory_notes.md 1.2, Perron-weighted, the simulator's synchronous rates):
    X_k <- X_k e^{-dt} + (1 - X_k) (1 - exp(-tanh(g sum_l W_kl(r) X_l)_+ dt)),   W(r) = E^PW - r I^PW,
for the M = 30 modules of partition P plus the lumped rest (results/pd3/modules/quotient_<P>.npz).
Grid: g at 160 ln points in [0.12, 30]; r = 0..6 in steps of 0.25.
Starts: all active; each module alone (31); each pair among the 8 modules with the largest Lambda_k (28); 200 random subsets
  (rng 7, each module on with p = 1/2). Two passes: silent modules exactly 0 (pass A) and silent modules at a one-neuron seed
  1/n_k (pass B). Started modules at X = 1.
Iteration to |dX|_inf < 1e-10 per step (checked every 100 steps), at most 2e5 steps. Per (g, r): converged end states of all
  starts are deduplicated (L_inf < 1e-6) and tested for stability: exact Jacobian spectral radius < 1 - 1e-9,
     J_kl = delta_kl (e^{-dt} - p(s_k)) + (1 - X_k) p'(s_k) g W_kl,  p'(s) = dt sech^2(s) e^{-tanh(s) dt} (s > 0), 0 (s < 0);
  rows of silent modules with s_k == 0 exactly (no input from the active set) use the one-sided cone map (perturbations of a
  silent module are >= 0), whose growth rate is obtained by power iteration (SL itself: g mu_+(W) < alpha_eff, mu_+ the cone
  spectral radius = rectified Perron root).
Labels: active set A = {k : X_k > 1e-3}: SL (empty), B_m (with identities), SH (all M + 1).
Non-stationary columns: a column whose residual stays > 1e-5 for 10 consecutive checks after step 20,000 is frozen and
  counted as unconverged (DEVIATION from 'at most 2e5 steps': a CPU probe at r = 1 found 29% of random-start columns at
  g >= 3 still at residual > 1e-6 after 2e5 steps, i.e. the closure itself does not settle there); a dt = 0.05 census at r = 1
  checks whether that is a synchronous-update artefact.
Continuation at each r: up-sweep from SL adding 1e-6 to every module at each g, down-sweep from all active at g = 30
  (non-stationary points store the 1000-step time average).
  A boundary along a sweep is first order if the state jumps (max_k |dX_k| > 0.02 between neighbouring g, ~3.5% apart) or if
  the up and down sweeps differ (hysteresis); otherwise second order (continuous).
Validation (--validate): the 2-module AL pair of theory_module2.json (PW_E_mean1 / PW_I_mean1) on its own grid and starts must
  reproduce its coarse_2module labels.
Output: results/pd3/modules/coarse_mf_<P>.npz (+ coarse_mf_<P>.json summary).
Usage: python pd3_coarse_mf.py P0|P1 [--validate]
"""
import argparse, json, time
import numpy as np
import torch

ROOT = "/MALECNS_ROOT"
ap = argparse.ArgumentParser(); ap.add_argument("P"); ap.add_argument("--validate", action="store_true")
ap.add_argument("--maxsteps", type=int, default=200000)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.float64
t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
dt = 0.2; dec = float(np.exp(-dt)); a_eff = (1 - dec) / dt


def iterate(E, I, X, gcol, rcol, maxsteps, tol=1e-10, chk=100, dev=dev, dtt=None, verbose=False):
    """iterate all columns of X [M, n] (each with its own g, r) to |dX|_inf < tol; returns X, residual, steps.
    A column whose residual stays > 1e-5 at 10 consecutive checks after step 20,000 is frozen as non-stationary."""
    dt_ = dt if dtt is None else dtt; dec_ = float(np.exp(-dt_))
    E_, I_ = torch.tensor(E, device=dev, dtype=DT), torch.tensor(I, device=dev, dtype=DT)
    X = torch.tensor(X, device=dev, dtype=DT); g = torch.tensor(gcol, device=dev, dtype=DT); gr = g * torch.tensor(rcol, device=dev, dtype=DT)
    n = X.shape[1]; res = np.full(n, np.inf); nst = np.zeros(n, np.int64); hi = np.zeros(n, np.int64)
    live = torch.arange(n, device=dev); st = 0
    while st < maxsteps and len(live):
        Xl = X[:, live]; gl = g[live]; grl = gr[live]
        for it in range(chk):
            s = (E_ @ Xl) * gl - (I_ @ Xl) * grl
            Xn = Xl * dec_ + (1 - Xl) * (-torch.expm1(-torch.tanh(s.clamp(min=0)) * dt_))
            if it == chk - 1:
                r_ = (Xn - Xl).abs().amax(0)
            Xl = Xn
        st += chk
        X[:, live] = Xl
        li = live.cpu().numpy(); rn = r_.cpu().numpy(); res[li] = rn; nst[li] = st
        hi[li] = np.where(rn > 1e-5, hi[li] + 1, 0)
        keep = (rn >= tol) & ~((st >= 20000) & (hi[li] >= 10))
        live = live[torch.tensor(keep, device=dev)]
        if verbose and st % 10000 == 0:
            log(f"  step {st}: live columns {len(live)}")
    return X.cpu().numpy(), res, nst


def tavg(E, I, X, gcol, rcol, nrec=1000, dev="cpu"):
    """time average and relative peak-to-peak amplitude of the module vector over nrec steps (non-stationary columns)"""
    E_, I_ = torch.tensor(E, device=dev, dtype=DT), torch.tensor(I, device=dev, dtype=DT)
    X = torch.tensor(X, device=dev, dtype=DT); g = torch.tensor(gcol, device=dev, dtype=DT); gr = g * torch.tensor(rcol, device=dev, dtype=DT)
    acc = torch.zeros_like(X); mx = torch.full((X.shape[1],), -np.inf, dtype=DT, device=dev); mn = -mx.clone()
    for _ in range(nrec):
        s = (E_ @ X) * g - (I_ @ X) * gr
        X = X * dec + (1 - X) * (-torch.expm1(-torch.tanh(s.clamp(min=0)) * dt)); acc += X
        tot = X.sum(0); mx = torch.maximum(mx, tot); mn = torch.minimum(mn, tot)
    avg = (acc / nrec).cpu().numpy()
    return avg, ((mx - mn) / torch.clamp(acc.sum(0) / nrec, min=1e-300)).cpu().numpy()


def mu_plus(W, iters=20000):
    """cone spectral radius (rectified Perron root) of W by shifted rectified power iteration"""
    x = np.ones(W.shape[0]) / np.sqrt(W.shape[0]); mu = 0.0
    for it in range(iters):
        y = np.clip(W @ x, 0, None); m = np.linalg.norm(y) / np.linalg.norm(x)
        if m == 0:
            return 0.0
        x = x + y / m; x /= np.linalg.norm(x)
        if it > 100 and abs(m - mu) < 1e-12 * m:
            return m
        mu = m
    return mu


def stability(X, g, W):
    """growth rate of the linearisation at fixed point X (one-sided rows for silent modules with s == 0)"""
    s = g * (W @ X); th = np.tanh(np.clip(s, 0, None)); p = -np.expm1(-th * dt)
    dp = np.where(s > 0, dt * (1 - th ** 2) * np.exp(-th * dt), 0.0)
    zero = s == 0
    J = np.diag(dec - p) + ((1 - X) * dp)[:, None] * g * W
    if not zero.any():
        return float(np.abs(np.linalg.eigvals(J)).max()), False
    v = np.random.default_rng(0).random(len(X)); v /= np.linalg.norm(v); lg = []
    for it in range(3000):
        y = J @ v
        y[zero] = dec * v[zero] + dt * np.clip(g * (W[zero] @ v), 0, None)
        nr = np.linalg.norm(y); v = y / nr
        if it >= 2000:
            lg.append(np.log(nr))
    return float(np.exp(np.mean(lg))), True


def label(X, names, thr=1e-3):
    A = [names[k] for k in range(len(X)) if X[k] > thr]
    if not A:
        return "SL"
    if len(A) == len(X):
        return "SH"
    return f"B{len(A)}:" + "+".join(A)


def census(E, I, names, G, R, starts, eps, maxsteps):
    """attractors per (r, g) from all starts (passes A and B)"""
    M = E.shape[0]; S = np.array(starts, float).T                    # [M, nS]
    SB = np.where(S > 0, S, eps[:, None])
    allS = np.concatenate([S, SB], 1); nS = allS.shape[1]
    cols_g = np.tile(np.repeat(G, nS), len(R)); cols_r = np.repeat(R, len(G) * nS)
    X0 = np.tile(allS, (1, len(G) * len(R)))
    log(f"census: {X0.shape[1]} columns on {dev}")
    Xf, res, nst = iterate(E, I, X0, cols_g, cols_r, maxsteps, verbose=True)
    log(f"census iterated: unconverged {int((res >= 1e-10).sum())} (resid > 1e-6: {int((res > 1e-6).sum())}), "
        f"max steps {nst.max()}")
    out = {}
    for ri, r in enumerate(R):
        W = E - r * I; mup = mu_plus(W)
        for gi, g in enumerate(G):
            base = (ri * len(G) + gi) * nS
            Xs, rs = Xf[:, base:base + nS].T, res[base:base + nS]
            uniq = []
            for j in np.argsort(rs):
                if rs[j] >= 1e-10:
                    continue
                if not any(np.abs(Xs[j] - u["X"]).max() < 1e-6 for u in uniq):
                    uniq.append({"X": Xs[j], "starts": [int(j)]})
                else:
                    for u in uniq:
                        if np.abs(Xs[j] - u["X"]).max() < 1e-6:
                            u["starts"].append(int(j)); break
            stab = []
            for u in uniq:
                if u["X"].max() == 0:
                    rho, one = dec + dt * g * mup, True           # SL: exact cone growth rate
                else:
                    rho, one = stability(u["X"], g, W)
                u["rhoJ"], u["one_sided"], u["stable"], u["label"] = rho, one, rho < 1 - 1e-9, label(u["X"], names)
                stab.append(u)
            out[(ri, gi)] = {"attractors": [u for u in stab if u["stable"]], "unstable": [u for u in stab if not u["stable"]],
                             "n_unconverged": int((rs >= 1e-10).sum()), "max_resid": float(rs.max())}
    return out


def sweeps(E, I, G, R, maxsteps):
    M = E.shape[0]; nR = len(R)
    up = np.zeros((nR, len(G), M)); dn = np.zeros((nR, len(G), M)); upres = np.zeros((nR, len(G))); dnres = np.zeros((nR, len(G)))
    Xu = np.zeros((M, nR)); Xd = np.ones((M, nR))
    for i in range(len(G)):
        Xu = Xu + 1e-6
        Xu, ru, _ = iterate(E, I, Xu, np.full(nR, G[i]), R, maxsteps, dev="cpu"); up[:, i] = Xu.T; upres[:, i] = ru
        j = len(G) - 1 - i
        Xd, rd, _ = iterate(E, I, Xd, np.full(nR, G[j]), R, maxsteps, dev="cpu"); dn[:, j] = Xd.T; dnres[:, j] = rd
        for Xs, rs, arr, gi in [(Xu, ru, up, i), (Xd, rd, dn, j)]:
            bad = rs >= 1e-10
            if bad.any():                                    # non-stationary: store the time average instead
                av, _ = tavg(E, I, Xs[:, bad], np.full(bad.sum(), G[gi]), R[bad])
                arr[bad, gi] = av.T
        if i % 20 == 0:
            log(f"sweeps {i+1}/{len(G)}: up unconverged {int((ru >= 1e-10).sum())}, down unconverged {int((rd >= 1e-10).sum())}")
    return up, dn, upres, dnres


def boundaries(up, dn, G, R, names):
    """classify changes of the active set along the sweeps: second order (continuous) / first order (jump or hysteresis)"""
    rows = []
    for ri, r in enumerate(R):
        for br, Xs in [("up", up[ri]), ("down", dn[ri])]:
            labs = [label(x, names) for x in Xs]
            for i in range(1, len(G)):
                if labs[i] != labs[i - 1]:
                    jump = float(np.abs(Xs[i] - Xs[i - 1]).max())
                    rows.append({"r": float(r), "branch": br, "g_lo": float(G[i - 1]), "g_hi": float(G[i]), "from": labs[i - 1],
                                 "to": labs[i], "max_jump": jump, "order": "first" if jump > 0.02 else "second"})
        hyst = np.abs(up[ri] - dn[ri]).max(1) > 1e-3
        if hyst.any():
            idx = np.where(hyst)[0]
            rows.append({"r": float(r), "branch": "hysteresis", "g_lo": float(G[idx[0]]), "g_hi": float(G[idx[-1]]),
                         "n_points": int(hyst.sum()), "order": "first"})
    return rows


def segments(cen, G, R):
    seg = {}
    for ri, r in enumerate(R):
        s, prev = [], None
        for gi, g in enumerate(G):
            labs = "|".join(sorted(u["label"] for u in cen[(ri, gi)]["attractors"])) or "none"
            if labs != prev:
                s.append([float(g), labs]); prev = labs
        seg[f"{r:g}"] = s
    return seg


if a.validate:
    t2 = json.load(open(f"{ROOT}/results/pd3/theory_module2.json"))
    E = np.array(t2["PW_E_mean1"]); I = np.array(t2["PW_I_mean1"]); cm = t2["coarse_2module"]
    G = np.array(cm["g"]); R = np.array(cm["r"]); eps = np.array([cm["eps"]["L"], cm["eps"]["R"]])
    # theory_module2 starts: (1, eps), (eps, 1), (1, 1), (eps, eps), iterated 30,000 steps; labels SL / SH / B1L / B1R
    X0 = np.array([[1, eps[1]], [eps[0], 1], [1, 1], [eps[0], eps[1]]]).T
    nS = 4; cols_g = np.tile(np.repeat(G, nS), len(R)); cols_r = np.repeat(R, len(G) * nS)
    Xf, res, nst = iterate(E, I, np.tile(X0, (1, len(G) * len(R))), cols_g, cols_r, 30000, tol=0)
    agree = tot = 0
    for ri, r in enumerate(R):
        labs = []
        for gi in range(len(G)):
            sset = set()
            for j in range(nS):
                xl, xr = Xf[:, (ri * len(G) + gi) * nS + j]
                sset.add("SL" if max(xl, xr) < 1e-6 else "SH" if min(xl, xr) > 1e-3 else ("B1L" if xl > xr else "B1R"))
            labs.append("+".join(sorted(sset)))
        ref = cm["phase"][f"{r:g}"]["labels"]
        agree += sum(x == y for x, y in zip(labs, ref)); tot += len(ref)
    log(f"VALIDATION vs theory_module2 coarse_2module: {agree}/{tot} (g, r) labels identical")
    json.dump({"validation_agree": agree, "validation_total": tot}, open(f"{ROOT}/results/pd3/modules/coarse_mf_validation.json", "w"))
    raise SystemExit

q = np.load(f"{ROOT}/results/pd3/modules/quotient_{a.P}.npz", allow_pickle=True)
E, I = q["E_PW"], q["I_PW"]; names = [str(x) for x in q["names"]]; nk = q["n"].astype(float); M = len(names)
Lam = q["Lambda"]
G = np.geomspace(0.12, 30, 160); R = np.round(np.arange(0, 6.001, 0.25), 3)
starts = [np.ones(M)] + [np.eye(M)[k] for k in range(M)]
top8 = [k for k in np.argsort(-Lam) if names[k] != "rest"][:8]
for i in range(8):
    for j in range(i + 1, 8):
        v = np.zeros(M); v[[top8[i], top8[j]]] = 1; starts.append(v)
rng = np.random.default_rng(7)
for _ in range(200):
    starts.append((rng.random(M) < 0.5).astype(float))
eps = 1 / nk
log(f"{a.P}: M+1 = {M} modules, {len(starts)} starts x 2 passes, grid {len(G)} g x {len(R)} r")
cen = census(E, I, names, G, R, starts, eps, a.maxsteps)
up, dn, upres, dnres = sweeps(E, I, G, R, a.maxsteps)
log("sweeps done")
bnd = boundaries(up, dn, G, R, names)
seg = segments(cen, G, R)
nattr = np.array([[len(cen[(ri, gi)]["attractors"]) for gi in range(len(G))] for ri in range(len(R))])
nunc = np.array([[cen[(ri, gi)]["n_unconverged"] for gi in range(len(G))] for ri in range(len(R))])
maxA = max(1, nattr.max())
AX = np.full((len(R), len(G), maxA, M), np.nan); AL_ = np.full((len(R), len(G), maxA), "", dtype=object)
AR = np.full((len(R), len(G), maxA), np.nan); AN = np.zeros((len(R), len(G), maxA), int)
for (ri, gi), c in cen.items():
    for j, u in enumerate(sorted(c["attractors"], key=lambda u: -u["X"].sum())):
        AX[ri, gi, j] = u["X"]; AL_[ri, gi, j] = u["label"]; AR[ri, gi, j] = u["rhoJ"]; AN[ri, gi, j] = len(u["starts"])
gSL = np.array([a_eff / max(mu_plus(E - r * I), 1e-300) for r in R])
# dt check: all-active and the 31 single-module starts at r = 1, dt = 0.05 (same model time budget, 4x steps)
S1 = np.array([np.ones(M)] + [np.eye(M)[k] for k in range(M)]).T; nS1 = S1.shape[1]
Xa, ra, _ = iterate(E, I, np.tile(S1, (1, len(G))), np.repeat(G, nS1), np.full(len(G) * nS1, 1.0), 4 * a.maxsteps, dtt=0.2)
Xb, rb, _ = iterate(E, I, np.tile(S1, (1, len(G))), np.repeat(G, nS1), np.full(len(G) * nS1, 1.0), 4 * a.maxsteps, dtt=0.05)
nonstat_dt02 = (ra.reshape(len(G), nS1) >= 1e-10).sum(1); nonstat_dt005 = (rb.reshape(len(G), nS1) >= 1e-10).sum(1)
log(f"dt check r=1: non-stationary columns dt 0.2: {int(nonstat_dt02.sum())}, dt 0.05: {int(nonstat_dt005.sum())}")
np.savez_compressed(f"{ROOT}/results/pd3/modules/coarse_mf_{a.P}.npz", g=G, r=R, names=np.array(names), n_attr=nattr,
                    n_unconverged=nunc, attr_X=AX, attr_label=AL_.astype(str), attr_rhoJ=AR, attr_nstarts=AN,
                    up=up, dn=dn, up_resid=upres, dn_resid=dnres, g_SL_edge=gSL, n_starts=len(starts),
                    dtcheck_nonstat_dt02=nonstat_dt02, dtcheck_nonstat_dt005=nonstat_dt005)
summ = {"P": a.P, "modules": names, "n_starts_per_pass": len(starts), "g_SL_edge": dict(zip([f"{r:g}" for r in R], gSL.tolist())),
        "max_n_attractors": int(nattr.max()), "n_cells_multistable": int((nattr > 1).sum()), "n_cells": int(nattr.size),
        "n_cells_with_unconverged": int((nunc > 0).sum()), "segments": seg, "sweep_boundaries": bnd,
        "g_range_cells_with_unconverged": [float(G[(nunc > 0).any(0)].min()), float(G[(nunc > 0).any(0)].max())] if (nunc > 0).any() else None,
        "dtcheck_r1": {"g_nonstationary_dt0.2": G[nonstat_dt02 > 0].tolist(), "g_nonstationary_dt0.05": G[nonstat_dt005 > 0].tolist(),
                       "n_dt0.2": int(nonstat_dt02.sum()), "n_dt0.05": int(nonstat_dt005.sum()), "n_columns_each": int(len(G) * nS1)},
        "first_order_boundaries": [b for b in bnd if b["order"] == "first"],
        "multistable_cells_examples": [(float(R[ri]), float(G[gi]), [u["label"] for u in cen[(ri, gi)]["attractors"]])
                                       for ri in range(len(R)) for gi in range(len(G)) if nattr[ri, gi] > 1][:60]}
json.dump(summ, open(f"{ROOT}/results/pd3/modules/coarse_mf_{a.P}.json", "w"), indent=1)
log(f"saved coarse_mf_{a.P}: max attractors {nattr.max()}, multistable cells {(nattr > 1).sum()}/{nattr.size}, "
    f"first-order sweep boundaries {len(summ['first_order_boundaries'])}")
