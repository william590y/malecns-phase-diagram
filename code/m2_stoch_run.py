"""m2_stoch_run.py -- tasks of lane m2/theory-stochastic (plan results/m2/theory/stochastic_plan.md).

usage: m2_stoch_run.py <task> [options]       (GPU unless noted; outputs results/m2/theory/stochastic/<task>/...)
  decay  --model M --r R [--graph F --tag T] [--gmin --gmax --ng] [--transient]
         QMF + resources at h = 0 (high branch, descending g), renormalised mode, chains (F / A / 2D; PF and CAL), regions,
         deterministic transient from the fully active start.
  driven --model M --r R [--orn 0.04] [--gmin --gmax --ng] [--lna]   ORN-driven fixed points (up and down sweeps) + LNA
  lnaqs  --model M --r R --g g1,g2,..                                LNA at h = 0 QS states (oscillation, variance)
  pulse  --model M --gaps 0,10,100 [--gmin --gmax --ng]              self-organised resources in the pulse protocol
  modes  (CPU)                                                        top excitatory modes, per-mode M1 / M2 single-mode onsets
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "/MALECNS_ROOT/code")
import m2_stoch_core as S

OUTR = f"{S.ROOT}/results/m2/theory/stochastic"


def kappa_cal():
    """M1 calibration (fixed in the plan): c_r = K_needed / K_noD (no-coincidence K) and kap_r = (K_needed / K_eff) /
    (K_needed / K_eff at r = 0), from results/analysis2/phase_complete.json (median K needed over M1 slow-window points)."""
    d = json.load(open(f"{S.ROOT}/results/analysis2/phase_complete.json"))
    return d


def m1_factors():
    txt = open(f"{S.ROOT}/results/analysis2/phase_complete.md").read().split("### MaleCNS")[1].split("### configuration")[0]
    rows = {}
    for line in txt.splitlines():
        if line.startswith("| ") and line[2].isdigit():
            c = [x.strip() for x in line.strip("|").split("|")]
            r = float(c[0]); Kn = float(c[12].split()[0]); Ke = c[13]
            Keff = float(Ke.split()[0]); KnoD = float(Ke.split("(")[1].rstrip(")"))
            rows[r] = {"K_needed": Kn, "K_eff": Keff, "K_noD": KnoD}
    return rows


def interp_factor(r, rows, key):
    rs = np.array(sorted(rows)); v = np.array([key(rows[x]) for x in rs])
    return float(np.interp(r, rs, v))


def shares(net, rho):
    rr = rho.cpu().numpy() if hasattr(rho, "cpu") else rho
    tot = rr.sum()
    out = {"rho_tot": float(tot), "rho_mean": float(tot / net.N)}
    if tot > 0:
        out["PR"] = float(tot ** 2 / (rr ** 2).sum())
        for k, m in net.masks.items():
            out[f"share_{k}"] = float(rr[m].sum() / tot)
            out[f"rho_{k}"] = float(rr[m].mean()) if m.any() else np.nan
    return out


R_CORE = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0)


def task_decay(a):
    """Setting (a). Profile: first-order branching closure (bqmf; r-independent) or the subtractive QMF (--closure qmf,
    at the given r). Chains for every r in R_CORE: PF (K, n* from the profile) and CAL (both x c_r, M1's K_needed/K_noD)."""
    torch = __import__("torch")
    net = S.Net(a.graph, model=a.model)
    rows = m1_factors()
    c_r = {r: interp_factor(r, rows, lambda x: x["K_needed"] / x["K_noD"]) for r in R_CORE}
    G = np.geomspace(a.gmin, a.gmax, a.ng)[::-1]
    h = torch.zeros(net.N, device=net.dev, dtype=torch.float64)
    res = []; st = None; v0 = u0 = None
    U, TAU = net.Upar, net.TAUpar
    for g in G:
        t0 = time.time()
        r0 = None if st is None else st["rho"]; O0 = None if st is None else st["O"]
        if a.closure == "bqmf":
            st = S.bqmf_solve(net, g, h, r0, O0, iters=a.iters, tol=1e-9, eta=0.5)
        else:
            st = S.qmf_solve(net, g, a.r, h, r0, O0, iters=a.iters, tol=1e-9, eta=0.5)
        rho = st["rho"]
        row = {"g": float(g), "qmf_it": st["it"], "qmf_res": st["res"]}
        row.update(shares(net, rho))
        if row["rho_tot"] < 1e-6:
            row["silent"] = True; res.append(row); log_row(row); st = None
            continue
        e = torch.where(rho > 1e-300, st["O"] / torch.clamp(rho, min=1e-300), torch.ones_like(rho))
        ea = e.cpu().numpy(); rr = rho.cpu().numpy()
        dn = net.dep_node
        row["e_mean_act"] = float((ea * rr)[dn].sum() / max(rr[dn].sum(), 1e-300)) if dn.any() else 1.0
        mAL = net.masks["AL_module"] & dn
        row["e_AL"] = float((ea * rr)[mAL].sum() / max(rr[mAL].sum(), 1e-300)) if mAL.any() else 1.0
        md, v, u = S.renorm_mode(net, g, e, rho, v0, u0)
        v0, u0 = v, u
        row.update({f"mode_{k}": val for k, val in md.items()})
        ns = md["n_mode"]; sig = md["sigma"]
        row["n_star"] = ns
        variants = {}
        if sig > 1 and ns > 0:
            K_pf = ns / (1 - 1 / sig)
            row["K_pf"] = K_pf
            for r in (R_CORE if a.closure == "bqmf" else (a.r,)):
                for name, fac in (("PF", 1.0), ("CAL", c_r[r])):
                    if name == "PF" and r != R_CORE[0] and a.closure == "bqmf":
                        continue                       # PF is r-independent under the first-order closure
                    par = {"lam": md["lam"], "lam_rest": md["lam_rest"], "X": md["X"], "n_star": fac * ns, "K": fac * K_pf}
                    lt = S.lifetimes(par, U, TAU, do2d=True)
                    lt.update({"K": fac * K_pf, "n_star": fac * ns})
                    variants[f"{name}_r{r:g}"] = lt
        row["chains"] = variants
        if a.peel > 0:
            row["islands"] = S.peel_islands(net, g, e, rho, a.peel)
            S.log("   islands: " + " | ".join(f"k{x['k']} sig {x['sigma']:.3f} K0 {x['K0']:.0f} n {x['n_mode']:.1f} AL {x['AL_v']:.2f} {x['top_superclass']}" for x in row["islands"]))
        if a.transient and (len(res) % a.tevery == 0):
            rec_t, rec, _ = S.qmf_dynamics(net, g, a.r, h, steps=a.tsteps, rec_every=5, closure=a.closure)
            ntr = np.array([float((u * x).sum()) / md["u2v"] for x in rec])
            tt = np.array(rec_t)
            m = tt >= 10
            k = int(np.argmin(ntr[m])) if m.any() else 0
            row["transient"] = {"n_min": float(ntr[m][k]), "t_min": float(tt[m][k]), "n_end": float(ntr[-1]),
                                "n_min_over_nstar": float(ntr[m][k] / ns) if ns > 0 else np.nan,
                                "n_t": ntr[::4].tolist(), "t": tt[::4].tolist()}
        row["sec"] = time.time() - t0
        res.append(row); log_row(row)
    out = {"task": "decay", "closure": a.closure, "model": a.model, "r_profile": a.r if a.closure == "qmf" else "any",
           "graph": a.graph, "tag": a.tag, "U": U, "TAU": TAU, "c_r": c_r,
           "kappa_dep": S.kappa(U, TAU) if U > 0 else 0.0, "N": net.N, "rows": res, "cross": {}}
    gg = np.array([x["g"] for x in res])[::-1]
    keys = sorted({k for x in res for k in x.get("chains", {})})
    for vk in keys:
        for lt in ("lt_F", "lt_A", "lt_2D"):
            y = np.array([x.get("chains", {}).get(vk, {}).get(lt, -np.inf) for x in res], float)[::-1]
            y = np.where(np.isnan(y), -np.inf, y)
            out["cross"][f"{vk}_{lt}"] = {"S": S.cross(gg, y, 200.0), "on": S.cross(gg, y, 3000.0)}
    save(out, f"decay/{a.closure}_{a.model}{'' if a.closure == 'bqmf' else f'_r{a.r:g}'}{a.tag}.json")


def log_row(row):
    ch = row.get("chains", {}).get("PF_r0", row.get("chains", {}).get(f"PF_r{row.get('r', 0):g}", {}))
    S.log(f"g {row['g']:.4f} rho {row.get('rho_tot', 0):.4g} PR {row.get('PR', 0):.0f} AL {row.get('share_AL_module', 0):.3f} "
          f"sig {row.get('mode_sigma', 0):.4f} n* {row.get('n_star', 0):.1f} K0 {row.get('mode_K0', 0):.0f} X {row.get('mode_X', 0):.3f} "
          f"ltF {ch.get('lt_F', np.nan):.2f} ltA {ch.get('lt_A', np.nan):.2f} lt2D {ch.get('lt_2D', np.nan):.2f}")


def save(obj, rel):
    p = f"{OUTR}/{rel}"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w"), default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    S.log("saved", p)


def task_driven(a):
    torch = __import__("torch")
    net = S.Net(a.graph, model=a.model)
    h = net.hvec(S.H_ORN if a.orn == 0.04 else float(__import__("m2_common").h_of_rho(a.orn)))
    G = np.geomspace(a.gmin, a.gmax, a.ng)
    if a.glist:
        G = np.array([float(x) for x in a.glist.split(",")])
    res = []
    st = None
    for g in G:                       # up-sweep from a quiet start (each g from zero when --glist is given)
        if a.glist:
            st = None
        t0 = time.time()
        st = S.qmf_solve(net, g, a.r, h, None if st is None else st["rho"], None if st is None else st["O"],
                         iters=a.iters, tol=1e-10, eta=0.5) if st is not None else \
            S.qmf_solve(net, g, a.r, h, torch.zeros(net.N, device=net.dev, dtype=torch.float64),
                        torch.zeros(net.N, device=net.dev, dtype=torch.float64), iters=a.iters, tol=1e-10, eta=0.5)
        row = {"g": float(g), "qmf_it": st["it"], "qmf_res": st["res"]}
        row.update(shares(net, st["rho"]))
        rr = st["rho"].cpu().numpy()
        row["G_gain"] = float(rr[net.masks["uPN_ant"]].mean() / rr[net.masks["ORN"]].mean())
        if a.lna:
            L = S.LNA(net, g, a.r, st)
            obs = L.obs_activity([net.masks["nonORN"], net.masks["AL_module"]])
            o = L.run(obs, maxlag=a.maxlag, ckpt=150, smax=a.smax, rtol=1e-9)
            nn = [int(net.masks["nonORN"].sum()), int(net.masks["AL_module"].sum())]
            for j, nm in enumerate(("nonORN", "AL")):
                var = float(o["var"][j]); mu = row[f"rho_{'nonORN' if nm == 'nonORN' else 'AL_module'}"]
                row[f"chi_{nm}"] = nn[j] * var; row[f"fano_{nm}"] = nn[j] * var / mu if mu > 0 else np.nan
                mtr = S.acf_metrics(o["acf"][:, j])
                row[f"acf_{nm}"] = mtr
                row[f"acf_{nm}_curve"] = (o["acf"][::5, j] / o["acf"][0, j]).tolist()
            row["lna_S"] = o["S"]; row["lna_conv"] = o["converged"]
            con, qon, varb = L.obs_onsets(net.masks["nonORN"])
            o2 = L.run(con, maxlag=100, ckpt=150, smax=a.smax, rtol=1e-9, qon=qon)
            var_on = float(o2["acf"][0, 0]) + varb
            cov = np.r_[var_on, o2["acf"][1:, 0] + o2["xq"][1:, 0]]
            m, rk = S.mr_from_cov(cov, var_on, 20)
            row["mr_m_step"] = m; row["mr_m_unit"] = m ** 5 if np.isfinite(m) else np.nan
            row["onset_rate"] = float(((1 - st["rho"]) * st["p"]).cpu().numpy()[net.masks["nonORN"]].sum())
            row["on_var"] = var_on; row["on_rk"] = rk.tolist()
        row["sec"] = time.time() - t0
        res.append(row)
        S.log(f"g {g:.4f} rho_nonORN {row['rho_nonORN']:.4g} G {row['G_gain']:.3f} AL {row['rho_AL_module']:.4g} "
              f"chi {row.get('chi_nonORN', np.nan):.3g} chiAL {row.get('chi_AL', np.nan):.3g} "
              f"tint {row.get('acf_nonORN', {}).get('tau_int', np.nan):.2f} P {row.get('acf_nonORN', {}).get('period', np.nan):.1f} "
              f"m {row.get('mr_m_unit', np.nan):.4f} S {row.get('lna_S')} {row['sec']:.0f}s")
    # down-sweep (QMF only) for mean-field bistability
    down = []; st = None
    for g in (G[::-1] if not a.glist else []):
        st = S.qmf_solve(net, g, a.r, h, None if st is None else st["rho"], None if st is None else st["O"],
                         iters=a.iters, tol=1e-10, eta=0.5)
        dd = shares(net, st["rho"]); dd["g"] = float(g); down.append(dd)
    out = {"task": "driven", "model": a.model, "r": a.r, "orn": a.orn, "graph": a.graph, "up": res, "down": down[::-1]}
    save(out, f"driven/{a.model}_r{a.r:g}_orn{a.orn:g}{a.tag}.json")


def task_lnaqs(a):
    torch = __import__("torch")
    net = S.Net(a.graph, model=a.model)
    h = torch.zeros(net.N, device=net.dev, dtype=torch.float64)
    gs = sorted([float(x) for x in a.g.split(",")], reverse=True)
    res = []; st = None
    for g in gs:
        st = S.qmf_solve(net, g, a.r, h, None if st is None else st["rho"], None if st is None else st["O"],
                         iters=a.iters, tol=1e-11, eta=0.5)
        row = {"g": g}; row.update(shares(net, st["rho"]))
        if row["rho_tot"] < 1e-6:
            res.append(row); continue
        L = S.LNA(net, g, a.r, st)
        obs = L.obs_activity([np.ones(net.N, bool), net.masks["AL_module"], net.masks["nonORN"]])
        o = L.run(obs, maxlag=a.maxlag, ckpt=150, smax=a.smax, rtol=1e-9)
        for j, nm in enumerate(("all", "AL", "nonORN")):
            row[f"var_{nm}"] = float(o["var"][j])
            row[f"acf_{nm}"] = S.acf_metrics(o["acf"][:, j])
            row[f"acf_{nm}_curve"] = (o["acf"][::5, j] / o["acf"][0, j]).tolist()
        row["cv_all"] = float(np.sqrt(o["var"][0]) / row["rho_mean"])
        row["lna_S"] = o["S"]; row["lna_conv"] = o["converged"]
        res.append(row)
        S.log(f"g {g} rho {row['rho_mean']:.4g} CV {row['cv_all']:.3f} P {row['acf_all']['period']:.1f} coh {row['acf_all']['coh_ratio']:.3f} "
              f"osc {row['acf_all']['osc']} tint {row['acf_all']['tau_int']:.1f} S {o['S']}")
    save({"task": "lnaqs", "model": a.model, "r": a.r, "rows": res}, f"lnaqs/{a.model}_r{a.r:g}{a.tag}.json")


def bicgstab(mv, b, x0=None, tol=1e-9, maxit=20000):
    """Plain BiCGSTAB for A x = b (A given by mv), torch vectors."""
    x = b.clone() if x0 is None else x0.clone()
    r = b - mv(x); rh = r.clone()
    rho = alpha = om = 1.0
    v = torch_zeros_like(b); p = torch_zeros_like(b)
    nb = float(b.norm())
    for it in range(maxit):
        rho1 = float((rh * r).sum())
        if rho1 == 0:
            break
        beta = (rho1 / rho) * (alpha / om) if it > 0 else 0.0
        p = r + beta * (p - om * v) if it > 0 else r.clone()
        v = mv(p)
        alpha = rho1 / float((rh * v).sum())
        s_ = r - alpha * v
        if float(s_.norm()) < tol * nb:
            x = x + alpha * p; return x, it + 1
        t = mv(s_)
        om = float((t * s_).sum()) / float((t * t).sum())
        x = x + alpha * p + om * s_
        r = s_ - om * t
        rho = rho1
        if float(r.norm()) < tol * nb:
            return x, it + 1
    return x, maxit


def torch_zeros_like(x):
    return x * 0


def task_pulse(a):
    """Pulse protocol (h = 0, one ORN seed when silent, gap G): quasi-static resources x between avalanches, solved
    self-consistently: M(x) = f(g E diag x)/alpha, F = (I - M)^{-1} sbar (mean onsets per avalanche per neuron), mean
    avalanche duration T_av from a Poisson Galton-Watson recursion with mean m = rho(M), onset rate nu = F/(T_av + T_gap),
    x = e(nu/alpha) (exact unit efficacy). Continuation in g (ascending), from rested resources at the first g."""
    torch = __import__("torch")
    net = S.Net(a.graph, model=a.model)
    orn = net.masks["ORN"]
    sbar = torch.tensor(orn / orn.sum(), device=net.dev, dtype=torch.float64)
    U, TAU = net.Upar, net.TAUpar
    gaps = [float(x) for x in a.gaps.split(",")]
    G = np.geomspace(a.gmin, a.gmax, a.ng)
    E = net.E
    res = []
    for gap in gaps:
        Tgap = S.DT if gap == 0 else S.DT / (1 - np.exp(-S.DT / gap))
        x = torch.ones(net.N, device=net.dev, dtype=torch.float64)
        v = None; F = None
        for g in G:
            t0 = time.time()
            hist = []; it = 0; dx = np.nan; nbi = 0
            for it in range(a.iters):
                vals = S.f_branch(g * E["val"] * x[E["col"]], torch) / S.A_EFF
                M = torch.sparse_csr_tensor(E["crow"], E["col"], vals, (net.N, net.N))
                lam, v, _ = S.perron(net, g, x, x0=v, iters=20000, tol=1e-10)
                m = lam / S.A_EFF
                if m < 1 - 1e-7:
                    F, nbi = bicgstab(lambda y: y - torch.mv(M, y), sbar, F if F is not None and float(F.min()) >= 0 else None)
                    F = torch.clamp(F, min=0)
                    big = False
                else:                                   # supercritical at these resources: an avalanche never ends
                    _, uu, _ = S.perron(net, g, x, True, v, iters=20000, tol=1e-10)
                    uu = uu / float((uu * v).sum())
                    F = sbar + v * float((uu * sbar).sum()) / 1e-7; big = True
                mm = min(m, 1 - 1e-7); qk = 1.0; ngen = 1.0
                for _ in range(2000000):
                    qk = 1 - np.exp(-mm * qk); ngen += qk
                    if qk < 1e-10:
                        break
                Tav = ngen / S.A_EFF
                nu = F / (Tav + Tgap)
                rho_eq = torch.clamp(nu / S.A_EFF, max=0.5)
                if U > 0:
                    xn = torch.tensor(S.eff_spike(rho_eq.cpu().numpy(), U, TAU), device=net.dev, dtype=torch.float64)
                    xn = torch.where(net.depm, xn, torch.ones_like(xn))
                else:
                    xn = torch.ones_like(x)
                dx = float((xn - x).abs().max())
                x = (1 - a.eta) * x + a.eta * xn
                hist.append(m)
                if it > 5 and dx < 1e-5 and not big:
                    break
            method = "direct"
            if not (dx < 1e-5 and not big) and U > 0:
                # near-critical fallback: bisection on eps with the mode form F = sbar + v (u.sbar)/eps
                method = "bisect"
                for _pass in range(3):
                    lam, v, _ = S.perron(net, g, x, x0=v, iters=20000, tol=1e-10)
                    _, uu, _ = S.perron(net, g, x, True, v, iters=20000, tol=1e-10)
                    uu = uu / float((uu * v).sum()); us = float((uu * sbar).sum())

                    def x_of(eps_):
                        qk = 1.0; ngen_ = 1.0; mm_ = 1 - eps_
                        for _ in range(2000000):
                            qk = 1 - np.exp(-mm_ * qk); ngen_ += qk
                            if qk < 1e-10:
                                break
                        Fm = sbar + v * us / eps_
                        rr_ = torch.clamp(Fm / (ngen_ / S.A_EFF + Tgap) / S.A_EFF, max=0.5)
                        xx = torch.tensor(S.eff_spike(rr_.cpu().numpy(), U, TAU), device=net.dev, dtype=torch.float64)
                        return torch.where(net.depm, xx, torch.ones_like(xx)), Fm, ngen_
                    lo, hi = 1e-8, 0.5
                    vv_ = v
                    for _b in range(40):
                        mid = np.sqrt(lo * hi)
                        xm, _, _ = x_of(mid)
                        lm, vv_, _ = S.perron(net, g, xm, x0=vv_, iters=20000, tol=1e-10)
                        if lm / S.A_EFF + mid - 1 > 0:     # still supercritical at this depletion -> need smaller eps
                            hi = mid
                        else:
                            lo = mid
                    eps_b = np.sqrt(lo * hi)
                    x, F, ngen = x_of(eps_b)
                    Tav = ngen / S.A_EFF
            lam, v, _ = S.perron(net, g, x, x0=v, iters=30000, tol=1e-11)
            _, u, _ = S.perron(net, g, x, True, v, iters=30000, tol=1e-11)
            u = u / float((u * v).sum())
            uv2 = float((u * v * v).sum()); u2v = float((u * u * v).sum())
            Ks = float(v.sum()) / uv2
            eps = 1 - lam / S.A_EFF
            Fn = F.cpu().numpy(); xa = x.cpu().numpy()
            row = {"gap": gap, "g": float(g), "m": lam / S.A_EFF, "eps": eps, "mean_size": float(Fn.sum()),
                   "T_av": Tav, "T_gap": Tgap, "n_gen": ngen, "K_s": Ks, "K0": 1 / (u2v * uv2), "it": it + 1,
                   "dx": dx, "bicg_it": nbi, "share_AL_F": float(Fn[net.masks["AL_module"]].sum() / Fn.sum()),
                   "x_AL": float(xa[net.masks["AL_module"] & net.dep_node].mean()) if U > 0 else 1.0,
                   "x_min": float(xa[net.dep_node].min()) if U > 0 else 1.0,
                   "S_eps": 1 / eps ** 2 if eps > 0 else np.inf, "S_dep": (Ks / U) ** (2 / 3) if U > 0 else np.inf,
                   "m_hist_tail": hist[-5:], "method": method, "sec": time.time() - t0}
            vv = (v * v).cpu().numpy(); row["PR_v"] = float(vv.sum() ** 2 / (vv ** 2).sum()); row["AL_v"] = float(vv[net.masks["AL_module"]].sum() / vv.sum())
            res.append(row)
            S.log(f"gap {gap} g {g:.4f} m {row['m']:.6f} eps {eps:.2e} <S> {row['mean_size']:.1f} Tav {Tav:.2f} Ks {Ks:.0f} "
                  f"S_eps {row['S_eps']:.3g} S_dep {row['S_dep']:.3g} AL_F {row['share_AL_F']:.2f} xAL {row['x_AL']:.3f} it {it+1} dx {dx:.1e} {method} {row['sec']:.0f}s")
    save({"task": "pulse", "model": a.model, "rows": res}, f"pulse/{a.model}{a.tag}.json")


def task_multistart(a):
    """Mean-field uniqueness test (de Candia coexistence / broken L-R symmetry): bQMF (and QMF at r = 0) fixed points at
    h = 0 (and ORN drive) from several starts; report pairwise L1 distances of the final activity vectors and lobe
    asymmetries."""
    torch = __import__("torch")
    net = S.Net(a.graph, model=a.model)
    N = net.N
    side = net.side
    AL = net.masks["AL_module"]
    ALL = AL & (side == "L"); ALR = AL & (side == "R")
    starts = {"all": np.full(N, 0.5), "AL_L": np.where(ALL, 0.5, 0.0), "AL_R": np.where(ALR, 0.5, 0.0),
              "CB": np.where(net.masks["CB"], 0.5, 0.0), "VNC": np.where(net.masks["VNC"], 0.5, 0.0),
              "rand1pc": (np.random.default_rng(1).random(N) < 0.01) * 0.5}
    res = []
    for drive in (0.0, S.H_ORN):
        hl = net.hvec(drive)
        hlam = torch.tanh(hl) * S.DT
        for g in [float(x) for x in a.g.split(",")]:
            fin = {}
            for nm, x0 in starts.items():
                r0 = torch.tensor(x0, device=net.dev, dtype=torch.float64)
                st = S.bqmf_solve(net, g, hlam, r0, r0.clone(), iters=a.iters, tol=1e-11, eta=0.5)
                fin[nm] = st["rho"].cpu().numpy()
            names = list(fin)
            tot = {k: float(v.sum()) for k, v in fin.items()}
            dist = {f"{x}|{y}": float(np.abs(fin[x] - fin[y]).sum() / max(fin[x].sum() + fin[y].sum(), 1e-300) * 2)
                    for i, x in enumerate(names) for y in names[i + 1:]}
            asym = {k: float((v[ALL].sum() - v[ALR].sum()) / max(v[ALL].sum() + v[ALR].sum(), 1e-300)) for k, v in fin.items()}
            row = {"g": g, "h_orn": drive, "rho_tot": tot, "max_rel_L1": max(dist.values()), "dist": dist, "AL_asym": asym}
            res.append(row)
            S.log(f"h {drive} g {g} totals {[round(t, 2) for t in tot.values()]} maxL1 {row['max_rel_L1']:.2e} asym {[round(x, 4) for x in asym.values()]}")
    save({"task": "multistart", "model": a.model, "rows": res}, f"multistart/{a.model}{a.tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--model", default="M2"); ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--graph", default=None); ap.add_argument("--tag", default="")
    ap.add_argument("--gmin", type=float, default=0.17); ap.add_argument("--gmax", type=float, default=12.0)
    ap.add_argument("--ng", type=int, default=48); ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--transient", action="store_true"); ap.add_argument("--tsteps", type=int, default=2000); ap.add_argument("--tevery", type=int, default=1)
    ap.add_argument("--orn", type=float, default=0.04); ap.add_argument("--lna", action="store_true")
    ap.add_argument("--maxlag", type=int, default=2500); ap.add_argument("--smax", type=int, default=30000)
    ap.add_argument("--g", default=""); ap.add_argument("--glist", default=""); ap.add_argument("--gaps", default="0,10,100")
    ap.add_argument("--closure", default="bqmf"); ap.add_argument("--peel", type=int, default=0); ap.add_argument("--eta", type=float, default=0.3); ap.add_argument("--nmax", type=int, default=200000)
    a = ap.parse_args()
    {"decay": task_decay, "driven": task_driven, "lnaqs": task_lnaqs, "pulse": task_pulse, "multistart": task_multistart}[a.task](a)
