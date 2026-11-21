"""E3 theory, third variant (GPU): MESOSCOPIC stochastic model = the birth-death chain of the onset theory with explicit,
slow per-neuron resources (spike-mode depression), simulated directly. Sits between the two analytic limits of
code/ald_dep_theory.py (frozen resource / adiabatic resource) and needs no assumption about which one applies.

 state per replica: n (active effective units, 0..n_top = floor(K)), resources x_j of the top-S mode neurons (by u_j v_j).
 per step dt = 0.2 (the simulator's step):
   rho_j = min(n <u^2,v> v_j, 0.5);  X = sum_j w_j x_j (w = u_j v_j over the support, normalised; x_j = 1 for
   non-depressing neurons);  births ~ Poisson(lam X n (1 - n/K) dt), deaths ~ Binomial(n, p_off), n capped at n_top;
   resources (Poisson onsets at rate nu_j = alpha_eff rho_j, spike mode, exact over the step):
   x_j <- x_inf + (x_j - x_inf) exp(-(1/TAU + U nu_j) dt),  x_inf = (1/TAU) / (1/TAU + U nu_j).
 start: n = n_top (fully active) with x_j = 1 - U (every neuron had an onset at t = 0), as in the simulator.
 output per g: censored-exponential MLE of the extinction time (censored at 3000), B replicas; quasi-stationary n among
 survivors (1500 < t <= 3000) and the AL-module activity it implies.
 CHECK: with U = 0 the model is the plain chain; its tau must match the exact recursion (log_tau) up to the discrete-time
 error (reported in the output as base_check).
Usage: python ald_dep_meso.py <spec.json> <out.json>   spec = {"kappa": {"0": k0, "1": k1}, "arms": {name: {r, U, TAU, group}}}
"""
import json, sys, time
import numpy as np
import torch

sys.path.insert(0, "/MALECNS_ROOT/code")
import ald_dep_theory as TH

dev = "cuda" if torch.cuda.is_available() else "cpu"
S_TOP, B, T = 1500, 1024, 3000.0
DT = 0.2; P_OFF = 1 - np.exp(-DT); A_EFF = P_OFF / DT
spec = json.load(open(sys.argv[1])); outp = sys.argv[2]
th = TH.Theory()
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:.0f}s]", *a, flush=True)
gen = torch.Generator(device=dev); gen.manual_seed(20261109)


def run(i, r, U, TAU, group, kappa):
    lam, K = th.lam[i], kappa * th.K[r][i]
    n_top = int(np.floor(K))
    uv, v, idx = th.uv[i], th.v[i], th.idx[i]
    o = np.argsort(-uv)[:S_TOP]
    w = uv[o] / uv[o].sum()
    dep = th.dep_mask(group)[idx[o]] if U > 0 else np.zeros(len(o), bool)
    modm = th.mod[idx[o]]
    tw = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)
    w_t, v_t, dep_t = tw(w), tw(v[o]), tw(dep.astype(float))
    n = torch.full((B,), float(n_top), device=dev)
    x = torch.where(dep_t > 0, 1 - U, 1.0)[None, :].repeat(B, 1)
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    ext = torch.full((B,), T, device=dev)
    steps = int(round(T / DT)); qs_sum = torch.zeros(B, device=dev); qs_k = 0
    rinv = 0.0 if not np.isfinite(TAU) else 1.0 / TAU
    for s in range(1, steps + 1):
        rho = torch.clamp(n[:, None] * th.u2v[i] * v_t[None, :], max=0.5)
        X = (x * w_t[None, :]).sum(1)
        b = lam * X * n * torch.clamp(1 - n / K, min=0) * DT
        births = torch.poisson(b, generator=gen)
        deaths = torch.binomial(n, torch.full_like(n, P_OFF), generator=gen)
        n = torch.clamp(n + births - deaths, min=0, max=n_top)
        if U > 0:
            nu = A_EFF * rho
            k = rinv + U * nu
            xin = torch.where(k > 0, rinv / torch.clamp(k, min=1e-30), torch.ones_like(k))
            xn = xin + (x - xin) * torch.exp(-k * DT)
            x = torch.where(dep_t[None, :] > 0, xn, x)
        died = alive & (n == 0)
        ext = torch.where(died, torch.full_like(ext, s * DT), ext); alive = alive & (n > 0)
        if s * DT > 1500:
            qs_sum += n; qs_k += 1
        if not alive.any():
            break
    d = int((~alive).sum()); E = float(ext.sum())
    tau = E / d if d > 0 else float("inf")
    al = alive.cpu().numpy()
    n_qs = float((qs_sum.cpu().numpy()[al] / max(qs_k, 1)).mean()) if al.any() else 0.0
    rho_all = np.clip(n_qs * th.u2v[i] * th.v[i], 0, 0.5)
    return {"g": float(th.g[i]), "tau": tau, "deaths": d, "alive": float(al.mean()), "tau_reg": E / max(d, 0.5),
            "n_qs": n_qs, "AL_rho_qs": float(rho_all[th.mod[th.idx[i]]].sum() / th.mod.sum()), "K": K, "n_top": n_top}


res = {"spec": spec, "S_TOP": S_TOP, "B": B, "T": T, "arms": {}}
for name, a in spec["arms"].items():
    kap = spec["kappa"][str(a["r"])]
    rows = []
    for i in range(min(len(th.g), int(__import__("os").environ.get("MESO_NG", 10 ** 6)))):
        rows.append(run(i, a["r"], a.get("U", 0.0), a.get("TAU", np.inf), a.get("group"), kap))
        if a.get("U", 0.0) == 0:
            rows[-1]["lt_exact"] = TH.log_tau_smooth(lambda nn: th.lam[i] * np.ones_like(nn), kap * th.K[a["r"]][i])
    lt = [np.log(x["tau_reg"]) for x in rows]
    gg = th.g[:len(rows)]
    res["arms"][name] = {"rows": rows, "S": TH.cross(gg, lt, 200.0), "on": TH.cross(gg, lt, 3000.0)}
    log(name, "S", res["arms"][name]["S"], "on", res["arms"][name]["on"])
    json.dump(res, open(outp, "w"), indent=1, default=float)
log("saved", outp)
