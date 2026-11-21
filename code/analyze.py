"""Slow-window analysis of the stochastic Wilson-Cowan runs (raw norm, real vs configuration null).
For each coupling g:
  - fate: lifetime (all replicas extinct) or sustained;
  - decay law of rho(t) from a fully active start, over the decaying stretch (t >= 5, rho > 0, before the last 10% of
    the lifetime): power law log rho = a + b log t versus exponential log rho = a - t/tau. R^2 of each is reported;
    the exponent b is the Griffiths diagnostic (it should vary continuously with g over a window);
  - localization of the activity that survives (occ, second half of the run): participation ratio, share in the top-1000
    neurons of the leading excitatory eigenvector (the antennal-lobe module), and top superclasses/types.
Spreading runs: survival exponent from P(t) ~ t^-delta, and the slope of N(t).
Writes results/analysis.json and prints a table.
"""
import json, os
import numpy as np

ROOT = "/MALECNS_ROOT"
R = f"{ROOT}/results"
C = "/MALECNS_CACHE"
sup = np.load(C + "/superclasses.npy"); typ = np.load(C + "/types.npy")
V = np.load(f"{R}/sym_exc_weighted_vecs_top10.npy")
lead = np.argsort(-V[:, 0] ** 2)[:1000]
mod_mask = np.zeros(len(sup), bool); mod_mask[lead] = True
top10 = np.zeros(len(sup), bool)
for j in range(10):
    top10[np.argsort(-V[:, j] ** 2)[:300]] = True


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)


def decay_fit(t, rho):
    life = t[np.argmax(rho == 0)] if (rho == 0).any() else np.inf
    end = 0.9 * life if np.isfinite(life) else t[-1]
    m = (t >= 5) & (t <= end) & (rho > 0)
    if m.sum() < 6:
        return life, np.nan, np.nan, np.nan, np.nan
    lt, lr = np.log(t[m]), np.log(rho[m])
    b, a = np.polyfit(lt, lr, 1); r2p = r2(lr, a + b * lt)
    k, c = np.polyfit(t[m], lr, 1); r2e = r2(lr, c + k * t[m])
    return life, b, r2p, -1 / k if k < 0 else np.inf, r2e


out = {}
for name in ["fine_decay_raw_r1", "fine_decay_raw_r1_null", "upper_decay_raw_r1"]:
    if not os.path.exists(f"{R}/{name}.npz"):
        print("missing", name); continue
    d = np.load(f"{R}/{name}.npz")
    t, g, rho, surv = d["t"], d["g"], d["rho"].mean(-1), d["surv"].mean(-1)
    occ = d["occ"].astype(np.float64) if "occ" in d.files else None
    rows = []
    print(f"\n== {name}")
    print(" g       life    sustained  b(power)  R2pow  tau(exp)  R2exp   rho(T)    PR(occ)  share_AL  share_top10modes  top superclasses")
    for i, gg in enumerate(g):
        life, b, r2p, tau, r2e = decay_fit(t, rho[i])
        row = {"g": float(gg), "lifetime": float(life), "survive_T": float(surv[i, -1]), "rho_T": float(rho[i, -1]),
               "b_power": float(b), "r2_power": float(r2p), "tau_exp": float(tau), "r2_exp": float(r2e)}
        if occ is not None and occ[i].sum() > 0:
            o = occ[i]
            row["PR_occ"] = float(o.sum() ** 2 / (o ** 2).sum())
            row["share_AL_module"] = float(o[mod_mask].sum() / o.sum())
            row["share_top10modes"] = float(o[top10].sum() / o.sum())
            top = np.argsort(-o)[:300]
            sc, cnt = np.unique(sup[top], return_counts=True)
            row["top_superclasses"] = dict(sorted(zip(sc.tolist(), cnt.tolist()), key=lambda x: -x[1])[:3])
            ty, tc = np.unique(typ[top], return_counts=True)
            row["top_types"] = dict(sorted(zip(ty.tolist(), tc.tolist()), key=lambda x: -x[1])[:5])
        rows.append(row)
        print(f"{gg:6.3f} {life:8.0f} {surv[i,-1]:8.2f}  {b:8.2f}  {r2p:6.3f}  {tau:8.1f}  {r2e:6.3f}  {rho[i,-1]:8.2e}  "
              f"{row.get('PR_occ', float('nan')):8.0f}  {row.get('share_AL_module', float('nan')):7.2f}  "
              f"{row.get('share_top10modes', float('nan')):7.2f}   {row.get('top_superclasses', '')}")
    out[name] = rows

for name in ["spread_raw_r1", "spread_raw_r1_null"]:
    if not os.path.exists(f"{R}/{name}.npz"):
        print("missing", name); continue
    d = np.load(f"{R}/{name}.npz")
    t, g = d["t"], d["g"]
    rows = []
    print(f"\n== {name}")
    for i, gg in enumerate(g):
        P = d["surv"][i].mean(-1); n = d["rho"][i].mean(-1) * 167216
        m = (t >= 5) & (P > 0)
        delta = -np.polyfit(np.log(t[m]), np.log(P[m]), 1)[0] if m.sum() > 5 else np.nan
        mm = (t >= 5) & (n > 0)
        eta = np.polyfit(np.log(t[mm]), np.log(n[mm]), 1)[0] if mm.sum() > 5 else np.nan
        rows.append({"g": float(gg), "P_end": float(P[-1]), "delta": float(delta), "eta": float(eta),
                     "N_end": float(n[-1])})
        print(f"g={gg:.3f}  P(T)={P[-1]:.4f}  delta={delta:.2f}  eta(N)={eta:.2f}  N(T)={n[-1]:.1f}")
    out[name] = rows
json.dump(out, open(f"{R}/analysis.json", "w"), indent=1)
print("\nsaved", f"{R}/analysis.json")
