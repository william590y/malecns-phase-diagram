"""Q3 of the AL data grounding: which regions come next in the spectral tail (top-20 modes of the r = 0 directed
excitatory operator, results/pd/tail_real_r0_dir.npz), i.e. the order in which localized regions would switch on in
the linear (independent-island) picture, g_k ~ alpha_eff / lambda_k.
Writes results/al_lit/grounding/tail20.json.
"""
import json
import numpy as np
from al_common import labels, describe_mode, ROOT, OUT, ALPHA_EFF

L = labels()
t = np.load(f"{ROOT}/results/pd/tail_real_r0_dir.npz", allow_pickle=True)
P = t["P"]
rows = []
for k in range(20):
    p = P[:, k].astype(np.float64); p /= p.sum()
    m = describe_mode(p, L, ntop_types=5)
    m.update({"k": k + 1, "lam": float(t["lam"][k]), "lam_im": float(t["lam_im"][k]),
              "g_on_naive": float(ALPHA_EFF / t["lam"][k]), "lead_region": max(m["region"], key=m["region"].get)})
    rows.append(m)
    im = "" if m["lam_im"] == 0 else f" {m['lam_im']:+.3f}i"
    print(f"{k+1:2d} lam {m['lam']:.3f}{im} g_on {m['g_on_naive']:.3f} 1/IPR {m['inv_ipr']:6.0f} side {max(m['side'], key=m['side'].get)} "
          f"{ {k2: round(v, 2) for k2, v in list(m['region'].items())[:3]} } types {list(m['types'])[:4]}")
order = []
for r in rows:
    if r["lead_region"] not in [o["region"] for o in order]:
        order.append({"region": r["lead_region"], "first_mode": r["k"], "lam": r["lam"], "g_on_naive": r["g_on_naive"]})
json.dump({"modes": rows, "first_appearance": order, "lam_400": float(t["lam"][399])}, open(f"{OUT}/tail20.json", "w"), indent=1)
print(order)
