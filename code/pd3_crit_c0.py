"""pd3 criticality C0 (CPU, no new simulation): field response of the AL module in the existing driven runs.

Inputs: results/pd2/drive_h{1e-5,1e-4,1e-3,1e-2,1e-1}_real.npz (swc.py --exp decay with drive h; r = 1, B = 32, T = 1000,
40 g in 0.1-3; occ[g, N] = per-neuron activity averaged over replicas and the second half t in (500, 1000], float16).
Per (g, h): n_AL = sum over the 526 AL-module neurons of occ (= mean number of active AL neurons), n_tot = sum over all
neurons, and from rho[g, t, B] (per replica) the mean and replica SE of the total active count over the recorded t > 500.
Local slopes d ln n / d ln h between consecutive h (one decade apart).
Writes results/pd3/criticality/c0_drive.json and prints a table at the three g that bracket g_lin = 0.16315.
"""
import json
import numpy as np

ROOT = "/MALECNS_ROOT"
G_LIN = 0.16315274988719286
hs = ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]
al = np.load(f"{ROOT}/results/module/module_graph.npz")["idx"]
rows = {}
gref = None
for hs_ in hs:
    d = np.load(f"{ROOT}/results/pd2/drive_h{hs_}_real.npz", allow_pickle=True)
    meta = json.loads(str(d["meta"])); g = d["g"]; t = d["t"]
    if gref is None:
        gref = g
    assert np.allclose(g, gref) and meta["r"] == 1.0 and meta["h"] == float(hs_)
    occ = d["occ"].astype(np.float64)
    N = occ.shape[1]
    sel = t > 500
    ntot_rep = d["rho"][:, sel, :].astype(np.float64).mean(1) * N            # [g, B] per-replica time mean
    rows[hs_] = {"h": float(hs_), "n_AL": occ[:, al].sum(1).tolist(), "n_tot_occ": occ.sum(1).tolist(),
                 "n_tot_rho_mean": ntot_rep.mean(1).tolist(),
                 "n_tot_rho_se": (ntot_rep.std(1, ddof=1) / np.sqrt(ntot_rep.shape[1])).tolist(),
                 "AL_share": (occ[:, al].sum(1) / np.maximum(occ.sum(1), 1e-300)).tolist()}
H = np.array([rows[h]["h"] for h in hs])
nAL = np.array([rows[h]["n_AL"] for h in hs])                                 # [h, g]
ntot = np.array([rows[h]["n_tot_rho_mean"] for h in hs])
slope_AL = np.diff(np.log(nAL), axis=0) / np.diff(np.log(H))[:, None]          # [h-1, g]
slope_tot = np.diff(np.log(ntot), axis=0) / np.diff(np.log(H))[:, None]
out = {"source": [f"results/pd2/drive_h{h}_real.npz" for h in hs], "g": gref.tolist(), "x": (gref / G_LIN).tolist(),
       "g_lin": G_LIN, "r": 1.0, "rows": rows, "h_mid": np.sqrt(H[1:] * H[:-1]).tolist(),
       "slope_nAL": slope_AL.tolist(), "slope_ntot": slope_tot.tolist(),
       "note": "occ is float16 (per-neuron values ~1e-5 are subnormal: ~0.6% resolution at 1e-5); n_AL has no replica SE "
               "(occ is a replica mean); n_tot from rho has a replica SE"}
json.dump(out, open(f"{ROOT}/results/pd3/criticality/c0_drive.json", "w"), indent=1)
print("g      x     | n_AL at h = " + "  ".join(hs) + " | slopes d ln n_AL / d ln h")
for k, gg in enumerate(gref):
    if 0.12 < gg < 0.45:
        print(f"{gg:.4f} {gg/G_LIN:.3f} | " + " ".join(f"{nAL[j, k]:8.4g}" for j in range(len(hs))) + " | "
              + " ".join(f"{slope_AL[j, k]:.3f}" for j in range(len(hs) - 1)) + "   tot: "
              + " ".join(f"{slope_tot[j, k]:.3f}" for j in range(len(hs) - 1)))
