"""Build explainer/flow.html (phase diagram with smooth theory curves, reduced-model flow, live module simulation) from the
theory and sweep outputs. Theory boundaries per r (real connectome, branching variant, T = 3000, phase.py's definitions):
  Q -> S where the module's predicted lifetime first reaches 200, S -> L where it reaches T = 3000 (theory_branch_gpu.py, dense r),
  L -> D where the mean-field (QMF) activity's participation ratio first reaches N/4 (theory_qmf.py, qmf_real + qmf_real_dense).
Simulated boundaries (phase.py / results/pd/phase.json): first g with all replicas alive at T, first D.
Module graph for the in-browser simulation: results/module/module_graph.npz (module_sim.py).
Writes explainer/flow.html and the scratchpad copy used for publishing.
"""
import base64, glob, json, os, shutil
import numpy as np

ROOT = "/MALECNS_ROOT"
PD = f"{ROOT}/results/pd"
SCR = "/MALECNS_ROOT/explainer_build"
parts = [json.load(open(f)) for f in sorted(glob.glob(f"{PD}/branch_real_dense_part*.json"))]
RS = parts[0]["r"]; g_lin = parts[0]["g_lin"]; a_eff = parts[0]["alpha_eff"]
rows = sorted([x for p in parts for x in p["rows"]], key=lambda x: x["g"])
G = np.array([x["g"] for x in rows])
fp = json.load(open(f"{PD}/flow_params.json"))
ph = json.load(open(f"{PD}/phase.json"))

# mean-field delocalization line from both QMF runs
gD = {}
for f in [f"{PD}/qmf_real.npz", f"{PD}/qmf_real_dense.npz"]:
    q = np.load(f); QG, QR, QPR, QRHO = q["g"], q["r"], q["PR"][:, 0, :], q["rho_T"][:, 0, :]
    for i, r in enumerate(QR):
        k = [j for j in range(len(QG)) if QRHO[i, j] > 1e-9 and QPR[i, j] >= 0.25]
        if k and k[0] > 0 and QRHO[i, k[0] - 1] > 1e-9:      # interpolate the N/4 crossing in log g
            a, b = QPR[i, k[0] - 1], QPR[i, k[0]]; fr = (0.25 - a) / (b - a)
            gD[round(float(r), 4)] = float(np.exp(np.log(QG[k[0] - 1]) + fr * np.log(QG[k[0]] / QG[k[0] - 1])))
        else:
            gD[round(float(r), 4)] = float(QG[k[0]]) if k else None
jR = int(np.argmin([abs(x["R"] - 2) for x in rows]))
K0 = rows[jR]["K_r0"]
theory, life_v = [], []
for r in RS:
    lt = np.array([x[f"logtau_r{r:g}"] for x in rows])
    def first(thr):                       # crossing of ln tau = ln thr, interpolated in log g between grid points
        hit = lt >= np.log(thr)
        if not hit.any():
            return None
        k = int(np.argmax(hit))
        if k == 0 or not np.isfinite(lt[k - 1]):
            return float(G[k])
        fr = (np.log(thr) - lt[k - 1]) / (lt[k] - lt[k - 1])
        return float(np.exp(np.log(G[k - 1]) + fr * np.log(G[k] / G[k - 1])))
    theory.append({"r": r, "gS": first(200), "gon": first(3000), "gD": gD.get(round(r, 4)),
                   "delta": K0 / rows[jR][f"K_r{r:g}"] - 1})
    life_v.append([float(v / np.log(10)) if np.isfinite(v) else -1.0 for v in lt])
# QMF r values that are not in the dense list still contribute to the delocalization curve
for r, v in gD.items():
    if not any(abs(t["r"] - r) < 1e-9 for t in theory):
        theory.append({"r": r, "gS": None, "gon": None, "gD": v, "delta": None})
theory.sort(key=lambda t: t["r"])
def monotone(key):                        # boundaries cannot move left as inhibition grows
    best = 0.0
    for t in theory:
        if t[key] is not None:
            t[key] = best = max(best, t[key])
for key in ["gS", "gon", "gD"]:
    monotone(key)
    # light smoothing in log g (two passes of a 1-2-1 filter, endpoints fixed): removes the jitter that the 99% truncation of
    # the Perron vectors puts into the coincidence term, without moving the curve
    idx = [i for i, t in enumerate(theory) if t[key] is not None]
    for _ in range(2):
        lv = [np.log(theory[i][key]) for i in idx]
        for j in range(1, len(idx) - 1):
            theory[idx[j]][key] = float(np.exp(0.25 * lv[j - 1] + 0.5 * lv[j] + 0.25 * lv[j + 1]))
    monotone(key)
dr = [t for t in theory if t["delta"] is not None]
for t in theory:                         # fill delta for QMF-only rows by linear interpolation
    if t["delta"] is None:
        t["delta"] = float(np.interp(t["r"], [d["r"] for d in dr], [d["delta"] for d in dr]))
sim = []
fsb = f"{ROOT}/results/analysis2/flow_sim_boundaries.json"
if os.path.exists(fsb):                   # refined sweeps, same lifetime criteria as the theory (analysis2_phase.py)
    for x in json.load(open(fsb))["rows"]:
        sim.append({"r": float(x["r"]), "S": x.get("S"), "on": x.get("on"), "D": x.get("D")})
else:                                     # stage-1 sweep, phase.py criteria
    for r, rr in sorted(ph["real"].items(), key=lambda kv: float(kv[0])):
        on = [x["g"] for x in rr if x["surv_T"] == 1]; dd = [x["g"] for x in rr if x["phase"] == "D"]
        sim.append({"r": float(r), "S": None, "on": on[0] if on else None, "D": dd[0] if dd else None})

# module graph: CSC by presynaptic neuron, so the browser pushes input only from active neurons
mg = np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)
n = len(mg["idx"]); pre, post, w = mg["pre"].astype(np.int64), mg["post"].astype(np.int64), mg["w"].astype(np.int64)
o = np.lexsort((post, pre)); pre, post, w = pre[o], post[o], w[o]
colptr = np.zeros(n + 1, np.int32); np.add.at(colptr, pre + 1, 1); colptr = np.cumsum(colptr).astype(np.int32)
top = np.argsort(-w)[:1500]
P = mg["pos"]; sides = mg["sides"]
xl, xr = P[sides == "L", 0].mean(), P[sides == "R", 0].mean()
lab_left, lab_right = ("fly's left antennal lobe", "fly's right antennal lobe") if xl < xr else ("fly's right antennal lobe", "fly's left antennal lobe")
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
graph = {"n": int(n), "c": float(mg["c"]), "colptr": b64(colptr.astype(np.int32)), "rows": b64(post.astype(np.int16)),
         "w": b64(w.astype(np.uint16)), "sign": b64(mg["sign"].astype(np.int8)), "px": b64(P[:, 0].astype(np.float32)),
         "py": b64(P[:, 1].astype(np.float32)), "top": b64(np.vstack([pre[top], post[top]]).T.ravel().astype(np.int32)),
         "leftLabel": lab_left, "rightLabel": lab_right}
v = np.load(f"{PD}/mode_real_r0_v.npy").astype(np.float64); v /= np.linalg.norm(v)
kappa = float((1 / rows[jR]["uv2"]) * v[mg["idx"]].sum() / n)
data = {"g_lin": g_lin, "theory": theory, "life": {"r": RS, "g": G.tolist(), "v": life_v}, "sim": sim, "graph": graph,
        "params": {"lambda_M": fp["lambda_M"], "n_M": fp["n_M"], "e_MB": fp["e_MB"], "e_BM": fp["e_BM"], "K0": K0, "alpha_eff": a_eff, "kappa": kappa}}
html = open(f"{ROOT}/explainer/flow_template.html").read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
open(f"{ROOT}/explainer/flow.html", "w").write(html)
if os.path.isdir(SCR):
    shutil.copy(f"{ROOT}/explainer/flow.html", f"{SCR}/flow.html")
print(f"flow.html {len(html)/1e3:.0f} kB; module n={n}, edges={len(pre)}; K0={K0:.1f}; kappa={kappa:.3f}")
for t in theory[::3]:
    print(f"r={t['r']:5.3f}  S {t['gS']}  on {t['gon']}  D {t['gD']}  delta {t['delta']:.2f}")
