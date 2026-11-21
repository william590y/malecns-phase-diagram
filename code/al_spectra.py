"""Leading eigenpairs of the whole-CNS excitatory operator under structural perturbations (AL data grounding, Q2/Q3).

Usage: al_spectra.py <variant> [<variant> ...]     (run `al_spectra.py list` for the registry)
Writes results/al_lit/grounding/spec_<variant>.json (top eigenvalues, localization of the top right eigenvectors, and for
'full' also the left eigenvectors) and spec_<variant>_P.npz (|v|^2 of the top right [and left] eigenvectors, float32).
c is always the simulation's global scale from the untransformed weights (1/742.7), so eigenvalues of variants that keep
the weights are directly comparable with lambda_PF(E) = 5.574 and with g_lin ~ alpha_eff / lambda.
'peel' removes the region holding the leading mode, re-solves, and repeats (switch-on order of regions beyond the AL).
"""
import sys, json, time
import numpy as np
import pandas as pd
from al_common import load, labels, exc_operator, pf, describe_mode, OUT, ALPHA_EFF, C

t0 = time.time()
d = load()
L = labels()
N = L["N"]
pre, post = d["pre"], d["post"]
grp = np.full(N, "nonAL", object)
grp[L["cat"] == "AL ORN"] = "ORN"
grp[L["cat"] == "AL HRN/TRN"] = "HRN/TRN"
grp[np.isin(L["cat"], ["AL uPN", "AL mPN"])] = "PN"
grp[L["cl"] == "ALLN"] = "LN"
grp[L["cat"] == "AL IN/ON"] = "IN/ON"
grp = grp.astype(str)
G = {g: grp == g for g in np.unique(grp)}
isAL = L["reg"] == "AL"


def drop_edges(a, b):
    return ~(G[a][pre] & G[b][post])


def ct_sign():
    """Sign from the CELL-TYPE-level NT prediction instead of the body-level consensus (fallback: consensus)."""
    bid = np.load(f"{C}/body_ids.npy")
    nt = pd.read_feather("/MALECNS_RAW/body-neurotransmitters-male-cns-v1.0.feather",
                         columns=["body", "celltype_predicted_nt"]).drop_duplicates("body").set_index("body")
    ct = nt.reindex(bid)["celltype_predicted_nt"].fillna("").astype(str).str.lower().values
    s = d["sign"].copy()
    s[ct == "acetylcholine"] = 1
    s[np.isin(ct, ["gaba", "glutamate", "histamine"])] = -1
    s[np.isin(ct, ["dopamine", "serotonin", "octopamine"])] = 0
    return s


V = {
    "full": {},
    "no_ORN": {"keep_node": ~G["ORN"]},
    "no_AL_sensory": {"keep_node": ~(G["ORN"] | G["HRN/TRN"])},
    "cap50": {"transform": lambda w: np.minimum(w, 50.0)},
    "cap20": {"transform": lambda w: np.minimum(w, 20.0)},
    "cap10": {"transform": lambda w: np.minimum(w, 10.0)},
    "log1p": {"transform": np.log1p},
    "sqrt": {"transform": np.sqrt},
    "binary": {"transform": lambda w: np.ones_like(w)},
    "soma_only": {"keep_node": np.isfinite(L["pos"]).all(1)},
    "soma_cap50": {"keep_node": np.isfinite(L["pos"]).all(1), "transform": lambda w: np.minimum(w, 50.0)},
    "noORN_cap50": {"keep_node": ~G["ORN"], "transform": lambda w: np.minimum(w, 50.0)},
    "LN_nonexc": {"keep_edge": ~G["LN"][pre]},
    "ct_nt": {"sign": "ct"},
    "no_AL": {"keep_node": ~isAL},
    "drop_ORN>LN": {"keep_edge": drop_edges("ORN", "LN")},
    "drop_ORN>PN": {"keep_edge": drop_edges("ORN", "PN")},
    "drop_ORN>ORN": {"keep_edge": drop_edges("ORN", "ORN")},
    "drop_PN>LN": {"keep_edge": drop_edges("PN", "LN")},
    "drop_LN>PN": {"keep_edge": drop_edges("LN", "PN")},
    "drop_PN<>LN": {"keep_edge": drop_edges("PN", "LN") & drop_edges("LN", "PN")},
    "drop_LN>LN": {"keep_edge": drop_edges("LN", "LN")},
    "drop_PN>PN": {"keep_edge": drop_edges("PN", "PN")},
    "drop_LN>ORN": {"keep_edge": drop_edges("LN", "ORN")},
    "drop_PN>ORN": {"keep_edge": drop_edges("PN", "ORN")},
    "drop_HRN/TRN>any": {"keep_edge": ~G["HRN/TRN"][pre]},
    "drop_PN_presyn": {"keep_edge": ~G["PN"][pre]},
    "peel": {},
}
if sys.argv[1] == "list":
    print("\n".join(V)); sys.exit()


def solve(name, M, want_left=False, k=8):
    vals, vecs, P = pf(M, k=k)
    res = {"variant": name, "lam": vals.real.tolist(), "lam_im": vals.imag.tolist(), "nnz": int(M.nnz),
           "g_lin_naive": float(ALPHA_EFF / vals.real[0]),
           "modes": [describe_mode(P[:, j], L) for j in range(k)]}
    # the best mode NOT carried by the AL (AL share < 0.5) among the top k; None -> below lam[k-1]
    res["best_nonAL"] = None
    for j, m in enumerate(res["modes"]):
        if m["region"].get("AL", 0) < 0.5:
            res["best_nonAL"] = {"j": j, "lam": float(vals.real[j]), "lead": max(m["region"], key=m["region"].get),
                                 "types": list(m["types"])[:4]}
            break
    res["al_share_top"] = [m["region"].get("AL", 0) for m in res["modes"]]
    out = {"P_right": P[:, :2].astype(np.float32)}
    if want_left:
        lv, lvec, LP = pf(M, k=k, left=True)
        res["lam_left"] = lv.real.tolist()
        res["left_modes"] = [describe_mode(LP[:, j], L) for j in range(2)]
        out["P_left"] = LP[:, :2].astype(np.float32)
        out["v_right"] = np.real(vecs[:, :2] * np.sign(vecs[np.argmax(np.abs(vecs[:, 0])), 0].real)).astype(np.float64)
        out["v_left"] = np.real(lvec[:, :2] * np.sign(lvec[np.argmax(np.abs(lvec[:, 0])), 0].real)).astype(np.float64)
    return res, out


for name in sys.argv[1:]:
    spec = V[name]
    if name == "peel":
        keep = np.ones(N, bool); steps = []; Pk = {}
        for step in range(8):
            M, c = exc_operator(d, keep_node=keep)
            res, out = solve(f"peel{step}", M)
            p = out["P_right"][:, 0].astype(float)
            # remove the coarse region that carries most of the leading mode; 'CB other' is too coarse to remove whole,
            # so there remove the neurons of the leading mode's top cell types (by |v|^2) until 90% of it is gone
            top_reg = max(res["modes"][0]["region"], key=res["modes"][0]["region"].get)
            if top_reg in ("CB other", "VNC", "AN/DN", "unannotated"):
                tshare = pd.Series(p).groupby(L["ty"]).sum().sort_values(ascending=False)
                cut = tshare.index[: int(np.searchsorted(tshare.cumsum().values, 0.9) + 1)]
                rm = np.isin(L["ty"], cut) & (L["reg"] == top_reg)
                res["removed"] = f"{top_reg} types {list(map(str, cut))[:12]}"
            else:
                rm = L["reg"] == top_reg
                res["removed"] = f"region {top_reg}"
            res["removed_n"] = int((rm & keep).sum())
            steps.append(res); Pk[f"P{step}"] = out["P_right"]
            print(f"[{time.time()-t0:.0f}s] peel {step}: lam {res['lam'][:3]} lead {res['modes'][0]['region']} "
                  f"-> remove {res['removed']} ({res['removed_n']})", flush=True)
            keep &= ~rm
        json.dump({"variant": "peel", "steps": steps}, open(f"{OUT}/spec_peel.json", "w"), indent=1)
        np.savez_compressed(f"{OUT}/spec_peel_P.npz", **Pk)
        continue
    if spec.get("sign") == "ct":
        d2 = dict(d); d2["sign"] = ct_sign()
        M, c = exc_operator(d2)
    else:
        M, c = exc_operator(d, **spec)
    res, out = solve(name, M, want_left=(name == "full"))
    res["c"] = c
    if name == "ct_nt":
        res["n_changed_sign"] = int((d2["sign"] != d["sign"]).sum())
        res["AL_changed_sign"] = int(((d2["sign"] != d["sign"]) & isAL).sum())
    json.dump(res, open(f"{OUT}/spec_{name.replace('/', '-')}.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/spec_{name.replace('/', '-')}_P.npz", **out)
    m0 = res["modes"][0]
    print(f"[{time.time()-t0:.0f}s] {name}: lam {np.round(res['lam'][:4], 3)}  1/IPR {m0['inv_ipr']:.0f}  "
          f"region {list(m0['region'].items())[:3]}  cat {list(m0['cat'].items())[:3]}", flush=True)
