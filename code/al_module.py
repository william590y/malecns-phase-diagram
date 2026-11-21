"""Q1 of the AL data grounding: what exactly is the 526-neuron module, and which classes carry the leading modes?

Module = results/module/module_graph.npz idx (union of the 99% supports of the two leading RIGHT eigenvectors of the
r = 0 directed excitatory operator). Leading-mode weights |v|^2 from al_spectra.py full (right = where activity sits;
left = which neurons' activity matters most for the mode, i.e. good seeds / feeders).
Writes results/al_lit/grounding/module.json.
"""
import json
import numpy as np
import pandas as pd
from al_common import load, labels, C, ROOT, OUT

d = load(); L = labels(); N = L["N"]
pre, post, w, sign = d["pre"], d["post"], d["weight"], d["sign"]
idx = np.load(f"{ROOT}/results/module/module_graph.npz")["idx"]
inmod = np.zeros(N, bool); inmod[idx] = True
F = np.load(f"{OUT}/spec_full_P.npz")
PR0, PR1, PL0 = (F["P_right"][:, 0].astype(float), F["P_right"][:, 1].astype(float), F["P_left"][:, 0].astype(float))
bid = np.load(f"{C}/body_ids.npy")
ann = pd.read_feather("/MALECNS_RAW/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                      columns=["bodyId", "itoleeHl", "statusLabel", "somaSide", "rootSide"]).set_index("bodyId").reindex(bid)
ntt = pd.read_feather("/MALECNS_RAW/body-neurotransmitters-male-cns-v1.0.feather").drop_duplicates(
    "body").set_index("body").reindex(bid)
out = {"n_module": int(len(idx))}


def table(lab, mask=None):
    """Per label: neurons in the module, in the CNS, and |v|^2 shares of right mode 1/2 and left mode 1."""
    rows = {}
    for u_ in np.unique(lab[inmod] if mask is None else lab[mask]):
        m = lab == u_
        rows[str(u_)] = {"n_module": int((m & inmod).sum()), "n_cns": int(m.sum()),
                         "right1": float(PR0[m].sum()), "right2": float(PR1[m].sum()), "left1": float(PL0[m].sum())}
    return dict(sorted(rows.items(), key=lambda kv: -kv[1]["right1"]))


out["by_superclass"] = table(L["sup"])
out["by_category"] = table(L["cat"], inmod | (PL0 > 1e-4) | (PR0 > 1e-4))
out["by_nt"] = table(L["nt"])
sid = np.where(np.isin(L["sides"], ["L", "R"]), L["sides"], "other")
out["by_side"] = table(sid)
out["by_type_top25"] = dict(list(table(L["ty"]).items())[:25])
out["by_hemilineage"] = table(ann["itoleeHl"].fillna("none").astype(str).values)
out["by_status"] = table(ann["statusLabel"].astype(str).values)
# glomeruli
g = L["glom"]
upn = L["cat"] == "AL uPN"
out["uPN_glomeruli_in_module"] = {str(k): int(v) for k, v in pd.Series(g[upn & inmod]).value_counts().items()}
out["uPN_glomeruli_by_right1"] = dict(list(table(np.where(upn, g, "-"), upn).items())[:15])
orn = L["cat"] == "AL ORN"
out["ORN_total"] = int(orn.sum()); out["ORN_in_module"] = int((orn & inmod).sum())
out["ORN_glomeruli_counts_top20"] = {str(k): int(v) for k, v in pd.Series(g[orn]).value_counts().head(20).items()}
out["ORN_n_glomeruli"] = int(pd.Series(g[orn]).nunique())
gl = pd.DataFrame({"g": g[orn], "left1": PL0[orn], "right1": PR0[orn]}).groupby("g").sum().sort_values("left1", ascending=False)
out["ORN_glomeruli_by_left1_top15"] = {k: {"left1": float(r.left1), "right1": float(r.right1),
                                          "n": int((g[orn] == k).sum())} for k, r in gl.head(15).iterrows()}
out["ORN_left1_total"] = float(PL0[orn].sum()); out["ORN_right1_total"] = float(PR0[orn].sum())
hrn = L["cat"] == "AL HRN/TRN"
out["HRN_TRN_in_module"] = {str(k): int(v) for k, v in pd.Series(L["ty"][hrn & inmod]).value_counts().items()}
# LN types in module with NT evidence
ln = (L["cl"] == "ALLN") & inmod
rows = []
for t in pd.Series(L["ty"][ln]).value_counts().index:
    m = (L["ty"] == t) & (L["cl"] == "ALLN")
    sub = ntt[m]
    rows.append({"type": str(t), "n_cns": int(m.sum()), "n_module": int((m & inmod).sum()),
                 "consensus_nt": pd.Series(L["nt"][m]).value_counts().to_dict(),
                 "predicted_nt": sub["predicted_nt"].value_counts().to_dict(),
                 "pred_conf_mean": float(sub["predicted_nt_confidence"].mean()),
                 "celltype_predicted_nt": sub["celltype_predicted_nt"].value_counts().to_dict(),
                 "celltype_conf": float(sub["celltype_predicted_nt_confidence"].mean()),
                 "ground_truth": sub["ground_truth"].dropna().astype(str).value_counts().to_dict(),
                 "hemilineage": ann["itoleeHl"][m].fillna("none").value_counts().to_dict(),
                 "right1": float(PR0[m].sum()), "left1": float(PL0[m].sum())})
out["LN_types"] = sorted(rows, key=lambda r: -r["right1"])
# all ALLN by consensus NT (CNS-wide), and how much ACh-predicted LN mass exists
alln = L["cl"] == "ALLN"
out["ALLN_consensus_nt_cns"] = pd.Series(L["nt"][alln]).value_counts().to_dict()
out["ALLN_ground_truth_cns"] = ntt["ground_truth"][alln].dropna().astype(str).value_counts().to_dict()
# where the module's excitatory input comes from
exc = sign[pre] > 0
mi = exc & inmod[post]
src = np.where(inmod[pre[mi]], "module", L["cat"][pre[mi]])
s = pd.Series(w[mi]).groupby(src).sum().sort_values(ascending=False)
out["module_exc_input_sources"] = {k: float(v / s.sum()) for k, v in s.head(12).items()}
mo = exc & inmod[pre]
dst = np.where(inmod[post[mo]], "module", L["reg"][post[mo]])
s = pd.Series(w[mo]).groupby(dst).sum().sort_values(ascending=False)
out["module_exc_output_targets"] = {k: float(v / s.sum()) for k, v in s.head(10).items()}
# all-sign input of the module (how inhibitory is it?)
ma = inmod[post]
s = pd.Series(w[ma]).groupby(np.where(sign[pre[ma]] > 0, "exc", np.where(sign[pre[ma]] < 0, "inh", "mod"))).sum()
out["module_input_by_sign"] = {k: float(v / s.sum()) for k, v in s.items()}
s = pd.Series(w[ma & inmod[pre]]).groupby(np.where(sign[pre[ma & inmod[pre]]] > 0, "exc",
                                                    np.where(sign[pre[ma & inmod[pre]]] < 0, "inh", "mod"))).sum()
out["module_internal_by_sign"] = {k: float(v / s.sum()) for k, v in s.items()}
json.dump(out, open(f"{OUT}/module.json", "w"), indent=1, default=str)
print(json.dumps({k: out[k] for k in ["by_category", "by_nt", "by_side", "ORN_in_module", "ORN_left1_total",
                                      "module_exc_input_sources", "module_input_by_sign", "ALLN_consensus_nt_cns",
                                      "ALLN_ground_truth_cns"]}, indent=1, default=str))
for r in out["LN_types"][:8]:
    print(r)
