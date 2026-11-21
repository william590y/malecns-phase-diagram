"""Why does the antennal lobe dominate the excitatory spectrum? (AL data grounding, Q2 + region ordering for Q3)

1. Connection-weight statistics of the edges inside each annotated region (AL, MB, CX, LH, OL columns, VNC, rest):
   synapses per connection, reciprocity, within-region in-strength, and the Perron root of each region's internal
   excitatory subgraph (= the coupling at which that region alone would switch on, g_R ~ alpha_eff / lambda_R).
2. Exact decomposition of lambda_PF over edge categories, lambda = sum_ij u_i M_ij v_j / (u.v) (u, v = left/right PF
   vectors of the full operator from al_spectra.py full), by AL group pair and by cell-type pair.
3. Small subgraphs of the AL (type-restricted) and their Perron roots, e.g. lLN1_bc alone.
4. The strongest excitatory connections and 2-cycles of the CNS.
Writes results/al_lit/grounding/edges.json.
"""
import json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from al_common import load, labels, scale_c, OUT, ALPHA_EFF

t0 = time.time()
d = load(); L = labels(); N = L["N"]
pre, post, w, sign = d["pre"], d["post"], d["weight"], d["sign"]
c = scale_c(post, w, N)
exc = sign[pre] > 0
reg, cat, ty, cl = L["reg"], L["cat"], L["ty"], L["cl"]
log = lambda *a: print(f"[{time.time()-t0:.0f}s]", *a, flush=True)

# reverse-edge lookup
key = pre * N + post
o = np.argsort(key); ks = key[o]
rk = post * N + pre
pos_ = np.searchsorted(ks, rk); pos_[pos_ >= len(ks)] = 0
has_rev = ks[pos_] == rk
w_rev = np.where(has_rev, w[o][pos_], 0.0)
log("reverse lookup done")


def perron(nodes_mask, emask=None, transform=None, k=3):
    """Perron root and localization of the internal excitatory subgraph on nodes_mask."""
    m = exc & nodes_mask[pre] & nodes_mask[post]
    if emask is not None:
        m &= emask
    idx = np.where(nodes_mask)[0]
    loc = -np.ones(N, np.int64); loc[idx] = np.arange(len(idx))
    ww = w[m] if transform is None else transform(w[m])
    M = sp.csr_matrix((ww * c, (loc[post[m]], loc[pre[m]])), shape=(len(idx), len(idx)))
    if M.nnz == 0:
        return {"lam": 0.0}
    if len(idx) < 3000:
        ev, V = np.linalg.eig(M.toarray())
    else:
        ev, V = sla.eigs(M, k=k, which="LR", tol=1e-8, maxiter=200000, ncv=24)
    j = np.argmax(ev.real)
    p = np.abs(V[:, j]) ** 2; p /= p.sum()
    tt = ty[idx]
    u, inv = np.unique(tt, return_inverse=True); s = np.bincount(inv, weights=p)
    top = {str(u[i]): float(s[i]) for i in np.argsort(-s)[:5]}
    return {"lam": float(ev.real[j]), "g_on_naive": float(ALPHA_EFF / ev.real[j]) if ev.real[j] > 0 else None,
            "inv_ipr": float(1 / (p ** 2).sum()), "n": int(len(idx)), "top_types": top}


def edge_stats(m, nodes_mask):
    ww = w[m]
    if len(ww) == 0:
        return {}
    n = int(nodes_mask.sum())
    me = m & exc
    ins_within = np.bincount(post[me], weights=w[me], minlength=N)[nodes_mask]
    ins_all = np.bincount(post[exc], weights=w[exc], minlength=N)[nodes_mask]
    ins_tot = np.bincount(post, weights=w, minlength=N)[nodes_mask]
    ok = ins_all > 0
    return {
        "n_neurons": n, "n_edges": int(m.sum()), "n_synapses": float(ww.sum()), "density": float(m.sum() / max(n, 1) ** 2),
        "edges_per_neuron": float(m.sum() / max(n, 1)),
        "syn_per_edge_mean": float(ww.mean()), "syn_per_edge_median": float(np.median(ww)),
        "syn_per_edge_p90": float(np.percentile(ww, 90)), "syn_per_edge_p99": float(np.percentile(ww, 99)),
        "syn_per_edge_max": float(ww.max()),
        "frac_edges_ge50": float((ww >= 50).mean()), "frac_syn_in_edges_ge50": float(ww[ww >= 50].sum() / ww.sum()),
        "frac_edges_ge100": float((ww >= 100).mean()),
        "reciprocity_unweighted": float(has_rev[m].mean()),
        "reciprocity_weighted": float(np.minimum(ww, w_rev[m]).sum() / ww.sum()),
        "exc_frac_of_syn": float(w[me].sum() / ww.sum()),
        "exc_in_within_mean_c": float(ins_within.mean() * c), "exc_in_within_p90_c": float(np.percentile(ins_within, 90) * c),
        "exc_in_within_max_c": float(ins_within.max() * c),
        "exc_in_within_frac_of_exc_in": float(ins_within[ok].sum() / ins_all[ok].sum()),
        "in_strength_total_mean_syn": float(ins_tot.mean()),
    }


# --- 1. regions ------------------------------------------------------------------------------------------------
R = {}
regions = ["AL", "MB", "CX", "LH", "OL intrinsic", "VPN/VCN", "VNC", "CB other", "JO/mechano (brain)"]
for rname in regions:
    nm = reg == rname
    m = nm[pre] & nm[post]
    R[rname] = {"stats": edge_stats(m, nm), "perron": perron(nm),
                "perron_cap50": perron(nm, transform=lambda x: np.minimum(x, 50.0)),
                "perron_binary": perron(nm, transform=np.ones_like)}
    log(rname, R[rname]["perron"]["lam"], R[rname]["stats"]["syn_per_edge_mean"])
# AL without its sensory neurons, and AL LNs + PNs only
for rname, nm in {"AL minus ORN/HRN/TRN": (reg == "AL") & ~np.isin(cat, ["AL ORN", "AL HRN/TRN"]),
                  "MB KC only": cat == "MB KC",
                  "PN->KC (calyx input) + KC": (cat == "MB KC") | np.isin(cat, ["AL uPN", "AL mPN"])}.items():
    m = nm[pre] & nm[post]
    R[rname] = {"stats": edge_stats(m, nm), "perron": perron(nm)}
    log(rname, R[rname]["perron"]["lam"])
nonAL = reg != "AL"
R["CNS minus AL"] = {"stats": edge_stats(nonAL[pre] & nonAL[post], nonAL)}
allm = np.ones(N, bool)
R["whole CNS"] = {"stats": edge_stats(np.ones(len(pre), bool), allm)}
# calyx-proxy edge set PN->KC alone (feedforward; stats only)
m = np.isin(cat, ["AL uPN", "AL mPN"])[pre] & (cat == "MB KC")[post]
R["edges PN->KC"] = {"stats": edge_stats(m, cat == "MB KC")}
m = (cat == "MB KC")[pre] & (cat == "MB KC")[post]
R["edges KC->KC"] = {"stats": edge_stats(m, cat == "MB KC")}

# --- 2. exact lambda decomposition -----------------------------------------------------------------------------
F = np.load(f"{OUT}/spec_full_P.npz")
v, u = F["v_right"][:, 0], F["v_left"][:, 0]
dec = {"v_min": float(v.min()), "u_min": float(u.min())}
contrib = np.where(exc, u[post] * w * c * v[pre], 0.0) / (u @ v)
dec["lam_check"] = float(contrib.sum())
grp = np.full(N, "nonAL", object)
grp[cat == "AL ORN"] = "ORN"; grp[cat == "AL HRN/TRN"] = "HRN/TRN"
grp[np.isin(cat, ["AL uPN", "AL mPN"])] = "PN"; grp[cl == "ALLN"] = "LN"; grp[cat == "AL IN/ON"] = "IN/ON"
grp = grp.astype(str)
for name, lab in [("by_group", grp), ("by_cat", cat), ("by_type", ty)]:
    kk = np.char.add(np.char.add(lab[pre[exc]], " > "), lab[post[exc]])
    uu, inv = np.unique(kk, return_inverse=True)
    s = np.bincount(inv, weights=contrib[exc]); syn = np.bincount(inv, weights=w[exc])
    oo = np.argsort(-np.abs(s))[:30]
    dec[name] = [{"pair": str(uu[i]), "lam_contrib": float(s[i]), "share": float(s[i] / dec["lam_check"]),
                  "exc_synapses": float(syn[i])} for i in oo]
# same for the second (near-degenerate) mode
v2, u2 = F["v_right"][:, 1], F["v_left"][:, 1]
c2 = np.where(exc, u2[post] * w * c * v2[pre], 0.0) / (u2 @ v2)
kk = np.char.add(np.char.add(grp[pre[exc]], " > "), grp[post[exc]])
uu, inv = np.unique(kk, return_inverse=True); s = np.bincount(inv, weights=c2[exc])
dec["by_group_mode2"] = [{"pair": str(uu[i]), "share": float(s[i] / c2.sum())} for i in np.argsort(-np.abs(s))[:12]]
dec["lam2_check"] = float(c2.sum())
log("decomposition", dec["lam_check"], dec["by_type"][:3])

# --- 3. type-restricted AL subgraphs -----------------------------------------------------------------------------
sub = {}
for name, nm in {
        "lLN1_bc only": ty == "lLN1_bc",
        "lLN1 (all) only": cat == "AL lLN1",
        "lLN1 + lLN2": np.isin(cat, ["AL lLN1", "AL lLN2"]),
        "all AL LNs": cl == "ALLN",
        "AL PNs only": np.isin(cat, ["AL uPN", "AL mPN"]),
        "AL LNs + PNs": (cl == "ALLN") | np.isin(cat, ["AL uPN", "AL mPN"]),
        "AL ORNs only": cat == "AL ORN",
        "AL minus lLN1_bc": (reg == "AL") & (ty != "lLN1_bc"),
        "AL minus lLN1": (reg == "AL") & (cat != "AL lLN1"),
        "AL minus lLN1, lLN2": (reg == "AL") & ~np.isin(cat, ["AL lLN1", "AL lLN2"]),
        "AL minus all LNs": (reg == "AL") & (cl != "ALLN"),
        "whole AL": reg == "AL"}.items():
    sub[name] = perron(nm)
    log(name, sub[name]["lam"], sub[name].get("inv_ipr"))
# lLN1_bc internal wiring
bc = np.where(ty == "lLN1_bc")[0]
mbc = (ty == "lLN1_bc")[pre] & (ty == "lLN1_bc")[post]
sid = L["sides"]
W = np.zeros((len(bc), len(bc))); loc = -np.ones(N, np.int64); loc[bc] = np.arange(len(bc))
np.add.at(W, (loc[post[mbc]], loc[pre[mbc]]), w[mbc])
same = sid[bc][:, None] == sid[bc][None, :]
sub["lLN1_bc_wiring"] = {
    "n": int(len(bc)), "sides": {s: int((sid[bc] == s).sum()) for s in np.unique(sid[bc])},
    "nt": {s: int((L["nt"][bc] == s).sum()) for s in np.unique(L["nt"][bc])},
    "edges": int(mbc.sum()), "frac_pairs_connected": float((W > 0).mean()),
    "mean_syn_same_side_pair": float(W[same & ~np.eye(len(bc), dtype=bool)].mean()),
    "mean_syn_cross_side_pair": float(W[~same].mean()) if (~same).any() else None,
    "max_edge": float(W.max()), "median_nonzero_edge": float(np.median(W[W > 0])),
    "exc_in_from_lLN1bc_mean_c": float(W.sum(1).mean() * c),
    "exc_in_total_mean_c": float(np.bincount(post[exc], weights=w[exc], minlength=N)[bc].mean() * c),
}
log("lLN1_bc", sub["lLN1_bc_wiring"])

# --- 4. strongest excitatory connections and 2-cycles ------------------------------------------------------------
oe = np.argsort(-np.where(exc, w, 0))[:200]
strong = {"top200_exc_edges_by_region_pair": {}, "top20": []}
for i in oe:
    k2 = f"{reg[pre[i]]} > {reg[post[i]]}"
    strong["top200_exc_edges_by_region_pair"][k2] = strong["top200_exc_edges_by_region_pair"].get(k2, 0) + 1
for i in oe[:20]:
    strong["top20"].append({"pre": str(ty[pre[i]]), "post": str(ty[post[i]]), "w": float(w[i]),
                            "rev_w": float(w_rev[i]), "pre_nt": str(L["nt"][pre[i]]), "post_nt": str(L["nt"][post[i]])})
# 2-cycles between two excitatory neurons: strength sqrt(w_ij w_ji) * c
both = exc & has_rev & (sign[post] > 0) & (pre < post)
cyc = np.sqrt(w[both] * w_rev[both]) * c
oc = np.argsort(-cyc)[:200]
bi = np.where(both)[0][oc]
strong["top200_exc_2cycles_by_region"] = {}
for i in bi:
    k2 = f"{reg[pre[i]]} <> {reg[post[i]]}"
    strong["top200_exc_2cycles_by_region"][k2] = strong["top200_exc_2cycles_by_region"].get(k2, 0) + 1
strong["top10_2cycles"] = [{"a": str(ty[pre[i]]), "b": str(ty[post[i]]), "w_ab": float(w[i]), "w_ba": float(w_rev[i]),
                            "sqrt_c": float(np.sqrt(w[i] * w_rev[i]) * c)} for i in bi[:10]]
# share of the strongest excitatory edges held by the AL, by percentile cut
for q in [99.9, 99.99]:
    thr = np.percentile(w[exc], q)
    mm = exc & (w >= thr)
    strong[f"exc_edges_above_p{q}"] = {"threshold_syn": float(thr), "n": int(mm.sum()),
                                       "frac_AL_internal": float(((reg[pre] == "AL") & (reg[post] == "AL"))[mm].mean())}
# per-neuron excitatory in-strength: top 30 neurons
ins_exc = np.bincount(post[exc], weights=w[exc], minlength=N)
on = np.argsort(-ins_exc)[:200]
strong["top200_exc_in_strength_by_region"] = {k: int(v) for k, v in zip(*np.unique(reg[on], return_counts=True))}
strong["top10_exc_in_strength"] = [{"type": str(ty[i]), "reg": str(reg[i]), "exc_in_syn": float(ins_exc[i])} for i in on[:10]]
log("strong", strong["top200_exc_edges_by_region_pair"])

json.dump({"c": c, "regions": R, "decomposition": dec, "al_subgraphs": sub, "strong": strong},
          open(f"{OUT}/edges.json", "w"), indent=1)
log("saved")
