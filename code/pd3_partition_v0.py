"""pd3 module_theory MT1/P0: partition 'islands-v0' of the 167,216 neurons into 34 modules (CPU, < 1 min).

Definition (plan.md section 3.1, fixed before any dynamics are run on it):
  modules 0, 1, 2 = AL-L, AL-R, AL-mid: the 526-neuron antennal-lobe module (results/module/module_graph.npz idx, the union of
                    the 99% |v|^2 supports of the two leading excitatory modes) split by soma side (sides.npy 'L' / 'R' / other);
  modules 3..32   = spectral islands 1..30 from the top-400 directed excitatory eigenmodes (results/pd/tail_real_r0_dir.npz),
                    taken in decreasing Re(lambda):
                      * island support = smallest set holding 90% of |v|^2 of the mode (P[:, k]);
                      * modes whose |v|^2 share on the 526-module is >= 0.5 are skipped (they are AL modes);
                      * a mode is merged into an accepted island if its support overlaps the 90% support of that island's SEED
                        (first) mode by more than 50% of the LARGER of the two supports (|S_a & S_b| > 0.5 max(|S_a|, |S_b|));
                        complex-conjugate pairs have identical |v|^2 and always merge.
                        DEVIATION (documented in module_theory.md): the first implementation compared each mode with the
                        growing UNION of an island's supports, normalised by the new mode's support only; because most later
                        modes have > 50% of their support inside mode 2's 2,293-neuron support, island 1 snowballed to 10,276
                        neurons (it absorbed all 395 later non-AL modes) and only 4 islands of >= 20 neurons existed. The
                        symmetric pairwise rule against every merged mode still chained (island 1 = 8,815 neurons, 11 islands
                        in all). Comparing with the seed mode only (no chaining) merges duplicates (conjugate pairs,
                        near-identical supports) only. A merged mode adds no neurons (island k = its seed mode's 90%
                        support minus neurons already assigned), so island 1 is exactly mode 2's 2,293-neuron support;
                      * neurons are assigned first-come in lambda order (AL first); a candidate island keeps only unassigned
                        neurons and is accepted if it has >= 20 of them (otherwise its neurons stay in the pool);
                      * stop at the 30th accepted island (modes that merge into the first 30 islands before the 31st new island
                        would start are still merged);
  module 33       = 'rest' (everything else).
Output: results/pd3/modules/partition_P0.npz
  membership int16[N] (module id 0..33), names[34], size[34], lam / lam_im / inv_ipr of the island's first mode, g_weak =
  alpha_eff / lam (weak-synapse threshold of the mode), modes (merged mode indices, ';'-separated), n_exc / n_inh / n_mod,
  table_json (per-module superclass and side mix), meta_json.
"""
import json, time
import numpy as np
from collections import Counter

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
t0 = time.time()
dt = 0.2; a_eff = (1 - np.exp(-dt)) / dt
sides = np.load(C + "/sides.npy", allow_pickle=True); sup = np.load(C + "/superclasses.npy", allow_pickle=True)
sign = np.load(C + "/sign.npy"); N = len(sign)
tail = np.load(f"{ROOT}/results/pd/tail_real_r0_dir.npz", allow_pickle=True)
P = tail["P"]; lam, lam_im, ipr = tail["lam"], tail["lam_im"], tail["ipr"]
mg = np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)
al_idx = mg["idx"]; AL = np.zeros(N, bool); AL[al_idx] = True

memb = np.full(N, -1, np.int16)
sd = sides[al_idx]
memb[al_idx[sd == "L"]] = 0; memb[al_idx[sd == "R"]] = 1; memb[al_idx[(sd != "L") & (sd != "R")]] = 2
names = ["AL-L", "AL-R", "AL-mid"]
info = [dict(modes=[], lam=np.nan, lam_im=np.nan, inv_ipr=np.nan) for _ in range(3)]

K = P.shape[1]
al_share = np.zeros(K); sup90 = []
for k in range(K):
    p = P[:, k].astype(np.float64); p /= p.sum()
    al_share[k] = p[AL].sum()
    o = np.argsort(-p, kind="stable"); cs = np.cumsum(p[o])
    sup90.append(o[: np.searchsorted(cs, 0.9) + 1])
skipped = [int(k) for k in range(K) if al_share[k] >= 0.5]
sset = [set(x.tolist()) for x in sup90]
# nesting statistic: share of non-AL modes whose 90% support lies > 50% inside mode 2's 90% support
nonal = [k for k in range(3, K) if al_share[k] < 0.5]
nest2 = np.array([len(sset[k] & sset[2]) / len(sset[k]) for k in nonal])
log_rows = []
islands = []          # dict(modes, support(set), id)
for k in range(K):
    if al_share[k] >= 0.5:
        log_rows.append((k, "skip_AL", float(al_share[k]), len(sup90[k]))); continue
    S = sup90[k]; Sset = set(S.tolist())
    merged = False
    for isl in islands:
        j = isl["modes"][0]; ov = len(Sset & sset[j]) / max(len(Sset), len(sset[j]))
        if ov > 0.5:
            isl["modes"].append(k)          # a merged mode is a duplicate of the seed: it adds no neurons
            log_rows.append((k, f"merge->{names[isl['id']]}", float(ov), 0)); merged = True; break
    if merged:
        continue
    if len(islands) == 30:
        log_rows.append((k, "stop (31st new island)", float(al_share[k]), len(S))); break
    new = S[memb[S] < 0]
    if len(new) >= 20:
        mid = 3 + len(islands); memb[new] = mid
        islands.append(dict(modes=[k], id=mid))
        names.append(f"isl{len(islands)}")
        info.append(dict(modes=[k], lam=float(lam[k]), lam_im=float(lam_im[k]), inv_ipr=float(1 / ipr[k]),
                         new_frac=len(new) / len(S), in_mode2=len(Sset & sset[2]) / len(Sset)))
        log_rows.append((k, f"new {names[-1]}", float(al_share[k]), int(len(new))))
    else:
        log_rows.append((k, "too_small", float(al_share[k]), int(len(new))))
for isl in islands:
    info[isl["id"]]["modes"] = isl["modes"]
REST = 3 + len(islands); memb[memb < 0] = REST; names.append("rest")
info.append(dict(modes=[], lam=np.nan, lam_im=np.nan, inv_ipr=np.nan))
M = len(names)
size = np.bincount(memb, minlength=M)
table = []
for m in range(M):
    sel = memb == m
    scm = Counter(sup[sel].tolist()); sdm = Counter(sides[sel].tolist())
    table.append({"id": m, "name": names[m], "size": int(size[m]), "modes": info[m]["modes"],
                  "lam": info[m]["lam"], "lam_im": info[m]["lam_im"], "inv_ipr": info[m]["inv_ipr"],
                  "g_weak": (a_eff / info[m]["lam"]) if np.isfinite(info[m]["lam"]) else None,
                  "support90_of_first_mode": int(len(sup90[info[m]["modes"][0]])) if info[m]["modes"] else None,
                  "first_mode_new_fraction": info[m].get("new_frac"), "first_mode_support_share_in_mode2_support": info[m].get("in_mode2"),
                  "n_exc": int((sign[sel] > 0).sum()), "n_inh": int((sign[sel] < 0).sum()), "n_mod": int((sign[sel] == 0).sum()),
                  "superclass_mix": {k: round(v / size[m], 4) for k, v in scm.most_common(4)},
                  "side_mix": {(k if k else "?"): round(v / size[m], 4) for k, v in sdm.most_common()}})
meta = {"alpha_eff": a_eff, "nonAL_modes_k>=3": len(nonal),
        "share_nonAL_modes_with_support_gt50pct_inside_mode2_support": float((nest2 > 0.5).mean()),
        "median_share_inside_mode2_support": float(np.median(nest2)), "skipped_AL_modes": skipped, "skipped_AL_modes_first6": [k for k in skipped if k < 6],
        "tail_AL_field_ge_0.5_first6": [int(k) for k in range(6) if tail["AL"][k] >= 0.5],
        "n_modes_processed": int(log_rows[-1][0]) + 1, "log": log_rows, "n_modules": M,
        "definition": __doc__.split("Output:")[0]}
np.savez_compressed(f"{ROOT}/results/pd3/modules/partition_P0.npz", membership=memb, names=np.array(names), size=size,
                    lam=np.array([d["lam"] for d in info]), lam_im=np.array([d["lam_im"] for d in info]),
                    inv_ipr=np.array([d["inv_ipr"] for d in info]),
                    g_weak=np.array([a_eff / d["lam"] if np.isfinite(d["lam"]) else np.nan for d in info]),
                    modes=np.array([";".join(map(str, d["modes"])) for d in info]),
                    n_exc=np.array([t["n_exc"] for t in table]), n_inh=np.array([t["n_inh"] for t in table]),
                    n_mod=np.array([t["n_mod"] for t in table]),
                    table_json=json.dumps(table), meta_json=json.dumps(meta))
print(f"[{time.time()-t0:.0f}s] skipped AL modes (first 6): {meta['skipped_AL_modes_first6']}  tail['AL'] rule: "
      f"{meta['tail_AL_field_ge_0.5_first6']}; processed {meta['n_modes_processed']} modes; {M} modules")
for t in table:
    print(f"{t['id']:2d} {t['name']:7s} n={t['size']:6d} modes={t['modes']} lam={t['lam']:.4g} 1/IPR={t['inv_ipr']:.4g} "
          f"g_weak={t['g_weak'] if t['g_weak'] is None else round(t['g_weak'], 4)} sup90={t['support90_of_first_mode']} "
          f"E/I/M={t['n_exc']}/{t['n_inh']}/{t['n_mod']} {list(t['superclass_mix'].items())[:2]} {t['side_mix']}")
print("saved results/pd3/modules/partition_P0.npz")
