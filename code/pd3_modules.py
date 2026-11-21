"""pd3 module_theory MT1/P1 + P2: community partitions of the MaleCNS connectome (CPU).

P1 (Leiden): graph A = E + E^T, the symmetrised EXCITATORY weighted graph (weights c * w on excitatory-presynaptic edges only,
  c = 1/mean in-strength), RBConfigurationVertexPartition (leidenalg 0.10.2, igraph 0.11.8) at resolution gamma in
  {0.5, 1, 2, 4} x seeds {0, 1, 2}. Reported per run: modularity (gamma = 1, weighted), number of modules, number of modules
  with >= 50 neurons, size quantiles; NMI across seeds (per gamma) and against P0. Choice (plan.md 3.1): the gamma with the
  highest mean seed-pair NMI that gives >= 20 modules of >= 50 neurons; within it the seed with the highest mean NMI to the
  other two seeds. Module ids are relabelled by decreasing size.
P2 (anatomy): superclass x side (superclasses.npy, sides.npy), ids by decreasing size.
Outputs: results/pd3/modules/partition_P1.npz, partition_P2.npz (membership int32[N], names, size, meta_json) and
  results/pd3/modules/leiden_runs.json.
"""
import json, time
import numpy as np
import igraph as ig
import leidenalg as la

C = "/MALECNS_CACHE"; ROOT = "/MALECNS_ROOT"
t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
ins = np.bincount(post, weights=w, minlength=N); c = 1 / ins[ins > 0].mean()
ex = sign[pre] > 0
a_, b_ = np.minimum(pre[ex], post[ex]), np.maximum(pre[ex], post[ex]); ww = c * w[ex]
loop = a_ == b_
key = a_[~loop] * N + b_[~loop]
uk, inv = np.unique(key, return_inverse=True)
wsum = np.bincount(inv, weights=ww[~loop])
edges = np.stack([uk // N, uk % N], 1)
log(f"excitatory directed edges {int(ex.sum())}, self-loops dropped {int(loop.sum())}, undirected edges {len(uk)}")
G = ig.Graph(n=N, edges=edges.tolist(), directed=False); G.es["weight"] = wsum.tolist()
del edges
log("igraph built", G.summary()[:80])
p0 = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz", allow_pickle=True)["membership"].astype(int)

runs = []; memb = {}
for gam in [0.5, 1.0, 2.0, 4.0]:
    for seed in [0, 1, 2]:
        t1 = time.time()
        part = la.find_partition(G, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=gam,
                                 seed=seed, n_iterations=5)
        m = np.array(part.membership)
        sz = np.bincount(m)
        o = np.argsort(-sz, kind="stable"); rel = np.empty_like(o); rel[o] = np.arange(len(o)); m = rel[m]; sz = sz[o]
        Q = G.modularity(m.tolist(), weights="weight")
        memb[(gam, seed)] = m
        row = {"gamma": gam, "seed": seed, "modularity_gamma1": Q, "quality": part.quality(), "n_modules": int(len(sz)),
               "n_modules_ge50": int((sz >= 50).sum()), "n_modules_ge20": int((sz >= 20).sum()),
               "largest": sz[:10].tolist(), "n_singletons": int((sz == 1).sum()),
               "nmi_vs_P0": ig.compare_communities(m.tolist(), p0.tolist(), method="nmi"), "sec": time.time() - t1}
        runs.append(row)
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
summ = {}
for gam in [0.5, 1.0, 2.0, 4.0]:
    nm = {}
    for s1, s2 in [(0, 1), (0, 2), (1, 2)]:
        nm[f"{s1}-{s2}"] = ig.compare_communities(memb[(gam, s1)].tolist(), memb[(gam, s2)].tolist(), method="nmi")
    rows = [r for r in runs if r["gamma"] == gam]
    summ[str(gam)] = {"seed_nmi": nm, "mean_seed_nmi": float(np.mean(list(nm.values()))),
                      "min_n_modules_ge50": int(min(r["n_modules_ge50"] for r in rows)),
                      "mean_modularity_gamma1": float(np.mean([r["modularity_gamma1"] for r in rows])),
                      "mean_nmi_vs_P0": float(np.mean([r["nmi_vs_P0"] for r in rows]))}
    log("gamma", gam, summ[str(gam)])
ok = [g for g in summ if summ[g]["min_n_modules_ge50"] >= 20]
gbest = max(ok, key=lambda g: summ[g]["mean_seed_nmi"]) if ok else max(summ, key=lambda g: summ[g]["min_n_modules_ge50"])
nmi = summ[gbest]["seed_nmi"]
seed_score = {s: np.mean([v for k, v in nmi.items() if str(s) in k.split("-")]) for s in [0, 1, 2]}
sbest = max(seed_score, key=seed_score.get)
m1 = memb[(float(gbest), sbest)]; sz1 = np.bincount(m1)
choice = {"gamma": float(gbest), "seed": int(sbest), "rule_satisfied": bool(ok), "eligible_gammas": ok, "seed_score": seed_score}
log("chosen", choice)
np.savez_compressed(f"{ROOT}/results/pd3/modules/partition_P1.npz", membership=m1.astype(np.int32),
                    names=np.array([f"L{k}" for k in range(len(sz1))]), size=sz1,
                    meta_json=json.dumps({"choice": choice, "graph": "A = E + E^T (excitatory, c*w)", "c": c}))
# P2: superclass x side
sup = np.load(C + "/superclasses.npy", allow_pickle=True); sides = np.load(C + "/sides.npy", allow_pickle=True)
lab = np.array([f"{a or '?'}|{b or '?'}" for a, b in zip(sup, sides)])
ul, inv = np.unique(lab, return_inverse=True); sz2 = np.bincount(inv); o = np.argsort(-sz2, kind="stable")
rel = np.empty_like(o); rel[o] = np.arange(len(o)); m2 = rel[inv]
np.savez_compressed(f"{ROOT}/results/pd3/modules/partition_P2.npz", membership=m2.astype(np.int32), names=ul[o], size=sz2[o],
                    meta_json=json.dumps({"definition": "superclass x side"}))
json.dump({"runs": runs, "per_gamma": summ, "choice": choice, "P2_n_groups": int(len(ul)),
           "P2_nmi_vs_P0": ig.compare_communities(m2.tolist(), p0.tolist(), method="nmi"),
           "P1_nmi_vs_P2": ig.compare_communities(m1.tolist(), m2.tolist(), method="nmi")},
          open(f"{ROOT}/results/pd3/modules/leiden_runs.json", "w"), indent=1)
log("saved partition_P1.npz, partition_P2.npz, leiden_runs.json")
