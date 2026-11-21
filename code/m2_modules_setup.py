"""m2_modules_setup.py -- lane m2/modules: region index sets, start sets and module subgraphs (CPU, 2026-11-09).

Reads the pd3 partitions (read-only) and the connectome cache (read-only) and writes
  results/m2/modules/groups/regions.npz   one key per region / start set (global cache ids, int64)
  results/m2/modules/groups/regions.json  region list (name, key, size, partition), start sets (key, size)
  results/m2/modules/graphs/<mod>.npz     isolated-module subgraphs for ald_sim --graph (pre, post local; w; sign; sub; c)
Partitions: P0 (34 pd3 spectral modules), P1 (Leiden gamma 4: the 50 modules >= 50 neurons + P1_small),
P2 (superclass x side: groups >= 50 neurons + P2_small); hemispheres HEM_L / HEM_R (side L / R).
c of the subgraphs = full-brain raw normalisation (1 / mean in-strength over neurons with input; = ald_sim's value).
"""
import json, os
import numpy as np

C = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/m2/modules"
GRP = f"{OUT}/groups"; GPH = f"{OUT}/graphs"
os.makedirs(GRP, exist_ok=True); os.makedirs(GPH, exist_ok=True)

sup = np.load(C + "/superclasses.npy", allow_pickle=True).astype(str)
sides = np.load(C + "/sides.npy", allow_pickle=True).astype(str)
N = len(sup)
keys = {}; regions = []; starts = {}


def safe(s):
    return s.replace("|", "_").replace("?", "U").replace(" ", "_")


def add_region(name, idx, part):
    k = "R_" + safe(name)
    keys[k] = np.asarray(idx, np.int64)
    regions.append({"name": safe(name), "key": k, "size": int(len(idx)), "partition": part})


# P0
p0 = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz", allow_pickle=True)
m0 = p0["membership"].astype(int); n0 = [str(x) for x in p0["names"]]
for k, nm in enumerate(n0):
    add_region("P0_" + nm, np.where(m0 == k)[0], "P0")
# P1
p1 = np.load(f"{ROOT}/results/pd3/modules/partition_P1.npz", allow_pickle=True)
m1 = p1["membership"].astype(int); s1 = np.bincount(m1)
big1 = [k for k in range(len(s1)) if s1[k] >= 50]
for k in big1:
    add_region(f"P1_L{k}", np.where(m1 == k)[0], "P1")
add_region("P1_small", np.where(~np.isin(m1, big1))[0], "P1")
# P2
p2 = np.load(f"{ROOT}/results/pd3/modules/partition_P2.npz", allow_pickle=True)
m2 = p2["membership"].astype(int); n2 = [str(x) for x in p2["names"]]; s2 = np.bincount(m2, minlength=len(n2))
big2 = [k for k in range(len(n2)) if s2[k] >= 50]
for k in big2:
    add_region("P2_" + n2[k], np.where(m2 == k)[0], "P2")
add_region("P2_small", np.where(~np.isin(m2, big2))[0], "P2")
# hemispheres
add_region("HEM_L", np.where(sides == "L")[0], "HEM")
add_region("HEM_R", np.where(sides == "R")[0], "HEM")

# start sets
AL = np.where(np.isin(m0, [n0.index("AL-L"), n0.index("AL-R"), n0.index("AL-mid")]))[0]
cb = np.array([s.startswith("cb_") or s.startswith("descending_neuron") for s in sup])
vnc = np.array([s.startswith("vnc_") or s.startswith("ascending_neuron") or s.startswith("sensory_ascending")
                or s == "efferent_ascending" for s in sup])
ol = np.array([s.startswith("ol_") or s.startswith("visual_projection") or s == "visual_centrifugal" for s in sup])
S = {"AL_L": np.where(m0 == n0.index("AL-L"))[0], "AL_R": np.where(m0 == n0.index("AL-R"))[0], "AL": AL,
     "hemiL": np.where(sides == "L")[0], "hemiR": np.where(sides == "R")[0],
     "nonAL": np.setdiff1d(np.arange(N), AL), "CB": np.where(cb)[0], "VNC": np.where(vnc)[0], "OL": np.where(ol)[0],
     "isl1": np.where(m0 == n0.index("isl1"))[0], "isl3": np.where(m0 == n0.index("isl3"))[0],
     "isl5": np.where(m0 == n0.index("isl5"))[0]}
assert not (cb & vnc).any() and not (cb & ol).any() and not (vnc & ol).any()
for k, v in S.items():
    keys["S_" + k] = np.asarray(v, np.int64); starts[k] = {"key": "S_" + k, "size": int(len(v))}
# extra regions for the start sets not already regions (confinement C needs counts inside S)
for k in ("AL", "nonAL", "CB", "VNC", "OL"):
    add_region("S_" + k, S[k], "START")

np.savez(f"{GRP}/regions.npz", **keys)
side_info = {"n_regions": len(regions), "regions": regions, "starts": starts,
             "unassigned_by_CB_VNC_OL": int(N - cb.sum() - vnc.sum() - ol.sum()),
             "superclass_rule": {"CB": "cb_* | descending_neuron*", "VNC": "vnc_* | ascending_neuron | sensory_ascending* | efferent_ascending",
                                 "OL": "ol_* | visual_projection* | visual_centrifugal"}}
json.dump(side_info, open(f"{GRP}/regions.json", "w"), indent=1)
print(f"{len(regions)} regions; starts", {k: v["size"] for k, v in starts.items()})
print("P0", sum(r["partition"] == "P0" for r in regions), "P1", sum(r["partition"] == "P1" for r in regions),
      "P2", sum(r["partition"] == "P2" for r in regions))

# isolated-module subgraphs (internal edges only; the rest of the brain is absent)
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float32); sign = np.load(C + "/sign.npy")
ins = np.bincount(post, weights=w.astype(np.float64), minlength=N); c = float(1 / ins[ins > 0].mean())
gi = {}
for nm, idx in [("AL", S["AL"]), ("AL_L", S["AL_L"]), ("AL_R", S["AL_R"]), ("isl1", S["isl1"]), ("isl3", S["isl3"]),
                ("isl5", S["isl5"]), ("CB", S["CB"]), ("VNC", S["VNC"]), ("OL", S["OL"])]:
    sel = np.zeros(N, bool); sel[idx] = True
    loc = -np.ones(N, np.int64); loc[idx] = np.arange(len(idx))
    e = sel[pre] & sel[post]
    np.savez(f"{GPH}/{nm}.npz", pre=loc[pre[e]], post=loc[post[e]], w=w[e], sign=sign[idx], sub=np.asarray(idx, np.int64),
             c=c, kind="module_alone", module=nm)
    gi[nm] = {"N": int(len(idx)), "n_edges": int(e.sum()), "n_exc_neurons": int((sign[idx] > 0).sum()),
              "n_inh_neurons": int((sign[idx] < 0).sum()), "frac_in_strength_internal":
              float(np.bincount(post[e], weights=w[e].astype(np.float64), minlength=N)[idx].sum() / ins[idx].sum())}
    print(nm, gi[nm])
json.dump({"c": c, "graphs": gi}, open(f"{GPH}/graphs.json", "w"), indent=1)
print("c =", c, "1/c =", 1 / c)
