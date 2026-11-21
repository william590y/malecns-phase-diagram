"""m2_stoch_graphs.py -- graph files (ald_sim --graph format) for the finite-size and module predictions of lane
m2/theory-stochastic. My own copies under results/m2/theory/stochastic/graphs/ (no other lane's files are read).
  cnull_N{N}_s{s}: m2_common.write_graph("cnull", N, s)  (pd3 configuration null of size N; c renormalised)
  realsub_N{N}_s{s}: m2_common.write_graph("realsub", N, s)  (AL module kept; c = full brain)
  usub_N{N}_s{s}: uniform random N-subset of all neurons, induced subgraph, c = full brain (rng 4242 + 1000 s + N % 997)
  mod_{AL,ALL,ALR,CB,VNC,OL}: induced subgraph of a block, c = full brain
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/MALECNS_ROOT/code")
import m2_common as MC

ROOT = MC.ROOT
OUT = f"{ROOT}/results/m2/theory/stochastic/graphs"
os.makedirs(OUT, exist_ok=True)
C = "/MALECNS_CACHE"
pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); NF = len(sign)
ins = np.bincount(post, weights=w, minlength=NF); c_full = 1 / ins[ins > 0].mean()
info = {}


def induced(S, name, kind):
    S = np.sort(np.asarray(S))
    loc = -np.ones(NF, np.int64); loc[S] = np.arange(len(S))
    e = (loc[pre] >= 0) & (loc[post] >= 0)
    np.savez(f"{OUT}/{name}.npz", pre=loc[pre[e]], post=loc[post[e]], w=w[e].astype(np.float32), sign=sign[S],
             sub=S.astype(np.int64), c=float(c_full), kind=kind, seed=0)
    info[name] = {"N": int(len(S)), "n_edges": int(e.sum()), "c": float(c_full)}


which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "fss"):
    for N in (5000, 10000, 20000, 40000, 80000):
        info[f"cnull_N{N}_s1"] = MC.write_graph("cnull", N, 1, f"{OUT}/cnull_N{N}_s1.npz")
    info["cnull_N167216_s0"] = MC.write_graph("cnull", 167216, 0, f"{OUT}/cnull_N167216_s0.npz")
    for N in (20000, 40000, 80000):
        info[f"realsub_N{N}_s1"] = MC.write_graph("realsub", N, 1, f"{OUT}/realsub_N{N}_s1.npz")
        rng = np.random.default_rng(4242 + 1000 + N % 997)
        induced(rng.choice(NF, N, replace=False), f"usub_N{N}_s1", "usub")
if which in ("all", "mod"):
    mod = np.load(f"{ROOT}/results/module/module_graph.npz", allow_pickle=True)["idx"]
    sides = np.load(C + "/sides.npy", allow_pickle=True).astype(str)
    sup = np.load(C + "/superclasses.npy", allow_pickle=True).astype(str)
    induced(mod, "mod_AL", "mod")
    induced(mod[sides[mod] == "L"], "mod_ALL", "mod")
    induced(mod[sides[mod] == "R"], "mod_ALR", "mod")
    induced(np.where(np.char.startswith(sup, "cb_"))[0], "mod_CB", "mod")
    induced(np.where(np.char.startswith(sup, "vnc_"))[0], "mod_VNC", "mod")
    induced(np.where(np.char.startswith(sup, "ol_") | np.char.startswith(sup, "visual_"))[0], "mod_OL", "mod")
json.dump(info, open(f"{OUT}/graphs_{which}.json", "w"), indent=1)
print(json.dumps(info, indent=1))
