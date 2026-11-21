"""pd3 redraw_fss: size-N graphs for the finite-size-scaling check. build(kind, N, seed) is a deterministic function.

kind = "cnull" (configuration null of size N): exactly the pd3_theory_nullK.py construction, so seeds 1 and 2 reproduce its
  K_nc table: rng = default_rng(1000*seed + N % 997); a uniformly random N-subset S (sorted); out-stubs = edges whose
  PRESYNAPTIC neuron is in S (they keep weight and the presynaptic sign); in-stubs = edges whose POSTSYNAPTIC neuron is in S;
  m = min(#out, #in) stubs matched at random (subsample the larger multiset without replacement, permute the in-stubs);
  c = 1 / mean in-strength of the new graph (over neurons with in-strength > 0, the simulator's raw normalisation).
  N = 167216: seed 0 = swc.py's null (post permuted with default_rng(0), all edges kept); seeds >= 1 = the same stub matching on
  the full set (S = all neurons, so it is a fresh permutation of the targets).
kind = "realsub" (real wiring, AL module always kept): rng = default_rng(7919 + 1000*seed + N % 997); S = the 526 AL-module
  neurons (results/module/module_graph.npz idx) plus a uniformly random (N - 526)-subset of the other neurons; induced
  subgraph (edges with pre and post in S); c fixed at the FULL-brain value 1/mean in-strength (= 1/742.7) so the AL's internal
  couplings are unchanged.
Returns dict: pre, post (local indices 0..N-1), w (synapse counts, float64), esign (per-edge presynaptic sign), c, N, kind,
seed, n_edges, and "sub" (global ids of the local neurons, or None for the full-size null) and "al_local" (local ids of AL
neurons inside the graph; for cnull only those that happen to be in the subset).
"""
import numpy as np
from pd3_common import load_cache, c_of, module_idx


def build(kind, N, seed, cache=None):
    pre, post, w, sign = cache if cache is not None else load_cache()
    NF = len(sign)
    al = module_idx()
    if kind == "cnull":
        if N == NF and seed == 0:
            rng0 = np.random.default_rng(0)
            po = post[rng0.permutation(len(post))]
            g = {"pre": pre.copy(), "post": po, "w": w.copy(), "esign": sign[pre].astype(np.int8), "sub": None}
            loc = np.arange(NF)
        else:
            rng = np.random.default_rng(1000 * seed + N % 997)
            S = np.sort(rng.choice(NF, N, replace=False)); loc = -np.ones(NF, np.int64); loc[S] = np.arange(N)
            out_e = np.where(loc[pre] >= 0)[0]; in_e = np.where(loc[post] >= 0)[0]
            m = min(len(out_e), len(in_e))
            oe = rng.choice(out_e, m, replace=False) if len(out_e) > m else out_e
            ie = rng.choice(in_e, m, replace=False) if len(in_e) > m else in_e
            ie = rng.permutation(ie)
            g = {"pre": loc[pre[oe]], "post": loc[post[ie]], "w": w[oe].copy(), "esign": sign[pre[oe]].astype(np.int8),
                 "sub": S}
        g["c"] = c_of(g["post"], g["w"], N)
    elif kind == "realsub":
        rng = np.random.default_rng(7919 + 1000 * seed + N % 997)
        rest = np.setdiff1d(np.arange(NF), al)
        S = np.sort(np.r_[al, rng.choice(rest, N - len(al), replace=False)])
        loc = -np.ones(NF, np.int64); loc[S] = np.arange(N)
        e = (loc[pre] >= 0) & (loc[post] >= 0)
        g = {"pre": loc[pre[e]], "post": loc[post[e]], "w": w[e].copy(), "esign": sign[pre[e]].astype(np.int8), "sub": S}
        g["c"] = c_of(post, w, NF)
    else:
        raise ValueError(kind)
    g.update({"kind": kind, "N": int(N), "seed": int(seed), "n_edges": int(len(g["pre"]))})
    a = loc[al]
    g["al_local"] = a[a >= 0]
    return g
