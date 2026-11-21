"""m2_nature_graphs.py -- lane m2/nature (results/m2/nature/plan.md section 1): finite-size-scaling graphs + exact g_lin.

  python m2_nature_graphs.py write      write every graph file (skips existing ones)            [CPU]
  python m2_nature_graphs.py glin       exact g_lin of every graph (pd3_common.g_lin_exact)      [GPU]

Families (graph files in results/m2/nature/graphs/):
  cnull_N{N}_s{s}.npz      configuration null of size N (m2_common.write_graph("cnull", ...), pd3 construction, own c)
  realsub_N{N}_s{s}.npz    real subsample keeping the AL module, c FIXED at the full-brain value (MODEL2 / pd3)
  realsubcn_N{N}_s{s}.npz  the same subsample, file WITHOUT c, so ald_sim uses 1/mean in-strength of the subgraph
                           (renormalised coupling; plan section 1)
  realsubu_N{N}_s{s}.npz   UNIFORM random N-subset of the real brain (AL not forced in), induced subgraph, file WITHOUT c
                           (renormalised coupling); rng = default_rng(4241 + 1000 s + N % 997)  (plan amendment 1)
  167k nulls are ald_sim --null --null_seed s (not written; their g_lin is computed from the same permutation).
g_lin uses the excitatory edges (presynaptic consensus sign +1) with the graph's c, exactly as pd3 (redraw_fss) did, so
the cnull s1-3 and realsub (fixed c) values can be checked against results/pd3/redraw_fss.json.
Output: results/m2/nature/graphs/graphs.json
"""
import json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import m2_common as M

D = f"{M.ROOT}/results/m2/nature/graphs"
CNULL_N = (5000, 10000, 20000, 40000, 80000)
CNULL_S = (1, 2, 3)
REAL_N = (20000, 40000, 80000)
REAL_S = (1, 2)
NFULL = 167216


def path(kind, N, s):
    return f"{D}/{kind}_N{N}_s{s}.npz"


def all_graphs():
    out = [("cnull", N, s) for N in CNULL_N for s in CNULL_S]
    out += [(k, N, s) for k in ("realsub", "realsubcn", "realsubu") for N in REAL_N for s in REAL_S]
    return out


def write():
    import pd3_fss_graph as F
    from pd3_common import load_cache
    os.makedirs(D, exist_ok=True)
    cache = load_cache()
    sign = cache[3]
    info = {}
    for kind, N, s in all_graphs():
        p = path(kind, N, s)
        if os.path.exists(p):
            continue
        if kind == "realsubu":
            pre0, post0, w0 = cache[0], cache[1], cache[2]
            S = np.sort(np.random.default_rng(4241 + 1000 * s + N % 997).choice(len(sign), N, replace=False))
            loc = -np.ones(len(sign), np.int64); loc[S] = np.arange(N)
            e = (loc[pre0] >= 0) & (loc[post0] >= 0)
            g = {"pre": loc[pre0[e]], "post": loc[post0[e]], "w": w0[e], "sub": S, "N": N, "n_edges": int(e.sum())}
        else:
            g = F.build("realsub" if kind == "realsubcn" else kind, N, s, cache=cache)
        sub = g["sub"] if g["sub"] is not None else np.arange(len(sign))
        kw = dict(pre=g["pre"].astype(np.int64), post=g["post"].astype(np.int64), w=g["w"].astype(np.float32),
                  sign=sign[sub], sub=sub.astype(np.int64), kind=kind, seed=s)
        if kind not in ("realsubcn", "realsubu"):
            kw["c"] = float(g["c"])
        tmp = p + ".tmp.npz"
        np.savez(tmp, **kw)
        os.replace(tmp, p)
        info[p] = {"N": int(g["N"]), "n_edges": int(g["n_edges"]), "c_file": kw.get("c")}
        print(p, info[p], flush=True)


def glin(only=None):
    import torch
    from pd3_common import load_cache, c_of, exc_csr_torch, g_lin_exact
    res = {}
    old = {}
    if only and os.path.exists(f"{D}/graphs.json"):
        old = json.load(open(f"{D}/graphs.json"))
        res.update(old["graphs"])
    for kind, N, s in all_graphs():
        if only and kind != only:
            continue
        G = np.load(path(kind, N, s))
        pre, post, w, sg = G["pre"], G["post"], G["w"].astype(np.float64), G["sign"]
        c = float(G["c"]) if "c" in G.files else c_of(post, w, N)
        Ed = exc_csr_torch(pre, post, w, sg[pre].astype(np.int8), c, N)
        gl, _ = g_lin_exact(Ed)
        res[f"{kind}_N{N}_s{s}"] = {"kind": kind, "N": N, "seed": s, "c": c, "g_lin": gl, "n_edges": int(len(pre))}
        print(kind, N, s, c, gl, flush=True)
        del Ed; torch.cuda.empty_cache()
    if only:
        old["graphs"] = res
        json.dump(old, open(f"{D}/graphs.json", "w"), indent=1)
        return
    pre, post, w, sign = load_cache()
    for s in (0, 1, 2):                        # ald_sim --null --null_seed s: post permuted with default_rng(s)
        po = post[np.random.default_rng(s).permutation(len(post))]
        c = c_of(po, w, NFULL)
        Ed = exc_csr_torch(pre, po, w, sign[pre].astype(np.int8), c, NFULL)
        gl, _ = g_lin_exact(Ed)
        res[f"cnull_N{NFULL}_s{s}"] = {"kind": "cnull_aldsim", "N": NFULL, "seed": s, "c": c, "g_lin": gl}
        print("null167k", s, c, gl, flush=True)
        del Ed; torch.cuda.empty_cache()
    c = c_of(post, w, NFULL)
    Ed = exc_csr_torch(pre, post, w, sign[pre].astype(np.int8), c, NFULL)
    gl, _ = g_lin_exact(Ed)
    res["real_full"] = {"kind": "real", "N": NFULL, "c": c, "g_lin": gl}
    # check against pd3 (identical constructions)
    pd = json.load(open(f"{M.ROOT}/results/pd3/redraw_fss.json"))["B_fss"]["per_graph"]
    chk = []
    for p in pd:
        if p["r"] != 0:
            continue
        k = f"{p['kind']}_N{p['N']}_s{p['seed']}"
        if k in res and p["N"] != NFULL:
            chk.append({"graph": k, "pd3": p["g_lin"], "here": res[k]["g_lin"],
                        "rel_dev": res[k]["g_lin"] / p["g_lin"] - 1})
    json.dump({"graphs": res, "pd3_check": chk,
               "pd3_check_max_abs_rel_dev": max(abs(x["rel_dev"]) for x in chk) if chk else None},
              open(f"{D}/graphs.json", "w"), indent=1)
    print("pd3 check", chk, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "glin" and len(sys.argv) > 2:
        glin(sys.argv[2])
    else:
        {"write": write, "glin": glin}[sys.argv[1]]()
