"""pd3 module_theory MT2 post-processing (CPU): corrected attractor census and boundary classification of the closure.

Two corrections to pd3_coarse_mf.py's census/boundaries, found when inspecting the P1 output (documented in module_theory.md):
 1. SL. Starts that collapse into silence through mutual inhibition decay at e^{-dt} per step with every module net-inhibited;
    at X ~ 1e-18 their per-step change is below the 1e-10 tolerance, so they were stored as converged 'SL' attractors with
    rho(J) = e^{-dt}. The silent state's stability is decided analytically instead: any converged state with max_k X_k < 1e-12
    is the SL fixed point, stable iff g < g_SL_edge(r) = alpha_eff / mu_+(W(r)) (cone spectral radius, as in de Candia's SL
    lines). Above the edge such states are transients, not attractors.
 2. Order of a boundary. A deterministic map cannot jump at a fold without an interval of coexistence, so a first-order line is
    identified by hysteresis (up and down sweeps differing, L_inf > 1e-3, on >= 2 consecutive g where both branches settled,
    residual < 1e-10) or by the edges of a census region with >= 2 stable attractors; changes of the active set with up = down
    are continuous (second order / imperfect crossovers), however steep. Large single-step changes without hysteresis are
    listed separately as 'steep' (max |dX| > 0.02 between neighbouring grid points, ~3.5% apart in g).
Output: results/pd3/modules/coarse_post_<P>.json (+ coarse_post_<P>.npz with n_attr_corr, labels_corr)
Usage: python pd3_coarse_post.py P0|P1
"""
import json, sys
import numpy as np

ROOT = "/MALECNS_ROOT"; P = sys.argv[1]
d = np.load(f"{ROOT}/results/pd3/modules/coarse_mf_{P}.npz", allow_pickle=True)
g, r, names = d["g"], d["r"], [str(x) for x in d["names"]]; gSL = d["g_SL_edge"]; M = len(names)
nA = d["n_attr"]; AX = d["attr_X"]; AL = d["attr_label"]
n_corr = np.zeros_like(nA); labs = np.empty(nA.shape, dtype=object); dropped = 0; sl_stable = np.zeros(nA.shape, bool)
for ri in range(len(r)):
    for gi in range(len(g)):
        keep = []
        for j in range(nA[ri, gi]):
            if np.nanmax(AX[ri, gi, j]) < 1e-12:
                if g[gi] < gSL[ri]:
                    keep.append("SL"); sl_stable[ri, gi] = True
                else:
                    dropped += 1
            else:
                keep.append(str(AL[ri, gi, j]))
        n_corr[ri, gi] = len(keep); labs[ri, gi] = "|".join(sorted(keep)) if keep else "none"


def lab(x, thr=1e-3):
    A = [names[k] for k in range(M) if x[k] > thr]
    return "SL" if not A else ("SH" if len(A) == M else f"B{len(A)}:" + "+".join(A))


bounds, hyst, steep = [], [], []
for ri in range(len(r)):
    up, dn, ur, dr = d["up"][ri], d["dn"][ri], d["up_resid"][ri], d["dn_resid"][ri]
    settled = (ur < 1e-10) & (dr < 1e-10)
    diff = np.abs(up - dn).max(1) > 1e-3
    h = diff & settled
    i = 0
    while i < len(g):
        if h[i]:
            j = i
            while j + 1 < len(g) and h[j + 1]:
                j += 1
            if j > i:
                hyst.append({"r": float(r[ri]), "g_lo": float(g[i]), "g_hi": float(g[j]), "n_points": int(j - i + 1),
                             "up_label_lo": lab(up[i]), "down_label_lo": lab(dn[i]), "max_Linf": float(np.abs(up[i:j + 1] - dn[i:j + 1]).max())})
            i = j + 1
        else:
            i += 1
    nonstat_up = (ur >= 1e-10); nonstat_dn = (dr >= 1e-10)
    for br, X, ns in [("up", up, nonstat_up), ("down", dn, nonstat_dn)]:
        L = [lab(x) for x in X]
        for i in range(1, len(g)):
            jump = float(np.abs(X[i] - X[i - 1]).max())
            if L[i] != L[i - 1]:
                bounds.append({"r": float(r[ri]), "branch": br, "g_lo": float(g[i - 1]), "g_hi": float(g[i]), "from": L[i - 1], "to": L[i],
                               "max_jump": jump, "in_hysteresis": bool(h[i] or h[i - 1]), "nonstationary": bool(ns[i] or ns[i - 1])})
            if jump > 0.02 and not (h[i] or h[i - 1]) and not (ns[i] or ns[i - 1]):
                steep.append({"r": float(r[ri]), "branch": br, "g_lo": float(g[i - 1]), "g_hi": float(g[i]), "max_jump": jump})
multi = n_corr >= 2
edges = []
for ri in range(len(r)):
    m = multi[ri]
    for i in range(1, len(g)):
        if m[i] != m[i - 1]:
            edges.append({"r": float(r[ri]), "g": float(np.sqrt(g[i] * g[i - 1])), "kind": "enter" if m[i] else "leave"})
seg = {}
for ri in range(len(r)):
    s, prev = [], None
    for gi in range(len(g)):
        if labs[ri, gi] != prev:
            s.append([float(g[gi]), labs[ri, gi]]); prev = labs[ri, gi]
    seg[f"{r[ri]:g}"] = s
nonst = d["n_unconverged"] > 0
out = {"P": P, "n_SL_entries_dropped_above_edge": int(dropped), "n_cells": int(n_corr.size),
       "n_cells_multistable_corrected": int(multi.sum()), "max_n_attractors_corrected": int(n_corr.max()),
       "n_cells_no_stable_attractor": int((n_corr == 0).sum()),
       "n_cells_no_stable_attractor_and_nonstationary": int(((n_corr == 0) & nonst).sum()),
       "multistable_r_values": sorted({float(r[ri]) for ri, gi in zip(*np.where(multi))}),
       "multistable_g_range": [float(g[multi.any(0)].min()), float(g[multi.any(0)].max())] if multi.any() else None,
       "SL_coexisting_with_active_cells": int(sum(1 for ri, gi in zip(*np.where(multi)) if "SL" in labs[ri, gi].split("|"))),
       "hysteresis_intervals": hyst, "n_hysteresis_intervals": len(hyst),
       "multistable_region_edges": edges[:200], "n_label_changes_up_or_down": len(bounds),
       "n_label_changes_continuous": int(sum(1 for b in bounds if not b["in_hysteresis"] and not b["nonstationary"])),
       "steep_jumps_without_hysteresis": steep[:100], "n_steep_jumps_without_hysteresis": len(steep),
       "segments_corrected": seg,
       "first_nonstationary_g_by_r": {f"{r[ri]:g}": (float(g[np.where(nonst[ri])[0][0]]) if nonst[ri].any() else None) for ri in range(len(r))},
       "g_SL_edge": {f"{x:g}": float(y) for x, y in zip(r, gSL)}}
json.dump(out, open(f"{ROOT}/results/pd3/modules/coarse_post_{P}.json", "w"), indent=1)
np.savez_compressed(f"{ROOT}/results/pd3/modules/coarse_post_{P}.npz", n_attr_corr=n_corr, labels_corr=labs.astype(str), sl_stable=sl_stable)
print(f"{P}: dropped {dropped} SL entries above the edge; multistable cells {multi.sum()}/{n_corr.size} (max {n_corr.max()}); "
      f"no stable attractor {(n_corr == 0).sum()} (of which non-stationary {((n_corr == 0) & nonst).sum()}); hysteresis intervals {len(hyst)}; "
      f"steep jumps w/o hysteresis {len(steep)}; SL+active cells {out['SL_coexisting_with_active_cells']}")
for rr in ["0", "1", "2", "2.5", "4", "6"]:
    print(" r", rr, [(round(a, 3), b[:70]) for a, b in seg[rr][:7]])
for h_ in hyst[:12]:
    print(" hyst", h_)
