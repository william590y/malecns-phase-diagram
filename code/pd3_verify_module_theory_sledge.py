"""pd3 verification of module_theory MT2 (CPU): the closure's silent-state (SL) edge computed DIRECTLY, without the rectified
power iteration of pd3_coarse_mf.py (whose single all-ones start does not converge for P1 at r >= 1.25 and returns mu_+ = 0
at r >= 5.75, i.e. 'no SL edge').

Method: the one-sided linearisation of the closure map at X = 0 is the positively homogeneous map
    L_g(x) = e^{-dt} x + dt g [W(r) x]_+ ,  x >= 0,  W(r) = E^PW - r I^PW (quotient_<P>.npz),
and SL is linearly unstable iff some x >= 0 grows under L_g. Growth rate = max over 132 starts (all ones, each module alone,
100 random subsets) of exp(mean log norm growth) over steps 3000..6000; the edge is the g where it crosses 1 (bisection in
ln g, 28 steps). Checked separately (see module_theory.md Verification) by iterating the NONLINEAR closure from a 1e-9 seed.
Output: results/pd3/verify_module_theory_sledge.json {P: {r: [g_edge_direct, g_edge_file]}}
"""
import json
import numpy as np
R3 = "/MALECNS_ROOT/results/pd3"
dt = 0.2; dec = np.exp(-dt); a_eff = (1 - dec) / dt
rng = np.random.default_rng(11)


def growth(W, g, starts, n=6000, burn=3000):
    X = starts.copy(); X /= X.sum(0, keepdims=True); lg = np.zeros(X.shape[1])
    for it in range(n):
        Y = dec * X + dt * g * np.clip(W @ X, 0, None); s = Y.sum(0); X = Y / s
        if it >= burn:
            lg += np.log(s)
    return np.exp(lg / (n - burn)).max()


out = {}
for P in ["P0", "P1"]:
    q = np.load(f"{R3}/modules/quotient_{P}.npz", allow_pickle=True); Ew, Iw = q["E_PW"], q["I_PW"]; M = Ew.shape[0]
    cp = json.load(open(f"{R3}/modules/coarse_post_{P}.json"))["g_SL_edge"]
    starts = np.concatenate([np.ones((M, 1)), np.eye(M) + 1e-9, (rng.random((M, 100)) < 0.3) + 1e-9], 1)
    out[P] = {}
    for r in np.round(np.arange(0, 6.001, 0.25), 3):
        W = Ew - r * Iw
        lo, hi = np.log(0.05), np.log(60.0)
        if growth(W, np.exp(hi), starts) < 1:
            out[P][f"{r:g}"] = [None, cp[f"{r:g}"]]; continue
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if growth(W, np.exp(mid), starts) > 1:
                hi = mid
            else:
                lo = mid
        ge = float(np.exp(0.5 * (lo + hi)))
        out[P][f"{r:g}"] = [ge, cp[f"{r:g}"]]
        print(P, r, "direct edge", round(ge, 4), "file", round(cp[f"{r:g}"], 4) if cp[f"{r:g}"] < 1e10 else "inf", flush=True)
    json.dump(out, open(f"{R3}/verify_module_theory_sledge.json", "w"), indent=1)
print("saved results/pd3/verify_module_theory_sledge.json")
