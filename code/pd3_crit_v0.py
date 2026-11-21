"""pd3 criticality V0: validate pd3_avalanche.py (frontier) against swc.py --exp spread (dense) on the full brain, r = 1.

Both start from one seed drawn uniformly from the 366 excitatory AL-module neurons (seeds_module_exc366.npz), h = 0.
Compared at every swc.py record step a (<= the avalanche cap):
  P(a) = fraction alive; E[n(a)] = mean number active over ALL replicas (extinct = 0).
  swc.py: B = 512 replicas, SE from the replica spread. pd3_avalanche: agg_alive / den, agg_n / den (den(a) = avalanches
  observed through age a), SE from agg_n2.
Pass (pre-registered, plan 4.2): every recorded t within 2 combined SE (two-sample z with the pooled variance) for P and E[n], and a two-sample KS test on the
extinction times (both discretised on the swc.py record grid; capped / alive-at-T = beyond the grid) with p > 0.01.
Also: timing, and the plausibility check against spreading.md P(100) = 0.486 [0.441, 0.531] (seeds ~ |v|^2; not a criterion).
Writes results/pd3/criticality/v0_validation.json.
"""
import json
import numpy as np
from scipy import stats

ROOT = "/MALECNS_ROOT"
R = f"{ROOT}/results/pd3/criticality"
out = {}
import os
REFS = [("0.20", "v0_swc_r1_g0.20.npz", "B512_preregistered"), ("0.36", "v0_swc_r1_g0.36.npz", "B512_preregistered"),
        ("0.20", "v0b_swc_r1_g0.20_B4096.npz", "B4096_followup"), ("0.36", "v0b_swc_r1_g0.36_B2048.npz", "B2048_followup")]
for gs, swf, lab in [x_ for x_ in REFS if os.path.exists(f"{R}/v0/{x_[1]}") and os.path.exists(f"{R}/v0/aval_v0_real_r1_g{x_[0]}.npz")]:
    sw = np.load(f"{R}/v0/{swf}", allow_pickle=True); av = np.load(f"{R}/v0/aval_v0_real_r1_g{gs}.npz", allow_pickle=True)
    msw = json.loads(str(sw["meta"])); mav = json.loads(str(av["meta"]))
    N = 167216; dt = 0.2
    rec = np.round(sw["t"] / dt).astype(int)
    rho = sw["rho"][0].astype(np.float64); surv = sw["surv"][0].astype(np.float64)       # [nt, B]
    Bsw = rho.shape[1]
    En_sw = (rho * N).mean(1); En_sw_se = (rho * N).std(1, ddof=1) / np.sqrt(Bsw)
    P_sw = surv.mean(1); P_sw_se = np.sqrt(np.maximum(P_sw * (1 - P_sw), 1.0 / Bsw ** 2) / Bsw)
    cap = int(av["cap"]); gstep = int(av["gstep"])
    s0 = av["rec_s0"]; se = av["rec_s_end"].copy(); stt_all = av["rec_status"]
    # final epoch complete (no avalanche still running at the final stop) -> every avalanche of it is fully observed
    final_complete = not np.any((stt_all == 2) & (se < 0))
    se[se < 0] = (gstep + cap + 1) if final_complete else gstep
    win = se - s0
    hw = np.bincount(np.minimum(win, cap), minlength=cap + 1)
    den = hw[::-1].cumsum()[::-1].astype(np.float64)                                        # den[a] = #{win >= a}
    En_av = av["agg_n"] / den; var = av["agg_n2"] / den - En_av ** 2; En_av_se = np.sqrt(np.maximum(var, 0) / den)
    P_av = av["agg_alive"] / den; P_av_se = np.sqrt(np.maximum(P_av * (1 - P_av), 1.0 / den ** 2) / den)
    k = rec <= cap
    a_ = rec[k]
    # two-sample z with the POOLED variance (under H0 both runs sample one distribution; avoids SE = 0 when all 512 dense
    # replicas are dead at late t)
    n1 = den[a_]; n2 = float(Bsw)
    pP = (P_av[a_] * n1 + P_sw[k] * n2) / (n1 + n2)
    zP = (P_av[a_] - P_sw[k]) / np.sqrt(np.maximum(pP * (1 - pP), 1e-300) * (1 / n1 + 1 / n2))
    v_av = np.maximum(av["agg_n2"][a_] / n1 - En_av[a_] ** 2, 0); v_sw = (rho[k] * N).var(1, ddof=1)
    mpool = (En_av[a_] * n1 + En_sw[k] * n2) / (n1 + n2)
    vpool = (n1 * (v_av + (En_av[a_] - mpool) ** 2) + n2 * (v_sw + (En_sw[k] - mpool) ** 2)) / (n1 + n2)
    zN = np.where(vpool > 0, (En_av[a_] - En_sw[k]) / np.sqrt(np.maximum(vpool, 1e-300) * (1 / n1 + 1 / n2)), 0.0)
    zP = np.where(pP > 0, zP, 0.0)
    # KS on extinction times discretised on the record grid
    full = win >= cap
    T = av["rec_T"][full]; stt = av["rec_status"][full]
    kav = np.searchsorted(rec, T, side="left").astype(float); kav[stt == 1] = len(rec)   # first record step >= T (dead there)
    dead_sw = surv == 0
    ksw = np.where(dead_sw.any(0), dead_sw.argmax(0), len(rec)).astype(float)
    ks = stats.ks_2samp(kav, ksw)
    # plausibility: P at t = 100 model time (step 500)
    a100 = 500
    row = {"g": float(gs), "x": float(gs) / 0.16315274988719286, "reference": swf, "swc_B": Bsw, "aval_n_full_window": int(full.sum()),
           "aval_n_records": int(len(s0)), "aval_status_counts": np.bincount(av["rec_status"], minlength=3).tolist(),
           "n_compared_t": int(k.sum()),
           "P_max_abs_z": float(np.max(np.abs(zP))), "P_n_beyond_2SE": int((np.abs(zP) > 2).sum()),
           "En_max_abs_z": float(np.max(np.abs(zN))), "En_n_beyond_2SE": int((np.abs(zN) > 2).sum()),
           "KS_D": float(ks.statistic), "KS_p": float(ks.pvalue),
           "P_at_100": {"aval": float(P_av[a100]), "aval_se": float(P_av_se[a100]), "swc": float(P_sw[np.argmin(np.abs(rec - a100))]), "swc_step": int(rec[np.argmin(np.abs(rec - a100))])},
           "En_at": {str(int(t)): {"aval": float(En_av[t]), "swc": float(En_sw[rec == t][0])} for t in [1, 5, 25, 100, 500, 2000, 3000] if (rec == t).any() and t <= cap},
           "timing": {"aval_ms_per_step": mav["ms_per_step_mean"], "aval_steps": mav["steps"], "aval_wall_s": mav["wall_s"],
                      "aval_gpu": mav["gpu"], "aval_ms_per_avalanche": 1e3 * mav["loop_s"] / max(len(s0), 1),
                      "aval_edges_total": mav["edges_total"]},
           "curves": {"t": (a_ * dt).tolist(), "P_av": P_av[a_].tolist(), "P_av_se": P_av_se[a_].tolist(), "P_sw": P_sw[k].tolist(),
                      "P_sw_se": P_sw_se[k].tolist(), "En_av": En_av[a_].tolist(), "En_av_se": En_av_se[a_].tolist(),
                      "En_sw": En_sw[k].tolist(), "En_sw_se": En_sw_se[k].tolist(), "zP": zP.tolist(), "zN": zN.tolist()}}
    row["pass_P_2SE_all_t"] = row["P_n_beyond_2SE"] == 0
    row["pass_En_2SE_all_t"] = row["En_n_beyond_2SE"] == 0
    row["pass_KS"] = row["KS_p"] > 0.01
    out[f"{gs}_{lab}"] = row
    print(f"g={gs} [{lab}]: P max|z| {row['P_max_abs_z']:.2f} ({row['P_n_beyond_2SE']}/{row['n_compared_t']} beyond 2 SE); E[n] max|z| "
          f"{row['En_max_abs_z']:.2f} ({row['En_n_beyond_2SE']} beyond); KS D {row['KS_D']:.4f} p {row['KS_p']:.3g}; "
          f"P(100) aval {row['P_at_100']['aval']:.4f} swc {row['P_at_100']['swc']}; "
          f"{row['timing']['aval_ms_per_step']:.1f} ms/step, {row['timing']['aval_ms_per_avalanche']:.3f} ms/avalanche on {row['timing']['aval_gpu']}")
out["note_followup"] = ("the B = 512 references (pre-registered) showed a correlated 1.5-3 SE excess of E[n(t)] at g = 0.20 and "
                       "0.36 while P(t) and the KS test agreed; the frontier matches the exact E[n(1 step)] computed from the "
                       "graph (1.1256 vs 1.1253 at g 0.20); larger dense references (B 4096 / 2048, other seeds) were run to "
                       "separate a heavy-tail sampling fluctuation of the B = 512 sample from a bias")
out["exact_En1"] = {"0.20": 1.1253, "0.36": 1.3696, "source": "per-seed sum over out-edges, computed in-session (V0 notes)"}
out["note"] = ("plausibility only: spreading.md P(100) = 0.486 [0.441, 0.531] at g = 0.36 used seeds ~ |v|^2 over the top-1000 "
               "neurons, not uniform over the 366 excitatory module neurons")
json.dump(out, open(f"{R}/v0_validation.json", "w"), indent=1)
