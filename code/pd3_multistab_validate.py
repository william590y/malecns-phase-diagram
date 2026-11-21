"""Validation of pd3_multistab.py (plan.md 5.1): all-active start, r = 1, B = 64 per g, T = 3000; tau_sim (censored-exponential
MLE, the phase_complete estimator) must fall inside the 95% CI of the pooled phase_complete point at the same g.
g values without a pooled point at the same g (0.344, 0.40) are compared with the CI envelope of the two neighbouring pooled points.
Extinction times: exact step (ext_step) and, for the like-for-like comparison, snapped up to swc.py's log record grid (phase_complete
uses the first recorded time with surv = 0 on that grid). Also a likelihood-ratio test of equal exponential rates (chi2, 1 dof)
against the pooled point when one exists.
Output: results/pd3/multistab/validation.json
"""
import json, sys
import numpy as np
from scipy import stats

sys.path.insert(0, "/MALECNS_ROOT/code")
from pd3_common import censored_mle

ROOT = "/MALECNS_ROOT"
DT = 0.2


def lr_test(d1, E1, d2, E2):
    def ll(d, E, lam):
        return d * np.log(lam) - lam * E if d > 0 else -lam * E
    if d1 + d2 == 0:
        return 1.0
    l1, l2, l0 = d1 / E1, d2 / E2, (d1 + d2) / (E1 + E2)
    LR = 2 * ((ll(d1, E1, l1) if d1 else 0.0) + (ll(d2, E2, l2) if d2 else 0.0) - ll(d1 + d2, E1 + E2, l0))
    return float(stats.chi2.sf(max(LR, 0.0), 1))


def main():
    pc = json.load(open(f"{ROOT}/results/analysis2/phase_complete.json"))["sim"]["real"]["1"]["points"]
    pg = np.array([p["g"] for p in pc])
    out = {"source_pooled": "results/analysis2/phase_complete.json sim.real['1'].points", "runs": {}}
    for tag in ["", "_split"]:
        f = f"{ROOT}/results/pd3/multistab/validate_r1{tag}.npz"
        z = np.load(f, allow_pickle=True); meta = json.loads(str(z["meta"]))
        G, B, T = meta["g"], meta["B"], meta["T"]
        steps = int(round(T / DT))
        rec = np.unique(np.round(np.logspace(0, np.log10(steps), 120)).astype(int))        # swc.py record grid
        ext = z["ext_step"]
        rows = []
        for i, g in enumerate(G):
            e = ext[i * B:(i + 1) * B]
            died = e >= 0
            t_exact = np.where(died, e * DT, T)
            snapped = np.where(died, rec[np.minimum(np.searchsorted(rec, np.maximum(e, 1)), len(rec) - 1)] * DT, T)
            m_ex = censored_mle(t_exact, died); m_sn = censored_mle(snapped, died)
            row = {"g": g, "B": B, "deaths": m_sn["deaths"], "tau_exact": m_ex["tau"], "tau_snapped": m_sn["tau"],
                   "tau_ci_snapped": m_sn["tau_ci"], "alive_T": float(1 - died.mean())}
            j = np.where(np.abs(pg - g) / g < 1e-4)[0]
            if len(j):
                p = pc[int(j[0])]
                Ep = p["tau_sim"] * p["deaths"] if p["deaths"] > 0 else None
                row["pooled"] = {"g": p["g"], "B": p["B"], "deaths": p["deaths"], "tau_sim": p["tau_sim"], "tau_ci": p["tau_ci"],
                                 "src": p["src"]}
                row["inside_pooled_ci"] = bool(p["tau_ci"][0] <= m_sn["tau"] <= p["tau_ci"][1]) if np.isfinite(m_sn["tau"]) else \
                    bool(m_sn["tau_ci"][0] <= p["tau_ci"][1])
                row["lr_p_equal_rate"] = lr_test(m_sn["deaths"], m_sn["E"], p["deaths"], Ep) if Ep else None
                row["comparison"] = "same-g pooled point"
            else:
                lo = pc[int(np.where(pg < g)[0][-1])]; hi = pc[int(np.where(pg > g)[0][0])]
                env = [min(lo["tau_ci"][0], hi["tau_ci"][0]), max(lo["tau_ci"][1], hi["tau_ci"][1])]
                row["neighbours"] = [{"g": q["g"], "tau_sim": q["tau_sim"], "tau_ci": q["tau_ci"], "src": q["src"]} for q in (lo, hi)]
                row["envelope"] = env
                row["inside_pooled_ci"] = bool(env[0] <= m_sn["tau"] <= env[1]) if np.isfinite(m_sn["tau"]) else bool(m_sn["tau_ci"][0] <= env[1])
                row["comparison"] = "envelope of the CIs of the two neighbouring pooled points"
            rows.append(row)
            print(f"{tag or 'fast'} g={g:.4g}: deaths {row['deaths']}/{B} tau={m_sn['tau']:.1f} (exact {m_ex['tau']:.1f}) "
                  f"CI {m_sn['tau_ci'][0]:.0f}-{m_sn['tau_ci'][1]:.0f}; pooled "
                  + (f"{row['pooled']['tau_sim']:.1f} [{row['pooled']['tau_ci'][0]:.0f}, {row['pooled']['tau_ci'][1]:.0f}] LR p={row['lr_p_equal_rate']}"
                     if "pooled" in row else f"envelope {env[0]:.0f}-{env[1]:.0f}") + f" -> inside {row['inside_pooled_ci']}")
        out["runs"]["fast" if not tag else "split"] = {"file": f.replace(ROOT + "/", ""), "gpu": meta["gpu"],
                                                       "seconds": meta["seconds"], "rows": rows,
                                                       "pass_spec_points": all(r["inside_pooled_ci"] for r in rows if r["g"] in (0.30, 0.344, 0.40)),
                                                       "pass_all_points": all(r["inside_pooled_ci"] for r in rows)}
    out["pass"] = all(v["pass_spec_points"] for v in out["runs"].values())
    json.dump(out, open(f"{ROOT}/results/pd3/multistab/validation.json", "w"), indent=1)
    print("PASS" if out["pass"] else "FAIL", {k: (v["pass_spec_points"], v["pass_all_points"]) for k, v in out["runs"].items()})


if __name__ == "__main__":
    main()
