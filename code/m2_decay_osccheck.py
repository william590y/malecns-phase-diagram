"""m2_decay_osccheck.py -- plan section 9 (lane m2/decay): is the r = 0 OSC flag a finite-size quasi-cycle or a
macroscopic oscillation? Reads the small-graph runs sim/x9_*.npz (index sim/osccheck_index.json) and, for N = 167,216,
the raw stage-1/2 runs of rows M2_cnull_r0 (configuration null) and M2_real_r0 (real graph). Writes
results/m2/decay/osccheck.json and figs/m2/decay_osc_fss.png.

Per (graph, N, g) with >= 4 replicas alive at T: x(t) = active fraction (alive replicas, 1500 < t <= 3000), the OSC
metrics of m2_obs, var_t(x) (mean over replicas) and rho_QS = mean x. Decision (pre-registered): slope of log var_t
against log N at matched rho_QS <= -0.7 -> quasi-cycle; >= -0.3 -> macroscopic; otherwise inconclusive.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_obs as O
import m2_decay_analyze as A

OUT = A.OUT; SIMD = A.SIMD; DT = 0.2
FIG = f"{A.ROOT}/figs/m2"


def points_of(path, N):
    d = np.load(path)
    T = float(json.loads(str(d["meta"]))["T"])
    names = [str(x) for x in d["reg_names"]]
    di = [names.index(n) for n in ("AL_module", "AL_rest", "CNS_rest")]
    res = []
    for i, g in enumerate(d["g"]):
        last = d["reg_last"][i][di].max(0)
        alive = last >= T - DT / 2
        if alive.sum() < 4:
            res.append({"g": float(g), "n_alive": int(alive.sum())}); continue
        x = d["cnt_act"][i][:, alive].astype(np.float64) / N
        m = O.osc_metrics(x, DT)
        res.append({"g": float(g), "n_alive": int(alive.sum()), "rho_qs": float(x.mean()), "var_t": float(x.var(0).mean()),
                    "N_var": float(N * x.var(0).mean()), "osc": m["osc"], "osc_strong": m["osc_strong"], "c_min": m["c_min"],
                    "c_peak": m["c_peak"], "period": m["period"], "coh": m["coh_ratio"], "cv_t": m["cv_t"]})
    return res


def main():
    ix = json.load(open(f"{SIMD}/osccheck_index.json"))
    fam = {"cnull": {}, "realsub": {}}
    for e in ix:
        if not os.path.exists(e["out"]):
            continue
        kind = "cnull" if e["graph"].startswith("cnull") else "realsub"
        fam[kind].setdefault(e["N"], []).extend(points_of(e["out"], e["N"]))
    # full size: raw runs of the rows (stage 1 + 2)
    idx = A.load_index()
    for kind, row in (("cnull", ("M2", "cnull", 0)), ("realsub", ("M2", "real", 0))):
        for b, e in idx.items():
            if (e["model"], e["graph"], e["r"]) == row and e["stage"] in ("s1", "s2") and os.path.exists(e["out"]):
                fam[kind].setdefault(167216, []).extend(points_of(e["out"], 167216))
    res = {"definitions": __doc__, "families": {}}
    for kind, byN in fam.items():
        Ns = sorted(byN)
        curves = {}
        for N in Ns:
            P = sorted([p for p in byN[N] if "rho_qs" in p], key=lambda p: p["rho_qs"])
            curves[N] = P
        # matched-rho slopes
        have = [N for N in Ns if len(curves[N]) >= 2]
        slopes = []
        if len(have) >= 2:
            lo = max(min(p["rho_qs"] for p in curves[N]) for N in have)
            hi = min(max(p["rho_qs"] for p in curves[N]) for N in have)
            if hi > lo:
                for rq in np.geomspace(lo, hi, 5):
                    lv = []
                    for N in have:
                        r_ = np.log([p["rho_qs"] for p in curves[N]]); v_ = np.log([p["var_t"] for p in curves[N]])
                        o = np.argsort(r_)
                        lv.append(np.interp(np.log(rq), r_[o], v_[o]))
                    sl = float(np.polyfit(np.log(have), lv, 1)[0])
                    slopes.append({"rho_qs": float(rq), "slope": sl, "log_var": [float(x) for x in lv]})
        med = float(np.median([s["slope"] for s in slopes])) if slopes else None
        verdict = (None if med is None else "quasi-cycle (finite-size)" if med <= -0.7
                   else "macroscopic oscillation" if med >= -0.3 else "inconclusive")
        res["families"][kind] = {"N": Ns, "points": {str(N): curves[N] for N in Ns},
                                 "osc_points": {str(N): [p["g"] for p in curves[N] if p["osc"]] for N in Ns},
                                 "matched_rho_slopes": slopes, "median_slope": med, "verdict": verdict,
                                 "sizes_used": have}
        print(kind, "N", Ns, "median slope", med, verdict)
    json.dump(A_clean(res), open(f"{OUT}/osccheck.json", "w"), indent=1)
    fig(res)


def A_clean(x):
    import m2_decay_report as R
    return R.clean(x)


def fig(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for j, (kind, lab) in enumerate((("cnull", "configuration nulls, size N"), ("realsub", "real subsamples keeping the AL"))):
        F = res["families"][kind]; cm = plt.get_cmap("viridis")
        Ns = F["N"]
        for i, N in enumerate(Ns):
            P = F["points"][str(N)]
            if not P:
                continue
            c = cm(i / max(len(Ns) - 1, 1))
            r = [p["rho_qs"] for p in P]
            axs[0, j].loglog(r, [p["var_t"] for p in P], "o-", color=c, ms=4, label=f"N = {N:,}")
            for p in P:
                if p["osc"]:
                    axs[0, j].plot(p["rho_qs"], p["var_t"], "o", ms=9, mfc="none", mec="red")
            axs[1, j].semilogx(r, [p["c_min"] for p in P], "o-", color=c, ms=4, label=f"N = {N:,} trough")
            axs[1, j].semilogx(r, [p["c_peak"] for p in P], "s--", color=c, ms=3, alpha=0.7)
        axs[0, j].set_xlabel("QS active fraction ρ_QS"); axs[0, j].set_ylabel("var_t of the active fraction")
        axs[0, j].set_title(f"{lab}\nmedian slope d log var / d log N = {F['median_slope'] if F['median_slope'] is None else round(F['median_slope'], 2)}"
                            f" → {F['verdict']}", fontsize=9)
        axs[0, j].legend(fontsize=7, frameon=False)
        axs[1, j].axhline(-0.1, color="black", ls=":", lw=0.8); axs[1, j].axhline(0.1, color="black", ls=":", lw=0.8)
        axs[1, j].set_xlabel("QS active fraction ρ_QS"); axs[1, j].set_ylabel("ACF trough (solid) and next peak (dashed)")
        axs[1, j].legend(fontsize=7, frameon=False)
    axs[0, 0].plot([], [], "o", mfc="none", mec="red", label="OSC flag"); axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("M2, r = 0, h = 0 decay: does the oscillation flag survive as N grows? (var ∝ 1/N = noise-driven quasi-cycle)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{FIG}/decay_osc_fss.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
