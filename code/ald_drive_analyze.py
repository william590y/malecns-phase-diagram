"""E1 analysis: ORN drive / antenna removal (2026-11-09). Reads results/aldef/drive/runs/*.npz (from ald_drive_args.py),
writes results/aldef/drive/summary.json, table.md and the per-(proto, r, level, g) rows used by ald_drive_figs.py.

Every quantity is defined in results/aldef/drive/plan.md (criteria fixed before the runs). Notation:
  driven      window 0 = 500 < t <= 1500 (ORN drive on), per-replica means from win_mean.
  G           rho_uPN_ant / rho_ORN (replica means; 95% CI = bootstrap over the 64 replicas, ratio of means).
  resid_w     mean uPN_ant activity in window w after removal / its driven value.
  t_1e, t_10  first recorded time after the switch at which the replica-mean uPN_ant activity is <= driven/e, <= driven/10
              (NaN = never within 3000 units).
  p_alive     fraction of replicas whose AL system (AL without ORNs, union AL module) is active at the end (t = 4500).
  ext_med     median time from removal to the last AL-system activity (censored at 3000).
  chi_time    n_module * mean_b var_t(rho_module) / mean rho_module in the driven window; chi_rep the same with the
              variance over replicas of the time means.
"""
import glob, json, os, re
import numpy as np

ROOT = "/MALECNS_ROOT"
D = f"{ROOT}/results/aldef/drive"
T1, TEND = 1500.0, 4500.0
GAIN_P, GAIN_B = (0.29, 1.45), (0.077, 1.67)
H0 = {0: {"g_lin": 0.1632, "S": 0.234, "onset": 0.266}, 1: {"g_lin": 0.1632, "S": 0.283, "onset": 0.344}}
rng = np.random.default_rng(0)
levels = json.load(open(f"{D}/levels.json"))


def boot_ratio(a, b, n=2000):
    i = rng.integers(0, len(a), (n, len(a)))
    r = a[i].mean(1) / np.maximum(b[i].mean(1), 1e-12)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


rows = []
files = sorted(glob.glob(f"{D}/runs/*.npz"))
for f in files:
    m = re.match(r"(all|ant)_r(\d)_o([\d.]+)_c(\d)\.npz", os.path.basename(f))
    proto, r, lev = m.group(1), int(m.group(2)), m.group(3)
    d = np.load(f)
    names = list(d["reg_names"]); ix = {n: i for i, n in enumerate(names)}
    sizes = d["reg_sizes"]
    wm, w2, wb = d["win_mean"], d["win_m2"], d["win_bounds"]          # [g, W, R, B]
    last = d["reg_last"]; t = d["t"]; reg = d["reg_rho"]               # reg [g, t, R, B]
    after = t > T1 + 1e-9
    for gi, g in enumerate(d["g"]):
        row = {"proto": proto, "r": r, "level": lev, "rho_iso": levels[lev]["rho_iso"], "g": float(g), "file": os.path.basename(f)}
        drv = {k: float(wm[gi, 0, ix[k]].mean()) for k in names}
        row["driven"] = drv
        oA, pA = wm[gi, 0, ix["ORN"]], wm[gi, 0, ix["uPN_ant"]]
        row["G"] = drv["uPN_ant"] / drv["ORN"]
        row["G_ci"] = boot_ratio(pA, oA)
        row["G_uPN"] = drv["uPN"] / drv["ORN"]
        row["G_PNall"] = drv["PN_all"] / drv["ORN"]
        # per-PN rates in the driven window (occ = per-neuron replica mean over 500 < t <= 1500)
        occ = d["occ"][gi].astype(np.float64)
        upn_idx = np.load(f"{D}/groups/uPN_ant.npz")["idx"]
        rp = occ[upn_idx] / max(drv["ORN"], 1e-12)
        row["uPN_ant_ratio_pct"] = [float(np.percentile(rp, q)) for q in (10, 50, 90)]
        row["uPN_ant_frac_in_primary"] = float(((rp >= GAIN_P[0]) & (rp <= GAIN_P[1])).mean())
        # removal
        base = max(drv["uPN_ant"], 1e-12)
        row["resid"] = {f"{wb[w, 0] - T1:g}-{wb[w, 1] - T1:g}": float(wm[gi, w, ix["uPN_ant"]].mean() / base)
                        for w in range(1, len(wb))}
        row["resid_module"] = {f"{wb[w, 0] - T1:g}-{wb[w, 1] - T1:g}": float(wm[gi, w, ix["AL_module"]].mean() /
                                                                          max(drv["AL_module"], 1e-12)) for w in range(1, len(wb))}
        row["after_abs"] = {k: {f"{wb[w, 0] - T1:g}-{wb[w, 1] - T1:g}": float(wm[gi, w, ix[k]].mean())
                                for w in range(1, len(wb))} for k in ("uPN_ant", "uPN", "uPN_palp", "AL_module", "CNS_rest", "ORN", "lLN1_bc", "LN_exc", "LN_inh")}
        tr = reg[gi, :, ix["uPN_ant"], :].mean(1)
        ta, xa = t[after] - T1, tr[after]
        for nm, thr in (("t_1e", np.exp(-1)), ("t_10", 0.1), ("t_1", 0.01)):
            hit = np.where(xa <= thr * base)[0]
            row[nm] = float(ta[hit[0]]) if (len(hit) and drv["uPN_ant"] > 0) else float("nan")
        la = last[gi, ix["ALsys"]]
        row["p_alive"] = float((la >= TEND - 1e-3).mean())
        row["p_alive_uPN"] = float((last[gi, ix["uPN_ant"]] >= TEND - 1e-3).mean())
        row["ext_med"] = float(np.median(np.clip(la - T1, 0, TEND - T1)))          # never active after removal -> 0
        row["ext_med_uPN"] = float(np.median(np.clip(last[gi, ix["uPN_ant"]] - T1, 0, TEND - T1)))
        row["ext_mean_dead"] = float((la[la < TEND - 1e-3] - T1).mean()) if (la < TEND - 1e-3).any() else float("nan")
        # module fluctuations in the driven window
        n_mod = float(sizes[ix["AL_module"]])
        mm, m2 = wm[gi, 0, ix["AL_module"]].astype(np.float64), w2[gi, 0, ix["AL_module"]].astype(np.float64)
        rho_m = max(mm.mean(), 1e-12)
        row["chi_time"] = float(n_mod * np.mean(np.maximum(m2 - mm ** 2, 0)) / rho_m)
        row["chi_rep"] = float(n_mod * mm.var(ddof=1) / rho_m)
        # criteria (plan.md)
        row["gain_primary"] = bool(GAIN_P[0] <= row["G"] <= GAIN_P[1])
        row["gain_broad"] = bool(GAIN_B[0] <= row["G"] <= GAIN_B[1])
        r100 = row["resid"]["100-300"]; r1000 = row["resid"]["1000-3000"]
        if proto == "all":
            row["S1"] = bool(r100 <= 0.10); row["S2"] = bool(row["p_alive"] <= 0.05)
            row["S1_1pct"] = bool(r100 <= 0.01)
            row["S2_uPN"] = bool(row["p_alive_uPN"] <= 0.05)
        else:
            row["S1p"] = bool(r100 <= 0.10); row["S2p"] = bool(r1000 <= 0.10)
            row["S1p_1pct"] = bool(r100 <= 0.01); row["S2p_1pct"] = bool(r1000 <= 0.01)
        rows.append(row)

rows.sort(key=lambda x: (x["proto"], x["r"], x["rho_iso"], x["g"]))
json.dump(rows, open(f"{D}/rows.json", "w"), indent=1)


def crossing(gs, y, thr, logy=False, rising=True):
    """First g where y crosses thr (linear interpolation in g; in log y if logy), scanning upward in g."""
    gs, y = np.asarray(gs, float), np.asarray(y, float)
    for i in range(len(gs) - 1):
        a, b = y[i], y[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if (rising and a < thr <= b) or (not rising and a > thr >= b):
            if logy and a > 0 and b > 0:
                a, b, thr_ = np.log(a), np.log(b), np.log(thr)
            else:
                thr_ = thr
            return float(gs[i] + (thr_ - a) / (b - a) * (gs[i + 1] - gs[i]))
    return None


def ranges(gs, ok):
    """Contiguous g intervals (as [first, last] grid points) where ok is True."""
    out, cur = [], None
    for g, o in zip(gs, ok):
        if o and cur is None:
            cur = [g, g]
        elif o:
            cur[1] = g
        elif cur is not None:
            out.append(cur); cur = None
    if cur is not None:
        out.append(cur)
    return out


summary = {"n_files": len(files), "n_rows": len(rows), "criteria": {"gain_primary": GAIN_P, "gain_broad": GAIN_B},
           "h0_boundaries": H0, "per": {}}
md = ["# E1 ORN drive / antenna removal: allowed g ranges (generated by code/ald_drive_analyze.py)", "",
      "Criteria: plan.md. 'all' = every ORN clamped at removal; 'ant' = antennal ORNs only (palps still driven).", ""]
for r in (0, 1):
    for lev in levels:
        A = [x for x in rows if x["proto"] == "all" and x["r"] == r and x["level"] == lev]
        P = {x["g"]: x for x in rows if x["proto"] == "ant" and x["r"] == r and x["level"] == lev}
        if not A:
            continue
        gs = [x["g"] for x in A]
        ok_gain = [x["gain_primary"] for x in A]; ok_gb = [x["gain_broad"] for x in A]
        ok_sil = [x["S1"] and x["S2"] for x in A]
        ok = [a and b for a, b in zip(ok_gain, ok_sil)]
        okb = [a and b for a, b in zip(ok_gb, ok_sil)]
        okp = [a and (g in P and P[g]["S1p"] and P[g]["S2p"]) for a, g in zip(ok, gs)]
        okpb = [a and (g in P and P[g]["S1p"] and P[g]["S2p"]) for a, g in zip(okb, gs)]
        key = f"r{r}_o{lev}"
        summary["per"][key] = {
            "r": r, "rho_iso": levels[lev]["rho_iso"], "g": gs,
            "gain_primary_ranges": ranges(gs, ok_gain), "gain_broad_ranges": ranges(gs, ok_gb),
            "silent_all_ranges": ranges(gs, ok_sil),
            "silent_S1_ranges": ranges(gs, [x["S1"] for x in A]), "silent_S2_ranges": ranges(gs, [x["S2"] for x in A]),
            "silent_ant_ranges": ranges(gs, [g in P and P[g]["S1p"] and P[g]["S2p"] for g in gs]),
            "allowed_primary": ranges(gs, ok), "allowed_broad": ranges(gs, okb),
            "allowed_primary_with_palp_test": ranges(gs, okp), "allowed_broad_with_palp_test": ranges(gs, okpb),
            "G_at": {f"{x['g']:g}": round(x["G"], 4) for x in A},
            "max_G_allowed_broad": max([x["G"] for x, o in zip(A, okb) if o], default=None),
            "chi_time_peak_g": gs[int(np.nanargmax([x["chi_time"] if 0.1 <= x["g"] <= 1.0 else np.nan for x in A]))],
            "chi_rep_peak_g": gs[int(np.nanargmax([x["chi_rep"] if 0.1 <= x["g"] <= 1.0 else np.nan for x in A]))],
        }
        s = summary["per"][key]
        Gv = [x["G"] for x in A]
        r100 = [x["resid"]["100-300"] for x in A]; pal = [x["p_alive"] for x in A]
        cx = {"G=0.29": crossing(gs, Gv, GAIN_P[0], True), "G=1.45": crossing(gs, Gv, GAIN_P[1], True),
              "G=0.077": crossing(gs, Gv, GAIN_B[0], True), "G=1.67": crossing(gs, Gv, GAIN_B[1], True),
              "S1 resid100-300=0.1": crossing(gs, r100, 0.1), "S1 resid100-300=0.01": crossing(gs, r100, 0.01),
              "S2 p_alive=0.05": crossing(gs, pal, 0.05), "p_alive=0.5": crossing(gs, pal, 0.5)}
        gp = sorted(P)
        cx["palp S1p resid100-300=0.1"] = crossing(gp, [P[g]["resid"]["100-300"] for g in gp], 0.1)
        cx["palp S2p resid1000-3000=0.1"] = crossing(gp, [P[g]["resid"]["1000-3000"] for g in gp], 0.1)
        cx["palp S2p resid1000-3000=0.01"] = crossing(gp, [P[g]["resid"]["1000-3000"] for g in gp], 0.01)
        s["crossings"] = cx
        up = [v for k2, v in cx.items() if k2 in ("S1 resid100-300=0.1", "S2 p_alive=0.05") and v is not None]
        s["allowed_interp_primary"] = [cx["G=0.29"], min([v for v in up + [cx["G=1.45"]] if v is not None], default=None)]
        s["allowed_interp_broad"] = [cx["G=0.077"], min([v for v in up + [cx["G=1.67"]] if v is not None], default=None)]
        pal_up = [cx["palp S1p resid100-300=0.1"], cx["palp S2p resid1000-3000=0.1"]]
        s["allowed_interp_primary_palp"] = [cx["G=0.29"], min([v for v in [s["allowed_interp_primary"][1]] + pal_up if v is not None], default=None)]
        s["allowed_interp_broad_palp"] = [cx["G=0.077"], min([v for v in [s["allowed_interp_broad"][1]] + pal_up if v is not None], default=None)]
        # recurrent amplification: G/g relative to its value at the smallest g > 0 (feed-forward slope)
        g0 = [x for x in A if x["g"] > 0][0]
        slope0 = g0["G"] / g0["g"]
        s["ff_slope_at_g"] = [g0["g"], slope0]
        s["amplification"] = {f"{x['g']:g}": round(x["G"] / (slope0 * x["g"]), 4) for x in A if x["g"] > 0}
        for x in A:
            if x["g"] > 0:
                x["amp"] = x["G"] / (slope0 * x["g"])
        md.append(f"## r = {r}, ORN level rho_iso = {lev} ({levels[lev]['ORN_Hz_at_tau10ms']:.1f} Hz at tau_u 10 ms; "
                  f"tau_u {levels[lev]['tau_u_ms_if_ORN_3.44Hz']:.1f} ms if ORN 3.44 Hz)")
        md.append(f"- gain in primary window at g: {s['gain_primary_ranges']}; broad: {s['gain_broad_ranges']}")
        md.append(f"- silent after removal (S1 and S2): {s['silent_all_ranges']}; palp-intact silent: {s['silent_ant_ranges']}")
        md.append(f"- ALLOWED (primary gain): {s['allowed_primary']}; broad gain: {s['allowed_broad']}; "
                  f"with palp test: {s['allowed_primary_with_palp_test']} / {s['allowed_broad_with_palp_test']}")
        md.append(f"- interpolated crossings: " + ", ".join(f"{k2} at {v:.3f}" if v is not None else f"{k2}: none"
                                                         for k2, v in cx.items()))
        md.append(f"- interpolated allowed interval: primary {s['allowed_interp_primary']}, broad {s['allowed_interp_broad']}; "
                  f"with palp test: primary {s['allowed_interp_primary_palp']}, broad {s['allowed_interp_broad_palp']}")
        md.append(f"- AL-module fluctuation peak (driven window, 0.1 <= g <= 1): chi_time at g = {s['chi_time_peak_g']}, "
                  f"chi_rep at g = {s['chi_rep_peak_g']}")
        md.append("")
        md.append("| g | rho_ORN | rho_uPN_ant | G [95% CI] | AL module | CNS rest | resid 20-50 | resid 100-300 | resid 1000-3000 | t_1/e | t_10% | p_alive | ext_med | palp-intact resid 100-300 / 1000-3000 | chi_time | ok |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x, o, ob in zip(A, ok, okb):
            pp = P.get(x["g"])
            pr = f"{pp['resid']['100-300']:.3g} / {pp['resid']['1000-3000']:.3g}" if pp else "–"
            md.append(f"| {x['g']:g} | {x['driven']['ORN']:.4g} | {x['driven']['uPN_ant']:.4g} | {x['G']:.3g} "
                      f"[{x['G_ci'][0]:.3g}, {x['G_ci'][1]:.3g}] | {x['driven']['AL_module']:.3g} | {x['driven']['CNS_rest']:.2g} | "
                      f"{x['resid']['20-50']:.3g} | {x['resid']['100-300']:.3g} | {x['resid']['1000-3000']:.3g} | "
                      f"{x['t_1e']:.3g} | {x['t_10']:.3g} | {x['p_alive']:.3g} | {x['ext_med']:.4g} | {pr} | "
                      f"{x['chi_time']:.3g} | {'YES' if o else ('broad' if ob else '')} |")
        md.append("")
json.dump(summary, open(f"{D}/summary.json", "w"), indent=1)
json.dump(rows, open(f"{D}/rows.json", "w"), indent=1)
open(f"{D}/table.md", "w").write("\n".join(md) + "\n")
print(f"{len(files)} files, {len(rows)} rows")
for k, s in summary["per"].items():
    print(k, "gainP", s["gain_primary_ranges"], "silent", s["silent_all_ranges"], "ALLOWED", s["allowed_primary"],
          "broad", s["allowed_broad"], "palp", s["allowed_primary_with_palp_test"], "chi", s["chi_time_peak_g"], s["chi_rep_peak_g"],
          "interp", np.round(s["allowed_interp_primary"], 3).tolist(), np.round(s["allowed_interp_broad"], 3).tolist(),
          "palp-interp", np.round(s["allowed_interp_primary_palp"], 3).tolist(), np.round(s["allowed_interp_broad_palp"], 3).tolist())
