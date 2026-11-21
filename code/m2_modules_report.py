"""m2_modules_report.py -- lane m2/modules: tables.md + key_numbers.json from the lane's JSON outputs (CPU, no simulation).
Every number in REPORT.md is in one of these tables with its source file.
"""
import json, os
import numpy as np

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/m2/modules"
R_CORE = (0, 0.25, 0.5, 1, 2, 4, 6)
KN = {}


def f(x, n=3):
    if x is None:
        return "–"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{n}g}"


def part_a(md):
    A = json.load(open(f"{OUT}/phase_A.json")); R = A["rows"]
    b = json.load(open(f"{ROOT}/results/m2/decay/boundaries.json"))["rows"]
    a2 = json.load(open(f"{ROOT}/results/analysis2/phase_complete.json"))["sim"]["real"]
    cur = json.load(open(f"{ROOT}/results/m2/driven/curves.json"))
    md.append("## Part A: module phases from existing occupancies (`phase_A.json`)\n")
    md.append("### A1. Setting (a), h = 0: modules on at the first sustained grid point, and the SH line (P0, 34 modules)\n")
    md.append("| r | model | onset g (tau 3000) | first sustained grid g | n on (1%) | n on (0.1%) | modules on (1%, by X) | P2 groups on (1%) | "
              "median module 1% crossing / onset | first SH g (1%) | SH / onset | D (PR N/4) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    KN["A"] = {}
    for r in R_CORE:
        for model, key in (("M1", "M1_pd3"), ("M2", "M2_lane"), ("M2s", "M2s_lane")):
            k = f"a|{key}|{r:g}"
            if k not in R:
                continue
            row = R[k]; g = np.array(row["g"])
            p0 = row["parts"]["P0"]["theta"]; p2 = row["parts"]["P2"]["theta"]
            n1 = np.array(p0["0.01"]["n_on"]); n01 = np.array(p0["0.001"]["n_on"])
            i = int(np.argmax(n1 > 0)) if (n1 > 0).any() else None
            if model == "M1":
                on = a2[f"{r:g}"]["on"]["g"]; D = a2[f"{r:g}"]["D"]["g"]
            else:
                on = b[f"{model}_real_r{r:g}"]["on"]["g"]; D = b[f"{model}_real_r{r:g}"]["D"]["g"]
            cr = np.array([x for x in p0["0.01"]["crossing"].values() if x is not None])
            med = float(np.median(cr) / on) if len(cr) else None
            sh = p0["0.01"]["first_SH_g"]
            ids = p0["0.01"]["ids_on"][i] if i is not None else []
            ids2 = p2["0.01"]["ids_on"][i] if i is not None else []
            md.append(f"| {r:g} | {model} | {f(on)} | {f(g[i]) if i is not None else '–'} | {n1[i] if i is not None else '–'} | "
                      f"{n01[i] if i is not None else '–'} | {', '.join(ids[:8])}{' …' if len(ids) > 8 else ''} | "
                      f"{', '.join(ids2[:6])}{' …' if len(ids2) > 6 else ''} | {f(med)} | {f(sh)} | {f(sh / on) if sh else '–'} | {f(D)} |")
            KN["A"][f"{model}_r{r:g}"] = {"onset": on, "first_g": float(g[i]) if i is not None else None,
                                         "n_on_1pct": int(n1[i]) if i is not None else None, "n_on_0.1pct": int(n01[i]) if i is not None else None,
                                         "ids_1pct": ids, "P2_ids_1pct": ids2, "median_cross_over_onset": med, "first_SH": sh, "D": D,
                                         "n_modules_cross_le_1.25_onset": int((cr <= 1.25 * on).sum()), "n_cross": int(len(cr))}
    md.append("\nSource: `phase_A.json` rows `a|M1_pd3|r`, `a|M2_lane|r`, `a|M2s_lane|r` (P0, theta 0.01 / 0.001); onset and D from "
              "`results/m2/decay/boundaries.json` (M2, M2s) and `results/analysis2/phase_complete.json` (M1).\n")
    # all 23 r: n on at onset
    md.append("### A2. Number of P0 modules on at the first sustained point, every r (setting a)\n")
    rs = sorted({float(k.split("|")[2]) for k in R if k.startswith("a|M2_lane|")})
    md.append("| r | " + " | ".join(f"{r:g}" for r in rs) + " |")
    md.append("|---|" + "---|" * len(rs))
    for model, key in (("M1 (1%)", "M1_pd3"), ("M2 (1%)", "M2_lane"), ("M2 (0.1%)", "M2_lane")):
        th = "0.001" if "0.1%" in model else "0.01"
        vals = []
        for r in rs:
            k = f"a|{key}|{r:g}"
            n = np.array(R[k]["parts"]["P0"]["theta"][th]["n_on"]) if k in R else np.array([0])
            vals.append(str(int(n[np.argmax(n > 0)])) if (n > 0).any() else "–")
        md.append(f"| {model} | " + " | ".join(vals) + " |")
    md.append("")
    # (b)
    md.append("### A3. Setting (b), ORN drive 0.04: modules on (1%) below and at DT->SS (P0)\n")
    md.append("| r | model | DT->SS | n on at 0.5 x DT->SS | ids | n on at the grid point nearest DT->SS | first SH g | max AL mean tilt (g) |")
    md.append("|---|---|---|---|---|---|---|---|")
    KN["B_drive"] = {}
    for r in (0, 0.25, 0.5, 1, 2, 4, 6):
        for model, s in (("M1", "A2"), ("M2", "A1"), ("M2s", "A4")):
            k = f"b|{model}|{r:g}"; ck = f"{s}|{model}|real|0.04|{r:g}|False"
            if k not in R or ck not in cur:
                continue
            row = R[k]; g = np.array(row["g"]); dt = cur[ck]["lines"].get("DT_SS")
            th = row["parts"]["P0"]["theta"]["0.01"]; n = np.array(th["n_on"])
            i5 = int(np.argmin(np.abs(np.log(g / (0.5 * dt))))); i1 = int(np.argmin(np.abs(np.log(g / dt))))
            tl = row["LR_tilt_max_where_pair_ge_1pct"]["AL"]
            md.append(f"| {r:g} | {model} | {f(dt)} | {n[i5]} (g {f(g[i5])}) | {', '.join(th['ids_on'][i5][:5])} | {n[i1]} (g {f(g[i1])}) | "
                      f"{f(th['first_SH_g'])} | {f(tl['max_abs'])} ({f(tl['g_at_max'])}) |")
            KN["B_drive"][f"{model}_r{r:g}"] = {"DT_SS": dt, "n_on_half": int(n[i5]), "ids_half": th["ids_on"][i5],
                                               "n_on_at_DTSS": int(n[i1]), "first_SH": th["first_SH_g"], "AL_tilt_max": tl["max_abs"]}
    md.append("\nSource: `phase_A.json` rows `b|M1|r`, `b|M2|r`, `b|M2s|r`; DT->SS from `results/m2/driven/curves.json`.\n")
    # tilt summary
    tl_all = {k: max((v["max_abs"] or 0) for v in R[k]["LR_tilt_max_where_pair_ge_1pct"].values() if v["n_points"])
              for k in R if R[k]["LR_tilt_max_where_pair_ge_1pct"]}
    KN["A_tilt_max_any_pair"] = tl_all
    KN["A_tilt_max_AL"] = {k: R[k]["LR_tilt_max_where_pair_ge_1pct"]["AL"]["max_abs"] for k in R}


def main():
    md = ["# m2/modules generated tables (code/m2_modules_report.py)\n"]
    part_a(md)
    for fname, fn in (("lr", part_lr), ("multistab", part_ms), ("iso", part_iso)):
        if os.path.exists(f"{OUT}/{fname}.json"):
            fn(md)
    open(f"{OUT}/tables.md", "w").write("\n".join(md) + "\n")
    json.dump(KN, open(f"{OUT}/key_numbers.json", "w"), indent=1, default=float)
    print("wrote tables.md, key_numbers.json")


def part_lr(md):
    J = json.load(open(f"{OUT}/lr.json")); C = J["cells"]
    md.append("## Part B: left/right symmetry (`lr.json`)\n")
    md.append("Criteria (plan.md 4): broken = c1 AND c2 AND c3; c4 = switching (|A| > 0.6 in >= 25% of late records for >= 25% of all-start "
              "survivors AND median corr_t(f_L, f_R) < -0.2).\n")
    md.append("| setting | model | r | pair | points | testable | broken | c1 | c2 | c3 | c4 | t_sym median range (one-sided start) | "
              "late mean A range (one-sided starts) | largest single-replica |late A| | corr_LR (all start) range |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    KN["LR"] = {}
    for key, v in C.items():
        st, model, r = key.split("|")
        for pair, (sL, sR) in (("AL", ("AL_L", "AL_R")), ("HEM", ("hemiL", "hemiR"))):
            rows = [rw[pair] for rw in v["rows"]]
            sm = J["summary"][f"{key}|{pair}"]
            ts = [x["starts"][s_]["t_sym"]["median"] for x in rows for s_ in (sL, sR) if s_ in x["starts"] and x["starts"][s_].get("t_sym")
                  and x["starts"][s_]["t_sym"]["median"] is not None]
            la = [x["starts"][s_]["late_A_mean"] for x in rows for s_ in (sL, sR) if s_ in x["starts"] and x["starts"][s_]["late_A_mean"] is not None]
            amax = [x["starts"][s_]["late_A_absmax"] for x in rows for s_ in x["starts"] if x["starts"][s_].get("late_A_absmax") is not None]
            cr = [x["starts"]["all"]["corr_LR_median"] for x in rows if "all" in x["starts"] and x["starts"]["all"]["corr_LR_median"] is not None]
            if not ts and not la:
                continue
            md.append(f"| {st} | {model} | {r} | {pair} | {sm['n_points']} | {sm['n_testable']} | {sm['n_broken']} | {sm['n_c1']} | {sm['n_c2']} | "
                      f"{sm['n_c3']} | {sm['n_c4']} | {f(min(ts)) if ts else '–'} – {f(max(ts)) if ts else '–'} | "
                      f"{f(min(la)) if la else '–'} – {f(max(la)) if la else '–'} | {f(max(amax)) if amax else '–'} | "
                      f"{f(min(cr)) if cr else '–'} – {f(max(cr)) if cr else '–'} |")
            KN["LR"][f"{key}|{pair}"] = dict(sm, t_sym_median_range=[min(ts), max(ts)] if ts else None,
                                             late_A_range=[min(la), max(la)] if la else None, max_abs_replica_A=max(amax) if amax else None,
                                             corr_range=[min(cr), max(cr)] if cr else None)
    tot = {k: sum(v[k] for v in J["summary"].values()) for k in ("n_points", "n_testable", "n_broken", "n_c1", "n_c2", "n_c3", "n_c4")}
    KN["LR_total"] = tot
    md.append(f"\nTotals over every (setting, model, r, g, pair): {tot}.\n")
    # smallest KS p and survivors detail
    md.append("Per point: `lr.json` cells[<setting>|<model>|<r>].rows[i][AL|HEM] (starts, ks_L_vs_R, bimodality_all, criteria).\n")


def part_ms(md):
    J = json.load(open(f"{OUT}/multistab.json")); C = J["cells"]
    md.append("## Part C / D(i): multistability across starts (`multistab.json`)\n")
    md.append("| setting | model | r | cells | multistable P0 / P1 / P2 | largest rel L1 between start means (P0) | module-alone states (start, g) | quiet-vs-all candidates (g) |")
    md.append("|---|---|---|---|---|---|---|---|")
    KN["MS"] = {}
    for key, sm in J["summary"].items():
        st, model, r = key.split("|")
        md.append(f"| {st} | {model} | {r} | {sm['n_cells']} | {sm['n_multistable_P0']} / {sm['n_multistable_P1']} / {sm['n_multistable_P2']} | "
                  f"{f(sm['max_rel_l1_P0'])} | {sm['module_alone'] or 'none'} | {sm['quiet_all_candidates'] or 'none'} |")
        KN["MS"][key] = sm
    md.append("")
    # per-start survival / confinement table, setting a, M2
    md.append("### Starts in setting (a): fraction alive at T, median confinement C (late share of activity inside the start set), "
              "median t_out (first time more activity outside the set than inside)\n")
    for key in sorted(C):
        st, model, r = key.split("|")
        if st != "a":
            continue
        v = C[key]
        md.append(f"**{model}, r = {r}** (g = {', '.join(f(rw['g']) for rw in v['rows'])})\n")
        md.append("| start | alive at T | C median | t_out median |")
        md.append("|---|---|---|---|")
        for s in v["starts"]:
            al = [rw["starts"].get(s, {}).get("alive_T") for rw in v["rows"]]
            cm = [rw["starts"].get(s, {}).get("confinement_median") for rw in v["rows"]]
            to = [rw["starts"].get(s, {}).get("t_out_median") for rw in v["rows"]]
            md.append(f"| {s} | {' / '.join(f(x, 2) for x in al)} | {' / '.join(f(x, 2) for x in cm)} | {' / '.join(f(x, 2) for x in to)} |")
        md.append("")
    md.append("### Setting (b): quiet vs fully active start (late nonORN activity, window 2000-3000)\n")
    md.append("| model | r | g | median quiet | median all | ratio | KS p | gap 1500-2000 | gap 2500-3000 | candidate |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(C):
        st, model, r = key.split("|")
        if st != "b":
            continue
        for rw in C[key]["rows"]:
            q = rw.get("quiet_vs_all")
            if q:
                md.append(f"| {model} | {r} | {f(rw['g'])} | {f(q['median_quiet'])} | {f(q['median_all'])} | {f(q['ratio'])} | {f(q['ks_p'], 2)} | "
                          f"{f(q['gap_1500_2000'], 2)} | {f(q['gap_2500_3000'], 2)} | {q['candidate']} |")
    md.append("")


def part_iso(md):
    J = json.load(open(f"{OUT}/iso.json"))
    md.append("## Part D(ii): modules alone (`iso.json`; isolated subgraph, h = 0, fully active start, B 64, T 3000, G_COARSE)\n")
    md.append("| module | N | model | r | S (tau 200) | onset (tau 3000) [bracket] | full-brain onset | onset / full-brain |")
    md.append("|---|---|---|---|---|---|---|---|")
    KN["ISO"] = {}
    order = ["AL", "AL_L", "AL_R", "isl1", "isl3", "isl5", "CB", "VNC", "OL"]
    for mod in order:
        for model in ("M1", "M2"):
            for r in (0, 1, 4):
                k = f"{model}_r{r}_{mod}"
                if k not in J:
                    continue
                v = J[k]
                on = v["on"]["g"]; br = v["on"]["bracket"]
                md.append(f"| {mod} | {v['N']} | {model} | {r} | {f(v['S']['g'])} | {f(on) if on else ('none <= 30' if v['on']['status'] == 'none' else v['on']['status'])} "
                          f"[{f(br[0])}, {f(br[1])}] | {f(v['full_brain_on'])} | {f(v['on_over_full'])} |")
                KN["ISO"][k] = {"N": v["N"], "S": v["S"]["g"], "on": on, "on_status": v["on"]["status"], "bracket": br,
                                "full_brain_on": v["full_brain_on"], "ratio": v["on_over_full"]}
    md.append("")


if __name__ == "__main__":
    main()
