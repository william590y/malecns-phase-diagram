"""m2_driven_report.py -- tables + quality checks of lane m2/driven (reads curves.json, summary.json, logs; 2026-11-09).
Writes results/m2/driven/table.md and checks.json. Every number in REPORT.md comes from these files.
"""
import glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_common as M

ROOT = M.ROOT
RES = f"{ROOT}/results/m2/driven"
C = json.load(open(f"{RES}/curves.json"))
S = json.load(open(f"{RES}/summary.json"))


def f(x, d=3):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if not np.isfinite(x):
        return "inf"
    return f"{x:.{d}g}"


def ci(c, k):
    v = c["lines_ci"].get(k)
    if not v or v["lo"] is None:
        return ""
    return f" [{f(v['lo'])}, {f(v['hi'])}]"


def sel(set_, model, graph, orn, palp=False):
    out = {}
    for k, c in C.items():
        kk = c["key"]
        if kk["set"] == set_ and kk["model"] == model and kk["graph"] == graph and abs(kk["orn"] - orn) < 1e-9 \
                and kk["palp"] == palp:
            out[kk["r"]] = c
    return dict(sorted(out.items()))


def peak_s(p):
    if p["g"] is None:
        return "–"
    s = f"{f(p['g'])}" + (" (resolved" if p["resolved"] else " (not resolved") + f", {p['frac_resolved']:.2f}"
    s += f", {p['phase_at_peak']})"
    return s


def at_line(c, g0, field):
    """value of a curve field at the grid point nearest to g0 (log distance)."""
    if g0 is None:
        return None
    g = np.array(c["g"]); k = int(np.argmin(np.abs(np.log(g / g0))))
    return c["curve"][field][k], float(g[k])


def main_table(cs, title):
    L = [f"### {title}", "",
         "| r | g range (n) | DQ->DA (G = 0.29) | allowed region [lo, hi] (hi set by) | silent lost (by) | tau_post = 200 | "
         "DT->SS (tau_post = 3000) | SS-L->SS-D | F_AL peak | F_non peak | tau_int peak | rho_ORN at allowed top | "
         "OSC / NS points |",
         "|" + "---|" * 13]
    for r, c in cs.items():
        Ln = c["lines"]; cv = c["curve"]
        top = at_line(c, Ln.get("allowed_hi"), "rho_ORN")
        nosc = sum(bool(x) for x in cv["osc"]); nns = sum(bool(x) for x in cv["NS"])
        L.append(f"| {r:g} | {f(min(c['g']))}–{f(max(c['g']))} ({c['n_g']}) | {f(Ln.get('DQ_DA'))}{ci(c, 'DQ_DA')} | "
                 f"{f(Ln.get('allowed_lo'))}–{f(Ln.get('allowed_hi'))}{ci(c, 'allowed_hi')} ({Ln.get('allowed_hi_by') or '–'}"
                 f"{'; runs ' + str(Ln.get('allowed_runs')) if (Ln.get('allowed_runs') or 0) > 1 else ''}) | "
                 f"{f(Ln.get('silent_lost'))} ({Ln.get('silent_lost_by') or '–'}) | {f(Ln.get('tau200'))} | "
                 f"{f(Ln.get('DT_SS'))}{ci(c, 'DT_SS')} | {f(Ln.get('SS_D'))} | {peak_s(c['peaks']['F_AL'])} | "
                 f"{peak_s(c['peaks']['F_non'])} | {peak_s(c['peaks']['tau_int'])} | "
                 f"{f(top[0]) if top else '–'} | {nosc} / {nns} |")
    return L + [""]


def checks():
    out = {}
    jobs = json.load(open(f"{RES}/jobs.json"))
    md5s = {}
    jids = set()
    for k, v in jobs.items():
        if isinstance(v, str) and v.isdigit():
            jids.add(v)
        elif isinstance(v, list):
            jids.update(str(x["job"]) for x in v if isinstance(x, dict) and "job" in x)
    out["job_ids"] = sorted(jids)
    for jid in jids:
        for lf in glob.glob(f"{ROOT}/logs/m2-driven-*_{jid}_*.out"):
            l0 = open(lf).readline()
            m = re.search(r"md5 ([0-9a-f]{8})", l0)
            md5s[m.group(1) if m else "?"] = md5s.get(m.group(1) if m else "?", 0) + 1
    out["sim_md5_in_logs"] = md5s
    idx = []
    for fi in sorted(glob.glob(f"{RES}/index_*.json")):
        idx += json.load(open(fi))
    out["tasks"] = len(idx)
    out["tasks_missing_output"] = [m["name"] for m in idx if not os.path.exists(m["out"])]
    mx, bad = 0.0, []
    lowg = []
    for k, c in C.items():
        cv = c["curve"]
        if not c["key"]["palp"]:
            v = [x for x in cv["ORN_after_removal_max"] if x is not None]
            if v and max(v) > 0:
                bad.append(k)
            mx = max([mx] + v)
        if c["g"][0] < 0.051 and c["key"]["graph"] == "real":
            lowg.append({"curve": k, "rho_ORN_g0.05": cv["rho_ORN"][0], "level": c["key"]["orn"],
                         "rel_err": cv["rho_ORN"][0] / c["key"]["orn"] - 1})
    out["ORN_after_removal_max_over_all"] = mx
    out["curves_with_ORN_activity_after_removal"] = bad
    out["rho_ORN_at_g0.05_rel_err_max"] = max(abs(x["rel_err"]) for x in lowg) if lowg else None
    out["rho_ORN_at_g0.05_rel_err_worst"] = sorted(lowg, key=lambda x: -abs(x["rel_err"]))[:3]
    # OSC / NS census
    n = nosc = nstr = nns = nex = 0
    osc_pts = []
    for k, c in C.items():
        cv = c["curve"]
        for i, g in enumerate(c["g"]):
            n += 1
            if cv["osc"][i]:
                nosc += 1; osc_pts.append([k, g, cv["phase"][i], cv["period"][i], cv["cv_t"][i]])
            nstr += bool(cv["osc_strong"][i]); nns += bool(cv["NS"][i]); nex += bool(cv["NS_exempt"][i])
    out.update(n_points=n, n_osc=nosc, n_osc_strong=nstr, n_NS=nns, n_NS_exempt=nex, osc_points=osc_pts[:50])
    json.dump(out, open(f"{RES}/checks.json", "w"), indent=1)
    return out


def main():
    L = ["# Lane m2/driven: generated tables (code/m2_driven_report.py; numbers from curves.json / summary.json)", "",
         "Lines are interpolated in ln g; [..] = 95 % replica bootstrap (2000 resamples). Peaks: grid argmax refined by a "
         "parabola in ln g; 'resolved' = interior and above both ends in >= 95 % of resamples (fraction shown).", ""]
    L += main_table(sel("A1", "M2", "real", 0.04), "A1 M2, real connectome, ORN 0.04 (main diagram)")
    L += main_table(sel("A2", "M1", "real", 0.04), "A2 M1 (old model), real, ORN 0.04")
    L += main_table(sel("A4", "M2s", "real", 0.04), "A4 M2s (per-target lLN1_bc signs), real, ORN 0.04")
    L += main_table(sel("A3", "M2", "cnull", 0.04), "A3 M2, configuration null (G, allowed: not meaningful)")
    L += main_table(sel("A3", "M2", "wnull", 0.04), "A3 M2, weight-shuffled null")
    for o in (0.01, 0.02, 0.07, 0.10):
        L += main_table(sel("B1", "M2", "real", o), f"B1 M2, real, ORN level {o:g}")
    for m in ("M2_allsyn", "M2_KW09", "M2_N15", "M2_tau30", "M1s"):
        L += main_table(sel("B2" if m != "M1s" else "B3", m, "real", 0.04), f"{m}, real, ORN 0.04")
    L += main_table(sel("B2", "M2_tau30", "real", 0.10), "M2_tau30, real, ORN 0.10")
    # MODEL2 5c: (a) vs (b) per r
    dec = S.get("decay_lane", {}).get("rows", {})
    L += [f"### MODEL2 5c: setting (a) (lane m2/decay, {S.get('decay_lane', {}).get('file')} md5 "
          f"{S.get('decay_lane', {}).get('md5')}, read {S.get('decay_lane', {}).get('mtime')}) vs setting (b) (this lane)", "",
          "| model, graph | r | g_lin | (a) S | (a) onset | (a) D | (b) DQ->DA | (b) allowed top | (b) silent lost | (b) DT->SS | "
          "DT->SS / (a) onset | allowed top / (a) S | F_AL max g (resolved) |", "|" + "---|" * 13]
    for (st, m, gr) in (("A1", "M2", "real"), ("A2", "M1", "real"), ("A4", "M2s", "real"), ("A3", "M2", "cnull"),
                        ("A3", "M2", "wnull")):
        for r, c in sel(st, m, gr, 0.04).items():
            d = dec.get(f"{m}_{gr}_r{r:g}", {})
            Ln = c["lines"]
            rat1 = (Ln.get("DT_SS") / d["on"]) if (Ln.get("DT_SS") and d.get("on")) else None
            rat2 = (Ln.get("allowed_hi") / d["S"]) if (Ln.get("allowed_hi") and d.get("S")) else None
            p = c["peaks"]["F_AL"]
            L.append(f"| {m} {gr} | {r:g} | {M.G_LIN:.3f} | {f(d.get('S'))} | {f(d.get('on'))} | {f(d.get('D'))} | "
                     f"{f(Ln.get('DQ_DA'))} | {f(Ln.get('allowed_hi'))} | {f(Ln.get('silent_lost'))} | {f(Ln.get('DT_SS'))} | "
                     f"{f(rat1)} | {f(rat2)} | {f(p['g'])} ({'yes' if p['resolved'] else 'no'}) |")
    L.append("")
    # palp
    L += ["### B4 palp-intact removal (S1' = uPN_ant 100-300 after removal / driven; S2' = 1000-3000; pass <= 0.1)", "",
          "| model | r | g where both pass | first g failing (after a pass) | S1' / S2' at that g |", "|---|---|---|---|---|"]
    for m in ("M2", "M2s"):
        for r, c in sel("B4", m, "real", 0.04, palp=True).items():
            ok = c["curve"]["palp_ok"]; g = c["g"]
            okg = [x for x, o in zip(g, ok) if o]
            fail = None
            if okg:
                k0 = ok.index(True)
                for k in range(k0, len(g)):
                    if not ok[k]:
                        fail = k; break
            L.append(f"| {m} | {r:g} | {f(min(okg)) if okg else '–'}–{f(max(okg)) if okg else '–'} | "
                     f"{f(g[fail]) if fail is not None else '–'} | "
                     f"{(f(c['curve']['S1p_resid'][fail]) + ' / ' + f(c['curve']['S2p_resid'][fail])) if fail is not None else '–'} |")
    L.append("")
    # field scaling
    fs = S.get("field_scaling", {})
    L += ["### Field scaling (M2 real, ORN levels 0.01-0.10)", "",
          "| r | s = 0.5 crossing [CI] | s_eff = 0.5 crossing [CI] (amendment 4) | chi_h peak g (resolved, frac) | "
          "s at g 0.05 / 0.3 / 1 / 3 | s_eff at g 0.05 / 0.3 / 1 / 3 |", "|---|---|---|---|---|---|"]
    for r, d in fs.items():
        g = np.array(d["g"]); s_ = np.array(d["s"]); se = np.array(d["s_eff"])
        pick = [s_[int(np.argmin(np.abs(np.log(g / x))))] for x in (0.05, 0.3, 1.0, 3.0)]
        picke = [se[int(np.argmin(np.abs(np.log(g / x))))] for x in (0.05, 0.3, 1.0, 3.0)]
        p = d["chi_h_peak"]
        def cis(k):
            return ('[' + f(d[k][0]) + ', ' + f(d[k][1]) + ']') if d.get(k) else ''
        L.append(f"| {float(r):g} | {f(d['s_half'])} {cis('s_half_ci')} | {f(d['s_eff_half'])} {cis('s_eff_half_ci')} | "
                 f"{f(p['g'])} ({'yes' if p['resolved'] else 'no'}, {p['frac_resolved']:.2f}) | "
                 f"{' / '.join(f(x) for x in pick)} | {' / '.join(f(x) for x in picke)} |")
    L.append("")
    # tracking
    L += ["### Do the peak lines track an instability line? (plan.md section 4)", "",
          "Entry = verdict (fraction of r within one grid step, n r compared; Spearman if the line varies).", "",
          "| set | observable | resolved r / all | vs g_lin | vs DT->SS | vs decay onset (a) | vs decay S (a) |",
          "|---|---|---|---|---|---|---|"]
    def v(x):
        if x is None:
            return "–"
        fr = x.get("frac_within_one_step")
        sp = x.get("spearman")
        return (f"{x['verdict']} ({'–' if fr is None else round(fr, 2)}, n {x['n']}"
                + (f", rho {sp:.2f}" if sp is not None else "") + ")")
    for sk, d in S.get("tracking", {}).items():
        for ob, t in d.items():
            nres = t.get("n_resolved", "")
            L.append(f"| {sk} | {ob} | {nres}{'/' + str(t.get('n_r')) if 'n_r' in t else ''} | {v(t.get('vs_g_lin'))} | "
                     f"{v(t.get('vs_DT_SS'))} | {v(t.get('vs_decay_on'))} | {v(t.get('vs_decay_S'))} |")
    L.append("")
    ck = checks()
    L += ["### Quality checks (checks.json)", "", "```", json.dumps({k: v for k, v in ck.items() if k != "osc_points"}, indent=1)[:3000], "```"]
    open(f"{RES}/table.md", "w").write("\n".join(L) + "\n")
    print("wrote table.md, checks.json")


if __name__ == "__main__":
    main()
