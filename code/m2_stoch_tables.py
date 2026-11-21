"""m2_stoch_tables.py -- print markdown tables from the prediction files of lane m2/theory-stochastic (for stochastic.md)."""
import json, os, sys
import numpy as np

T = "/MALECNS_ROOT/results/m2/theory"


def J(p):
    return json.load(open(f"{T}/{p}"))


def f(x, n=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{n}g}"


on = J("stochastic_onset.json")
print("### onset table (setting a)\n")
print("| model | r | single 2D S / on | single F on | single A on | pooled 2D S / on | PR at single on | AL share at single on |")
print("|---|---|---|---|---|---|---|---|")
for m in ("M2", "M2_KW09", "M2_N15", "M2_tau30", "M1"):
    if m not in on["models"]:
        continue
    b = on["models"][m]["boundaries"]
    for r in ("0", "0.25", "0.5", "1", "2", "4", "6"):
        k = "PF_r0" if r == "0" else f"CAL_r{r}"
        s2 = b.get(f"single_{k}_lt_2D", {}); sF = b.get(f"single_{k}_lt_F", {}); sA = b.get(f"single_{k}_lt_A", {})
        p2 = b.get(f"pooled_{k}_lt_2D", {})
        a = s2.get("at_onset", {}) or {}
        print(f"| {m} | {r}{' (PF)' if r == '0' else ' (CAL)'} | {f(s2.get('S'))} / {f(s2.get('on'))} | {f(sF.get('on'))} | "
              f"{f(sA.get('on'))} | {f(p2.get('S'))} / {f(p2.get('on'))} | {f(a.get('PR'))} | {f(a.get('share_AL_module'), 2)} |")
isl = on.get("islands_M2")
if isl:
    print("\nislands crossings:", json.dumps(isl["cross"]))
print("\n### transient (M2)\n")
for t in on["models"]["M2"]["transient"]:
    print(f"g {t['g']:.3f}: n_min/n* {t['n_min_over_nstar']:.3f} at t {t['t_min']:.1f}")
if os.path.exists(f"{T}/stochastic_qs.json"):
    qs = J("stochastic_qs.json")
    print("\n### QS LNA\n")
    print("| model | g | n_tot | PR | AL share | CV | Fano(count) | ln tau (coherent LNA) | period | coherence | OSC | OSC AL | tau_int |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m, d in qs.items():
        if m == "notes":
            continue
        for r in d["rows"]:
            print(f"| {m} | {r['g']:g} | {f(r['rho_tot'])} | {f(r['PR'])} | {f(r['share_AL'], 2)} | {f(r['cv_all'])} | {f(r['fano_count'])} | "
                  f"{f(r['lt_LNA_gauss'])} | {f(r['period'])} | {f(r['coh'], 2)} | {r['osc']} | {r['osc_AL']} | {f(r['tau_int'])} |")
if os.path.exists(f"{T}/stochastic_driven.json"):
    dv = J("stochastic_driven.json")
    print("\n### driven summaries\n")
    for k, d in dv.items():
        if k == "notes":
            continue
        print(k, json.dumps(d["summary"]))
if os.path.exists(f"{T}/stochastic_aval.json"):
    av = J("stochastic_aval.json")
    print("\n### pulse\n")
    for k, d in av.items():
        if k == "notes":
            continue
        for gk, gd in d["by_gap"].items():
            print(f"{k} {gk}: critical-like g {gd['critical_like_g']} span {gd['critical_like_span']}")
            for r in gd["rows"]:
                print(f"   g {r['g']:.3f} m {r['m']:.6f} eps {r['eps']:.2e} S_eps {f(r['S_eps'])} S_dep {f(r['S_dep'])} K_s {f(r['K_s'])} "
                      f"<S> {f(r['mean_size'])} AL_F {f(r['share_AL_F'], 2)} xAL {f(r['x_AL'], 2)} {r.get('method')}")
if os.path.exists(f"{T}/stochastic_fss.json"):
    fs = J("stochastic_fss.json")
    print("\n### FSS / modules\n")
    print("| graph | N | single PF r0 S / on / W | single CAL r1 on | single CAL r4 on | pooled PF r0 S / on / W |")
    print("|---|---|---|---|---|---|")
    for k, d in fs["graphs"].items():
        b = d["boundaries"]
        s = b.get("single_PF_r0_lt_2D", {}); p = b.get("pooled_PF_r0_lt_2D", {})
        print(f"| {k} | {d['N']} | {f(s.get('S'))} / {f(s.get('on'))} / {f(s.get('W'))} | {f(b.get('single_CAL_r1_lt_2D', {}).get('on'))} | "
              f"{f(b.get('single_CAL_r4_lt_2D', {}).get('on'))} | {f(p.get('S'))} / {f(p.get('on'))} / {f(p.get('W'))} |")
