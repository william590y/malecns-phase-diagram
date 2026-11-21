#!/usr/bin/env python
"""pd3 completeness critic (2026-11-09): label fixes for figs/pd3/phase_diagram_v3.png and the two lifetime heat maps.

The original scripts (code/pd3_synthesis.py, code/pd3_redraw.py) are NOT modified. This script reads their source, applies the
exact string replacements listed in FIXES (each must match exactly once), executes the patched source as a non-__main__
module, and calls only the figure functions:
  - pd3_synthesis.figure_phase_diagram()  -> figs/pd3/phase_diagram_v3.png
  - pd3_redraw.main()                      -> figs/pd3/phase_lifetime_heatmap.png, phase_lifetime_heatmap_nulls.png
    (its npz / meta outputs are redirected to results/pd3/critic_regen/ and compared with results/pd3/redraw/, which is
    left untouched).
Pre-critic PNGs are copied to figs/pd3/verify_orig/<name>_precritic.png first (only if no such copy exists yet).
Data, lines and colours are unchanged; only texts and the top margin of phase_diagram_v3 change.
"""
import json
import os
import shutil

import numpy as np

ROOT = "/MALECNS_ROOT"
CODE = f"{ROOT}/code"
FIG = f"{ROOT}/figs/pd3"
BK = f"{FIG}/verify_orig"
REGEN = f"{ROOT}/results/pd3/critic_regen"
os.makedirs(BK, exist_ok=True)
os.makedirs(REGEN, exist_ok=True)

# AL share of activity in B_AL (onset .. island-1 1% line), from code/pd3_critic_alshare.py
alj = json.load(open(f"{ROOT}/results/pd3/critic_alshare.json"))["results"]
sh = [row["AL_share"] for r, v in alj.items() for row in v["rows"] if v["g_on"] * 0.999 <= row["g"] <= v["g_isl1_1pct"]]
SHLO, SHHI = min(sh), max(sh)
print(f"AL share of all activity inside B_AL (r = 0, 1, 6): {SHLO:.3f}-{SHHI:.3f} over {len(sh)} pooled points")

FIXES = {
    "pd3_synthesis.py": [
        ('top=0.94, bottom=0.03)', 'top=0.915, bottom=0.03)'),
        ('fig.suptitle("MaleCNS stochastic Wilson-Cowan: de Candia-style (g, r) phase diagram.  Solid = continuous "\n'
         '                 "(second-order, finite-size rounded) line; dashed = first-order line; dotted = crossover",\n'
         '                 fontsize=12, y=0.985)',
         'fig.suptitle("MaleCNS stochastic Wilson-Cowan (Model 1): de Candia-style (g, r) diagram.\\n"\n'
         '                 "Solid = g_lin, the linear instability of silence (continuous in mean field; rounded by the "\n'
         '                 "finite AL module in simulation, not a sharp transition); dashed = first-order line (none found); "\n'
         '                 "dotted = crossover", fontsize=11.5, y=0.993)'),
        ('label="g_lin = 0.1632: SL edge, the only critical line"',
         'label="g_lin = 0.1632: linear instability of silence (theory; r- and T-independent)"'),
        ('Patch(fc=C_B, label="B_AL (+ islands): localized, active")',
         f'Patch(fc=C_B, label="B_AL (+ islands): PR < N/4; AL = {100 * SHLO:.0f}-{100 * SHHI:.0f}% of activity")'),
        ('label="M1 test inside a closure B1 window"',
         'label="M1 test inside the 2-module closure\'s B1 window"'),
        # do not draw the median-island / rest 1% lines through rows where they were not measured (simulated to g = 2.6 only)
        ('ax.plot(ISLMED[m], R[m], ls=(0, (1, 2)), lw=1.4, color=OK["green"], marker=".", ms=4, zorder=4)',
         'ax.plot(ISLMED, R, ls=(0, (1, 2)), lw=1.4, color=OK["green"], marker=".", ms=6, zorder=4)'),
        ('ax.plot(REST[m], R[m], ls=(0, (1, 2)), lw=1.4, color="0.35", marker=".", ms=4, zorder=4)',
         'ax.plot(REST, R, ls=(0, (1, 2)), lw=1.4, color="0.35", marker=".", ms=6, zorder=4)'),
        ('"hatched rows (r = 2.25-5.5):\\nsimulated only to g = 2.6"',
         '"hatched rows (r = 2.25-5.5):\\nsimulated only to g = 2.6;\\n1% lines broken where not measured"'),
    ],
    "pd3_redraw.py": [
        ('OUT = f"{RES}/pd3/redraw"', f'OUT = "{REGEN}"'),
        ('mean-field critical line\\n(silence loses stability; r-independent)',
         'linear instability of silence\\n(exact linearisation; r-independent)'),
    ],
}


def patched(fname):
    src = open(f"{CODE}/{fname}").read()
    for old, new in FIXES[fname]:
        n = src.count(old)
        assert n == 1, f"{fname}: replacement target found {n} times: {old[:80]!r}"
        src = src.replace(old, new)
    return src


def backup(name):
    dst = f"{BK}/{name[:-4]}_precritic.png"
    if not os.path.exists(dst):
        shutil.copy2(f"{FIG}/{name}", dst)


for n in ("phase_diagram_v3.png", "phase_lifetime_heatmap.png", "phase_lifetime_heatmap_nulls.png"):
    backup(n)

g1 = {"__name__": "pd3_synthesis_critic", "__file__": f"{CODE}/pd3_synthesis.py"}
exec(compile(patched("pd3_synthesis.py"), f"{CODE}/pd3_synthesis.py(patched)", "exec"), g1)
g1["figure_phase_diagram"]()

g2 = {"__name__": "pd3_redraw_critic", "__file__": f"{CODE}/pd3_redraw.py"}
exec(compile(patched("pd3_redraw.py"), f"{CODE}/pd3_redraw.py(patched)", "exec"), g2)
g2["main"]()

# the redirected heat-map data must equal the verified originals
same = {}
for kind in ("real", "null", "wnull"):
    a = np.load(f"{ROOT}/results/pd3/redraw/heatmap_{kind}.npz"); b = np.load(f"{REGEN}/heatmap_{kind}.npz")
    same[kind] = all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
print("regenerated heat-map arrays identical to results/pd3/redraw:", same)
json.dump({"fixes": {k: [[o, n] for o, n in v] for k, v in FIXES.items()}, "AL_share_in_B_AL": [SHLO, SHHI],
           "heatmap_arrays_identical": same}, open(f"{REGEN}/critic_figs_log.json", "w"), indent=1)
