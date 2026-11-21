"""AL-defect synthesis figure: phase bars (h = 0, fully active start) for the published model and its fixes, with the
coupling range that the antennal-lobe deafferentation data allow (ORN-driven protocol of E1) drawn above each row.

Input: results/aldef/synth/synth_numbers.json (code/ald_synth_calc.py). Output: figs/aldef/synth_phase.png.
Phase colours: one-hue ordinal ramp (validated with the dataviz validator, --ordinal): slow window / sustained, PR < N/4 /
sustained, PR >= N/4; quiescent = neutral grey; not simulated = hatched.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = '/MALECNS_ROOT'
d = json.load(open(f'{ROOT}/results/aldef/synth/synth_numbers.json'))
E1, E2, E3, RM, GL = d['E1_allowed'], d['E2_signs'], d['E3_depression'], d['E3_removal'], d['g_lin']

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C_Q, C_S, C_L, C_D = '#e6e5e1', '#86b6ef', '#2a78d6', '#104281'
XMIN, XMAX, GRID_END = 0.08, 22.0, 8.0


def pct(x):
    return f'{100 * x:.0f}%' if x >= 0.01 else f'{100 * x:.1f}%'


def sign_row(key, label, glin):
    v = E2[key]
    return dict(label=label, S=v['S'], on=v['on'], D=v['D'], end=None, glin=glin,
                site=f"first sustained: AL module {pct(v['ignition_module_share'])} of activity, "
                     f"~{v['ignition_PR_neurons'] / 1000:.1f}k neurons")


def dep_row(key, label, allowed=None, allowed_solid=False):
    v = E3[key]
    fs = v['first_sustained']
    return dict(label=label, S=v['S'], on=v['on'], D=v.get('D_interp'), end=None if v.get('D_interp') else GRID_END,
                glin=GL['base'], allowed=allowed, allowed_solid=allowed_solid,
                site=f"first sustained: AL module {pct(fs['AL_share'])} of activity, "
                     f"~{fs['PR_over_N'] * 167216 / 1000:.1f}k neurons")


rows = {0: [], 1: []}
for r in (0, 1):
    b = sign_row(f'base_r{r}', 'Model as published', GL['base'])
    b['allowed'] = (E1[str(r)]['lower'][0], E1[str(r)]['upper_with_palp'][1], E1[str(r)]['upper'][1])
    rows[r].append(b)
rows[0] += [sign_row('lLN1bc_zero_r0', 'lLN1_bc made non-chemical (sign 0)', GL['lLN1bc_off']),
            sign_row('lLN1bc_lLN2_inh_r0', 'lLN1_bc + 40 ACh lLN2 made inhibitory', GL['lLN1bc_lLN2ach_off']),
            dep_row('sp_NW16_r0', 'Depression, all synapses (NW16)')]
rows[1] += [sign_row('lLN1bc_inh_r1', 'lLN1_bc made inhibitory', GL['lLN1bc_off']),
            sign_row('lLN1bc_zero_r1', 'lLN1_bc made non-chemical (sign 0)', GL['lLN1bc_off']),
            sign_row('lLN1bc_lLN2_inh_r1', 'lLN1_bc + 40 ACh lLN2 made inhibitory', GL['lLN1bc_lLN2ach_off']),
            dep_row('spg_lLN1bc_r1', 'Depression on lLN1_bc only (NW16)'),
            dep_row('spg_exc_r1', 'Depression, excitatory outputs (NW16)', RM['rm_spg_exc_r1']['allowed']),
            dep_row('sp_NW16_r1', 'Depression, all synapses (NW16)', RM['rm_sp_NW16_r1']['allowed']),
            dep_row('sp_KW09_r1', 'Depression, all synapses (KW09)', RM['rm_sp_KW09_r1']['allowed'])]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2})
heights = [len(rows[0]), len(rows[1])]
fig, axes = plt.subplots(2, 1, figsize=(12.5, 0.72 * sum(heights) + 2.7), sharex=True,
                         gridspec_kw={'height_ratios': heights, 'hspace': 0.18})
BH = 0.36  # phase-bar height
for r, ax in zip((0, 1), axes):
    R = rows[r]
    for i, row in enumerate(R):
        y = len(R) - 1 - i
        segs = [(XMIN, row['S'], C_Q), (row['S'], row['on'], C_S)]
        top = row['D'] if row['D'] else row['end']
        segs.append((row['on'], top, C_L))
        if row['D']:
            segs.append((row['D'], XMAX, C_D))
        for a, bnd, c in segs:
            ax.barh(y, bnd - a, left=a, height=BH, color=c, edgecolor='white', linewidth=1.5)
        if not row['D']:
            ax.barh(y, XMAX - row['end'], left=row['end'], height=BH, color='white', edgecolor='#a8a7a2',
                    hatch='////', linewidth=0.6)
        ax.plot([row['glin']] * 2, [y - BH / 2 - 0.05, y + BH / 2 + 0.05], color=INK, lw=2.2,
                solid_capstyle='butt')
        al = row.get('allowed')
        if al:
            ya = y + 0.36
            if len(al) == 3:  # published model: solid = passes palp test too; dashed = gain + silence only
                ax.plot([al[0], al[1]], [ya, ya], color=INK, lw=3.2, solid_capstyle='butt')
                ax.plot([al[1], al[2]], [ya, ya], color=INK, lw=1.6, ls=(0, (2, 1.5)))
            else:
                ax.plot([al[0], al[1]], [ya, ya], color=INK, lw=1.6, ls=(0, (2, 1.5)))
        ax.text(XMAX * 1.04, y - 0.08, row['site'].replace('first sustained: ', ''), va='center', ha='left',
                fontsize=9, color=INK2)
        ax.text(row['on'] * 1.03 if row['D'] is None or row['D'] / row['on'] > 1.6 else row['on'], y,
                f"{row['on']:.2f}", va='center', ha='left', fontsize=8.2, color='white', fontweight='bold')
    ax.set_yticks(range(len(R)))
    ax.set_yticklabels([row['label'] for row in reversed(R)], fontsize=9.6, color=INK)
    ax.set_ylim(-0.55, len(R) - 0.25)
    ax.set_xscale('log')
    ax.set_xlim(XMIN, XMAX)
    ax.grid(axis='x', color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_title(f'r = {r} (inhibition strength)', loc='left', fontsize=11, color=INK, fontweight='bold')
    ax.text(XMAX * 1.04, len(R) - 0.35, 'where activity first sustains itself', fontsize=8.6, color=INK, ha='left',
            fontweight='bold')
ticks = [0.1, 0.163, 0.25, 0.5, 1, 2, 4, 8, 16]
axes[-1].set_xticks(ticks)
axes[-1].set_xticklabels(['0.1', 'g_lin\n0.163', '0.25', '0.5', '1', '2', '4', '8', '16'])
axes[-1].minorticks_off()
axes[0].minorticks_off()
axes[-1].set_xlabel('coupling g (raw units, log scale)')
leg = [Patch(color=C_Q, label='quiescent at h = 0 (dies fast, lifetime < 200)'),
       Patch(color=C_S, label='slow window (lifetime 200-3000)'),
       Patch(color=C_L, label='self-sustained, participation < N/4'),
       Patch(color=C_D, label='self-sustained, participation >= N/4 (delocalized)'),
       Patch(facecolor='white', edgecolor='#a8a7a2', hatch='////', label='not simulated (grid ends at g = 8)'),
       Line2D([], [], color=INK, lw=0, marker='|', markersize=12, markeredgewidth=2.2, label='g_lin of that network (linear instability of silence)'),
       Line2D([], [], color=INK, lw=3.2, label='allowed by deafferentation data, incl. palps-intact test'),
       Line2D([], [], color=INK, lw=1.6, ls=(0, (2, 1.5)),
              label='allowed by gain + silence after ORN removal only (palp test failed or not run)')]
fig.legend(handles=leg, loc='lower center', ncol=2, frameon=False, fontsize=8.8, bbox_to_anchor=(0.45, 0.0))
fig.suptitle('Is the self-sustained antennal lobe a model defect? Phase bars at h = 0 (white numbers = sustained onset)',
             x=0.02, ha='left', fontsize=12, color=INK, fontweight='bold')
fig.subplots_adjust(left=0.245, right=0.715, top=0.935, bottom=0.135)
fig.savefig(f'{ROOT}/figs/aldef/synth_phase.png', dpi=150, facecolor='white')
print('wrote figs/aldef/synth_phase.png')
