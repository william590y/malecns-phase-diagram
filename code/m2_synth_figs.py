"""m2_synth_figs.py -- final Model-2 phase-diagram figures (lane m2/synth; plan results/m2/synth/plan.md sec. 3).

Reads only lane outputs (results/m2/{decay,driven,theory,nature,modules}, results/m2/synth/agreement.json) and writes
figs/m2/final_{decay,decay_nulls,driven,m1_vs_m2,agreement}.png. Colors: the dataviz reference palette (categorical
slots in fixed order; one-hue blue ramp for the ordered phases).
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = '/MALECNS_ROOT'
R = f'{ROOT}/results/m2'
F = f'{ROOT}/figs/m2'

dec = json.load(open(f'{R}/decay/boundaries.json'))
drv = json.load(open(f'{R}/driven/curves.json'))
mfp = json.load(open(f'{R}/theory/meanfield_predictions.json'))
mfpts = json.load(open(f'{R}/theory/meanfield_points.json'))
sto = json.load(open(f'{R}/theory/stochastic_onset.json'))
sfss = json.load(open(f'{R}/theory/stochastic_fss.json'))
sdrv = json.load(open(f'{R}/theory/stochastic_driven.json'))
agr = json.load(open(f'{R}/synth/agreement.json'))
where = json.load(open(f'{R}/decay/where.json'))
glin = 0.16315

R_ALL = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5,
         5.5, 6]
R_CORE = [0, 0.25, 0.5, 1, 2, 4, 6]

# palette (dataviz reference, light mode)
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4',
     'green': '#008300', 'violet': '#6250d6', 'red': '#e34948'}
SEQ = ['#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95']

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'legend.fontsize': 7.5,
                     'legend.frameon': False, 'figure.facecolor': 'white', 'axes.facecolor': SURF,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 2,
                     'axes.spines.top': False, 'axes.spines.right': False})


def fr(r):
    return f'{r:g}'


def dline(prefix, b, rs, src=None):
    src = src or dec['rows']
    g, lo, hi, rr, lower = [], [], [], [], []
    for r in rs:
        row = src.get(f'{prefix}_r{fr(r)}')
        if row is None or row.get(b) is None:
            continue
        v = row[b]
        if v.get('g') is None:
            br = v.get('bracket') or [None]
            if br[0] is not None:
                lower.append((br[0], r))
            continue
        tot = v.get('total') or [v['g'], v['g']]
        g.append(v['g']); lo.append(tot[0] if tot[0] is not None else v['g']); hi.append(tot[1] if tot[1] is not None else v['g'])
        rr.append(r)
    return np.array(g), np.array(lo), np.array(hi), np.array(rr), lower


def sim_band(ax, g, lo, hi, rr, color, label, ls='-', marker='o', ms=3.5, lw=2, alpha=0.22, zorder=4):
    if len(g) == 0:
        return
    ax.fill_betweenx(rr, lo, hi, color=color, alpha=alpha, lw=0, zorder=zorder - 1)
    ax.plot(g, rr, ls=ls, color=color, lw=lw, marker=marker, ms=ms, label=label, zorder=zorder)


def phase_fill(ax, rows_S, rows_on, rows_D, rs, xmin=0.05, xmax=30):
    """shade Q / S / L / D between the simulated lines (stepwise per r, interpolated between r values)."""
    S = np.interp(rs, rows_S[3], rows_S[0])
    on = np.interp(rs, rows_on[3], rows_on[0])
    Dg = np.full(len(rs), xmax)
    if len(rows_D[0]):
        Dg = np.interp(rs, rows_D[3], rows_D[0])
        Dg = np.where(rs <= rows_D[3].max(), Dg, xmax)  # no D shading beyond the last r where D was reached
    ax.fill_betweenx(rs, xmin, S, color=SEQ[0], lw=0, zorder=0)
    ax.fill_betweenx(rs, S, on, color=SEQ[1], lw=0, zorder=0)
    ax.fill_betweenx(rs, on, Dg, color=SEQ[2], lw=0, zorder=0)
    ax.fill_betweenx(rs, Dg, xmax, color=SEQ[3], lw=0, zorder=0)


def fmt_g(ax, xmin=0.1, xmax=30, ylab=True):
    ax.set_xscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.15, 6.15)
    ax.set_xlabel('coupling g (log)')
    if ylab:
        ax.set_ylabel('inhibition scale r')
    ax.axvline(glin, color=INK, lw=1.2, ls=(0, (1, 1.5)), zorder=5)


# ======================================================================================== figure 1: decay (setting a)
def fig_decay():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axs[0]
    S = dline('M2_real', 'S', R_ALL)
    on = dline('M2_real', 'on', R_ALL)
    D = dline('M2_real', 'D', R_ALL)
    rs = np.linspace(0, 6, 200)
    phase_fill(ax, S, on, D, rs)
    sim_band(ax, *S[:4], INK2, 'sim: S (tau = 200)', ls='--', marker='o', ms=3)
    sim_band(ax, *on[:4], INK, 'sim: onset (tau = 3000)', marker='o', ms=3.5)
    sim_band(ax, *D[:4], C['violet'], 'sim: D (PR = N/4)', marker='s', ms=3.5)
    # theory: meanfield
    P3 = mfp['P3_decay_M2']
    eq = np.array([P3[fr(r)]['primary']['on_equal_size'] for r in R_ALL])
    cut = np.array([P3[fr(r)]['primary']['on'] for r in R_ALL])
    chA = np.array([P3[fr(r)]['chain_AL_mode']['frozen_kcal']['on_adiabatic'] for r in R_ALL])
    chF = np.array([P3[fr(r)]['chain_AL_mode']['frozen_kcal']['on_frozen'] for r in R_ALL])
    mfD = [(P3[fr(r)]['primary']['D'], r) for r in R_ALL]
    ax.plot(eq, R_ALL, color=C['orange'], lw=2, ls='--', label='mean field: onset, equal-size rule (heuristic)', zorder=6)
    ax.plot(cut, R_ALL, color=C['orange'], lw=1.3, ls=':', label='mean field: cutoff (lower bound)', zorder=6)
    ax.fill_betweenx(R_ALL, chA, chF, facecolor='none', edgecolor=C['yellow'], hatch='...', lw=0, zorder=2)
    ax.plot(chA, R_ALL, color=C['yellow'], lw=1.5, ls='-.', label='mean field: AL-mode chain, adiabatic .. frozen', zorder=6)
    ax.plot(chF, R_ALL, color=C['yellow'], lw=1.5, ls='-.', zorder=6)
    dd = np.array([(d, r) for d, r in mfD if d is not None])
    ax.plot(dd[:, 0], dd[:, 1], ls='none', marker='D', ms=5, mfc='white', mec=C['violet'], mew=1.4,
            label='mean field: D (open: none up to 30 at the other r)', zorder=7)
    nod = [r for d, r in mfD if d is None]
    ax.plot([29] * len(nod), nod, ls='none', marker='>', ms=5, color=C['violet'], alpha=0.6, zorder=7)
    # theory: stochastic (core r)
    SB = sto['models']['M2']['boundaries']
    st_on = [SB['single_PF_r0_lt_2D']['on']] + [SB[f'single_CAL_r{fr(r)}_lt_2D']['on'] for r in R_CORE[1:]]
    po = []
    for r in R_CORE:
        tag = 'PF' if r == 0 else 'CAL'
        v = SB.get(f'pooled_{tag}_r{fr(r)}_lt_2D') or {}
        if v.get('on') is None:
            v = SB.get(f'pooled_{tag}_r{fr(r)}_lt_F') or {}
        po.append(v.get('on'))
    ax.plot(st_on, R_CORE, color=C['aqua'], lw=2, ls='-', marker='^', ms=6,
            label='stochastic: single renormalised mode (pre-registered; r > 0 M1-calibrated)', zorder=6)
    ax.plot(po, R_CORE, color=C['aqua'], lw=1.3, ls=':', marker='v', ms=5, label='stochastic: pooled state (lower bound; 2-D chain where solved, else frozen)',
            zorder=6)
    fmt_g(ax, 0.15, 30)
    ax.text(glin * 1.04, 6.0, 'g_lin', color=INK, fontsize=8, va='top')
    for txt, x in (('Q', 0.5), ('S', None), ('L', 6.6), ('D', 22)):
        if x:
            ax.text(x, 5.55, txt, fontsize=12, color=SEQ[6], fontweight='bold', ha='center')
    ax.set_title('(a) Model 2, h = 0 decay: simulation (lines + 95% bands) vs theory')
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color=SEQ[0], label='Q: tau < 200'), Patch(color=SEQ[1], label='S: 200 <= tau < 3000'),
                Patch(color=SEQ[2], label='L: sustained, PR < N/4'), Patch(color=SEQ[3], label='D: sustained, PR >= N/4')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.52), ncol=2, fontsize=7)

    # right: lifetimes per r? -> ratio panel of theory/sim onset
    ax = axs[1]
    simon = on[0]
    simr = on[3]
    sim_at = dict(zip(simr, simon))
    ax.axhspan(0.9, 1.1, color=GRID, zorder=0, lw=0)
    ax.axhline(1, color=MUTED, lw=1)
    ax.plot(R_ALL, eq / np.array([sim_at[r] for r in R_ALL]), color=C['orange'], ls='--', marker='o', ms=4,
            label='mean field equal-size')
    ax.plot(R_ALL, chA / np.array([sim_at[r] for r in R_ALL]), color=C['yellow'], ls='-.', marker='o', ms=3,
            label='mean field AL chain (adiabatic)')
    ax.plot(R_ALL, chF / np.array([sim_at[r] for r in R_ALL]), color=C['yellow'], ls=':', marker='o', ms=3,
            label='mean field AL chain (frozen)')
    ax.plot(R_ALL, cut / np.array([sim_at[r] for r in R_ALL]), color=C['orange'], ls=':', marker='.', ms=4,
            label='mean field cutoff')
    ax.plot(R_CORE, np.array(st_on) / np.array([sim_at[r] for r in R_CORE]), color=C['aqua'], marker='^', ms=6,
            label='stochastic single mode')
    ax.plot(R_CORE, np.array(po, float) / np.array([sim_at[r] for r in R_CORE]), color=C['aqua'], ls=':', marker='v',
            ms=5, label='stochastic pooled')
    ax.set_yscale('log')
    ax.set_yticks([0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3])
    ax.set_yticklabels(['0.2', '0.3', '0.5', '0.7', '1', '1.5', '2', '3'])
    ax.set_xlabel('inhibition scale r')
    ax.set_ylabel('predicted onset / simulated onset (log)')
    ax.set_title('(b) onset: theory / simulation (grey band = +-10%)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.42), fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f'{F}/final_decay.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


# ======================================================================================== figure 2: nulls
def fig_nulls():
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 5.4), sharey=True)
    P8 = mfp['P8_variants_nulls']
    for ax, (gk, mk, ttl) in zip(axs, (('cnull', 'M2_nulltargets', 'configuration null (targets permuted)'),
                                         ('wnull', 'M2_nullweights', 'weight-shuffled null (wiring kept)'),
                                         ('M2s', 'M2s', 'M2s (per-target lLN1_bc signs)'))):
        pre = f'M2_{gk}' if gk != 'M2s' else 'M2s_real'
        S = dline(pre, 'S', R_CORE)
        on = dline(pre, 'on', R_CORE)
        D = dline(pre, 'D', R_CORE)
        rs = np.linspace(0, 6, 100)
        phase_fill(ax, S, on, D, rs)
        sim_band(ax, *S[:4], INK2, 'sim S', ls='--', ms=3)
        sim_band(ax, *on[:4], INK, 'sim onset', ms=3.5)
        sim_band(ax, *D[:4], C['violet'], 'sim D', marker='s', ms=3.5)
        ron = dline('M2_real', 'on', R_ALL)
        ax.plot(ron[0], ron[3], color=MUTED, lw=1.2, ls='-', label='M2 real onset (reference)')
        eqk = 'on_equal_size'
        th_eq, th_cut, th_D = [], [], []
        for r in R_CORE:
            v = P8.get(f'decay|{mk}|r{fr(r)}|cqmf') or P8.get(f'decay|{mk}|r{fr(r)}|qmf') or {}
            th_eq.append(v.get(eqk)); th_cut.append(v.get('on')); th_D.append(v.get('D'))
        ax.plot(th_eq, R_CORE, color=C['orange'], ls='--', marker='o', ms=4,
                label='mean field onset, equal-size' + (' (M1-real calibration)' if gk != 'M2s' else ''))
        ax.plot(th_cut, R_CORE, color=C['orange'], ls=':', lw=1.3, label='mean field cutoff (lower bound)')
        dd = [(d, r) for d, r in zip(th_D, R_CORE) if d is not None]
        if dd:
            ax.plot(*zip(*dd), ls='none', marker='D', ms=5, mfc='white', mec=C['violet'], mew=1.4, label='mean field D')
        nod = [r for d, r in zip(th_D, R_CORE) if d is None]
        if nod:
            ax.plot([28] * len(nod), nod, ls='none', marker='>', ms=5, color=C['violet'], alpha=0.6,
                    label='mean field: no D up to 30')
        if gk == 'cnull':
            g167 = sfss['graphs']['M2_cnull_N167216_s0']['boundaries']
            pts = [(g167['single_PF_r0_lt_2D']['on'], 0)] + [(g167[f'single_CAL_r{r}_lt_2D']['on'], r) for r in (1, 4)]
            ax.plot(*zip(*pts), ls='none', marker='^', ms=7, color=C['aqua'], label='stochastic single mode, own 167k null')
        if gk == 'M2s':
            ttl = ttl + '\nsim: = M2 within 6%; mean field: within 0-5%'
        fmt_g(ax, 0.15, 30, ylab=(gk == 'cnull'))
        ax.set_title(ttl)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6.8, ncol=2)
    fig.suptitle('Model 2, h = 0 decay: null graphs and the sign variant, simulation vs theory (7 core r values)',
                 fontsize=10.5, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{F}/final_decay_nulls.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


# ======================================================================================== figure 3: driven (setting b)
PH_COL = {'DQ': SEQ[0], 'DA': SEQ[1], 'DT': SEQ[3], 'SS-L': SEQ[4], 'SS-D': SEQ[6]}


def phase_of(p):
    for k in ('SS-L', 'SS-D', 'DQ', 'DA', 'DT'):
        if p.startswith(k):
            return k
    return None


def fig_driven():
    fig = plt.figure(figsize=(13.5, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, :2])
    # tiles
    for i, r in enumerate(R_ALL):
        v = drv[f'A1|M2|real|0.04|{fr(r)}|False']
        g = np.array(v['g'])
        ph = v['curve']['phase']
        al = np.array(v['curve']['allowed'], bool)
        edges = np.sqrt(g[1:] * g[:-1])
        lo_e = np.r_[g[0] / 1.09, edges]
        hi_e = np.r_[edges, g[-1] * 1.09]
        rlo = (R_ALL[i - 1] + r) / 2 if i > 0 else r - 0.0625
        rhi = (R_ALL[i + 1] + r) / 2 if i < len(R_ALL) - 1 else r + 0.25
        for j in range(len(g)):
            k = phase_of(ph[j])
            if k is None:
                continue
            ax.fill_between([lo_e[j], hi_e[j]], rlo, rhi, color=PH_COL[k], lw=0, zorder=0)
            if al[j]:
                ax.fill_between([lo_e[j], hi_e[j]], rlo, rhi, facecolor='none', edgecolor=C['green'], hatch='////',
                                lw=0, zorder=1)
    L = {k: [] for k in ('DQ_DA', 'allowed_hi', 'DT_SS', 'SS_D')}
    for r in R_ALL:
        v = drv[f'A1|M2|real|0.04|{fr(r)}|False']['lines']
        for k in L:
            L[k].append(v.get(k))
    ax.plot(L['DQ_DA'], R_ALL, color=INK, lw=1.6, ls='--', label='sim: G = 0.29 (DQ -> DA)')
    ax.plot(L['allowed_hi'], R_ALL, color=C['green'], lw=2, label='sim: top of data-allowed region (S1 lost)')
    ax.plot(L['DT_SS'], R_ALL, color=INK, lw=2, label='sim: DT -> SS (activity outlives ORN removal)')
    ax.plot(L['SS_D'], R_ALL, color=C['violet'], lw=1.6, marker='s', ms=3, label='sim: SS-L -> SS-D')
    on = dline('M2_real', 'on', R_ALL)
    ax.plot(on[0], on[3], ls='none', marker='o', ms=4, mfc='white', mec=C['red'], mew=1.3,
            label='sim, h = 0: decay onset (setting a)')
    S = dline('M2_real', 'S', R_ALL)
    ax.plot(S[0], S[3], ls='none', marker='|', ms=7, color=C['red'], mew=1.5, label='sim, h = 0: decay S line')
    P6 = mfp['P6_driven_M2_ORN0.04']
    mg = [P6[fr(r)]['primary']['DQ_DA_G029'] for r in R_ALL]
    ms_ = [P6[fr(r)]['primary']['SS_equal_size'] for r in R_ALL]
    ax.plot(mg, R_ALL, color=C['orange'], lw=1.8, ls='--', marker='o', ms=3, label='mean field: G = 0.29')
    ax.plot(ms_, R_ALL, color=C['magenta'], lw=2, ls='-.', label='mean field: removal-SS, equal-size rule')
    srows = sdrv['M2_orn0.04']['rows']
    sg = np.array([x['g'] for x in srows]); sG = np.array([x['G_gain'] for x in srows])

    def xup(lvl):
        k = np.nonzero(sG >= lvl)[0][0]
        f = (lvl - sG[k - 1]) / (sG[k] - sG[k - 1])
        return float(np.exp(np.log(sg[k - 1]) + f * np.log(sg[k] / sg[k - 1])))
    ax.plot([xup(0.29)] * 2, [-0.1, 0.35], color=C['aqua'], lw=3, solid_capstyle='butt',
            label='stochastic LNA (r = 0 only): G = 0.29 and G = 1.45')
    ax.plot([xup(1.45)] * 2, [-0.1, 0.35], color=C['aqua'], lw=3, solid_capstyle='butt')
    fmt_g(ax, 0.05, 30)
    ax.set_title('(a) Model 2 driven by spontaneous ORN activity (ORN level 0.04): phases, simulated lines and theory')
    hs, ls_ = ax.get_legend_handles_labels()
    hs = [Patch(color=PH_COL[k], label=k) for k in PH_COL] + \
        [Patch(facecolor='white', edgecolor=C['green'], hatch='////', label='data-allowed (G 0.29-1.45, S1, S2)')] + hs
    ax.legend(handles=hs, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=7)

    # (b) gain curves sim vs theory
    ax = fig.add_subplot(gs[1, 0])
    for r, col in ((0, C['blue']), (1, C['orange']), (4, C['aqua'])):
        v = drv[f'A1|M2|real|0.04|{fr(r)}|False']
        g = np.array(v['g']); G = np.array(v['curve']['G'], float)
        ax.plot(g, G, ls='none', marker='o', ms=3.2, color=col, label=f'sim r = {r}')
        pts = [p for p in mfpts[f'driven_M2_cqmf_r{fr(r)}' if r else 'driven_M2_qmf_r0']['points'] if p.get('ic') == 'quiet']
        pg = np.array([p['g'] for p in pts]); pG = np.array([p['G'] for p in pts])
        o = np.argsort(pg)
        ax.plot(pg[o], pG[o], color=col, lw=1.6, ls='--', label=f'mean field r = {r}')
    ax.plot(sg, sG, color=INK, lw=1.4, ls=':', label='stochastic LNA (r-independent)')
    ax.axhspan(0.29, 1.45, color=C['green'], alpha=0.08, lw=0)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.05, 12); ax.set_ylim(0.03, 8)
    ax.set_xlabel('coupling g (log)'); ax.set_ylabel('PN/ORN gain G (log)')
    ax.set_title('(b) gain: simulation vs theory')
    ax.legend(fontsize=6.3, ncol=2, loc='upper left')

    # (c) fluctuations at r = 0
    ax = fig.add_subplot(gs[1, 1])
    v = drv['A1|M2|real|0.04|0|False']
    g = np.array(v['g'])
    ax.plot(g, np.array(v['curve']['F_non'], float), ls='none', marker='o', ms=3.2, color=C['blue'],
            label='sim nonORN Fano')
    ax.plot(sg, [x['fano_nonORN'] for x in srows], color=C['blue'], lw=1.6, ls='--', label='LNA nonORN Fano')
    ax.plot(g, np.array(v['curve']['tau_int'], float), ls='none', marker='s', ms=3.2, color=C['orange'],
            label='sim tau_int (units)')
    ax.plot(sg, [x['tau_int'] for x in srows], color=C['orange'], lw=1.6, ls='--', label='LNA tau_int')
    for r, mk in ((1, '^'), (4, 'v')):
        vv = drv[f'A1|M2|real|0.04|{r}|False']
        ax.plot(vv['g'], np.array(vv['curve']['tau_int'], float), ls='none', marker=mk, ms=3, color=C['orange'],
                alpha=0.45, label=f'sim tau_int r = {r}')
    ax.axvline(v['lines']['DT_SS'], color=INK, lw=1, ls=':')
    ax.text(v['lines']['DT_SS'] * 1.05, 1.0, 'DT->SS', fontsize=7, color=INK2)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.05, 12); ax.set_ylim(0.8, 150)
    ax.set_xlabel('coupling g (log)'); ax.set_ylabel('value (log)')
    ax.set_title('(c) fluctuations, r = 0: simulation vs LNA')
    ax.legend(fontsize=6.3, loc='upper left', ncol=2)

    # (d) allowed region vs r: M2, M1, M2s, ORN levels at r = 1
    ax = fig.add_subplot(gs[1, 2])
    for (setk, mdl, col, lab, dy) in (('A2', 'M1', MUTED, 'M1', -0.27), ('A1', 'M2', C['blue'], 'M2', -0.09),
                                      ('A4', 'M2s', C['magenta'], 'M2s', 0.09)):
        for yi, r in enumerate(R_CORE):
            v = drv.get(f'{setk}|{mdl}|real|0.04|{fr(r)}|False')
            if v is None:
                continue
            lo, hi = v['lines'].get('allowed_lo'), v['lines'].get('allowed_hi')
            if lo and hi:
                ax.plot([lo, hi], [yi + dy, yi + dy], color=col, lw=4, solid_capstyle='butt',
                        label=lab if r == 0 else None)
            ax.plot(v['lines']['DT_SS'], yi + dy, marker='|', ms=8, mew=2, color=col)
    a1 = [mfp['P6_driven_M2_ORN0.04'][fr(r)]['primary'].get('allowed_A1_intervals') or [] for r in R_CORE]
    for yi, (r, a) in enumerate(zip(R_CORE, a1)):
        if a:
            ax.plot([a[0]['from'], a[0]['to']], [yi + 0.27] * 2, color=C['orange'], lw=1.6, ls='--',
                    label='mean field gain window (A1, M2)' if r == 0 else None)
    ax.set_xscale('log'); ax.set_xlim(0.07, 3); ax.set_ylim(-0.6, len(R_CORE) - 0.4)
    ax.set_yticks(range(len(R_CORE))); ax.set_yticklabels([fr(r) for r in R_CORE])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('coupling g (log)'); ax.set_ylabel('inhibition scale r (core values)')
    ax.set_title('(d) data-allowed region (bars) and DT->SS (ticks)')
    ax.legend(fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, -0.42), ncol=2)
    fig.savefig(f'{F}/final_driven.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


# ======================================================================================== figure 4: M1 vs M2
def fig_m1_vs_m2():
    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22)
    ref = dec['m1_reference']
    for k, (ttl, pre, src) in enumerate((('(a) Model 1 (no depression, h = 0): analysis2 lines', 'M1ref_real', ref),
                                          ('(b) Model 2 (with depression, h = 0): this program', 'M2_real',
                                           dec['rows']))):
        ax = fig.add_subplot(gs[0, k])
        S = dline(pre, 'S', R_ALL, src)
        on = dline(pre, 'on', R_ALL, src)
        D = dline(pre, 'D', R_ALL, src)
        rs = np.linspace(0, 6, 200)
        phase_fill(ax, S, on, D, rs, xmin=0.05, xmax=30)
        sim_band(ax, *S[:4], INK2, 'S (tau = 200)', ls='--', ms=3)
        sim_band(ax, *on[:4], INK, 'onset (tau = 3000)', ms=3.5)
        sim_band(ax, *D[:4], C['violet'], 'D (PR = N/4)', marker='s', ms=3.5)
        if D[4]:
            lb = np.array(D[4])
            ax.plot(lb[:, 0], lb[:, 1], ls='none', marker='>', ms=6, color=C['violet'],
                    label='D not reached up to this g')
        if k == 1:
            o1 = dline('M1ref_real', 'on', R_ALL, ref)
            ax.plot(o1[0], o1[3], color=MUTED, lw=1.2, label='M1 onset (for scale)')
        fmt_g(ax, 0.1, 30)
        ax.set_title(ttl)
        if k == 0:
            ax.legend(loc='upper right', fontsize=7)
        else:
            ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0.12), fontsize=6.5, framealpha=0.85, frameon=True,
                      edgecolor='none')
    # (c) where at onset: AL share and PR/N
    ax = fig.add_subplot(gs[1, 0])
    for mdl, col, mk in (('M1', MUTED, 's'), ('M2', C['blue'], 'o'), ('M2s', C['magenta'], 'D'), ('M1s', C['yellow'], 'v')):
        rr, al, pr = [], [], []
        for r in R_ALL:
            w = where.get(f'{mdl}_real_r{fr(r)}')
            if not w or not w.get('on'):
                continue
            rr.append(r); al.append(w['on']['region_share']['AL_module']); pr.append(w['on']['PRN'])
        if rr:
            ax.plot(rr, np.array(al) * 100, color=col, marker=mk, ms=5, lw=1.6, label=f'{mdl}: AL-module share')
    ax.set_xlabel('inhibition scale r'); ax.set_ylabel('% of all activity in the AL module (526 neurons)')
    ax.set_ylim(0, 60)
    ax.set_title('(c) where the first sustained state lives (h = 0)')
    ax2 = ax.inset_axes([0.5, 0.2, 0.46, 0.4])
    for mdl, col, mk in (('M1', MUTED, 's'), ('M2', C['blue'], 'o')):
        rr, pr = [], []
        for r in R_ALL:
            w = where.get(f'{mdl}_real_r{fr(r)}')
            if not w or not w.get('on'):
                continue
            rr.append(r); pr.append(w['on']['PR'])
        ax2.plot(rr, pr, color=col, marker=mk, ms=4, lw=1.4, label=mdl)
    ax2.set_yscale('log'); ax2.set_title('neurons carrying it (PR)', fontsize=7.5, fontweight='normal')
    ax2.tick_params(labelsize=7); ax2.legend(fontsize=6.5, loc='center right')
    ax.legend(loc='center left', fontsize=7)
    # (d) driven: allowed region and SS line M1 vs M2 (23 r for M2)
    ax = fig.add_subplot(gs[1, 1])
    for (setk, mdl, col, rs_) in (('A2', 'M1', MUTED, R_CORE), ('A1', 'M2', C['blue'], R_ALL)):
        lo, hi, ss, rr = [], [], [], []
        for r in rs_:
            v = drv.get(f'{setk}|{mdl}|real|0.04|{fr(r)}|False')
            if v is None:
                continue
            lo.append(v['lines']['allowed_lo']); hi.append(v['lines']['allowed_hi']); ss.append(v['lines']['DT_SS'])
            rr.append(r)
        ax.fill_betweenx(rr, lo, hi, color=col, alpha=0.3, lw=0, label=f'{mdl}: data-allowed region')
        ax.plot(ss, rr, color=col, lw=2, label=f'{mdl}: DT -> SS (= its h = 0 onset)')
    ax.axvline(glin, color=INK, lw=1.2, ls=(0, (1, 1.5)))
    ax.text(glin * 1.04, 6.0, 'g_lin', fontsize=8, va='top')
    ax.set_xscale('log'); ax.set_xlim(0.07, 3); ax.set_ylim(-0.15, 6.15)
    ax.set_xlabel('coupling g (log)'); ax.set_ylabel('inhibition scale r')
    ax.set_title('(d) ORN-driven: where the real fly may sit (ORN level 0.04)')
    ax.legend(fontsize=7, loc='lower right')
    fig.savefig(f'{F}/final_m1_vs_m2.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


# ======================================================================================== figure 5: agreement
def fig_agreement():
    rows = agr['rows']
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    specs = [
        (axs[0], 'a', [('on', 'meanfield', 'CQMF equal-size (A1, heuristic)', C['orange'], 'o', 'MF onset (equal-size)'),
                       ('S', 'meanfield', 'CQMF equal-size (A1, heuristic)', C['orange'], 'x', 'MF S (equal-size)'),
                       ('on', 'stochastic', None, C['aqua'], '^', 'stochastic onset (single mode)'),
                       ('S', 'stochastic', None, C['aqua'], '+', 'stochastic S (single mode)')],
         '(a) h = 0: S and onset'),
        (axs[1], 'a', [('D', 'meanfield', 'CQMF PR = N/4', C['violet'], 'D', 'MF D')], '(b) h = 0: D line'),
        (axs[2], 'b', [('DQ->DA (G = 0.29)', 'meanfield', 'CQMF', C['orange'], 'o', 'MF G = 0.29'),
                       ('DT->SS (removal self-sustained)', 'meanfield', None, C['orange'], 's', 'MF removal-SS (equal-size)'),
                       ('allowed top', 'meanfield', None, C['yellow'], 'v', 'MF allowed top (grid)'),
                       ('DQ->DA (G = 0.29)', 'stochastic', None, C['aqua'], '^', 'LNA G = 0.29 (r-independent)')],
         '(c) ORN-driven lines'),
    ]
    for ax, setting, series, ttl in specs:
        ax.axhspan(0.9, 1.1, color=GRID, lw=0, zorder=0)
        ax.axhline(1, color=MUTED, lw=1)
        for b, lane, var, col, mk, lab in series:
            pts = [x for x in rows if x['setting'] == setting and x['boundary'] == b and x['lane'] == lane and
                   x['graph'] == 'real' and (var is None or x['variant'] == var) and 'bracket' not in x['variant']]
            ok = sorted((x['r'], x['ratio'] if x['ratio'] is not None else np.nan) for x in pts)
            none = [x['r'] for x in pts if x['ratio'] is None and x['sim'] is not None]
            if ok:
                ax.plot(*zip(*ok), color=col, marker=mk, ms=5, lw=1.4, label=lab)
            if none:
                ax.plot(none, [3.6] * len(none), ls='none', marker='^', ms=6, mfc='white', mec=col, mew=1.3,
                        label=lab + ': theory none up to g 30')
        ax.set_yscale('log'); ax.set_ylim(0.35, 4.0)
        ax.set_yticks([0.4, 0.5, 0.7, 1, 1.5, 2, 3])
        ax.set_yticklabels(['0.4', '0.5', '0.7', '1', '1.5', '2', '3'])
        ax.set_xlabel('inhibition scale r')
        ax.set_title(ttl)
        ax.legend(fontsize=6.8, loc='lower left' if ax is axs[1] else 'upper left')
    P3 = mfp['P3_decay_M2']
    simon = {x['r']: x['sim'] for x in rows if x['setting'] == 'a' and x['boundary'] == 'on' and x['graph'] == 'real'
             and x['lane'] == 'meanfield' and x['variant'].startswith('CQMF equal-size')}
    rr = sorted(simon)
    axs[0].plot(rr, [P3[fr(r)]['chain_AL_mode']['frozen_kcal']['on_adiabatic'] / simon[r] for r in rr],
                color=C['yellow'], ls='-.', marker='.', ms=5, lw=1.3,
                label='MF AL-mode chain, adiabatic (meant as upper bound)')
    axs[0].legend(fontsize=6.8, loc='upper left')
    axs[0].set_ylabel('theory / simulation (log; grey = +-10%)')
    fig.suptitle('Model 2: theory / simulation for every graded boundary of the real connectome', fontsize=10.5,
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{F}/final_agreement.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_decay()
    fig_nulls()
    fig_driven()
    fig_m1_vs_m2()
    fig_agreement()
    print('done')
