"""m2_synth_table.py -- theory vs simulation agreement table for Model 2 (lane m2/synth, plan results/m2/synth/plan.md).

Reads only lane output files (paths below) and grades every boundary with the pre-registered rule of
results/m2/theory/stochastic_plan.md sec. 3 (point: inside the simulated total interval or within +-10%; bracket:
lower <= sim <= upper; r-trend: on(6)/on(0) within +-20%). Writes results/m2/synth/agreement.json and agreement.csv.
"""
import csv
import json

import numpy as np

ROOT = '/MALECNS_ROOT'
R = f'{ROOT}/results/m2'
OUTJ = f'{R}/synth/agreement.json'
OUTC = f'{R}/synth/agreement.csv'

dec = json.load(open(f'{R}/decay/boundaries.json'))
drv = json.load(open(f'{R}/driven/curves.json'))
mfp = json.load(open(f'{R}/theory/meanfield_predictions.json'))
mfpts = json.load(open(f'{R}/theory/meanfield_points.json'))
sto = json.load(open(f'{R}/theory/stochastic_onset.json'))
sfss = json.load(open(f'{R}/theory/stochastic_fss.json'))
sdrv = json.load(open(f'{R}/theory/stochastic_driven.json'))
nat = json.load(open(f'{R}/nature/summary.json'))
iso = json.load(open(f'{R}/modules/iso.json'))
where = json.load(open(f'{R}/decay/where.json'))
rchk = json.load(open(f'{R}/synth/recheck.json'))

R_ALL = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5,
         5.5, 6]
R_CORE = [0, 0.25, 0.5, 1, 2, 4, 6]
rows = []


def fr(r):
    return f'{r:g}'


def grade_point(th, sim, tot):
    if sim is None and th is None:
        return 'agree (both none)', None
    if th is None:
        return 'fail (theory: none <= 30)', None
    if sim is None:
        return 'fail (sim: none <= 30)', None
    ratio = th / sim
    inside = tot is not None and tot[0] is not None and tot[1] is not None and tot[0] <= th <= tot[1]
    return ('agree' if inside or abs(ratio - 1) <= 0.10 else 'fail'), ratio


def grade_bracket(lo, hi, sim):
    if sim is None or lo is None:
        return 'n/a', None
    if hi is None:
        return ('holds (open above)' if lo <= sim else 'fails'), None
    return ('holds' if lo <= sim <= hi else 'fails'), hi / lo


def add(setting, boundary, graph, r, sim, tot, lane, variant, th, src, kind='point', lo=None, hi=None, note=''):
    if kind == 'point':
        gr, ratio = grade_point(th, sim, tot)
        width = None
    else:
        gr, width = grade_bracket(lo, hi, sim)
        ratio = None
    rows.append(dict(setting=setting, boundary=boundary, graph=graph, r=r, sim=sim,
                     sim_lo=(tot[0] if tot else None), sim_hi=(tot[1] if tot else None), lane=lane, variant=variant,
                     theory=th, lo=lo, hi=hi, ratio=ratio, bracket_width=width, grade=gr, source=src, note=note))


def drow(key):
    return dec['rows'].get(key)


def dval(key, b):
    row = drow(key)
    if row is None or row.get(b) is None:
        return None, None
    v = row[b]
    return v.get('g'), v.get('total')


# ----------------------------------------------------------------------------------------- setting (a), M2 real
P3 = mfp['P3_decay_M2']
SB = sto['models']['M2']['boundaries']


def sto_single(r, b):
    k = f'single_PF_r0_lt_2D' if r == 0 else f'single_CAL_r{fr(r)}_lt_2D'
    v = SB.get(k)
    return None if v is None else v.get(b)


def sto_pooled(r, b):
    tag = 'PF' if r == 0 else 'CAL'
    for lt in ('2D', 'F'):
        v = SB.get(f'pooled_{tag}_r{fr(r)}_lt_{lt}')
        if v and v.get(b) is not None:
            return v.get(b), lt
    return None, None


for r in R_ALL:
    key = f'M2_real_r{fr(r)}'
    p = P3[fr(r)]['primary']
    ch = P3[fr(r)]['chain_AL_mode']['frozen_kcal']
    for b, mfk in (('S', 'S_equal_size'), ('on', 'on_equal_size')):
        sim, tot = dval(key, b)
        add('a', b, 'real', r, sim, tot, 'meanfield', 'CQMF equal-size (A1, heuristic)', p.get(mfk),
            'theory/meanfield_predictions.json P3_decay_M2.<r>.primary')
        if b == 'on':
            add('a', b, 'real', r, sim, tot, 'meanfield', 'bracket [cutoff, AL-chain adiabatic]', None,
                'P3 primary.on / chain_AL_mode.frozen_kcal.on_adiabatic', kind='bracket', lo=p.get('on'),
                hi=ch.get('on_adiabatic'))
            add('a', b, 'real', r, sim, tot, 'meanfield', 'bracket [cutoff, AL-chain frozen]', None,
                'P3 primary.on / chain_AL_mode.frozen_kcal.on_frozen', kind='bracket', lo=p.get('on'),
                hi=ch.get('on_frozen'))
        if r in R_CORE:
            add('a', b, 'real', r, sim, tot, 'stochastic', 'single mode 2-D chain' + (' (PF)' if r == 0 else ' (CAL)'),
                sto_single(r, b), 'theory/stochastic_onset.json models.M2.boundaries.single_*_lt_2D',
                note='pre-registered primary')
            lo, lt = sto_pooled(r, b)
            add('a', b, 'real', r, sim, tot, 'stochastic', f'bracket [pooled {lt}, single 2-D]', None,
                'stochastic_onset.json pooled_* / single_*', kind='bracket', lo=lo, hi=sto_single(r, b))
    sim, tot = dval(key, 'D')
    add('a', 'D', 'real', r, sim, tot, 'meanfield', 'CQMF PR = N/4', p.get('D'), 'P3 primary.D')

# r-trend on(6)/on(0)
sim_tr = dval('M2_real_r6', 'on')[0] / dval('M2_real_r0', 'on')[0]
trend = {
    'sim': sim_tr,
    'meanfield_equal_size': P3['6']['primary']['on_equal_size'] / P3['0']['primary']['on_equal_size'],
    'meanfield_chain_adiabatic': P3['6']['chain_AL_mode']['frozen_kcal']['on_adiabatic'] /
    P3['0']['chain_AL_mode']['frozen_kcal']['on_adiabatic'],
    'stochastic_single': sto_single(6, 'on') / sto_single(0, 'on'),
}
trend['grades'] = {k: ('agree' if abs(v / sim_tr - 1) <= 0.2 else 'fail') for k, v in trend.items() if k != 'sim'}

# ----------------------------------------------------------------------------------------- setting (a), nulls, M2s, variants
P8 = mfp['P8_variants_nulls']
for r in R_CORE:
    for gk, mk in (('cnull', 'M2_nulltargets'), ('wnull', 'M2_nullweights')):
        key = f'M2_{gk}_r{fr(r)}'
        v = P8.get(f'decay|{mk}|r{fr(r)}|cqmf') or P8.get(f'decay|{mk}|r{fr(r)}|qmf') or {}
        sim, tot = dval(key, 'on')
        add('a', 'on', gk, r, sim, tot, 'meanfield', 'CQMF equal-size (M1-real calibration)', v.get('on_equal_size'),
            f'P8_variants_nulls decay|{mk}|r{fr(r)}', note='calibration from the real M1 graph')
        add('a', 'on', gk, r, sim, tot, 'meanfield', 'CQMF cutoff (lower bound)', None, 'P8 .on', kind='bracket',
            lo=v.get('on'), hi=None)
        sim, tot = dval(key, 'D')
        add('a', 'D', gk, r, sim, tot, 'meanfield', 'CQMF PR = N/4', v.get('D'), f'P8 decay|{mk}|r{fr(r)}.D')
    key = f'M2s_real_r{fr(r)}'
    v = P8.get(f'decay|M2s|r{fr(r)}|cqmf') or {}
    for b, mfk in (('S', 'S_equal_size'), ('on', 'on_equal_size')):
        sim, tot = dval(key, b)
        add('a', b, 'M2s', r, sim, tot, 'meanfield', 'CQMF equal-size', v.get(mfk), f'P8 decay|M2s|r{fr(r)}')
    sim, tot = dval(key, 'D')
    add('a', 'D', 'M2s', r, sim, tot, 'meanfield', 'CQMF PR = N/4', v.get('D'), f'P8 decay|M2s|r{fr(r)}.D')

# stochastic: configuration null at full size (r 0 PF; r 1, 4 CAL)
g167 = sfss['graphs']['M2_cnull_N167216_s0']['boundaries']
for r in (0, 1, 4):
    k = 'single_PF_r0_lt_2D' if r == 0 else f'single_CAL_r{r}_lt_2D'
    for b in ('S', 'on'):
        sim, tot = dval(f'M2_cnull_r{r}', b)
        add('a', b, 'cnull', r, sim, tot, 'stochastic', 'single mode 2-D chain, own 167k null', (g167.get(k) or {}).get(b),
            f'theory/stochastic_fss.json graphs.M2_cnull_N167216_s0.{k}')

for var in ('M2_KW09', 'M2_N15', 'M2_allsyn', 'M2_tau30'):
    for r in (0, 1, 4):
        key = f'{var}_real_r{r}'
        sim, tot = dval(key, 'on')
        v = P8.get(f'decay|{var}|r{r}|cqmf') or {}
        if r == 4:
            v = P8.get(f'decay|{var}|r4|cqmf') or {}
        add('a', 'on', var, r, sim, tot, 'meanfield', 'CQMF equal-size', v.get('on_equal_size'),
            f'P8 decay|{var}|r{r}')
        if var != 'M2_allsyn' and var in sto['models']:
            sb = sto['models'][var]['boundaries']
            k = 'single_PF_r0_lt_2D' if r == 0 else f'single_CAL_r{r}_lt_2D'
            add('a', 'on', var, r, sim, tot, 'stochastic', 'single mode 2-D chain' + (' (PF)' if r == 0 else ' (CAL)'),
                (sb.get(k) or {}).get('on'), f'stochastic_onset.json models.{var}.{k}')

# ----------------------------------------------------------------------------------------- setting (b), M2 real
P6 = mfp['P6_driven_M2_ORN0.04']


def lines(ck):
    v = drv[ck]
    out = {}
    for k in ('DQ_DA', 'DT_SS', 'allowed_hi', 'SS_D'):
        val = v['lines'].get(k)
        ci = v['lines_ci'].get(k) or {}
        br = v['lines_bracket'].get(k)
        tot = None
        if ci.get('lo') is not None and br and br[0] is not None and br[1] is not None:
            tot = [min(ci['lo'], br[0]), max(ci['hi'], br[1])]
        out[k] = (val, tot)
    return out


for r in R_ALL:
    L = lines(f'A1|M2|real|0.04|{fr(r)}|False')
    p = P6[fr(r)]['primary']
    add('b', 'DQ->DA (G = 0.29)', 'real', r, *L['DQ_DA'], 'meanfield', 'CQMF', p.get('DQ_DA_G029'),
        'P6_driven_M2_ORN0.04.<r>.primary.DQ_DA_G029')
    add('b', 'DT->SS (removal self-sustained)', 'real', r, *L['DT_SS'], 'meanfield', 'CQMF equal-size (A1)',
        p.get('SS_equal_size'), 'P6 primary.SS_equal_size')
    a1 = p.get('allowed_A1_intervals') or []
    add('b', 'allowed top', 'real', r, *L['allowed_hi'], 'meanfield', 'CQMF A1 window top (grid point)',
        (a1[0]['to'] if a1 else None), 'P6 primary.allowed_A1_intervals[0].to',
        note='grid value; MF strict allowed region (S1 and S2) is empty at every r > 0')

# stochastic LNA (r-independent first-order theory) at r = 0
srows = sdrv['M2_orn0.04']['rows']
sg = np.array([x['g'] for x in srows])
sG = np.array([x['G_gain'] for x in srows])


def cross_up(g, y, lvl):
    k = np.nonzero(y >= lvl)[0]
    if len(k) == 0 or k[0] == 0:
        return None
    k = k[0]
    f = (lvl - y[k - 1]) / (y[k] - y[k - 1])
    return float(np.exp(np.log(g[k - 1]) + f * (np.log(g[k]) - np.log(g[k - 1]))))


L0 = lines('A1|M2|real|0.04|0|False')
add('b', 'DQ->DA (G = 0.29)', 'real', 0, *L0['DQ_DA'], 'stochastic', 'LNA, r-independent', cross_up(sg, sG, 0.29),
    'theory/stochastic_driven.json M2_orn0.04.rows G_gain')
g145_sim = rchk['driven']['A1|M2|real|0.04|0|False']['mine']['G_145']
add('b', 'G = 1.45', 'real', 0, g145_sim, None, 'stochastic', 'LNA, r-independent', cross_up(sg, sG, 1.45),
    'stochastic_driven.json; sim from synth/recheck.json (own crossing)')
for r in (1, 4, 6):
    L = lines(f'A1|M2|real|0.04|{r}|False')
    add('b', 'DQ->DA (G = 0.29)', 'real', r, *L['DQ_DA'], 'stochastic', 'LNA, r-independent (r = 0 closure)',
        cross_up(sg, sG, 0.29), 'stochastic_driven.json', note='theory has no r-dependence at first order')
# driven fluctuation peaks at r = 0 (sim curves.json peaks)
pk = drv['A1|M2|real|0.04|0|False']['peaks']
fano = np.array([x['fano_nonORN'] for x in srows])
tint = np.array([x['tau_int'] for x in srows])
fal = np.array([x['fano_AL'] for x in srows])
for name, arr, sk in (('nonORN Fano peak height', fano, 'F_non'), ('tau_int peak height', tint, 'tau_int'),
                      ('AL Fano peak height', fal, 'F_AL')):
    i = int(np.argmax(arr))
    add('b', name, 'real', 0, pk[sk]['value'], None, 'stochastic', 'LNA', float(arr[i]),
        'stochastic_driven.json rows; sim driven/curves.json peaks', note=f'sim at g {pk[sk]["g"]:.3f}, theory at g {sg[i]:.3f}')
    add('b', name.replace('height', 'location g'), 'real', 0, pk[sk]['g'], pk[sk].get('ci'), 'stochastic', 'LNA',
        float(sg[i]), 'stochastic_driven.json rows; sim curves.json peaks.ci')
for r in (1, 4):
    pkr = drv[f'A1|M2|real|0.04|{r}|False']['peaks']
    i = int(np.argmax(tint))
    add('b', 'tau_int peak height', 'real', r, pkr['tau_int']['value'], None, 'stochastic', 'LNA (r-independent)',
        float(tint[i]), 'stochastic_driven.json; curves.json', note='theory has no r-dependence')

# ----------------------------------------------------------------------------------------- modules alone (r = 0, stochastic)
mods = {'AL': 'M2_mod_AL', 'AL_L': 'M2_mod_ALL', 'AL_R': 'M2_mod_ALR', 'CB': 'M2_mod_CB', 'VNC': 'M2_mod_VNC',
        'OL': 'M2_mod_OL'}
for m, sk in mods.items():
    sim = iso[f'M2_r0_{m}']
    simv = sim['on']['g']
    br = sim['on']['bracket']
    sb = sfss['graphs'][sk]['boundaries']['single_PF_r0_lt_2D']
    pb = sfss['graphs'][sk]['boundaries'].get('pooled_PF_r0_lt_2D') or {}
    add('a-module', 'on (module alone)', m, 0, simv, br, 'stochastic', 'single mode 2-D chain (PF)', sb.get('on'),
        f'stochastic_fss.json graphs.{sk}; sim modules/iso.json M2_r0_{m}',
        note=f'N sim {sim["N"]}, theory {sfss["graphs"][sk]["N"]}; pooled {pb.get("on")}')

# ----------------------------------------------------------------------------------------- FSS of the configuration null (r = 0)
cn = nat['T1_onset']['r0']['cnull']
for N, on, W in zip(cn['N'], cn['onset'], cn['W']):
    key = {5000: 'M2_cnull_N5000_s1', 10000: 'M2_cnull_N10000_s1', 20000: 'M2_cnull_N20000_s1',
           40000: 'M2_cnull_N40000_s1', 80000: 'M2_cnull_N80000_s1', 167216: 'M2_cnull_N167216_s0'}[N]
    sb = sfss['graphs'][key]['boundaries']['single_PF_r0_lt_2D']
    gr, ratio = grade_point(sb['on'], on, None)
    gr = 'agree' if abs(ratio - 1) <= 0.15 else 'fail'  # pre-registered FSS rule: null onsets within +-15%
    rows.append(dict(setting='a-FSS', boundary='null onset', graph=f'cnull N={N}', r=0, sim=on, sim_lo=None, sim_hi=None,
                     lane='stochastic', variant='single mode 2-D chain (PF), own graphs', theory=sb['on'], lo=None, hi=None,
                     ratio=ratio, bracket_width=None, grade=gr,
                     source='nature/summary.json T1_onset.r0.cnull; theory/stochastic_fss.json', note='rule +-15%'))
    rows.append(dict(setting='a-FSS', boundary='width W', graph=f'cnull N={N}', r=0, sim=W, sim_lo=None, sim_hi=None,
                     lane='stochastic', variant='single mode 2-D chain (PF)', theory=sb['W'], lo=None, hi=None,
                     ratio=sb['W'] / W, bracket_width=None, grade='(descriptive)', source='same', note=''))

# ----------------------------------------------------------------------------------------- where activity lives at the onset
where_rows = []
for r in R_CORE:
    w = where.get(f'M2_real_r{fr(r)}', {})
    first = w.get('first_sustained') or w.get('on') or {}
    sim_pr = first.get('PR')
    sim_al = (first.get('region_share') or {}).get('AL_module')
    # meanfield at its equal-size onset: interpolate PR and AL share of decay_M2_cqmf (qmf at r 0) points in ln g
    kk = 'decay_M2_qmf_r0' if r == 0 else f'decay_M2_cqmf_r{fr(r)}'
    pts = [p for p in mfpts[kk]['points'] if p.get('ic') == 'decay']
    gg = np.array([p['g'] for p in pts])
    pr = np.array([p.get('PR') or 0 for p in pts], float)
    al = np.array([(p.get('share') or {}).get('AL_module', np.nan) if isinstance(p.get('share'), dict) else np.nan
                   for p in pts], float)
    o = np.argsort(gg)
    gon = P3[fr(r)]['primary']['on_equal_size']
    mf_pr = float(np.exp(np.interp(np.log(gon), np.log(gg[o]), np.log(np.maximum(pr[o], 1e-9)))))
    mf_al = float(np.interp(np.log(gon), np.log(gg[o]), al[o]))
    st = SB.get('single_PF_r0_lt_2D' if r == 0 else f'single_CAL_r{fr(r)}_lt_2D', {}).get('at_onset', {})
    gsim = first.get('g')
    mf_pr_simg = float(np.exp(np.interp(np.log(gsim), np.log(gg[o]), np.log(np.maximum(pr[o], 1e-9)))))
    mf_al_simg = float(np.interp(np.log(gsim), np.log(gg[o]), al[o]))
    sto_pr_simg = sto_al_simg = None
    if r == 0:
        prof = sto['models']['M2']['profile']
        pg = np.array([q['g'] for q in prof]); ppr = np.array([q['PR'] for q in prof]); pal = np.array([q['share_AL_module'] for q in prof])
        oo = np.argsort(pg)
        sto_pr_simg = float(np.interp(np.log(gsim), np.log(pg[oo]), ppr[oo]))
        sto_al_simg = float(np.interp(np.log(gsim), np.log(pg[oo]), pal[oo]))
    where_rows.append(dict(r=r, sim_g=first.get('g'), sim_PR=sim_pr, sim_AL=sim_al, mf_g=gon, mf_PR=mf_pr, mf_AL=mf_al,
                           sto_PR=st.get('PR'), sto_AL=st.get('share_AL_module'),
                           mf_PR_at_sim_g=mf_pr_simg, mf_AL_at_sim_g=mf_al_simg, sto_PR_at_sim_g=sto_pr_simg,
                           sto_AL_at_sim_g=sto_al_simg,
                           mf_PR_ok=(sim_pr is not None and 0.5 <= mf_pr / sim_pr <= 2),
                           mf_AL_ok=(sim_al is not None and 0.5 <= mf_al / sim_al <= 2),
                           sto_PR_ok=(sim_pr is not None and st.get('PR') is not None and 0.5 <= st['PR'] / sim_pr <= 2),
                           sto_AL_ok=(sim_al is not None and st.get('share_AL_module') is not None and
                                      0.5 <= st['share_AL_module'] / sim_al <= 2)))

# ----------------------------------------------------------------------------------------- tallies
tally = {}
for x in rows:
    gcls = x['graph'] if x['graph'] in ('real', 'cnull', 'wnull', 'M2s') else ('variant' if x['graph'].startswith('M2_') else x['graph'])
    k = f"{x['lane']}|{x['setting']}|{x['boundary']}|{gcls}|{x['variant']}"
    t = tally.setdefault(k, {})
    t[x['grade']] = t.get(x['grade'], 0) + 1
    if x['ratio'] is not None:
        t.setdefault('_ratios', []).append(round(x['ratio'], 3))
for t in tally.values():
    if '_ratios' in t:
        t['ratio_min'] = min(t['_ratios'])
        t['ratio_max'] = max(t['_ratios'])
        del t['_ratios']

json.dump({'rule': 'point: inside sim total interval or within +-10%; bracket: lo <= sim <= hi; trend +-20%; FSS +-15%',
           'rows': rows, 'trend_on6_over_on0': trend, 'where_at_onset': where_rows, 'tally': tally},
          open(OUTJ, 'w'), indent=1, default=float)
with open(OUTC, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for x in rows:
        w.writerow(x)
for k, v in sorted(tally.items()):
    print(k, v)
print('trend', trend)
for x in where_rows:
    print(x)
