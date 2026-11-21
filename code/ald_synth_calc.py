"""AL-defect synthesis: collect the boundary numbers used in the synthesis report and section 12 of the explainer.

Reads only result files written by the E1-E4 lanes (no simulation). Writes results/aldef/synth/synth_numbers.json.
Derived quantities computed here (not in the lane files):
  - D (participation ratio = N/4) for the depression arm sp_NW16_r0, by linear interpolation of PR/N in ln g between the
    two grid points that bracket 0.25 (results/aldef/depression/points.json);
  - PR/N at g = 8 (grid end) for r = 1 arms (D not reached on the E3 grid);
  - slow-window width on/S and on/g_lin for every condition;
  - E1 allowed ranges pooled over the five ORN levels.
"""
import json, math, os
R = '/MALECNS_ROOT/results/aldef'
out = {}
E1 = json.load(open(f'{R}/drive/summary.json'))
E2 = json.load(open(f'{R}/signs/summary.json'))
E3k = json.load(open(f'{R}/depression/key_numbers.json'))
E3p = json.load(open(f'{R}/depression/points.json'))
E4t = open(f'{R}/weights/table.md').read()

GLIN = {'base': E2['spec']['base']['g_lin'], 'lLN1bc_off': E2['spec']['lLN1bc_off']['g_lin'],
        'lLN1bc_lLN2ach_off': E2['spec']['lLN1bc_lLN2ach_off']['g_lin']}
out['g_lin'] = GLIN

# E1 allowed ranges
e1 = {}
for r in (0, 1):
    per = {k: v for k, v in E1['per'].items() if v['r'] == r}
    lo = [v['allowed_interp_primary'][0] for v in per.values()]
    hi = [v['allowed_interp_primary'][1] for v in per.values()]
    hip = [v['allowed_interp_primary_palp'][1] for v in per.values()]
    p1 = [v['crossings']['palp S2p resid1000-3000=0.01'] for v in per.values()]
    chi = [v['chi_time_peak_g'] for v in per.values()]
    e1[r] = {'lower': [min(lo), max(lo)], 'upper': [min(hi), max(hi)], 'upper_with_palp': [min(hip), max(hip)],
             'palp_leak_1pct_at': [min(p1), max(p1)], 'chi_time_peak_g': [min(chi), max(chi)],
             'h0': E1['h0_boundaries'][str(r)]}
out['E1_allowed'] = e1

# E2 deterministic sign conditions
e2 = {}
for k, v in E2['det'].items():
    b = v['bounds']; ig = v['ignition']
    gl = GLIN['base'] if k.startswith('base') else (GLIN['lLN1bc_lLN2ach_off'] if 'lLN2' in k else GLIN['lLN1bc_off'])
    e2[k] = {'S': b['S'], 'on': b['on'], 'on_ci': b['on_ci'], 'D': b['D'],
             'on_over_S': b['on'] / b['S'], 'on_over_glin': b['on'] / gl, 'g_lin': gl,
             'ignition_g': ig['g'], 'ignition_module_share': ig['module_share'], 'ignition_PR_neurons': ig['PR_neurons'],
             'ignition_PR_frac': ig['PR_frac'], 'ignition_n90': ig['n90'],
             'ignition_top_regions': sorted(ig['region'].items(), key=lambda x: -x[1])[:4]}
out['E2_signs'] = e2

# E3 depression arms
def D_interp(pts, thr=0.25):
    pts = sorted([p for p in pts if p['PR'] is not None and p['alive'] and p['alive'] > 0.9], key=lambda p: p['g'])
    for a, b in zip(pts, pts[1:]):
        if a['PR'] < thr <= b['PR']:
            f = (thr - a['PR']) / (b['PR'] - a['PR'])
            return math.exp(math.log(a['g']) + f * (math.log(b['g']) - math.log(a['g']))), [a['g'], b['g']]
    return None, None
e3 = {}
for k in ['base_r0', 'base_r1', 'sp_NW16_r1', 'sp_KW09_r1', 'sp_N15_r1', 'sp_slow_r1', 'sp_NW16_r0', 'rt_NW16_r1',
          'spg_exc_r1', 'spg_lLN1bc_r1', 'spg_AL_r1', 'spg_LN_r1', 'spg_PN_r1', 'spg_notAL_r1', 'spg_ORN_r1',
          'iso_base_ALonly_r1', 'iso_base_noAL_r1', 'iso_sp_NW16_ALonly_r1', 'iso_sp_NW16_noAL_r1',
          'iso_sp_KW09_ALonly_r1', 'iso_sp_KW09_noAL_r1']:
    b = E3k['boundaries'][k]
    d = {'S': b['S']['g'], 'on': b['on']['g'], 'on_total': b['on']['total'], 'on_over_S': b['on']['g'] / b['S']['g'],
         'on_over_glin': b['on']['g'] / GLIN['base'], 'on_ratio_vs_base': b.get('on_ratio_vs_base')}
    if k in E3p:
        D, br = D_interp(E3p[k]); d['D_interp'] = D; d['D_bracket'] = br
        g8 = [p for p in E3p[k] if abs(p['g'] - 8) < 1e-6]
        d['PR_frac_at_g8'] = g8[0]['PR'] if g8 else None
    fs = E3k['at_first_sustained'].get(k)
    if fs:
        d['first_sustained'] = fs
        d['first_sustained_PR_neurons'] = fs['PR_over_N'] * 167216  # N from CLAUDE.md
    e3[k] = d
out['E3_depression'] = e3
out['E3_removal'] = {k: {'allowed': v['allowed_interval'], 'ratio_hi_lo': v['allowed_ratio_hi_lo']}
                     for k, v in E3k['removal_protocol'].items()}
out['E3_driven_linear'] = E3k['driven_linear_response']

# simple cross-lane comparisons
b1 = e2['base_r1']['on']; b0 = e2['base_r0']['on']
out['ratios'] = {
    'sign0_r0_on_over_base': e2['lLN1bc_zero_r0']['on'] / b0,
    'sign0_r1_on_over_base': e2['lLN1bc_zero_r1']['on'] / b1,
    'iii_r0_on_over_base': e2['lLN1bc_lLN2_inh_r0']['on'] / b0,
    'iii_r1_on_over_base': e2['lLN1bc_lLN2_inh_r1']['on'] / b1,
    'NW16_r1_on_over_base_E3': e3['sp_NW16_r1']['on'] / e3['base_r1']['on'],
    'KW09_r1_on_over_base_E3': e3['sp_KW09_r1']['on'] / e3['base_r1']['on'],
    'NW16_r0_on_over_base_E3': e3['sp_NW16_r0']['on'] / e3['base_r0']['on'],
    'lLN1bc_only_dep_r1_on_over_base_E3': e3['spg_lLN1bc_r1']['on'] / e3['base_r1']['on'],
    'E1_upper_over_h0_onset_r0': [e1[0]['upper'][0] / e1[0]['h0']['onset'], e1[0]['upper'][1] / e1[0]['h0']['onset']],
    'E1_upper_over_h0_onset_r1': [e1[1]['upper'][0] / e1[1]['h0']['onset'], e1[1]['upper'][1] / e1[1]['h0']['onset']],
    'E3_NW16_allowed_upper_over_dep_onset': out['E3_removal']['rm_sp_NW16_r1']['allowed'][1] / e3['sp_NW16_r1']['on'],
    'E3_KW09_allowed_upper_over_dep_onset': out['E3_removal']['rm_sp_KW09_r1']['allowed'][1] / e3['sp_KW09_r1']['on'],
    'base_allowed_upper_over_noAL_onset_r1': e1[1]['upper'][1] / e3['iso_base_noAL_r1']['on'],
    'base_allowed_palp_upper_over_noAL_onset_r1': e1[1]['upper_with_palp'][1] / e3['iso_base_noAL_r1']['on'],
    'NW16_allowed_upper_over_ALalone_onset': out['E3_removal']['rm_sp_NW16_r1']['allowed'][1] / e3['iso_sp_NW16_ALonly_r1']['on'],
    'NW16_allowed_upper_over_noAL_onset': out['E3_removal']['rm_sp_NW16_r1']['allowed'][1] / e3['iso_sp_NW16_noAL_r1']['on'],
    'KW09_allowed_upper_over_ALalone_onset': out['E3_removal']['rm_sp_KW09_r1']['allowed'][1] / e3['iso_sp_KW09_ALonly_r1']['on'],
    'KW09_allowed_upper_over_noAL_onset': out['E3_removal']['rm_sp_KW09_r1']['allowed'][1] / e3['iso_sp_KW09_noAL_r1']['on'],
    'exc_allowed_upper_over_exc_onset': out['E3_removal']['rm_spg_exc_r1']['allowed'][1] / e3['spg_exc_r1']['on'],
    'NW16_ALalone_onset_over_allowed_upper': e3['iso_sp_NW16_ALonly_r1']['on'] / out['E3_removal']['rm_sp_NW16_r1']['allowed'][1],
    'base_h0_onset_over_allowed_upper_r1': [e1[1]['h0']['onset'] / e1[1]['upper'][1], e1[1]['h0']['onset'] / e1[1]['upper_with_palp'][0]],
    'base_h0_onset_over_allowed_upper_r0': [e1[0]['h0']['onset'] / e1[0]['upper'][1], e1[0]['h0']['onset'] / e1[0]['upper_with_palp'][0]],
    'NW16_r0_over_r1_onset': e3['sp_NW16_r0']['on'] / e3['sp_NW16_r1']['on'],
    'NW16_r0_D_over_base_D': (e3['sp_NW16_r0']['D_interp'] or float('nan')) / e3['base_r0']['D_interp'],
}
# recurrent amplification (E1 'amplification' = G / feed-forward G) at the interpolated upper end of each allowed range
amp = {}
for k, v in E1['per'].items():
    gs = sorted(float(x) for x in v['amplification'])
    up = v['allowed_interp_primary'][1]
    for a, b in zip(gs, gs[1:]):
        if a <= up <= b:
            fa, fb = v['amplification'][f'{a:g}'], v['amplification'][f'{b:g}']
            amp[k] = fa + (up - a) / (b - a) * (fb - fa)
out['E1_amplification_at_allowed_upper'] = amp
# E2 per-type ensemble: draws with all 30 lLN1_bc excitatory, and how many of them have the AL as top ignition region
ens = {}
for k in ['ens_type_r0', 'ens_type_r1', 'ens_type_r3']:
    v = E2['ens'][k]; n_exc = n_top = 0
    for run, c in zip(v['runs'], v['lLN1bc_exc_counts']):
        reg = run['ignition']['region']
        if c == 30:
            n_exc += 1; n_top += int(max(reg, key=reg.get) == 'AL')
    ens[k] = {'n_draws': v['n_draws'], 'n_lLN1bc_exc': n_exc, 'n_AL_top_at_ignition': n_top}
out['E2_ensemble_top'] = ens
os.makedirs(f'{R}/synth', exist_ok=True)
json.dump(out, open(f'{R}/synth/synth_numbers.json', 'w'), indent=1, default=float)
print(json.dumps({'E1': e1, 'ratios': out['ratios'],
                  'E3_D': {k: (v.get('D_interp'), v.get('D_bracket'), v.get('PR_frac_at_g8'), round(v['on_over_S'], 3), round(v['on_over_glin'], 2)) for k, v in e3.items()},
                  'E2_w': {k: (round(v['on_over_S'], 3), round(v['on_over_glin'], 2), round(v['ignition_PR_frac'], 4)) for k, v in e2.items()}}, indent=1, default=float))
