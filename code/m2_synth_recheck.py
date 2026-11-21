"""m2_synth_recheck.py -- independent recheck of the M2 decay and driven boundary lines from the RAW simulator outputs.

Lane m2/synth (plan: results/m2/synth/plan.md, sec. 1). Imports no project module: own lifetime estimator, own PAVA,
own crossing rule. Compares with the lane values in results/m2/decay/boundaries.json and results/m2/driven/curves.json.
Output: results/m2/synth/recheck.json (+ a printed table).

Usage: python code/m2_synth_recheck.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = '/MALECNS_ROOT'
DEC = f'{ROOT}/results/m2/decay'
DRV = f'{ROOT}/results/m2/driven'
OUT = f'{ROOT}/results/m2/synth/recheck.json'
N_ALL = 167216
N_NONORN = 164577
T_DEC = 3000.0
T_POST = 3000.0


def pava(y, w=None):
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    blocks = []  # [value, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    return np.concatenate([[b[0]] * b[2] for b in blocks])


def cross(g, y, level, decreasing=False):
    """first crossing of `level` by the PAVA fit of y(g), linear in ln g; None if never."""
    g = np.asarray(g, float)
    y = np.asarray(y, float)
    o = np.argsort(g)
    g, y = g[o], y[o]
    if decreasing:
        y, level = -y, -level
    f = pava(y)
    k = np.nonzero(f >= level)[0]
    if len(k) == 0:
        return None
    k = int(k[0])
    if k == 0:
        return float(g[0])
    a, b = f[k - 1], f[k]
    fr = 0.0 if b == a else (level - a) / (b - a)
    return float(np.exp(np.log(g[k - 1]) + fr * (np.log(g[k]) - np.log(g[k - 1]))))


def tau_cens(ext, T):
    ext = np.asarray(ext, float)
    dead = np.isfinite(ext) & (ext < T)
    d = int(dead.sum())
    tot = float(np.minimum(np.where(np.isfinite(ext), ext, T), T).sum())
    return (tot / d if d else np.inf), d


# --------------------------------------------------------------------------------------------- decay
def decay_rows():
    idx = []
    for f in sorted(glob.glob(f'{DEC}/sim/index_s1.json')) + sorted(glob.glob(f'{DEC}/sim/index_s2*.json')):
        idx += json.load(open(f))
    rows = defaultdict(list)
    for e in idx:
        rows[(e['model'], e['graph'], float(e['r']))].append(e['out'])
    return rows


def decay_row(files):
    per_g = defaultdict(lambda: {'ext': [], 'occ': [], 'nrep': 0})
    for fn in files:
        z = np.load(fn)
        t = z['t']
        surv = z['surv']  # [g, rec, B]
        occ = z['occ'].astype(np.float64)
        for i, g in enumerate(z['g']):
            s = surv[i]
            dead = s <= 0
            first = np.where(dead.any(0), np.argmax(dead, 0), -1)
            ext = np.where(first >= 0, t[np.maximum(first, 0)], np.inf)
            key = round(float(g), 6)
            per_g[key]['ext'].append(ext)
            per_g[key]['occ'].append(occ[i] * s.shape[1])
            per_g[key]['nrep'] += s.shape[1]
    gs = np.array(sorted(per_g))
    tau, deaths, pr = [], [], []
    for g in gs:
        ext = np.concatenate(per_g[g]['ext'])
        tt, d = tau_cens(ext, T_DEC)
        tau.append(tt)
        deaths.append(d)
        o = np.sum(per_g[g]['occ'], 0) / per_g[g]['nrep']
        s2 = (o * o).sum()
        pr.append(float(o.sum() ** 2 / s2) if s2 > 0 else 0.0)
    tau = np.array(tau)
    pr = np.array(pr)
    lt = np.log(np.where(np.isfinite(tau), tau, 1e9))
    S = cross(gs, lt, np.log(200.0))
    on = cross(gs, lt, np.log(3000.0))
    sus = np.isfinite(tau) & (tau >= 3000) | ~np.isfinite(tau)
    D = cross(gs[sus], pr[sus], N_ALL / 4) if sus.sum() >= 2 else None
    return {'S': S, 'on': on, 'D': D, 'n_g': int(len(gs))}


def lane_decay(b, model, graph, r):
    key = f'{model}_{graph}_r{r:g}'
    row = b['rows'].get(key)
    if row is None:
        return key, None
    out = {}
    for k in ('S', 'on', 'D'):
        v = row.get(k) or {}
        out[k] = {'g': v.get('g'), 'total': v.get('total')}
    return key, out


# --------------------------------------------------------------------------------------------- driven
def driven_row(files):
    per_g = defaultdict(lambda: {'orn': [], 'upn': [], 'upn_post': [], 'ext': []})
    for fn in files:
        z = np.load(fn)
        names = [str(x) for x in z['reg_names']]
        iO, iU, iN = names.index('ORN'), names.index('uPN_ant'), names.index('nonORN')
        wm = z['win_mean']  # [g, win, R, B]
        wb = z['win_bounds']
        k_drv = int(np.nonzero((wb[:, 0] == 500) & (wb[:, 1] == 1500))[0][0])
        k_post = int(np.nonzero((wb[:, 0] == 1600) & (wb[:, 1] == 1800))[0][0])
        last = z['reg_last']  # [g, R, B]
        for i, g in enumerate(z['g']):
            key = round(float(g), 6)
            per_g[key]['orn'].append(wm[i, k_drv, iO])
            per_g[key]['upn'].append(wm[i, k_drv, iU])
            per_g[key]['upn_post'].append(wm[i, k_post, iU])
            tl = last[i, iN].astype(float) - 1500.0
            ext = np.where(tl >= T_POST - 1e-6, np.inf, np.maximum(tl, 0.0))
            per_g[key]['ext'].append(ext)
    gs = np.array(sorted(per_g))
    G, tp, s1 = [], [], []
    for g in gs:
        orn = np.concatenate(per_g[g]['orn']).mean()
        upn = np.concatenate(per_g[g]['upn']).mean()
        upp = np.concatenate(per_g[g]['upn_post']).mean()
        G.append(upn / orn if orn > 0 else np.nan)
        tt, _ = tau_cens(np.concatenate(per_g[g]['ext']), T_POST)
        tp.append(tt)
        s1.append(upp / upn if upn > 0 else np.nan)
    G = np.array(G)
    tp = np.array(tp)
    s1 = np.array(s1, float)
    ltp = np.log(np.where(np.isfinite(tp), tp, 1e9))
    return {
        'DQ_DA': cross(gs, G, 0.29),
        'G_145': cross(gs, G, 1.45),
        'DT_SS': cross(gs, ltp, np.log(3000.0)),
        'tau200': cross(gs, ltp, np.log(200.0)),
        'S1_fail': cross(gs, s1, 0.1),  # first g where the post-removal uPN residual exceeds 10% of driven
        'n_g': int(len(gs)),
    }


def driven_files(set_, model, graph, r):
    pat = re.compile(rf'^{re.escape(model)}_{graph}_o0\.04_r{re.escape(f"{r:g}")}(_ref\d+)?_c\d+\.npz$')
    return sorted(f for f in glob.glob(f'{DRV}/runs/{set_}/*.npz') if pat.match(os.path.basename(f)))


def verdict(mine, lane, total=None):
    if mine is None or lane is None:
        return 'n/a' if (mine is None) != (lane is None) else 'both none'
    rel = mine / lane - 1
    ok = abs(rel) <= 0.02 or (total is not None and total[0] is not None and total[0] <= mine <= total[1])
    return 'pass' if ok else 'FAIL'


def main():
    import sys
    only_driven = '--only_driven' in sys.argv
    res = {'decay': {}, 'driven': {}, 'rule': 'pass if within 2% of the lane value or inside the lane total interval'}
    b = json.load(open(f'{DEC}/boundaries.json'))
    rows = decay_rows()
    R_ALL = sorted({r for (m, gph, r) in rows if m == 'M2' and gph == 'real'})
    if only_driven:
        res['decay'] = json.load(open(OUT))['decay']
    want = [('M2', 'real', r) for r in R_ALL]
    want += [(m, gph, r) for (m, gph, r) in sorted(rows) if (m, gph) in
             (('M2', 'cnull'), ('M2', 'wnull'), ('M2s', 'real'), ('M1', 'real'))]
    for (m, gph, r) in ([] if only_driven else want):
        mine = decay_row(rows[(m, gph, r)])
        key, lane = lane_decay(b, m, gph, r)
        ent = {'mine': mine, 'lane': lane, 'verdict': {}}
        for k in ('S', 'on', 'D'):
            lv = lane[k]['g'] if lane else None
            ent['verdict'][k] = verdict(mine[k], lv, lane[k]['total'] if lane else None)
        res['decay'][key] = ent
        print('decay', key, {k: (mine[k], lane[k]['g'] if lane else None, ent['verdict'][k]) for k in ('S', 'on', 'D')},
              flush=True)

    cur = json.load(open(f'{DRV}/curves.json'))
    sets = [('A1', 'M2', 'real', r) for r in R_ALL]
    for r in (0, 0.25, 0.5, 1, 2, 4, 6):
        sets += [('A2', 'M1', 'real', r), ('A3', 'M2', 'cnull', r), ('A3', 'M2', 'wnull', r), ('A4', 'M2s', 'real', r)]
    for (s, m, gph, r) in sets:
        files = driven_files(s, m, gph, r)
        if not files:
            print('no files', s, m, gph, r)
            continue
        mine = driven_row(files)
        ck = f'{s}|{m}|{gph}|0.04|{r:g}|False'
        lane = cur[ck]['lines'] if ck in cur else {}
        ci = cur[ck]['lines_ci'] if ck in cur else {}
        ent = {'mine': mine, 'lane': {k: lane.get(k) for k in ('DQ_DA', 'DT_SS', 'tau200', 'allowed_hi', 'silent_lost')},
               'files': len(files), 'verdict': {}}
        for k in ('DQ_DA', 'DT_SS', 'tau200'):
            tot = None
            if k in ci and ci[k].get('lo') is not None:
                br = cur[ck]['lines_bracket'].get(k) or [ci[k]['lo'], ci[k]['hi']]
                tot = [min(ci[k]['lo'], br[0]), max(ci[k]['hi'], br[1])]
            ent['verdict'][k] = verdict(mine[k], lane.get(k), tot)
        # allowed top: MODEL2 5b = G in [0.29, 1.45] AND S1 AND S2 (tau_post not used). Mine: first of S1 loss, G = 1.45.
        cands = [x for x in (mine['S1_fail'], mine['G_145']) if x is not None]
        mine['allowed_hi_approx'] = min(cands) if cands else None
        ent['lane']['allowed_hi_by'] = lane.get('allowed_hi_by')
        tot = None
        if 'allowed_hi' in ci and ci['allowed_hi'].get('lo') is not None:
            br = cur[ck]['lines_bracket'].get('allowed_hi') or [ci['allowed_hi']['lo'], ci['allowed_hi']['hi']]
            tot = [min(ci['allowed_hi']['lo'], br[0]), max(ci['allowed_hi']['hi'], br[1])]
        ent['verdict']['allowed_hi'] = verdict(mine['allowed_hi_approx'], lane.get('allowed_hi'), tot)
        res['driven'][ck] = ent
        print('driven', ck, {k: (mine.get(k if k != 'allowed_hi' else 'allowed_hi_approx'), lane.get(k), ent['verdict'][k])
                             for k in ('DQ_DA', 'DT_SS', 'allowed_hi')}, flush=True)

    tally = defaultdict(lambda: defaultdict(int))
    for part in ('decay', 'driven'):
        for ent in res[part].values():
            for k, v in ent['verdict'].items():
                tally[part][f'{k}:{v}'] += 1
    res['tally'] = {p: dict(v) for p, v in tally.items()}
    print(json.dumps(res['tally'], indent=1))
    json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
