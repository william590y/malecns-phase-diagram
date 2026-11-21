"""pd3 criticality C1: event-sparse ("frontier") single-seed avalanche simulator (GPU), exactly swc.py's update rules.

Model (swc.py --norm raw, h = 0): binary units, synchronous dt = 0.2. An active unit turns off with probability 1 - e^{-dt};
a quiescent unit turns on with probability 1 - exp(-tanh(s)_+ dt), s = g sum_j c w_ij sigma_j a_j, sigma = +1 exc, -r inh,
0 modulatory (presynaptic sign); every probability uses the old state (synchronous update). c = the graph's raw normalisation.

Frontier algorithm. The graph is stored by presynaptic neuron (out_ptr, out_tgt, out_coef = c w sigma). B slots run
independent avalanches; the active set is one sorted int64 key list key = slot * N + neuron. One step:
  1. expand the out-edges of every active (neuron, slot) pair (repeat_interleave; chunked to <= --chunk edges);
  2. segment-sum per target key (torch.unique(return_inverse) + index_add_), s = g * sum;
  3. quiescent targets with s > 0 switch on with p = 1 - exp(-tanh(s) dt) (active-set membership by searchsorted on the
     sorted keys); active pairs survive with p = e^{-dt}; all from the old state;
  4. a slot with no active unit ends (extinct); a slot whose age reaches --cap is closed as censored (capped). Freed slots
     are re-seeded at once (slot recycling), unless seeding has stopped (N_av issued or --seed_min wall minutes) or the
     number of out-edges of the current active set exceeds --edge_budget (bounds the per-step cost at high activity; the
     decision depends only on the past, so the sample of seeded avalanches stays i.i.d.).

Definitions (de Candia et al. 2026): one seed neuron at age 0 (it counts as one spike); size S = number of 0->1 transitions
incl. the seed; duration T = number of steps until the step at which the last active unit turned off (x dt = model time);
profile = spikes per step at ages 0..T-1 (the seed at age 0).

Seeds: real / module graphs: uniform over the 366 excitatory neurons of the 526-neuron AL module. cnull graphs: uniform over
excitatory neurons with out-degree > 0 inside the 99% |v|^2 support of the leading mode of B(g_lin) (see the meta: support
size, 1/IPR).

Saved per avalanche (records): s0 (global step at seeding), T (steps; for status 2 = the age reached), status (0 extinct,
1 capped at --cap, 2 still running when observation stopped), s_end (global step at which observation of this avalanche's
epoch ended; -1 = the final step), seed (global neuron id), spikes in AL-L / AL-R / AL-mid / rest (P0 membership), size S,
max active, n_active at ages 5, 10, 20, 35, 50 steps (0 once extinct; for the bootstrap of the growth rate), skewness of
the spike-weighted and the activity-weighted time profile (third standardised moment; POSITIVE =
fast rise, slow decay = de Candia's "left-skewed").
Online aggregates:
  (a) per age a = 0..cap: sums over avalanches of n_active(a), n_active(a)^2, spikes(a), 1[n_active(a) > 0] (extinct = 0).
      E[n(a)] = agg_n[a] / den(a), den(a) = #{avalanches: s0 + a <= s_end} (computed from the records; unbiased because the
      observation window does not depend on an avalanche's own fate);
  (b) log duration bins (edges 5 * 1.15^k steps, up to the cap; extinct avalanches only): the profile on 40 points of t/T
      (point k samples step floor((k + 0.5) / 40 * T)), channels [total, AL-L, AL-R, other, seed lobe, other lobe, majority
      lobe, minority lobe] (lobe channels for AL-seeded avalanches only; majority by total L vs R spikes, ties -> L),
      sum, sum of squares (total), counts;
  (c) exact-duration mean profiles for T in {10, 20, 40, 80, 160, 320, 640} steps (extinct only), channels [AL-L, AL-R,
      other] + sum of squares of the total;
  (d) reservoir sample of --nsamp avalanches (extinct or capped) with T >= --samp_minT steps: per-step spikes [T, 3]
      (AL-L, AL-R, other).
Checkpoint (all of the above, consistent at the checkpoint step) every --ckpt_min minutes; a resubmitted task resumes from
it (records of the interrupted epoch get s_end = the checkpoint step).

Usage: pd3_avalanche.py --graph real --r 0 --x 1.0 [--B 16384 --cap 5000 --nav 2e6] --tag real_r0_x1.00
       pd3_avalanche.py --graph module --r 1 --x 1.0 ...      (526-neuron AL module alone, module_graph.npz; own g_lin)
       pd3_avalanche.py --graph cnull --N 167216 --gseed 0 --r 0 --x 1.0 ...  (pd3_fss_graph.build; own g_lin)
       pd3_avalanche.py --graph complete --K 76 --r 0 --x 1.0 ...  (one-mode control: K all-to-all excitatory units, every
         edge c w = 1/K; its synchronous SIS dynamics is the discrete-time version of the logistic birth-death chain
         b(n) = lam n (1 - n/K), d(n) = alpha n used by the theory; same code path and estimators as the connectome runs)
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd3_common import A_EFF, DT, ROOT, load_cache, c_of, exc_csr_torch, B_mats, perron_t, k_nc, g_lin_exact

ap = argparse.ArgumentParser()
ap.add_argument("--graph", choices=["real", "module", "cnull", "complete"], required=True)
ap.add_argument("--K", type=int, default=76, help="complete: number of units (one-mode control = finite-K MF-DP chain)")
ap.add_argument("--N", type=int, default=167216); ap.add_argument("--gseed", type=int, default=0)
ap.add_argument("--r", type=float, required=True)
ap.add_argument("--x", type=float, default=None, help="g = x * (the graph's own g_lin)")
ap.add_argument("--g", type=float, default=None, help="absolute g (overrides --x)")
ap.add_argument("--B", type=int, default=16384); ap.add_argument("--cap", type=int, default=5000)
ap.add_argument("--nav", type=float, default=2e6)
ap.add_argument("--seed_min", type=float, default=100.0, help="stop seeding after this many wall minutes")
ap.add_argument("--wall_min", type=float, default=110.0, help="stop stepping after this many wall minutes")
ap.add_argument("--ckpt_min", type=float, default=10.0)
ap.add_argument("--chunk", type=float, default=3e7); ap.add_argument("--edge_budget", type=float, default=6e7)
ap.add_argument("--nsamp", type=int, default=2000); ap.add_argument("--samp_minT", type=int, default=50)
ap.add_argument("--rng", type=int, default=0)
ap.add_argument("--tag", required=True)
ap.add_argument("--outdir", default=f"{ROOT}/results/pd3/criticality")
a = ap.parse_args()
dev = "cuda"; t0 = time.time()
log = lambda *x: print(f"[{time.time()-t0:.1f}s]", *x, flush=True)
out_path = f"{a.outdir}/aval_{a.tag}.npz"; part_path = f"{a.outdir}/partial/aval_{a.tag}.partial.npz"
os.makedirs(f"{a.outdir}/partial", exist_ok=True)
if os.path.exists(out_path):
    log("exists, nothing to do:", out_path); sys.exit(0)
gpu = torch.cuda.get_device_name(0)
log("GPU", gpu, "task", a.tag, vars(a))
G_LIN_REAL = 0.16315274988719286          # results/analysis2/phase_complete.json g_lin.real
G_LIN_NULL0 = 0.8286653139685044          # phase_complete.json g_lin.null (swc.py null = cnull N=167216 seed 0)

# ------------------------------------------------------------------------------------------------------------- graph
P0 = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz")["membership"].astype(np.int64)
chan_global = np.where(P0 <= 2, P0, 3)    # 0 AL-L, 1 AL-R, 2 AL-mid, 3 rest
meta = {"graph": a.graph, "r": a.r}
if a.graph == "complete":
    N = a.K; ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij"); m_ = ii != jj
    pre = jj[m_].astype(np.int64); post = ii[m_].astype(np.int64); w = np.ones(len(pre)); c = 1.0 / N
    nsign = np.ones(N, np.int64); esign = nsign[pre]; gid = np.arange(N)
elif a.graph == "module":
    mg = np.load(f"{ROOT}/results/module/module_graph.npz")
    gid = mg["idx"].astype(np.int64); N = len(gid)
    pre = mg["pre"].astype(np.int64); post = mg["post"].astype(np.int64); w = mg["w"].astype(np.float64)
    nsign = mg["sign"].astype(np.int64); esign = nsign[pre]; c = float(mg["c"])
else:
    cache = load_cache()
    if a.graph == "real":
        pre, post, w, sign = cache; N = len(sign); esign = sign[pre]; gid = np.arange(N); c = c_of(post, w, N)
        nsign = sign.astype(np.int64)
    else:
        from pd3_fss_graph import build
        Gr = build("cnull", a.N, a.gseed, cache)
        pre, post, w, esign, c, N = Gr["pre"], Gr["post"], Gr["w"], Gr["esign"].astype(np.int64), Gr["c"], Gr["N"]
        gid = np.arange(N) if Gr["sub"] is None else Gr["sub"].astype(np.int64)
        nsign = np.zeros(N, np.int64)                      # presynaptic sign of each local neuron (from its out-edges)
        nsign[pre] = esign
    del cache
chan = chan_global[gid] if a.graph != "complete" else np.full(N, 3, np.int64)
log(f"graph {a.graph}: N={N} edges={len(pre)} c={c:.6g}")

# own g_lin and (cnull) the leading mode for the seed pool
if a.graph == "real":
    g_lin = G_LIN_REAL; meta["g_lin_source"] = "phase_complete.json g_lin.real"
    pool = gid[(chan <= 1) & (nsign > 0)]
    mod = np.load(f"{ROOT}/results/module/module_graph.npz")["idx"]
    pool = np.intersect1d(pool, mod)
    meta["pool"] = "366 excitatory neurons of the 526-neuron AL module (uniform)"
else:
    Ed = exc_csr_torch(pre, post, w, esign, c, N, dev)
    if a.graph == "cnull" and a.N == 167216 and a.gseed == 0:
        g_lin = G_LIN_NULL0; meta["g_lin_source"] = "phase_complete.json g_lin.null"
        lam, v, _ = perron_t(B_mats(Ed, g_lin)[0])
    else:
        g_lin, v = g_lin_exact(Ed, rtol=1e-6); meta["g_lin_source"] = "computed (pd3_common.g_lin_exact, rtol 1e-6)"
        lam, v, _ = perron_t(B_mats(Ed, g_lin)[0], v)
    _, u, _ = perron_t(B_mats(Ed, g_lin)[1], v)
    K_g, ipr_g = k_nc(u, v)
    meta.update({"R_at_g_lin": lam / A_EFF, "K_nc_glin": K_g, "inv_ipr_glin": ipr_g})
    vv = (v.abs().double().ravel() ** 2).cpu().numpy(); vv /= vv.sum()
    o = np.argsort(-vv); sup = o[: np.searchsorted(np.cumsum(vv[o]), 0.99) + 1]
    del Ed; torch.cuda.empty_cache()
    if a.graph == "module":
        pool = np.where(nsign > 0)[0]
        meta["pool"] = "366 excitatory neurons of the 526-neuron AL module (uniform)"
    elif a.graph == "complete":
        pool = np.arange(N); meta["pool"] = "all K units (uniform)"
    else:
        outdeg_exc = np.bincount(pre[esign > 0], minlength=N)
        cand = np.zeros(N, bool); cand[sup] = True
        pool = np.where(cand & (outdeg_exc > 0))[0]
        meta["pool"] = "excitatory neurons with out-degree > 0 inside the 99% |v|^2 support of the leading mode of B(g_lin)"
    meta["support99_size"] = int(len(sup))
    log(f"g_lin {g_lin:.7g} R {lam/A_EFF:.6f} K_nc {K_g:.4g} 1/IPR {ipr_g:.4g} 99% support {len(sup)}")
pool_local = pool                                           # local ids (real: local = global)
meta.update({"g_lin": g_lin, "N": int(N), "c": c, "n_edges": int(len(pre)), "pool_size": int(len(pool_local))})
g = a.g if a.g is not None else a.x * g_lin
x = g / g_lin
meta.update({"g": g, "x": x})
log(f"g = {g:.6g} (x = {x:.4f}); seed pool {len(pool_local)}")

sig = np.where(esign > 0, 1.0, np.where(esign < 0, -a.r, 0.0))
coef = sig * w * c
keep = coef != 0
o = np.argsort(pre[keep], kind="stable")
pre_k = pre[keep][o]; tgt_k = post[keep][o]; coef_k = coef[keep][o].astype(np.float32)
outdeg = np.bincount(pre_k, minlength=N)
out_ptr = np.r_[0, np.cumsum(outdeg)]
T_ = lambda arr, dt=torch.int64: torch.tensor(arr, device=dev, dtype=dt)
out_ptr_t = T_(out_ptr); out_deg_t = T_(outdeg); out_tgt_t = T_(tgt_k); out_coef_t = T_(coef_k, torch.float32)
chan_t = T_(chan)
ch3_t = torch.where(chan_t >= 2, torch.full_like(chan_t, 2), chan_t)   # 3-channel history: AL-L, AL-R, other (mid + rest)
pool_t = T_(pool_local); gid_t = T_(gid)
meta["n_nonzero_edges"] = int(keep.sum()); meta["pool_outdeg_mean"] = float(outdeg[pool_local].mean())
del pre, post, w, coef, keep, o, pre_k, tgt_k, coef_k
log(f"CSR by presynaptic neuron on GPU: {meta['n_nonzero_edges']} edges; mean out-degree of seeds {meta['pool_outdeg_mean']:.1f}")

# ------------------------------------------------------------------------------------------------------- state
B, CAP = a.B, a.cap
p_off = 1 - np.exp(-DT)
NBIN_EDGES = [5.0]
while NBIN_EDGES[-1] < CAP:
    NBIN_EDGES.append(NBIN_EDGES[-1] * 1.15)
NBIN = len(NBIN_EDGES) - 1
TSET = [10, 20, 40, 80, 160, 320, 640]
NPT = 40
ff = lambda *s: torch.zeros(*s, device=dev, dtype=torch.float64)
agg = {"agg_n": ff(CAP + 1), "agg_n2": ff(CAP + 1), "agg_sp": ff(CAP + 1), "agg_alive": ff(CAP + 1),
       "prof_sum": ff(NBIN, NPT, 8), "prof_sq": ff(NBIN, NPT), "prof_cnt": ff(NBIN), "prof_cnt_al": ff(NBIN),
       "prof_T": ff(NBIN), "prof_S": ff(NBIN),
       "ex_sum": ff(len(TSET), max(TSET), 3), "ex_sq": ff(len(TSET), max(TSET)), "ex_cnt": ff(len(TSET))}
REC_KEYS = ["s0", "T", "status", "s_end", "seed", "sp4", "S", "maxact", "skew_sp", "skew_act", "n_at"]
AGES = [5, 10, 20, 35, 50]
recs = {k: [] for k in REC_KEYS}
samples = []; samp_meta = []          # reservoir
st = {"gstep": 0, "issued": 0, "n_elig": 0, "resumes": 0, "epochs": []}
if os.path.exists(part_path):
    Pp = np.load(part_path, allow_pickle=True)
    for k in agg:
        agg[k] = torch.tensor(Pp[k], device=dev, dtype=torch.float64)
    for k in REC_KEYS:
        recs[k] = [Pp["rec_" + k]]
    s_ck = int(Pp["gstep"])
    se = recs["s_end"][0]; se[se < 0] = s_ck                # interrupted epoch: observation ended at the checkpoint
    st = json.loads(str(Pp["state"])); st["resumes"] += 1; st["epochs"].append(s_ck)
    samp_meta = list(json.loads(str(Pp["samp_meta"])))
    so = Pp["samp_off"]; sd = Pp["samp_data"]
    samples = [sd[so[i]:so[i + 1]] for i in range(len(so) - 1)]
    log(f"RESUMED from checkpoint at gstep {s_ck}: {st['issued']} issued, {sum(len(x) for x in recs['T'])} records")
gen = torch.Generator(device=dev); gen.manual_seed(1_000_003 * a.rng + 7919 * st["resumes"] + 17)
nprng = np.random.default_rng(31 * a.rng + 7 * st["resumes"] + 5)

inuse = torch.zeros(B, dtype=torch.bool, device=dev)
age = torch.zeros(B, dtype=torch.int64, device=dev); s0 = torch.zeros_like(age); seedn = torch.zeros_like(age)
S4 = torch.zeros(B, 4, dtype=torch.int64, device=dev); maxact = torch.zeros_like(age)
msp = ff(B, 4); mact = ff(B, 4)
nck = torch.zeros(B, len(AGES), dtype=torch.int32, device=dev)
lut = torch.full((CAP + 2,), -1, dtype=torch.int64, device=dev); lut[torch.tensor([x_ for x_ in AGES if x_ <= CAP], device=dev)] = torch.arange(len([x_ for x_ in AGES if x_ <= CAP]), device=dev)
hist = torch.zeros(B * (CAP + 1) * 3, dtype=torch.int32, device=dev)
A = torch.zeros(0, dtype=torch.int64, device=dev)
ar4 = torch.arange(4, device=dev)


def moments_skew(m):
    m0 = m[:, 0].clamp(min=1e-300); mu = m[:, 1] / m0; var = m[:, 2] / m0 - mu ** 2
    third = m[:, 3] / m0 - 3 * mu * var - mu ** 3
    sk = third / var.clamp(min=1e-300) ** 1.5
    return torch.where(var > 1e-9 * (mu ** 2 + 1), sk, torch.full_like(sk, float("nan")))


def tpow(t):
    t = t.double()
    return torch.stack([torch.ones_like(t), t, t * t, t * t * t], 1)


def seed_slots(slots):
    global A
    k = slots.numel()
    if k == 0:
        return
    nn = pool_t[torch.randint(len(pool_t), (k,), generator=gen, device=dev)]
    inuse[slots] = True; age[slots] = 0; s0[slots] = st["gstep"]; seedn[slots] = nn
    S4[slots] = 0; S4[slots, chan_t[nn]] = 1; maxact[slots] = 1; nck[slots] = 0
    msp[slots] = tpow(torch.zeros(k, device=dev))                         # spike at t = 0
    mact[slots] = tpow(torch.full((k,), 0.5, device=dev))                 # 1 active over [0, 1)
    hist[(slots * (CAP + 1)) * 3 + ch3_t[nn]] = 1
    agg["agg_n"][0] += k; agg["agg_n2"][0] += k; agg["agg_sp"][0] += k; agg["agg_alive"][0] += k
    A = torch.sort(torch.cat([A, slots * N + nn])).values
    st["issued"] += k


def update(A):
    nA = A.numel()
    if nA == 0:
        return A, A, 0
    slot = torch.div(A, N, rounding_mode="floor"); nr = A - slot * N
    deg = out_deg_t[nr]; cum = torch.cumsum(deg, 0); Etot = int(cum[-1])
    if Etot <= a.chunk:
        cuts = [0, nA]
    else:
        tg = torch.arange(1, int(Etot // a.chunk) + 1, device=dev, dtype=torch.int64) * int(a.chunk)
        cuts = sorted(set([0, nA] + torch.searchsorted(cum, tg).tolist()))
    U, SV = [], []
    for i0, i1 in zip(cuts[:-1], cuts[1:]):
        d = deg[i0:i1]; Ec = int(d.sum())
        if Ec == 0:
            continue
        rep = torch.repeat_interleave(torch.arange(i1 - i0, device=dev), d, output_size=Ec)
        excl = torch.cumsum(d, 0) - d
        eidx = out_ptr_t[nr[i0:i1]][rep] + (torch.arange(Ec, device=dev) - excl[rep])
        tk = slot[i0:i1][rep] * N + out_tgt_t[eidx]
        uk, inv = torch.unique(tk, sorted=True, return_inverse=True)
        s = torch.zeros(uk.numel(), device=dev, dtype=torch.float32).index_add_(0, inv, out_coef_t[eidx])
        U.append(uk); SV.append(s)
        del rep, excl, eidx, tk, inv
    if len(U) > 1:
        uk, inv = torch.unique(torch.cat(U), sorted=True, return_inverse=True)
        s = torch.zeros(uk.numel(), device=dev, dtype=torch.float32).index_add_(0, inv, torch.cat(SV))
    elif len(U) == 1:
        uk, s = U[0], SV[0]
    else:
        uk = torch.zeros(0, dtype=torch.int64, device=dev); s = torch.zeros(0, device=dev)
    pos = s > 0
    uk = uk[pos]; s = s[pos]
    if uk.numel():
        p = torch.searchsorted(A, uk).clamp(max=nA - 1)
        q = A[p] != uk
        uk = uk[q]; s = s[q]
        pon = 1 - torch.exp(-torch.tanh(g * s) * DT)
        new_on = uk[torch.rand(uk.numel(), generator=gen, device=dev) < pon]
    else:
        new_on = uk
    surv = A[torch.rand(nA, generator=gen, device=dev) >= p_off]
    return torch.sort(torch.cat([surv, new_on])).values, new_on, Etot


def close(idx, status):
    """Close the avalanches in slots idx (extinct: status 0, capped: 1, running at stop: 2 -> records only)."""
    if idx.numel() == 0:
        return {}
    Tst = age[idx]
    S = S4[idx].sum(1)
    rec = {"s0": s0[idx], "T": Tst, "status": torch.full_like(Tst, status), "s_end": torch.full_like(Tst, -1),
           "seed": gid_t[seedn[idx]], "sp4": S4[idx], "S": S, "maxact": maxact[idx],
           "skew_sp": moments_skew(msp[idx]).float(), "skew_act": moments_skew(mact[idx]).float(), "n_at": nck[idx]}
    rec = {k: v.cpu().numpy() for k, v in rec.items()}
    if status == 2:
        return rec
    rows = (idx * (CAP + 1)).view(-1, 1)
    if status == 0:
        # (b) rescaled profiles in log duration bins
        m = Tst >= 5
        if m.any():
            ii = idx[m]; TT = Tst[m]; nb = ii.numel()
            b = torch.clamp(torch.floor(torch.log(TT.double() / 5.0) / np.log(1.15)).long(), 0, NBIN - 1)
            kk = torch.floor((torch.arange(NPT, device=dev, dtype=torch.float64) + 0.5).view(1, -1) / NPT * TT.view(-1, 1).double()).long()
            base = (ii * (CAP + 1)).view(-1, 1) + kk                      # [nb, 40]
            h3 = torch.stack([hist[base * 3 + cch] for cch in range(3)], 2).double()   # [nb, 40, 3]
            tot = h3.sum(2)
            sl = chan_t[seedn[ii]]; isal = sl <= 1
            slc = torch.clamp(sl, max=1)
            seedlobe = torch.where(isal.view(-1, 1), torch.gather(h3[:, :, :2], 2, slc.view(-1, 1, 1).expand(nb, NPT, 1)).squeeze(2), torch.zeros_like(tot))
            othlobe = torch.where(isal.view(-1, 1), torch.gather(h3[:, :, :2], 2, (1 - slc).view(-1, 1, 1).expand(nb, NPT, 1)).squeeze(2), torch.zeros_like(tot))
            Lmaj = (S4[ii, 0] >= S4[ii, 1]).view(-1, 1)
            major = torch.where(Lmaj, h3[:, :, 0], h3[:, :, 1]); minor = torch.where(Lmaj, h3[:, :, 1], h3[:, :, 0])
            ch8 = torch.stack([tot, h3[:, :, 0], h3[:, :, 1], h3[:, :, 2], seedlobe, othlobe, major, minor], 2)
            agg["prof_sum"].index_add_(0, b, ch8); agg["prof_sq"].index_add_(0, b, tot * tot)
            agg["prof_cnt"].index_add_(0, b, torch.ones(nb, device=dev, dtype=torch.float64))
            agg["prof_cnt_al"].index_add_(0, b, isal.double())
            agg["prof_T"].index_add_(0, b, TT.double()); agg["prof_S"].index_add_(0, b, S4[ii].sum(1).double())
        # (c) exact durations
        for j, Tx in enumerate(TSET):
            mm = Tst == Tx
            if mm.any():
                ii = idx[mm]
                base = (ii * (CAP + 1)).view(-1, 1) + torch.arange(Tx, device=dev).view(1, -1)
                h3 = torch.stack([hist[base * 3 + cch] for cch in range(3)], 2).double()
                agg["ex_sum"][j, :Tx] += h3.sum(0); agg["ex_sq"][j, :Tx] += (h3.sum(2) ** 2).sum(0)
                agg["ex_cnt"][j] += ii.numel()
    # (d) reservoir sample (extinct or capped, T >= samp_minT)
    el = torch.nonzero(Tst >= a.samp_minT).ravel()
    if el.numel():
        Tel = Tst[el].cpu().numpy(); iel = idx[el]
        take = []
        for q in range(len(Tel)):
            st["n_elig"] += 1; k = st["n_elig"]
            if k <= a.nsamp:
                take.append((q, len(samples))); samples.append(None); samp_meta.append(None)
            else:
                jj = int(nprng.integers(k))
                if jj < a.nsamp:
                    take.append((q, jj))
        for q, pos_ in take:
            sl_ = int(iel[q]); TT = int(Tel[q])
            row = hist[(sl_ * (CAP + 1)) * 3:(sl_ * (CAP + 1) + TT) * 3].view(TT, 3).cpu().numpy().copy()
            samples[pos_] = row
            samp_meta[pos_] = {"T": TT, "status": status, "seed": int(gid[int(seedn[sl_])]),
                               "seed_chan": int(chan[int(seedn[sl_])]), "S": int(S4[sl_].sum()), "s0": int(s0[sl_])}
    # clear history rows
    Tmax = int(Tst.max()) + 1
    base = rows + torch.arange(min(Tmax, CAP + 1), device=dev).view(1, -1)
    for cch in range(3):
        hist[base * 3 + cch] = 0
    return rec


def snapshot(final):
    """Assemble a consistent output at the current step (running slots -> status-2 records)."""
    run_idx = torch.nonzero(inuse).ravel()
    extra = close(run_idx, 2)
    R = {}
    for k in REC_KEYS:
        parts = recs[k] + ([extra[k]] if extra else [])
        R[k] = np.concatenate(parts) if parts else np.zeros(0)
    if extra:
        if final:
            pass                                                  # s_end = -1 = the final step
        else:
            R["s_end"][-len(extra["T"]):] = -1                    # provisional (resolved on resume / final)
    offs = np.r_[0, np.cumsum([len(s) for s in samples if s is not None])].astype(np.int64)
    data = np.concatenate([s for s in samples if s is not None]) if samples else np.zeros((0, 3), np.int32)
    out = {("rec_" + k): v for k, v in R.items()}
    out.update({k: v.cpu().numpy() for k, v in agg.items()})
    out.update({"samp_off": offs, "samp_data": data, "samp_meta": json.dumps([m for m in samp_meta if m is not None]),
                "gstep": st["gstep"], "state": json.dumps(st), "bin_edges": np.array(NBIN_EDGES), "tset": np.array(TSET),
                "dt": DT, "cap": CAP, "B": B, "g": g, "x": x, "g_lin": g_lin, "r": a.r,
                "final_complete": bool(final and run_idx.numel() == 0)})
    return out


# -------------------------------------------------------------------------------------------------------- main loop
step_times = []; edges_tot = 0; Emax = 0; last_ck = time.time(); seeding = True; last_log = time.time()
t_loop = time.time(); Eprev = 0
free = torch.nonzero(~inuse).ravel()
nseed = int(min(len(free), a.nav - st["issued"]))
seed_slots(free[:max(nseed, 0)])
while True:
    ts = time.time()
    st["gstep"] += 1
    A, new_on, Etot = update(A)
    edges_tot += Etot; Emax = max(Emax, Etot); Eprev = Etot
    age[inuse] += 1
    nact = torch.bincount(torch.div(A, N, rounding_mode="floor"), minlength=B)
    on_slot = torch.div(new_on, N, rounding_mode="floor"); on_ch = chan_t[new_on - on_slot * N]
    spk4 = torch.bincount(on_slot * 4 + on_ch, minlength=4 * B).view(B, 4)
    spk = spk4.sum(1)
    S4 += spk4; maxact = torch.maximum(maxact, nact)
    tp = tpow(age)
    msp += spk.double().view(-1, 1) * tp
    mact += nact.double().view(-1, 1) * tpow(age.double() + 0.5)
    # history (per-step spikes, 3 channels) of in-use slots
    if new_on.numel():
        hidx = ((on_slot * (CAP + 1) + age[on_slot]) * 3 + ch3_t[new_on - on_slot * N])
        hist.index_add_(0, hidx, torch.ones_like(hidx, dtype=torch.int32))
    iu = torch.nonzero(inuse).ravel()
    ai = age[iu]; ni = nact[iu].double()
    kk = lut[ai]; sel = kk >= 0
    if sel.any():
        nck[iu[sel], kk[sel]] = nact[iu[sel]].int()
    agg["agg_n"].index_add_(0, ai, ni); agg["agg_n2"].index_add_(0, ai, ni * ni)
    agg["agg_sp"].index_add_(0, ai, spk[iu].double()); agg["agg_alive"].index_add_(0, ai, (ni > 0).double())
    ended = torch.nonzero(inuse & (nact == 0)).ravel()
    capped = torch.nonzero(inuse & (nact > 0) & (age >= CAP)).ravel()
    if capped.numel():
        cm = torch.zeros(B, dtype=torch.bool, device=dev); cm[capped] = True
        A = A[~cm[torch.div(A, N, rounding_mode="floor")]]
    for idx_, stt in ((ended, 0), (capped, 1)):
        rec = close(idx_, stt)
        if rec:
            for k in REC_KEYS:
                recs[k].append(rec[k])
    inuse[ended] = False; inuse[capped] = False
    el_min = (time.time() - t0) / 60
    if seeding and (st["issued"] >= a.nav or el_min >= a.seed_min):
        seeding = False
        log(f"seeding stopped at gstep {st['gstep']}: issued {st['issued']} ({'N_av' if st['issued'] >= a.nav else 'wall'})")
    if seeding and Eprev < a.edge_budget:
        free = torch.nonzero(~inuse).ravel()
        nseed = int(min(free.numel(), a.nav - st["issued"]))
        if nseed > 0:
            seed_slots(free[:nseed])
    step_times.append(time.time() - ts)
    if time.time() - last_log > 60:
        nrec = sum(len(x) for x in recs["T"])
        log(f"gstep {st['gstep']} issued {st['issued']} closed {nrec} in use {int(inuse.sum())} active {A.numel()} "
            f"edges/step {Etot:.3g} ({1e3*np.mean(step_times[-200:]):.1f} ms/step)")
        last_log = time.time()
    if not inuse.any() and not seeding:
        break
    if el_min >= a.wall_min:
        log("wall limit: stopping with running avalanches censored (status 2)")
        break
    if time.time() - last_ck > 60 * a.ckpt_min:
        snap = snapshot(False)
        tmp = part_path + ".tmp.npz"; np.savez(tmp, **snap); os.replace(tmp, part_path)
        log(f"checkpoint at gstep {st['gstep']}"); last_ck = time.time()

snap = snapshot(True)
meta.update({"args": vars(a), "gpu": gpu, "wall_s": time.time() - t0, "loop_s": time.time() - t_loop,
             "steps": st["gstep"], "ms_per_step_mean": 1e3 * float(np.mean(step_times)) if step_times else None,
             "edges_total": float(edges_tot), "edges_max_step": float(Emax), "issued": st["issued"],
             "resumes": st["resumes"], "epochs": st["epochs"], "torch": torch.__version__})
snap["meta"] = json.dumps(meta)
tmp = out_path + ".tmp.npz"
np.savez_compressed(tmp, **snap); os.replace(tmp, out_path)
if os.path.exists(part_path):
    os.remove(part_path)
nrec = len(snap["rec_T"])
log(f"saved {out_path}: {nrec} records ({int((snap['rec_status'] == 0).sum())} extinct, {int((snap['rec_status'] == 1).sum())} capped, "
    f"{int((snap['rec_status'] == 2).sum())} running at stop); {st['gstep']} steps, {meta['ms_per_step_mean']:.2f} ms/step")
