"""pd3 module_theory MT3: neuron-level quenched mean field (QMF) of the FULL network from many starts + continuation (GPU).

Map (exactly theory_qmf.py's, h = 0, raw normalisation c = 1/mean in-strength, dt = 0.2, all 167,216 neurons):
    rho_i <- rho_i e^{-dt} + (1 - rho_i) (1 - exp(-tanh(g sum_j W_ij(r) rho_j)_+ dt)),   W(r) = c (E - r I).
Mixed precision (DEVIATION, for speed: a pure float64 run was 7x slower on an A40 in smoke test 175264): the state rho is
float64; the bulk of the iteration uses a float32 sparse product and p_on (expm1), which stalls at residuals ~1e-8 (float32
rounding cycles; first production attempt 175422, cancelled); columns are therefore iterated in float32 until their residual is
< 1e-7 and then POLISHED with the exact float64 map (<= 1000 steps) until < 1e-9. Every stored residual is the exact float64
map's max_i |F(rho)_i - rho_i|.

Pass 1 (many starts): g at 48 ln points in [0.15, 30] x 32 starts (plan.md 3.3): 1 all active; 2 AL (526) only; 3 AL-L only;
  4 AL-R only; 5 rest only (all neurons outside the AL); 6-25 P0 islands 1-20 alone; 26-29 four random half-subsets of the 34
  P0 modules (rng 2026+i); 30 AL + island 1; 31 uniform 1e-3; 32 AL at 0.05. Started modules at rho = 1, the rest exactly 0.
  <= 25,000 steps; every 500 steps a column whose max_i |d rho_i| (one step) < 1e-7 is frozen (then polished to 1e-9, see
  above); a column whose residual stayed > 1e-3 at 10 consecutive checks after step 10,000 is frozen as oscillating. Per column: mean rho, the 34
  P0 module means, PR/N, AL share, residual, steps, and the growth rate rho(J) of the linearisation (300 power iterations):
     J v = (e^{-dt} - p_on(s)) v + (1 - rho) p'(s) g W v,  p'(s) = dt sech^2(s) e^{-tanh(s) dt} for s > 0, 0 for s < 0, and
     for s == 0 exactly (a silent neuron with no active input) the ONE-SIDED derivative: dt [g W v]_+ (perturbations of a
     silent neuron are >= 0). DEVIATION from the plan's p'(0) = 0: with p'(0) = 0 the silent state would be 'stable' at every g.
     rho(J) = geometric-mean norm growth over the last 100 of 300 iterations (complex leading pairs allowed).
Pass 2 (continuation, two columns run together): up-sweep over 192 ln g points in [0.15, 30] from silence, adding eps = 1e-6
  to every neuron at each new g; down-sweep from the converged all-active state at g = 30 (no eps). At each g: >= 3000 steps
  (plan); if a column's residual is still in (1e-7, 1e-4] (slow convergence, not oscillation) the point is extended up to
  15,000 steps (DEVIATION: avoids spurious up/down gaps from critical slowing next to continuous transitions); early stop
  when residual < 1e-12. rho(J) along both branches. Pass 1 also stores, for the all-active start, the per-neuron temporal
  std over 256 steps (which neurons oscillate).
Oscillation diagnostics (added after the smoke test found no convergence at g >= ~1.4, r = 1): every column / branch point
  that has not converged is iterated 256 more steps while the global mean and the 34 module means are recorded; stored are
  their time averages (used for all comparisons between starts / branches), the relative peak-to-peak amplitude of the
  global mean and its dominant period (FFT, in steps).
Pass 3 (time-step check of any oscillation): the all-active start at the 48 pass-1 g values, map with dt = 0.05 (same rates;
  continuous-time limit) for 2000 model time, same diagnostics. A period-2 oscillation at dt = 0.2 that vanishes at dt = 0.05
  is an artefact of the synchronous update, one that persists is a property of the mean-field ODE.
Usage: python pd3_qmf_ic.py --r R [--smoke]  ->  results/pd3/module_theory/qmf_r<R>.npz
"""
import argparse, json, os, time
import numpy as np
import torch

C = "/MALECNS_CACHE"; ROOT = "/MALECNS_ROOT"
ap = argparse.ArgumentParser()
ap.add_argument("--r", type=float, required=True)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = "cuda"; t0 = time.time(); dt = 0.2; decay = float(np.exp(-dt)); DT = torch.float64; F32 = torch.float32
log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
out_path = a.out or f"{ROOT}/results/pd3/module_theory/qmf_r{a.r:g}{'_smoke' if a.smoke else ''}.npz"

pre = np.load(C + "/pre.npy").astype(np.int64); post = np.load(C + "/post.npy").astype(np.int64)
w = np.load(C + "/weight.npy").astype(np.float64); sign = np.load(C + "/sign.npy"); N = len(sign)
ins = np.bincount(post, weights=w, minlength=N); c = 1.0 / ins[ins > 0].mean()
s = np.where(sign[pre] > 0, 1.0, np.where(sign[pre] < 0, -a.r, 0.0)); keep = s != 0
idx = torch.tensor(np.vstack([post[keep], pre[keep]]), device=dev)
W64 = torch.sparse_coo_tensor(idx, torch.tensor(s[keep] * w[keep] * c, device=dev, dtype=DT), (N, N)).coalesce().to_sparse_csr()
W = W64.to(F32)
del idx, pre, post, w
part = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz", allow_pickle=True)
memb = part["membership"].astype(np.int64); names = list(part["names"]); MOD = len(names)
REST = names.index("rest"); AL = np.isin(memb, [0, 1, 2])
log(f"graph N={N} nnz={int(keep.sum())} r={a.r} c={c:.6g}; {MOD} P0 modules; GPU {torch.cuda.get_device_name()}")
memb_t = torch.tensor(memb, device=dev)
onehot = torch.zeros(N, MOD, device=dev, dtype=DT); onehot[torch.arange(N, device=dev), memb_t] = 1
onehot32 = onehot.float()
msize = onehot.sum(0)
AL_t = torch.tensor(AL, device=dev)

# ---------------- starts ----------------
starts, snames = [], []
def add(nm, v):
    starts.append(np.asarray(v, np.float64)); snames.append(nm)
add("all", np.ones(N)); add("AL", AL * 1.0); add("AL-L", (memb == 0) * 1.0); add("AL-R", (memb == 1) * 1.0)
add("rest_only", (~AL) * 1.0)
for k in range(1, 21):
    add(f"isl{k}", (memb == names.index(f"isl{k}")) * 1.0)
half_sets = []
for i in range(4):
    sel = np.sort(np.random.default_rng(2026 + i).choice(MOD, MOD // 2, replace=False)); half_sets.append(sel.tolist())
    add(f"half{i}", np.isin(memb, sel) * 1.0)
add("AL+isl1", (AL | (memb == names.index("isl1"))) * 1.0)
add("uniform1e-3", np.full(N, 1e-3)); add("AL@0.05", AL * 0.05)
S0 = np.array(starts).T                                    # [N, 32]
G1 = np.geomspace(0.15, 30, 48); MAXST, CHK, TOL, TOL32 = 25000, 500, 1e-9, 1e-7
if a.smoke:
    G1 = np.geomspace(0.15, 30, 4); MAXST = 2000
NS = S0.shape[1]
cols = [(gi, si) for si in range(NS) for gi in range(len(G1))]           # column order: start-major
ncol = len(cols)


def step(X, gv, dt_=dt):
    """one synchronous QMF step; X float64 [N, cols], gv float32 [1, cols]"""
    s_ = torch.sparse.mm(W, X.float()) * gv
    p = -torch.expm1(-torch.tanh(s_.clamp(min=0)) * dt_)
    return X * float(np.exp(-dt_)) + (1 - X) * p.double()


def step64(X, gv):
    """the exact map in float64 (polishing and the true residual)"""
    s_ = torch.sparse.mm(W64, X) * gv.double()
    return X * decay + (1 - X) * (-torch.expm1(-torch.tanh(s_.clamp(min=0)) * dt))


def polish(X, gv, maxst=1000, chk=100):
    """float64 iteration of columns whose float32-phase residual is small; returns X and the TRUE (float64) residual"""
    Xn = step64(X, gv); r_ = (Xn - X).abs().amax(0); X = Xn
    live = torch.where((r_ >= TOL) & (r_ < 1e-5))[0]; st = 0
    while len(live) and st < maxst:
        Xl = X[:, live]; gl = gv[:, live]
        for _ in range(chk - 1):
            Xl = step64(Xl, gl)
        Xn = step64(Xl, gl); rl = (Xn - Xl).abs().amax(0); X[:, live] = Xn; r_[live] = rl; st += chk
        live = live[rl >= TOL]
    return X, r_.cpu().numpy()


def osc_diag(X, gv, nrec=256, dt_=dt):
    """iterate nrec steps recording global and module means: time averages, rel. peak-to-peak amplitude, period (steps)"""
    gm, xm = [], []
    for _ in range(nrec):
        X = step(X, gv, dt_)
        Xf = X.float(); gm.append(Xf.mean(0)); xm.append((onehot32.T @ Xf) / msize.float()[:, None])
    gm = torch.stack(gm).double().cpu().numpy(); xm = torch.stack(xm).double().cpu().numpy()   # [t, cols], [t, MOD, cols]
    avg = gm.mean(0); amp = (gm.max(0) - gm.min(0)) / np.maximum(np.abs(avg), 1e-300)
    f = np.abs(np.fft.rfft(gm - avg, axis=0)); f[0] = 0
    k = f.argmax(0); per = np.where(k > 0, nrec / np.maximum(k, 1), np.inf)
    return X, avg, xm.mean(0).T, amp, per


def summarize(X):
    tot = X.sum(0)
    xm = (onehot.T @ X) / msize[:, None]                   # [MOD, cols]
    pr = (tot ** 2 / (X ** 2).sum(0).clamp(min=1e-300) / N)
    al = X[AL_t].sum(0) / tot.clamp(min=1e-300)
    return (tot / N).cpu().numpy(), xm.T.cpu().numpy(), pr.cpu().numpy(), al.cpu().numpy()


def growth_rate(X, gv, iters=300, seed=0):
    """rho(J) of the one-sided linearisation at the states X (columns), by power iteration."""
    X = X.float(); gv = gv.float()
    s_ = torch.sparse.mm(W, X) * gv
    sp_ = s_.clamp(min=0); th = torch.tanh(sp_)
    p_on = -torch.expm1(-th * dt)
    dp = torch.where(s_ > 0, dt * (1 - th ** 2) * torch.exp(-th * dt), torch.zeros_like(s_))
    zero = s_ == 0
    diag = decay - p_on; fac = (1 - X) * dp
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    v = torch.rand(X.shape, device=dev, dtype=F32, generator=gen)
    v /= v.norm(dim=0, keepdim=True)
    logs = []
    for it in range(iters):
        Wv = torch.sparse.mm(W, v) * gv
        y = diag * v + fac * Wv
        y = torch.where(zero, decay * v + dt * Wv.clamp(min=0), y)   # silent neurons with exactly zero input: one-sided
        nrm = y.norm(dim=0).clamp(min=1e-30)
        if it >= iters - 100:
            logs.append(torch.log(nrm))
        v = y / nrm
    return torch.exp(torch.stack(logs).double().mean(0)).cpu().numpy()


# ---------------- pass 1 ----------------
res = {k: np.full(ncol, np.nan) for k in ["mean", "pr", "al", "resid", "steps", "rhoJ", "tavg_mean", "amp", "period"]}
res["X"] = np.full((ncol, MOD), np.nan); res["tavg_X"] = np.full((ncol, MOD), np.nan)
all_active_final = np.zeros((len(G1), N), np.float16); all_active_std = np.zeros((len(G1), N), np.float16)
order = np.arange(ncol)
for c0 in range(0, ncol, a.chunk):
    cid = order[c0:c0 + a.chunk]
    X = torch.tensor(S0[:, [cols[j][1] for j in cid]], device=dev, dtype=DT)
    gv = torch.tensor(G1[[cols[j][0] for j in cid]], device=dev, dtype=F32)[None, :]
    live = torch.arange(len(cid), device=dev); Xall = X.clone(); resid = np.full(len(cid), np.inf); nst = np.zeros(len(cid))
    hi = np.zeros(len(cid), int)
    st = 0
    while st < MAXST and len(live):
        Xl = Xall[:, live]; gl = gv[:, live]
        for _ in range(CHK - 1):
            Xl = step(Xl, gl)
        Xn = step(Xl, gl)
        r_ = (Xn - Xl).abs().amax(0)
        st += CHK
        Xall[:, live] = Xn
        li = live.cpu().numpy(); rn = r_.cpu().numpy(); resid[li] = rn; nst[li] = st
        hi[li] = np.where(rn > 1e-3, hi[li] + 1, 0)
        done = (r_ < TOL32) | torch.tensor((st >= 10000) & (hi[li] >= 10), device=dev)   # converged / persistently oscillating
        live = live[~done]
    Xall, resid = polish(Xall, gv)                       # float64 polishing; resid is now the exact-map residual
    m_, xm, pr, al = summarize(Xall)
    rj = growth_rate(Xall, gv)
    res["mean"][cid] = m_; res["X"][cid] = xm; res["pr"][cid] = pr; res["al"][cid] = al
    res["tavg_mean"][cid] = m_; res["tavg_X"][cid] = xm; res["amp"][cid] = 0.0; res["period"][cid] = np.nan
    unc = np.where(resid >= TOL)[0]
    if len(unc):
        ut = torch.tensor(unc, device=dev)
        _, avg, xavg, amp, per = osc_diag(Xall[:, ut].clone(), gv[:, ut], 1024)
        res["tavg_mean"][cid[unc]] = avg; res["tavg_X"][cid[unc]] = xavg; res["amp"][cid[unc]] = amp; res["period"][cid[unc]] = per
    res["resid"][cid] = resid; res["steps"][cid] = nst; res["rhoJ"][cid] = rj
    a0 = [jj for jj, j in enumerate(cid) if cols[j][1] == 0]
    for jj in a0:
        all_active_final[cols[cid[jj]][0]] = Xall[:, jj].float().cpu().numpy().astype(np.float16)
    if a0:                                                  # per-neuron temporal std (256 steps), all-active start
        at = torch.tensor(a0, device=dev); Xa = Xall[:, at].clone(); s1 = torch.zeros_like(Xa); s2 = torch.zeros_like(Xa)
        for _ in range(256):
            Xa = step(Xa, gv[:, at]); s1 += Xa; s2 += Xa * Xa
        sd = (s2 / 256 - (s1 / 256) ** 2).clamp(min=0).sqrt().float().cpu().numpy()
        for q, jj in enumerate(a0):
            all_active_std[cols[cid[jj]][0]] = sd[:, q].astype(np.float16)
        del Xa, s1, s2
    log(f"pass1 chunk {c0//a.chunk+1}/{(ncol+a.chunk-1)//a.chunk}: steps<= {st}, unconverged {int((resid >= TOL).sum())}, "
        f"max resid {resid.max():.2e}")
    del X, Xall; torch.cuda.empty_cache()
shape = (NS, len(G1))
p1 = {k: (v.reshape(shape) if v.ndim == 1 else v.reshape(shape + (MOD,))) for k, v in res.items()}
log("pass1 all-start mean rho at g=" + " ".join(f"{g:.3g}" for g in G1[::8]) + " : " +
    " ".join(f"{v:.3g}" for v in p1["mean"][0, ::8]))

# ---------------- pass 2: continuation ----------------
G2 = np.geomspace(0.15, 30, 192); MIN2, MAX2 = 3000, 15000
if a.smoke:
    G2 = np.geomspace(0.15, 30, 12); MIN2, MAX2 = 300, 600
nG = len(G2)
# down-sweep start: all-active state at g = 30
Xd = torch.ones(N, 1, device=dev, dtype=DT); gtop = torch.tensor([[G2[-1]]], device=dev, dtype=F32)
rd = np.inf
for st in range(1, 25001):
    Xn = step(Xd, gtop)
    if st % 100 == 0:
        rd = float((Xn - Xd).abs().max())
        if rd < 1e-12:
            Xd = Xn; break
    Xd = Xn
log(f"down-sweep start: {st} steps, last residual {rd:.2e}")
X2 = torch.cat([torch.zeros(N, 1, device=dev, dtype=DT), Xd], 1)        # col 0 up, col 1 down
up_states = torch.zeros(N, nG, device=dev, dtype=F32); dn_states = torch.zeros(N, nG, device=dev, dtype=F32)
r2 = {k: np.full((2, nG), np.nan) for k in ["mean", "pr", "al", "resid", "steps", "tavg_mean", "amp", "period"]}
r2["X"] = np.full((2, nG, MOD), np.nan); r2["tavg_X"] = np.full((2, nG, MOD), np.nan)
for i in range(nG):
    gv = torch.tensor([[G2[i], G2[nG - 1 - i]]], device=dev, dtype=F32)
    X2[:, 0] += 1e-6; X2[:, 0].clamp_(max=1.0)
    st = 0; rr = np.array([np.inf, np.inf])
    while st < MAX2:
        for _ in range(99):
            X2 = step(X2, gv)
        Xn = step(X2, gv); rr = (Xn - X2).abs().amax(0).cpu().numpy(); X2 = Xn; st += 100
        if st >= MIN2 and ((rr < TOL32) | (rr > 1e-4)).all():
            break                                           # converged (float32 phase), or every unconverged column oscillates
    X2, rr = polish(X2, gv)
    m_, xm, pr, al = summarize(X2)
    tm, txm, amp, per = m_.copy(), xm.copy(), np.zeros(2), np.full(2, np.nan)
    if (rr >= 1e-9).any():
        X2, avg, xavg, amp_, per_ = osc_diag(X2, gv)
        for col in range(2):
            if rr[col] >= 1e-9:
                tm[col], txm[col], amp[col], per[col] = avg[col], xavg[col], amp_[col], per_[col]
    for col, gi in [(0, i), (1, nG - 1 - i)]:
        r2["mean"][col, gi] = m_[col]; r2["pr"][col, gi] = pr[col]; r2["al"][col, gi] = al[col]
        r2["resid"][col, gi] = rr[col]; r2["steps"][col, gi] = st; r2["X"][col, gi] = xm[col]
        r2["tavg_mean"][col, gi] = tm[col]; r2["tavg_X"][col, gi] = txm[col]; r2["amp"][col, gi] = amp[col]; r2["period"][col, gi] = per[col]
    up_states[:, i] = X2[:, 0].float(); dn_states[:, nG - 1 - i] = X2[:, 1].float()
    if i % 16 == 0 or i == nG - 1:
        log(f"pass2 {i+1}/{nG}: up g={G2[i]:.4g} mean={m_[0]:.4g} res={rr[0]:.1e} | down g={G2[nG-1-i]:.4g} mean={m_[1]:.4g} "
            f"res={rr[1]:.1e} steps={st}")
rj_up = growth_rate(up_states, torch.tensor(G2[None, :], device=dev, dtype=F32))
rj_dn = growth_rate(dn_states, torch.tensor(G2[None, :], device=dev, dtype=F32))
log("growth rates along branches done")
del up_states, dn_states; torch.cuda.empty_cache()

# ---------------- pass 3: dt = 0.05 check (all-active start) ----------------
DT3 = 0.05; ST3 = int(2000 / DT3) if not a.smoke else 4000
X3 = torch.ones(N, len(G1), device=dev, dtype=DT); g3 = torch.tensor(G1[None, :], device=dev, dtype=F32)
for st in range(ST3 - 1):
    X3 = step(X3, g3, DT3)
Xn = step(X3, g3, DT3); res3 = (Xn - X3).abs().amax(0).cpu().numpy(); X3 = Xn
m3, xm3, pr3, al3 = summarize(X3)
X3, avg3, xavg3, amp3, per3 = osc_diag(X3, g3, 1024, DT3)
log("pass3 (dt 0.05) resid: " + " ".join(f"{v:.1e}" for v in res3[::6]) + " | amp: " + " ".join(f"{v:.2g}" for v in amp3[::6]))
meta = dict(r=a.r, N=N, c=c, dt=dt, h=0.0, starts=snames, half_sets=half_sets, modules=names, maxsteps=MAXST, check=CHK,
            tol=TOL, min2=MIN2, max2=MAX2, eps_up=1e-6, smoke=a.smoke, gpu=torch.cuda.get_device_name(),
            seconds=time.time() - t0)
tmp = out_path + ".tmp.npz"
np.savez_compressed(tmp, g1=G1, g2=G2, starts=np.array(snames), modules=np.array(names),
                    p1_mean=p1["mean"], p1_X=p1["X"], p1_pr=p1["pr"], p1_al=p1["al"], p1_resid=p1["resid"],
                    p1_steps=p1["steps"], p1_rhoJ=p1["rhoJ"], all_active_final=all_active_final, all_active_std=all_active_std,
                    up_mean=r2["mean"][0], dn_mean=r2["mean"][1], up_X=r2["X"][0], dn_X=r2["X"][1],
                    up_pr=r2["pr"][0], dn_pr=r2["pr"][1], up_al=r2["al"][0], dn_al=r2["al"][1],
                    up_resid=r2["resid"][0], dn_resid=r2["resid"][1], up_steps=r2["steps"][0], dn_steps=r2["steps"][1],
                    up_rhoJ=rj_up, dn_rhoJ=rj_dn,
                    p1_tavg_mean=p1["tavg_mean"], p1_tavg_X=p1["tavg_X"], p1_amp=p1["amp"], p1_period=p1["period"],
                    up_tavg_mean=r2["tavg_mean"][0], dn_tavg_mean=r2["tavg_mean"][1], up_tavg_X=r2["tavg_X"][0],
                    dn_tavg_X=r2["tavg_X"][1], up_amp=r2["amp"][0], dn_amp=r2["amp"][1], up_period=r2["period"][0],
                    dn_period=r2["period"][1],
                    p3_dt=DT3, p3_mean=m3, p3_X=xm3, p3_pr=pr3, p3_resid=res3, p3_tavg_mean=avg3, p3_tavg_X=xavg3,
                    p3_amp=amp3, p3_period=per3, meta=json.dumps(meta))
os.replace(tmp, out_path)
log("saved", out_path)
