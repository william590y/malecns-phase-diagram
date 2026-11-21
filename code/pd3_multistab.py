"""pd3 program 'multistability' (plan.md section 5): coexisting states, broken L/R symmetry, hysteresis -- STOCHASTIC model.

Simulator: the dense synchronous update of swc.py, unchanged rules (raw normalisation c = 1 / mean in-strength = 1/742.7,
dt = 0.2, sigma = +1 excitatory / -r inhibitory / 0 modulatory, active -> silent with prob 1 - e^{-dt}, silent -> active with
prob 1 - exp(-tanh(s)_+ dt), s_i = g sum_j c w_ij sigma_j a_j + h). Differences from swc.py, none of which changes the rules:
  * every column (replica) carries its own (g, r, h), so up- and down-sweeps (or several starts / g values) share one sparse
    product; when all columns of a segment share one r the combined matrix W(r) = c (E - r I) is used (as swc.py), otherwise
    E and I are multiplied separately (r-sweeps);
  * p_on is computed as -expm1(-tanh(s)_+ dt) (swc.py: 1 - exp(...)); identical in exact arithmetic, more accurate in float32
    for tiny drives (h = 1e-5 gives p_on ~ 2e-6, where 1 - exp loses ~1% to rounding);
  * starts are sets of P0 modules (results/pd3/modules/partition_P0.npz: AL-L, AL-R, AL-mid, islands 1-30, rest; 34 modules)
    fully active, the rest silent; also Bernoulli(p) starts (QMF fixed points) and bit-packed snapshots of earlier runs;
  * a run is a SCHEDULE of segments (dwells); the state is carried over from one segment to the next.
Recording (per run):
  modc   int32 [n_rec, 34, B]  per-module active counts every `rec_every` steps (5 steps = 1 model time; record 0 = t = 0);
  fine   int16 [fine_steps + 1, 3, B]  AL-L / AL-R / AL-mid counts at EVERY step for the first fine_steps steps (al_sym mode);
  per segment (occupancy accumulated over the last `occ_frac` of the segment, per replica, float32 [N, B] on the GPU):
     rho [seg, B] (mean fraction active), Xk [seg, 34, B] (mean fraction active per module), PR_rep [seg, B]
     ((sum occ)^2 / sum occ^2 / N of the replica's own occupancy), and per GROUP of columns: occ_mean float16 [seg, G, N]
     (replica-mean occupancy), PR_mean [seg, G] (PR of the replica-mean occupancy, the phase_complete.md definition) and
     PR_boot [seg, G, nb] (same, for nb multinomial replica resamples whose weights wb [G, Bg, nb] are fixed for the whole run
     and stored, so offline bootstraps of rho etc. can use the same resamples);
     snap uint8 [seg, ceil(N/8), B]: np.packbits of the final state of the segment (axis 0), for post-hoc partitions;
  ext_step / ext_al_step int32 [B]: first step with no active neuron in the network / in the 526-neuron AL module (-1: never);
  optional region (M3b): per-replica occupancy of a neuron subset over model-time windows (float16 [W, n_reg, B]) and the
     projections z_p = (x_block - rho_a) . d_p / |d_p|^2 of block means (block = `blk` model time) on QMF difference vectors
     d_p = rho_b - rho_a (float32 [n_blk, P, B]).
All outputs are written atomically (tmp file + os.replace); a task whose output exists is skipped (resubmission reruns only
missing outputs).

Modes (one per program part; args files written by pd3_multistab_args.py):
  validate --r 1 --g ...          all-active start, B = 64 per g, T = 3000 (one segment, per-column g)
  alsym    --r R --g ...          M1: starts AL-L / AL-R / AL (both) / all, B = 64 each, T = 1000, h = 0, one g at a time
  sweep    --r R --h H --D D      M2a: 40 ln g in [0.15, 30]; cols 0..B-1 up from silence, B..2B-1 down from all-active at g = 30
           [--down_only]          (h = 0 control: down branch only)
  rsweep   --g G --h H --D D      M2c: 25 r in [0, 6]; cols 0..B-1: r 0 -> 6 -> 0 from all-active at r = 0; cols B..2B-1:
                                  r 6 -> 0 -> 6 from all-active at r = 6 (extension)
  ic       --r R --g G            M3: 13 starts x B = 32, T = 2000, h = 0
  icq      --cells r:g,r:g,...    M3b: at QMF multistable cells, the MT3 starts named in qmf_multi.json (B = 64) and Bernoulli
                                  samples of the QMF fixed points themselves (B = 32), T = 5000, h = 0, region projections
  m2b      --sweep FILE --seg K --T T   M2b: 64 replicas from the up-branch end-state and 64 from the down-branch end-state
                                  of segment K of an M2a sweep, at that segment's g (r, h from the sweep)
"""
import argparse, json, os, sys, time
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False                 # module counts / occupancy products in exact float32

CACHE = "/MALECNS_CACHE"
ROOT = "/MALECNS_ROOT"
OUTDIR = f"{ROOT}/results/pd3/multistab"
DT = 0.2
P_OFF = float(1 - np.exp(-DT))
T0 = time.time()


def log(*x):
    print(f"[{time.time() - T0:.0f}s]", *x, flush=True)


# ------------------------------------------------------------------------------------------------------------- network
class Net:
    def __init__(self, dev="cuda"):
        self.dev = dev
        pre = np.load(CACHE + "/pre.npy").astype(np.int64); post = np.load(CACHE + "/post.npy").astype(np.int64)
        w = np.load(CACHE + "/weight.npy").astype(np.float64); sign = np.load(CACHE + "/sign.npy")
        self.N = N = len(sign)
        ins = np.bincount(post, weights=w, minlength=N)
        self.c = float(1.0 / ins[ins > 0].mean())
        self.pre, self.post, self.w, self.psign = pre, post, w, sign[pre]
        self._W, self._EI = {}, None
        part = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz", allow_pickle=True)
        self.memb = part["membership"].astype(np.int64)
        self.names = [str(x) for x in part["names"]]
        self.M = len(self.names)
        self.size = np.bincount(self.memb, minlength=self.M)
        oh = torch.zeros(self.M, N, device=dev)
        oh[torch.tensor(self.memb, device=dev), torch.arange(N, device=dev)] = 1.0
        self.onehotT = oh                                          # [34, N] float32 (exact 0/1)
        self.AL = np.isin(self.memb, [0, 1, 2])
        self.AL_idx_t = torch.tensor(np.where(self.AL)[0], device=dev)
        al_mod = np.load(f"{ROOT}/results/module/module_graph.npz")["idx"]
        assert set(al_mod.tolist()) == set(np.where(self.AL)[0].tolist()), "P0 AL modules != module_graph idx"
        log(f"net N={N} edges={len(pre)} c={self.c:.6g} (1/c={1 / self.c:.2f}); P0 modules {self.M}: "
            f"AL-L {self.size[0]} AL-R {self.size[1]} AL-mid {self.size[2]} rest {self.size[-1]}")

    def _csr(self, coef):
        keep = coef != 0
        idx = torch.tensor(np.vstack([self.post[keep], self.pre[keep]]), device=self.dev)
        M = torch.sparse_coo_tensor(idx, torch.tensor(coef[keep], device=self.dev, dtype=torch.float32),
                                    (self.N, self.N)).coalesce().to_sparse_csr()
        del idx
        return M

    def W(self, r):
        """combined matrix c (E - r I) exactly as swc.py builds it (float32 coefficients sigma * w * c)."""
        r = float(r)
        if r not in self._W:
            if len(self._W) >= 2:
                self._W.clear(); torch.cuda.empty_cache()
            coef = np.where(self.psign > 0, 1.0, np.where(self.psign < 0, -r, 0.0)) * self.w * self.c
            self._W[r] = self._csr(coef)
            log(f"built W(r={r:g}) nnz={self._W[r]._nnz()}")
        return self._W[r]

    def EI(self):
        if self._EI is None:
            self._W.clear(); torch.cuda.empty_cache()
            E = self._csr(np.where(self.psign > 0, 1.0, 0.0) * self.w * self.c)
            I = self._csr(np.where(self.psign < 0, 1.0, 0.0) * self.w * self.c)
            self._EI = (E, I)
            log(f"built split E (nnz {E._nnz()}) / I (nnz {I._nnz()})")
        return self._EI


# -------------------------------------------------------------------------------------------------------------- starts
def half_set(net, i):
    """MT3's random half-subsets of the 34 P0 modules (pd3_qmf_ic.py: rng 2026 + i)."""
    return np.sort(np.random.default_rng(2026 + i).choice(net.M, net.M // 2, replace=False))


def start_prob(net, spec):
    """activation probability vector [N] for a start spec (deterministic 0/1 for module sets)."""
    m, nm = net.memb, net.names
    isl = lambda k: m == nm.index(f"isl{k}")
    if spec == "all":
        return np.ones(net.N, np.float32)
    if spec == "silent":
        return np.zeros(net.N, np.float32)
    if spec == "AL":
        return net.AL.astype(np.float32)
    if spec in ("AL-L", "AL-R", "AL-mid", "rest"):
        return (m == nm.index(spec)).astype(np.float32)
    if spec == "rest_only":                                   # MT3 definition: every neuron outside the 526 AL module
        return (~net.AL).astype(np.float32)
    if spec.startswith("isl") and spec[3:].isdigit():
        return isl(int(spec[3:])).astype(np.float32)
    if spec.startswith("half") and spec[4:].isdigit():
        return np.isin(m, half_set(net, int(spec[4:]))).astype(np.float32)
    if spec == "AL+isl1":
        return (net.AL | isl(1)).astype(np.float32)
    if spec == "AL+isl1-5":
        return (net.AL | np.isin(m, [nm.index(f"isl{k}") for k in range(1, 6)])).astype(np.float32)
    if spec.startswith("isl1-5"):
        return np.isin(m, [nm.index(f"isl{k}") for k in range(1, 6)]).astype(np.float32)
    if spec.startswith("uniform"):
        return np.full(net.N, float(spec.split("@")[1]), np.float32)
    if spec.startswith("AL@"):
        return (net.AL * float(spec[3:])).astype(np.float32)
    if spec.startswith("fp:"):                                 # QMF fixed point (qmf_multi_vectors.npz key)
        v = np.load(f"{ROOT}/results/pd3/module_theory/qmf_multi_vectors.npz")[spec[3:]]
        return np.clip(v.astype(np.float32), 0, 1)
    raise ValueError(spec)


def build_X0(net, col_specs, seed):
    """col_specs: list (len B) of start specs or ('snap', file, seg, col). Returns float32 [N, B] on the GPU."""
    B = len(col_specs)
    X = torch.zeros(net.N, B, device=net.dev)
    g = torch.Generator(device=net.dev); g.manual_seed(seed + 7919)
    cache = {}
    for j, sp in enumerate(col_specs):
        if isinstance(sp, tuple) and sp[0] == "snap":
            _, f, seg, col = sp
            key = (f, seg)
            if key not in cache:
                cache[key] = np.load(f)["snap"][seg]
            bits = np.unpackbits(cache[key][:, col], count=net.N).astype(np.float32)
            X[:, j] = torch.tensor(bits, device=net.dev)
            continue
        if sp not in cache:
            cache[sp] = torch.tensor(start_prob(net, sp), device=net.dev)
        p = cache[sp]
        if bool(((p == 0) | (p == 1)).all()):
            X[:, j] = p
        else:
            X[:, j] = (torch.rand(net.N, device=net.dev, generator=g) < p).float()
    return X


# ----------------------------------------------------------------------------------------------------------- simulator
def simulate(net, col_specs, groups, segments, seed, rec_every=5, fine_steps=0, snapshots=True, store_occ_mean=True,
             region=None, nb=400, occ_frac=0.5):
    """col_specs: per-column start; groups: list of (name, list of cols); segments: list of dicts with 'steps' and per-column
    numpy arrays 'g', 'r', 'h' (len B). region (optional): dict(idx [n_reg] int64, windows [(t0, t1) model time],
    blk (model time), V [n_reg, P] float32, off [P] float32)."""
    dev = net.dev; B = len(col_specs); M = net.M
    torch.manual_seed(seed); np_rng = np.random.default_rng(seed)
    X = build_X0(net, col_specs, seed)
    tot_steps = int(sum(s["steps"] for s in segments))
    n_rec = tot_steps // rec_every + 1
    modc = torch.zeros(n_rec, M, B, dtype=torch.int32, device=dev)
    modc[0] = torch.mm(net.onehotT, X).round().int()
    fine = torch.zeros(fine_steps + 1, 3, B, dtype=torch.int16, device=dev) if fine_steps else None
    if fine is not None:
        fine[0] = torch.mm(net.onehotT[:3], X).round().short()
    ext = torch.full((B,), -1, dtype=torch.int32, device=dev)
    ext_al = torch.full((B,), -1, dtype=torch.int32, device=dev)
    # replica resample weights per group (fixed for the run)
    wb = []
    for gname, cols in groups:
        cnt = np_rng.multinomial(len(cols), np.full(len(cols), 1.0 / len(cols)), size=nb).T.astype(np.float32)  # [Bg, nb]
        wb.append(cnt)
    wb_t = [torch.tensor(x, device=dev) for x in wb]
    gcols_t = [torch.tensor(cols, device=dev) for _, cols in groups]
    S = len(segments); G = len(groups)
    out = {k: [] for k in ["rho", "Xk", "PR_rep", "PR_mean", "PR_boot", "occ_mean", "snap", "seg_t_end"]}
    if region is not None:
        ridx = torch.tensor(region["idx"], device=dev)
        Vt = torch.tensor(region["V"], device=dev); offt = torch.tensor(region["off"], device=dev)
        wins = region["windows"]; blk_steps = int(round(region["blk"] / DT))
        racc = torch.zeros(len(wins), len(region["idx"]), B, device=dev); rcnt = [0] * len(wins)
        bacc = torch.zeros(len(region["idx"]), B, device=dev); bcount = 0
        proj = torch.zeros(tot_steps // blk_steps, Vt.shape[1], B, device=dev); pk = 0
    gs = 0                                                  # global step
    hzero_rest = [all(float(np.max(np.abs(s["h"]))) == 0 for s in segments[k:]) for k in range(S)]
    dead_all = False
    for si, seg in enumerate(segments):
        steps = int(seg["steps"]); occ_start = steps - int(round(steps * occ_frac))
        gcol = torch.tensor(np.asarray(seg["g"], np.float32), device=dev)[None, :]
        hv = np.asarray(seg["h"], np.float32); hcol = torch.tensor(hv, device=dev)[None, :] if np.any(hv != 0) else None
        rv = np.asarray(seg["r"], np.float64)
        if np.all(rv == rv[0]):
            Wr = net.W(rv[0]); split = False
        else:
            E, I = net.EI(); rcol = torch.tensor(rv.astype(np.float32), device=dev)[None, :]; split = True
        occ = torch.zeros(net.N, B, device=dev); n_occ = steps - occ_start
        tseg = time.time()
        for s in range(steps):
            if dead_all:                                    # h = 0 everywhere from here on and every column absorbed
                gs += steps - s
                break
            if split:
                inp = torch.sparse.mm(E, X); inp.sub_(torch.sparse.mm(I, X) * rcol)
            else:
                inp = torch.sparse.mm(Wr, X)
            inp.mul_(gcol)
            if hcol is not None:
                inp.add_(hcol)
            p_on = torch.expm1(torch.tanh(inp.clamp_(min=0)).mul_(-DT)).neg_()
            u = torch.rand_like(X)
            X = torch.where(X > 0, (u >= P_OFF), (u < p_on)).float()
            gs += 1
            if s >= occ_start:
                occ.add_(X)
            n = X.sum(0)
            ext = torch.where((n == 0) & (ext < 0), torch.full_like(ext, gs), ext)
            nal = X.index_select(0, net.AL_idx_t).sum(0)
            ext_al = torch.where((nal == 0) & (ext_al < 0), torch.full_like(ext_al, gs), ext_al)
            if gs % rec_every == 0:
                modc[gs // rec_every] = torch.mm(net.onehotT, X).round().int()
            if fine is not None and gs <= fine_steps:
                fine[gs] = torch.mm(net.onehotT[:3], X).round().short()
            if region is not None:
                xr = X.index_select(0, ridx)
                tmod = gs * DT
                for wi, (a0, a1) in enumerate(wins):
                    if a0 < tmod <= a1:
                        racc[wi].add_(xr); rcnt[wi] += 1
                bacc.add_(xr); bcount += 1
                if bcount == blk_steps:
                    if pk < proj.shape[0]:
                        proj[pk] = torch.mm(Vt.T, bacc / bcount) - offt[:, None]; pk += 1
                    bacc.zero_(); bcount = 0
            if hzero_rest[si] and s % 200 == 199 and not bool((ext < 0).any()):
                dead_all = True
        # ---- segment summary
        occm = occ / max(n_occ, 1)
        tot = occm.sum(0)
        out["rho"].append((tot / net.N).cpu().numpy())
        out["Xk"].append((torch.mm(net.onehotT, occm) / torch.tensor(net.size, device=dev, dtype=torch.float32)[:, None]).cpu().numpy())
        sq = (occm * occm).sum(0)
        out["PR_rep"].append(torch.where(sq > 0, tot * tot / sq.clamp(min=1e-30) / net.N, torch.zeros_like(tot)).cpu().numpy())
        prm, prb, om = [], [], []
        for gi in range(G):
            sub = occm.index_select(1, gcols_t[gi])                  # [N, Bg]
            mean = sub.mean(1)
            s1 = float(mean.sum()); s2 = float((mean * mean).sum())
            prm.append(s1 * s1 / s2 / net.N if s2 > 0 else 0.0)
            bm = torch.mm(sub, wb_t[gi]) / len(groups[gi][1])          # [N, nb]
            b1 = bm.sum(0); b2 = (bm * bm).sum(0)
            prb.append(torch.where(b2 > 0, b1 * b1 / b2.clamp(min=1e-30) / net.N, torch.zeros_like(b1)).cpu().numpy())
            if store_occ_mean:
                om.append(mean.half().cpu().numpy())
            del bm
        out["PR_mean"].append(np.array(prm)); out["PR_boot"].append(np.array(prb))
        if store_occ_mean:
            out["occ_mean"].append(np.array(om))
        if snapshots:
            out["snap"].append(np.packbits(X.bool().cpu().numpy(), axis=0))
        out["seg_t_end"].append(gs * DT)
        del occ, occm
        rmean = out["rho"][-1]
        log(f"seg {si + 1}/{S}: g[{float(gcol.min()):.4g}..{float(gcol.max()):.4g}] r[{rv.min():g}..{rv.max():g}] "
            f"h[{hv.min():g}..{hv.max():g}] {steps} steps {time.time() - tseg:.1f}s "
            f"({1e3 * (time.time() - tseg) / max(steps, 1):.2f} ms/step); "
            + " ".join(f"{gn}:rho={rmean[c].mean():.3g},alive={float((ext[c] < 0).float().mean()):.2f}"
                       for (gn, c), _ in zip(groups, range(G))))
    res = {"modc": modc.cpu().numpy(), "ext_step": ext.cpu().numpy(), "ext_al_step": ext_al.cpu().numpy(),
           "rho": np.array(out["rho"], np.float32), "Xk": np.array(out["Xk"], np.float32),
           "PR_rep": np.array(out["PR_rep"], np.float32), "PR_mean": np.array(out["PR_mean"], np.float32),
           "PR_boot": np.array(out["PR_boot"], np.float32), "wb": _pad_wb(wb),
           "seg_t_end": np.array(out["seg_t_end"]), "rec_dt": rec_every * DT}
    if store_occ_mean:
        res["occ_mean"] = np.array(out["occ_mean"], np.float16)
    if snapshots:
        res["snap"] = np.array(out["snap"], np.uint8)
    if fine is not None:
        res["fine"] = fine.cpu().numpy()
    if region is not None:
        res["reg_occ"] = np.stack([(racc[k] / max(rcnt[k], 1)).half().cpu().numpy() for k in range(len(wins))])
        res["reg_idx"] = np.asarray(region["idx"]); res["reg_windows"] = np.array(wins, float)
        res["proj"] = proj[:pk].cpu().numpy(); res["proj_blk"] = region["blk"]
    return res


def _pad_wb(wb):
    """replica resample weights per group, zero-padded to the largest group: float32 [G, max Bg, nb]."""
    mx = max(x.shape[0] for x in wb)
    out = np.zeros((len(wb), mx, wb[0].shape[1]), np.float32)
    for i, x in enumerate(wb):
        out[i, :x.shape[0]] = x
    return out


def save(path, res, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, meta=json.dumps(meta), **res)
    os.replace(tmp, path)
    log(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def seg_list(B, n, steps, g, r, h):
    """helper: n segments with per-column arrays from callables of the segment index."""
    return [{"steps": steps, "g": np.asarray(g(k), np.float64) * np.ones(B), "r": np.asarray(r(k), np.float64) * np.ones(B),
             "h": np.asarray(h(k), np.float64) * np.ones(B)} for k in range(n)]


# ----------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "alsym", "sweep", "rsweep", "ic", "icq", "m2b"])
    ap.add_argument("--r", type=float, default=None)
    ap.add_argument("--g", type=float, nargs="*", default=None)
    ap.add_argument("--h", type=float, default=0.0)
    ap.add_argument("--D", type=float, default=200.0)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--down_only", action="store_true")
    ap.add_argument("--force_split", action="store_true", help="validate: use the split E/I product (r-sweep code path)")
    ap.add_argument("--cells", default=None, help="icq: comma-separated r:g list (QMF multistable cells)")
    ap.add_argument("--sweep", default=None); ap.add_argument("--seg", type=int, default=None)
    ap.add_argument("--rsweep", default=None, help="m2b on an r-sweep file: --seg K = index of r in its r grid (fwd 0->6->0 run)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    meta = {"args": vars(a), "dt": DT, "argv": sys.argv}
    os.makedirs(OUTDIR, exist_ok=True)

    def done(p):
        if os.path.exists(p) and not a.force:
            log(f"output exists, skipping: {p}"); return True
        return False

    if a.mode == "validate":
        r = 1.0 if a.r is None else a.r; G = a.g or [0.30, 0.344, 0.345, 0.39, 0.40, 0.405]; B = a.B or 64
        T = a.T or 3000.0
        out = a.out or f"{OUTDIR}/validate_r{r:g}{'_split' if a.force_split else ''}.npz"
        if done(out): return
        net = Net(); log(f"GPU {torch.cuda.get_device_name()}")
        cols = ["all"] * (B * len(G)); gv = np.repeat(G, B)
        rv = np.full(len(cols), r)
        if a.force_split:                                   # force the split path with a tiny per-column r perturbation of 0
            net.EI()
        seg = [{"steps": int(round(T / DT)), "g": gv, "r": rv, "h": np.zeros(len(cols))}]
        if a.force_split:
            seg[0]["r"] = rv.copy(); seg[0]["r"][0] = r + 1e-12        # not all-equal -> split product (r differs by 1e-12)
        groups = [(f"g{g:g}", list(range(i * B, (i + 1) * B))) for i, g in enumerate(G)]
        res = simulate(net, cols, groups, seg, a.seed, snapshots=False, store_occ_mean=True)
        meta.update(mode="validate", r=r, g=G, B=B, T=T, gpu=torch.cuda.get_device_name(), seconds=time.time() - T0)
        save(out, res, meta)

    elif a.mode == "alsym":
        r = a.r; G = a.g; B = a.B or 64; T = a.T or 1000.0
        out = a.out or f"{OUTDIR}/al_sym_r{r:g}.npz"
        if done(out): return
        net = Net(); log(f"GPU {torch.cuda.get_device_name()}")
        starts = ["AL-L", "AL-R", "AL", "all"]
        cols = [s for s in starts for _ in range(B)]
        groups = [(s, list(range(i * B, (i + 1) * B))) for i, s in enumerate(starts)]
        per_g = []
        for gi, g in enumerate(G):
            seg = seg_list(len(cols), 1, int(round(T / DT)), lambda k: g, lambda k: r, lambda k: 0.0)
            res = simulate(net, cols, groups, seg, a.seed * 100 + gi, fine_steps=500, snapshots=True, store_occ_mean=True)
            per_g.append(res)
        keys = ["modc", "fine", "ext_step", "ext_al_step", "rho", "Xk", "PR_rep", "PR_mean", "PR_boot", "occ_mean", "snap", "wb"]
        res = {k: np.stack([p[k] for p in per_g]) for k in keys}
        res["g"] = np.array(G); res["rec_dt"] = per_g[0]["rec_dt"]
        meta.update(mode="alsym", r=r, g=list(G), B=B, T=T, starts=starts, sizes={"AL-L": int(net.size[0]), "AL-R": int(net.size[1]),
                    "AL-mid": int(net.size[2])}, gpu=torch.cuda.get_device_name(), seconds=time.time() - T0)
        save(out, res, meta)

    elif a.mode == "sweep":
        r, h, D = a.r, a.h, a.D; B = a.B or 32
        G = np.geomspace(0.15, 30.0, 40)
        out = a.out or f"{OUTDIR}/sweep_r{r:g}_h{h:g}_D{D:g}{'_down' if a.down_only else ''}.npz"
        if done(out): return
        net = Net(); log(f"GPU {torch.cuda.get_device_name()}")
        K = len(G); steps = int(round(D / DT))
        if a.down_only:
            cols = ["all"] * B; groups = [("down", list(range(B)))]
            segs = seg_list(B, K, steps, lambda k: G[K - 1 - k], lambda k: r, lambda k: h)
        else:
            cols = ["silent"] * B + ["all"] * B; groups = [("up", list(range(B))), ("down", list(range(B, 2 * B)))]
            segs = [{"steps": steps, "g": np.r_[np.full(B, G[k]), np.full(B, G[K - 1 - k])], "r": np.full(2 * B, r),
                     "h": np.full(2 * B, h)} for k in range(K)]
        res = simulate(net, cols, groups, segs, a.seed, snapshots=True, store_occ_mean=True)
        res["g_grid"] = G
        meta.update(mode="sweep", r=r, h=h, D=D, B=B, g_grid=G.tolist(), down_only=a.down_only,
                    layout="segment k: up columns at g_grid[k], down columns at g_grid[K-1-k]",
                    gpu=torch.cuda.get_device_name(), seconds=time.time() - T0)
        save(out, res, meta)

    elif a.mode == "rsweep":
        g, h, D = a.g[0], a.h, a.D; B = a.B or 32
        R = np.linspace(0.0, 6.0, 25); K = len(R)
        out = a.out or f"{OUTDIR}/rsweep_g{g:g}.npz"
        if done(out): return
        net = Net(); log(f"GPU {torch.cuda.get_device_name()}")
        fwd = np.r_[R, R[::-1]]; rev = np.r_[R[::-1], R]            # 50 segments each
        cols = ["all"] * (2 * B); groups = [("fwd", list(range(B))), ("rev", list(range(B, 2 * B)))]
        steps = int(round(D / DT))
        segs = [{"steps": steps, "g": np.full(2 * B, g), "r": np.r_[np.full(B, fwd[k]), np.full(B, rev[k])],
                 "h": np.full(2 * B, h)} for k in range(2 * K)]
        res = simulate(net, cols, groups, segs, a.seed, snapshots=True, store_occ_mean=True)
        res["r_fwd"] = fwd; res["r_rev"] = rev
        meta.update(mode="rsweep", g=g, h=h, D=D, B=B, r_grid=R.tolist(), layout="segment k: fwd cols at r_fwd[k], rev cols at r_rev[k]",
                    gpu=torch.cuda.get_device_name(), seconds=time.time() - T0)
        save(out, res, meta)

    elif a.mode == "ic":
        r, g = a.r, a.g[0]; B = a.B or 32; T = a.T or 2000.0
        out = a.out or f"{OUTDIR}/ic_r{r:g}_g{g:g}.npz"
        if done(out): return
        net = Net(); log(f"GPU {torch.cuda.get_device_name()}")
        starts = ["all", "AL", "rest_only"] + [f"isl{k}" for k in range(1, 6)] + ["AL+isl1-5", "half0", "half1", "half2", "AL+isl1"]
        cols = [s for s in starts for _ in range(B)]
        groups = [(s, list(range(i * B, (i + 1) * B))) for i, s in enumerate(starts)]
        segs = seg_list(len(cols), 1, int(round(T / DT)), lambda k: g, lambda k: r, lambda k: 0.0)
        res = simulate(net, cols, groups, segs, a.seed, snapshots=True, store_occ_mean=True)
        meta.update(mode="ic", r=r, g=g, B=B, T=T, starts=starts, half_sets=[half_set(net, i).tolist() for i in range(3)],
                    gpu=torch.cuda.get_device_name(), seconds=time.time() - T0)
        save(out, res, meta)

    elif a.mode == "icq":
        cells = [tuple(float(x) for x in c.split(":")) for c in a.cells.split(",")]
        T = a.T or 5000.0
        qm = json.load(open(f"{ROOT}/results/pd3/module_theory/qmf_multi.json"))["per_r"]
        vec = np.load(f"{ROOT}/results/pd3/module_theory/qmf_multi_vectors.npz")
        net = None
        for (r, g) in cells:
            out = f"{OUTDIR}/icq_r{r:g}_g{g:.4f}.npz"
            if done(out): continue
            if net is None:
                net = Net(); log(f"GPU {torch.cuda.get_device_name()}")
            pairs = [p for p in qm[f"{r:g}"]["pairs"] if abs(p["g"] - g) < 1e-3]
            assert pairs, (r, g)
            gk = pairs[0]["g"]
            names = []
            for p in pairs:
                for s in (p["A"], p["B"]):
                    if s not in names:
                        names.append(s)
            keys = [f"r{r:g}_g{gk:.4f}_{s}" for s in names]
            assert all(k in vec.files for k in keys), keys
            fps = {s: vec[k].astype(np.float64) for s, k in zip(names, keys)}
            # region = union of neurons with |rho_a - rho_b| > 0.05 over all pairs (as qmf_multi.json n_diff_0.05)
            reg = np.zeros(net.N, bool)
            for p in pairs:
                reg |= np.abs(fps[p["A"]] - fps[p["B"]]) > 0.05
            ridx = np.where(reg)[0]
            V, off, plabels = [], [], []
            for p in pairs:
                d = (fps[p["B"]] - fps[p["A"]])[ridx]; dd = float(d @ d)
                V.append(d / dd); off.append(float(fps[p["A"]][ridx] @ d / dd)); plabels.append(f"{p['A']}->{p['B']}")
            region = {"idx": ridx, "windows": [(1000.0, 3000.0), (3000.0, 5000.0)], "blk": 10.0,
                      "V": np.array(V, np.float32).T.copy(), "off": np.array(off, np.float32)}
            Bm, Bf = (a.B or 64), max((a.B or 64) // 2, 1)
            specs = [(f"mt3:{s}", s, Bm) for s in names] + [(f"fp:{s}", f"fp:{k}", Bf) for s, k in zip(names, keys)]
            cols, groups = [], []
            for lab, sp, bb in specs:
                groups.append((lab, list(range(len(cols), len(cols) + bb)))); cols += [sp] * bb
            segs = seg_list(len(cols), 1, int(round(T / DT)), lambda k: gk, lambda k: r, lambda k: 0.0)
            res = simulate(net, cols, groups, segs, a.seed + int(1000 * g) + int(100 * r), rec_every=25, snapshots=True,
                           store_occ_mean=True, region=region)
            res["fp_reg"] = np.array([fps[s][ridx] for s in names], np.float32)
            res["fp_Xk"] = np.array([[fps[s][net.memb == k].mean() for k in range(net.M)] for s in names], np.float32)
            meta2 = dict(meta, mode="icq", r=r, g=gk, T=T, starts=[x[0] for x in specs], attractors=names, keys=keys,
                         pairs=plabels, n_region=int(len(ridx)), B_mt3=Bm, B_fp=Bf, rec_every_model_time=5.0,
                         gpu=torch.cuda.get_device_name(), seconds=time.time() - T0)
            save(out, res, meta2)

    elif a.mode == "m2b" and a.rsweep:
        # M2b on an r-sweep (added): the fwd (0 -> 6 -> 0) run reaches r = R[k] going up in segment k and going down in segment 2K-1-k
        sw = np.load(a.rsweep, allow_pickle=True); sm = json.loads(str(sw["meta"]))
        R = np.array(sm["r_grid"]); K = len(R); k = a.seg; r = float(R[k]); g, h = sm["g"], sm["h"]; B0 = sm["B"]
        T = a.T or 5000.0; B = a.B or 64
        out = a.out or f"{OUTDIR}/m2b_rsweep_g{g:g}_r{r:g}.npz"
        if done(out): return
        net = Net(); log(f"GPU {torch.cuda.get_device_name()}")
        cols = [("snap", a.rsweep, k, j % B0) for j in range(B)] + [("snap", a.rsweep, 2 * K - 1 - k, j % B0) for j in range(B)]
        groups = [("from_up", list(range(B))), ("from_down", list(range(B, 2 * B)))]
        segs = seg_list(2 * B, 1, int(round(T / DT)), lambda q: g, lambda q: r, lambda q: h)
        res = simulate(net, cols, groups, segs, a.seed, snapshots=True, store_occ_mean=True, occ_frac=0.5)
        meta.update(mode="m2b_rsweep", r=r, h=h, D=sm["D"], g=float(g), seg_up=k, seg_down=2 * K - 1 - k, B=B, T=T, sweep=a.rsweep,
                    gpu=torch.cuda.get_device_name(), seconds=time.time() - T0)
        save(out, res, meta)

    elif a.mode == "m2b":
        sw = np.load(a.sweep, allow_pickle=True); sm = json.loads(str(sw["meta"]))
        r, h, k = sm["r"], sm["h"], a.seg; B0 = sm["B"]; G = np.array(sm["g_grid"]); K = len(G)
        # up columns are at G[k] in segment k; the down branch reaches G[k] in segment K-1-k
        g = G[k]; kd = K - 1 - k; T = a.T or 5000.0; B = a.B or 64
        out = a.out or f"{OUTDIR}/m2b_r{r:g}_h{h:g}_D{sm['D']:g}_g{g:.4f}.npz"
        if done(out): return
        net = Net(); log(f"GPU {torch.cuda.get_device_name()}")
        cols = [("snap", a.sweep, k, j % B0) for j in range(B)] + [("snap", a.sweep, kd, B0 + j % B0) for j in range(B)]
        groups = [("from_up", list(range(B))), ("from_down", list(range(B, 2 * B)))]
        segs = seg_list(2 * B, 1, int(round(T / DT)), lambda q: g, lambda q: r, lambda q: h)
        res = simulate(net, cols, groups, segs, a.seed, snapshots=True, store_occ_mean=True, occ_frac=0.5)
        meta.update(mode="m2b", r=r, h=h, D=sm["D"], g=float(g), seg_up=k, seg_down=kd, B=B, T=T, sweep=a.sweep,
                    gpu=torch.cuda.get_device_name(), seconds=time.time() - T0)
        save(out, res, meta)


if __name__ == "__main__":
    main()
