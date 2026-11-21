"""Unit tests for the Model-2 options of ald_sim.py (small graphs, exact or analytic expectations; 2026-11-09).
Usage: python m2_sim_test.py <ald_sim module file> [<old ald_sim module file for the regression test>]
       -> results/m2/sim/validation/unit_tests_m2.json   (GPU if available)

 U1 --edge_sign   end to end through main() on a --graph whose neurons map (sub) to real cache neurons: K copies of
                  A (lLN1_bc, clamped on) -> B (ALPN), A -> C (ALLN). Stationary rates from the exact 2-state chain,
                  rho = p_on(s) / (p_on(s) + p_off) with s the edge input: no override; ->ALPN = 0 (B exactly silent);
                  ->ALLN = -1 with r = 0.25 and h_C = 0.5 (s = 0.25: checks the r scaling); later flag wins; edge override
                  on top of a node --sign_override. The graph file's own c is used (recomputed c would differ 3.7x).
 U2 --null_seed   same null graph (edge hash) for different --seed with --null_seed, different without (old behaviour);
                  both null kinds; null differs from the real graph.
 U3 --rec_counts  exact identities: counts at record steps = rho * N; region counts sum to the total; offs = act(s-1) +
                  on(s) - act(s) lie in [0, act(s-1)]; --counts_win slice = full-run slice (same seed); rho/surv/occ with
                  counts on = counts off (bitwise, general path = plain path). Statistical: E[offs] = p_off act(s-1),
                  E[onsets of isolated driven units] = p_on (n - act(s-1)) (z-tests).
 U4 --pulse       N = 1 per replica: renewal rho = E[L] / (E[L] + E[Q]), L ~ Geom(p_off), Q ~ Geom(p) (gap 0: Q = 1
                  exactly; gap 2; refractory n: Q = n + 1 exactly); every avalanche has size 1; clamped neurons never
                  seeded; a run that never goes silent is bitwise equal to the same run without --pulse (main RNG stream
                  untouched); seeds deplete the resource in spike mode: x = (1 - U)^(number of onsets) exactly.
 U5 --ramp        rate-mode depression on a clamped-on A driving B, g schedule [0.5, 1, 2, 1, 0.5] (loop) with 20-step
                  dwells: seg_reg(B) against the exact deterministic-resource recursion (carried across dwells), seg_x(A)
                  equal to the exact x(t); main(): schedule, loop, T override, dwell-end records; a single-g ramp with
                  D = T is bitwise equal to the plain run; early stop leaves later dwells at 0.
 U6 regression    every pre-existing simulate() configuration (drive, rate/spike depression, adaptation, refractory,
                  clamp_on, switch + windows + reg_last + occ window + extra records) gives bitwise identical outputs in
                  the new and the old module (spmm engine, same seed).
 U7 real cache    sizes of the M2 populations and of the M2s edge sets (recorded for MODEL2.md).
 U8 --init       quiet start with a selected set active (all, or each with probability F): active fraction after
                 step 1 = F (1 - p_off) in the set (z-test), exactly 0 outside; spike-mode resource of the initial
                 actives = 1 - U exactly; --init with --exp decay is an error.
"""
import importlib.util, json, os, sys, tempfile, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(path, name):
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader(name, path)       # any file name (e.g. ald_sim.py.bak-m2-pre)
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec); loader.exec_module(m)
    return m


S = load(sys.argv[1], "ald_new")
OLD = load(sys.argv[2], "ald_old") if len(sys.argv) > 2 else None
OUT = "/MALECNS_ROOT/results/m2/sim/validation"
TMP = tempfile.mkdtemp(dir=OUT, prefix="tmp_unit_")
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
DT, BETA = 0.2, 1.0
P_OFF = 1 - np.exp(-DT)
R = {"module_file": sys.argv[1], "old_module_file": sys.argv[2] if OLD else None, "device": dev}
t_start = time.time()


def pon(s):
    return 1 - np.exp(-BETA * np.tanh(np.maximum(s, 0)) * DT)


def rho2(s):
    p = pon(s)
    return p / (p + P_OFF)


def chain_tau(s):
    lam = 1 - pon(s) - P_OFF
    return (1 + lam) / (1 - lam)


def csr(rows, cols, vals, N):
    if len(rows) == 0:
        rows, cols, vals = [0], [0], [0.0]
    i = torch.tensor(np.vstack([rows, cols]), device=dev)
    return torch.sparse_coo_tensor(i, torch.tensor(np.asarray(vals, np.float32), device=dev), (N, N)).coalesce().to_sparse_csr()


def regions_of(masks, names):
    return {"names": names, "Rm": torch.tensor(np.array(masks), dtype=torch.float32, device=dev)}


def report(name, ok, **kw):
    def cv(v):
        if isinstance(v, (np.floating, np.integer, float, int)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    R[name] = {"pass": bool(ok), **{k: cv(v) for k, v in kw.items()}}
    print(f"{'PASS' if ok else 'FAIL'} {name}: " + ", ".join(f"{k}={cv(v)}" for k, v in kw.items()), flush=True)


def run_main(args, tag):
    out = f"{TMP}/{tag}.npz"
    S.main([str(x) for x in args] + ["--out", out, "--device", dev])
    return dict(np.load(out, allow_pickle=True)), json.load(open(out[:-4] + ".json"))


# real-cache labels (for --graph sub)
C = S.C
ty = np.load(f"{C}/types.npy", allow_pickle=True).astype(str)
cl = np.load(f"{C}/classes.npy", allow_pickle=True).astype(str)
sg = np.load(f"{C}/sign.npy")
gA = int(np.where(ty == "lLN1_bc")[0][0]); gB = int(np.where((cl == "ALPN") & (ty != "lLN1_bc"))[0][0])
gC = int(np.where((cl == "ALLN") & (ty != "lLN1_bc"))[0][0]); gD = int(np.where(cl == "olfactory")[0][0])

# ------------------------------------------------------------------------------------------------- U1 edge_sign
K = 300
loc = lambda k, j: 5 * k + j                       # per copy: 0 A, 1 B, 2 C, 3 D (ORN), 4 D2 (ORN)
pre, post, w = [], [], []
for k in range(K):
    pre += [loc(k, 0), loc(k, 0), loc(k, 3)]; post += [loc(k, 1), loc(k, 2), loc(k, 4)]; w += [10.0, 10.0, 90.0]
sub = np.tile([gA, gB, gC, gD, gD], K)
gfile = f"{TMP}/motif.npz"
np.savez(gfile, pre=np.array(pre), post=np.array(post), w=np.array(w, np.float32), sign=sg[sub], sub=sub, c=0.1)
c_recomputed = 1 / np.mean([10, 10, 90])
base = ["--graph", gfile, "--exp", "quiet", "--norm", "raw", "--g", 1.0, "--T", 400, "--B", 64, "--no_default_regions",
        "--clamp_on", "lLN1_bc", "--region", "B=ALPN", "--region", "C=ALLN&~lLN1_bc", "--seed", 11]
cases = {
    "none": ([], 1.0, 1.0, 1.0),
    "PN0": (["--edge_sign", "lLN1_bc->ALPN=0"], 1.0, 0.0, 1.0),
    "PN0_LNm1_r0.25_hC0.5": (["--edge_sign", "lLN1_bc->ALPN=0", "--edge_sign", "lLN1_bc->ALLN=-1", "--r", 0.25,
                              "--drive", "ALLN&~lLN1_bc=0.5"], 1.0, 0.0, 0.25),
    "later_wins": (["--edge_sign", "lLN1_bc->ALPN=0", "--edge_sign", "lLN1_bc->all=1"], 1.0, 1.0, 1.0),
    "on_top_of_node_override": (["--sign_override", "lLN1_bc=0", "--edge_sign", "lLN1_bc->ALLN=1"], 1.0, 0.0, 1.0),
}
for name, (extra, _, sB, sC) in cases.items():
    res, side = run_main(base + extra, f"U1_{name}")
    t = res["t"]; late = t > 200
    mB = res["reg_rho"][0][late][:, 0].mean(); mC = res["reg_rho"][0][late][:, 1].mean()
    ok = True; info = {}
    for lab, m, s in (("B", mB, sB), ("C", mC, sC)):
        ex = rho2(s) if s > 0 else 0.0
        nrec = late.sum()
        se = np.sqrt(max(ex * (1 - ex), 1e-12) * chain_tau(max(s, 1e-9)) / (K * 64 * 5))   # >= 5 independent windows
        ok &= (m == 0.0) if s <= 0 else abs(m - ex) < 5 * se + 2e-3
        info[f"{lab}_meas"] = m; info[f"{lab}_exact"] = ex
    n_ed = [e["n_edges"] for e in side.get("edge_sign", [])]
    info["n_edges_per_override"] = n_ed
    if name == "later_wins":
        ok &= n_ed == [K, 2 * K]
    elif extra:
        ok &= all(n == K for n in n_ed)
    report(f"U1_edge_sign_{name}", ok, **info)
report("U1_graph_file_c_used", abs(rho2(1.0) - R["U1_edge_sign_none"]["B_meas"]) < 0.01
       and abs(rho2(10 * c_recomputed) - rho2(1.0)) > 0.05,
       rho_with_file_c=rho2(1.0), rho_if_c_recomputed=rho2(10 * c_recomputed), measured=R["U1_edge_sign_none"]["B_meas"])

# ------------------------------------------------------------------------------------------------- U2 null_seed
rng = np.random.default_rng(3)
Nn, En = 400, 4000
gf2 = f"{TMP}/rand.npz"
np.savez(gf2, pre=rng.integers(0, Nn, En), post=rng.integers(0, Nn, En), w=rng.integers(1, 20, En).astype(np.float32),
         sign=rng.choice([1, -1, 0], Nn))
b2 = ["--graph", gf2, "--exp", "decay", "--norm", "raw", "--g", 0.5, "--T", 2, "--B", 2, "--null"]
h = {}
for kind in ("targets", "weights"):
    for tag, extra in (("ns5_s1", ["--null_seed", 5, "--seed", 1]), ("ns5_s2", ["--null_seed", 5, "--seed", 2]),
                       ("s1", ["--seed", 1]), ("s2", ["--seed", 2])):
        _, side = run_main(b2 + ["--null_kind", kind] + extra, f"U2_{kind}_{tag}")
        h[f"{kind}_{tag}"] = side["null_edges_md5"]
import hashlib
G0 = np.load(gf2)
real_md5 = hashlib.md5(np.ascontiguousarray(G0["post"].astype(np.int64)).tobytes() + np.ascontiguousarray(G0["w"]).tobytes()).hexdigest()
ok = all(h[f"{k}_ns5_s1"] == h[f"{k}_ns5_s2"] and h[f"{k}_s1"] != h[f"{k}_s2"] and h[f"{k}_ns5_s1"] != real_md5
         for k in ("targets", "weights"))
report("U2_null_seed", ok, **h, real=real_md5)

# ------------------------------------------------------------------------------------------------- U3 counts
nA, nB = 1000, 1000
N = nA + nB
rows = nA + np.arange(nB); cols = np.arange(nA)
hv = np.r_[np.full(nA, 0.3), np.zeros(nB)]
mA = np.r_[np.ones(nA, bool), np.zeros(nB, bool)]
Wm = csr(rows, cols, np.full(nB, 0.5), N)
regs = regions_of([mA, ~mA], ["A", "B"])
kw = dict(exp="quiet", T=300.0, dt=DT, B=64, beta=BETA, dev=dev, regions=regs)
P = S.make_params(N, DT, dev, h=0.0, hv=hv)
torch.manual_seed(7)
r1 = S.simulate(Wm, N, [1.0], P, rng=np.random.default_rng(1), counts={"s0": 0, "s1": 1500, "reg": True}, **kw)
torch.manual_seed(7)
r2 = S.simulate(Wm, N, [1.0], P, rng=np.random.default_rng(1), counts={"s0": 500, "s1": 1000, "reg": False}, **kw)
torch.manual_seed(7)
r0 = S.simulate(Wm, N, [1.0], P, rng=np.random.default_rng(1), **kw)
ca, co = r1["cnt_act"][0].astype(np.int64), r1["cnt_on"][0].astype(np.int64)
steps_rec = np.round(r1["t"] / DT).astype(int)
ok_rec = np.array_equal(np.round(r1["rho"][0] * N).astype(np.int64), ca[steps_rec - 1])
ok_reg = np.array_equal(r1["cnt_reg_act"][0].sum(1), ca) and np.array_equal(r1["cnt_reg_on"][0].sum(1), co)
off = ca[:-1] + co[1:] - ca[1:]
ok_bal = bool((off >= 0).all() and (off <= ca[:-1]).all())
ok_win = np.array_equal(r2["cnt_act"][0], r1["cnt_act"][0][500:1000]) and np.array_equal(r2["cnt_on"][0], r1["cnt_on"][0][500:1000])
ok_same = all(np.array_equal(r0[k], r1[k]) for k in ("rho", "surv", "occ", "reg_rho"))
z_off = (off.sum() - P_OFF * ca[:-1].sum()) / np.sqrt(P_OFF * (1 - P_OFF) * ca[:-1].sum())
aA = r1["cnt_reg_act"][0][:, 0].astype(np.int64); oA = r1["cnt_reg_on"][0][:, 0].astype(np.int64)
pA = pon(0.3)
exp_on = pA * (nA - aA[:-1]).sum()
z_on = (oA[1:].sum() - exp_on) / np.sqrt(pA * (1 - pA) * (nA - aA[:-1]).sum())
report("U3_counts_exact_identities", ok_rec and ok_reg and ok_bal and ok_win and ok_same, records_equal_rhoN=ok_rec,
       regions_sum_to_total=ok_reg, off_balance=ok_bal, window_slice_equal=ok_win, rho_surv_occ_unchanged=ok_same)
report("U3_counts_statistics", abs(z_off) < 5 and abs(z_on) < 5, z_off=z_off, z_on_isolated_driven=z_on)
# plain path (decay, scalar h = 0) vs general path forced by counts: bitwise identical
Wc = csr(np.arange(1, 300), np.arange(0, 299), np.full(299, 3.0), 300)
P0 = S.make_params(300, DT, dev, h=0.0)
kw0 = dict(exp="decay", T=60.0, dt=DT, B=32, beta=BETA, dev=dev)
torch.manual_seed(5); q0 = S.simulate(Wc, 300, [0.6, 1.2], P0, rng=np.random.default_rng(0), **kw0)
torch.manual_seed(5); q1 = S.simulate(Wc, 300, [0.6, 1.2], P0, rng=np.random.default_rng(0), counts={"s0": 0, "s1": 300}, **kw0)
report("U3_counts_plain_vs_general_bitwise", all(np.array_equal(q0[k], q1[k]) for k in ("rho", "surv", "occ", "t")),
       early_stop_steps=float(q1["rho"].shape[1]))

# ------------------------------------------------------------------------------------------------- U4 pulse
def pulse_run(N, gap, nref=0, off=None, std=None, T=2000.0, B=4096, seed=3, hv=None, regions=None, counts=True,
              std_mode="spike", exp="quiet"):
    P = S.make_params(N, DT, dev, h=0.0, hv=hv, nref=nref, off=off, std_spec=std, std_mode=std_mode)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    seedable = np.arange(N) if off is None else np.where(~off)[0]
    pp = 1.0 if gap <= 0 else 1 - np.exp(-DT / gap)
    pul = {"idx": torch.tensor(seedable, device=dev), "p": pp, "gen": gen}
    torch.manual_seed(seed)
    return S.simulate(csr([], [], [], N), N, [1.0], P, exp=exp, T=T, dt=DT, B=B, beta=BETA,
                      rng=np.random.default_rng(seed), dev=dev, regions=regions,
                      counts={"s0": 0, "s1": int(round(T / DT))} if counts else None, pulse=pul), pp


import m2_obs as OBS  # noqa: E402  (code/ is on sys.path when run from code/)
for gap, nref in ((0.0, 0), (2.0, 0), (0.0, 3)):
    res, pp = pulse_run(1, gap, nref)
    EL = 1 / P_OFF; EQ = (nref + 1) if gap <= 0 else 1 / pp
    ex = EL / (EL + EQ)
    occm = res["occ"][0].astype(np.float64).mean()
    ca = res["cnt_act"][0]
    av = OBS.avalanches(ca, res["cnt_on"][0])
    # silent runs (between avalanches)
    sil = OBS.avalanches((ca == 0).astype(np.int64), None, thr=0)["dur"]
    n_av = len(av["size"])
    se_rho = np.sqrt(ex * (1 - ex) * 2 * max(EL, EQ) / (4096 * 5000))
    se_L = np.sqrt((1 - P_OFF) / P_OFF ** 2 / n_av)
    ok = abs(occm - ex) < 5 * se_rho + 1e-3 and np.all(av["size"] == 1) and abs(av["dur"].mean() - EL) < 5 * se_L
    if gap <= 0:
        ok &= bool(np.all(sil == nref + 1))
    else:
        ok &= abs(sil.mean() - 1 / pp) < 5 * np.sqrt((1 - pp) / pp ** 2 / len(sil))
    report(f"U4_pulse_renewal_gap{gap:g}_ref{nref}", ok, rho_meas=occm, rho_exact=ex, mean_dur_steps=av["dur"].mean(),
           E_L=EL, mean_silent_steps=sil.mean(), E_Q=EQ, all_sizes_1=bool(np.all(av["size"] == 1)), n_av=n_av)
off = np.array([True, True, False, False])
res, _ = pulse_run(4, 0.0, off=off, T=400.0, B=1024)
o = res["occ"][0].astype(np.float64)
report("U4_pulse_respects_clamp_off", o[0] == 0 and o[1] == 0 and abs(o[2] - o[3]) < 0.02 and o[2] > 0.1, occ=o)
# never silent -> bitwise identical to the run without pulse (main stream untouched)
Nd = 200
hvd = np.full(Nd, 5.0)
Pd = S.make_params(Nd, DT, dev, h=0.0, hv=hvd)
torch.manual_seed(9); a0 = S.simulate(csr([], [], [], Nd), Nd, [1.0], Pd, exp="decay", T=100.0, dt=DT, B=64, beta=BETA,
                                      rng=np.random.default_rng(9), dev=dev)
gen = torch.Generator(device=dev); gen.manual_seed(1)
torch.manual_seed(9); a1 = S.simulate(csr([], [], [], Nd), Nd, [1.0], Pd, exp="decay", T=100.0, dt=DT, B=64, beta=BETA,
                                      rng=np.random.default_rng(9), dev=dev,
                                      pulse={"idx": torch.arange(Nd, device=dev), "p": 1.0, "gen": gen})
same = all(np.array_equal(a0[k], a1[k]) for k in ("rho", "surv", "occ"))
report("U4_pulse_main_stream_untouched", same and a0["surv"][0][1:].min() == 1, identical=same)
# seeds deplete (spike mode, no recovery): x_b(T) = (1 - U)^(onsets of b)
# (U = 0.1 keeps x >= ~1e-3: the float32 recovery update 1 + (x - 1) q has an absolute error ~6e-8 per step, so
#  tiny x (e.g. 0.5^31) cannot be compared in relative terms; that is a property of the E3 update, not of --pulse)
res, _ = pulse_run(1, 0.0, std=(np.array([0.1]), np.array([np.inf])), T=40.0, B=512,
                   regions=regions_of([np.ones(1, bool)], ["A"]))
n_on = res["cnt_on"][0].sum(0)
x_exact = np.mean(0.9 ** n_on)
x_meas = res["res_x"][0][-1, 0]
report("U4_pulse_seeds_deplete", abs(x_meas - x_exact) < 1e-4 * x_exact, x_meas=x_meas, x_exact=x_exact,
       rel_err=abs(x_meas - x_exact) / x_exact, mean_onsets=n_on.mean())

# ------------------------------------------------------------------------------------------------- U5 ramp
nm, wAB, U, tau = 2000, 1.2, 0.05, 50.0
N = 2 * nm; A = np.arange(nm); Bn = nm + A
on = np.zeros(N, bool); on[A] = True
Uv = np.zeros(N); Uv[A] = U; Tv = np.full(N, np.inf); Tv[A] = tau
P = S.make_params(N, DT, dev, h=0.0, std_spec=(Uv, Tv), on=on)            # rate mode (deterministic x for clamped A)
sched = [0.5, 1.0, 2.0, 1.0, 0.5]; Ld, Av = 20, 10
torch.manual_seed(4)
res = S.simulate(csr(Bn, A, np.full(nm, wAB), N), N, [sched[0]], P, exp="decay", T=len(sched) * Ld * DT, dt=DT, B=256,
                 beta=BETA, rng=np.random.default_rng(4), dev=dev, regions=regions_of([on, ~on], ["A", "B"]),
                 ramp={"g": sched, "L": Ld, "A": Av})
k1 = 1 / tau + U; xinf = (1 / tau) / k1; ea = np.exp(-k1 * DT)
x, rB = 1.0, 1.0
seg_ex, xend = np.zeros(len(sched)), np.zeros(len(sched))
for s in range(1, len(sched) * Ld + 1):
    j, pos = divmod(s - 1, Ld)
    gcur = sched[j]
    rB = rB * (1 - P_OFF) + (1 - rB) * pon(gcur * wAB * x)
    x = xinf + (x - xinf) * ea
    if pos >= Ld - Av:
        seg_ex[j] += rB / Av
    if pos == Ld - 1:
        xend[j] = x
measB = res["seg_reg"][:, 1, :].mean(1)
se = np.sqrt(seg_ex * (1 - seg_ex) / (nm * 256))
z = (measB - seg_ex) / se
# what a reset of x at each dwell would give (shows the test is sensitive to the carry-over)
report("U5_ramp_carry_over_exact", np.abs(z).max() < 5 and np.allclose(res["seg_x"][:, 0], xend, atol=1e-5)
       and np.array_equal(res["seg_g"], sched),
       max_abs_z=np.abs(z).max(), seg_B_meas=measB, seg_B_exact=seg_ex, seg_x=res["seg_x"][:, 0], x_exact=xend)
res, side = run_main(["--graph", gf2, "--exp", "decay", "--norm", "raw", "--g", 0.3, 0.6, 1.2, "--ramp", 4,
                      "--ramp_loop", "--B", 4, "--seed", 1], "U5_main_loop")
ok = (side["ramp"]["schedule"] == [0.3, 0.6, 1.2, 0.6, 0.3] and abs(side["ramp"]["T"] - 20.0) < 1e-9
      and res["seg_rho"].shape == (5, 4) and all(np.isclose(res["t"], 4.0 * k).any() for k in range(1, 6)))
report("U5_ramp_main_schedule", ok, schedule=side["ramp"]["schedule"], T=side["ramp"]["T"])
torch.manual_seed(5); p0 = S.simulate(Wc, 300, [1.2], P0, rng=np.random.default_rng(0), **{**kw0, "T": 40.0})
torch.manual_seed(5); p1 = S.simulate(Wc, 300, [1.2], P0, rng=np.random.default_rng(0), **{**kw0, "T": 40.0},
                                      ramp={"g": [1.2], "L": 200, "A": 100})
report("U5_single_dwell_ramp_bitwise_plain", all(np.array_equal(p0[k], p1[k]) for k in ("rho", "surv", "occ", "t")))
torch.manual_seed(5); p2 = S.simulate(Wc, 300, [0.0], P0, rng=np.random.default_rng(0), **{**kw0, "T": 40.0},
                                      ramp={"g": [0.0, 0.0, 5.0, 5.0], "L": 50, "A": 25})
report("U5_ramp_early_stop", np.all(p2["seg_rho"][2:] == 0) and np.all(p2["seg_alive"][2:] == 0),
       seg_rho_mean=p2["seg_rho"].mean(1), seg_alive=p2["seg_alive"].mean(1))

# ------------------------------------------------------------------------------------------------- U6 regression
if OLD is not None:
    def both(fn):
        outs = []
        for M in (OLD, S):
            torch.manual_seed(21)
            outs.append(fn(M))
        a, b = outs
        keys = sorted(a.keys())
        return all(np.array_equal(a[k], b[k], equal_nan=True) for k in keys if k != "sec_per_step"), keys
    nA = 600; N = 2 * nA
    rows = nA + np.arange(nA); cols = np.arange(nA)
    W6 = lambda: csr(np.r_[rows, cols], np.r_[cols, rows], np.r_[np.full(nA, 0.8), np.full(nA, -0.4)], N)
    mA = np.r_[np.ones(nA, bool), np.zeros(nA, bool)]
    hv = np.r_[np.full(nA, 0.2), np.zeros(nA)]
    regs6 = lambda: regions_of([mA, ~mA], ["A", "B"])
    confs = {
        "drive": dict(P=dict(hv=hv)),
        "std_rate": dict(P=dict(hv=hv, std_spec=(np.full(N, 0.3), np.full(N, 20.0)))),
        "std_spike": dict(P=dict(hv=hv, std_spec=(np.full(N, 0.25), np.full(N, 157.0)), std_mode="spike")),
        "adapt_ref": dict(P=dict(hv=hv, adapt_spec=(np.full(N, 0.02), np.full(N, 50.0)), nref=2)),
        "clamp_on_std": dict(P=dict(on=mA, std_spec=(np.full(N, 0.3), np.full(N, 5.0))), exp="decay"),
        "switch_windows": dict(P=dict(hv=hv, std_spec=(np.full(N, 0.25), np.full(N, 100.0)), std_mode="spike"),
                               switch=True, windows=[(100, 400), (500, 700)], reg_last=True, occ_steps=(100, 400),
                               rec_extra=np.arange(10, 1000, 10)),
    }
    for name, c in confs.items():
        def fn(M, c=c):
            P = M.make_params(N, DT, dev, h=0.0, **c["P"])
            sw = None
            if c.get("switch"):
                P2 = M.make_params(N, DT, dev, h=0.0, hv=np.zeros(N), off=mA)
                sw = {"s": 400, "P": P2}
            return M.simulate(W6(), N, [0.5, 1.5], P, exp=c.get("exp", "quiet"), T=200.0, dt=DT, B=32, beta=BETA,
                              rng=np.random.default_rng(2), dev=dev, regions=regs6(), switch=sw,
                              windows=c.get("windows"), reg_last=c.get("reg_last", False),
                              occ_steps=c.get("occ_steps"), rec_extra=c.get("rec_extra"))
        ok, keys = both(fn)
        report(f"U6_regression_old_vs_new_{name}", ok, keys=keys)
    # build_matrix on the real cache: default (no edge override) bitwise equal to the old module
    pre_ = np.load(f"{C}/pre.npy").astype(np.int64); post_ = np.load(f"{C}/post.npy").astype(np.int64)
    w_ = np.load(f"{C}/weight.npy").astype(np.float32)
    sc = S.node_scale(post_, w_, w_, len(sg), "raw", "renorm")
    sig = S.sigma_vector(sg, 1.0)
    Wn, nn = S.build_matrix(pre_, post_, w_, sc, sig, len(sg), dev)
    Wo, no = OLD.build_matrix(pre_, post_, w_, sc, sig, len(sg), dev)
    same = (nn == no and torch.equal(Wn.crow_indices(), Wo.crow_indices()) and torch.equal(Wn.col_indices(), Wo.col_indices())
            and torch.equal(Wn.values(), Wo.values()))
    # identity override (V = existing sign) leaves the operator bitwise unchanged
    L = S.Labels(len(sg), sign=sg)
    em = S.select("lLN1_bc", L)[pre_] & S.select("ALPN", L)[post_]
    Wi, ni = S.build_matrix(pre_, post_, w_, sc, sig, len(sg), dev, edge_ov=[(em, 1.0)])
    same_i = torch.equal(Wi.values(), Wn.values()) and torch.equal(Wi.col_indices(), Wn.col_indices())
    Wz, nz = S.build_matrix(pre_, post_, w_, sc, sig, len(sg), dev, edge_ov=[(em, 0.0)])
    report("U6_build_matrix_real_cache", same and same_i and nz == nn - int(em.sum()), nnz=nn, nnz_lLN1bc_PN0=nz,
           n_edges_lLN1bc_to_ALPN=int(em.sum()))
    del Wn, Wo, Wi, Wz

# ------------------------------------------------------------------------------------------------- U8 --init
if "--init" in open(sys.argv[1]).read():
    Ni = 1000
    gi = f"{TMP}/iso.npz"
    # one dummy edge between two neurons outside the init set (they never activate: h = 0, no active input)
    np.savez(gi, pre=np.array([Ni - 1]), post=np.array([Ni - 2]), w=np.array([1.0], np.float32), sign=np.ones(Ni, np.int64))
    half = f"{TMP}/half.npz"; np.savez(half, idx=np.arange(Ni // 2))
    bi = ["--graph", gi, "--exp", "quiet", "--norm", "raw", "--g", 1.0, "--T", 0.4, "--B", 256, "--seed", 3,
          "--region", f"A=idx:{half}", "--region", f"B=~idx:{half}", "--std", 0.25, "inf", "--std_mode", "spike"]
    ok = True; info = {}
    for frac in (1.0, 0.3):
        res, side = run_main(bi + ["--init", f"idx:{half}", "--init_frac", frac], f"U8_init_{frac}")
        a1 = res["reg_rho"][0][0, 0, :].mean(); b1 = res["reg_rho"][0][0, 1, :].max()
        ex = frac * (1 - P_OFF)
        se = np.sqrt(ex * (1 - ex) / (Ni // 2 * 256))
        ok &= abs(a1 - ex) < 5 * se and b1 == 0 and side["init"]["n"] == Ni // 2
        info[f"frac{frac}_A_step1"] = a1; info[f"frac{frac}_expected"] = ex; info[f"frac{frac}_B_max"] = b1
        if frac == 1.0:            # every init neuron counts as an onset at t = 0: x = 1 - U, no recovery (TAU inf)
            xa, xb = res["res_x"][0, 0, 0, 0], res["res_x"][0, 0, 0, 1]          # [draw, g, t, R]
            ok &= abs(xa - 0.75) < 1e-6 and abs(xb - 1.0) < 1e-6
            info["x_A_after_step1"] = xa; info["x_B"] = xb
    try:
        run_main(["--graph", gi, "--exp", "decay", "--g", 1.0, "--T", 0.4, "--B", 2, "--init", f"idx:{half}"], "U8_bad")
        bad = False
    except ValueError:
        bad = True
    report("U8_init", ok and bad, init_requires_quiet=bad, **info)

# ------------------------------------------------------------------------------------------------- U7 real cache sets
L = S.Labels(len(sg), sign=sg)
pre_ = np.load(f"{C}/pre.npy").astype(np.int64); post_ = np.load(f"{C}/post.npy").astype(np.int64)
w_ = np.load(f"{C}/weight.npy").astype(np.float32)
sets = {k: S.select(v, L) for k, v in {"exc_sign+1": "sign:1", "inh_sign-1": "sign:-1", "mod_sign0": "sign:0",
                                        "ORN_olfactory": "olfactory", "lLN1_bc": "lLN1_bc", "ALPN": "ALPN",
                                        "ALLN": "ALLN", "module": "module"}.items()}
info = {f"n_{k}": int(m.sum()) for k, m in sets.items()}
src = sets["lLN1_bc"][pre_]
for tgt in ("ALPN", "ALLN"):
    e = src & sets[tgt][post_]
    info[f"lLN1_bc->{tgt}_edges"] = int(e.sum()); info[f"lLN1_bc->{tgt}_synapses"] = float(w_[e].sum())
e = src & ~sets["ALPN"][post_] & ~sets["ALLN"][post_]
info["lLN1_bc->other_edges"] = int(e.sum()); info["lLN1_bc->other_synapses"] = float(w_[e].sum())
oth_cls = np.load(f"{C}/classes.npy", allow_pickle=True).astype(str)[post_[e]]
u, c = np.unique(oth_cls, return_counts=True)
info["lLN1_bc->other_by_class_edges"] = {str(a): int(b) for a, b in sorted(zip(u, c), key=lambda z: -z[1])[:10]}
report("U7_real_cache_sets", info["n_ORN_olfactory"] == 2639 and info["n_lLN1_bc"] == 30, **info)

R["_meta"] = {"n_pass": sum(v["pass"] for k, v in R.items() if isinstance(v, dict) and "pass" in v),
              "n_tests": sum(1 for k, v in R.items() if isinstance(v, dict) and "pass" in v),
              "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "wall_s": time.time() - t_start}
R["all_pass"] = R["_meta"]["n_pass"] == R["_meta"]["n_tests"]
json.dump(R, open(f"{OUT}/unit_tests_m2.json", "w"), indent=1, default=str)
print(json.dumps(R["_meta"]), "ALL PASS" if R["all_pass"] else "SOME FAILED", flush=True)
