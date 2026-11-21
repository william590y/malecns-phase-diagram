"""Tests of the E1 options of ald_sim.py (--switch_at / --sw_* / --rec_lin / --win / --reg_last / --occ_win), 2026-11-09.
Run on a GPU node (falls back to CPU): python ald_drive_test.py [--cli]  -> results/aldef/drive/validation/drive_tests.json

Small graphs of ISOLATED units (no edges), exact expectations:
 D1 switch          three groups driven at h = 0.5 for T1, then A clamped off, B drive set to 0, C unchanged:
                    A exactly 0 after T1; E[B(T1 + k dt)] = rho_ss (1 - p_off)^k (isolated units cannot re-activate);
                    C stays at rho_ss = p_on/(p_on + p_off).
 D2 windows         win_mean over a stationary window = rho_ss (z-test, autocorrelation-corrected);
                    win_m2 = rho_ss^2 + rho_ss (1 - rho_ss)/n for n independent units (z-test).
 D3 reg_last        A: last activity time = T1 exactly (A active at T1 in every replica); C: = T;
                    B: E[last - T1] from the exact max-of-geometric distribution, CDF(k) = (1 - rho_ss q^(k+1))^n.
 D4 early stop      phase 2 without drive is absorbing: the run stops early, records after the stop are 0, windows count
                    the skipped steps as silent.
 D5 driven phase 2  phase 1 at h = 0 (quiet, would be absorbing) and phase 2 driven: NO early stop in phase 1;
                    activity rises to rho_ss after the switch.
 D6 occ window      occ over a window = rho_ss per group.
--cli (full cache, needs the GPU): one ald_sim.py CLI run at g = 0 and 0.3 with ORN drive + antenna removal; checks that
       ORNs follow the isolated-unit formula, are exactly 0 after the switch with reg_last(ORN) = T1, and that every
       non-ORN region is silent at g = 0. Also: swc.py vs ald_sim.py with --rec_lin/--win/--reg_last switched on (no drive):
       rho/surv at swc.py's record times and occ must be bit-identical.
"""
import json, os, subprocess, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ald_sim as S

OUT = f"{S.ROOT}/results/aldef/drive/validation"
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
DT, BETA = 0.2, 1.0
P_OFF = 1 - np.exp(-DT); Q = 1 - P_OFF
results = {}


def pon(s):
    return 1 - np.exp(-BETA * np.tanh(np.maximum(s, 0)) * DT)


def empty(N):
    i = torch.tensor([[0], [0]], device=dev)
    return torch.sparse_coo_tensor(i, torch.tensor([0.0], device=dev), (N, N)).coalesce().to_sparse_csr()


def report(name, ok, **kw):
    results[name] = {"pass": bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v)
                                          for k, v in kw.items()}}
    print(f"{'PASS' if ok else 'FAIL'} {name}: " + ", ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


t0 = time.time()
n, B, h = 200, 256, 0.5
N = 3 * n
pa = pon(h); rho_ss = pa / (pa + P_OFF)
grp = np.repeat(np.arange(3), n)
masks = [grp == k for k in range(3)]
regions = {"names": ["A", "B", "C"], "Rm": torch.tensor(np.array(masks), dtype=torch.float32, device=dev)}
T1, T = 40.0, 60.0
s_sw, steps = int(round(T1 / DT)), int(round(T / DT))
hv = np.full(N, h)
P1 = S.make_params(N, DT, dev, h=0.0, hv=hv)
hv2 = hv.copy(); hv2[masks[1]] = 0.0
P2 = S.make_params(N, DT, dev, h=0.0, hv=hv2, off=masks[0])
wins = [(int(round(20 / DT)), s_sw), (s_sw, steps)]
rec_extra = np.concatenate([[s_sw], s_sw + np.arange(1, 41), np.arange(5, steps + 1, 5)])
res = S.simulate(empty(N), N, [1.0], P1, exp="quiet", T=T, dt=DT, B=B, beta=BETA, rng=np.random.default_rng(0), dev=dev,
                 regions=regions, switch={"s": s_sw, "P": P2}, rec_extra=rec_extra, windows=wins, reg_last=True,
                 occ_steps=wins[0])
t = res["t"]; reg = res["reg_rho"][0]          # [t, R, B]
after = t > T1 + 1e-9
# D1
okA = bool(np.all(reg[after, 0, :] == 0))
iT1 = np.argmin(abs(t - T1))
zs = []
for k in (1, 2, 5, 10, 20, 40):
    i = np.argmin(abs(t - (T1 + k * DT)))
    exp_b = rho_ss * Q ** k
    m = reg[i, 1, :].mean()
    se = np.sqrt(exp_b * (1 - exp_b) / (n * B))
    zs.append((k, exp_b, m, (m - exp_b) / se))
okB = all(abs(z) < 4.5 for *_, z in zs)
mC = reg[after, 2, :].mean()
nC = after.sum()
seC = np.sqrt(rho_ss * (1 - rho_ss) / (n * B * nC)) * np.sqrt((1 + Q - pa) / (1 - Q + pa))   # AR(1) correction (rough)
report("D1_switch_clampA_zero", okA, max_A_after=float(reg[after, 0, :].max()))
report("D1_switch_B_decay", okB, checks=[{"k": k, "expected": e, "measured": float(m), "z": float(z)} for k, e, m, z in zs])
report("D1_switch_C_unchanged", abs(mC - rho_ss) < 5 * seC + 1e-3, expected=rho_ss, measured=mC, se=seC)
# D2 windows
wm, w2 = res["win_mean"][0], res["win_m2"][0]        # [W, R, B]
nstep = wins[0][1] - wins[0][0]
lam = Q - pa                                          # per-step autocorrelation of one unit
se_w = np.sqrt(rho_ss * (1 - rho_ss) / (n * B * nstep) * (1 + lam) / (1 - lam))
okw = all(abs(wm[0, r].mean() - rho_ss) < 5 * se_w for r in range(3))
report("D2_window_mean_phase1", okw, expected=rho_ss, measured=[float(wm[0, r].mean()) for r in range(3)], se=se_w)
e2 = rho_ss ** 2 + rho_ss * (1 - rho_ss) / n
m2 = w2[0].mean()
se2 = w2[0].std() / np.sqrt(3 * B)
report("D2_window_m2_phase1", abs(m2 - e2) < 5 * se2, expected=e2, measured=m2, se=se2)
okA2 = bool(np.all(wm[1, 0] == 0))
report("D2_window_A_after_zero", okA2)
# D3 reg_last
L = res["reg_last"][0]                                # [R, B]
okLA = bool(np.all(np.isclose(L[0], T1)))
okLC = bool(np.all(np.isclose(L[2], T)))
j = np.arange(1, 400)
surv = 1 - (1 - rho_ss * Q ** j) ** n                 # P(last_step - s_sw >= j)
E_last = surv.sum() * DT
mL = (L[1] - T1).mean(); seL = (L[1] - T1).std() / np.sqrt(B)
report("D3_reg_last_A_equals_T1", okLA, values=np.unique(L[0]).tolist())
report("D3_reg_last_C_equals_T", okLC, values=np.unique(L[2]).tolist())
report("D3_reg_last_B_extinction", abs(mL - E_last) < 5 * seL, expected=E_last, measured=mL, se=seL)
# D6 occ window
occ = res["occ"][0].astype(np.float64)
okocc = all(abs(occ[masks[r]].mean() - rho_ss) < 5 * se_w + 2e-3 for r in range(3))
report("D6_occ_window", okocc, expected=rho_ss, measured=[float(occ[masks[r]].mean()) for r in range(3)])

# D4 early stop: everything switched off at T1 -> absorbing
P2b = S.make_params(N, DT, dev, h=0.0, hv=np.zeros(N))
T4 = 400.0
res4 = S.simulate(empty(N), N, [1.0], P1, exp="quiet", T=T4, dt=DT, B=B, beta=BETA, rng=np.random.default_rng(1),
                  dev=dev, regions=regions, switch={"s": s_sw, "P": P2b}, rec_extra=[s_sw], windows=[(s_sw, int(T4 / DT))],
                  reg_last=True)
steps_run = res4["sec_per_step"]  # not informative; use records
t4 = res4["t"]; r4 = res4["rho"][0]
late = t4 > T1 + 60
ok4 = bool(np.all(r4[late] == 0)) and bool(np.all(res4["reg_last"][0] < T1 + 60))
# expected window mean after switch: sum over k of rho_ss Q^k / (window steps)
exp_w = rho_ss * Q / (1 - Q) / int(T4 / DT - s_sw)
m4 = res4["win_mean"][0, 0].mean()
report("D4_early_stop_absorbing", ok4 and abs(m4 - exp_w) < 0.05 * exp_w + 5e-5, window_mean_expected=exp_w,
       window_mean_measured=m4, max_last=float(res4["reg_last"][0].max()))

# D5 phase 1 undriven (absorbing on its own) -> phase 2 driven: must not stop early
P1q = S.make_params(N, DT, dev, h=0.0, hv=np.zeros(N))
P2d = S.make_params(N, DT, dev, h=0.0, hv=hv)
res5 = S.simulate(empty(N), N, [1.0], P1q, exp="quiet", T=T, dt=DT, B=B, beta=BETA, rng=np.random.default_rng(2),
                  dev=dev, regions=regions, switch={"s": s_sw, "P": P2d}, rec_extra=[s_sw], windows=[(s_sw + 50, steps)])
m5 = res5["win_mean"][0, 0].mean()
report("D5_no_early_stop_before_driven_phase2", abs(m5 - rho_ss) < 0.01 and bool(np.all(res5["rho"][0][res5["t"] <= T1] == 0)),
       expected=rho_ss, measured=m5)

# ------------------------------------------------------------------------------------- CLI on the full cache (GPU)
if "--cli" in sys.argv:
    PY = sys.executable
    D = f"{S.ROOT}/results/aldef/drive"
    hO = 0.03590103426967425                                  # isolated-unit rho 0.038 (ORN 3.44 Hz at tau_u 10 ms)
    paO = pon(hO); rhoO = paO / (paO + P_OFF)
    f = f"{OUT}/cli_switch.npz"
    cmd = [PY, f"{S.ROOT}/code/ald_sim.py", "--exp", "quiet", "--norm", "raw", "--r", "1", "--g", "0", "0.3",
           "--T", "60", "--B", "16", "--seed", "7", "--drive", f"olfactory={hO}", "--switch_at", "40",
           "--sw_clamp_off", "olfactory", "--rec_lin", "1", "--win", "10:40", "--win", "40:60", "--reg_last",
           "--occ_win", "10:40", "--region", "ORN=olfactory", "--region", "PN=class:ALPN", "--out", f]
    p = subprocess.run(cmd, capture_output=True, text=True)
    open(f"{OUT}/cli_switch.log", "w").write(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    if p.returncode != 0:
        report("CLI_switch_runs", False, err=p.stderr[-2000:])
    else:
        d = np.load(f)
        names = list(d["reg_names"]); iO = names.index("ORN")
        wm = d["win_mean"]                                     # [g, W, R, B]
        nO = int(d["reg_sizes"][iO])
        mO = wm[0, 0, iO].mean()
        seO = np.sqrt(rhoO * (1 - rhoO) / (nO * 16 * 150) * (2 - P_OFF - paO) / (P_OFF + paO))
        others0 = [names.index(k) for k in ("AL_module", "CNS_rest", "PN")]     # AL_rest contains the ORNs
        silent_g0 = bool(np.all(d["reg_rho"][0][:, others0, :] == 0))
        tt = d["t"]; aft = tt > 40 + 1e-9
        orn_zero = bool(np.all(d["reg_rho"][:, aft, iO, :] == 0))
        last_ok = bool(np.all(np.isclose(d["reg_last"][:, iO, :], 40.0)))
        report("CLI_ORN_isolated_rate_g0", abs(mO - rhoO) < 5 * seO, expected=rhoO, measured=mO, se=seO)
        report("CLI_nonORN_silent_g0", silent_g0)
        report("CLI_ORN_zero_after_removal", orn_zero and last_ok, reg_last_ORN=np.unique(d["reg_last"][:, iO, :]).tolist())
        report("CLI_PN_activity_g0.3", True, PN_driven=float(wm[1, 0, names.index("PN")].mean()),
               PN_after=float(wm[1, 1, names.index("PN")].mean()))
    # swc.py vs ald_sim with the extra outputs switched on (no drive): identical at swc.py's record times.
    # One g per run: with several g values the global torch RNG stream of the next g depends on WHEN the early stop of a
    # dying g is detected, i.e. on the record grid (swc.py checks only at its own record points).
    for gtag, gval in (("dies", "0.25"), ("survives", "0.35")):
        base = f"--exp decay --norm raw --r 1 --g {gval} --T 300 --B 16 --seed 3".split()
        fs, fa = f"{OUT}/eq_swc_{gtag}.npz", f"{OUT}/eq_ald_extra_{gtag}.npz"
        p1 = subprocess.run([PY, f"{S.ROOT}/code/swc.py"] + base + ["--out", fs], capture_output=True, text=True)
        p2 = subprocess.run([PY, f"{S.ROOT}/code/ald_sim.py"] + base + ["--rec_lin", "2", "--win", "100:300",
                                                                         "--reg_last", "--out", fa], capture_output=True, text=True)
        open(f"{OUT}/eq_{gtag}.log", "w").write(p1.stdout + p1.stderr + p2.stdout + p2.stderr)
        A, Bz = np.load(fs), np.load(fa)
        idx = np.searchsorted(Bz["t"], A["t"])
        same_t = bool(np.allclose(Bz["t"][idx], A["t"]))
        eq = {k: bool(same_t and np.array_equal(A[k], Bz[k][:, idx])) for k in ("rho", "surv")}
        eq["occ"] = bool(np.array_equal(A["occ"], Bz["occ"]))
        # occ after an early stop is divided by the steps actually run (swc.py quirk), which depend on the record grid
        died = bool(A["surv"][0, -1].sum() == 0)
        ok = eq["rho"] and eq["surv"] and (eq["occ"] or died)
        report(f"EQ_swc_vs_ald_extra_outputs_g{gval}", ok, same_record_times=same_t, identical=eq, died_early=died,
               n_t_swc=len(A["t"]), n_t_ald=len(Bz["t"]))

results["_meta"] = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
                    "seconds": time.time() - t0, "n_pass": sum(v["pass"] for k, v in results.items() if k != "_meta"),
                    "n_tests": sum(1 for k in results if k != "_meta")}
json.dump(results, open(f"{OUT}/drive_tests{'_cli' if '--cli' in sys.argv else ''}.json", "w"), indent=1, default=str)
print(json.dumps(results["_meta"]))
