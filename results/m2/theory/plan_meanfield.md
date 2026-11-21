# Lane m2/theory, part "meanfield": plan and pre-registered criteria

Written 2026-11-09 ~17:40, after the smoke test (array m2-theory-smoke 209306; outputs `runs/smoke_*`) and BEFORE any
production run. I have not opened any output of the experiment lanes (`results/m2/{decay,driven,nature,modules}`,
`figs/m2/*`) and will not before the predictions file `meanfield_predictions.json` is written. The M1 reference
(`results/analysis2/flow_sim_boundaries.json`, `results/pd3/*`, `results/aldef/*`) is used, and only as named below.

Spec followed: `results/m2/MODEL2.md` (model, variants, protocols, windows, phase definitions, grids). Deviations are listed
at the end.

## 1. What is computed

| piece | code | output |
|---|---|---|
| single-unit per-spike TM in the synchronous update: exact moment map and renewal efficacy, Poisson limit | `code/m2_mf_core.py` | `derivation.md` §1-2, tests |
| g_lin (exact linearisation of the stochastic process about the RESTED silent state), g_lin(x) = g_lin/x, QMF silent edge alpha/mu(r) | `code/m2_mf_glin.py` | `meanfield_glin.json` |
| network mean field with resources, two closures (below), protocols decay_h0 and driven (+ removal), all M2 r values | `code/m2_mf_run.py` (GPU array `m2-theory-*`) | `runs/*.npz` |
| linearisation: exact JVP (QMF) / central differences (CQMF), ring-down of population sums -> matrix-pencil modes; renormalised branching ratio sigma_x | same | `runs/*.npz` |
| branching birth-death chain with resources (theory_branch rows + mode vectors; frozen and adiabatic resource limits) | `code/m2_mf_chain.py` (CPU) | `meanfield_chain.json` |
| analytic slow-fast (activity-resource) reduction: no Hopf, damped-oscillation period and coherence, crash after the fully active start | `derivation.md` §5-6 | numbers in `meanfield_predictions.json` |
| analysis / phase lines / predictions | `code/m2_mf_analyze.py` | `meanfield_*.json`, `figs/m2/meanfield_*.png` |

### Closures (the only approximations of the network mean field)
- **QMF**: p_i = f(E[s_i]) (mean input; exactly M1's `theory_qmf.py` closure, extended with resources).
- **CQMF** (coincidence-corrected): with S_E, S_I >= 0 the excitatory and inhibitory input, E[(S_E - S_I)_+] = mu_E -
  E[min(S_E, S_I)], and for independent inputs E[min] <= min(mu_E P(S_I > 0), mu_I P(S_E > 0)). CQMF uses
  s_i = mu_E - min(mu_E pi_I, mu_I pi_E) (pi = 1 - prod(1 - rho) over the inputs of each sign). It equals QMF at r = 0 and
  at dense activity, and, unlike QMF, keeps inhibition at second order near silence, as the exact linearisation does
  (g_lin is r-independent). Its drive is a tighter lower bound on E[(S)_+] than QMF's.

## 2. Pre-registered criteria

**Closure validation on M1 (V1, decides the primary closure for the network predictions).** M1 runs of both closures,
R_CORE, decay protocol. D_MF(r) = first g (isotonic, ln g interpolation as `m2_obs.first_crossing`) with PR_all >= N/4
over the window 1500-3000, among MF-sustained points. Compare with the M1 simulated D (`flow_sim_boundaries.json`:
1.47 / 2.50 / 4.44 / 14.5 at r = 0 / 0.25 / 0.5 / 1; none up to 30 at r = 2, 4, 6). Score = mean |ln(D_MF / D_sim)| over
r in {0.25, 0.5, 1} (a "none" counts as |ln(30 / D_sim)|) plus 0.5 per wrong "none / reached" call at r in {2, 4, 6}.
Lower score = primary closure. Both closures are reported everywhere.
V2 (sanity, not used for selection): M1 driven at ORN 0.04, the MF gain curve G(g) and its G = 0.29 crossing vs E1's
allowed lower ends (r = 0 ~0.09-0.10, r = 1 ~0.10-0.12; `results/aldef/REPORT.md`).

**Setting (a), decay_h0, protocol IC (fully active, e = 1, x = 1 - U), MF + cutoff.** n_all(t) = sum_i rho_i(t).
- tau_MF = first t with n_all(t) < n_c, n_c = 1 (0.1 and 10 reported as sensitivity); censored at 3000.
- Q: tau_MF < 200; S: 200 <= tau_MF < 3000; sustained: never below n_c by T = 3000; L / D: PR_all of the occupancy
  over 1500-3000 < / >= N/4.
- S_MF, on_MF, D_MF: first crossings as above.
- Interpretation fixed now: P(alive at t) <= E[n(t)], so a dip of the mean-field count below 1 bounds survival; but the
  mean field ignores fluctuation-driven extinction of a small sustained state. **S_MF and on_MF are therefore lower
  bounds for the simulated S and onset** (heuristic: the QMF overestimates activity for r = 0 by Jensen; for r > 0 QMF
  over-inhibits, so the bound may fail there; CQMF is the closure for which the bound is expected to hold).
- **Bistability**: decay-IC vs eps-IC (silent + 1e-6, rested) end states. Bistable if both are stationary (relative
  peak-to-peak of n_all over the last 500 units < 1e-3), max(n) >= 1 and |n_A - n_B| / max(n_A, n_B) > 0.1. The g edges
  of bistable intervals are the saddle-node (fold) bracket.
- **Mean-field oscillation (Hopf / limit cycle)**: relative peak-to-peak of n_all over 2000-3000 > 1e-3 AND the amplitude
  over 2500-3000 >= 0.5 x that over 2000-2500 AND a dominant period <= 333 (>= 3 cycles). Hopf line = first g flagged.

**Setting (b), driven (ORN h = 0.037926 etc.), protocol IC quiet; MF + cutoff after removal.**
- rho_ORN, rho_drv (nonORN), G = rho_uPN_ant / rho_ORN, PR_nonORN from the 500-1500 means (MODEL2 5b).
- tau_post = first t > 1500 with n_nonORN < 1, minus 1500 (censored at 3000). S1: uPN mean over 1600-1800 <= 10% of the
  driven value. S2: tau_post < 3000 or n_ALsys(4500) < 0.05.
- Removal class and phases exactly as MODEL2 5b (silent / transient / self-sustained; SS, DT, DA, DQ; -D, -NS, -O with
  the MF oscillation rule on n_nonORN over 500-1500, period <= 333).
- Allowed region = G in [0.29, 1.45] AND S1 AND S2. Because removal in the MF ignores finite-size extinction, the MF
  removal classes and SS / DT lines are lower bounds (the simulated SS / DT lines should lie at or above them); the G
  curve itself is a direct prediction.
- Bistability: quiet-IC vs decay-IC (fully active + drive) at t = 1500, same rule.

**Linear stability (R_CORE; decay end state, driven state at T1).** Ring-down of the linearised map for 1000 units after
(i) a demographic kick (delta rho_i ~ N(0, rho_i(1 - rho_i))) and (ii) a uniform 0.1% kick; population sums per region.
Matrix pencil (model order <= 16) on the nonORN (driven) / all (decay) sum, 1-unit sampling. Reported: slowest real mode
tau_1 = -1/ln|z|, the leading oscillatory pair (largest amplitude with period <= 375) with period P and per-period damping
D_P = |z|^P. A mode is called robust if the two kicks give periods within 10%.
- Predicted "OSC-strong possible" only if a limit cycle exists or the leading oscillatory pair has D_P >= 0.7.
- The slow-fast analytic reduction predicts D_P <= exp(-2 pi / sqrt(alpha TAU)) = 0.59 for M2 (derivation §5): a
  pre-registered analytic prediction that the network computation tests.
- sigma_x = lambda_PF(f(g E diag(x_next)))/alpha_eff (frozen resources of the analysed state; excitatory branching
  operator) and sigma_1 (rested, x = 1) are reported; no criterion attached (descriptive).

**Branching chain with resources (`m2_mf_chain.py`).** The M1 dense theory rows (`results/pd/branch_real_dense_part*.json`:
lambda(g), K_r(g) for 23 r) and mode vectors (`results/aldef/depression/theory_vecs.npz`) with the exact M2 efficacy e(rho):
frozen-resource chain (birth lambda X* n (1 - n/K)) and adiabatic chain (lambda X(n)); lifetime from the top state; S and on
lines at tau = 200 / 3000. Primary: frozen chain with K = kappa_r K_eff, kappa_r fitted per r so that the undepressed chain
reproduces the M1 simulated onset (`flow_sim_boundaries.json`, the aldef procedure). Alternates: kappa = 1; adiabatic.

## 3. Grids and runs
- g: `m2_common.G_COARSE` (40 points 0.05-30). r: R_ALL (23) for M2 decay and driven at ORN 0.04, both closures;
  R_CORE for M1, the variants (M2s, M1s, M2_allsyn, M2_KW09, M2_N15, M2_tau30), the other ORN levels and the nulls
  (configuration / weight-shuffled, null_seed 0).
- Main M2 and M1 runs carry both initial conditions; the variant / level / null runs carry the protocol IC only.

## 4. Deviations from MODEL2.md (and why)
- The mean field has no replicas: tau is a cutoff time, not a censored MLE; PR uses the deterministic occupancy.
- No FSS / avalanche / Levina predictions here (they need fluctuations: the stochastic lane). A mean-field hysteresis
  (ramp) computation is optional and done only if time allows.
- The oscillation thresholds for the deterministic trajectory (1e-3, >= 3 cycles) are mine; MODEL2's OSC flag is an ACF
  rule for noisy data and is not applicable to a deterministic trajectory.
- MODEL2.md §1 states c = 1/742.7; the simulator (and this code, which calls its functions) uses c = 1/745.91.

## 5. Amendment A1 (2026-11-09 ~17:55; written before the M1 decay validation runs had finished and, as before, without any experiment-lane output)
The smoke and first variant runs (my own mean-field outputs) confirm that the deterministic mean field keeps a small
sustained state at every g above its silent edge, so its cutoff lines are weak lower bounds. One calibrated finite-size
heuristic is added, with its own validation:
- **Equal-size rule.** n*_MF(g) = expected number of active neurons in the QS window of the decay protocol (decay IC).
  From the M1 runs (R_CORE, each closure): n_cal(r) = n*_MF at the M1 simulated onset (flow_sim_boundaries.json, ln g
  interpolation). **Valid** only if max/min of n_cal over the R_CORE r values is <= 3; then n_cal = their geometric mean
  and the predicted onset is the first g with n*_MF >= n_cal (S analogously with the M1 S line). Otherwise the rule is
  reported as invalid and not used.
- Expected direction (stated now): depression makes the sustained state near-critical (smaller restoring drift), so at
  equal size it dies faster; the equal-size rule should therefore UNDER-estimate the M2 onset. It is a heuristic point
  estimate between the cutoff lower bound and the AL-chain upper bound.
- The same rule on the driven runs: the predicted removal SS line is the first g where n_nonORN at T = 4500 >= n_cal.

## 6. Correction C1 (2026-11-09 ~18:25; my own outputs only, no experiment output read)
The first CQMF version let inhibition cancel only the synaptic excitation g mu_E, not the external drive h. So under
CQMF the driven ORNs were shielded from LN inhibition (measured rho_ORN stayed at 0.040 at every g, unlike QMF). Fixed in
code/m2_mf_core.py: A = h + g mu_E, s = A - min(A pi_I, g mu_I P(A > 0)) with P(A > 0) = 1 where h > 0. Every h = 0 run
(decay protocol, V1, A1) is unchanged. All 90 CQMF driven runs were redone (array m2-theory-drv2 211907); the superseded
files are in runs/superseded_cqmf_driven_v1/ and are not used.
