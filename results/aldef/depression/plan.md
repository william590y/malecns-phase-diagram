# E3 — synaptic depression, adaptation, refractoriness: plan and pre-registered criteria

Written 2026-11-09 ~07:45, before any production run (args: `code/ald_dep_args.py` → `args_stage1.txt`, `index_stage1.json`).

## Question
Does short-term synaptic depression (or adaptation / refractoriness) turn the self-sustained AL state at h = 0 into a
transient one? Hypothesis under test (from the workflow, not assumed): depression leaves the linear onset g_lin unchanged
(at silence all resources are full) but lowers the active-state amplitude and the metastable lifetime.

## Simulator change (new option, default unchanged)
`ald_sim.py --std_mode spike` (Tsodyks–Markram per spike; one active epoch = one spike; U = fraction used per epoch =
1 − f). The existing `--std` (now `--std_mode rate`, the default) depletes continuously DURING every active epoch, so even a
fully recovered neuron's first epoch is weakened by φ = x∞ + (1 − x∞)(1 − e^{−dt})/(1 − e^{−(1+k)dt}); in rate mode the
silent-state branching operator is scaled by φ and g_lin moves up by ≈ 1/φ (0.82 → ×1.22 for U 0.25, τ 89; 0.77 → ×1.30
for U 0.34, τ 240). The hypothesis "g_lin unchanged" therefore holds only for the spike mode; both are run.

## Runs (stage 1; all: decay from fully active start, raw norm, T = 3000, B = 64, push engine, dt 0.2)
g grid: 16 log-spaced points 0.15–0.6 + 11 log-spaced points 0.6–8 (27 points), 3 interleaved tasks per arm.
r = 1 unless noted. Literature parameters (`results/aldef/lit_biophysics_calc.json`, τ_u = 10 ms):
- base (r = 1, r = 0)
- spike mode, all presynaptic neurons, U = 1 − f: KW09 (0.28, 240), N15 (0.22, 89), NW16 (0.25, 157; main), slow
  (0.09, 63); weak series U 0.25 with TAU 3, 10, 30
- rate mode, all neurons, U from `U_geometric_epoch`: KW09 (0.34, 240), N15 (0.25, 89), NW16 (0.29, 157), slow (0.09, 63)
- spike mode at the main point (0.25, 157) on one population only: excitatory outputs (sign +1), ORNs, PNs (class ALPN),
  LNs (class ALLN), lLN1_bc only, the AL (al ∪ module) only, everything except the AL
- adaptation, all neurons, TAUA 50: D·TAUA ∈ {0.1, 0.3, 1, 3}; TAUA 10 with D·TAUA = 1; LNs only (ALLN) D·TAUA = 1
- refractory N ∈ {5, 10, 25, 50} steps (1, 2, 5, 10 time units)
- r = 0 copies: spike NW16, rate NW16, adaptation D·TAUA = 1 (TAUA 50), refractory 10
- driven linear response (quiet start, h on every neuron, g 0.08–0.42, B 64, T 3000, window 1000–3000): base, spike NW16,
  spike KW09, rate NW16 at h = 1e-5 and 1e-4.
Stage 2 (refinement): extra g points where a boundary falls in a coarse grid cell (> ×1.15) or beyond the grid.

## Measures
- Lifetime: per replica extinction time (first record with no active neuron), censored at 3000; τ = censored-exponential
  MLE = Σ min(t_i, T)/deaths with the chi-square CI; boundaries S (τ = 200) and on (τ = 3000) by the analysis2 method
  (isotonic ln τ, first crossing interpolated in ln g, bracket + replica bootstrap). Also τ_late: same MLE from t0 = 100
  on, among replicas alive at t0 (removes the transient from the fully active start).
- Quasi-stationary amplitude: AL-module active fraction averaged over 1500 < t ≤ 3000 among replicas alive at T.
- AL leadership: share of the activity in the AL module (occupancy, second half), AL-module rate / CNS-rest rate, and
  for dying replicas the fraction in which the AL module is the last region with activity (reg_last).
- Resource: mean efficacy of active depressing neurons in the AL module (res_x_act) in the window.

## Pre-registered reading
1. Linear onset: spike mode — driven response at g ≤ 0.9 g_lin equal to base (ratio CI includes 1); rate mode — shifted
   by ≈ 1/φ. Theory: spike mode has the same λ(g) by construction.
2. Amplitude: QS AL activity lower than base at every g where both are sustained.
3. Lifetime: S and on move up (ratio > 1 with non-overlapping CIs) — "depression shortens lifetimes".
4. "Self-sustained → transient": no point with τ ≥ 3000 on the grid (up to g = 8) for the arm, OR on moves above the
   whole E1 allowed range (r = 1 upper end 0.289) AND the AL is not the sustained region any more. A move of the onset
   alone is a quantitative change, not a qualitative one; reported as such.
5. Theory check: the depression-extended birth–death chain (frozen-x and adiabatic variants, `code/ald_dep_theory.py`)
   predicts the RATIO on(arm)/on(base) and S(arm)/S(base) within the sim CIs (pass), within ×1.25 (rough), else fail.
   The base theory misses r = 1 by ~×1.8 (K too small), so ratios are the fair test; r = 0 absolute values also compared.
