# Program `criticality`: is the steep, smooth crossover a (finite-size-rounded) second-order transition? (pd3, 2026-11-09)

Plan section 4 of `results/pd3/plan.md` (C0-C4 + pre-registered Q2 criteria). Every number below is read from one of the files
named next to it; the machine-readable summary is `results/pd3/criticality.json` (built by `code/pd3_crit_summary.py` from
`results/pd3/criticality/aval_analysis.json`, `qs_analysis.json`, `v0_validation.json`, `c0_drive.json`). Job ids:
`results/pd3/criticality/JOBS.md`. Notation: x = g / g_lin of the graph in question (real brain g_lin = 0.16315, AL module alone
0.16428, configuration nulls their own g_lin); time in model units (1/alpha); K = effective module size of the one-mode logistic
birth-death chain (finite-K mean-field directed percolation, MF-DP); K_needed = 75.8 (r = 0), 36.6 (r = 1), 16.2 (r = 6)
(`phase_complete.json`). "Post hoc" marks every analysis that was not in the pre-registered plan.

## 0. Short answer, and how strongly each question is answered

| question | what this program found | strength |
|---|---|---|
| **Q2** "is this steep but smooth crossover a second-order phase transition?" | **Partly, and only as a finite-size-rounded crossover** [wording changed in verification]. (i) No first-order signature: no bimodal quasi-stationary (QS) distribution at any x or r tested, and (module_theory, quoted) no mean-field hysteresis at the onset (in a K ~ 10-90 system unimodality is weak evidence of continuity). (ii) The linear critical point is g_lin, approximately r-independent: the growth-rate zeros x* follow the *exact-linearisation* finite-K chain at K_needed at r = 0, 1 and 6 and not the net-input closure (x* = 1.42 / 2.10 predicted at r = 1 / 6; measured 1.27-1.35 / 1.63-1.72); a verification fit with K free (independent of K_needed) gives an effective branching scale c(r)/c(0) = 0.98-1.00 at r = 1 (closure 0.91) but 0.94-0.95 at r = 6 (closure 0.84; poor fit), so r-independence holds at r <= 1 and is approximate (~5% in R) at r = 6 (section 3.1); large-K configuration nulls have x* = 1.018. (iii) Lifetime at g_lin ~ K^0.50 +- 0.01 (r = 0) and K^0.46 +- 0.04 (r = 1) holds only against the *fitted* K (14-2,900; fitted from the same chain's lifetimes at x 1.2-2.0, so partly circular); against the structural K of the pre-registered test (K_n, K_nc) the within-family slopes are 0.37-0.38 (r = 0) and 0.25 (r = 1), below the chain's own 0.51-0.56 over the same K (section 6). **But** (iv) it is *not* MF-DP of a single mode at the level of avalanches: the mean size grows with duration with a local slope peaking at 2.10-2.35 (same-estimator chain 1.58-1.69), small avalanches are in excess (size-exponent MLE 1.74-1.83 with S_min = 2, but 1.46-1.57 with a KS-chosen S_min of 10-30 spikes, i.e. near 3/2 above ~20 spikes; section 3.2), the duration distribution needs a K 1.7-3.3x below K_needed (0.9-1.6x with
mode-weighted seeds) and still deviates from the chain's shape by 17-49%, the QS order parameter rises faster than the chain (slope 1.15-1.28 vs 0.53-0.79 in the pre-registered window; the data exceed the chain in every window, but the slope itself is window-dependent, 0.59-1.28), the module-susceptibility peak
sits at the chain's g only at r = 0 (+7%; +18% at r = 1), and the field response has no slope-1/2 window. The degree/weight-preserving configuration nulls deviate the same way (more strongly), which points to network heterogeneity (hypothesis), not AL competition. | no first-order signature: **moderate** (QS unimodal, mean field continuous; stochastic hysteresis **not tested here**); location at g_lin: **supported** at r <= 1, **approximate** at r = 6; rounding by K: **only with fitted K** (fails with the pre-registered structural K at r = 1, marginal at r = 0); "MF-DP universality class": **not supported** (clear negative, several observables); "phase transition of the brain": **no** (K small and fixed). |
| **Q1** de Candia-style diagram: broken-symmetry (B1) signatures? | Avalanche shapes at g_lin are nearly symmetric (skewness -0.07 ... +0.04 at T ~ 10, ~0 at T ~ 35-45) - far from de Candia's +0.4 on SL-B1 - and the minority antennal lobe's share of spikes **rises** late in long avalanches at every r (no competition). Inhibition only lowers the cross-lobe spread slightly. | **strong negative** for an L/R broken-symmetry mechanism at the AL onset. |
| **Q3b** finite-size check | Contributes the nested-AL-sub-module points (K_n 16-67) to the lifetime-vs-K plot with the FSS nulls (`figs/pd3/lifetime_vs_K_criticality.png`); they collapse onto the chain only with fitted K (section 6). | supporting (with fitted K only). |

**Pre-registered verdict.** With the literal criteria of plan 4.6 the outcome is *a fails, b fails, c fails at r = 1 (structural K,
which plan 4.5 specifies: slope 0.21) and passes only literally at r = 0 (0.42, produced by merging two families whose own slopes are
0.37-0.38), passing at both r only with post-hoc fitted K; d fails, e passes (QS unimodality; stochastic hysteresis untested)*
[c corrected in verification; the original text read "c passes (with fitted K)"]. None of the plan's four clean verdicts applies; the
nearest row, "c fails", reads "the rounding is not set by K" - true for the structural K, not for an effective K fitted to lifetimes. Criterion a ("x* in [0.97, 1.03]") was mis-specified: the plan's own chain predicts x* = 1.14 / 1.29 / 1.76
for K_needed at r = 0 / 1 / 6 (finite-K saturation makes E[n(age)] decay at R = 1); measured against that prediction the
location test passes (section 3.1). The failures of b and d are genuine (section 3.2-3.4, 4, 5).

**One-line reading** [revised in verification]. The crossover is the finite-K image of the linear instability at g_lin (r-independent
at r <= 1, within ~5% in R at r = 6), with no first-order signature, rounded by an effective module size K ~ 10-90 (fits: 9.5-23 from durations, 16-76 K_needed, 23-92 from growth rates) that must be fitted
(lifetime ~ K_fit^0.5; the structural K does not predict the lifetimes with the chain's exponent); its avalanche statistics are not
single-mode MF-DP and look like those of a heterogeneous branching network (and so do the nulls'). Because that K is small and does not
grow with brain size, this is a finite-size crossover, not a phase transition of the brain.

---------------------------------------------------------------------------------------------------------------------

## 1. Pre-registered Q2 criteria (plan 4.6), evaluated (`criticality.json` `Q2_criteria`)

| criterion | pre-registered test | result | verdict |
|---|---|---|---|
| **Q2-a** location | growth-rate zero x* in [0.97, 1.03] for r = 0, 1, 6 | x* = 1.186 [1.184, 1.190] (r 0), 1.345 [1.341, 1.349] (r 1), 1.722 [1.717, 1.728] (r 6; from the post-hoc x extension) | **FAIL** as written; the finite-K chain predicts 1.138 / 1.291 / 1.757 (section 3.1) |
| | chi_QS peak within +-10% of g = 0.195 (r 0) / 0.209 (r 1) | chi_AL peak g = 0.2080 (r 0, +6.7%), 0.2473 (r 1, +18.3%); (r 6: 0.336 vs chain 0.233) | r 0 **PASS**, r 1 **FAIL** |
| **Q2-b** finite-K MF-DP | P(T) at x = 1 matches the chain with one K (compensated deviation <= 10% up to T_c, or chi2/dof <= 2) | K_fit = 23.1 (r 0), 13.7 (r 1); max compensated deviation 0.17 / 0.18; chi2/dof 4,784 / 4,121 (N = 2e6) | **FAIL** |
| | that K within x1.5 of K_needed | 23.1 / 75.8 = 0.30; 13.7 / 36.6 = 0.37 | **FAIL** |
| | <S>(T) local-slope peak 1.5-1.75 on the real graph, rising toward 2 with K | 2.349 [2.329, 2.367] (r 0), 2.242 [2.224, 2.259] (r 1); module 2.17; nulls 2.58-2.68 (robust to the binning: 2.27-2.41 for r 0 with bin ratio 1.15-1.4 or a 5-bin slope; verification) | **FAIL** (peaks above 2; they do rise with K) |
| **Q2-c** rounding by K | tau_top(x = 1) ~ K^(0.5 +- 0.1) over nested sub-modules + nulls (plan 4.5: against K_n, K_nc) | unfitted K (K_n, K_nc), as pre-registered: 0.421 +- 0.010 (r 0), 0.206 +- 0.010 (r 1); within families: sub-modules 0.376 / 0.247, nulls 0.372 / 0.253 (r 0 / r 1), chain's own local slope 0.56 (K 16-67) / 0.51 (K 103-5,631); post-hoc fitted K: 0.496 +- 0.007 (r 0), 0.465 +- 0.036 (r 1) | **FAIL** as pre-registered at r 1; r 0 passes only literally (merged families); PASS with fitted K (partly circular, section 6) [verdict corrected in verification] |
| **Q2-d** exponents | n_QS within 20% of the chain at the same K | ratio data/chain 0.47 (x 0.95) ... 1.32 (x 3.1) at r 0; 0.44 ... 1.75 at r 1 | **FAIL** (within 20% only for x ~ 1.4-2.2 at r 0, ~1.4-1.7 at r 1) |
| | field slope <= 0.65 at x = 1 while >= 0.85 at x = 0.9 | r 0: x 1 min 0.645 (only in the last h interval, n_AL 27-55); x 0.9 min 0.727. r 1: x 1 min 0.771; x 0.9 min 0.820 | **FAIL** |
| **Q2-e** not first order | unimodal QS histograms; no hysteresis | all QS n_AL histograms unimodal (interior test, section 4); mean-field up/down sweeps identical at the onset (module_theory.md); stochastic sweeps not run here | **PASS** (QS, mean field); stochastic hysteresis: multistability program |

---------------------------------------------------------------------------------------------------------------------

## 2. V0 validation of the new frontier simulator (`v0_validation.json`)

`code/pd3_avalanche.py` (event-sparse, synchronous, swc.py's rules) against swc.py `--exp spread` on the full brain, r = 1, seeds
uniform over the 366 excitatory AL-module neurons (`criticality/seeds_module_exc366.npz`), recorded steps <= 3,000 (94 points).

| g | dense reference | P(t): points beyond 2 SE, max abs z | E[n(t)]: beyond 2 SE, max abs z | KS on extinction times | verdict |
|---|---|---|---|---|---|
| 0.20 | swc.py B 512 (pre-registered) | 4 / 94, 2.48 | 9 / 94, 3.04 | D 0.034, p 0.58 | E[n] fails |
| 0.36 | swc.py B 512 (pre-registered) | 0 / 94, 1.28 | 17 / 94, 2.73 | D 0.026, p 0.90 | E[n] fails |
| 0.20 | swc.py B 4,096 (follow-up, other seed) | 0 / 94, 1.53 | 0 / 94, 1.34 | D 0.012, p 0.60 | **pass** |
| 0.36 | swc.py B 2,048 (follow-up, other seed) | 2 / 94, 2.16 | 0 / 94, 1.55 | D 0.026, p 0.20 | pass except 2 P points at abs z 2.1-2.2 |

- Both B = 512 references had a correlated excess of E[n(t)] (heavy-tailed: a few large avalanches dominate a 512-replica mean).
  Two checks showed the frontier is right: (i) the exact E[n(1 step)], computed from the graph by summing each seed's out-edges, is
  1.1253 (g 0.20) and 1.3696 (g 0.36); the frontier gives 1.1256 (200,000 avalanches) and 1.3613 (10,000); (ii) larger dense
  references agree within 2 SE at every recorded t for E[n]. The remaining 2 / 94 P(t) points at abs z <= 2.16 (g 0.36) are within
  what 94 correlated tests give by chance; strictly, the "every t within 2 SE" criterion is missed there.
- z-scores use the pooled two-sample variance (a dense run with all 512 replicas dead has SE 0 otherwise).
- Timing: 11.7 ms/step and 0.042 ms per avalanche at g 0.20 (RTX PRO 6000 Blackwell); 106 ms/step and 32 ms per avalanche at
  g 0.36 (RTX 6000 Ada; ~2,100 of 10,000 avalanches reach the cap of 3,000 steps). Production at x = 1: 2e6 avalanches in 14-64 s
  per task (wall time incl. loading).
- Plausibility (not a criterion): P(alive at t = 100) at g 0.36 = 0.226 +- 0.004 (frontier) vs 0.237-0.244 (dense); the earlier
  `spreading.md` 0.486 used seeds ~ |v|^2, not uniform seeds.

## 3. C1: single-seed avalanches (`aval_analysis.json`, `figs/pd3/aval_*.png`)

Production: 35 spec tasks (real r 0 and 1 at 10 x each; r 0.5 / 2 / 4 / 6 at x 1; r 6 at 0.98 / 1.02; AL module alone r 0 / 1;
full configuration null r 0 at x 0.98 / 1 / 1.02 and r 1 at x 1; FSS nulls N 5k / 20k / 80k), plus post-hoc controls: complete
graphs of K = 37 / 76 units at the same 10 x and K = 300 / 2,000 at x 1 (the synchronous SIS dynamics of a complete graph is the
discrete-time version of the one-mode logistic chain, run by the same code and estimators), and real r 6 at x 1.20-2.30 (6 tasks).
N_av = 2e6 (x <= 1.05), 5e5 (1.10), 2e5 (>= 1.2); nulls 1e6. Every avalanche ended or reached the cap (status counts in the json).

### 3.1 Where is the critical point? (growth rate of E[n(age)], `aval_critical_point.png`)

gamma = slope of ln E[n(age)] over ages 5, 10, 20, 35, 50 steps (per model time; CLT bootstrap of the per-avalanche n(age)).
At finite K even R = 1 gives gamma < 0 (saturation n^2/K), so x* > 1 is *predicted*; the exact continuous chain with the
connectome's R(x) gives the reference values.

| | x*: uniform seeds (pre-registered) | x*: seeds weighted by v (post hoc) | chain, exact linearisation (K_needed) | chain, net-input closure (R x mu(r)/mu(0)) |
|---|---|---|---|---|
| real r 0 | 1.186 [1.184, 1.190] | 1.123 [1.120, 1.126] | 1.138 (K 75.8) | = exact at r 0 |
| real r 1 | 1.345 [1.341, 1.349] | 1.271 [1.266, 1.276] | 1.291 (K 36.6) | 1.418 |
| real r 6 | 1.722 [1.717, 1.728] | 1.627 [1.620, 1.635] | 1.757 (K 16.2) | 2.102 |
| complete graph, 76 / 37 units | 1.126 [1.125, 1.127] / 1.273 [1.271, 1.275] | - | 1.137 (K 76) / 1.287 (K 37) | - |
| full configuration null r 0 | 1.018 [1.016, 1.020] | - | 1.035 (K 300) ... 1.005 (K 2,000) | - |

- Seeds weighted by v_seed (the right Perron vector of the module's branching operator at its g_lin) project the initial state
  on the leading mode, which is what the chain's "n = 1" means; uniform seeds also load the slowly decaying second AL mode
  (lambda_2/lambda_1 = 5.389/5.574) and low-importance neurons.
- The exact-linearisation chain lies inside or next to the range spanned by the two seed weightings at every r; the one-mode
  control sits 0.011-0.014 below the continuous chain (discrete-time effect), comparable to the v-weighted offsets at r 0 and 1
  (0.015, 0.020); at r 6 the v-weighted zero is 0.13 and the uniform-seed zero 0.035 below the chain. The
  net-input closure misses by +0.07 (r 1) and +0.38 (r 6) beyond even the uniform-seed values.
- Estimator sensitivity: regressing on all ages 5-50 instead of 5 ages moves x* to 1.190 / 1.357 / 1.768 (r 0 / 1 / 6).
- r-dependence at x = 1 (fig. panel b; `critical_point.r_dependence_x1`): gamma = -0.113 / -0.150 / -0.170 / -0.196 / -0.223 /
  -0.238 for r = 0 / 0.5 / 1 / 2 / 4 / 6 (v-weighted -0.073 / -0.110 / -0.128 / -0.153 / -0.182 / -0.195); exact linearisation +
  chain saturation at K_needed(r): -0.084 / -0.114 / -0.137 / -0.167 / -0.201 / -0.224; closure: -0.084 / -0.148 / -0.190 /
  -0.239 / -0.280 / -0.306. The data's r-dependence (-0.12 to -0.13 from r 0 to 6) matches the saturation term (-0.14), not a
  shift of the linear threshold (closure -0.22).
- **Reading.** The linear critical point is g_lin (theory P1 / T1); what moves with r is mostly the finite-K saturation.
  The pre-registered window [0.97, 1.03] ignored that term.
- **Caveat (verification).** The chain predictions above use K_needed(r), which `phase_complete` derived from lifetimes *assuming*
  the exact linearisation; the closure was not given its own K, so the comparison above is not symmetric. A 2-parameter fit of the
  data gamma(x) with the chain, R = c R_exact(x) and K free (`code/pd3_verify_criticality.py`, `results/pd3/verify_criticality.json`
  `vw_and_Kc_fit`, `figs/pd3/verify_criticality.png` c) gives c(1)/c(0) = 0.978 (uniform seeds) / 0.997 (v-weighted), against
  0.911 for the closure and 1 for the exact linearisation, and c(6)/c(0) = 0.942 / 0.948 against 0.837 / 1, with free
  K = 92 / 44 / 23 (uniform, r 0 / 1 / 6) and 79 / 38 / 23 (v-weighted). The fits are imperfect (chi2/dof 9-32 at r 0-1, 119-131 at
  r 6, with SEs of 0.0005-0.0015). So the closure is rejected at r = 1, but at r = 6 the effective linear branching rate sits ~5%
  below its r = 0 value (a threshold ~5% above g_lin), about a third of the way to the closure - r-independence is approximate there,
  or the one-mode chain is inadequate at r = 6.

### 3.2 Sizes and durations at x = 1 vs the chain (`aval_size_duration.png`)

| graph (x = 1) | <S> | P(T > 10) | K_fit (P(T), uniform seeds) | max compensated dev. at K_fit (at K_ref) | MLE size exponent, [2, S_c/3] | MLE with KS-chosen S_min (verification) | K_fit with v-weighted seeds (post hoc) |
|---|---|---|---|---|---|---|---|
| real r 0 | 10.1 | 0.045 | 23.1 | 0.17 (0.56 at 75.8) | 1.770 | 1.457 (S_min 30) | 48.5 (dev 0.45) |
| real r 1 | 7.35 | 0.036 | 13.7 | 0.18 (0.51 at 36.6) | 1.804 | 1.569 (S_min 25) | 28.1 (dev 0.47) |
| real r 6 | 5.65 | 0.026 | 9.5 | 0.20 (0.32 at 16.2) | 1.825 | (window [2, 35] too short) | 18.0 (dev 0.49) |
| AL module alone r 0 / r 1 | 6.20 / 4.71 | 0.036 / 0.028 | 21.8 / 11.8 | 0.36 / 0.29 | 1.741 / 1.801 | 1.481 (S_min 10) / - | 45.9 / 25.4 (dev 0.20 / 0.21) |
| config. null r 0 (K_nc 5,631) | 44.0 | 0.040 | 130 | 0.58 | 1.770 | 1.562 (S_min 30, upper bound 3,000) | - |
| FSS nulls 5k / 20k / 80k (r 0) | 18.6 / 24.3 / 38.4 | 0.042 / 0.037 / 0.040 | 45.5 / 40.4 / 107 | 0.40 / 0.45 / 0.57 | 1.81 / 1.83 / 1.78 | - | - |
| complete graph 37 / 76 / 300 / 2,000 units | 8.2 / 11.5 / 22.4 / 56.8 | 0.077 / 0.091 / 0.102 / 0.106 | 45.7 / 89.1 / 331 / 2,088 | 0.12 / 0.11 / 0.10 / 0.10 | 1.470 / 1.492 / 1.513 / 1.523 | K 76: 1.351 (S_min 20) | - |

- K_fit = one-parameter multinomial fit of the chain's duration distribution, conditional on T > 1, at the run's R; deviation =
  max |P_data(T > t)/P_chain(T > t) - 1| over t in [1, T_c(K)]. chi2/dof is 1,273-5,164 for every x = 1 run including the
  one-mode controls (1,273-1,929) (N = 1e6-2e6 resolves any model imperfection), so the 10% deviation is the usable criterion; the controls themselves
  sit at 0.10-0.12.
- The controls recover their K within x1.04-1.25 and the MF-DP size exponent (1.47-1.52): the estimators work. The real brain
  and the nulls do not: their avalanches end sooner than the chain at the K that reproduces their lifetimes, and they have an
  excess of small avalanches.
- **Correction (verification).** The original text said the size distribution is "steeper than S^-3/2 below the cutoff". That holds
  only for S below ~10-30 spikes. The truncated MLE depends strongly on S_min (real r 0: 1.770 at S_min 2, 1.666 at 5, 1.554 at 10,
  1.483 at 20, 1.450 at 40, upper bound 166); choosing S_min by the minimum KS distance (Clauset et al.) gives 1.457 (r 0), 1.569
  (r 1), 1.481 (module r 0), 1.562 (configuration null), and the least-squares slope of the log-binned density between 20 spikes and
  the upper bound is -1.48 / -1.57 / -1.50 / -1.57. Above ~20 spikes the connectome's size exponent is therefore close to 3/2, and
  *steeper* than the one-mode control at the same window (K 76: 1.351 at S_min 20, a finite-K bump), by ~0.1-0.2. What the S_min = 2
  number measures is the excess of 1-10-spike avalanches (plausibly uniform seeds that rarely propagate; not tested). `figs/pd3/verify_criticality.png` a.
- Mode-weighted seeds remove part of the gap (K_fit 48.5 vs 75.8 at r 0, within x1.56; 28.1 vs 36.6 at r 1, within x1.30) and
  bring the survival curve onto the K = 76 control up to t ~ 15 (panel c), but the shape deviation stays 0.45-0.49.

### 3.3 Mean size vs duration (`aval_ST_slope.png`)

Local slope of ln <S> vs ln T (3-bin regression, bins x1.25). Peak values [95% CI]:

| graph | peak | post hoc, v-weighted seeds |
|---|---|---|
| real r 0 / 0.5 / 1 / 2 / 4 / 6 | 2.35 / 2.28 / 2.24 / 2.20 / 2.13 / 2.10 | r 0 2.14, r 1 2.01, r 6 1.88 |
| AL module alone r 0 / r 1 | 2.17 / 2.06 | 1.92 / 1.85 |
| configuration nulls (N 5k-167k) | 2.58-2.68 | - |
| complete graphs 37 / 76 / 300 / 2,000 | 1.570 / 1.679 / 1.804 / 1.922 | - |
| exact chain 37 / 75.8 / 300 / 2,000 (plan) | 1.55 / 1.66 / 1.79 / 1.91 (same estimator on the chain MC rows: 1.58 / 1.69 / 1.90 / 2.01) | - |

The peak sits at T ~ 6-10 model time in every graph. The connectome's avalanches grow *faster* than T^2 there (MF-DP's
asymptote) and the excess over the chain at the same K is +0.69 (real r 0 and r 1) and +0.52 (module r 0); +0.46-0.48 with
v-weighted seeds. The degree- and weight-preserving nulls show it more strongly. A one-mode description fails here; a heavy-tailed effective offspring distribution (strong single synapses, hub
neurons) is a candidate explanation that is shared with the nulls (hypothesis, not tested).

### 3.4 Shapes and skewness (`aval_shapes.png`)

Mean skewness of the spike profile (third standardised moment of time weighted by spikes; + = fast rise, slow decay = de Candia's
"left-skewed"), mean +- SE:

| graph (x = 1) | T ~ 9.6 | T ~ 35 | T ~ 44 |
|---|---|---|---|
| real r 0 / 1 / 6 | +0.026 +- 0.002 / +0.032 +- 0.002 / +0.040 +- 0.002 | -0.008 +- 0.006 (r 0) | -0.002 +- 0.010 / +0.057 +- 0.020 / -0.003 +- 0.037 |
| AL module alone r 0 / 1 | -0.066 +- 0.002 / -0.057 +- 0.002 | -0.016 +- 0.007 / +0.001 +- 0.012 | -0.014 +- 0.011 / +0.010 +- 0.020 |
| configuration null r 0 / FSS nulls | +0.058 / +0.054-0.072 | +0.096 / +0.07-0.12 | +0.074 / +0.073-0.088 |
| complete graphs 37 / 76 / 300 / 2,000 | -0.018 / -0.025 / -0.029 / -0.033 | +0.003 ... -0.010 | -0.007 ... +0.001 |
| de Candia et al. 2026, SL-B1 line | | almost +0.4 at T ~ 40 (400 ms = 40/alpha) | |

- The discrete-time one-mode controls are slightly negative at short T (-0.02 to -0.03 at T ~ 10; the continuous chain is
  0 +- 0.02, `theory_bd.json`), so the reference is not exactly 0.
- The real brain exceeds the controls by at most +0.07 at T ~ 5-15 (+0.10 with v-weighted seeds: +0.074 at r 0, T ~ 10) and is
  ~0 for T >= 20; the AL module alone is ~0 or slightly
  negative. The small positive skew at T ~ 10 grows weakly with r (+0.026 -> +0.040, r 0 -> 6) and appears only with the rest of
  the brain attached; the nulls are *more* left-skewed (+0.05 to +0.12), i.e. a structural (degree / weight heterogeneity) effect of the kind
  described by Gleeson & Durrett 2017 (`literature.md`), not competition.
- Normalised shapes (panel b) peak at t/T ~ 0.4 for T ~ 6-13 and flatten for T >= 20. No single exponent collapses them,
  because the <S>(T) slope is not constant (deviation from the plan's T^(1 - gamma_ST) collapse; normalised by the mean instead).

### 3.5 Recruitment of the two lobes and of the rest (`aval_recruitment.png`, `aval_examples.png`)

Shares of spikes in long avalanches (T = 10-50 model time), early (t/T 0.05-0.2) / middle (0.4-0.6) / late (0.8-1.0):

| r (x = 1) | minority-lobe share of AL spikes | seed-lobe share | share outside the AL | late - middle minority share (reservoir sample, 95% CI) |
|---|---|---|---|---|
| 0 | 0.100 / 0.114 / 0.190 | 0.917 / 0.817 / 0.730 | 0.317 / 0.386 / 0.506 | +0.064 [0.046, 0.083] |
| 0.5 | 0.092 / 0.100 / 0.169 | 0.920 / 0.832 / 0.753 | 0.310 / 0.374 / 0.499 | +0.055 [0.039, 0.071] |
| 1 | 0.090 / 0.099 / 0.165 | 0.922 / 0.835 / 0.760 | 0.309 / 0.370 / 0.493 | +0.062 [0.046, 0.078] |
| 2 | 0.084 / 0.091 / 0.153 | 0.924 / 0.845 / 0.775 | 0.308 / 0.369 / 0.486 | +0.067 [0.049, 0.083] |
| 4 | 0.083 / 0.087 / 0.139 | 0.925 / 0.847 / 0.786 | 0.310 / 0.367 / 0.484 | +0.073 [0.054, 0.092] |
| 6 | 0.082 / 0.089 / 0.148 | 0.927 / 0.846 / 0.781 | 0.312 / 0.364 / 0.496 | +0.061 [0.043, 0.079] |
| module alone 0 / 1 | 0.096 / 0.102 / 0.155; 0.087 / 0.088 / 0.136 | 0.920 / 0.835 / 0.768; 0.926 / 0.850 / 0.791 | 0.002-0.003 | +0.058 / +0.052 |

- de Candia's mechanism (cooperative recruitment, then inter-module competition) predicts a late *fall* of the minority share that
  strengthens with inhibition. Observed: the minority lobe's share *rises* late at every r, by the same amount (+0.05 to +0.07,
  CIs overlapping, also at r = 0 where there is no inhibition), and the seed lobe's share falls steadily: activity spreads
  cooperatively from the seed lobe to the other. Inhibition only lowers the level of cross-lobe spread (late minority share 0.190 at
  r 0 vs 0.139-0.165 at r >= 1). This agrees with the neuron-level theory (no B1 state; competition ratio Gamma ~ 0.003).
- At x = 1, 35-37% of all spikes are outside the AL (real brain), rising from ~0.31 to ~0.50 over an avalanche.

## 4. C2: quasi-stationary order parameter and susceptibility (`qs_analysis.json`, `qs_order_parameter.png`, `qs_susceptibility.png`)

QS method (de Oliveira & Dickman; B 256 replicas x 16 stored configurations; burn-in 1,000, measurement 4,000 model time), x at
24 ln points in [0.95, 3.5]. Chain comparison without refitting: QS distribution of the logistic chain at K_needed, chain units x 1.94
= AL neurons.

| graph | chi_AL peak: g (x), height | chain at K_needed: peak g, height | whole-brain chi_QS peak g | beta (chain, same window) | n_AL / chain, x 0.95 -> 3.1 | tau_QS at x 1.58 (chain) |
|---|---|---|---|---|---|---|
| real r 0 | 0.208 (1.275), 8.33 | 0.195, 7.65 (K 75.8) | 0.515 (two maxima: 14.1 at g 0.218, 14.7 at g 0.510) | 1.20 (0.79) | 0.47 -> 1.32 | 1,727 (1,636) |
| real r 1 | 0.247 (1.516), 7.84 | 0.209, 4.94 (K 36.9) | 0.256 | 1.28 (0.68) | 0.44 -> 1.75 | 51.8 (75.0) |
| real r 6 | 0.336 (2.060), 7.00 | 0.233, 2.85 (K 16.2) | 0.350 | 1.15 (0.53) | 0.35 -> 1.81 | 11.5 (13.5) |
| AL module alone r 0 | 0.218 (1.326), 7.47 | 0.196, 7.65 (K 75.8) | = chi_AL | 1.18 (0.79) | 0.59 -> 1.24 | 950 (1,611) |
| AL module alone r 1 | 0.255 (1.551), 7.00 | 0.211, 4.94 (K 36.9) | = chi_AL | 1.21 (0.68) | 0.63 -> 1.77 | 54.4 (74.5) |

(chi heights in neuron units, chain x 1.94; peak = parabola in ln x through the three highest points; replica-bootstrap bands in
`qs_analysis.json`. For the module alone the brain's K_needed is used; its own fitted K is ~70 / ~37, `redraw_fss.md`.)

- **Location.** At r 0 the module-susceptibility peak is where the chain puts it (g 0.208 vs 0.195, +7%), with a similar height
  (8.3 vs 7.7). With inhibition the measured peak moves to larger g than the chain at K_needed (r 1 +18%, r 6 +44%) and is higher
  (7.8 vs 4.9; 7.0 vs 2.9): the QS fluctuations of the inhibited module are not those of one mode with K_needed (which was fitted to
  extinction times from a fully active start).

- **Unimodality.** Tested on n_AL >= 3 (smoothing sigma 1): every histogram has one interior maximum. Two boundary features are not
  coexistence: P(n_AL = 0) (samples with only non-AL activity; the QS state is conditioned on total activity) falls from 0.21 (r 0,
  x 0.95) to 0.001 at x 1.58 and below 0.001 above; and at every x, also in the module alone, n_AL = 1 is ~1.5x the n = 2-3 level (the exact chain QSD has
  no such edge excess; a plausible cause is a last active neuron that cannot propagate - 160 of 526 module neurons are inhibitory or
  modulatory - hypothesis).
- **Whole-brain chi_QS** (= N var(rho)/<rho>) has a second maximum near g ~ 0.5 at r 0 (grid maximum 14.7 at g 0.510; parabola 0.515), close to the first
  non-AL island threshold (0.482, `theory_notes.md` 1.7): the island cascade is a second rounded crossover, not a jump.
- **Exponent beta** (slope of ln n_AL vs ln(R - 1) over (R - 1) sqrt(K) in [2, 8]) is 1.15-1.28 in every graph, while the chain
  gives 0.53-0.79 in the same window (MF-DP asymptote 1): the AL fills up faster than one mode with constant K.
  [verification] This is an effective slope across the finite-K crossover (R - 1 = 0.26-1.9), not an asymptotic exponent, and it
  depends on the window: data / chain = 0.98 / 0.64 ([1, 4]), 1.20 / 0.79 ([2, 8]), 0.79 / 0.58 ([4, 16]), 0.59 / 0.42 ([8, 32]) at
  r 0; at r 1 0.98 / 0.56, 1.28 / 0.68, 0.88 / 0.52, 0.63 / 0.40. The robust statement is that the data exceed the chain's slope in
  every window (by 0.2-0.6), not the value 1.15-1.28.
- QS lifetimes are ~25-40% shorter than the chain's near threshold (r 0: 4.9 vs 7.1 at x 0.95; r 1: 51.8 vs 75.0 at x 1.58) and
  reach it at larger x (r 0: 1,727 vs 1,636 at x 1.58; r 1: 7,877 vs 6,701 at x 2.22).

## 5. C3: field scaling (`qs_analysis.json` C3, `field_scaling.png`)

Drive h on every neuron, from silence, burn-in 500, measurement 2,000 model time, B 128, h = 1e-7 ... 3e-2 (12 values).

| r, x | n_AL at h = 1e-5 / 1e-3 / 1e-2 / 3e-2 | local slope over the last three h intervals (n_AL at mid-interval) | exact driven chain (K_needed), minimum slope |
|---|---|---|---|
| r 0, 0.90 | 0.036 / 2.91 / 22.1 / 49.1 | 0.926 (4.8), 0.839 (13.3), 0.727 (33.0) | 0.53 |
| r 0, 1.00 | 0.042 / 4.20 / 27.3 / 55.4 | 0.871 (6.8), 0.758 (17.3), 0.645 (38.9) | 0.45 |
| r 0, 1.10 | 0.093 / 6.71 / 33.8 / 62.0 | 0.778 (10.3), 0.632 (23.1), 0.553 (45.8) | 0.38 |
| r 1, 0.90 | 0.023 / 2.08 / 14.2 / 34.9 | 0.843 (3.3), 0.826 (8.6), 0.820 (22.2) | 0.56 |
| r 1, 1.00 | 0.040 / 2.63 / 16.3 / 38.1 | 0.812 (4.1), 0.774 (10.3), 0.771 (25.0) | 0.48 |
| r 1, 1.10 | 0.054 / 3.61 / 18.9 / 41.6 | 0.730 (5.4), 0.710 (12.3), 0.717 (28.1) | 0.42 |

- The local slope d ln n_AL / d ln h stays near 1 (linear response, non-overlapping avalanches) while n_AL < ~1 at all x, and falls
  only once n_AL reaches ~10-50, i.e. ~K, at every x. The driven K-unit chain dips to 0.38-0.56; the AL does not (its minimum, 0.55-0.82, is reached at the largest h). There is no
  window with slope ~1/2 (delta_h = 2) and no clear separation between x = 0.9 and 1.0 at matched n_AL (r 0, n_AL ~ 10-27:
  0.84 at x 0.9 vs 0.76 at x 1.0). With K ~ 40-80 the window sqrt(K) << n << K is less than a decade, and the drive also
  activates all 526 AL neurons directly, which the K-unit chain does not represent. Below h ~ 1e-6, n_AL ~ 1e-3 and the slopes are
  noise-dominated.
- C0 (existing r = 1 driven runs, `c0_drive.json`): at x = 0.948 / 1.035 the slopes are 0.90-0.93 / 0.96-1.09 for h 1e-5 -> 1e-3 and
  0.77-0.79 / 0.73-0.76 for 1e-3 -> 1e-1; n_AL < 1 at h <= 1e-4.

## 6. C4: lifetime at g_lin against module size (`qs_analysis.json` C4, `lifetime_vs_K_criticality.png`)

Nested AL sub-modules (top-n by P[:,0] + P[:,1], internal edges, full-brain c), module-alone decay from fully active, B 4,096, T 3,000.

| n | 50 | 75 | 100 | 150 | 200 | 300 | 400 | 526 | 800 | 1,200 |
|---|---|---|---|---|---|---|---|---|---|---|
| g_lin,n | 0.2214 | 0.2061 | 0.1893 | 0.1756 | 0.1701 | 0.1667 | 0.1653 | 0.1643 | 0.1640 | 0.1639 |
| K_n | 16.4 | 20.3 | 25.0 | 51.8 | 61.7 | 64.0 | 64.2 | 66.6 | 66.9 | 67.1 |
| tau(x = 1), r 0 | 10.7 | 12.4 | 14.0 | 16.4 | 17.8 | 18.4 | 18.8 | 19.2 | 19.6 | 19.8 |
| K_fit (x 1.2-2.0), r 0 | 19.5 | 24.4 | 29.5 | 55.0 | 65.4 | 72.5 | 75.1 | 80.5 | 75.1 | 80.5 |
| tau(x = 1), r 1 | 10.6 | 11.5 | 12.3 | 13.8 | 14.4 | 15.1 | 15.2 | 15.2 | 15.8 | 15.7 |
| K_fit, r 1 | 18.8 | 20.5 | 22.7 | 32.7 | 35.1 | 36.3 | 36.9 | 36.9 | 36.9 | 36.9 |

- K_n saturates at ~67 by n = 200, so the sub-module family spans only K ~ 16-67 (x4); the plan's prediction at K_n
  (exact chain at R = 1 and K_n: 7.6-16.6) is 16-29% below the measured lifetimes.
- Log-log slopes of tau(x = 1): sub-modules alone 0.376 +- 0.025 (r 0, vs K_n), 0.390 +- 0.021 (r 0, vs K_fit), 0.247 +- 0.016 (r 1, vs K_n),
  0.504 +- 0.034 (r 1, vs K_fit). Merged with the 21 configuration nulls per r (K_nc 103-5,631; `redraw_fss` tau_x1): **0.496 +- 0.007 (r 0)
  and 0.465 +- 0.036 (r 1) with fitted K**; 0.421 +- 0.010 / 0.206 +- 0.010 with unfitted K. Inhibition decouples the unfitted K
  from the lifetime (r 1), as in `redraw_fss.md`.
- K_fit of the 526-neuron sub-module is 80.5 (r 0) and 36.9 (r 1), next to K_needed 75.8 / 36.6 of the whole brain.
- **Verification.** (1) The pre-registered analysis (plan 4.5) is against the structural K (K_n, K_nc). Within each family the slopes
  are 0.376 (sub-modules) and 0.372 (nulls, K_nc 103-5,631) at r 0, 0.247 and 0.253 at r 1, while the exact chain's own mean
  extinction time from n = K at R = 1 has local slope 0.557 over K 16-67 and 0.512 over K 103-5,631 (banded solve,
  `verify_criticality.json` c4). The merged r 0 slope 0.421 passes [0.4, 0.6] only because the two families are offset from each
  other; neither family alone does. (2) The fitted K (sub-modules: from tau at x 1.2-2.0; nulls: `redraw_fss` K_fit, from tau in
  [200, 3,000] at x > 1) is obtained by inverting the same chain's lifetime law, so tau(x = 1) ~ K_fit^0.5 tests whether the chain
  links lifetimes at x = 1 and x > 1, not rounding by an independently known K; the fitted-K range is 14-2,900, not "10-5,000". (3) At
  r 1 the null lifetimes at x = 1 are 1.32-2.06x the chain's tau_top at their own K_fit (r 0: 1.00-1.17), so even the fitted-K
  version misses the amplitude with inhibition (visible in `lifetime_vs_K_criticality.png` a). `figs/pd3/verify_criticality.png` b.

## 7. What this means for the user's questions

**Q2** [revised in verification]. The "steep but smooth crossover" (tau = 200 ... 3,000 at 1.4-2.1 g_lin) is not a critical point;
it is where the finite-K module's extinction time crosses the observation time. Underneath it is the linear instability of the AL
mode, and the onset there is consistent with a continuous, absorbing-state-type onset of a small system:
- located at the linear threshold g_lin: growth rates and their zeros follow the exact-linearisation chain at K_needed at
  r = 0, 1, 6 and not the net-input closure; with K free the effective branching scale is r-independent at r = 1 (0.98-1.00 of its
  r = 0 value; closure 0.91) and ~5% lower at r = 6 (closure 16% lower), so the threshold is r-independent at r <= 1 and within ~5%
  at r = 6; nulls of large K have their zero at 1.02 g_lin;
- rounded by a small effective module size that has to be fitted (lifetime at g_lin ~ K_fit^0.5, fitted K 14-2,900); the structural
  K (K_n, K_nc) predicts the lifetimes only with a slope of 0.37 (r 0) / 0.25 (r 1), so "rounding set by K" holds for an effective K,
  not for the structural one; the module's K ~ 37-80 does not grow with brain size (`redraw_fss.md`);
- without first-order signatures in the QS distributions or the mean field (stochastic hysteresis not tested here).
Since K is small and fixed, the brain has no phase transition here in the strict sense: "finite-size-rounded crossover around g_lin"
is the accurate term, and "second order" can only mean "continuous, with no first-order signature" in this finite system. The onset
is **not** in the single-mode MF-DP universality class at the level of avalanche statistics (S(T) slope 2.1-2.35, excess of small
avalanches below ~20 spikes with a size exponent of 1.46-1.57 above, QS order parameter rising faster than the chain in every window,
no delta_h = 2 window). The configuration nulls share this, which suggests it reflects the heterogeneous weights and degrees of the
connectome rather than the AL module or inhibition (hypothesis; not tested directly).

**Q1.** No broken-symmetry (B1) signature at the AL onset: shapes nearly symmetric, the minority lobe is recruited late rather
than suppressed, at every r. Of de Candia's diagram, only the SL edge (a second-order, finite-size-rounded line at g_lin,
vertical in (g, r)) is realised here.

## 8. Deviations from the spec, and why

1. **V0 follow-up references** (B 4,096 / 2,048): the pre-registered B 512 references failed the E[n] criterion through
   heavy-tail sampling (section 2); the frontier was confirmed by the exact one-step value and by the larger references.
2. **Seeding with an edge budget** (6e7 out-edges per step) and **an observation-window estimator** instead of draining slots: the
   per-step cost is bounded at high activity; E[n(age)] uses the avalanches whose window covers that age (unbiased because the
   window does not depend on the avalanche's fate); distributions use avalanches observed to the cap (all of them in every run).
3. **Per-avalanche n_active at ages 5/10/20/35/50** (record added after V0; dynamics unchanged) for the bootstrap of gamma; gamma is
   a 5-age regression (the all-ages aggregate is reported as a check).
4. **Own g_lin** for the AL module alone (0.16428, computed; +0.7% vs the brain) and for every null; null seeds always restricted
   to the 99% support of the leading mode (1/IPR < N/20 in all four null graphs).
5. **Added controls**: complete graphs (22 runs) as identical-code one-mode references; real r 6 at x 1.20-2.30 (6 runs) because
   gamma < 0 at x 0.98-1.02 left x* unbracketed.
6. **Post-hoc v-weighted seeds** (reweighting of the recorded seeds; no new runs) because uniform seeds load the second AL mode and
   weak seeds, which the chain's "n = 1 in the mode" does not.
7. **QS rerun**: the first submission kept rare stale configurations of the fully active start in the store (p = 0.5 random
   overwrites leave ~0.2% of slots untouched), inflating var(n_tot) up to 1e5x in single replicas; fixed by a round-robin refresh of
   all slots during the burn-in (measurement unchanged). The 14 QS tasks were cancelled and rerun.
8. **Field rerun**: float32 `1 - exp(-tanh(h) dt)` is 0 for h <= 3e-7 and quantised by ~15% at 1e-6; field mode uses `-expm1`.
   The h grid was extended to 3e-3, 1e-2, 3e-2 (the spec's top 1e-3 lies below the sqrt(K) << n << K window) and B = 128 (spec 64).
9. **Unimodality** tested on interior n_AL (>= 3); the boundary features are reported separately (section 4).
10. **Shape collapse** normalised by the mean profile, because no single gamma_ST exists (section 3.4).
11. **Chi2/dof <= 2** is unattainable at N = 1e6-2e6 even for the one-mode controls; the compensated-deviation criterion is used, with
    the controls' 0.10-0.12 as the baseline.
12. **Exponent windows**: S_c = 3.4 K (x 1.94 for AL graphs) and T_c from the chain at the graph's reference K (K_needed, module
    K_fit 70 / 36.5, null K_nc), S_min = 2, T_min = 5 steps.
13. `lifetime_vs_K_criticality.png` is a new figure; `redraw_fss`'s `lifetime_vs_K.png` was not overwritten (its `load_c4()` expects
    a json; the C4 points are in `qs_analysis.json` C4).
14. -t 03:00:00 for the QS / field / sub-module arrays (spec 02:00:00).

## 9. Files

- Code: `code/pd3_avalanche.py` (frontier simulator), `pd3_qs.py` (QS + field), `pd3_submodule.py` (C4), `pd3_crit_c0.py`,
  `pd3_crit_v0.py`, `pd3_aval_analyze.py`, `pd3_crit_qs_analyze.py`, `pd3_crit_figs.py`, `pd3_crit_summary.py`; runner `pd3_array.sbatch`.
- Results: `results/pd3/criticality.json`, this file; `results/pd3/criticality/` (aval_*.npz, qs_*.npz, field_*.npz, submodule_r*.npz,
  v0/, aval_analysis.json, qs_analysis.json, v0_validation.json, c0_drive.json, JOBS.md, args_*.txt).
- Figures (all inspected): `figs/pd3/aval_size_duration.png`, `aval_ST_slope.png`, `aval_critical_point.png`, `aval_shapes.png`,
  `aval_examples.png`, `aval_recruitment.png`, `qs_order_parameter.png`, `qs_susceptibility.png`, `field_scaling.png`,
  `lifetime_vs_K_criticality.png`; verification figure `figs/pd3/verify_criticality.png`.
- Verification: `code/pd3_verify_criticality.py` -> `results/pd3/verify_criticality.json` (section 10).

---------------------------------------------------------------------------------------------------------------------

## 10. Verification (adversarial check, 2026-11-09)

Independent script `code/pd3_verify_criticality.py` (imports no pd3 code; its own chain numerics: dense generator + `scipy.linalg.expm`
propagators, QSD by power iteration, mean extinction times by a banded solve; its own Perron vector of the 526-neuron module by dense
eigendecomposition, lambda / alpha_eff = 1.0000000). Output `results/pd3/verify_criticality.json`, figure `figs/pd3/verify_criticality.png`
(inspected).

**Recomputed from the raw outputs (all agree):**

| quantity | program | recomputed |
|---|---|---|
| x*, uniform seeds, r 0 / 1 / 6 (5-age gamma, linear up-crossing) | 1.186 / 1.345 / 1.722 | 1.1865 [1.1836, 1.1893] / 1.3448 [1.3410, 1.3494] / 1.7222 [1.7169, 1.7276] |
| x*, v-weighted seeds (own v) | 1.123 / 1.271 / 1.627 | 1.1230 / 1.2711 / 1.6274 |
| x* all ages 5-50 (estimator sensitivity) | 1.190 / 1.357 / 1.768 | 1.190 / 1.357 / 1.768 |
| x* complete 37 / 76; full configuration null r 0 | 1.273 / 1.126; 1.018 | 1.2731 / 1.1261; 1.0182 |
| chain x*: exact linearisation (K_needed) r 0 / 1 / 6; closure r 1 / 6; complete-graph chain 37 / 76 | 1.138 / 1.291 / 1.757; 1.418 / 2.102; 1.287 / 1.137 | 1.1376 / 1.2910 / 1.7573; 1.4180 / 2.1019; 1.2881 / 1.1373 |
| <S>, P(T > 10) at x 1, real r 0 / 1 / 6 | 10.1, 0.045 / 7.35, 0.036 / 5.65, 0.026 | 10.12, 0.0451 / 7.35, 0.0357 / 5.65, 0.0263 |
| <S>(T) local-slope peak: real r 0 / 1 / 6; module r 0 / 1; null; complete 37 / 76 / 300 / 2,000 | 2.35 / 2.24 / 2.10; 2.17 / 2.06; 2.62; 1.570 / 1.679 / 1.804 / 1.922 | 2.349 / 2.242 / 2.101; 2.166 / 2.063; 2.624; 1.570 / 1.679 / 1.804 / 1.922 (bin ratio 1.15-1.4 or 5-bin: r 0 2.27-2.41) |
| K_fit of P(T) (own bins and chain): real r 0 / 1 / 6; module r 0 / 1; complete 37 / 76 | 23.1 / 13.7 / 9.5; 21.8 / 11.8; 45.7 / 89.1 | 22.7 / 13.7 / 9.5; 21.4 / 11.8; 44.9 / 88.0 |
| max compensated deviation at K_fit (at K_ref), real r 0 / 1 / 6 | 0.17 (0.56) / 0.18 (0.51) / 0.20 (0.32) | 0.156 (0.561) / 0.177 (0.505) / 0.201 (0.321); controls 0.107-0.128 |
| size MLE [2, S_c/3], real r 0 / 1 / 6, module r 0, complete 37 / 76 | 1.770 / 1.804 / 1.825, 1.741, 1.470 / 1.492 | 1.770 / 1.804 / 1.825, 1.741, 1.470 / 1.492 |
| skewness at T 8-12, real r 0 / 1 / 6; module r 0 / 1; null; complete 37 / 76 | +0.026 / +0.032 / +0.040; -0.066 / -0.057; +0.058; -0.018 / -0.025 | +0.026 / +0.032 / +0.040; -0.066 / -0.057; +0.058; -0.018 / -0.025 (SE 0.0014-0.004) |
| late - middle minority share, r 0 / 1 / 6; module r 0 / 1 | +0.064 [0.046, 0.083] / +0.062 / +0.061 [0.043, 0.079]; +0.058 / +0.052 | +0.064 [0.047, 0.082] / +0.062 / +0.061 [0.044, 0.080]; +0.058 / +0.052 |
| QS chi_AL peak g, r 0 / 1 / 6; chain peak g | 0.208 / 0.247 / 0.336; 0.195 / 0.209 / 0.233 | 0.2080 / 0.2473 / 0.3362; 0.1951 / 0.2084 / 0.2325 |
| QS slope beta [2, 8], data / chain, r 0 / 1 / 6; n_AL / chain x 0.95 -> 3.1 | 1.20 / 0.79, 1.28 / 0.68, 1.15 / 0.53; 0.47 -> 1.32, 0.44 -> 1.75 | identical; 0.47 -> 1.32, 0.44 -> 1.75, 0.35 -> 1.81 |
| QS histograms unimodal (interior) | all | all (own test: smoothing sigma 1.5, 5% height, 20% dip) |
| field minimum slope (SE < 0.25, n_AL >= 1), r 0 x 0.9 / 1 / 1.1; r 1 | 0.727 / 0.645 / 0.553; 0.820 / 0.771 / 0.710 | identical |
| C4 tau(x = 1); slope vs K_n r 0 / 1; merged unfitted | 10.7-19.8, 10.6-15.7; 0.376 / 0.247; 0.421 / 0.206 | identical; chain prediction at K_n 7.56-16.56 (program 7.6-16.6) |
| V0 exact E[n(1 step)] at g 0.20 / 0.36 (from the connectome) | 1.1253 / 1.3696 | 1.12531 / 1.36963 |

**Methods checked and found sound.** All 63 avalanche runs are `final_complete` with no status-2 records; capped avalanches occur
only at x >= 1.35 and do not enter gamma (ages <= 50 steps) or the x = 1 distributions. The program's CLT bootstrap of gamma is valid
(avalanches are i.i.d.; an i.i.d. bootstrap by random-permutation groups gives the same SE, e.g. 0.0010 at r 0 x 1.2). Note for reuse:
the records are stored in closure order and, within a seeding wave, sorted by duration, so a block bootstrap in record or s0 order is
*not* valid (it inflates the SE ~10x; found and discarded during this verification). The exact chain QSD has no n = 1 edge excess
(p(1) is the minimum below the mode at x 1.13-1.58), as the program states. C0 slopes match `c0_drive.json`.

**Changes made to this file (each marked "[verification]" or "revised in verification" in place):**
1. Section 0, Q2 row and strength column: "Partly yes" -> "Partly, and only as a finite-size-rounded crossover"; location qualified
   (r-independent at r <= 1, ~5% at r = 6, from the free (K, c) fit); K-rounding qualified (fitted K 14-2,900, partly circular;
   structural-K slopes 0.25-0.38); the size-exponent claim corrected (see 4); the beta slope marked window-dependent; "so this is
   network heterogeneity" -> "points to ... (hypothesis)"; strength "continuity / location / K-rounding: strong" replaced by graded
   statements. Q3b row: "collapse" only with fitted K.
2. Pre-registered verdict: Q2-c changed from "passes (with fitted K)" to "fails at r = 1 as pre-registered (structural K, plan 4.5),
   passes only literally at r = 0 (merged families with own slopes 0.37-0.38), passes with post-hoc fitted K". Section 1 table row
   updated accordingly. Reason: plan 4.5 specifies tau vs K_n merged with the nulls (K ~ 10-5,500, i.e. K_nc), and K_fit is derived
   from the same chain's lifetime law (partly circular).
3. One-line reading and section 7: "a continuous absorbing-state transition located at the exact, r-independent g_lin ... rounded by
   sqrt(K)" -> a finite-size-rounded crossover around g_lin, consistent with a continuous onset of a small system, with the location
   r-independent at r <= 1 and within ~5% at r = 6 and the rounding set by a fitted (not the structural) K; "the brain has no phase
   transition here" kept and made the headline. Reason: K ~ 10-90 is fixed and does not grow with N, so no transition can be
   established; continuity rests on unimodality in a small system and on the mean field.
4. Section 3.2: added the KS-chosen-S_min exponents (1.457 / 1.569 / 1.481 / 1.562 for real r 0 / r 1 / module / null vs 1.351 for
   the K 76 control); the statement "size distribution steeper than S^-3/2 below the cutoff" corrected: the steepness (1.74-1.83) is
   confined to S below ~10-30 spikes; above that the exponent is close to 3/2 (though ~0.1-0.2 above the finite-K control).
5. Section 3.1: added the caveat that the chain-vs-closure comparison used K_needed (itself derived under the exact linearisation),
   and the free 2-parameter fit: c(1)/c(0) = 0.978 / 0.997 (uniform / v-weighted; closure 0.911), c(6)/c(0) = 0.942 / 0.948
   (closure 0.837), fits imperfect (chi2/dof 9-32 at r <= 1, 119-131 at r 6).
6. Section 4: beta marked as a window-dependent effective slope (data 0.59-1.28, chain 0.40-0.79 across windows [1, 4]-[8, 32]);
   the robust statement is data > chain in every window.
7. Section 6: added within-family slopes vs structural K, the chain's own local exponent (0.557 over K 16-67, 0.512 over 103-5,631),
   the circularity of K_fit, the correct fitted-K range (14-2,900), and the r 1 amplitude miss (null tau / chain tau at K_fit
   1.32-2.06).
8. Section 1, <S>(T) row: noted robustness of the peak to the binning.
9. Sections 9 / 10: verification files added.

**Not changed, noted.** (a) Q2-a's chi criterion was evaluated on chi_AL; the whole-brain chi_QS (the plan's wording) has its first
local maximum at g 0.2148 at r 0 (+10.2% vs 0.195, parabola through the grid maxima), i.e. a borderline fail instead of a pass; at
r 1 both fail. (b) The Q2-b 10% compensated-deviation criterion is not met by the one-mode controls either (0.107-0.128), so the
real brain's 0.16-0.20 is a modest excess; the K_fit / K_needed ratio (0.30-0.59) is the cleaner failure. (c) Seed weighting moves x*
by 0.06-0.10 (uniform vs v-weighted), larger than the chain-vs-data differences at r 0 and 1; the location test is only as good as
this systematic. (d) The skewness and recruitment results (Q1) reproduce exactly and their reading stands.

