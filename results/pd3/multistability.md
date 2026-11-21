# Program `multistability`: broken L/R symmetry, hysteresis, coexisting attractors (stochastic model)

2026-11-09, workflow wf_98fb990d-4fb, plan `results/pd3/plan.md` section 5. Every number below is read from the file named next to it;
all of them are also collected, with their source, in `results/pd3/multistability.json` (built by `code/pd3_multistab_report.py`).
Job ids, statuses and reruns: `results/pd3/multistability/JOBS.md`. Raw outputs: `results/pd3/multistab/`.

## 0. Bottom line

| question | result | strength |
|---|---|---|
| **Q1. Is there neuron-level broken symmetry (de Candia's B1)?** | **No.** At 48 (g, r) points (r = 0 to 6, g = 1 to 8 x the simulated onset, plus the closure's predicted B1 windows), starting one AL lobe only, the other lobe is recruited within a median of 1.0 to 7.0 model time. At t in [500, 1000] the AL-L-start and AL-R-start replicas end in the same symmetric state: 0 of 48 points meet any of the three pre-registered broken-symmetry criteria (`analysis_m1.json`). | Strong for the tested window: 64 replicas per start, T = 1000. In 3 of the 21 closure-B1-window points every replica died before the end of the run, T = 1000 [verification: at r = 4, g = 0.29 and 0.32 all by t = 165 and 342; at r = 6, g = 0.43 all by t = 934, with 16 of 256 still alive at t = 500], and a 4th kept at most 3 survivors per start, so the closure's B1 window cannot be tested there: those points lie below the stochastic model's sustained activity. |
| **Q1. Is there any first-order line (coexistence / hysteresis)?** | **None found.** Adiabatic g-sweep loop areas fall 3.6- to 15.5-fold between D = 50 and D = 800 [verification: 3-point log-log slopes -0.46 to -0.99. These are descriptive only. The D = 800 points sit at or near the noise floor, so they are not scaling exponents]. At D = 800 every loop is within 3 SE of its noise floor except r = 0, h = 1e-5 (0.263 +- 0.055 against a floor of 0.080), which is still shrinking (A(800)/A(200) = 0.36), so it does not meet the pre-registered criterion [verification: a label-permutation test finds that loop still open at D = 800 (p < 0.001), and r = 0.5, h = 1e-5 marginally open (p = 0.02); all other D = 800 loops have p >= 0.12]. All 14 loop centres re-run from both branches' end-states (M2b) relax to one state (late KS p >= 0.093) [verification: only 2 of the 14 centres are at the onset (r = 4, g = 0.388 and the r-sweep centre r = 5.75, g = 0.6); 5 lie below the tau = 200 line, where activity is drive-dominated, and 7 lie above 1.25 x g_on, so M2b is only a thin direct test at the onset itself]. The 40 (g, r) cells where the neuron-level mean field (QMF) had coexisting fixed points show no coexistence in the stochastic model (M3b), and the module-subset starts converge to a single non-silent attractor in all 23 surviving cells (M3). | Strong against coexistence at dwell times up to 800 and run times up to 5,000 model time (32 to 64 replicas). It cannot exclude hysteresis smaller than the noise floor (loop area about 0.02 to 0.13), or metastable states living longer than 5,000. |
| **Q2-e. Is the steep onset first order?** | **No** first-order signature at the onset: there is no coexistence and no hysteresis that survives longer dwells. The h = 0 down-sweeps go extinct close to where the measured lifetime crosses the dwell (tau = D) [verification: this agrees with the lifetime map but does not discriminate between the readings, since a metastable branch near a spinodal would also die where tau ~ D. The observed extinction g also lies systematically about 0.6 to 1 grid step below the prediction, section 3.1]. Within the detection limits this argues against the first-order alternative. It does not decide between a rounded *second-order* transition and a smooth rare-region crossover; that is for the criticality program. | Moderate to strong (the same caveats as above). |

The pre-registered predictions fared as follows.
- Neuron-level theory, "symmetrisation within tens of model time everywhere": **confirmed**, and faster than predicted (median 1.0 to 7.0).
- de Candia closure, "persistent one-lobe states for r >= 2.5 (r = 4 from g 0.287)": **refuted** wherever activity survives.
- M3, "a single non-silent attractor": **confirmed**.
- M3, "rest- and island-only starts ignite the AL": **half right**. The rest-only start does. The single-island starts mostly die at g = 0.4 at every r, and at g = 0.7 for r >= 1 (h = 0 is absorbing) [verification: at r = 0, g = 0.7 they survive in 72% to 100% of replicas]. When they survive, they reach the common attractor.

## 1. What was run

| part | spec | code / mode | outputs |
|---|---|---|---|
| validation | all-active start, r = 1, g in {0.30, 0.344, 0.40}, B = 64, T = 3000 (+ g 0.345, 0.39, 0.405, which have same-g pooled points) | `pd3_multistab.py validate`; checked by `pd3_multistab_validate.py` | `validate_r1.npz`, `validate_r1_split.npz`, `validation.json` |
| M1 | 7 r x (6 multiples of g_on(r) + the closure windows), 4 starts x 64, T = 1000, h = 0 | `alsym` | `al_sym_r{0,0.5,1,2,3,4,6}.npz` |
| M2a | 6 r x h {1e-5, 1e-4} x D {50, 200, 800}: up from silence and down from all-active at g = 30, over 40 ln g in [0.15, 30], B = 32 per branch; + 6 down-sweeps at h = 0, D = 200 | `sweep` | `sweep_r*_h*_D*.npz` (42) |
| M2b | at loop centres: 64 replicas from each branch's end-state, T = 5000 | `m2b` (+ `m2b --rsweep`) | `m2b_*.npz` (14) |
| M2c | g {0.6, 2, 8}, r 0 -> 6 -> 0 over 25 points (+ 6 -> 0 -> 6, added), D = 200, h = 1e-5, B = 32 | `rsweep` | `rsweep_g{0.6,2,8}.npz` |
| M3 | r {0, 1, 2, 4} x g {0.4, 0.7, 1.2, 2.5, 6, 15}, 13 starts x 32, T = 2000, h = 0 | `ic` | `ic_r*_g*.npz` (24) |
| M3b | every QMF multistable cell (40 cells, `module_theory/qmf_multi.json`): the MT3 starts (B = 64) + Bernoulli samples of the QMF fixed points (B = 32, added), T = 5000 | `icq` | `icq_r*_g*.npz` (40) |

All 132 outputs exist and no task failed, was preempted or timed out (sacct; `multistability/JOBS.md`). In-script GPU time totals
about 19.4 h (sum of `seconds` in the output metadata). Analysis: `code/pd3_multistab_analyze.py` -> `results/pd3/multistab/analysis_{m1,m2a,m2b,m2c,m3,m3b}.json`
and `summary.json`. Figures: `code/pd3_multistab_figs.py`.

### 1.1 Simulator `code/pd3_multistab.py` and its validation

It uses the dense synchronous update of swc.py with unchanged rules: dt 0.2, sigma +1 / -r / 0, active -> silent with
probability 1 - e^-dt, silent -> active with probability 1 - exp(-tanh(s)_+ dt).
- Each column (replica) carries its own (g, r, h). An up- and a down-sweep, or several starts, therefore share one sparse product.
- With one r per segment, the combined W(r) is built exactly as swc.py builds it. With per-column r (the r-sweeps), E and I are
  multiplied separately.
- Starts are sets of P0 modules (`results/pd3/modules/partition_P0.npz`, 34 modules), Bernoulli(p) vectors, or bit-packed snapshots.
- The recording follows the spec.
  - Per-module counts every model time (int32 [rec, 34, B]). M3b records every 5 model time; see deviation 7.
  - Per-replica occupancy is accumulated over the last half of each dwell and gives rho, X_k, n_AL / n_L / n_R and PR, plus the
    PR of the replica-mean occupancy with a bootstrap.
  - A bit-packed snapshot is saved at the end of each dwell.
  - Exact extinction steps are recorded.
- Writes are atomic, and a task whose output exists is skipped.

Validation (`results/pd3/multistab/validation.json`) uses the censored-exponential MLE with extinction times snapped to swc.py's
record grid, as in `phase_complete`. It passes at every g on both code paths.

| g (r = 1) | tau, fast path [95% CI] | tau, split E/I path | pooled `phase_complete` point [95% CI] | inside |
|---|---|---|---|---|
| 0.30 | 373 [292, 484] | 303 | 301 [236, 391] (LR p 0.23 / 0.97) | yes / yes |
| 0.344 | 2,884 [2,141, 3,985] | 3,704 | no same-g point; envelope of neighbours 0.34 / 0.345: [1,181, 5,267] | yes / yes |
| 0.345 | 3,683 | 2,729 | 3,171 [2,030, 5,267] (LR p 0.60 / 0.58) | yes / yes |
| 0.39 | 30,720 | 30,236 | 20,702 [8,086, 75,981] | yes / yes |
| 0.40 | 46,460 | 46,681 | envelope [8,086, 3.46e6] | yes / yes |
| 0.405 | 63,707 | 95,776 | 87,702 [15,741, 3.46e6] | yes / yes |

Speed: 14.8 ms per step at B = 384 on an RTX 6000 Ada (fast path), and 47.8 ms per step on an A6000 (split path)
(`multistability/JOBS.md`).

## 2. M1: left/right AL symmetry (the decisive B1 test)

Source: `results/pd3/multistab/analysis_m1.json` (`summary.json` -> m1). Figure: `figs/pd3/multistab_AL_symmetry.png`.

Definitions:
- f_L = n_L/248, f_R = n_R/275, A = (f_L - f_R)/(f_L + f_R).
- t_sym is the first time |A| < 0.3. The AL-L/AL-R counts are recorded every step for t <= 100 and every model time after that.
- Late window: t in [500, 1000]. A survivor is alive at T and has an active AL in >= 50% of the window samples.

**Symmetrisation is fast at every point.**
- The median t_sym over the AL-L-only or AL-R-only replicas is 1.0 to 7.0 model time over all 48 points. It shortens with g: the
  mean of the L- and R-start medians is 3.8 to 4.9 at the onset and 1.1 to 2.0 at 8 x the onset (panel b).
- The largest upper quartile is 12.6, and the largest single-replica t_sym is 59 (r = 4, g = 0.32, a dying run).
- No replica was censored. 71 of the 6,144 one-lobe-start replicas lost their AL before reaching |A| < 0.3.

**There are no persistent one-lobe states.**

| criterion (pre-registered) | points meeting it (of 48) | extreme value |
|---|---|---|
| c1: >= 25% of surviving AL-L-start replicas keep late mean A > 0.6 | 0 | the largest fraction is 0 (the AL-R mirror is also 0) |
| c2: AL-L vs AL-R start late A distributions differ, KS p < 0.001 | 0 | smallest p = 0.021 (r = 0, g = 0.665) over 44 testable points |
| c3: the all-active start's late A is bimodal (2-Gaussian BIC > 10, weights >= 0.1, separation > 2 SD) | 0 | largest delta-BIC = 2.6 (r = 2, g = 1.608) |

The late mean A of every start with >= 10 survivors lies between -0.001 and +0.047, and no single replica exceeds |A| = 0.189.
The small positive A is common to all four starts (`late_A_mean`): it is a fixed slight left bias of the one symmetric state, not
a broken symmetry.

**The closure's B1 windows.** de Candia's closure for the AL pair (`theory_module2.json` coarse_2module) has B1L | B1R from
g = 0.247 (r = 3), 0.287 (r = 4) and 0.428 (r = 6) up to the end of its grid at g = 3.0.
- 21 of the M1 points lie in these windows.
- At r = 4, g = 0.29 and 0.32, and at r = 6, g = 0.43, every one of the 256 replicas died before the end of the run (T = 1000)
  [verification: corrected from "before t = 500". At r = 4 the last replica died at t = 165 and 342. At r = 6, g = 0.43, 16 of the
  256 were still alive at t = 500 and the last died at t = 934 (`results/pd3/verify_multistability.json` F1)]. At r = 4, g = 0.40,
  only 1 to 3 survived per start.
- The reason: these g values lie below or barely above the stochastic model's tau = 200 line (S = 0.373 at r = 4 and 0.436 at
  r = 6, `results/analysis2/phase_complete.json`). The closure's silent edge (0.270 at r = 4 and 0.386 at r = 6,
  `theory_module2.json`) is far below where the stochastic AL can sustain itself.
- At the other 17 window points (>= 10 survivors per one-lobe start), both one-lobe starts symmetrise and end in the same state
  (KS p 0.059 to 0.94).

## 3. M2: adiabatic sweeps, hysteresis and its confirmation

Sources: `analysis_m2a.json` (`decisions`, `runs`, `h0`), `analysis_m2b.json`, `analysis_m2c.json`. Figures:
`figs/pd3/multistab_hysteresis.png` and `figs/pd3/multistab_m2b.png`.

### 3.1 M2a, loop area against dwell

A_rho = integral |ln rho_up - ln rho_down| d ln g over the 40 sweep points. SE is the replica bootstrap. The noise floor (added)
is the same area between random halves of one branch, scaled to 32 vs 32 replicas.

| r | h = 1e-5: A(D = 50) | A(200) | A(800) +- SE | floor(800) | log-log slope vs D | h = 1e-4: A(50) | A(200) | A(800) +- SE | floor(800) | slope |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.71 | 0.728 | 0.263 +- 0.055 | 0.080 | -0.67 | 0.266 | 0.035 | 0.019 +- 0.010 | 0.022 | -0.95 |
| 0.5 | 1.55 | 0.398 | 0.167 +- 0.043 | 0.087 | -0.80 | 0.249 | 0.050 | 0.029 +- 0.009 | 0.025 | -0.77 |
| 1 | 1.76 | 0.486 | 0.119 +- 0.048 | 0.106 | -0.97 | 0.152 | 0.084 | 0.038 +- 0.010 | 0.030 | -0.50 |
| 2 | 1.39 | 0.351 | 0.139 +- 0.046 | 0.128 | -0.83 | 0.194 | 0.053 | 0.041 +- 0.010 | 0.034 | -0.57 |
| 4 | 1.23 | 0.489 | 0.081 +- 0.039 | 0.130 | -0.98 | 0.139 | 0.061 | 0.039 +- 0.009 | 0.036 | -0.46 |
| 6 | 1.57 | 0.182 | 0.101 +- 0.038 | 0.132 | -0.99 | 0.178 | 0.060 | 0.043 +- 0.009 | 0.039 | -0.51 |

**Reading the table.**
- Every loop shrinks as the dwell grows. At h = 1e-5 the area falls 6.5 to 15.5-fold from D = 50 to 800.
- Measured against the noise floor, the h = 1e-5 loops exceed it by 5.3 to 9.9 SE at D = 50 and by -1.2 to 3.3 SE at D = 800.
  Only r = 0 stays above 3 SE (3.3), and that loop is still shrinking (ratio 0.36, slope -0.67).
- At h = 1e-5 the residual loop sits at the onset. The largest |ln rho_up - ln rho_down| at D = 800 is 0.96 (r = 0, g = 0.225),
  against 3.01 at D = 50 (`analysis_m2a.json` runs, d_ln_rho).
- The up branch lags because ignition from silence waits for the drive to seed the AL: the up-branch rho rises late and the down
  branch survives into the slow window (panel a). [verification: corrected from "That is a lag loop, the signature of a continuous
  transition". A rate-dependent lag loop appears at any transition swept at finite rate. What argues against a first-order loop is
  that the area keeps shrinking as D grows; a first-order loop would keep a finite area.]
- The stronger drive h = 1e-4 seeds faster, and its loops are at the noise floor from D = 200 on [verification: label-permutation
  p >= 0.29 at D = 200 except r = 1 (p = 0.02); p >= 0.12 for all six r at D = 800].
- [verification] **The slopes in the table are not exponents.** Each is a 3-point fit, and the two segments disagree (for example,
  r = 0, h = 1e-4: -1.46 for D = 50 -> 200 and -0.44 for 200 -> 800; r = 6, h = 1e-5: -1.55 and -0.42). The flattening comes from
  the noise floor: at r = 4 and 6, h = 1e-5, A(800) is already below the floor. Read them only as "the loops shrink".
- [verification] **Label-permutation test (independent of the floor estimate).** Whole up- and down-branch replica columns were
  swapped at random (2,000 permutations; `code/pd3_verify_multistability.py`, `results/pd3/verify_multistability.json` V3).
  - At D = 800 the h = 1e-5 loop is still significantly open at r = 0 (p < 0.001) and marginally at r = 0.5 (p = 0.02); r = 1, 2, 4
    and 6 give p = 0.36, 0.32, 0.95 and 0.80.
  - Both open loops are still shrinking (r = 0: A(800)/A(200) = 0.36; r = 0.5: 0.42). That is a kinetic lag not yet closed at
    D = 800, not evidence of a finite first-order loop.
- The PR loop areas are small throughout: A_PR at D = 800 is 0.0034 to 0.0172 (in units of PR/N x ln g). No candidate exceeds its
  noise floor (computed from the bootstrap SDs) by 3 SE.

**Pre-registered hysteresis criterion.** A candidate needs A(800) > 3 SE and A(800)/A(200) > 0.6.
- It fires for A_rho at r = 2 and r = 4 (h = 1e-4, whole sweep and onset window), at r = 6 (h = 1e-4), and in the window above
  the onset at r = 2 and 6 (h = 1e-5).
- Applied to A_PR, it also fires at r = 0 (h = 1e-4) and r = 6 (h = 1e-5 and 1e-4).
- **Every candidate's area is within 3 SE of its noise floor.** For example, r = 2, h = 1e-4 has A(800) = 0.041 +- 0.010 with
  floor 0.034.
- The criterion fires because a |difference| area cannot shrink below its noise floor, so the "shrinks by < 40%" condition is
  met by noise once the loop has closed.
- With the floor subtracted, **no** (r, h, window) is a candidate for A_rho or A_PR (`any_candidate_noise_corrected` false for
  both). As the spec requires, every candidate went to M2b anyway.

**h = 0 down-sweeps (D = 200).** All 32 replicas die at every r. The extinction point is compared with the tau = D = 200 crossing
of the lifetime map (`results/pd3/redraw/heatmap_real.npz`, ln_tau_iso) and with the extinction distribution predicted from that
curve (survival exp(-D/tau(g)) per dwell). The sweep grid steps by a factor 1.146 in g.

| r | extinction g: median [IQR] | tau = 200 crossing | predicted median | KS p (observed vs predicted) |
|---|---|---|---|---|
| 0 | 0.225 [0.225, 0.225] | 0.234 | 0.225 | 1.00 |
| 0.5 | 0.258 [0.258, 0.258] | 0.265 | 0.258 | 1.00 |
| 1 | 0.296 [0.258, 0.296] | 0.283 | 0.296 | 0.80 |
| 2 | 0.296 [0.296, 0.339] | 0.320 | 0.339 | 0.12 |
| 4 | 0.388 [0.339, 0.388] | 0.373 | 0.388 | 0.65 |
| 6 | 0.445 [0.388, 0.509] | 0.435 | 0.445 | 0.93 |

Where the down branch goes silent is fixed by the extinction time against the dwell.

[verification] The original text concluded "It is a lifetime crossover that moves with the observation time, not a spinodal."
That is weakened, for three reasons.
- **Only one dwell was run at h = 0.** With D = 200 only, the shift with observation time is inferred from tau(g), not observed.
- **The check does not discriminate.** A metastable active branch ending at a spinodal would also die where its own lifetime ~ D.
  The agreement shows only that the sweep's extinction is governed by the same lifetimes as the fully-active-start runs. The
  evidence against a spinodal is elsewhere: no coexistence (M2b, M3, M3b) and loops that close with D.
- **The KS p-values above are on tied, discrete grid values, so they are conservative.** An exact Monte-Carlo multinomial G-test
  of the observed extinction-segment counts against the predicted distribution gives p = 1.0, 1.0, 0.64, 0.33, 0.80 and 0.38
  (r = 0, 0.5, 1, 2, 4, 6), so there is no misfit.
  - However, the mean ln g_ext is *below* the prediction at all 6 r, by 0.08 to 0.14 in ln g (0.6 to 1.0 grid step).
  - Each shift alone gives one-sided p 0.06 to 0.25. Combined: sign test p = 0.031, Stouffer p = 0.002 (`verify_multistability.json` F2).
  - So the swept active branch lives somewhat longer than the fully-active-start lifetime map predicts. This is small and does
    not change any conclusion, but "dies where tau = D" holds only to within about one grid step.

### 3.2 M2b, coexistence test at the loop centres

There are 14 tasks (`analysis_m2b.json`):
- the 6 rho-loop centres of the spec candidates;
- 4 extra centres from the PR candidates;
- 1 control, the largest remaining loop (r = 0, h = 1e-5, g = 0.225), which is not a candidate;
- 3 r-sweep loop centres (section 3.3).

Each runs 64 replicas from the up-branch end-state and 64 from the down-branch end-state for T = 5000 at fixed (g, r, h).
- **No coexistence anywhere.** On per-replica rho and PR over t in [2500, 5000], the smallest KS p is 0.093 (rho, r = 6,
  h = 1e-5, g = 0.296) and 0.21 (PR).
- The two groups' mean active counts are statistically indistinguishable already in the first 10 model time (KS p >= 0.059 in
  every task, `windows_active_count`).
- So the D = 800 end-states of the two branches are effectively the same state.
- [verification] **Where the centres lie.** The centres were chosen by the pre-registered rule (largest |difference| of flagged
  loops), and most flagged loops were noise. As a result, only 2 of the 14 centres are at the onset, between the tau = 200 line
  S and 1.25 x g_on: r = 4, g = 0.388 and the r-sweep centre r = 5.75, g = 0.6.
  - 5 lie below S, where the drive dominates: 2 to 33 neurons active per replica on average over t in [2500, 5000], and the AL is
    completely silent in 54% to 99% of the late records (`verify_multistability.json` F5).
  - 7 lie above 1.25 x g_on, deep in the active phase (`verify_multistability.json` F3).
  - So M2b's "no coexistence" is mostly a test away from the onset. At the onset the evidence is the closing onset-window loops
    plus M1, M3 and M3b.
- [verification] "One state" means one stationary distribution, the same for both groups. Near the onset that distribution
  includes intermittent AL silences, which give a left tail in the per-replica late rho. At r = 5.75, g = 0.6 the AL is silent in
  up to 43% of a replica's late records (KS up vs down on that fraction p = 0.84). This is not two coexisting active states.

### 3.3 M2c, sweeps in r

The table gives loop areas in r at D = 200 (`analysis_m2c.json`).

| g | 0 -> 6 -> 0 run: A +- SE (floor) | 6 -> 0 -> 6 run (added): A +- SE (floor) |
|---|---|---|
| 0.6 | 0.101 +- 0.026 (0.115) | 0.132 +- 0.033 (0.105) |
| 2 | 0.037 +- 0.009 (0.041) | 0.046 +- 0.008 (0.039) |
| 8 | 0.029 +- 0.008 (0.035) | 0.030 +- 0.007 (0.035) |

- Every loop sits at its noise floor. With one dwell (as specified) the D-scaling half of the criterion cannot be evaluated. M2b
  at each run's loop centre (r = 5.75, 5, 5) finds no coexistence (KS p 0.30 to 0.94).
- AL activity falls smoothly with r on both branches. At g = 0.6 the AL fraction drops from 0.279 at r = 0 to 0.076 at r = 6
  (`XAL_up`).
- **One-lobe dwell audit.** 14 of the 3,200 replica-dwells at g = 0.6 (0 at g = 2 and 8) have a dwell-mean |A| > 0.6.
  - In every one of them the AL was completely silent in >= 56.5% of the dwell's records: these are AL extinction events whose
    left-over occupancy is lopsided, not one-lobe states.
  - They occur at r >= 2.75, where g = 0.6 is close to the stochastic onset.

## 4. M3: module-subset starts

Source: `analysis_m3.json`. Figure: `figs/pd3/multistab_ic_attractors.png`.

The 13 starts are: all; AL; rest_only (MT3's definition: every neuron outside the 526-neuron AL); islands 1 to 5 alone;
AL + islands 1 to 5; MT3's random half-subsets half0 to half2; AL + island 1.

The late vector X_k is the fraction active in each of the 34 P0 modules over t in [1000, 2000]. Relative L1 is MT3's
|a - b|_1 / max(|a|_1, |b|_1).

- **A single non-silent attractor in every cell.** At r = 4, g = 0.4 all 416 replicas died (that g is near the S line at r = 4).
  In each of the other 23 cells, every surviving replica falls into one cluster (average linkage, cut 0.2).
- The largest relative L1 between two start means is 0.077 (r = 4, g = 0.7), against the 0.2 threshold. The largest
  within-start spread is 0.051 (r = 2, g = 0.4). No start pair is distinct.
- **Which starts die (h = 0, absorbing).** Single-island starts die at low g. At r = 1, g = 0.4, islands 1 to 5 alone keep 0% to
  6% of replicas, against 88% for rest_only and 97% for all-active. At r = 0, g = 0.4, island 1 keeps 19% and island 3 keeps 28%;
  islands 2, 4 and 5 keep 0%.
- From g = 1.2 on, every single-island start keeps >= 72% of its replicas (r = 0 to 4), and the survivors reach the common
  attractor.
- So the rest-only start does ignite the AL, at survival comparable to the all-active start. A single island usually cannot at
  g = 0.4 (any r) or at g = 0.7 for r >= 1: it dies before it recruits the AL. [verification: corrected from "at g <= 0.7". At
  r = 0, g = 0.7 the single-island starts survive in 72% to 100% of replicas (islands 1 to 5: 100, 72, 100, 94, 97%). At r = 1, 2
  and 4, g = 0.7 they survive in 0% to 41%. Surviving island starts carry the same AL fraction as the all-active start in every
  cell, so they do ignite the AL.]

## 5. M3b: do the QMF's coexisting fixed points survive noise?

Source: `analysis_m3b.json`. Figure: `figs/pd3/multistab_qmf_cells.png`.

The QMF (MT3) found >= 2 coexisting stable fixed points at 40 (g, r) cells for r = 0.5 to 6, each pair differing in one connected
region of 8 to 9,274 neurons (`results/pd3/module_theory.md` section 4.4). For every pair of QMF attractors (a, b), each surviving
replica's late (t in [3000, 5000]) occupancy on that region is projected as z = (x - rho_a).d/|d|^2. Here z = 0 is attractor a and
z = 1 is attractor b.

- **No coexistence in any of the 40 cells.**
  - Replicas started at a and at b reach the same z. This holds both for MT3's module starts and for Bernoulli samples of the QMF
    fixed points themselves.
  - The largest difference of median z between the a-start and b-start groups is 0.034.
  - The smallest of the 133 KS p-values is 0.006 (r = 4, g = 0.727, all vs isl3, MT3 starts, 59 vs 16 survivors), with medians
    0.407 vs 0.384.
- The common state is a single noise-averaged state at an intermediate projection: median z ranges from -0.012 to 1.047 over
  cells. It is not one of the two QMF branches.
- The early window ([1000, 3000]) and the late window agree within 0.039 in median z, so no slow drift is left.
- **5 cells (r = 6, g = 0.330 to 0.518) are extinct** in every replica by T = 5000. At r = 6, g = 0.580, only 0% to 12.5% survive.
  The QMF's r = 6 multistability starts below the stochastic onset (0.579, `phase_complete.json`), where the stochastic AL cannot
  sustain itself.

## 6. Answers

**Q1: a de Candia-style diagram with broken-symmetry phases and first-order lines.** At the neuron level the stochastic model has
neither.
- The left and right antennal lobes are always co-active. A one-lobe start re-symmetrises in a few model time, at every r up to
  6, including inside the closure's B1L | B1R windows wherever activity survives.
- No (g, r) shows hysteresis that persists as the dwell grows, coexistence at loop centres, or more than one non-silent attractor.
  [verification: corrected from "hysteresis beyond noise". At D = 800 the r = 0, h = 1e-5 loop is still above noise (permutation
  p < 0.001) and r = 0.5 marginally (p = 0.02), but both are still shrinking.]
- The QMF's coexisting fixed points (40 cells) are removed by demographic noise.

The B1 phase and the dashed first-order lines that de Candia's closure predicts for these modules
(`results/pd3/module_theory.md` section 3) are therefore properties of the closure, not of this connectome's dynamics. This is
consistent with the theory audit (`theory_notes.md` section 1.5): Gamma about 0.003 and no dead zone. On the final diagram there
is **no dash-dot (first-order) line**.

**Q2-e: is the steep crossover first order?** No first-order signature was found.
- At the onset the up/down loops shrink with the dwell: at h = 1e-5 the area falls 6.5- to 15.5-fold from D = 50 to 800.
  [verification: the descriptive 3-point slopes -0.67 to -0.99 are not exponents, see section 3.1.]
- All but two of those loops reach the noise floor by D = 800 [verification: r = 0 (permutation p < 0.001) and r = 0.5 (p = 0.02)
  are still open, and both are still shrinking].
- No loop centre, including r = 0, shows two states (M2b).
- [verification: corrected from "This rules out the first-order reading and leaves the finite-size-rounded second-order reading
  standing".] Within the detection limits this argues against a first-order onset. The M2b test is thin at the onset itself (2 of
  14 centres, section 3.2), and it does not cover metastability living longer than 5,000. It leaves a continuous onset standing:
  either a rounded second-order transition or a smooth rare-region (localized) crossover with no sharp transition. This program
  cannot tell those apart; the criticality program has to.

**Contribution to Q2 / Q3a.** The h = 0 down-sweeps go silent close to where the measured lifetime equals the dwell (tau = D) at all
6 r (section 3.1). That is consistent with extinction governed by the lifetime map. [verification: corrected from "the behaviour
expected of a lifetime crossover whose position depends on the observation time". With one dwell (D = 200) the dependence on
observation time is inferred, not observed. A metastable branch near a spinodal would also die where tau ~ D, so this is a
consistency check, not evidence for "crossover, not wall". The observed extinction g also sits 0.6 to 1 grid step below the
prediction at all 6 r.]

## 7. Deviations from the spec (all deliberate)

1. **Normalisation constant.** swc.py's c is 1/mean in-strength over the 166,499 neurons with nonzero in-strength, which is
   1/745.91 (computed here from the cache). The project notes and the task text call it "1/742.7", but 742.71 is total
   in-strength / N, every neuron counted. pd3_multistab.py uses swc.py's formula exactly; only then is the validation like for
   like. The 0.43% difference is a documentation inconsistency, not a model change.
2. **Per-column (g, r, h)** and a split E/I product for r-sweeps. Both code paths were validated (section 1.1). p_on is computed
   with expm1, which is identical in exact arithmetic and more accurate in float32 for h = 1e-5.
3. **Validation points.** g = 0.344 and 0.40 have no same-g pooled point, so they are compared with the CI envelope of the two
   neighbouring pooled points. Three extra g values with same-g pooled points (0.345, 0.39, 0.405) were added, with an LR test of
   equal rates.
4. **M1 definitions** (not fixed by the spec). The survivor definition, the bimodality test (2-Gaussian BIC, delta > 10, weights
   >= 0.1, separation > 2 SD; Sarle's coefficient also stored), and "AL died first" as its own t_sym outcome.
5. **M2a noise floor and the A_PR criterion (added).** The spec criterion is applied literally to A_rho and also to A_PR, and every
   candidate is sent to M2b. The noise-corrected criterion (A - floor > 3 SE) is reported next to it because the literal one fires
   on noise once a loop has closed. A control loop was added to M2b. M2b was also run at the M2c loop centres (`m2b --rsweep`),
   because M2c has a single dwell and the spec's D-scaling criterion cannot be evaluated there.
6. **M2c** also runs the reverse loop 6 -> 0 -> 6 (32 extra columns in the same task, at almost no cost).
7. **M3b.** Bernoulli samples of the QMF fixed points (B = 32) were added next to the MT3 starts (B = 64). That tests each QMF
   fixed point directly, without the transient from a module start. Per-module counts are recorded every 5 model time instead of
   every 1, for storage. Coexistence is judged by the projection onto the QMF difference region, because the regions (8 to 9,274
   neurons) can be too small to move a 34-module vector by 10%.
8. **M3 starts.** The listed starts add up to 12. MT3's "AL + island 1" was added as the 13th. "rest only" uses MT3's definition
   (all neurons outside the AL, islands included), for comparability with MT3.
9. **Closure-window flag.** A point counts as "closure predicts B1" only for g <= 3.0, the end of the closure grid in
   `theory_module2.json`.
10. **Wrapper.** The spec's `code/pd3_array.sbatch` was used unchanged. Job names are `pd3-multistability-<part>`.

## 8. Caveats and limits (negative results stated plainly)

- **Detection limits.**
  - Hysteresis smaller than the noise floor at 32 replicas per branch cannot be seen. The floor is about 0.08 to 0.13 at
    h = 1e-5 and 0.02 to 0.04 at h = 1e-4, at D = 800.
  - Dwell times were at most 800 and runs at most 5,000 model time. Metastable states living much longer, or switching faster
    than the averaging windows (which would appear as a single mixed state), are not excluded. No bimodality was seen where it was
    tested (M1).
- **The closure's B1 windows at r = 4 (g < 0.4) and r = 6 (g < 0.58)** could not be tested, because the stochastic AL dies there.
  The comparison of the two theories rests on the 17 window points with survivors.
- **The QMF's r = 6 multistability is untestable at g <= 0.52** for the same reason (5 extinct cells).
- **M3 compares 34-module vectors.** A difference confined to a few neurons inside a module would be invisible there. M3b covers
  this for the regions the QMF identified, but not for unknown ones.
- **Drive.** M1, M3 and M3b run at h = 0, where silence is absorbing, so "dies" is a legitimate end-state. The sweeps use
  h = 1e-5 and 1e-4.
- **Partition.** Module identities follow P0 (`module_theory.md` section 2.1: islands 2 to 30 are fragments of one shared region).
- [verification] **M2b placement.** 12 of the 14 M2b centres are not at the onset (5 below the tau = 200 line, 7 above 1.25 x g_on;
  section 3.2), because the pre-registered rule places them where noise in the flagged loops peaked. A dedicated M2b-style test at
  the tau = 3000 onset for each r was not run.
- [verification] **Loop-area slopes** (-0.46 to -0.99) are 3-point descriptive fits limited by the noise floor, not scaling
  exponents. No power law in D is claimed.
- [verification] **h = 0 extinction vs the lifetime map** agrees only to within about one grid step. There is a systematic shift
  to lower g, with sign consistent at all 6 r (section 3.1).

## 9. Files

- **Code:** `code/pd3_multistab.py` (simulator and modes), `pd3_multistab_args.py` (args files, missing-output lists, M2b
  centres), `pd3_multistab_validate.py`, `pd3_multistab_analyze.py`, `pd3_multistab_figs.py`, `pd3_multistab_report.py`.
- **Args and jobs:** `results/pd3/multistability/` (`args_*.txt`, `outputs_*.txt`, `JOBS.md`).
- **Data:** `results/pd3/multistab/` (`validate_r1*.npz`, `al_sym_r*.npz`, `sweep_*.npz`, `rsweep_g*.npz`, `ic_r*_g*.npz`,
  `icq_r*_g*.npz`, `m2b_*.npz`, `validation.json`, `analysis_*.json`, `summary.json`, `analysis_run.log`; `smoke/` holds the
  smoke test only).
- **Summary:** `results/pd3/multistability.json`.
- **Figures:** `figs/pd3/multistab_AL_symmetry.png`, `multistab_hysteresis.png`, `multistab_ic_attractors.png`,
  `multistab_qmf_cells.png`, `multistab_m2b.png`.
- **Verification:** `code/pd3_verify_multistability.py` -> `results/pd3/verify_multistability.json`; `code/pd3_verify_multistability_figs.py`
  (regenerates `multistab_AL_symmetry.png` with the corrected panel-(f) label).

## 10. Verification (adversarial check, 2026-11-09)

Independent recomputation from the raw `results/pd3/multistab/*.npz` by `code/pd3_verify_multistability.py`, which does not
import the program's code (numpy / scipy / sklearn only). Output: `results/pd3/verify_multistability.json`. I also checked the
methods, looked at all five figures, and checked each claim above against the data.

### 10.1 Headline numbers recomputed (all reproduce)

| claim | recomputed | ok |
|---|---|---|
| swc.py raw normalisation 1/c = 745.91 (742.71 = total/N) | 745.910 over 166,499 nonzero-in neurons; total/N 742.712 (V0) | yes |
| validation r = 1, g = 0.30: tau 373 [292, 484] vs pooled 301 [236, 391] | 372.8 [291.9, 484.0] snapped (exact 357.7), 64/64 deaths; pooled 301.1 [235.8, 391.0] (V1) | yes |
| M1: 0 of 48 points meet c1 / c2 / c3 | 0 / 0 / 0 of 48 (own GMM: max delta-BIC 2.63 at r = 2, g = 1.608) (V2) | yes |
| M1: t_sym medians 1.0 to 7.0; largest q75 12.6; largest replica 59; 71 AL-died-first; 0 censored | 1.0-7.0; 12.6; 59; 71; 0. Coarse (every-model-time) records only: 1.0-8.0 (V2) | yes |
| M1: smallest KS p (L vs R) 0.021 at r = 0, g = 0.665 over 44 testable points | 0.0215 at (0, 0.665), 44 tests (V2) | yes |
| M1: late mean A of starts with >= 10 survivors in [-0.001, 0.047]; largest replica 0.189 | [-0.00096, 0.0466]; 0.1886 (V2) | yes |
| M1: 21 closure-window points, 17 with >= 10 survivors per one-lobe start, KS 0.059 to 0.94 | 21, 17, 0.059-0.944 (V2) | yes |
| M2a: A_rho table (D = 50 / 200 / 800, all 12 rows) | identical to 3 decimals (deterministic). Own bootstrap SEs and random-half floors agree within resampling noise, e.g. r = 0, h = 1e-5, D = 800: 0.263 +- 0.057, floor 0.072 (V3) | yes |
| M2a: h = 1e-5 loops 5.3 to 9.9 SE above floor at D = 50, -1.2 to 3.3 at D = 800 | own floor/SE: 5.4 to 9.4 at D = 50; -1.4 to 3.4 at D = 800 (differences are resampling noise) (V3) | yes |
| M2a h = 0: median extinction g / tau = 200 crossing per r | identical medians and IQRs; crossings 0.234, 0.265, 0.283, 0.320, 0.373, 0.435 (V3) | yes |
| M2b: 0 of 14 coexist; smallest late KS p 0.093 (rho) and 0.21 (PR); first-10 KS p >= 0.059 | 0.0935 (r = 6, h = 1e-5, g = 0.296); 0.2115; 0.0592 (V4) | yes |
| M2c: areas 0.101 / 0.132 / 0.037 / 0.046 / 0.029 / 0.030; X_AL 0.279 -> 0.076; 14 of 3,200 one-lobe dwells | identical areas; 0.2793 -> 0.0762; 14 of 3,200 (V5) | yes |
| M3: largest rel L1 between start means 0.077 (r = 4, g = 0.7); r = 4, g = 0.4 all dead; one cluster | 0.0766 at r4_g0.7; r4_g0.4 the only all-dead cell; every survivor within rel L1 0.094 of its cell's grand mean (V6) | yes |
| M3b: 40 cells, 5 all-dead (r = 6, g 0.330-0.518); 133 KS, min p 0.006 (r = 4, g = 0.727, all->isl3, MT3, 59 vs 16, medians 0.407 / 0.384); max median-z difference 0.034; z medians -0.012 to 1.047; early/late drift <= 0.039; r = 6, g = 0.58 survival 0-12.5% | all identical (V7) | yes |
| GPU time ~19.4 h over 132 outputs | 19.37 h over 132 npz (F4) | yes |

### 10.2 Errors corrected

1. **Section 0 (row Q1 broken symmetry) and section 2.** The text said "every replica died before t = 500" at r = 6, g = 0.43. In
   fact all 256 died before T = 1000, but 16 were alive at t = 500 and the last died at t = 934 (F1). The r = 4, g = 0.29 / 0.32
   statements were right (last deaths at t = 165 / 342). The conclusion (no survivors, untestable) is unchanged.
2. **Figure `multistab_AL_symmetry.png`, panel (f).** It carried the same hardcoded "extinct before t = 500" label. It was
   regenerated from `code/pd3_verify_multistability_figs.py`, a copy of `pd3_multistab_figs.py` that only computes this label from
   the raw extinction times; all other panels are unchanged. The original script was not modified.
3. **Section 0 (predictions) and section 4: island starts.** "A single island usually cannot [ignite the AL] at g <= 0.7" is
   false at r = 0, g = 0.7, where the five island starts survive in 72% to 100% of replicas. It now reads "at g = 0.4 (any r) and
   at g = 0.7 for r >= 1" (V6).
4. **Section 0 Q2-e, section 3.1 and section 6: "transition" vs "crossover" language.**
   - "This removes / rules out the first-order alternative" was overstated. It is now "argues against within the detection
     limits".
   - "Leaves the finite-size-rounded second-order reading standing" now also names the alternative it leaves standing: a smooth
     rare-region crossover with no sharp transition.
   - "A lag loop, the signature of a continuous transition" was wrong: lag loops appear at any transition swept at finite rate.
     What argues against first order is the loop closing with D.
5. **Section 3.1 and section 6: the h = 0 down-sweep reading.**
   - "a lifetime crossover that moves with the observation time, not a spinodal" was overstated. Only D = 200 was run, so the
     shift with observation time is inferred. A metastable branch would also die where tau ~ D, so this is a consistency check,
     not a discriminating test.
   - The KS p-values on tied grid data are conservative. An exact Monte-Carlo multinomial G-test gives p 0.33 to 1.0 (no misfit).
   - However, the mean ln g_ext is below the tau(g) prediction at all 6 r, by 0.6 to 1.0 grid step (sign test p = 0.031,
     Stouffer p = 0.002; F2). "Dies where tau = D" holds only to within about one grid step.
6. **Section 0 (row Q1 first-order lines) and section 6: dwell scaling.** "Shrink as about D^-0.5 to D^-1" read like a scaling law.
   It is now the fold change (3.6 to 15.5 from D = 50 to 800), with the 3-point slopes labelled descriptive and floor-limited.
   The two segment slopes disagree, for example r = 6, h = 1e-5: -1.55 then -0.42 (V3).
7. **Section 6, Q1: "No (g, r) shows hysteresis beyond noise"** is now "that persists as the dwell grows". A label-permutation test
   (2,000 whole-column swaps, not using the noise-floor estimate) finds two loops still open at D = 800, both still shrinking:
   r = 0, h = 1e-5 (p < 0.001) and r = 0.5, h = 1e-5 (p = 0.02). All other D = 800 loops have p >= 0.12 (V3).
8. **Section 3.2: M2b placement (added).** Only 2 of the 14 M2b centres are at the onset. 5 sit below the tau = 200 line, where
   the drive dominates (2 to 33 active neurons, AL silent in 54% to 99% of late records), and 7 sit above 1.25 x g_on (F3, F5).
   M2b's "no coexistence" is therefore mostly a test away from the onset. A caveat was added to section 8.
9. **Section 3.2: "one state".** It now says one shared stationary distribution. Near the onset that distribution has intermittent
   AL silences: up to 43% of a replica's late records at r = 5.75, g = 0.6, the same for both groups (KS p = 0.84) (F5).
10. **Section 3.1, h = 1e-4 at D = 200.** "At the noise floor from D = 200 on" now carries an exception: r = 1 has permutation
    p = 0.02 at D = 200. At D = 800 all six have p >= 0.12.

### 10.3 Methods checked, no change needed

- **Simulator.** `pd3_multistab.py` uses the same update rule, dt, p_off, sign convention and normalisation as swc.py
  (`--norm raw`).
- **Branch alignment.** Up columns at g_grid[k] in segment k, down columns at g_grid[K-1-k]. Extinction segment (e - 1) // steps.
  M2b snapshots are taken from the matching segments.
- **M2b replica independence.** M2b duplicates each of the 32 end-state snapshots twice (64 = 2 x 32 per branch). That is
  harmless after T = 5000 of independent noise, but the starts are 32 distinct states per branch, not 64.
- **Pre-registered criteria.** M1 c1-c3, the M2a candidate rule and the M3 clustering rule were applied as pre-registered; the
  noise-floor variant is clearly labelled as an addition.
- **PR loop areas.** The PR/N of the replica-mean occupancy at g below about 0.2 grows with D (sparse drive-induced activations;
  `multistab_hysteresis.png` panel b). The PR loop areas therefore do not shrink with D (A_PR(800)/A_PR(200) = 0.38 to 5.8, `analysis_m2a.json` decisions PR_ratio_800_200). They
  sit at their noise floor, so the conclusion stands, but the D-scaling half of the criterion is uninformative for A_PR.
- **Figures.** All five figures were read. Labels and colours are colour-blind-safe (Okabe-Ito and cividis). Apart from panel (f)
  of `multistab_AL_symmetry.png` (fixed), the plotted values match the analysis JSON and the recomputation.

### 10.4 Verdict

The program's main negative results reproduce from the raw data:
- no left/right symmetry breaking;
- no coexistence at the 14 M2b centres;
- one non-silent attractor in M3;
- none of the QMF multistability survives noise in M3b.

The corrections are one factual error (the t = 500 extinction claim and its figure label), one over-generalisation (island
starts), and over-strong interpretive language (rules out, lag-loop signature, crossover not wall, D-power laws). The verification
also adds two findings: two h = 1e-5 loops are still significantly open at D = 800 although still shrinking, and the h = 0
extinction sits systematically about one grid step below the lifetime-map prediction.
