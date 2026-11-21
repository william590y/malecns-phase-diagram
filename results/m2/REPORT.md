# Model 2 phase diagrams: verified synthesis (lane m2/verify, 2026-11-09)

I tried to refute the synthesis. All of its simulated numbers held up: every line I recomputed from the raw files matched, and 16 of 16 lines rerun on the GPU with new random seeds matched. The main negative results also held. One of them is the palp-intact failure at r 0, which I recomputed with 256 new replicas.

Nine statements needed correcting. Most are about how results were described: null allowed regions, the mean-field D line, oscillations, the time base of the branching ratio, the palp limit, how blind the theory lanes were, and the r 0 finite-size cut. None changes a phase boundary or a main verdict.

All paths are relative to /MALECNS_ROOT.

## 0. How I checked
- **Plan written first.** `results/m2/verify/plan.md` was written at 20:10, before I computed any number.
- **Recomputed from raw files** with my own code (`code/m2_verify.py`; it uses only numpy and scipy and no project module) → `results/m2/verify/verify.json`. Everything I checked matched the lane values within the 2% rule:
  - **Decay (h = 0).** S, onset and D at r 0, 0.25, 1, 4 and 6, plus both nulls.
    - r 0: 0.981 / 1.179 / 6.29.
    - r 1: 1.210 / 1.570 / 13.88.
    - r 6: 1.557 / 2.018 / 8.03.
    - Real/null onset: 0.733 (r 0) and 1.421 (r 6).
    - At the first sustained g, the AL share is 6.1% (r 0), 6.2% (r 1) and 5.0% (r 6). The PR is 13.6k, 14.6k and 19.1k neurons.
  - **Driven.** DQ→DA, the allowed top and DT→SS at all 23 r.
    - r 0: 0.216 / 0.968 / 1.211.
    - r 1: 0.317 / 1.234 / 1.570.
    - DT→SS / decay onset is 0.97–1.03; allowed top / decay S is 0.94–1.03; DT→SS / allowed top is 1.23–1.33.
    - M1, both nulls, M2s and the 0.01 / 0.10 ORN levels all reproduce.
  - **FSS.** Configuration null, W(167k)/W(5k): 0.506 (r 0), 0.168 (r 1), 0.187 (r 4). Real uniform subsamples, W(full)/W(20k): 0.61–0.67.
  - **Hysteresis.** A(50)/A(800) is 17–39 in the 4 cells I checked.
  - **Branching ratio.** My own multistep-regression code gives m per step = 0.98805, identical to the lane.
  - **Lifetimes, real r 1, g 1.57.** Late lifetime 4194, 13.3% dead by t = 200, residual-lifetime KS p = 0.15.
  - **Modules alone, r 0.** AL 2.18, CB 1.41, VNC 3.36, OL 5.00.
  - **Variants.** TAU 30 ×0.486, KW09 ×1.41, N15 ×0.73, all-synapse ×1.02, M2s ×1.06.
  - **g_lin.** I recomputed it from the read-only cache with sparse eigensolves, including the exact nonlinear branching operator: 0.1631526.
  - **Theory/sim ratios.** MF equal-size r 0: 0.534. Stochastic single mode r 0: 1.409. On(6)/on(0): sim 1.712, MF 1.928, stochastic 2.768.
  - **LNA at r 0.** G = 0.29 at 0.213 (sim 0.215); G = 1.45 at 1.029 (sim 1.050); Fano peak 59.3 at 0.908 (sim 55.2 at 0.993).
- **New GPU runs with new seeds** (array m2-verify-sim 219854, 15/15 tasks, 2.9 GPU-hours, ald_sim md5 61580c76) → `results/m2/verify/reruns.json`. All 16 lines replicate:

  | line | rerun | lane |
  |---|---|---|
  | r 0 S (B = 128) | 0.981 | 0.981 |
  | r 0 onset (B = 128) | 1.197 | 1.182 |
  | r 1 S | 1.222 | 1.210 |
  | r 1 onset | 1.538 | 1.571 |
  | r 6 onset | 2.087 | 2.018 (inside its total interval) |
  | r 0 D | 6.299 | 6.294 |
  | configuration null r 0, same graph | 1.573 | 1.607 |
  | configuration null r 0, new graph (null_seed 2) | 1.592 | real/null 0.742, vs 0.735 |
  | driven r 0 DQ→DA / top / DT→SS | 0.215 / 0.978 / 1.208 | 0.216 / 0.968 / 1.211 |
  | driven r 1 DQ→DA / top | 0.318 / 1.219 | 0.317 / 1.234 |
  | CB alone, fine grid | 1.43 | 1.42 |
  | AL alone, fine grid | 2.16 | 2.19 |

  The palp-intact rerun is in correction 6.
- **Simulator.** Live copies of the unit tests passed 29/29, and the observable tests passed 11/11, on the current ald_sim.py (job 219855; outputs in `results/m2/verify/simtest/`). An independent check from production files: the measured ORN efficacy at g 0.05 is 0.41404, against 0.41413 from the exact renewal formula (difference 0.02%).
- **Artefact checks:**
  - **Estimator.** Using exact extinction times instead of log-grid times moves the onset by ≤ 0.3%.
  - **Finite T.** Truncating the T = 30000 runs at 3000 moves the onset by ≤ 0.9% (r 1: 1.547 vs 1.546; r 4: 1.961 vs 1.944).
  - **Replica count.** B = 128 reruns give the same lines.
  - **Fit window.** Residual-lifetime KS p is 0.15–0.41 at 3 long-run points.
- **Pre-registration.** Plan times and outputs agree to within about a minute. Nature amendment 1 (15:06) came about 10 s after the first null-graph FSS output; the family it introduces first ran at 15:39. All amendments are labelled.

## 1. Model 2 phase diagrams (unchanged unless marked [corrected])

**1a. h = 0 decay from a fully active start**
- The synthesis table stands: S 0.98–1.57, onset 1.18–2.02 (7.2–12.4 × g_lin), D 6.3–14.2, AL share 5.0–6.3%, PR 13.6k–19.1k. The phase order is Q → S → L → D at every r, and onset/S is 1.20–1.33.
- Configuration null onset 1.37–1.61. Real/null onset is 0.74 at r 0, and the same on a second null graph (0.742); it rises to 1.42 at r 6.
- M2s is within 6.2% of M2 on every line. The variants are as stated.

**1b. Driven at ORN level 0.04**
- The synthesis table stands, and every driven line is the corresponding h = 0 line seen through ORN removal.
- [corrected] **Palp-intact limit.** M2 passes the palp-intact test only for g ≲ 0.20, not "≤ 0.186":
  - interpolation gives 0.202 on the lane's data and 0.201 on the rerun;
  - 0.186 is just the last passing coarse grid point;
  - the gain edge is 0.216, so no g satisfies all three data constraints at r 0 or r 1. Confirmed.
- [corrected] **Why the palp test fails.** At the gain edge it is the late residual (1000–3000 units after removal) that fails, and clearly: 0.114 [0.112, 0.116]. The early residual is marginal: 0.096 [0.091, 0.102]. The gap between the palp limit and the gain edge is only 7% in g.
- [corrected] **M2s.** The sliver at r 0 runs from 0.235 to between 0.258 and about 0.30. The upper end is set by interpolation between grid points.
- [corrected] **Nulls.** The configuration null has no data-allowed region at any core r. The weight-shuffled null, which keeps the ORN→PN wiring, has a small one at r 0 (0.84–1.15) and at r 0.25 (1.01–1.15), and none at r ≥ 0.5.

## 2. Theory against simulation
All grades stand except those below.
- [corrected] **Mean-field D line.**
  - It agrees within ±10% at 3 of 23 r: r 0 (6.08 vs 6.29), r 2.5 (11.25 vs 11.01) and r 2.75 (9.64 vs 10.62). At r 2.5 and 2.75 it agrees only because the curve crosses over from too high to too low.
  - It is too high at r 0.125 (×1.96) and r 1.75–2.25 (×1.33–2.36), and absent up to g 30 at r 0.25–1.5.
  - It is too low by 19–35% at r ≥ 3 (r 3: 0.81; r 4: 0.69; r ≥ 4.5: 0.65).
  - Over-inhibition explains only r 0.125–2.25. At r ≥ 3 the error has the opposite sign, and those values lie in the mean field's own oscillating regime, which that lane flagged as low-confidence.
- [corrected] **Blindness.** Neither theory lane was blind. Both lanes' instructions included reading STATE.md, which by 17:40 summarised all four experiment lanes. The stochastic lane disclosed this. The meanfield lane (plan ~17:40, predictions 19:43) states only that it opened no experiment file. The pre-registered point estimates failed anyway (×0.53–0.80 and ×1.41–2.28), so the exposure did not produce any agreement.
- The rest stands: brackets, r-trend, LNA at r 0 only, finite-size null onsets 5/6, the qualitative grades, and "real above null at r 0" being wrong (predicted ×1.30, simulated ×0.74, confirmed on a second null graph).

## 3. M1 against M2
The table stands, with one row corrected:
- [corrected] **Oscillations.** On the real M2 graph, damped noise quasi-cycles appear only at r 0, at g 1.15–1.31 (period 73–83 units). The flags at r 0.25, 0.5 and 4 are all on the weight-shuffled null.

## 4. Answers to the program's questions
These stand, with two corrections:
- [corrected] **Branching ratio.** It is 0.94 per unit at most under drive, which is 0.988 per step. Per step, that misses MODEL2's |1 − m| < 0.01 mark by only 0.002 (4 real-M2 points have m per step ≥ 0.985, an autocorrelation time of about 17 units). The pulse maximum is 0.977 per unit (0.995 per step). There is no LHG marker under either convention, but "never within 0.01 of 1" holds only with the per-unit convention the nature lane chose.
- [corrected] **OSC flags.** They are threshold-marginal: the autocorrelation peak is 0.10–0.15 against a threshold of 0.10.
  - With an unbiased-lag autocorrelation estimate, 9 real-graph points are flagged instead of 7.
  - The one real-graph "OSC-strong" point (7 replicas, coherence 0.78) gives −0.17 with the point estimate C(2P)/C(P), so that label is an estimator artefact.
  - The verdict stands: damped, never strong, not a phase.

## 5. Open problems
These stand, with three changes:
- [corrected] **Item 6.** The KS bug in m2_obs.powerlaw_fit affects no other user. A grep finds it only in m2_obs.py, m2_nature_aval.py (as a comparison), m2_smoke_check.py and m2_obs_test.py. pd3 never used it, so pd3 needs no recheck.
- [corrected] **Item 7.** The oscillation finite-size verdict on the real graph depends on how the graph is subsampled:
  - uniform subsamples (the pre-registered primary family) give variance ∝ N^-1.19 at r 0, a quasi-cycle;
  - the AL-keeping families give slopes −0.03 and +0.34, which the nature lane's rule labels "macroscopic" but which are confounded by 2–4× AL over-coupling. The nature lane's summary left these out;
  - the decay lane's AL-keeping check gave −0.57 (inconclusive).
- [corrected] **Configuration-null FSS at r 0.** The W ratio is 0.502 (lane) or 0.506 (mine). A bootstrap over the 3 graph seeds gives 95% interval [0.47, 0.55], with P(ratio ≤ 0.5) = 0.39, so whether it passes the cut is undetermined. The robust feature is that W is flat for N ≥ 40k (0.143, 0.148, 0.136). The verdict stays "inconclusive".

## Every correction
1. "Neither null has a data-allowed region" → true only for the configuration null. The weight-shuffled null has 0.84–1.15 (r 0) and 1.01–1.15 (r 0.25). The synthesis recheck had marked these rows n/a.
2. "Mean-field D agrees only at r 0; 31–36% low at r ≥ 3; same over-inhibition" → agrees at r 0, 2.5 and 2.75; 19–35% low at r ≥ 3, which is the opposite sign; ×1.3–2.4 high at r 0.125 and 1.75–2.25.
3. "Quasi-cycles at r ≤ 0.5" → on the real graph, r 0 only. The flags are threshold-marginal, and the one "strong" flag is an estimator artefact.
4. "Oscillation FSS on the real graph inconclusive" → the verdict depends on the subsample family: −1.19 uniform, −0.03 / +0.34 AL-keeping, −0.57 in the decay lane.
5. "Branching ratio never within 0.01 of 1" → depends on the time base. The driven maximum is 0.988 per step.
6. "Palp passes only for g ≤ 0.186, marginal at about 0.22" → limit about 0.20. The failure is confirmed with 256 new replicas and is set by the late residual (0.114 [0.112, 0.116]). The early residual is marginal. The M2s sliver may reach about 0.30.
7. "pd3 and other users should recheck the KS fit" → no other user exists.
8. "Predictions written before the results" → neither theory lane was blind to the STATE.md summaries, and the meanfield lane did not disclose this.
9. "Configuration null fails the r 0 FSS cut narrowly" → undetermined within seed noise.

## Key numbers

Recompute from raw files (results/m2/verify/verify.json); all within 2% or inside the lane interval:
- decay S / onset / D: r 0 0.981 / 1.179 / 6.294; r 1 1.210 / 1.570 / 13.877; r 6 1.557 / 2.018 / 8.032;
- configuration-null onset: r 0 1.607, r 6 1.420; real/null 0.733 and 1.421;
- at the first sustained g: AL share 6.1 / 6.2 / 5.0%, PR 13.6k / 14.6k / 19.1k (r 0 / 1 / 6);
- driven lines at all 23 r: r 0 0.216 / 0.968 / 1.211; DT→SS / decay onset 0.971–1.027; allowed top / S 0.935–1.033; DT→SS / top 1.226–1.335;
- FSS W ratio: null 0.506 / 0.168 / 0.187 (r 0 / 1 / 4); real uniform subsamples 0.61–0.67;
- hysteresis A(50)/A(800): 17–39; branching ratio per step 0.98805 (per unit 0.942);
- lifetimes, real r 1, g 1.57: late lifetime 4194, early 13.3%, residual KS p 0.15;
- modules alone: AL 2.18, CB 1.41; g_lin 0.1631526;
- ORN efficacy: measured 0.41404 vs renewal 0.41413;
- estimator effect ≤ 0.3%; finite-T effect ≤ 0.9%.

New-seed reruns (results/m2/verify/reruns.json; job 219854; 2.9 GPU-h), 16/16 replicate:
- r 0 S 0.981, onset 1.197; r 1 S 1.222, onset 1.538; r 6 onset 2.087; D r 0 6.299;
- configuration null 1.573 (same graph), 1.592 (new graph; real/null 0.742);
- driven r 0: 0.215 / 0.978 / 1.208; driven r 1: 0.318 / 1.219;
- CB alone 1.43, AL alone 2.16;
- palp r 0 (256 replicas): CONFIRMED. Late residual 0.114 [0.112, 0.116] at the gain edge 0.216; palp limit 0.201.

Tests: unit 29/29, observables 11/11 PASS (job 219855).

Corrections (results/m2/verify/corrections.json):
- weight-shuffled null allowed region 0.84–1.15 (r 0), 1.01–1.15 (r 0.25);
- mean-field D agrees at r 0 / 2.5 / 2.75 and is ×0.65–0.81 at r ≥ 3;
- real-graph OSC flags only at r 0; the "strong" point's coherence is −0.17 by point estimate (lane 0.78);
- oscillation FSS slope −1.19 (uniform), −0.03 / +0.34 (AL-keeping);
- driven branching ratio maximum 0.988 per step;
- configuration-null r 0 W ratio: 95% seed interval [0.47, 0.55], P(≤ 0.5) 0.39.

## Open issues

- No REPORT.md file was written, because the harness blocks report files. The corrected report is in this summary, the corrections are listed in results/m2/verify/corrections.json, and an entry was added to STATE.md. The workflow or the explainer step (explainer/model2.html) should carry the corrected text.
- The palp-intact failure at r 0 is statistically firm but small. The late residual is 0.114 against a 0.10 threshold at the gain edge, and the palp limit (about 0.20) sits only 7% below the gain edge. The verdict therefore depends on E1's 10% convention.
- The OSC flag counts (86 of 3,410 pairs, 3 strong) depend on the autocorrelation estimator (biased vs unbiased lag normalisation; max over a window vs the point C(2P)) because the flags sit at the 0.10 threshold. MODEL2's flag should fix a single estimator before it is used again.
- The branching-ratio criterion's time base is not fixed in MODEL2. Per step, the driven maximum (0.988) misses |1 - m| < 0.01 by only 0.002. Any reuse of that criterion should say which unit it uses.
- I did not recompute these from raw files: tau_int peak locations, avalanche exponents and KS values, left/right symmetry counts, multistability clusters, or the mean-field and stochastic internals beyond the prediction JSONs. They rest on the lane files and the synthesis recheck.
- Figures were not regenerated. The final_*.png figures do not show the weight-shuffled null's allowed region, and they do not show the mean-field D agreeing at r 2.5 and 2.75. Check the figure captions against corrections 1–3.
