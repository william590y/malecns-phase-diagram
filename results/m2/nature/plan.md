# Lane m2/nature: the nature of Model 2's transitions (plan, written 2026-11-09 15:00 EST, before any run of this lane)

Spec followed: `results/m2/MODEL2.md` (md5 2781e09d), sections 1-7, especially section 6 (pre-registered criteria for the
nature of the transitions), copied below and only tightened. Simulator `code/ald_sim.py` md5 61580c76 (checked in every
task log by `code/m2_array.sbatch`). Argument lines from `m2_common.args_line()` (md5 a9f3ee1e) or, for `--graph` runs,
the same protocol flags with the extra regions removed (MODEL2 open issue: a region selector that matches nothing on a
small graph raises). Observables from `m2_obs` (md5 8be25a8b). Jobs `m2-nature-*`. Code `code/m2_nature_*.py`. Results
here, figures `figs/m2/nature_*.png`.

Question: at each M2 boundary, is the change a smooth crossover, a continuous (finite-size-rounded) transition, a
first-order transition, or self-organized (critical over a range of g)? Plus: is there a Hopf-like oscillation onset?

## 0. Inputs: where the boundaries are (not re-estimated on the full graph)

The decay and driven lanes have finished full 40 + 24-point sweeps at every r on the full graph, so I take the
full-graph boundaries from their files instead of running my own coarse full-graph sweeps (that would duplicate ~100
GPU-h). I do run my own coarse sweeps, then refinements, on every finite-size graph (section 1), where nothing exists yet.
- setting (a): `results/m2/decay/boundaries.json` (md5 176f8aea), rows `M2_real_r*`, `M1_real_r*` (M1 r 0/1/4) and
  `m1_reference` (M1 other r, analysis2), `M2_cnull_r*`, `M2_cnull1_r1`. R_CORE values used here:

| r | M2 S | M2 onset | M2 D | M1 S | M1 onset | M2 cnull (null_seed 0) S / onset |
|---|---|---|---|---|---|---|
| 0 | 0.981 | 1.18 | 6.29 | 0.233 | 0.262 | 1.41 / 1.61 |
| 0.25 | 1.10 | 1.33 | 13.1 | 0.251 | 0.287 | 1.30 / 1.48 |
| 0.5 | 1.18 | 1.43 | 14.1 | 0.265 | 0.306 | 1.28 / 1.44 |
| 1 | 1.21 | 1.57 | 13.9 | 0.286 | 0.347 | 1.26 / 1.37 |
| 2 | 1.34 | 1.72 | 11.9 | 0.320 | 0.402 | 1.29 / 1.37 |
| 4 | 1.47 | 1.94 | 9.29 | 0.383 | 0.495 | 1.30 / 1.40 |
| 6 | 1.56 | 2.02 | 8.03 | 0.436 | 0.579 | 1.31 / 1.42 |

- setting (b): `results/m2/driven/` (summary.json md5 5351baf9, curves.json md5 9415d915, table.md): DT->SS = (a) onset
  x0.97-1.03 at every r; DQ->DA 0.22-0.47; allowed top ~ (a) S.
- So M2 has, per r, three candidate "transitions" to characterise: (T1) the h = 0 onset of sustained activity (S -> onset
  window; = the driven DT -> SS line), (T2) the localized -> delocalized line D, (T3) whatever happens to avalanches below
  T1 (the self-organized-criticality question). T2 is a PR convention line far above T1 (D/onset 4-10); I test it only
  by the order parameter / hysteresis pictures, not by FSS (cost; section 6 of MODEL2 asks FSS for the (a) crossover).

## 1. Finite-size scaling (setting a)

**Graphs** (`code/m2_nature_graphs.py`, written once, then read-only; exact g_lin of every graph computed with
`pd3_common.g_lin_exact`, and checked against pd3's values for the identical cnull / realsub graphs):
- **F-cnull**: configuration nulls `write_graph("cnull", N, s)` (pd3 construction, own c), N in {5k, 10k, 20k, 40k, 80k},
  s in {1, 2, 3}; and N = 167,216 as `ald_sim --null --null_seed s`, s in {0, 1, 2} (s = 0 at r 0, 1, 4 and s = 1 at r 1
  are the decay lane's rows `M2_cnull_r*` / `M2_cnull1_r1`, reused; the rest run here).
- **F-realsub** (as MODEL2): real subsamples keeping the AL module, `write_graph("realsub", N, s)`, c fixed at the
  full-brain value, N in {20k, 40k, 80k}, s in {1, 2}; full brain = decay lane rows `M2_real_r*`.
- **F-realsub-cn** (ADDED, deviation explained): the same subsamples with c renormalised to the subgraph's own mean
  in-strength (graph file without `c`, so ald_sim computes 1/mean in-strength of the subgraph). Why: in M1 the onset
  was carried by the AL module, which the fixed-c subsample keeps intact. In M2 the first sustained state is brain-wide
  (AL 5-6 % of activity, decay lane), and the fixed-c subsample cuts every non-AL neuron's input to ~N/167k of its
  value, so that family changes the coupling as well as N (decay lane x9: realsub 80k r 0 onset ~1.9 vs 1.18 for the
  full brain). With c renormalised, the mean input per neuron is kept. Both families are reported; each is judged
  separately by the same rules.
- r: F-cnull and F-realsub-cn at r in {0, 1, 4}; F-realsub (fixed c) at r in {0, 1} (it is the less informative family).

**Runs**: protocol `decay_h0` (M2 flags, B 64, T 3000, total counts 1500:3000), default regions only on `--graph` runs.
- Stage 1: g = G_COARSE points in [0.69, 11.2] (18 points) for N <= 80k; for the 167k nulls G_COARSE in [0.96, 3.02]
  (8 points). If a curve has not reached tau = 3000 by the top point, it is extended by 4 coarse points.
- Stage 2: `refine(lo, hi, 8)` inside the bracketing coarse cell of tau = 200 and of tau = 3000 (one set if the same cell).

**Per graph and r** (same estimators as the decay lane): tau = censored MLE (`m2_obs.extinction_times`, `tau_mle`);
isotonic ln tau in g (weights deaths + 0.5), first crossings of 200 (S) and 3000 (onset) in ln g (`first_crossing`).
- **W = ln(g_on / g_S)** (MODEL2 section 6). Family value at each N = mean over seeds (SD reported).
- Secondary (reported, not used in the decisions): W' = ln(g(tau = 3000)/g(tau = 500)) (less sensitive to the early-death
  mode the decay lane found at t ~ 30-100); onset location g_on(N) and x_on = g_on/g_lin(graph); QS density rho_QS and
  chi = N var_t(x) of survivors at each g (order parameter and its fluctuation; height of the chi maximum vs N).

**Decisions (pre-registered; MODEL2 section 6, tightened)**:
- **Sharpening** (cnull): seed-mean W falls monotonically with N, Spearman rho(W, N) <= -0.9 over the 6 sizes, AND
  W(167k)/W(5k) <= 0.5.
- For a real-subsample family (sizes 20k, 40k, 80k, full):
  - **finite-K-rounded crossover** if every seed-mean W(N) is within +-20 % of W(full);
  - **sharpening** if Spearman rho(W, N) = -1 (4 sizes) AND W(full)/W(20k) <= 0.6 (added rule: MODEL2 gives none for
    realsub);
  - otherwise **inconclusive / drifting**, with the direction stated.
- These are applied at each r separately and summarised.

## 2. Driven finite-size check (setting b)

MODEL2 section 6: "height of the chi_AL (or chi_nonORN) maximum against N, as a growth exponent".
- Graphs: F-cnull N in {10k, 20k, 40k, 80k}, s in {1, 2}; F-realsub-cn N in {20k, 40k, 80k}, s in {1, 2}. Full-size
  points from the driven lane (A1 real, A3 cnull `--null_seed 0`; same protocol). r in {0, 1}.
- Protocol `driven` (M2, ORN 0.04, B 64), regions: default + ORN + nonORN (extra regions dropped on subgraphs).
  g = G_COARSE in [0.42, 11.2] (20 points).
- chi_nonORN = n_nonORN <var_t rho_nonORN>_b over the driven window (500, 1500]; chi_AL as well on realsub-cn (AL kept).
  Peak = maximum over g of the replica-bootstrap mean (only interior maxima count; a maximum at the grid edge is reported as
  "edge").
- **Growth exponent a**: least-squares slope of ln chi_max (seed mean) vs ln N.
  - a >= 0.5 AND the peak g approaches the family's DT->SS line -> "divergence-like (transition)";
  - a <= 0.2 -> "no divergence (crossover)";
  - otherwise inconclusive.
  (Reference: pd3 M1 chi_win slopes 0.65 cnull, 0.15 realsub.)
- DT->SS on these graphs: tau_post = 3000 crossing (MODEL2 5b), reported for the peak-tracking statement.

## 3. Hysteresis (MODEL2 section 6, tightened)

- Protocols: `ramp_driven` (ORN 0.04; quiet start; g ascending then back) and `ramp_h0_down` (h = 0; fully active start
  at the top g; g descending then back up; pulses on ORNs with gap G = 10, i.e. a silent replica is re-seeded after a mean
  of ~10 time units, short against every dwell).
- Models M1 and M2, r in R_CORE (7), dwell D in {50, 200, 800}, B = 32, one seed each.
  Extensions (tightening): D = 3200 for M2 at r in {0, 1, 4}, both protocols; a second seed at D = 800 for M2 at
  r in {0, 1, 4}, both protocols (reproducibility of A(800)).
- g grid per (model, r): g = x * g_on(model, r) with x = geomspace(0.5, 2.0, 17) (Delta ln g = 0.0866); g_on from section 0.
  This window contains S, the onset / DT->SS, and the silent-lost line of the driven lane (~S), for both models.
- Signal: ramp_driven: nonORN active fraction (`seg_reg`, region nonORN); ramp_h0_down: total active fraction
  (`seg_rho`). Each value = mean over the last half of the dwell.
- **Loop area** A(D) = sum over the 16 g visited twice of |<x>_up - <x>_down| * Delta ln g (replica means; the same 32
  replicas go up then down). Also reported normalised by sum_g (<x>_up + <x>_down)/2 * Delta ln g.
- **Noise floor**: 2000 label permutations, each swapping up/down independently per (replica, g); p = fraction of
  permuted areas >= observed; floor = 99th percentile.
- **First-order (hysteresis)** if A(800) > the 99th percentile of its permutation distribution AND A(800)/A(200) >= 0.7
  AND (where run) A(3200)/A(800) >= 0.7. **Finite-rate effect** if A(800)/A(200) < 0.5 (or A(3200)/A(800) < 0.5).
  Otherwise inconclusive.
- **Bimodality**: per (g, branch), the 32 per-replica values; bimodality coefficient BC = (skew^2 + 1)/(excess kurtosis +
  3 (n-1)^2/((n-2)(n-3))) > 0.555 AND both sides of the largest gap in the sorted values (Otsu split) hold >= 10 % of the
  replicas. Counted per run. Only ramp_driven bimodality counts as first-order evidence: in ramp_h0_down an extinct
  replica (re-seeded avalanches only) next to an alive one is the absorbing state, expected in any finite system.
  First-order support from bimodality needs >= 2 adjacent bimodal g on the same branch at D = 800.

## 4. Avalanches and the self-organized-criticality test (MODEL2 section 6)

**(i) Pulse protocol** `aval_pulse` (h = 0, quiet start, one ORN seeded in a silent replica; gap G), B 64, T 3000.
- Models M1 and M2, r in R_CORE, G in {0, 10, 100}. g = G_COARSE in [0.1, 3] (21 points, 0.1135-3.019), truncated per
  (model, r) at g <= 1.25 x g_on(model, r): above the onset the network does not return to silence and avalanches stop.
- Avalanche = maximal run of steps with activity; size S = onsets incl. the seed; duration T in steps
  (`m2_obs.avalanches`, censored runs dropped).
- Fits per point: `powerlaw_fit` on S (Clauset KS scan) and on T; <S>(T) slope gamma (`mean_size_given_T`, T >= 2, >= 5
  avalanches per T); crackling prediction gamma_pred = (tau_T - 1)/(tau_S - 1); MF-DP: tau_S 3/2, tau_T 2, gamma 2.
- Shape collapse (reported): mean profile of onsets over avalanches with T in [T0, 2 T0] for T0 in {10, 20, 40, 80} steps,
  rescaled a/T^(gamma - 1) vs t/T; gamma_collapse = the gamma minimising the variance between rescaled profiles.
- Branching ratio: MR estimator (`mr_branching`) on the per-step onset counts pooled over replicas, m_step; reported per
  model time unit m_unit = m_step^5 (5 steps = 1 unit = the mean active epoch).
- Stage 2: points with < 2000 avalanches in the tail-fit range or n_tail < 1000 but otherwise critical-like get up to 3
  extra seeds (pooled).
- **Critical-like** g (MODEL2): tau_S in [1.3, 1.7], KS <= 0.03, xmin <= 10, n_tail >= 1000, largest size / xmin >= 100.
- **LHG test** per (r, G) cell, span = max/min of the critical-like g values in the longest run of consecutive coarse grid
  points (the all-points max/min is also reported):
  - supported: M2 span >= 1.5 AND (M1 span < 1.2 or M1 has none);
  - not supported: M2 span < 1.2 (including no critical-like g in M2 while M1 has some);
  - inconclusive: neither model has any critical-like g;
  - mixed: anything else.
  Overall verdict = the majority class over the 21 cells (counts reported). Only coarse-grid points enter the span
  (identical grids for M1 and M2; any extra points are reported separately).

**(ii) Driven regime** (no new runs; the driven lane's raw files: A1 M2 23 r, A2 M1 R_CORE, A3 nulls, A4 M2s; all have
per-step region counts in (500, 1500]):
- MR branching ratio m of the nonORN onset counts (`cnt_reg_on`, region nonORN), m_unit = m_step^5;
- thresholded avalanches of the nonORN active count (threshold = each replica's median; size = area above it; Poil et
  al. 2012), with the same fits.
- **Marker**: |1 - m_unit| < 0.01 over a g range of factor >= 1.5 (tighter than MODEL2's per-step |1 - m| < 0.01,
  which at dt = 0.2 is met by any autocorrelation time > 20 units). Per-step values are reported too.

## 5. Oscillations (Hopf-like onset?)

- Data: decay lane per-point OSC metrics (`results/m2/decay/osc.json`, M2 real 23 r x 64 g, nulls, M1) and its
  finite-size check (`osccheck.json`, r 0); driven lane OSC (points.csv); my FSS runs (r 0, 1, 4; nulls and
  real subsamples, counts 1500:3000).
- Per r: OSC / OSC-strong points, period P(g), amplitude std_t(rho) and CV_t, coherence C(2P)/C(P), periodogram peak
  ratio, against g.
- **Hopf-like onset** at g_H in a row if, for g >= g_H on >= 3 consecutive grid points, OSC-strong holds AND the variance
  does not fall with N (slope of ln var_t vs ln N at matched rho_QS >= -0.3, decay lane rule).
- **Noise-driven quasi-cycle** if OSC (not strong) and the slope <= -0.7. Else inconclusive.
- The amplitude-vs-g law is fitted (A^2 linear in g - g_H) only if a Hopf-like onset is found.

## 6. Order parameter (MODEL2 section 6)

QS active fraction (decay_h0, survivors, 1500-3000) vs g from the decay lane's sum files (M2 real, R_CORE) and my FSS
runs. A first-order onset needs a jump in rho_QS together with hysteresis or ramp_driven bimodality; a jump alone does not
count (survivor conditioning makes apparent jumps).

## 7. Verdict rules (per transition, stated before seeing results)

For T1 (the onset of sustained activity, = DT->SS) at each r:
- **first-order** if section 3's first-order rule holds (either protocol), or ramp_driven bimodality (section 3) holds;
- otherwise **continuous transition, finite-size rounded** if F-cnull sharpens AND at least one real-subsample family
  sharpens or drifts toward a sharper crossover with N;
- otherwise **finite-K-rounded crossover** if a real-subsample family is finite-K-rounded (W within +-20 %);
- otherwise **inconclusive** (with the direction stated).
For T3 (avalanches below T1): **self-organized (LHG)** if section 4's overall verdict is "supported"; "not supported" or
"inconclusive" otherwise. The driven m marker is reported next to it.
For oscillations: **Hopf-like**, **quasi-cycle**, or **none / inconclusive** per section 5.

## 8. Costs and job hygiene

Estimates (MODEL2 9.2 costs; small graphs 5-10x cheaper): FSS ~25-30 GPU-h, driven FSS ~5, hysteresis ~11, avalanches
~8 GPU-h. Arrays with `%` limits ~40-60; preempted / failed tasks are rerun with the same line and seed (missing-output
scan, `code/m2_nature_rerun.py`). Job ids in `results/m2/nature/JOBS.md`. GPU partitions gpu, ellis only;
`--constraint="ampere|ada|hopper|blackwell"`, the standard exclusions, `--nice=500`.

## Amendment 1 (2026-11-09 15:06 EST; written after the graph files and their g_lin were computed, BEFORE any FSS output existed)

The g_lin job (`results/m2/nature/graphs/graphs.json`) shows that renormalising c on the AL-keeping subsample
multiplies c by 4.1 (20k), 3.2 (40k), 2.0 (80k): the AL module keeps ALL its internal synapses, so its coupling is boosted
by that factor (g_lin falls to 0.040-0.083) and the AL could again lead the onset, as in M1. Neither AL-keeping family is
then a clean "same network, fewer neurons" family for M2's brain-wide onset (fixed c dilutes the bulk; renormalised c
over-couples the AL). Added before any result:
- **F-realsubu** (new, PRIMARY real-graph family): a UNIFORM random N-subset of the real brain (AL not forced in), induced
  subgraph, c renormalised to the subgraph's mean in-strength, N in {20k, 40k, 80k}, s in {1, 2}, r in {0, 1, 4}; full brain
  = decay lane rows. Every neuron keeps, in expectation, its mean input; all modules (the AL included) are thinned alike.
  Also run in the driven FSS (section 2).
- Section 7 is amended: the real-graph statement in the T1 verdict is made from F-realsubu (sharpening / finite-K-rounded /
  inconclusive by the section-1 rules). F-realsub and F-realsub-cn keep their own verdicts and are reported next to it as
  checks of how the AL-keeping construction changes the answer. The cnull rule is unchanged.

## Amendment 2 (2026-11-09 15:22 EST; before any avalanche result was analysed or looked at beyond one smoke file)

`m2_obs.powerlaw_fit` evaluates the empirical CCDF at every sorted sample, so tied values (most avalanche sizes are small
integers) inflate its KS distance. On 1e5 exactly critical Galton-Watson avalanches its KS is 0.37 / 0.09 / 0.048 at
xmin 1 / 5 / 10 and its scan picks xmin = 55, so the pre-registered critical-like rule (KS <= 0.03 AND xmin <= 10) rejects
a perfect critical branching process (`results/m2/nature/pl_check.json`, `code/m2_nature_aval.py plcheck`). I do not edit
the shared m2_obs.py; this lane uses `pl_fit` in `code/m2_nature_aval.py`: the same MLE, the KS evaluated on unique
values, xmin candidates 1..10 then 20 log-spaced values up to the value leaving 50 tail points (Clauset scan, minimum KS).
Check: critical GW (m = 1) -> tau_S 1.504, xmin 7, KS 0.003, critical-like TRUE; m = 0.9 -> KS 0.048, FALSE; m = 0.8 ->
FALSE. The thresholds of the rule are unchanged. (One smoke file, M1 r 0.25 gap 0, had been fitted with the old function
while testing the code; nothing from it enters a decision.)

## Amendment 3 (2026-11-09 15:23 EST; POST HOC, after seeing the first hysteresis table)

The pre-registered first-order rule fired once, for the ORN-driven M1 ramp at r = 1 (A(800) = 8.8e-6 just above its
permutation 99th percentile 8.5e-6, p = 0.0065; A(800)/A(200) = 0.72). There A(200) and A(50) are themselves at the noise
floor (p = 0.41 and 0.62), so the ratio test compares two noise values: the rule did not anticipate a loop that is absent
at every dwell. 28 (protocol, model, r) tests at the 1 % level give about 0.3 false positives. Added, labelled post hoc in
every table: two more seeds at D = 800 and one D = 3200 run for that cell, and D = 3200 runs for the M1 h = 0 cells whose
D = 800 loop is still above its floor (r 0, 0.25, 0.5, 2; plus a second D = 800 seed at r 2). Reading: if the extra
seeds' A(800) fall to the floor, the trigger was noise. The pre-registered verdict is still reported as it came out.
