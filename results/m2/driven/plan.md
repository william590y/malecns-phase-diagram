# Lane m2/driven: the ORN-driven phase diagram of Model 2 (plan, written 2026-11-09 11:10, before any run)

Spec followed: `results/m2/MODEL2.md` (sections 1, 2, 4 "driven", 5b, 5c, 7). Simulator `code/ald_sim.py` md5 61580c76
(checked in every task log). Argument lines from `m2_common.args_line()`, observables from `m2_obs`. Jobs `m2-driven-*`.
Code: `code/m2_driven_args.py` (task lines), `code/m2_driven_analyze.py` (per-point table, boundaries, peaks),
`code/m2_driven_figs.py` (figures `figs/m2/driven_*.png`).

## 1. What runs

Protocol `driven` exactly as MODEL2 section 4: quiet start, ORNs driven for 0 < t <= 1500, every ORN clamped to 0 at
t = 1500, run to 4500; B = 64; windows 500:1500 / 1600:1800 / 2500:4500; occupancy over 500:1500; `--rec_counts
--counts_win 500:1500 --counts_reg` (per-step region counts in the driven window, needed for the nonORN autocorrelation,
oscillation flag and stationarity check).

| set | model | graph | ORN level | r | g | points |
|---|---|---|---|---|---|---|
| A1 main | M2 | real | 0.04 | R_ALL (23) | G_COARSE (40, 0.05-30) | 920 |
| A2 reference | M1 | real | 0.04 | R_CORE (7) | G_COARSE | 280 |
| A3 nulls | M2 | configuration null, weight-shuffled null (`--null_seed 0`, as the smoke runs) | 0.04 | R_CORE | G_COARSE | 560 |
| A4 comparison | M2s | real | 0.04 | R_CORE | G_COARSE | 280 |
| B1 ORN levels | M2 | real | 0.01, 0.02, 0.07, 0.10 | R_CORE | G_COARSE, g <= 11.21 (34) | 952 |
| B2 depression sensitivity | M2_allsyn, M2_KW09, M2_N15, M2_tau30 | real | 0.04 (M2_tau30 also 0.10) | 0, 1, 4 | G_COARSE, g <= 11.21 | 510 |
| B3 attribution | M1s | real | 0.04 | 0, 1, 4 | G_COARSE, g <= 11.21 | 102 |
| B4 palp-intact removal | M2, M2s | real | 0.04 | 0, 1 | G_COARSE in [0.096, 3.02] (21) | 84 |
| C refinement | as needed | | | R_CORE | `refine(lo, hi, 8)` inside bracketing cells (section 4) | ~ 300 |

**Deviations from MODEL2, and why.**
- MODEL2 asks for R_CORE for variants; the task asks for "a subset". B2/B3 use r = 0, 1, 4 (cost). M2s (the named
  comparison variant) gets all of R_CORE.
- B1-B3 stop at g = 11.21 (34 of 40 grid points): the dense high-g points cost the most (~250 s each) and the lines
  these sets are for (DQ->DA, allowed top, DT, SS onset) are expected at g < 4 (M2 r = 1 SS ~ 3 in the smoke run;
  M1's r = 6 onset was 1.7x its r = 1 value). If any B curve has not reached SS by 11.21 it is extended. Consequence:
  the SS-L/SS-D line is mapped only for A1-A4.
- M2_tau30 is also run at ORN 0.10, the closest level to its own mapping (MODEL2 section 2). Under that mapping the
  removal windows (100-300 units = 5-15 s) are not E1's 1-3 s, so its allowed region is reported but not interpreted
  as a data constraint.
- The M1 reference (A2) and M1s (B3) at ORN 0.04 are added: an M1 driven diagram over r does not exist (E1 ran r = 0, 1).
- Seeds: one per task line (`index.json`); the null graphs are fixed by `--null_seed 0` across tasks.

## 2. Observables per point (MODEL2 5b, exactly)

Driven window W_d = (500, 1500]; replica means unless stated.
- rho_drv = win_mean[nonORN], rho_ORN (measured, reported next to every point), rho_uPN_ant, rho_AL_module.
- **Gain** G = rho_uPN_ant / rho_ORN.
- **Fluctuation** chi_X = n_X <var_t rho_X>_b (var_t = win_m2 - win_mean^2 per replica), X = nonORN (n 164,577) and
  AL_module (526); chi_X / <rho_X>.
- **Autocorrelation time** tau_int, tau_e of rho_nonORN(t) from cnt_reg_act[nonORN] (`m2_obs.osc_metrics`, dt 0.2).
- **PR_nonORN** of occ over the 164,577 nonORN neurons; -D if PR_nonORN >= 164,577/4.
- **OSC** flag (`osc_metrics` on rho_nonORN(t) in W_d, all replicas; P <= 250), OSC-strong, CV_t (bursting if >= 0.3).
  Also computed on the AL module counts (reported, not used for the label).
- **Stationarity**: |mean(500-1000) - mean(1000-1500)| <= 10 % of the W_d mean of the nonORN counts, else -NS.
  Points with rho_nonORN * 164,577 < 1 active neuron on average are not given -NS (the 10 % rule is noise there);
  this exemption is reported.
- **Deafferentation**: tau_post = censored MLE (`tau_mle`, censoring at 3000) of t = reg_last[nonORN] + dt - 1500
  (replicas active at 4500 censored); S1 = uPN_ant over 1600-1800 <= 10 % of its driven value; S2 = at most 3/64
  replicas with ALsys active at 4500. Removal class: silent (tau_post < 200 and S1 and S2), self-sustained
  (tau_post >= 3000), transient (else).
- **Phase**: SS (-L / -D by PR_nonORN), DT, DA (silent, G >= 0.29), DQ (silent, G < 0.29); modifiers -O, -D, -NS.
- **Data-allowed**: G in [0.29, 1.45] and S1 and S2. Reported alongside: broad gain window [0.077, 1.67]; S1 at 1 %;
  palp-intact test (B4 only: S1' uPN_ant 100-300 after removal <= 10 % of driven, S2' uPN_ant 1000-3000 after removal
  <= 10 % of driven, E1 definitions).
- Extra (not used for labels): where activity lives (region shares of occ, top superclasses), resources res_x.

## 3. Lines (boundaries) per (set, r)

All crossings are interpolated linearly in ln g between grid points; the uncertainty is the bracketing grid cell and a
95 % replica bootstrap (2000 resamples, replicas resampled independently at every g).
- **DQ->DA**: first upward crossing of G = 0.29 (raw curve, first grid cell where G goes from < 0.29 to >= 0.29).
- **Allowed region**: the set of grid points that pass; its top = the first failing point after the first passing run,
  interpolated on the criterion that fails first (G = 1.45, S1 residual = 0.1, or the S2 alive fraction = 3.5/64).
  If the allowed points are not contiguous, all runs are reported.
- **DA->DT** (loss of "silent"): first g where the removal class is not silent; interpolated on whichever of
  tau_post = 200 (isotonic ln tau_post), S1 residual = 0.1, S2 alive fraction = 3.5/64 crosses first.
- **DT->SS**: tau_post = 3000 crossing of the isotonic ln tau_post (as M1's tau boundaries; `first_crossing`).
- **SS-L->SS-D**: PR_nonORN/n_nonORN = 0.25 crossing (isotonic).
- Refinement (set C): at R_CORE for A1 (and A4/A2 at r = 0, 1) 8 log points inside the bracketing cells of DQ->DA,
  the allowed top, DT->SS and the chi_AL/rho peak (if resolved).

## 4. Fluctuation and susceptibility peaks, and "does a peak line track an instability?" (pre-registered)

**Peak observables** along g at fixed (set, r): F_AL = chi_AL/<rho_AL> (MODEL2's near-critical marker), F_non =
chi_nonORN/<rho_nonORN>, chi_nonORN, and tau_int.
- **Peak** = grid argmax; location refined by a parabola in ln g through the three highest consecutive points.
- **Resolved** if, in >= 95 % of 2000 replica-bootstrap resamples of the whole curve, the argmax is an interior grid
  point AND its value exceeds the curve's values at both ends of the g range (0.05 and the last g). Otherwise "no
  resolved peak" (e.g. monotone rise into the SS phase).

**Static susceptibility from the ORN levels (B1, plus A1 at 0.04)**, at R_CORE:
- chi_h(g) = d rho_nonORN / d h_ORN by central difference between ORN levels 0.02 and 0.07 (h 0.018533, 0.068798);
- field slope s(g) = least-squares slope of ln rho_nonORN against ln rho_ORN (MEASURED) over the 5 levels.
  Linear response gives s ~ 1, a mean-field critical point s = 1/delta = 0.5, a drive-insensitive self-sustained
  state s -> 0. Line: g where s crosses 0.5 (isotonic decreasing).
- chi_h peak: grid argmax + resolution as above (bootstrap over replicas of both levels).

**Instability candidates** (each is a g line per r):
- (i) g_lin = 0.1632 (linear instability of silence; r-independent; unchanged by spike-mode depression);
- (ii) the DT->SS line of this lane (the driven state becomes self-sustaining without input);
- (iii) the setting-(a) onset g_on(r) (tau = 3000) from lane m2/decay if it exists when this lane is analysed; otherwise
  only the aldef E3 reference at r = 1 (1.57 [1.53, 1.63]);
- (iv) mean-field lines (Hopf / saddle-node) from lane m2/meanfield, only if written by then (comparison, not a test).

**Criterion.** A peak line *tracks* an instability line if, over the r values where the peak is resolved (at least 4
needed, else "inconclusive"), |ln(g_peak/g_line)| <= ln 1.178 (one coarse grid step; refined values where available)
at >= 80 % of them, AND, when the instability line itself changes by more than one grid step across those r, the
Spearman rank correlation of g_peak and g_line across r is >= 0.8. It *does not track* if the location test holds at
< 50 % of them; in between = "partial". Same rule for the s = 0.5 line.

## 5. Expectations written before the runs (to be tested, not assumed)

From aldef E1/E3 (+ verification) and the smoke runs (r = 1, ORN 0.04):
- DQ->DA near g 0.25-0.32; allowed region about 0.32-1.24 (upper end interpolated across 1.15-1.5); smoke: DA at 0.5,
  DT at 1.5, SS-L at 3.0. Allowed top at or below the setting-(a) S line (1.24) and the DT->SS line near the (a)
  onset (1.57).
- The AL-module F_AL with depression was flat (1.2-2.3 over g 0.06-2.6) in E3's removal runs, so a resolved F_AL peak
  is NOT expected below the SS line; if a peak exists we expect it at the DT->SS line (driven bistability / slow
  switching), not at g_lin.
- Measured rho_ORN falls with g (0.037 / 0.030 / 0.024 at g 0.5 / 1.5 / 3.0), so G must use the measured rho_ORN.
- M1 reference: allowed 0.10-0.29 at r = 1 and 0.09-0.23 at r = 0 (E1), F_AL peak at 0.20-0.24 (r = 1).
- Nulls: the configuration null destroys ORN->PN wiring, so its G and allowed region are not meaningful; for nulls only
  the SS line, PR, chi peaks and OSC are interpreted.

## 6. Quality checks before any number is reported
- Every task log: md5 61580c76; every npz exists and has the expected g list; preempted/failed tasks rerun.
- ORN clamp: ORN activity after removal exactly 0 at every point.
- rho_ORN at g = 0.05 within 5 % of the isolated value (0.04 at level 0.04), i.e. h is right.
- The smoke points (r = 1, g 0.5 / 1.5 / 3.0) are reproduced statistically by A1's neighbouring grid points (no exact
  match expected: different g and seed).
- Bootstrap CIs reported for every boundary; boundaries that sit in the first or last grid cell are reported as bounds.
- Results that are negative or inconclusive are reported as such.

## Amendments (written after the interim look at 13:35, before the final analysis)
1. **Top of the allowed region.** Section 3 said "the first failing point after the first passing run". At large r the
   gain G stays within a few % of 0.29 over several refined g points (A1 r = 4: G 0.287-0.302 for g 0.45-0.50), so in
   some bootstrap resamples a noisy dip below 0.29 ended the "first run" right at the lower edge (CI of the top
   [0.467, 1.52]). The top is now the first failure of an UPPER criterion (G > 1.45, S1 or S2) after the first allowed
   point; the lower edge stays the G = 0.29 crossing. The number of separate allowed runs is still reported.
2. Descriptive, not pre-registered: local maxima (prominence >= 10 %) of every peak observable are listed next to the
   pre-registered global-peak test, because F_AL turned out to have more than one hump on some curves.
3. Refinement waves C_1 and C_2 cover A1 at all R_CORE and A2/A4 at r = 0, 1 (as planned). Preempted tasks (Slurm does
   not requeue them here) are rerun with the same lines and seeds by code/m2_driven_rerun.py.
4. **Field slope (written after the interim look, 13:55).** The pre-registered slope s = dln rho_nonORN / dln rho_ORN
   is ~0.40-0.44 already at g = 0.05 (r = 0, 1, 4), where the response must be linear. Cause: ORN-synapse depression
   compresses the transmitted input (rho_ORN * e_ORN, e = 0.74 / 0.59 / 0.41 / 0.29 / 0.22 at levels 0.01-0.10 at
   g 0.05, r 1; Poisson mean field gives a log-slope 1/(1 + U nu TAU) = 0.41 at rho 0.04). So "s = 0.5" is not a
   mean-field-critical marker here and is reported as not interpretable. Added: s_eff = slope against the transmitted
   input rho_ORN * e_ORN (measured mean efficacy of active ORN epochs, `res_x_act`), which is ~0.9 at g = 0.05; its
   0.5 crossing gets the same tracking test. This is post hoc and is labelled so in the report.
