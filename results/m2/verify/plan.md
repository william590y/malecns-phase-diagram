# Lane m2/verify: adversarial verification of the Model-2 restart (plan, pre-registered)

Written 2026-11-09 20:10 EST, before any recomputation, rerun or new number of this lane. Target: the synthesis text of
lane m2/synth (workflow wf_aae614ed-511) and the six lane reports behind it. Goal: try to REFUTE them.
Code: `code/m2_verify.py` (CPU recompute from raw outputs; imports no project module: numpy/scipy only),
`code/m2_verify_args.py` (GPU rerun argument lines), `code/m2_verify_sim.py` (analysis of the reruns).
Results here (`results/m2/verify/`), figures `figs/m2/verify_*.png`. Jobs `m2-verify-*`. All paths relative to
`/MALECNS_ROOT`.

## A. Recompute >= 15 key numbers from RAW outputs with my own code
Own estimators: censored-exponential MLE tau = sum(min(t_i, T))/deaths with t_i = first record time where the replica's
rho is 0 (MODEL2 5a); own PAVA (weights deaths + 0.5, as the lanes); first crossing interpolated linearly in ln g; own PR;
own driven per-point quantities from win_mean / reg_last; own power-law / branching / KS code where needed.
Numbers (lane value in brackets; source lane):
1. M2 real r 0: S (0.981), onset (1.18), D (6.29) [decay]
2. M2 real r 1: S (1.21), onset (1.57), D (13.9) [decay]
3. M2 real r 6: S (1.567), onset (2.02), D (8.03) [decay]
4. configuration-null onset r 0 and r 6, and real/null onset ratio (0.74 at r 0, 1.42 at r 6) [decay, synth]
5. AL-module share and PR at the first sustained g, r 0 and r 6 (6.1% / 13.6k; 5.0% / 19.1k) [decay, synth]
6. driven r 0: DQ->DA (0.216), allowed top (0.968), DT->SS (1.21) [driven]
7. driven r 1: DQ->DA (0.317), allowed top (1.23), DT->SS (1.57) [driven]
8. DT->SS / decay onset over all 23 r (0.97-1.03) [driven, synth]
9. palp-intact test: largest passing g and residuals at the gain edge, r 0 and r 1 (passes only g <= 0.186) [driven]
10. driven r 0: non-ORN Fano peak (55 at g 0.99), tau_int peak location (g 0.69-0.96 over r) [driven, synth]
11. FSS, configuration null: W(5k)->W(167k) at r 0 (0.271 -> 0.136, ratio 0.502) and r 1 (0.451 -> 0.075) [nature]
12. FSS, real uniform subsample: W(20k)/W(full) at r 0 (0.62) [nature]
13. hysteresis: M2 loop-area ratio A(50)/A(800) for at least 2 cells (11-90) [nature]
14. branching ratio (multistep regression) at the M2 pulse maximum (<= 0.977 per unit) and its time base [nature]
15. lifetimes: residual-lifetime exponential test after t = 200 and late lifetime at real r 1, g 1.57 (tau_late 4190,
    early-dead 13%) [decay]
16. OSC flags: re-derive the flag (trough, peak, period) at >= 3 flagged points and the per-row flag counts on the real
    M2 graph [decay]
17. isolated modules, r 0: AL alone (2.19), central brain alone (1.42) [modules]
18. g_lin = 0.1632 from the Perron root of the excitatory operator (sparse eigs on the read-only cache) [all]
19. theory/simulation ratios: MF equal-size onset r 0 (0.53), stochastic single-mode r 0 (1.41), r-trend 1.71 / 1.93 /
    2.77, MF D vs sim at every r where both exist [synth]
20. spike-mode depression check from production files: ORN active-epoch efficacy at g 0.05 against the exact renewal
    formula e = (1 - M)/(1 - (1 - U) M) at the measured ORN rate [simulator]
**Pass rule (same as the synthesis recheck):** within 2% of the lane value or inside the lane's total interval; ratios
within 2%; counts exact. A failure is reported with both values; the lane value is not silently replaced.

## B. GPU reruns with NEW seeds (>= 4 key points; code/ald_sim.py md5 61580c76 unchanged)
| id | what | g | B |
|---|---|---|---|
| V1/V2 | decay M2 real r 0 around S and onset | 0.93-1.10 / 1.12-1.29 (6 each) | 128 |
| V3/V4 | decay M2 real r 1 around S and onset | 1.13-1.36 / 1.47-1.73 | 128 |
| V5 | decay M2 real r 6 around onset | 1.85-2.19 | 128 |
| V6 | decay M2 real r 0 around D | 5.6-7.0 | 64 |
| V7 | decay M2 configuration null r 0, same graph (null_seed 0), new run seed | 1.50-1.73 | 64 |
| V8 | decay M2 configuration null r 0, NEW graph (null_seed 2) | 1.40-1.85 | 64 |
| V9/V10 | driven M2 r 0 around DQ->DA, allowed top, DT->SS | 0.20-0.99 / 1.15-1.28 | 64 |
| V11 | driven M2 r 1 around DQ->DA and allowed top | 0.30-1.28 | 64 |
| V12/V13 | palp-intact removal M2 r 0, fine grid across the gain edge (two seeds) | 0.19-0.243 (7) | 2 x 128 |
| V14/V15 | isolated CB and AL (modules lane graphs), r 0, fine grid | 1.20-1.58 / 1.85-2.46 | 64 |
| T | live unit tests of the simulator options (copy of m2_sim_test.py / m2_obs_test.py writing to results/m2/verify/simtest) | | |
**Replication rule:** a rerun line replicates if the new-seed crossing (from the rerun alone, my code) lies inside the
lane's total interval or within 3% of the lane value. Palp: the claim "no g satisfies gain >= 0.29 AND the palp-intact
test at r 0" is REFUTED if any rerun g with pooled G >= 0.29 has both palp residuals <= 0.10 with the upper end of a
95% replica-bootstrap CI <= 0.10; CONFIRMED if at every rerun g with G >= 0.29 a residual's bootstrap CI lies wholly
above 0.10; otherwise MARGINAL (reported as such, and the synthesis wording changed to "marginal").

## C. Audits (descriptive, no pass mark)
- Pre-registration: each plan's stated writing time and its file mtime vs the earliest output of the lane; amendments
  labelled; theory predictions vs experiment exposure (STATE.md summaries existed before both theory plans).
- Criteria consistency across lanes: onset definitions (decay log-grid tau vs driven exact reg_last), "first sustained g"
  vs crossings, S1 rule, the branching-ratio time base, the FSS W definition, the KS bug in m2_obs.powerlaw_fit.
- Artefacts: finite T (tau = 3000 onset from the T = 30000 long runs truncated at 3000 vs 30000), fit windows (lifetime
  tests after t = 200 vs 500), replica counts (flags at points with 4-7 survivors; bootstrap widths), grid limits
  (coarse-only lines such as the palp test and the isolated-module onsets), seed counts (FSS W with 3 seeds; jackknife).
- Synthesis text vs lane files: every claim I can trace; mis-statements listed as corrections.
