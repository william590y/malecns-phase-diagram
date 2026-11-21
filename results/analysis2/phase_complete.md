# Complete phase diagram: simulation vs theory (MaleCNS, stochastic Wilson–Cowan)

Generated 2026-11-09 by `code/analysis2_phase.py` (numbers, `results/analysis2/phase_complete.json`, the tables below) and
`code/analysis2_phase_figs.py` (figures). Every number here is read from that json, from the files it names, or from the three
verified analyses (`spreading.md`, `longruns.md`, `driven.md`, with their verification corrections applied). Figures:
`figs/pd2/phase_real.png`, `phase_nulls.png`, `phase_driven.png`, `tau_sim_vs_theory.png`. Flow-page data:
`results/analysis2/flow_sim_boundaries.json` (per r: S, on, D, their uncertainty intervals and g_half; null = not reached).
An independent critic review (last section) recomputed the boundaries from the raw files; sentences it changed are marked [critic].

## Criteria (identical for simulation and theory, T = 3000)

| boundary | simulation | theory |
|---|---|---|
| S starts | τ_sim = 200 | predicted module lifetime τ = 200 (exact birth–death recursion, dense branch rows) |
| sustained onset (S → L) | τ_sim = 3000; g_half (half the replicas alive at T) reported alongside | τ = 3000 |
| D | participation ratio of the occupancy PR = N/4, among sustained points | QMF fixed point alive and its PR = N/4 |

- τ_sim is the censored-exponential MLE: (sum of per-replica extinction times, censored at T) / deaths. An extinction time is
  the first recorded time a replica is silent (log time grid, 8.4% per step in the T = 3000 runs, 9.3% / 10.5% in the
  T = 8000 / 30000 runs pooled at r = 1; this biases τ_sim up by at most one step, i.e. by ≤ 0.6% in g, because ln τ_sim
  rises about 18-fold faster than ln g near the onset at r = 1; with interval-midpoint death times every onset moves by
  ≤ 0.4% [critic]).
- All runs with norm raw, h = 0, dt 0.2 and T ≥ 3000 were pooled per (graph, r): stage 1 (`results/pd/decay_*`), pd2
  (`onset_*`, `deloc_*`), and at r = 1 also the fine, upper, original and long sweeps truncated at T = 3000. PR uses only
  T = 3000 runs (same occupancy window). In total 23 real r values (40–173 g points each), 6 configuration-null and 6
  weight-shuffled-null r values.
- Crossings: ln τ_sim, the alive fraction and PR were made monotone in g (weighted isotonic regression), then the first
  crossing was interpolated in ln g. This moved no crossing by more than 0.17% compared with the raw first crossing.
- Uncertainty ("total" in the tables) = envelope of the two grid points around the crossing and the 95% replica bootstrap
  (2000 resamples). PR has no per-replica data, so the D uncertainty is essentially the grid bracket.
- Theory crossings use the raw dense rows (`branch_<graph>_dense_part0-7.json`, all 8 parts for all three graphs; my
  re-implementation of the recursion reproduces the stored ln τ exactly). Rows marked * (configuration null, r = 0) lie below
  the first theory row (g = 0.873); there the same recursion was run with R interpolated linearly from R = 1 at g_lin and K
  from the first row. QMF D lines merge `qmf_<graph>.npz`, `qmf_real_dense.npz` and `pd2/qmf_<graph>_r*.npz` (23 r values per
  graph); duplicated r agree to < 1% except configuration null r = 1.5, 2 and 3 (7.59 vs 7.35, 12.7 vs 12.0, and none
  by g = 30 vs 28.6; the pd2 value is used).
- Refinement jobs: pd2-sim array 168141 (54 tasks). Tasks 5 (onset_real_r0.625) and 43 (onset_wnull_r0.5) were preempted
  and rerun as 169200_5 and 169836_43; th-qmf2-wnull 168150 (r 3.5–6) was preempted and rerun as 169201. All 54 outputs
  and all 7 pd2 QMF files exist.

## Headline, in plain language

1. **The theory gets the shape right.** Every graph goes quiescent → slow window → sustained, the slow window starts well
   above the linear threshold g_lin (real: S at 1.43–2.7 × g_lin), and the configuration null goes straight from the slow
   window to whole-brain activity (D at the onset for all 6 r tested), as P4 predicted.
2. **At r = 0 the boundaries nearly agree.** Real graph: S 0.245 (theory) vs 0.234 [0.228, 0.244] (simulation), onset 0.291 vs
   0.266 [0.260, 0.278], D 1.45 vs 1.47 [1.44, 1.54]. Theory is 5% (S) and 9% (onset) high, just outside the simulation's
   interval; the QMF D value is inside it, but QMF runs on a 20% g grid and only places D in the cell 1.31–1.58 [critic].
   Configuration null: theory S 0.837 and onset 0.866 (inputs extrapolated below the first theory row) lie inside the
   simulated brackets, but the simulation puts both S and onset in one 8% grid cell (τ_sim = 145 at g = 0.838, no deaths
   at 0.905), so agreement is established only to that cell. The interpolated 0.841 / 0.867 depend on how the zero-death
   point is regularised and do not support a 0.1–0.5% match [critic]. Weight-shuffled null: S 0.719 vs 0.709, onset
   0.766 vs 0.734, D 1.18 (QMF cell 1.09–1.31) vs 1.21.
   - The remaining real-graph offset at r = 0 is about the size of the excitatory coincidence term's effect [critic: was
     "comes from"]. The same recursion with no coincidence term at all (K = 1/(⟨u²,v⟩⟨u,v²⟩), 70.5–109) gives S 0.234 and onset 0.270, both inside the simulation's intervals.
     The lifetimes needed at r = 0 correspond to K ≈ 75.8 (72.7–80.4), slightly above even that no-coincidence K (70.7 at
     the same g) and well above the theory's K_eff(r = 0) = 58.5.
3. **With inhibition, the theory fails, and the failure grows with r.** Theory/simulation ratio of the onset: 1.26
   (r = 0.125), 1.38 (0.25), 1.56 (0.5), 1.78 (1), 2.10 (2), 2.61 (4), 4.08 (6; 3.57 without the recursion's floor(K)
   jitter). For S: 1.16, 1.23, 1.33, 1.49, 1.62, 1.77, 1.77. Inhibition does move the simulated boundaries, but far less:
   from r = 0 to r = 6 the simulated onset rises 2.17-fold (0.266 → 0.579) and the theory's 8.1-fold (0.291 → 2.36).
   - In the theory's own terms the miss is in the module size. With the theory's λ(g) and recursion, the simulated lifetimes need K about twice
     the theory's K_eff for every r ≥ 0.25: K_needed/K_eff = 1.30 (r = 0), 1.65 (0.125), 1.91 (0.25), 2.09–2.22
     (r = 0.375–1.25), 1.86–2.12 (r = 1.5–6). The configuration null shows the same factor (1.87–2.06 at r = 0.5–6).
     The weight-shuffled null does not show a constant factor (2.93, 2.85, 2.15, 1.59, 1.36 at r = 0.5, 1, 2, 4, 6, from one
     to three slow-window points each), so "the same factor on all three graphs" is not supported [critic]. A plausible
     reading is that the factorised coincidence term overstates how much inhibition shrinks the island, by about a factor
     of 2 in K for r ≥ 0.25. This is a hypothesis, not a finding: at r = 0 the needed K (75.8) exceeds even the K with no
     coincidence term at all (70.7), so part of the miss lies elsewhere (λ(g), the one-mode reduction or the birth–death
     closure), and every K here assumes λ(g) and the one-mode reduction are right [critic].
   - In lifetimes, τ_sim/τ_theory (geometric mean over slow-window points with ≥ 3 deaths) is 4.1 at r = 0 (n = 4,
     2.5–8.0) and 17–54 for r ≥ 0.25 (40 at r = 1, n = 20, 9.1–289). In `tau_sim_vs_theory.png` every point lies above the
     diagonal (smallest ratio 2.5 real, 3.3 configuration null, 1.26 weight-shuffled null); 86% of the 159 real points
     are above 10× and 18% above 100×.
4. **The mean-field D line also overstates inhibition.** Real graph: QMF D vs simulated D is 1.45 vs 1.47 (r = 0), 2.02 vs
   1.91 (0.125), 3.01 vs 2.50 (0.25), 4.72 vs 3.29 (0.375), 7.67 vs 4.44 (0.5), 13.8 vs 6.05 (0.625); the QMF grid cells are 1.31–1.58, 1.90–2.28, 2.74–3.30, 3.96–4.76,
   6.88–8.28 and 12.0–14.4, so from r = 0.25 up the whole cell lies above the simulated interval, and at r = 0.125 it
   overlaps it [critic]. For r = 0.75–1.5 QMF
   never reaches N/4 up to g = 30, but the simulation delocalizes at 8.33, 11.0, 14.5, 21.8 and 28.9. For r = 1.75, 2, 3,
   4 and 6 neither does by g = 30 (for r = 2.25, 2.5, 2.75, 3.5, 4.5, 5, 5.5 the simulation only reaches g = 2.6, so D is
   untested there). Configuration null: QMF puts D at ≤ 2.28 (r = 0.5; QMF is still silent at the previous grid point
   1.90, so this is only an upper bound), 4.20 (cell 3.96–4.76, r = 1) and 12.0 (cell 12.0–14.4, r = 2), against simulated
   1.05, 1.11 and 1.16 (1.8–2.2×, 3.8×, 10×) [critic]; it is silent below g_QMF = α_eff/μ(r), e.g. μ(1) = 0.264.
5. **The weight-shuffled null is not like the configuration null.** P4 expected both nulls to go straight to D. The
   configuration null does (PR 0.40 at the first active point for r = 0, 0.42 at g = 1.09 for r = 1). The weight-shuffled
   null has an L phase by the N/4 criterion: at r = 0 the PR is 0.11 at onset (g 0.757) and reaches 0.25 at 1.21; at r = 1
   it is 0.02 at onset and reaches 0.25 only at 4.68.
6. **The slow window is narrow and one-module.** The simulated window (onset/S) is 1.14 wide at r = 0 and 1.22 at r = 1
   (theory 1.19 and 1.45). The long-run and spreading analyses below show it is one metastable module with exponential
   lifetimes, not a Griffiths phase with power-law decay.
7. **Driven plane (r = 1):** the fluctuation peak at h = 1e-5 sits at g = 0.311 (grid cell [0.285, 0.339]); P5 with the
   measured h = 0 lifetime predicts 0.305 (inside), with the theory lifetime 0.448 (1.44× too high, for the same reason as
   item 3). No peak is resolved for h ≥ 1e-3.


## Where the theory works, and where it fails (numbers)

**Works:**
- r = 0, all three graphs: S and onset within 0–9% (table above; on the configuration null only to one 8% grid cell).
  Real D (QMF) consistent with the simulated interval (1.45, QMF cell 1.31–1.58, vs 1.47 [1.44, 1.54]); weight-shuffled
  null D consistent (1.18, cell 1.09–1.31, vs 1.21 [1.14, 1.24]) [critic].
- g_lin = 0.1632 lies below every simulated boundary at every r (P1 is a lower bound, as stated).
- Ordering and qualitative r-dependence: S and onset rise monotonically with r in both simulation and theory.
- Configuration null delocalized at onset at all 6 r (P4).
- Driven h = 1e-5: P5 with the measured lifetime falls in the peak's grid cell (0.305 vs 0.311 [0.285, 0.339]).

**Fails:**
- Every r ≥ 0.125 on the real graph: theory S and onset lie outside the simulated intervals, too high by 16–79% (S) and
  26–308% (onset). Configuration null r ≥ 0.5: 7–18% (S; inside the interval only at r = 0.5), 11–25% (onset). Weight-shuffled null r ≥ 0.5: 9–15% (S),
  10–23% (onset). The relative misses in g are smaller on the nulls because their K is larger (theory K_eff 51–138 at r ≥ 0.5,
  against 8.6–30 on the real graph at r ≥ 0.25), so ln τ rises more steeply with g and the same factor ≈ 2 in K moves the
  crossing less.
- QMF D line for r ≥ 0.25 on the real graph (21–128% high where it exists, with the whole QMF grid cell above the simulated
  interval; at r = 0.125 the cell 1.90–2.28 overlaps the simulated [1.91, 2.00]; absent for r = 0.75–1.5 where the
  simulation delocalizes), and for r ≥ 0.5 on both nulls (1.34–10× high; the configuration-null r = 0.5 value is an upper
  bound) [critic].
- Weight-shuffled null: P4 (no L phase) fails, see headline item 5.
- At r ≥ 2.75 the theory's onset curve is jagged because the recursion truncates the chain at floor(K) (K ≈ 6–11 there). A variant
  that interpolates the truncation level gives 1.00 instead of 1.04 (r = 2.75), 1.17 instead of 1.25 (3.5), 1.44 instead of
  1.60 (4.5) and 2.07 instead of 2.36 (6). This does not change any conclusion.

## Agreement tables

Columns: theory boundary, simulated boundary with its total uncertainty, ratio theory/simulation; g_half = half-survival
point (1.00–1.06 × the τ_sim = T onset everywhere); τ_sim/τ_th = geometric mean over slow-window points with ≥ 3 deaths and
200 < τ_sim; K needed = the K that reproduces τ_sim with the theory's own λ(g) and recursion, median (range) over the same
points; last column = theory K_eff at those points (in brackets the K with no coincidence term). "> x" = not reached up to the
largest simulated g. * = theory crossing below the first dense row (inputs interpolated from g_lin). D theory: QMF value
interpolated in ln g, with its grid cell in brackets (QMF grid = 32 log points in 0.1–30, 20% steps); "≤" = QMF jumps
from silent straight to PR ≥ N/4 between two grid points, so only an upper bound [critic].

### MaleCNS (g_lin = 0.1632)

| r | S theory | S sim [total] | ratio | on theory | on sim [total] | ratio | g_half sim | D theory (QMF; grid cell) | D sim [total] | ratio | τ_sim/τ_th geomean (n, range) | K needed median (range) | K_eff theory (D = 0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.245 | 0.234 [0.228, 0.244] | 1.05 | 0.291 | 0.266 [0.26, 0.278] | 1.09 | 0.271 | 1.45 (1.31–1.58) | 1.47 [1.44, 1.54] | 0.984 | 4.13 (4, 2.52–8.02) | 75.8 (72.7–80.4) | 58.5 (70.7) |
| 0.125 | 0.281 | 0.241 [0.228, 0.246] | 1.16 | 0.348 | 0.276 [0.26, 0.279] | 1.26 | 0.28 | 2.02 (1.90–2.28) | 1.91 [1.91, 2] | 1.05 | 8.02 (3, 4.01–20.5) | 63 (61.7–66.8) | 38.3 (70.7) |
| 0.25 | 0.308 | 0.251 [0.244, 0.256] | 1.23 | 0.395 | 0.287 [0.278, 0.297] | 1.38 | 0.291 | 3.01 (2.74–3.30) | 2.5 [2.44, 2.6] | 1.21 | 17.2 (5, 4.75–60.3) | 57.7 (54.4–59.8) | 30.1 (70.7) |
| 0.375 | 0.332 | 0.258 [0.244, 0.262] | 1.29 | 0.437 | 0.295 [0.278, 0.3] | 1.48 | 0.301 | 4.72 (3.96–4.76) | 3.29 [3.11, 3.65] | 1.43 | 26.3 (4, 5.91–130) | 53.7 (49.2–55.8) | 25.6 (70.8) |
| 0.5 | 0.353 | 0.265 [0.26, 0.278] | 1.33 | 0.476 | 0.306 [0.3, 0.317] | 1.56 | 0.311 | 7.67 (6.88–8.28) | 4.44 [4.3, 4.76] | 1.73 | 27 (4, 8.97–69.7) | 48.6 (45.1–49.6) | 22.7 (70.8) |
| 0.625 | 0.372 | 0.268 [0.26, 0.278] | 1.39 | 0.512 | 0.317 [0.297, 0.321] | 1.62 | 0.324 | 13.8 (12.0–14.4) | 6.05 [5.94, 6.98] | 2.28 | 33.8 (4, 10.2–146) | 44.8 (44.4–46.9) | 20.6 (70.8) |
| 0.75 | 0.39 | 0.274 [0.26, 0.282] | 1.42 | 0.547 | 0.329 [0.317, 0.339] | 1.66 | 0.336 | none | 8.33 [8.21, 9.65] | – | 29.4 (5, 7.97–123) | 40.5 (39.9–41.7) | 19 (70.9) |
| 0.875 | 0.406 | 0.278 [0.26, 0.287] | 1.46 | 0.581 | 0.341 [0.334, 0.361] | 1.7 | 0.348 | none | 11 [9.65, 11.3] | – | 25.4 (5, 7.84–133) | 38.7 (37.1–39.9) | 17.9 (70.9) |
| 1 | 0.421 | 0.283 [0.28, 0.289] | 1.49 | 0.613 | 0.344 [0.34, 0.35] | 1.78 | 0.35 | none | 14.5 [14.4, 15.7] | – | 40.3 (20, 9.14–289) | 36.6 (34.9–40.3) | 16.8 (70.9) |
| 1.25 | 0.449 | 0.295 [0.278, 0.3] | 1.52 | 0.675 | 0.358 [0.339, 0.364] | 1.88 | 0.366 | none | 21.8 [21.7, 25.5] | – | 53.5 (6, 8.65–375) | 34.2 (33.3–38.5) | 15.4 (71) |
| 1.5 | 0.474 | 0.308 [0.302, 0.317] | 1.54 | 0.734 | 0.386 [0.362, 0.39] | 1.9 | 0.394 | none | 28.9 [25.5, 30] | – | 45.7 (7, 10.6–252) | 30.4 (29.7–33.3) | 14.3 (71) |
| 1.75 | 0.496 | 0.314 [0.297, 0.323] | 1.58 | 0.787 | 0.392 [0.384, 0.412] | 2.01 | 0.402 | none | > 30 | – | 33 (6, 8.86–117) | 28.4 (27.9–29.5) | 13.5 (71.1) |
| 2 | 0.517 | 0.32 [0.313, 0.339] | 1.62 | 0.842 | 0.402 [0.386, 0.415] | 2.1 | 0.42 | none | > 30 | – | 33.9 (7, 9.98–94.5) | 26.7 (25.9–27.6) | 12.9 (71.1) |
| 2.25 | 0.538 | 0.33 [0.317, 0.341] | 1.63 | 0.908 | 0.418 [0.407, 0.44] | 2.17 | 0.434 | none | > 2.6 | – | 34.8 (6, 9.52–176) | 25.5 (24.8–26.3) | 12.3 (71.1) |
| 2.5 | 0.556 | 0.341 [0.331, 0.361] | 1.63 | 0.953 | 0.444 [0.427, 0.47] | 2.15 | 0.455 | none | > 2.6 | – | 44.7 (6, 9.64–258) | 24.3 (22.9–25.3) | 11.8 (71.2) |
| 2.75 | 0.573 | 0.344 [0.337, 0.361] | 1.67 | 1.04 | 0.443 [0.432, 0.47] | 2.34 | 0.458 | none | > 2.6 | – | 54.3 (7, 14–283) | 23.6 (22.5–25.2) | 11.4 (71.3) |
| 3 | 0.593 | 0.346 [0.336, 0.362] | 1.72 | 1.08 | 0.456 [0.44, 0.47] | 2.36 | 0.466 | none | > 30 | – | 35 (10, 10.2–147) | 22.2 (21.3–23.1) | 11.1 (71.3) |
| 3.5 | 0.623 | 0.369 [0.361, 0.386] | 1.69 | 1.25 | 0.473 [0.462, 0.502] | 2.63 | 0.488 | none | > 2.6 | – | 35.1 (6, 13.8–107) | 21.1 (20.2–21.6) | 10.5 (71.4) |
| 4 | 0.658 | 0.373 [0.362, 0.386] | 1.77 | 1.33 | 0.508 [0.495, 0.525] | 2.61 | 0.524 | none | > 30 | – | 30 (9, 11.3–93.9) | 19.5 (18.6–20.4) | 10.1 (71.4) |
| 4.5 | 0.685 | 0.399 [0.386, 0.413] | 1.72 | 1.6 | 0.519 [0.502, 0.536] | 3.09 | 0.533 | none | > 2.6 | – | 35.8 (7, 10–143) | 18.4 (18–19.2) | 9.63 (71.5) |
| 5 | 0.727 | 0.406 [0.386, 0.419] | 1.79 | 1.69 | 0.547 [0.532, 0.573] | 3.08 | 0.563 | none | > 2.6 | – | 33.9 (8, 8.37–123) | 17.5 (17.3–18.8) | 9.29 (71.6) |
| 5.5 | 0.75 | 0.426 [0.412, 0.44] | 1.76 | 1.78 | 0.583 [0.562, 0.612] | 3.06 | 0.603 | none | > 2.6 | – | 28.8 (7, 10.3–61.5) | 16.7 (15.6–17.5) | 8.99 (71.7) |
| 6 | 0.772 | 0.436 [0.422, 0.44] | 1.77 | 2.36 | 0.579 [0.561, 0.615] | 4.08 | 0.615 | none | > 30 | – | 41.3 (13, 12.2–125) | 16.2 (14.3–17) | 8.64 (72) |


### configuration null (wiring shuffled) (g_lin = 0.8287)

| r | S theory | S sim [total] | ratio | on theory | on sim [total] | ratio | g_half sim | D theory (QMF; grid cell) | D sim [total] | ratio | τ_sim/τ_th geomean (n, range) | K needed median (range) | K_eff theory (D = 0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.837 * | 0.841 [0.838, 0.905] | 0.995 | 0.866 * | 0.867 [0.838, 0.905] | 0.999 | 0.871 | ≤ 0.91 (silent at 0.757) | 0.867 [0.838, 0.905] (at onset) | ≤ 1.05 | – | – | – |
| 0.5 | 1.05 | 0.984 [0.979, 1.06] | 1.07 | 1.17 | 1.05 [0.979, 1.06] | 1.11 | 1.06 | ≤ 2.28 (silent at 1.90) | 1.05 [0.979, 1.06] (at onset) | ≤ 2.18 | 18.3 (1, 18.3–18.3) | 264 (264–264) | 138 (5.66e+03) |
| 1 | 1.12 | 1.01 [1, 1.03] | 1.1 | 1.27 | 1.11 [1.1, 1.13] | 1.14 | 1.12 | 4.2 (3.96–4.76) | 1.11 [1.1, 1.13] (at onset) | 3.8 | 8.8 (12, 3.3–36) | 182 (164–186) | 95.7 (5.66e+03) |
| 2 | 1.19 | 1.06 [0.979, 1.07] | 1.13 | 1.37 | 1.16 [1.14, 1.24] | 1.19 | 1.18 | 12 (12.0–14.4) | 1.16 [1.14, 1.24] (at onset) | 10.3 | 6.56 (2, 3.78–11.4) | 134 (133–135) | 71.8 (5.67e+03) |
| 4 | 1.26 | 1.09 [1.06, 1.1] | 1.15 | 1.48 | 1.2 [1.14, 1.24] | 1.23 | 1.21 | none | 1.2 [1.14, 1.24] (at onset) | – | 12.6 (3, 4.19–59.1) | 113 (109–125) | 57 (5.67e+03) |
| 6 | 1.3 | 1.1 [1.06, 1.14] | 1.18 | 1.54 | 1.23 [1.14, 1.24] | 1.25 | 1.25 | none | 1.23 [1.14, 1.24] (at onset) | – | 13.8 (2, 7.03–27.2) | 105 (104–107) | 51.1 (5.68e+03) |

Theory-only rows (r without a simulation): r=0.125: S 0.942, on 1.02, D 1.09; r=0.25: S 0.99, on 1.08, D 1.31; r=0.375: S 1.02, on 1.13, D 1.58; r=0.625: S 1.07, on 1.2, D 2.74; r=0.75: S 1.09, on 1.22, D 3.3; r=0.875: S 1.1, on 1.25, D 3.55; r=1.25: S 1.14, on 1.3, D 5.7; r=1.5: S 1.16, on 1.33, D 7.35; r=1.75: S 1.17, on 1.35, D 9.73; r=2.25: S 1.2, on 1.39, D 15.5; r=2.5: S 1.21, on 1.41, D 19.3; r=2.75: S 1.22, on 1.42, D 24.5; r=3: S 1.23, on 1.43, D 28.6; r=3.5: S 1.25, on 1.46, D –; r=4.5: S 1.27, on 1.5, D –; r=5: S 1.28, on 1.51, D –; r=5.5: S 1.29, on 1.53, D –

### weight-shuffled null (g_lin = 0.663)

| r | S theory | S sim [total] | ratio | on theory | on sim [total] | ratio | g_half sim | D theory (QMF; grid cell) | D sim [total] | ratio | τ_sim/τ_th geomean (n, range) | K needed median (range) | K_eff theory (D = 0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.719 | 0.709 [0.664, 0.717] | 1.01 | 0.766 | 0.734 [0.717, 0.757] | 1.04 | 0.737 | 1.18 (1.09–1.31) | 1.21 [1.14, 1.24] | 0.976 | 1.26 (1, 1.26–1.26) | 762 (762–762) | 615 (690) |
| 0.5 | 0.83 | 0.728 [0.717, 0.775] | 1.14 | 0.921 | 0.788 [0.775, 0.838] | 1.17 | 0.805 | 3.34 (3.30–3.96) | 2.49 [2.48, 2.69] | 1.34 | 14.2 (1, 14.2–14.2) | 433 (433–433) | 148 (691) |
| 1 | 0.873 | 0.762 [0.757, 0.775] | 1.14 | 0.984 | 0.799 [0.775, 0.838] | 1.23 | 0.807 | 15.9 (14.4–17.3) | 4.68 [4.63, 4.76] | 3.4 | 6.61 (1, 6.61–6.61) | 309 (309–309) | 108 (691) |
| 2 | 0.917 | 0.798 [0.775, 0.838] | 1.15 | 1.05 | 0.861 [0.838, 0.905] | 1.22 | 0.872 | none | > 5 | – | 7.08 (1, 7.08–7.08) | 181 (181–181) | 84.1 (691) |
| 4 | 0.961 | 0.871 [0.838, 0.905] | 1.1 | 1.12 | 0.957 [0.91, 0.979] | 1.16 | 0.964 | none | 21.8 [20.8, 25] | – | 7.51 (3, 3.69–26.5) | 108 (105–124) | 68.2 (692) |
| 6 | 0.985 | 0.901 [0.838, 0.919] | 1.09 | 1.15 | 1.05 [0.979, 1.06] | 1.1 | 1.07 | none | > 5 | – | 3.61 (3, 2.69–6.3) | 83.8 (79.8–91) | 61.7 (692) |

Theory-only rows (r without a simulation): r=0.125: S 0.768, on 0.832, D 1.48; r=0.25: S 0.794, on 0.871, D 1.9; r=0.375: S 0.815, on 0.899, D 2.49; r=0.625: S 0.842, on 0.94, D 4.74; r=0.75: S 0.854, on 0.956, D 6.71; r=0.875: S 0.864, on 0.97, D 9.95; r=1.25: S 0.886, on 1, D –; r=1.5: S 0.898, on 1.02, D –; r=1.75: S 0.908, on 1.03, D –; r=2.25: S 0.924, on 1.06, D –; r=2.5: S 0.931, on 1.07, D –; r=2.75: S 0.937, on 1.08, D –; r=3: S 0.943, on 1.09, D –; r=3.5: S 0.953, on 1.1, D –; r=4.5: S 0.968, on 1.13, D –; r=5: S 0.974, on 1.13, D –; r=5.5: S 0.979, on 1.14, D –

## Driven plane, r = 1 (`phase_driven.png`)

Refined runs (`results/pd2/drive_h*_real.npz`, B 32, 40 g in 0.1–3, T 1000; the stage-1 runs analysed in `driven.md` had
B 16, 32 g). Peak = argmax of χ_rep/ρ in 0.1 ≤ g ≤ 1, resolved if interior and the bootstrap puts it above the window's last
point in ≥ 95% of resamples.

| h | sim peak g [grid cell] | χ_rep/ρ at peak | resolved | P5, measured h = 0 τ (n = 526) | P5, theory τ r = 1 (n = 526) |
|---|---|---|---|---|---|
| 1e-5 | 0.311 [0.285, 0.339] | 27.4 | yes (global maximum; largest value above the window 4.76) | 0.305 (inside) | 0.448 (1.44×) |
| 1e-4 | 0.285 [0.261, 0.311] | 4.36 | yes in the window, but only a local maximum (4.55 at g = 2.52) | 0.246 (below the cell) | 0.316 (just above the cell) |
| 1e-3 | – (maximum at the window's last point, 0.966) | 1.12 | no (P = 0.00) | 0.179 | 0.200 |
| 1e-2 | – (argmax 0.743; F at the window's last point is as high in 45% of resamples) | 0.67 | no (P = 0.55) | 0.165 | 0.167 |
| 1e-1 | – (maximum at the window's last point, 0.966) | 0.27 | no (P = 0.00) | 0.163 | 0.164 |
| null, 1e-3 | – (argmax 2.05 in 0.5–5) | 3.22 | no (P = 0.82) | 0.938 | 0.996 |

P5 values are from `driven.json` (theory P5 table). The refined runs reproduce the stage-1 peaks (0.304, cell [0.275, 0.331],
at h = 1e-5; 0.287 at h = 1e-4). In the figure the theory curve h_peak(g) = 1/(n p_est(g) τ(g)) is drawn continuously from the
dense theory rows (solid) and from the measured h = 0 lifetimes (dashed). The measured-lifetime curve passes through the
h = 1e-5 peak; the theory-lifetime curve runs to the right of it, by the same lifetime underestimate as in the (g, r) plane.
For h ≥ 1e-3 the fluctuations rise smoothly with g and no module peak appears near the predicted g ≈ g_lin.

## Folded-in verified analyses

### Single-seed spreading, r = 1 (`spreading.md`, corrected)
- Seed replay: 2,654 replicas seeded on non-excitatory neurons, 0 of them ever switched anyone on.
- Establishment matches the single-type branching prediction only at g ≥ 0.42: P(100) for excitatory module seeds 0.606,
  0.689, 0.709 against p_est 0.609, 0.671, 0.725 at g = 0.42, 0.50, 0.60; at g = 0.36 it is 0.486 [0.441, 0.531] against
  0.545.
- Survival to T is badly underpredicted: real module seeds P(T) = 0.27, 0.545, 0.637, 0.660 at g = 0.36, 0.42, 0.50, 0.60
  against p_est·e^(−T/τ) = 3.9e-18, 1.3e-7, 0.0072, 0.225. The cause is the lifetime: late τ_sim 15, 33, 130, 730, 6,270 at
  g = 0.20–0.36 against the r = 1 theory's 10.4, 15.2, 24.2, 41.7, 76 (K needed 27–39 against K_eff(r = 1) ≈ 17).
- Configuration null (corrected theory source): τ_sim/τ_theory 1.24, 4.6, 17.8, 80.6 at g = 0.95, 1.05, 1.11, 1.15, K needed
  119–200 against 96. Establishment there is about twice p_est (0.498 vs 0.251 at g = 1.11) and depends on the seed's
  out-strength (0.27 / 0.47 / 0.75 by tercile).
- The pre-registered Griffiths test fails for all four data sets (0/8 real module, 0/5 null module, 0/5 each random); the
  late decay is exponential (at g = 0.32: R² 0.995, τ 731, vs power law 0.781) on g = 0.28–0.36; at g = 0.24 the two fits tie.
- Random seeds: on the real graph 1 and 3 survivors at g = 0.45 and 0.60 (1.77 and 2.02 expected from direct hits on the
  1,000-neuron set); only 2 of the 4 were in the set, the other 2 were ORNs projecting into it. On the configuration null 25
  and 22 survived against 1.27 and 1.66 expected (catchment 19,600 and 13,200 neurons; 44 of 46 started outside the set).
- Caveat: `mspread_real_a` and `_b` used the same seed, so the g pairs (0.20, 0.36), (0.24, 0.42), (0.28, 0.50) and
  (0.32, 0.60) share seed neurons and noise.

### Long runs and lifetime shape, r = 1 (`longruns.md`, corrected)
- τ_sim (T = 30,000, 32 replicas) 71.5, 147, 329, 925, 1,647 at g = 0.26–0.34; the fine sweep continues to 3,447 (0.345),
  8,510 (0.36), 2.2e4 (0.375) and 5.7e4 (0.39).
- Lifetimes in the slow window are exponential: 0 KS rejections at 9 distinct real g; on the null one outlier (g = 1.10,
  CV 1.53, one replica lived 8.7 τ). ρ(t) decays exponentially for t ≥ 20 (10/10 real, 8/8 null); the earlier
  "continuously varying power-law exponent" came from fitting the t ≥ 5 window, which includes the initial transient.
- τ_sim/τ_theory grows with g roughly as g^12: geometric mean 29 (8.7–226) on the real graph, 11 (3.3–126) on the null.
  A nearly constant K of 36.9 (34–40) reproduces every real lifetime (theory K_eff(r = 1) 16.9); the null needs 181
  (167–204) against 95.7. If the whole miss is put into the coincidence term, its inhibitory part is overstated 4.3–5.2×.
- The apparent onset drifts with T less than the theory says: g_half = 0.351 (T 3000), 0.366 (8000), 0.386–0.399 (30,000,
  model-dependent), a ×1.10–1.14 drift against the theory's ×1.40. Null ×1.035–1.06 vs ×1.08.
- Cross-check with this analysis (pooled data, same r = 1): g_half(T = 3000) = 0.350 here vs 0.3508 there; median K needed
  36.6 here (slow-window points of all r = 1 sweeps truncated at T = 3000) vs 36.9 there.

### Driven plane (`driven.md`, corrected; stage-1 data)
- The h = 1e-5 peak is a single grid point (honest bracket = grid cell [0.275, 0.331]); the refined runs above confirm it
  at 0.311 [0.285, 0.339].
- Telegraph model with the measured lifetime and n = 526 gives χ_rep at the peak of 0.70× the measured value (range
  0.49–1.55 from the simulation's CI), i.e. consistent with 1. At g = 0.30, OFF snapshots average 2.7 active neurons and ON
  snapshots 71. About half (53%) of the excess occupancy at the peak lies in the top-1,000 module neurons.
- The on-fraction midpoint moves down with h (0.316 → 0.274 from h = 1e-5 to 1e-4, Δln g −0.125 to −0.143 depending on
  threshold) but less than predicted (−0.216 with the measured lifetime).
- Mean field matches the simulated density below g_lin (medians 1.00–1.09; single points up to 21% off) and underpredicts
  it by about 2.2–2.35× for g = 0.5–3 at h ≤ 1e-3. The exact first-order response including inhibition fits at h ≥ 1e-3
  (χ² 7.2 and 0.4 over 3 points) where the excitation-only operator fails (384 and 3,762); at h = 1e-4 even the exact
  response gives χ² 9.0 (p 0.029).
- Configuration null at h = 1e-3: no resolved replica-fluctuation peak (stage 1: maximum at 2.72; refined runs here: 2.05,
  P = 0.82), against P5's 0.94–1.00.

## Caveats
- The simulated S boundary uses the mean lifetime from a fully active start; at the low end this includes the time the whole
  brain takes to fall silent (τ_sim ≈ 13–18 below g_lin at r = 1), so S is a lifetime threshold, not a pure module property.
- Earlier comparisons in STATE.md ("r = 0 theory vs simulation all match", 23:40) used phase.py's simulation criteria (time of
  the last replica's death ≥ 200; all replicas alive at T). With the matched τ-based criteria the simulated boundaries are
  lower and the r = 0 theory is 5–9% high.
- D has no replica bootstrap (per-neuron occupancy is stored only as a replica mean). For real r = 2.25, 2.5, 2.75, 3.5, 4.5,
  5 and 5.5 the simulation stops at g = 2.6, so their D is only "> 2.6"; weight-shuffled null r = 2 and 6 stop at g = 5.
- Null boundaries at r = 0 rest on few grid points near the crossing (config null bracket 0.838–0.905), so their total
  intervals are mostly grid brackets.
- K_needed uses the theory's own λ(g) and birth–death recursion; attributing the whole miss to K (rather than λ or the
  one-mode reduction) is an assumption.
- The theory's L band on the nulls (`phase_nulls.png`) combines two theory pieces that disagree there: the branching theory
  says sustained, while QMF is still silent or below N/4.

## Files
- Code: `code/analysis2_phase.py` (analysis, ~1 min on CPU, 2000 bootstrap resamples), `code/analysis2_phase_figs.py` (figures).
- Results: `results/analysis2/phase_complete.json` (all boundaries per graph and r, per-point τ_sim with CI, alive fraction,
  PR, ln τ_theory, K_needed; theory variants; driven-plane maps and P5 curves), `phase_complete_table.md` (generated tables),
  `flow_sim_boundaries.json` (flow page), `phase_complete_run.log`.
- Figures: `figs/pd2/phase_real.png`, `phase_nulls.png`, `phase_driven.png`, `tau_sim_vs_theory.png`.

## Critic review (2026-11-09)

Code: `code/analysis2_critic.py` (independent recomputation, ~2 min on CPU), `code/analysis2_critic_fig.py`. Results:
`results/analysis2/critic.json`, `critic_run.log`. New figure: `figs/pd2/critic_boundaries.png`.

**Method.** I recomputed the boundaries from the raw npz/json files with methods that deliberately differ from
`analysis2_phase.py`:
- my own loader, with activity read from ρ > 0 and death time = geometric midpoint between the last active and the first
  silent record;
- crossings from a local censored-exponential likelihood fit of ln τ = a + b ln g (Newton; replica bootstrap, 400
  resamples), instead of isotonic regression plus interpolation;
- a model-free 95% data bracket for every crossing: the largest g whose χ² CI of τ lies wholly below the threshold, and
  the smallest g whose CI lies wholly above it;
- the extinction time as the closed-form double sum instead of the downward recursion;
- theory crossings by input interpolation (λ and K interpolated in ln g, ln τ evaluated on a fine grid) as well as
  output interpolation;
- QMF D crossings together with their grid cells;
- the driven peaks recomputed from the raw drive files;
- every quoted ratio re-derived from `phase_complete.json`.

In all, I recomputed S, onset and D for all 35 simulated (graph, r) sets, theory S and onset for 23 r on each of the three
graphs, QMF D for every r and file, and the 6 driven peaks.

### What the recomputation confirms
1. **Theory numerics.** The stored ln τ in all 3 × 1,104 (row, r) entries equal the closed-form sum to ≤ 1.1e-11.
   Input-interpolated and output-interpolated crossings differ by ≤ 0.5%. The narrow (`branch_real.json`), wide and dense
   branching runs agree in λ and K to < 0.07% at common g and give the same crossings (e.g. real r = 1: S 0.422 / 0.421,
   onset 0.613).
2. **Real-graph simulated boundaries.** My local-fit S and onset are within −1.9…+2.1% (S) and −2.2…+1.1% (onset) of
   the values above, for all 23 r. D is identical, because both use the same occupancy data.
3. **Data brackets.** Every simulated S and onset in the tables (35 sets × 2) lies inside its model-free 95% data bracket.
4. **Theory against the data brackets.** On the real graph the theory S and onset lie above the bracket at every r,
   including r = 0 (S 0.245 against a bracket upper end of 0.244; onset 0.291 against 0.278). On the configuration null
   the theory is inside the bracket at r = 0 (S and onset) and at r = 0.5 (S only), and above it everywhere else. On the
   weight-shuffled null it is inside at r = 0 (S only) and above elsewhere. "Theory too high, more so with r" therefore
   does not depend on the interpolation.
5. **Death-time convention.** Using interval-midpoint instead of first-silent-record death times moves every onset by
   ≤ 0.4%.
6. **Common random numbers.** Every run used `--seed 0`, so runs at different r with the same g list and B share one
   random stream. The per-replica extinction times are nevertheless not materially correlated across neighbouring r: the
   Spearman correlation has median −0.03 and range −0.45 to 0.59 over 29 g values (r pairs 0/0.125, 1/1.25, 4/4.5). The
   smoothness of the simulated curves in r is therefore not a shared-noise artefact.
7. **Driven peaks.** All reproduce exactly: 0.311 (h = 1e-5); 0.285 (1e-4); the window edge at h = 1e-3 and 1e-1; 0.743
   at 1e-2; 2.05 on the null.
   - The driven QMF density (`results/pd/qmf_drive_real.npz`, now panel d of `phase_driven.png`) matches the simulation
     below g_lin (median sim/QMF 1.00–1.11).
   - For g = 0.5–3 it underpredicts the simulation. The median sim/QMF ratio there is 2.38, 2.37, 2.24, 1.46 and 1.04 at
     h = 1e-5, 1e-4, 1e-3, 1e-2 and 1e-1.
8. **Quoted numbers.** All of these match the text up to rounding:
   - onset and S ratios;
   - the onset rise, ×2.17 in simulation against ×8.13 in theory;
   - K_needed/K_eff for every r;
   - the τ-ratio statistics (159 points, minimum 2.52, 85.5% above 10×, 17.6% above 100×);
   - g_half/onset (1.02–1.06 on the real graph, 1.00–1.02 on the nulls);
   - the PR values at onset;
   - the P5 predictions;
   - the duplicate-QMF disagreements (7.59 vs 7.35, 12.7 vs 12.0, none vs 28.6, each pair in the same grid cell where both
     exist).

### Problems found, and what I changed
- **QMF D uncertainty was not reported.**
  - QMF runs on 32 log-spaced g in 0.1–30, i.e. 20% steps. Every theory D is therefore known only to a grid cell, about as
    wide as the simulated D intervals.
  - For the configuration null at r ≤ 0.75, QMF jumps from silent straight to PR ≥ N/4 between two grid points, so the
    quoted D (0.91, 2.28, …) is only an upper bound.
  - *Changed:* grid cells added to the D columns of the agreement tables, to headline items 2 and 4, and to the Works and
    Fails lists. The config-null r = 0 and 0.5 values are marked "≤". The QMF r = 0.125 cell overlaps the simulated D,
    so the "fails" range for the QMF D line now starts at r = 0.25. The cells are drawn as translucent bars on
    `phase_real.png` and `phase_nulls.png`.
- **Configuration-null r = 0 "agrees to 0.1–0.5%" was not supported.**
  - The simulated S and onset both lie in one grid cell: τ_sim = 145 at g = 0.838 (32/32 died), and no deaths at 0.905.
  - The interpolated 0.841 / 0.867 are set by regularising the zero-death point as E/0.5. A different regularisation moves
    the onset within that 8% cell.
  - *Changed:* headline item 2 and the Works list now say the theory agrees only to the cell.
- **"Same factor ≈ 2 in K on all three graphs" was too strong.**
  - The weight-shuffled null gives K_needed/K_eff = 2.93, 2.85, 2.15, 1.59 and 1.36 at r = 0.5, 1, 2, 4 and 6, from one
    to three points each. That is not a constant.
  - The attribution to the factorised coincidence term is a hypothesis. At r = 0 the needed K (75.8) exceeds even the K
    with no coincidence term (70.7).
  - *Changed:* headline item 3.
- **Time-grid statement.** The record grid is 8.4% per step at T = 3000, but 9.3% and 10.5% in the T = 8000 and 30000 runs
  pooled at r = 1, not "≤ 10%". *Changed:* the Criteria bullet now says this; the effect is negligible (item 5 above).

### The operational phase.py definitions vs the τ criteria (not in the original comparison)
The project's fixed operational phases are:
- S: the last replica dies after t = 200, or only some replicas survive;
- L/D: every replica is alive at T.

Both depend on the replica count B. For exponential lifetimes they correspond to:
- B = 32: S at τ ≈ 52, L at τ ≈ 1.38e5;
- B = 16 (stage 1): S at τ ≈ 63, L at τ ≈ 6.9e4.

The τ-based criteria used above (τ = 200 and τ = 3000) are the matched choice for comparing with the theory. However, the
location of the simulated L boundary depends strongly on which definition is used. On the B = 32 pd2 runs, L_op lies above
the τ = T onset by:
- ×1.11–1.19 at r = 0;
- ×1.37–1.46 at r = 1;
- ×1.57–1.68 at r = 6.

The table compares each operational cell with the theory evaluated at the matching τ, from the dense rows (`critic.json`
`operational`, `theory_operational`). In the cell columns, (a, b] means b is the first grid point that satisfies the
criterion and a the point before it.

| graph | r | S: τ_sim = 200 | S_op cell (B 32) | theory τ = 200 | theory τ = 52 | onset: τ_sim = 3000 | L_op cell (B 32) | theory τ = 3000 | theory τ = 1.38e5 |
|---|---|---|---|---|---|---|---|---|---|
| MaleCNS | 0 | 0.234 | (0.2, 0.214] | 0.245 | 0.214 | 0.266 | (0.297, 0.317] | 0.291 | 0.351 |
| MaleCNS | 0.5 | 0.265 | (0.228, 0.244] | 0.353 | 0.29 | 0.306 | (0.361, 0.386] | 0.476 | 0.713 |
| MaleCNS | 1 | 0.283 | (0.244, 0.26] | 0.421 | 0.335 | 0.344 | (0.47, 0.502] | 0.613 | 1.18 |
| MaleCNS | 2 | 0.32 | (0.278, 0.297] | 0.517 | 0.395 | 0.402 | (0.536, 0.573] | 0.842 | > 4.89 |
| MaleCNS | 4 | 0.373 | (0.317, 0.339] | 0.658 | 0.475 | 0.508 | (0.698, 0.745] | 1.33 | > 4.89 |
| MaleCNS | 6 | 0.436 | (0.317, 0.339] | 0.772 | 0.54 | 0.579 | (0.908, 0.97] | 2.36 | > 4.89 |
| config null | 0 | 0.841 | (0.775, 0.838] | 0.837 | < first row | 0.867 | (0.838, 0.905] | 0.866 | 0.879 |
| config null | 1 | 1.01 | (0.905, 0.979] | 1.12 | 0.998 | 1.11 | (1.14, 1.24] | 1.27 | 1.43 |
| config null | 4 | 1.09 | (0.979, 1.06] | 1.26 | 1.1 | 1.2 | (1.33, 1.44] | 1.48 | 1.74 |
| weight-shuffled null | 0 | 0.709 | (0.664, 0.717] | 0.719 | < first row | 0.734 | (0.717, 0.775] | 0.766 | 0.804 |
| weight-shuffled null | 1 | 0.762 | (0.717, 0.775] | 0.873 | 0.782 | 0.799 | (0.838, 0.905] | 0.984 | 1.11 |
| weight-shuffled null | 4 | 0.871 | (0.775, 0.838] | 0.961 | 0.849 | 0.957 | (0.979, 1.06] | 1.12 | 1.3 |

Under the operational definitions the theory is again too high on the real graph: for L_op at every r, and for S_op at
every r ≥ 0.125. It is more too high for L: at r = 1 it gives 1.18 against (0.47, 0.502]. The reason is that the theory's
ln τ rises too slowly with g, so reaching τ = 1.4e5 takes it much further out than reaching τ = 3000. Over all 35 sets
(`critic_run.log`), the theory falls inside the operational cell only in these cases:
- real r = 0 S_op, at the cell edge 0.214;
- configuration-null S_op at r = 0.5 (0.945 in (0.905, 0.979]) and r = 2 (1.05 in (0.979, 1.06]);
- configuration-null r = 0 L_op (0.879 in (0.838, 0.905]).

Everywhere else it lies above the cell.

The conclusion does not depend on the definition. `critic_boundaries.png` shows (a) the data brackets with the
branching and pair-operator theory, and (b) the operational cells against the matched theory.

### Theory variant not in the comparison: the pair operator
`branch_{real,null,wnull}.json` also contain the pair-operator refinement of P1 (the theory page's dynamical-correlation
correction). It was computed only on the narrow grids: real g ≤ 0.59, nulls r = 0, 1, 4. Wherever it is defined, it moves
S and onset up by 3–16%, i.e. further from the simulation:

| graph | r | S (pair) | onset (pair) | S (branching) | onset (branching) |
|---|---|---|---|---|---|
| MaleCNS | 0 | 0.253 | 0.300 | 0.245 | 0.291 |
| MaleCNS | 0.25 | 0.330 | 0.433 | 0.308 | 0.395 |
| MaleCNS | 0.5 | 0.388 | 0.552 | 0.353 | 0.476 |
| MaleCNS | 1 | 0.484 | > 0.59 | 0.421 | 0.613 |
| config null | 1 | 1.152 | 1.312 | 1.12 | 1.27 |
| weight-shuffled null | 0 | 0.734 | 0.781 | 0.719 | 0.766 |
| weight-shuffled null | 1 | 0.896 | 1.013 | 0.873 | 0.984 |

### A failure of my own method (reported so it is not mistaken for a finding)
On the nulls, my local log-linear fit puts S 0.6–6% lower than the values above (configuration null r = 0: 0.808 vs
0.841). Between adjacent grid points τ jumps by more than an order of magnitude there, which a local log-linear model
cannot follow; at weight-shuffled null r = 0.5 my estimate even falls outside the data bracket. The interpolated values
in the tables are the better estimates. The honest uncertainty is the data bracket (`critic.json`
`sim.<graph>.<r>.S_data_bracket`).

### Files changed by the critic
- `phase_complete.md`: the [critic]-marked sentences in the header, the Criteria bullet, headline items 2–4 (incl. "comes from" → "is about the size of" in item 2), the Works
  and Fails lists, the table legend, the D columns of the agreement tables, and this section.
- `phase_complete.json`: keys added, none changed (listed under `critic_review.added_keys`):
  - QMF grid cells and upper-bound flags for every theory D;
  - the data bracket, local-fit estimate and theory position for every S and onset;
  - the operational-criteria comparison.

  Rerunning `analysis2_phase.py` drops these keys; rerun `analysis2_critic.py` after it to restore them.
- `code/analysis2_phase_figs.py`:
  - QMF grid-cell bars on the theory D line, with a legend entry, in `phase_real.png` and `phase_nulls.png`;
  - `phase_driven.png` re-laid out 2 × 2, with a new panel (d) comparing simulated and QMF density, the
    configuration-null h = 1e-3 fluctuation curve in (c), and h tick labels written as 1e-5 … 1e-1.

  I regenerated all four figures and looked at each one. `tau_sim_vs_theory.png` is unchanged.
- New files: `code/analysis2_critic.py`, `code/analysis2_critic_fig.py`, `results/analysis2/critic.json`,
  `critic_run.log`, `figs/pd2/critic_boundaries.png`.
- Not changed: `analysis2_phase.py` (not rerun), `phase_complete_table.md` (its generated D columns still lack the grid
  cells), `flow_sim_boundaries.json`, and the raw results.

### Open items the critic could not fix (they need new runs)
- **Theory D is known only to a 20% cell.** A QMF run on a fine g grid around each crossing (a GPU job) would turn the
  cells into values.
- **D untested at several r.**
  - Real graph, r = 2.25, 2.5, 2.75, 3.5, 4.5, 5 and 5.5: the simulation stops at g = 2.6.
  - Weight-shuffled null, r = 2 and 6: it stops at g = 5.
  - Real graph, r = 1.75–6: D is not reached by g = 30.
- **Driven plane.** It exists only at r = 1 for MaleCNS, at one h (1e-3) for the configuration null, and not at all for
  the weight-shuffled null. It therefore has no inhibition axis and no proper null comparison. `driven.md` / `driven.json`
  still describe the stage-1 driven runs.
- **No replica bootstrap for PR (D).** Per-replica occupancy is not stored.
- **Configuration-null theory at r = 0.** It lies below the first dense theory row and rests on extrapolated inputs (R
  linear from g_lin, K from the first row). A theory run starting closer to g_lin would remove that.
- **Pair-operator theory.** It exists only on the narrow grids (table above); it was not computed densely or for all r.
- **Cause of the factor of about 2 in K.** It is not identified. The data show that removing the coincidence term entirely
  is not enough at r = 0, so the one-mode birth–death reduction itself is a candidate.
