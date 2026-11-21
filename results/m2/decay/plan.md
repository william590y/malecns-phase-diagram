# Lane m2/decay: the h = 0 decay phase diagram of Model 2 (plan, pre-registered)

Written 2026-11-09 11:10, before any run of this lane. Spec: `results/m2/MODEL2.md` (followed exactly; deviations are
listed in section 7). Simulator `code/ald_sim.py` md5 61580c76 (checked at submission; every task log prints the md5).
Code: `code/m2_decay_args.py` (argument lines), `code/m2_decay_analyze.py` (analysis), `code/m2_decay_figs.py` (figures).
Results here (`results/m2/decay/`), figures `figs/m2/decay_*.png`. Jobs `m2-decay-*`.

## 1. Questions
1. Q/S/L/D boundaries of M2 in (g, r) on the real connectome (23 r of `R_ALL`), with intervals, against M1 (analysis2).
2. The same on the configuration null and the weight-shuffled null (`R_CORE`), against M1's nulls.
3. M2s (per-target lLN1_bc signs) against M2 (`R_CORE`). Attribution arm M1s (signs fix alone) at r = 0, 1, 4.
4. Oscillation flag (MODEL2 5a) at every point with at least 4 replicas alive at T.
5. Where activity lives at each boundary (S, onset, D): AL-module share, standard regions, top-5 superclasses with
   enrichment, PR.
6. Shape of the lifetime distribution in the slow window: exponential (one metastable state, as in M1) or heavy-tailed
   (rare regions, Griffiths-like)?
7. Depression sensitivity: M2_allsyn, M2_KW09, M2_N15, M2_tau30 at r = 0, 1, 4.

## 2. Runs (protocol `decay_h0` of `m2_common.protocol`, B = 64, T = 3000, push engine, standard regions)
Every run: `--rec_counts --counts_win 1500:3000` (oscillation flag). `--counts_reg` in addition on the real-graph rows of
M2 and M2s at `R_CORE` and on the M1 rows (AL-module and nonORN signals). Distinct `--seed` per task (no common random
numbers across r or g sets); seeds are listed in `sim/index_*.json`.

| stage | rows (model, graph, r) | g | tasks |
|---|---|---|---|
| 1 coarse | M2 real, 23 r (`R_ALL`) | `G_COARSE` (40 log points 0.05–30), split into 4 interleaved sets of 10 | 92 |
| 1 | M2s real, `R_CORE` | same | 28 |
| 1 | M2 configuration null (`--null --null_seed 0`) and weight-shuffled null (`--null --null_kind weights --null_seed 0`), `R_CORE` | same | 56 |
| 1 | the same two nulls with `--null_seed 1`, r = 1 (graph-to-graph variation) | same | 8 |
| 1 | M2_allsyn, M2_KW09, M2_N15, M2_tau30 at r = 0, 1, 4 | same | 48 |
| 1 | M1s at r = 0, 1, 4 | same | 12 |
| 1 | M1 at r = 0, 1, 4 (pipeline check against analysis2 and an M1 oscillation reference; `--null_seed` not used) | same | 12 |
| 2 refine | every stage-1 row: `m2_common.refine(lo, hi, 8)` inside the bracketing grid cell of each of S, onset and D (when the boundary is crossed inside the grid) | 8 per boundary | one task per (row, boundary) |
| 3a long runs | M2 real r = 0, 1, 4 and M2 configuration null r = 1: 6 g at the stage-1 isotonic τ = 100, 200, 400, 800, 1600, 3000 | T = 30000, B = 2 × 64 | 48 |
| 3b where at S | M2 and M2s real, both nulls, `R_CORE`: g at the S crossing (after stage 2), `--occ_win 50:400` | 1 g | ≤ 28 |

The configuration null with `--null_seed 0` is the graph of M1's analysis2 null runs (all M1 runs used `--seed 0`, and
`--null_seed 0` equals `--seed 0`, test V3), so M1 and M2 null rows are on the same graph.

## 3. Definitions (MODEL2.md section 5a; identical to analysis2 so that M1 and M2 compare like for like)
- Extinction time: first record time with no active unit on the default log record grid (`m2_obs.extinction_times`);
  τ = censored-exponential MLE Σ min(t_i, T)/deaths. All runs of a row (stages 1 and 2) are pooled per g.
- Boundaries: isotonic regression of ln τ in g with weights deaths + 0.5 (zero deaths regularised as E/0.5 for this
  purpose only, as analysis2), first crossing of ln 200 (S) and ln 3000 (onset) interpolated in ln g. D: isotonic PR/N
  among points at or above the onset crossing, first crossing of 1/4; "at the onset" if the first sustained point is
  already delocalized. PR over all N neurons from `occ` (replica mean, t > 1500). Also g_half (half the replicas alive at T).
- Uncertainty: envelope of the bracketing grid cell and a 95% replica bootstrap (2000 resamples). D has no per-replica
  data, as in M1.
- Per-point phase for the tile diagrams: Q τ < 200; S 200 ≤ τ < 3000; L τ ≥ 3000 and PR < N/4; D τ ≥ 3000 and PR ≥ N/4
  (point estimate, not isotonic; M1 tiles from the per-point values in `results/analysis2/phase_complete.json`).
- Diagnostic (reported, not a boundary): late hazard τ_late = exposure/deaths over t ∈ [100, T] among replicas alive at
  t = 100, to separate the metastable state's lifetime from the transient of the fully active start.

## 4. Pre-registered criteria
**4.1 Oscillation (MODEL2 5a, `m2_obs.osc_metrics`, thresholds unchanged).** Window 1500 < t ≤ 3000, replicas alive at
T (last active time from `reg_last` ≥ 3000 − dt), at least 4 of them. Signal: `cnt_act / N`; with `--counts_reg` also
AL_module and nonORN. OSC = trough ≤ −0.10 AND peak ≥ 0.10 AND period ≤ 375; OSC-strong = OSC AND C(2P)/C(P) ≥ 0.7;
bursting if CV_t ≥ 0.3, otherwise quasi-cycle. Any OSC point is re-checked on a smaller graph if one can be built for it
(section 6 of MODEL2); if not, it is reported as unchecked.

**4.2 Lifetime shape and the Griffiths-like window (stage 3a; exact extinction times).**
- Extinction time per replica = max over the three default disjoint regions of `reg_last` + dt (step resolution),
  censored at T = 30000.
- Per g: (a) KS distance to a shifted exponential (t0 = earliest death, τ = MLE above t0), p-value by parametric
  bootstrap (2000) that reproduces censoring and the re-fit; also the unshifted version. (b) CV of t − t0 when no replica
  is censored, against the exponential's 95% range from the same bootstrap. (c) Kaplan–Meier survival on log axes.
- ρ(t) = mean over all replicas (dead = 0). Fits of ln ρ: exponential (linear in t) and power law (linear in ln t) on
  two windows [t0, t_end] with t0 = 50 and t0 = 200 (after the fully active transient), t_end = last record with at least
  5 replicas alive. Power-law exponent also fitted on the two halves (in ln t) of each window.
- **Griffiths signature at one g**: lifetimes broader than exponential (shifted-exponential KS p < 0.01 AND CV above the
  exponential's 97.5% point) AND ρ(t): power-law R² ≥ exponential R² over ≥ 1 decade of t, with the half-window
  exponents within 30% of each other, in BOTH windows.
- **Griffiths-like window**: the signature at ≥ 3 consecutive long-run g of a row AND the power-law exponent changing
  monotonically by ≥ 20% across them.
- **Exponential (no Griffiths window)**: no g with the signature AND the shifted-exponential KS p ≥ 0.05 with CV inside
  the exponential range at ≥ 2/3 of the g with τ ≥ 200.
- Anything else: **inconclusive**, reported as such, with what failed.
- Screen on all rows (stages 1–2, B = 64): per point with τ ≥ 200 and ≥ 10 deaths, the same KS test (500 bootstrap) on
  the log-grid extinction times, Fisher-combined per row. A screen only; the verdict uses stage 3a.

**4.3 Where activity lives.** At the onset: the first pooled grid point with τ ≥ 3000 ("first sustained g", as aldef).
At D: the first point with PR ≥ N/4. At S: the stage-3b run. Reported: AL-module share Σ_module o / Σ o, the standard
region shares, top-5 superclasses by share with enrichment = share / size fraction, PR and PR/N, and the quasi-stationary
active fraction per region (`win_mean` 1500–3000 among replicas alive at T). "AL-led" = AL-module share ≥ 25% at the
first sustained point (M1 r = 1: 47%; aldef M2-equivalent arm: 6%).

**4.4 Variants.** Reported: ratio of S, onset and D to M2 at the same r; whether the phase sequence changes (an L phase
appears or disappears; D at the onset); whether any variant is AL-led (4.3) or shows OSC. These are descriptive; no pass
mark.

**4.5 M1 pipeline check.** My code on the stage-1 M1 rows (coarse grid only) must put analysis2's S and onset inside my
total interval at r = 0, 1 and 4. If it fails, the analysis code is wrong until shown otherwise.

**4.6 Order parameter.** QS active fraction among survivors (1500–3000) against g, per row. A jump at the onset is
reported but is not taken as evidence of a first-order transition (survivor conditioning creates apparent jumps;
hysteresis and FSS belong to lane `nature`).

## 5. Figures (`figs/m2/decay_*.png`, each looked at before it is reported)
- `decay_phase_tiles.png`: per-point Q/S/L/D tiles, M1 (analysis2) | M2 side by side, real graph, 23 r (style of
  `figs/pd_phase_gr.png`), OSC points marked.
- `decay_phase_lines.png`: boundaries with intervals in (g, r), M1 | M2 side by side (style of `figs/pd2/phase_real.png`), g_lin.
- `decay_phase_nulls.png`: tiles of both nulls, M1 | M2.
- `decay_order.png`: QS active fraction, PR/N and AL-module share heat maps, M1 | M2 (style of `figs/pd_order_gr.png`).
- `decay_m2s.png`: M2 vs M2s (and M1s) boundaries and AL share.
- `decay_variants.png`: sensitivity variants.
- `decay_where.png`: activity location at the boundaries.
- `decay_lifetimes.png`: survival curves and ρ(t) in the slow window (M2, null; M1 long runs for reference).
- `decay_osc.png`: autocorrelations at sustained points.

## 6. What would change the conclusions
- An OSC-strong point or a bursting point: a new phase label O in setting (a).
- A Griffiths-like window (4.2): the original target of the project, now on the corrected model.
- An AL-led first sustained point in M2 at any r: the defect would not be removed at that r.

## 7. Deviations from MODEL2.md
- Nulls, M2s and variants on `R_CORE` or {0, 1, 4} rather than `R_ALL` (MODEL2 section 7 assigns `R_ALL` to the two main
  diagrams; M1's nulls had 6 r). M2s may be extended to `R_ALL` if it differs from M2.
- MODEL2 asks for "the first recorded time with no active unit on the default log grid" for τ; I follow it. For the
  lifetime SHAPE I use the exact (step-resolution) last-active time from `reg_last`, which removes the ~8% grid
  quantisation from the KS/CV tests; the τ boundaries are unaffected. (Amended about 11:12, before any output existed: the
  stage 1–2 lifetime screen of 4.2 also uses the exact times, not the log-grid times.)
- Where at S needs a dedicated run (3b), because at τ = 200 almost every replica is dead before the t > 1500 occupancy
  window.

## 8. Post-hoc additions (written about 11:50, after seeing the stage-1 lifetime screen; NOT pre-registered)
The screen showed that, in M2, deaths from a fully active start come in two groups: early deaths at t ≈ 30–100 (the
network falls silent while the resources depleted by the start are still recovering, TAU = 157) and later deaths from the
quasi-stationary state. A shifted exponential with t0 = earliest death cannot fit such a mixture, so the pre-registered
KS test (4.2) rejects for a reason that has nothing to do with rare regions. Added, and labelled post hoc wherever reported:
- late-conditional test: among replicas alive at t_c = 200 (and 500), the residual lifetimes t − t_c, KS against an
  exponential through the origin (parametric bootstrap with censoring), and CV when uncensored;
- the early-death fraction P(t < t_c) and the late lifetime τ_late (constant-hazard MLE after t_c);
- the empirical hazard h(t) in log-spaced time bins (long runs): a plateau = one metastable state; a hazard that keeps
  falling like 1/t = power-law survival (rare regions).
The pre-registered verdict of 4.2 is still computed and reported as is.

## 9. Oscillation finite-size check (written 12:33, after the interim OSC table and before any of these runs)
The interim analysis flags OSC (not strong, apart from one 7-replica point) at r = 0 only: on the real graph near the
onset (g ≈ 1.15–1.25) and on the configuration null at g ≈ 1.6–3.0. MODEL2 5a requires a check on a smaller graph.
- Graphs: configuration nulls N = 20k, 40k, 80k (`write_graph("cnull", N, 1)`; the 167k point is the null_seed-0 null
  already run); real subsamples that keep the AL module, N = 40k, 80k (`write_graph("realsub", N, 1)`; the 167k point
  is the real graph). Only the default regions are used on these graphs.
- Runs: M2, r = 0, decay_h0 with total counts, B = 64, T = 3000, the 22 `G_COARSE` points from 0.8 to 30.
- Measures at each (N, g) with ≥ 4 replicas alive at T: OSC metrics of the total active fraction (alive replicas,
  1500–3000), var_t(x) (replica mean), and the QS density ρ_QS.
- Decision, at matched ρ_QS (log var_t interpolated in log ρ_QS within each N; the common ρ_QS range of the sizes):
  slope of log var_t against log N ≤ −0.7 → noise-driven quasi-cycle (finite-size fluctuation, var ∝ 1/N);
  ≥ −0.3 → macroscopic population oscillation; in between → inconclusive. The configuration null is the clean family
  (same construction at every N, own normalisation); the real subsamples change the network as well as N (c is fixed at
  the full-brain value), so they are reported as a weaker check.
