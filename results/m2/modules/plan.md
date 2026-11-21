# Lane m2/modules: de Candia-style module phases under Model 2 (plan, pre-registered)

Written 2026-11-09 15:00, before any run of this lane. Spec: `results/m2/MODEL2.md` (followed; deviations in section 8).
Simulator `code/ald_sim.py` md5 61580c76 (checked at submission; every task log prints the md5). M1 baseline:
`results/pd3/REPORT.md`, `results/pd3/multistability.md`, `results/pd3/module_theory.md` (MT4, MT5).
Code: `code/m2_modules_*.py`. Results here (`results/m2/modules/`), figures `figs/m2/modules_*.png`. Jobs `m2-modules-*`.

## 1. Questions
- **Q1 Module phase diagram.** In (g, r), which modules are active under M2? Classify every point as symmetric low (SL),
  broken-symmetry B_m (m of the M modules on, with identities) or symmetric high (SH), in setting (a) (h = 0, fully active
  start) and setting (b) (ORN drive 0.04). Compare with M1 on the same pipeline.
- **Q2 Left/right symmetry.** Does M2 have a broken L/R symmetry (de Candia's B1L | B1R) for the two antennal lobes, and,
  new here, for the two hemispheres? Do one-sided starts re-symmetrise, and how fast? Is there L/R competition
  (alternation, anti-correlation)?
- **Q3 Multistability across initial conditions.** Do different starts (module subsets, hemispheres, random densities, quiet
  vs fully active) reach one state or several, in (a) and (b)? In (b) silence is not absorbing, so a quiet-start driven
  state and a fully-active-start state could coexist near the self-sustaining line: this is the main new coexistence test.
- **Q4 Which modules can be ON alone?** (i) In the network, at h = 0: does a start confined to one module survive and stay
  confined (a B1-like state), die, or spread to the common state? (ii) In isolation (the module's own subgraph, the rest of
  the brain absent): at what g does each module sustain itself (M1 vs M2)?
- **Q5 M1 → M2.** How do the answers to Q1–Q4 change from M1 to M2 (and M2s, M2_tau30 for the L/R test)?

## 2. Partitions, regions and start sets (`code/m2_modules_setup.py` → `results/m2/modules/groups/`)
Reused from pd3 (`results/pd3/modules/`), nothing recomputed:
- **P0** (34 modules, pd3's spectral islands; primary, comparable with pd3 M3): AL-L 248, AL-R 275, AL-mid 3, isl1–isl30,
  rest 158,745.
- **P1** (Leiden, γ = 4, seed 1): the 50 modules with ≥ 50 neurons, plus one lumped `P1_small` group.
- **P2** (superclass × side): the groups with ≥ 50 neurons, plus one lumped `P2_small` group. Gives identities and the
  explicit L/R pairs (e.g. cb_intrinsic|L vs |R).
- **Hemispheres:** HEM_L = all neurons with side L, HEM_R = side R (all superclasses, AL included).
- The 5 standard M2 regions and ald_sim's 3 default regions are kept.
All of them are passed as `--region NAME=idx:groups/regions.npz:KEY` (global ids), so every run records per-replica window
means (`--win`) and time courses (`reg_rho`, `--rec_lin 5`) for every module.

**Start sets** (`--init KEY`, `--exp quiet`; `all` uses `--exp decay` in (a), `--init all` in (b)):
AL-L, AL-R (P0), AL (the 526 module), hemiL, hemiR, nonAL (everything outside the 526 module = pd3 MT3's rest_only),
CB (superclass cb_* and descending_neuron*), VNC (vnc_*, ascending_neuron, sensory_ascending*, efferent_ascending),
OL (ol_*, visual_projection*, visual_centrifugal), isl1, isl3, isl5 (P0), rand10 / rand1 (`--init all --init_frac 0.1 / 0.01`),
and in (b) also `quiet` (the standard driven start).

## 3. Part A: module phase diagram from existing outputs (no new simulation; `code/m2_modules_phase.py`)
Data: `occ[g, N]` of the decay lane (M2 real 23 r, M2s 7 r, M1 r 0/1/4; second half, dead replicas count 0) and of the
driven lane (driven window 500–1500: M2 23 r, M1 7 r, M2s 7 r), and pd3's M1 decay files (`results/pd/`, `results/pd2/`,
45 files, 23 r, the inputs of `pd3_module_occ.py`). Duplicate g are pooled weighted by B, as pd3.
- X_k = mean occupancy of module k. In (a) among survivors: X_k / surv2 (surv2 = fraction alive over t ≥ 1500), only at
  points with surv2 ≥ 0.5 (pd3 MT4). In (b) all replicas (nothing is absorbing).
- **Module "on"**: X_k ≥ θ. θ = 1% (primary, = pd3) and 0.1% (secondary, because M2 activity is sparser).
- **Point labels.** (a): SL if surv2 < 0.5 or no module is on; SH if all M modules are on; otherwise B_m (m = number on).
  (b): the same, with SL = no module on. M counts every module of the partition (P0: 34, incl. AL-mid and rest).
- **Activation lines**: per module, the first g where X_k ≥ θ (crossing in ln g, pd3's rule), per r.
- **Mean-level L/R asymmetry**: A = (X_L − X_R)/(X_L + X_R) for AL-L/AL-R and every P2 L/R pair with X_L + X_R ≥ θ.
  Replica means hide spontaneous breaking (half the replicas L, half R), so this only detects structural tilt; Part B is the
  test of spontaneous breaking.
- Outputs `phase_A.json`; figures `modules_phase_map.png` (number of modules on over (g, r), M1 vs M2, settings a and b),
  `modules_activation.png` (activation lines per module, M1 vs M2).

## 4. Part B: left/right symmetry (new simulations)
Runs (B = 64 per start; T = 2000 in (a), 3000 in (b)):
- (a) h = 0: starts all, AL-L, AL-R, hemiL, hemiR at g = {1, 1.25, 1.6, 2.5, 4, 8} × the onset of the row (decay lane,
  τ = 3000 crossing; pd3 used the same multiples of the M1 onset). M2 at R_CORE (7 r); M2s and M2_tau30 (AL-L, AL-R only)
  at r 0, 1, 4; M1 (same 5 starts) at r 0, 1, 4 with pd3's g values (pipeline check against pd3 M1).
- (b) ORN 0.04 (`--drive olfactory=0.037926`, on for the whole run, no removal): starts quiet, all, AL-L, AL-R, hemiL, hemiR
  at g = {0.5, 0.8, 0.95, 1.05, 1.25, 1.6, 2.5, 4} × DT→SS of the row (driven lane, τ_post = 3000 line). M2 at r 0, 1, 4;
  M1 at r 0, 1 (starts quiet, all, AL-L, AL-R).
Measures, for each L/R pair (AL lobes: f = active fraction of AL-L / AL-R; hemispheres: HEM_L / HEM_R):
A(t) = (f_L − f_R)/(f_L + f_R) (undefined when both are silent).
- t_sym = first record time with |A| < 0.3 after a one-sided start (default log grid + every 5 units).
- Late window: (a) t ∈ [1000, 2000]; (b) t ∈ [2000, 3000]. Survivor (a) = alive at T and the pair active in ≥ 50% of the
  late records; (b) every replica.
- **Pre-registered criteria (pd3's c1–c3, unchanged):** (c1) ≥ 25% of surviving L-start replicas keep late mean A > 0.6;
  (c2) L-start vs R-start late mean-A distributions differ, KS p < 0.001; (c3) the all-start late mean A is bimodal
  (2-Gaussian BIC gain > 10, both weights ≥ 0.1, |μ1 − μ2| > 2 max σ). **Broken symmetry** = c1 AND c2 AND c3.
- **Added, switching (rivalry-like) symmetry breaking:** (c4) in ≥ 25% of the all-start survivors, ≥ 25% of the late
  records have |A| > 0.6; AND the replica-median L/R correlation corr_t(f_L, f_R) over the late records is < −0.2.
  Reported separately from "broken symmetry".
- Reported: alive fractions, late f_L, f_R, mean A per start, t_sym quantiles, corr_t(f_L, f_R).

## 5. Part C: multistability across initial conditions (new simulations)
- (a) h = 0, B = 32 per start, T = 2000, the 6 g multiples of Part B, M2 at r 0, 1, 2, 4: starts AL, nonAL, CB, VNC, OL,
  isl1, isl3, isl5, rand10, rand1; together with the Part B starts (all, AL-L, AL-R, hemiL, hemiR; B = 64) of the same cell.
  M1 reference at r 0, 1 (starts AL, nonAL, CB, VNC, OL, isl1, rand10; pd3 g multiples 1.25, 2.5, 8).
- (b) ORN 0.04, B = 32 per start, T = 3000, Part B's 8 g, M2 at r 0, 1, 4: starts nonAL, CB, VNC, OL, isl1, rand10, together
  with Part B's (quiet, all, AL-L, AL-R, hemiL, hemiR; B = 64). Quiet and all also at r 0.25, 0.5, 2, 6.
- Late vector per replica: the P0 34-module active fractions averaged over the late window (`win_mean`); P1 and P2 vectors
  as secondary.
- **Pre-registered (pd3 M3 rule):** relative L1 = |a − b|_1 / max(|a|_1, |b|_1); average-linkage clustering of all
  survivors (a) / all replicas (b) of a cell, cut 0.2. **Multistable** = ≥ 2 clusters each holding ≥ 50% of the replicas
  (of all B) of some start. Also reported: the largest relative L1 between start means.
- **(b) quiet vs all, coexistence candidate:** KS p < 0.001 on the per-replica late nonORN activity AND median(all) /
  median(quiet) ≥ 1.5 (or ≤ 1/1.5) AND the gap of medians in [2500, 3000] is ≥ 0.5 × the gap in [1500, 2000] (not closing).
  **Stage 2** for every candidate: T = 10000 from both starts, B = 64; **confirmed coexistence** if KS p < 0.001 in
  [8000, 10000] and the gap there is ≥ 0.7 × the gap in [4000, 6000]. A closing gap = a slow transient, not coexistence.

## 6. Part D: which modules can be ON alone
- **(i) in network** (from Part C (a) runs): per start S, alive fraction at T; confinement C = late active count inside S
  / late active count (survivors); t_out = first record time at which the activity outside S exceeds the activity inside S
  (counts). **Module-alone state of S** at (g, r) (pre-registered): alive ≥ 25% AND median C ≥ 0.8 AND the S-start
  survivors' cluster differs from the all-start cluster (relative L1 of the means > 0.2).
- **(ii) in isolation** (new runs; `--graph` subgraph of the module, full-brain c = 1/745.91, h = 0, fully active start,
  B = 64, T = 3000, G_COARSE): modules AL (526), AL-L, AL-R, isl1, isl3, isl5, CB, VNC, OL; M1 and M2; r 0, 1, 4.
  τ = censored MLE (`m2_obs`), onset = isotonic τ = 3000 crossing in ln g; S = τ = 200. Bracket = grid cell (×1.178).
  "Can sustain alone at g" = g ≥ its own onset. Compared with the full-brain onset of the same model and r, and with pd3
  MT5 (M1 AL / islands at r 0, 1).

## 7. Predictions written before the runs (to be scored, not used as criteria)
- P1: under M2 the AL is not the first module on in (a); several large modules switch on together at the onset (M1: the AL
  first, then a smeared island cascade).
- P2: no L/R broken symmetry for the AL lobes or the hemispheres in M2 (c1–c3 none), fast re-symmetrisation (t_sym of
  order 10 model time or less), and no switching (c4 none): the L–R coupling is excitatory and depression acts per neuron.
- P3: (a) one non-silent attractor per cell (as M1). (b) No confirmed quiet/all coexistence (null hypothesis); a
  depression-limited up state could in principle coexist with the driven state near DT→SS, which is what the test looks for.
- P4: in isolation under M2 the AL needs g ≈ 2.9–3 at r 1 (aldef all-synapse value 2.92–3.00; excitatory-only expected
  similar or lower), far above the full-brain onset 1.57; no single brain division sustains alone at the full-brain onset.

## 8. Deviations from MODEL2 and reasons
- (a) runs use T = 2000 (not 3000) and late window [1000, 2000], as pd3 M3 (T 2000); the L/R test in pd3 used T = 1000.
  T = 2000 is ≈ 13 TAU, enough for resources to settle. The onset multiples refer to the decay lane's T = 3000 onset.
- (b) uses ORN drive for the whole run (no removal phase): this is a coexistence protocol, not the driven-phase protocol;
  the driven phases come from the driven lane.
- Many more regions than the standard ones (needed for module vectors); `--rec_lin 5` for the L/R time courses.
- Isolation runs use `--no_default_regions` and no standard regions (selectors would be empty on small subgraphs, MODEL2
  9.1 open issue); extinction times come from `surv` on the record grid (`m2_obs.extinction_times`), as the decay lane.
- c: ald_sim derives c = 1/745.91 from the cache (MODEL2's text says 1/742.7; CLAUDE.md documents the 0.43% difference).
  Subgraph files carry the full-brain c.

## 9. Added during the run (after the plan was written; none changes a criterion)
- 15:20: the isolated small modules (AL, AL_L, AL_R, isl1, isl3, isl5; <= 2,292 neurons) ran on the thickstun CPU partition
  (`--device cpu`, code/m2_modules_cpu.sbatch) instead of GPU, because the GPU queue was full; CB, VNC and OL stayed on GPU.
  Same simulator, same lines plus `--device cpu`. The M1 results reproduce pd3 MT5 (AL 0.268 vs 0.271 at r 0, 0.338 vs 0.340 at r 1;
  isl1 0.975 vs 0.973; isl5 1.15 vs 1.16; isl3 2.86 vs 2.89).
- Two LRa tasks (3, 14) were preempted by Slurm and rerun with the same lines and seeds (job 205284).
- Part C cells are matched by g value across starts (M1's module starts use 3 of the 6 g multiples).
- Part B setting (b): the ORN drive is not L/R symmetric in the connectome (ORNs with side R 1,344, L 884, unlabelled 411), so
  hemisphere and AL asymmetries in (b) include an explicit field; c1-c3 test spontaneous breaking on top of it.
- No (b) quiet-vs-all candidate occurred, so stage 2 (T = 10000) was not needed and not run.
- Added post hoc (descriptive): `lr_extremes.json` (smallest L-vs-R KS p, largest bimodality BIC gain per setting/model/pair).
