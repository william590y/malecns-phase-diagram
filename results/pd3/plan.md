# Phase diagram v3 (pd3): plan for four GPU programs

Written 2026-11-09 for workflow wf_98fb990d-4fb (design step). Inputs used:
- the research outputs `results/pd3/literature.md` and `results/pd3/theory_notes.md` (with `theory_*.json`);
- `results/analysis2/phase_complete.{md,json}`, `CLAUDE.md`, `STATE.md`;
- `code/swc.py`, `module_sim.py`, `theory_qmf.py`, `analysis2_phase.py`, `pd3_theory_nullK.py`;
- `refs/decandia2026_symmetry_breaking_modular.txt`.

Every number in this plan is either a design parameter (grid, replica count, run length) or is quoted with its source file.
Cost figures marked "estimate" are budgets built from the measured step times below; they are not results.

## 0. What the plan must deliver

### 0.1 The user's questions

1. **Q1.** A de Candia-style phase diagram for the real connectome, more complete and more compelling than a Griffiths region:
   - symmetric-low / broken-symmetry / symmetric-high phases, with the identities of the modules involved;
   - second-order lines versus first-order lines.
2. **Q2.** "Is this steep but smooth crossover a second-order phase transition?"
3. **Q3 (accepted offers).**
   - (a) A heat map of measured lifetime over (g, r). g_lin is the only solid line; the τ = 200, τ = 3000 and PR = N/4 lines are labelled
     dashed contours ("crossovers, not phase walls").
   - (b) A finite-size check: configuration nulls of several sizes N sharpen with N, while the real AL crossover does not.

### 0.2 What is already established (do not redo; cite)

**Thresholds and the AL module**
- g_lin = 0.1632 (0.16315), with α_eff = 0.906 and c = 1/742.7. g_lin is r-independent (`phase_complete.json` g_lin; theory P1).
- The 526-neuron AL module (248 L, 275 R, 3 midline; 366 excitatory, 121 inhibitory, 39 modulatory) carries the onset
  (`theory_module.json`). The two leading modes are 70/30 L/R hybrids, not one mode per lobe (`theory_notes.md` §1.5).

**Simulated boundaries (`phase_complete.md`)**

| boundary | r = 0 | r = 0.5 | r = 1 | r = 2 | r = 4 | r = 6 |
|---|---|---|---|---|---|---|
| onset (τ = 3000) | 0.266 | 0.306 | 0.344 | 0.402 | 0.508 | 0.579 |
| S (τ = 200) | 0.234 | — | 0.283 | — | — | — |
| D | 1.47 | 4.44 | 14.5 | > 30 | > 30 | > 30 |

- At r = 1.5, D = 28.9. For r ≥ 1.75, D lies above 30 wherever it was tested.
- Onset/g_lin is 1.63 at r = 0 and 2.11 at r = 1.

**Finite-K chain (`theory_bd.json`, `theory_bd2.json`)**
- One K per r reproduces the measured lifetimes: K_needed = 75.8 (r = 0), 36.6–36.9 (r = 1), 16.2 (r = 6).
- Predictions at g_lin for K = 37 / 54.7 / 75.8:
  - τ_top = 12.0 / 14.8 / 17.7;
  - S_c = 240 / 360 / 505 spikes;
  - T_c = 13.3 / 16.4 / 19.5.
- The local slope of ⟨S⟩(T) peaks at 1.55 (K = 37) and 1.66 (K = 75.8); it reaches 1.79 at K = 300 and 1.91 at K = 2000.
- The QS susceptibility peak is at g ≈ 0.195 / 0.201 / 0.209 for K = 75.8 / 54.7 / 36.9.
- Avalanche shapes of any one-mode chain are exactly symmetric.

**Mean-field structure (`theory_module*.json`)**
- de Candia's coarse closure predicts broken symmetry for the AL pair:
  - r = 2.25: B1 and SH coexist at g 0.224–0.273;
  - r ≥ 2.5: B1L | B1R bistability; at r = 4 from g 0.287, with B1 + SH from 0.317; at r = 6 from 0.428 and 0.578.
- The neuron-level QMF of the module has no B1 state at any r ≤ 6. The competition ratio is Γ ≈ 0.003 (winner-take-all needs Γ > 1).
- Island thresholds (weak-synapse):
  - AL pair: 0.163 and 0.168;
  - first non-AL island: 0.482 (λ 1.879, 1/IPR 632, central-brain intrinsic);
  - next: 0.521 (76% AL), 0.529, 0.580.
- Islands below g = 0.3 / 0.5 / 1 / 1.47 / 2 / 3: 2 / 3 / 22 / 56 / 118 / 266.
- Inter-module crossover ratios: r₁*(rest←AL) = 2.04 and r₁*(AL←rest) = 9.6.

**Null-family predictions (`theory_nullK.json`, `theory_bd.json` null_family)**
- K_nc of size-N stub-matched nulls (seeds 1 / 2):

  | N | K_nc, seed 1 | K_nc, seed 2 |
  |---|---|---|
  | 5k | 212 | 248 |
  | 10k | 426 | 212 |
  | 20k | 416 | 839 |
  | 40k | 888 | 1164 |
  | 80k | 2511 | 2706 |
  | full null | 5539 | — |

- Predicted onset/g_lin at r = 0: 1.76 (2k), 1.39 (5k), 1.25 (10k), 1.16 (20k), 1.11 (40k), 1.07 (80k), 1.045 (167k).
- The configuration null has g_lin = 0.8287 and simulated onsets 0.867 (r = 0) and 1.11 (r = 1).

**Step times (measured)**
- Full brain, dense, B = 32: 1.26 ms per step on an RTX PRO 6000 Blackwell (18.9 s for 15,000 steps at a sustained g;
  `logs/pd2-sim_168141_1.out`).
- A6000: 2 ms at B = 16 and 11 ms at B = 256 (`STATE.md` stage 2).

### 0.3 Shape of the answer we are testing (pre-registered)

**Q1.** The connectome realises de Candia's diagram only in a broken, heterogeneous form:
- SL ends at g_lin, the symmetric-low edge. This is a second-order (transcritical, MF-DP) line, vertical in (g, r).
- The AL module is the "B1-like" phase (L). It is protected by heterogeneity, not by inhibition.
- Further modules switch on through an imperfect, smeared cascade. These are dotted lines, not sharp ones.
- D is the SH-like global phase.
- First-order lines exist only if multistability or hysteresis is found at the neuron level. The theory audit predicts none.
- Alongside it we show the diagram de Candia's own closure would predict for the same modules, which has broken symmetry and
  first-order lines. The contrast is part of the answer.

**Q2.** The crossover is the finite-K-rounded image of a second-order MF-DP point at g_lin, with K ≈ 37–76:
- the visible steep crossover is the module's extinction-time crossover, 1.6–2.1 × g_lin;
- it drifts with T;
- it does not sharpen with N.

The tests are listed in §4.6 (criteria Q2-a to Q2-e). Each can come out negative, and negatives are reported as such.

## 1. Shared infrastructure (build first; all programs use it)

| file | what | resources |
|---|---|---|
| `code/pd3_array.sbatch` | generic GPU array; line `SLURM_ARRAY_TASK_ID+1` of the args file = `<pd3 script> <args>`; partitions `gpu,ellis,default_partition`, `--gres=gpu:1`, `--constraint="ampere\|ada\|hopper\|blackwell"`, `--nice=500`, `--exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01`, `-c 4 --mem=48G -t 02:00:00`, log `logs/%x_%A_%a.out` | GPU |
| `code/pd3_cpu.sbatch` | the same for CPU scripts: `-p thickstun -c 16 --mem=128G -t 03:00:00` | CPU |
| `code/pd3_common.py` | shared helpers, so that no existing code is modified (copy, never import, `analysis2_phase.py`, which has no `__main__` guard). Contents: graph loading (`load_graph(kind, r, N=None, seed=None)` returning a torch CSR in the swc.py convention, coef = σ·w·c), the censored-exponential MLE, `iso`, `first_cross`, `log_tau`, `log_tau_smooth`, `k_needed` (copied from `analysis2_phase.py`), the module membership loader, and the exact g_lin by bisection (λ_PF(B(g)) = α_eff with B_ij = f(g E_ij), f(x) = (1 − e^{−tanh(x)·dt})/dt; GPU power iteration, relative tolerance 1e-5). | — |
| `code/pd3_fss_graph.py` | builder for size-N graphs (used by redraw_fss and criticality), see §2.2 | — |
| `code/pd3_partition_v0.py` | partition P0 (minutes, CPU), see §3.1; needed by criticality (per-module spike split) and multistability. **Run it first.** | CPU |

**Submission.**
- Command: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch -J pd3-<prog> --array=0-(n-1)%K code/pd3_array.sbatch <argsfile>`.
- Each program keeps `results/pd3/<prog>/JOBS.md` (job ids, status, reruns). The synthesis step merges these into `STATE.md`; programs do not
  edit `STATE.md` concurrently.

**Conventions.**
- The update rule is the same as in swc.py:
  - synchronous dt = 0.2;
  - an active unit turns off with probability 1 − e^{−dt};
  - a quiescent unit turns on with probability 1 − exp(−tanh(s)₊ dt), where s = g Σ_j c w_ij σ_j a_j + h;
  - σ = +1 for excitatory, −r for inhibitory, 0 for modulatory;
  - raw normalisation.
- Time is in model units (1/α).
- Every new simulator must pass the validation test listed under its program before production runs.
- Figures:
  - Okabe–Ito colours for lines and categories;
  - cividis or viridis for sequential maps;
  - labelled axes with units and readable fonts (≥ 9 pt at the saved size);
  - every PNG is opened and checked by eye before it is reported.

---------------------------------------------------------------------------------------------------------------------

## 2. Program `redraw_fss`: lifetime heat map, and finite-size scaling of null onsets

### 2.1 Part A: continuous lifetime heat map (CPU only; no new simulation)

The script `code/pd3_redraw.py` runs on CPU in minutes.

**Inputs.** Exactly the simulation files and pooling rules of `analysis2_phase.py`:
- `results/pd/decay_r*_*.npz`, `results/pd2/onset_*`, `deloc_*`;
- at r = 1, also the fine, upper, original and long sweeps, truncated at T_c = 3000.

**Theory curves.** Taken from `phase_complete.json` "theory": S and onset from the P2 recursion with K_eff, and the QMF D.

**Computation.**
1. Per (graph, r) and per g, pool replicas as in `load_sim`:
   - deaths d and exposure E;
   - τ_sim = E/d. With d = 0 the point is censored: flag it and keep the lower bound 2E/χ²₀.₉₅(2);
   - PR/N from occ (T = 3000 runs only).
2. Per r row:
   - isotonic ln τ_sim in g, with weights d + 0.5;
   - linear interpolation in ln g onto a common grid of 320 ln-uniform points in [0.1, 30];
   - outside the measured g range of that row: blank, with grey hatching labelled "not simulated".
   - Do not interpolate across r. Each of the 23 real r values is drawn as a band between the midpoints of neighbouring r values.
3. Panel A (lifetime):
   - colour = log₁₀ τ_sim on [0, log₁₀ 3000] (cividis);
   - censored cells (τ > 3000 within T = 3000) in a separate flat light colour, labelled "sustained over T = 3000".
4. Panel B (sustained region): log₁₀(PR/N) from 1e-3 to 1 (viridis); non-sustained cells greyed.
5. Lines (identical on both panels):
   - g_lin = 0.1632 as a **solid** black vertical line, labelled "g_lin: mean-field critical line (silence loses stability; r-independent)";
   - **dashed**, from the verified `phase_complete.json` sim boundaries: τ = 200, τ = 3000 and PR = N/4, each with its 95% total-uncertainty
     band (thin shading) and an inline label "(crossover)";
   - **dotted**, lighter: the matching theory lines (P2 τ = 200 / 3000, QMF PR = N/4).
6. Panel C (inset), drift of the onset with observation time at r = 1:
   - g_half = 0.351 (T 3000), 0.366 (8000), 0.386–0.399 (30,000) (`longruns.md`);
   - chain crossings with K = 36.9: 0.346 and 0.395 (`theory_bd.json` real_crossings).
   - It shows that the "onset" moves with T, i.e. a crossover rather than a wall.
7. Consistency check. Run matplotlib `contour` on the interpolated field at τ = 200 and 3000 and compare with the `phase_complete.json`
   boundaries row by row. Report max |Δ ln g|. If it exceeds one heat-map grid step (ln 300/319 = 0.018 in ln g), find out why
   before finishing.
8. A second figure gives the same treatment to the configuration and weight-shuffled nulls (6 r each, with their own g_lin = 0.8287 / 0.663).

**Outputs.**
- `results/pd3/redraw/heatmap_{real,null,wnull}.npz`: grid, log10τ, censored mask, PR/N, row edges;
- `results/pd3/redraw/heatmap_meta.json`: inputs and the consistency check;
- `figs/pd3/phase_lifetime_heatmap.png` (panels A, B, C) and `figs/pd3/phase_lifetime_heatmap_nulls.png`.

**Acceptance.** Every dashed contour coincides with the verified boundary within its uncertainty band, and g_lin is the only solid line.

### 2.2 Part B: finite-size scaling (GPU array `pd3-fss`)

**Graph builder** `code/pd3_fss_graph.py`. The graph is a deterministic function of (kind, N, seed) and is built inside each task.

- `cnull` (configuration null of size N): exactly the `pd3_theory_nullK.py` construction, so that seeds 1 and 2 reproduce its K_nc table.
  - rng = `default_rng(1000*seed + N % 997)`; draw a uniformly random N-subset.
  - Out-stubs are the edges whose presynaptic neuron is in the subset; they keep their weight and presynaptic sign. In-stubs are the edges
    whose postsynaptic neuron is in the subset.
  - Match m = min(#out, #in) stubs at random. c = 1/(mean in-strength of the new graph).
  - For N = 167,216: seed 0 is swc.py's null (post permuted with `default_rng(0)`); seeds 1 and 2 are stub-matching on the full set.
- `realsub` (real wiring, AL always kept): a random N-subset that contains all 526 AL-module neurons (`module_graph.npz` idx), with the
  induced subgraph.
  - c is fixed at the full-brain value 1/742.7, so the AL's internal couplings are unchanged.
  - This is a control: it shows the onset does not depend on N while the AL is intact. The informative real-network size family is the
    nested AL sub-modules (criticality C4).

**Per-graph diagnostics (GPU, at task start; saved with the output).**
- λ_PF(E) and the weak-synapse g, g_lin computed exactly by bisection, the Perron vectors at g_lin, K_nc = 1/(⟨u²,v⟩⟨u,v²⟩), 1/IPR,
  edge count and c.
- R(x) = λ_PF(B(x·g_lin))/α_eff on the simulation grid (used for the chain overlay).
- Validation:
  - cnull seeds 1 and 2 at N = 5k–80k must reproduce `theory_nullK.json` λ_PF and K_nc to 1e-6 relative;
  - the full null must give g_lin = 0.8287 (`phase_complete`).

**Simulation** `code/pd3_fss_sim.py` (adapted copy of swc.py `--exp decay`).
- Start fully active; h = 0; T = 3000; B = 64 replicas.
- Record:
  - the exact extinction step per replica (int32 [g, B]; not only on the log grid);
  - rho[g, t, B] and surv on swc.py's log grid;
  - occ[g, N] (float16, second half);
  - per-replica mean and variance of n_active among survivors over [T/2, T], sampled every 5 steps.
- The g grid is relative to each graph's own g_lin:
  - r = 0: x = g/g_lin at 30 ln-spaced points in [0.90, 2.50];
  - r = 1: 36 points in [0.90, 8.0]. The chain predicts onset/g_lin = 5.74 at N = 10k and "not reached" below that size
    (`theory_notes.md` §2.4 vi). An onset above 8 is reported as "> 8".

**Array layout.** One task = one (graph, r) with all g, for 54 tasks in total:
- cnull: N ∈ {2500, 5000, 10000, 20000, 40000, 80000, 167216} × seeds {1, 2, 3} (167216: {0, 1, 2}) = 21 graphs;
- realsub: N ∈ {5000, 20000, 80000} × seeds {1, 2} = 6 graphs;
- each graph at r ∈ {0, 1}.

Submit with `%16`, `-t 02:00:00`. Cost estimate: ≤ 30 g × 15,000 steps × ~2 ms ≈ 15 min for the largest tasks, since only sustained points run
to T; tasks with N ≤ 40k take a few minutes.

**Analysis** `code/pd3_fss_analyze.py` (CPU). Per graph and r:
- τ_sim(x) by censored MLE.
- The onset x_on (τ = 3000), S (τ = 200) and g_half: isotonic fit plus ln interpolation, as in `analysis2_phase.py`, with a 2000-resample
  replica bootstrap.
- Steepness s = d ln τ/d ln g at τ = 1000 (finite difference on the isotonic fit). This measures sharpening without the T convention.
- τ(x = 1), the lifetime at the graph's own g_lin. All 64 replicas die there, so it is uncensored.
- K_fit: the single K for which `log_tau_smooth(R(x)·α_eff, K)` best fits ln τ_sim over the slow-window points (200 ≤ τ ≤ 3000, ≥ 3 deaths),
  by least squares in ln τ.
- χ_surv(x) = var(n)/⟨n⟩ among survivors: location and height of the peak.
- Overlays:
  - the chain with each realisation's K_nc (meaningful at r = 0 only; at r = 1 the inhibitory coincidence term dominates: theory K_eff 95.7
    against no-coincidence 5,660 for the full null, `phase_complete.md`);
  - the κN rule (`theory_bd.json` null_family).

**Figures** (`figs/pd3/`).
- `fss_onset_vs_N.png`: two panels (r = 0, 1).
  - (x_on − 1) against N, log–log: seeds as markers, the seed mean as a line, the chain prediction as a band;
  - the real full brain as horizontal lines at 1.63 and 2.11, and the realsub points.
- `fss_lifetime_curves.png`: τ(x) per N (sequential colour by N), with the real curve overlaid.
- `fss_steepness.png`: s against K_fit.
- `lifetime_vs_K.png`: τ(x = 1) against K, shared with criticality C4.

**Decision criteria (pre-registered).**
- **FSS-1, the null sharpens.**
  - At r = 0 the seed-mean x_on decreases with N (Spearman ρ ≤ −0.9 over the 7 sizes), with x_on(167k) ≤ 1.10 and x_on(10k) ≥ 1.15.
  - At r = 1 the same monotone trend holds.
- **FSS-2, the real crossover does not sharpen.** realsub x_on lies within ±5% of 1.63 (r = 0) and 2.11 (r = 1) at every N.
- **FSS-3, the mechanism is module size.**
  - Nulls: K_fit ∝ N^a with a ≥ 0.7;
  - τ(x = 1) ∝ K^{0.5 ± 0.1} across nulls and nested AL sub-modules;
  - s increases with K_fit.
- **Negative branch.** If the nulls do not sharpen (FSS-1 fails), report it as a result: hub-star modes or the ×2 seed scatter of K/N
  (`theory_nullK.json`) dominate. The finite-size argument then rests on the nested-sub-module family (C4) alone.

---------------------------------------------------------------------------------------------------------------------

## 3. Program `module_theory`: modules, quotient mean field, neuron-level QMF from many starts

### 3.1 MT1: partitions and module statistics (CPU)

**P0 "islands-v0"** (`code/pd3_partition_v0.py`; fixed now, minutes; run before everything else).
- Modules AL-L (248), AL-R (275) and AL-mid (3) are the 526-module split by `sides`.
- Then take the spectral islands of `tail_real_r0_dir.npz` in decreasing Re λ:
  - island k = the 90% |v|² support of P[:, k];
  - skip modes with AL share ≥ 0.5 (indices 0, 1, 3, 5 among the first six);
  - merge modes whose supports overlap > 50%. The complex pairs share supports, e.g. λ 1.308 appears twice;
  - assign neurons first-come in λ order;
  - keep the first 30 islands with ≥ 20 neurons.
- Everything else is "rest", which gives 34 modules.
- Island 1 is mode 2 (λ 1.879, 1/IPR 632, 90% support 2,293 neurons, cb_intrinsic). Many later islands are VNC motor or intrinsic.
- Save: membership int16[N] and a table of size, λ, α_eff/λ, superclass mix and side.
- Output: `results/pd3/modules/partition_P0.npz`.

**P1, Leiden** (`code/pd3_modules.py`, `pd3_cpu.sbatch`, about 1 h).
- Install: `pip install --user igraph==0.11.8 leidenalg==0.10.2` (PyPI is reachable from this host; checked 2026-11-09).
- Graph: the symmetrised excitatory weighted graph A = E + Eᵀ, with weights c·w on excitatory-presynaptic edges only.
- Method: RBConfigurationVertexPartition with resolution γ ∈ {0.5, 1, 2, 4} × seeds {0, 1, 2}.
- Report: modularity, module count, size distribution, NMI across seeds and against P0.
- Use the γ with the highest mean NMI across seeds that gives ≥ 20 modules of ≥ 50 neurons.
- Fallback if the install fails: `networkx.community.louvain_communities` on the excitatory graph restricted to w ≥ 3 (count the edges
  first), seeds 0–2, resolution {1, 2}.

**P2, anatomy.** Superclass × side.

**Per partition**, for the top M = 30 modules by Perron root Λ_k of E_kk, plus "rest":
1. Thresholds and size:
   - Λ_k and the weak threshold α_eff/Λ_k;
   - the exact branching threshold g_lin,k (bisection on the restricted block);
   - K_k = 1/(⟨u²,v⟩⟨u,v²⟩) at g_lin,k.
2. Quotient matrices (definitions in `theory_notes.md` §1.2):
   - uniform E^UN_kl and I^UN_kl;
   - Perron-weighted E^PW_kl = ⟨u^k, E_kl v^l⟩ and I^PW_kl, with v mean-1 and ⟨u, v⟩ = 1.
3. Equitability error (Schaub et al. 2016): for each target k and source l, the coefficient of variation over i ∈ k of
   Σ_{j∈l} c w_ij, separately for excitatory and inhibitory input. 0 means equitable, the condition behind de Candia's exact reduction.
4. r₁*(k←l) = E_kl/I_kl and r₀*(k) = E_kk/I_kk, under both closures.
5. Dead-zone screen. This generalises `neuron_dead_zone` in `pd3_theory_module2.py`.
   - For every ordered pair l → k among the top 30 with any coupling, and r ∈ {0.5, 1, 2, 4, 6}: the fraction of k's neurons, and the
     u^k-weighted fraction, with net-positive input Σ_{j∈l}(E_ij − r I_ij) v^l_j > 0.
   - A **de Candia candidate pair** has r₁*(k←l) < r₀*(k) and a net-positive share < 5% at some r ≤ 6. Only such pairs can host neuron-level
     broken symmetry. For the AL pair the share is 22–38% (`theory_module2.json`), so that pair is not a candidate.

**Outputs.** `results/pd3/modules/{partition_P0,P1,P2}.npz`, `modules_table.json`, `quotient_{P}.npz`, `deadzone_{P}.json`.

**Figure** `figs/pd3/modules_partition.png`:
- module size against g_lin,k, coloured by superclass;
- equitability errors;
- dead-zone share against r for the 10 pairs with the strongest |I^PW|.

### 3.2 MT2: de Candia's closure on connectome modules (CPU, minutes)

Script `code/pd3_coarse_mf.py`.

**Model.** The Perron-weighted M-module map with the simulator's synchronous rates:
X_k ← X_k e^{−dt} + (1 − X_k)(1 − exp(−tanh(g Σ_l W_kl(r) X_l)₊ dt)), with W = E^PW − r I^PW. Use M = the top 30 modules plus the lumped
rest, for P0 and P1.

**Grid.** g at 160 ln points in [0.12, 30]; r from 0 to 6 in steps of 0.25 (25 values).

**Starts.**
- all active;
- each module alone (31);
- each pair among the top 8 (28);
- 200 random subsets (seeded).
- Two passes: exact zeros on the silent modules, and a one-neuron seed 1/n_k on them (as in `theory_module2.json` coarse_2module).

**Procedure.**
1. Iterate to |ΔX|∞ < 1e-10 (at most 2·10⁵ steps).
2. Deduplicate attractors (L∞ < 1e-6) and test stability with the exact Jacobian eigenvalues (spectral radius < 1 − 1e-9).
3. Label each attractor by its active set A = {k : X_k > 1e-3}: SL (∅), B_m (|A| = m, with identities), SH (all modules).
4. Continue each r upward and downward in g from the previous attractor. That classifies the boundaries:
   - second order: X_k grows continuously from 0 (transcritical);
   - first order: a branch ends in a fold with a finite jump.

**Outputs.**
- `results/pd3/modules/coarse_mf_{P}.npz`: attractor list per (g, r).
- `figs/pd3/modules_coarse_phase.png`, a de Candia-style diagram of the closure:
  - regions coloured by attractor set and multistability count;
  - solid second-order lines and dashed first-order lines;
  - an inset placing our (g, r) rays in de Candia's (w0, w1) plane for the AL pair and AL↔rest (from `theory_module2.json` PW matrices).
- Purpose: show what de Candia's theory would predict for this connectome. It generalises the AL-pair result to 30 modules.

### 3.3 MT3: neuron-level QMF of the full network from many starts, with continuation (GPU array `pd3-qmf`)

This is the decisive deterministic test. Script `code/pd3_qmf_ic.py`.

**Map.** The exact map of `theory_qmf.py`: ρ_i ← ρ_i e^{−dt} + (1 − ρ_i)(1 − exp(−tanh(g Σ_j W̃_ij(r) ρ_j + h)₊ dt)), with h = 0, on all
167,216 neurons.

**Tasks.** One per r ∈ {0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6}, giving 9 tasks, `-t 03:00:00`.

**Pass 1: many starts.** g at 48 ln points in [0.15, 30] × 32 starts = 1,536 columns:
1. all active;
2. AL module (526) only;
3. AL-L only;
4. AL-R only;
5. rest only (all neurons outside the AL);
6–25. each of P0 islands 1–20 alone;
26–29. four random half-subsets of the 34 P0 modules (seeded);
30. AL + island 1;
31. uniform ρ = 1e-3;
32. AL at ρ = 0.05.

Procedure:
- Iterate at most 25,000 steps (5,000 model time). Test convergence every 500 steps: max |Δρ_i| per step < 1e-9. Store the residual of
  columns that have not converged; critical slowing near g_lin is expected.
- At the end, for each column store:
  - global ρ̄, the 34 per-module means, PR/N, AL share and the residual;
  - the Jacobian spectral radius ρ(J) from 300 power iterations of
    Jv = (e^{−dt} − p_on)⊙v + (1 − ρ)⊙p_on′(s)⊙(g W̃ v), with spmm on all columns at once. Here
    p_on′(s) = dt·sech²(s)·e^{−tanh(s)·dt} for s > 0 and 0 otherwise. ρ(J) is the norm growth rate; complex leading pairs are allowed.
- Save the final ρ vectors of the all-active start (float16, 48 × N) for downstream use.

**Pass 2: continuation** (same task).
- Up-sweep: g over 192 ln points in [0.15, 30], 3,000 steps per point. At each point add ε = 1e-6 to the previous state, which escapes the
  absorbing silent state.
- Down-sweep: from the all-active fixed point at g = 30, with no ε.
- Store ρ_up(g), ρ_down(g), PR and X_k on both branches.

**Cost estimate.** 1,536 columns at about 70 ms per step (scaled from 11 ms at B = 256) gives about 30 min for pass 1. Pass 2 has 1–2 columns
at about 2 ms per step × 576k steps per direction, about 20 min.

**Decision criteria.**
- **MT3-multi.** Neuron-level multistability holds at (g, r) if two starts converge (residual < 1e-7) to fixed points with ρ(J) < 1 whose
  34-module activity vectors differ by > 10% (relative L1). Example: an AL-L-only state with AL-R < 10% of AL-L, which would be a B1 state.
- **MT3-order.**
  - First order: a hysteresis interval in which ρ_up and ρ_down differ by > 5% on ≥ 2 consecutive g points, with a jump, and ρ(J) → 1 at
    the end of the branch.
  - Otherwise every boundary is continuous: a transcritical point or a crossover.
- **Prediction** (theory audit): no multistability and no hysteresis at any r. All non-silent starts converge to the all-active branch.
  Any positive finding is passed to multistability M3b for a stochastic test.

**Figure** `figs/pd3/modules_qmf_phase.png`, per r:
- per-module X_k(g) on the up and down branches (heat strip, modules ordered by g_lin,k);
- a map over (g, r) of the number of distinct attractors;
- ρ(J) against g, showing critical slowing at g_lin and at any fold.

### 3.4 MT4: module occupancy map from existing simulations (CPU, no GPU)

Script `code/pd3_module_occ.py`.

**Data.**
- From `occ[g, N]` of every T = 3000 real run (`results/pd/decay_r*_real.npz`, `results/pd2/onset_real_r*.npz`, `deloc_real_r*.npz`),
  compute the per-module mean occupancy X_k^sim(g, r) for P0 and P1. This needs no new runs.
- The same quantity from MT3 (QMF, all-active start).

**Activation points.**
- In the coupled system, module k switches on (imperfectly) at the maximum of d ln X_k/d ln g, in both simulation and QMF.
- The isolated threshold g_lin,k (MT1) is drawn as a vertical reference. In the exact theory each module's own onset is r-independent.

**Map.** The "B_m with identities" map: over (g, r), the number of modules with X_k^sim ≥ 1% and ≥ 10%, labelled with the first modules to
activate.

**Outputs.** `results/pd3/modules/module_occ_{P}.json` and `figs/pd3/modules_sim_occupancy.png`.

### 3.5 MT5 (optional, if GPU time allows): is any island "a second AL"?

- Run each of P0 islands 1–5 alone (module_sim convention: internal edges only, full-brain c), r ∈ {0, 1}.
- g at 24 ln points in [1, 4] × g_lin,k; start fully active; B = 64; T = 3000. That is 10 GPU tasks of a few minutes each.
- Output: each island's own finite-K lifetime curve. Several long-lived islands would be the ingredient a Griffiths regime needs
  (Cota et al. 2018). One dominant island confirms the single-rare-region picture.

**Deliverable to synthesis.** The module skeleton of the final diagram:
- SL edge at g_lin;
- B_AL (L);
- island-activation lines, dotted and imperfect;
- D;
- any first-order line confirmed by MT3 and multistability.

It is shown beside the closure's diagram (MT2), so the reader sees exactly where de Candia's picture holds and where it fails.

---------------------------------------------------------------------------------------------------------------------

## 4. Program `criticality`: avalanches, quasi-stationary order parameter, field scaling, lifetime vs K

### 4.1 C0: zero-cost look at existing driven runs (CPU)

- Files: `results/pd2/drive_h{1e-5,1e-4,1e-3,1e-2,1e-1}_real.npz` (r = 1, B = 32, T = 1000, 40 g in 0.1–3).
- Their g grid contains 0.1417, 0.1547 and 0.1688, which bracket g_lin.
- Compute the module occupancy Σ_{i∈AL} occ_i against h at these g, and its local slope d ln n_AL/d ln h. This is a first look at the
  field exponent before C3.

### 4.2 C1: new avalanche simulator `code/pd3_avalanche.py` (GPU)

**Algorithm.** An event-sparse ("frontier") synchronous update with exactly swc.py's rules.
- The graph is stored by presynaptic neuron: `out_ptr`, `out_tgt` int32, `out_coef` float32 = c·w·σ.
- B slots each run an independent avalanche. One step:
  1. Expand the out-edges of all active (neuron, slot) pairs (`repeat_interleave`). Chunk the expansion to ≤ 3·10⁷ edges and merge the
     chunk sums.
  2. Segment-sum the inputs per key = slot·N + target (`torch.unique(return_inverse)` + `index_add_`), giving s = g·Σ out_coef.
  3. Quiescent targets with s > 0 switch on with probability 1 − exp(−tanh(s)·dt). Membership in the active set is tested with
     `searchsorted` on the sorted active keys. Active pairs survive with probability e^{−dt}. Every probability uses the old state, as
     in a synchronous update.
  4. A slot with no active neuron ends its avalanche. A slot whose age reaches the cap is closed as censored. Either way it is re-seeded at
     once, so slot recycling keeps the GPU busy and there is no batch-waiting bias.
- h = 0 only, because the frontier method needs an absorbing silent state.

**Seeds.** One neuron per avalanche, drawn uniformly from the 366 excitatory neurons of the 526-module. This follows de Candia's "one
neuron of one of the modules chosen randomly". The seed id is saved, so seeds can be reweighted later, e.g. ∝ u_i.
- Nulls and sub-module graphs: uniform over excitatory neurons with out-degree > 0, restricted to the 99% support of the leading mode when
  that mode is localized.

**Definitions** (as in de Candia).
- Size S = number of 0→1 transitions, including the seed.
- Duration T = dt × (number of steps until the step at which the last active neuron turns off).
- Profile s(t) = spikes per step.

**Saved per avalanche.** S, T in steps, the censored flag, the seed id, spikes in AL-L / AL-R / AL-mid / rest (P0), and the maximum
number active.

**Online aggregates.**
- (a) Σ over avalanches of n_active(age) and spikes(age), for age < cap. Extinct avalanches count as 0, so this gives the mean growth
  E[n(age)].
- (b) Rescaled profiles in log duration bins (ratio 1.15) from 5 steps to the cap:
  - the profile interpolated on 40 points of t/T (sum, sum of squares, count);
  - the per-avalanche skewness, i.e. the third standardised moment of time weighted by s(t). **Positive skewness = fast rise and slow
    decay = de Candia's "leftward / left-skewed".** Store its mean and SE per bin.
- (c) Exact-duration mean profiles for T ∈ {10, 20, 40, 80, 160, 320, 640} steps, total and split L / R / rest (the analogue of de Candia
  Fig. 7).
- (d) 2,000 sampled raw avalanches with T ≥ 50 steps, with per-step spikes split L-AL / R-AL / rest.

**Defaults.**
- B = 16,384 slots; cap 5,000 steps (1,000 model time).
- Stop after N_av avalanches or 110 min of wall time; checkpoint every 10 min.
- For x = g/g_lin ≥ 1.2: N_av = 2·10⁵ and cap 2,500 steps (many seeds establish there).

**Validation V0 (before production; 2 tasks).**
- Full brain, r = 1, g ∈ {0.20, 0.36}. Run pd3_avalanche (cap 3,000) and a dense reference: swc.py `--exp spread`, B = 512, with a
  `--seed_set` npz of the 366 excitatory module neurons, uniform p (written under `results/pd3/aval/`).
- Pass if:
  - the survival curves P(t) agree within 2 SE at every recorded t (KS on extinction times p > 0.01);
  - the mean number active, E[n(t)] over all replicas with dead ones counted as 0, agrees within 2 SE at every recorded t. swc.py
    stores ρ per replica on its log grid, not avalanche sizes, so sizes are not compared.
- Also log the timing per step and per avalanche.
- `spreading.md` P(100) = 0.486 [0.441, 0.531] at g = 0.36 used seeds ∝ |v|² over the top 1,000 neurons. It is a plausibility check
  only, not a pass criterion.

**Production array `pd3-aval`.** x = g/g_lin with g_lin = 0.1632; 35 tasks, `%12`, `-t 02:00:00`.

| graph | r | x | tasks | N_av |
|---|---|---|---|---|
| real full brain | 0 | 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20, 1.35, 1.50 | 10 | 2·10⁶ (x ≤ 1.05), 5·10⁵ (1.10), 2·10⁵ (≥ 1.2) |
| real full brain | 1 | same 10 | 10 | same |
| real full brain | 0.5, 2, 4, 6 | 1.00 | 4 | 2·10⁶ |
| real full brain | 6 | 0.98, 1.02 | 2 | 2·10⁶ |
| AL module alone (526, `module_graph.npz`) | 0, 1 | 1.00 | 2 | 2·10⁶ |
| configuration null (full, seed 0; g_lin 0.8287) | 0 | 0.98, 1.00, 1.02 | 3 | 10⁶ |
| configuration null | 1 | 1.00 | 1 | 10⁶ |
| FSS cnull seed 1, N ∈ {5k, 20k, 80k}, own g_lin | 0 | 1.00 | 3 | 10⁶ |

**Analysis** `code/pd3_aval_analyze.py` (CPU).
- Size and duration distributions:
  - log-binned P(S) and P(T), compensated S^{3/2}P(S) and T²P(T);
  - discrete power-law MLE exponents (Clauset et al.) in [S_min, S_c/3] and [T_min, T_c/3];
  - fits are secondary. The primary comparison is the exact chain distribution at K ∈ {37, 54.7, 75.8} (`theory_bd.json` aval_R1),
    plus a one-parameter chain fit of K to P(T).
- ⟨S⟩(T) and its local slope. Predicted peak: 1.55 at K = 37 and 1.66 at K = 75.8. The null (K ~ 5,000) should come closer to 2;
  the chain gives 1.91 at K = 2000.
- Empirical critical point:
  - γ(x) = slope of ln E[n_active(age)] over ages 5–50 steps;
  - x* = the zero of γ(x), with a bootstrap CI.
  - Exact theory: x* = 1 for every r. Under the net-input closure the r = 6 runs would be strongly subcritical at g_lin (theory T1).
- Shapes:
  - fixed-T mean profiles;
  - collapse ⟨s(t/T)⟩·T^{1−γ_ST}, with γ_ST from the ⟨S⟩(T) fit;
  - mean skewness against T, with a bootstrap CI.
  - Compare real full brain, module alone, configuration null and the chain (0 ± 0.02, `theory_bd2.json`) against de Candia (left skew up
    to ~0.4 at T ≈ 400 ms = 40/α along SL–B1).
- Recruitment: the L-AL / R-AL / rest spike shares against age inside long avalanches.
  - de Candia's mechanism (cooperative recruitment, then competition) would show the minority lobe's share falling late.
  - The neuron-level theory predicts both lobes co-recruited.

**Figures** (`figs/pd3/`).
- `aval_size_duration.png`: real r = 0 and 1 at x = 1, the null, and chain curves.
- `aval_ST_slope.png`.
- `aval_critical_point.png`: γ(x) for r = 0, 1, 6, with x* marked.
- `aval_shapes.png`: collapse, plus skewness against T for real, module, null and chain.
- `aval_examples.png`: L / R / rest traces.

### 4.3 C2: quasi-stationary order parameter and susceptibility

Script `code/pd3_qs.py` (GPU; dense swc-like; full brain, module or FSS graph).

**QS method** (de Oliveira & Dickman 2005).
- Each of B = 256 replicas keeps M = 16 stored configurations (uint8 [N, B, M] on the GPU, ~0.7 GB for the full brain).
- Every 25 steps, a random stored slot of each live replica is overwritten with probability 0.5.
- On absorption, the replica is reset to a random stored configuration, and the event is counted.
- The store is filled during the first 400 steps from a fully active start. Discard t < 1,000 model time; measure over 4,000 model time
  (20,000 steps).

**Observables.** Every 5 steps:
- n_total, n_AL, n_AL-L, n_AL-R;
- ⟨ρ⟩ and ⟨n_AL⟩;
- χ_QS = N(⟨ρ²⟩ − ⟨ρ⟩²)/⟨ρ⟩ and the module χ = var(n_AL)/⟨n_AL⟩;
- τ_QS = model time / number of absorption events;
- a histogram of n_AL (unimodal or bimodal);
- PR of the occupancy.

**Grid.**
- Real full brain, r ∈ {0, 1, 6}, x at 24 ln points in [0.95, 3.5]. This covers g_lin, the predicted χ peaks (0.195 at r = 0, 0.209 at
  r = 1), the τ = 200 / 3000 crossovers and the L region. 6 x per task gives 12 tasks.
- Module alone (526), r ∈ {0, 1}, same x: 2 tasks.
- Cost estimate: ≈ 11 ms per step at B = 256 × 25,000 steps ≈ 5 min per x, about 30 min per task.

**Analysis.**
- n_AL,QS against R − 1, with R(g) from the dense branch rows.
  - Compare without new fitting with the exact chain QS mean at K = 75.8 / 36.9 / 16.2 (r = 0 / 1 / 6), and with the mean-field
    amplitudes 115 / 35 / 18.5 × (g/g_lin − 1) (`theory_notes.md` §2.1, T10).
  - Order-parameter exponent: the slope of ln n_QS against ln(R − 1) over x_K = (R − 1)√K ∈ [2, 8]. MF-DP gives 1, with large
    corrections at this K.
- χ peak location and height against the predictions: g 0.195 / 0.209; heights 3.9 / 2.5 in chain units.
- A bimodal histogram would be a first-order signature. None is expected.

**Figures.** `figs/pd3/qs_order_parameter.png` and `qs_susceptibility.png`.

### 4.4 C3: field scaling at g_lin

`pd3_qs.py --h` mode (no reactivation; h > 0).
- Real full brain, r ∈ {0, 1}, x ∈ {0.90, 1.00, 1.10}.
- h ∈ {1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3}, run in sequence within one task. Each h starts silent: burn-in 500, then
  measure 2,000 model time, B = 64. That gives 6 tasks of about 20 min.

Observable: n_AL(h) and its local slope d ln n_AL/d ln h.

MF-DP predictions:

| x | expected slope |
|---|---|
| 1.00 | 1/2 (δ_h = 2), but only in the window √K ≲ n_AL ≲ K, under one decade at K ≈ 50–76 |
| 0.90 | 1 (linear response) |
| 1.10 | → 0 at small h |

Compare with the exact driven chain. Its birth rate is λn(1 − n/K) + h_eff(K − n) and its death rate αn; solve its stationary
distribution on CPU at the K_needed values. A tricritical point would give slope 1/3 (δ_h = 3).

**Figure.** `figs/pd3/field_scaling.png`.

### 4.5 C4: lifetime at g_lin against module size

Script `code/pd3_submodule.py` (GPU, 2 tasks: r = 0 and r = 1).

**Nested AL sub-modules.**
- The top-n neurons ranked by P[:,0] + P[:,1] (`tail_real_r0_dir.npz`), n ∈ {50, 75, 100, 150, 200, 300, 400, 526, 800, 1200}.
- Induced internal subgraph, full-brain c (module_sim convention).

**Per sub-module.**
- Exact g_lin,n and K_n = 1/(⟨u²,v⟩⟨u,v²⟩) at g_lin,n.
- Module-alone stochastic decay from fully active at x = g/g_lin,n ∈ {1.0, 1.2, 1.5, 2.0}, B = 4,096, T = 3,000, with the exact extinction
  step per replica.

**Analysis.**
- τ_top(x = 1) against K_n, with the prediction 1.90·K^0.503/α_eff (12.0 / 14.8 / 17.7 model time at K = 37 / 54.7 / 75.8).
- Merge with the FSS nulls (§2.2) on one log–log plot, `figs/pd3/lifetime_vs_K.png`. This is the √K collapse across real sub-modules and
  nulls (K ≈ 10 to 5,500).
- ln τ against K at x = 1.5 and 2 (exponential in K·Φ(R)).

### 4.6 Decision criteria for Q2 (pre-registered)

**Criteria.**
- **Q2-a (location).**
  - The avalanche growth-rate zero x* lies in [0.97, 1.03] for r = 0 and r = 1, and also for r = 6 (r-independence).
  - The χ_QS peak lies within ±10% of the chain-predicted g (0.195 at r = 0, 0.209 at r = 1).
- **Q2-b (finite-K MF-DP).**
  - P(T) at x = 1 matches the chain with one K: compensated deviations ≤ 10% up to T_c, or χ²/dof ≤ 2.
  - That K agrees with K_needed (75.8 / 36.9) within a factor 1.5.
  - The ⟨S⟩(T) local-slope peak is 1.5–1.75 on the real graph and rises toward 2 with K across module, real and nulls.
- **Q2-c (√K rounding).** τ_top(x = 1) ∝ K^{0.5 ± 0.1} over nested sub-modules and nulls.
- **Q2-d (exponents).**
  - n_QS agrees with the chain at the same K within 20%.
  - The field slope at x = 1 dips to ≤ 0.65 in the window, while at x = 0.9 it stays ≥ 0.85.
- **Q2-e (not first order).** QS histograms are unimodal everywhere, and multistability finds no hysteresis at the onset.

**Verdicts.**

| outcome | verdict |
|---|---|
| a–e all pass | "Finite-K-rounded continuous (MF-DP) transition at g_lin of a ~50-unit module. The steep crossover is its extinction-time image (1.6–2.1 × g_lin), drifts with T and does not sharpen with N. In the strict sense it is not a phase transition of the brain." |
| a fails | The critical point is not at g_lin, so theory P1 is wrong. |
| b–d fail, with bimodality or hysteresis | First-order or tricritical. |
| c fails | The rounding is not set by K. |

**Shapes.**

| observed skewness | reading |
|---|---|
| 0 within 2 SE at all T | One-mode dynamics. |
| negative (right skew) | Intra-module E/I (de Candia's Δ_k). |
| positive (left skew) | Contradicts the no-competition result, unless the configuration null shows the same skew, which would make it a degree or weight heterogeneity effect (Gleeson & Durrett 2017). |

---------------------------------------------------------------------------------------------------------------------

## 5. Program `multistability`: coexisting states, broken symmetry, hysteresis (stochastic)

### 5.1 Simulator

Script `code/pd3_multistab.py` (GPU; dense swc.py update with the same rules).
- **Starts.** Defined by module sets of P0 (listed modules fully active, the rest silent).
- **Schedules.** Adiabatic: a list of (g, dwell) or (r, dwell), with the state carried over.
- **Drive.** Optional h.
- **Recording.**
  - per-module active counts every 1 model time (int32 [rec, 34, B]);
  - per-replica ρ, n_AL, n_L, n_R and PR from per-replica occupancy accumulated over the last half of each dwell (float32 [N, B];
    21 MB at B = 32);
  - a bit-packed final snapshot per replica at the end of each dwell or start, so P1 (Leiden) can be applied post hoc.
- **Validation.** All-active start at r = 1, g ∈ {0.30, 0.344, 0.40}, B = 64, T = 3000. The τ_sim must fall inside the 95% CI of the
  pooled `phase_complete` points.

### 5.2 M1: left/right AL symmetry (the decisive B1 test)

**Starts.** AL-L only (248 active), AL-R only (275), both AL (526), all active.

**Grid.**
- r ∈ {0, 0.5, 1, 2, 3, 4, 6}, with g = g_on(r) × {1.0, 1.25, 1.6, 2.5, 4, 8}. g_on(r) is the simulated onset (`flow_sim_boundaries.json`):
  0.266, 0.306, 0.344, 0.402, 0.456, 0.508 and 0.579.
- Added points in the closure's predicted B1 windows: r = 4 at g ∈ {0.29, 0.32, 0.40}, and r = 6 at g ∈ {0.43, 0.58, 0.70}
  (`theory_module2.json` coarse_2module).
- B = 64, T = 1000, h = 0. 7 tasks, one per r, about 20 min each (estimate).

**Observables.**
- f_L = n_L/248, f_R = n_R/275, and the asymmetry A = (f_L − f_R)/(f_L + f_R).
- t_sym per replica: the first time |A| < 0.3 after starting from |A| = 1.
- The late (t ∈ [500, 1000]) distribution of per-replica mean A among survivors, compared across starts (two-sample KS).

**Decision criteria.**
- **Broken symmetry (B1-like)** at a (g, r) requires all three:
  - ≥ 25% of surviving L-only replicas keep mean A > 0.6 over [500, 1000];
  - the late A distributions of the L-only and R-only starts differ (KS p < 0.001);
  - the all-active start gives a bimodal A distribution.
- **Predictions.**
  - Neuron-level theory: t_sym of the order of the module relaxation time (tens of model time) and identical late distributions everywhere.
  - de Candia closure: persistent one-lobe states for r ≥ 2.5 (at r = 4 from g 0.287).

### 5.3 M2: adiabatic sweeps (hysteresis = first-order test)

**M2a, sweeps in g.**
- r ∈ {0, 0.5, 1, 2, 4, 6}; g at 40 ln points in [0.15, 30].
- Up-sweep: from silence, with h ∈ {1e-5, 1e-4}. The drive keeps the absorbing state from trapping the up branch; 1e-5 is the
  driven-plane value.
- Down-sweep: from fully active at g = 30, with the same h.
- Dwell D ∈ {50, 200, 800} model time per point. Average ρ, n_AL, X_k and PR over the last half of each dwell. B = 32.
- Also a down-sweep at h = 0, D = 200. Its extinction point is compared with the τ = D crossover from the lifetime map.
- Tasks: one task holds the up and down sweeps of one (r, h, D), giving 36 tasks plus 6 h = 0 tasks = 42.
- Cost estimate: ≤ 160,000 steps × ~1.5 ms ≈ 4 min per sweep at D = 800.

**M2c, sweeps in r** (de Candia's lines are crossed by changing w1, i.e. r).
- g ∈ {0.6, 2, 8}; r from 0 to 6 and back over 25 points; D = 200; h = 1e-5; B = 32. 3 tasks.

**Loop measures.**
- A_ρ = ∫|ln ρ_up − ln ρ_down| d ln g and A_PR = ∫|PR_up − PR_down|/N d ln g (in r for M2c), with a replica bootstrap.
- How the loop area scales with D.

**Decision criteria.**
- **Hysteresis candidate:** the loop area at D = 800 exceeds 3 bootstrap SE and shrinks by < 40% from D = 200 to 800. Continuous
  transitions give loops that shrink as the dwell grows.
- **M2b confirmation:** at the loop centre, run 64 replicas from the up-branch end-state and 64 from the down-branch end-state, T = 5000.
  - If their per-replica ρ / PR distributions remain distinct (KS p < 0.001), it is coexistence: a first-order line, drawn dash-dot on the
    final diagram.
  - Otherwise it is a crossover.

### 5.4 M3: module-subset starts (coexisting attractors with different active sets)

**Grid.**
- r ∈ {0, 1, 2, 4}; g ∈ {0.4, 0.7, 1.2, 2.5, 6, 15}. At r = 0 this spans L, the island cascade 0.48–1.5 and D.
- 13 starts:
  - all active;
  - AL only;
  - rest only;
  - each of P0 islands 1–5 alone;
  - AL + islands 1–5;
  - 3 random half-subsets (seeded).
- B = 32, T = 2000, h = 0. 24 tasks, one per (r, g), about 5–10 min each (estimate).

**M3b.** Any (g, r, start) at which MT3 found QMF multistability is added as a task with B = 64 and T = 5000.

**Analysis.**
- Take the per-module activity vectors of surviving replicas over t ∈ [1000, 2000], for each start.
- Cluster across starts. A distinct attractor has relative L1 distance > 0.2 and replica spread < 0.05.
- Record which starts die. h = 0 is absorbing, so "rest only" at low g can die.

**Decision criteria.**
- **Multistability:** ≥ 2 distinct non-silent clusters, each holding ≥ 50% of its start's replicas at T = 2000.
- **Prediction:** a single non-silent attractor. Rest-only and island-only starts should ignite the AL through excitatory projections,
  since r₁*(AL←rest) = 9.6 (`theory_module.json` B_AL_rest).

### 5.5 Outputs

- `results/pd3/multistab/{al_sym_r*.npz, sweep_*.npz, rsweep_g*.npz, ic_r*_g*.npz, summary.json}`.
- `figs/pd3/multistab_AL_symmetry.png`: A(t) traces, t_sym against r, late A histograms per start, with the closure's B1 windows shaded.
- `figs/pd3/multistab_hysteresis.png`: up/down ρ and PR against g (and r), and loop area against D.
- `figs/pd3/multistab_ic_attractors.png`: late module-activity heat strips per start.

---------------------------------------------------------------------------------------------------------------------

## 6. Order of work, dependencies, budget

| step | depends on | GPU tasks | wall (estimate, shared GPUs, nice 500) |
|---|---|---|---|
| `pd3_common.py`, `pd3_array.sbatch`, `pd3_partition_v0.py` (P0) | — | 0 | 30 min |
| redraw A (heat map) | — | 0 | 1 h |
| fss graph builder + validation (theory_nullK reproduction) | common | 0 (or 1) | 30 min |
| pd3-fss (54) | builder | 54 | 1.5–3 h |
| MT1 P1 Leiden (CPU) + quotient / dead-zone | P0 | 0 | 1–1.5 h |
| MT2 coarse closure (CPU) | MT1 | 0 | 30 min |
| pd3-qmf MT3 (9) | P0 | 9 | 1–2 h |
| MT4 occupancy map (CPU) | P0, MT3 | 0 | 30 min |
| pd3-aval V0 (2) then production (35) | P0 | 37 | 2–3 h |
| pd3-qs C2 (14) + C3 (6) | — | 20 | 1–2 h |
| pd3-sub C4 (2) | common | 2 | 30 min |
| pd3-ms M1 (7), M2 (45), M3 (24) (+ M2b, M3b) | P0, MT3 for M3b | 76 + | 2–3 h |

- In total about 200 GPU tasks, most under 30 min. Submit each program's array with a concurrency cap (`%12`–`%16`).
- While a program waits on GPUs, it writes and tests its analysis scripts on partial outputs. Poll with until-loops of ≤ 9 min.
- **Preemption.** Every task writes its output atomically (to a temporary file, then rename). The resubmission script only reruns array
  indices whose output file is missing.

## 7. The final diagram (input to synthesis)

`figs/pd3/phase_diagram_v3.png`:
- **Background:** the lifetime heat map (§2.1, panel A), with the PR/N map in the sustained region (panel B).
- **Lines:**
  - g_lin, **solid**: the second-order MF-DP critical line, i.e. the SL edge; finite-K rounded; r-independent.
  - τ = 200, τ = 3000 and PR = N/4, **dashed**, labelled "crossover".
  - Island activation points, **dotted**: imperfect onsets from MT3 and MT4.
  - First-order lines, **dash-dot**: only where M2b confirms coexistence.
  - de Candia-closure lines in a separate panel (MT2) for contrast.
- **Region labels:**
  - SL (silent);
  - "rounded critical band" (g_lin < g < τ = 3000; finite-K MF-DP, where criticality applies);
  - B_AL (L, "B1-like": the AL module active, protected by heterogeneity);
  - the B_{AL+islands} cascade;
  - D (global, SH-like).
- **Side panels:**
  - FSS: (x_on − 1) against N for nulls against the real network;
  - lifetime against K (√K collapse);
  - avalanche P(T) compensated at g_lin with the chain;
  - skewness against T;
  - AL-symmetry test.

## 8. Risks and fallbacks

| risk | fallback |
|---|---|
| Frontier simulator too slow when activity is high (x ≥ 1.2, nulls) | Lower B and the cap, chunk the expansion. Supercritical shapes are secondary. |
| Leiden install fails | networkx Louvain on the thresholded excitatory graph. P0 alone is enough for every simulation. |
| QMF convergence near g_lin is slow (critical slowing) | Report residuals and mark unconverged columns. Continuation supplies the branch. |
| Small-N nulls have no clean onset (hub-star modes, ×2 K scatter) | Report seed by seed and use the per-realisation K_fit rather than N as the abscissa. |
