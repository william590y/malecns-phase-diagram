# Program module_theory (MT1-MT5): connectome modules, de Candia's closure, and the neuron-level mean field from many starts

2026-11-09, workflow wf_98fb990d-4fb, plan `results/pd3/plan.md` section 3. Every number below is read from a file named next to it
(all of them are also collected, with the file they came from, in `results/pd3/module_theory.json`, built by
`code/pd3_module_theory_summary.py`). Job ids, failures and reruns: `results/pd3/module_theory/JOBS.md`.

## 0. Bottom line

| question | what this program shows | strength |
|---|---|---|
| Q1: a de Candia-style diagram (SL / Bm with identities / SH; 2nd- vs 1st-order lines)? | **de Candia's own closure, applied to the connectome's modules, does give one** (MT2): an SL edge that is second order and moves with r, B_m states with named modules (the AL module first, then a cascade), coexisting B1 states of the left and right AL for r >= 2.5, first-order (hysteresis) intervals, and at strong coupling a region with no stable fixed point [verification: was "no stationary state"]. **The exact neuron-level mean field of the whole brain (MT3) does not reproduce the lobe-level broken symmetry** (no B1 state of the antennal lobe, AL, at any of 9 r values), but it is **not** the single-attractor, purely continuous picture predicted either: it has coexisting stable fixed points at r = 0.5 and 1.5-6 with the pre-registered 10% criterion (none at r = 1), and at every r >= 0.25 with a 1% criterion (only r = 0 has a single attractor) [verification]; coexisting states differ in one connected region of 8-9,274 neurons (VNC/central circuits at r = 0.5-2; subsets of AL local neurons, giving coexisting left- and right-tilted AL states, at r = 3-6), fixed-point hysteresis at r = 0.5, and for r >= 0.5 (dt-robust) a non-stationary (oscillating) mean field above g ~ 1-2, carried by 26k-43k neurons of which optic-lobe intrinsic neurons are the largest group (43-89%) [verification]. | closure: firm for P0 (validated against theory_module2, 1525/1525 labels); P1's stored SL edge was wrong for r >= 1.25 and is corrected in section 3.2 [verification]. QMF findings: firm as statements about the mean field (converged to residual < 1e-7 with estimated distance < 1e-6; 95-100% of the counted states below 1e-9 [verification]; distinct states checked neuron by neuron). Whether any of them survives noise in the stochastic model is NOT tested here (handed to multistability M3b). |
| Q2: is the steep, smooth crossover second order? | At the AL onset the neuron-level mean field is continuous at every r: one attractor, up- and down-sweeps identical, the silent state loses stability transcritically (QMF SL edge, r-dependent, 0.159-0.198). All multistability and hysteresis found lie at g >= 0.33 (r = 6) and otherwise at g >= 0.65 (0.73 with the 10% criterion), i.e. above the QMF onset; they are also above the simulated onsets except at r = 6, where QMF multistability starts at g = 0.33, below the simulated tau = 200 and tau = 3000 lines (0.436 and 0.579, `phase_complete.json`) [verification]. At r = 0 the whole diagram is continuous. Coupled-system module activation (MT4) is smeared: at r = 0 every island turns on below its own isolated threshold, driven by the active AL; at r >= 1 the islands with low isolated thresholds (isl1-3, isl5, isl12, isl28) reach 1% only near or above that excitation-only threshold [verification]. | is consistent with a continuous onset at mean-field level, but cannot tell a finite-size-rounded transition from a smooth crossover of a finite module; does not by itself test the finite-size statements (criticality / redraw_fss programs; criticality.md rejects the MF-DP class). |
| Q3 (accepted offers: heat map, finite-size check) | not part of this program (redraw_fss). | — |

**Griffiths ingredients (MT5).** Alone, island 1 (2,292 neurons) and island 5 (221) are AL-like long-lived regions at r = 0 (sustained at
1.46 and 1.62 times their own threshold, g ~ 0.97 and 1.16), but at r = 1 only the AL is (none of islands 1-5 reaches tau = 3000 up
to 4 times its own threshold). Among the AL and islands 1-5 (the only modules simulated alone; islands 6-30 and the rest were not) there are
one to three long-lived regions, not a broad distribution of them [verification: scope stated].

## 1. What was run

| step | code | compute | output |
|---|---|---|---|
| MT1 P0 partition | `code/pd3_partition_v0.py` | local CPU, 4 s | `results/pd3/modules/partition_P0.npz` (+ `.log`) |
| MT1 P1 Leiden (+ P2) | `code/pd3_modules.py` | CPU job 175284 (thickstun), 44 min | `partition_P1.npz`, `partition_P2.npz`, `leiden_runs.json` |
| MT1 items 1-5 | `code/pd3_module_stats.py P0/P1/P2` | local (P0), CPU jobs 176327/176328 | `quotient_<P>.npz`, `deadzone_<P>.json`, `modules_table.json` |
| MT2 closure | `code/pd3_coarse_mf.py P0/P1` (+ `pd3_coarse_post.py`) | GPU jobs 176386 (P0), 176402 (P1), on `-p ellis` | `coarse_mf_<P>.{npz,json}`, `coarse_post_<P>.{npz,json}` |
| MT3 QMF | `code/pd3_qmf_ic.py --r R` | GPU array 175683 (8 tasks) + 176342 (r = 0.25 rerun) | `results/pd3/module_theory/qmf_r<R>.npz` |
| MT3 analysis | `code/pd3_qmf_analyze.py` | CPU | `module_theory/qmf_analysis.json` |
| MT3b (added) | `code/pd3_qmf_multi.py` | GPU job 177888 | `module_theory/qmf_multi.json`, `qmf_multi_vectors.npz` |
| MT4 | `code/pd3_module_occ.py P0` | CPU | `results/pd3/modules/module_occ_P0.json` |
| MT5 | `code/pd3_island_sim.py` | GPU array 175546 (12 tasks) | `module_theory/island_<m>_r<r>.npz`, `island_lifetimes.json` |
| figures | `pd3_modules_fig.py`, `pd3_coarse_fig.py`, `pd3_qmf_analyze.py`, `pd3_module_occ_fig.py`, `pd3_island_fig.py`, `pd3_skeleton_fig.py` | CPU | `figs/pd3/modules_*.png` |

Validation: the closure code reproduces the 2-module AL closure of `theory_module2.json` exactly (1525/1525 (g, r) labels,
`coarse_mf_validation.json`); the module statistics reproduce the AL-pair numbers of `theory_module2.json` (Perron-weighted
E_LR / E_RL = 0.080 / 0.066, I_LR / I_RL = 0.034 / 0.033; dead-zone shares R<-L 84% / 36% / 25% and L<-R 86% / 38% / 22% at
r = 0 / 1 / 6; `module_theory.json` MT1.P0.AL_PW, AL_pairs); MT5's thresholds equal MT1's to 1e-9; MT3b recomputed every
attractor and matched the pass-1 mean activity to <= 1.5e-8 relative (`qmf_multi.json` max_mean_check).

## 2. MT1: partitions and module statistics

### 2.1 P0 'islands-v0' (34 modules) - and a structural finding

AL-L 248, AL-R 275, AL-mid 3 (the 526-module split by side); skipped AL modes 0, 1, 3, 5 (both the 526-module share and the tail's
own AL field give the same four); islands 1-30 from the first 302 of the 400 modes; rest 158,745 (`partition_P0.log`). Island 1 is
mode 2's 90% support (2,292 neurons; one of its 2,293 neurons belongs to the AL module), 39% vnc_intrinsic and 25% cb_intrinsic.

**Deviation and finding.** The spec's merge rule ("merge modes whose supports overlap > 50%"), applied to the growing union of an
island's supports, absorbed all 395 later non-AL modes into island 1 (10,276 neurons; only 4 islands). The reason is structural:
**87% of the 395 non-AL top-400 modes have more than half of their 90% support inside mode 2's support** (median share 0.63;
`partition_P0.npz` meta). The top-400 excitatory modes are individually localized (1/IPR <= 650) but they are not disjoint rare
regions: most of them live on one VNC/central-brain region around mode 2. P0 therefore merges a mode only with the SEED mode of an
island when the overlap exceeds 50% of the larger support (conjugate pairs and near-duplicates), merged modes add no neurons, and
later islands are the parts of later modes not already assigned (first-come). Islands 2-30 are thus mostly fragments
(20-791 neurons) of that shared region. [verification: the 87% / 0.63 recomputed exactly; 99% of these 395 modes have > 50% of their
90% support inside the union of the supports of the modes before them.]

### 2.2 P1 (Leiden) and P2 (superclass x side)

Leiden on A = E + E^T (13,632,893 undirected excitatory edges), RBConfiguration, gamma {0.5, 1, 2, 4} x seeds {0, 1, 2}
(`leiden_runs.json`): modules with >= 50 neurons 5-7 / 9-11 / 24-28 / 50-53; mean seed-pair NMI 0.899 / 0.846 / 0.841 / 0.857;
only gamma 2 and 4 give >= 20 modules of >= 50 neurons, so gamma = 4 (seed 1, highest mean NMI to the other seeds) was chosen:
50 modules >= 50 neurons, largest 8,622, modularity (gamma = 1) 0.648, 1,886 singletons (neurons without excitatory edges).
NMI with P0 is 0.05 (P0 is 95% 'rest'); P2 has 70 groups, NMI(P2, P0) 0.067, NMI(P1, P2) 0.496.

### 2.3 Thresholds of the modules alone (top 30 by Lambda_k + lumped rest; `modules_table.json`)

| partition | module | n | Lambda_k | g_lin,k (alone) | K_k |
|---|---|---|---|---|---|
| P0 | AL-R | 275 | 5.464 | 0.1665 | 36.5 |
| P0 | AL-L | 248 | 5.392 | 0.1687 | 36.6 |
| P0 | isl1 | 2,292 | 1.385 | 0.665 | 103.5 |
| P0 | isl5 | 221 | 1.274 | 0.715 | 78.3 |
| P0 | isl3 | 234 | 1.296 | 0.754 | 7.7 |
| P0 | isl2 | 75 | 0.955 | 1.001 | 9.0 |
| P0 | lumped rest | 158,792 | 0.998 | 0.916 | 268.5 |
| P1 | L14 (contains the AL; 61% cb_sensory) | 4,483 | 5.546 | 0.1640 | 76.1 |
| P1 | L29 (VNC) | 2,054 | 1.713 | 0.543 | 11.2 |
| P1 | lumped rest | 95,233 | 0.495 | 1.854 | 92.5 |
| P2 | cb_intrinsic, R / L | 16,052 / 16,028 | 5.466 / 5.399 | 0.1664 / 0.1685 | 36.6 / 36.8 |
| P2 | vnc_intrinsic, L | 6,473 | 0.935 | 1.009 | 30.9 |

Top-30 modules with g_lin,k below 0.3 / 0.5 / 1 / 1.47 / 2 / 3 / 30: P0 2 / 2 / 5 / 7 / 7 / 10 / 26; P1 1 / 1 / 11 / 18 / 25 / 30 / 30.
In every partition one module (the AL, or the lobe halves) switches on alone near the whole-brain g_lin = 0.1632; everything else
needs g >= 0.54 alone. (P1's L14, which contains both lobes and their sensory inputs, has K = 76.1; the 2-lobe module of theory_bd
needed K = 75.8 at r = 0. This may be a coincidence and is not used anywhere.)

### 2.4 Is de Candia's reduction applicable? Equitability and Perron weighting

de Candia's reduction is exact when every neuron of module k receives the same input from module l (CV = 0). Median coefficient of
variation of sum_{j in l} c w_ij over i in k (`quotient_<P>.npz`): P0 excitatory self 1.23, cross 3.49; inhibitory self 1.64, cross
3.62; P1 1.87 / 8.57 / 1.37 / 7.30; P2 1.87 / 5.57 / 1.77 / 4.81 (AL-L <- AL-L 1.18, AL-L <- AL-R 2.62). The inputs within a module
vary by more than their mean: the closure (MT2) is an uncontrolled approximation for this connectome, which is why MT3 (no closure)
is the decisive test. [verification: medians recomputed exactly; the cross CVs are mostly sparsity - in the median coupled ordered
pair 74% of the target module's neurons receive no excitatory input from the source, and among receivers only the median excitatory
cross CV is 1.52.]

### 2.5 Dead-zone screen (where could neuron-level broken symmetry live?)

For each ordered pair l -> k: r1*_PW(k <- l) < r0*_PW(k) (the de Candia window) and the share of k's neurons with net-positive input
from l's Perron pattern at r in {0.5, 1, 2, 4, 6} (`deadzone_<P>.json`):

| | coupled pairs | window r1* < r0* | plain candidates (share < 5%) | strict (u-weighted and receiver shares < 5%) | meaningful | mutual meaningful pairs |
|---|---|---|---|---|---|---|
| P0 | 879 | 491 | 134 | 53 | 13 | 0 |
| P1 | 903 | 388 | 206 | 37 | 22 | 1 (L46 <-> rest) |
| P2 | 622 | 242 | 107 | 61 | 6 | 0 |

'Meaningful' (added, see section 9) also requires coupling >= 1e-3 Lambda_k, at least one receiver of the Perron pattern and both
modules able to switch on alone below g = 30. The plain criterion is passed trivially by sparsely coupled pairs, and most strict
candidates had a Perron-pattern drive of exactly 0. The AL pair is not a candidate (u-weighted net-positive share 0.39-0.41 at
r = 6). The meaningful candidates are one-directional suppressions of weak modules (P0: isl7, isl10, isl16 <- isl1; isl2 and
isl28 <- rest) or pairs inside inhibition-dominated modules (P1: r0* = 0.045-0.26 for L19, L46, L47, L48); none is a pair of two
strong modules inhibiting each other. Figure: `figs/pd3/modules_partition.png`.

## 3. MT2: de Candia's closure on connectome modules

Map X_k <- X_k e^{-dt} + (1 - X_k)(1 - exp(-tanh(g sum_l W_kl X_l)_+ dt)), W = E^PW - r I^PW, 31 modules (top 30 + rest), 160 g x 25 r,
520 starts per (g, r) (all, 31 singles, 28 top-8 pairs, 200 random subsets; zeros and one-neuron seeds), float64, up/down sweeps.
Raw output `coarse_mf_<P>.{npz,json}`; the corrected census and boundary classification (section 9, deviations 6-7) in
`coarse_post_<P>.json`.

### 3.1 P0 closure (AL-L, AL-R, 28 spectral islands, lumped rest)

- **SL edge** (second order, transcritical; g = alpha_eff / mu_+(W(r))): 0.1645 (r = 0), 0.1835 (1), 0.2071 (2), 0.2198 (2.5),
  0.2696 (4), 0.3863 (6) (`coarse_post_P0.json` g_SL_edge). It moves 2.35x with r; the exact process's edge g_lin does not.
- **r = 0** (single attractor in every cell): SL -> B2 (both AL lobes) at g = 0.170 -> + island 1 at 0.209 -> a continuous cascade
  (6 modules at 0.391, 11 at 0.419) -> all 31 modules above 1e-3 from g = 1.80 on (SH) [verification: was "at g = 30"]. No
  multistability, no hysteresis.
- **Broken AL symmetry, r >= 2.5**: states with only AL-L and only AL-R active coexist in 1,426 of the 4,000 cells, from
  g = 0.224 (r = 2.5), 0.286 (r = 4) and 0.434 (r = 6) upward (`module_theory.json` MT2.P0.corrected.AL_lobe_broken_first_g_by_r).
  This is theory_module2's 2-module result (B1L | B1R from 0.287 at r = 4, from 0.428 at r = 6) surviving in the 31-module closure.
- **Combinatorial multistability**: 2,484 of 4,000 (g, r) cells have >= 2 stable attractors; the maximum per cell is 1 (r = 0),
  38 (0.5), 55 (1), 14 (2), 98 (3), 103 (4), 133 (6), 155 overall. Islands behind dead zones switch on or off independently
  (e.g. r = 1, g = 0.84: island 1 or island 3 active, not both). Here attractors count as distinct when any module differs by
  > 1e-6; with the QMF's criterion (section 4.4: > 10% relative L1 of the module vectors) there are 2,304 multistable cells and at
  most 61 attractors per cell [verification], which is the number to compare with the QMF.
- **First-order lines**: 46 up/down hysteresis intervals (both branches settled), at every r from 0.25 to 6
  (`coarse_post_P0.json` hysteresis_intervals). (557 further single-step jumps > 0.02 without hysteresis are not resolved as first
  order on this grid: a fold pair narrower than one grid step (3.5% in g) cannot be excluded; deviation 7 [verification: was "are
  steep continuous growth, not first-order"].)
- **No stable fixed point** in 450 cells, at r = 0.75-6 (none at r = 0.5) [verification, corrected]: in each of them at least 273
  of the 520 starts (median 519) do not settle, and in the 124 such cells at r >= 3.5 the other starts collapse towards the silent
  state above its edge, so 'no stationary state' means 'no stable fixed point found', not 'every start keeps moving'. First such g:
  2.46 (r = 0.75), 5.29 (1), 2.38 (2), 4.60 (3), 1.11 (4), 0.705 (6). (The values previously quoted here, 1.80 (r = 0.5), 3.03 (1),
  2.38 (2), 1.51 (3), 1.11 (4), 0.705 (6), are where the FIRST start fails to settle, `coarse_post_P0.json`
  first_nonstationary_g_by_r.) At r = 1, 1,710 of 5,120 check columns do not settle at dt = 0.2 and 323 at dt = 0.05 (all at
  g >= 4.0), so most of the closure's non-stationarity is a synchronous-update artefact, but not all.

### 3.2 P1 closure (30 Leiden modules + rest)

- SL edge 0.1634 (r = 0), 0.1897 (1), 0.2254 (2), 0.3609 (4), 0.6577 (5.5), 0.7625 (5.75), 0.9074 (6) [verification, corrected:
  computed directly from the growth rate of the one-sided linearised closure, `code/pd3_verify_module_theory_sledge.py`, and
  confirmed with the nonlinear closure from a 1e-9 single-module seed (grows at 1.02x, decays at 0.98x the edge). The stored
  values (0.2257, 0.3599, 0.6656, "none <= 30 at r >= 5.75") come from a rectified power iteration from the all-ones vector that
  does not converge for P1 at r >= 1.25 (residual 0.3-0.75) and returns mu_+ = 0 at r >= 5.75, where the all-ones vector is
  inhibited everywhere although module L14 alone grows; the error is up to 6% (r = 5: 0.516, not 0.550). P0's stored edges are
  correct at all 25 r.]
- r = 0: SL -> B1 (L14, the AL-containing module) at 0.164 -> continuous cascade -> 31 modules at g = 30; single attractor.
- Only 76 multistable cells (max 4 attractors) and 4 hysteresis intervals, all at r = 0.25-1 (r = 0.5: g 0.64-1.68, module L29
  (VNC) on or off); no lobe-broken states (both lobes are inside L14).
- No stable fixed point in 2,772 cells with the corrected SL edge (2,562 with the stored one) [verification, corrected]: in 2,531 of
  them some starts collapse towards the silent state (X < 1e-12) above its edge and the others do not settle; only 219 have every
  start unsettled. First such g: 0.28 (r = 2.75) to 4.15 (r = 0.25) [was "0.29-1.74"]. At r = 1, 3,204 of 5,120 check columns are
  unsettled at dt = 0.2 and 2,019 at dt = 0.05, g >= 3.37.

### 3.3 Reading

Applied to the connectome's own modules, de Candia's closure produces every ingredient of their Figure 2: a second-order SL edge,
B_m states with named modules (the AL first), broken-symmetry states of the AL lobes for r >= 2.5, and first-order (hysteresis)
lines - plus a large region without a stable fixed point (mostly non-stationary) that equal modules cannot have. But the closure's multistability is a property of the
partition (P0: 2,484 multistable cells with up to 155 attractors; P1: 76 cells with up to 4), and the reduction is not exact for
this connectome (CV 1.2-8.6, section 2.4). The neuron-level QMF (section 4) is the arbiter: where the closure has B1L | B1R
(1,426 cells), the QMF has at most weakly tilted AL states, and its multistability is far sparser (0-15 of 48 g per r at the 10%
criterion, 0-27 at 1%; the closure has 2,304 multistable cells of 4,000 at the same 10% criterion) [verification].
Figures: `figs/pd3/modules_coarse_phase.png` (diagram, attractor counts, P1 diagram, (w0, w1) rays, an r = 4 sweep) and the right
panel of `figs/pd3/modules_skeleton.png`. In the (w0, w1) plane the AL pair sits at |w1| << w0 for every r (de Candia's point A
region; w1 changes sign at r1*_PW = 2.33 / 2.01), while AL-L <- rest has w1 < 0 already for r > 0.73 (r1*_PW(AL-L <- rest),
`deadzone_P0.json`; dotted rays).

## 4. MT3: neuron-level QMF of the full network from 32 starts, with continuation

Exact theory_qmf.py map on all 167,216 neurons, h = 0, r in {0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6}; pass 1: 48 g x 32 starts (<= 25,000
steps, float32 phase to residual 1e-7 then float64 polish to 1e-9); pass 2: up/down sweeps over 192 g (>= 3,000 steps per point);
pass 3: dt = 0.05 check of the all-active start (`qmf_r<r>.npz`, `qmf_analysis.json`).

### 4.1 Silent edge and D

The up-sweep leaves silence between consecutive grid points (`qmf_r<r>.npz` up_mean > 1e-8): r = 0 [0.1586, 0.1630], 0.25 [0.1676,
0.1723], 0.5 [0.1723, 0.1772], 1 [0.1772, 0.1821], 1.5-2 [0.1821, 0.1873], 3-4 [0.1873, 0.1925], 6 [0.1925, 0.1980]. This is the
net-input (rectified) closure's silent edge at neuron level; it moves by about 1.2x from r = 0 to 6, whereas the exact process's
g_lin = 0.1632 does not (theory T1). In every case the transition is continuous: away from the edge up- and down-sweeps agree (at r = 0 to 1.8e-8 relative
everywhere else), and below g = 0.5 the only points differing by > 5% are the 1-2 grid points at the edge itself (critical
slowing; rho(J) within 0.008 of 1), never two consecutive points away from it.
QMF D (PR = N/4, `qmf_analysis.json`): r = 0 1.450 (sim 1.47), r = 0.25 3.01-3.05 (sim 2.50), r = 0.5 7.66-8.75 (sim 4.44), none up
to g = 30 for r >= 1 (sim 14.5 at r = 1) - the known over-inhibition of the QMF D line (`phase_complete.md`).

### 4.2 r = 0: one attractor, continuous everywhere

All 32 starts converge to the same fixed point at all 48 g (max fixed-point attractors 1; also at a 1% criterion [verification]),
no hysteresis (largest up/down difference 1.8e-8 relative away from the critical grid point g = 0.163), no non-stationarity. The r = 0 mean-field diagram is SL -> (continuous) AL -> (continuous) cascade -> D.

### 4.3 r >= 0.25: a non-stationary mean field at strong coupling

Above a few times the onset, the QMF does not settle: the all-active start keeps changing by > 1e-5 per step with a dominant period
of 6-512 steps (1.2-102 model time; `qmf_r<r>.npz` p1_period). First non-stationary g (all-active start, pass 1) and the same with
dt = 0.05 (pass 3):

| r | 0.25 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|---|---|
| dt = 0.2 | 5.53 (2 g only) | 1.43 | 1.02 | 1.28 | 1.43 | 0.91 | 1.60 | 1.28 |
| dt = 0.05 | none | 1.60 | 1.14 | 1.43 | 1.43 | 1.02 | 2.24 | 1.79 |
| neurons with temporal std > 1e-3 (max over g) | 863 | 26,198 | 40,880 | 43,277 | 43,049 | 35,630 | 40,437 | 36,447 |

At r = 0.25 the effect is a synchronous-update artefact (gone at dt = 0.05). For r >= 0.5 it persists at dt = 0.05, i.e. it is a
property of the mean-field ODE: sustained oscillation / irregular dynamics of a large population, of which optic-lobe intrinsic
neurons are the largest group (15,536-23,518 neurons = 43-89% of the oscillating ones; a majority at r = 0.5-2 and 4, but only 47%
at r = 3 and 43% at r = 6 [verification: was "mostly"]), then central-brain and VNC intrinsic (`oscillating_composition_at_max`).
[verification] At the first flagged g the relative peak-to-peak amplitude of the global mean is small (7e-5 to 5e-3 at dt = 0.05)
and pass 3 covers only 2,000 model time, so the onset g of the non-stationarity depends on the detection threshold and a slowly
damped oscillation cannot be excluded at the lowest flagged g; the amplitude grows with g (to 1-5% at g ~ 6-8). de Candia's
equal modules cannot oscillate (all neurons of a module see the same input); the heterogeneous connectome's E-I loops can. Whether
the stochastic model shows corresponding rhythms is not tested here.

### 4.4 Multistability of fixed points (MT3-multi) - the prediction 'none anywhere' fails

Two or more converged (residual < 1e-7 and estimated distance residual / (1 - rho(J)) < 1e-6), stable (rho(J) < 1) fixed points
whose 34-module vectors differ by > 10% relative L1, from different starts at the same (g, r):

| r | 0 | 0.25 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|---|---|---|
| g with >= 2 fixed points | none | none | 11 g in 0.91-5.53 | none | 2.51, 2.81 | 2.01 | 0.81-1.28 (3 g) | 0.73-1.60 (8 g) | 0.33-1.60 (15 g) |
| max number | 1 | 1 | 4 | 1 | 3 | 2 | 2 | 3 | 4 |
| [verification] same, single-linkage clustering (order-independent) | none | none | 9 g in 0.91-3.52 | none | 2.51, 2.81 | 2.01 | 3 g | 8 g | 15 g |
| [verification] > 1% instead of > 10% apart: g (max number) | none (1) | 27 g in 0.65-30 (4) | 22 g in 0.73-7.76 (11) | 5 g in 0.91-2.81 (3) | 5 g in 0.91-2.81 (4) | 8 g in 0.91-2.24 (5) | 6 g in 0.81-1.43 (3) | 8 g in 0.73-1.60 (3) | 15 g in 0.33-1.60 (4) |

[verification] The pre-registered 10% threshold is on the relative L1 distance of the 34 module MEANS, which weights the 3-neuron
AL-mid like the 158,745-neuron rest and normalises by the sum of all module means; it hides coexisting fixed points at r = 0.25 and 1
that are of the same kind as those at r = 0.5. Examples (converged to residual < 1e-9, rho(J) 0.97-0.99): r = 0.25, g = 1.28, four
stable fixed points in which island 2 (75 VNC neurons) has mean activity 0.28 or 0.05 (relative L1 6.8%); r = 1, g = 0.91, island 2
at 1e-4 or 0.046. Only r = 0 is single-attractor at every threshold. The greedy clustering of the published counts is
order-dependent: at r = 0.5, g = 3.94 and 5.53 all 32 starts converge to 7-11 fixed points more than 1% apart that form a chain of
< 10% steps, so single linkage at 10% gives 9 instead of 11 multistable g. r = 2's single multistable g (2.01) and r = 6's g = 1.60
do not survive residual < 1e-9 or rho(J) < 0.995 (r = 2's states reach rho(J) = 0.9987); all other entries do.

What differs (MT3b, every pair of coexisting attractors recomputed and compared neuron by neuron, `qmf_multi.json`):
- each pair differs in **one connected region** (connected components of the synaptic graph on the neurons with |d rho| > 0.05:
  1 in 78 of 80 pairs; the other two have a second component of 1 and 11 neurons);
- r = 0.5: g 0.91-1.28, a 107-195-neuron VNC region around island 2 (vnc_intrinsic, vnc_efferent, vnc_motor); g 2.0-5.5,
  337-9,274 neurons (island 1, central-brain and VNC intrinsic, descending neurons) - large alternative states of the VNC/premotor
  system;
- r = 1.5 and 2: 78-821 neurons, the same VNC/central circuits (at r = 1.5 also 17% optic-lobe intrinsic neurons [verification]);
- r = 3-6: 8-362 neurons, mostly **AL local interneurons** (lLN2P_a/b/c, lLN2X12, lLN2F, lLN2T types) and HRN_VP1d receptor
  neurons; at r = 4 also 8-30 neurons of island 8 (ascending neurons).
- [verification] The differences are small at large r: summed over neurons, |rho_A - rho_B| totals 1.5-86 neuron-equivalents of
  activity at r = 4 and 2.9-43 at r = 6 (25.7-1,768 at r = 0.5); 15 of the 80 pairs (19%) differ by < 10% relative L1 at neuron
  level (`qmf_multi_vectors.npz`). E.g. r = 6, g = 0.33: total activity ~29 neuron-equivalents, the two states differ in 20 neurons.

### 4.5 No AL lobe symmetry breaking (B1), but tilted AL states at r >= 3

No AL-L-only start ever ended with AL-R below 10% of AL-L (B1 criterion; `g_B1_like_AL_L_start` empty at all 9 r). The coexisting
AL states at r = 3-6 differ in the left/right balance: lobe asymmetry A = (X_L - X_R)/(X_L + X_R) of the coexisting fixed points
ranges -0.011..0.050 (r = 3), 0.028..0.139 (r = 4) and -0.108..0.170 (r = 6), with a spread up to 0.26 within one g (r = 6;
`module_theory.json` MT3.AL_lobe_asymmetry_of_coexisting_states). Example r = 6, g = 0.518: three states with
(X_L, X_R) = (0.078, 0.071), (0.084, 0.060), (0.067, 0.076) - a nearly balanced, a left-tilted and a right-tilted state, pairwise
differing in 51-99 neurons, mostly lLN2-type local interneurons (`qmf_multi_vectors.npz`, types.npy). This is broken symmetry in a weak, partial sense (both lobes stay active; de Candia's B1 has the
minority module silent, |A| -> 1). It appears only at r >= 3, where the closure (theory_module2) predicted B1L | B1R from r = 2.5.

### 4.6 Hysteresis (MT3-order)

Up/down sweeps differing by > 5% on >= 2 consecutive g, excluding critical-slowing points; classified by what the two branches are
(`hysteresis_points_by_type`):

| r | 0 | 0.25 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|---|---|---|
| A: both branches stable fixed points | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| B: fixed point vs non-stationary branch | 0 | 0 | 10 | 13 | 0 | 0 | 2 | 0 | 2 |
| C: both non-stationary (time averages over 256 steps) | 0 | 0 | 34 | 74 | 66 | 51 | 92 | 79 | 93 |

Type A (genuine fixed-point hysteresis) occurs only at r = 0.5 (e.g. g 3.26-3.75, 6 points, up to 18% difference); type B (a stable
fixed point on one branch, an oscillating state on the other) at r = 0.5 and r = 1 (g 1.82-3.0) and 2 points each at r = 3 and 6.
Type C differences are between time averages of irregular states and are not evidence of distinct attractors.

### 4.7 Verdict on the MT3 predictions

- 'No B1 anywhere': **holds** (no lobe-level B1 at any r).
- 'No multistability or hysteresis anywhere': **fails** at the mean-field level for r >= 0.5 with the 10% criterion (except
  r = 1) and for every r >= 0.25 with a 1% criterion [verification]: coexisting fixed points (one
  connected region each, 8-9,274 neurons), fixed-point hysteresis at r = 0.5, fixed point / oscillation coexistence at r = 0.5-1, and
  weakly tilted coexisting AL states at r >= 3. All of it lies at g >= 0.33 (r = 6) or g >= 0.65 (r < 6; 0.73 with the 10%
  criterion), above the QMF's AL onset - but at r = 6 below the simulated onset (tau = 3000 at g = 0.579) [verification]; the
  onset itself is continuous at every r. Handed to multistability M3b (stochastic test) - list of (r, g, starts) in
  `qmf_analysis.json` (`census`, `g_multistable_fixed_points`) and `qmf_multi.json`.
- Figure: `figs/pd3/modules_qmf_phase.png` (per-module X_k(g) on both branches at r = 0, 1, 4; the (g, r) map of fixed-point counts
  with non-stationary cells marked; rho(J) and the mean activity on both branches).

## 5. MT4: which modules are active where (existing simulations)

From occ[g, N] of all 9 + 23 + 13 real-graph T = 3000 runs (45 files, 23 r; `module_occ_P0.json`), module occupancy among survivors
(X_k / fraction alive in the second half; only g with >= half the replicas alive). The plan's activation point (max of
d ln X_k / d ln g) lands, for every module, at the sustained onset, because the replica-mean occupancy of every module jumps there;
the threshold crossing X_k >= 1% among survivors is reported instead (both are in the json).

| r | AL (first sustained g) | isl1 | isl2 | isl3 | isl5 | median of 30 islands | rest | own g_lin,k (isl1 / isl2 / isl3 / isl5 / rest) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.278 | 0.382 | 0.525 | 0.487 | 0.521 | 0.460 | 0.701 | 0.665 / 1.001 / 0.754 / 0.715 / 0.916 |
| 1 | 0.362 | 0.559 | 1.009 | 1.014 | 0.946 | 0.847 | 2.37 | same |
| 4 | 0.524 | 0.826 | 1.82 | 1.69 | 1.49 | 1.49 | 6.11 | same |
| 6 | 0.612 | 0.949 | 2.18 | 1.94 | 1.69 | 1.73 | 7.45 | same |

All 30 islands reach 1% at every r; at r = 0 they do so at g 0.38-0.92 (median 0.46; QMF 0.37-0.90), although their own thresholds
are 0.665-336 (the 28 islands among the top 30 of MT1; isl15 and isl23 have no g_lin,k because they are outside the top 30).
The coupled system switches modules on by drive from the already active AL (imperfect bifurcations), so island 'lines' are smeared
crossovers (dotted in the skeleton), not sharp transitions. The number of modules above 1% rises steeply at r = 0, from 4 (g = 0.41)
through 8, 9, 24 and 30 to 31 (g = 0.52) on the pooled grid (`n_ge_1pct`) - steep, but resolved, not a jump [verification: was
"jumps from 3-4 to 31"]. Figure `figs/pd3/modules_sim_occupancy.png`.
[verification] "Far below its own threshold" holds at r = 0 only (all 28 top-30 islands cross below their g_lin,k). At r >= 1 the
islands with low isolated thresholds reach 1% near or above them, as the table above already shows: r = 1 isl2 1.009 (own 1.001),
isl3 1.014 (0.754), isl5 0.946 (0.715), isl28 6.77 (2.98); r = 4 and 6 also isl1 0.826 / 0.949 (0.665) and isl12 1.49 / 1.65 (1.39).
g_lin,k is the excitation-only (r-independent) threshold, so inhibition raises the effective threshold; the activation is still
imperfect (driven), but not "far below" it.

## 6. MT5: islands simulated alone (is any island 'a second AL'?)

module_sim convention, B = 64, T = 3000, 24 g in [1, 4] x g_lin,k (`island_lifetimes.json`):

| module | n | g_lin,k | r = 0: tau(x = 1); x at tau = 200 / 3000 (g at 3000) | r = 1: tau(x = 1); x at tau = 200 / 3000 |
|---|---|---|---|---|
| AL (526) | 526 | 0.1643 | 19.9; 1.45 / 1.65 (0.271) | 16.5; 1.76 / 2.07 (0.340) |
| isl1 | 2,292 | 0.665 | 28.7; 1.33 / 1.46 (0.973) | 15.5; 3.08 / > 4 |
| isl5 | 221 | 0.715 | 18.9; 1.41 / 1.62 (1.160) | 8.2; > 4 / > 4 |
| isl3 | 234 | 0.754 | 11.8; 3.02 / 3.83 (2.89) | 9.2; > 4 / > 4 |
| isl4 | 127 | 2.485 | 10.0; 3.52 / > 4 | 7.5; > 4 / > 4 |
| isl2 | 75 | 1.001 | 7.6; > 4 / > 4 | 7.3; > 4 / > 4 |

At r = 0 islands 1 and 5 are AL-like (sustained at 1.46-1.62 x their own threshold, like the AL's 1.65); at r = 1 only the AL is.
The AL alone sustains at g = 0.271 (r = 0) and 0.340 (r = 1), matching the whole-brain onsets 0.266 and 0.344 (`phase_complete.md`).
So among the AL and islands 1-5 there are one (r = 1) to three (r = 0) long-lived regions, at very different g - not the broad
spectrum of rare regions a Griffiths phase needs. Islands 6-30 and the rest were not simulated alone, so this is a statement about
the six largest-Lambda modules only [verification: was "the connectome has one (r >= 1) to three ..."; only r = 0 and 1 were run].
Figure `figs/pd3/modules_island_lifetimes.png`.

## 7. Deliverable: the module skeleton

`figs/pd3/modules_skeleton.png`: (a) neuron level - SL edge at g_lin (solid), simulated tau = 3000 and PR = N/4 crossovers (dashed),
imperfect island activation (dotted: island 1, median island, rest reaching 1%), the QMF's non-stationary band, its multistable
fixed points and its hysteresis intervals (markers; mean-field candidates, not confirmed first-order lines); (b) the closure's
diagram for the same modules. No first-order line is drawn as confirmed: that needs the stochastic test (multistability M3b).
[verification: figure regenerated by `code/pd3_verify_module_theory_figs.py` (original in `figs/pd3/verify_orig/`): the SL edge
legend no longer says "MF-DP" (criticality.md rejects the MF-DP class; g_lin is the exact linear instability of silence), the
"rounded critical band" label is now "finite-K rounded onset", QMF fixed points coexisting at > 1% are added as small dots, and
"no stable state (non-stationary)" is now "no stable fixed point found".]

## 8. Answers

**Q1.** A de Candia-type diagram with named modules exists for this connectome only in de Candia's closure (MT2). The exact neuron-level
mean field keeps the SL edge continuous, has no lobe-level B1, and replaces the closure's clean B_m / SH structure by (i) a smeared,
driven module cascade, (ii) localized multistability of particular circuits (VNC/premotor at r = 0.25-2 [verification: was 0.5-2;
r = 0.25 and 1 only at a 1% criterion], AL local neurons at r >= 3,
giving weakly tilted L/R states) and (iii) a non-stationary mean field at strong coupling for r >= 0.5. These are mean-field results;
the first-order candidates are passed to M3b.

**Q2.** At mean-field level the onset is continuous at every r (single attractor, no hysteresis, transcritical edge), consistent with
the finite-size-rounded second-order reading (but equally with a smooth crossover of a finite module; the mean field cannot tell them
apart); the multistability found sits at higher g than the QMF onset, and - except at r = 6 - than the simulated onset
[verification]. MT4 shows the 'onset' of the brain is the AL's, and every further module joins smoothly. Finite-size scaling itself
is tested elsewhere.

## 9. Deviations from the spec (all deliberate, with reasons)

1. **P0 merge rule** (section 2.1): seed-mode, larger-support normalisation, merged modes add no neurons; the literal rule snowballed.
   P0 has 34 modules as specified.
2. **Own sbatch wrappers** `code/pd3_mt_array.sbatch` / `pd3_mt_cpu.sbatch` (same resources as the plan's) to avoid editing files
   shared with other programs; the closure jobs ran on `-p ellis` (a subset of the allowed partitions) after two preemptions on
   owner nodes (176013, 176275).
3. **QMF precision** (`pd3_qmf_ic.py` header): a float64 run was 7x slower (smoke 175264); float32 alone stalls at residual ~1e-8
   (first attempt 175422, cancelled); final scheme float32 phase to 1e-7, float64 polish to 1e-9.
4. **One-sided Jacobian** at silent neurons with exactly zero input (the plan's p'(0) = 0 would make silence 'stable' at every g).
5. **QMF extras**: non-stationary columns are frozen after 10 checks > 1e-3 past step 10,000; time averages / amplitudes / periods
   for non-stationary columns; per-neuron temporal std of the all-active start; pass 3 at dt = 0.05; pass-2 points extended to
   <= 15,000 steps only while converging slowly; convergence also requires residual / (1 - rho(J)) < 1e-6 (removes slowly drifting
   columns next to the SL edge, e.g. r = 0.25, g = 0.168, rho(J) = 0.9998); MT3b (`pd3_qmf_multi.py`) added to identify what
   differs between coexisting attractors.
6. **Closure non-stationarity**: columns frozen after 10 checks > 1e-5 past step 20,000 (a CPU probe found 29% of random-start columns
   at g >= 3, r = 1 still unsettled after 2e5 steps); dt = 0.05 check at r = 1.
7. **Closure SL and boundary classification** (`pd3_coarse_post.py`): states with max X < 1e-12 are the SL fixed point, stable iff
   g < alpha_eff / mu_+(W) (they had been stored as converged with rho(J) = e^{-dt}); first order is identified by hysteresis
   intervals and multistable-region edges, not by single-step jumps (a deterministic fold implies coexistence).
8. **Dead-zone 'meaningful' criterion** (section 2.5) added because the plain and strict criteria are dominated by trivial pairs.
9. **MT4 activation point**: the 1% crossing among survivors replaces the max-slope point for interpretation (both stored).
10. MT5 also simulated the AL module itself (reference) - 12 tasks instead of 10.

## 10. Caveats and negative results

- Everything in sections 3-4 is mean field (closure or QMF). Finite-K fluctuations can destroy multistability of small regions
  (8-362 neurons at r >= 3) and turn coexistence into metastable switching; nothing here says which.
- The QMF over-inhibits relative to the stochastic model (D line, onset; `phase_complete.md`), so the r-scale of the QMF findings
  (multistability from r = 0.25 at a 1% criterion / r = 0.5 at 10%, AL tilt from r = 3) may not transfer one-to-one to the simulation.
- Type-C hysteresis is not interpreted (time averages of irregular states over 256 steps).
- P0 islands 2-30 are fragments of one shared region (section 2.1); module identities in P0 are therefore partly arbitrary. P1
  (Leiden) and P2 give the same leading structure (one AL-containing module at g_lin ~ 0.164-0.169, everything else >= 0.54).
- MT4 uses existing runs with B = 16-32; the 1% crossings are on irregular pooled g grids.

## 11. Files

Code: `code/pd3_partition_v0.py`, `pd3_modules.py`, `pd3_module_stats.py`, `pd3_coarse_mf.py`, `pd3_coarse_post.py`, `pd3_qmf_ic.py`,
`pd3_qmf_analyze.py`, `pd3_qmf_multi.py`, `pd3_module_occ.py`, `pd3_island_sim.py`, figure scripts `pd3_modules_fig.py`,
`pd3_coarse_fig.py`, `pd3_module_occ_fig.py`, `pd3_island_fig.py`, `pd3_skeleton_fig.py`, summary `pd3_module_theory_summary.py`,
sbatch `pd3_mt_array.sbatch`, `pd3_mt_cpu.sbatch`.
Results: `results/pd3/modules/` (partitions, quotients, dead zones, closure, occupancy), `results/pd3/module_theory/` (QMF, MT3b, MT5,
JOBS.md), `results/pd3/module_theory.json`.
Figures: `figs/pd3/modules_partition.png`, `modules_coarse_phase.png`, `modules_qmf_phase.png`, `modules_sim_occupancy.png`,
`modules_island_lifetimes.png`, `modules_skeleton.png`.

## 12. Verification (adversarial verifier, 2026-11-09)

Independent recomputation with `code/pd3_verify_module_theory.py` (imports no module_theory code; output
`results/pd3/verify_module_theory.json`) and `code/pd3_verify_module_theory_sledge.py` (closure SL edge computed directly;
`results/pd3/verify_module_theory_sledge.json`); the two corrected figures were regenerated by `code/pd3_verify_module_theory_figs.py`
(adapted copies of the original figure scripts, which are unchanged; original PNGs in `figs/pd3/verify_orig/`). The verified and
corrected numbers are also in `module_theory.json` under `verification` (and `MT2.P1.g_SL_edge_CORRECTED_direct`). Every figure of
this program was looked at.

### 12.1 Reproduced from the raw inputs / outputs with my own code

| headline (published) | recomputed | from |
|---|---|---|
| 87% of the 395 non-AL top-400 modes have > 50% of their 90% support in mode 2's; median 0.63 | 0.866, 0.625; mode-2 support 2,293 (39% vnc_intrinsic, 25% cb_intrinsic, 1 AL neuron); max 1/IPR 632 | `pd/tail_real_r0_dir.npz` |
| AL-L / AL-R alone: g_lin,k 0.1687 / 0.1665, K 36.6 / 36.5 | 0.16871 / 0.16647, K 36.55 / 36.54 (AL 526: 0.16428, = MT5) | connectome, dense eig + bisection |
| P0 median CV: E self 1.23 / cross 3.49, I 1.64 / 3.62 | identical (own top-30 grouping identical to `quotient_P0.npz`) | connectome |
| AL Perron-weighted pair E^PW, I^PW (0.080 / 0.066, 0.034 / 0.033) | identical to 1e-15 | connectome, own Perron vectors |
| closure P0 SL edge 0.1645 (r = 0) / 0.3863 (r = 6) | 0.16451 / 0.38632; the direct growth-rate edge equals the stored one at all 25 r; the nonlinear closure from a 1e-9 seed decays at 0.99x and grows at 1.01x the edge (r = 0, 1, 2.5, 4, 6) | `quotient_P0.npz` |
| closure P0: 2,484 multistable cells, max 155, B1L and B1R coexist in 1,426 cells from g 0.224 / 0.286 / 0.434 (r 2.5 / 4 / 6), 46 hysteresis intervals at every r 0.25-6, r = 0 cascade 2 / 3 / 6 / 11 modules at g 0.170 / 0.209 / 0.391 / 0.419 | identical | `coarse_mf_P0.npz` |
| closure P1: 76 multistable cells, max 4, 4 hysteresis intervals (r 0.25-1; r = 0.5: g 0.64-1.68, L29) | identical | `coarse_mf_P1.npz` |
| theory_module2 B1L and B1R from 0.287 (r = 4), 0.428 (r = 6) | 0.2871, 0.4280 (r = 2.5: 0.2236) | `theory_module2.json` |
| QMF silent-edge brackets, r = 0 [0.1586, 0.1630] ... r = 6 [0.1925, 0.1980] | identical; alpha_eff / mu(r) of the earlier, independent `theory_qmf.py` run (`pd/qmf_real.npz`) lies inside 8 of 9 brackets (r = 0.5: 0.17208, 0.14% below it). The down-sweep returns to silence in the same bracket or one grid point (2.8%) lower (r = 0.25, 0.5, 1.5), at the predicted edge: critical slowing, not hysteresis | `qmf_r<r>.npz` |
| QMF D 1.450 (r = 0; sim 1.47), 3.01-3.05 (0.25; sim 2.50), 7.66-8.75 (0.5; sim 4.44); sim 14.5 at r = 1 | 1.4499 (sim 1.4708), 3.01-3.05 (2.4992), 7.66-8.75 (4.4406); 14.47 | `qmf_r<r>.npz`, `analysis2/phase_complete.json` |
| QMF r = 0: one attractor at all 48 g | also at a 1% criterion | `qmf_r0.npz` |
| QMF multistable g and max count per r (10%, published clustering) | identical when re-implemented; see 12.2 for single linkage and threshold | `qmf_r<r>.npz` |
| QMF hysteresis by type: A 20 (r = 0.5 only), B 10 / 13 / 2 / 2 (r 0.5 / 1 / 3 / 6), C as tabled | identical (A at g 1.27-1.31, 3.26-3.75, 4.42-6.00; max 17.7%; rho(J) <= 0.988) | `qmf_r<r>.npz` |
| QMF non-stationarity: first g at dt 0.2 / 0.05 (table 4.3) and 863-43,277 oscillating neurons | identical | `qmf_r<r>.npz` |
| MT3b: one connected component in 78 of 80 pairs; 8-9,274 differing neurons; AL asymmetry ranges r = 3 / 4 / 6; the r = 6, g = 0.518 example | identical (own synaptic graph, own components); identities of the differing neurons (VNC around isl2 at low g, r = 0.5; central/VNC/descending at high g; AL lLN2 types + HRN_VP1d at r >= 3) confirmed from `types.npy` | `qmf_multi_vectors.npz` |
| MT4: table of section 5, island 1% crossings at r = 0 0.38-0.92 (median 0.46; QMF 0.37-0.90) | identical (own pooling of the raw occ / surv of the 45 runs) | `pd/decay_*`, `pd2/onset_*`, `deloc_*` |
| MT5: every tau(x = 1), x at tau = 200 / 3000, g at tau = 3000 of the table | identical (own censored-exponential MLE); 10-58 of 64 replicas die at the grid points bracketing tau = 3000, so the crossings are well determined | `island_<m>_r<r>.npz` |
| Leiden: 50 modules >= 50 neurons, largest 8,622, 1,886 singletons, NMI(P1, P0) 0.052; 13,632,893 undirected excitatory edges | identical (the 13,632,893 exclude 54 self-loops); the gamma choice follows the plan's rule | `partition_P1.npz`, connectome |
| dead-zone counts (section 2.5 table) and AL-pair shares | identical, recounted from the pair rows | `deadzone_<P>.json` |

### 12.2 Errors found and corrected (each marked [verification] in the text)

1. **P1 closure SL edge (section 3.2, json).** The stored edge comes from a rectified power iteration from the all-ones vector
   that does not converge for P1 at r >= 1.25 (residual 0.3-0.75) and returns mu_+ = 0 at r >= 5.75 ("no SL edge <= 30"). Computed
   directly: 0.2254 (r = 2), 0.3609 (4), 0.516 (5; stored 0.550, the largest error, 6%), 0.6577 (5.5), 0.7625 (5.75), 0.9074 (6);
   confirmed with the nonlinear closure from single-module seeds. With the corrected edge P1 has 2,772 (not 2,562) cells without a
   stable fixed point; multistable cells unchanged (76). P0 is unaffected (its iteration converges; edges identical at all r).
2. **"No stationary state ... every start keeps moving" (3.1, 3.2).** False: in every such P0 cell 1-247 of the 520 starts
   converge (to unstable states or to the silent state above its edge); it means "no stable fixed point found". The P0 first-g
   list quoted (1.80 at r = 0.5, 3.03 at r = 1, 1.51 at r = 3) was where the first start fails to settle; the first cells without a
   stable fixed point are 2.46 (r = 0.75), 5.29 (1), 2.38 (2), 4.60 (3), 1.11 (4), 0.705 (6), and there are none at r = 0.5. P1's
   range "0.29-1.74" is 0.28-4.15.
3. **QMF multistability depends on the 10% threshold (bottom line, 4.4, 4.7, 8, 10).** The relative L1 distance of the 34 module
   means weights a 3-neuron module like the 158,745-neuron rest; coexisting converged stable fixed points that differ by 1-10%
   exist at r = 0.25 (27 of 48 g, up to 4) and r = 1 (5 g, up to 3) - island 2 on or off, the same VNC switch as at r = 0.5 - so
   "for r >= 0.5" becomes "r = 0.5 and 1.5-6 at 10%, every r >= 0.25 at 1%"; only r = 0 is single-attractor at every threshold.
   The published greedy clustering is order-dependent (single linkage at 10%: 9 instead of 11 g at r = 0.5); r = 2's only
   multistable g and r = 6's g = 1.60 do not survive residual < 1e-9 or rho(J) < 0.995.
4. **"Above the simulated onsets" (Q2 row, 4.7, 8).** At r = 6 QMF multistability starts at g = 0.33, below the simulated
   tau = 200 (0.436) and tau = 3000 (0.579) lines (`phase_complete.json`); the statement holds for r < 6 only.
5. **"Every island turns on far below its own threshold" (Q2 row, section 5).** True at r = 0 only; at r = 1, 4, 6 isl1-3, isl5,
   isl12 and isl28 reach 1% near or above their (excitation-only) g_lin,k, as section 5's own table shows.
6. **"Mostly optic-lobe" oscillating neurons (bottom line, 4.3).** Optic-lobe intrinsic neurons are the largest group (43-89%) but
   not a majority at r = 3 (47%) and r = 6 (43%). Added: the global-mean oscillation amplitude at the first flagged g is only
   7e-5 to 5e-3 (dt = 0.05), so the onset g of the non-stationarity is detection-threshold dependent.
7. **Closure vs QMF comparison used different distinctness criteria (3.1, 3.3).** Closure attractors were deduplicated at 1e-6
   (L_inf), QMF states at 10% relative L1. At the QMF's criterion the P0 closure has 2,304 multistable cells and at most 61 (not
   155) attractors; the qualitative contrast stands.
8. **"557 steep jumps ... are steep continuous growth, not first-order" (3.1)** -> not resolved as first order at the 3.5% grid; a
   fold pair narrower than one grid step cannot be excluded.
9. **Convergence statement (bottom line).** States are counted at residual < 1e-7 with estimated distance < 1e-6, not < 1e-9;
   95-100% of the counted states are below 1e-9 (max 4.3e-9) and rho(J) <= 0.9987.
10. **MT4 "jumps from 3-4 to 31" (section 5)** -> a steep but resolved rise (4 -> 8 -> 9 -> 24 -> 30 -> 31 between g = 0.41 and 0.52).
11. **MT5 scope (Griffiths paragraph, section 6).** Only the AL and islands 1-5 were simulated alone, and only at r = 0 and 1:
    "the connectome has one (r >= 1) to three long-lived regions" -> "among the AL and islands 1-5, one (r = 1) to three (r = 0)".
12. **Q2 strength wording.** "supports a finite-size-rounded second-order onset" -> "consistent with a continuous onset at mean-field
    level; the mean field cannot tell a rounded transition from a smooth crossover of a finite module" (criticality.md rejects the
    MF-DP class).
13. **`figs/pd3/modules_skeleton.png`.** The legend called the g_lin line "SL edge (MF-DP, exact, r-independent)" and the region
    beside it "rounded critical band"; criticality.md finds the MF-DP class NOT supported. Now "linear instability of silence (exact,
    r-independent)" and "finite-K rounded onset"; QMF fixed points coexisting at > 1% added as small dots; "no stable state
    (non-stationary)" -> "no stable fixed point found".
14. **`figs/pd3/modules_coarse_phase.png`.** Panel (c) now uses the directly computed P1 SL edge (the line now reaches r = 6) and
    the corrected no-stable-fixed-point cells; legends and panel (b) say "no stable fixed point found" instead of
    "non-stationary / none settled"; panel (f) gives the corrected P1 count and the P0 count at the 10% criterion.
15. Smaller: r = 0 closure is SH from g = 1.80 (not "at g = 30"); section 4.2's "1.8e-8" holds away from the critical grid point; r = 1.5
    difference sets include 17% optic-lobe neurons; the coexisting QMF states at r = 4-6 differ by only 1.5-86 neuron-equivalents
    of summed activity (19% of the 80 pairs by < 10% relative L1 at neuron level) - added because it bears on whether noise will
    wash them out.

### 12.3 Methods checked and found sound

- Closure map, Jacobian with one-sided rows and the analytic SL decision (post-processing deviation 7) are correct for P0.
- QMF convergence: residual plus distance criterion is adequate (counted states are at most ~3e-6 from their fixed point, while the
  differences reported are >= 1%); the one-sided Jacobian at silent neurons is the right linearisation of the rectified map.
- No power-law exponents are fitted in this program. MT5 uses the censored-exponential MLE, appropriate given the exponential
  lifetimes established earlier (STATE.md); B = 64 gives 10-58 deaths at the crossings.
- Transition vs crossover: "transcritical / second order / first order (hysteresis)" is used only for the deterministic closure
  and QMF, where it is correct; simulated module activation is called a crossover throughout. The QMF's continuity at the onset
  does not by itself make the stochastic onset a phase transition (wording 12.2-12 adjusted).

### 12.4 Not verified

- The closure and QMF runs themselves were not re-simulated (only their stored outputs, plus the independent SL-edge
  computations); `coarse_mf_validation.json` (1525/1525) was taken as stored; the Leiden runs were not repeated.
- The dt = 0.05 closure check and the MT3 pass-3 runs were read, not rerun.
