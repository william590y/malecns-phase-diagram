# Phase diagram v3 (pd3): de Candia-style diagram, the order of the crossover, heat map and finite-size check

Synthesis of workflow wf_98fb990d-4fb, 2026-11-09. Model: **Model 1** (stochastic Wilson-Cowan, `code/swc.py`, h = 0 unless
stated; see the caveat in section 8). This report assembles four programs, each adversarially verified:
`redraw_fss.md`, `module_theory.md`, `criticality.md` and `multistability.md` (all in `results/pd3/`). Their verification
corrections are applied here.

**Sourcing.** Every number is taken from one of those files (section number given), or from the JSON named next to it.
Numbers I computed or re-read for this synthesis are listed with their source key in `results/pd3/synthesis.json`, which is
written by `code/pd3_synthesis.py` (CPU only; no new simulation). Notation: x = g / g_lin, with g_lin = 0.1632 (0.16315,
`analysis2/phase_complete.json`); r = inhibition scale; τ = lifetime from a fully active start (model time 1/α).

---------------------------------------------------------------------------------------------------------------------

## 1. Plain-language answers

**Q1. Can we recover a phase diagram like de Candia et al. 2026?** Partly. The result is more complete than the old
"Griffiths region", and part of it is negative.
- **What exists.** The connectome has one critical line, at g_lin = 0.1632. It is vertical in (g, r) because inhibition
  cannot move it. Below it is the quiescent phase (de Candia's SL). Above it come, in order:
  - a window where the antennal lobe (AL) is active but metastable, and dies at a finite time;
  - a sustained AL-only phase. This is the analogue of de Candia's B1 with *one* active module, the whole AL (left and right
    lobes together). Per neuron the rest of the brain is nearly silent, but in total it carries 51–71% of the activity: a
    B1-like state with heavy spill-over, not de Candia's B1 with exactly silent modules;
  - a smeared cascade in which further modules ("islands") switch on one after another;
  - at r ≤ 1.5, a global delocalized phase D, the analogue of de Candia's SH.
- **What does not exist** in the stochastic model:
  - no broken left/right symmetry (no state with only one lobe active);
  - no coexisting states;
  - no first-order (dashed) lines.
- **Where those features come from.** de Candia's own mean-field closure, applied to our modules, does predict one-lobe
  states (B1L | B1R) and first-order lines (figure 1b). Every stochastic test refutes them (section 5). They come from the
  closure's assumption that all neurons in a module receive the same input. That assumption fails badly here.
- **Not a Griffiths phase** either. Among the six modules simulated alone, one to three are long-lived, at very different g.
  A Griffiths phase needs a broad distribution of rare regions.

**Q2. Is the steep but smooth crossover a second-order phase transition?** Not in the strict sense. The underlying onset is
continuous, but the steep crossover itself is not a phase transition.
- **What the crossover is.** The steep crossover sits at 1.43–1.63 × g_lin (r = 0) and 1.73–2.11 × g_lin (r = 1). That is
  where the lifetime of one small module (the AL) crosses the observation time. It moves when the observation time changes.
- **Where the transition is.** It is consistent with the finite-size-rounded image of a *continuous* onset at g_lin. The
  exact-linearisation chain is continuous there (the neuron-level mean field at its own nearby edge), and no first-order
  signature was found in the simulations.
- **Why it is not a phase transition of the brain.** The rounding is set by a fixed, small effective module, about 70 units at
  r = 0 and about 36 at r = 1. It does not grow with brain size, so the crossover never sharpens:
  - the AL alone (526 neurons) has the same onset to within 1.8% as the full brain (167,216 neurons);
  - delocalized random-wiring nulls (configuration nulls) do sharpen toward g_lin as N grows.
- **Universality.** The onset is not described by a single finite-K mean-field directed-percolation (MF-DP) mode, the class
  de Candia report: the avalanche statistics deviate from it in several ways. No asymptotic exponent can be measured in a
  module of K ≈ 10–90, so the universality class itself is undetermined. The one exponent with an asymptotic-looking
  window, the size exponent above ~20 spikes (1.46–1.57), is close to MF-DP's 3/2.
- **"Second order" therefore only means** "continuous, with no first-order signature, consistent with a rounded
  transcritical point at g_lin". Strength: section 4.

**Q3 (offers you accepted).**
- **(a) Heat map.** Done: `figs/pd3/phase_lifetime_heatmap.png`. g_lin is the only solid line. τ = 200, τ = 3000 and
  PR = N/4 are dashed and labelled as crossovers.
- **(b) Finite-size check.** Done, with 54 + 24 GPU runs. Null onsets fall monotonically toward g_lin as N grows, from 2.5k to
  167k neurons. Real subsamples that keep the AL, from 5k to 80k neurons, stay within 1.1% of the full brain. The
  pre-registered scaling of the fitted module size failed, narrowly (section 6).

---------------------------------------------------------------------------------------------------------------------

## 2. Figures (all in `figs/pd3/`; I looked at each)

| figure | what it shows |
|---|---|
| **`phase_diagram_v3.png`** (new) | **The diagram, in de Candia style.** (a) The real connectome in the stochastic model: named regions SL / M / B_AL (+ islands) / D. g_lin is the only solid line. Crossovers are dotted (τ = 200, τ = 3000, PR = N/4, 1% island activation). The legend has a dashed "first-order line" entry, marked "none found". Theory is overlaid: the one-mode chain onset, the QMF D line, the QMF silent edge, and the QMF coexistence cells that noise removes. (b) de Candia's closure on the same 31 modules: SL edge, B_m, SH, B1L \| B1R, multistable cells, first-order intervals. The stochastic left/right-symmetry tests (M1) and the simulated τ = 3000 line are overlaid. |
| **`q2_second_order_evidence.png`** (new) | The Q2 evidence in six panels: (a) order parameter: QS simulation, finite-K chain, mean-field amplitude; (b) AL susceptibility; (c) location of the growth-rate zero x* (data, chain with the threshold at g_lin, net-input closure, one-mode controls, null); (d) onset distance from g_lin against N; (e) local slope of ⟨S⟩(T) against MF-DP controls; (f) hysteresis loop area against dwell time. |
| **`fss_summary.png`** (new) | Finite-size check: lifetime curves τ(x) for nulls coloured by N against the real subsamples, the full brain and the AL alone (r = 0 and 1); x_on − 1 against N; survivor fluctuations against N. |
| `phase_lifetime_heatmap.png` (redraw_fss, verified) | The heat map you asked for (Q3a): measured lifetime over (g, r), PR/N where activity is sustained, and the drift of the onset with T (panel C). Nulls: `phase_lifetime_heatmap_nulls.png`. |
| supporting (program figures) | `fss_onset_vs_N.png`, `fss_lifetime_curves.png`, `lifetime_vs_K_criticality.png`, `aval_*.png`, `qs_*.png`, `field_scaling.png`, `modules_skeleton.png`, `modules_coarse_phase.png`, `modules_qmf_phase.png`, `multistab_*.png`, `verify_*.png` |

**Line convention.** The two diagrams use different dash conventions.
- The heat map follows your spec: crossovers are dashed.
- `phase_diagram_v3.png` follows de Candia's Fig. 2, where dashed means first-order, so crossovers are dotted there.
- In both, solid marks only g_lin.

---------------------------------------------------------------------------------------------------------------------

## 3. Definitions: every phase and every line

### 3.1 Phases (regions of `phase_diagram_v3.png` a)

| name | definition (operational) | where | de Candia analogue | source |
|---|---|---|---|---|
| **SL**: quiescent | g < g_lin. The silent state is linearly stable: λ_PF of the excitatory branching operator is below α_eff. From a fully active start, all activity dies quickly. At g_lin itself the lifetime is only τ(x = 1) = 19.9 (r = 0) and 16.5 (r = 1) model time for the AL alone, and 19.4–22.0 (r = 0) for the real subsamples. SL and the lower part of M (g_lin < g < the τ = 200 line) cannot be told apart by any T-independent simulation observable: τ is short (about 15–200) on both sides of g_lin. The SL/M edge is the theoretical g_lin; simulation supports it only indirectly (the growth-rate zeros x* follow the chain with its threshold at g_lin, the large-K null has x* = 1.018, and null onsets converge toward g_lin). | g < 0.1632 at every r | SL | `island_lifetimes.json` AL_r*.tau_x1; `redraw_fss.md` §2.2 table |
| **M**: metastable AL (the finite-K rounded onset band) | g_lin < g < g_on(T), with g_on defined by τ_sim(g_on) = T = 3000. Silence is linearly unstable and the mean field has an active AL fixed point. In the stochastic model the AL activity persists for an **exponentially distributed** time τ(g) and then dies, because h = 0 makes silence absorbing. The sub-band τ ≥ 200 is the old "slow window S". This is not a thermodynamic phase: its upper edge moves with T. | r = 0: 0.1632–0.266 (S at 0.234). r = 1: 0.1632–0.344 (S at 0.283). r = 6: 0.1632–0.579 (S at 0.436). | B1 (the AL) rounded by finite size | `phase_complete.json` sim; `longruns.md` (0 KS rejections of exponential lifetimes at 9 g) |
| **B_AL**: sustained, localized, AL-led | g > g_on(3000) and PR < N/4, before the island cascade. Both AL lobes are active together: the late lobe asymmetry is −0.001 to +0.047, and a one-lobe start re-symmetrises in a median 1.0–7.0 model time. Per neuron the rest is nearly silent (mean occupancy 2e-4 to 1.3e-3, against 0.05–0.19 per AL neuron), but in total it carries 51–71% of the activity (`critic_alshare.json`). B_AL is a B1-like state with heavy spill-over, as literature.md §0.1 anticipated; it is not de Candia's B1 with exactly silent modules. | from g_on up to the island-1 1% line (0.382 at r = 0, 0.559 at r = 1, 0.949 at r = 6) | B1 with m = 1 module = the whole AL, the rest = the silent modules (only approximately silent: spill-over) | `multistability.md` §2; `module_occ_P0.json` |
| **B_AL + islands**: cascade | Modules switch on one by one as smeared, driven (imperfect) activations. Island 1 (2,292 neurons) reaches 1% occupancy first, then the median of 30 islands (0.460 at r = 0, 0.847 at r = 1), then the rest of the brain (0.701 at r = 0, 2.37 at r = 1, 7.45 at r = 6). It stays localized (PR < N/4). At r = 0 the number of modules above 1% rises 4 → 8 → 9 → 24 → 30 → 31 between g = 0.41 and 0.52: steep, but resolved, not a jump. | between the dotted 1% lines | B_m with m growing | `module_theory.md` §5; `module_occ_P0.json` |
| **D**: global (SH-like) | Sustained, with participation ratio PR ≥ N/4. | g ≥ 1.47 (r = 0), 2.50 (0.25), 4.44 (0.5), 14.5 (1), 28.9 (1.5). Not reached up to g = 30 at r = 1.75, 2, 3, 4, 6. | SH | `phase_complete.json` sim D |
| *absent*: B1L \| B1R (one lobe active), coexistence phases | Searched for and not found (section 5). | — | B1 with broken L/R symmetry; coexistence regions | `multistability.md` |

Rows r = 2.25, 2.5, 2.75, 3.5, 4.5, 5 and 5.5 were simulated only up to g = 2.6 (hatched in the figure).

### 3.2 Lines

| line | style | definition | values | status |
|---|---|---|---|---|
| **g_lin** | **solid** (the only one) | The exact linear instability of silence: λ_PF(B(g)) = α_eff with B_ij = f(g E_ij), excitatory synapses only. It is r-independent because inhibition does not enter the linearisation of the stochastic process near silence. | 0.1632 at every r | The edge of SL. The exact-linearisation one-mode chain is continuous (transcritical) at g_lin by construction; the neuron-level QMF (net-input closure) is continuous at its own silent edge, 0.159–0.163 at r = 0 moving to 0.193–0.198 at r = 6. The data favour the chain's location (section 4, row 4), so "continuous at g_lin in mean field" holds for the chain, and for the QMF only at r = 0. In the stochastic model it is rounded by finite size and becomes visible only as the crossovers below. It is the only line that does not move with T. |
| **τ = 3000** | dotted (crossover) | τ_sim = 3000. τ_sim is the censored-exponential MLE from a fully active start, with T = 3000 runs pooled. | 0.266 (r = 0), 0.344 (r = 1), 0.579 (r = 6) | Convention. At r = 1, g_half moves 0.351 → 0.366 → 0.386–0.399 for T = 3000 → 8000 → 30000 (`phase_complete.md`, longruns). |
| **τ = 200** | dotted | τ_sim = 200 | 0.234 (r = 0), 0.283 (r = 1), 0.436 (r = 6) | Convention. The window g_on/S is 1.14 (r = 0) and 1.22 (r = 1). |
| **PR = N/4** | dotted | Participation ratio of the occupancy = N/4, among sustained points | values above (D) | Convention; no jump in PR |
| **island / rest 1%** | dotted (thin) | First g at which a module's occupancy among survivors reaches 1% | values above | Driven, smeared activation. At r = 0 every island crosses 1% below its own isolated threshold. At r ≥ 1 the islands with low thresholds cross near or above them. |
| **first-order lines** | dashed | Coexistence or hysteresis that survives longer dwells | **none found** | section 5 |
| theory: one-mode chain τ = 3000 | thin, open squares | Chain onset with the computed K_eff (theory P2, unfitted) | 0.291 (r = 0), 0.613 (r = 1), 2.36 (r = 6) | Matches at r = 0 (onset 9% high). For r > 0 inhibition is overstated: 1.78× (r = 1), 4.08× (r = 6). |
| theory: QMF PR = N/4 | thin, open diamonds | Neuron-level mean field (QMF) participation ratio = N/4 | 1.45 (r = 0; simulation 1.47), 3.01 (r = 0.25; simulation 2.50), 7.67 (r = 0.5; simulation 4.44). None up to g = 30 for r ≥ 0.75. | Too inhibitory for r ≥ 0.25 |
| theory: QMF silent edge | grey ticks | Silent edge of the net-input (rectified) closure, on the up-sweep | 0.159–0.163 (r = 0) to 0.193–0.198 (r = 6) | Moves 1.2× with r, whereas the exact g_lin does not move. The exact g_lin is the right threshold (section 4, row 4). |

Sources: `phase_complete.json` (sim and theory), `module_occ_P0.json`, `qmf_analysis.json` and `module_theory.md` §4.1, all
collected in `synthesis.json`.

### 3.3 de Candia's closure on our modules (`phase_diagram_v3.png` b; theory only)

This is the closure map X_k ← X_k e^(−dt) + (1 − X_k)(1 − e^(−tanh(g Σ_l W_kl X_l)₊ dt)), with W = E^PW − r I^PW. It runs on
31 modules: AL-L, AL-R, 28 spectral islands and the rest (`module_theory.md` §3.1).
- **Closure SL edge** (solid, second order): 0.1645 (r = 0) rising to 0.3863 (r = 6). Unlike g_lin, it moves with r.
- **B1L | B1R**: left-only and right-only AL states coexist in 1,426 of 4,000 cells. They start at g = 0.224 (r = 2.5),
  0.286 (r = 4) and 0.434 (r = 6).
- **First-order intervals** (up/down hysteresis, dashed): 46, at every r from 0.25 to 6.
- **Combinatorial multistability** (islands switching on or off): 2,484 multistable cells, with at most 155 attractors.
  With the QMF's 10% criterion this becomes 2,304 cells and at most 61.
- **No stable fixed point found** in 450 cells.
- **The amount of multistability depends on the partition.** The Leiden partition (P1) gives 76 multistable cells, at most 4
  attractors.
- **The closure is uncontrolled here.** Within-module input varies by more than its mean: median coefficient of variation 1.23
  (excitatory self-input) and 3.49 (cross-module) in P0, against 0 for de Candia's exact case (`module_theory.md` §2.4).

---------------------------------------------------------------------------------------------------------------------

## 4. Q2 in detail: is the steep smooth crossover a second-order transition?

**Answer.** Not as a phase transition of the brain. It is consistent with the finite-size-rounded image of a continuous
onset of one small module at g_lin. The strength of each strand is graded in the table below.

| # | question | evidence | strength |
|---|---|---|---|
| 1 | Is there a sharp transition in some large-size limit, and where? | Configuration nulls sharpen with N. The onset x_on falls 1.336 → 1.061 (r = 0) and 3.71 → 1.336 (r = 1) from N = 2.5k to 167k, Spearman −1 at both r (FSS-1 PASS). The r = 0 extrapolation is consistent with x_∞ = 1, but the free fit ranges 0.93–1.02 with weighting and N range, and fixing x_∞ = 1 costs only Δχ² = 0.5. The r = 1 fit gives an unphysical 0.81 and is not usable. (`redraw_fss.md` §2.2, V3) | **moderate**: the approach is clear, the limit is located only to a few % |
| 2 | Is the onset continuous at mean-field level? | Yes, at its own threshold: the exact-linearisation chain at g_lin by construction, the neuron-level QMF at its silent edge (= g_lin only at r = 0). The neuron-level QMF at r = 0 has one attractor at all 48 g, and its up and down sweeps agree to 1.8e-8 away from the critical grid point. At every r the silent state loses stability without a jump. (`module_theory.md` §4.1–4.2) | **firm for the mean field**. The mean field cannot distinguish a rounded transition from a smooth crossover. |
| 3 | Is there a first-order signature in the stochastic model? | **None found.** All QS histograms are unimodal (interior test, n_AL ≥ 3). At h = 1e-5 the sweep loop areas shrink 6.5–15.5× from dwell D = 50 to 800. At D = 800 every loop is within 3 SE of its noise floor except r = 0 (0.263 ± 0.055 against a floor of 0.080, permutation p < 0.001) and, marginally, r = 0.5 (p = 0.02); both are still shrinking. No coexistence at the 14 loop centres re-run from both branches (smallest late KS p = 0.093), but **only 2 of those 14 centres lie at the onset**. M3: one active attractor in 23 cells. M3b: 0 of 40 mean-field coexistence cells survive noise. (`criticality.md` §4; `multistability.md` §3–5, §10) | **moderate**. Detection limits: loop areas below ≈ 0.08–0.13, and metastability beyond 5,000 model time. Unimodality is weak evidence in a system of K ≈ 10–90. |
| 4 | Is the linear critical point at g_lin and independent of r? | The growth-rate zeros x* of E[n(age)] are 1.186 / 1.345 / 1.722 (uniform seeds) and 1.123 / 1.271 / 1.627 (mode-weighted) at r = 0 / 1 / 6. The finite-K chain with the threshold at g_lin predicts 1.138 / 1.291 / 1.757; the net-input closure predicts 1.418 (r = 1) and 2.102 (r = 6). A free (K, c) fit gives c(1)/c(0) = 0.98–1.00 (closure 0.91) and c(6)/c(0) = 0.94–0.95 (closure 0.84), with a poor fit at r = 6. The one-mode controls recover their chain x*: 1.126 vs 1.137 (K = 76), 1.273 vs 1.287 (K = 37). The large-K null has x* = 1.018. (`criticality.md` §3.1, §10) | **supported at r ≤ 1, approximate (~5%) at r = 6**. Caveat: changing the seed weighting moves x* by 0.06–0.10. |
| 5 | Is the steep part set by a fixed finite module? | Real subsamples of 5k, 20k and 80k neurons that keep the AL give x_on within 1.1% of the full brain (1.632 at r = 0, 2.110 at r = 1), and the AL alone (N = 526) is within 1.8% (1.652 / 2.072). The fitted effective size is K_fit 69.0–72.3 (r = 0) and 36.0–37.2 (r = 1), against the full brain's K_needed of 75.8 and 36.9. Survivor fluctuations at the crossover grow only weakly in the real subsamples (N^0.15 ± 0.06), against N^0.65 ± 0.015 in the r = 0 nulls. These are effective slopes over ≤ 1.8 decades. (`redraw_fss.md` §2.2, §3, V8) | **strong** for "a fixed module; it does not move or sharpen as unrelated neurons are added" |
| 6 | Does the rounding follow the √K law of a finite MF-DP system? | Only with the *fitted* K: τ(x = 1) ∝ K_fit^(0.50 ± 0.01) at r = 0 and K_fit^(0.46 ± 0.04) at r = 1. With the pre-registered *structural* K the within-family slopes are 0.37–0.38 (r = 0) and 0.25 (r = 1), against the chain's own 0.51–0.56. FSS-3: K_fit ∝ N^(0.667 ± 0.027), which fails the pre-registered a ≥ 0.7. (`criticality.md` §6, §10; `redraw_fss.md` §2.3) | **weak / partial**. Q2-c failed as pre-registered at r = 1. The fitted-K version is partly circular. |
| 7 | Is it in the MF-DP universality class (de Candia's claim for their critical lines)? | **Not as a single finite-K MF-DP mode**; the class itself is undetermined at K ≈ 10–90. Measured against one-mode controls run with the same code: <br>• the local slope of ⟨S⟩(T) peaks at 2.35 (r = 0) and 2.24 (r = 1), against 1.57–1.92 for complete-graph controls with K = 37–2000; the nulls peak higher still, at 2.58–2.68; <br>• P(T) needs K_fit 23.1 / 13.7, against K_needed 75.8 / 36.6, with shape deviations of 0.16–0.20 (controls 0.11–0.13); <br>• the quasi-stationary order parameter rises faster than the chain in every fit window; <br>• the field response has no slope-½ window (minimum slope 0.645 at r = 0, x = 1, against 0.45 for the driven chain); <br>• the size exponent is near 3/2 above ~20 spikes (1.46–1.57 with KS-chosen S_min), with an excess of small avalanches below that. <br>Because the nulls deviate the same way, the likely cause is network heterogeneity; that is a hypothesis, not tested. (`criticality.md` §3.2–3.3, §4–5, §10) | **clear negative** for single-mode MF-DP |
| 8 | Where is the module's susceptibility peak? | At r = 0 the χ_AL peak (g = 0.208) is 6.7% above the chain's (0.195), with a similar height. At r = 1 it is 18% above (0.247 vs 0.209). The peak is smooth and finite. (`criticality.md` §4) | consistent at r = 0, misses at r = 1 |

**Exponents against de Candia's MF-DP list** (added by the completeness critic):

| quantity | MF-DP (de Candia) | here | source | reading |
|---|---|---|---|---|
| order parameter | β = 1 | effective QS slope 0.59–1.28 depending on the window (r = 0, 1); same-window chain 0.40–0.79 | criticality §4 | finite-K crossover, no asymptotic window |
| susceptibility | γ = 1 (diverges) | one finite peak: χ_AL 8.3 (r = 0), 7.8 (r = 1) | criticality §4 | divergence is impossible at fixed K |
| size exponent | 3/2 | 1.74–1.83 with S_min = 2; 1.46–1.57 with a KS-chosen S_min of 10–30 spikes | criticality §3.2 | MF-DP-like above ~20 spikes, excess below |
| duration exponent | 2 | not measured as an exponent; P(T) fits the chain only with K 1.7–3.3× below K_needed | criticality §3.2 | none |
| ⟨S⟩(T) | ∝ T² | local slope peaks at 2.10–2.35 near T ≈ 6–10, no plateau; one-mode controls 1.57–1.92 | criticality §3.3 | deviates |
| field response | n ∝ h^½ | no slope-½ window; minimum slope at x = 1 is 0.645 (r = 0) and 0.771 (r = 1), against the driven chain's 0.45 / 0.48 | criticality §5 | deviates |
| finite size | onset → g_lin; τ(g_lin) ∝ √K | null onsets: x_on − 1 ∝ N^−0.42 with the limit fixed at g_lin (free fit 0.93–1.02); τ(x = 1) ∝ K_fit^0.50 (r = 0); structural K gives 0.37 | redraw_fss §2 | partial; the real brain has no size family |

**Pre-registered Q2 criteria** (plan §4.6; `criticality.md` §1, as verified):
- **a (location in [0.97, 1.03]): FAIL as written.** The criterion ignored the finite-K offset that the plan's own chain
  predicts. Against the chain it passes at r ≤ 1.
- **b (finite-K MF-DP): FAIL.**
- **c (√K rounding): FAIL at r = 1** with the structural K; passes literally at r = 0 only because two offset families are
  merged; passes at both r only with the post-hoc fitted K.
- **d (exponents): FAIL.**
- **e (not first order): PASS** (QS unimodality; stochastic hysteresis is covered by multistability, row 3).

**What would change this answer.** These would argue for a genuine discontinuity:
- loops that stop shrinking at dwell times above 800;
- a dedicated coexistence test at the τ = 3000 onset for each r (not run: M2b's centres were mostly away from the onset);
- metastable states living longer than 5,000 model time.

Nothing in hand points that way.

---------------------------------------------------------------------------------------------------------------------

## 5. Q1 in detail: which of de Candia's structures exist here

| de Candia structure | closure on our modules (theory) | neuron-level mean field (QMF) | stochastic model (simulation) |
|---|---|---|---|
| SL and its second-order edge | Yes. The edge moves with r: 0.1645 → 0.3863. | Yes. Continuous edge at 0.159–0.198. | Yes, at g_lin = 0.1632 (r-independent), rounded |
| B1 = one module active, the others silent | B_AL from 0.170 (r = 0) | AL first; the rest is not silent (spill-over) | **B_AL**, with the whole AL as the module and both lobes active |
| B_m cascade | A continuous cascade at r = 0; combinatorial multistability at r > 0 | Smeared cascade | Smeared, driven cascade (dotted 1% lines) |
| SH | All 31 modules active from g = 1.80 (r = 0) | QMF D 1.45 (r = 0) | D: 1.47 (r = 0), 14.5 (r = 1); none up to 30 at r ≥ 1.75 |
| **Broken L/R symmetry (B1L \| B1R)** | **Yes**, 1,426 cells, r ≥ 2.5 | **No** lobe-level B1 at any of 9 r. At r = 3–6 there are only weakly tilted coexisting AL states, with asymmetry −0.108 to +0.170 at r = 6 (B1 would be ±1). | **No**: 0 of 48 points meet any of the three pre-registered criteria. At the 17 closure-window points that have survivors, the two lobes end in the same symmetric state. At 4 of the 21 closure-window points too few replicas survived to test. |
| **First-order lines** | **Yes**, 46 hysteresis intervals | Fixed-point hysteresis only at r = 0.5 (20 points, g 1.27–6.0). Coexisting fixed points at r = 0.5 and 1.5–6 (10% criterion) or every r ≥ 0.25 (1%). | **None**: see section 4, row 3 |
| Avalanche shape on SL–B1 left-skewed (≈ +0.4 at T ≈ 40) | — | — | Nearly symmetric: skewness +0.026 / +0.032 / +0.040 at T ≈ 10 (r = 0 / 1 / 6), |skewness| ≤ 0.06 at T ≈ 35–45 (largest: r = 1, T ≈ 44, +0.057 ± 0.020). The minority lobe's share *rises* late at every r (+0.064 [0.046, 0.083] at r = 0, +0.061 [0.043, 0.079] at r = 6): cooperative spread, not competition. |
| No oscillations (equal modules) | No stable fixed point in 450 cells (mostly non-stationary) | **Non-stationary (oscillating) mean field** for r ≥ 0.5 above g = 1.0–2.2 (first non-stationary g at dt = 0.05), robust to dt. It is carried by 26k–43k neurons, of which optic-lobe intrinsic neurons are the largest group (43–89%), a majority only at r = 0.5–2 and 4, with small amplitude at onset. | **not tested** |

Sources: `module_theory.md` §3–4 and §12; `multistability.md` §2–6 and §10; `criticality.md` §3.4–3.5.

**Why the broken symmetry disappears.**
- The exact linearisation of the stochastic process near silence contains only excitatory branching. Inhibition acts only on
  neurons that are already receiving excitation, because the rate is β·tanh(s)₊.
- The two lobes are coupled excitatorily, as near-copies (Perron-weighted E_LR/E_RL = 0.080/0.066).
- The competition ratio of the exact two-mode normal form is Γ ≈ 0.003, while winner-take-all needs Γ > 1
  (`theory_notes.md` §1.5).
- The closure's B1 instead comes from treating each lobe as a block in which every neuron receives the same net input. The
  connectome's modules are far from that (CV 1.2–8.6).

**What the literature says** (from `literature.md` §0; no new numbers):
- de Candia's exact reduction needs an equitable partition, in which all neurons of a module get the same input (Schaub et al. 2016). Connectome modules are far from that (CV 1.2–8.6), so B1L | B1R and first-order lines were not expected to survive outside the closure. They did not.
- A localized principal eigenvector gives no strict threshold: activity stays on a finite set of nodes and spreads through a smooth crossover (Goltsev et al. 2012; Ferreira et al. 2016; St-Onge et al. 2021).
- A Griffiths phase needs a broad distribution of rare-region sizes (Vojta 2006; Moretti & Muñoz 2013). One dominant region gives exponential lifetimes, which is what was measured.
- Finite logistic populations have an extinction time ∝ √N at threshold and exponential in N above it (Doering, Sargsyan & Sander 2005). This is the rounding used throughout the report.
- Known first-order mechanisms in these models:
  - a convex activation function (Alvankar Golpayegan & de Candia 2023);
  - E/I populations receiving different inputs (Piuvezam et al. 2023);
  - inhibitory nodes (Corral López et al. 2022);
  - net-inhibitory coupling between modules (de Candia et al. 2026).

  Our concave tanh₊ rules out the first, and none of the tests found a first-order signature.
- Power laws alone do not establish criticality (Touboul & Destexhe 2017; Priesemann & Shriki 2018).

The Griffiths question is closed negatively for the same structural reason. 87% of the 395 non-AL top-400 modes have most of
their support inside one shared VNC/central region (mode 2's), so they are not disjoint rare regions (`module_theory.md` §2.1).

---------------------------------------------------------------------------------------------------------------------

## 6. Finite-size check (Q3b), numbers

Source: `redraw_fss.md` §2 and Verification; `fss_analysis.json`.

| | N = 2.5k | 5k | 10k | 20k | 40k | 80k | 167k |
|---|---|---|---|---|---|---|---|
| null x_on, r = 0 (seed mean) | 1.336 | 1.282 | 1.211 | 1.145 | 1.122 | 1.079 | 1.061 |
| null x_on, r = 1 | 3.71 | 2.76 | 2.39 | 1.90 | 1.745 | 1.474 | 1.336 |
| real subsample x_on, r = 0 / 1 | – | 1.644 / 2.106 | – | 1.640 / 2.101 | – | 1.631 / 2.119 | full brain 1.632 / 2.110 |

- **FSS-1 PASS.** The null onset decreases strictly at both r (Spearman −1).
- **FSS-2 PASS.** Every real subsample is within 1.1% of the full brain. In absolute g, the subsamples give 0.266–0.271
  (r = 0) and 0.343–0.348 (r = 1).
- **FSS-3 mixed.** K_fit ∝ N^(0.667 ± 0.027) at r = 0 and N^(0.574 ± 0.022) at r = 1, both below the pre-registered 0.7: a
  FAIL. The unfitted K_nc scales as N^(0.89 ± 0.05).
- **Steepness.** The model-free secant steepness of the null crossover rises 29.5–33.8 (2.5k) → 55.7–57.8 (167k) and saturates
  above 40k. The measured steepness is −1% to +30% relative to the one-mode chain's (like-for-like MLE; model-free secants −6% to +39%) (V4).
- **Lifetime at g_lin.** The chain predicts the lifetime at g_lin within 1.00–1.17× for the r = 0 nulls. The r = 1 nulls sit
  1.32–2.05× above it, and the real subsamples 1.13–1.45× above it (V5).
- **Refined number.** The full configuration null's r = 0 onset is now g = 0.8778 [0.8765, 0.8811]. It was 0.867, known only
  to an 8% grid cell.

---------------------------------------------------------------------------------------------------------------------

## 7. Theory against simulation, in one place

| theory | where it agrees | where it fails |
|---|---|---|
| g_lin (exact linearisation; P1) | It lies below every simulated boundary at every r. x* follows the chain with this threshold at r ≤ 1, within ~5% at r = 6. Large-K nulls sit at x* = 1.018. | — |
| One-mode chain with computed K_eff (P2) | r = 0: S 0.245 vs 0.234, onset 0.291 vs 0.266 | r > 0: inhibition overstated. The onset ratio is 1.78 (r = 1) and 4.08 (r = 6); reproducing the lifetimes needs K about 2× K_eff for r ≥ 0.25. Cause open (`phase_complete.md`). |
| Same chain with K fitted per r (K_needed) | S, onset and the T-drift at r = 0 and 1 (e.g. r = 1: chain 0.346 / 0.395 at T = 3000 / 30000, against measured 0.351 / 0.386–0.399) | Avalanche shapes and statistics (section 4, row 7). K is fitted to the same lifetimes. |
| QMF (neuron-level mean field) | D at r = 0 (1.45 vs 1.47); no lobe-level B1 (matches the simulation) | D too inhibitory for r ≥ 0.25. Its coexisting fixed points (40 cells) are removed by noise. The silent edge moves with r (net-input closure). |
| de Candia closure on modules | the SL edge exists | B1L \| B1R and first-order lines are refuted wherever activity survives. Its SL edge is up to 2.4× too high at r = 6 (0.386 vs g_lin 0.163). |

Sources: `phase_complete.md`; `criticality.md` §3.1; `module_theory.md` §3–4; `multistability.md` §2, §5; heat-map panel C
(`redraw_fss.md` §1).

---------------------------------------------------------------------------------------------------------------------

## 8. Caveats, weak points and open items (stated plainly)

1. **Model 1 only.** Everything here uses the original model: no synaptic depression, h = 0, c = 1/745.91.
   - The verified AL-defect study (`results/aldef/REPORT.md`) found that M1's self-sustained AL is a **defect of the
     model's synapse dynamics**.
   - With per-spike depression the onset rises 2.3–4.8×, and the first sustained state is brain-wide, with the AL carrying
     only 5–11% of the activity (CLAUDE.md, STATE.md "AL-defect" entries).
   - The phases M and B_AL above are therefore properties of M1. They should not be read as predictions for the fly.
   - The Model-2 program (`results/m2/`) is redoing this diagram.
2. **"Second order" is not established for the stochastic system.**
   - Continuity rests on the mean field, on unimodal QS histograms in a small system (K ≈ 10–90), and on hysteresis loops
     that shrink.
   - The data are not described by a single finite-K MF-DP mode; the universality class itself is undetermined.
   - The √K rounding law holds only with a fitted K.
3. **First-order tests have limits.**
   - Loop areas below the noise floor (≈ 0.08–0.13 at h = 1e-5) cannot be seen, and dwell times went only to D = 800.
   - Only 2 of the 14 M2b coexistence centres lie at the onset.
   - The r = 0, h = 1e-5 loop is still open at D = 800, though it is shrinking.
   - The h = 0 down-sweep extinction sits about one grid step below the lifetime-map prediction at all 6 r (sign test
     p = 0.031).
4. **FSS-3 failed** as pre-registered (a = 0.667 < 0.7). The null-onset extrapolation does not locate the limit to better than
   a few %.
5. **Several analyses were not pre-registered**, and their numbers carry less weight:
   - the mode-weighted seeds and the free (K, c) fit (criticality);
   - the fine-grid FSS runs, the window steepness s_fit and χ_win (redraw_fss);
   - the noise-floor criterion and the permutation test (multistability).
6. **Untested mean-field features.** The QMF predicts a non-stationary (oscillating) mean field for r ≥ 0.5 above g = 1.0–2.2 (first non-stationary g at dt = 0.05).
   Its amplitude is small at onset, and whether the stochastic model has a corresponding rhythm was not tested.
7. **Scope of the island simulations.** Only the AL and islands 1–5 were simulated alone (MT5), and only at r = 0 and 1.
   - Islands 2–30 of partition P0 are fragments of one shared region, so module identities in P0 are partly arbitrary.
   - P1 (Leiden) and P2 give the same leading structure: one AL-containing module at g_lin ≈ 0.164–0.169, and every other
     module needs g ≥ 0.54 alone.
8. **Documentation fix.** swc.py's normalisation is c = 1/745.91: the mean over the 166,499 neurons with nonzero in-strength.
   The project notes' "1/742.7" is total in-strength divided by all N. The difference is 0.43%, and every pd3 simulator used
   swc.py's value.

---------------------------------------------------------------------------------------------------------------------

## 9. Files and jobs

**Synthesis (this step).**
- Code: `code/pd3_synthesis.py`.
- Output: `results/pd3/synthesis.json` (76 sourced numbers).
- Figures: `figs/pd3/phase_diagram_v3.png`, `q2_second_order_evidence.png`, `fss_summary.png`.
- No GPU jobs.

**Program reports** (verified): `results/pd3/{redraw_fss,module_theory,criticality,multistability}.md` and their `.json`.
- Verification scripts: `code/pd3_verify_*.py`.
- Verification outputs: `results/pd3/verify_*.json` and `results/pd3/redraw_fss/verify/`.
- Research inputs: `results/pd3/literature.md`, `theory_notes.md`, `plan.md`.

**GPU jobs** (all completed, none left in the queue; details in each program's `JOBS.md`).

| program | jobs | notes |
|---|---|---|
| redraw_fss | 175132 (smoke); 175172 (54-task FSS array); 175681, 175816, 176126, 176564, 176756 (fine-grid batches) | reruns: 176015 (preempted 175172_35); 175816_0 (preempted 175681_8) |
| module_theory | 175683 + 176342 (QMF, 9 r); 176386 and 176402 (closure P0 and P1, on ellis); 175546 (MT5, 12 tasks); 177888 (MT3b); CPU 175284, 176327, 176328 | cancelled or superseded: 175264, 175376, 175422, 175633, 175507, 176013, 176275, 176515 |
| criticality | 178550, 178697, 178922 (smoke tests); 178665, 178827 (V0); 179269 + 179638 + 180126 (avalanches: 63 outputs); 179306 + 181954 (QS, 14); 178729_20-21 (C4); 179627 (field, 6) | cancelled for bugs: 178729_0-19 (stale QS store; float32 field drive) |
| multistability | 179161 (smoke); 179202 (validation); 179561 (M1); 179562 (M2a, 42 tasks); 179563 (M2c); 179564 (M3); 179565 (M3b); 180689, 180780, 181423 (M2b) | 132 outputs, no failures, ~19.4 GPU-h |

---------------------------------------------------------------------------------------------------------------------

## Critic review (completeness critic, 2026-11-09)

**Verdict.**
- The report answers all three of your questions: Q1, Q2, Q3a and Q3b.
- Its numbers match the program reports and the raw data: six key numbers were recomputed independently and all reproduce.
- Its grading of the Q2 evidence is sound.
- Below: one factual regression, one claim that is misleading in total activity, three over-strong sentences, minor numeric precision fixes, two additions, and three relabelled figures.

### A. Numbers recomputed from raw outputs
- Script: `code/pd3_critic_recompute.py`, written for this check. It has its own isotonic fit (PAVA), crossing rule, censored-exponential estimate and Perron root, and imports no project code.
- Output: `results/pd3/critic_recompute.json`.

| quantity | report | recomputed | raw source |
|---|---|---|---|
| g_lin | 0.1632 (0.16315) | 0.1631526 (published 0.1631527) | connectome cache: λ_PF(B(g)) = α_eff = 0.90635, 1/c = 745.91 |
| τ = 200 / τ = 3000 crossings, r = 0 | 0.234 / 0.266 | 0.23358 / 0.26622 | 93 pooled g points (pd/, pd2/) |
| same, r = 1 | 0.283 / 0.344 | 0.28285 / 0.34433 | 173 pooled g points |
| D (PR = N/4), r = 0 | 1.47 | 1.4708 | occupancies of the T = 3000 runs |
| null x_on (seed mean), N = 2.5k → 167k | 1.336 → 1.061 (r = 0); 3.71 → 1.336 (r = 1) | 1.3357 → 1.0609; 3.7131 → 1.3361 | ext_step of the 78 fss_*.npz files |
| real subsamples: maximum deviation from the full brain | 1.1% | 1.10% (r = 0), 0.99% (r = 1) | same |
| M1 late lobe asymmetry, 48 points; criterion c1 | −0.001 to +0.047; 0 | −0.00096 to +0.0466; 0 | al_sym_r*.npz, modc |
| island-1 1% crossing, r = 0 / 1 / 6 | 0.382 / 0.559 / 0.949 | 0.3819 / 0.5588 / 0.9485 | occupancies + partition_P0 |

### B. Figures (fixed; looked at after regeneration)
`code/pd3_critic_figs.py` makes the fixes. It applies exact string replacements to in-memory copies of `pd3_synthesis.py` and `pd3_redraw.py`; both originals are unchanged. It also records every replacement and keeps the pre-critic PNGs:
- replacement log: `results/pd3/critic_regen/critic_figs_log.json`;
- pre-critic PNGs: `figs/pd3/verify_orig/*_precritic.png`.

The regenerated heat-map arrays (`results/pd3/critic_regen/`) are bit-identical to `results/pd3/redraw/`.

1. **`phase_diagram_v3.png`**
   - The suptitle called g_lin a "continuous (second-order, finite-size rounded) line". It now reads "linear instability of silence (continuous in mean field; rounded by the finite AL module in simulation, not a sharp transition)".
   - The g_lin legend entry "the only critical line" is now "linear instability of silence (theory; r- and T-independent)".
   - The yellow M1 markers are now labelled "inside the 2-module closure's B1 window". They use theory_module2's windows, not the 31-module closure drawn in panel (b).
   - The B_AL legend entry now carries the AL activity share (29–49%; section C.2).
   - The median-island and rest-of-brain 1% lines were drawn straight through rows r = 2.25–2.75, 3.5 and 4.5–5.5. Those rows were simulated only to g = 2.6, so the crossings were never measured there. The lines are now broken at those rows, and the hatch note says so.
2. **`phase_lifetime_heatmap.png` and `phase_lifetime_heatmap_nulls.png`**
   - The g_lin label "mean-field critical line (silence loses stability; r-independent)" contradicted §3.2: the mean-field (QMF) silent edge moves with r (0.159 → 0.198).
   - It now reads "linear instability of silence (exact linearisation; r-independent)".

### C. Corrections to the report text
All nine corrections were applied to the text above when this file was saved (by the main session, 2026-11-09):
the oscillation-row regression, the B_AL spill-over share, three over-strong sentences, the two mean fields, the SL
operational definition, precision fixes, the exponent table (section 4) and the literature paragraph (section 5).

**Files.**
- Code: `code/pd3_critic_recompute.py`, `code/pd3_critic_alshare.py`, `code/pd3_critic_figs.py`.
- Results: `results/pd3/critic_recompute.json`, `results/pd3/critic_alshare.json`, `results/pd3/critic_regen/`.
- STATE.md now ends with a dated "pd3 completeness critic" entry.
- No GPU jobs were run.
