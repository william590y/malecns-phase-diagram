# Program `redraw_fss`: lifetime heat map and finite-size scaling (pd3, 2026-11-09)

This program covers plan section 2 (`results/pd3/plan.md`). Every number below is read from one of the files named next to it, or was computed by the code listed under Files.
- Machine-readable summary: `results/pd3/redraw_fss.json`, built by `code/pd3_redraw_fss_summary.py` from `results/pd3/redraw/heatmap_meta.json` (part A) and `results/pd3/redraw_fss/fss_analysis.json` (part B).
- Job ids: `results/pd3/redraw_fss/JOBS.md`.

## 0. What this answers, and how strongly

| user question | what this program delivers | strength |
|---|---|---|
| **Q3a**: redraw the diagram as a continuous lifetime heat map. g_lin is the only solid line; τ = 200, τ = 3000 and PR = N/4 are dashed "crossovers, not phase walls". | Done: `figs/pd3/phase_lifetime_heatmap.png` (real network, 23 r) and `phase_lifetime_heatmap_nulls.png` (both nulls, 6 r each). The interpolated field reproduces every verified boundary to within 0.4% in g. | Complete. This is a re-presentation of verified data; no new physics. |
| **Q3b**: finite-size check. Nulls should sharpen with N; the real crossover should not. | 54 GPU decay runs (configuration nulls at 7 sizes × 3 seeds, real subsamples that keep the AL at 3 sizes × 2 seeds, r = 0 and 1), plus 24 fine-grid runs. | **FSS-1 passes** (nulls sharpen; Spearman −1 at r = 0 and r = 1). **FSS-2 passes** (the real crossover sits within 1.1% of the full-brain value at every N). **FSS-3 is mixed**: 2 of 3 sub-criteria pass; K_fit ∝ N^0.67 ± 0.03 fails the pre-registered a ≥ 0.7. |
| **Q2**: is the steep but smooth crossover a second-order transition? | The same dynamics sharpens toward g_lin when the module size K grows with N (nulls): x_on falls from 1.336 (2.5k) to 1.061 (167k), and an extrapolation is consistent with x_∞ = 1 (fixing x_∞ = 1 costs Δχ² = 0.5), but the free-fit value depends on weighting and N range (0.93–1.02; see Verification V3), so "1.007 ± 0.010" is not a precise location. In the real brain the rounding is set by a fixed module of about 70 units (r = 0) or about 36 (r = 1), which does not change across the realsub family (5k–80k, AL kept). The one-mode logistic chain with one fitted K per graph reproduces the slow-window lifetimes, and the lifetime at g_lin for the r = 0 nulls; it under-predicts the lifetime at g_lin by 1.3–2.1× for the r = 1 nulls and 1.1–1.45× for the real subsamples, and the measured window steepness is a median 8–11% steeper than the chain's (Verification V4, V5). | **Strong** for "a fixed-size module sets the crossover; it does not move or sharpen as unrelated neurons are added". **Moderate** for "it is the rounded image of a transition at g_lin" (null onsets approach g_lin; continuity itself is assumed through the chain and not tested here). **Not supported here** for "MF-DP class specifically": the chain is MF-DP by construction, K is fitted, no exponent is measured, and the criticality program (`criticality.md`, not yet verified) reports avalanche and QS statistics that deviate from single-mode MF-DP. |
| **Q1**: de Candia-style diagram | Panel A is the continuous version of the (g, r) diagram. One critical line (g_lin) is drawn and no first-order lines are drawn; first-order behaviour was not tested in this program (multistability's job). This program does not test broken symmetry (module_theory and multistability do). | Indirect only. |

## 1. Part A: continuous lifetime heat map (`code/pd3_redraw.py`, CPU, no new simulation)

**Inputs: identical to `analysis2_phase.py`.**
- The same 70 simulation files and the same pooling (T_c = 3000; at r = 1 the T = 8000 and 30000 sweeps are truncated at the last record ≤ 3000).
- Checked: the file list equals `phase_complete.json` `inputs.simulation`.
- The per-row point counts equal `phase_complete.json` for all 35 rows: 23 real, 6 configuration null, 6 weight-shuffled null (`heatmap_meta.json` `rows`).
- Real network: 1,568 pooled (g, r) points, 1,114 of them with zero deaths.

**Construction (as specified).**
- τ_sim = E/d. Isotonic ln τ in g with weights d + 0.5, using analysis2's E/max(d, 0.5) regularisation.
- Linear interpolation in ln g onto 320 ln-uniform points in [0.1, 30] (step 0.01788 in ln g).
- No interpolation across r: each r is a band between the midpoints of its neighbours. Cells outside a row's simulated range are hatched as "not simulated".
- Panel B: isotonic PR among the points at or above the τ = 3000 crossing (the analysis2 D rule), interpolated in ln g.

**Consistency check (spec item 7).** I ran a matplotlib `contour` of the interpolated ln τ field at ln 200 and ln 3000 and read off each row (`heatmap_meta.json` `consistency_check`):

| graph | max abs(Δ ln g), τ = 200 | max abs(Δ ln g), τ = 3000 | rows inside the verified 95% total band |
|---|---|---|---|
| real | 0.0040 (r = 6) | 0.0039 (r = 1.5) | 23/23 and 23/23 |
| configuration null | 0.0020 | 0.0029 | 6/6 and 6/6 |
| weight-shuffled null | 0.0006 | 0.0003 | 6/6 and 6/6 |

- All are well below one grid step (0.0179). No row has more than one crossing. The acceptance criterion holds: every dashed line coincides with its verified boundary within its band.

**Figure** (`figs/pd3/phase_lifetime_heatmap.png`; I looked at it).
- Panel A shows log10 τ_sim (cividis), with a flat light colour where the isotonic τ ≥ 3000 ("sustained over T = 3000").
- Panel B shows log10(PR/N) in the sustained region (viridis), grey elsewhere.
- Lines:
  - g_lin = 0.1632 is the only solid line;
  - simulated τ = 200, τ = 3000 and PR = N/4 are dashed (Okabe–Ito vermillion, blue, reddish purple), each with its 95% total band and a "(crossover)" label;
  - the theory P2 τ lines and the QMF PR = N/4 line are dotted.
- Panel C (T-drift at r = 1):
  - measured g_half = 0.3508 (T = 3000), 0.3659 (8000) and 0.386–0.399 (30000) (`longruns.md`);
  - the K = 36.9 chain is drawn as a continuous curve (theory λ(g) from `results/pd/branch_real_dense_part*.json`). It crosses at 0.3456 (T = 3000) and 0.3944 (T = 30000), reproducing `theory_bd.json` `real_crossings` (0.3457 and 0.3947).
  - The apparent onset rises with log T while g_lin does not move.
- Nulls figure (`phase_lifetime_heatmap_nulls.png`): their own g_lin are 0.8287 and 0.6630. On the configuration null, PR = N/4 sits at the onset at every r and is labelled "(at the onset)".

**Notes on the flat "sustained" colour.**
- It marks cells where the isotonic τ ≥ 3000, not only points with zero deaths. 93.1% of the real sustained cells (97.7% null, 97.6% weight-shuffled) have a zero-death nearest point.
- No cell whose nearest point has zero deaths lies below τ = 3000 (`heatmap_meta.json` `sustained_cells_*`).

## 2. Part B: finite-size scaling

### 2.1 Graphs, validation and runs

**Configuration nulls (`code/pd3_fss_graph.py`, kind `cnull`).**
- The exact `pd3_theory_nullK.py` construction. The full-size seed 0 is the swc.py null.
- Validation, 22 (graph, r) runs (all theory_nullK rows × 2 r): λ_PF(E), K_nc(E), 1/IPR, c and the edge count reproduce `theory_nullK.json` to a maximum relative deviation of 4.4e-16 (λ) and 1.2e-15 (K), with identical edge counts (`redraw_fss.json` `validation_summary`).
- Exact g_lin of the full null: 0.8286651, against `phase_complete.json` 0.8286653.

**Real subsamples (`realsub`).**
- The 526 AL-module neurons plus a random rest; induced subgraph; full-brain c.
- Their leading mode at g_lin keeps ≥ 99.6% of its |v|² on the AL. g_lin is 0.16374–0.16426, against 0.16315 for the full brain. K_nc(g_lin) is 66–68.
- For the nulls, the AL share of |v|² is 0.013–0.075.

**Simulation (`code/pd3_fss_sim.py`).**
- An adapted copy of swc.py decay: fully active start, h = 0, dt 0.2, T = 3000, B = 64. The exact extinction step is stored per replica.
- x = g/g_lin: r = 0 uses 30 points in [0.90, 2.50]; r = 1 uses 36 points in [0.90, 8.0].
- All 54 tasks finished, on A40, 3090, A5000, A6000, 6000 Ada and PRO 6000 Blackwell GPUs. The longest took 1 h 11 min (full-size null, r = 1).

**Fine-grid follow-up (`code/pd3_fss_fine.py`, not in the spec; see Deviations).**
- 16 extra points across each coarse run's [x_S lower bracket, x_on upper bracket], for every null with N ≥ 20k at both r (24 tasks).
- Pooled with the coarse points per graph.

### 2.2 Onset vs N (pre-registered FSS-1, FSS-2)

Seed means of x_on (τ_sim = 3000; isotonic + ln interpolation; `fss_analysis.json` `seed_means`). The bracketed CI is the 2,000-resample replica bootstrap of the seed mean. The seed SD (graph-to-graph scatter) is listed separately and dominates at small N.

| N | null r = 0: x_on mean [CI] (seed SD) | K_nc(g_lin) | K_fit | τ(x = 1) | null r = 1: x_on mean (seed SD) | K_fit (r = 1) |
|---|---|---|---|---|---|---|
| 2,500 | 1.336 [1.331, 1.341] (0.045) | 187 / 181 / 103 | 201 / 207 / 146 | 29.5 / 34.6 / 28.3 | 3.71 (0.30) | 17.1 / 14.4 / 14.4 |
| 5,000 | 1.282 (0.051) | 214 / 252 / 105 | 269 / 293 / 175 | 37.4 / 38.0 / 28.6 | 2.76 (0.29) | 22.5 / 26.0 / 18.7 |
| 10,000 | 1.211 (0.028) | 434 / 216 / 351 | 421 / 302 / 377 | 46.6 / 41.1 / 48.1 | 2.39 (0.22) | 30.7 / 23.8 / 31.1 |
| 20,000 | 1.145 (0.020) | 421 / 884 / 891 | 523 / 749 / 773 | 53.4 / 58.6 / 63.0 | 1.90 (0.14) | 38.2 / 51.4 / 52.5 |
| 40,000 | 1.122 (0.009) | 919 / 1194 / 824 | 847 / 988 / 860 | 62.8 / 69.1 / 65.5 | 1.745 (0.012) | 57.7 / 57.2 / 60.3 |
| 80,000 | 1.079 (0.002) | 2564 / 2775 / 3093 | 1743 / 1796 / 1880 | 92.4 / 91.2 / 100.7 | 1.474 (0.032) | 97.7 / 109.1 / 115.9 |
| 167,216 | 1.061 [1.060, 1.062] (0.0015) | 5631 / 5325 / 5192 | 2868 / 2606 / 2648 | 131.4 / 116.2 / 124.8 | 1.336 (0.002) | 179.0 / 174.0 / 174.5 |
| **realsub** 5k / 20k / 80k | 1.644 / 1.640 / 1.631 | 66 / 66 / 68 | 69.0–72.3 | 19.4–22.0 | 2.106 / 2.101 / 2.119 | 36.0–37.2 |
| AL module alone (526; `module_theory/island_lifetimes.json`) | 1.652 | – | – | 19.9 | 2.072 | – |
| **real full brain** (`phase_complete.json`) | 1.632 | – | 75.8 (K_needed) | – | 2.110 | 36.9 (`longruns.md`) |

**FSS-1 (nulls sharpen): PASS at both r** (`decisions.FSS1_r0/r1`).
- At r = 0 the seed-mean x_on decreases strictly over all 7 sizes (Spearman −1.0); x_on(167k) = 1.061 ≤ 1.10 and x_on(10k) = 1.211 ≥ 1.15.
- At r = 1 it decreases strictly as well (Spearman −1.0), from 3.71 to 1.34.

**FSS-2 (real does not sharpen): PASS** (`decisions.FSS2_r0/r1`).
- Every realsub seed lies within 1.1% of the full-brain x_on: maximum abs deviation 0.011 at r = 0 and 0.0099 at r = 1.
- The same holds in absolute g. realsub onsets are 0.266–0.271 (r = 0) and 0.343–0.348 (r = 1), all inside the full brain's verified intervals [0.260, 0.278] and [0.340, 0.350].
- The AL module alone (N = 526, module_theory MT5) gives 1.652 / 2.072, i.e. +1.2% / −1.8%.
- From N = 526 to 167,216 the real crossover therefore does not move.

**Supplementary (not pre-registered): where the null onset extrapolates** (`redraw_fss.json` `supplementary_onset_extrapolation`).
- A fit of x_on = x_∞ + A N^−b to the r = 0 seed means (weights: between-seed SE) gives x_∞ = 1.007 ± 0.010 and b = 0.45 ± 0.05 (χ² 11.5 on 4 dof). ~~The sharp limit is at g_lin.~~ *[Verification V3: this location is not robust. χ²/dof = 2.9, so the quoted ±0.010 is too small (≈ ±0.017 after rescaling); the free-fit x_∞ is 0.933 ± 0.062 unweighted, 1.020 ± 0.009 without the 2.5k point, 1.024 ± 0.009 for N ≥ 10k and 0.83 ± 0.16 without the 167k point. Fixing x_∞ = 1 raises χ² only from 11.48 to 11.97. Correct reading: the r = 0 null onsets approach g_lin and are consistent with a limit at g_lin, but the extrapolation cannot place the limit to better than a few percent (`results/pd3/redraw_fss/verify/verify_redraw_fss.json` onset_extrapolation; `figs/pd3/verify_redraw_fss.png` A).]*
- With x_∞ fixed at 1, (x_on − 1) ∝ N^−0.42 at r = 0 and N^−0.49 at r = 1.
- At r = 1 the three-parameter fit gives x_∞ = 0.81 ± 0.06. That is below 1, where the chain is subcritical, so this functional form does not identify the r = 1 limit. The r = 1 data show only the monotone approach.

**Theory overlays** (`fss_onset_vs_N.png`).
- At r = 0 the chain with each realisation's own R(x) and unfitted K_nc predicts x_on with median abs relative error 1.6% (max 11.9%, for the low-K seed-3 graphs at 2.5k and 5k).
- The miss has a systematic sign. The chain is too high at small N, with (x_on − 1) ratio 1.08–1.47. It is too low at full size, with ratio 0.70–0.72, because K_fit/K_nc falls from 1.07–1.67 (N ≤ 5k) to 0.49–0.51 (167k).
- The κN rule of `theory_bd.json` (r = 0) tracks the r = 0 data.
- At r = 1 the κN rule (κ from the full null's K_needed 181) **fails at small N**. It predicted no onset below x = 8 for N ≤ 5k and 5.74 at 10k; the simulations give 3.71 (2.5k), 2.76 (5k) and 2.39 (10k). The reason: the effective K under inhibition grows as N^0.57, not ∝ N. K_fit(r = 1) is 14–17 at 2.5k against κN = 2.7.
- At full size the r = 1 K_fit is 174–179, matching the earlier K_needed of 182 (164–186) (`phase_complete.md`).

**Refinement of an earlier number.**
- The full configuration null at r = 0 (seed 0 = the swc.py null) now has onset g = 0.8778 [0.8765, 0.8811] and S g = 0.850.
- Earlier the onset was 0.867, known only to the 8% cell [0.838, 0.905] (`phase_complete.md`, critic). The new value lies in that cell; the theory onset 0.866 is 1.4% below it.
- At r = 1: 1.1056 [1.0983, 1.1106], against 1.107 [1.104, 1.133] earlier.

### 2.3 Mechanism (pre-registered FSS-3) and lifetime at g_lin

`decisions.FSS3_r0/r1`; nulls only unless stated otherwise.

| sub-criterion | r = 0 | r = 1 | verdict |
|---|---|---|---|
| K_fit ∝ N^a, a ≥ 0.7 | a = 0.667 ± 0.027 (n 21, R² 0.97) | a = 0.574 ± 0.022 | **FAIL** at the pre-registered threshold. K_nc itself scales as N^0.89 ± 0.05 |
| τ(x = 1) ∝ K^(0.5 ± 0.1) | K_fit^0.505 ± 0.011 (R² 0.99); with realsub 0.483 ± 0.009. Unfitted K_nc: K_nc^0.372 ± 0.011. *[V5: the chain's own local exponent over the same K_fit range is 0.512; the r = 0 null points lie 1.00–1.17× the chain]* | K_fit^0.405 ± 0.010 (just inside); K_nc^0.25. *[V5: the chain's own exponent over this K range is 0.545, and the r = 1 null points lie 1.32–2.05× above the chain: nominal pass, but not agreement with the chain]* | **PASS** with K_fit (r = 1 nominal only); **fails** with K_nc. The pre-registration did not say which K; the C4 sub-modules it named were not included here (criticality reports them separately) |
| steepness rises with K_fit | Spearman(s_fit, K_fit) = 0.89; spec s1000: 0.84 | 0.97; s1000: 0.51 | **PASS** |

**Reading FSS-3.**
- The lifetime at each graph's own g_lin was not used to fit K: the K_fit window is 200 ≤ τ ≤ 3000, while τ(x = 1) is 14–131. So τ(x = 1) ∝ K_fit^0.505 is an out-of-sample √K law.
- The exact chain at R = 1, with no free parameter beyond K, passes through the **r = 0** null points (ratio τ_sim/τ_chain 1.00–1.17). *[V5: it does not pass through the r = 1 null points (1.32–2.05× above, largest at small K) or the real subsamples (1.13–1.30× at r = 0, 1.24–1.45× at r = 1); `figs/pd3/verify_redraw_fss.png` C.]*
- ~~Window steepness collapses onto the same chain. s_fit ... lies on the chain's s(K) computed with the measured R(x) for every graph, real or null, at r = 0 and r = 1 (`fss_steepness.png` A).~~ *[V4: the chain curve in `fss_steepness.png` A is a secant ln(100)/ln(x(10⁴)/x(100)), a different estimator from s_fit, and it is drawn with one graph's R(x). Compared like for like (each graph's own R(x) and K_fit, the same window, the same censored MLE), the measured steepness is 0.99–1.15× the chain's (median 1.08) at r = 0 and 1.05–1.30× (median 1.11) at r = 1; model-free secants give 0.94–1.23× (median 1.13) and 1.01–1.39× (median 1.14). So steepness tracks the chain to within ~10–30%, with the data systematically steeper; it does not lie on it. The apparent drop of the 167k nulls below the chain curve in panel A is an estimator artefact: their τ ∈ [100, 10⁴] window starts below g_lin (x = 0.966, since τ(x = 1) = 116–131).]* The real subsamples sit where K ≈ 70 places them.
- What fails is the simple proportionality K ∝ N. The fitted K grows more slowly than the unfitted K_nc (ratio falling from about 1.1–1.7 to 0.5). Inhibition at r = 1 cuts K by ×7–13 (2.5k) to ×30–31 (167k) relative to K_nc. *[V6: relative to the r = 0 K_fit of the same graph (which is itself below K_nc at large N) the r = 1 reduction is ×10–14 (2.5k) and ×15–16 (167k).]*
- *[V7: K_fit at N ≤ 10k (r = 0) rests on 1–3 slow-window points (one point for 10k seed 1, where the fit is exact by construction); the fine grid was added only for N ≥ 20k (5–12 points). Local slopes of ln K_fit vs ln N between adjacent sizes zig-zag between 0.40 and 1.01 (r = 0); fits on N ≤ 20k and N ≥ 20k give 0.62 and 0.69. The FSS-3 "a ≥ 0.7" failure holds under every split.]*
- So "the mechanism is module size" holds when K is the effective (fitted) size. It does not hold with K = κN, and the plan's negative branch for a < 0.7 applies only in that narrow sense.

**Supplementary: survivor fluctuations** (`fss_analysis.json` `supplementary_chi_win`). This is the maximum of χ_surv = var(n)/⟨n⟩ inside [x_S, 1.25 x_on].
- Nulls at r = 0: grows from 14–16 (2.5k) to 201–236 (167k), i.e. ∝ N^0.65 ± 0.015.
- Nulls at r = 1: N^0.19 ± 0.02.
- Real subsamples: 4.8–9.3 (N^0.15 ± 0.06 at r = 0; N^0.10 ± 0.03 at r = 1).
- The pre-registered global χ_surv peak (`chi_peak`) is mostly at the edge of the window (first x with ≥ 3 survivors, or x = 8 at r = 1), so its location is not informative.

## 3. Interpretation for Q2 ("is the steep smooth crossover second order?")

1. **The crossover in the real brain is set by a fixed-size module.**
   - Its position (x_on = 1.63 at r = 0, 2.11 at r = 1) is the same for the AL alone (526 neurons), for real subsamples of 5k–80k neurons that keep the AL, and for the full brain, to within 1.8%.
   - The fitted module size is the same in all of them: K ≈ 69–72 at r = 0 and 36–37 at r = 1, against the full-brain K_needed of 75.8 and 36.9.
   - ~~The fluctuation peak does not grow with N.~~ *[V8]* The fluctuation peak grows only weakly with N in the real subsamples (4.8–9.3; N^0.15 ± 0.06 at r = 0, N^0.10 ± 0.03 at r = 1), against N^0.65 ± 0.015 in the r = 0 nulls.
   - ~~In the thermodynamic sense the real brain has **no** phase transition here: adding neurons does not sharpen the crossover.~~ *[V9]* Adding unrelated real neurons around an intact AL (5k–80k) does not move or sharpen the crossover, so it is not a bulk transition of the brain. This family holds the AL fixed by design, so it shows that the rest of the brain does not alter the AL crossover; a single finite connectome has no thermodynamic limit, and "no phase transition" in that sense cannot be tested.
2. **The same dynamics sharpens toward g_lin when K grows.**
   - In delocalized configuration nulls K grows with N.
   - The τ = 3000 onset approaches g_lin roughly as N^−0.42 (r = 0). It is consistent with a limit at g_lin, but the free extrapolation gives 0.93–1.02 depending on weighting and N range (V3), so it does not locate the limit precisely.
   - The crossover steepens from 2.5k to 20k: s_fit rises from 25.1–30.8 (2.5k) to 36.9–44.7 (20k). Above 20k s_fit is not monotone (40k 37.8–48.0, 80k 36.1–50.2, 167k 39.5–44.3), because for the largest nulls the τ ∈ [100, 10⁴] window extends below g_lin. The model-free secant ln(100)/ln(x(10⁴)/x(100)) rises from 29.5–33.8 (2.5k) to 55.7–57.8 (167k) and saturates above 40k (52.1–56.1) *[V4]*.
   - Survivor fluctuations at the crossover grow as N^0.65 (r = 0; only N^0.19 at r = 1).
   - These are the generic finite-size signatures of a transition located near the linear threshold. They do not by themselves distinguish a continuous from a discontinuous transition.
   - With a fully active start, a first-order transition with an active branch extending below g_lin would put the limit at the fold, below g_lin; the r = 0 null onsets approach g_lin from above, which argues against that. Neither a first-order transition nor hysteresis was tested here; that is the multistability program's job.
3. **The lifetimes follow the finite-K one-mode chain only approximately.**
   - With the measured R(x) and one K per graph, the logistic birth–death chain reproduces the slow-window lifetimes (rms of ln τ ≤ 0.28, from 1–13 points per graph with one free parameter), the out-of-sample lifetime at g_lin for the **r = 0 nulls** (1.00–1.17× the chain; exponent 0.505 vs the chain's 0.512 over the same K range), and the window steepness to within −1% to +15% (r = 0) and +5% to +30% (r = 1), with the data systematically steeper *[V4]*.
   - It under-predicts the lifetime at g_lin by 1.32–2.05× for the r = 1 nulls (exponent 0.405 vs the chain's 0.545) and by 1.13–1.45× for the real subsamples *[V5]*.
   - Limits: K is fitted (the unfitted K_nc gives the √K law only to exponent 0.37), and the chain is MF-DP by construction, so this is a consistency test of the finite-size one-mode form, not an exponent measurement. The criticality program (`criticality.md`, not yet verified) reports avalanche statistics that deviate from single-mode MF-DP, so the MF-DP class is not supported by this program.
4. **Verdict.** The steep smooth crossover is the finite-size-rounded image of the instability at g_lin (where the deterministic mean field has its transcritical point), observed through the module's extinction time. Continuity is assumed through the chain and not tested here. In the real network the rounding scale (K ≈ 70 at r = 0, ≈ 36 at r = 1) is the AL module, and it does not change as unrelated neurons are added. It is therefore correctly drawn as a crossover (dashed), with g_lin the only critical line (solid), as in the heat map.

## 4. Deviations from the spec, and why

1. **realsub c = 1/745.91, not 1/742.7.**
   - 1/745.91 is the simulator's raw normalisation (1/mean in-strength over neurons with in-strength > 0). It is also the c stored in `module_graph.npz` and is what keeps the AL couplings unchanged.
   - 742.7 is the mean over all 167,216 neurons, including those with zero in-strength.
2. **realsub RNG.** The spec did not fix it. I used `default_rng(7919 + 1000*seed + N % 997)`.
3. **Main-array time limit.** `-t 03:00:00` instead of 02:00:00, as a safety margin. The largest task took 1 h 11 min.
4. **Fine-grid follow-up (24 extra GPU tasks).**
   - On the coarse r = 0 grid (3.5% steps), x_S and x_on of the N ≥ 20k nulls fell into one or two cells. K_fit rested on 1–2 points, and the τ = 1000 finite difference would have been set by the zero-death regularisation.
   - Fine points were pooled with the coarse ones per graph (same graph, same g_lin to 1e-9).
5. **Steepness.**
   - The pre-registered s1000 (isotonic finite difference at τ = 1000) is reported, with a 95% lower-bound rule for a censored upper endpoint. No case needed it after pooling.
   - It is noise-dominated on the fine grids (e.g. 80k r = 0: 156.5 / 69.1 / 124.9 across seeds).
   - I therefore added s_fit, the censored-exponential MLE slope over τ ∈ [100, 10⁴] with its asymptotic SE, and use it as the primary measure. The FSS-3 steepness criterion passes with either.
6. **χ_surv.** The global peak is reported as specified, but it is mostly at the edge of the window. The windowed maximum above is supplementary.
7. **The "sustained" flat colour** covers cells with isotonic τ ≥ 3000, not only zero-death points (section 1).
8. **Panel C** is a separate third panel rather than an inset, for readability. The chain is drawn as a continuous curve and is checked against `theory_bd.json`.
9. **`lifetime_vs_K.png` has no nested AL sub-module (criticality C4) points.** No C4 output existed when this ran.
   - `load_c4()` in `pd3_fss_analyze.py` looks for `results/pd3/criticality/*submodule*.json` (rows with K_n and tau_x1). It must be adapted to C4's real format, and the figure regenerated.
   - The AL module alone (module_theory MT5, an unverified program output at the time) is used only as a cross-check point in `fss_onset_vs_N.png`.
10. **g_lin bisection tolerance** is 1e-6 relative (the spec asked for 1e-5).
11. **Preemptions.** Two tasks were preempted without requeue: 175172_35 and 175681_8. Both were rerun from their per-g checkpoints (176015, 175816_0).
12. **New helpers in `pd3_common.py`.**
    - `log_tau_fast` / `log_tau_smooth_fast`: a vectorised closed form of the same birth–death chain, equal to `log_tau` to 1.2e-15 relative and about 30× faster. It is used by the FSS analysis.
    - The verified originals are kept unchanged alongside.

## 5. Files

**Code.**
- `code/pd3_common.py` (helpers copied from analysis2_phase.py, plus exact g_lin and the vectorised chain)
- `pd3_array.sbatch`
- `pd3_redraw.py` (part A)
- `pd3_fss_graph.py`, `pd3_fss_sim.py`, `pd3_fss_fine.py`, `pd3_fss_fine_args.py`
- `pd3_fss_analyze.py` (part B; `NBOOT` environment override, default 2000)
- `pd3_redraw_fss_summary.py`

**Results.**
- `results/pd3/redraw/heatmap_{real,null,wnull}.npz`, `heatmap_meta.json`
- `results/pd3/redraw_fss/fss_*.npz` (54 coarse and 24 fine), `fss_analysis.json`, `fss_analysis.log`, `JOBS.md`, `args_*.txt`
- `results/pd3/redraw_fss.json`

**Figures** (all inspected).
- `figs/pd3/phase_lifetime_heatmap.png`
- `phase_lifetime_heatmap_nulls.png`
- `fss_onset_vs_N.png`
- `fss_lifetime_curves.png`
- `fss_steepness.png`
- `lifetime_vs_K.png`

## Verification (adversarial re-check, 2026-11-09)

**How it was checked.**
- `code/pd3_verify_redraw_fss.py` is an independent script. It imports none of `pd3_common.py`, `pd3_fss_analyze.py` or `pd3_redraw.py`.
- It has its own PAVA, crossing rule, birth–death chain, K fit (grid + golden section), censored-exponential MLE (Nelder–Mead/BFGS) and replica bootstrap. The chain agrees with a direct linear solve of the generator to 3e-13 in ln τ.
- It reads the raw per-replica extinction steps (`ext_step`) of all 54 coarse and 24 fine `fss_*.npz` files, plus `phase_complete.json`, `theory_bd.json`, `theory_nullK.json`, `module_theory/island_lifetimes.json` and the heat-map npz files.
- Output: `results/pd3/redraw_fss/verify/verify_redraw_fss.json` (log alongside) and `figs/pd3/verify_redraw_fss.png`.
- All 7 figures were looked at; the 3 part-B figures with defects were regenerated after the patches below.

**Reproduced exactly** (to the printed digits, from raw data):
- **Part A.**
  - Max abs(Δ ln g) is 0.0040 / 0.0039 (real), 0.0020 / 0.0029 (configuration null) and 0.0006 / 0.0003 (weight-shuffled null), against a grid step of 0.01788.
  - 23/23, 6/6 and 6/6 rows lie inside the total bands, and no row has more than one crossing.
  - 1,568 real points, 1,114 of them with zero deaths.
- **FSS-1.**
  - Seed-mean x_on at r = 0: 1.336, 1.282, 1.211, 1.145, 1.122, 1.079, 1.061.
  - Seed-mean x_on at r = 1: 3.713 → 1.336.
  - Spearman −1 and strictly decreasing at both r.
  - Seed-mean replica-bootstrap CI: [1.332, 1.341] at 2.5k and [1.060, 1.062] at 167k.
- **FSS-2.**
  - Max abs deviation 0.0110 (r = 0) and 0.0099 (r = 1).
  - Absolute g: 0.266–0.271 and 0.343–0.348.
  - AL alone: +1.2% / −1.8% in x, and +1.9% / −1.1% in absolute g.
- **FSS-3.**
  - a = 0.667 ± 0.027 (r = 0) and 0.574 ± 0.022 (r = 1); K_nc ∝ N^0.887 ± 0.053.
  - τ(x = 1) slopes: 0.505 ± 0.011 and 0.405 ± 0.010 against K_fit; 0.372 and 0.253 against K_nc.
  - Spearman(s_fit, K_fit) = 0.89 / 0.97.
  - realsub K_fit: 69.0–72.3 and 36.0–37.2.
  - K_fit/K_nc runs from 1.07–1.67 to 0.49–0.51.
  - Maximum K-fit rms of ln τ: 0.284.
- **χ_win.**
  - Slopes N^0.652 ± 0.015 (nulls, r = 0) and N^0.194 ± 0.022 (nulls, r = 1).
  - realsub: N^0.152 ± 0.061 (r = 0) and N^0.103 ± 0.033 (r = 1).
- **Extrapolation** (weighted, all N): x_∞ = 1.0072 ± 0.0098, b = 0.451, χ² = 11.48. At r = 1: 0.810 ± 0.065.
- **Full null, seed 0.**
  - r = 0: onset 0.8778, bracket [0.8770, 0.8811], bootstrap [0.8765, 0.8789], so total [0.8765, 0.8811]; S 0.8504.
  - r = 1: 1.1056, total [1.0983, 1.1109]. The md's 1.1106 differs only by bootstrap Monte-Carlo noise.
- **Validation and diagnostics.**
  - Max relative deviation 1.2e-15 over 22 runs; full-null g_lin 0.8286651.
  - AL share of |v|²: ≥ 0.9957 in realsub, 0.013–0.075 in the nulls.
  - realsub g_lin 0.16374–0.16426 and K_nc 65.9–67.9.
- **Chain overlays and κN rule.**
  - K_nc chain overlay: median abs error 1.6%, max 11.9%; (x_on − 1) ratios 1.08–1.47 and 0.70–0.72.
  - κN rule (r1_needed): no onset for N ≤ 5k, and 5.74 at 10k.
- **Run time.** The longest task took 4,262 s (1 h 11 min).

**Methods checked and found sound.**
- Censoring:
  - d = 0 points enter the isotonic fit as E/0.5 with weight 0.5, the analysis2 rule;
  - τ_sim = E/d is the correct censored-exponential MLE;
  - the K_fit window (200 ≤ τ ≤ 3000, ≥ 3 deaths) excludes x = 1, so τ(x = 1) really is out of sample.
- Fine and coarse points pool on the same graph (same g_lin).
- The realsub c is 1/745.91 (from `diag`).
- FSS-1/2/3 thresholds were applied as pre-registered.

**Corrections and qualifications** (each is also marked *[Vn]* in the text above):
1. **V1, Part A.** No change. The heat map, its consistency check and its "crossover" labelling are correct.
2. **V2, power-law language.** Every "exponent" in part B is an effective log–log OLS slope over at most 1.8 decades (N 2.5k–167k; K 14–2,900), and no alternative form was tested.
   - Local slopes between adjacent sizes vary widely: K_fit 0.40–1.01 (r = 0); χ_win about 0.49–0.92, from seed geometric means.
   - Read N^0.67, N^0.65 and similar as summary slopes, not scaling laws. The OLS SEs (±0.015–0.03) understate this model uncertainty.
3. **V3, onset extrapolation.** "x_∞ = 1.007 ± 0.010; the sharp limit is at g_lin" is not robust.
   - χ²/dof = 2.9.
   - x_∞ from other fits: 0.933 ± 0.062 unweighted; 1.020 ± 0.009 without 2.5k; 1.024 ± 0.009 for N ≥ 10k; 0.83 ± 0.16 without 167k.
   - Fixing x_∞ = 1 costs only Δχ² = 0.5.
   - Corrected reading: the onset is consistent with a limit at g_lin but is not located to better than a few percent. Changed in §0, §2.2 and §3.
4. **V4, steepness vs the chain.** The chain curve in `fss_steepness.png` A is a secant with one graph's R(x), while s_fit is a death-weighted censored MLE over a window that, for the 167k nulls, starts below g_lin (x = 0.966).
   - Like for like, with each graph's own R(x), K_fit and window, data/chain = 0.99–1.15 (median 1.08) at r = 0 and 1.05–1.30 (median 1.11) at r = 1.
   - Secants give 0.94–1.23 and 1.01–1.39.
   - So "lies on the chain for every graph" is replaced by "tracks it within ~10–30%, systematically steeper".
   - s_fit is not monotone in N above 20k, so "steepens from 2.5k to 167k" now quotes the secant (29.5–33.8 → 55.7–57.8, saturating above 40k).
5. **V5, lifetime at g_lin vs the chain.** "The chain at R = 1 passes through the null points" holds only at r = 0: ratio 1.00–1.17, and the chain's own exponent over the same K range is 0.512 against the data's 0.505.
   - At r = 1 the null lifetimes are 1.32–2.05× the chain. The data exponent is 0.405 against the chain's 0.545, so the pre-registered [0.4, 0.6] pass at r = 1 is nominal only.
   - The real subsamples sit 1.13–1.30× (r = 0) and 1.24–1.45× (r = 1) above the chain.
   - `figs/pd3/verify_redraw_fss.png` C.
6. **V6, inhibition's effect on K.** The "×30–31 relative to K_nc" mixes in the r = 0 fit/K_nc mismatch. Relative to the same graph's r = 0 K_fit, the reduction is ×10–14 (2.5k) and ×15–16 (167k).
7. **V7, K_fit support at small N.** At r = 0, K_fit for N ≤ 10k rests on 1–3 points; 10k seed 1 has one point, so its fit is exact. Fits on N ≤ 20k and N ≥ 20k give a = 0.62 and 0.69. The FSS-3 "a ≥ 0.7" failure holds either way.
8. **V8, realsub fluctuations.** "The fluctuation peak does not grow with N" is replaced by "grows weakly (4.8 → 9.3; N^0.15 ± 0.06 at r = 0, N^0.10 ± 0.03 at r = 1)".
9. **V9, thermodynamic language.** "The real brain has no phase transition here" is replaced.
   - The realsub family holds the AL fixed by design. It shows that unrelated neurons do not move or sharpen the AL crossover.
   - A single finite connectome has no thermodynamic limit, so that statement cannot be tested.
10. **V10, Q2 strength.**
    - "Continuous (transcritical)" is now stated as assumed through the chain, not tested here.
    - The growing fluctuations and steepening are generic finite-size signatures and do not identify the order.
    - "Moderate for MF-DP" is downgraded to "not supported here". This also cross-references `criticality.md` (not yet verified), which reports avalanche and QS statistics deviating from single-mode MF-DP. I did not re-check that program.
    - The first-order argument now says why a fully active start would put a first-order limit at a fold below g_lin.
11. **V11, Q1 row.** "No first-order lines" is replaced by "none drawn; first-order behaviour not tested here".
12. **V12, figures.**
    - `fss_onset_vs_N.png`: the log y axes had a single labelled decade; explicit tick labels were added.
    - `fss_steepness.png` A: the chain curves are now labelled as secants, with a legend title pointing to the like-for-like comparison.
    - `lifetime_vs_K.png`: the shorter chain label keeps the legend off the curve, and panel B's title notes that r = 1 and realsub lie 1.1–2.1× above the chain.
    - How they were regenerated: a minimal patch to `figures()` in `code/pd3_fss_analyze.py` (pre-patch copy in `results/pd3/redraw_fss/verify/`), run through `code/pd3_verify_redraw_fss_figs.py`. That driver re-runs the same seeded analysis and writes only PNGs; `fss_analysis.json` is unchanged, and its per-graph log is identical.
    - Pre-patch figures are kept in `results/pd3/redraw_fss/verify/figs_before/`.
    - New: `figs/pd3/verify_redraw_fss.png`, with panels for (A) extrapolation sensitivity, (B) like-for-like steepness ratio and (C) lifetime at g_lin over the chain.
13. **V13, minor.** The κN rule's "no onset below x = 8 for N ≤ 5k" is true, but `theory_bd.json`'s null grid reaches x = 12, so the prediction is "no onset up to x = 12".
14. **JSON.** A `verification` block was added to `results/pd3/redraw_fss.json`, with the pre-edit copy in `verify/`. Rerunning `pd3_redraw_fss_summary.py` would drop it.

**What stands.**
- Part A as delivered.
- FSS-1 and FSS-2 pass, and their numbers are exact.
- FSS-3: the pre-registered a ≥ 0.7 fails. τ(x = 1) ∝ K_fit^0.5 holds at r = 0 and is nominal at r = 1. The steepness criterion passes.
- The real crossover is set by the fixed AL module (K ≈ 70 / 36) and does not move from N = 526 to 167k.
- Null onsets approach g_lin as N grows.

**What is weakened.**
- The precise extrapolated location (1.007 ± 0.010).
- "Every graph lies on the one-mode chain": it holds for the slow window by construction, and at g_lin only for the r = 0 nulls.
- The MF-DP identification.
