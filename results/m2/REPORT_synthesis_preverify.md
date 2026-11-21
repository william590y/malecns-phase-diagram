# Model 2 phase diagrams: synthesis of the restart

Lane m2/synth, workflow wf_aae614ed-511, 2026-11-09. No new simulation was run, only CPU analysis. The plan, with the recheck rule and the agreement rule, was fixed before any number was computed: `results/m2/synth/plan.md`. All paths below are relative to `/MALECNS_ROOT`.

## 0. How I checked the lanes

**Recheck from the raw simulator files** (`code/m2_synth_recheck.py` → `results/m2/synth/recheck.json`).
- I used my own lifetime estimator, my own isotonic (PAVA) fit and my own crossing rule, and imported no project code.
- **Decay (h = 0).** The S, onset and D lines were recomputed for:
  - M2 on the real graph, all 23 r;
  - both nulls and M2s, 7 core r each;
  - M1 at r 0, 1 and 4.
  - Result: 47/47 S, 47/47 onset and 46/46 D pass the 2% rule. One row has no D in either version. Most lines agree to within 0.3%.
- **Driven (ORN level 0.04).** The lines recomputed were G = 0.29 (DQ→DA), DT→SS, τ_post = 200 and the top of the data-allowed region.
  - Result: 51/51, 51/51, 51/51 and 39/39 pass.
  - My first try at the allowed top used a binary S1 rule, and 25 rows "failed" (kept in `recheck_v1_binaryS1.json`). With MODEL2's own definition (S1 residual crossing 10%, or G crossing 1.45), all 39 pass.
- **One new fact from the recheck: neither null has a data-allowed region.** In both nulls G reaches 0.29 only at g 1.4–3.3, after S1 is already lost at 1.15–1.36.
- I also re-read these verdicts from their files:
  - hysteresis: M2 14/14 cells finite-rate (`nature/summary.json`);
  - avalanches: 0 critical-like points in M1 or M2;
  - left/right symmetry: 0 of 220 testable points broken (`modules/key_numbers.json`);
  - oscillation flags: 86 of 3,410 (point, signal) pairs flagged, 3 strong (`decay/key_numbers.json`);
  - driven checks (`driven/checks.json`);
  - palp-intact limits (`driven/curves.json` B4).
- The driven lane's ratios to the decay lines use the current `decay/boundaries.json` (md5 176f8aea), so they are not stale.

**Theory against simulation** (`code/m2_synth_table.py` → `results/m2/synth/agreement.json`, `agreement.csv`).
- The rule is the one pre-registered in `theory/stochastic_plan.md` sec. 3, applied to both theory lanes:
  - a boundary agrees if the prediction lies inside the simulated interval or within ±10% of the simulated value;
  - a bracket holds if lower ≤ sim ≤ upper;
  - the r-trend agrees if on(6)/on(0) is within ±20%;
  - finite-size null onsets agree within ±15%.

## 1. The complete M2 phase diagrams

### 1a. Setting (a): h = 0, decay from a fully active start

Figure: `figs/m2/final_decay.png`. Source: `results/m2/decay/boundaries.json` and `where.json`, rechecked.

| r | S (τ = 200) | onset (τ = 3000) [95%] | D (PR = N/4) | AL share at first sustained g | neurons carrying it (PR) |
|---|---|---|---|---|---|
| 0 | 0.981 | 1.18 [1.17, 1.20] | 6.29 | 6.1% | 13.6k |
| 0.25 | 1.10 | 1.33 [1.30, 1.36] | 13.1 | 6.3% | 13.6k |
| 0.5 | 1.18 | 1.43 [1.40, 1.46] | 14.1 | 6.3% | 14.0k |
| 1 | 1.21 | 1.57 [1.54, 1.60] | 13.9 | 6.2% | 14.6k |
| 2 | 1.34 | 1.72 [1.70, 1.75] | 11.9 | 5.8% | 15.7k |
| 4 | 1.47 | 1.94 [1.91, 1.97] | 9.29 | 5.3% | 17.8k |
| 6 | 1.56 | 2.02 [1.99, 2.10] | 8.03 | 5.0% | 19.1k |

- **Phases.** Every r shows the same order: Q (τ < 200) → S (slow window) → L (sustained, PR < N/4) → D.
- **Position.** The onset sits 7.2–12.4 × g_lin (g_lin = 0.1632, unchanged from M1).
- **Slow window.** onset/S is 1.20–1.33.
- **The L phase is bounded at every r.** D is 6.3–14.2, and reaching D means the optic lobe is recruited.
- **Nulls** (`final_decay_nulls.png`):
  - configuration null: onset 1.37–1.61, already delocalized at the onset (D = onset);
  - weight-shuffled null: onset 1.41–1.59, D 2.77–5.36;
  - real onset / configuration-null onset: 0.74 (r 0), 0.90 (r 0.25), 1.00 (r 0.5), 1.14 (r 1), 1.26 (r 2), 1.38 (r 4), 1.42 (r 6).
- **M2s** matches M2 within 6% on every line.
- **Variants** (onset as a multiple of M2's, r 0/1/4; decay lane):
  - KW09 1.16–1.41;
  - N15 0.73–0.85;
  - all-synapse depression 0.87–1.02;
  - TAU 30 0.49–0.71, with the AL at 12–18%.
  - None of them is AL-led.

### 1b. Setting (b): driven by spontaneous ORN activity (ORN level 0.04)

Figure: `figs/m2/final_driven.png`. Source: `results/m2/driven/curves.json`, rechecked.

| r | DQ→DA (G = 0.29) | data-allowed region [CI of top] | DT→SS | SS-L→SS-D | M1 allowed | M1 DT→SS |
|---|---|---|---|---|---|---|
| 0 | 0.216 | 0.216–0.968 [0.963, 0.978] | 1.21 | 6.25 | 0.092–0.215 | 0.268 |
| 0.25 | 0.241 | 0.241–1.08 | 1.32 | 12.8 | 0.097–0.230 | 0.291 |
| 0.5 | 0.267 | 0.267–1.13 | 1.43 | 13.9 | 0.101–0.248 | 0.306 |
| 1 | 0.317 | 0.317–1.23 [1.18, 1.26] | 1.57 | 13.9 | 0.114–0.279 | 0.347 |
| 2 | 0.400 | 0.400–1.37 | 1.73 | 12.0 | 0.132–0.306 | 0.409 |
| 4 | 0.463 | 0.463–1.48 | 1.95 | 9.43 | 0.164–0.340 | 0.507 |
| 6 | 0.474 | 0.474–1.57 [1.50, 1.61] | 2.07 | 8.20 | 0.187–0.382 | 0.615 |

- **Phases.** DQ → DA → DT → SS-L → SS-D at all 23 r.
- **Each driven line is an h = 0 line seen through ORN removal.** Over the 23 r:
  - DT→SS / h = 0 onset = 0.97–1.03;
  - allowed top / h = 0 S line = 0.94–1.03;
  - SS-L→SS-D / h = 0 D = 0.98–1.02.
  - So ORN drive adds no new transition. The realistic regime is everything between G = 0.29 and the start of the h = 0 slow window.
- **ORN level, at r = 1:** allowed region 0.137–0.73 (level 0.01), 0.187–1.18 (0.02), 0.317–1.23 (0.04), 0.63–1.25 (0.07), and none at 0.10 (none for any r ≥ 1). DT→SS stays at 1.54–1.57.
- **Negative result.** With the palp ORNs still driven, M2 passes the palp-intact test (Olsen 2007) only for g ≤ 0.186 at r 0 and r 1. The gain edge is 0.216 (r 0) and 0.317 (r 1), so **no g satisfies all three data constraints at r 0 or r 1.**
  - The failure is marginal at the lower edge (about 10% residual at g ≈ 0.22) and clear elsewhere (17–34%).
  - M2s leaves a sliver at r 0 only (0.235–0.258).
  - This test is reported alongside the main allowed region, not as part of it (MODEL2 5b).

## 2. Theory against simulation

Figures: `figs/m2/final_agreement.png`, plus the overlays in `final_decay.png`, `final_decay_nulls.png` and `final_driven.png`. Source: `results/m2/synth/agreement.json`.

### 2a. h = 0 onset, real graph (theory value, with theory/sim in brackets)

| r | sim [95%] | mean field, equal-size rule | MF cutoff / AL-chain adiabatic / AL-chain frozen | stochastic single mode (pre-reg.) | stochastic pooled |
|---|---|---|---|---|---|
| 0 | 1.18 [1.17, 1.20] | 0.63 (0.53) | 0.32 / 1.33 / 2.93 | 1.66 (1.41, PF) | 0.73 |
| 0.25 | 1.33 | 0.83 (0.63) | 0.32 / 1.28 / 2.71 | 1.87 (1.41, CAL) | 0.92 |
| 0.5 | 1.43 | 1.05 (0.73) | 0.32 / 1.31 / 2.72 | 2.08 (1.45) | 0.96 |
| 1 | 1.57 | 1.23 (0.78) | 0.27 / 1.39 / 2.82 | 2.49 (1.59) | 0.88 |
| 2 | 1.72 | 1.22 (0.71) | 0.27 / 1.53 / 3.01 | 3.09 (1.79) | 1.15 |
| 4 | 1.94 | 1.20 (0.62) | 0.23 / 1.83 / 3.46 | 3.97 (2.04) | 1.06 |
| 6 | 2.02 | 1.21 (0.60) | 0.23 / 2.02 / 3.76 | 4.60 (2.28) | 1.36 |

### 2b. Grades by boundary

**h = 0 onset and S line, real graph**
- **No pre-registered point estimate of the onset agrees at any r.**
  - The mean-field equal-size rule is 0.53–0.80 of the simulation (fails at 23/23 r). Its direction was pre-registered: it was expected to under-estimate.
  - The stochastic single mode is 1.41–2.28 of the simulation (fails at 7/7 r). It was pre-registered as an upper bound in g, so its direction is also right.
- **The brackets hold, but they are wide.**
  - Mean field [cutoff, AL-chain frozen]: holds at 23/23 r, but spans 9–16× in g.
  - Stochastic [pooled, single]: holds at 7/7 r, spans 2.3–3.4×.
- **The AL-chain adiabatic value is the closest number,** within ±10% at 16/23 r. But it was meant as an upper bound, and the simulation lies above it at 21/23 r. It also models the AL alone, while the simulated onset is brain-wide. I read this match as probably a coincidence.
- **S line:** mean field 0.58–0.79 of the simulation; stochastic 1.20–1.98. Both fail at every r.
- **Trend with r**, on(6)/on(0): simulation 1.71; mean-field equal-size 1.93 (agrees); AL-chain adiabatic 1.52 (agrees); stochastic 2.77 (fails, because its M1-calibrated inhibition factor is too strong).

**D line**
- Mean field agrees only at r 0 (6.08 against 6.29).
- At r 0.25–1.5 the mean field has no D up to g 30, while the simulation has D at 13–14.
- At r ≥ 3 it is 31–36% low.
- This is the same mean-field over-inhibition seen in M1.

**Nulls, M2s and variants**
- Mean-field null onsets, which reuse the M1 real-graph calibration: 0.75–0.90 of the simulation (fail at 14/14).
- The stochastic theory on its own 167k-neuron null:
  - r 1: agrees (1.08);
  - r 0: fails (0.80);
  - r 4: fails (1.24).
- **Finite size, configuration null at r 0 (stochastic):** null onsets for 5k–80k neurons agree within 15% (5/6). The 167k null fails (0.80). The width W is over-estimated by 14–60%, but its sign of change with N is right.
- **Real vs null.** The stochastic theory predicted the real graph ignites above its null at r 0 (×1.30). The simulation gives ×0.74. The stochastic lane's own post-hoc check called this "agree"; it is not. At r 1 and r 4 the direction is right but the size is off (1.68 and 2.28 predicted, 1.14 and 1.38 simulated).
- **M2s.** The mean field correctly says M2s ≈ M2 (within 0–5%), but its absolute M2s onsets fail like M2's (0.51–0.81).
- **Variants:**
  - mean-field equal-size: 0.44–0.81, fails at 12/12;
  - stochastic single mode: agrees at 2/9 (TAU 30 at r 0: 1.02; r 4: 0.94), others 0.84–2.73.
- **Modules alone, r 0 (stochastic single mode):**
  - agree: AL 0.95, optic lobe 1.04, left AL lobe 1.07;
  - narrow fails: right AL lobe 1.105, central brain 1.18;
  - large fail: nerve cord 7.2.
  - The theory and the modules lane define the central-brain and nerve-cord blocks slightly differently (N 37k vs 39k, 20k vs 23k).

**Where activity lives at the simulated onset**
- Both theories are within ×2 for PR and for AL share at all 7 core r.
- Evaluated at its own predicted onset g, the mean field is 2–2.5× too AL-heavy for r ≤ 4.

**Driven, at ORN level 0.04**
- **The stochastic linear-noise approximation (LNA) works at r = 0:**
  - G = 0.29 at g 0.213 (sim 0.216);
  - G = 1.45 at g 1.03 (sim 1.05);
  - non-ORN Fano peak 59 at g 0.91 (sim 55 at 0.99);
  - τ_int peak height 11.0 (sim 10.1);
  - AL Fano peak height 2.20 (sim 2.19).
  - The g positions of the τ_int and AL Fano peaks are 13–17% low.
- **The LNA has no r-dependence,** so at r ≥ 1 it fails: DQ→DA is 0.45–0.67 of the simulation, and the τ_int peak is 2.0–2.3×.
- **Mean-field G = 0.29 line:** agrees at 10/23 r (r ≤ 0.625 and r ≥ 4.5). In between it has a spurious bump, up to 2.5× at r 1.75.
- **Mean-field removal-SS line** (equal-size rule): 0.53–0.81 of DT→SS, fails at 23/23.
- **Mean-field allowed top** (on a coarse grid): agrees at 7/23 r.
- **The mean field's strict allowed region (S1 and S2) is empty at every r > 0,** while the simulation's is wide.

**Qualitative predictions**
- **Agree:**
  - no bistability or coexistence (both theories);
  - no left/right symmetry breaking;
  - no Griffiths phase. The stochastic theory predicted exponential lifetimes plus extra early deaths, which is exactly what the simulation shows;
  - damped noise quasi-cycles, never strong. Predicted period 55–150 against simulated 53–152; predicted flag at r 0, g 1.0–1.2, against simulated g 1.15–1.31;
  - branching ratio: predicted maximum 0.948 per unit, simulated 0.94 (driven);
  - M2 has no critical-like avalanche range;
  - bursts appear from g ≈ 0.55 (the simulation's first large-avalanche bump is at 0.585, r 0).
- **Disagree:** the mean field's large limit cycles for every r > 0 (from g about 1.0–1.6) do not appear in the simulation. There is no OSC flag at r ≥ 0.625 apart from one 4-replica point, and no Hopf candidate at any r.
- The deterministic mean field also puts a tiny active state at its silent edge (about 5–8 neurons at g 0.30). In the stochastic network that state cannot survive, so g_lin is not a visible boundary.

**Verdict on theory.**
- The exact pieces hold: g_lin, the single-unit resource law (κ = 35.3), and the r = 0 driven LNA.
- The h = 0 onset of a brain-wide, depression-limited state is not predicted to ±10% by any pre-registered estimate. It is only bracketed, by a factor of 2–3 (stochastic) or 9–16 (mean field).
- As in M1, inhibition is where the theory fails: the D line, the driven lines at r > 0, and the stochastic r-trend.

## 3. M1 (old) and M2 (new) side by side

Figure: `figs/m2/final_m1_vs_m2.png`.

**Changed**

| item | M1 | M2 |
|---|---|---|
| Scale of the h = 0 lines | S 0.234–0.436, onset 0.266–0.583 (1.6–3.6 × g_lin) | S 0.98–1.57, onset 1.18–2.02 (7.2–12.4 × g_lin): ×3.4–4.8 |
| Where the first sustained state lives | AL module 40–50% of activity, PR about 1k neurons | AL 5.0–6.3%, PR 13.6k–19.1k; central-brain intrinsic 48–56%, nerve-cord intrinsic 21–23%, descending ×5.6–8.8, motor ×9–10 |
| AL alone (r 0/1/4) | equals the full brain (0.268 / 0.338 / 0.493) | 2.19 / 11.5 / none up to 30, against the full brain 1.18 / 1.57 / 1.94. No module sustains alone at the brain onset |
| D line | 1.47 (r 0), rising to 28.9 (r 1.5); none up to 30 at r 1.75–6 where tested | 6.3–14.2, bounded at every r; falls for r ≥ 1 (unexplained) |
| Real vs configuration null | real ignites at 0.31–0.47 × the null onset | 0.74 (r 0) to 1.42 (r 6): reversed for r ≥ 1 |
| Finite-size scaling, real graph | onset pinned by the AL within 1.1% | not pinned: onset 4.36 → 1.18 from 20k to the full brain (r 0); W ~ N^-0.2 |
| Oscillations | none | noise-driven quasi-cycles at r ≤ 0.5 near the onset, period 0.5–1.5 s; variance ~ N^-1 on the nulls; not a phase |
| Driven allowed region | 0.092–0.215 (r 0) to 0.187–0.382 (r 6), width ×2.0–2.5 | 0.216–0.968 to 0.474–1.57, width ×3.1–4.5. Lies wholly above g_lin (1.3–9.6 × g_lin): a rested network would be linearly unstable there, and depression is what holds it |
| Driven fluctuation marker | sharp AL Fano peak at 0.55–0.66 × the SS line, tracking it | no peak tracks any line; AL Fano is a plateau |

**Survived**
- The phase order Q → S → L → D.
- The relative width of the slow window (onset/S 1.14–1.37 in M1, 1.20–1.33 in M2).
- No Griffiths phase. The reason changed:
  - M1: one rare region, the AL;
  - M2: one brain-wide metastable state. Lifetimes after t = 200 are exponential at 21/21 long-run points, and no ρ(t) follows a power law.
- "Continuous, finite-size-rounded, no first-order signature." The part that did not survive is "of one fixed module at g_lin."
- No first order. Hysteresis loops shrink with slower ramps: 3.6–15.5× in M1, 11–90× in M2, down to the noise floor by dwell 800–3200. There is no bimodality.
- No left/right broken symmetry (M1 0/48, M2 0/220). In M2 the two sides are positively correlated (AL 0.05–0.65).
- No coexistence: one cluster of end states in 96 + 72 cells.
- DT→SS equals the h = 0 onset.
- The top of the allowed region sits just below the self-sustaining line: DT→SS / top is 1.23–1.61 in M1 and 1.23–1.33 in M2.
- Theory works without inhibition and fails with it.

## 4. Answers to the program's questions

**What phases exist (M2).**
- With h = 0:
  - Q: dies in under 2 s;
  - S: slow window, 2–30 s;
  - L: sustained, brain-wide but not global. PR < N/4 still means 13k–42k neurons, so "L" is a convention, not one module;
  - D: global, with the optic lobe recruited.
- Under ORN drive:
  - DQ;
  - DA: amplifying, silent after removal;
  - DT: reverberates 2–30 s after removal;
  - SS-L and SS-D.
- None of these exist in M2:
  - module-only phases: no AL-only or one-lobe state, and no module sustains alone at the brain onset;
  - broken-symmetry or coexistence phases;
  - an oscillatory phase;
  - a Griffiths phase.

**Which boundaries are true transitions, and of what order.**
- **g_lin = 0.1632** is the only parameter-free line: the exact linear instability of the rested silent state. In M2 nothing observable happens there. The mean-field active state is a few neurons, and the stochastic network kills it.
- **The h = 0 onset of self-sustained activity** (= DT→SS) is the one candidate for a genuine transition. It behaves as a continuous (absorbing-state-like) onset that finite size rounds heavily:
  - in the configuration nulls the crossover sharpens with N (W 0.45 → 0.075 at r 1);
  - in the real graph W only drifts (0.19–0.28 at full size, about N^-0.2), so at the fly's size it is a broad crossover;
  - where it would sit in an infinitely large brain cannot be determined;
  - there is no first-order evidence;
  - the pre-registered finite-size verdict is "inconclusive" at r 0, 1 and 4.
- **Under ORN drive this transition is smoothed away.** No driven observable is singular at DT→SS (χ_nonORN has no peak there).
- **The τ = 200 and τ = 3000 lines** are lifetime conventions on that one crossover.
- **L→D** is a recruitment convention (PR = N/4). Its nature was not tested.
- **DQ→DA** is a data convention, not a transition.

**Does depression self-organise criticality?** No, not in this model.
- **Levina–Herrmann–Geisel test:** pre-registered verdict "inconclusive" (21/21 cells), because neither M1 nor M2 has a single critical-like point.
- **M2's avalanches are not power laws:**
  - where the size exponent is 1.3–1.7, the KS distance is 0.031–0.080;
  - a bump of large population bursts appears from 0.5–0.68 × the onset;
  - the crackling relation fails: measured slope 1.96–2.24, predicted 2.36–2.77.
- **Branching ratio:** at most 0.977 per unit (pulse) and 0.94 (driven), never within 0.01 of 1.
- **Theory agrees:** depression divides every mode's effective size by 36, and the AL's self-organised mode cuts avalanches off at sizes 62–95.
- **The only LHG-like hint fails the fit rule.** Ignoring fit quality, M2's size exponent sits in [1.3, 1.7] over a g range of up to ×1.93 (M1 ≤ ×1.39).
- **Untested:** the stochastic theory predicts that the configuration null would be critical-like for g 0.91–3.0.

**Where the data-allowed region sits.**
- In DA, between G = 0.29 (0.216 at r 0, 0.474 at r 6) and the h = 0 slow-window line.
- That puts it 1.23–1.33× below the self-sustaining line, entirely above g_lin, and in the regime where depression sets the response: the slope against the transmitted ORN input falls below 0.5 for g ≥ 0.19–0.29.
- It shrinks with ORN drive and vanishes at ORN level 0.10 for r ≥ 1.
- With the palp-intact test added, it is empty at r 0 and r 1 in M2. That is a sign M2 still misses something in the AL: electrical eLN→PN coupling, LN→LN dynamics, or per-glomerulus detail.

**What this implies for the wake/sleep interpretation.**
- The realistic regime of M2 is a driven, subcritical, depression-limited amplifier.
  - It has no critical point, no Griffiths phase and no fluctuation peak that tracks a transition.
  - Activity dies within about 2 s of removing the ORNs.
- So the model gives no phase-transition signature that could separate wake from sleep inside the allowed region. The original "Griffiths ≈ wake" hypothesis has no support in M2.
- The AL-only regime, already dropped as a sleep correlate by aldef, does not exist at all in M2.
- The nearest qualitative change is crossing into brain-wide self-sustained reverberation. That would need about a 25–33% rise in effective gain, for example from neuromodulation, which the model sets to 0. This is a speculation about what *could* produce a state change; it was not tested, and it lies outside what the deafferentation data allow.
- All of this concerns the model, not the fly.

## 5. Remaining open problems

1. **No g in M2 passes the gain test and the palp-intact test together** at r 0 or r 1 (M2s: a sliver at r 0). The AL synapse model needs electrical coupling, measured LN→LN depression, or glomerulus-level detail.
2. **No theory predicts the h = 0 onset of the brain-wide depressed state to ±10%.**
   - The mean field's equal-size rule is calibrated on M1.
   - The stochastic single mode treats the network as AL-seeded, but simulated isolated blocks contradict that: the central brain alone ignites at 1.42 and the AL alone at 2.19 (r 0), while the whole brain ignites at 1.18. So the regions re-seed each other collectively.
   - A theory of that re-seeding is missing. The stochastic "fluctuation-based" estimate (1.22–1.32 at r 0) was post hoc.
3. **Inhibition in theory.**
   - The stochastic r-dependence comes only from an M1-calibrated factor.
   - The LNA has no r-dependence.
   - The mean-field D line and its Hopf predictions over-inhibit.
   - Whether the mean-field limit cycles are a closure artefact or a large-N effect is untested.
4. **Large-N behaviour of the real-graph onset.** Where it sits and whether W → 0 are undetermined, because the real subsample families change the network as well as N.
5. **The L→D line.** Its nature (finite-size scaling, ramps) is untested, and why D falls with r for r ≥ 1 is unexplained.
6. **The Levina test is undecided by design.**
   - M1 had no critical-like points.
   - The gap-100 runs have too few avalanches.
   - The null's predicted critical-like range is unsimulated.
   - `m2_obs.powerlaw_fit` inflates KS for tied sizes. The nature lane used a tie-correct fit, but pd3 and any other users of the old fit should recheck.
7. **Oscillations on the real graph.** The finite-size check is inconclusive there. MODEL2's CV ≥ 0.3 "bursting" rule depends on N and should be retired as a phase label.
8. **The τ_int maximum at g 0.69–0.96.** The LNA reproduces it at r 0, but at r ≥ 1 its height halves, and that is unexplained.
9. **Lifetime verdict.** The pre-registered test is "inconclusive" at r 0, r 1 and the null, because of two-stage deaths. The "exponential" verdict rests on the post-hoc residual-lifetime test.
10. **Sensitivity.** The numbers depend on the time mapping and the ORN level:
    - TAU 30 lowers the onset to 0.49–0.71× and raises the AL share to 12–18%;
    - under that mapping, real ORN rates would be ρ ≈ 0.19, beyond the ORN grid;
    - the measured ORN rate falls with g (0.022–0.042).
11. **Sampling gaps.**
    - Variants and nulls were run on core r only.
    - Each null uses one fixed graph.
    - The driven B sets stop at g 11.2.
    - The M1 onset check at r 4 is off by 2.6% (push vs spmm engine not tested at high r).
12. **The wake/sleep question needs neuromodulation and electrical coupling,** neither of which is in the model.

## 6. Figures and files

**Figures**
- `figs/m2/final_decay.png`: h = 0 phase diagram (all 23 r) with simulation bands, the mean-field equal-size, cutoff and AL-chain lines, the mean-field D, and the stochastic single and pooled lines; plus theory/sim ratios against r.
- `figs/m2/final_decay_nulls.png`: configuration null, weight-shuffled null and M2s, simulation against mean field (and the stochastic 167k null).
- `figs/m2/final_driven.png`, four panels:
  - (a) driven phase tiles (23 r), allowed region, simulated lines, h = 0 onset and S overlay, mean-field and LNA lines;
  - (b) gain G(g): simulation, mean field and LNA;
  - (c) Fano and τ_int at r 0: simulation and LNA;
  - (d) allowed regions of M1, M2 and M2s, and the mean-field window.
- `figs/m2/final_m1_vs_m2.png`: M1 and M2 h = 0 diagrams on the same axes; AL share and PR at the onset; driven allowed regions and DT→SS for M1 and M2.
- `figs/m2/final_agreement.png`: theory/sim ratio against r for S, onset, D and the driven lines.

**Data and code**
- `results/m2/synth/{plan.md, recheck.json, recheck_v1_binaryS1.json, agreement.json, agreement.csv}`.
- `code/m2_synth_{recheck,table,figs}.py`.
- A lane entry was appended to `STATE.md`. No REPORT.md was written; this text is the report.
