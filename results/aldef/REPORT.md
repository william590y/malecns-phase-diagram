# Is the antennal-lobe result a model defect? Verified synthesis (2026-11-09)

I tried to break the synthesis. The core answer holds up, but eight parts needed correcting. The biggest change: the evidence does not rank depression above the lLN1_bc sign fix, because within the AL the two have the same effect. Several other claims overreached or relied on a coarse grid.

## How I checked
All paths below are under /MALECNS_ROOT.
- **Recomputed from the raw runs.** `code/ald_verify.py` (output `results/aldef/verify/verify.json`) has its own lifetime estimator and its own crossing rule.
  - It recomputed 28 sustained onsets across E2 and E3; all are within 0.4% of the lane values.
  - Three onset definitions (mean lifetime 3000, half of replicas alive, and lifetime counted from t = 100) change the key ratios by 2% or less. There is no fit-window artifact.
  - It also reproduced: the E1 allowed ranges (r = 1, ORN level 0.04: 0.113–0.278, lane 0.114–0.278); the E4 causal ratios (2.16, 3.04, 0.99/2.21, 0.97/3.59); D = 6.29 at r = 0 with depression; 20 of 24 type-ensemble draws with all 30 lLN1_bc excitatory; g_lin = α_eff/λ1 for every spectral condition; the palp leak (2.3–8.9%).
- **New GPU runs with new seeds** (array ald-verify-sim 183656, 7/7 tasks done; `code/ald_verify_simcheck.py`, outputs in `results/aldef/verify/sim/`):
  - The current ald_sim.py (md5 92c2e0d3) is bit-identical to swc.py in 3 new cases (`sim/bitid.json`). The work copy behind E3's earlier check no longer exists, so this closes that gap.
  - The push engine and the standard engine agree: onset 0.3454 vs 0.3459, with 128 replicas each.
  - Five onsets replicate within 1.3–2.8%: NW16 1.363 vs 1.384; lLN1_bc sign 0 0.991 vs 1.003; AL alone with NW16 2.92 vs 3.00; brain without the AL with NW16 1.458 vs 1.481; baseline 0.345 vs 0.346.
- **Citations.** I re-read the quoted sentences in fresh downloads of Olsen 2007, Kazama & Wilson 2009, Yaksi & Wilson 2010, Kazama 2011, Eckstein 2024, Tobin 2017, Kazama & Wilson 2008 and Benayoun 2010. All quotes are accurate. For Nagel & Wilson 2016 the fresh download was blocked by a captcha, so I checked the literature lane's saved copy.
- **Confounds checked:**
  - Run lengths are the same (T = 3000) in every decay arm.
  - Every lane uses the same criteria as analysis2.
  - E4 reports both normalizations.
  - Replica counts are adequate, apart from the E2 per-draw ensembles (16 replicas each).
- **Two real problems:** the ranking confound (correction 1) and the grid-limited upper ends (correction 3).

## Corrected report

**Short answer**
1. **It is a defect of the model's synapse dynamics, not of its wiring, and not a real prediction.**
   - Two model features can each remove it, and our simulations cannot say which matters more:
     - the missing short-term depression;
     - lLN1_bc acting as plain chemical excitation onto every target.
   - At r = 1, every fix that takes away the AL's lead puts the onset at 0.94–1.04, which is where the rest of the brain ignites on its own (1.04):
     - depression on lLN1_bc only: 0.944;
     - depression on the whole AL: 1.011;
     - lLN1_bc sign 0: 1.003;
     - lLN1_bc inhibitory: 1.036;
     - lLN1_bc plus the cholinergic lLN2 inhibitory: 1.042;
     - AL switched off: 1.042.
   - All-synapse depression goes further (1.38) only because it also depresses the rest of the brain, whose own onset moves from 1.04 to 1.48 (`verify.json` → `ratios_by_definition`).
   - Depression has the broader literature support, and it also works at r = 0. At r = 0, sign 0 alone still leaves the AL igniting first (23% of activity).
   - Neither fix has been tested in a realistic form. Depression of LN→LN synapses has never been measured, and the simulator cannot set signs per target.
   - The weights are not the cause (E4), and the ACh label is probably right (P = 0.753).
2. **The real fly sits below its own ignition point, driven by its ORNs.**
   - Published model: g = 0.10–0.29 at r = 1. Olsen's palps-intact condition caps this at 0.18–0.19.
   - With NW16 depression: about 0.25–1.18. The upper end is interpolated between grid points 1.15 and 1.5.
   - The amplification is moderate: 1.3–3.1× at the top of the published model's range.
   - With depression the allowed range lies entirely above g_lin (0.163), so a fully rested network would be linearly unstable there. Depression is what keeps activity bounded.
   - **New result:** under ORN drive with depression, the AL module's fluctuation measure stays flat at 1.2–2.3 over g = 0.06–2.6. Without depression it peaks at 4.2 near g = 0.22. So the depressed AL is a quiet, moderate amplifier, not a near-critical one. These numbers come from the existing E3 removal runs (`verify.json` → `E3_removal_driven_fluct`).
3. **The AL-only regime should be dropped as a sleep correlate.**
   - In the published model it lies 1.15–1.9× in g above the allowed range.
   - With literature depression it does not appear at any g we simulated (r = 0 and 1). The AL alone needs 2.0–2.6× the top of the allowed range.
   - This depends on the time unit. With a recovery time of 30 units, which is what a 50 ms unit would give, the onset rises only ×2.5 and the AL still holds 18% of the activity at onset. AL-alone runs were not done at that setting.
4. **Phase diagram after the fixes.**
   - The slow window keeps its relative width: onset/S is 1.21–1.27 with depression, against 1.19–1.21 in the published model.
   - The onset sits 7–10× above the AL's g_lin.
   - Where activity first lasts, it is broad: 9,000–18,000 neurons, with the AL module holding 5–11% of activity.
   - D at r = 0 moves from 1.47 to about 6.3.
   - Silence is far from linearly stable at the new onset: at g = 1.38, 47 of the top 400 excitatory modes (43 of them outside the AL) are past threshold. What holds activity back below the onset is depression, not the absence of an instability.
   - Whether that onset is a smooth crossover, a rounded continuous transition or a first-order jump was not tested.
5. **Recommended default and the deciding experiment.** These are unchanged, except for two numbers:
   - ORN level 0.04 corresponds to 3.6 Hz, not 3.4 Hz;
   - the upper end of the allowed range (0.32–1.24) is interpolated.

## Every correction
1. **Depression ranked first, signs second.** This was confounded, because all-synapse depression also depresses the rest of the brain. Restricted to the AL, the two fixes are equal (×2.92 vs ×2.89). Now a tie.
2. **"Every ratio is independent of the time mapping."** False for depression: the recovery time in model units is τ_rec/τ_u. With a recovery time of 30 units the onset rises only ×2.5 and the AL keeps 18% of activity at onset.
3. **Depressed allowed-range upper ends (1.18, 1.39, 1.24) and the ratios built on them.** These were interpolated across a coarse grid step from 1.15 to 1.5.
   - Upper end / onset: 0.79–0.85 → bracket 0.83–1.08 (NW16), 0.69–0.90 (KW09), 0.73–0.96 (excitatory only).
   - "About 0.8 of global ignition" → 0.7–0.95.
   - "Widens twofold" → 1.6–1.9× by interpolation.
   - "AL alone 2.6×" → 2.0–2.6×.
4. **"Far above any linear instability … a fluctuation-limited crossover."** Wrong: 47 modes are past threshold at the depressed onset (`modes_past_naive_threshold`). The claim is withdrawn.
5. **"Above the slow window the model fails both tests (PNs 2–4.6× ORN)."** That gain figure is for ORN level 0.04 only. At the highest drive levels the gain test passes just above the slow window (r = 1, level 0.10: G = 1.15 at g = 0.36). Only the silence test always fails (31–95% of driven activity remains). Likewise, "passes both deafferentation tests" now reads "passes the rate test and the remove-all-ORNs test".
6. **Kazama 2011 "strengthens eLN→PN coupling severalfold."** The "severalfold" refers to odor-evoked lateral excitation onto deafferented PNs; the coupling was "significantly larger". The palps were intact, so it is not a test of activity without input.
7. **"Lifetimes 17–70 units between g_lin and 1."** Should be 15–70 (NW16, KW09). That is barely longer than the 12–15 units the fully active start takes to die out on its own; N15 reaches 270.
8. **Smaller fixes:**
   - 3.4 Hz → 3.6 Hz.
   - AL share at onset with literature depression: 5–13% → 5–11% (13% applies only when depression is on lLN1_bc alone).
   - Weak-drive gain ratio: also 0.95–0.96 ± 0.01 at h = 1e-4.
   - "AL at 0.63–0.84 of its ignition" → "of the h = 0 onset" (0.64–0.86 of the AL-alone onset).
   - The frozen-resource theory matches only once its module size is calibrated on the published-model runs.
   - "One mismatch inside the range" → the palps leak caps the range; below 0.165–0.197 it stays under the pre-registered 10% tolerance.
   - The depression_removal figure description: 2.5× → 2.4–5.6×.

## Files changed
- `results/aldef/section.html` was fixed in place. The synthesis version is kept as `section.html.pre_verify`.
- Checked with Playwright on a scratch copy of the explainer with section 12 inserted: no horizontal scroll at 390 or 1280 px, no broken anchors, all six images load.
- `explainer/index.html` was not modified, and nothing was published.
- A short entry was added to `STATE.md`.