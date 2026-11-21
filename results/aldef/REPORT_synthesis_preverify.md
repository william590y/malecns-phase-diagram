# Is the antennal-lobe result a model defect? Synthesis (2026-11-09)

**Inputs.** I re-read and checked the files from every lane, then cross-checked the key numbers against them:
- Literature: `results/aldef/lit_biophysics.md` and `lit_biophysics_calc.json`.
- E1, ORN drive and antenna removal: `results/aldef/drive/summary.json`, `rows.json` and `table.md`.
- E2, transmitter signs: `results/aldef/signs/summary.json`.
- E3, depression: `results/aldef/depression/key_numbers.json`, `points.json` and `table.md`.
- E4, weights: `results/aldef/weights/table.md`.
- Background: `results/pd3/criticality.md` and `multistability.md`.

**New files.**
- `code/ald_synth_calc.py` writes `results/aldef/synth/synth_numbers.json`. It collects the lane numbers and computes the few derived ones: ratios, an interpolated delocalization line, amplification at the edge of the allowed range, and counts from the sign ensembles.
- `code/ald_synth_fig.py` makes `figs/aldef/synth_phase.png`.
- `results/aldef/section.html` is the new explainer section.

**Conventions.** g is the coupling. "Onset" means the coupling at which mean lifetime reaches 3000. Unless I say otherwise, numbers are for r = 1 (inhibition strength 1). Times are in model units, and 1 unit ≈ 10 ms; that is an assumption, and every ratio below is independent of it.

## Short answer
1. **Yes, it is a defect of the model, in its synapse dynamics rather than its wiring.** It is not a defect of the weights, probably not of the transmitter label, and it is not a real prediction. The main cause is the missing short-term synaptic depression. A close second is that the cholinergic lLN1_bc clique acts as plain excitatory chemical synapses onto every target. Running at h = 0 with g above the range the data allow is why the defect shows up at all. The synapse-count weights are not the cause.
2. **The real fly sits below its own ignition point, driven by the ORNs, with the AL acting as a moderate amplifier.** In the published model that means g = 0.10–0.29. With literature depression it means g = 0.25–1.18, just below a brain-wide ignition point. The "AL alone is active" regime that we had read as sleep-like is outside the allowed range, and it does not exist at all once depression is added. That mapping should be dropped.
3. **After the fixes the phase diagram still has a quiescent region, a slow window of the same relative width, a broad sustained phase and a delocalized phase. The AL-only localized phase is gone.** Sustained activity now starts 7–10 times above the linear instability g_lin. It is a crossover limited by fluctuations, not a near-critical module. No criticality or hysteresis tests were run on the corrected models.
4. **New default:** per-spike depression on excitatory outputs, ORN drive, and g chosen inside the allowed range. The next code change is per-target signs for lLN1_bc. **The deciding fly experiment:** measure the lLN1_bc→lLN1_bc synapse (its sign, whether it is electrical, how strong it is, and how it depresses), plus a reverberation test in fully deafferented flies.

## 1. Defect of the model, of its inputs, or a real prediction? Causes ranked

| rank | cause | effect of fixing it | literature support |
|---|---|---|---|
| 1 | **No short-term depression (c)** | Per-spike depression at literature values moves the h = 0 onset from 0.346 to 1.14 (N15), 1.38 (NW16) or 1.66 (KW09), a factor of 3.3–4.8. At r = 0 (NW16) it goes from 0.262 to 1.21, ×4.6. Depressing only the 30 lLN1_bc gives 0.944 (×2.7). Depressing only neurons outside the AL gives 0.342, no change. The AL alone now needs g = 3.00 (NW16) or 4.53 (KW09), while the brain without the AL sustains at 1.48 or 1.74, so the AL no longer leads. Weak-drive gain below g_lin is unchanged (ratio 1.02 ± 0.04). | Strong for measured synapses. ORN→PN and ORN→LN keep 72–91% of their strength per spike and recover over 0.6–2.4 s. LN→LN, including lLN1_bc→lLN1_bc, has never been measured. |
| 2 | **lLN1_bc as uniform excitatory chemical synapses onto every target (a+d)** | Setting lLN1_bc to sign 0 moves the onset 0.263 → 0.448 at r = 0 (×1.7) and 0.348 → 1.003 at r = 1 (×2.9). Making lLN1_bc and 40 cholinergic lLN2 inhibitory gives ×2.2 at r = 0 and ×3.0 at r = 1. At r ≥ 1 the AL then holds only 0.3–9% of the first sustained activity. The leading eigenvalue drops 5.574 → 3.174, or to 1.879 when the lLN2 are included. | eLN→PN is "mainly or purely electrical", while eLN→iLN is cholinergic (Yaksi & Wilson 2010). The ACh label itself is probably right: P(excitatory) = 0.753 under our mapping. The realistic per-target version has not been tested. |
| 3 | **h = 0 everywhere, so g is set too high (e)** | With ORN drive, the unmodified model passes both deafferentation tests for g = 0.090–0.231 (r = 0) and 0.100–0.289 (r = 1). At those g, though, the rest of the brain is at most 0.28 of its own onset (1.04), so nothing outside the AL can ever sustain itself. | Strong. Resting PN input rate of 74.9 Hz ≈ 3.44 Hz × 17.4 ORNs, and PNs are silent 10–20 min after the antennae are removed (Olsen 2007). |
| 4 | **Linear synapse-count weights (b)** | Not the cause. Every weight rule still self-sustains at h = 0. At equal leading eigenvalue the onset is 0.32–0.36 against 0.346, and 0.20 for binary weights. With linear weights the AL alone ignites the brain: switching it off multiplies the onset by 2.16 (r = 0) or 3.04 (r = 1). Strong compression shares ignition with other regions, and binary weights hand it to mushroom-body Kenyon cells. | Count-proportional strength has only been measured for ORN→PN (Tobin 2017). |
| – | Adaptation and refractoriness | They also raise the onset (0.42–1.71 and 0.47–2.97), but their strengths are not constrained by data. | PNs do not adapt (104% of their rate over 500 ms). Realistic refractoriness is about 1 model step. |

Sources for the table:
- Row 1: `depression/key_numbers.json` and `table.md`.
- Row 2: `signs/summary.json`.
- Row 3: `drive/summary.json`, plus `iso_base_noAL_r1` in `depression/key_numbers.json`.
- Row 4: `weights/table.md`.
- Ratios: `synth/synth_numbers.json`.
- Literature values: `lit_biophysics.md`.

**Is it a defect of the inputs?**
- **Weights:** no (E4).
- **The ACh label:** probably not.
  - Under the release's own transmitter confidences, lLN1_bc stays excitatory in 149 of 200 per-type spectral draws.
  - In the 20 of 24 simulated draws where it stayed excitatory, the AL was the top region at ignition in 20 of 20 at r = 0, 17 at r = 1 and 11 at r = 3 (`synth_numbers.json` → `E2_ensemble_top`).
  - The label's provenance is still open. The release's `ground_truth = ACh` for lLN1_bc does not come from Eckstein 2024's ground-truth table, and Eckstein points to lLN2T_a/b/c as the better match to known cholinergic LNs (`lit_biophysics.md`).

**Is it a real prediction?** No. The literature lane found no fly recording of a self-sustaining AL. The counter-evidence is weak:
- isolated-brain PN spiking, reported as "data not shown" (Gu & O'Dowd 2006);
- pan-neuronal calcium correlations with the antennae removed (Mann 2017).

Two things from E1 point the same way. First, the model's own self-sustaining regime breaks both measurements: above the slow window PNs fire at 2–4.6 times the ORN rate, and they keep 51–80% of their activity long after every ORN is removed (E1, r = 0, ORN level 0.04). Second, removing the antennae chronically for 2–5 days strengthens eLN→PN coupling "severalfold", yet only larger fluctuations were reported, not self-sustained firing (Kazama 2011). That fits a large safety margin below ignition. It is consistency, not a test, because that coupling is electrical and the model has no gap junctions.

## 2. Where the real fly sits, and what that does to our earlier reading
The constraints were set before the runs (`drive/plan.md`). The PN/ORN gain G must lie in 0.29–1.45, and PNs must be silent soon after the ORNs are removed.

| model (r = 1) | allowed g | h = 0 onset | onset of AL alone / of brain without AL | fly position |
|---|---|---|---|---|
| published | 0.10–0.29 (0.18–0.19 with the palps test) | 0.344 | 0.337 / 1.04 | AL at 0.63–0.84 of its onset; rest of the brain at ≤ 0.28 of its own |
| depression NW16, all synapses | 0.25–1.18 | 1.38 | 3.00 / 1.48 | whole brain at 0.85 of its onset; AL at ≤ 0.39 of its own |
| depression KW09, all synapses | 0.35–1.39 | 1.66 | 4.53 / 1.74 | 0.83; AL at ≤ 0.31 |
| depression NW16, excitatory outputs only | 0.32–1.24 | 1.57 | not run | 0.79 |

Sources: `drive/summary.json`, `depression/key_numbers.json` (`removal_protocol`, `isolation`), and `synth_numbers.json` (`ratios`).

Inside the allowed range (published model, r = 0, ORN level 0.04, g = 0.12–0.20):
- After ORN removal, PN activity falls to 1/e in 2–7 units. The AL goes fully silent after a median of 5–24 units (`drive/table.md`).
- At the lowest g, gain grows roughly in proportion to g: G/g is 2.4–2.7, close to the pure feed-forward estimate of 2.55.
- At the top of the allowed range, recurrence multiplies the gain by 1.3–3.1, depending on the ORN level (`synth_numbers.json` → `E1_amplification_at_allowed_upper`). So the AL is a moderate amplifier, not a diverging one.
- Under realistic ORN drive, the AL's fluctuation peak sits at g = 0.16–0.18 (r = 0) and 0.20–0.24 (r = 1), inside the range. The old peak at g ≈ 0.30 came from h = 1e-5 applied to every neuron, about 3,400 times weaker than one real ORN.

**One mismatch remains inside the range.** With the palps still driven, antennal PNs keep 2–9% of their driven activity at g = 0.10–0.16, through lateral input from local neurons (`drive/rows.json`). That share passes 10% from g = 0.165–0.197, whereas Olsen 2007 saw no spontaneous PN activity. This again points to lateral excitation onto PNs being too strong (cause 2). It has not been tested with depression.

**Consequence for the section-11 interpretation** ("AL lit, rest quiet" = sleep-like; recruitment beyond the AL = wake):
- **Published model.** The AL-only regime lies above the allowed range for the recorded state, but only by a factor of 1.15–1.9 in g (`synth_numbers.json` → `ratios`). A modest increase in gain during sleep could still reach it.
- **Corrected model.** The regime does not exist at any g. With NW16 depression the AL alone needs 2.6 times the top of the allowed range, and the rest of the brain ignites first.
- **Weaker inhibition does not bring it back.** Lowering r from 1 to 0, the bee rest-state mechanism, only moves the brain-wide onset from 1.38 to 1.21, with the AL holding 5% of the activity.

So the "AL-only = sleep-like" mapping should be dropped. Only the first half of section 11's literature-guided mapping B survives: the AL is an ORN-driven amplifier held below its ignition point. "Recruitment beyond the AL" is still a sensible order parameter, but it has to be studied with input switched on.

With depression, the allowed range reaches about 0.79–0.85 of the brain-wide onset. A corrected model is therefore compatible with a brain sitting just below global ignition. That is a possibility, not a finding, and it is no longer about the AL.

## 3. The phase diagram after the most defensible fixes
The full picture is in `figs/aldef/synth_phase.png`.
- **Quiescent region.** It now reaches g ≈ 1.1–1.3 at r = 1 with depression (S = 1.14 for NW16, 1.33 for KW09, 1.24 for excitatory-only), against about 0.29 before. With ORN drive, this is where the real AL operates.
- **The slow window survives with the same relative width.** The ratio of onset to the window's lower edge S (lifetime 200) is:
  - 1.21–1.27 with literature depression;
  - 1.18–1.21 with sign changes;
  - 1.19–1.21 in the published model (r = 1).

  Its position, however, is far above the linear instability: onset/g_lin is 8.5 for NW16 and 10.2 for KW09, against 2.1 before. Between g_lin and g ≈ 1, depressed lifetimes stay at only 17–70 units (`depression/points.json`).
- **No AL-only localized phase.**
  - Where activity first lasts, it already spans about 9,000–18,000 neurons (participation ratio), against 1,000–1,400 before. There are two exceptions: lLN1_bc sign 0 at r = 0 (4,700 neurons, AL 23%) and the weak "slow-component" depression (5,000 neurons, AL 21%).
  - The AL module holds 5–13% of the activity with literature depression, and 0.3–9% with sign changes at r ≥ 1. The rest is mostly central brain and nerve cord, plus optic lobe once the clique is silenced.
  - With depression, AL neurons are still the most active per neuron, at 18–39 times the rate of the rest of the brain. They no longer start the ignition.
  - "Localized" survives only in the weak sense that participation stays below N/4.
- **Delocalization.**
  - Depression, r = 0: the delocalization line (participation = N/4) moves from 1.47 to about 6.3. This is my interpolation between g = 4.995 and 6.322 in `depression/points.json`, a shift of ×4.3.
  - Depression, r = 1: not reached by g = 8, the end of the grid. Participation at g = 8 is 0.185, the same as the published model's 0.186.
  - Sign changes: the line barely moves (r = 0: 1.47 → 1.48–1.64; r = 1: 14.4 → 14.5–17.0).
- **Crossover or transition?**
  - In the published model the onset is a continuous transition located at g_lin and rounded by the module's finite size, with no first-order signatures (`pd3/criticality.md`, `pd3/multistability.md`).
  - After the fixes, sustained activity begins in a broad network far above any linear instability: a crossover limited by fluctuations, not a near-critical module.
  - The frozen-resource birth–death theory reproduces the AL-alone onsets with depression (2.83 vs 3.00; 4.37 vs 4.53).
  - Not tested: avalanche, quasi-stationary and hysteresis tests on the corrected models. Whether the broad onset is sharp, or even first order, is open.
- **With ORN drive and depression,** the allowed range widens about twofold. Its high/low ratio goes from 2.5 to 4.0–4.8, and its upper edge stays at 0.79–0.85 of the onset. The "driven, just below ignition" reading needs much less fine-tuning.

## 4. What should change in the model, and which fly experiment decides
**Recommended default.**
1. Per-spike depression on excitatory outputs: `--std_mode spike`, U = 0.25, TAU = 157 units ≈ 1.6 s (Nagel & Wilson 2016 ORN→LN). Use all-synapse depression and the KW09/N15 values as sensitivity checks. Do not use the default rate mode: it also weakens a rested neuron's first spike and moves g_lin by a factor of 1.2–1.3.
2. Drive the ORNs instead of using h = 0: all 2,639 ORNs at isolated-unit activity 0.04 (h ≈ 0.038; 3.4 Hz at 10 ms per unit).
3. Choose g inside the allowed range (0.32–1.24 at r = 1 for this setting). Report results relative to that range and to the brain-wide onset (1.57).
4. Keep linear synapse counts and the released transmitter labels.
5. Next code changes:
   - per-edge signs: lLN1_bc→PN set to 0 or weak, lLN1_bc→LN kept at +1;
   - rerun the palps test with depression;
   - add drive to the other sensory inputs.

**Fly experiments.**
1. **Measure the lLN1_bc→lLN1_bc synapse**, which carries 42.2% of the leading eigenvalue (`results/al_lit/data_grounding.md`).
   - Find lLN1_bc genetically with a split-GAL4 line matched to its morphology, and stain for ChAT or VAChT. That settles the label.
   - Make paired recordings from lLN1_bc onto lLN1_bc, onto PNs and onto GABAergic LNs.
   - Separate chemical from electrical transmission with mecamylamine and a shakB gap-junction mutant.
   - Use 10–20 Hz trains. This gives the sign per target, the electrical share, the strength, and the depression (fraction retained per spike and recovery time).
   - In the model, depression with f ≈ 0.75 on lLN1_bc outputs alone moves the onset 2.7-fold.
2. **Functional test.**
   - Remove both antennae and palps, then drive lLN1_bc optogenetically with conditioning trains of different lengths. Record PNs and LNs after light-off.
   - The difference I expect (my reasoning; I did not simulate it): with depression, after-activity is short and gets shorter after longer trains. Without depression, it does not depend on train length, and a gain increase of only 1.15–1.9 times would make it self-sustaining.

## Caveats
- The time mapping is an assumption: τ_u = 10 ms caps rates at 45 Hz. Rate ratios are unaffected.
- Depression of LN→LN and PN→LN synapses has never been measured, so ORN-synapse values stand in for all synapses.
- The simulator has no gap junctions.
- The sign ensemble assumes the non-consensus probability is shared among transmitters in proportion to their frequency in the release.
- Most depression arms were run at r = 1 only, and the removal protocol at a single ORN level (0.04).
- Non-olfactory inputs stay at h = 0 in the ORN-drive runs.
- Boundary uncertainties are about ±2–10%.

## Deliverables
- `results/aldef/section.html` is explainer section 12, "Is the antennal-lobe result a model defect?":
  - It uses only the page's existing classes and tokens and links into section 11's reference anchors.
  - Its six figures were copied to `explainer/figs/aldef/`.
  - I checked it with Playwright on a scratchpad copy of index.html with the section inserted: no horizontal page scroll at 390 px or 1280 px, and every anchor resolves.
  - `explainer/index.html` itself was not modified, and nothing was published.