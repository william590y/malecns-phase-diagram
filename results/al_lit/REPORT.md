# The antennal lobe in our MaleCNS model, and what the literature says about it

Synthesis written 2026-11-09 for the user's question about the antennal lobe's (AL's) special status, a possible link between the AL and sleep, and whether our simulated findings match any empirical results on consciousness or brain phase transitions involving this region.

**Sources.**
- Literature line 1: `line1_al_sleep_arousal.md`.
- Literature line 2: `line2_anesthesia_arousal_consciousness.md`.
- Literature line 3: `line3_phase_transitions_olfactory_wholebrain.md`.
- Data grounding on the MaleCNS cache: `data_grounding.md` and `.json`, with figures in `figs/al/`.
- Four independent fact-checks, appended to those files as "Verification" sections. One of them recomputed about 40 numbers with `code/al_verify_3.py`.
- Our simulations: `STATE.md`, `CLAUDE.md`, and `results/analysis2/{phase_complete,spreading,longruns,driven}.md`.

Only citations rated **verified** or **partly verified** are used. Appendix A lists what was dropped and every correction applied.

**Labels.**
- Scope: **AL-direct** (measured or manipulated in the insect AL); **olfactory-general** (other insects' olfactory pathway, or the mammalian olfactory bulb (OB) and piriform cortex); **whole-brain/other** (including "the AL was not covered"); **theory/model**.
- Strength: **strong** = causal or replicated, read in full text; **moderate** = one clear study; **weak** = preprint, abstract only, indirect, or a different species or state definition.

## 1. Short answer

1. In our model the AL is special for a concrete structural reason. In each lobe, 15 cholinergic local neurons (lLN1_bc) are wired all-to-all, with a median of about 200 synapses per connection. With the reciprocal PN-LN loops, this gives the AL an excitatory feedback strength about 3 times that of the next-strongest circuit in the CNS.
2. It is not a sensory-convergence effect. The module contains none of the 2,639 olfactory receptor neurons (ORNs), and deleting them leaves the leading eigenvalue at 5.55 of 5.57. The lead does depend on the multi-synapse weights and, decisively, on the cholinergic label the release gives these local neurons.
3. A real but modest literature links the AL to sleep:
   - olfactory input and two GABAergic AL outputs change how much, and when, flies sleep;
   - AL synapses and glia track sleep need, but no more than the rest of the brain does;
   - sleep is needed for glia to clear damaged olfactory axons from the AL.
4. None of it shows the AL generating or gating sleep, wake, anesthesia or consciousness. The fly control points are the fan-shaped body, ellipsoid body, protocerebral bridge, mushroom body and clock neurons. The one study comparing AL activity with other regions after sleep loss found "little change" in the AL.
5. The AL keeps processing odors during sleep: flies wake selectively to salient odors. In mammals the OB is the least state-gated stage, and state changes act downstream of it.
6. Nobody has recorded the fly AL across verified sleep or under general anesthesia, so the most direct test of our model is still open.
7. Several empirical results line up with the model:
   - the AL has the most spontaneous spiking in fly brain recordings;
   - olfactory regions are the most intercorrelated at rest;
   - the AL is among the most reciprocally wired neuropils;
   - its dynamics are controlled by local-neuron inhibition;
   - an independent fly-connectome simulation also found no Griffiths phase.
8. The biggest tension is mechanism. Real spontaneous AL activity is fed by receptor neurons, and projection neurons go silent without antennae. Our model's AL keeps itself going with no input. The model also lacks gap junctions, neuromodulation and any sleep-pressure variable.
9. The literature does not place seizures in the AL (the mushroom body α/β lobes are the only initiation site) or insect consciousness there (the central complex is the leading candidate).
10. The closest data come from a honeybee AL that is more internally coupled at rest, modelled with weaker local inhibition. They suggest that our "AL lit, rest of the brain quiet" regime reads better as sleep-like than as wake-like, with wakefulness corresponding to recruitment beyond the AL. That is an interpretation, not a finding.
11. Honestly, direct links are sparse. The model does make specific, feasible predictions for isoflurane, sleep loss, removal of sensory input, and recordings across sleep (section 7).

## 2. Why the AL is special in our model

### 2.1 What the module is

The module is the union of the 99% supports of the two leading right eigenvectors of the r = 0 excitatory operator. That operator is M = c·w on cholinergic presynaptic neurons, with c = 1/745.9. It has 526 neurons.

| component | count | share of leading mode \|v\|² |
|---|---:|---:|
| lLN1 (30 lLN1_bc, 15 per lobe, all ACh) | 34 | 0.52 (lLN1_bc 0.51) |
| lLN2 (all 92 in the CNS) | 92 | 0.29 |
| uniglomerular PNs (50 of 66 glomeruli) | 217 | 0.09 |
| v2LN/vLN/il3LN | 34 | 0.05 |
| multiglomerular PNs | 53 | 0.02 |
| other (LH 30, APL/DPM 4, ALIN/ALON, HRN/TRN, ...) | 96 | < 0.03 |
| ORNs | 0 of 2,639 | 0.001 (3.3% of the left, driver eigenvector) |

- By transmitter, the module has 366 ACh, 96 GABA, 32 "unclear", 25 Glu and 7 monoaminergic neurons.
- The leading pair (5.57 and 5.39) is one bilateral system. Each lobe alone gives 5.49 (right) and 5.41 (left), mixed weakly via bilateral ORN axons.
- ORNs supply 39% of the module's excitatory input, but only as input; they are not part of the loop.

### 2.2 What makes it dominant

- **A complete clique.** All 420 of 420 same-side ordered lLN1_bc pairs are connected, with a median of 204 synapses and no cross-side edges.
  - The clique alone has λ = 3.92 (the uniform-clique value is 14 × 206.8 / 745.9 = 3.88).
  - Adding the cholinergic lLN2 types and the PN↔LN loops brings λ to 5.57. That is 2.97 times the best non-AL mode (1.88, VNC leg premotor and motor neurons).
- **Exact decomposition of λ.** LN→LN 64.7% (lLN1_bc→lLN1_bc alone 42.2%), PN→LN 16.8%, LN→PN 16.5%, and any ORN edge 0.87%. This holds even though ORN→LN (633k synapses) and ORN→PN (503k) are the largest synapse pools.
- **Heavy reciprocal pairs.** 161 of the CNS's 200 strongest reciprocal excitatory pairs are in the AL.
  - The AL core (1,144 cells) averages 12.0 synapses per connection. 4.9% of its connections have 50 or more synapses and hold 46% of its synapses. Its weighted reciprocity is 0.51.
  - Correction: the mushroom body's reciprocity (0.53) is slightly higher, but it is built from 1-2-synapse KC-KC contacts. What sets the AL apart is having heavy weights, reciprocity and excitatory local neurons together.
- **Each region alone (Perron root).** AL 5.57, VNC 1.70, other central brain 1.47 (AVLP016/079/076), central complex 1.28 (ExR6, ExR4, EPG, EL, PEN_a), optic lobe 0.52, LH 0.49, mushroom body 0.47.

### 2.3 Biology or artefact?

| perturbation (whole CNS) | λ₁ | does the AL still lead? |
|---|---:|---|
| none | 5.57 | yes (2.97×) |
| ORNs removed | 5.55 | yes (2.95×) |
| only neurons with a soma (drops every afferent) | 5.50 | yes (2.94×) |
| cell-type-level transmitter signs | 5.57 | yes (2.98×) |
| weights capped at 50/20/10 | 3.34/2.04/1.30 | yes (2.21×/1.76×/1.44×) |
| sqrt / log1p weights | 0.77/0.48 | yes (1.49×/1.32×) |
| binarized weights | 0.28 | no: the MB's KC-KC wiring leads (0.280 vs 0.249) |
| AL local neurons made non-excitatory | 1.88 | no: the AL leaves the top 8 modes |

- **Sensory convergence is ruled out.**
- **Synapse counting matters moderately.** The AL still leads with weights capped at 10. Binary wiring delocalizes, and degree-preserving rewiring removes the localization.
- **The cholinergic identity of lLN1_bc and several lLN2 types is decisive.**
  - The release gives lLN1_bc as ACh 30/30 in its consensus, body-level, cell-type-level and ground-truth fields, with confidence 0.75.
  - Cholinergic excitatory LNs (eLNs) exist (Shang 2007), with reciprocal eLN-PN and eLN-eLN excitation through cholinergic synapses and gap junctions (Huang 2010). But most AL LNs are GABAergic. CNS-wide, the 420 AL LNs are labelled 137 ACh, 115 GABA, 99 Glu and 67 unclear.
  - Whether lLN1_bc are the physiologically characterized eLNs is unchecked.
- **Tracing.** 228 of the 526 module neurons, including all lLN1_bc, are "Prelim Roughly traced" and carry 80% of the mode (22% of neurons CNS-wide have this status). The clique is stereotyped (15 vs 15) and matches the hemibrain and FlyWire types.
- **Normalization.** Per-neuron input normalization gives optic-lobe two-neuron modes instead.
- **Missing biology.** There are no gap junctions and no neuromodulation, and DA/OA/5-HT/unclear neurons have sign 0. eLN→PN excitation is largely electrical and also recruits inhibitory LNs (Yaksi & Wilson 2010), so the real loop could be stronger or weaker.

**Verdict.** The dominance is a property of the reconstructed wiring: a stereotyped, heavily weighted, reciprocal cholinergic LN network, not receptor-neuron convergence. That this makes the AL able to sustain activity on its own rests on unsettled modelling choices: chemical synapses only, the LN signs, and global normalization.

Independent support: Lin 2024 (FlyWire) lists the AL with the central complex as the high-reciprocity neuropils and reports strong left-right AL reciprocal coupling. Their counts include inhibitory neurons, and no paper ranks the AL as the densest neuropil.

### 2.4 How the AL shapes the simulated dynamics

- **Silence becomes unstable** at g_lin = 0.163, independent of r.
- **Slow window.** One metastable module with exponential lifetimes, not a Griffiths phase.
  - It spans g 0.234-0.266 at r = 0 and 0.283-0.344 at r = 1. Its width (onset/start) is 1.14 and 1.22.
  - At r = 1 the mean lifetime rises from about 70 to about 1,650 time units between g = 0.26 and 0.34.
- **Sustained onset rises with inhibition:** 0.266 (r = 0), 0.344 (r = 1), 0.579 (r = 6).
- **The module alone reproduces the brain's onset.** The point where all replicas survive at r = 1 is 0.446 for the module vs 0.436 for the whole brain; at r = 4 it is 0.66 vs 0.63.
- **Random single seeds ignite only via the AL.** The surviving seeds were DL2v_adPN, ORN_DP1m, ORN_DL4 and lLN1_bc.
- **Switch-on order.**
  - At r = 0: AL 0.26, then a motor tier (VNC/GNG plus a VNC+CX ExR6/ExR4 mode) at about 0.50, then abdominal VNC at 0.63.
  - At r = 1: 0.34, then 1.0-1.35, then 2.28.
  - MB and LH carry 19% and 12% of activity at g = 0.30 only because PNs drive them.
  - The central-complex ring switches on about 4.3× above the AL's spectral threshold (about 1.9× for the whole tier in simulation).
- **Delocalization** happens at g 1.47 (r = 0) and 14.5 (r = 1).
- **Weak drive.** With h = 1e-5 at r = 1, the fluctuation peak is at g ≈ 0.30-0.31, where the module flickers. There is no resolved peak for h ≥ 1e-3.

## 3. What is known empirically about the AL and sleep or arousal

### 3.1 Causal manipulations (AL-direct)

- **Hsu, Choi & Sehgal 2021, Sleep** (verified). Strength: moderate, causal, one lab.
  - Adult Kir2.1 silencing of Orco ORNs produced a "significant decrease in both daytime and nighttime sleep" with no rebound. Removing the antennae reduced next-day sleep in young flies.
  - ORN activation and 16 odors did not raise sleep, and GH146 excitatory PNs had no effect.
  - Activating Mz699 GABAergic inhibitory PNs at night "dramatically increased sleep". Mz699 also labels vlPr neurons outside the AL.
  - "No single neuronal population ... bidirectionally regulate[s] sleep". LNs and arousal threshold were not tested.
- **Alpert 2020, Curr Biol.** A posterior-AL cold glomerulus feeds GABAergic TPN-IIs onto DN1a clock neurons, which reshape morning sleep. Strength: moderate to strong, but thermosensory, not about overall AL activity.
- **Singh & Donlea 2020.** Antennal injury raises daytime sleep for under 24 h, but ORN silencing does not. Sleep deprivation slows clearance of injured ORN terminals in the AL.
- **Cook 2025, bioRxiv** (partly verified).
  - The abstract supports ORN-channel-specific effects on day and night sleep, mating-status dependence, and 5-HT receptor heterogeneity.
  - Unchecked and provisional: that BoNT-C silencing of Orco or Ir8a ORNs increased sleep (the opposite sign to Hsu), the single-glomerulus signs, and the CSD-neuron VMAT result.

Net: the sign of the ORN effect on sleep is inconsistent. The clear routes are GABAergic outputs, and no study has tested AL LNs in sleep.

### 3.2 Does AL activity or structure track sleep?

| finding | source | scope; strength |
|---|---|---|
| After sleep deprivation, a CaMPARI snapshot shows the protocerebral bridge most reliably up, EB and MB up, and "little change in signal in the antenna lobe (AL)" | Dai 2024, Sci Adv | AL-direct, negative; moderate (the only region-level test that included the AL) |
| After 16 h SD, AL BRP rose 69 ± 11%, like the MB β lobes, EB and central brain (no sleep-by-region interaction, P = 0.86); AL volume +15% | Gilestro 2009, Science | AL, not specific; strong |
| After 12 h SD, cholinergic BRP rose ~21-39% in each of 37 neuropils; DA/Glu unchanged; GABA up in the prow only; "within transmitter class" | Weiss 2024, PNAS | whole-brain; strong |
| AL and SMP astrocyte Ca²⁺ rise with sleep need | Blum 2021 | AL, one of two regions; moderate |
| Sleep-dependent glial clearance of injured ORN axons in the AL | Stanhope 2020; Zhang 2026 | AL glia, slow; moderate |
| Bushey 2011 (structural synaptic homeostasis) did not examine the AL | | coverage gap |
| Honeybee AL PNs: higher glomerular functional connectivity at rest; a model reproduces this with correlated noise plus reduced LN inhibition, giving a less specific odor code | Moguilner 2025, Neural Netw | AL, bee; weak to moderate (n = 9; rest defined by immobility) |

### 3.3 Does the AL keep working during sleep?

- **French 2021, Nature.** Sleeping flies discriminate odors and wake selectively: 11 of 26 conditions woke them above control, and starvation raised waking to food odors. There are "at least two gate points": one in the ORNs (DTKR in Or85a) and one in the depth of the MB (PAM DANs, MB011B, FSB R38E07). Strength: strong, indirect for the AL.
- **Kirszenblat 2025, preprint** (abstract only). KC odor identity stays intact in sleep while DAN value coding shifts.
- **Zwaka 2015.** In bees, odor in deep sleep enhances memory.
- **Mammals.**
  - Gating occurs "in a majority of olfactory cortex neurons, but in only a small percentage of olfactory bulb neurons" (Murakami 2005, urethane).
  - In slow-wave sleep, cortex drives the OB top-down (Manabe 2011).
  - OB gamma vanishes in slow-wave and REM sleep (Mori 2013).
  - Piriform responses are reduced in slow-wave states (Gaeta & Wilson 2022).
  - So the first relay's dynamics change, but transmission gating is cortical.
- **Species contrast.** Humans are rarely woken by odors (Carskadon & Herz 2004).

### 3.4 The awake AL is spontaneous and tightly coupled

- PNs fire at 1-5 Hz, correlated through divergence from shared ORNs (Kazama & Wilson 2009).
- Ongoing PN activity is organized and plastic (Franco & Yaksi 2021).
- The olfactory cluster has "some of the strongest observed correlations, even in the absence of olfactory input" (Mann 2017).
- The AL/LH and MB are more coupled than wiring predicts (Turner 2021).
- PN components are spontaneously active (Aimon 2019).
- AL PNs also carry auditory activity (Pacheco 2021).

### 3.5 Coverage gaps

- Tainton-Heap 2021 imaged about 100 µm from the dorsal surface and never mentions the AL.
- Yap 2017 recorded the central brain; Raccuglia 2019 recorded R5.
- Schaffer 2023 excluded the AL from its volume.
- No study records the fly AL across verified sleep, tests AL LNs in sleep, or images the AL under anesthesia. Two searches found no counterexample.

**Bottom line.** The evidence fits "the AL is active in both wake and sleep, and state changes propagation beyond the first relay plus downstream neuromodulatory gating". It does not fit "AL active = awake".

## 4. Anesthesia, consciousness and seizures

- **No insect AL recording under anesthesia exists.**
  - The fly isoflurane LFP probe went in laterally through the eye, covering optic lobe then central brain (Cohen 2016, 2018; Leung 2021, 2025; Muñoz 2020). Zero "antennal" hits in those papers.
  - Troup 2023 imaged from the back of the head.
  - Two PubMed queries were rerun (16 and 23 hits) and found nothing.
- **What isoflurane does.**
  - Central power falls more than optic-lobe power, and feedback is preferentially cut (Cohen 2018).
  - Integration "collapsed to isolated clusters" (Leung 2021).
  - Recordings become less complex and more reversible (Muñoz 2020).
  - About 18% of neurons are active awake vs 15% under isoflurane, and patterns fragment (Troup 2023). Wake and sleep both show about 23-24% active (Tainton-Heap 2021).
  - The only generalizing marker is a longer autocorrelation timescale in wake, at the central channel (Leung 2025, 49 blinded flies).
  - Isoflurane impairs cholinergic release and spares GABA/Glu (Hines 2024, MBONs); syntaxin release effects were seen at the larval NMJ (Zalucki 2015).
  - The fan-shaped body pathway controls both sleep and isoflurane sensitivity (Kottler 2013).
  - Graded doses give a succession of endpoints (van Swinderen 2006).
- **The AL in awake recordings.**
  - It has "the most spiking" and the most power above 15 Hz; odors raise 15-100 Hz coherence there; only GABA and the 7B line raised AL 50-100 Hz (Paulk 2013, more than 2,000 sites).
  - AL and LH oscillate spontaneously (Aimon 2019); Aimon 2023 found no AL-specific state claim.
  - But the 74.9 Hz PN EPSC rate equals the ORN rate times the ORN count (Kazama & Wilson 2009), so AL activity is fed forward.
- **Mammalian OB under anesthesia.** It is not silenced.
  - Responses are broader under anesthesia (Rinberg 2006). Wakefulness enhances granule-cell inhibition (Kato 2012).
  - Ketamine-xylazine gives an 80-130 Hz OB rhythm (Średniawa 2021); subanesthetic ketamine gives OB-originating high-frequency oscillations (Hunt 2019).
  - The OB stays respiration-locked under light anesthesia (Fontanini & Bower 2005).
  - Tsuno 2008 reports OB inhibition largest in slow-wave sleep, the opposite direction to Kato.
- **Consciousness.** The central complex is the insect substrate; the AL is "primary odor processing" (Barron & Klein 2016). Merrick 2014 says the "olfactory bulb is unnecessary for endogenic olfactory consciousness" and that a conscious content "must participate in a network that is large-scale". Mori 2013 is a gamma-coupling hypothesis. Neither privileges the first relay.
- **Seizures.**
  - Initiation is at the MB α/β lobes (c739, 117Y, 7B and 121Y give 78-90%). EB, giant fibre, FSB and Gad1 activation did not trigger seizures, and no AL driver was tested (Saras 2017).
  - In Hekmat-Scafe 2010, GH146 rescue was modest (63% still bang-sensitive) and PN tetanus toxin had no effect (97%).
  - The giant fibre is critical in the circuit (Kroll 2015).
  - jus is expressed in the optic lobe, subesophageal zone and VNC (Horne 2017), plus a few AL neurons (Dean 2018, expression only).
  - prickle seizures arise in a brainstem-like region (Lilienthal 2026).
  - The AL is not implicated.

## 5. Criticality and state transitions

- **Ódor 2022** (partly verified). A Kuramoto model on the hemibrain is mean-field-like.
  - Raw graph: "no signs ... [of a] GP".
  - Weights "not enough to introduce visible GP effects, or maybe a very weak ones"; "hard to decide if there is a narrow GP" on the normalized graph. Modularity Q ≈ 0.002.
  - Their raw-weight run was "not smoother", so normalization alone does not explain differences from our result.
  - This is hedged agreement with our single metastable region.
- **Ódor 2025.** Kuramoto on about 10⁵ nodes; the abstract does not mention a Griffiths phase.
- **Pospisil 2024.** Signed-matrix eigenvectors are sparse and localized. The top one is in the lobula plate; the olfactory eigenvectors include AL→LH PNs. A different matrix from ours.
- **Shiu 2024.** An input-driven LIF model with 0 Hz baseline; silent on AL self-sustainment.
- **Theory.**
  - Goltsev 2012: a localized principal eigenvector confines activity, then a crossover follows. This is our diagram.
  - Pastor-Satorras & Castellano 2016: hub vs dense-subgraph localization. Ours is the subgraph type; the null is the hub type.
  - Moretti & Muñoz 2013: a Griffiths phase needs hierarchical rare regions, so one region gives metastability only.
  - Kinouchi & Copelli 2006: dynamic range peaks at criticality (general theory).
- **AL dynamics are odor-evoked and inhibition-shaped.**
  - Picrotoxin removes synchrony; desynchronization impairs fine discrimination (MacLeod & Laurent 1996; Stopfer 1997; Bazhenov 2001).
  - Fly LN-dependent oscillations appear with no spontaneous oscillation (Tanaka 2009).
  - Winnerless competition (Rabinovich 2001).
  - Activity returns to baseline after the odor (Mazor & Laurent 2005).
  - Assisi 2011; Ito 2009 (moth frequency drops suddenly as ORN output falls).
  - The spontaneous state differs from the driven one (Daly 2011: 25-55 vs 55-85 Hz).
  - The bee AL keeps an odor-specific spontaneous-correlation trace for minutes, interpreted as "odor-specific attractors" (Galán 2006). This is the nearest analogue to our slow window, but it rides on driven activity.
- **Drosophila PNs are silent without antennae** (Olsen 2007, palps intact). Inhibition dominates and scales with input (Olsen & Wilson 2008). Silencing a third of the LNs left odor patterns unchanged (Strube-Bloss 2017).
- **Mammalian OB.**
  - Mitral-cell bistability (Heyward 2001); an "intrinsic propensity to oscillate" (Fontanini 2003); state-dependent gamma (Kay 2003); chaos (Skarda & Freeman 1987, abstract only).
  - Sleep-like coherence is cortical (Wilson & Yan 2010).
  - Olfactory rhythms do not drive the slow oscillation (Viczko 2014).
  - Bulbectomy does not abolish an aroused EEG (Watanabe 1980).
- **No avalanche or criticality data exist** for the AL, OB, piriform cortex or whole fly brain. In zebrafish, input moves the brain "to a more ordered regime" (Ponce-Alvarez 2018), the opposite of our weak-drive peak.
- **Persistent fly circuits.** The EB heading bump (Seelig & Jayaraman 2015) and pCd neurons (Jung 2020, added by a checker). Neither is in the AL.

## 6. Alignments, tensions, and two state mappings

### 6.1 Aligns
1. The bilateral, reciprocal AL module matches Lin 2024 (high reciprocity, strong left-right AL coupling) and Pospisil 2024 (localized eigenvectors).
2. AL first and dominant, with MB and LH PN-driven, matches Paulk 2013, Mann 2017, Turner 2021 and Aimon 2019. The outcome matches; the mechanism differs.
3. Inhibition raising the onset (0.266 to 0.579) matches Olsen & Wilson 2008, MacLeod & Laurent 1996, Tanaka 2009 and Paulk 2013. Complicated by Strube-Bloss 2017.
4. No Griffiths phase matches Ódor 2022 (hedged), Moretti & Muñoz and Goltsev.
5. The AL as the lowest-threshold circuit, last to drop out, matches French 2021, Kirszenblat 2025, Murakami 2005, Rinberg 2006 and Kato 2012. Untested in the insect AL.
6. Weaker inhibition making self-sustained AL activity easier matches the Moguilner 2025 bee rest state, which makes the AL-only regime look rest-like.
7. Slow near-onset dynamics match Leung 2025's longer timescales in wake, in kind only and in the central brain.
8. Loss of recruitment as gain falls matches fragmentation in Troup 2023 and Leung 2021.
9. Hines 2024 (a cholinergic g cut) and Weiss 2024 (uniform cholinergic upscaling after SD) give direct parameter perturbations.
10. Galán 2006's odor trace lasting minutes is a partial analogue of the slow window.
- Speculative: the module contains 25 GABAergic PNs, mostly `*_vPN` types (my unverified quick check), driven by the cholinergic core. Hsu 2021 found that activating GABAergic inhibitory PNs increases sleep, so AL reverberation might drive a sleep-promoting output. Correspondence to Mz699 is unchecked.

### 6.2 Tensions
1. **Fed forward vs self-sustained** (Olsen 2007; Kazama & Wilson 2009; Mazor & Laurent 2005). Our AL holds activity at h = 0 above g ≈ 0.27/0.34. Either the real gain is below the AL onset (the real AL is a driven relay) or the model overstates recurrence.
2. **Sleep deprivation.** It upscales cholinergic synapses by 21-39% (Weiss) and AL BRP by 69% (Gilestro). That exceeds our slow-window width (1.14-1.22), yet Dai 2024 saw little AL change. The real AL is probably well below its intrinsic onset. Caveats: BRP is not weight, and CaMPARI integrates over time.
3. **Sleep and anesthesia are not silence** (23-24% and 15-18% of neurons active). h = 0 absorbing silence is not a model of either; compare the weakly driven regime instead.
4. **No sleep homeostat dynamics.** The empirical control points are the dFB (Kottler 2013; French 2021), EB R5 (Raccuglia 2019; Blum 2021), the PB (Dai 2024), DN1a (Alpert 2020) and MB DANs (French; Kirszenblat). The model has no neuromodulation, and the CX ring is second tier (about 4.3×).
5. **Missing gap junctions** (Yaksi & Wilson 2010).
6. **The direction of first-relay inhibition across states is contested.** Kato 2012, Rinberg 2006 and Moguilner 2025 point one way, Tsuno 2008 the other.
7. **The fluctuation peak needs h = 1e-5** and is absent at h ≥ 1e-3, while real ORN drive is continuous.
8. **Consciousness (CX) and seizures (MB α/β) point elsewhere.** Our supercritical side is delocalization, not an AL runaway.
9. **Fragile inputs:** the lLN1_bc sign, preliminary tracing (80% of the mode), and the normalization choice.

### 6.3 Two mappings onto behavioural states

**Mapping A** (the user's original hypothesis):
- wake = the AL near its onset;
- sleep and anesthesia = below onset, the AL quiet;
- seizure = above onset.
- For: only Leung 2025, in kind and in the central brain.
- Against: French 2021, Tainton-Heap 2021, Troup 2023, Dai 2024, Saras 2017, Barron & Klein 2016.

**Mapping B** (literature-guided):
- wake = the AL held near threshold by stronger LN inhibition, driven by ORNs, with propagation beyond the AL;
- sleep = the AL on and more self-coupled (weaker inhibition), with downstream propagation gated;
- anesthesia = cholinergic gain cut, so recruitment beyond the AL is lost first and the AL is last to go;
- seizure is not AL-specific.
- For: Moguilner 2025, French 2021, Kirszenblat 2025, Murakami 2005, Kato 2012, Rinberg 2006, Troup 2023, Hines 2024.
- Against: Tsuno 2008, Dai 2024, and the fact that isoflurane also hits ORN→PN synapses.

Mapping B is better supported but untested at the AL. Under B the useful model order parameter is recruitment outside the module (activity share, participation ratio, AL→MB/LH transfer), not AL activity.

## 7. Testable predictions

1. **Deafferentation reverberation.** Remove the antennae and the maxillary palps (Olsen 2007 left the palps intact). Image cholinergic LNs and PNs, and pulse the AL LNs. The model predicts AL activity outlasts the MB's and LH's, with memoryless lifetimes that grow steeply with cholinergic gain. Refuted if the AL decays as fast as other regions.
2. **Graded isoflurane with the AL in view.** Use Paulk 2013 registered LFP or Aimon 2019 light-field imaging at 0-0.6% isoflurane. Measure per-region power, autocorrelation (the Leung 2025 marker) and fragmentation. The model predicts the CX and motor tiers (about 2-4× the AL threshold) and MB/LH drive drop first, the AL last, with a peak in AL fluctuations at a light dose. Refuted if the AL falls first; note that ORN→PN synapses are cholinergic too.
3. **The AL across verified sleep.** Image the AL plus the MB calyx with sleep verified by arousal threshold (the French 2021 odor-puff assay). Measure glomerular coupling and node degree (Moguilner method) and the PN→KC ratio. Mapping A predicts a quieter AL; mapping B predicts a preserved, more coupled AL with reduced transmission.
4. **AL LN manipulation vs sleep and arousal** (never tested). Use adult-restricted GABAergic vs cholinergic LN lines in the Hsu 2021 design. Measure sleep amount, bouts, arousal threshold and glomerular coupling. In the model, r from 6 to 0 lowers the AL onset from 0.58 to 0.27, and non-excitatory LNs remove the AL from the top modes. Conditional on mapping B: less GABAergic inhibition gives a more rest-like AL and possibly consolidated sleep or a higher odor arousal threshold. Caveat: Strube-Bloss 2017.
5. **Sleep deprivation.** A 21-39% uniform cholinergic gain increase exceeds the 1.14-1.22 window. Compare AL off-response decay and the persistence of the Galán-type correlation trace after 12 h SD vs rested, against MB and LH. Dai 2024 already argues against the AL sitting near its onset.
6. **Recruitment order at arousal transitions.** The model predicts AL first, then the VNC/GNG plus CX ExR tier, then abdominal VNC, with MB/LH lagging. Use whole-brain imaging that includes the AL during waking or emergence from anesthesia. Refuted if the CX or PB leads.
7. **Ignition sites.** Stimulating lLN1_bc-type LNs, or DP1m and DL2v channels, should give longer AL reverberation than other PN types. The uPN mode weights are largest for DP1m, DP1l, DM1, VL2p and DC1. Cook 2025's DP1l and VL2p sleep effects are provisional leads only.
8. **Octopamine disinhibition.** Octopamine raises spontaneous AL output (Rein 2013, bee), so AL self-coupling alone is not a wake signature. Downstream recruitment should separate wake from rest.

## 8. What to simulate next

1. **ORN-only drive and deafferentation.** Fit g so that PNs are silent without ORN input (Olsen 2007) but fire at 1-5 Hz with it (Kazama & Wilson). This bounds the realistic gain relative to the AL onset (0.27/0.34). Repeat the fluctuation-peak analysis with realistic drive.
2. **eLN-PN gap-junction sweep** (Yaksi & Wilson 2010).
3. **Sign and tracing robustness in the dynamics:**
   - lLN1_bc as GABA, as sign 0, and with confidence-weighted signs;
   - weights capped at 10-20;
   - "unclear" LNs set to ±.
4. **Isoflurane.** Scale cholinergic weights (including ORN→PN) by (1 − δ) and keep GABA and Glu. Report regional dropout order, autocorrelation time, fragmentation and active fraction under weak drive.
5. **Sleep pressure.**
   - Apply a uniform +21-39% cholinergic gain (SD) and ask which region changes first.
   - Add a slow homeostatic variable S(t) on dFB (23E10 types), R5 and PB SpsP neurons, feeding back as a gain reduction or as a gate on PN→KC/LH. Test whether "AL on, propagation gated" emerges.
   - Add DN1a circadian drive (Alpert 2020).
6. **Neuromodulators as region-specific g and r:**
   - octopamine as AL disinhibition (Rein 2013);
   - 5-HT from the CSD neurons (Cook, provisional);
   - DANs at the MB (French; Kirszenblat).
7. **Wake order parameter.** Use recruitment outside the module and AL→MB/LH transfer.
8. **Bridges to the literature.**
   - Glomerular coupling and node degree in the slow window vs above onset (Moguilner).
   - In-weight-normalized W (to compare with Ódor 2022).
   - Pospisil's ordering on the MaleCNS.
   - Whether the module's GABAergic vPN types match the Mz699 iPNs (Hsu 2021).
   - Whether the VP2/VP3 PNs match the TPN-IIs (Alpert 2020). Only 3 of 35 VP2/VP3 PNs are in the module, and the two GABAergic VP3 vPNs are not (my quick check, unverified).
9. **Seizure analogue in the MB α/β lobes** (Saras 2017): lower r and raise g there. Can the model produce a runaway that is not centred on the AL?

## Appendix A. How the evidence was checked

- **Scope.** Four fact-checks:
  - Line 1: 41 papers.
  - Line 2: 35 papers plus 6 cited only in the notes.
  - Line 3: 58 DOIs (55 resolved in Europe PMC; 3, plus Ódor 2025, via Crossref or arXiv).
  - Data grounding: 7 claims, with about 40 numbers recomputed (`code/al_verify_3.py`, jobs 175573 and 175597).
  - The lines overlap.
- **Dropped citations: 0.** No cited study was rated "not found" or "misreported".
  - The only "not found" entry was line 1's set of negative claims (no AL sleep recording, no AL LN sleep test, no AL anesthesia imaging). It is used only as a search gap.
  - Excluded by choice: Crowe & Gong, FENS 2026 poster (title only, never verified).
  - Gaeta & Wilson 2022 is cited only for its piriform finding, because it mischaracterizes Hsu 2021 as "bidirectional".
- **Partly verified, used with qualifiers:**
  - Cook 2025: details provisional.
  - Weiss 2024: cholinergic changes, within transmitter class.
  - Tanaka 2009: issue 26.
  - van Swinderen & Greenspan 2003: not relied on.
  - Ódor 2022: hedges and the raw-weight run restored.
  - Lin 2024: reciprocity only; EB/FB lead only on the internal-weight fraction.
  - Ódor 2025: no Griffiths phase in the abstract.
- **Corrections applied:**
  - Seizure initiation is placed in the MB α/β lobes only.
  - Ódor 2022 is not "the only" fly-connectome criticality simulation.
  - The OB's dynamics do change with state; what is unsupported is the first relay driving or gating the transition.
  - The CX is the best-known persistent fly circuit, not the only one (pCd).
  - Olsen 2007 left the palps intact.
  - Hunt 2019 used a subanesthetic dose.
  - Data grounding:
    - c = 1/745.9, not 1/742.7 (no λ affected);
    - the clique value is 3.88, not 3.90;
    - MB reciprocity (0.526) exceeds the AL core's (0.507);
    - "other central" (1.47) exceeds the CX (1.28);
    - the CX is about 4.3× spectrally (1.9× in simulation), not 3×.
- **New in this synthesis and unverified:** the GABAergic PN count (25, mostly `*_vPN`) and the VP2/VP3 count (3 of 35), from `results/module/module_graph.npz` and the cache `nt`/`types` arrays.

## References (DOIs; all verified or partly verified)

**AL and sleep or arousal**
- Alpert 2020, Curr Biol, 10.1016/j.cub.2020.04.038
- Blum 2021, Curr Biol, 10.1016/j.cub.2020.10.012
- Bushey 2011, Science, 10.1126/science.1202839
- Cook 2025, bioRxiv, 10.1101/2025.09.11.675654 (partly verified)
- Dai 2024, Sci Adv, 10.1126/sciadv.ads0652
- French 2021, Nature, 10.1038/s41586-021-03954-w
- Gilestro 2009, Science, 10.1126/science.1166673
- Hsu 2021, Sleep, 10.1093/sleep/zsaa265
- Kirszenblat 2025, bioRxiv, 10.1101/2025.07.04.663253
- Moguilner 2025, Neural Netw, 10.1016/j.neunet.2025.107575
- Raccuglia 2019, Curr Biol, 10.1016/j.cub.2019.08.070
- Singh & Donlea 2020, Curr Biol, 10.1016/j.cub.2019.12.065
- Stanhope 2020, Curr Biol, 10.1016/j.cub.2020.02.057
- Tainton-Heap 2021, Curr Biol, 10.1016/j.cub.2020.10.081
- Weiss 2024, PNAS, 10.1073/pnas.2312664121 (partly verified)
- Yap 2017, Nat Commun, 10.1038/s41467-017-02024-y
- Zhang 2026, PLoS Genet, 10.1371/journal.pgen.1011999
- Zwaka 2015, Curr Biol, 10.1016/j.cub.2015.09.069

**AL physiology**
- Aimon 2019, PLoS Biol, 10.1371/journal.pbio.2006732
- Aimon 2023, eLife, 10.7554/eLife.85202
- Franco & Yaksi 2021, Cell Rep, 10.1016/j.celrep.2021.110165
- Huang 2010, Neuron, 10.1016/j.neuron.2010.08.025
- Kazama & Wilson 2009, Nat Neurosci, 10.1038/nn.2376
- Mann 2017, Curr Biol, 10.1016/j.cub.2017.06.076
- Olsen 2007, Neuron, 10.1016/j.neuron.2007.03.010
- Olsen & Wilson 2008, Nature, 10.1038/nature06864
- Pacheco 2021, Nat Neurosci, 10.1038/s41593-020-00743-y
- Paulk 2013, J Neurophysiol, 10.1152/jn.00414.2013
- Rein 2013, J Comp Physiol A, 10.1007/s00359-013-0805-y
- Schaffer 2023, Nat Commun, 10.1038/s41467-023-41261-2
- Shang 2007, Cell, 10.1016/j.cell.2006.12.034
- Strube-Bloss 2017, Sci Rep, 10.1038/s41598-017-08090-y
- Turner 2021, Curr Biol, 10.1016/j.cub.2021.03.004
- Yaksi & Wilson 2010, Neuron, 10.1016/j.neuron.2010.08.041

**AL dynamics**
- Assisi 2011, Neuron, 10.1016/j.neuron.2010.12.019
- Bazhenov 2001, Neuron, 10.1016/S0896-6273(01)00286-0
- Daly 2011, Front Neuroeng, 10.3389/fneng.2011.00012
- Galán 2006, Neural Comput, 10.1162/089976606774841558
- Ito 2009, Neuron, 10.1016/j.neuron.2009.10.004
- MacLeod & Laurent 1996, Science, 10.1126/science.274.5289.976
- Mazor & Laurent 2005, Neuron, 10.1016/j.neuron.2005.09.032
- Rabinovich 2001, PRL, 10.1103/PhysRevLett.87.068102
- Stopfer 1997, Nature, 10.1038/36335
- Tanaka 2009, J Neurosci 29(26), 10.1523/JNEUROSCI.1455-09.2009

**Anesthesia, consciousness and seizures**
- Barron & Klein 2016, PNAS, 10.1073/pnas.1520084113
- Cohen 2016, eNeuro, 10.1523/ENEURO.0116-16.2016
- Cohen 2018, eNeuro, 10.1523/ENEURO.0329-17.2018
- Dean 2018, Fly, 10.1080/19336934.2017.1402993
- Hekmat-Scafe 2010, Genetics, 10.1534/genetics.109.109074
- Hines 2024, J Neurosci, 10.1523/JNEUROSCI.0588-23.2024
- Horne 2017, Genetics, 10.1534/genetics.116.199083
- Kottler 2013, Curr Biol, 10.1016/j.cub.2013.02.021
- Kroll 2015, Genetics, 10.1534/genetics.115.177600
- Leung 2021, PLoS Comput Biol, 10.1371/journal.pcbi.1008722
- Leung 2025, PLoS Biol, 10.1371/journal.pbio.3003217
- Lilienthal 2026, J Neurogenet, 10.1080/01677063.2026.2742881
- Muñoz 2020, PRR, 10.1103/PhysRevResearch.2.023219
- Saras 2017, Genetics, 10.1534/genetics.116.194779
- Troup 2023, J Neurosci, 10.1523/JNEUROSCI.1657-22.2023
- van Swinderen 2006, J Neurobiol, 10.1002/neu.20300
- van Swinderen & Greenspan 2003, Nat Neurosci, 10.1038/nn1054 (partly verified)
- Zalucki 2015, Anesthesiology, 10.1097/ALN.0000000000000629

**Mammalian olfactory system**
- Carskadon & Herz 2004, Sleep, 10.1093/sleep/27.3.402
- Fontanini 2003, J Neurosci, 10.1523/JNEUROSCI.23-22-07993.2003
- Fontanini & Bower 2005, J Neurophysiol, 10.1152/jn.01320.2004
- Gaeta & Wilson 2022, Front Neural Circuits, 10.3389/fncir.2022.1076354
- Heyward 2001, J Neurosci, 10.1523/JNEUROSCI.21-14-05311.2001
- Hunt 2019, Neuropsychopharmacology, 10.1038/s41386-018-0173-y
- Kato 2012, Neuron, 10.1016/j.neuron.2012.09.037
- Kay 2003, J Integr Neurosci, 10.1142/S0219635203000196
- Manabe 2011, J Neurosci, 10.1523/JNEUROSCI.6578-10.2011
- Merrick 2014, Front Psychol, 10.3389/fpsyg.2013.01011
- Mori 2013, Front Psychol, 10.3389/fpsyg.2013.00743
- Murakami 2005, Neuron, 10.1016/j.neuron.2005.02.025
- Rinberg 2006, J Neurosci, 10.1523/JNEUROSCI.0884-06.2006
- Skarda & Freeman 1987, BBS, 10.1017/S0140525X00047336
- Średniawa 2021, Sci Rep, 10.1038/s41598-021-85705-5
- Tsuno 2008, J Neurosci, 10.1523/JNEUROSCI.1576-08.2008
- Viczko 2014, J Neurosci, 10.1523/JNEUROSCI.3581-13.2014
- Watanabe 1980, Tohoku J Exp Med, 10.1620/tjem.130.41
- Wilson & Yan 2010, J Neurophysiol, 10.1152/jn.00711.2010

**Connectomes, models and theory**
- Goltsev 2012, PRL, 10.1103/PhysRevLett.109.128702
- Jung 2020, Neuron, 10.1016/j.neuron.2019.10.028 (added by a checker)
- Kakaria & de Bivort 2017, Front Behav Neurosci, 10.3389/fnbeh.2017.00008
- Kinouchi & Copelli 2006, Nat Phys, 10.1038/nphys289
- Lappalainen 2024, Nature, 10.1038/s41586-024-07939-3
- Lin 2024, Nature, 10.1038/s41586-024-07968-y (partly verified)
- Moretti & Muñoz 2013, Nat Commun, 10.1038/ncomms3521
- Ódor 2022, PRR, 10.1103/PhysRevResearch.4.023057 (partly verified)
- Ódor 2025, arXiv:2503.20708 / 10.1016/j.chaos.2026.119150 (partly verified)
- Pastor-Satorras & Castellano 2016, Sci Rep, 10.1038/srep18847
- Ponce-Alvarez 2018, Neuron, 10.1016/j.neuron.2018.10.045
- Pospisil 2024, Nature, 10.1038/s41586-024-07982-0
- Seelig & Jayaraman 2015, Nature, 10.1038/nature14446
- Shiu 2024, Nature, 10.1038/s41586-024-07763-9