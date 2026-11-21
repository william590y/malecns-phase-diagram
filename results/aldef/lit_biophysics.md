# AL-defect question: literature notes on AL biophysics (2026-11-09)

Notes file (not a report) for the question "is the antennal lobe's self-sustained ignition at h = 0 a defect of our
stochastic Wilson-Cowan model?". It covers literature only; no simulations. Arithmetic that turns published numbers
into model units: `code/ald_lit_calc.py` -> `results/aldef/lit_biophysics_calc.json` (cited below as **[calc]**).

**How sources were read.** Full texts come from NCBI efetch, PMC HTML or Europe PMC XML. Downloaded copies (not
project files) are in the session scratchpad `.../scratchpad/lit/*.txt`. Each entry says whether I read the full text
or only the abstract. Quotes are copied from those texts. Two papers were blocked by publisher bot protection
(cell.com, journals.physiology.org), so only their abstracts are used: Huang et al. 2010 and Seki et al. 2010.
Yang et al. 2011 is also abstract only.

Project numbers reused (not re-derived here): `results/al_lit/data_grounding.md` and `REPORT.md` (λ 5.57; the
lLN1_bc clique; λ 1.88 when AL local neurons are made non-excitatory; 420 ALLNs by consensus 137 ACh / 115 GABA /
99 Glu / 67 unclear), `CLAUDE.md` (c = 1/742.7, dt = 0.2, α_eff = 0.906), and `STATE.md` (driven-plane peak g ≈ 0.30
at h = 1e-5; no resolved peak for h ≥ 1e-3).

---------------------------------------------------------------------------------------------------------------------

## 1. Identity of lLN1_bc, lLN1_a and lLN2: are they the cholinergic eLNs?

**Bottom line.** It is plausible that lLN1_bc are cholinergic, and they may be (part of) the physiologically
characterised excitatory LNs (eLNs). No paper ties the EM type lLN1_bc to the krasavietz eLNs, and there is no direct
staining of lLN1_bc. The best EM-transmitter study points to lLN2T_a/b/c as the closest morphological match to the
krasavietz eLNs, and it predicts lLN1_bc as cholinergic, with some members predicted dopaminergic. Immunostaining
gives a cholinergic LN fraction (~4-8 %) far below the maleCNS consensus labels (33 %).

### 1.1 EM type definitions
- **Schlegel et al. 2021, eLife 10:e66018** (hemibrain olfactory connectome). Full text (PMC8298098).
  - "We find 196 ALLNs in the right hemisphere which we assign to 5 lineages, 4 morphological classes, 25 anatomical
    groups and 74 cell types".
  - "ALLNs derive from three main neuroblast clones: the lateral neuroblast lineage ('l' and 'l2' from ALl1), the
    ventral neuroblast lineage ('v' from ALv1) and the ventral ALLN-specific lineage ('v2' from ALv2)".
  - "Strong ALLN-ALLN connectivity arises mostly from the broad ALLNs of the lateral lineage (Figure 5C). They may act
    as master regulators of the ALLN network." This matches our module, where the lateral-lineage lLN1_bc/lLN2 carry the
    mode.
  - The main text never mentions "lLN1", and it assigns no transmitter to LN types. Its transmitter guesses use a
    hemilineage rule ("if brain neurons share a hemilineage they will share their fast-acting transmitter
    expression"), applied to third-order neurons. Status: verified, but it does not settle the lLN1_bc transmitter.
- **Schlegel et al. 2024, Nature** (FlyWire cell typing) and **Dorkenwald et al. 2024, Nature**. Full texts
  (PMC11446831, PMC11446842) contain no lLN1 or excitatory-LN statement. The type name lLN1_bc comes from these
  annotation releases. That the maleCNS lLN1_bc annotation matches hemibrain lLN1_b/lLN1_c and FlyWire lLN1_bc is from
  `data_grounding.md`.

### 1.2 EM neurotransmitter predictions
- **Eckstein et al. 2024, Cell 187:2574** ("Neurotransmitter classification from electron microscopy images at
  synaptic sites in Drosophila melanogaster"). Full text (PMC11106717), including the STAR Methods section "Working
  through antennal lobe local neuron expression". Status: verified. Key sentences:
  - "From the literature, we expected ∼ 54 local neurons per hemisphere of which ∼ 90% would be GABAergic. We also
    expected perhaps as many as ∼ 15 cholinergic local neurons."
  - "we find ∼ 126 local neurons per hemisphere for 'ALl1 dorsal'. We found that 143 (left: 74, right: 69) are
    predicted GABAergic, 48 (left: 20, right: 28) cholinergic, 13 (left: 9, right: 4) dopaminergic, 14 (left: 7,
    right: 7) glutamatergic and 37 (left: 18, right: 19) serotonergic."
  - "The predicted cholinergic neurons are unilateral and 'broad', i.e., pan-glomerular, in accordance with previous
    descriptions but more numerous. Positioned closely among them are other local lLN1_bc neurons that are predicted to
    be dopaminergic." So the predicted-cholinergic broad group includes lLN1_bc, and part of lLN1_bc was predicted DA,
    which the authors suspect is a misprediction caused by neuropeptide co-expression.
  - "Specifically, among them are ∼ 13 lLN2T_abc neurons that have a cell body placement and general morphology most
    similar to reported cholinergic local neurons." These are among the serotonin-mispredicted neurons. Figure S5F
    legend: the most ventral ALl1 cell bodies "are the most similar to described Krasavietz positive cholinergic local
    neurons ... We suspect they all should be predicted for acetylcholine."
  - "A subset of lLN1 neurons is known to express both GABA and glutamate to effect synaptic plasticity", citing
    Das et al. 2011 PNAS.
  - Patchy LNs: "Half the local neurons that had previously been categorized as LN2 'patchy' local neurons ... were
    predicted glutamatergic (cell type: lLN2P_a) and half GABAergic (cell types: lLN2P_b, lLN2P_c)." Our maleCNS cache
    labels lLN2P_a GABA (`data_grounding.md`), so the two datasets disagree on that type.
  - ALv2: "133 (left: 67, right: 66) glutamatergic neurons alongside a newly identified sub-population of 28 (left: 14,
    right: 16) potential cholinergic neurons."
  - "Surprisingly, 18%–27% of antennal lobe local neurons may be cholinergic, suggesting that lateral excitation is a
    more prominent feature of antennal lobe processing than previously thought."
  - These AL LNs "seem to break Dale's law and Lacin's law ..., with similar morphology types predicted to express
    different transmitters". The authors list the AL among the worst-predicted regions ("along with the antennal lobe we
    considered it one of the worst predicted parts of the brain in the HemiBrain dataset").
  - Ground truth. The paper's ground-truth table (Data S, `mmc3.csv`, 5,603 neurons with `known_nt`) contains only
    3 AL LN entries (v2LN38 GABA, v2LN34E GABA, vLN28/29 GABA) and no lLN1 or lLN2T. So the maleCNS `ground_truth = ACh`
    for all 30 lLN1_bc (`data_grounding.md`) does not come from Eckstein's ground truth. I could not find where the
    maleCNS ground_truth field comes from: the cache docs (`~/malecns-flygym/docs/data.md`) do not say, and no maleCNS
    methods text turned up in search. **Open.**

### 1.3 Direct transmitter evidence (immunostaining, driver lines)
- **Shang et al. 2007, Cell 128:601** ("Excitatory local circuits ..."). Full text (PMC2866183). Verified.
  - "We describe a previously unidentified population of cholinergic local neurons (LNs) with multiglomerular
    processes."
  - "The line termed krasavietz-GAL4 marks a dozen or so LNs in the antennal lobe and many KCs in the mushroom body".
  - "The fractions of cholinergic neurons in the GFP-expressing cell populations were substantial at 71 and 63 % in
    KL107-GAL4 and krasavietz-GAL4, respectively (Table S1)."
  - With a ChAT-promoter reporter: "eight to 11 large, brightly fluorescent neurons extending a visible process into the
    lobe were identified as cholinergic by the presence of dsRed". Figure 7A: "dense innervation of all glomeruli from a
    laterally situated cluster of 8 LNs".
  - So there are about 8-11 (antibody count ~15 krasavietz cells) cholinergic, pan-glomerular LNs per AL, in a lateral
    cluster. This is the same order as 15 lLN1_bc per lobe.
- **Chou et al. 2010, Nat Neurosci 13:439** (LN diversity). Full text (PMC2847188). Verified.
  - "the lower bound for Drosophila LNs (labeled by our Gal4 lines) is ∼100 ipsilaterally projecting and ∼100
    bilaterally projecting LNs for each antennal lobe."
  - "We found that the vast majority of LNs are GABAergic; however, there are a few cholinergic cells, and a larger
    minority that are neither GABAergic nor cholinergic".
  - "Staining for additional neurotransmitter types identifies a few dopaminergic and a large number of (mostly
    ventral) glutamatergic LNs".
  - Antibody specificity was checked in Gad1 and Cha mutant MARCM clones. Exact counts are in Supplementary Table 1,
    which I did not open.
- **Seki et al. 2010, J Neurophysiol** (doi:10.1152/jn.00249.2010). Abstract only. "we found that LNs labeled by a
  specific enhancer trap line (GAL4-Krasavietz), which had previously been reported as cholinergic LNs, were mostly
  GABAergic." This directly conflicts with Shang 2007.
- **Yaksi & Wilson 2010, Neuron 67:1034.** Full text (PMC2954501). Verified.
  - "about half of the LNs labeled by krasavietz-Gal4 are GABA-negative (58–61%; Chou et al., 2010; Shang et al.,
    2007)".
  - "Shang et. al. (2007) reported that GABA-negative krasavietz LNs are immunopositive for choline acetyltransferase
    (Cha), although a later study reported that not all are Cha-positive (Chou et al., 2010)".
  - Soma position: "GABA-negative krasavietz LN somata were located just ventro-lateral to the antennal lobe neuropil
    ... most of the GABA-positive labeled somata were dorso-lateral".
- **Das et al. 2011, Neural Syst Circuits 1:4** (glutamatergic LN lineage). Full text (PMC3257541). Verified.
  - "a discrete cluster of approximately 65 labelled cells located ventral and slightly lateral to the neuropil of the
    antennal lobe" (OK371, DVGLUT-positive, GABA-negative).
  - Lateral-lineage LNs are "largely inhibitory (GABAergic) with a few excitatory (cholinergic) cells".
- **Nagel & Wilson 2016, J Neurosci 36:4325.** Full text (PMC4829653). "Within each antennal lobe, ∼50 GABAergic LNs
  express Gal4, whereas the remaining ∼50 GABAergic LNs do not (Chou et al., 2010)." That gives ~100 GABAergic LNs
  per AL.
- **Task et al. 2022, eLife 11:e72599.** Abstract checked. It maps chemoreceptor co-receptor expression in ORNs. It has
  no LN transcriptomics and no LN transmitter data, so it is **not relevant** to LN identity. I found no single-cell
  transcriptomic census of adult AL LN transmitters.
- **Tobin, Wilson & Lee 2017, eLife 6:e24838.** Full text (PMC5440167). An EM study of glomerulus DM6 (ORN→PN
  wiring). It defines no LN types. It is used in section 3 for weights.

### 1.4 Fractions of AL LN transmitters (my arithmetic, rough)
| source | GABA | Glu | ACh | other / unclear |
|---|---|---|---|---|
| Immunostaining/genetics (Chou 2010 ≥ ~200 LNs/AL; ~100 GABA per Nagel & Wilson 2016; ~65 Glu per Das 2011; ~8-15 ACh per Shang 2007) | ~50 % | ~30 % | ~4-8 % | "larger minority neither GABA nor ChAT" (Chou), a few DA |
| EM prediction, FlyWire (Eckstein 2024) | majority of ALl1-dorsal | ALv2 mostly Glu | **18-27 %** | 5-HT/DA mispredictions |
| maleCNS consensus, 420 ALLNs (`data_grounding.md`) | 115 (27 %) | 99 (24 %) | **137 (33 %)** | 67 unclear (16 %) |

The model's cholinergic LN fraction (33 %) is the high end of every estimate. Immunostaining (Shang, Chou) supports
only about 10-15 cholinergic LNs per AL.

---------------------------------------------------------------------------------------------------------------------

## 2. How eLNs act

- **eLN → PN is electrical, not cholinergic.** Yaksi & Wilson 2010 (full text):
  - "eLN-to-PN synapses transmit both hyperpolarization and depolarization, are not diminished by blocking chemical
    neurotransmission, and are abolished by a gap-junction mutation" (abstract).
  - "the nicotinic acetylcholine receptor antagonist mecamylamine had no effect, although two other nicotinic
    antagonists had weak effects in some experiments".
  - "Our results suggest that eLNs do not release acetylcholine onto PNs."
  - "eLN-to-iLN synapses are largely cholinergic, whereas eLN-to-PN synapses are mainly or purely electrical."
  - Connectivity: of "74 dual recordings from PNs and krasavietz LNs, ... 37 showed an excitatory LN-PN connection, 17
    showed an inhibitory LN-PN connection"; and "each eLN is connected to most or all PNs".
- **PN → eLN is mixed.** "These results imply that PN-to-eLN synapses are mixed chemical-electrical synapses." "PNs
  release acetylcholine onto eLNs, in addition to coupling electrically to eLNs."
- **eLNs recruit iLNs.** Abstract: "eLNs have two opposing effects on PNs, driving both direct excitation and indirect
  inhibition. We propose that when stimuli are weak, lateral excitation promotes sensitivity, whereas when stimuli are
  strong, lateral excitation helps recruit inhibitory gain control." Eliminating these synapses "boosts some PN odor
  responses".
- **Huang et al. 2010, Neuron 67:1021.** Abstract only. "Paired recordings of krasavietz eLNs and PNs showed reciprocal
  excitatory connections mediated by dendrodendritic cholinergic synapses and gap junctions. Reciprocal connections were
  also found between two krasavietz eLNs but were rare between krasavietz eLNs and inhibitory LNs."
  - This conflicts with Yaksi & Wilson on two points: a cholinergic component of eLN→PN, and how common eLN→iLN
    connections are. Yaksi & Wilson found the eLN→iLN depolarization "in many of these pairs".
  - eLN-eLN coupling exists (Huang), but the abstract gives no mechanism or strength, and I found no measurement of it.
- **Tonic inhibition of eLNs.** eLNs "were always barraged by spontaneous inhibitory postsynaptic potentials (IPSPs)"
  (Yaksi & Wilson). Cd2+ "blocks spontaneous IPSPs and so increases eLN excitability". Kazama et al. 2011 (Fig. 3
  legend): "Note large spontaneous IPSPs (arrowhead), which are typical of eLNs."
- **Overall strength.** Kazama, Yaksi & Wilson 2011 (J Neurosci 31:7619; full text PMC3119471): "In the normal circuit,
  excitatory connections between glomeruli are weak." Olsen & Wilson 2008 (Nature 452:956; full text PMC2824883):
  "interglomerular inhibition dominates over interglomerular excitation".
- **Lateral excitation saturates.** Olsen et al. 2007 (Neuron 54:89; full text PMC2048819): "increasing the rate of
  incoming ORN spikes from 50 to 150 spikes/second had little effect on the amount of lateral excitatory input that was
  broadcast to other glomeruli".
- **Are eLNs spontaneously active?** Not quantified in any text I could read.
  - Chou 2010 (lines included krasavietz = Line9; antennae intact): "All these LNs fired spontaneous action potentials,
    and their spiking was always modulated by odors." Also: "On average, pan-glomerular LNs had significantly higher
    spontaneous firing rates than other LNs (Fig. 5b)."
  - Yang et al. 2011 (J Neurophysiol 106:2127; abstract only): "a cluster of large local interneurons (LNs) located
    laterally to each antennal lobe ... exhibit extensive arborizations throughout the whole antennal lobe. Dual
    recordings showed that these large LNs demonstrated rhythmic spontaneous activities that correlated with other LNs
    and projection neurons (PNs)". The preparation (isolated brain or in vivo, adult or pupa) could not be checked.
    Weak.
- **Persistent or reverberant AL activity without input.**
  - Honeybee: Galán et al. 2006 (Neural Comput 18:10; author PDF read). "during the first few minutes after odor
    presentation, correlations between the spontaneous activity fluctuations suffice to reconstruct the stimulus". The
    authors call this "a type of reverberation that is rather distinct from persistent activity". The antennae were
    intact, so this is not self-sustained activity without input.
  - Drosophila: I found no in vivo report of self-sustained AL activity without afferents. The closest items are in
    section 4: isolated-brain PN spiking (Gu & O'Dowd 2006, data not shown), and AL calcium fluctuations with the
    antennae and proboscis removed (Mann et al. 2017).

---------------------------------------------------------------------------------------------------------------------

## 3. Synaptic dynamics, intrinsic properties and firing rates

### 3.1 Short-term depression (STD)
Convention: published fits multiply the amplitude by f after each spike and recover with time constant τ. f is not the
release probability.
- **ORN→PN, Kazama & Wilson 2008, Neuron 58:401.** Full text (PMC2429849).
  - "Estimated release probability was consistently high, with a mean value of 0.79. The number of release sites was
    also high, with a mean value of 51."
  - "Averaged across PNs, uEPSP amplitude was 6.19 ± 0.45 mV (n = 23)."
  - "Strong inputs rapidly depress ORN-PN synapses". At 7 Hz, depression has "a mainly presynaptic origin".
  - Abstract: "strong stimuli produce short-term depression at this synapse. This helps explain why PN odor responses
    are transient".
- **ORN→PN, Kazama & Wilson 2009, Nat Neurosci 12:1136.** Full text (PMC2751859). Methods: "The parameters α = 0.72 and
  τ = 2.4 s were obtained from a least-squares regression using these data". This was a DM4 nerve stimulated at the
  spontaneous ORN rate.
- **ORN→PN, Nagel, Hong & Wilson 2015, Nat Neurosci 18:56.** Full text (PMC4289142).
  - "f = 0.78 and τ = 893 ms" (10 Hz train, n = 19 PNs).
  - Two components: "For the IMI-resistant component, these parameters were f = 0.77, and τ = 1006 ms, whereas for the
    curare-resistant component, these parameters were f = 0.91, τ = 629 ms".
- **ORN→LN, Nagel & Wilson 2016.** Full text.
  - "Values of f and τ are 0.75 and 1566 ms for LNs; 0.78 and 893 ms for PNs."
  - "EPSCs measured in LNs showed more pronounced depression than those measured in PNs did." These LNs are mostly
    GABAergic (NP3056 and others).
- **iLN→LN.** "excitatory synapses onto LNs are fast and depressing, whereas inhibitory synapses are slow and
  facilitating" (Nagel & Wilson 2016). No parameters given.
- **LN→PN.** This inhibition is largely presynaptic, on ORN terminals.
  - Nagel 2015: "The time course of inhibition could be fit with an alpha function with a time constant of about 25 ms."
  - Wilson & Laurent 2005 (J Neurosci 25:9069; full text PMC6725763): "Whereas GABAA receptors shape PN odor responses
    during the early phase of odor responses, GABAB receptors mediate odor-evoked inhibition on longer time scales."
    Also: "These odor-evoked inhibitory epochs can last from ∼100 ms to several seconds."
  - Olsen & Wilson 2008 abstract: the inhibition "scales with total feedforward input to the entire antennal lobe" and
    is "mediated by both ionotropic and metabotropic receptors on the same nerve terminal".
- **Not found:** STD parameters for PN→LN, eLN→PN (electrical; coupling coefficients only in figures), eLN→eLN,
  LN→LN cholinergic, and PN→PN. **The lLN1_bc→lLN1_bc synapse, which carries 42 % of λ, has never been measured.**

**[calc] Steady-state depression at real rates** (literature parameters only; Poisson-train values;
`steady_state_A` in the JSON):
- **ORN→PN at the DM4 spontaneous rate of 3.44 Hz:** 0.30 (K&W 2009 fit), 0.60 (Nagel 2015), 0.56 (fast component),
  0.84 (slow component).
- **ORN→LN at 3.44 Hz:** 0.43.
- **At 20 Hz:** 0.07 / 0.20 / 0.18 / 0.47 (ORN→PN, same order) and 0.11 (ORN→LN).
- **At 50 Hz:** 0.03 / 0.09 / 0.08 / 0.26 and 0.05.

So AL excitatory synapses already run at about 30-60 % of their rested strength at rest. At rates of tens of Hz, the
fast component falls to under about 20 %.

### 3.2 Intrinsic properties
- **PNs show no spike-frequency adaptation.** Kazama & Wilson 2008: "Even when injected current was sufficient to
  produce firing rates >100 spikes/s, PN responses did not decline over the course of 500 ms (final firing rates were
  104.2% of the initial rate, n = 8 cells)." Input resistance is "598.0 ± 69.3 MΩ, n = 14, measured with antennae
  removed" (Gouwens & Wilson 2009, J Neurosci 29:6239; full text PMC2709801).
- **LNs adapt mildly, and it differs by class.**
  - Nagel & Wilson 2016: "a prolonged light stimulus produced ongoing spiking with mild adaptation". Some LNs fire
    regularly and others in bursts; bursty cells sit at more hyperpolarized resting potentials.
  - Seki 2010 (abstract): the LN classes differ in "firing patterns, degree of spike adaptation, and amplitude of spike
    afterhyperpolarization"; "one class of LNs had characteristic burst firing properties, whereas the others were
    tonically active".
  - Kazama 2011: after chronic deafferentation, "There is no significant change in eLN excitability."
- **Refractoriness and maximum rates.** Evoked PN rates exceed 150 spikes/s (Kazama & Wilson 2009: "a powerful
  increase in firing rate (>150 spikes/sec)"). Palp ORN dose-response curves reach "saturation at ∼225 spikes/sec"
  (de Bruyne et al. 1999, J Neurosci 19:4520; full text PMC6782632). So the absolute refractory period is short, a few
  ms.

### 3.3 Firing rates
**ORN spontaneous:**
- DM4: 3.44 ± 0.16 Hz (Kazama & Wilson 2009).
- DM4 "4 spikes/s" and VM2 "about 7 spikes/s" (R.I.W., unpublished observations, cited in Kazama & Wilson 2008).
- VM7: "Each VM7 ORN fires spontaneously at ~10 spikes/s" (Olsen & Wilson 2008).
- Palp ORNs: "most neurons exhibiting frequencies between 3 and 13 spikes/sec"; pb2B "32 ± 7 spikes/sec" (de Bruyne
  1999).

**PN spontaneous:**
- "In the absence of odors, PNs fire spontaneously (typically 1–5 spikes/sec)" (Kazama & Wilson 2009).
- Spontaneous EPSC rate: "74.9 ± 8.6 Hz, n = 8, one antenna removed" ≈ 3.44 Hz × "17.4 ± 0.9 per antenna" (Kazama &
  Wilson 2009).
- Spontaneous PN spikes are ORN-driven: "this drives spontaneous action potentials in PNs" (Gouwens & Wilson 2009).

**LN spontaneous:**
- 2.3 ± 0.2 spikes/s, cell-attached, n = 12 (Wilson & Laurent 2005).
- "4.6 ± 2.8 spikes/s, mean ± s.d. across cells" (Nagel 2015; Nagel & Wilson 2016).
- Pan-glomerular LNs fire above the mean (Chou 2010).

**Evoked:** PNs above 150 spikes/s; ORNs "50 to 150 spikes/second" (Olsen 2007) and up to ~225 spikes/s (de Bruyne
1999).

### 3.4 Synapse-count weights (cause b)
- Tobin et al. 2017:
  - "On average, unitary ORN→PN connections were composed of about 23 synapses".
  - "This result argues that synapse number is the main contributor to connection strength variations".
  - "Most connections had summation efficacies near 0.9, indicating weakly sublinear summation."
- So **within ORN→PN**, near-linear count weighting is supported. Nothing comparable exists for LN dendrodendritic
  synapses. At the network level, the eLN-mediated lateral excitation saturates (Olsen 2007, section 2).

---------------------------------------------------------------------------------------------------------------------

## 4. Deafferentation (input removal) evidence

**Acute, in vivo (minutes):**
- **Olsen et al. 2007.**
  - "PNs postsynaptic to antennal glomeruli in antennae-less flies show no spontaneous activity". Figure 1D legend:
    "Spontaneous activity is abolished".
  - Timing: "ORN input was removed acutely (~10–20 minutes before recording)".
  - The palps were intact; with antennae and palps both removed, "No depolarization is observed" to odors.
- **Kazama & Wilson 2008:** spontaneous EPSCs "are completely absent when direct ORN input to a glomerulus is removed
  while keeping the lateral inputs intact".
- **Kazama et al. 2011:** "we observed that spike-driven spontaneous EPSPs are absent in postsynaptic PNs after ORN
  somata are removed (Fig. 1B)".
- **Kazama & Wilson 2009:** PN pairs deprived of direct ORN input still share correlated central input, "these
  fluctuations are much smaller than the correlated membrane potential fluctuations driven by ORNs".
- **Antennae are removed routinely to quiet the AL.** Yaksi & Wilson 2010: "In all dual recording experiments, the
  antennae were removed just before the experiment in order to minimize spontaneous activity." Kazama 2011: "Palps were
  removed acutely to minimize spontaneous activity in the network."

**Chronic, in vivo (2-5 days):**
- **Kazama, Yaksi & Wilson 2011.**
  - Configuration: "either antennae or palps were removed within several hours after eclosion, and recordings were
    performed 2 d later".
  - Main result: "odor-evoked lateral excitatory input to deafferented projection neurons was potentiated severalfold.
    This was caused, at least in part, by strengthened electrical coupling from excitatory local neurons onto projection
    neurons, as well as increased activity in excitatory local neurons."
  - Spontaneous activity: "spontaneous membrane potential fluctuations in these PNs were larger after chronic antennal
    removal compared with acute removal". "Spontaneous activity is also potentiated after chronic palp removal".
  - Waiting 5 d produced "even larger changes".
  - The trigger is axon severing and glia, not silence: "Merely silencing receptor neurons was not sufficient".
  - Interpretation: "this mechanism might help boost activity to compensate for lost sensory input". Without input, the
    AL becomes *more* excitatory over days (homeostatic). Self-sustained firing was not reported.

**Isolated brain (no sensory organs, < 1 h):**
- **Gu & O'Dowd 2006, J Neurosci 26:265.** Full text (PMC6674319).
  - Kenyon cells "are members of a spontaneously active neural circuit in the isolated brain". nAChRs "are responsible
    for most of the spontaneous excitatory drive in this circuit in the absence of normal sensory input".
  - "recordings from projection neurons in our isolated brain preparation show that these cells fire spontaneous action
    potentials under the same recording conditions reported in the present study (data not shown)."
  - This is the only Drosophila statement I found of PN spiking with no sensory organs attached. It is data not shown,
    in saline ex vivo, and the source of the drive is unknown. Weak.
- **Yang 2011** (abstract only; preparation not checked): rhythmic spontaneous activity of large lateral LNs.

**Whole-brain imaging, input removed acutely:**
- **Mann, Gallen & Clandinin 2017, Curr Biol 27:2389.** Full text (PMC5967399).
  - Setup: "the proboscis, antennae, and surrounding cuticle were removed" (the palps sit on the proboscis). "Removal
    of antennae prevented possible olfactory or auditory inputs."
  - Olfactory regions "represent some of the strongest observed correlations, even in the absence of olfactory input".
  - The signal is pan-neuronal calcium. It shows AL fluctuations are coherent with LH/MB without afferents. It does not
    give spike rates and does not show self-sustained firing.

**Not found:** any Drosophila recording of PN or eLN firing rates over days to weeks after complete removal of both
antennae and palps.

---------------------------------------------------------------------------------------------------------------------

## 5. Parameter ranges for the model, and the time mapping

### 5.1 Mapping assumption (stated explicitly)
- **Model facts.** A unit goes 1→0 at α = 1 and 0→1 at β·tanh(s)₊ with β = 1. Steps are synchronous with dt = 0.2, so
  p_off = 0.181 per step, α_eff = 0.906, and the mean active epoch is 1.10 time units [calc].
- **Assumption.** One model time unit = τ_u ms, and one active epoch = one spike (with its refractory/integration
  window). Then a neuron's rate is ν = α_eff·ρ/τ_u, and with saturating drive ρ_max = 0.5, so ν_max = 0.453/τ_u.
- **Mapping A (recommended), τ_u = 10 ms.** This is Benayoun et al. 2010's own choice (PLoS Comput Biol 6:e1000846;
  full text PMC2900286, equations read from the PLOS formula images): "α = 0.1 ms⁻¹, corresponding to an active state
  with a time constant of τ = α⁻¹ = 10 ms (1 ms for the action potential plus 9 ms to approximate a refractory period
  where neurons are hyperpolarized). This choice of α constrains neuronal firing rates to be no greater than 100 Hz."
  - In our discrete model: ν = 90.6·ρ Hz, ν_max = 45 Hz [calc].
  - It covers every spontaneous rate in section 3.3. It cannot represent odor-evoked PN rates above 150 Hz; that would
    need τ_u ≤ 0.453/150 Hz ≈ 3 ms (my arithmetic).
- **Mapping B (reject for rates), τ_u = 50 ms.** This is the unitary-EPSP half-width (Nagel 2015 Methods: "a unitary
  EPSP decayed with a half-width of about 50 ms, in agreement with published data").
  - It gives ν_max = 9.1 Hz [calc], so it cannot hold the 10 Hz VM7 ORN rate or 13-32 Hz palp ORNs (ρ > 0.5).
  - Use it only as an upper bound on τ_u.
- **Drive equivalents.** An isolated unit with drive h has ρ = p_on/(p_on+p_off) [calc].
  - h = 1e-5 (our driven plane) ↔ ν = 0.001 Hz in mapping A.
  - h = 1e-3 ↔ 0.10 Hz.
  - A real ORN at 3.44 Hz ↔ h ≈ 0.036 (mapping A).
  - **The h used for the driven-plane fluctuation peak is about 3,400× weaker per neuron than real ORN spontaneous
    firing.** One caveat: the driven plane put h on every neuron, while real drive enters through ORNs only.
- **[calc] ORN drive onto a PN, order of magnitude only.**
  - Inputs: 2 × 17.4 ORNs (bilateral) × ~23 synapses per connection × c = 1/742.7 × 3.44 Hz.
  - Result: PN rates of 0.66 Hz at g = 0.163 and 1.2 Hz at g = 0.30, nearly the same under both mappings.
  - That is the right order for observed PN spontaneous rates (1-5 Hz). It ignores PN recurrence, LN inhibition,
    depression and the maleCNS's actual synapse counts.

### 5.2 Table
| quantity | literature value | model, mapping A (τ_u = 10 ms) | mapping B (50 ms) | source |
|---|---|---|---|---|
| ORN spontaneous rate | 3.44 ± 0.16 Hz (DM4); ~4 (DM4), ~7 (VM2), ~10 (VM7); palp 3-13, up to 32 ± 7 | ρ 0.033-0.14; isolated-unit h 0.031-0.155 (3.44 Hz: ρ 0.038, h 0.036) | ρ 0.17-0.19 at 3-3.44 Hz; > 0.5 for ≥ 10 Hz (not representable) | K&W 2009; K&W 2008; Olsen & Wilson 2008; de Bruyne 1999 |
| PN spontaneous rate | typically 1-5 Hz; 74.9 ± 8.6 Hz spontaneous EPSCs (one antenna) | ρ 0.011-0.055 | ρ 0.055-0.28 | K&W 2009 |
| PN spontaneous rate, input removed | none ("no spontaneous activity"; spike-driven sEPSPs absent) | ρ → 0 | ρ → 0 | Olsen 2007; Kazama 2011 |
| LN (mostly GABAergic) spontaneous rate | 2.3 ± 0.2 Hz; 4.6 ± 2.8 Hz | ρ 0.025-0.051 | ρ 0.13-0.25 | Wilson & Laurent 2005; Nagel 2015/2016 |
| eLN spontaneous rate | not quantified (spontaneously spiking; barraged by IPSPs) | — | — | Chou 2010; Yaksi & Wilson 2010 |
| ORN→PN depression, per-spike retention f → ald_sim U | f 0.72-0.78 (fast 0.77, slow 0.91) | U 0.25-0.34 (slow 0.089) [calc, geometric epoch] | same U | K&W 2009; Nagel 2015 |
| ORN→PN recovery τ_rec | 0.893-2.4 s (components 0.63, 1.0 s) | TAU 89-240 (63, 101) | TAU 18-48 (13, 20) | K&W 2009; Nagel 2015 |
| ORN→LN depression | f 0.75, τ 1.566 s | U 0.29, TAU 157 | U 0.29, TAU 31 | Nagel & Wilson 2016 |
| ORN→PN release probability (per site) | p 0.79, N 51, q 1.05 pA; uEPSP 6.19 mV | (not U; use f) | | K&W 2008 |
| iLN→LN | slow, facilitating; no parameters | test U ≈ 0 (no STD) | | Nagel & Wilson 2016 |
| LN→PN inhibition (presynaptic on ORN terminals) | rise ~ alpha function τ ≈ 25 ms; GABA_B epochs 0.1-several s | 2.5 units; GABA_B 10-several hundred units | 0.5 units; 2-100 units | Nagel 2015; Wilson & Laurent 2005 |
| eLN→PN | electrical; Cd²⁺- and mecamylamine-insensitive; small somatic coupling coefficients | treat as non-chemical (sign 0 or weak) | | Yaksi & Wilson 2010 |
| eLN→iLN | largely cholinergic (Yaksi) vs rare (Huang) | keep as excitatory | | Yaksi & Wilson 2010; Huang 2010 |
| eLN↔eLN, lLN1_bc→lLN1_bc | reciprocal connections exist; mechanism and strength unknown | **unconstrained**: scan | | Huang 2010 (abstract) |
| PN spike-frequency adaptation | none over 500 ms at > 100 Hz | D = 0 | | K&W 2008 |
| LN adaptation | mild; class-dependent | small D, TAUA ~ 10-100 units (guess, not measured) | | Nagel & Wilson 2016; Seki 2010 |
| absolute refractory period | short (PNs > 150 Hz; ORNs ~225 Hz) | already inside the ~11 ms active epoch; extra --refractory not needed | | K&W 2009; de Bruyne 1999 |

**[calc] Mean-field resource x* = 1/(1+UρTAU), mapping A, ORN→PN parameters:**
- At ρ = 0.04 (≈ 3.6 Hz, rest): 0.23-0.53 (slow component 0.82).
- At ρ = 0.1: 0.11-0.31 (slow 0.64).
- At ρ = 0.3: 0.04-0.13 (slow 0.37).

---------------------------------------------------------------------------------------------------------------------

## 6. Synthesis: defect or not?

Plain answer: the literature favours "the self-sustained AL at h = 0 is a defect of the model", meaning a regime the
real AL does not show. The model's exact mechanism is still untested, because the decisive synapse
(lLN1_bc→lLN1_bc) has never been measured.

1. **The real AL is input-driven at rest and goes quiet without input** (strong; replicated, full texts).
   - Spontaneous PN spiking comes from ORNs. PN EPSC rate equals ORN rate × number of ORNs.
   - It disappears within 10-20 min of removing ORN input, even with palps and lateral input intact.
   - Experimenters remove antennae precisely to quiet the network.
   - "In the normal circuit, excitatory connections between glomeruli are weak."
   - Counter-evidence is weak: isolated-brain PN spiking (data not shown); rhythmic large-LN activity (abstract,
     preparation unknown); pan-neuronal calcium fluctuations without antennae. None shows a self-sustaining recurrent
     AL state.
2. **Cause (a), the cholinergic label.**
   - Cholinergic broad LNs exist (8-15 per AL by immunostaining), and EM predictions place lLN1_bc among them. So "ACh"
     for lLN1_bc is plausible but unproven; Eckstein flags lLN2T_a/b/c as the closer soma match.
   - The bigger problem is not the label but what an ACh label means in the model. Real eLN output onto PNs is
     electrical and not blocked by nicotinic antagonists. Real eLN cholinergic output goes largely onto inhibitory LNs.
     eLNs sit under constant GABAergic IPSP barrage.
   - The model instead treats every lLN1_bc synapse (to PNs, to LNs, to each other) as strong, non-depressing chemical
     excitation scaled by synapse count. Also, the maleCNS cholinergic LN fraction (33 %) is above all other estimates
     (immunostaining ~4-8 %; FlyWire EM 18-27 %).
3. **Cause (b), linear count weights.** Supported within ORN→PN (Tobin: summation efficacy ≈ 0.9). There is no data for
   LN-LN synapses. Lateral excitation saturates at the network level.
4. **Cause (c), missing STD and adaptation.** This is very likely to matter for the size and lifetime of the active
   state.
   - Every measured excitatory AL synapse depresses strongly and recovers slowly (0.6-2.4 s, i.e. 60-240 units in
     mapping A).
   - In a sustained state at ρ ≥ 0.1, mean-field resources fall to ≤ 0.31 for 4 of 5 published parameter sets [calc].
     That cut is larger than the AL's 2.97× lead over the next mode.
   - Caveats. STD does not change g_lin: at silence x = 1. It acts on the active state, where it should shorten
     lifetimes or turn sustained activity into bursts, raising the sustained onset. That is a prediction for the
     simulation lane, not a result. LN→LN depression is unmeasured. PNs do not adapt, so adaptation is a weaker lever
     than STD.
5. **Cause (d), electrical eLN→PN and eLN→iLN recruitment.** Supported (full text). The real eLN pathway brings its own
   inhibition, which grows with stimulus strength. Gap junctions transmit hyperpolarization as well as depolarization,
   so they cannot act as a non-depressing excitatory chemical drive.
6. **Cause (e), h = 0 everywhere.** This is the clearest mismatch.
   - Real ORNs fire at 3-13 Hz (up to ~30 Hz), about h ≈ 0.03-0.15 per ORN in mapping A. That is orders of magnitude
     above the 1e-5 used for the driven-plane peak.
   - The crude ORN→PN estimate gives PN rates of 0.6-1.2 Hz at g 0.16-0.30, the right order for the observed 1-5 Hz.
   - The literature therefore fits "a strongly driven, sub-threshold AL with strong feedforward amplification":
     - feedforward: "weak ORN odor responses are amplified in PNs" through strong, high-p synapses (K&W 2008 abstract);
     - lateral: lateral excitation "promotes sensitivity" for weak stimuli (Yaksi & Wilson 2010);
     - homeostatic: chronic deafferentation *raises* eLN coupling (Kazama 2011).
   - Whether it sits "just below ignition" in a critical sense has no direct test in the literature. No avalanche or
     criticality data exist for the fly AL (al_lit `REPORT.md`).
7. **What to change in the model (for the simulation lanes).**
   - Give ORNs h ≈ 0.03-0.15 (mapping A) and leave the rest at 0.
   - Add STD with U ≈ 0.25-0.34 and TAU ≈ 90-240 on excitatory outputs (ORN and, as a guess, LN/PN). No STD on
     inhibitory LN outputs; they facilitate.
   - Make lLN1_bc→PN non-chemical (sign 0) as a gap-junction proxy, and keep lLN1_bc→iLN excitatory.
   - Then ask whether the AL still self-sustains when ORNs are clamped off. The literature predicts it should not, and
     predicts that PN rates under ORN drive should land at 1-5 Hz.

## 7. Verification table
| citation | access | status | key sentence used |
|---|---|---|---|
| Schlegel et al. 2021 eLife 10:e66018 | full text | verified | "Strong ALLN-ALLN connectivity arises mostly from the broad ALLNs of the lateral lineage" |
| Eckstein et al. 2024 Cell 187:2574 | full text + ground-truth CSV | verified | "18%–27% of antennal lobe local neurons may be cholinergic"; lLN1_bc / lLN2T_abc sentences |
| Shang et al. 2007 Cell 128:601 | full text | verified | "63 % in ... krasavietz-GAL4"; "eight to 11 ... cholinergic" |
| Chou et al. 2010 Nat Neurosci 13:439 | full text | verified | "vast majority of LNs are GABAergic; however, there are a few cholinergic cells" |
| Seki et al. 2010 J Neurophysiol | abstract | verified (abstract) | krasavietz LNs "were mostly GABAergic" |
| Huang et al. 2010 Neuron 67:1021 | abstract | verified (abstract) | eLN↔PN "cholinergic synapses and gap junctions"; eLN↔eLN reciprocal; eLN-iLN rare |
| Yaksi & Wilson 2010 Neuron 67:1034 | full text | verified | "eLN-to-PN synapses are mainly or purely electrical" |
| Kazama, Yaksi & Wilson 2011 J Neurosci 31:7619 | full text | verified | "In the normal circuit, excitatory connections between glomeruli are weak." |
| Das et al. 2011 Neural Syst Circuits 1:4 | full text | verified | "approximately 65 labelled cells" (glutamatergic) |
| Nagel & Wilson 2016 J Neurosci 36:4325 | full text | verified | "0.75 and 1566 ms for LNs; 0.78 and 893 ms for PNs" |
| Nagel, Hong & Wilson 2015 Nat Neurosci 18:56 | full text | verified | "f = 0.78 and τ = 893 ms"; "4.6 ± 2.8 spikes/s" |
| Kazama & Wilson 2008 Neuron 58:401 | full text | verified | "release probability ... 0.79"; "104.2%" |
| Kazama & Wilson 2009 Nat Neurosci 12:1136 | full text | verified | "α = 0.72 and τ = 2.4 s"; "typically 1–5 spikes/sec"; 74.9/3.44/17.4 |
| Olsen, Bhandawat & Wilson 2007 Neuron 54:89 | full text | verified | "no spontaneous activity"; "~10–20 minutes"; saturation 50→150 |
| Olsen & Wilson 2008 Nature 452:956 | full text | verified | "interglomerular inhibition dominates"; VM7 "~10 spikes/s" |
| Wilson & Laurent 2005 J Neurosci 25:9069 | full text | verified | LN 2.3 ± 0.2 spikes/s; GABA_B "longer time scales" |
| Gouwens & Wilson 2009 J Neurosci 29:6239 | full text | verified | 598 MΩ; ORN input "drives spontaneous action potentials in PNs" |
| Tobin, Wilson & Lee 2017 eLife 6:e24838 | full text | verified | "about 23 synapses"; "summation efficacies near 0.9" |
| de Bruyne et al. 1999 J Neurosci 19:4520 | full text | verified | "between 3 and 13 spikes/sec"; "32 ± 7"; "∼225 spikes/sec" |
| Gu & O'Dowd 2006 J Neurosci 26:265 | full text | verified (claim is data not shown) | PNs in isolated brain "fire spontaneous action potentials" |
| Yang et al. 2011 J Neurophysiol 106:2127 | abstract | verified (abstract); preparation unknown | large lateral LNs with "rhythmic spontaneous activities" |
| Mann, Gallen & Clandinin 2017 Curr Biol 27:2389 | full text | verified | antennae and proboscis removed; "even in the absence of olfactory input" |
| Galán et al. 2006 Neural Comput 18:10 | author PDF | verified | "rather distinct from persistent activity" |
| Benayoun et al. 2010 PLoS Comput Biol 6:e1000846 | full text + formula images | verified | α = 0.1 ms⁻¹, τ = 10 ms (1 + 9 ms), ≤ 100 Hz |
| Task et al. 2022 eLife 11:e72599 | abstract | verified; not relevant | ORN co-receptor map; no LN data |

## 8. Open
- Where the maleCNS `ground_truth = ACh` for lLN1_bc comes from (not Eckstein's ground truth; source not found).
- Direct physiology of lLN1_bc vs lLN2T_a/b/c: which of them are the krasavietz eLNs (soma position: eLNs
  ventrolateral per Yaksi & Wilson).
- Any measurement of eLN↔eLN synapse strength or dynamics. Huang 2010 full text (blocked) may have the eLN-eLN
  mechanism and spontaneous rates.
- Chou 2010 Supplementary Table 1 counts, which I did not open.
