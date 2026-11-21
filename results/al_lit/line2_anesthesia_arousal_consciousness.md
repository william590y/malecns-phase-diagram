# Line 2: anesthesia, arousal, consciousness proxies and seizures in flies and insects, and where the antennal lobe fits

Literature agent for line 2, 2026-11-09. The question: do our MaleCNS Wilson-Cowan findings line up with empirical work on
anesthesia, arousal, consciousness or seizures in the fly? The model's findings are that the antennal-lobe (AL) module
holds the leading eigenvector, is the first and dominant region to sustain activity, and alone reproduces the onset.

How each source was read:
- **FULL**: full text read, either downloaded from PMC (NCBI E-utilities `efetch db=pmc`, or the PMC HTML page) and searched for
  the passages quoted below, or read through the fetch tool on the publisher or PMC page.
- **PART**: full-text page read through the fetch tool, which summarised it. Methods and results were covered; I did not read every line.
- **ABS**: abstract only, from the PubMed record retrieved with NCBI E-utilities (`efetch db=pubmed`). The URL given is the PubMed page for that record.

Evidence classes: **(a)** direct evidence about the AL; **(b)** olfactory processing in general (insect AL, mammalian olfactory
bulb); **(c)** whole-brain or other-region evidence; **(T)** theory or model.

---------------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **No study I could find has recorded the AL under general anesthesia, in flies or in any insect.** This is a negative
   result of the search, not a negative experimental result. The fly anesthesia electrophysiology (Cohen 2016, 2018;
   Leung 2021, 2025; Muñoz 2020) used a linear probe pushed laterally through the eye, covering the optic lobe and then
   the central brain. None of these papers names the AL as a recorded structure. The one whole-brain calcium-imaging study
   under isoflurane (Troup 2023) imaged through a window at the back of the head and does not mention the AL. PubMed query
   `(anesthetic OR anesthesia OR isoflurane OR halothane OR propofol OR sevoflurane) AND "antennal lobe"` (16 hits) and
   `(isoflurane OR halothane OR anesthetic) AND Drosophila AND (olfactory OR odor OR olfaction)` (23 hits) gave nothing
   that records the AL under anesthesia. Several web searches came back the same way.
2. **Where fly anesthesia does act, the central brain is hit harder than the sensory periphery (optic lobe).** Cohen
   2016 and 2018 and Leung 2021 report this, and Leung 2025 finds the only marker that generalised at the deepest (central)
   channel. The fly brain is not silenced: about 18% of imaged neurons are active awake versus about 15% under isoflurane
   (Troup 2023), and correlations fragment. Isoflurane impairs excitatory cholinergic release but spares GABA and
   glutamate synapses (Hines 2024).
3. **Spontaneous activity data that include the AL do single it out.** In Paulk 2013 (more than 2,000 registered LFP sites), the
   AL had the most spontaneous spiking and more power above 15 Hz than the central brain. Only GABAergic activation (and one
   mixed line) raised 50-100 Hz power in the AL. In Aimon 2019 (light-field near-whole brain), AL projection-neuron (PN)
   components were spontaneously active without odour, and the AL and lateral horn showed spontaneous oscillations. This is
   the closest empirical match to "the AL is the first and dominant region to sustain activity". **Caveat:** Kazama & Wilson
   2009 show that spontaneous PN input is a barrage of EPSCs driven by spontaneously firing ORNs, with the EPSC rate equal to
   the ORN rate times the number of ORNs. The biological AL activity is therefore largely fed forward from the antenna. Our
   model's AL holds activity by recurrence with h = 0.
4. **Seizures (the supercritical side) do not point to the AL.** Fly seizure-locus work implicates the mushroom body (MB)
   alpha/beta lobes, the ellipsoid body (EB) and the giant-fibre pathway. [VERIFICATION: only the MB alpha/beta lobes are an
   *initiation* site (Saras 2017). EB and giant fibre come from rescue (Hekmat-Scafe 2010) and suppression (Kroll 2015); in Saras 2017,
   optogenetic activation through c232 (EB) and a307 (giant fibre) did not trigger seizures.] Restoring kcc in AL PNs (GH146) gave only modest
   rescue (63% bang-sensitive). Silencing PN output with tetanus toxin had no effect (97%) (Hekmat-Scafe 2010). The only AL
   link is expression: a seizure gene (jus) overlaps the rescuing GAL4 line in "a small number of neurons in the antennal
   lobe" (Dean 2018).
5. **Consciousness accounts do not place the AL at the centre.** Barron & Klein 2016 put the substrate of insect subjective
   experience in the central complex (CX). The AL appears only as primary odour processing. Leung 2021 found slightly *higher*
   integrated information in peripheral (optic-lobe) channels, which the authors tie to IIT's emphasis on sensory areas.
   That periphery is visual, not olfactory.
6. **Theory.** A Kuramoto study on the hemibrain connectome (Ódor, Deco, Kelling 2022) found a mean-field-like transition
   and "no clear" Griffiths phase. This agrees with our "one metastable region, not a Griffiths phase". However, they
   normalised each node's incoming weights. My inference is that this probably removes the synapse-count heterogeneity
   that localises our leading eigenvector on the AL. [VERIFICATION: they also simulated the raw, unnormalised network and found
   its transition "not smoother than the homeostatic one"; see D1.]

---------------------------------------------------------------------------------------------------------------------------

## A. Direct AL evidence (a)

### A1. Paulk, Zhou, Stratton, Liu, van Swinderen (2013). Multichannel brain recordings in behaving Drosophila reveal oscillatory activity and local coherence in response to sensory stimulation and circuit activation. J Neurophysiol 110(7):1703-1721. doi:10.1152/jn.00414.2013. PMC4868374. [FULL]
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4868374/ (HTML downloaded and searched)
- More than 2,000 registered LFP sites in more than 200 flies. Sites were mapped to three classes: "central brain (a higher-order
  region), antennal lobes (the primary olfactory sensory regions), and optic lobes". **The AL was recorded.**
- Spontaneous activity: "Our brain registration method revealed decreased spike intervals (or increased spike rates) in the
  central brain, with the most spiking in the antennal lobes (Fig. 2G)". The Fig. 2G legend says spike intervals "were shorter in the
  protocerebrum and antennal lobes compared with other brain regions".
- Spontaneous spectra: "There was significantly increased power in the higher frequencies >15 Hz in the antennal lobes compared
  with the central brain, whereas central brain channels displayed increased low-frequency oscillations" (N = 32 wild-type flies).
- Odour: "Olfactory stimuli increased coherence from 15 to 100 Hz specifically in the antennal lobes ... but not elsewhere".
  Odours also raised 30-100 Hz power in the central brain.
- Circuit activation (TrpA1): GABA (Gad1) and 7B-Gal4 (MB plus AL) were predicted to act in the AL, and "Remarkably, only these two
  lines increased high-frequency (50–100 Hz) activity in the antennal lobes". Elsewhere: "only GABAergic activation altered
  LFP activity in the antennal lobes, whereas other Gal4s only had effects in the optic lobes".
- **No anesthesia or sleep condition in this paper.**
- Relevance: this is the strongest empirical support for "the AL is the most spontaneously active region". It also
  shows that AL fast dynamics are under GABAergic (LN) control, consistent with our result that inhibition shifts the onset.

### A2. Aimon, Katsuki, Jia, Grosenick, Broxton, Deisseroth, Sejnowski, Greenspan (2019). Fast near-whole-brain imaging in adult Drosophila during responses to stimuli and behavior. PLoS Biol 17(2):e2006732. doi:10.1371/journal.pbio.2006732. PMC6395010. [FULL]
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6395010/ (JATS XML via efetch, searched)
- Light-field imaging, pan-neuronal GCaMP6 and ArcLight. **The AL is in the field of view.**
- "Components likely representing the activity of antennal projection neurons were spontaneously active in the absence of odor,
  but some of them became more active when the odor was presented".
- "Other patterns of spontaneous activity included oscillations in the antennal lobe and lateral horn and fast, ongoing activity in
  the ellipsoid body and the protocerebral bridge". The authors add: "More work is necessary to establish the conditions and
  consistency of these patterns".
- Behaviour: "brain activity increased on a global scale when the fly walked but not when it groomed".
- Relevance: an independent method (imaging rather than LFP) that also shows spontaneous AL activity. No anesthesia or sleep condition.

### A3. Mann, Gallen, Clandinin (2017). Whole-Brain Calcium Imaging Reveals an Intrinsic Functional Network in Drosophila. Curr Biol 27(15):2389-2396. doi:10.1016/j.cub.2017.06.076. PMC5967399. [FULL]
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC5967399/ (efetch, searched)
- Resting-state correlations between atlas regions of the central brain. The AL is an atlas region (AL_R appears in the figure).
- "regions involved in olfactory processing (antennal lobe (AL), lateral horn (LH), MB, SMP, superior lateral protocerebrum (SLP),
  crepine (CRE)) showed a high degree of interconnectivity".
- Relevance: the AL sits inside a strongly co-fluctuating olfactory cluster at rest. This fits our result that the AL and its
  targets form the module that lights up first. No state manipulation.

### A4. Kazama & Wilson (2009). Origins of correlated activity in an olfactory circuit. Nat Neurosci 12(9):1136-1144. doi:10.1038/nn.2376. PMC2751859. [FULL]
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC2751859/ (efetch, searched)
- Whole-cell PN recordings "reveal an ongoing barrage of large, fast, spontaneous excitatory postsynaptic currents" and "ORNs are
  the only source of large, fast EPSCs in PNs".
- The spontaneous EPSC rate in DM4 PNs (74.9 ± 8.6 Hz, one antenna removed) "is not substantially different from the mean firing
  rate of DM4 ORNs (3.44 ± 0.16 Hz ...) multiplied by the mean number of DM4 ORNs (17.4 ± 0.9 per antenna)".
- "Reciprocal PN-PN connections make a smaller contribution to correlations" (abstract).
- Relevance, as a **caveat**: real spontaneous AL activity is mostly driven by spontaneous ORN firing, which is peripheral
  noise, not self-sustained recurrent excitation. Our model has the AL sustain itself with h = 0. A fair comparison needs
  an ORN-noise drive (our weak-drive runs come closer). The prediction should be phrased as "the AL recurrent module
  amplifies ORN noise most", not "the AL ignites spontaneously".

### A5. Tanaka, Ito, Stopfer (2009). Odor-evoked neural oscillations in Drosophila are mediated by widely branching interneurons. J Neurosci 29(~~27~~ 26):8595-8603 [VERIFICATION: issue is 26, per PubMed 19571150]. doi:10.1523/JNEUROSCI.1455-09.2009. PMC2753235. [ABS]
URL: https://pubmed.ncbi.nlm.nih.gov/19571150/
- Odours evoke LFP oscillations in the MB that are "reversibly abolished by application of the GABA(a) blocker picrotoxin". AL LNs
  and PNs show "subthreshold membrane potential oscillations that were tightly phase locked to LFP oscillations recorded downstream
  in the MBs". "only the more widely branching population of LNs is necessary for generating odor-elicited oscillations".
- Relevance: a dynamical, synchrony-type transition located in the AL and generated by inhibitory LNs. It is odour-evoked,
  not a state transition.

### A6. Hekmat-Scafe, Mercado, Fajilan, Lee, Hsu, Mount, Tanouye (2010). Seizure sensitivity is ameliorated by targeted expression of K+-Cl- cotransporter function in the mushroom body of the Drosophila brain. Genetics 184(1):171-183. doi:10.1534/genetics.109.109074. PMC2815914. [PART]
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC2815914/
- Bang-sensitivity of kcc^DHS1 relative to sibling controls (lower means more rescue): OK107 (MB) 30%, c232 (EB) 17%,
  104Y (fan-shaped body) 76%, **GH146 (AL projection neurons) 63%**, the "modest" group. Also MB247 32% and c772 23%.
- **GH146 > tetanus toxin "produces no significant difference in the bang sensitivity of kccDHS1 flies (97% BS ...)"**. The
  authors used this to rule out the MB effect being an upstream olfactory-input effect.
- Abstract: "phenotypic rescue is largely accounted for by targeted, circumscribed expression in the mushroom bodies (MBs) and the
  ellipsoid body (EB)", acting "via a weakening of chemical synaptic inhibition by GABAergic transmission".
- Relevance: **direct AL evidence, against AL involvement in seizure susceptibility**. AL PNs are at most minor contributors.

### A7. Dean, Weinstein, Amin, Karno, McAvoy, Hoy, Recknagel, Jarvis, Deitcher (2018). Extending julius seizure, a bang-sensitive gene, as a model for studying epileptogenesis: Cold shock, and a new insertional mutation. Fly 12(1):55-61. doi:10.1080/19336934.2017.1402993. PMC5927695. [PART]
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC5927695
- MiMIC-tagged Jus: "limited expression in the central brain surrounding the antennal lobes". It overlaps the rescuing GAL4 in "a small number
  of neurons in the antennal lobe", and the Fig. 2 legend says "Most of the overlapping neurons are in the brain surrounding the antennal lobes".
- Relevance: weak. This is expression in or near the AL for a seizure gene, with no evidence that seizures start there.
  The companion paper, Horne et al. 2017 Genetics (doi:10.1534/genetics.116.199083, [ABS], https://pubmed.ncbi.nlm.nih.gov/28082408/),
  says the fosmid fusion "was expressed in the optic lobes and the subesophageal and thoracic abdominal ganglia". There,
  knockdown in GABAergic or cholinergic neurons gave partial bang-sensitivity, and knockdown in glutamatergic neurons gave none.

### A8. Negative search results for the AL (what was NOT found)
- No recording of the AL (LFP, imaging or single unit) under isoflurane, propofol, halothane, sevoflurane or CO2 in any
  insect. Queries are listed in section 0.
- No study naming the AL as a seizure-initiation site. One web search for `Drosophila seizure initiation brain region "antennal lobe"` returned
  only MB (Saras 2017; Hekmat-Scafe 2010) and expression-level jus hits.
- No AL-specific analysis in the insect-consciousness debate papers I read (Barron & Klein 2016; Adamo 2016; Key et al. 2016).

---------------------------------------------------------------------------------------------------------------------------

## B. Olfactory processing in general (b)

### B1. van Swinderen & Greenspan (2003). Salience modulates 20-30 Hz brain activity in Drosophila. Nat Neurosci 6(6):579-586. doi:10.1038/nn1054. [ABS]
URL: https://pubmed.ncbi.nlm.nih.gov/12717438/
- "Coupling of salience effects (odor, heat or novelty) to these stimuli modulated LFPs in the 20-30 Hz range by evoking a transient,
  selective increase". Odour acted as a salience or arousal cue that raised a central-brain perceptual LFP correlate. Recording site is not in the abstract.

### B2. Rein, Mustard, Strauch, Smith, Galizia (2013). Octopamine modulates activity of neural networks in the honey bee antennal lobe. J Comp Physiol A 199(11):947-962. doi:10.1007/s00359-013-0805-y. PMC3825135. [ABS]
URL: https://pubmed.ncbi.nlm.nih.gov/23681219/
- Honeybee AL (insect AL, but not fly): "octopamine increases background activity in olfactory output neurons, while reducing average calcium
  levels". The proposed mechanism is octopamine receptors "on a subpopulation of inhibitory neurons in a disinhibitory pathway".
- Relevance: an arousal neuromodulator raises spontaneous AL output, possibly through disinhibition. In our model this would be
  a move toward lower r or higher effective g, so toward or past the AL onset.

### B3. Murakami, Kashiwadani, Kirino, Mori (2005). State-dependent sensory gating in olfactory cortex. Neuron 46(2):285-296. doi:10.1016/j.neuron.2005.02.025. [ABS]
URL: https://pubmed.ncbi.nlm.nih.gov/15848806/
- Rats under urethane alternate between slow-wave and fast-wave states. Olfactory cortex neurons respond robustly in the fast-wave state and weakly in the slow-wave state,
  and this "was observed in a majority of olfactory cortex neurons, but in only a small percentage of olfactory bulb neurons".
- Relevance: in mammals the AL analogue (olfactory bulb) keeps responding across anesthetic or sleep-like states while cortex
  is gated. This matches the qualitative picture that the primary olfactory module is the last to drop out.

### B4. Rinberg, Koulakov, Gelperin (2006). Sparse odor coding in awake behaving mice. J Neurosci 26(34):8857-8865. doi:10.1523/JNEUROSCI.0884-06.2006. [ABS]
URL: https://pubmed.ncbi.nlm.nih.gov/16928875/
- "odor-elicited changes in mitral cell firing rate were larger and more frequently observed in the anesthetized than in the awake
  condition. Only 27% of mitral cells that showed a response to odors in the anesthetized state were also odor responsive in the
  awake state."

### B5. Kato, Chu, Isaacson, Komiyama (2012). Dynamic sensory representations in the olfactory bulb: modulation by wakefulness and experience. Neuron 76(5):962-975. doi:10.1016/j.neuron.2012.09.037. [ABS]
URL: https://pubmed.ncbi.nlm.nih.gov/23217744/
- "wakefulness greatly enhances the activity of inhibitory granule cells and makes principal mitral cell odor responses more sparse
  and temporally dynamic". Also, "the expression of this experience-dependent plasticity is prevented by anesthesia".
- Relevance, as a **tension**: in the mouse olfactory bulb, *wakefulness = more inhibition*, and anesthesia gives broader principal-cell
  activity. In our model, more inhibition raises the onset and lowers activity. Mapping "awake" onto "near onset" therefore needs excitation
  and inhibition to rise together in the awake state, not g alone.

### B6. Merrick, Godwin, Geisler, Morsella (2014). The olfactory system as the gateway to the neural correlates of consciousness. Front Psychol 4:1011. doi:10.3389/fpsyg.2013.01011. [ABS] (theory/review)
URL: https://pubmed.ncbi.nlm.nih.gov/24454300/
- Proposes that the olfactory system is "a unique portal" for contrasting conscious and unconscious processing. It hypothesises that for olfactory content to be
  conscious it "must participate in a network that is large-scale". This is mammalian and theoretical. In this view the olfactory module
  alone is not sufficient for consciousness.

---------------------------------------------------------------------------------------------------------------------------

## C. Whole-brain or other-region evidence on anesthesia, arousal and seizures (c)

### Anesthesia in the fly: what was recorded, and where

**C1. van Swinderen (2006).** A succession of anesthetic endpoints in the Drosophila brain. J Neurobiol 66(11):1195-1211. doi:10.1002/neu.20300. [ABS]
URL: https://pubmed.ncbi.nlm.nih.gov/16858694/
- "At low doses, isoflurane uncoupled brain activity from ongoing movement, followed by a sudden attenuation in neural correlates of
  perception. Average LFP activity in the brain was more gradually attenuated with higher doses, followed by loss of movement".
  "anesthetic sensitivities in the fly brain are, like other arousal states, influenced by dopaminergic activity".
- The abstract does not give the recording site. The lab's earlier LFPs were "in the medial brain between the mushroom bodies" (Nitz 2002, C10).
- Relevance: the perceptual correlate drops *suddenly* while average LFP declines *gradually*. That pattern is compatible with a
  threshold-like loss of a specific dynamical mode (our onset), with background activity left over.

**C2. Cohen, Zalucki, van Swinderen, Tsuchiya (2016).** Local Versus Global Effects of Isoflurane Anesthesia on Visual Processing in the Fly Brain. eNeuro 3(4):ENEURO.0116-16.2016. doi:10.1523/ENEURO.0116-16.2016. PMC4967815. [FULL]
URL: https://www.eneuro.org/content/3/4/ENEURO.0116-16.2016 (read with the fetch tool); full text also downloaded from PMC via efetch
- 16-site linear probe "inserted laterally to the eye". Channels 1-6 are periphery (optic lobe) and 9-14 are central brain. **The AL is not mentioned.**
- Isoflurane "reduced power and coherence at the tagging frequency (13 or 17 Hz) in central brain regions", and "increased power and
  coherence at twice the tag frequency (26 or 34 Hz) in the optic lobes". The 0.6% dose abolished air-puff responsiveness.

**C3. Cohen, van Swinderen, Tsuchiya (2018).** Isoflurane Impairs Low-Frequency Feedback but Leaves High-Frequency Feedforward Connectivity Intact in the Fly Brain. eNeuro 5(1):ENEURO.0329-17.2018. doi:10.1523/ENEURO.0329-17.2018. PMC5846228. [FULL]
URL: https://www.eneuro.org/content/5/1/ENEURO.0329-17.2018 (fetch tool); also efetch PMC5846228
- Same data, spontaneous activity. "This probe covers approximately half of the fly brain". The periphery is the optic lobe, the centre is the central brain, and the **AL is not recorded or named**.
- "For the center, power was significantly reduced across almost all frequencies tested". The periphery showed a smaller reduction.
  Feedback (centre to periphery) dominates at 0.1-5 Hz and isoflurane "preferentially reduced FB". Centre-periphery coherence was reduced at about 10-25 Hz.

**C4. Leung, Cohen, van Swinderen, Tsuchiya (2021).** Integrated information structure collapses with anesthetic loss of conscious arousal in Drosophila melanogaster. PLoS Comput Biol 17(2):e1008722. doi:10.1371/journal.pcbi.1008722. PMC7946294. [FULL]
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7946294 (fetch tool plus efetch, searched)
- Same probe ("inserted laterally into the fly's eye", 15 bipolar channels). **The AL is not named.**
- "integrated interactions among populations of neurons during wakefulness collapsed to isolated clusters of interactions during
  anesthesia", "with their collapse occurring all throughout the fly brain".
- "Opposite to our expectation, however, we found a trend indicating that peripheral channel sets tended to have slightly but
  significantly greater system-level integrated information ... centrally located channel sets seemed to be more affected by anesthesia".
  The authors also say "The finding of greater system-level integrated information in the peripheral, sensory processing, areas is also consistent with an
  indirect prediction of IIT that sensory areas are more important for consciousness".
- Relevance: the nearest empirical analogue of "a sensory module carries the integrated activity". It concerns the visual periphery, not the AL.

**C5. Muñoz, Leung, Zecevik, Pollock, Cohen, van Swinderen, Tsuchiya, Modi (2020).** General anesthesia reduces complexity and temporal asymmetry of the informational structures derived from neural recordings in Drosophila. Phys Rev Research 2:023219. doi:10.1103/PhysRevResearch.2.023219. arXiv:1905.13173. [ABS]
URL: https://arxiv.org/abs/1905.13173v3
- Statistical complexity drops under isoflurane "even in single channel" data, and "anesthetized brains become both less structured and
  more reversible". The abstract page does not state the recording sites.

**C6. Leung, Mahmoud, Jeans, Fulcher, van Swinderen, Tsuchiya (2025).** Wakefulness can be distinguished from general anesthesia and sleep in flies using a massive library of univariate time series analyses. PLoS Biol 23:e3003217. doi:10.1371/journal.pbio.3003217. PMC12244653. [FULL]
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12244653/ (efetch, searched)
- Registered report. Discovery set N = 13, evaluation N = 49 across isoflurane and sleep. The probe was "inserted laterally into the fly's eye",
  "Channel 1 refers to the channel positioned furthest into the fly brain". **The AL is not named.**
- "only 47 time-series features, applied to recordings obtained from the center of the fly brain, to also significantly classify
  wake from anesthesia or sleep in all 4 of these evaluation datasets. Most of these were related to autocorrelation, and they
  indicated that signals during wakefulness remained correlated to their past for a longer timescale than during anesthesia and
  sleep". Complexity and spectral features "failed to generalize".
- Relevance, our strongest dynamical point of contact: **wake has longer autocorrelation timescales**. In our model, slow dynamics
  (long lifetimes, peak fluctuations) sit at the AL flicker point near onset. Wake as the near-onset, slow regime and
  anesthesia and sleep as faster-decorrelating, subcritical regimes is the shape the user's hypothesis predicts. The caveat is that this is measured in the central brain, not the AL.

**C7. Troup, Tainton-Heap, van Swinderen (2023).** Neural Ensemble Fragmentation in the Anesthetized Drosophila Brain. J Neurosci 43(14):2537-2551. doi:10.1523/JNEUROSCI.1657-22.2023. PMC10082453. [FULL]
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10082453/ (HTML downloaded and searched)
- Two-photon, pan-neuronal nuclear GCaMP. "we imaged the fly's central brain through a window cut into the back of the head".
  "Some flies had small regions of optic lobe neurons in the periphery of the recording, and these areas were masked". **The AL is never mentioned.**
  The AL is an anterior structure, so it was probably outside the imaged volume. That is my inference, not stated by the authors.
- "the percentage of active neurons decreased moderately yet significantly, from ∼18% active during wakefulness to 15% active under
  isoflurane". Average activity of active neurons was unchanged (p = 0.847). Correlation patterns "become more fragmented and less
  diverse under anesthesia but remain wake-like during induced sleep".
- Relevance: anesthesia in the fly is not a silent absorbing state. The model's subcritical regime with h = 0 is silent, so
  anesthesia should be compared with the *weakly driven* subcritical regime, not with h = 0.

**C8. Hines, Kewin, Van De Poll, Anggono, Bademosi, van Swinderen (2024).** Synapse-Specific Trapping of SNARE Machinery Proteins in the Anesthetized Drosophila Brain. J Neurosci 44 (2024). doi:10.1523/JNEUROSCI.0588-23.2024. PMC11170680. [ABS]
URL: https://pubmed.ncbi.nlm.nih.gov/38749704/
- In MB output neurons, "isoflurane impaired synaptic release and presynaptic protein dynamics in excitatory cholinergic synapses. In
  contrast, isoflurane had little to no effect on inhibitory GABAergic or glutamatergic synapses".
- Relevance: this gives a mechanistic map to our parameters. Isoflurane lowers excitatory gain (g) and leaves inhibition intact
  (effectively a higher r). Both moves push the model below onset. ORN-to-PN and PN synapses are cholinergic, but this was measured in MBONs, not the AL.

**C9. Kottler et al. (2013) Curr Biol 23(7):594-598, doi:10.1016/j.cub.2013.02.021 [ABS] (https://pubmed.ncbi.nlm.nih.gov/23499534/).**
"We uncover a common pathway in the fly brain controlling both sleep duration and isoflurane sensitivity, centered on monoaminergic
modulation of sleep-promoting neurons of the fan-shaped body." The anesthesia-sensitivity circuit is in the central complex, not the AL.
Related papers, all [ABS]: Zalucki et al. 2015 Anesthesiology 122:1060-74 (doi:10.1097/ALN.0000000000000629; syntaxin1A mutants
that are resistant or hypersensitive; isoflurane "slightly attenuates synaptic release"). Bademosi et al. 2018 Cell Rep 22:427-440
(doi:10.1016/j.celrep.2017.12.054; propofol traps syntaxin1A in nanoclusters). Zalucki & van Swinderen 2016 Conscious Cogn 44:72-88
(doi:10.1016/j.concog.2016.06.017), a review: "more complex behaviors ... are more sensitive to volatile general anesthetics".

**C10. Arousal-state LFP (medial protocerebrum), [ABS].** Nitz, van Swinderen, Tononi, Greenspan 2002 Curr Biol 12:1934-40
(doi:10.1016/s0960-9822(02)01300-3; https://pubmed.ncbi.nlm.nih.gov/12445387/): LFPs recorded "in the medial brain between the
mushroom bodies" correlate with activity state. van Swinderen, Nitz, Greenspan 2004 Curr Biol 14:81-7 (https://pubmed.ncbi.nlm.nih.gov/14738728/):
uncoupling of brain LFP from movement defines an intermediate arousal state. Yap et al. 2017 Nat Commun 8:1815 (doi:10.1038/s41467-017-02024-y;
https://pubmed.ncbi.nlm.nih.gov/29180766/): a 7-10 Hz central-brain oscillation in transitional sleep. None of these used AL sites.

### Behaviour-state whole-brain imaging: does it include the AL?
- **Schaffer et al. 2023**, Nat Commun 14:5572, doi:10.1038/s41467-023-41261-2, PMC10495430 [FULL, searched]. Imaged "a parallelepiped-shaped volume
  spanning the dorsal third of the central brain". **The AL is not imaged** (zero mentions). "The activity of most neurons correlated (or
  anticorrelated) with running and flailing".
- **Aimon, Cheng, Gjorgjieva, Grunwald Kadow 2023**, eLife 12:e85202, doi:10.7554/eLife.85202, PMC10168698 [FULL, searched]. Light-field whole brain;
  the component table includes "AL Whole antennal lobe" and several PN components. The abstract reports "a global change in activity that tightly correlated with
  spontaneous bouts of walk". I found no AL-specific state claim in the text.
- Aimon 2019 and Mann 2017: see A2 and A3.

### Insect consciousness debate
**C11. Barron & Klein (2016).** What insects can tell us about the origins of consciousness. PNAS 113(18):4900-4908. doi:10.1073/pnas.1520084113. PMC4983823. [FULL; the fetch tool covered the first ~100k of 143k characters]
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4983823/
- The central complex is the proposed substrate: "One of the crucial functions of the CX is thus to generate a neural simulation of the state of the
  moving insect in space", which they treat as analogous to the vertebrate tectum and colliculus. The MB is cast as learning and memory.
- The AL appears only in the Fig. 2 legend: "Primary odor processing is performed in the antennal lobes (AL)". No role in experience is given. Anesthesia
  is discussed only for vertebrates. Selective attention is used as evidence.
- Replies [FULL letters, via the fetch tool]. Key, Arlinghaus, Browman 2016 PNAS 113:E3813 (doi:10.1073/pnas.1606835113; PMC4941505)
  names no insect structures beyond a midbrain analogue. Adamo 2016 PNAS 113:E3812 (doi:10.1073/pnas.1606942113; PMC4941479)
  says the definition is "too narrow" (robots would qualify) [VERIFICATION: the letter's actual words are "define consciousness very narrowly"; "too narrow" is a paraphrase, not a quote] and gives no regions, olfaction or anesthesia.
  Klein & Barron 2016, the reply, PNAS 113:E3814-5 (doi:10.1073/pnas.1607409113): **found but NOT read** (PMC returned a captcha and PNAS a 403).

### Seizure models (the supercritical side)
**C12. Saras, Wu, Brawer, Tanouye (2017).** Investigation of Seizure-Susceptibility in a Drosophila melanogaster Model of Human Epilepsy with Optogenetic Stimulation. Genetics 206(4):1739-1746. doi:10.1534/genetics.116.194779. PMC5560784. [PART]
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC5560784/
- ReaChR in para^bss1. These drivers induced paralysis or seizures: elav (100%), Cha, c739 MB (78%), 117Y MB (90%), 7B MB alpha/beta (85%), 121Y alpha/beta (82%).
  These did not: a307 (giant fibre), 104Y (fan-shaped body), c232 (EB), Gad1, c305a (alpha'/beta', 0%), 1471 (gamma, 0%).
- "We tentatively identify the mushroom body as one apparent locus of optogenetic seizure initiation". **No AL, PN or ORN driver was tested.**
  Note that 7B-Gal4 also expresses in the AL (Paulk 2013, A1), but the authors attribute its effect to the MB alpha/beta lobes.

**C13. Kroll, Wong, Siddiqui, Tanouye (2015).** Genetics 201(3):1087-1102, doi:10.1534/genetics.115.177600 [ABS] (https://pubmed.ncbi.nlm.nih.gov/26341658/).
"we identified the giant fiber neuron as critical in the seizure circuit and sufficient to suppress seizures."

**C14. Pavlidis & Tanouye (1995).** J Neurosci 15(8):5810-9, doi:10.1523/JNEUROSCI.15-08-05810.1995 [ABS] (https://pubmed.ncbi.nlm.nih.gov/7643221/).
"electrical shocks delivered to the brain cause seizure-like activity in the dorsal longitudinal flight motor neurons". The classic readout
is thoracic (giant-fibre to DLM), so it carries no information about which region of the brain initiates.

**C15. Lilienthal, Nukala, Williquett, Manak (2026).** J Neurogenet, doi:10.1080/01677063.2026.2742881 [ABS] (https://pubmed.ncbi.nlm.nih.gov/42900470/).
In prickle (pk^sple) mutants, seizures are suppressed by expression "in a) glutamatergic neurons or b) a region analogous to the human brain stem".
"removal of the head of pksple mutants also significantly reduces seizures". The seizures are brain-derived; the AL is not mentioned in the abstract.

**C16. Mituzaite, Petersen, Claridge-Chang, Baines (2021).** eNeuro 8 (2021), doi:10.1523/ENEURO.0079-21.2021 [ABS] (https://pubmed.ncbi.nlm.nih.gov/34330816/).
A methods comparison: "electroshock is the most consistent method of seizure induction". It has no regional data.

---------------------------------------------------------------------------------------------------------------------------

## D. Theory and models (T)

**D1. Ódor, Deco, Kelling (2022).** Differences in the critical dynamics underlying the human and fruit-fly connectome. Phys Rev Research 4:023057. doi:10.1103/PhysRevResearch.4.023057. arXiv:2201.11084. [FULL via ar5iv and the fetch tool]
URL: https://ar5iv.arxiv.org/html/2201.11084 (and https://arxiv.org/abs/2201.11084)
- Kuramoto model on the hemibrain v1.0.1 largest component (21,615 nodes, 3.4 M weighted directed edges). **Incoming weights were normalised per node**.
- The transition is mean-field-like (K_c = 1.70, eta = 0.70). The region is wider than in Erdős–Rényi graphs but much narrower than in the KKI-18 human connectome. Modularity is tiny
  (Q_w = 1.2e-4), and the topological dimension is d = 5.4, which "renders this model into the mean-field region".
- Griffiths phase: the authors call it "hard to decide" whether a narrow one exists. The weight tail "does not seem to be enough to introduce visible GP
  effects". They also write that "the fly brain's simpler structure does not allow the appearance of the complex sub-critical dynamical phenomena".
- No eigenvector localisation and no brain regions are discussed.
- Relevance: an independent model-level agreement with our "no Griffiths phase". The difference matters: with raw synapse
  counts (our "norm raw") we do find a localised AL module and a metastable window. In-weight normalisation would probably
  suppress the AL localisation. [VERIFICATION: Ódor et al. also ran the raw, unnormalised hemibrain (their Fig. 3; K_c = 0.00090)
  and write that "the transition of the raw case is not smoother than the homeostatic one". So raw synapse counts did not widen
  their transition. Whether normalisation removes AL eigenvector localisation is still an untested inference, but normalisation
  alone does not explain the agreement or disagreement with our onset.] This is a testable inference: rerun our onset with in-weight-normalised W.

---------------------------------------------------------------------------------------------------------------------------

## E. How our simulated findings line up

| our finding | empirical counterpart | verdict |
|---|---|---|
| AL module is the first and dominant region to sustain activity; it alone reproduces the onset | Paulk 2013: AL has the most spontaneous spiking and more >15 Hz power. Aimon 2019: spontaneous PN activity and AL oscillations. Mann 2017: tight olfactory resting cluster | **consistent** in outcome. **Mechanism differs**: Kazama & Wilson 2009 show the spontaneous AL input is ORN-driven, fed forward |
| sub-onset regime as an anesthesia or sleep analogue | Troup 2023: the brain stays active (18% to 15%) and correlations fragment. Cohen 2016/18, Leung 2021: the central brain is hit harder than the periphery. Hines 2024: excitatory synapses are impaired, inhibitory ones spared | **partly consistent**. Lower excitatory g and relatively stronger inhibition do move the model below onset. But real anesthesia is not silent; compare with the weakly driven model |
| slow window and fluctuation peak near AL flicker as the "awake" analogue | Leung 2025: wake signals keep longer autocorrelation (the only generalising marker) | **consistent in kind** (slow dynamics in wake), but measured in the central brain, not the AL |
| AL is the last region to stay lit (implied by "first to ignite") | Murakami 2005 (rat OB not gated by state); Rinberg 2006 and Kato 2012 (OB more broadly active under anesthesia) | **consistent** for the mammalian AL analogue. **Untested in insects**: no AL recording under anesthesia exists |
| inhibition raises the onset | Paulk 2013 (only GABA activation drives AL 50-100 Hz). Tanaka 2009 (LN GABA makes the oscillations). Kato 2012 (wake = more OB inhibition) | mixed. Inhibition clearly shapes AL dynamics, but in mouse OB "awake = more inhibition", so "awake" cannot just be higher g |
| supercritical side as seizures | Saras 2017 (MB alpha/beta initiates); Hekmat-Scafe 2010 (MB/EB rescue; GH146 modest, PN TNT none); Kroll 2015 (giant fibre) | **not consistent**: no fly evidence for AL-initiated seizures. Our model has no seizure-like AL runaway distinct from global spread either |
| no Griffiths phase, one metastable region | Ódor 2022: hemibrain Kuramoto is mean-field-like with no clear GP | **consistent** (theory) |
| consciousness tied to an AL-centred critical regime | Barron & Klein 2016 put it in the central complex; Leung 2021 found higher Phi in the *visual* periphery | **no support** for an AL-specific role. The literature places arousal and consciousness signatures centrally |

### Testable predictions this suggests (from existing setups)
1. Put registered multichannel LFP sites (Paulk 2013 method) or light-field imaging that covers the AL (Aimon 2019) under graded isoflurane.
   The model predicts that AL spontaneous activity and >15 Hz power are the **last** to fall, and that AL fluctuations or autocorrelation
   **peak at a light dose** before collapse.
2. Remove the antennae (cutting ORN drive) and repeat. If the AL still holds activity, recurrent self-sustainment (our model) is supported.
   If it goes silent, as Kazama & Wilson 2009 suggest for PN input, the empirical "AL dominance" is fed forward and the model's onset should
   be re-run with ORN noise as h, not h = 0.
3. In the model, an isoflurane-like perturbation should cut excitatory (cholinergic) weights only and spare GABA and Glu (per Hines 2024).
   Check whether the AL is still the last module standing.

---------------------------------------------------------------------------------------------------------------------------

## Verification (independent fact-check, 2026-11-09)

Method. Each citation's authors, year, title, journal, volume, issue, pages and DOI were checked against PubMed or PMC `esummary`
(NCBI E-utilities), or against Crossref and the arXiv API for the two Phys Rev Research papers. Quoted findings were checked by
exact-string search: PubMed abstracts (`efetch db=pubmed`); PMC full text (`efetch db=pmc` XML, or the PMC HTML page when the
XML held only front matter); arXiv PDF or ar5iv HTML. Hekmat-Scafe 2010 was behind a PMC captcha for curl, so it was read through
the fetch tool. Both PubMed negative-search queries in section 0 were rerun and gave the same counts (16 and 23). The 16 + 23
titles include no AL recording under general anesthesia. "Absent" claims (for example "the AL is not named") were checked as zero
hits for "antennal" in the full text. This is a literature summary, so no data-recompute script was written.

| # | citation | status | note |
|---|---|---|---|
| A1 | Paulk 2013 J Neurophysiol | verified | Metadata matches. All quotes found in the full text: ">2,000 registered recording sites ... >200" flies, "most spiking in the antennal lobes", ">15 Hz in the antennal lobes compared with the central brain", "specifically in the antennal lobes", "only these two lines" (GABA and 7B). The only mention of anesthesia is cold anesthesia during handling; no sleep condition. |
| A2 | Aimon 2019 PLoS Biol | verified | Both PN and AL-oscillation quotes are exact; "walked but not when it groomed" is exact. |
| A3 | Mann 2017 Curr Biol | verified | The "high degree of interconnectivity" quote is exact. |
| A4 | Kazama & Wilson 2009 Nat Neurosci | verified | Quotes are exact. 74.9 ± 8.6 Hz (one antenna removed), 3.44 Hz and 17.4 per antenna all confirmed. |
| A5 | Tanaka, Ito, Stopfer 2009 J Neurosci | partly verified | All three abstract quotes are exact. **The issue is 26, not 27** (29(26):8595-603); annotated in place. |
| A6 | Hekmat-Scafe 2010 Genetics | verified | OK107 30%, c232 17%, 104Y 76%, GH146 63% ("only modest reductions"), MB247 32%, c772 23%; GH146>TNT "no significant difference ... (97% BS)". The abstract quotes are exact. |
| A7 | Dean 2018 Fly | verified | All three AL quotes are exact in the full text; cold sensitivity is the paper's topic. |
| A7b | Horne 2017 Genetics | verified | Both abstract quotes are exact. |
| B1 | van Swinderen & Greenspan 2003 Nat Neurosci | partly verified | The quote is exact. The salience cues (odour, heat, novelty) were *coupled to visual stimuli*. The abstract says "neural correlates of perception" and does not say "attention". It does not give the recording site, so "not from the AL" cannot be checked from the abstract. |
| B2 | Rein 2013 J Comp Physiol A | verified | The quotes are exact, including "while reducing average calcium levels". |
| B3 | Murakami 2005 Neuron | verified | Urethane; robust responses in FWS, weak in SWS; the OB quote is exact. |
| B4 | Rinberg 2006 J Neurosci | verified | Both quotes are exact. |
| B5 | Kato 2012 Neuron | verified | Both quotes are exact. |
| B6 | Merrick 2014 Front Psychol | verified | "unique portal" and "must participate in a network that is large-scale" are exact. |
| C1 | van Swinderen 2006 J Neurobiol | verified | Both quotes are exact; the abstract mentions dopaminergic modulation. |
| C2 | Cohen 2016 eNeuro | verified | "inserted laterally to the eye", both tag-frequency quotes, and "0.6% isoflurane abolishes behavioral responsiveness" (air puff) are confirmed. Zero "antennal" hits. |
| C3 | Cohen 2018 eNeuro | verified | "covers approximately half of the fly brain", the "For the center" quote, "preferentially reduced FB" at 0.1-5 Hz, and centre-periphery coherence reduced "around the 10–25 Hz range" are confirmed. Zero "antennal" hits. |
| C4 | Leung 2021 PLoS Comput Biol | verified | All quotes are exact, including the IIT sensory-areas sentence. Zero "antennal" hits. |
| C5 | Muñoz 2020 Phys Rev Research | verified | Metadata matches via arXiv and Crossref. The abstract quotes are exact. Zero "antennal" hits in the full arXiv PDF. |
| C6 | Leung 2025 PLoS Biol | verified | Discovery N = 13; blinded evaluation N = 49. "only 47 time-series features" are significant "only at the deepest channel, Channel 1"; the autocorrelation quote is exact. Zero "antennal" hits. |
| C7 | Troup 2023 J Neurosci | verified | "window cut into the back of the head", "∼18% ... to 15%" (p = 0.0063) and "unchanged" activity (p = 0.847) are confirmed, and the fragmentation quote is exact. Zero "antennal" hits. |
| C8 | Hines 2024 J Neurosci | verified | 44(24). Both quotes are exact; the measurements are in MBONs. |
| C9 | Kottler 2013 Curr Biol | verified | The quote is exact. |
| C9b | Zalucki 2015 Anesthesiology | verified | The quotes are exact. Synaptic release was measured at the *larval NMJ*. |
| C10 | Nitz 2002 Curr Biol | verified | Both quotes are exact. |
| C10b | Schaffer 2023 Nat Commun | verified | The parallelepiped-volume quote and the running/flailing quote are exact. Zero "antennal" hits. |
| C10c | Aimon 2023 eLife | verified | "AL Whole antennal lobe" and PN components appear in the component table only; there is no AL-specific state claim in the text. The abstract quote is exact. |
| C11 | Barron & Klein 2016 PNAS | verified | The CX quote is exact. "antennal" occurs once, in the Fig. 2 legend quote. Zero hits for "sleep" or "arousal". Anesthesia is discussed only for vertebrate cortex and subcortex. |
| C11b | Adamo 2016 PNAS | verified | Content is correct: robots would qualify, far fewer neurons, no emotion circuits, objection to the label. The actual wording is "define consciousness very narrowly", **not** the quoted "too narrow" in the notes (annotated in place). |
| C11c | Key, Arlinghaus, Browman 2016 PNAS | verified | Only the vertebrate midbrain and its insect analogue are discussed; zero hits for "antennal", "olfact", "sleep" or "anesthe". |
| C12 | Saras 2017 Genetics | verified | c739 78%, 117Y 90%, 7B 85%, 121Y 82%; c305a and 1471 0%; Cha positive; a307, 104Y, c232 and Gad1 negative. The locus quote is exact. No GH146 or AL driver was used. |
| C13 | Kroll 2015 Genetics | verified | The quote is exact. |
| C14 | Pavlidis & Tanouye 1995 J Neurosci | verified | The quote is exact; failure is at the PSI-DLMmn synapse. |
| C15 | Lilienthal 2026 J Neurogenet | verified | PMID 42900470 exists (J Neurogenet 2026 Oct). The quotes are exact and the abstract does not mention the AL. |
| D1 | Ódor, Deco, Kelling 2022 Phys Rev Research | partly verified | The published title matches; the arXiv title has the prefix "What makes us humans:". 21,615-node LCC, in-weight normalisation (Eq. 10), K_c = 1.70(2), eta = 0.70(5), Q_w = 1.184e-4, d = 5.4(5), "hard to decide", "does not seem to be enough" and "simpler structure" are all confirmed. **Omitted:** the authors also ran the raw, unnormalised network (Fig. 3, K_c = 0.00090) and say its transition is "not smoother than the homeostatic one". That weakens the claim that in-weight normalisation explains the difference from our result (annotated in place). |

Extra citations that appear only in the notes (not in the summary items) were checked for existence and key phrase. Bademosi
2018 Cell Rep 22:427-440 (nanoclusters), Zalucki & van Swinderen 2016 Conscious Cogn 44:72-88 (quote exact), van Swinderen, Nitz,
Greenspan 2004 Curr Biol 14:81-7 (uncoupling, intermediate arousal), Yap 2017 Nat Commun 8:1815 (7-10 Hz), Mituzaite 2021 eNeuro 8(4)
(quote exact) and Klein & Barron 2016 PNAS 113:E3814-5 (exists, not read) are all real.

**Summary-level corrections.** (i) Summary point (4), "initiation is placed in the MB alpha/beta lobes, the ellipsoid body and the
giant fibre", overstates the evidence. Only the MB alpha/beta lobes are an initiation locus (Saras 2017). In that study,
optogenetic activation of the EB (c232) and giant fibre (a307) did *not* trigger seizures. EB and giant fibre enter through
rescue (Hekmat-Scafe 2010) and suppression (Kroll 2015). (ii) Point (6): the inference that in-weight normalisation hides
AL heterogeneity has to be weighed against Ódor's own raw-weight run, which was no smoother. (iii) Point (3), "only GABA activation raised 50-100 Hz
power in the AL": Paulk 2013 also reports the mixed 7B-Gal4 line doing so. The item text has this right; the summary sentence drops it.
Everything else was confirmed as stated.
