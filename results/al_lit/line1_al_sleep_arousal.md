# Line 1: the antennal lobe and sleep/arousal (Drosophila and other insects)

Compiled 2026-11-09 for the maleCNS Wilson-Cowan / Griffiths project. Each entry gives the
citation, a URL/DOI that was actually opened, what was read (full text vs abstract only), the
finding (quoted or closely paraphrased), and a short "relevance to us" note. Evidence classes:

- (a) direct evidence about the antennal lobe (AL) itself (ORNs in the AL, PNs, LNs, AL glia, AL neuropil)
- (b) olfactory processing in general (insect olfaction beyond the AL, mammalian olfactory bulb/cortex, worms)
- (c) whole-brain or other-region evidence (including negative results: "imaged but no AL effect" or "AL not covered")

Access notes: PMC HTML was intermittently blocked by a captcha, so most PMC full texts were read via
NCBI efetch (`eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=...`); abstracts via the
Europe PMC REST API. bioRxiv returned HTTP 429 for most of the session. Cook et al. 2025 was read
once via WebFetch (full text, summarized). Kirszenblat et al. 2025 is ABSTRACT ONLY. Raw text dumps
are in the session scratchpad, not here.

---------------------------------------------------------------------------------------------------

## Bottom line (read this first)

1. **No study shows the AL is a sleep or wake generator.** The fly sleep-homeostat/drive literature
   points to the dFB, the EB R5 ring neurons, the protocerebral bridge, the MB and clock neurons, not the AL.
   The one study that compared calcium in the AL with other neuropils after sleep deprivation found
   "little change" in the AL while the PB, EB and MB went up (Dai et al. 2024, Sci Adv).
2. **The AL keeps working during sleep.** Sleeping flies still tell odors apart and wake selectively
   to salient ones (French et al. 2021, Nature). The state-dependent gates sit in the ORNs (tachykinin
   receptor in Or85a ORNs) and deeper in the MB (PAM DANs to MBON MB011B to the FSB layers). They do not
   sit at a global AL shut-off. A 2025 preprint (abstract only) reports that KC odor-identity coding is
   intact in sleep while DAN value coding changes. In mammals the same split appears at a different
   relay: OB responses barely change between slow-wave and fast-wave states, but olfactory-cortex responses
   are strongly gated (Murakami et al. 2005).
3. **Olfactory input and AL outputs adjust sleep amount, not consciousness.** The sign is not consistent
   across labs. Kir2.1 silencing of Orco ORNs *reduced* sleep (Hsu et al. 2021). BoNT-C silencing of Orco or
   Ir8a ORNs *increased* sleep (Cook et al. 2025 preprint). ORN silencing did not reproduce the
   injury-induced sleep increase (Singh & Donlea 2020). Individual glomerular channels push sleep in
   different directions, and removing serotonin from the AL's CSD neurons increased night sleep (Cook et al.
   2025). Activating GABAergic inhibitory PNs (iPNs, Mz699) at night strongly increases sleep (Hsu et al.
   2021). One thermosensory part of the AL (the posterior AL "cold" glomerulus, read by GABAergic TPN-IIs)
   inhibits the DN1a clock neurons and reshapes morning sleep in the cold (Alpert et al. 2020).
4. **Sleep need does reach the AL's structure and glia, but no more than it reaches other regions.** After
   16 h of sleep deprivation, Bruchpilot (BRP) in the AL rose +69% (similar to MB, EB and the whole central
   brain; no region-by-sleep interaction) and AL volume rose +15% (Gilestro et al. 2009). Astrocyte Ca2+ in
   the AL and SMP rises with sleep need (Blum et al. 2021). Sleep speeds up glial clearance of injured ORN
   axons from the AL (Singh & Donlea 2020; Stanhope et al. 2020; Zhang et al. 2026). Brain-wide BRP scaling
   after sleep loss is uniform across 37 neuropils within each transmitter class (Weiss et al. 2024).
5. **The best single match to our dynamics comes from honeybees.** Two-photon imaging of AL PNs in
   head-fixed bees at night shows higher glomerular functional connectivity (node degree) in rest
   epochs than in motion epochs. A spiking AL model reproduces this with weakly correlated input noise
   plus *reduced* GABAergic LN conductance, and in the model sleep-like states have less specific odor
   codes (Moguilner et al. 2025, Neural Networks). Their rest epochs were defined by motion only, with no
   arousal-threshold test, and n = 9 bees.
6. **The AL is spontaneously active and tightly coupled even without odor.** PNs fire 1-5 spikes/s
   spontaneously, and the correlations come mostly from shared ORN input (Kazama & Wilson 2009). PN ongoing
   activity is spatiotemporally organized and plastic (Franco & Yaksi 2021). In resting-state whole-brain
   imaging, the olfactory regions (AL, LH, MB, SMP, SLP, CRE) are among the most strongly intercorrelated
   "even in the absence of olfactory input" (Mann et al. 2017). The AL/LH and MB have *more* functional
   coupling than the connectome predicts (Turner et al. 2021). Whole-brain light-field imaging finds PN
   components "spontaneously active in the absence of odor" (Aimon et al. 2019).
7. **Whole-brain sleep imaging has not resolved the AL.** Tainton-Heap et al. 2021 (pan-neuronal nuclear
   GCaMP, ~100 µm from the top of the brain, 18 planes) report no AL or region-specific analysis. The LFP
   sleep-stage studies (Yap et al. 2017; Raccuglia et al. 2019) record central-brain or R5 signals, not AL
   signals. No study found reports AL activity across spontaneous fly sleep stages.

**Implication for our hypothesis (interpretation, not a finding):** the evidence does not support "AL
active = awake". It fits better with "AL active in both wake and sleep; what changes with state is
propagation beyond the first relay and neuromodulatory gating downstream". On that reading, our "AL lit,
rest of brain quiet" regime (just above g ~ 0.16-0.27 at r = 0) looks more like a sleep-like or OB-in-SWS
state. The wake analogue would be recruitment beyond the AL, not the AL ignition point itself. The
bee data (more glomerular coupling at rest, modelled by weaker LN inhibition) also fit an AL that is
*more* self-coupled in rest-like states.

---------------------------------------------------------------------------------------------------

## (a) Direct evidence about the antennal lobe

### A1. Hsu, Choi & Sehgal 2021 — olfactory circuit manipulations and sleep amount
- Citation: Hsu CT, Choi JTY, Sehgal A. (2021) "Manipulations of the olfactory circuit highlight the role of
  sensory stimulation in regulating sleep amount." *Sleep* 44(5): zsaa265. doi:10.1093/sleep/zsaa265
- URL opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC8343592 (WebFetch; read ~100k of 122k characters,
  i.e. abstract, results and discussion; the unread tail is references). FULL TEXT.
- Findings:
  - Removing or gluing the antennae reduced sleep the next day in young flies (4-5 d). Older flies showed no
    change in total sleep, though night sleep fell.
  - Silencing ORNs (Or83b/Orco-Gal4 > Kir2.1, adult-restricted with tub-Gal80ts) "resulted in a significant
    decrease in both daytime and nighttime sleep relative to controls": shorter bouts and no rebound
    (the authors read this as reduced sleep need). The effect was strongest on day 1 and faded with
    prolonged silencing.
  - Activating ORNs (TrpA1) did not increase sleep. A panel of 16 odors did not reliably increase sleep.
  - Excitatory PNs (GH146): silencing had no effect, and activation did not truly increase sleep (multibeam
    monitors showed reduced activity, not more sleep).
  - **Inhibitory PNs (iPNs, Mz699-Gal4):** "activating the iPNs at night dramatically increased sleep when
    tested using both the single and multibeam system", mainly through longer bouts and shorter latency.
    Daytime activation did not, and silencing iPNs did not significantly reduce sleep.
    [Verification 2026-11-09: Hsu et al. themselves note that Mz699 also labels ventrolateral protocerebrum
    (vlPr) neurons that "do not innervate the antennal lobe at all", so the iPN attribution is the likeliest
    reading, not a clean AL-only result.]
  - "No single neuronal population in the olfactory processing pathway was found to bidirectionally
    regulate sleep." Of the LHONs screened, only LH180 reduced sleep when silenced. MB ablation did not block
    the sleep loss after antennectomy.
  - **AL local interneurons were not tested.** Arousal threshold was not measured.
- Relevance: the most direct "AL to sleep" causal result. ORN-driven AL input supports sleep, the opposite
  of an "AL activity = wake" reading, and a GABAergic AL output pathway (iPNs, mlALT) promotes sleep.
  Worth checking whether Mz699-type iPNs are in our 526-neuron module. Our module comes from the
  *excitatory* connectome, so GABAergic iPNs probably are not, which would make them a natural
  "sleep-promoting inhibitory output" of the AL in our r > 0 runs.

### A2. French, Geissmann, Beckwith & Gilestro 2021 — sensory processing during sleep
- Citation: French AS, Geissmann Q, Beckwith EJ, Gilestro GF. (2021) "Sensory processing during sleep in
  *Drosophila melanogaster*." *Nature* 598: 479-482. doi:10.1038/s41586-021-03954-w
- URL opened: accepted manuscript PDF from Imperial Spiral,
  https://spiral.imperial.ac.uk/bitstreams/e505ab5e-6171-47a1-8fe0-9e0356eb8ef5/download (record:
  https://spiral.imperial.ac.uk/entities/publication/1a8e7fb3-8c07-4079-95d7-420545c5a2cd). FULL TEXT
  (accepted manuscript); abstract also checked on Europe PMC (PMID 34588694).
- Findings:
  - A robotic ethoscope delivered odor-laden air puffs to individual flies after 5 min of immobility.
    Attractive acetic acid concentrations (1-5%) woke flies more than humidified air. Aversive (10%) or
    neutral (30%) concentrations did not wake them more, despite the higher intensity.
  - Arousal depended on time of day. Siesta had the highest arousal threshold to mechanical stimuli but
    "has a greater discriminatory power towards salience than deep, early-night sleep".
  - In a 26-condition odor panel, 11 odors elicited waking above controls. Aversive (trap-avoided) odors
    were more likely to wake satiated flies.
  - Internal state: ethanol and overnight sleep deprivation made flies *less* responsive to 5% acetic acid.
    Starvation increased waking specifically to food odors (14/18 food vs 2/8 non-food), described as
    "specific modulation of discrimination at a sensory level rather than a global adjustment in arousal".
  - Circuit: of the MBONs that synapse on FSB layers, silencing MB011B increased arousal to 5% acetic acid.
    So did silencing FSB line R38E07 (layers 5, 8, 9). The dFB 23E10 neurons had "no role for those
    neurons in olfactory processing during sleep". PAM DANs (gamma5) gate the starvation effect.
  - **Peripheral gate in the ORNs:** knocking down the tachykinin receptor (DTKR) in Or85a ORNs masked the
    starvation effect, and silencing Or85a ORNs made sleeping flies respond to high acetic acid. The
    authors' summary: "at least two gate points: a peripheral one taking action in the ORNs and a central
    one in the depth of the mushroom bodies".
  - AL physiology during sleep was **not recorded**. The paper describes the AL only as the relay
    ("the antennal lobe receives and collates olfactory signals ... and relays organised information towards
    the calyx").
- Relevance: during sleep the olfactory pathway through the AL stays functional enough to discriminate
  valence. Gating is distributed (ORNs plus MB/FSB). This argues against the AL going silent in sleep, and
  against the AL's on/off state being the wake/sleep switch.

### A3. Cook ... Dacks 2025 (bioRxiv preprint) — individual ORN channels and the AL serotonergic neuron shape sleep
- Citation: Cook OM, Pursley BT, Pauls G, Roeder S, Aites IA, Coates KE, Dacks AM. (2025) "Individual
  olfactory channels shape distinct parameters of sleep architecture." *bioRxiv*
  doi:10.1101/2025.09.11.675654 (not peer reviewed).
- Opened: Europe PMC record PPR1082462 (abstract), then https://www.biorxiv.org/content/10.1101/2025.09.11.675654v1.full
  via WebFetch. That covered the first 100k of 292k characters (results, figure legends, discussion and
  methods, per the fetch). FULL TEXT, but read through WebFetch's summarizer, so verify the exact wording
  before quoting in a paper.
- Findings:
  - Abstract: "broadly inhibiting olfactory receptor neurons (ORNs) similarly disrupts sleep/wake cycles";
    "the activity of distinct ORN types are important for day and nighttime sleep and heterogeneously shape
    parameters of sleep architecture"; effects depend on mating status and are "in part shaped by
    heterogeneous serotonin (5-HT) receptor expression".
  - **Direction differs from Hsu et al.:** constitutive BoNT-C silencing in mated females *increased* sleep.
    Ir8a-GAL4 (~10-12 posterior glomeruli) increased day and night sleep, Orco-GAL4 increased only night
    sleep ("silencing Orco+ ORNs only increased nighttime sleep"), and Ir25a had no effect on duration.
  - Single ORN types act differently and not additively. Silencing Ir31a (VL2p, DM3), Ir75a (DP1l) or Or35a
    (VC3, VM6l) decreased daytime sleep; silencing Ir41a (VC5) or Ir76a (VM4) increased and consolidated
    night sleep. Many daytime effects were absent in males and virgins.
  - 5-HT2B is expressed in nearly all ORN classes, and its knockdown reshapes day and night sleep.
  - **AL serotonergic CSD neurons:** R60F02-GAL4 > VMAT-RNAi (which removes AL 5-HT staining) increased and
    consolidated nighttime sleep, and SerT dominant-negative did similarly.
  - PNs and LNs were not manipulated.
  - [Verification 2026-11-09: the checker could confirm only the bibliographic record and the abstract-level
    claims (Europe PMC PPR1082462; bioRxiv API). bioRxiv returned HTTP 429 to every full-text request, so
    the direction of the BoNT-C effects, the single-ORN results and the CSDn VMAT-RNAi result are NOT
    independently verified. Check them against the PDF before relying on the sign contrast with Hsu 2021.]
- Relevance: the strongest AL-specific sleep result found. Sleep architecture depends on which glomerular
  channels are active, and on serotonergic modulation inside the AL (CSDn, 5-HT2B on ORNs). The sign
  (silencing increases sleep here, decreases it in Hsu 2021) depends on method, sex and mating state, so
  "AL input promotes wake" vs "AL input promotes sleep" is unresolved. The glomeruli named here (VC5, VM4,
  DP1l, VL2p, DM3, VC3, VM6l) could be checked against the glomeruli carrying weight in our leading
  eigenvector (e.g. DL2v).

### A4. Singh & Donlea 2020 — antennal injury, sleep and synapse pruning in the AL
- Citation: Singh P, Donlea JM. (2020) "Bidirectional regulation of sleep and synapse pruning after neural
  injury." *Current Biology* 30(6): 1063-1076.e3. doi:10.1016/j.cub.2019.12.065
- Opened: NCBI efetch of PMC7199647 (author manuscript). FULL TEXT.
- Findings: "Daytime sleep is elevated after antennal or wing injury, but sleep returns to baseline levels
  within 24 hours"; "Similar increases in sleep are not observed when olfactory receptor neurons are silenced
  or when other sensory organs are severed". Neuroprotective hiw/bsk manipulations in ORNs weaken the
  post-injury sleep increase. "pre-synaptic active zones are preferentially removed from severed axons within
  hours after injury, and ... sleep depriving recently injured flies slows the removal of both active zones
  and damaged axons" (BRP-STaR in Or22a ORN terminals in the AL).
- Relevance: (i) AL debris clearance needs sleep. (ii) Their ORN silencing did *not* raise sleep, and they
  did not report the decrease Hsu et al. found. The direction of "ORN input to sleep" is not settled. For
  us this is a reminder that AL state and sleep are linked through slow structural and glial processes, not
  only through fast dynamics.

### A5. Stanhope ... Keene 2020 — sleep regulates glial engulfment in the AL
- Citation: Stanhope BA, Jaggard JB, Gratton M, Brown EB, Keene AC. (2020) "Sleep regulates glial plasticity
  and expression of the engulfment receptor Draper following neural injury." *Current Biology* 30(6):
  1092-1101.e3. doi:10.1016/j.cub.2020.02.057
- Opened: Europe PMC abstract (PMID 32142708). ABSTRACT ONLY.
- Finding: after ORN axotomy, "Mechanical and genetic sleep deprivation impaired the clearance of damaged
  neurites" and gaboxadol accelerated it. In sleep-deprived animals, JAK-STAT activation, Draper
  upregulation and "innervation of the antennal lobe by glial membranes" were delayed.
- Follow-up: Zhang J, Brown EB, Lloyd E, Yeragi E, Farhy-Tselnicker I, Keene AC. (2026) "Sleep rescues
  age-associated loss of glial engulfment." *PLoS Genetics* 22(1): e1011999. doi:10.1371/journal.pgen.1011999 (Europe PMC
  abstract, PMID 41529082; ABSTRACT ONLY): pharmacological or genetic sleep induction restores clearance of
  axotomized ORN axons by olfactory ensheathing glia in aged flies.
- Relevance: AL glia are sleep-sensitive. Not a dynamical-state result.

### A6. Gilestro, Tononi & Cirelli 2009 — synaptic markers vs sleep/wake, including the AL
- Citation: Gilestro GF, Tononi G, Cirelli C. (2009) "Widespread changes in synaptic markers as a function of
  sleep and wakefulness in *Drosophila*." *Science* 324(5923): 109-112. doi:10.1126/science.1166673
- Opened: NCBI efetch of PMC2715914 (author manuscript). FULL TEXT.
- Findings: BRP immunofluorescence after 16 h of sleep deprivation rose in "4 major structures of the fly
  brain, including antennal lobes (mean of % ± SEM, 69 ± 11), beta lobes of the mushroom bodies (40 ± 11),
  ellipsoid body of the central complex (67 ± 18), and entire central brain (76 ± 15)". The effect was due to
  sleep loss (F = 16.27, P < 0.0001), with no tissue effect and no sleep-by-tissue interaction (F = 0.24,
  P = 0.86). Also: "In flies that were sleep deprived for 16h, antennal lobes were slightly but
  significantly bigger than in flies that had slept (+15%, p = 0.01)".
- Relevance: the AL takes part in wake-related synaptic upscaling (SHY), but not uniquely; the effect is
  brain-wide. Our model's AL dominance comes from synapse counts. If wake raises AL synapse number about as
  much as everywhere else, the AL's relative dominance should be roughly preserved across sleep/wake.
  Strictly that is a prediction, not a finding.

### A7. Blum ... Wu 2021 — astroglial Ca2+ encodes sleep need (imaged in the AL)
- Citation: Blum ID, Keleş MF, Baz ES, Han E, Park K, Luu S, Issa H, Brown M, Ho MCW, Tabuchi M, Liu S, Wu MN.
  (2021) "Astroglial calcium signaling encodes sleep need in *Drosophila*." *Current Biology* 31(1): 150-162.e7.
  doi:10.1016/j.cub.2020.10.012
- Opened: NCBI efetch of PMC8442851. FULL TEXT (results paragraph on the AL/SMP CaMPARI read).
- Finding: astrocyte CaMPARI2 was measured ex vivo in "two different neuropil regions (superior medial
  protocerebrum, SMP, and antennal lobe, AL)". In astrocyte processes, Ca2+ was elevated at ZT12-15 and
  "further increased at ZT0-3 following SD". In cell bodies it was "markedly elevated at ZT0-3 following SD".
  Daytime SD as a stress control did not raise it. The downstream pathway runs to R5 neurons via Spätzle/Toll.
- Relevance: AL astrocytes track sleep need, but the AL was chosen as one of two sample neuropils; the
  effect is not presented as AL-specific.

### A8. Alpert ... Gallio 2020 — posterior-AL "cold" glomerulus and GABAergic TPN-IIs onto DN1a sleep timing
- Citation: Alpert MH, Frank DD, Kaspi E, Flourakis M, Zaharieva EE, Allada R, Para A, Gallio M. (2020)
  "A circuit encoding absolute cold temperature in *Drosophila*." *Current Biology* 30(12): 2275-2288.e5.
  doi:10.1016/j.cub.2020.04.038
- Opened: NCBI efetch of PMC7314653. FULL TEXT.
- Findings: arista thermoreceptors converge on the posterior antennal lobe (PAL) "hot" and "cold"
  glomeruli. TPN-II projection neurons from the cold glomerulus show sustained firing that scales with
  absolute cold below ~25 °C. "TPN-IIs express the inhibitory neurotransmitter GABA", synapse onto DN1a
  clock neurons (GRASP), and cold "all but shut down" DN1a firing. A cold step in the morning increases
  sleep, and blocking DN1a output at 25 °C "significantly increases morning sleep partially mimicking cold
  conditions". The authors describe a circuit providing "powerful and persistent inhibition to brain centers
  involved in regulating sleep and activity".
- Relevance: a concrete, AL-originating, GABAergic projection that regulates sleep timing. Like Hsu's
  iPNs, the AL-to-sleep route runs through *inhibitory* PNs. Thermosensory PAL glomeruli may or may not
  be in our module. Worth checking for VP2/VP3 PNs.

### A9. Moguilner ... Haase 2025 — honeybee AL network state at rest vs motion (night)
- Citation: Moguilner S, Tiraboschi E, Fantoni G, Strelevitz H, Soleimani H, Del Torre L, Hasson U, Haase A.
  (2025) "Neuronal correlates of sleep in honey bees." *Neural Networks* 189: 107575.
  doi:10.1016/j.neunet.2025.107575 (bioRxiv 2024: doi:10.1101/2024.10.11.617548)
- Opened: open-access publisher PDF from the University of Trento repository,
  https://iris.unitn.it/retrieve/334c3c3a-ee3f-4154-88b7-698db2291c27/1-s2.0-S089360802500454X-main.pdf
  (record https://iris.unitn.it/handle/11572/456933). FULL TEXT (methods, results, abstract read).
- Findings: two-photon imaging of PNs (Fura-2-dextran injected into the antenno-protocerebral tracts),
  one AL per head-fixed forager bee, through the whole night, n = 9 bees, rest vs motion classified from
  video. A classifier separated rest from motion at "93 %" out-of-sample accuracy, and "a feature importance
  analysis suggested network features to be decisive. Accordingly, the glomerular connectivity was found to
  be significantly increased in the rest-state patterns"; the node degree "shows the strongest deviation".
  Glomeruli show spontaneous oscillations "on the order of tenths of Hz". A leaky spiking AL model
  reproduces the rest state with "weakly correlated input noise and a reduction of synaptic conductance of
  the inhibitive local neurons (LNs)", and the simulated sleep-like state shows "a decreased specificity of
  the odour code". There was no arousal-threshold test, so "rest" is behavioural immobility at night,
  not verified sleep. Odor responses during rest were simulated, not measured.
- Relevance: the closest empirical match to our AL-centric dynamics. In a rest/sleep-like state the AL
  becomes *more* internally coupled and more synchronous, and the proposed mechanism is weaker LN
  inhibition. In our model, lower inhibition lowers the onset, so at fixed g the AL sits deeper in its
  self-sustained, coherent regime. This supports reading the AL-lit regime as sleep-like, not wake-like.
  Caveats: bee not fly, n = 9, rest is not verified sleep.

### A10. Kazama & Wilson 2009 — spontaneous, correlated PN activity originates in ORNs
- Citation: Kazama H, Wilson RI. (2009) "Origins of correlated activity in an olfactory circuit."
  *Nature Neuroscience* 12(9): 1136-1144. doi:10.1038/nn.2376
- Opened: NCBI efetch of PMC2751859. FULL TEXT.
- Finding: "In the absence of odors, PNs fire spontaneously (typically 1-5 spikes/sec)". "Dual recordings
  showed synchronized spontaneous spikes in projection neurons (PNs) postsynaptic to the same type of olfactory
  receptor neuron (ORN)"; "The primary origin of correlations lies in the divergence of each ORN onto every PN
  in its glomerulus". Heterotypic PNs are uncorrelated, and PN-PN connections contribute less.
- Relevance: the quiet-state AL is spontaneously active and driven by ORNs, as in our model where ORNs
  feed the module. The strong correlations come from convergent ORN-to-PN synapse counts, the same
  feature that localizes our eigenvectors. This is an awake-preparation result, not a sleep result.

### A11. Franco & Yaksi 2021 — organized, plastic ongoing activity in AL PNs
- Citation: Franco LM, Yaksi E. (2021) "Experience-dependent plasticity modulates ongoing activity in the
  antennal lobe and enhances odor representations." *Cell Reports* 37(13): 110165.
  doi:10.1016/j.celrep.2021.110165
- Opened: NCBI efetch of PMC8739562 (GH146 > GCaMP6m). FULL TEXT (abstract, results skimmed).
- Finding: "projection neurons of the fruit fly antennal lobe exhibit spatiotemporally organized ongoing
  activity". After repeated odor exposure there is a "gradual and cumulative decrease in the amplitude and
  number of calcium events occurring in the absence of odor stimulation, as well as a reorganization of
  correlations between olfactory glomeruli", and functional clusters of glomeruli stay stable across epochs.
  The paper does not examine sleep. The authors note that octopamine regulates inhibitory LNs and could
  carry internal state.
- Relevance: ongoing AL activity exists and is state- and experience-dependent. No sleep data.

### A12. Shang et al. 2007; Yaksi & Wilson 2010 — excitatory LNs and electrical coupling inside the AL (mechanism)
- Shang Y, Claridge-Chang A, Sjulson L, Pypaert M, Miesenböck G. (2007) "Excitatory local circuits and their
  implications for olfactory processing in the fly antennal lobe." *Cell* 128(3): 601-612.
  doi:10.1016/j.cell.2006.12.034. Europe PMC abstract (PMC2866183). ABSTRACT ONLY. Finding: "a previously
  unidentified population of cholinergic local neurons (LNs) with multiglomerular processes"; "PNs are driven
  by a combination of glomerulus-specific ORN afferents and diffuse LN excitation".
- Yaksi E, Wilson RI. (2010) "Electrical coupling between olfactory glomeruli." *Neuron* 67(6): 1034-1047.
  doi:10.1016/j.neuron.2010.08.041. Europe PMC abstract (PMC2954501). ABSTRACT ONLY. Finding: eLN-to-PN
  lateral excitation "is mediated by electrical synapses", "abolished by a gap-junction mutation". eLNs also
  synapse onto inhibitory LNs, so they have "two opposing effects on PNs".
- Relevance: these are not sleep papers, but they bear on our module. The connectome-based model has
  chemical synapses only, so it omits eLN-PN gap junctions; the real AL may be *more* recurrently excitatory
  than our weights say. The "lLN" excitatory local neurons in our leading eigenvector match the cholinergic
  eLNs described here. eLNs also recruit inhibitory LNs, which our r-sweep models only coarsely.

### A13. Aimon et al. 2019 — light-field whole-brain imaging; PN components spontaneously active
- Citation: Aimon S, Katsuki T, Jia T, Grosenick L, Broxton M, Deisseroth K, Sejnowski TJ, Greenspan RJ.
  (2019) "Fast near-whole-brain imaging in adult *Drosophila* during responses to stimuli and behavior."
  *PLoS Biology* 17(2): e2006732. doi:10.1371/journal.pbio.2006732
- Opened: NCBI efetch of PMC6395010. FULL TEXT.
- Findings: "brain activity increased on a global scale when the fly walked but not when it groomed"
  (global ΔF/F vs walking R² = 0.37 ± 0.19, N = 6). Odor puffs activated the AL, LH, MB and superior
  neuropils. "Components likely representing the activity of antennal projection neurons were spontaneously
  active in the absence of odor, but some of them became more active when the odor was presented". Flies
  were awake (walk/rest/groom); sleep was not assessed.
- Relevance: the AL has ongoing activity in quiet, awake rest. Behavioural arousal (walking) raises
  activity globally, not AL-specifically.

### A14. Pacheco et al. 2021 — AL PNs carry auditory activity (AL is not purely olfactory)
- Citation: Pacheco DA, Thiberge SY, Pnevmatikakis E, Murthy M. (2021) "Auditory activity is diverse and
  widespread throughout the central brain of *Drosophila*." *Nature Neuroscience* 24: 93-104.
  doi:10.1038/s41593-020-00743-y
- Opened: NCBI efetch of PMC7783861. FULL TEXT.
- Finding: auditory responses appeared in 33/36 major regions, including "neuropils of the olfactory pathway
  (the antennal lobe (AL), the medial antennal lobe tract containing olfactory projection neurons (PNs), the
  mushroom body ... and the lateral horn (LH))". The authors "confirmed that activity within the olfactory
  pathway was present within the somas of individual AL PNs" (GH146 > GCaMP6s).
- Relevance: the AL receives non-olfactory (mechanosensory/auditory) drive, so under "weak external
  drive" it can be excited by more than odors. No sleep data.

---------------------------------------------------------------------------------------------------

## (b) Olfactory processing in general (beyond the AL proper; other species)

### B1. Kirszenblat, Someya & Kazama 2025 (bioRxiv preprint) — odor coding in the MB during fly sleep
- Citation: Kirszenblat L, Someya M, Kazama H. (2025) "Sleep promotes aversive value coding in *Drosophila*."
  *bioRxiv* doi:10.1101/2025.07.04.663253 (not peer reviewed).
- Opened: Europe PMC record PPR1047973 (abstract). bioRxiv full text and PDF returned HTTP 429. ABSTRACT ONLY.
- Finding (abstract): recordings "from multiple neuron types during sleep and wakefulness in the *Drosophila*
  olfactory association center" (the MB). "baseline activity of dopamine neurons (DANs) is distinct between
  active, rest and sleep states". "Optogenetic inhibition of DANs acutely induces sleep and lowers
  odor-evoked arousal". "whereas coding of odor identity in Kenyon cells (KCs) remains intact, coding of odor
  value in DANs is altered", with more aversive representations during sleep; "the brain does not simply
  suppress the propagation of sensory signals during sleep".
- Relevance: if KC odor identity coding is intact in sleep, AL PN output must still carry odor identity
  during sleep, which is indirect evidence that the AL is "on" in sleep. Whether AL PNs were recorded
  directly is unknown from the abstract.

### B2. Murakami, Kashiwadani, Kirino & Mori 2005 — state-dependent gating in olfactory cortex, not bulb (rat)
- Citation: Murakami M, Kashiwadani H, Kirino Y, Mori K. (2005) "State-dependent sensory gating in olfactory
  cortex." *Neuron* 46(2): 285-296. doi:10.1016/j.neuron.2005.02.025
- Opened: Europe PMC abstract (PMID 15848806). ABSTRACT ONLY.
- Finding: in urethane-anesthetized rats alternating between slow-wave (SWS) and fast-wave (FWS) EEG states,
  olfactory cortex neurons "showed robust spike responses to adequate odorants during FWS, whereas they showed
  only weak responses during SWS. The state-dependent change ... was observed in a majority of olfactory cortex
  neurons, but in only a small percentage of olfactory bulb neurons."
- Relevance: the mammalian analogue of the AL (the OB) keeps responding across slow-wave and fast-wave states;
  gating happens one synapse later. Same pattern as flies (A2, B1). Caveat: urethane-induced SWS, not natural
  sleep.

### B3. Tsuno, Kashiwadani & Mori 2008 — OB inhibition varies with behavioural state (rat)
- Citation: Tsuno Y, Kashiwadani H, Mori K. (2008) "Behavioral state regulation of dendrodendritic synaptic
  inhibition in the olfactory bulb." *J Neurosci* 28(37): 9227-9238. doi:10.1523/JNEUROSCI.1576-08.2008
- Opened: Europe PMC abstract (PMC6670912). ABSTRACT ONLY.
- Finding: granule-to-mitral inhibition "was larger and longer when slow waves were present"; in freely
  behaving rats "the inhibition was the largest during slow-wave sleep state, and successively smaller during
  light sleep, awake immobility, and awake moving states", regulated by cholinergic tone.
- Relevance: in the OB, local inhibition is *strongest* in deep sleep. This is the opposite direction from the
  bee AL model inference (A9, weaker LN inhibition at rest). Treat the direction of LN inhibition across
  states as unresolved.

### B4. Kato, Chu, Isaacson & Komiyama 2012 — OB in awake vs anesthetized mice
- Citation: Kato HK, Chu MW, Isaacson JS, Komiyama T. (2012) "Dynamic sensory representations in the olfactory
  bulb: modulation by wakefulness and experience." *Neuron* 76(5): 962-975. doi:10.1016/j.neuron.2012.09.037
- Opened: Europe PMC abstract (PMC3523713). ABSTRACT ONLY.
- Finding: "wakefulness greatly enhances the activity of inhibitory granule cells and makes principal mitral
  cell odor responses more sparse and temporally dynamic" compared with anesthesia. Experience-dependent
  weakening of mitral responses "is prevented by anesthesia".
- Relevance: under anesthesia the OB's inhibitory interneurons are less active and principal cells respond
  more broadly. This agrees with the bee model direction (less LN inhibition, less specific codes) and with our
  model, where less inhibition means easier sustained relay activity. It is anesthesia, not sleep, and
  conflicts in direction with B3.

### B5. Manabe, Kusumoto-Yoshida, Ota & Mori 2011 — OB receives synchronized top-down input in SWS (rat)
- Citation: Manabe H, Kusumoto-Yoshida I, Ota M, Mori K. (2011) "Olfactory cortex generates synchronized
  top-down inputs to the olfactory bulb during slow-wave sleep." *J Neurosci* 31(22): 8123-8133.
  doi:10.1523/JNEUROSCI.6578-10.2011
- Opened: Europe PMC abstract (PMC6622849). ABSTRACT ONLY.
- Finding: "The olfactory cortex is functionally isolated from the external odor world during slow-wave
  sleep". Piriform sharp waves drive "synchronized top-down inputs to the olfactory bulb".
- Relevance: in sleep the first olfactory relay is driven from above, not silenced. In fly terms, look for
  feedback (e.g. from LH/MB or modulatory neurons) into the AL during sleep. Our model has no such
  structured feedback beyond the connectome.

### B6. Carskadon & Herz 2004 — odors rarely wake sleeping humans
- Citation: Carskadon MA, Herz RS. (2004) "Minimal olfactory perception during sleep: why odor alarms will not
  work for humans." *Sleep* 27(3): 402-405. doi:10.1093/sleep/27.3.402
- Opened: Europe PMC abstract (PMID 15164891). ABSTRACT ONLY.
- Finding: n = 6. Responses to odors on 92% of stage 1 trials. "Peppermint was ineffective in stages 2, 4, and
  REM sleep. Pyridine produced behavioral threshold on 45% of stage 2 trials, none in stage 4", while tones
  worked on at least 75% of trials.
- Relevance: species difference. Human olfaction is a poor arousal channel, but flies wake selectively to
  odors (A2). Do not assume the fly AL behaves like a human sensory pathway in sleep.

### B7. Zwaka ... Menzel 2015 — context odor during deep sleep enhances bee memory
- Citation: Zwaka H, Bartels R, Gora J, Franck V, Culo A, Götsch M, Menzel R. (2015) "Context odor presentation
  during sleep enhances memory in honeybees." *Current Biology* 25(21): 2869-2874. doi:10.1016/j.cub.2015.09.069
- Opened: Europe PMC abstract (PMID 26592345). ABSTRACT ONLY.
- Finding: "exposure to an odor during deep sleep that has been present during learning improves memory
  performance the following day"; odor during wake, or novel odors during sleep, did not.
- Relevance: odor information enters the sleeping insect brain effectively enough to trigger consolidation,
  so the AL relay works during deep sleep. Indirect; there is no AL recording.

### B8. Other olfactory-general items (abstract only, for completeness)
- Tanizawa F, Takemoto H. (2021) "Sleep contributes to preference for novel food odours in *Drosophila
  melanogaster*." *Sci Rep* 11: 9395. doi:10.1038/s41598-021-88967-1 (Europe PMC abstract PMC8087676).
  Night-time sleep deprivation increased preference for, and responses to, the rearing-environment odour
  (apple cider vinegar). Behaviour only, no AL physiology.
- Chandra R, ..., L'Étoile ND. (2023) "Sleep is required to consolidate odor memory and remodel olfactory
  synapses." *Cell* 186(13): 2911-2928.e20. doi:10.1016/j.cell.2023.05.006 (Europe PMC abstract PMC10354834).
  In *C. elegans*, sleep plus odor training is required "to diminish inhibitory synaptic connections between the
  AWC chemosensory neurons and the AIYs". First-relay olfactory synapses are remodelled by sleep, in a worm.
- Gaeta G, Wilson DA. (2022) "Reciprocal relationships between sleep and smell." *Front Neural Circuits* 16:
  1076354. doi:10.3389/fncir.2022.1076354 (NCBI efetch PMC9813672; FULL TEXT skimmed). Review: piriform
  odor-evoked activity "is greatly reduced during slow-wave activity" (anesthetized) and during nREM in
  unanesthetized rats (citing Barnes et al. 2011). It cites Hsu et al. 2021 as fly evidence that olfactory
  pathway neurons regulate sleep. [Verification 2026-11-09: Gaeta & Wilson describe Hsu 2021 as
  showing that olfactory-pathway neurons "can regulate sleep in a bidirectional manner", which contradicts
  Hsu's own conclusion that "No single neuronal population ... was found to bidirectionally regulate sleep".
  Cite Hsu directly, not via this review.]
- Krishnan B, Dryer SE, Hardin PE. (1999) "Circadian rhythms in olfactory responses of *Drosophila
  melanogaster*." *Nature* 400: 375-378. doi:10.1038/22566. Tanoue S, Krishnan P, Krishnan B, Dryer SE, Hardin PE.
  (2004) "Circadian clocks in antennal neurons are necessary and sufficient for olfaction rhythms in
  *Drosophila*." *Curr Biol* 14(8): 638-649. doi:10.1016/j.cub.2004.04.009 (both Europe PMC abstracts,
  ABSTRACT ONLY). Electroantennogram responses are circadian, and antennal-neuron clocks are "both necessary
  and sufficient for olfaction rhythms". This is periphery-level time-of-day modulation of the AL's input,
  not a sleep-state effect.
- Lou L, Tu ZJ, Lahondère C, Vinauger C. (2024) "Rhythms in insect olfactory systems: underlying mechanisms and
  outstanding questions." *J Exp Biol* 227(21): jeb244182. doi:10.1242/jeb.244182 (Europe PMC abstract only).
  Review of circadian olfactory rhythms across insects.

---------------------------------------------------------------------------------------------------

## (c) Whole-brain / other-region evidence, including negative results for the AL

### C1. Dai, Le, Ma & Rosbash 2024 — CaMPARI snapshot after sleep deprivation: little change in the AL (NEGATIVE for AL)
- Citation: Dai X, Le JQ, Ma D, Rosbash M. (2024) "Four SpsP neurons are an integrating sleep regulation hub in
  *Drosophila*." *Science Advances* 10(47): eads0652. doi:10.1126/sciadv.ads0652
- Opened: NCBI efetch of PMC11584021. FULL TEXT.
- Finding: in pan-neuronal (nSyb > CaMPARI) whole-brain snapshots after SD, "the increase of CaMPARI signal
  after sleep deprivation in the PB was the most robust and reliable"; the EB also increased; "There was little
  change in signal in the antenna lobe (AL) and the optic lobe (fig. S1D)". In the discussion: "only a minimal
  effect in the vision processing optic lobe and the olfactory processing AL".
- Relevance: the clearest direct test of whether AL activity tracks sleep pressure, and the answer is
  essentially no. Under this readout the AL is a sensory relay whose integrated activity does not encode
  sleep drive.

### C2. Tainton-Heap et al. 2021 — whole-brain calcium during spontaneous and dFB-induced sleep (AL not analysed)
- Citation: Tainton-Heap LAL, Kirszenblat LC, Notaras ET, Grabowska MJ, Jeans R, Feng K, Shaw PJ, van Swinderen B.
  (2021) "A paradoxical kind of sleep in *Drosophila melanogaster*." *Current Biology* 31(3): 578-590.e6.
  doi:10.1016/j.cub.2020.10.081
- Opened: PMC HTML PMC8993047 via curl (full text extracted). FULL TEXT.
- Findings: pan-neuronal nuclear GCaMP6f, "18 optical slices of the central brain at 3.3Hz" over "~100μm depth
  from the top of the brain". Mean % active neurons was not different between wake and sleep overall
  ("average % active wake = 24.05 ± 4.25, average % active sleep = 23.21% ± 9.790; n=7, p = 0.6875"). Early
  sleep (0-5 min) is wake-like, late sleep (>10 min) has the lowest activity, and connectivity (mean degree of
  correlated active neurons) is lower in sleep overall. dFB-induced sleep keeps wake-like activity but blocks
  visual responses. **No AL or olfactory analysis appears anywhere in the text**; analyses are per neuron, not
  per region.
- Relevance: a negative or uncovered result for the AL. The global finding of fewer correlated active
  neurons in deep sleep is a whole-brain result. Our model's "slow window" (one long-lived metastable module)
  could be compared with their "fewer neurons, lower degree" late-sleep stage only at the whole-brain level.

### C3. Weiss, Blundell, Singh & Donlea 2024 — brain-wide presynaptic scaling after sleep loss is uniform across neuropils
- [Verification 2026-11-09: the heading overstates it. The paper's claim is "within neurotransmitter class".
  After 12 h SD only cholinergic BRP rose in every region. GABAergic BRP shows a significant group-by-region
  interaction (prow only), and after 24 h SD the dopaminergic and glutamatergic data show condition-by-region
  interactions. Read it as "cholinergic upscaling is roughly uniform across 37 neuropils".]
- Citation: Weiss JT, Blundell MZ, Singh P, Donlea JM. (2024) "Sleep deprivation drives brain-wide changes in
  cholinergic presynapse abundance in *Drosophila melanogaster*." *PNAS* 121(13): e2312664121.
  doi:10.1073/pnas.2312664121
- Opened: NCBI efetch of PMC10990117. FULL TEXT.
- Finding: after 12 h SD, cholinergic BRP rose "significantly in each individual neuropil region by ~21 to 39%",
  with no regional changes in dopaminergic or glutamatergic neurons. GABAergic neurons rose only in the prow
  (~18%). "Within neurotransmitter class and waking context, BRP changes are similar across the 37 neuropil
  domains". The text does not single out the AL. Male-male social pairing effects were "enriched in neuropil
  regions associated with visual and olfactory processing".
- Relevance: if wake scales excitatory (cholinergic) synapses roughly uniformly, a sleep-deprived connectome
  would raise effective g everywhere. In our model that pushes the AL past onset first. This is a testable
  model prediction, not an empirical AL result.

### C4. Bushey, Tononi & Cirelli 2011 — structural SHY evidence (AL not studied; NEGATIVE for coverage)
- Citation: Bushey D, Tononi G, Cirelli C. (2011) "Sleep and synaptic homeostasis: structural evidence in
  *Drosophila*." *Science* 332(6037): 1576-1581. doi:10.1126/science.1202839
- Opened: NCBI efetch of PMC3128387. FULL TEXT.
- Finding: the three circuits studied were the small LNvs (PDF), MB gamma neurons (MARCM), and VS1 visual
  neurons; synapse size or number rises with wake and falls only with sleep. **The AL was not examined.**

### C5. Mann, Gallen & Clandinin 2017; Turner, Mann & Clandinin 2021 — resting-state functional connectivity
- Mann K, Gallen CL, Clandinin TR. (2017) "Whole-brain calcium imaging reveals an intrinsic functional network
  in *Drosophila*." *Current Biology* 27(15): 2389-2396.e4. doi:10.1016/j.cub.2017.06.076.
  NCBI efetch PMC5967399. FULL TEXT. Finding: "regions involved in olfactory processing (antennal lobe (AL),
  lateral horn (LH), MB, SMP, superior lateral protocerebrum (SLP), crepine (CRE)) showed a high degree of
  interconnectivity"; olfactory regions "are highly interconnected and represent some of the strongest observed
  correlations, even in the absence of olfactory input". Awake, task-free flies; sleep not assessed.
- Turner MH, Mann K, Clandinin TR. (2021) "The connectome predicts resting-state functional connectivity across
  the *Drosophila* brain." *Current Biology* 31(11): 2386-2394.e3. doi:10.1016/j.cub.2021.03.004.
  NCBI efetch PMC8519013. FULL TEXT. Finding: structural (hemibrain) connectivity predicts resting FC, but
  "brain regions belonging to the mushroom body and the lateral horn & antennal lobe have a higher average
  difference" (FC above what SC predicts), and "indirect anatomical pathways ... exert an important influence
  on functional connectivity, especially for regions like the mushroom body, antennal lobe, and lateral horn".
- Relevance: strong empirical support that the AL is a spontaneously co-active, tightly coupled module at
  rest, more so than direct structure predicts. This fits our eigenvector localization and the
  "AL-first, AL-dominant" dynamics. These are awake resting states, not sleep.

### C6. Schaffer et al. 2023; Mann et al. 2021; Brezovec et al. 2024 — behaviour-state whole-brain imaging
- Schaffer ES, ..., Abbott LF, Axel R. (2023) "The spatial and temporal structure of neural activity across the
  fly brain." *Nat Commun* 14: 5572. doi:10.1038/s41467-023-41261-2. NCBI efetch PMC10495430. FULL TEXT.
  "The activity of most neurons correlated (or anticorrelated) with running and flailing". **The AL is not
  mentioned anywhere in the text** (not covered or not analysed).
- Mann K, Deny S, Ganguli S, Clandinin TR. (2021) "Coupling of activity, metabolism and behaviour across the
  *Drosophila* brain." *Nature* 593: 244-248. doi:10.1038/s41586-021-03497-0. NCBI efetch PMC10544789. FULL
  TEXT. The AL appears only as the site of optogenetic PN stimulation for the activity-to-ATP coupling
  experiment. Initiating even minimal movement causes large-scale activity and metabolic changes. No AL state
  result.
- Brezovec BE, Berger AB, Hao YA, Chen F, Druckmann S, Clandinin TR. (2024) "Mapping the neural dynamics of
  locomotion across the *Drosophila* brain." *Current Biology* 34(4): 710-726.e4. doi:10.1016/j.cub.2023.12.063.
  Europe PMC abstract only (PMID 38242122). Walking-related signals are topographically mapped across many
  regions "involved in navigation, memory, sensory processing, and motor control". The abstract makes no
  AL-specific claim.
- Relevance: awake behavioural arousal (running) produces global, not AL-specific, activity changes. That
  argues against the AL as an arousal hub.

### C7. LFP sleep-stage work — no AL recordings
- Yap MHW, Grabowska MJ, Rohrscheib C, Jeans R, Troup M, Paulk AC, van Alphen B, Shaw PJ, van Swinderen B.
  (2017) "Oscillatory brain activity in spontaneous and induced sleep stages in flies." *Nat Commun* 8: 1815.
  doi:10.1038/s41467-017-02024-y. NCBI efetch PMC5704022. FULL TEXT. A 7-10 Hz central-brain oscillation in
  transitional sleep; gaboxadol sleep has low-amplitude LFPs. **No AL mention** (the only olfactory mention is a
  general remark on insect LFP oscillations).
- Raccuglia D, Huang S, Ender A, Heim MM, Laber D, Suárez-Grimalt R, Liotta A, Sigrist SJ, Geiger JRP, Owald D.
  (2019) "Network-specific synchronization of electrical slow-wave oscillations regulates sleep drive in
  *Drosophila*." *Current Biology* 29(21): 3611-3621.e3. doi:10.1016/j.cub.2019.08.070. Europe PMC abstract only.
  Delta-band compound oscillations in EB R5 neurons increase with sleep need and support "sensory gating"
  (light-induced waking). An R5 network result, not an AL result.

### C8. Kaiser & Steiner-Kaiser 1983 — first insect neuronal sleep correlate was visual, not olfactory
- Kaiser W, Steiner-Kaiser J. (1983) "Neuronal correlates of sleep, wakefulness and arousal in a diurnal
  insect." *Nature* 301: 707-709. doi:10.1038/301707a0. Europe PMC abstract only. Sensitivity of optomotor
  interneurons in the bee optic lobes oscillates with sleep/wake. Not AL.

---------------------------------------------------------------------------------------------------

## Searched but not found (record of negative searches)

- No paper found that records **Drosophila AL PN or LN activity across spontaneous sleep vs wake** (with an
  arousal-verified sleep criterion). The closest are the bee AL rest-vs-motion study (A9) and the MB-level
  fly sleep recordings (B1, abstract only).
- No paper found showing that **AL local interneurons (GABAergic or glutamatergic LNs) regulate sleep** in
  flies. Neither Hsu et al. 2021 nor Cook et al. 2025 tested LNs. The AL-to-sleep effects found run through
  ORNs, iPNs (Mz699), thermosensory GABAergic TPN-IIs, and serotonergic CSD neurons.
- No study found of **isoflurane or other anesthetic effects on Drosophila AL odor responses** (one WebSearch;
  the anesthesia LFP work found concerns visual responses). This belongs to Line 2.
- The CaMPARI SD screen (C1) is the only direct region-level comparison that included the AL, and it was
  negative for the AL.

## Suggested checks in our data prompted by this literature
1. Is the Mz699-type GABAergic iPN population (mlALT, multiglomerular) in or adjacent to the 526-neuron
   module, and how strongly does it target LH/other regions? (Hsu 2021: iPN activation increases sleep.)
2. Are the posterior-AL thermosensory glomeruli (VP2/VP3, TPN-II GABAergic) part of the module?
   (Alpert 2020: these PNs regulate morning sleep.)
3. Compute pairwise correlation and network degree among AL glomeruli in the "slow window" vs above
   onset. The bee data predict higher glomerular coupling in the rest-like state (Moguilner 2025).
4. Test whether the AL-only regime (lit AL, quiet elsewhere) looks more like the empirical sleep pattern
   (first relay on, downstream gated) than like wake. That would call for a measure of AL-to-rest-of-brain
   propagation, rather than AL activity alone, as the wake order parameter.
5. The model lacks eLN-PN gap junctions (Yaksi & Wilson 2010), so the real AL is probably *more* recurrently
   excitatory than the connectome weights imply.

---------------------------------------------------------------------------------------------------

## Verification (independent fact-check, 2026-11-09)

Method: every DOI was resolved through the Europe PMC REST API, and authors, year, title, journal,
volume and pages were compared with the citation. Quoted findings were then searched for word for word in
full text where it was available: NCBI efetch PMC XML for 20 papers; the PMC HTML page for Hsu 2021; the
Imperial Spiral accepted-manuscript PDF for French 2021; the IRIS UniTN PDF for Moguilner 2025. Otherwise
they were checked against the Europe PMC abstract. bioRxiv returned HTTP 429 to every request (login node,
compute node and WebFetch), so the full text of neither preprint could be re-read. Two WebSearches were run
to probe the negative claims (AL LNs in sleep; AL imaging in sleeping flies). Neither found a counterexample.

Bottom line: all 41 cited papers exist with the stated authors, year, title and venue. No fabricated
citation was found. Quoted strings matched the source verbatim, apart from trivial elisions (for example,
Gilestro's "ellipsoid body of the central complex (67 ± 18)" is quoted as "ellipsoid body (67 ...)"). Three
things need correcting or qualifying:
1. **Weiss 2024 is overgeneralized** in the summary bullet ("presynaptic upscaling ... uniform across 37
   neuropils") and in the C3 heading. The paper says the changes are similar "within neurotransmitter class".
   After 12 h SD, cholinergic BRP rose in every region. Dopaminergic and glutamatergic BRP did not change,
   and GABAergic BRP had a group-by-region interaction (prow only). After 24 h SD, the TH and vGlut data show
   condition-by-region interactions. Correct wording: "cholinergic presynaptic upscaling after 12 h SD is
   roughly uniform across 37 neuropils".
2. **The details of Cook et al. 2025 are unverified.** Existence, authors and the abstract-level claims are
   confirmed. The specific results (BoNT-C silencing *increases* sleep, the single-ORN signs, CSDn VMAT-RNAi
   increasing night sleep) rest on the original session's WebFetch summary alone. The "inconsistent sign vs
   Hsu 2021" argument in the summary depends on them.
3. **Gaeta & Wilson 2022 calls Hsu 2021 "bidirectional"**, which contradicts Hsu's own conclusion. This is
   a property of the review, not an error in the notes. Annotated in B8.

Minor caveats, not errors:
- Hsu says the Mz699 driver also labels vlPr neurons that do not innervate the AL. "GABAergic AL output
  (iPNs) promotes sleep" is the likeliest reading, not a clean AL-only result.
- The French 2021 all-day stats show 30% acetic acid at p = .041 vs air. The text says the aversive and
  neutral concentrations were "largely statistically similar or inferior", so "did not wake more" is a fair
  paraphrase of the late-night windows.
- The negative-coverage claims (no AL analysis in Tainton-Heap 2021, Bushey 2011, Schaffer 2023, Yap 2017)
  were confirmed by full-text search: "antennal" and "olfact" occur zero times in Tainton-Heap and Bushey,
  "antenna" zero times in Schaffer, and in Yap only a generic discussion sentence. The global "no study
  found" claims cannot be proven, but two searches turned up no counterexample.

| # | Citation | Status | Note |
|---|----------|--------|------|
| A1 | Hsu, Choi & Sehgal 2021, Sleep 44(5):zsaa265 | verified | Full text (PMC HTML). Exact quotes found: Kir2.1 "significant decrease in both daytime and nighttime sleep", no rebound, 16-odor panel, iPN "dramatically increased sleep", LH180, "No single neuronal population ... bidirectionally regulate sleep". Young flies 4-5 d. No arousal-threshold or LN test. Mz699 also labels vlPr neurons (authors' caveat). |
| A2 | French et al. 2021, Nature 598:479-482 | verified | Accepted MS PDF. "eleven of the 26 odourants", food 14/18 vs non-food 2/8, MB011B, R38E07 (FSB layers 5, 8, 9), 23E10, PAM gamma5, DTKR in Or85a, "at least two gate points ..." all found. Puffs after 5 min of inactivity. |
| A3 | Cook et al. 2025, bioRxiv 10.1101/2025.09.11.675654 | partly verified | Record and authors confirmed (Europe PMC PPR1082462; bioRxiv API v1 2025-09-11, not yet published). The abstract supports distinct ORN types shaping day/night sleep, mating-status dependence and 5-HT receptor heterogeneity. The BoNT-C sign, the single-ORN signs and the CSDn VMAT result could not be re-read (HTTP 429). |
| A4 | Singh & Donlea 2020, Curr Biol 30(6):1063 | verified | Full text. All three quoted sentences are verbatim (injury raises daytime sleep; ORN silencing does not; SD slows active-zone and axon removal). hiw/bsk confirmed. |
| A5 | Stanhope et al. 2020, Curr Biol 30(6):1092 | verified | Abstract. Quotes match (SD impaired clearance; gaboxadol accelerated it; delayed JAK-STAT, Draper and "innervation of the antennal lobe by glial membranes"). |
| A5b | Zhang et al. 2026, PLoS Genet 22(1):e1011999 | verified | Abstract (PMC12858066). Antennal axotomy; ensheathing glia engulf ORN neurites; pharmacological or genetic sleep induction restores clearance and Draper induction in aged flies. |
| A6 | Gilestro, Tononi & Cirelli 2009, Science 324:109 | verified | Full text. AL 69 ± 11, beta lobes 40 ± 11, EB 67 ± 18, CB 76 ± 15; sleep×tissue P = 0.86; AL volume +15% (p = 0.01) after 16 h SD, all verbatim. |
| A7 | Blum et al. 2021, Curr Biol 31(1):150 | verified | Full text. CaMPARI2 in SMP and AL, ex vivo; "further increased at ZT0-3 following SD", "markedly elevated"; daytime-SD control no rise; Spätzle to R5. SD was 12 h. |
| A8 | Alpert et al. 2020, Curr Biol 30(12):2275 | verified | Full text. PAL cold glomerulus; TPN-IIs "express the inhibitory neurotransmitter GABA"; cold "all but shut down" DN1a; blocking DN1a at 25 °C "significantly increases morning sleep"; morning cold step increases sleep. |
| A9 | Moguilner et al. 2025, Neural Netw 189:107575 | verified | Full PDF. Fura-2-dextran, n = 9, 93% accuracy, degree feature, "tenths of Hz" oscillations, correlated noise plus reduced LN conductance, less specific sleep-like odor code. "So far, we have only assessed whole-body motion" confirms the rest definition. |
| A10 | Kazama & Wilson 2009, Nat Neurosci 12:1136 | verified | Full text. "typically 1-5 spikes/sec"; "primary origin of correlations lies in the divergence of each ORN onto every PN in its glomerulus"; heterotypic pairs uncorrelated (abstract: "only weakly correlated"). |
| A11 | Franco & Yaksi 2021, Cell Rep 37:110165 | verified | Full text. Both quotes verbatim. "sleep" appears only in the reference list. |
| A12 | Yaksi & Wilson 2010, Neuron 67:1034; Shang et al. 2007, Cell 128:601 | verified | Abstracts. Electrical eLN-PN synapses, abolished by gap-junction mutation, "two opposing effects on PNs"; Shang "diffuse LN excitation". Shang's authors (not listed in the summary): Shang, Claridge-Chang, Sjulson, Pypaert, Miesenböck. |
| A13 | Aimon et al. 2019, PLoS Biol 17:e2006732 | verified | Full text. Global increase with walking, not grooming; R² = 0.37 ± 0.19 (N = 6); PN components "spontaneously active in the absence of odor"; odor responses in AL, LH, superior neuropils, MB. |
| A14 | Pacheco et al. 2021, Nat Neurosci 24:93 | verified | Full text. "33 out of 36 major brain regions"; the AL/mALT/MB/LH sentence and "within the somas of individual AL PNs" are verbatim. |
| C1 | Dai et al. 2024, Sci Adv 10:eads0652 | verified | Full text. "most robust and reliable" (PB); "little change in signal in the antenna lobe (AL) and the optic lobe (fig. S1D)"; "only a minimal effect in ... the olfactory processing AL". |
| C2 | Tainton-Heap et al. 2021, Curr Biol 31:578 | verified | Full text. 18 slices, ~100 µm; 24.05 ± 4.25 vs 23.21 ± 9.79, n = 7, p = 0.6875. Zero occurrences of "antennal" or "olfact". |
| C3 | Weiss et al. 2024, PNAS 121:e2312664121 | verified (item); summary bullet overgeneralized | Full text. "~21 to 39%" cholinergic after 12 h SD; prow-only GABAergic; "similar across the 37 neuropil domains" is qualified by "Within neurotransmitter class". The summary bullet drops that qualifier; see correction 1. |
| C4 | Bushey, Tononi & Cirelli 2011, Science 332:1576 | verified | Full text. s-LNv, MB gamma neurons and VS1 studied; zero occurrences of "antennal" or "olfactory". |
| C5a | Mann, Gallen & Clandinin 2017, Curr Biol 27:2389 | verified | Full text. "high degree of interconnectivity" and "strongest observed correlations, even in the absence of olfactory input", both verbatim. |
| C5b | Turner, Mann & Clandinin 2021, Curr Biol 31:2386 | verified | Full text. The "higher average difference" quote is verbatim, and the difference is defined as z(FC) - z(SC), so the AL/LH and MB have relatively more functional than structural connectivity. Indirect-pathway quote verbatim. |
| C6 | Schaffer et al. 2023 Nat Commun 14:5572; Mann et al. 2021 Nature 593:244; Brezovec et al. 2024 Curr Biol 34:710 | verified | Schaffer full text: "correlated (or anticorrelated) with running and flailing" verbatim, "antenna" 0 hits. Mann 2021 full text: the AL appears only as the optogenetic PN-stimulation site. Brezovec abstract: no AL-specific claim. |
| C7 | Yap et al. 2017 Nat Commun 8:1815; Raccuglia et al. 2019 Curr Biol 29:3611 | verified | Yap full text: 7-10 Hz transitional stage; gaboxadol gives low-amplitude LFPs; no AL recordings. Raccuglia abstract: R5 delta oscillations, sleep need, sensory gating. |
| B1 | Kirszenblat, Someya & Kazama 2025, bioRxiv 10.1101/2025.07.04.663253 | verified (abstract only) | Abstract quotes verbatim ("Optogenetic inhibition of DANs acutely induces sleep and lowers odor-evoked arousal"; KC identity intact, DAN value altered; "does not simply suppress the propagation"). Not peer reviewed. |
| B2 | Murakami et al. 2005, Neuron 46:285 | verified | Abstract. Urethane SWS/FWS; robust responses in FWS, weak in SWS; "majority of olfactory cortex neurons, but in only a small percentage of olfactory bulb neurons". |
| B3 | Tsuno, Kashiwadani & Mori 2008, J Neurosci 28:9227 | verified | Abstract. Inhibition "largest during slow-wave sleep state, and successively smaller during light sleep, awake immobility, and awake moving states"; cholinergic regulation. |
| B4 | Kato et al. 2012, Neuron 76:962 | verified | Abstract. Wakefulness enhances granule-cell activity and makes mitral responses sparser; plasticity "prevented by anesthesia". |
| B5 | Manabe et al. 2011, J Neurosci 31:8123 | verified | Abstract. Both quotes verbatim. |
| B6 | Carskadon & Herz 2004, Sleep 27:402 | verified | Abstract. n = 6 (3 M, 3 F); 92% stage 1; peppermint ineffective in stages 2, 4 and REM; pyridine 45% of stage 2 trials, none in stage 4; tones at least 75%. |
| B7 | Zwaka et al. 2015, Curr Biol 25:2869 | verified | Abstract. Quote verbatim; wake-phase context odor and novel odor during sleep did not help. |
| B8a | Gaeta & Wilson 2022, Front Neural Circuits 16:1076354 | verified (with caveat) | Full text. The piriform "greatly reduced during slow-wave activity" quote and the Barnes 2011 nREM citation are verbatim. The review calls Hsu 2021 "bidirectional", contradicting Hsu itself (annotated in B8). |
| B8b | Krishnan, Dryer & Hardin 1999, Nature 400:375; Tanoue et al. 2004, Curr Biol 14:638 | verified | Abstracts. Circadian olfactory electrophysiology rhythms abolished in per01/tim01; antennal neurons "both necessary and sufficient for olfaction rhythms". (The EAG method is from the papers' figures, not the abstracts.) |
| B8c | Tanizawa & Takemoto 2021, Sci Rep 11:9395; Chandra et al. 2023, Cell 186:2911 | verified | Abstracts. Night SD increases ACV preference and responses; worm AWC-to-AIY inhibitory connections diminished, quote verbatim. |
| B8d | Lou et al. 2024, J Exp Biol 227:jeb244182 (notes only) | verified (metadata) | Record exists as cited. |
| C8 | Kaiser & Steiner-Kaiser 1983, Nature 301:707 | verified | Abstract. Optomotor interneurons in the bee optic lobe; circadian sensitivity oscillation read as sleep/wake correlates. |
| neg | "No study records fly AL across verified sleep / tests AL LNs in sleep / images fly AL under anesthesia" | not disproven | Two WebSearches found no counterexample. A negative claim cannot be fully verified. |

Not applicable: this is a literature summary, so no data-recomputation script (al_verify_0.py) was written.
In-place annotations were added to A1 (Mz699 caveat), A3 (Cook unverified details), B8 (Gaeta) and C3
(Weiss heading). Pre-edit backup in the checker's session scratchpad.
