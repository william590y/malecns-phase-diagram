# Line 3: phase transitions, criticality and state changes in olfactory circuits and whole-fly-brain models

Written 2026-11-09 by a literature subagent for the question "is the antennal lobe (AL) special, and do our simulated
transitions match empirical results on consciousness, sleep or phase transitions in this region?".

How to read this file:
- Every entry gives the citation, the URL or DOI I actually opened, and how much I read: "full text", "abstract only",
  or "full text, keyword search" (I fetched the full text and searched it, rather than reading every page).
- "Kind" sorts the evidence into four groups:
  - (a) AL direct: evidence about the insect antennal lobe itself.
  - (b) olfactory general: the AL of other insects, or the mammalian olfactory bulb (OB) and piriform cortex.
  - (c) whole-brain / other region.
  - (t) theory / model.
- Negative findings are listed explicitly in section 6.
- Our numbers come from `STATE.md`, `CLAUDE.md` and `results/analysis2/phase_complete.md`.

Abbreviations: PN projection neuron, LN local neuron, ORN olfactory receptor neuron, CX central complex,
EB ellipsoid body, GP Griffiths phase, SWS slow-wave sleep.

---------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **The structure matches.** The one paper that compares network statistics across all neuropils of the whole fly
   connectome singles out the AL, but for *reciprocity*, not density.
   - Lin et al. 2024 (Nature) list the AL with the central complex as the high-reciprocity neuropils.
   - They report strong reciprocal connections between the left and right ALs.
   - They find many neuropil-specific, highly reciprocal rich-club neurons in the AL.
   - That fits our result that the leading eigenvectors sit on a bilateral (left plus right) AL module that a
     degree-preserving rewiring destroys.
   - I found no paper saying the AL is the most synapse-dense or most strongly connected neuropil. Lin et al. give the
     EB and fan-shaped body the highest fraction of internal connections, ~~and the EB one of the highest average
     connection strengths~~.
     [VERIFICATION 2026-11-09: Lin's text ranks the EB/FB only on *fraction of internal connection weight*. A
     per-neuropil connection-strength ranking is not stated in the text (it is only plotted in Extended Data Fig. 6b,
     which was not checked). Do not say Lin rank the EB/FB highest for density or strength.]
2. **The "no Griffiths phase" result matches.** The only published criticality simulation on a fly connectome found no
   Griffiths phase.
   [VERIFICATION: "only" is inconsistent with Ódor, Papp & Deco 2025, listed below. That paper is now published in Chaos,
   Solitons & Fractals (Dec 2026, doi:10.1016/j.chaos.2026.119150). Say "the only published criticality study of the
   GP question on a fly connectome". Ódor 2022 also hedges: "or maybe a very weak ones", and, on the homeostatic graph,
   "It is hard to decide if there is a narrow GP in the 1.5 ≤ K ≤ 1.7 region". The "no signs" quote refers to the raw
   graph.]
   - Ódor, Deco & Kelling 2022 ran the Kuramoto model on the hemibrain, weighted by synapse counts.
   - They found a mean-field-like transition with "no signs of sub-critical PL-s, corresponding to a GP".
   - They blame the near-absence of modularity (Q = 0.002).
   - This agrees with our "one metastable region, not a GP".
   - Theory (Moretti & Muñoz 2013) says a GP needs a hierarchy of rare regions. One dominant module therefore gives one
     metastable lifetime, not a GP. Goltsev et al. 2012 predict that a localized leading eigenvector gives activity
     confined to the module above 1/Λ1, followed by a gradual crossover to network-wide activity, which is the shape
     of our onset.
3. **The dynamics clash with Drosophila physiology.**
   - In Drosophila the AL does **not** keep itself active without input. PNs in flies without antennae "show no
     spontaneous activity" (Olsen et al. 2007).
   - Spontaneous PN activity and correlations come mainly from ORN spikes (Kazama & Wilson 2009).
   - Interglomerular inhibition dominates interglomerular excitation (Olsen & Wilson 2008).
   - Lateral excitation exists, but it runs mainly through gap junctions from excitatory LNs (Yaksi & Wilson 2010),
     which a chemical-synapse connectome model does not contain. It also recruits inhibitory LNs.
   - So in vivo the AL behaves like a driven, sub-threshold relay. That matches our *driven* regime (h > 0), not our
     self-sustained AL regime.
   - Our result that inhibition raises the onset does match the empirical role of LNs (gain control; GABA-A blockers
     remove oscillatory synchrony).
4. **No empirical result links the AL or OB to the level of consciousness or to a sleep/anesthesia phase transition.**
   [VERIFICATION: too strong as worded. Papers cited in this file show the OB's dynamics do change with sleep and
   anesthesia state:
   - Mori 2013: sniff-paced OB gamma disappears in SWS and REM.
   - Kato 2012: OB granule-cell inhibition and mitral-cell responses differ between awake and anesthetized animals.
   - Średniawa 2021: ketamine-xylazine produces a prominent 80–130 Hz OB rhythm while 30–65 Hz gamma diminishes.
   - Fontanini & Bower 2005: oscillation frequencies shift with anesthetic depth.
   - Manabe 2011: OB sharp waves appear during SWS, driven top-down by cortex.
   What the evidence supports: no study shows the first relay *generating* or *gating* the state transition or the level
   of consciousness, and state gating of odor responses is mainly cortical (Murakami 2005).]
   - In mammals the first olfactory relay is the *least* state-gated stage. Murakami et al. 2005 found odor responses
     gated by SWS-like vs fast-wave states in most olfactory-cortex neurons but in "only a small percentage of olfactory
     bulb neurons".
   - Under light anesthesia only the OB keeps tracking respiration (Fontanini & Bower 2005).
   - Sleep-related olfactory phenomena are cortical: piriform sharp waves drive the OB top-down during SWS
     (Manabe et al. 2011).
   - The olfactory-consciousness literature is mostly argument. It points to piriform and orbitofrontal cortex, and
     says the OB may be unnecessary for imagined smells (Merrick et al. 2014).
   - Fly whole-brain sleep and anesthesia studies either did not image the AL or did not report it (Tainton-Heap 2021,
     Troup 2023, Cohen 2016, Leung 2021).
   - Intact flies show some of the strongest spontaneous functional correlations among olfactory regions even with the
     antennae removed (Mann et al. 2017). This is the only fly data point that looks like intrinsic AL-network
     coherence.
5. **Partial dynamical analogues exist, but only as stimulus-evoked or cellular phenomena.**
   - Honeybee AL spontaneous activity carries an odor-specific correlation trace for minutes after a 4-s odor
     (Galán et al. 2006, attractor interpretation).
   - Rat mitral cells are intrinsically bistable (Heyward et al. 2001).
   - The olfactory bulb and cortex have an "intrinsic propensity to oscillate" under ketamine-xylazine
     (Fontanini et al. 2003).
   - Locust and moth AL dynamics are transient, inhibition-shaped trajectories (winnerless competition, frequency
     transitions), not phase transitions of the whole brain.
6. **No avalanche or criticality measurement exists for the insect AL, the OB or the piriform cortex, nor for any
   whole-Drosophila-brain recording**, as far as my searches reached. Kinouchi & Copelli 2006 is the only theory that
   puts criticality in olfactory glomeruli (maximal dynamic range).
7. **The one persistent-activity circuit known in the fly is the central complex, not the AL.**
   [VERIFICATION: "the one" is an overstatement. For example, pCd neurons are persistently active and are needed for a
   persistent courtship/aggression state (Jung Y. et al. 2020, Neuron 105:322–333, doi:10.1016/j.neuron.2019.10.028;
   checked in Europe PMC). Better wording: "the best-known self-sustaining circuit".]
   - The ellipsoid-body heading bump persists in darkness (Seelig & Jayaraman 2015).
   - A ring attractor emerges in the PB–EB model (Kakaria & de Bivort 2017).
   - ~~Lin 2024 rank the EB highest for reciprocity and internal connections.~~ [VERIFICATION: Lin's text lists the CX
     (FB, EB, NO) *and the ALs* together as high-reciprocity regions, with no "highest" ranking. The EB/FB lead only on
     fraction of internal connection weight.]
   - Why our weighted, excitatory-only matrix localizes on the AL rather than the CX is worth checking. Likely reasons
     are ORN convergence, synapse counts per edge, and the CX's strong inhibition.

Check suggested by this literature (not done):
- Which AL neurons carry the leading eigenvector, and what transmitter the maleCNS predicts for them?
- Real Drosophila AL LNs are mostly GABAergic, with a cholinergic excitatory-LN minority (Shang 2007, Huang 2010).
- Much lateral excitation is electrical (Yaksi & Wilson 2010).
- If cholinergic LN types or ORN→PN/LN weights drive the module, its self-sustaining loop may be stronger in the model
  than in vivo.

---------------------------------------------------------------------------------------------------------------------

## 1. Fly connectome structure and whole-brain models

### Lin A., Yang R., Dorkenwald S., Matsliah A., Sterling A.R., Schlegel P., ... Jefferis G.S.X.E., Murthy M. 2024. "Network statistics of the whole-brain connectome of Drosophila." Nature 634:153–165. doi:10.1038/s41586-024-07968-y
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC11446825/ (PMC full text; the first ~100k of 154k characters, which
  covers Results, Discussion and figure legends).
- Read: full text (first ~65%). Kind (c), with AL-specific statements.
- Findings:
  - Rich club: total-degree cut-off of 37, 40,218 neurons (~30%). Central-brain neurons are over-represented; rich-club
    neurons are more likely GABAergic. There is no rich club for out-degree alone. "Highly connected hub neurons do
    not seem to be present."
  - AL-specific statements:
    - "regions with high reciprocity probabilities, including the central complex (FB, EB and noduli (NO)) and the ALs".
    - "Strong reciprocal connectivity also occurs across the midline, such as between the left and right ALs".
    - Neuropil-specific highly reciprocal neurons (NSRNs; rich-club, mostly inhibitory, 54% GABA): "In some
      neuropils, such as the AL and EB, there are many NSRNs".
    - Three-node motifs 7–10 are under-represented in the ALs, which the authors link to few unidirectional edges and
      high reciprocity.
    - In the spectral analysis of a reverse random walk, the most-visited "repeller" nodes "include many with synapses
      in the antennal lobes (AL) and medullae (ME)".
    - ALLNs are described as providing global feedback inhibition.
  - What the paper does NOT say:
    - It does not rank the AL highest for density or connection strength.
    - It gives the EB and FB the highest fraction of internal connections, ~~and the EB one of the highest average
      connection strengths~~ [VERIFICATION: not stated in the text; only Extended Data Fig. 6b plots per-neuropil
      strength, which was not checked].
- Relevance: independent support that the AL is a strongly reciprocal, bilaterally coupled module. That is the kind
  of structure that localizes a leading eigenvector (cf. Pastor-Satorras & Castellano 2016), and it matches our
  near-degenerate L/R top pair (4647, 4385). Lin's reciprocity counts include inhibitory LNs, whereas our localization
  is in the excitatory matrix.

### Pospisil D.A., Aragon M.J., Dorkenwald S., ... Murthy M., Pillow J.W. 2024. "The fly connectome reveals a path to the effectome." Nature 634:201–209. doi:10.1038/s41586-024-07982-0
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC11446844/ (also NCBI efetch of the full XML).
- Read: full text. Kind (c), with an olfactory statement.
- Method: eigendecomposition of the *signed* synapse-count matrix of FlyWire (connections under 5 synapses
  thresholded), with eigenvectors ordered by |λ|.
- Findings:
  - The top eigenvectors are extremely sparse: about 50 neurons for the top 10 on average.
  - They are mostly anatomically localized. The first eigenvector has a negative real eigenvalue and 75% of its
    synapses in the lobula plate (visual).
  - "The top olfactory eigenvectors were also anatomically localized and contained mushroom body neurons (row 1) and
    projection neurons from the antennal lobe to the lateral horn (row 2)."
  - One non-localized example (Extended Data Fig. 7) spans "the antennal lobes, lateral horn, and lateral accessory
    lobe".
  - Eigenvalues decay slowly; the 1000th is about 1/10 of the largest.
- Relevance:
  - It confirms that fly-connectome eigenvectors are localized rather than delocalized.
  - It does not find the AL at the top, but it uses a different object: signed weights with inhibitory 2-cycles that
    dominate |λ|. Our STATE notes say the ordinary signed spectrum is misleading for our onset.
  - A direct comparison would need their |λ| ordering applied to the maleCNS.

### Dorkenwald S. et al. 2024. "Neuronal wiring diagram of an adult brain." Nature 634:124–138. doi:10.1038/s41586-024-07558-y
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC11446842/
- Read: full text, first 200k of 232k characters. Kind (c).
- Findings:
  - The whole brain has 0.0175 mm³ of neuropil and about 130 million synapses, i.e. "7.4 synapses per µm3, a much
    higher density than that of mammalian cortex".
  - No per-neuropil synapse density appears in the text I read. Extended Data Fig. 2c,d show the proofread-attachment
    fraction per neuropil, not density.
  - In the information-flow UMAP, "all antennal lobe local neurons and all central complex neurons" cluster together.
- Relevance: a negative result for "the AL is the densest neuropil". The AL is not singled out for density.

### Shiu P.K. et al. 2024. "A Drosophila computational brain model reveals sensorimotor processing." Nature 634:210–219. doi:10.1038/s41586-024-07763-9
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/
- Read: full text, first 100k of 145k characters. Kind (c) model.
- Model: a leaky integrate-and-fire model of the whole FlyWire brain.
  - Excitatory and inhibitory synapses have equal magnitude.
  - Baseline firing is 0 Hz.
  - The single free weight W_syn was set "such that activation of sugar GRNs at 100 Hz resulted in roughly 80% of
    maximal MN9 firing".
  - The authors note that circuits with basal inhibition "may be poorly simulated".
- Findings: no olfactory or AL stimulation, and no discussion of runaway or self-sustained activity in the part I read.
- Relevance: the published whole-brain fly simulation was deliberately run in a regime where activity is input-driven.
  Nothing in it speaks for or against an AL-first ignition.

### Lappalainen J.K. et al. 2024. "Connectome-constrained networks predict neural activity across the fly visual system." Nature 634:1132–1140. doi:10.1038/s41586-024-07939-3
- Opened: Europe PMC record (abstract), PMC11525180.
- Read: abstract only. Kind (c).
- Findings: the model covers only the motion pathways of the optic lobe (64 cell types). There is no whole-brain or AL
  content.
- Relevance: not applicable to the AL question.

### Schlegel P. et al. 2021. "Information flow, cell types and stereotypy in a full olfactory connectome." eLife 10:e66018. doi:10.7554/eLife.66018
- Opened: NCBI efetch full XML of PMC8298098.
- Read: full text, keyword search. Kind (a).
- Findings:
  - A complete hemibrain olfactory-system annotation: ~2600 ALRNs onto 58 right-AL glomeruli.
  - ALLNs have "complex inhibitory or excitatory synaptic interactions with all other neuron types".
  - It reports "highly lateralised circuits in the antennal lobe".
  - My keyword search found no statement that the AL is the densest or most connected neuropil.
- Relevance: background for module composition. Its lateralisation result applies to ORNs, not to PNs.

---------------------------------------------------------------------------------------------------------------------

## 2. Criticality and Griffiths phases on connectomes (theory, plus the fly-specific simulations)

### Ódor G., Deco G., Kelling J. 2022. "Differences in the critical dynamics underlying the human and fruit-fly connectome." Phys. Rev. Research 4:023057. doi:10.1103/PhysRevResearch.4.023057
- The arXiv v2 title is "What makes us humans: ...".
- Opened: https://arxiv.org/pdf/2201.11084 (PDF, text extracted).
- Read: full text. Kind (c)+(t).
- Setup:
  - Hemibrain v1.0.1: 21,662 nodes, 3,413,160 directed edges weighted by synapse counts (1 to 4,299).
  - The fly network is "rather homogeneous, almost structureless" in the adjacency plot. Its modularity quotient is
    "very low: Q = 0.002264".
  - Kuramoto model.
- Findings:
  - "the synchronization transition is mean-field like, and the width of the transition region is larger than in
    random graphs, but much smaller than as for the KKI-18 human connectome."
  - On the raw graph: "no signs of sub-critical PL-s, corresponding to a GP have been found".
  - "Although the link weight distribution of FF exhibits a fat tail, it does not seem to be enough to introduce
    visible GP effects, or maybe a very weak ones."
  - "It is hard to decide if there is a narrow GP in the 1.5 ≤ K ≤ 1.7 region."
- Relevance: the closest prior result to ours, and it **agrees**: no GP in a fly connectome.
  - Caveats: the hemibrain is a partial brain (right AL plus part of the left), and Kuramoto is not Wilson–Cowan.
  - Our finding adds that the weight heterogeneity does produce one localized module (the AL) with a metastable
    window, which their phase-oscillator model could not see.

### Ódor G., Papp I., Deco G. 2025. "Fluctuation-dissipation of the Kuramoto model on fruit-fly connectomes." arXiv:2503.20708; listed as accepted in Chaos, Solitons & Fractals
- Opened: https://arxiv.org/abs/2503.20708
- Read: abstract only. Kind (c)+(t).
- Findings:
  - A full fly connectome of "≃10^5" nodes (the page does not name FlyWire).
  - Raw, homeostatic and random-inhibitory weight variants; non-equilibrium FDT/aging analysis.
  - Spectral dimension d_s ≈ 2.
  - ~~The abstract mentions the GP only as a possible regime.~~ [VERIFICATION: the current arXiv abstract does not
    mention a Griffiths phase at all. It says d_s ≈ 2 < d_c suggests "non mean-field scaling behavior", and ends:
    "fluctuations are maximal in the resting state of the critical brain". The arXiv comment says "accepted version in
    Chaos & Solitons & Fractals". Crossref lists it as published there (Dec 2026, doi:10.1016/j.chaos.2026.119150).]
    No localization or region results are given.
- Relevance: the only whole-fly-brain criticality simulation I found. It has no AL content.

### Moretti P., Muñoz M.A. 2013. "Griffiths phases and the stretching of criticality in brain networks." Nat. Commun. 4:2521. doi:10.1038/ncomms3521
- Opened: https://ar5iv.arxiv.org/html/1308.6661 (arXiv version).
- Read: full text. Kind (t).
- Findings:
  - A GP is an extended region where "activity can transitorily linger for long times within rare active regions",
    producing generic power laws.
  - It requires hierarchical-modular disorder across "very different scales". Plain modular networks cannot support a
    GP, and "a finite topological dimension D is an excellent indicator".
  - Weights "are not necessary".
  - Spectral counterpart: "localized eigenvectors and Lifshitz tails which are the spectral counterparts of rare
    regions and GPs".
  - It shows power-law avalanches on the human (998-node) and C. elegans connectomes.
- Relevance:
  - It explains our negative GP test. We have essentially ONE large rare region (the AL module), not a broad
    distribution of rare-region sizes, so we get one exponential lifetime, i.e. metastability without a GP.
  - Our in-normalized signed spectrum does show a ladder of ~2-neuron localized optic-lobe modes (a Lifshitz-tail-like
    picture), but the dynamics are dominated by the AL module.

### Goltsev A.V., Dorogovtsev S.N., Oliveira J.G., Mendes J.F.F. 2012. "Localization and spreading of diseases in complex networks." Phys. Rev. Lett. 109:128702. doi:10.1103/PhysRevLett.109.128702
- Opened: https://ar5iv.arxiv.org/html/1202.4411
- Read: full text. Kind (t).
- Findings:
  - The inverse participation ratio (IPR) of the principal eigenvector decides the outcome.
    - If it is localized (IPR = O(1)), prevalence just above λc = 1/Λ1 is O(1/N): activity stays confined to a finite
      set.
    - Hubs and "edges with a large weight" are localization centres.
  - In the localized case "a strict epidemic threshold in this case is actually absent". A finite-fraction epidemic
    appears only after a crossover at higher λ.
- Relevance: the textbook account of our phase diagram.
  - Synapse-count weights localize the leading eigenvector on the AL.
  - Silence becomes unstable at g_lin (λ_PF), with activity confined to the AL.
  - Other regions are then recruited gradually (crossover, not a sharp transition).

### Pastor-Satorras R., Castellano C. 2016. "Distinct types of eigenvector localization in networks." Sci. Rep. 6:18847. doi:10.1038/srep18847
- Opened: Europe PMC abstract (PMC4709588).
- Read: abstract only. Kind (t).
- Findings: there are two localization types.
  - On the largest hub (degree exponent γ > 5/2).
  - On a mesoscopic subgraph, the max K-core shell (γ < 5/2).
  - Both are seen in real networks.
- Relevance: our real graph shows the mesoscopic-subgraph type (a module of 100–1000 neurons), while the configuration
  null shows hub-star localization (STATE.md). That split is exactly the dichotomy described here.

### Kinouchi O., Copelli M. 2006. "Optimal dynamical range of excitable networks at criticality." Nature Physics 2:348–351. doi:10.1038/nphys289
- Opened: https://arxiv.org/abs/q-bio/0601037
- Read: abstract only. Kind (t), with an olfactory motivation.
- Findings:
  - A network of excitable elements has maximal sensitivity and dynamic range at the critical point of a
    non-equilibrium phase transition.
  - The authors relate this to "the role of gap junctions in olfactory glomeruli" and retinal ganglion cell output.
- Relevance: the only theory I found that puts criticality inside an olfactory glomerular network. It predicts a
  functional benefit of the AL sitting near its activation threshold under drive, which is our flicker peak at
  g ≈ 0.30 with weak h. It is not evidence that the AL actually is critical.

---------------------------------------------------------------------------------------------------------------------

## 3. Insect antennal lobe dynamics (direct AL evidence)

### MacLeod K., Laurent G. 1996. "Distinct mechanisms for synchronization and temporal patterning of odor-encoding neural assemblies." Science 274:976–979. doi:10.1126/science.274.5289.976
- Opened: https://authors.library.caltech.edu/52457 (abstract).
- Read: abstract only. Kind (a), locust.
- Findings:
  - Picrotoxin in the AL blocked the fast LN→PN GABA synapse, removing synchronization while sparing each PN's slow
    temporal pattern.
  - "Fast GABA-mediated inhibition, therefore, appears to underlie neuronal synchronization but not response tuning."
- Relevance: AL collective dynamics are organised by LN inhibition, which is consistent with inhibition setting our
  onset. The phenomenon is odor-evoked oscillation, not a sustained-activity transition.

### Stopfer M., Bhagavan S., Smith B.H., Laurent G. 1997. "Impaired odour discrimination on desynchronization of odour-encoding neural assemblies." Nature 390:70–74. doi:10.1038/36335
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (a), honeybee.
- Findings:
  - Honeybee PN assemblies oscillate and synchronize. Picrotoxin abolishes the synchrony.
  - Desynchronized bees cannot discriminate molecularly similar odours, though they can still discriminate dissimilar
    ones.
- Relevance: AL synchrony is functional (fine discrimination), but it is a sensory, not arousal-level, function.

### Bazhenov M., Stopfer M., Rabinovich M., Abarbanel H.D.I., Sejnowski T.J., Laurent G. 2001. "Model of cellular and network mechanisms for odor-evoked temporal patterning in the locust antennal lobe." Neuron 30:569–581. doi:10.1016/S0896-6273(01)00286-0
- Opened: https://www.bazhlab.ucsd.edu/wp-content/uploads/2014/04/Neuron2001b.pdf (PDF, text extracted).
- Read: full text. Kind (a)+(t).
- Findings:
  - In the model, slow inhibitory LN→PN connections produce the slow, odor-specific PN patterning.
  - The same GABAergic synapses, through fast receptors, give 20 Hz oscillatory synchronization.
  - Slow patterning is "resistant to blockade of fast inhibition". LN–LN lateral inhibition shapes stimulus-specific
    LN activity.
- Relevance: an established E–I model of the AL in which LN inhibition controls the regime, which is analogous to our
  r dependence.

### Rabinovich M., Volkovskii A., Lecanda P., Huerta R., Abarbanel H.D.I., Laurent G. 2001. "Dynamical encoding by networks of competing neuron groups: winnerless competition." Phys. Rev. Lett. 87:068102. doi:10.1103/PhysRevLett.87.068102
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (t), AL-motivated.
- Findings: inhibitory networks inspired by insect and fish olfaction encode inputs as trajectories near heteroclinic
  connections between saddles ("winnerless competition"), with large capacity.
- Relevance: the dominant dynamical theory of the AL is transient heteroclinic dynamics, not fixed-point
  bistability or criticality.

### Laurent G., Stopfer M., Friedrich R.W., Rabinovich M.I., Volkovskii A., Abarbanel H.D.I. 2001. "Odor encoding as an active, dynamical process: experiments, computation, and theory." Annu. Rev. Neurosci. 24:263–297. doi:10.1146/annurev.neuro.24.1.263
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (b) review.
- Findings: argues that "early olfactory relays are active and dynamical networks" and introduces winnerless
  competition.
- Relevance: background.

### Mazor O., Laurent G. 2005. "Transient dynamics versus fixed points in odor representations by locust antennal lobe projection neurons." Neuron 48:661–673. doi:10.1016/j.neuron.2005.09.032
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (a).
- Findings:
  - PN population trajectories have an on-transient (1–2 s), a fixed point stable for at least 8 s during long odors,
    and an off-transient lasting a few seconds "as activity returns to baseline".
  - Stimulus separation is best in the transients.
- Relevance: the locust AL holds an odor-driven fixed point only while input lasts, then returns to baseline. There
  is no input-free sustained state. This is closest to our driven regime.

### Assisi C., Stopfer M., Bazhenov M. 2011. "Using the structure of inhibitory networks to unravel mechanisms of spatiotemporal patterning." Neuron 69:373–386. doi:10.1016/j.neuron.2010.12.019
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (a)+(t).
- Findings: in an insect AL model, graph colorings of the inhibitory LN network predict which LN groups switch together
  between activity and quiescence.
- Relevance: AL dynamics are dictated by LN network structure. It supports a structure-to-dynamics link, but for the
  inhibitory subnetwork.

### Tanaka N.K., Ito K., Stopfer M. 2009. "Odor-evoked neural oscillations in Drosophila are mediated by widely branching interneurons." J. Neurosci. 29:8595–8603. doi:10.1523/JNEUROSCI.1455-09.2009
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC2753235/
- Read: full text. Kind (a), Drosophila.
- Findings:
  - Odor-evoked ~10 Hz LFP oscillations (recorded in the MB calyx) synchronize PNs.
  - They require the widely branching LN2 population, not LN1.
  - They are "reversibly abolished by application of the GABAa blocker picrotoxin".
  - The oscillations build up over repeated odor presentations. Only odor-evoked activity was studied; no spontaneous
    oscillation was reported.
- Relevance: in Drosophila too, LN inhibition creates the AL's collective dynamics. No input-free AL state is reported.

### Ito I., Bazhenov M., Ong R.C., Raman B., Stopfer M. 2009. "Frequency transitions in odor-evoked neural oscillations." Neuron 64:692–706. doi:10.1016/j.neuron.2009.10.004
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (a), moth.
- Findings: during long odor pulses, AL oscillations "suddenly slowed as net olfactory receptor neuron (ORN) output
  decreased". The frequency is set by ORN adaptation and saturation.
- Relevance: an input-driven dynamical transition inside the AL, controlled by afferent drive. It is not a
  sleep/consciousness transition.

### Daly K.C., Galán R.F., Peters O.J., Staudacher E.M. 2011. "Detailed characterization of local field potential oscillations and their relationship to spike timing in the antennal lobe of the moth Manduca sexta." Front. Neuroeng. 4:12. doi:10.3389/fneng.2011.00012
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (a), moth.
- Findings:
  - The frequency of weakly distributed AL oscillations "was state dependent (spontaneous activity = 25-55 Hz;
    odor-driven = 55-85 Hz)".
  - Spike–LFP phase locking was strongest during spontaneous activity.
  - Bicuculline lowered the frequency.
- Relevance: the AL has spontaneous collective dynamics distinct from its odor-driven ones. "State" here means
  spontaneous vs odor-driven, not sleep vs wake.

### Galán R.F., Weidert M., Menzel R., Herz A.V.M., Galizia C.G. 2006. "Sensory memory for odors is encoded in spontaneous correlated activity between olfactory glomeruli." Neural Comput. 18:10–25. doi:10.1162/089976606774841558
- Opened: https://epub.ub.uni-muenchen.de/14819/1/Galan06_NeuralComp.pdf (text extracted).
- Read: full text. Kind (a), honeybee.
- Findings:
  - After a single 4-s odor, the across-glomerular correlations of spontaneous PN calcium activity are biased toward
    the odor's pattern.
  - "during the first few minutes after odor presentation, correlations between the spontaneous activity fluctuations
    suffice to reconstruct the stimulus". The decay time is a few minutes.
  - The authors interpret this as the AL being "a network with odor-specific attractors and a high level of spontaneous
    activity".
- Relevance: the nearest empirical analogue to our slow window, where the AL holds structured activity long after
  input. But here it is a correlation trace riding on input-driven spontaneous activity, not survival of activity
  without input.

### Olsen S.R., Bhandawat V., Wilson R.I. 2007. "Excitatory interactions between olfactory processing channels in the Drosophila antennal lobe." Neuron 54:89–103. doi:10.1016/j.neuron.2007.03.010
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC2048819/
- Read: full text. Kind (a).
- Findings:
  - "PNs postsynaptic to antennal glomeruli in antennae-less flies show no spontaneous activity."
  - Lateral interglomerular input to PNs is "predominantly excitatory and can be strong enough to trigger a train of
    action potentials" when other ORNs are odor-driven.
- Relevance: **key empirical counterpoint.** Without ORN drive the real AL output is silent, so the in vivo AL is
  below any self-sustaining threshold. Our model has the AL self-sustaining above g = 0.163 even without input.
  - This constrains the biologically plausible g to below the AL onset, or else argues that the model overestimates
    recurrent AL excitation.
  - Lateral excitation is real and can drive spikes, which supports an excitatory intra-AL loop of some strength.

### Kazama H., Wilson R.I. 2009. "Origins of correlated activity in an olfactory circuit." Nat. Neurosci. 12:1136–1144. doi:10.1038/nn.2376
- Opened: https://kazamalab.riken.jp/pdf/NatNeurosci_Kazama&Wilson_2009.pdf (text extracted).
- Read: full text. Kind (a).
- Findings:
  - Spontaneous spikes are synchronized in PNs of the same glomerulus.
  - "The primary origin of correlations lies in the divergence of each ORN onto every PN in its glomerulus. Reciprocal
    PN–PN connections make a smaller contribution."
  - Spontaneous EPSC rate in DM4 PNs (74.9 Hz with one antenna removed) ≈ ORN rate (3.44 Hz) × ORN number (17.4).
  - Contralateral homotypic PNs also show coincident events, because ORNs project bilaterally.
- Relevance:
  - Spontaneous AL activity is ORN-driven: a constant external drive h > 0 on the module in vivo.
  - Bilateral ORN axons couple the left and right ALs, consistent with our bilateral module containing ORNs.

### Olsen S.R., Wilson R.I. 2008. "Lateral presynaptic inhibition mediates gain control in an olfactory circuit." Nature 452:956–960. doi:10.1038/nature06864
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (a).
- Findings: removing interglomerular input broadens PN tuning, "implying that interglomerular inhibition dominates over
  interglomerular excitation". The inhibition "scales with total feedforward input to the entire antennal lobe" and
  acts partly presynaptically.
- Relevance: supports our finding that inhibition raises the onset (0.27 → 0.58 as r goes from 0 to 6). The real AL
  has strong input-scaled inhibitory gain control, which should keep it from running away.

### Shang Y., Claridge-Chang A., Sjulson L., Pypaert M., Miesenböck G. 2007. "Excitatory local circuits and their implications for olfactory processing in the fly antennal lobe." Cell 128:601–612. doi:10.1016/j.cell.2006.12.034
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (a).
- Findings: "a previously unidentified population of cholinergic local neurons (LNs) with multiglomerular processes".
  These excitatory LNs respond broadly and provide diffuse lateral excitation of PNs.
- Relevance: the cholinergic LNs in our excitatory AL module have a real counterpart.

### Huang J., Zhang W., Qiao W., Hu A., Wang Z. 2010. "Functional connectivity and selective odor responses of excitatory local interneurons in Drosophila antennal lobe." Neuron 67:1021–1033. doi:10.1016/j.neuron.2010.08.025
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (a).
- Findings:
  - "reciprocal excitatory connections mediated by dendrodendritic cholinergic synapses and gap junctions" between
    krasavietz eLNs and PNs, and between eLNs.
  - eLN connections to inhibitory LNs were rare.
- Relevance: recurrent eLN–PN excitation exists, so a positive-feedback loop is biologically real.

### Yaksi E., Wilson R.I. 2010. "Electrical coupling between olfactory glomeruli." Neuron 67:1034–1047. doi:10.1016/j.neuron.2010.08.041
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (a).
- Findings:
  - Lateral excitation of PNs is mediated by electrical synapses from eLNs; a gap-junction mutation eliminates it.
  - eLNs also excite inhibitory LNs, "driving both direct excitation and indirect inhibition".
  - Weak stimuli give sensitivity; strong stimuli recruit inhibitory gain control.
- Relevance: much of the AL's lateral excitation is electrical, which is absent from a chemical-synapse connectome.
  The model's AL excitation therefore comes from chemical (ORN→PN/LN, cholinergic LN) synapses, and the
  real-circuit balance includes eLN-driven inhibition.

### Strube-Bloss M.F., Grabe V., Hansson B.S., Sachse S. 2017. "Calcium imaging revealed no modulatory effect on odor-evoked responses of the Drosophila antennal lobe by two populations of inhibitory local interneurons." Sci. Rep. 7:7854. doi:10.1038/s41598-017-08090-y
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (a).
- Findings: silencing two GABAergic multiglomerular LN populations (about 1/3 of LNs) with shi(ts) did not change
  odor-evoked input or output patterns. The authors suggest these LNs act "on a global and longer time scale".
- Relevance: a negative or nuanced result. Not every LN class shapes odor coding, so the inhibition that matters may be
  slow or global.

---------------------------------------------------------------------------------------------------------------------

## 4. Fly whole-brain state, sleep and anesthesia

### Mann K., Gallen C.L., Clandinin T.R. 2017. "Whole-brain calcium imaging reveals an intrinsic functional network in Drosophila." Curr. Biol. 27:2389–2396. doi:10.1016/j.cub.2017.06.076
- The authors are Mann, Gallen and Clandinin, not Gordus/Bargmann as the task text guessed.
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC5967399/
- Read: full text. Kind (c), with an AL statement.
- Setup: pan-neuronal GCaMP6m, two 17.4-min spontaneous sessions in darkness, **antennae removed**.
- Findings:
  - Olfactory regions (AL, LH, MB, SMP, SLP, CRE) "showed a high degree of interconnectivity".
  - They "represent some of the strongest observed correlations ... even in the absence of olfactory input".
  - The SMP and MB medial lobe are the most broadly connected regions.
- Relevance: the only fly data showing intrinsic, input-independent coherence that includes the AL.
  - It is consistent with the AL taking part in a strongly coupled module.
  - It is not evidence of self-sustained AL firing. Calcium correlations can come from small fluctuations, and
    Olsen 2007 shows PNs are silent without antennae.

### Tainton-Heap L.A.L., Kirszenblat L.C., Notaras E.T., Grabowska M.J., Jeans R., Feng K., Shaw P.J., van Swinderen B. 2021. "A paradoxical kind of sleep in Drosophila melanogaster." Curr. Biol. 31:578–590. doi:10.1016/j.cub.2020.10.081
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC8993047/
- Read: full text, first 100k of 132k characters. Kind (c).
- Findings:
  - Two-photon imaging of "18 optical slices of the central brain" spanning ~100 µm from the dorsal surface.
  - "average % active wake = 24.05 ± 4.25, average % active sleep = 23.21% ± 9.790" (n.s.). Late sleep is typically
    lowest.
  - dFB-induced sleep keeps wake-like activity levels but silences visually responsive neurons.
  - The antennal lobe, the olfactory system and the mushroom body are never mentioned.
- Relevance: **negative for an AL link**. Whole-brain sleep imaging reported no AL-specific effect and probably did not
  image the ventral AL. Sleep did not globally silence neurons, which argues against "sleep = below onset, silent".

### Troup M., Tainton-Heap L.A.L., van Swinderen B. 2023. "Neural ensemble fragmentation in the anesthetized Drosophila brain." J. Neurosci. 43:2537–2551. doi:10.1523/JNEUROSCI.1657-22.2023
- Opened: Europe PMC abstract (PMC10082453; the full text was not retrievable).
- Read: abstract only. Kind (c).
- Findings:
  - "Neurons in the Drosophila brain remain active during general anesthesia as well as induced sleep."
  - Correlation patterns "become more fragmented and less diverse under anesthesia but remain wake-like during induced
    sleep".
- Relevance: anesthesia in the fly shows up as loss of integration, not silence. Our model has no analogue of
  fragmentation; the nearest would be the PR (participation ratio) / D boundary. No AL information.

### Cohen D., Zalucki O.H., van Swinderen B., Tsuchiya N. 2016. "Local versus global effects of isoflurane anesthesia on visual processing in the fly brain." eNeuro 3:ENEURO.0116-16.2016. doi:10.1523/ENEURO.0116-16.2016
- Opened: Europe PMC abstract, plus NCBI efetch full text (methods searched).
- Read: abstract and full text, keyword search. Kind (c).
- Findings:
  - Isoflurane reduced power and coherence at the visual tag frequency in central brain regions and increased
    2f responses in the optic lobes.
  - The probe was "inserted laterally", with channels grouped as optic lobe vs central brain. The AL was not sampled.
- Relevance: negative for an AL link. The fly anesthesia electrophysiology did not record the AL.

### Leung A., Cohen D., van Swinderen B., Tsuchiya N. 2021. "Integrated information structure collapses with anesthetic loss of conscious arousal in Drosophila melanogaster." PLoS Comput. Biol. 17:e1008722. doi:10.1371/journal.pcbi.1008722
- Opened: Europe PMC abstract, plus NCBI efetch full text (PMC7946294, methods searched).
- Read: abstract and full text, keyword search. Kind (c).
- Findings:
  - "integrated interactions among populations of neurons during wakefulness collapsed to isolated clusters of
    interactions during anesthesia". The collapse was uniform across the brain.
  - The 16-electrode probe was "inserted laterally into the fly's eye", spanning optic lobe to central brain. The AL
    was not sampled; "olfactory" appears only in the references and review text.
- Relevance: the strongest fly evidence on consciousness-like states. It concerns integration across the
  visual–central axis, with no AL data.

### Seelig J.D., Jayaraman V. 2015. "Neural dynamics for landmark orientation and angular path integration." Nature 521:186–191. doi:10.1038/nature14446
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (c).
- Findings: with no visual or self-motion cues, the EB heading representation "is maintained in this network through
  persistent activity", suggestive of a ring attractor.
- Relevance: the empirically established self-sustaining circuit in the fly is the CX/EB, not the AL. A contrast for
  our AL-first ignition.

### Kakaria K.S., de Bivort B.L. 2017. "Ring attractor dynamics emerge from a spiking model of the entire protocerebral bridge." Front. Behav. Neurosci. 11:8. doi:10.3389/fnbeh.2017.00008
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (c)+(t).
- Findings:
  - A LIF model of the light-microscopy PB–EB connectivity reproduces ring-attractor behaviors, including persistence
    without stimuli.
  - The attractor is robust and arises from "local excitation and long-range inhibition".
- Relevance: the only fly region-specific "self-sustained activity from wiring" model I found. It is in the CX and is
  not built on an EM connectome.

### Singh P., Donlea J.M. 2020. "Bidirectional regulation of sleep and synapse pruning after neural injury." Curr. Biol. 30:1063–1076. doi:10.1016/j.cub.2019.12.065
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (a), sleep.
- Findings:
  - Removing the antennae transiently raises daytime sleep (back to baseline within 24 h).
  - "Similar increases in sleep are not observed when olfactory receptor neurons are silenced", so the cause is
    clearance of damaged ORN axons, not sensory deprivation.
- Relevance: a sleep–AL link through injury and glia, not AL activity level. Silencing AL input does not change sleep,
  which argues against "AL activity level sets arousal".

### French A.S., Geissmann Q., Beckwith E.J., Gilestro G.F. 2021. "Sensory processing during sleep in Drosophila melanogaster." Nature 598:479–482. doi:10.1038/s41586-021-03954-w
- Opened: Europe PMC abstract. A WebSearch summary of press coverage says odour puffs were delivered to sleeping flies.
- Read: abstract only; full text paywalled. Kind (b), sleep.
- Findings: sleeping flies "discern the quality of sensory stimuli and are more likely to wake up in response to salient
  stimuli". Salience is modulated by internal state.
- Relevance: olfactory processing continues during fly sleep. I could not verify which AL neurons are involved.

---------------------------------------------------------------------------------------------------------------------

## 5. Mammalian olfactory bulb and piriform cortex: bistability, slow oscillations, state dependence, consciousness

### Heyward P., Ennis M., Keller A., Shipley M.T. 2001. "Membrane bistability in olfactory bulb mitral cells." J. Neurosci. 21:5311–5320. doi:10.1523/JNEUROSCI.21-14-05311.2001
- Opened: Europe PMC abstract (PMC6762867).
- Read: abstract only. Kind (b).
- Findings:
  - "The majority of mitral cells are bistable", alternating spontaneously between upstate and downstate (~10 mV
    apart) "in the absence of ionotropic excitatory or inhibitory synaptic inputs".
  - Olfactory nerve input can switch a cell to the upstate.
- Relevance: bistability in the first olfactory relay, but at the single-cell (intrinsic) level, not a network phase.

### Fontanini A., Spano P., Bower J.M. 2003. "Ketamine-xylazine-induced slow (< 1.5 Hz) oscillations in the rat piriform (olfactory) cortex are functionally correlated with respiration." J. Neurosci. 23:7993–8001. doi:10.1523/JNEUROSCI.23-22-07993.2003
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC6740491/
- Read: full text. Kind (b), anesthesia.
- Findings:
  - Slow oscillations in OB and piriform field potentials "are all correlated with the natural breathing cycle".
  - Tracheotomy disrupts them, but slow patterns sometimes persist, "supporting the view that the olfactory system has
    an intrinsic propensity to oscillate".
  - Piriform pyramidal cells show up/down states (−76 vs −90 mV).
- Relevance: under anesthesia the olfactory system shows UP/DOWN slow-wave dynamics that are driven by input but also
  partly intrinsic. This is a state-related dynamical regime in olfactory circuits.

### Fontanini A., Bower J.M. 2005. "Variable coupling between olfactory system activity and respiration in ketamine/xylazine anesthetized rats." J. Neurophysiol. 93:3573–3581. doi:10.1152/jn.01320.2004
- Opened: https://researchconnect.stonybrook.edu/en/publications/variable-coupling-between-olfactory-system-activity-and-respirati/
- Read: abstract only. Kind (b), anesthesia.
- Findings: in deep anesthesia the whole olfactory system is highly correlated with respiration. In light anesthesia
  "only the olfactory bulb" tracks respiration. Bulb–cortex coupling depends on anesthetic depth.
- Relevance: anesthetic depth changes cortex–bulb coupling, while the bulb itself stays input-locked. The state
  dependence sits downstream of the first relay.

### Murakami M., Kashiwadani H., Kirino Y., Mori K. 2005. "State-dependent sensory gating in olfactory cortex." Neuron 46:285–296. doi:10.1016/j.neuron.2005.02.025
- This is in Neuron, not Nat Neurosci.
- Opened: Europe PMC abstract (PMID 15848806).
- Read: abstract only. Kind (b), sleep-like state.
- Findings:
  - In urethane-anesthetized rats cycling between slow-wave and fast-wave states, olfactory-cortex odor responses were
    robust in the fast-wave state and weak in the slow-wave state.
  - The change appeared "in a majority of olfactory cortex neurons, but in only a small percentage of olfactory bulb
    neurons".
- Relevance: **important negative for an AL/OB-centred state transition.** The mammalian analogue of the AL is mostly
  not gated by brain state; gating happens in piriform cortex, without a thalamus.

### Manabe H., Kusumoto-Yoshida I., Ota M., Mori K. 2011. "Olfactory cortex generates synchronized top-down inputs to the olfactory bulb during slow-wave sleep." J. Neurosci. 31:8123–8133. doi:10.1523/JNEUROSCI.6578-10.2011
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC6622849/
- Read: full text. Kind (b), sleep.
- Findings:
  - Anterior piriform sharp waves "were absent during waking states but occurred repeatedly during sleep states",
    were most frequent in SWS and were suppressed in REM.
  - During SWS, OB sharp waves are synchronized with them, i.e. top-down input to the OB.
- Relevance: SWS in the mammalian olfactory system is a cortex-driven synchronous regime imposed on the bulb, not a
  bulb-intrinsic transition.

### Wilson D.A., Yan X. 2010. "Sleep-like states modulate functional connectivity in the rat olfactory system." J. Neurophysiol. 104:3231–3239. doi:10.1152/jn.00711.2010
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (b), sleep-like state.
- Findings:
  - In slow-wave-like states the piriform cortex has reduced odor sensitivity and enhanced coherence with other
    forebrain structures.
  - Granger causality runs hippocampus → piriform.
- Relevance: sleep-like states raise long-range coherence of the olfactory cortex. Again this is cortical.

### Kato H.K., Chu M.W., Isaacson J.S., Komiyama T. 2012. "Dynamic sensory representations in the olfactory bulb: modulation by wakefulness and experience." Neuron 76:962–975. doi:10.1016/j.neuron.2012.09.037
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (b), anesthesia vs awake.
- Findings: "wakefulness greatly enhances the activity of inhibitory granule cells and makes principal mitral cell odor
  responses more sparse and temporally dynamic". Anesthesia also blocks experience-dependent plasticity.
- Relevance: in the first relay, the awake state means *more* inhibition. In our model, more inhibition (higher r)
  raises the onset. So if anything, anesthesia would lower inhibition and make the AL more, not less, active.
  - This pulls against a simple "anesthesia = sub-onset, silent AL" mapping.
  - It suggests that the awake/anesthetized axis in the first relay may map onto r rather than g.

### Kay L.M. 2003. "Two species of gamma oscillations in the olfactory bulb: dependence on behavioral state and synaptic interactions." J. Integr. Neurosci. 2:31–44. doi:10.1142/S0219635203000196
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (b).
- Findings:
  - Sniff-locked type 1 gamma (65–100 Hz) vs type 2 gamma (35–65 Hz), the latter during alert immobility.
  - Mice lacking granule-cell inhibition lose type 2 gamma.
- Relevance: OB dynamical regimes depend on behavioral state and on inhibition.

### Mori K., Manabe H., Narikiyo K., Onisawa N. 2013. "Olfactory consciousness and gamma oscillation couplings across the olfactory bulb, olfactory cortex, and orbitofrontal cortex." Front. Psychol. 4:743. doi:10.3389/fpsyg.2013.00743
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC3797617/
- Read: full text, first 100k of 141k characters. Kind (b), consciousness hypothesis.
- Data findings:
  - In the OB, "sniff-paced gamma oscillations occur throughout the waking states, but disappear during slow-wave sleep
    and REM-sleep", except in micro-arousals.
  - OB–APC–OFC gamma coupling occurs occasionally when awake and is absent during sleep.
  - The OFC is "only three synapses distant from the olfactory sensory neurons".
- Hypotheses: a tufted/mitral binding model for olfactory consciousness. The authors acknowledge that conscious
  olfactory perception in rodents is controversial.
- Relevance: the closest "olfactory bulb and consciousness" paper. It ties waking to sniff-paced gamma *coupling*
  across areas, not to a transition inside the bulb.

### Merrick C., Godwin C.A., Geisler M.W., Morsella E. 2014. "The olfactory system as the gateway to the neural correlates of consciousness." Front. Psychol. 4:1011. doi:10.3389/fpsyg.2013.01011
- The authors are Merrick, Godwin, Geisler and Morsella, not Gottfried.
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC3887364/
- Read: full text, first half. Kind (b), consciousness review.
- Findings:
  - "Consciousness of some sort does not require the first-order thalamic nuclei, at least for olfactory experiences."
  - "the olfactory bulb is unnecessary for endogenic olfactory consciousness" (from lesion and imaging cases).
  - "the current literature does not permit one to draw strong conclusions regarding the neuroanatomical regions that
    are critical". OFC lesion cases are suggestive.
- Relevance: the olfactory-consciousness literature does not single out the first relay. If anything it leans toward
  OFC and piriform.

### Skarda C.A., Freeman W.J. 1987. "How brains make chaos in order to make sense of the world." Behav. Brain Sci. 10:161–173. doi:10.1017/S0140525X00047336
- Opened: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/how-brains-make-chaos-in-order-to-make-sense-of-the-world/3811D1F3B8203EFE80CAF7449D0792DF
- Read: abstract only. Kind (b)+(t).
- Findings: from rabbit OB EEG, "chaotic behavior serves as the essential ground state for the neural perceptual
  apparatus". A reference title lists "chaotic and limit cycle attractors in olfactory bulb"; I did not read that
  paper.
- Relevance: historically the main claim of attractor transitions in the first olfactory relay. Freeman's
  inhalation-triggered state transitions are not verified here beyond the abstract.

### Tort A.B.L., Brankačk J., Draguhn A. 2018. "Respiration-entrained brain rhythms are global but often overlooked." Trends Neurosci. 41:186–197. doi:10.1016/j.tins.2018.01.007
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (b)/(c) review.
- Findings: nasal respiration entrains oscillations "in several regions of the rodent brain" and modulates 70–120 Hz
  gamma, most prominently in frontal regions.
- Relevance: the OB as a pacemaker for brain-wide rhythms. This is input-driven entrainment, not a phase transition.

### Biskamp J., Bartos M., Sauer J.F. 2017. "Organization of prefrontal network activity by respiration-related oscillations." Sci. Rep. 7:45508. doi:10.1038/srep45508
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC5368652/
- Read: full text. Kind (b)/(c).
- Findings:
  - A 1–5 Hz prefrontal respiration rhythm appears during awake immobility.
  - "Bilateral bulbectomy resulted in a strong reduction of PRR power", while theta and gamma were unchanged.
  - Sleep and anesthesia were not studied.
- Relevance: the OB drives a waking cortical rhythm, but nothing here is about consciousness level.

### Karalis N., Sirota A. 2022. "Breathing coordinates cortico-hippocampal dynamics in mice during offline states." Nat. Commun. 13:467. doi:10.1038/s41467-022-28090-5
- Opened: Europe PMC abstract, plus NCBI efetch full text (PMC8786964, keyword search).
- Read: full text, keyword search. Kind (b)/(c), sleep.
- Findings:
  - During offline states (including sleep), respiration couples hippocampal ripples to cortical DOWN/UP transitions.
  - Olfactory deafferentation abolished OB-driven prefrontal gamma entrainment. Yet it "left most prefrontal and
    thalamic neurons modulated by breathing", which implies a central respiratory corollary discharge.
- Relevance: a sleep-related UP/DOWN coordination exists that is partly OB-dependent and partly not. The OB is not
  required for the breathing–sleep coupling.

### Viczko J., Sharma A.V., Pagliardini S., Wolansky T., Dickson C.T. 2014. "Lack of respiratory coupling with neocortical and hippocampal slow oscillations." J. Neurosci. 34:3937–3946. doi:10.1523/JNEUROSCI.3581-13.2014
- Opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC6705278/
- Read: full text. Kind (b)/(c), negative.
- Findings:
  - "no such broad forebrain coupling takes place during slow-wave activity patterns" (ketamine-xylazine, urethane,
    natural SWS).
  - "respiratory-related oscillatory activities are likely limited to primary olfactory structures".
  - Coupling rose in REM-like states.
- Relevance: **negative.** Olfactory slow rhythms do not drive the brain-wide slow oscillation of sleep or anesthesia.

### Hunt M.J., Adams N.E., Średniawa W., Wójcik D.K., Simon A., Kasicki S., Whittington M.A. 2019. "The olfactory bulb is a source of high-frequency oscillations (130–180 Hz) associated with a subanesthetic dose of ketamine in rodents." Neuropsychopharmacology 44:435–442. doi:10.1038/s41386-018-0173-y
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (b), anesthetic drug.
- Findings:
  - Ketamine HFO "was of larger magnitude and was phase-advanced in the OB relative to ventral striatum". Granger
    causality points to the OB as the source.
  - Local OB inhibition and naris blockade attenuate it.
- Relevance: an anesthetic-class drug produces a network rhythm that originates in the first olfactory relay and
  spreads. This is the clearest case of a drug-induced brain-state signature with its locus in the OB analogue.
  It concerns a subanesthetic dose, not loss of consciousness.

### Średniawa W., Wróbel J., Kublik E., Wójcik D.K., Whittington M.A., Hunt M.J. 2021. "Network and synaptic mechanisms underlying high frequency oscillations in the rat and cat olfactory bulb under ketamine-xylazine anesthesia." Sci. Rep. 11:6390. doi:10.1038/s41598-021-85705-5
- Opened: https://eprints.whiterose.ac.uk/172477/1/s41598_021_85705_5.pdf (text extracted).
- Read: full text. Kind (b), anesthesia.
- Findings:
  - Ketamine-xylazine anesthesia gives "a prominent 80–130 Hz rhythm in the olfactory bulb", while 30–65 Hz gamma is
    diminished.
  - The rhythm depends on nasal respiration and "persisted despite surgical excision of the piriform cortex".
  - It depends on excitatory–inhibitory synaptic activity, not on gap junctions.
- Relevance: under anesthesia the OB keeps a vigorous local E–I rhythm. The first relay does not go silent under
  anesthesia, which pulls against "anesthesia = AL below onset".

### Watanabe S., Fukuda T., Ueki S. 1980. "Changes in electroencephalogram of the rat following olfactory bulbectomy." Tohoku J. Exp. Med. 130:41–48. doi:10.1620/tjem.130.41
- Opened: https://www.jstage.jst.go.jp/article/tjem1920/130/1/130_1_41/_article/-char/ja
- Read: abstract only. Kind (b).
- Findings: after bilateral bulbectomy:
  - the neocortex gradually showed an arousal-type, low-voltage fast EEG;
  - the amygdala went low-voltage fast at once;
  - hippocampal theta was synchronized for about a week.
- Relevance: removing the OB does not abolish cortical activation. If anything the EEG looks more aroused, so the OB is
  not required for a waking-type EEG.

### Carskadon M.A., Herz R.S. 2004. "Minimal olfactory perception during sleep: why odor alarms will not work for humans." Sleep 27:402–405. doi:10.1093/sleep/27.3.402
- Opened: Europe PMC abstract (PMID 15164891).
- Read: abstract only. Kind (b), human sleep.
- Findings:
  - Odors aroused sleepers on 92% of stage 1 trials.
  - Peppermint was ineffective in stages 2, 4 and REM. Pyridine worked on 45% of stage 2 trials, none in stage 4, and
    a third in REM. Tones worked on ≥75% of trials.
  - "human olfaction is not reliably capable of alerting a sleeper."
- Relevance: in human sleep the olfactory channel is strongly gated. Behavioral data only, with no circuit locus.

### Ponce-Alvarez A., Jouary A., Privat M., Deco G., Sumbre G. 2018. "Whole-brain neuronal activity displays crackling noise dynamics." Neuron 100:1446–1459. doi:10.1016/j.neuron.2018.10.045
- Opened: Europe PMC abstract.
- Read: abstract only. Kind (c), zebrafish.
- Findings:
  - Light-sheet whole-brain imaging of larval zebrafish shows "scale-invariant neuronal avalanches".
  - Gap junctions help maintain criticality.
  - Sensory input and behavior transiently push the system "to a more ordered regime".
- Relevance: the only whole-brain cellular criticality result I found; it is in a vertebrate. Note that input moves the
  system away from criticality toward order. Our driven fluctuation peak is the opposite convention: maximal
  fluctuations under weak drive.

---------------------------------------------------------------------------------------------------------------------

## 6. Negative findings and gaps (searches that returned nothing relevant)

- **No avalanche or criticality study** of the insect AL, the mammalian OB or the piriform cortex. Searched "neuronal
  avalanches olfactory bulb OR piriform OR antennal lobe criticality power law"; every hit was neocortex or theory.
- **No avalanche or criticality analysis of Drosophila whole-brain or whole-CNS recordings**: adult LFP or imaging,
  or larval light-sheet (Lemon et al. 2015 is methods only; I did not open it). Fly anesthesia studies use
  complexity, integrated information or coherence measures instead (Leung 2021; Muñoz et al. 2020 PRR, which I did
  not open).
- **No paper states that the AL is the most synapse-dense or most strongly connected neuropil.**
  - Dorkenwald 2024 gives only a whole-brain density.
  - Lin 2024 single out the AL for reciprocity, while the EB and FB lead on internal connections ~~and strength~~
    [VERIFICATION: the strength ranking is not in the text].
  - A hemibrain per-neuropil density table was not found.
  - This could be computed directly from the maleCNS ROI volumes.
- **No Wilson–Cowan, branching-process or threshold-model criticality study on hemibrain or FlyWire** besides Ódor's
  Kuramoto papers.
- **No study of isoflurane or other anesthetics recorded in the insect AL.**
- **Fly sleep and anesthesia whole-brain studies do not report the AL.**
  - Tainton-Heap 2021 imaged the dorsal 100 µm.
  - Cohen 2016 and Leung 2021 inserted their probes through the eye.
  - The Troup 2023 abstract has no regional result.
- A FENS 2026 poster, Crowe A. & Gong P., "Multifractal connectome eigenmodes link structural heterogeneity to
  multiscale neural representations" (Drosophila connectome), is directly on topic. Only its title and keywords were
  visible at https://world-wide.org/fens-26/multifractal-connectome-eigenmodes-1c956b76, so it is NOT used as
  evidence.

## 7. How each of our results lines up

| our result | aligned evidence | conflicting / complicating evidence |
|---|---|---|
| Leading excitatory eigenvectors localize on a bilateral AL module; weights, not degrees, cause it | Lin 2024: AL has high reciprocity, strong L–R AL reciprocal coupling, many NSRNs. Pospisil 2024: fly eigenvectors are sparse and anatomically localized, including AL→LH PN modes. Goltsev 2012 and Pastor-Satorras & Castellano 2016 (weight-induced and mesoscopic-subgraph localization) | Pospisil's top eigenvector is the lobula plate (signed matrix, ordered by abs λ). No source names the AL as densest. Lin rank the EB/FB higher for internal connections |
| Not a Griffiths phase; one metastable region | Ódor 2022: hemibrain Kuramoto, "no signs of ... GP", low modularity. Moretti & Muñoz 2013: a GP needs hierarchical rare regions | none found |
| AL alone reproduces onset; AL self-sustains activity | Galán 2006: honeybee AL reverberation for minutes. Mann 2017: olfactory regions strongly coherent without input. Shang 2007, Huang 2010: recurrent cholinergic eLN–PN excitation exists | Olsen 2007: PNs silent without antennae. Kazama & Wilson 2009: spontaneous activity is ORN-driven. Mazor & Laurent 2005: activity returns to baseline after odor. Yaksi & Wilson 2010: lateral excitation is largely electrical and recruits inhibition |
| Inhibition raises onset | Olsen & Wilson 2008: inhibition dominates and scales with total input. MacLeod & Laurent 1996, Tanaka 2009: GABA-A LNs control collective dynamics. Kato 2012: awake OB has more inhibition | Strube-Bloss 2017: silencing 1/3 of LNs left odor responses unchanged |
| Weak-drive fluctuation peak where the AL flickers ("awake" analogue) | Kinouchi & Copelli 2006 (theory: dynamic range peaks at criticality, olfactory glomeruli). Kazama & Wilson 2009: the real AL is always ORN-driven (h > 0) | No empirical criticality or avalanche data for the AL. Ponce-Alvarez 2018: input moves zebrafish brain away from criticality |
| Hypothesis: awake = near-critical AL, sleep or anesthesia = sub-onset | none direct | Murakami 2005: the OB is barely state-gated, gating is cortical. Fontanini & Bower 2005: the OB stays respiration-locked under light anesthesia. Średniawa 2021, Hunt 2019: the OB is vigorously rhythmic under ketamine(-xylazine). Tainton-Heap 2021, Troup 2023: fly neurons stay active in sleep and anesthesia. Singh & Donlea 2020: ORN silencing does not change sleep. Viczko 2014: olfactory rhythms do not drive the sleep slow oscillation. Merrick 2014: the OB may be unnecessary for olfactory consciousness |

---------------------------------------------------------------------------------------------------------------------

## Verification (independent fact-check, 2026-11-09)

### Method
- **Metadata.** Every DOI in this file's headings (58) was queried in Europe PMC (REST, `resultType=core`).
  - 55 resolve there, with matching authors, year, title, journal, volume and pages.
  - The other 3 are not indexed in Europe PMC and were confirmed instead through Crossref and arXiv: Ódor 2022 (PRR),
    Kinouchi & Copelli 2006 (Nat Phys) and Skarda & Freeman 1987 (BBS).
  - Ódor 2025 was confirmed through the arXiv API and Crossref.
- **Quotes.** Every quoted phrase was string-matched, after normalization, against the abstract plus a full text where
  one was available. Full texts came from:
  - the Europe PMC `fullTextXML` service;
  - NCBI efetch (PMC);
  - PMC HTML pages;
  - arXiv PDFs (Ódor 2022, Moretti & Muñoz 2013, Goltsev 2012);
  - author PDFs (Galán 2006, Kazama & Wilson 2009).
  Phrases that did not match were then read in context. All of those failures were caused by tokenization (for example
  "(AL)", "(row 1)" or "(LNs)" inside the quote), not by the quote being wrong.
- **Paraphrases.** Claims that are not quotes were checked by reading the matching abstract or full-text passage.
- **Not checked.** The numbers our model produces (g = 0.163, onset 0.27 → 0.58, the eigenvalue pair) are not
  literature claims and were not checked. No data recomputation was done; this is a literature summary.
- **Summary count.** Overall: 56 verified, 2 partly verified, 0 misreported, 0 not found. The partly verified items
  are Lin 2024, where the summary's wording overreaches, and Ódor 2025, where the abstract claim is wrong.

### Table

| citation | status | note |
|---|---|---|
| Lin et al. 2024 Nature 634:153 | partly verified | All quotes are in the PMC full text: L/R AL reciprocity; "include many with synapses in the antennal lobes (AL)"; rich club 40,218 neurons (~30%); hubs "do not seem to be present"; many NSRNs in the AL and EB. **Overreach:** the summary says Lin "rank the ellipsoid and fan-shaped bodies highest" on density/strength. The text ranks them only on *fraction of internal connections*. Lin also do not rank the EB "highest for reciprocity": the CX and the ALs are listed together. |
| Pospisil et al. 2024 Nature 634:201 | verified | Threshold of fewer than 5 synapses; "around 50 neurons" on average for the top 10 eigenvectors; eigenvector 1 has 75% of its synapses in the lobula plate and a negative real eigenvalue; olfactory-eigenvector quote matches (the authors' "(row 1)" was omitted). |
| Dorkenwald et al. 2024 Nature 634:124 | verified | "7.4 synapses per µm3" whole brain; UMAP quote present; no per-neuropil density in the text. |
| Shiu et al. 2024 Nature 634:210 | verified | Single free parameter W_syn sets the size of every excitatory and inhibitory synapse; baseline 0 Hz; tuned so sugar GRNs at 100 Hz give about 80% of maximal MN9 firing; "may be poorly simulated" quote present; olfactory terms appear only in the references. |
| Ódor, Deco & Kelling 2022 PRR 4:023057 | verified | Hemibrain v1.0.1, 21,662 nodes; Q = 0.002264; mean-field quote; "no signs of sub-critical PL-s ..." (raw graph). The quoted "not enough to introduce visible GP effects" is clipped: the original continues "or maybe a very weak ones". The paper also says it is "hard to decide if there is a narrow GP in the 1.5 ≤ K ≤ 1.7 region" (homeostatic graph). The summary drops these hedges. The arXiv title differs ("What makes us humans: ..."). |
| Ódor, Papp & Deco 2025 arXiv:2503.20708 | partly verified | Exists. The abstract confirms about 10^5 nodes, raw/homeostatic/random-inhibitory variants and d_s ≈ 2. **Wrong:** the abstract does not mention a Griffiths phase at all. It is now published in Chaos Solitons Fractals (Crossref: Dec 2026, doi:10.1016/j.chaos.2026.119150), which makes the summary's "the only published criticality simulation on a fly connectome (Ódor 2022)" inaccurate. |
| Moretti & Muñoz 2013 Nat Commun 4:2521 | verified | "disorder needs to be present at very different scales"; "Plain modular networks ... are not able to support GPs"; weights "not necessary"; "localized eigenvectors and Lifshitz tails which are the spectral counterparts of rare regions and GPs". |
| Goltsev et al. 2012 PRL 109:128702 | verified | IPR of order O(1) means localized; "hubs and edges with large weights are centers of localization"; "a strict epidemic threshold in this case is actually absent"; smooth crossover at higher λ. |
| Pastor-Satorras & Castellano 2016 Sci Rep 6:18847 | verified | Localization on the hub (γ > 5/2) or on the max-K-core mesoscopic subgraph (γ < 5/2); both are seen in real networks. |
| Kinouchi & Copelli 2006 Nat Phys 2:348 | verified | Abstract as stated. Note: the result is for generic excitable networks, and olfactory glomeruli are cited only as "compatible". The summary's "argues ... in olfactory glomeruli" is slightly strong. |
| MacLeod & Laurent 1996 Science 274:976 | verified | Quote is in the abstract. |
| Stopfer et al. 1997 Nature 390:70 | verified | Honeybee; picrotoxin; fine but not coarse discrimination impaired. |
| Bazhenov et al. 2001 Neuron 30:569 | verified | Abstract matches. The claim that LN–LN inhibition shapes LN activity was not separately checked. |
| Rabinovich et al. 2001 PRL 87:068102 | verified | Abstract matches (heteroclinic, insects and fish, capacity e(N-1)!). |
| Mazor & Laurent 2005 Neuron 48:661 | verified | Phases of 1–2 s, at least 8 s and "a few seconds as activity returns to baseline". |
| Assisi, Stopfer & Bazhenov 2011 Neuron 69:373 | verified | Graph colorings predict LN groups switching between activity and quiescence. |
| Tanaka, Ito & Stopfer 2009 J Neurosci 29:8595 | verified | About 10 Hz in the MB calyx; "reversibly abolished" by picrotoxin; only the widely branching LN population (LN2) is necessary; oscillations "emerged gradually during repeated presentations"; "spontaneous" never appears. |
| Ito et al. 2009 Neuron 64:692 | verified | Moth; "suddenly slowed as net olfactory receptor neuron (ORN) output decreased". |
| Daly et al. 2011 Front Neuroeng 4:12 | verified | State-dependent frequency of 25–55 Hz spontaneous vs 55–85 Hz odor-driven; phase locking strongest when spontaneous; bicuculline lowers frequency. |
| Galán et al. 2006 Neural Comput 18:10 | verified | 4-second odor; both quotes present in the author PDF. |
| Olsen, Bhandawat & Wilson 2007 Neuron 54:89 | verified | Both quotes present. Caveat: the maxillary palps were intact. "No spontaneous activity" applies to PNs of antennal glomeruli, and palp odors still depolarized them through lateral input. The AL was not fully deafferented. |
| Kazama & Wilson 2009 Nat Neurosci 12:1136 | verified | Quote present. 74.9 ± 8.6 Hz spontaneous EPSC rate in DM4 PNs ≈ 3.44 Hz × 17.4 ORNs. ORNs project bilaterally. |
| Olsen & Wilson 2008 Nature 452:956 | verified | Both quotes are in the abstract. |
| Shang et al. 2007 Cell 128:601 | verified | Quote present (the original reads "cholinergic local neurons (LNs) with multiglomerular processes"). |
| Huang et al. 2010 Neuron 67:1021 | verified | Quote is in the abstract. |
| Yaksi & Wilson 2010 Neuron 67:1034 | verified | Gap-junction mutation abolishes eLN→PN transmission and odor-evoked lateral excitation; "driving both direct excitation and indirect inhibition". |
| Strube-Bloss et al. 2017 Sci Rep 7:7854 | verified | LN1/LN2 make up one third of LNs; no change in odor patterns; "on a global and longer time scale". |
| Schlegel et al. 2021 eLife 10:e66018 | verified | About 2600 ALRNs (2644) in 58 right-AL glomeruli; "highly lateralised circuits in the antennal lobe" (abstract); LN quote present; no density ranking. |
| Mann, Gallen & Clandinin 2017 Curr Biol 27:2389 | verified | GCaMP6m; 17-minute sessions in complete darkness; antennae removed; all three quotes present. SMP has "high functional connectivity to the rest of the brain" and MB_ML is "highly correlated with much of the fly brain". |
| Tainton-Heap et al. 2021 Curr Biol 31:578 | verified | 18 optical slices, about 100 µm from the top of the brain; wake/sleep % active quotes present; "antennal" and "olfact" appear nowhere in the full text. |
| Troup et al. 2023 J Neurosci 43:2537 | verified | Both quotes are in the abstract; the full text mentions olfaction only in a reference title. |
| Cohen et al. 2016 eNeuro 3:0116-16 | verified | Power and coherence at the tag frequency reduced centrally; doubled-frequency increase in the optic lobes; probe inserted laterally; channels grouped as optic lobe vs central brain. |
| Leung et al. 2021 PLoS Comput Biol 17:e1008722 | verified | Collapse quote present; "inserted laterally into the fly's eye"; "collapsed uniformly across the brain". |
| Seelig & Jayaraman 2015 Nature 521:186 | verified | Persistence quote is in the abstract. The citation is correct, but the summary's "the one self-sustaining circuit known in the fly" is an overstatement: pCd persistent activity (Jung et al. 2020 Neuron 105:322) is another example. |
| Kakaria & de Bivort 2017 Front Behav Neurosci 11:8 | verified | LIF PB–EB model built from light-microscopy wiring; persistence recapitulated; "local excitation and long-range inhibition". |
| Lappalainen et al. 2024 Nature 634:1132 | verified | 64 cell types in the optic-lobe motion pathways. |
| Singh & Donlea 2020 Curr Biol 30:1063 | verified | Quote and 24-hour return to baseline are in the abstract. |
| French et al. 2021 Nature 598:479 | verified | Quote is in the abstract. The Imperial College and EurekAlert press releases confirm odour puffs. The circuit was not checked (paywalled). |
| Heyward et al. 2001 J Neurosci 21:5311 | verified | Bistable, about 10 mV apart, independent of ionotropic synaptic input; ON input switches cells to the up state. |
| Fontanini, Spano & Bower 2003 J Neurosci 23:7993 | verified | "intrinsic propensity to oscillate"; tracheotomy disrupts the oscillations; up state -76.08 mV vs down state -89.54 mV. |
| Fontanini & Bower 2005 J Neurophysiol 93:3573 | verified | Only the OB is respiration-correlated under light anesthesia; the whole system under deep anesthesia. |
| Murakami et al. 2005 Neuron 46:285 | verified | Quote is in the abstract. |
| Manabe et al. 2011 J Neurosci 31:8123 | verified | Quote present; "OC-SPWs were suppressed during REM sleep". |
| Wilson & Yan 2010 J Neurophysiol 104:3231 | verified | Abstract matches (hippocampus → piriform Granger direction). |
| Kato et al. 2012 Neuron 76:962 | verified | Quote is in the abstract. |
| Kay 2003 J Integr Neurosci 2:31 | verified | Type 1 (65–100 Hz, sniff-locked) and type 2 (35–65 Hz, alert immobility); selective loss of type 2 when granule-cell inhibition is deleted. |
| Mori et al. 2013 Front Psychol 4:743 | verified | Quotes present; OFC "only three synapses distant"; coupling "occasionally during awake periods, but is absent during sleep". Note: its OB-gamma-vanishes-in-sleep finding is itself a sleep-state change in the first relay (see the annotation on bottom line 4). |
| Merrick et al. 2014 Front Psychol 4:1011 | verified | All three quotes present. |
| Skarda & Freeman 1987 BBS 10:161 | verified | Crossref metadata correct. The Cambridge abstract contains "chaotic behavior serves as the essential ground state for the neural perceptual apparatus". The abstract says "olfactory bulb" EEG but does not name the rabbit. |
| Tort, Brankačk & Draguhn 2018 Trends Neurosci 41:186 | verified | Abstract matches (70–120 Hz, frontal). |
| Biskamp, Bartos & Sauer 2017 Sci Rep 7:45508 | verified | Bulbectomy quote present (1–5 Hz; theta and gamma unchanged). |
| Karalis & Sirota 2022 Nat Commun 13:467 | verified | "the olfactory deafferentation (OD) left most prefrontal and thalamic neurons modulated by breathing"; respiratory corollary discharge. |
| Viczko et al. 2014 J Neurosci 34:3937 | verified | Quotes present (the original has "respiratory-related oscillatory *neural* activities ... during slow-wave forebrain states"); coupling enhanced in REM and REM-like states. |
| Hunt et al. 2019 Neuropsychopharmacology 44:435 | verified | Abstract matches. Note: the dose is *subanesthetic*, so this is not an anesthesia result even though the summary groups it with anesthesia. |
| Średniawa et al. 2021 Sci Rep 11:6390 | verified | Both quotes are in the abstract; depends on E–I synapses, "but not gap junctions". |
| Watanabe, Fukuda & Ueki 1980 Tohoku J Exp Med 130:41 | verified | Abstract matches. Hippocampal theta was "markedly synchronized during the period of a week or so", i.e. it *became* synchronized; "stays synchronized" is slightly off. Europe PMC misspells the third author as "Ueko". |
| Carskadon & Herz 2004 Sleep 27:402 | verified | 92% in stage 1; peppermint ineffective in stages 2, 4 and REM; pyridine 45%, 0% and one third; tones at least 75%. |
| Ponce-Alvarez et al. 2018 Neuron 100:1446 | verified | Both quotes are in the abstract; gap junctions maintain criticality. |

### Corrections to the summary (and to this file; inline annotations above)
1. "Lin et al. rank the ellipsoid and fan-shaped bodies highest on those measures [density/strength]". The text ranks
   them only on fraction of internal connection weight. The "EB highest for reciprocity" in bottom line 7 is also not
   in the paper.
2. "The only published criticality simulation on a fly connectome (Ódor 2022)" contradicts Ódor, Papp & Deco 2025.
   That paper is now published in Chaos Solitons Fractals (Dec 2026, doi:10.1016/j.chaos.2026.119150). Ódor 2022's
   no-GP result is also hedged ("or maybe a very weak ones"; "hard to decide if there is a narrow GP" on the
   homeostatic graph).
3. Ódor 2025: the abstract does not mention a Griffiths phase.
4. "No empirical result ties the AL or OB to ... a sleep/anesthesia transition" is too strong. The OB's dynamics change
   with state (Mori 2013, Kato 2012, Średniawa 2021, Fontanini & Bower 2005, Manabe 2011). What is unsupported is the
   first relay *driving* or *gating* the state transition or consciousness level.
5. "The one self-sustaining circuit known in the fly is the central complex" is an overstatement. Counterexample:
   pCd persistent activity (Jung et al. 2020 Neuron 105:322).
6. Small caveats:
   - Olsen 2007: the palps were intact, so the AL was not fully deafferented.
   - Hunt 2019 used a subanesthetic dose.
   - Kinouchi & Copelli is general excitable-network theory.
   - "Confirmed against Europe PMC" holds for 55 of 58 DOIs; the other 3, and Ódor 2025, were confirmed through
     Crossref and arXiv.
