# Literature for phase diagram v2 (pd3): modular phases, transition order, finite-size rounding

Compiled 2026-11-09 for workflow wf_98fb990d-4fb (literature step). Scope: the user's questions
(1) can we recover a de Candia et al. 2026-style phase diagram (SL / SH / Bm, second-order lines with MF-DP avalanches,
first-order dashed lines) for the stochastic Wilson-Cowan (SWC) model on the maleCNS connectome, and
(2) is the steep but smooth crossover a second-order transition; plus support for (3) the lifetime heat map and the
finite-size check on nulls.

**Access level of each entry.** [F] = full text read. [P] = key sections of the full text read, from a downloaded PDF or
XML. [A] = abstract or metadata only, via search results or the abstract page. Every number below is quoted from the
cited source at that access level, or from our own files (named where used). Nothing is extrapolated. Where a search
summary could not be checked against the paper, the entry says so.

---------------------------------------------------------------------------------------------------------------------

## 0. Synthesis: what the literature implies for our questions

### 0.1 What carries over from de Candia et al. 2026, and what does not
- **The exact module reduction needs identical input within a module.** de Candia's M-module equations are exact
  because "the connections depend only on the pre-synaptic neuron, so that all the neurons of a given module receive the
  same input". The same property rules out Hopf oscillations in their model (all Jacobian eigenvalues real).
  - In graph terms this is an *external equitable partition*: every node in block k receives the same total weight
    from block l (Schaub et al. 2016, §d).
  - A real connectome module is not equitable. So the symmetric SL/SH/Bm structure is *explicitly broken* from the start.
- **In a heterogeneous network the SL boundary is set by the strongest module.** Silence goes unstable where the most
  strongly self-coupled module goes unstable. In our model that is g_lin = 0.1632, the Perron root of the excitatory
  branching operator, carried by the antennal-lobe (AL) module. Sources: our CLAUDE.md and
  results/analysis2/phase_complete.md.
  - So **our g_lin is the analogue of de Candia's SL–B1 critical line**, with "B1" = AL active and the rest silent. That
    is our localized (L) regime.
  - Further modules switch on one after another at their own local thresholds (our P3 "islands"). They do not switch on
    together at one symmetric line. This is the network-localization picture of Goltsev et al. 2012 (§c).
- **Bm states with exactly silent modules need net non-positive input into every silent neuron.** de Candia's Eq. 9b
  requires f(m w1 Σ) = 0, i.e. w1 ≤ 0.
  - In our per-neuron model, any excitatory projection from an active module onto a neuron of a "silent" module drives
    that neuron (rate β tanh(s)_+ > 0). An inhibitory input does nothing to a silent neuron unless excitation coincides
    (our theory.html, §3).
  - So exact Bm states require targets that receive no net excitation. Generically we expect *B1-like* states: localized
    activity with spill-over. This is a qualitative difference to test. Do the spill-over neurons stay at O(h)-like
    occupancy (B-like), or do they get recruited (SH-like)?
- **Sources of first-order (dashed) lines.** The literature gives four mechanisms:
  1. A convex region of the activation function. In SWC, f = β tanh(s + γ s²) is continuous for γ = 0 (no hysteresis)
     and discontinuous with hysteresis at γ = 4 (Alvankar Golpayegan & de Candia 2023). Our f = β tanh(s)_+ is concave,
     so a single homogeneous population cannot give a first-order line.
  2. E/I populations receiving different inputs. This gives saddle-node lines (T3/T4) and tricritical (TDP) points in
     SWC (Piuvezam et al. 2023).
  3. Inhibitory nodes in a branching process, which give hysteresis (Corral López et al. 2022).
  4. Net-inhibitory coupling between modules (de Candia et al. 2026). There the dashed lines mark Bm fixed points losing
     stability discontinuously.

  In our model, candidates for first-order lines are inhibition-mediated competition between modules (r > 0) and E/I
  structure inside modules. The test is hysteresis sweeps in g or r at fixed N, as in Alvankar Golpayegan & de Candia
  2023, Fig. 8.
- **Avalanche-shape signature.** Symmetric or right-skewed shapes at the SL–SH line; left-skewed at the SL–B1 line for
  intermediate durations, from cooperative recruitment followed by inter-module competition (de Candia et al. 2026).
  Network structure can also skew shapes:
  - heavy-tailed degrees (Gleeson & Durrett 2017);
  - a large human connectome (Ódor 2016, as cited by de Candia);
  - inhibitory hubs with plasticity (Zaccariello et al. 2025, as cited by de Candia).

  A concrete prediction to test: what is the avalanche shape for single seeds at g ≈ g_lin in the AL module, and does
  it change with r?

### 0.2 Is the steep but smooth crossover a second-order transition?
- **Deterministic (mean-field) level: yes.** With a concave tanh_+ rate the onset of a self-coupled population is a
  transcritical bifurcation. The order parameter is linear in the distance (β = 1), fluctuations diverge with γ = 1, and
  the class is MF-DP.
  - Sources: de Candia et al. 2021 and 2026; Piuvezam et al. 2023 (T1, field exponent δ_h = 2, i.e. ρ ∝ h^{1/2} at the
    critical point); Alvankar Golpayegan & de Candia 2023 (γ = 0: continuous, no hysteresis).
  - Our QMF has the same structure at g_lin (theory.html §4: transcritical, ε* ∝ R − 1).
- **Stochastic level, finite module: no sharp transition, only a rounded crossover.** For a logistic birth-death
  population of size N, Doering, Sargsyan & Sander 2005 give the mean extinction time:
  - below threshold: ≈ ln N / (1 − Λ);
  - at threshold: ≈ 1.5687 √(π/2) √N;
  - above threshold: ≈ [Λ/(Λ−1)²] √(2π/N) · exp[N(ln Λ − 1 + 1/Λ)].

  The exponent is exactly the Φ(R) = ln R − 1 + 1/R of our P2 (theory.html), with N → K_eff. So in the lifetime the
  crossover has width ∝ K^{-1/2}, as in our P2 window √(2 ln(αT)/K_eff).
  - K is fixed by the AL module. It is 526 neurons in all; the theory's K_eff is 58.5 at r = 0 and 16.8 at r = 1, and the
    simulations need 75.8 and 36.6 (phase_complete.md). So **the crossover cannot sharpen as the whole brain grows.**
- **Whole-brain thermodynamic sense: localized activity has no strict threshold.** Goltsev et al. 2012 [F]: "if the
  principal eigenvector … is localized, then at the infection rate λ right above the threshold 1/Λ1, the disease is
  mainly localized on a finite number of vertices. Importantly, a strict epidemic threshold in this case is actually
  absent, and a real epidemic affecting a finite fraction of vertices occurs after a smooth crossover."
  - Ferreira et al. 2016 (PRE 94:062305) show the localized states are *metastable*: their longevity and scale vary
    smoothly from localized to delocalized near the threshold, with no sharp transition. This is our slow window in SIS
    language.
  - St-Onge et al. 2021 describe the same "mesoscopic localization" with a *smeared* transition, for groups.
  - Pastor-Satorras & Castellano 2016 and 2018 discuss localization of the principal eigenvector in real networks,
    which makes the steady-state density possibly non-extensive.
- **Griffiths phase vs one rare region.** A Griffiths phase (power-law decay with continuously varying exponents) needs
  a *broad distribution* of rare-region sizes: the exponentially rare large regions have exponentially long lifetimes
  (Vojta 2006, class B; Muñoz et al. 2010; Moretti & Muñoz 2013; Cota et al. 2018).
  - One dominant rare region gives an exponential lifetime distribution, which is what we measured (CLAUDE.md, "Not a
    Griffiths phase in this model").
- **How to show a second-order transition in one finite network.** These are the literature's operational tests:
  1. Lifetime scaling with effective size at the threshold (τ ∝ √K) vs above it (ln τ ∝ K) (Doering et al. 2005). This
     needs a family of sizes: configuration nulls of several N, replicated or subsampled modules, or de Candia's M
     identical modules.
  2. Field scaling at the critical point, ρ ∝ h^{1/2} (δ_h = 2, MF-DP), vs δ_h = 3 at a tricritical point and
     bistability or hysteresis at a first-order point (Piuvezam et al. 2023, Fig. 5).
  3. The quasi-stationary (QS) distribution and the QS susceptibility χ = N(⟨ρ²⟩ − ⟨ρ⟩²)/⟨ρ⟩: its peak locates the
     size-dependent threshold, and its height scales as a power of N (Ferreira et al. 2012; de Oliveira & Dickman 2005).
     A bimodal QS distribution plus hysteresis indicates a first-order transition (Alvankar Golpayegan & de Candia 2023).
  4. Avalanche exponents (size 3/2, duration 2, ⟨S⟩(T) ~ T²), with cutoffs that scale with system size (de Candia et al.
     2021), plus shape collapse (Friedman et al. 2012; Sethna et al. 2001).
  5. A caveat: power laws alone can arise without criticality (Touboul & Destexhe 2017; Priesemann & Shriki 2018; Benayoun
     et al. 2010). Use 1–3, not 4 alone.
- **Answer the literature supports.** The onset at g_lin is a mean-field second-order (transcritical, MF-DP) bifurcation
  of the AL module. In the actual finite stochastic system it is a finite-K-rounded crossover whose width does not shrink
  with brain size. In the whole-brain thermodynamic sense it is a localization phenomenon with no strict threshold
  (Goltsev et al. 2012). It would be a true critical point only in a family where the module itself grows.
  - The configuration null, whose leading mode is delocalized, should instead sharpen with N, as the QMF threshold does on
    delocalized networks (Ferreira et al. 2012; Castellano & Pastor-Satorras 2020).

### 0.3 Links to wakefulness, sleep and anesthesia (only where a paper makes them)
- Mammals:
  - Wakefulness progressively pushes cortex away from criticality and sleep restores it (Xu et al. 2024).
  - Avalanche sizes differ by state: slow-wave sleep > wake > REM, consistent with a slightly subcritical regime
    (Priesemann et al. 2013).
  - Unconsciousness under propofol or anesthesia is a departure from critical dynamics (Tagliazucchi et al. 2016; Toker et
    al. 2022).
  - Criticality appears at an intermediate cortical state between synchronized and desynchronized (Fontenele et al. 2019).
  - Review: Hengen & Shew 2025.
- A Griffiths (extended-critical) regime as an individual-difference / cognition correlate was proposed for human
  connectome models (Wu et al. 2025 preprint; Moretti & Muñoz 2013).
- Flies:
  - Integrated-information structure collapses under isoflurane in Drosophila (Leung et al. 2021).
  - **No paper was found that measures avalanches, criticality or a phase diagram in Drosophila brain activity** (three
    searches; negative result).
  - The only fly-connectome criticality papers found are Kuramoto models on fly connectomes (Ódor et al. 2022, 2025).
    The first found a mean-field-like synchronization transition and attributed it to a lack of modular structure in the
    21,662-node fly graph.

---------------------------------------------------------------------------------------------------------------------

## (a) Stochastic Wilson-Cowan model and close relatives

**de Candia A, Conte D, Golpayegan HA, Scarpetta S (2026). Symmetry breaking and avalanche shapes in modular neural
networks. Front. Comput. Neurosci. 20:1744991.** doi:10.3389/fncom.2026.1744991 [F] (refs/decandia2026_symmetry_breaking_modular.txt)

- **Model.** SWC with binary units; decay α; activation β tanh(s)_+, identical to ours. M equal modules with N_E
  excitatory and N_I inhibitory units each. Intra-module and inter-module E/I strengths are w^E_0, w^I_0, w^E_1, w^I_1.
  Fixed points depend only on the imbalances w0 = w^E_0 − w^I_0 and w1 = w^E_1 − w^I_1. Avalanche statistics also depend
  on the sums w^S = w^E + w^I.
- **Phases.**
  - SL: the absorbing quiescent state is attractive.
  - SH: all modules active.
  - Bm: m modules active, the rest silent. These exist only for w1 ≤ 0 and w0 ≥ α/β − (m−1)w1, with C(M,m) copies.
- **Critical lines (solid).** w0 = α/β for w1 ≤ 0 (SL–B1), and w0 = α/β − (M−1)w1 for w1 ≥ 0 (SL–SH). On these one
  Jacobian eigenvalue is zero. The order parameter vanishes continuously with exponent 1, and fluctuations diverge with
  exponent 1, as in MF-DP.
- **First-order lines (dashed).** Where fixed points lose stability discontinuously.
- **No Hopf bifurcation.** All neurons of a module get the same input, so the Jacobian eigenvalues are real.
- **Simulations.** Gillespie; M = 3; α = β = 0.1 ms⁻¹; N_E = N_I = 10⁵; h = 0, restarting from one random active neuron.
  Points A (w0, w1) = (1, 0), B (0.8, 0.1), C (1, −0.15).
- **Avalanche shapes.**
  - Symmetric (purely excitatory) or right-skewed (strong E and I) in a single module, and along the SL–SH line.
  - Left-skewed for 100 < T < 1000 ms along SL–B1 at large w^S_1. Skewness reaches almost 0.4 at w^S_1 = 10 and T ≈ 400
    ms, and grows with M.
  - Mechanism: the inter-module input stays positive for about the first 50 ms, while excitatory activity leads, so all
    modules are recruited cooperatively. After about 100–150 ms inter-module inhibition selects one winner.
- **Speculation in the paper.** A hierarchical modular extension would shift the transition, "controlled by the relevant
  eigenmodes of the block-structured coupling matrix", and make the symmetry breaking multi-stage.
- **Relevance.** This is the target diagram. Its exact reduction assumes equitable modules and symmetric permutation, and
  neither holds in the connectome. Its central mechanism (net inhibition between modules gives Bm states and left-skewed
  avalanches) is testable in our (g, r) plane.

**Benayoun M, Cowan JD, van Drongelen W, Wallace E (2010). Avalanches in a stochastic model of spiking neurons. PLoS
Comput. Biol. 6:e1000846.** doi:10.1371/journal.pcbi.1000846 [A]

- The original SWC model (two-state Markov neurons, Gillespie simulation). In the large-N limit it obeys the
  Wilson-Cowan equations, which do not themselves produce avalanches.
- Avalanches arise when E and I are balanced so that "small fluctuations in the difference of population activities feed
  forward into large fluctuations in the sum": weakly stable, functionally feedforward (non-normal) dynamics. Hence their
  claim that "a network need not be 'critical' for the production of avalanches".
- **Relevance.** This is our model's ancestor. It is also the canonical warning that avalanches alone do not prove
  criticality, so any MF-DP claim at g_lin needs exponent, cutoff and field tests.

**de Candia A, Sarracino A, Apicella I, de Arcangelis L (2021). Critical behaviour of the stochastic Wilson-Cowan model.
PLoS Comput. Biol. 17:e1008884.** doi:10.1371/journal.pcbi.1008884 [P] (Europe PMC full text)

- With identical inputs to E and I, the fixed points depend on w0 = w_E − w_I. The critical point is w0c = α/β, a
  transcritical bifurcation of the quiescent fixed point.
- Linear-noise (van Kampen) expansion: the feed-forward term w_ff couples Σ and Δ. Relaxation times τ1 and τ2 diverge
  only for h → 0 at w0c.
- **Finite-size scaling.** At the critical point (their parameters w0 = 0.1, h = 10⁻⁶) the exponential cutoff of the size
  distribution "scales with N" (N = 10³–10⁷). The maximum correlation time saturates with N to the linear-theory value at
  criticality and decreases away from it.
- **Exponents.** Size 3/2 and duration 2 (MF branching). The fitted ⟨S⟩(T) exponent is 2.1 ± 0.05 (50% inhibitory) and
  2.02 ± 0.05 (20%), on T ∈ [80, 200]. Shape collapse holds, with symmetric shapes in that paper.
- **Relevance.** The protocol for showing that an SWC transition is "second-order-like" in finite systems: cutoff
  scaling with N, diverging relaxation time, MF-DP exponents. These are the tests to apply to the AL module at g_lin.

**Alvankar Golpayegan H, de Candia A (2023). Bistability and criticality in the stochastic Wilson-Cowan model. Phys. Rev.
E 107:034404.** doi:10.1103/PhysRevE.107.034404 [P] (downloaded PDF)

- Activation f(s) = β tanh(s + γ s²) for s > 0.
  - γ = 0: a continuous transition at w0 = α/β. A hysteresis sweep (N = 10⁵, w0 from 0.7 to 1.1 and back) shows no
    history dependence.
  - Larger γ (γ = 4 shown): an active solution appears discontinuously, with bistability and hysteresis.
- Scale-free avalanches persist on the spinodal line where the low-activity state goes unstable, with a "bump" of huge
  avalanches. For moderate N the system switches between the low and high states; their Fig. 7 shows lifetimes vs γ
  and N.
- **Relevance.** Direct evidence that the *pure tanh* SWC has a continuous (second-order) onset. First-order behaviour
  needs a convex region of f, or other structure. Their hysteresis protocol and bimodal rate distribution are the
  first-order diagnostics to reuse.

**Piuvezam HC, Marin B, Copelli M, Muñoz MA (2023). Unconventional criticality, scaling breakdown, and diverse
universality classes in the Wilson-Cowan model of neural dynamics. Phys. Rev. E 108:034110.**
doi:10.1103/PhysRevE.108.034110; arXiv:2301.06839 [P] (downloaded PDF)

- SWC with Φ(s) = tanh(s)_+, as in ours, but with four free couplings w_EE, w_EI, w_IE, w_II, so E and I populations
  receive different inputs.
- Piecewise-smooth dynamics (switching manifolds s_E = 0, s_I = 0). There are 8 quiescent-to-active transition types:
  - T1, transcritical: DP, β = 1, δ_h = 2.
  - T2, saddle-node-transcritical: tricritical DP, β = 1/2, δ_h = 3.
  - T3, saddle-node: discontinuous with bistability and hysteresis.
  - T4: discontinuous from an "excitable quiescent" phase.
  - T5, Hopf-tricritical: δ_h = 2, with anomalous P(S|T) collapse at γ = 4.
  - T6–T8: continuous at h = 0 but bistable once h > 0.
- A Hopf line separates standard quiescence from an excitable (non-normal, reactive) quiescent phase.
- Their T1 matches de Candia et al. 2021 when w_EE = w_IE and w_EI = w_II.
- **Relevance.** The menu of transition types that E/I structure can produce. The field exponent gives a clean
  single-network test of second order vs tricritical vs first order: measure ρ(h) at g_lin and check δ_h = 2.

**Apicella I, Scarpetta S, de Arcangelis L, Sarracino A, de Candia A (2022). Power spectrum and critical exponents in the
2D stochastic Wilson-Cowan model. Sci. Rep. 12:21870.** doi:10.1038/s41598-022-26392-8 [A]

- SWC on a 2-D lattice with connections decaying exponentially with distance. The power-spectrum exponent and the ⟨S⟩(T)
  exponent are "around 1 and 1.3", against ≈ 2 in mean field.
- The avalanche size and duration exponents decrease below their MF values of 1.5 and 2. The abstract gives the values
  as formulas; a search summary reported 1.29 ± 0.01 and 1.37 ± 0.01, not checked against the paper.
- Suggests a finite-dimensional universality class.
- **Relevance.** Our model on a sparse, spatially embedded graph may deviate from MF-DP exponents. Graph and spectral
  dimension matter: Ódor et al. 2025 report spectral dimension ≈ 2 for a full fly connectome.

**Mariani B, Nicoletti G, Bisio M, Maschietto M, Vassanelli S, Suweis S (2022). Disentangling the critical signatures of
neural activity. Sci. Rep. 12:10770.** doi:10.1038/s41598-022-13686-0 [A]

- In rat barrel cortex MEA data, power-law avalanches and scale-free spatial correlations have different origins.
  Avalanches can come from shared external stochastic modulation, and correlations from internal interactions.
- **Relevance.** A caution for the driven plane (h > 0): avalanche-like statistics can be produced by the drive.

**Corral López R, Buendía V, Muñoz MA (2022). Excitatory-inhibitory branching process: a parsimonious view of cortical
asynchronous states, excitability, and criticality. Phys. Rev. Research 4:L042027.** doi:10.1103/PhysRevResearch.4.L042027;
arXiv:2203.16374 [P]

- Adding inhibitory nodes to the branching process gives an intermediate "asynchronous" phase between quiescence and
  saturation.
- In the inhibition-dominated case, the mean-field phase diagram has a continuous line, a discontinuous line with
  bistability, and a tricritical point. It also shows collective excitability, hysteresis, tilted (non-parabolic)
  avalanche shapes and partial synchronization.
- Strong non-normality, quantified by a Henrici index, marks the asynchronous phase.
- **Relevance.** A minimal model of how inhibition (our r) can turn a DP line into a first-order or tricritical one, and
  skew avalanche shapes. Useful to interpret r-dependence.

**di Santo S, Burioni R, Vezzani A, Muñoz MA (2016). Self-organized bistability associated with first-order phase
transitions. Phys. Rev. Lett. 116:240601.** doi:10.1103/PhysRevLett.116.240601; arXiv:1605.05161 [A]

**Buendía V, di Santo S, Villegas P, Burioni R, Muñoz MA (2020). Self-organized bistability and its possible relevance
for brain dynamics. Phys. Rev. Research 2:013318.** doi:10.1103/PhysRevResearch.2.013318 [A]

- Self-organized bistability: systems self-tune to the edge of a *first-order* transition. This gives scale-invariant
  avalanches coexisting with huge anomalous ones, bimodal activity, and "self-organized collective oscillations".
- In the infinitely slow synaptic limit SOB and cortical models coincide, but the required time-scale separation looks
  biologically implausible.
- **Relevance.** If our r-sweeps reveal first-order lines, scale-free avalanches near their spinodals (as in Alvankar
  Golpayegan 2023) are expected. They must not be read as evidence for a second-order point.

**Girardi-Schappo M, Brochini L, Costa AA, Carvalho TTA, Kinouchi O (2020). Synaptic balance due to homeostatically
self-organized quasicritical dynamics. Phys. Rev. Research 2:012042(R).** doi:10.1103/PhysRevResearch.2.012042 [A]

- E/I mean-field network with a critical point at synaptic balance. Inhibitory short-term depression plus threshold
  adaptation self-organizes the system near it (quasicriticality), producing E/I current cancellation and asynchronous
  irregular firing. The follow-up is J. Phys. Complex. 2:045001 (2021), per a search result.
- de Candia 2026 cites it for "criticality in E–I networks is not uniquely tied to a canonical balanced condition".
- **Relevance.** Adaptation and homeostasis can hold a network near an absorbing transition. A possible mechanism to put
  the fly network near g_lin, not explored by us.

**Kinouchi O, Copelli M (2006). Optimal dynamical range of excitable networks at criticality. Nat. Phys. 2:348–351.**
doi:10.1038/nphys289 [A]

**Larremore DB, Shew WL, Restrepo JG (2011). Predicting criticality and dynamic range in complex networks: effects of
topology. Phys. Rev. Lett. 106:058101.** doi:10.1103/PhysRevLett.106.058101; arXiv:1008.0022 [P]

- Kinouchi & Copelli: excitable networks maximize sensitivity and dynamic range at the critical point.
- Larremore et al.: the critical point is set by the largest eigenvalue of the weighted adjacency matrix (λ = 1), and
  dynamic range peaks there. Homogeneous networks reach a higher dynamic range than heterogeneous ones.
- **Relevance.** The spectral-threshold principle behind our g_lin (Perron root of the branching operator = α_eff).
  Their nonlinear analysis uses further spectral properties beyond λ, which fits our K_eff built from the left and right
  Perron vectors.

**Larremore DB, Shew WL, Ott E, Sorrentino F, Restrepo JG (2014). Inhibition causes ceaseless dynamics in networks of
excitable nodes. Phys. Rev. Lett. 112:138103.** doi:10.1103/PhysRevLett.112.138103; arXiv:1307.7658 [P]

- Inhibitory nodes that reduce neighbours' activation probability make activity self-sustaining. The activity-dependent
  "branching function" Λ(S) exceeds 1 at small S. Inside the self-sustaining region a critical line carries avalanches
  embedded in ceaseless activity.
- **Relevance.** A contrasting inhibition rule. In our additive-input, rectified model an inhibitory spike onto a silent
  neuron does nothing (theory.html §3), so this mechanism is absent. Worth stating when comparing.

**Wallace E, Benayoun M, van Drongelen W, Cowan JD (2011). Emergent oscillations in networks of stochastic spiking
neurons. PLoS ONE 6:e14804.** doi:10.1371/journal.pone.0014804 [A]

- SWC E–I networks give population oscillations without single-neuron oscillators: noisy limit cycles if the
  deterministic system has a limit cycle, quasi-cycles if it has a weakly stable focus.
- **Relevance.** de Candia's "no Hopf" result relies on identical inputs within a module. Our heterogeneous E–I
  connectome has complex signed eigenvalues (STATE.md: Re 1090 ± 1317i), so quasi-cycles are possible. An oscillatory
  phase should be checked in the v2 diagram, not assumed away.

---------------------------------------------------------------------------------------------------------------------

## (b) Criticality and phase structure on modular, hierarchical and real connectomes

**Moretti P, Muñoz MA (2013). Griffiths phases and the stretching of criticality in brain networks. Nat. Commun. 4:2521.**
doi:10.1038/ncomms3521; arXiv:1308.6661 [A]

- In hierarchical-modular networks, and in the human and C. elegans connectomes, the critical point is replaced by an
  extended critical-like region (a Griffiths phase). Their argument is that this helps self-organization find critical
  attributes.
- **Relevance.** The project's starting hypothesis. Our data show one dominant rare region (exponential lifetimes), not
  the broad rare-region distribution a Griffiths phase needs.

**Muñoz MA, Juhász R, Castellano C, Ódor G (2010). Griffiths phases on complex networks. Phys. Rev. Lett. 105:128701.**
doi:10.1103/PhysRevLett.105.128701; arXiv:1009.0395 [A]

- Contact process with quenched disorder on networks gives Griffiths phases and slow relaxation. Topological
  heterogeneity alone suffices in networks with *finite topological dimension*.
- **Relevance.** Finite topological dimension is the condition. Ódor et al. 2022 report a high topological dimension for
  a fly connectome, which disfavours Griffiths phases there.

**Ódor G, Dickman R, Ódor G (2015). Griffiths phases and localization in hierarchical modular networks. Sci. Rep.
5:14451.** doi:10.1038/srep14451; arXiv:1503.06307 [A]

- Contact and threshold models on Kaiser–Hilgetag hierarchical-modular networks. A Griffiths phase appears when
  connection probabilities are reduced toward the percolation threshold.
- Quenched mean field shows steady-state localization, which can occur even in highly connected small-world networks
  "in case of link disorder".
- **Relevance.** Weight (link) disorder alone can localize, as in our finding that weights, not degrees, localize the
  leading mode (STATE.md: the binary graph is delocalized).

**Cota W, Ódor G, Ferreira SC (2018). Griffiths phases in infinite-dimensional, non-hierarchical modular networks. Sci.
Rep. 8:9144.** doi:10.1038/s41598-018-27506-x; arXiv:1801.06406 [A]

- Loosely coupled modules act as effective rare regions, giving extended parameter regions with continuously varying
  dynamical exponents. These hold "for single network realizations" and survive finite-size analysis. Avalanche sizes
  have robust power-law tails.
- **Relevance.** The key requirement is *many* loosely coupled modules with a spread of lifetimes. Our spectrum has 400
  localized modes (CLAUDE.md), but the dynamics are dominated by one (the AL). Whether other modules can hold activity
  at larger g, especially with drive, decides whether a Griffiths-like regime exists anywhere in the (g, r, h) space.

**Ódor G (2013). Spectral analysis and slow spreading dynamics on complex networks. Phys. Rev. E 88:032109.**
arXiv:1306.3401 [A]

**Ódor G (2014). Localization transition, Lifschitz tails and rare-region effects in network models.** arXiv:1405.7622
(journal reference not verified; a talk listing suggests PRE 89:042102) [A]

- The QMF spectral decomposition of the weighted adjacency matrix predicts rare-region effects and slow dynamics. The
  2014 paper studies IPR distributions and Lifshitz tails of eigenvalues as predictors of the SIS Griffiths phase.
- **Relevance.** The spectral method we use (top-400 eigenpairs, IPR). Cite for the Lifshitz-tail argument.

**Ódor G (2016). Critical dynamics on a large human Open Connectome network. Phys. Rev. E 94:062411.**
doi:10.1103/PhysRevE.94.062411; arXiv:1604.02127 [A]

- Threshold models on an 836,733-node human connectome give non-universal power-law avalanches over an extended control
  region. If node input sensitivity is equalized, simple threshold models show a sharp discontinuous transition.
- de Candia 2026 cites it for a left-skewed avalanche profile on a modular human connectome.
- **Relevance.** On a real connectome, normalizing inputs changes the transition type. This parallels our in-norm vs
  raw-norm contrast (STATE.md).

**Ódor G, Deco G, Kelling J (2022). Differences in the critical dynamics underlying the human and fruit-fly connectome.
Phys. Rev. Research 4:023057.** doi:10.1103/PhysRevResearch.4.023057; arXiv:2201.11084 [A]

- Kuramoto model on a 21,662-node fruit-fly connectome (dataset not named on the abstract page). The synchronization
  transition is "mean-field like", with a transition region wider than in random graphs but much narrower than in the
  KKI-18 human connectome.
- The authors attribute this to the fly graph being "less heterogeneous, lacking modular structure and exhibiting high
  topological dimension".
- **Relevance.** The only prior criticality study on a fly connectome. Our weighted maleCNS graph *does* have a strongly
  localized weighted module (AL), while its binary graph is delocalized (STATE.md). That is consistent with their
  "no modular structure" being a property of the unweighted or normalized graph.

**Ódor G, Papp I, Deco G (2025/2026). Fluctuation-dissipation of the Kuramoto model on fruit-fly connectomes.**
arXiv:2503.20708 (accepted in Chaos, Solitons & Fractals per arXiv) [A]

- Kuramoto on a full fly connectome of ≈10⁵ neurons: raw weights, homeostatic weights, and random inhibitory edges.
  The graph dimension is ≈ 6, but the spectral dimension relevant to synchronization is ≈ 2, below the upper critical
  dimension, implying non-mean-field fluctuation effects.
- **Relevance.** A low spectral dimension can make exponents deviate from MF-DP on fly connectomes. Measured exponents
  at g_lin should be compared with both MF-DP and finite-dimensional values.

**Rusch FR, Kinouchi O, Roque AC (2025). Influence of topology on the critical behavior of hierarchical modular neuronal
networks. Commun. Phys. 8:168.** doi:10.1038/s42005-025-02074-5 [A]

- Stochastic adaptive spiking neurons on hierarchical modular networks. Sparse modular architectures preserve
  criticality more robustly than fully connected ones. Inhomogeneities produce quasicritical, Griffiths-like phases.
- **Relevance.** Context. Their heterogeneity spreads rather than isolates; contrast with our single dominant module.

**Russo R, Herrmann HJ, de Arcangelis L (2014). Brain modularity controls the critical behavior of spontaneous activity.
Sci. Rep. 4:4312.** doi:10.1038/srep04312; arXiv:1403.6339 [A]

- On a modular network, avalanche sizes are power-law only up to the size of one module, then decay exponentially.
  Criticality returns when inter-module links make the structure more random.
- **Relevance.** Directly predicts that our avalanche cutoff at g_lin is set by the AL module, not by N. This is the
  same finite-K argument as §0.2.

**Kaiser M, Hilgetag CC (2010). Optimal hierarchical modular topologies for producing limited sustained activation of
neural networks. Front. Neuroinform. 4:8.** doi:10.3389/fninf.2010.00008; arXiv:1003.3081 [A]

- Hierarchical modular networks support "limited sustained activity" (activity that neither dies nor takes over the
  network) over a large parameter range, and random or small-world networks do not. Optimal configurations have
  intermediate numbers of levels and many modules.
- **Relevance.** Our localized-active (L) phase is "limited sustained activity" in their terms. This is the
  neuroscience precedent for calling it a phase of the modular network.

**Angiolelli M, Scarpetta S, Sorrentino P, et al. (2025). Role of criticality in the structure-function relationship in
the human brain. Phys. Rev. Research 7:043153.** doi:10.1103/PhysRevResearch.7.043153 (citation as in de Candia 2026) [A]

- A modular network of LIF neurons (one group per cortical region; *not* SWC, per the conference abstracts found) has an
  uncorrelated low-rate state, a strongly correlated state, and an "extended critical region" between them. That region
  maximizes the structure–function correlation and reproduces MEG avalanche sequence repertoires.
- **Relevance.** A modular-brain precedent for an extended intermediate regime between low and high activity.

**Myrov V, et al. (2024). Hierarchical whole-brain modeling of critical synchronization dynamics in human brains.
bioRxiv 2024.05.08.593146** (per a Helsinki portal, later peer-reviewed in 2026) [A]

- A hierarchical Kuramoto model, with each area a population of oscillators, gives an extended critical regime with
  long-range temporal correlations. Human MEG resembles its subcritical side.
- **Relevance.** Context for modular "extended critical" regimes.

**Cirunay MT, Batac RC, Ódor G (2025). Learning and criticality in a self-organizing model of connectome growth. Sci. Rep.
15:31890.** doi:10.1038/s41598-025-16377-8; arXiv:2504.01565 [A]

- A sandpile with Hebbian learning on hierarchical modular networks gives power-law weight distributions and lognormal
  degrees and strengths.
- **Relevance.** Heavy-tailed weights (ours: median 2, max 2,591 synapses per edge; STATE.md) can come from growth plus
  learning. Weights are what localize our leading mode.

**Buendía V, Villegas P, Burioni R, Muñoz MA (2022). The broad edge of synchronisation: Griffiths effects and collective
phenomena in brain networks. Phil. Trans. R. Soc. A 380:20200424.** doi:10.1098/rsta.2020.0424; arXiv:2109.11783 [A]

- Synthetic hierarchical-modular and core-periphery networks give intermediate, flexible synchrony: "broad Griffiths-like
  phases". Hybrid-type transitions are another route to rich dynamics.
- **Relevance.** A review-level statement of the Griffiths picture we are testing against.

**Pretel J, Buendía V, Torres JJ, Muñoz MA (2024). From asynchronous states to Griffiths phases and back: structural
heterogeneity and homeostasis in excitatory-inhibitory networks. Phys. Rev. Research 6:023018.**
doi:10.1103/PhysRevResearch.6.023018; arXiv:2310.02369 [A]

- An E–I network with tunable heterogeneity shows both asynchronous states and Griffiths phases in one phase diagram.
  Structural disorder breaks local E/I balance, and plasticity or homeostasis restore it.
- **Relevance.** E–I plus heterogeneity can give an asynchronous phase, a Griffiths phase, or both, depending on
  homeostasis. Our model has no homeostasis, which is one reason we see a single-rare-region regime.

**Wu K, Chialvo DR, Zhou C, Yu L (2025). Optimal Griffiths phase in heterogeneous human brain networks.** arXiv:2512.03409
(preprint) [A]

- With Human Connectome Project data, structural modularity plus regional heterogeneity of excitability give a Griffiths
  phase spanning a wide range of global excitability, with an optimal point. An individual's position in it relates to
  functional connectivity and cognition.
- **Relevance.** The most recent claim linking a Griffiths regime to cognition. It needs heterogeneity of *local
  excitability* in addition to modularity, an ingredient we lack (uniform α, β).

**Haimovici A, Tagliazucchi E, Balenzuela P, Chialvo DR (2013). Brain organization into resting state networks emerges at
criticality on a model of the human connectome. Phys. Rev. Lett. 110:178101.** doi:10.1103/PhysRevLett.110.178101 [A]

- A Greenberg–Hastings model on a 998-node human connectome reproduces fMRI resting-state networks, diverging correlation
  length and anomalous correlation scaling only when tuned to criticality.
- **Relevance.** Classic connectome-criticality precedent.

**Zarepour Nasir Abadi M, Perotti JI, Billoni OV, Chialvo DR, Cannas SA (2019). Universal and nonuniversal neural dynamics
on small world connectomes: A finite-size scaling analysis. Phys. Rev. E 100:052138.** arXiv:1905.05280 [A]

- Greenberg–Hastings on connectome-like small-world networks. Depending on topology the transition is mean-field
  percolation, non-universal, or discontinuous (noncritical) into a percolated state. It uses finite-size scaling across
  network sizes.
- **Relevance.** Topology alone can make the activity transition first order. FSS across sizes is the standard method.

**Deco G, Jirsa VK (2012). Ongoing cortical activity at rest: criticality, multistability, and ghost attractors. J.
Neurosci. 32:3366–3375.** doi:10.1523/JNEUROSCI.2523-11.2012 [A]

**Golos M, Jirsa V, Daucé E (2015). Multistability in large scale models of brain activity. PLoS Comput. Biol.
11:e1004644.** doi:10.1371/journal.pcbi.1004644 [A]

- Deco & Jirsa: resting-state networks are structured noise around a stable low-activity equilibrium near a bifurcation,
  with latent "ghost" multistable attractors.
- Golos et al.: a connectome-coupled graded-response Hopfield model (up to ~1000 nodes) has large fixed-point attractor
  sets, counted by sampling initial conditions uniformly. The best empirical fit, and peak attractor entropy, sit at the
  edge of multistability.
- **Relevance.** Connectome-scale precedents for computing *many* fixed points (our Bm analogues) and placing the brain at
  the edge of multistability. Their uniform-sampling attractor count is a baseline method.

---------------------------------------------------------------------------------------------------------------------

## (c) Finite-size rounding, lifetimes, quasi-stationary methods, criticality tests

**Doering CR, Sargsyan KV, Sander LM (2005). Extinction times for birth-death processes: exact results, continuum
asymptotics, and the failure of the Fokker-Planck approximation. Multiscale Model. Simul. 3:283–299.**
doi:10.1137/030602800; arXiv:q-bio/0401016 [P]

- Exact mean extinction time and large-N asymptotics. For the logistic SIS model λ_n = Λn(1 − n/N), μ_n = n:
  - superthreshold: τ ≈ Λ/(Λ−1)² · √(2π/N) · e^{N(ln Λ − 1 + 1/Λ)};
  - threshold: τ ≈ 1.5687 · √(π/2) · √N;
  - subthreshold: τ ≈ ln N / (1 − Λ).
- The Fokker–Planck approximation is valid only near threshold.
- **Relevance.** The theoretical backbone of our P2 lifetime law: the exponent N·Φ(Λ) with Φ(R) = ln R − 1 + 1/R, and
  N → K_eff. It also gives the *scaling forms* to test finite-size rounding: ln τ ∝ K above, τ ∝ √K at, ln K below.

**Nåsell I (2001). Extinction and quasi-stationarity in the Verhulst logistic model. J. Theor. Biol. 211:11–27;** and
**Nåsell I (2011). Extinction and Quasi-stationarity in the Stochastic Logistic SIS Model (Springer)** [A]

- Three parameter regions: a long time to extinction with an approximately normal QS distribution; a short time with a
  truncated-geometric QS distribution; and a transition region. A later paper citing the 2011 book places the critical
  regime at |μ − λ| = O(N^{-1/2}).
- **Relevance.** The N^{-1/2} width of the critical window, i.e. why the AL crossover has a finite width ∝ K^{-1/2}.

**Dickman R, Vidigal R (2002). Quasi-stationary distributions for stochastic processes with an absorbing state. J. Phys.
A 35:1147–1166.** arXiv:cond-mat/0110557 [A]

**de Oliveira MM, Dickman R (2005). How to simulate the quasistationary state. Phys. Rev. E 71:016129.**
arXiv:cond-mat/0407797 [P]

- QS solutions for the contact process on a complete graph (mean-field DP) and related models. The QS simulation method:
  when the process is about to hit the absorbing state, jump to a surviving configuration drawn from its own history.
  The QS lifetime is τ = 1/p₁. It is about an order of magnitude more efficient at the critical point.
- **Relevance.** A practical way to measure the QS density, the moment ratio and the lifetime of the AL module across g
  without waiting for extinctions. It replaces the censored-MLE τ at large g.

**Ferreira SC, Castellano C, Pastor-Satorras R (2012). Epidemic thresholds of the susceptible-infected-susceptible model
on networks: a comparison of numerical and theoretical results. Phys. Rev. E 86:041125.** arXiv:1206.6728 [P]

- QS simulations locate the size-dependent threshold from the peak of χ = N(⟨ρ²⟩ − ⟨ρ⟩²)/⟨ρ⟩, which is more clear-cut
  than the standard N(⟨ρ²⟩ − ⟨ρ⟩²). The peak height scales as χ_max ~ N^{γ′/ν̄}.
- QMF predicts the threshold's N-scaling with the right exponent but often the wrong prefactor.
- **Relevance.** The susceptibility to use for the configuration-null finite-size check (peak position and height vs N).
  It also matches our finding that QMF gets the shape but not the prefactor (theory vs simulation, phase_complete.md).

**Boguñá M, Castellano C, Pastor-Satorras R (2013). Nature of the epidemic threshold for the susceptible-infected-
susceptible dynamics in networks. Phys. Rev. Lett. 111:068701.** arXiv:1305.4819 [A]

**Castellano C, Pastor-Satorras R (2020). Cumulative merging percolation and the epidemic transition of the
susceptible-infected-susceptible model in networks. Phys. Rev. X 10:011070.** doi:10.1103/PhysRevX.10.011070;
arXiv:1906.06300 [A]

- Boguñá et al.: a survival-time (lifespan) method locates thresholds where QS methods fail. Long-range reinfection
  among hubs makes the SIS threshold vanish for power-law networks.
- Castellano & Pastor-Satorras: SIS survival is a long-range percolation among hubs that reactivate each other through
  chains of low-degree nodes. The FSS exponent of the threshold takes a non-trivial value over a long pre-asymptotic
  range.
- **Relevance.** The general picture: isolated active regions (hubs; for us, modules) have finite lifetimes, and a true
  transition needs mutual reactivation among many of them. With one AL module (two lobes) there is little to merge, which
  is consistent with a rounded crossover. Mutual reactivation between L and R AL, or between AL and other islands under
  drive, is the mechanism to look for if a sharper collective transition exists.

**Goltsev AV, Dorogovtsev SN, Oliveira JG, Mendes JFF (2012). Localization and spreading of diseases in complex networks.
Phys. Rev. Lett. 109:128702.** doi:10.1103/PhysRevLett.109.128702; arXiv:1202.4411 [F] (5-page arXiv version read;
PRL reference from search indexing, APS page not opened)

- SIS in QMF on weighted and unweighted networks. If the principal eigenvector is localized (IPR finite as N → ∞), then
  just above λ_c = 1/Λ1 the infection sits on a finite number of vertices. Hubs, heavy edges and dense subgraphs are the
  centres of localization.
- "A strict epidemic threshold in this case is actually absent, and a real epidemic affecting a finite fraction of
  vertices occurs after a smooth crossover." If the eigenvector is delocalized, a finite fraction is infected just above
  λ_c.
- **Relevance.** The single most direct literature answer to Q2: localized leading mode (ours: AL, set by heavy weights)
  means a smooth crossover with no strict threshold. It predicts the contrast with the configuration null (delocalized),
  whose onset should sharpen with N.

**Ferreira RS, da Costa RA, Dorogovtsev SN, Mendes JFF (2016). Metastable localization of diseases in complex networks.
Phys. Rev. E 94:062305.** doi:10.1103/PhysRevE.94.062305; arXiv:1606.05624 [A]

- SIS on heterogeneous networks with hubs as localization centres. Below the epidemic threshold the localized states are
  *metastable*. Their longevity and scale change smoothly from localized to delocalized as the threshold is approached,
  with no sharp transition. On a random regular graph with one hub, the shape of the distribution of the number of
  infected nodes detects localization.
- **Relevance.** The closest published analogue of our slow window: a metastable, localized active state with finite,
  smoothly varying lifetime, around a crossover rather than a sharp transition. It suggests a diagnostic to add, the shape
  of the QS distribution of the active count in the AL module across g.

**Pastor-Satorras R, Castellano C (2016). Distinct types of eigenvector localization in networks. Sci. Rep. 6:18847.**
doi:10.1038/srep18847 [A]

**Pastor-Satorras R, Castellano C (2018). Eigenvector localization in real networks and its implications for epidemic
spreading. J. Stat. Phys. 173:1110–1123.** doi:10.1007/s10955-018-1970-8; arXiv:1801.05649 [A]

- On power-law networks, the principal eigenvector localizes on the largest hub (γ > 5/2) or on the max-index K-core
  (γ < 5/2). In real networks it sits on the hub star or on a dense subgraph.
- With localization, "the density of infected individuals in the steady state … might not be an extensive quantity,
  casting doubts on the validity of [QMF]".
- **Relevance.** Our AL localization is the dense-subgraph type (STATE.md: in the null, localized modes become single-hub
  stars). Their caution on QMF's extensivity matches our reading of the whole-brain density near g_lin.

**Martin T, Zhang X, Newman MEJ (2014). Localization and centrality in networks. Phys. Rev. E 90:052808.**
arXiv:1401.5093 [A]

- Hub-driven localization of the adjacency eigenvector comes from a hub reflecting centrality off its neighbours. The
  non-backtracking (Hashimoto) matrix removes hub localization, but not localization on dense subgraphs (per a later
  paper).
- **Relevance.** A check on whether AL localization is an echo artifact. It should survive the non-backtracking operator
  because it is a dense module. Our pair operator (theory.html) plays a similar role.

**Mata AS, Ferreira SC (2013). Pair quenched mean-field theory for the susceptible-infected-susceptible model on complex
networks. EPL 103:48003.** doi:10.1209/0295-5075/103/48003; arXiv:1305.5153 [A]

- Pair QMF gives the same threshold scaling as one-vertex QMF but is quantitatively much more accurate against QS
  simulations.
- **Relevance.** The literature counterpart of our pair operator P(g). In our data the pair variant moved boundaries up
  3–16%, away from the simulations (STATE.md, "Critic review of the complete phase diagram"), so the SIS result does not
  transfer directly.

**Vojta T (2006). Rare region effects at classical, quantum and nonequilibrium phase transitions. J. Phys. A 39:R143–R205.**
doi:10.1088/0305-4470/39/22/R01; arXiv:cond-mat/0602312 [P]

- Classification by rare-region dimension d_RR vs the lower critical dimension d_c⁻:
  - A (d_RR < d_c⁻): weak, exponential Griffiths effects; conventional critical point.
  - B (d_RR = d_c⁻): strong power-law Griffiths effects; infinite-randomness, activated scaling. This includes the
    contact process with point defects, counting time as a dimension.
  - C (d_RR > d_c⁻): rare regions order independently and the transition is *smeared*. In its tail, "statically ordered
    islands … coexist with the disordered bulk".
- Power-law Griffiths behaviour needs a density of rare regions that falls exponentially with volume, balanced against
  exponentially growing lifetimes.
- **Relevance.** Vocabulary for Q2. A single finite AL module gives activated (exponential-in-K) lifetimes but no
  distribution, so no power-law Griffiths phase. Since the module is finite it cannot order statically, so this is not a
  true smeared transition either. It is a finite-size-rounded crossover.

**St-Onge G, Thibeault V, Allard A, Dubé LJ, Hébert-Dufresne L (2021). Social confinement and mesoscopic localization of
epidemics on networks. Phys. Rev. Lett. 126:098301.** arXiv:2003.05924 [A]

- With group (higher-order) structure, infection can stay at low global prevalence while concentrated in large groups
  that self-sustain it. Global prevalence grows fastest well above the threshold, "a defining feature of smeared phase
  transitions".
- **Relevance.** A clean modern example of mesoscopic localization giving a smeared global transition. Analogous to the
  AL module carrying activity while the global density stays tiny.

**Hinrichsen H (2000). Non-equilibrium critical phenomena and phase transitions into absorbing states. Adv. Phys.
49:815–958.** doi:10.1080/00018730050198152; arXiv:cond-mat/0001070 [A]

**Lübeck S (2004). Universal scaling behavior of non-equilibrium phase transitions.** arXiv:cond-mat/0501259 (journal
venue not verified here) [A]

- Standard references for DP: universality, scaling with an external field, finite-size scaling, and universal scaling
  functions.
- **Relevance.** Background for the MF-DP exponents and the field-scaling test.

**Sethna JP, Dahmen KA, Myers CR (2001). Crackling noise. Nature 410:242–250.** doi:10.1038/35065675 (as cited by de
Candia 2026) [A]

**Friedman N, Ito S, Brinkman BAW, Shimono M, DeVille REL, Dahmen KA, Beggs JM, Butler TC (2012). Universal critical
dynamics in high resolution neuronal avalanche data. Phys. Rev. Lett. 108:208102.** doi:10.1103/PhysRevLett.108.208102 [A]

- Avalanche-shape collapse across durations as a criticality test, the crackling-noise scaling relation, and
  sub/supercritical samples in cultures.
- **Relevance.** The shape-collapse test to apply at g_lin, alongside de Candia's skewness-vs-duration measure.

**Gleeson JP, Durrett R (2017). Temporal profiles of avalanches on networks. Nat. Commun. 8:1227.**
doi:10.1038/s41467-017-01212-0 [A]

- Branching-process theory for mean avalanche shapes on configuration-model networks. At criticality shapes collapse,
  but heavy-tailed degree distributions give non-symmetric shapes.
- **Relevance.** Shape asymmetry at g_lin could come from weight or degree heterogeneity rather than de Candia's
  competition mechanism. A null comparison (weight-shuffled vs configuration) separates them.

**Touboul J, Destexhe A (2017). Power-law statistics and universal scaling in the absence of criticality. Phys. Rev. E
95:012413.** arXiv:1503.08033 [A]

**Priesemann V, Shriki O (2018). Can a time varying external drive give rise to apparent criticality in neural systems?
PLoS Comput. Biol. 14:e1006081.** doi:10.1371/journal.pcbi.1006081 [A]

- Power laws and even scaling collapse can arise from non-critical irregular regimes or from time-varying drive.
  Priesemann & Shriki: inhomogeneous Poisson drive gives approximate power laws with realistic cutoffs, and separating
  genuine from apparent criticality needs interventions.
- **Relevance.** Our criticality claims must rest on lifetime and field scaling and on an onset that is identified
  independently (g_lin), not on avalanche power laws alone.

**Wilting J, Priesemann V (2018). Inferring collective dynamical states from widely unobserved systems. Nat. Commun.
9:2325.** doi:10.1038/s41467-018-04725-4 [A]

**Zierenberg J, Wilting J, Priesemann V (2018). Homeostatic plasticity and external input shape neural network dynamics.
Phys. Rev. X 8:031018.** doi:10.1103/PhysRevX.8.031018 [A]

- Wilting & Priesemann: the multistep-regression estimator of the branching ratio is consistent under subsampling. In
  rat, cat and monkey it finds a "reverberating" state between asynchronous-irregular and critical.
- Zierenberg et al.: with rate homeostasis, input strength sets the dynamic state.
- **Relevance.** For the driven plane, the distance to criticality depends on h. They also give an estimator to compare
  model branching ratios with data, if fly data become available.

**Muñoz MA (2018). Colloquium: Criticality and dynamical scaling in living systems. Rev. Mod. Phys. 90:031001.**
doi:10.1103/RevModPhys.90.031001; arXiv:1712.04499 [A]

- A review of the criticality hypothesis, including Griffiths phases on networks and the evidence in neural systems.
- **Relevance.** General citation.

---------------------------------------------------------------------------------------------------------------------

## (d) Multistability, first-order transitions, symmetry breaking; threshold-linear networks; computing many fixed points

**Curto C, Geneson J, Morrison K (2019). Fixed points of competitive threshold-linear networks. Neural Comput.**
(DOI 10.1162/neco_a_01151 per a listing; volume and pages not verified). arXiv:1804.00794 [A]

- Characterizes all fixed points of competitive threshold-linear networks (TLNs) by *support*, using a sign condition per
  support and a domination criterion.
- For combinatorial TLNs (connectivity from a directed graph) it proves parameter-independent "graph rules" that read
  off the fixed points from the graph.
- **Relevance.** Our model linearized above threshold is a rectified network. Fixed points labelled by which modules are
  active are exactly "supports", the analogue of de Candia's Bm. Their sign conditions give an enumeration method that
  avoids blind root-finding when the module graph is small (≤ ~20 modules).

**Parmelee C, Moore S, Morrison K, Curto C (2022). Core motifs predict dynamic attractors in combinatorial threshold-linear
networks. PLoS ONE 17:e0264456.** doi:10.1371/journal.pone.0264456 [A]

- Fixed points supported on "core motifs" predict both static and dynamic attractors, checked on all 5-node networks.
- **Relevance.** A rule to choose which module supports to test for stability. Dynamic (cyclic) attractors are possible
  in rectified networks with inhibition, where de Candia's symmetric model has none.

**Hahnloser RHR, Seung HS, Slotine JJ (2003). Permitted and forbidden sets in symmetric threshold-linear networks. Neural
Comput. 15:621–638.** doi:10.1162/089976603321192103 [A]

- For symmetric TLNs: convergence to attractive fixed points if and only if the matrix is copositive; multiple attractors
  if and only if it is not positive semidefinite.
- "Permitted sets" (co-active groups at a stable steady state) are closed under subsets; forbidden sets are closed under
  supersets.
- **Relevance.** The permitted-set lattice is the threshold-linear version of "which Bm states exist". It is a useful
  organizing concept for the multistability map in the (g, r) plane.

**Litwin-Kumar A, Doiron B (2012). Slow dynamics and high variability in balanced cortical networks with clustered
connections. Nat. Neurosci. 15:1498–1505.** doi:10.1038/nn.3220 [A]

**Schaub MT, Billeh YN, Anastassiou CA, Koch C, Barahona M (2015). Emergence of slow-switching assemblies in structured
neuronal networks. PLoS Comput. Biol. 11:e1004196.** doi:10.1371/journal.pcbi.1004196 [A]

- Litwin-Kumar & Doiron: clustered excitation in balanced E–I networks gives clusters that transiently switch up or
  down. Rewiring only 3% of excitatory connections changes the dynamics substantially.
- Schaub et al.: slow-switching assemblies are tied to leading eigenvalues separated by a gap from the bulk of the
  asymmetric weight matrix, with Schur vectors partly localized on neuron groups.
- **Relevance.** E–I precedents for metastable switching among modules ("Bm with noise"). Schaub et al. make the same
  spectral argument as ours (gapped, localized leading modes control slow dynamics). With h > 0, our model might switch
  between the L and R AL lobes or between islands. That is worth testing as a de Candia-like broken-symmetry regime.

**Schwalger T, Deger M, Gerstner W (2017). Towards a theory of cortical columns: from spiking neurons to interacting neural
populations of finite size. PLoS Comput. Biol. 13:e1005507.** doi:10.1371/journal.pcbi.1005507; arXiv:1611.00294 [A]

- Mesoscopic stochastic population equations for interacting populations of 50–2000 neurons, with finite-size noise.
  They predict finite-size-induced stochastic transitions in multistable networks.
- **Relevance.** A principled module-level stochastic reduction for modules of a few hundred neurons, the AL's scale.
  Useful if we build an M-module reduced SWC with per-module finite-size noise.

**Schaub MT, O'Clery N, Billeh YN, Delvenne JC, Lambiotte R, Barahona M (2016). Graph partitions and cluster
synchronization in networks of oscillators. Chaos 26:094821.** arXiv:1608.04283 [A]

- External equitable partitions (EEPs) give exact coarse-graining of network dynamics onto quotient graphs, extended to
  signed networks (sEEP).
- **Relevance.** The formal condition behind de Candia's exact module reduction. On the connectome we can measure how far
  a partition is from equitable. The variance of the block-input sums within blocks quantifies how badly the symmetric
  SL/SH/Bm picture is broken.

**Farrell PE, Birkisson Á, Funke SW (2015). Deflation techniques for finding distinct solutions of nonlinear partial
differential equations. SIAM J. Sci. Comput. 37:A2026–A2045.** doi:10.1137/140984798 [A]

- Deflation modifies the residual so that Newton iterations converge to *new* solutions from the same initial guess.
  Preconditioning keeps the cost flat as solutions are deflated. It is implemented in BifurcationKit.jl, per its
  documentation.
- **Relevance.** A practical tool to find coexisting QMF fixed points (Bm analogues) of the N = 167k or module-reduced
  equations on GPU. It complements uniform random restarts (Golos et al. 2015) and support enumeration (Curto et al.
  2019).

**Scarpetta S, Apicella I, Minati L, de Candia A (2018). Hysteresis, neural avalanches, and critical behavior near a
first-order transition of a spiking neural network. Phys. Rev. E 97:062305.** doi:10.1103/PhysRevE.97.062305 (as cited
by de Candia 2026; not opened) [A]

- **Relevance.** Hysteresis and avalanches near a first-order line. Cited for method only.

---------------------------------------------------------------------------------------------------------------------

## (e) Links to wakefulness, sleep and anesthesia (only where the paper makes the link)

**Xu Y, Schneider A, Wessel R, Hengen KB (2024). Sleep restores an optimal computational regime in cortical networks. Nat.
Neurosci. 27:328–338.** doi:10.1038/s41593-023-01536-9 [A]

- Rat visual cortex, recorded continuously for 10–14 days. Waking progressively pushes dynamics away from criticality and
  sleep restores it. The deviation predicts sleep/wake behaviour better than slow-wave activity.

**Priesemann V, Valderrama M, Wibral M, Le Van Quyen M (2013). Neuronal avalanches differ from wakefulness to deep sleep.
PLoS Comput. Biol. 9:e1002985.** doi:10.1371/journal.pcbi.1002985 [A]

- Human intracranial recordings: power-law avalanches in every vigilance state. Slow-wave sleep has larger avalanches,
  wake intermediate, REM smaller. The authors model the brain as slightly subcritical, with state changes from small
  global changes in synaptic strength.

**Tagliazucchi E, et al. (2016). Large-scale signatures of unconsciousness are consistent with a departure from critical
dynamics. J. R. Soc. Interface 13:20151027.** doi:10.1098/rsif.2015.1027 [A]

**Toker D, et al. (2022). Consciousness is supported by near-critical slow cortical electrodynamics. PNAS
119:e2024455119.** doi:10.1073/pnas.2024455119 [A]

- Tagliazucchi et al.: under propofol, frontothalamic long-range temporal correlations and structure–function coupling
  drop. A model with a phase transition reproduces this as a move away from criticality.
- Toker et al.: in ECoG and MEG, slow cortical oscillations move away from an edge-of-chaos critical point under
  anesthesia, and move toward it with psychedelics.

**Fontenele AJ, et al. (2019). Criticality between cortical states. Phys. Rev. Lett. 122:208101.**
doi:10.1103/PhysRevLett.122.208101 [A]

- Urethane-anesthetized rats: the crackling-noise scaling relation holds at an intermediate value of spiking variability
  (CV), between synchronized and desynchronized states. Exponents are not all MF-DP.

**Hengen KB, Shew WL (2025). Is criticality a unified setpoint of brain function? Neuron 113:2582–2598.**
doi:10.1016/j.neuron.2025.05.020 [A]

- Review and meta-analysis (preprint: 143 datasets, 2003–2024) arguing that criticality is a homeostatic set-point.

**Leung A, Cohen D, van Swinderen B, Tsuchiya N (2021). Integrated information structure collapses with anesthetic loss
of conscious arousal in Drosophila melanogaster. PLoS Comput. Biol. 17:e1008722.** doi:10.1371/journal.pcbi.1008722 [A]

- Fly brain recordings: integrated interactions during wake collapse to isolated clusters under isoflurane, uniformly
  across the brain. The authors argue this rules out purely feedforward fly circuits.
- **Relevance.** The only fly study found that links a brain-wide dynamical measure to an arousal state. It uses no
  criticality or phase-diagram framework.

**Negative result.** Searches for Drosophila avalanches, criticality, branching or Griffiths analyses of brain activity,
and for anesthesia effects on fly criticality, returned none. A parallel workflow, wf_eebe6fe0-5e5 (AL literature),
covers the AL–sleep question in depth.

---------------------------------------------------------------------------------------------------------------------

## (f) Fly connectome data, whole-brain fly models, and AL biology (context for interpreting the AL module)

**Berg S, et al. (2025/2026). Sexual dimorphism in the complete connectome of the Drosophila male central nervous system.**
bioRxiv doi:10.1101/2025.10.09.680999; per a search result, published in Cell 2026 (doi:10.1016/j.cell.2026.08.015, not
opened) [A]

- The maleCNS: central brain, optic lobes and VNC. The preprint graph has 166,391 neurons and 25.6M edges. Our v1.0
  cache: 167,216 neurons and 25,587,572 edges (CLAUDE.md).

**Dorkenwald S, et al. (2024). Neuronal wiring diagram of an adult brain. Nature 634:124–138.**
doi:10.1038/s41586-024-07558-y [A] — FlyWire: 139,255 neurons and 5×10⁷ synapses (female).

**Scheffer LK, et al. (2020). A connectome and analysis of the adult Drosophila central brain. eLife 9:e57443.** [A] — the
hemibrain.

**Lin A, et al. (2024). Network statistics of the whole-brain connectome of Drosophila. Nature 634:153–165.**
doi:10.1038/s41586-024-07968-y [A] — rich-club organization (≈30% of neurons highly connected).

**Pospisil DA, et al. (2024). The fly connectome reveals a path to the effectome. Nature 634:201–209.**
doi:10.1038/s41586-024-07982-0 [A]

- Eigen-decomposition of the FlyWire connectome under a linear model gives "eigencircuits": the dominant circuits involve
  *relatively small populations of neurons*, and "fly whole-brain dynamics appear to be generated by a large collection
  of small circuits that operate largely independently".
- **Relevance.** Independent support, on a different fly connectome with a linear model, for our finding that all top-400
  excitatory modes are localized (CLAUDE.md). It also supports reading the fly brain as many weakly coupled modules
  rather than one bulk. For a de Candia-style diagram this means many heterogeneous, weakly coupled "modules", which
  favours sequential island recruitment over symmetric lines.

**Shiu PK, et al. (2024). A Drosophila computational brain model reveals sensorimotor processing. Nature 634:210–219.**
doi:10.1038/s41586-024-07763-9 [A]

- A whole-brain LIF model built from FlyWire connectivity and predicted transmitters. Neurons fire when excitation
  outweighs inhibition. It predicted feeding and grooming circuits that were validated experimentally.
- **Relevance.** A whole-fly-brain dynamics precedent with signed, connectome-derived weights and the same sign
  convention issues, but no phase-diagram analysis.

**Lappalainen JK, et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system.
Nature 634:1132–1140.** doi:10.1038/s41586-024-07939-3 [A]

- Connectome-constrained model of 64 optic-lobe cell types, with unknown single-cell and synapse parameters fit by deep
  learning to motion detection. It predicts activity across 26 studies.
- **Relevance.** Context: connectome plus task fitting, not a statistical-physics phase analysis.

**Eckstein N, et al. (2024). Neurotransmitter classification from electron microscopy images at synaptic sites in
Drosophila melanogaster. Cell 187:2574–2594.** doi:10.1016/j.cell.2024.03.016 [A]

- EM-based transmitter predictions: 87% accuracy per synapse and 94% per neuron. These are the source of connectome sign
  assignments.
- **Relevance.** Our sign map rests on predicted transmitters. Sign errors enter the r-axis.

**Liu WW, Wilson RI (2013). Glutamate is an inhibitory neurotransmitter in the Drosophila olfactory system. PNAS
110:10294–10299.** PMC3690841 [A]

- In the AL, glutamate hyperpolarizes projection neurons and GABAergic local neurons via GluClα. About one third of AL
  local neurons are glutamatergic.
- **Relevance.** Supports our "glutamate = inhibitory" choice inside the AL module, the region that dominates the
  dynamics. Elsewhere glutamate can be excitatory (CLAUDE.md lists this as a model limitation).

**Olsen SR, Bhandawat V, Wilson RI (2007). Excitatory interactions between olfactory processing channels in the
Drosophila antennal lobe. Neuron 54:89–103.** PMC2048819 [A]

- Projection neurons receive substantial lateral *excitatory* input from other glomeruli, stereotyped across flies.
  Glomeruli are interconnected by local interneurons.
- **Relevance.** Biological context for why the AL forms a strongly recurrently excited module in the weighted graph.

**Dahmen D, Grün S, Diesmann M, Helias M (2019). Second type of criticality in the brain uncovers rich multiple-neuron
dynamics. PNAS 116:13051.** doi:10.1073/pnas.1818972116 [A]

- A second type of criticality, set by the bulk spectral radius of the connectivity (weight variance), at which the
  number of active neurons stays roughly constant.
- **Relevance.** Our bulk vs localized-tail distinction: g_lin is a tail (localized) threshold. A bulk spectral-radius
  criterion would apply to the D line.

---------------------------------------------------------------------------------------------------------------------

## Pointers for the plan (literature-derived; for the design step, not results)
1. **Module-level de Candia diagram.**
   - Partition: spectral islands, or Louvain on the weighted excitatory graph.
   - Quotient matrices: E_kl and I_kl, with the equitability error reported (Schaub et al. 2016).
   - Fixed points: QMF fixed points by active-module support (Curto et al. 2019; deflation, Farrell et al. 2015), with
     Jacobian stability. Map (g, r) regions by which supports are stable.
   - Simulation: hysteresis sweeps in g and r to find first-order lines (Alvankar Golpayegan & de Candia 2023).
2. **Order of the g_lin onset.**
   - Field scaling ρ(h) ~ h^{1/2} at g_lin (Piuvezam et al. 2023).
   - QS density and χ vs g (de Oliveira & Dickman 2005; Ferreira et al. 2012).
   - Lifetime scaling vs effective size, ln τ vs K and τ vs √K (Doering et al. 2005). Use nulls of several N, or M
     replicas of the AL module.
   - Avalanche exponents, cutoff vs K, shape and skewness at g_lin (de Candia et al. 2021, 2026; Friedman et al. 2012),
     with Touboul & Destexhe and Priesemann & Shriki as caveats.
3. **Finite-size check on nulls.** The configuration null (delocalized leading mode) should show χ-peak drift and height
   growth with N, and a sharpening onset (Ferreira et al. 2012; Goltsev et al. 2012). The real AL crossover width is set
   by K, not N (Goltsev et al. 2012; Russo et al. 2014; Doering et al. 2005).
