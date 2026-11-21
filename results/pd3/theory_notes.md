# Theory audit: de Candia's modular theory on the MaleCNS connectome, and whether the crossover is second order

2026-11-09. Theory only (no new simulations). Light CPU computations only:

| Code | Output |
|---|---|
| `code/pd3_theory_module.py` | `results/pd3/theory_module.json` (+ `.log`) |
| `code/pd3_theory_module2.py` | `results/pd3/theory_module2.json` |
| `code/pd3_theory_bd.py` | `results/pd3/theory_bd.json` |
| `code/pd3_theory_bd2.py` | `results/pd3/theory_bd2.json` |
| `code/pd3_theory_nullK.py` | `results/pd3/theory_nullK.json` |
| `code/pd3_theory_figs.py` | `figs/pd3/theory_audit.png` |

Inputs (read only):
- the cache;
- `results/module/module_graph.npz` (the 526-neuron AL module);
- `results/pd/tail_real_r0_dir.npz`, `branch_{real,null}_dense_part*.json`, `mode_*.json`, `flow_params.json`;
- `results/analysis2/phase_complete.{md,json}` and `flow_sim_boundaries.json`.

Every number below is either taken from those files (source given) or computed by the pd3 scripts (json key given).

Notation:
- α_eff = (1 − e^{−dt})/dt = 0.906 (dt = 0.2). β = 1.
- f(x) = (1 − e^{−tanh(x)₊ dt})/dt is the discrete-time switch-on rate. Fixed points of the simulator's synchronous map equal those of
  α_eff ρ = (1 − ρ) f(s).
- E and I are the excitatory and inhibitory weight matrices scaled by c = 1/mean in-strength (c = 1.3406e-3). The simulator uses
  W̃(r) = E − rI.
- g_lin = 0.16315: the coupling at which λ_PF(B(g)) = α_eff, with B_ij(g) = f(g E_ij)
  (`branch_real_dense_part*.json`).

---------------------------------------------------------------------------------------------------------------------

## 0. Short answers

1. **Coarse-graining.** de Candia's M-module theory coarse-grains to heterogeneous modules without trouble. Our knobs map onto
   theirs as w⁰_k = g(E_kk − r I_kk)/α_eff and w¹_kl = g(E_kl − r I_kl)/α_eff. Raising r rotates the ray traced by g in the
   (w0, w1) plane. Inter-module imbalance changes sign at r₁* = E_kl/I_kl. For the left/right antennal-lobe (AL) pair that is
   r₁* = 2.0–2.3, while intra-module imbalance stays positive up to r₀* ≈ 10 (Perron-weighted). So *in de Candia's closure*, a
   broken-symmetry window r₁* < r < r₀* exists. The coarse 2-module model computed here does show it: B1L/B1R bistability for
   r ≥ 2.5, and B1 + SH coexistence (first-order lines) above that.

2. **But no B1 at the neuron level.** That closure is not valid for our network, and the broken-symmetry phase disappears at the
   neuron level:
   - The exact linearization of the stochastic process near silence contains only excitatory branching (verified earlier; P1).
     So at linear order the inter-module coupling is never inhibitory.
   - The neuron-level mean field (QMF) of the 526-neuron module has **no B1 state at any r ≤ 6**. All starting conditions (one AL
     active, the other AL active, both) converge to the same both-active state.
   - The exact two-mode normal form gives a competition ratio Γ ≈ 0.003, far below the winner-take-all threshold of 1.
   - The AL pair is de Candia's "point A/B" case: two weakly, excitatorily coupled near-copies with an SL→SH (symmetric) line.
     Prediction: no left-skewed avalanches from L/R competition.

3. **Second order?** The silent state loses stability at g_lin through a transcritical bifurcation with mean-field
   directed-percolation (MF-DP) exponents. The one-mode birth–death reduction is exactly finite-K MF-DP: the logistic SIS process
   on a complete graph of K units.
   - The "steep but smooth crossover" at g_on ≈ 0.27 (r = 0) to 0.58 (r = 6) is **not the critical point**. It is the
     T-dependent extinction crossover of a K ≈ 37–76 module, sitting R_on ≈ 1.6–2.1× above g_lin.
   - In the strict sense it is not a phase transition. K is fixed by the AL anatomy, not by the brain size N, so the brain's
     crossover never sharpens.
   - It is the finite-K image of a second-order (MF-DP) transition: one constant K per r reproduces τ(g) across the whole window,
     including its drift with observation time.
   - Section 2.4 lists the finite-size signatures that would establish this (and that a "mere crossover" would not show), with
     numbers. Section 3 lists the tests.

---------------------------------------------------------------------------------------------------------------------

## 1. De Candia's modular mean field, generalised to connectome modules

### 1.1 Same rules, different regime

Both models share the same rules:
- binary units;
- 1→0 at rate α;
- 0→1 at rate β tanh(s)₊, with s = Σ_j w_ij a_j + h.

de Candia et al. 2026 use M equal modules of N_E = N_I = 10⁵ neurons, all-to-all, with weights w^E_kl/N_E and −w^I_kl/N_I. Each
input is then a sum of ~10⁵ tiny terms. It is deterministic to O(N^-1/2), equal for E and I neurons of a module, and the
net-input closure s_k = Σ_l (w^E_kl x_l − w^I_kl y_l) is exact.

Our connectome is the opposite regime:
- about 150 inputs per neuron (25.6M edges / 167,216 neurons);
- single synapses up to g·3.5 (theory.html section 8).

A lone inhibitory spike onto a silent neuron does nothing (F(negative) = 0). The exact linear operator at silence is therefore the
excitatory branching matrix B(g), not E − rI (theory.html section 3). This was verified on the configuration null: the net-input
mean field puts the r = 1 onset at g = 3.43, but simulation shows sustained activity at 1.11 (`phase_complete.md`). Everything
below is written for both closures, so that the difference is explicit.

### 1.2 Coarse-grained equations for heterogeneous modules

Partition the neurons into modules k = 1..M of sizes n_k (AL left/right; AL vs rest; or the spectral islands). Each module has an
excitatory population (presynaptic sign +1) and an inhibitory population (sign −1); modulatory neurons have no outgoing weight.
Two coarse-grainings:

**Uniform (de Candia's literal closure).** Let X^a_k be the active fraction of population a ∈ {E, I} of module k.

    E^a_kl = (1/n^a_k) Σ_{i∈(k,a)} Σ_{j∈l, exc} c w_ij,     I^a_kl = (1/n^a_k) Σ_{i∈(k,a)} Σ_{j∈l, inh} c w_ij
    dX^a_k/dt = −α X^a_k + (1 − X^a_k) β tanh( g Σ_l [E^a_kl X^E_l − r I^a_kl X^I_l] )₊

This is de Candia's Eq. 4–5 with w^E_kl ↔ g E_kl and w^I_kl ↔ g r I_kl. When E- and I-neurons of a module receive the same input
(E^E = E^I), fixed points have X^E = X^I (their Δ_k = 0) and collapse onto one variable per module with W_kl(r) = E_kl − r I_kl.

**Perron-weighted (needed for heterogeneous modules).** Let v^k and u^k be the right and left Perron vectors of E restricted to
module k (Perron root Λ_k), with v^k normalised to mean 1 over the module and ⟨u^k, v^k⟩ = 1. Write ρ_i ≈ X_k v^k_i on module k.

    E^PW_kl = ⟨u^k, E_kl v^l⟩ (so E^PW_kk = Λ_k),   I^PW_kl = ⟨u^k, I_kl v^l⟩,   W(r) = E^PW − r I^PW
    dX_k/dt = −α X_k + (1 − X_k) β tanh( g Σ_l W_kl(r) X_l )₊

The uniform closure misplaces the threshold of a heterogeneous module by a factor ≈ 2. For the AL sides,
E^UN_kk = 2.62 / 2.50 against Λ_k = 5.39 / 5.46, which gives thresholds 0.35 / 0.36 against 0.168 / 0.166
(`theory_module.json` `A_AL_pair`). The Perron weighting is therefore required. It reproduces λ_PF: the 2×2 E^PW has eigenvalues
5.509 / 5.347, against 5.536 / 5.374 for the module and 5.574 / 5.389 for the whole brain.

### 1.3 Mapping (g, r) onto (w0, w1)

In de Candia's units (critical w0 = α/β = 1):

    w0_k(g, r) = g [E_kk − r I_kk] / α_eff,     w1_kl(g, r) = g [E_kl − r I_kl] / α_eff

- At fixed r, g moves the point radially along a ray from the origin.
- r rotates the ray: w1_kl changes sign at r₁*(kl) = E_kl/I_kl and w0_k at r₀*(k) = E_kk/I_kk.
- If r₁* < r₀*, the window r₁* < r < r₀* is de Candia's w1 < 0 < w0 half-plane, where the broken-symmetry phases Bm live.
- Inter-module connections that are relatively more inhibitory than intra-module ones are exactly what makes this window exist.

### 1.4 Stability analysis, general heterogeneous M (rectified-linear)

**Silent state (SL).** Near X = 0, dX/dt = −αX + βg[W X]₊. This map is piecewise linear and positively homogeneous. Its growth
rate is the cone spectral radius

    μ₊(W) = max over supports A consistent with rectification of λ_PF(W_AA)

(consistent means W_{A^c,A} v_A ≤ 0 off the support). SL is stable iff βgμ₊ < α. Two cases:
- W_kl ≤ 0 for all k ≠ l: μ₊ = max_k W_kk. This is de Candia's SL–B1 line w0 = 1 for w1 ≤ 0. For heterogeneous modules it is
  set by the strongest module alone.
- W_kl ≥ 0: μ₊ = λ_PF(W). This is the SL–SH line, w0 + (M − 1)w1 = 1 for w1 ≥ 0, in the equal-module case.

Both lines are transcritical, i.e. second order: the order parameter grows continuously from zero.

**Bm states (active set A).** A fixed point X_A > 0 of the A-subsystem is a Bm state if:
- (existence, the "dead zone") Σ_{l∈A} W_kl X_l ≤ 0 for every k ∉ A, so f(s_k) = 0 and X_k = 0 is a fixed point;
- (internal stability) J_AA = −diag(α + βtanh S_k) + diag((1 − X_k) β sech² S_k) g W_AA is Hurwitz.

A strictly negative input gives transverse stability for free. The rate is −α, and perturbations smaller than the dead zone
|Σ_l W_kl X_l| / W_kk are absorbed. Three kinds of boundary:
1. Transcritical lines inside A, where the A-subsystem's Perron root crosses α/(βg). These are second order.
2. Loss of the dead zone: an inactive module's net input reaches 0 from below. If that module is supercritical on its own, it
   jumps to high activity. This is first order: de Candia's dashed lines, e.g. B1 at w1 → 0⁻.
3. Instability of a mixed-sign eigenvector of J_AA, for example the antisymmetric mode of SH when w0 − w1 is large. This is also
   first order, with coexistence of SH and B1.

With heterogeneity, the degenerate C(M, m) Bm states split into a cascade ordered by Λ_k. The symmetric first-order lines split
into one line per module pair.

**The exact (sparse) version.** Replace the silent-state operator by the excitatory branching operator. Section 4 of theory.html
gives the amplitude equations for modes A_k (ρ ≈ Σ_k A_k v^k):

    dA_k/dt = Σ_l (𝔅_kl − α δ_kl) A_l − Σ_{l,m} c^k_lm A_l A_m,    𝔅_kl = ⟨u^k, B(g) v^l⟩ ≥ 0

where c^k_lm combines:
- the (1 − ρ) saturation;
- the coincidence terms Σ_{j<m} |Δ^i_jm| ρ_j ρ_m, i.e. E–E concavity and coincident inhibition, the only place r enters.

Consequences:
- At linear order every inter-module coupling is ≥ 0. The SL line is g_lin, vertical in (g, r).
- de Candia's w1 < 0 does not exist at the silent state.
- A broken-symmetry state can only come from second-order competition: the Lotka–Volterra criterion
  Γ = a_kl a_lk / (a_kk a_ll) > 1, with a_kk = c^k_kk and a_kl = c^k_kl + c^k_lk. It must also beat the linear seeding 𝔅_kl > 0,
  which makes every such bifurcation imperfect.

### 1.5 The left/right antennal lobes as a module pair

Data: `theory_module.json` `A_AL_pair`, `mode0`, `mode1`; `theory_module2.json`.

**Composition.** The 526-module splits into 248 L, 275 R and 3 midline neurons. By sign: 366 excitatory, 121 inhibitory,
39 modulatory.

**Correction to the working description.** The two leading modes are *not* one per lobe:
- mode 0 (λ = 5.574) has 69% of |v|² on R and 31% on L;
- mode 1 (λ = 5.389) has 70% on L and 30% on R;
- their 99% supports overlap in 395 neurons.

The reason is that the lobes are hybridised:

| quantity | value |
|---|---|
| per-side Perron roots Λ_L / Λ_R | 5.392 / 5.464 (detuning 0.072, 1.3%) |
| Perron-weighted excitatory cross-coupling E_LR / E_RL | 0.073 / 0.072 |
| coupling / half-detuning | 2.01 |

A two-level system with coupling twice the half-detuning mixes the two lobes roughly 70/30 (tan 2θ ≈ 2.0), which is what the
eigenvectors show. The same ratio, 2.03, appears in the exact branching operator at g_lin: 𝔅 = [[0.8767, 0.0118], [0.0120, 0.8884]]
(`C_normal_form`). Only 5.2–5.8% of each lobe's excitatory module-internal in-strength comes from the other lobe, and 11.8–12.3% of
its inhibitory in-strength (`in_strength_share_from_other_side`).

**Inhibitory couplings (Perron-weighted).** I_LL = 0.508, I_RR = 0.553, I_LR = 0.031, I_RL = 0.036 (unit-norm Galerkin). With
mean-1 normalisation the cross terms are 0.034 / 0.033 and E_LR / E_RL = 0.080 / 0.066 (`theory_module2.json`). Thresholds:
- r₀* = 10.6 (L), 9.9 (R);
- r₁* = 2.33 (L←R), 2.01 (R←L);
- uniform closure: r₀* = 4.4 / 4.1, r₁* = 2.04 / 1.60.

Above r = 1.58 the L lobe overtakes R in net-input gain, because I_RR > I_LL.

**(w0, w1) of the AL pair** (de Candia units; Perron-weighted, mean-1 normalisation; from `theory_module2.json` PW_E/PW_I_mean1):

| g | r = 0 | r = 1 | r = 2 | r = 3 | r = 4 | r = 6 |
|---|---|---|---|---|---|---|
| 0.163 (g_lin) | w0 0.97/0.98, w1 +0.014/+0.012 | 0.88/0.88, +0.008/+0.006 | 0.79/0.78, +0.002/0.000 | 0.70/0.69, −0.004/−0.006 | 0.61/0.59, −0.010/−0.012 | 0.42/0.39, −0.023/−0.024 |
| 0.266 (sim onset r = 0) | 1.58/1.60, +0.024/+0.019 | 1.43/1.44, +0.013/+0.010 | 1.28/1.28, +0.003/0.000 | 1.14/1.12, −0.007/−0.010 | 0.99/0.95, −0.017/−0.019 | 0.69/0.63, −0.037/−0.039 |
| 0.579 (sim onset r = 6) | 3.44/3.49, +0.051/+0.042 | 3.12/3.14, +0.029/+0.021 | 2.80/2.78, +0.007/0.000 | 2.47/2.43, −0.015/−0.021 | 2.15/2.08, −0.037/−0.042 | 1.50/1.37, −0.081/−0.084 |

(w0 L/R; w1 L←R/R←L.)

In de Candia units the AL pair sits very close to their point A (w1 ≈ 0): |w1|/w0 ≤ 0.062 everywhere (about 0.01 at r = 1).
Their avalanche points use |w1|/w0 = 0.125 (point B: w0 0.8, w1 +0.1) and 0.15 (point C: w0 1, w1 −0.15), i.e. 2–15 times
larger.

**Computed phase diagrams of the AL pair (figure panel a).**

*(a) Coarse 2-module model, de Candia closure* (`theory_module2.json` `coarse_2module`). Attractors from four starts, each with a
one-neuron (1/n_k) seed on every silent side:
- r ≤ 2: SL → SH, continuous, at the SL edge g = α_eff/μ₊.
- r = 2.25: B1L + SH coexist at g = 0.224–0.273, then SH.
- r ≥ 2.5: B1L | B1R bistable just above the edge, then B1 + SH coexistence. For example, r = 4: SL up to 0.273, B1L | B1R from
  0.287, B1 + SH from 0.317. r = 6: B1L | B1R from 0.428, B1 + SH from 0.578.
- At r ≥ 3 a narrow band just above the edge also keeps SL against symmetric one-neuron seeds (mutual dead zones).
- The edge itself rises from 0.1645 (r = 0) to 0.386 (r = 6), because w0 falls with r in this closure.

This is a de Candia-type diagram: second-order SL–SH and SL–B1 lines, and first-order B1/SH lines.

*(b) Neuron-level mean field (QMF) of the same 526 neurons* (`theory_module.json` `D_module_qmf`): **no B1 anywhere.**
- For r = 0, 0.5, 1, 2, 3, 4, 6 and g = 0.15–3 (41 log points), the L-only, R-only and both-active starts end in the same state
  (largest difference 8e-11) with X_L ≈ X_R.
- The onset is continuous. The first active grid point carries 0.1–1.5% activity. The onset lies in the grid cell 0.162–0.174
  (r = 0, 0.5), 0.174–0.188 (r = 1, 2) and 0.188–0.202 (r = 3, 4, 6).
- Mechanism: the dead zone does not exist at the neuron level (`theory_module2.json` `neuron_dead_zone`). Take each lobe's target
  neurons and drive them from the other lobe's Perron pattern. The share receiving net-POSITIVE input is:

  | r | R←L | L←R | u-weighted |
  |---|---|---|---|
  | 0 | 84% | 86% | — |
  | 1 | 36% | 38% | 50–54% |
  | 6 | 25% | 22% | 39–41% |

  The active lobe always seeds the silent one through these neurons, and the seeded lobe's own gain (Λ ≈ 5.4) dwarfs the cross
  inhibition (≈ 0.035 r).

*(c) Exact two-mode normal form* (`theory_module.json` `C_normal_form`; factorised coincidences, g = g_lin, 0.268, 0.35, 0.5, 0.8):
- Γ = 1e-4 at r = 0, and 0.0025–0.0035 for r = 0.5–6. Cross-saturation is 0.3–6% of self-saturation, so the lobes coexist rather
  than compete.
- The same computation gives the module's single-mode K_eff: 58.9 / 54.1 / 50.3 / 43.7 / 33.2 at r = 0, and 17.9 / 17.4 / 17.0 /
  16.2 / 14.5 at r = 1. The whole-brain dense rows give 62.5 at g = 0.175 and 57.9 at 0.268 (r = 0), and 17.2 at 0.268 (r = 1),
  which confirms the implementation.

**Conclusion for the AL pair.** The left/right lobes are a natural symmetric pair. In de Candia's language they are weakly,
excitatorily coupled modules near point A, with a single SL→SH (hybrid-mode) second-order line at g_lin and no broken-symmetry
phase in our model. Their mechanism for left-skewed avalanches (cooperative recruitment, then inter-module competition) therefore
has nothing to act on. Expected avalanche shapes at the AL onset are symmetric (one-mode, section 2.4) or right-skewed (intra-module
E/I imbalance, de Candia Fig. 3), not left-skewed.

### 1.6 The AL vs the rest of the brain (de Candia's SH ↔ our D)

Data: `theory_module.json` `B_AL_rest`.

| quantity | value |
|---|---|
| Uniform input to a rest neuron from the fully active AL | exc 0.0083, inh 0.0041 |
| Rest neurons receiving any AL input | 19,542 (exc 0.071, inh 0.035 on average among them) |
| r₁*(rest←AL) | 2.04 |
| Share of receivers net-inhibited at r = 0.5 / 1 / 2 / 4 / 6 | 17% / 28% / 43% / 60% / 65% |
| Perron-weighted input into the AL from the fully active rest | exc 4.96, inh 0.52 |
| r₁*(AL←rest) | 9.6 |

The module picture of the simulated phases:
- Q + S = finite-K rounding of SL.
- L = the AL module active and the rest nearly silent, the analogue of B1.
- D = global activity, the analogue of SH.

At r < 2 the L phase is not a symmetry-broken phase protected by inhibition. It is protected by heterogeneity: the rest is
subcritical on its own, and the AL's drive onto it is weak and excitatory (an imperfect bifurcation). That fits the smooth growth
of PR (950 → 2,800 → 24,000; `STATE.md`) and a smooth L→D crossover. For r > 2 the *mean* AL→rest coupling is net inhibitory,
which in de Candia's closure would turn the L→D boundary into a first-order line with L/D coexistence. At the neuron level,
57% (r = 2) to 35% (r = 6) of receivers are not net-inhibited, which argues against it, as in the AL pair. This is a direct test (T6).

### 1.7 Many modules: the island cascade

The top-400 excitatory eigenmodes are all localized (1/IPR ≤ 632, median 68, down to λ₄₀₀ = 0.242, i.e. g = 3.74). Treat them as
heterogeneous modules. Their weak-synapse thresholds g_k = α_eff/λ_k are:
- 0.163, 0.168 (the AL pair);
- then 0.482 (island 3: 1/IPR 632, AL share 0), 0.521 (76% AL), 0.529, 0.580 (96% AL), 0.670, 0.687, ...

Number of islands with g_k below 0.3 / 0.5 / 1.0 / 1.47 / 2 / 3: 2 / 3 / 22 / 56 / 118 / 266 (`E_islands`).

With decoupled modules this would be a cascade of second-order SL→B_m lines. The linear excitatory seeding from already-active
islands (𝔅_kl > 0) makes each of them imperfect, so the cascade is a smeared recruitment, not a sequence of sharp lines. This is
what the simulation shows (no jump; theory.html P3). The global D line is not an island threshold: at r = 0, D = 1.47 has only
56 islands below it, with summed 1/IPR 6,586 neurons ≪ N/4 = 41,804. It is the collective mean-field threshold of the driven rest, which the QMF D line
reproduces at r = 0 (1.45 vs 1.47; `phase_complete.md`).

---------------------------------------------------------------------------------------------------------------------

## 2. Is the steep smooth crossover a second-order transition?

### 2.1 Mean field: transcritical at g_lin with MF-DP exponents

The amplitude equation (theory.html section 4), with R = λ_PF(B(g))/α_eff and R − 1 ≈ g/g_lin − 1 near g_lin:

    dε/dt = (λ − α) ε − [λ⟨u, v²⟩ + D(g, r)] ε²   ⇒   ε* = (R − 1) / (R⟨u, v²⟩ + D/α)

- The order parameter is linear in g − g_lin: exponent β_op = 1.
- Linear response to a drive h below threshold, ρ = βh⟨u, 1⟩ v / (α − λ(g)), and the relaxation rate |λ − α| on both sides give
  γ = 1 and ν_∥ = 1.

These are the MF-DP / branching exponents de Candia quote for their critical lines.

Number of active neurons in the mean-field active state:

    N_act = ε* Σ v ≈ A_r (g/g_lin − 1),   A_r = 115 (r = 0), 35 (r = 1), 18.5 (r = 6)

(`C_normal_form` at g_lin: Σv/(R uv2 + D/α_eff).) Check against the neuron-level QMF of the module at g = 0.1742, where
R − 1 = 0.064 relative to the module's own threshold 0.1637: predicted 7.4 active neurons, QMF 7.8.

### 2.2 Finite K: the one-mode reduction is finite-size MF-DP

The chain b(n) = λn(1 − n/K), d(n) = αn is the stochastic logistic / SIS process on a complete graph of K sites, i.e. MF-DP at
size K. Its finite-size scaling, computed exactly (`theory_bd.json`, `theory_bd2.json`; time in units of 1/α; divide by
α_eff = 0.906 for model time):

| quantity at R = 1 (g = g_lin) | law | fitted (K ≥ 1024) | numbers for K = 37 / 54.7 / 75.8 |
|---|---|---|---|
| mean lifetime from a fully active start | τ_top ∝ K^1/2 | 1.90 K^0.503 | 12.0 / 14.8 / 17.7 model time |
| quasi-stationary (QS) lifetime | τ_QS ∝ K^1/2 | 0.91 K^0.499 | 6.2 / 7.5 / 8.8 model time |
| QS mean (chain units) | n_QS ∝ K^1/2 | 0.69 K^0.497 | 4.41 / 5.29 / 6.17 → ×1.94 = 8.6 / 10.3 / 12.0 active neurons |
| QS density | ρ_QS = n_QS/K ∝ K^-1/2 | 0.69 K^-0.503 | 0.119 / 0.097 / 0.081 |
| size cutoff (P/P_∞ = 1/2) | S_c ∝ K | K^1.03 (K = 64–512) | 123 / 186 / 260 chain units → ×1.94 = 240 / 360 / 505 neuron spikes |
| duration cutoff ((1 + αt)P(T > t) = 1/2) | T_c ∝ K^1/2 | K^0.51 (K = 64–2048); T_c ≈ 2.0 K^1/2 | 13.3 / 16.4 / 19.5 model time |
| mean avalanche duration from one seed | ∝ ln K | — | 2.5 / 2.7 / 2.8 (1/α) |
| QS susceptibility χ = var(n)/⟨n⟩ | peak ∝ K^1/2 at x = (R − 1)K^1/2 = 1.7 | K^0.52 (K ≥ 300), K^0.55 (all) | peak 2.5 / 3.2 / 3.9 at R = 1.279 / 1.230 / 1.195 |

- **Size cutoff: S_c ∝ K, not K².** This is SIS-type depletion: the saturation acts on the currently active, n²/K. Over a time t
  that drift accumulates to t³/K, which balances n ∼ t at t ∼ K^1/2, giving S ∼ t² ∼ K. An SIR-type process, with depletion by
  cumulative activations, would give S_c ∝ K^2/3 and T_c ∝ K^1/3. Our units turn off and can be re-activated, so SIS applies.
- **Chain units to neuron spikes.** Multiply by ⟨u², v⟩ Σv = 1.94 (module leading mode at g_lin, `C_normal_form` `sbar_sum_v`).
- **Corrections to scaling are large at K ≈ 40–80.** τ/K^1/2 does not collapse on x = (R − 1)K^1/2 beyond x ≈ 1. At x = 3,
  ln(τ/√K) = 2.07 (K = 37), 2.38 (75.8), 3.12 (1000), 3.33 (5000) (`collapse`). K Φ(R) = x²/2 − (2/3)x³/√K + … So data from the
  real module must be compared with the exact chain, not with the asymptotic scaling function.

### 2.3 What the crossover is

A finite system has no sharp transition, so the question is whether the crossover is *the finite-K image of an MF-DP critical
point*. Three facts:
1. **Location.** The critical point of the module's mean field is g_lin = 0.163, where the lifetime is only ≈ 12–18 model time
   (table above). The steep crossover that looks like an onset is where τ(g) reaches T, R_on ≈ 1.6–2.1× above g_lin. It drifts
   with T as √(ln T): a metastable finite region, not a critical point.
2. **Size.** The relevant size is the module's K (≈ 37 at r = 1, ≈ 76 at r = 0 needed by the data; 54.7–62.5 at r = 0 and
   16.9–17.9 at r = 1 from theory), not the brain's N. Adding neurons elsewhere does not sharpen it. In the thermodynamic sense the
   brain has no phase transition here: it has a rounded MF-DP transition of a ~50-unit module. In de Candia's model the analogous
   rounding is K^-1/2 ≈ 0.003 (N_E = 10⁵ per module). Here it is of order 1.
3. **Consistency with finite-K MF-DP, from existing data.** One constant K per r reproduces the measured τ(g) through the whole
   window:
   - K = 36.9 (34–40) at r = 1 (`longruns.md`) and 75.8 (72.7–80.4) at r = 0 (`phase_complete.md`).
   - With these K and the connectome's λ(g), the chain gives (`theory_bd.json` `real_crossings`):

     | r | quantity | chain | simulation |
     |---|---|---|---|
     | 0 | S (τ = 200) | 0.231 | 0.234 [0.228, 0.244] |
     | 0 | onset (τ = 3000) | 0.265 | 0.266 [0.260, 0.278] |
     | 1 | S | 0.283 | 0.283 |
     | 1 | onset | 0.346 | 0.344 |
     | 1 | τ = 30,000 crossing | 0.395 | g_half(T = 30,000) = 0.386–0.399 |

   - The T-drift ×1.14 (0.346 → 0.395) matches the observed ×1.10–1.14. The theory's ×1.40 came from its too-small K_eff (16.9).
   - Lifetimes above threshold are exponential, as the chain's quasi-stationary decay requires (0 KS rejections at 9 g values;
     `phase_complete.md`).
   - **Caveat:** K_needed was fitted to these same lifetimes, so this checks the one-parameter *form*, not an independent
     prediction. The factor ≈ 2 between K_needed and the computed K_eff remains unexplained (`phase_complete.md`, open items).

**Answer.** The crossover is a finite-K-rounded second-order (transcritical, MF-DP) transition of the AL module, located at g_lin
and observed through its extinction-time crossover. It is not a bulk phase transition and does not sharpen with brain size. It is
more than a "mere crossover" only to the extent that the finite-size signatures of section 2.4 hold. Most of them are cheap to
test on the module alone.

### 2.4 Finite-size signatures that discriminate

Each signature below separates "MF-DP critical point rounded by finite K" from "a smooth crossover without an underlying critical
point" (for example an imperfect bifurcation or a gradual change of gain).

**(i) At g = g_lin.**
- τ_top ≈ 1.9 √K/α_eff = 12.0 / 14.8 / 17.7 model time; τ_QS = 6.2 / 7.5 / 8.8.
- The QS state holds 8.6 / 10.3 / 12.0 active neurons.
- A smooth crossover without a critical point has no reason to show √K scaling.

**(ii) Single-seed avalanches at g_lin** (seeds ∝ module v).
- P(S) ∝ S^-3/2 from S ≈ 2 to S_c ≈ 3.4K: 240 / 360 / 505 neuron spikes for K = 37 / 54.7 / 75.8.
- Just before the cutoff P(S) sits *above* the infinite-K law: P/P_∞ peaks at 1.29 at S ≈ 100 for K = 75.8 (`psi_at`), then the
  cutoff follows.
- P(T > t) ≈ 1/(1 + α_eff t) up to T_c ≈ 13–20 model time: about one decade of T^-2 (panels b, c).

**(iii) ⟨S⟩(T).** The exponent 2 is only asymptotic. At K = 37 and 75.8 the local slope d ln⟨S⟩/d ln T peaks at about 1.55 and
1.66, never 2. It reaches 1.79 at K = 300 and 1.91 at K = 2000 (`mc_R1`, `theory_bd2.json` `mc`; panel d).
- A measured ⟨S⟩ ∼ T^1.5–1.7 on the module is therefore *expected* under MF-DP and is not evidence against it.
- The sharp test is the K-dependence: slopes rise toward 2 in larger-K nulls.

**(iv) Avalanche shape.**
- For any one-mode reduction, i.e. any 1-D birth–death chain, the duration-conditioned mean profile is exactly time-symmetric.
  Proof: every birth–death chain is reversible (π(n)b(n) = π(n+1)d(n+1)). An avalanche path starts and ends at n = 1, so its
  density equals that of its time reverse, holding times included.
- Monte Carlo confirms it: 400,000 avalanches per K. The mean skewness per duration bin is consistent with 0: |mean| ≤ 0.021, and
  3 of 53 bins exceed 2 SEM (max 2.7 SEM), about what chance gives. The centroid offset is ≤ 0.0025 (`mc_R1`).
- Skewness in the real module therefore measures a second dynamical variable:
  - right skew: intra-module E–I imbalance, de Candia's Δ_k;
  - left skew: inter-module competition, which section 1.5 says is absent here.

**(v) Fluctuations.**
- The QS susceptibility χ = var(n)/⟨n⟩ peaks at x = (R − 1)√K ≈ 1.7, i.e. g ≈ 0.195 / 0.201 / 0.209 for K = 75.8 / 54.7 / 36.9
  (R(g) from the dense rows). Peak heights are 3.9 / 3.2 / 2.5 (chain units), and the module lifetime there is 37 / 30 / 23 model
  time.
- This zero-drive (QS) peak lies *below* the τ = 200 boundary. The driven peak at h = 1e-5 (0.311) is set by ignition versus
  extinction instead (theory.html P5).

**(vi) Null family (delocalized leading mode).**
- Configuration nulls of size N built by stub matching on a random N-subset (`pd3_theory_nullK.py`) keep
  λ_PF(E) = 1.07–1.15 (weak-synapse g = 0.79–0.89).
- Their K_nc grows as N^0.92 (K/N = 0.021–0.050, ×2 seed-to-seed scatter); 1/IPR grows as N^0.61. "K ∝ N" holds to within that
  scatter.
- With K = κN and the full-size null's R(g), the chain predicts (`null_family`):

| N | 1,000 | 2,000 | 5,000 | 10,000 | 20,000 | 40,000 | 80,000 | 167,216 |
|---|---|---|---|---|---|---|---|---|
| r = 0 (κ = 5031/167216): K | 30 | 60 | 150 | 301 | 602 | 1204 | 2407 | 5031 |
| onset/g_lin (τ = 3000) | 2.37 | 1.76 | 1.39 | 1.25 | 1.16 | 1.11 | 1.07 | 1.045 |
| τ(g_lin), model time | 10.7 | 15.6 | 25.4 | 36.5 | 52.1 | 74.1 | 105 | 153 |
| r = 1 (κ from K_needed 181): onset/g_lin | – | not reached | not reached | 5.74 | 2.91 | 1.99 | 1.57 | 1.34 |

- The real brain's onset/g_lin is 1.63 (r = 0) and 2.11 (r = 1) at its full N, set by the AL, and would stay so under subsampling
  that keeps the AL.
- A null of about 2,000 neurons (r = 0) or 40,000–80,000 (r = 1) has the same relative window as the whole real brain.
- At full size the predicted null onsets (0.866 at r = 0, 1.112 at r = 1) match the simulated 0.867 and 1.11. At r = 0 the
  agreement holds only to the simulation's 8% grid cell [0.838, 0.905] (critic, `phase_complete.md`). At r = 1 it is by
  construction, since K_needed came from those runs.
- The window shrinks as K^-1/2 ∝ N^-1/2 only asymptotically; at N ≤ 20,000 the corrections of section 2.2 dominate.
- Real r = 0 R(g) is used for every N. This assumes λ_PF and the mode shape do not depend on N; the λ_PF spread above (±5%) moves
  g_lin(N) by the same amount.

---------------------------------------------------------------------------------------------------------------------

## 3. Concrete, computable predictions (with numbers)

Each line gives the prediction, the numbers, and the cheapest test.

- **T1. SL boundary is r-independent.** It lies at g_lin = 0.1632 (exact linearization). The de Candia/net-input closure instead
  moves it from 0.1645 (r = 0) to 0.386 (r = 6) (coarse model), or to 0.163 → 0.194 (whole-brain rectified QMF, `mode_real.json`).
  *Test:* single-seed module avalanches at g = 0.163 for r = 0, 1, 6 should look equally critical (τ_top ≈ 12–18,
  P(T > t) ≈ 1/(1 + α_eff t) up to T_c). Under the net-input closure the r = 6 runs would be strongly subcritical.

- **T2. Lifetime at g_lin ∝ K^1/2.** From a fully active start, the module alone (526 neurons) at g = 0.1632 has
  τ ≈ 14.8 (K = 54.7), within 12.0–17.7 for K = 37–76. *Test:* module_sim-type run, 1,000 replicas, T = 200, r = 0 and 1.

- **T3. Avalanche statistics at g_lin** (module seeds, r = 0):
  - P(S) ∝ S^-3/2 for 2 ≲ S ≲ 360 neuron spikes, with a bump to 1.29× near S ≈ 200 spikes, then the cutoff;
  - P(T) ∝ T^-2 up to T_c ≈ 16 model time;
  - ⟨S⟩(T) local slope ≤ 1.7.
  - At r = 1, use K ≈ 37: cutoffs 240 spikes and 13 model time.

- **T4. Shapes.** Symmetric (skewness 0 ± 0.02) for the one-mode description. A measured skew of either sign at the AL onset
  measures a second variable: positive skewness (left-skewed) would contradict section 1.5's no-competition result.

- **T5. QS susceptibility peak** below the S line: g ≈ 0.195 (r = 0, K 75.8) and 0.209 (r = 1, K 36.9), height ∝ K^1/2.
  *Test:* QS method (reactivate on absorption) or h → 0 driven runs on the module, 15–20 g points in 0.17–0.30.

- **T6. No AL broken symmetry / no hysteresis.** Run the module from "L only" and "R only" starts at r = 3, 4, 6 and
  g = 0.25–0.6:
  - neuron-level theory: both lobes active within the module lifetime, with no lasting one-lobe state;
  - de Candia closure: B1L | B1R bistability (r = 4: g ≥ 0.287), and B1 + SH coexistence above 0.317 (r = 4) and 0.578 (r = 6).

  Whole-brain analogue: a QMF and a simulation from an "AL only" start vs a fully active start at r = 2–6, g = 1–30, to look for
  L/D hysteresis (first-order L→D). The neuron-level argument predicts none.

- **T7. Null family finite-size scaling** (configuration nulls, N = 2k–167k, r = 0): onset/g_lin = 1.76, 1.39, 1.25, 1.16, 1.11,
  1.07, 1.045 for N = 2k, 5k, 10k, 20k, 40k, 80k, 167k. τ(g_lin) grows as √N (15.6 → 153 model time). Measure K (or 1/IPR) of
  each realization, because K/N scatters ×2 between seeds.

- **T8. Real-brain subsampling.** Remove neurons outside the AL (keep ≥ the 526-module and its inputs): the onset ratio stays
  ≈ 1.63 (r = 0).

- **T9. Drift with T.** At r = 1 the τ = T crossing moves 0.346 → 0.395 from T = 3,000 to 30,000 (K = 36.9). At r = 0 it moves
  0.265 → 0.289 (K = 75.8) (`real_crossings`).

- **T10. Mean-field amplitude.** QMF/factorised active count ≈ 115(g/g_lin − 1) at r = 0, 35(g/g_lin − 1) at r = 1, and
  18.5(g/g_lin − 1) at r = 6, for g just above g_lin.

- **T11. Island cascade.** The next non-AL island switches on (weak-synapse) at g = 0.482. Islands below g = 1.0 / 1.47 / 2:
  22 / 56 / 118. Their recruitment should appear as small steps in PR(g) between 0.48 and 1.5 at r = 0, all imperfect (no
  thresholds visible as kinks if the AL drive dominates).

- **T12. Effective r-dependence (no theory fits it).**
  - Simulated onset rises ×2.17 from r = 0 to 6.
  - The exact threshold does not move (×1.0).
  - The K_eff extinction theory rises ×8.1.
  - The de Candia-closure SL edge rises ×2.35, close in total but wrong in shape: onset/edge = 1.62, 1.87, 1.94, 1.88, 1.50 at
    r = 0, 1, 2, 4, 6.

  None reproduces it quantitatively; this is still open, together with the factor ≈ 2 in K.

---------------------------------------------------------------------------------------------------------------------

## 4. Negative results and limits

- **No broken-symmetry phase.** de Candia's B1/B2 phases with first-order lines appear for the AL pair only in the coarse
  net-input closure. The neuron-level mean field and the exact two-mode normal form both say no (sections 1.5b, c).
- **What would be needed for a de Candia diagram.** Modules with strong, *dense* mutual inhibition reaching all their neurons
  (no net-excited receivers), comparable in size to the modules' own gain. The AL pair has |w1|/w0 ≤ 0.06, and 22–38% of receivers
  are net-excited even at r = 6. A different module definition (Louvain, 400 islands) could be screened with the same
  `neuron_dead_zone` statistic before running any dynamics.
- **The two leading modes are hybrid L/R modes (≈ 70/30), not one per lobe.** The "left and right AL" description in the task
  text should be read as "the 526-module spans both lobes".
- **Coarse-model labels.** The coarse 2-module phase labels depend on the seed size (one neuron) used to probe the dead zone. At
  r ≥ 3, SL is kept just above the edge only against symmetric seeds.
- **What the finite-K analysis assumes.** The one-mode chain assumes that both λ(g) and the reduction itself are right. The
  measured lifetimes need K about 2× the computed K_eff (r ≥ 0.25), and at r = 0 even more than the no-coincidence K
  (75.8 vs 70.7). The constants in section 2.2 are exact *for the chain*; the K to use for the real module is the empirical one.
- **Null family caveats.** The predictions use the full-size null's λ(g) for every N, and the theory-K/N ratio from one
  realization. K/N scatters ×2 between realizations (`theory_nullK.json`).
- **Module definition.** Recomputing the module's 99% supports from the float16 tail swaps one neuron at a tie on the boundary;
  the saved `module_graph.npz` index set was used throughout (`recomputed_support_symdiff` = 2).

## 5. Files

- Code:
  - `code/pd3_theory_module.py` (A–E, about 11 min on 2 CPUs; D dominates);
  - `code/pd3_theory_module2.py` (about 1 min);
  - `code/pd3_theory_bd.py` (about 15 s);
  - `code/pd3_theory_bd2.py` (about 10 s);
  - `code/pd3_theory_nullK.py` (about 25 s);
  - `code/pd3_theory_figs.py`.
- Results: `results/pd3/theory_{module,module2,bd,bd2,nullK}.json` and the matching `.log`.
- Figure: `figs/pd3/theory_audit.png`:
  - (a) the AL-pair (g, r) diagram under the de Candia closure, with g_lin, the neuron-level onset and the simulated onset;
  - (b) compensated P(S) at g_lin;
  - (c) compensated survival at g_lin;
  - (d) ⟨S⟩(T) local slope vs K;
  - (e) QS susceptibility;
  - (f) null-family onset ratio vs N.
