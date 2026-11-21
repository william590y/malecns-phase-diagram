# Stochastic and spectral theory of Model 2 (lane m2/theory-stochastic)

Written 2026-11-09 by the stochastic-theory lane of workflow wf_aae614ed-511. Plan and pre-registration:
`results/m2/theory/stochastic_plan.md`. Code: `code/m2_stoch_core.py` (library), `code/m2_stoch_run.py` (GPU tasks),
`code/m2_stoch_modes.py` (per-mode table, CPU), `code/m2_stoch_graphs.py` (my own FSS and module graphs),
`code/m2_stoch_test.py` (tests), `code/m2_stoch_predict.py` (assembles the prediction files). Raw outputs:
`results/m2/theory/stochastic/`. Prediction files: `results/m2/theory/stochastic_*.json` (md5 sums in section 9).

**Blindness.** These predictions are not blind; see `stochastic_plan.md` section 0. The latest sections of `STATE.md`,
which I was told to read, already summarise the finished M2 experiment lanes. I did not open any file under
`results/m2/{decay,driven,modules,nature}/` before the prediction files were written. No parameter is fitted to M2 data.
The only calibration is taken from Model 1 (`results/analysis2/phase_complete.json`) and is labelled **CAL**;
everything else is **PF** (parameter-free).

---------------------------------------------------------------------------------------------------------------------

## 1. The model in mean variables

Per step (dt = 0.2, p_off = 1 − e^(−dt) = 0.1813, α = p_off/dt = 0.906), with the simulator's order (recovery, then the
onset takes e = x and depletes x ← (1 − U) x; the input uses the previous state's a_j e_j):

| variable | meaning |
|---|---|
| ρ_i = E[a_i] | activity |
| O_i = E[a_i e_i] | mean output (depressing units) |
| R1_i = E[a_i x_i], R0_i = E[(1 − a_i) x_i] | resource, split by state |

    ρ'  = (1 − p_off) ρ + p (1 − ρ)
    O'  = (1 − p_off) O + p A0,            A0 = q R0 + (1 − q)(1 − ρ),  A1 = q R1 + (1 − q) ρ,  q = e^(−dt/TAU)
    R1' = (1 − p_off) A1 + p (1 − U) A0
    R0' = p_off A1 + (1 − p) A0

For one unit at a fixed onset probability p these moment equations are exact (x changes linearly given the state). Their
stationary solution gives the exact mean efficacy e(ρ) = O/ρ. It agrees with the renewal formula used by the aldef lane
to 7e-14 (test T1, `results/m2/theory/stochastic/test/tests.json`). Expanding at low activity,

    e(ρ) = 1 − κ ρ + O(ρ²),   κ = U · p_off q G_L(q)/(1 − q),  G_L(q) = p_off q/(1 − (1 − p_off) q)
    κ(M2) = 35.30  (Poisson limit U·TAU·α = 35.57)

**Closures across units.**
- (QMF) p_i = 1 − exp(−β tanh(s_i)₊ dt) with s_i = g Σ_j c w_ij σ_j out_j: the mean input enters the nonlinearity, and
  inhibition is subtracted from it.
- (bQMF, first-order branching, "noisy-OR") each active excitatory input j triggers i independently with probability
  π_ij = 1 − exp(−tanh(g c w_ij e_j) dt) per step:

      p_i = 1 − Π_j (1 − ρ_j π_ij)

  This is exact to first order in the activity. Inhibition enters only through coincidences, which are second order.

**Why both, and which is used where (deviation from the plan, logged).** The plan named the QMF. In my own smoke run (M2,
r = 1; `results/m2/theory/stochastic/decay/M2_r1_smoke.json`), the subtractive QMF crushes the network to about 60
active neurons in total, even at g = 6, and leaves a renormalised Perron root of σ~ = 10.7 on neurons that the closure
holds silent. This is the documented M1 failure of mean-field inhibition: the QMF D line overstated inhibition by 1.2–10×
(`results/analysis2/phase_complete.md`, item 4). For r > 0 I therefore use the bQMF profile, with inhibition carried
only by the M1-calibrated factor c_r (section 4.4). The QMF is kept at r = 0, where the two closures coincide to first
order, as a cross-check, and as the basis of the linear-noise approximation. I made this change after seeing my own theory
output, before seeing any M2 data file.

## 2. Silence: the linear threshold does not move (PF, exact)

A fully recovered unit transmits e = 1, so the branching operator about silence is unchanged:
B(g)_ij = f(g c w_ij), with f(x) = (1 − e^(−tanh(x₊) dt))/dt. Inhibition does not enter the linearisation of the
stochastic process about silence, so

    g_lin(M2) = g_lin(M1) = α / λ_PF(B(g_lin)) = 0.1632, for every r, every U, TAU (spike mode), every ORN level.

2.1 **M2s is the exception.** Setting lLN1_bc→ALPN to 0 removes 6,658 excitatory edges, so E itself changes and
g_lin(M2s) is slightly higher. In M1 the LN→PN edges carry 16.5% of λ₁, but only part of that is lLN1_bc→PN, and the
lLN1_bc clique (LN→LN) is intact. I did not compute M2s; the expected effect is small.

## 3. Depression renormalises every mode: K → K/(1 + κ) (PF)

**The M1 one-mode chain.** With a mode (λ, right v, left u, ⟨u, v⟩ = 1) of the excitatory branching operator, put
activity ρ_j = n ⟨u², v⟩ v_j. The M1 chain is b(n) = λ n (1 − n/K), d(n) = α n, with K₀ = 1/(⟨u², v⟩⟨u, v²⟩) when
coincidences are neglected.

**Adding depression.** Each active presynaptic neuron transmits e(ρ_j), so the birth rate gains the mode-weighted
efficacy:

    X(n) = Σ_j u_j v_j e(ρ_j) / Σ_j u_j v_j = 1 − κ n ⟨u², v⟩⟨u, v²⟩ + …

This uses one fact: the left vector u of the excitatory operator vanishes on every neuron without excitatory outputs, so
every j that carries weight is depressing. Hence

    b(n) = λ n [1 − (1 + κ) n / K₀ + …],     K_dep = K₀ / (1 + κ) = K₀ / 36.3     (every mode, every g)

The effective size of every mode shrinks 36-fold. The M1 AL module had K₀ ≈ 71 (`phase_complete.md`), so
K_dep ≈ 2: under depression it cannot hold a metastable state on its own near g_lin. That is the spectral reason why
M1's AL-led slow window disappears.

This small-amplitude statement assumes the resources follow the activity (adiabatic). The resources are in fact slow
(TAU = 157 ≫ 1/α = 1.1), and e(ρ) is a hyperbola, not a line. Section 4 handles both.

## 4. The active state, the renormalised branching operator and its lifetime

### 4.1 Renormalised (frozen-resource) operator on the active state
At the mean-field active state (ρ*, e* = O*/ρ*), freezing the resources:

    B~(g) = f(g E diag(e*)),   λ~ = λ_PF(B~),   σ~ = λ~/α   (frozen-resource branching ratio)
    λ_rest = ⟨u~, B(g) v~⟩      (the same mode, rested),   X* = λ~/λ_rest   (mode efficacy)
    n* = ⟨u~, ρ*⟩/⟨u~², v~⟩     (activity in mode units),  K = n*/(1 − 1/σ~)   (all saturation in the profile)

**Self-organised near-criticality.** In the frozen picture the active state obeys λ~ (1 − n*/K) = α. So
σ~ − 1 = n*/(K − n*). This is small whenever depression, rather than the (1 − ρ) factor, holds the activity down. The
network therefore always sits just above criticality of its own depressed operator, at every g above the onset. This is
the mechanism of Levina, Herrmann and Geisel (2007). It is a statement about σ~, not about avalanches (section 7) or about
the measured branching ratio under drive (section 6).

### 4.2 Three lifetimes (from the active state)
- **(F) frozen:** the logistic chain (λ~, K). This is the limit TAU → ∞.
- **(A) adiabatic:** the 2-D chain's exact fast-resource manifold, (1 − y) = U TAU λ_rest y² n (1 − n/K)/N_y, inserted
  into b(n) = λ_rest y(n) n (1 − n/K). This is the limit TAU → 0 at fixed U·TAU/N_y. Test: the 2-D solver converges to
  it, ln τ 5.266 vs 5.274 at TAU = 0.02 (`tests.json`).
- **(2D) piecewise-deterministic chain in (n, y), the primary prediction:**

      n → n + 1 at b = λ_rest y n (1 − n/K),   n → n − 1 at α n,   dy/dt = (1 − y)/TAU − U y b / N_y

  Here N_y = U TAU α X* n*/(1 − X*), so that its deterministic fixed point is (n*, X*). The mean extinction time from
  (n*, X*) is computed by a sparse direct solve with an upwind grid in y.

The 2-D lifetime can lie below the frozen one: slow negative feedback amplifies fluctuations. In the test case,
ln τ is 3.95 (2D) against 4.07 (F) and 5.27 (A).

### 4.3 Linear analysis of the 2-D chain: quasi-cycles and the Gaussian lifetime
Linearising about (n*, y*), with σ~ − 1 = n*/(K − n*) and N_y fixed by the balance:

    J11 = −α (σ~ − 1),   J12 = α n*/y*,   J21 ≈ −(1 − y*)/(TAU n*),   J22 = −(2 − y*)/(TAU y*)
    det ≈ (α/(TAU y*)) [(1 − y*) + (σ~ − 1)(2 − y*)],   tr = −[α (σ~ − 1) + (2 − y*)/(TAU y*)]

When depression dominates (σ~ − 1 ≪ 1 − y*), the eigenvalues are complex. The result is a noise-driven quasi-cycle
with

    period P ≈ 2π √(TAU y* / (α (1 − y*))),   damping γ = −tr/2,   coherence C(2P)/C(P) ≈ e^(−γ P)

Examples: y* = 0.3 gives P ≈ 54; y* = 0.5 gives P ≈ 83 model units. Its amplitude is that of a birth-death noise, so
var(ρ) ∝ 1/N. This is a quasi-cycle (Benayoun et al. 2010; here the slow variable is the resource), not a limit cycle.
With noise only in n (D = α n*), the stationary variance is Var n = D (J22² + det)/(|tr| det). The Gaussian estimate of
the lifetime is ln τ ≈ n*²/(2 Var n). Near σ~ → 1 it agrees with the WKB result at leading order.

### 4.4 Inhibition (CAL)
The bQMF profile carries no inhibition. M1 measured how much inhibition shrinks the effective module: the K needed to
reproduce the simulated lifetimes, divided by the no-coincidence K, is c_r = K_needed/K_noD (`phase_complete.md`
table; 1.07 at r = 0, 0.82 at 0.25, 0.69 at 0.5, 0.52 at 1, 0.38 at 2, 0.27 at 4, 0.23 at 6). The CAL variant multiplies
n*, K and N_y by c_r, keeping σ~ and X*. Assumption: inhibition shrinks the brain-wide active state by the same factor
that it shrank the AL module in M1.

### 4.5 Validation on Model 1 (r = 0)
The same pipeline without resources (QMF profile → Perron mode of B(g) → n*, K → logistic chain) reproduces the M1 r = 0
boundaries. From the smoke run (`decay/M1_r0_smoke.json`; coarse grid): S ≈ 0.236 (simulation 0.234 [0.228, 0.244]) and
onset ≈ 0.276 (simulation 0.266 [0.26, 0.278]); the K implied by the profile is about 67, against M1's K_noD of 70.7.
The production values are in section 8.

## 5. The deterministic transient from a fully active start (PF)

The decay protocol starts with every unit active, e = 1 and x = 1 − U. In the mean map the activity collapses within a
few units while every excitatory resource is depleted. It then recovers on the scale TAU. The minimum of the mode
amplitude during this recovery, n_min, measured against the stationary level n*, predicts excess early extinctions:
replicas that die before the resources recover. Their effect is a τ_MLE pulled below the stationary lifetime near the
onset, and survival curves with an early drop and an exponential tail.

## 6. Linear-noise approximation (LNA) under ORN drive

z = (ρ, O, R1, R0) for depressing units and ρ for the rest; A = one-step Jacobian of the mean map at the driven fixed
point; Q = covariance of the Bernoulli on/off events, with resources at their conditional means:

    Q_i = ρ p_off(1 − p_off) d1 d1ᵀ + (1 − ρ) p(1 − p) d0 d0ᵀ
    d1 = (1, e, y1, −y1),   y1 = q x1 + 1 − q
    d0 = (1, y0, (1 − U) y0, −y0),   y0 = q x0 + 1 − q

The stationary covariance is C = A C Aᵀ + Q. For an observable cᵀz:

    Var = Σ_s w_sᵀ Q w_s,  with w_s = (Aᵀ)^s c
    ACF(k) = cᵀ A^k (C c),  with C c from a checkpointed reverse pass

For the onset count, the covariance at lag k ≥ 1 is c_onᵀ A^k C c_on + c_onᵀ A^(k−1) q_on, which includes the
same-step noise of the onsets and the state. The MR branching ratio is obtained by fitting r_k = b m^k for k = 1..20 steps,
exactly as m2_obs.mr_branching does. All of this costs one sparse product per step, with no N × N matrix.

**Test** (`tests.json`; 400-unit random graph, drive, depression with TAU 20, against an exact stochastic simulation of
the same synchronous rules):
- variance 3.877e-4 vs 3.818e-4;
- ACF at lags 1, 5, 20, 50, 100: 0.865 / 0.480 / 0.034 / −0.0137 / −0.0055, vs 0.864 / 0.476 / 0.032 / −0.0130 / −0.0053;
- onset-count variance 8.71 vs 8.63, and lag covariances within 3%;
- MR m 0.757 vs 0.742.

Inhibition: the LNA uses the QMF at r = 0. At first order the driven fluctuations do not depend on r, so r = 0 is the
prediction for every r. Where the simulation differs with r, that difference is the size of the coincidence effects.

## 7. Pulse protocol: self-organised branching and the Levina test (PF)

Separation of timescales: one ORN seed whenever the network is silent, after a gap of mean T_gap (0.2, 10.1 or 100.1).
Between avalanches the resources are treated as quasi-static at x. The branching matrix is M(x) = f(g E diag x)/α. The
expected onsets per avalanche are F = (I − M)⁻¹ s̄, with s̄ uniform over ORNs. The mean duration T_av comes from a
Poisson Galton–Watson recursion with mean m = ρ(M). The onset rate is ν = F/(T_av + T_gap), and x = e(ν/α). Solving this
self-consistently gives the self-organised distance from criticality, ε = 1 − ρ(M(x*)).

**Two cutoffs.**
- **Branching cutoff.** A process at distance ε has S_ε ~ 1/ε².
- **Within-avalanche depletion.** A neuron that fires again inside the same avalanche transmits (1 − U) times less.
  After S onsets spread over the mode with weights ∝ v~_j, the mode's branching ratio has dropped by
  δ(S) ≈ U S ⟨u~, v~²⟩/Σv~ = U S/K_s. A critical avalanche of size S survives only while δ(S) ≲ S^(−1/2), so

      S_dep ≈ (K_s / U)^(2/3)

**Prediction rule (fixed in the plan).** A g counts as "critical-like" only if min(S_ε, S_dep) ≥ 1000, the size MODEL2's
rule needs: largest/x_min ≥ 100 with x_min ≤ 10. When the self-organised mode is the AL mode (K_s ~ 10²), S_dep ~ 40:
no power law can span two decades.

## 8. Results and predictions

### 8.1 Validation on Model 1 (same pipeline, no resources)
File: `stochastic_onset.json` → models.M1. bQMF profile → Perron mode of B(g) → logistic chain.
| r | predicted S / onset | M1 simulation S / onset (`phase_complete.md`) |
|---|---|---|
| 0 (PF) | 0.233 / 0.265 | 0.234 [0.228, 0.244] / 0.266 [0.26, 0.278] |
| 1 (CAL) | 0.277 / 0.330 | 0.283 [0.28, 0.289] / 0.344 [0.34, 0.35] |
| 6 (CAL) | 0.393 / 0.507 | 0.436 [0.422, 0.44] / 0.579 [0.561, 0.615] |

The "pooled" variant (section 8.4) gives 0.205 / 0.238 at r = 0, 11% low: in M1 it overstates the lifetime, as expected
for a one-module state. M1 AL module alone: 0.237 / 0.272 against 0.233 / 0.265 for the full brain (pd3 found the two
within 1.8%). One lobe alone: 0.29 / 0.35. **So the single-mode chain is validated on M1 at r = 0, and within 2–12% at
r > 0 with the M1 calibration (which was derived from M1, so r > 0 is only a consistency check).**

### 8.2 Which excitatory modes go unstable where (`stochastic_modes.json`)
- The top 80 eigenvalues of E have 26 real and 54 complex. Only two real modes have positive eigenvectors (sign purity
  ≥ 0.9), so only for them is the island approximation reliable:
  - k = 0, the AL mode: λ 5.574, K₀ 70.1, PR 72, AL share 0.99;
  - k = 2, a broad non-AL central-brain mode: λ 1.879, K₀ 561, PR 632, AL share 0.00.
- Depression shrinks every mode by 1 + κ = 36.3: K_dep is 1.9 for the AL mode and 15.4 for mode 2 (PF).
- Single-mode onsets (τ = 3000; r = 0, PF):

| mode | g_lin,k (saturating kernel) | M1 onset | M2 onset, 2-D [frozen, adiabatic] |
|---|---|---|---|
| 0 (AL) | 0.163 | 0.271 | 2.77 [3.15, 1.59] |
| 2 (broad CB) | 0.484 | 0.564 | 1.50 [1.68, 0.99] |

  **Depression reverses the ranking of the modes.** In M1 the AL mode (largest λ) ignites first. In M2 the lifetime
  scales with K₀/(1 + κ), while λ enters only logarithmically, so a broad, weaker mode outlives the AL mode.
- At the 26-real-mode level, many sub-threshold or mixed-sign modes are VNC motor modes (λ 0.6–1.3). Their |v|-based K₀
  is unreliable and is not used.
- Counting "modes past threshold" (the aldef observation, 47 of 400 at g = 1.38) is therefore not the right picture of the
  onset. Silence is linearly unstable from g_lin = 0.163 onward, and depression decides which unstable mode can hold
  activity long enough.

### 8.3 The active state that depression selects (bQMF, PF, r-independent at first order)
Files: `stochastic_onset.json` → models.M2.profile, islands_M2; raw `decay/bqmf_M2.json`, `decay/bqmf_M2_peel.json`.
- **Where activity lives.**
  - At g = 1.21: 606 active-neuron equivalents spread with PR 12,587. Shares: AL 6.5%, central brain 57%, VNC 25%,
    optic lobe 8%.
  - At g = 1.66: PR 15,500 and AL 5.2%.
  - The AL share falls monotonically with g (27% at g = 0.50, 2% at g = 8).
- **Efficacy.** Active depressing neurons transmit e = 0.49–0.58 on average in this range. The AL transmits 0.19–0.24.
- **Renormalised operator.**
  - Its Perron mode is still the AL (AL share of v~² 0.91), but broadened: K₀ 157–168 against 70 at rest.
  - It sits just above criticality, σ~ = 1.14–1.25 (g 1.0–1.64). The rested gain of the same mode is λ_rest/α = 6–9.5,
    and its efficacy is X* = 0.13–0.19.
- **Islands (successive renormalised modes on disjoint supports).**
  - Below g ≈ 2.1 the AL is the only self-sustaining island (σ~ > 1).
  - The next islands are small VNC modules (K₀ 11–23) with σ~ = 0.90–0.98.
  - A large central-brain island (K₀ 317 → 2,326) crosses σ~ = 1 between g = 2.0 and 2.2.
  - **Prediction:** at the predicted onset the brain-wide activity is spill-over from a depressed AL "engine", amplified
    by a near-critical bulk (σ~ 0.95–0.98). It is not a second self-sustaining module.
- **Mean-field uniqueness** (`stochastic_multistart.json`).
  - Six starts give one fixed point at every g tested (all at 0.5; AL-L only; AL-R only; central brain only; VNC only;
    1% random), at h = 0 and under ORN drive, for M2 and M1. The largest relative L1 distance is 3e-10.
  - The driven up- and down-sweeps coincide (relative difference 2.5e-9).
  - Hence no coexistence, no first-order line and no broken L/R state are predicted.
  - Lobe asymmetry (L − R)/(L + R) is −0.02 to −0.05 at h = 0 and −0.065 to −0.08 under ORN drive: the right lobe is
    slightly more active, because the ORNs are right-biased in the connectome.

### 8.4 The onset and the slow window (setting a)
The pre-registered primary is the single renormalised mode with the 2-D chain. It ignores re-seeding from the rest of the
brain, so it is an **upper bound on the onset g**. The pooled variant (added after I saw that my own renormalised mode is
localised while the activity is broad) is a **lower bound** for non-interacting islands. The coherent-LNA estimate uses
the exact linear-noise variance of the total activity at the QS state (r = 0) and lies between them.

| r | single 2-D S / onset [frozen, adiabatic onset] | islands-max onset | pooled 2-D S / onset | coherent LNA onset | PR / AL share at single onset |
|---|---|---|---|---|---|
| 0 (PF) | **1.18 / 1.66** [2.05, 0.96] | 1.66 | 0.58 / 0.74 | 1.22 (1.32 with κ = 1/τ_int); S ≤ 1.0 | 15,500 / 5.2% |
| 0.25 (CAL) | 1.32 / 1.87 | – | – | – | 16,800 / 4.8% |
| 0.5 (CAL) | 1.46 / 2.08 | – | – | – | 18,000 / 4.5% |
| 1 (CAL) | 1.74 / 2.49 | 2.49 | 0.67 / 0.88 | – | 20,200 / 4.1% |
| 2 (CAL) | 2.17 / 3.09 | – | – | – | 23,400 / 3.6% |
| 4 (CAL) | 2.69 / 3.97 | 3.14 | 0.81 / 1.06 | – | 28,000 / 3.2% |
| 6 (CAL) | 3.08 / 4.60 | 3.29 | – | – | 31,100 / 2.9% |

Derived predictions:
- Width of the slow window: onset/S = 1.41 (single) and 1.28 (pooled) at r = 0, against 1.14 in M1.
- r-trend: onset(r = 6)/onset(r = 0) = 2.77 (single CAL). Pooled CAL gives onset(4)/onset(0) = 1.44.
- Ratio to M1: 6.3× (single) and 2.8× (pooled) at r = 0.
- g_lin stays 0.163 at every r, so the M2 onset sits 4.5–10× above g_lin (r = 0, across the three estimates).
- The QMF closure at r = 0 gives the same single-mode values (1.18 / 1.65; `decay/qmf_M2_r0.json`).

**The lifetime distribution in the slow window** is predicted to be exponential, because there is a single metastable
state (the 2-D chain) with no distribution of rare regions. So no Griffiths phase is predicted. The deterministic transient
from the fully active start (`transient`) dips to n_min/n* of
- ≤ 0.13 for g ≤ 0.38;
- 0.42–0.72 for g = 0.57–1.05;
- 0.78–0.86 for g = 1.29–1.94.

The minimum falls at t ≈ 31–42 model units. The prediction is excess early extinctions (t ≲ 100) that grow as g falls
below the onset. They would pull τ_MLE below the stationary lifetime there.

### 8.5 Sensitivity variants (single 2-D at r = 0, PF; ratio to M2)
| variant | S / onset | onset / M2 | pooled onset (ratio) | AL share at onset |
|---|---|---|---|---|
| KW09 (U 0.28, TAU 240) | 1.73 / 2.50 | 1.51 | 0.90 (1.23) | 4.4% |
| N15 (U 0.22, TAU 89) | 0.76 / 1.03 | 0.62 | 0.59 (0.80) | 6.9% |
| tau30 (TAU 30) | 0.45 / 0.59 | 0.35 | 0.43 (0.59) | 16% |

M2_allsyn: at first order (inhibition through coincidences only) it equals M2. With inhibition included, depressing
inhibitory outputs weakens the coincidence suppression, so the onset should be lower than M2 (direction only).

### 8.6 Driven state (LNA, ORN 0.04; r = 0 closure, i.e. first-order r-independent)
File `stochastic_driven.json`.
- **Gain.** The PN/ORN gain G = 0.29 at g = 0.214 and G = 1.45 at g = 1.03. For ORN levels 0.01 / 0.02 / 0.07 / 0.10 the window is [0.12, 0.42] / [0.15, 0.63] / [0.31, 1.72] / [0.41, 2.55], so it moves to higher g with drive. M1 under the same closure gives [0.091, 0.208]; the old E1 measurement (aldef, r = 0) was 0.090–0.231, which validates the closure's gain. The data-allowed region is that window
  intersected with "silent after removal" (g < S). So its top is 1.03 (gain-limited) on the single-mode S (1.18), or
  0.58 (S-limited) on the pooled S.
- **Fluctuations (no susceptibility peak at a transition line).**
  - Fano χ/ρ of nonORN peaks at 59 at g ≈ 0.91.
  - τ_int of nonORN peaks at 11.0 units at g ≈ 0.65.
  - χ_nonORN rises monotonically to the grid end (0.68 at g = 5); there is no interior peak.
  - The AL Fano factor has a broad plateau of 1.9–2.2 over g 0.23–0.91 (maximum 2.2 at g ≈ 0.39) and falls to 0.9 by g = 5.
  - None of these maxima sits at g_lin (0.163) or at the predicted onsets. They sit at 0.4–0.75× the single-mode onset.
- **Oscillation.**
  - The nonORN ACF has a deep negative lobe (−0.25 to −0.39 for g ≥ 0.65) at lag ≈ P/2, with P = 55–130 units.
  - The second peak is ≤ 0.10, so the OSC flag is marginal or absent, and OSC-strong never occurs (coherence ≤ 0.21).
  - Interpretation: a heavily damped activity–resource quasi-cycle.
- **Branching ratio (MR, nonORN onsets, k = 1..20 steps).**
  - Maximum m = 0.989 per step (0.948 per unit) at g ≈ 0.77.
  - |1 − m| < 0.01 nowhere; the closest is 0.011.
  - **No Levina-type m ≈ 1 range under drive.**
- **Other ORN levels** (0.01, 0.02, 0.07, 0.10):
  - the same Fano peak (57–66 at g ≈ 0.85) and τ_int peak (10–11 at g ≈ 0.6);
  - the gain window moves to higher g with stronger drive (see `summary.g_at_G_*`);
  - max m per unit 0.946–0.948.
- **M1 under the same LNA** (r = 0): AL Fano peak 3.0 at g = 0.146, nonORN Fano peak 13.4 and τ_int 6.7 at g = 0.51,
  max m per unit 0.84.

### 8.7 Oscillation at h = 0 (QS LNA, r = 0; `stochastic_qs.json`)
- **M2.**
  - Deep negative ACF lobe (−0.18 to −0.39) at every g from 0.6 to 12.
  - The OSC flag passes only at g = 1.0 and 1.2 (P 82 and 72; coherence 0.07). It also passes at g = 0.35, where the LNA
    is invalid (CV 2.4).
  - OSC-strong never occurs.
  - The period falls with g: 148 (g 0.6), 100 (0.8), 82 (1.0), 72 (1.2), 63 (1.5), 56 (2–3). The 2-D chain formula
    P ≈ 2π√(TAU y*/(α(1 − y*))) gives 80–130 for the pooled y* = 0.5–0.7.
  - The amplitude is that of a noise-driven quasi-cycle: var ∝ 1/N, while the normalised ACF does not depend on N.
- **M1.** No negative lobe at any g, so no oscillation.
- **Prediction.** OSC points, if any, sit near the onset and at low r. They are damped quasi-cycles, P ≈ 60–100,
  var·N constant across sizes. r-dependence is not computed: the closure is r = 0.

### 8.8 Avalanches and the Levina test (pulse protocol; `stochastic_aval.json`)
**Real M2, gap 0 (gap 100 similar).**
- For g < g_lin, ε = 1 − σ_rest: 0.39 at g = 0.10, 0.105 at g = 0.149.
- For g just above g_lin the quasi-static resources self-organise to ε = 1e-5 to 1e-2 (g 0.18–0.41). The self-organised
  mode is the AL: 44–59% of the onsets, AL resource 0.45–0.96, K_s = 123–233. So the within-avalanche depletion cutoff
  is S_dep = 62–95 onsets.
- At g ≈ 0.49 the self-organised mode broadens: K_s 4,717, S_dep 709, AL resource 0.13.
- For g ≥ 0.60 no subcritical quasi-static state exists: a rested, rarely visited island is supercritical. The prediction
  is system-size bursts (a large-avalanche bump) from g ≈ 0.55–0.6, that is ≈ 0.35× the single-mode onset or ≈ 0.8× the
  pooled onset.

**Exponents.** Within S ≲ S_dep the theory gives the mean-field branching values τ_S = 3/2, τ_T = 2 and γ = 2.

**Critical-like (min(S_ε, S_dep) ≥ 1000).** Not met at any g for the real M2.

**M1.** No self-organisation. Critical-like only within ~3% below g_lin (ε ≤ 0.032), a span ≤ 1.03. Above g_lin the
avalanches do not end.

**Predicted verdict under MODEL2's rule:** M2 span < 1.2, so **"not supported"**. If the M1 grid misses the 3% band,
neither model has a critical-like g and the rule says "inconclusive".

**Configuration null (full size), the contrast prediction.**
- K_s = 33,600–64,000 and S_dep = 2,600–4,000.
- Above the null's g_lin (≈ 0.83), ε = 1.5e-5 to 4.5e-4 for g = 0.91–3.0.
- So the null **is** critical-like from g ≈ 0.91 to at least 3.0, a span ≥ 3.3.
- **Depression would self-organise the delocalised null to criticality but not the real, localised connectome.**

### 8.9 Finite size and modules (`stochastic_fss.json`)
**Configuration nulls** (single 2-D, r = 0, PF):

| N | 5k | 10k | 20k | 40k | 80k | 167k |
|---|---|---|---|---|---|---|
| onset | 3.65 | 2.91 | 2.30 | 1.76 | 1.50 | 1.28 |
| W = ln(onset/S) | 0.309 | 0.286 | 0.254 | 0.229 | 0.20 | 0.175 |

- W falls monotonically (Spearman −1) as about N^(−0.16).
- W(167k)/W(5k) = 0.57, which narrowly fails the pre-registered "sharpening" criterion (≤ 0.5).
- The onset moves toward the null's g_lin, so the transition is continuous and strongly rounded.
- The pooled variant gives the same trend.
- Real/null onset ratio at full size: 1.30 (r 0, PF), 1.68 (r 1, CAL), 2.28 (r 4, CAL).
- **M2 reverses M1:** the real brain now ignites above its configuration null. In M1 the null's onset was 2–3× higher.

**Real subsamples.**
- Uniform random subsamples drift strongly with N, because they are diluted:
  - single onsets 21.6 (40k) → 7.14 (80k) → 1.66 (full);
  - pooled onsets 14.0 (20k) → 4.72 (40k) → 1.94 (80k) → 0.74 (full);
  - W 0.37–0.42 (single) and 0.26–0.30 (pooled), with no monotone trend.
- AL-kept subsamples: onset 2.02 → 1.98 → 1.88 → 1.66, with W 0.36–0.38.
- **Prediction: "drifting / inconclusive"**, neither sharpening nor fixed.

**Modules alone** (c fixed at the full-brain value; single onset, with the pooled onset in brackets):

| module | M2 onset | ratio to full brain | M1 onset |
|---|---|---|---|
| full brain | 1.66 (0.74) | 1 | 0.265 |
| AL module | 2.08 (1.70) | 1.25 | 0.272 |
| one AL lobe | 3.66–3.72 (2.87–2.95) | 2.2 | 0.344–0.349 |
| central brain | 1.67 (0.93) | 1.01 | – |
| optic lobe | 5.22 (2.75) | – | – |
| VNC | 24.1 (1.86) | – | – |

- Nothing but the central brain sustains alone at the full-brain onset, and the central brain only on the single-mode
  estimate.
- One lobe needs about 2.2× the onset of both lobes, so no one-lobe state is expected at the full-brain onset.


## 9. Files, md5 sums and deviations

The prediction files were written at 18:42 on 2026-11-09 (CPU job 211815, `code/m2_stoch_predict.py`). md5 (first 8):

| file | md5 |
|---|---|
| `results/m2/theory/stochastic_onset.json` (setting a: boundaries, profile, transient, islands) | 4a35165c |
| `results/m2/theory/stochastic_modes.json` (per-mode table) | 915a5dce |
| `results/m2/theory/stochastic_driven.json` (driven LNA) | 6bca4561 |
| `results/m2/theory/stochastic_qs.json` (h = 0 LNA: oscillation, coherent lifetime) | 10d58dc4 |
| `results/m2/theory/stochastic_aval.json` (pulse protocol / Levina) | a9fa0279 |
| `results/m2/theory/stochastic_fss.json` (FSS graphs and modules) | 79fa7a41 |
| `results/m2/theory/stochastic_multistart.json` (mean-field uniqueness) | cd9abe7c |

Figures: `figs/m2/stoch_lifetime.png`, `stoch_modes.png`, `stoch_driven.png`, `stoch_aval.png`.

**Jobs.** All are named `m2-stoch-*`.
- GPU: smoke 209790; decay 209856; driven 209880; lnaqs 209891; multistart 210403; pulse 210428; peel 210638.
- CPU (thickstun): modes 209874; graphs 209815; small-graph decay 210650; assembly 211815.
- Cancelled by me to save GPU time, with the informative rows kept:
  - pulse 210428_0/2/4: rows parsed from the logs by `code/m2_stoch_pulselog.py`; the null gap-100 sweep is incomplete;
  - driven ORN 0.02 part 1 and M2_tau30 driven;
  - lnaqs for tau30 and KW09;
  - the M1 big-graph decays and the M1 QMF cross-check (the M1 r = 0 QMF smoke run is kept).

**Deviations from the plan (all decided before opening any experiment file).**
1. **Closure.** bQMF (first-order branching) replaces the subtractive QMF for the setting-(a) profile, because the QMF
   crushes the network at r = 1 (section 1). The QMF is kept at r = 0 as a cross-check (1.18 / 1.65 vs bQMF
   1.18 / 1.66).
2. **CAL variant.** It uses c_r = K_needed/K_noD, as the plan says. An early code version used a ratio of K_eff instead;
   that version was replaced before any production run.
3. **"Pooled" and "islands-max" variants** were added after I saw that the renormalised Perron mode is localised on the
   AL while the activity is broad. The coherent-LNA lifetime was added at the same time. The pre-registered primary is
   unchanged.
4. **Numerical fixes.** The 2-D solver now uses Richardson extrapolation (M = 240/480) after a grid-convergence test
   showed a 0.3–0.6 error in ln τ at n* ≈ 150 for M = 240. The adiabatic bracket now uses the 2-D chain's exact
   fast-resource manifold.
5. **Driven LNA** was run at r = 0 only. With the first-order closure the theory has no r-dependence under drive.
6. **Not done:**
   - M2s;
   - the LNA for variants;
   - the oscillation's r-dependence;
   - the QS LNA at r > 0;
   - the avalanche size distribution itself (only its cutoffs).


## 10. Assumptions behind the predictions (one list, referenced by number)

A1. **Mean-field closure of the active profile.** The bQMF profile treats inputs as independent, and inhibition only
    through coincidences (it ignores them at first order). It is exact to first order in activity, and in M1 at r = 0 it
    reproduces the simulated S and onset to within 1% (section 8.1).
A2. **One-dimensional (or 2-D with a resource) reduction of extinction.** "single" uses the leading renormalised mode only.
    It ignores re-seeding from the rest of the brain, so it is a lower bound on lifetime and an upper bound on the onset
    g. "pooled" treats the whole active state as one well-mixed pool of PR units, which is an upper bound on lifetime for
    islands that do not interact. The coherent-LNA estimate uses the exact linear-noise variance of the total activity
    and a Gaussian tail, and is valid only when the QS mean is at least 2 SD above zero.
A3. **Resource closure.** One effective resource y per mode, with N_y set by the depletion balance at the QS state.
    Resource shot noise is neglected (deterministic drift between births). Efficacy uses the exact single-unit stationary
    e(ρ).
A4. **Inhibition (CAL only).** M1's empirical shrink factor c_r = K_needed/K_noD is transferred from the AL module of M1 to
    the M2 active state. This is untested, and is the weakest assumption in every r > 0 number.
A5. **Driven LNA.** QMF closure at r = 0 (no inhibition), Gaussian fluctuations about a stable fixed point, resources at
    their conditional means in the noise covariance. Validated on a 400-unit graph (section 6) but not on the connectome.
A6. **Pulse protocol.** Quasi-static mean resources between avalanches (a mean field over avalanches; real resources
    fluctuate from avalanche to avalanche). Poisson Galton–Watson durations. Within-avalanche depletion enters only
    through the cutoff S_dep (an order-of-magnitude scaling, prefactor 1).
A7. **Time units and parameters.** U = 0.25, TAU = 157, α = 0.906; the ORN drive h = 0.037926. Variants change only
    (U, TAU). M2s is not computed separately (section 2.1).

## 11. Post-hoc check against the STATE.md summary (written after the md5 sums above; not used for any prediction)

I had seen these summary lines before starting (section 0). I compare against the STATE.md text only. No experiment file
was opened. The synthesis lane should redo this from the experiment files.
- **Onset, r = 0** (sim 1.18).
  - Single-mode: 1.66 (+41%).
  - Pooled: 0.74 (−38%).
  - Coherent LNA: 1.22 (+3%), or 1.32 (+12%) with the other prefactor.
  - The pre-registered primary fails the ±10% rule. The bracket [0.74, 1.66] contains the simulation.
- **r-trend.** The simulated onset rises ×1.71 from r 0 to r 6. The single CAL theory rises ×2.77 (islands-max ×1.98);
  the M1-calibrated inhibition is too strong in M2 as well.
- **Where activity lives.** At onset the simulation has PR 13.6k–19.1k and AL 5.0–6.3%. The theory has PR 15.5k and AL
  5.2% (single onset), and AL 6.6% at g = 1.2. Agrees.
- **Variants** (onset relative to M2). Simulation: KW09 1.16–1.41, N15 0.73–0.85, tau30 0.49–0.71. Theory, single
  1.51 / 0.62 / 0.35; pooled 1.23 / 0.80 / 0.59. The direction is right; the single mode exaggerates the sensitivity.
- **Configuration null.**
  - The real-above-null reversal is predicted (×1.30–2.28) and is in the simulation (×1.15–1.42 at r ≥ 1).
  - Null onsets: theory 1.28–1.74; simulation 1.37–1.61.
- **Oscillation.** The damped quasi-cycle (P 60–150, never strong, var ∝ 1/N) agrees qualitatively with the simulated OSC
  points (P 53–152, coherence mostly 0.1–0.5, var ~ N^−1). My r = 0 coherence (≤ 0.07 near onset) is lower than the
  simulated values.
- **Lifetimes.** Early deaths and an exponential tail: agrees.
- **Driven.**
  - Allowed region at r = 0: simulation 0.216–0.968; theory 0.214–1.03.
  - At r ≥ 1 the simulated lower edge rises (0.317), which the first-order theory cannot give.
  - Peak of τ_int: simulation 0.69–0.96; theory 0.65.
  - No χ peak at the transition: agrees. AL Fano plateau: agrees.
  - MR m ≤ 0.94 per unit, never within 0.01 of 1: theory max 0.948, never within 0.01. Agrees.
- **Avalanches.**
  - No critical-like point in either model, so the verdict is "inconclusive": consistent with the predicted absence of
    critical-like points in M2.
  - The bump starts at 0.5–0.68× onset in the simulation; the theory has it at g ≈ 0.55–0.6 ≈ 0.5× of the simulated
    onset.
  - The simulated fitted τ_S passes through 1.3–1.7 at 0.16–0.32× onset (KS fails), in the theory's narrow power-law
    range (S ≲ 60–95).
- **FSS.**
  - Null W ratio from 5k to 167k: simulation 0.50; theory 0.57.
  - The real uniform-subsample family drifts in both, but the theory drifts much more strongly (onset 21.6 at 40k
    against a simulated 4.4 at 20k): dilution is overstated by the bQMF profile on small subsamples.
- **Modules alone** (r = 0).
  - AL: theory 2.08, simulation 2.19 (−5%).
  - Central brain: 1.67 against 1.42 (+18%).
  - Optic lobe: 5.22 against 5.0 (+4%).
  - VNC: 24 (single) or 1.86 (pooled) against 3.36. The single-mode VNC value is wrong.
- **L/R broken symmetry, coexistence, hysteresis that survives slow ramps.** None predicted, none found.
