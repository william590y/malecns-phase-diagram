# Lane m2/theory-stochastic: plan and pre-registration (written 2026-11-09 ~17:10, before any computation of this lane)

Spec followed: `results/m2/MODEL2.md` (model M2 = M1 + per-spike depression on ACh outputs, U 0.25, TAU 157, + ORN drive
h = 0.037926; variants M2_KW09, M2_N15, M2_tau30, M2_allsyn; protocols decay_h0, driven, aval_pulse). Simulator semantics
read from `code/ald_sim.py` (md5 61580c76): per step, recovery x <- 1 + (x - 1) q (q = e^{-dt/TAU}), then onsets take
e = x and deplete x <- (1 - U) x; the input uses the previous state's a_j e_j. My equations use exactly this order.

Outputs: derivations `results/m2/theory/stochastic.md`; predictions `results/m2/theory/stochastic_*.json`;
intermediate files `results/m2/theory/stochastic/`; code `code/m2_stoch_*.py`; jobs named `m2-stoch-*`; figures
`figs/m2/stoch_*.png`. The parallel mean-field lane writes `results/m2/theory/{args,runs,...}`; I do not touch those files.

## 0. Honest disclosure (blindness)

The task said to write predictions before reading experiment results. The task also told me to read the latest sections
of `STATE.md`, and those sections already summarise the finished M2 experiment lanes (decay, driven, modules, nature),
with their headline numbers (onsets per r, PR and AL share at onset, the OSC points and periods, early deaths in the
lifetime runs, avalanche fits, branching ratios, FSS widths). I read them before writing this plan. **These predictions
are therefore not blind.** What I do about it:
- I do **not** open any file under `results/m2/{decay,driven,modules,nature}/` until all prediction JSONs are written
  and their md5 sums are recorded in `stochastic.md`.
- No free parameter of the theory is fitted to M2 data. Every input is either a model constant (U, TAU, dt, alpha,
  beta, h), a connectome quantity, or a calibration taken from **Model 1** results (`results/analysis2/phase_complete.json`),
  fixed below before computing.
- The methods and the comparison rules below are fixed now. If I change a method after seeing its output, I log the change
  and the reason in `stochastic.md` ("deviations"), and I keep the first version's numbers too.
- In the final report every prediction carries one of two labels: **parameter-free** (no M1 calibration, no choice made
  after seeing anything about M2 beyond MODEL2.md) or **M1-calibrated**.

## 1. What the theory must deliver (task)

1. A spectral account of the depressed onset: which excitatory modes go unstable where; the depression-renormalised
   branching operator B(g, x*) on the active state; the effective module size K; lifetimes from a birth-death chain with
   a resource variable (2-D chain, plus the frozen and adiabatic limits as brackets) -> predicted S (tau = 200) and onset
   (tau = 3000) lines.
2. Driven fluctuations: linear-noise approximation (LNA) around the driven fixed point -> chi_nonORN, chi_AL, Fano-like
   chi/rho, tau_int, oscillation (period, coherence C(2P)/C(P)), the multistep-regression branching ratio m; and whether
   a Levina-type self-organised critical regime is predicted (pulse protocol: self-organised branching ratio sigma*(g),
   avalanche cutoffs).
3. Finite-size and module predictions where feasible.

## 2. Methods (fixed now)

**M-A. Per-mode renormalisation (parameter-free).** For a mode (lambda, right v, left u, <u, v> = 1) of the excitatory
branching operator, the one-mode logistic chain of M1 (theory_branch.py) is b(n) = lambda n (1 - n/K), d(n) = alpha n,
K_0 = 1/(<u^2, v><u, v^2>) without the coincidence term. With per-spike depression the birth rate is multiplied by the
mode-weighted efficacy X(n) = sum_j u_j v_j e(rho_j)/sum_j u_j v_j, rho_j = n <u^2, v> v_j, with e(rho) the exact
stationary efficacy of a unit (renewal formula, = ald_dep_theory.eff_spike). Expanding, e = 1 - kappa rho + ...,
kappa = U p_off q/(1 - q) G_L(q) ~= U TAU alpha_eff = 35.6, so the small-amplitude effective size becomes
K_dep = K_0/(1 + kappa) (exact for modes whose left vector vanishes off excitatory neurons, which holds for every mode of
the excitatory operator). Computed for the top ~60 real modes of E (left and right eigenvectors), with lambda_k(g) from the
Rayleigh quotient of f(gE).

**M-B. QMF with exact resources (setting a and b).** Per neuron: rho (activity); per depressing neuron also
O = E[a e], R1 = E[a x], R0 = E[(1 - a) x]. The one-step mean map (simulator order) is exact for a single unit at fixed
input; the network closure is the usual quenched mean field p_i = 1 - exp(-beta tanh(s_i)_+ dt), s_i = g sum_j c w_ij
sigma_j out_j + h_i, out_j = O_j (depressing) or rho_j. Fixed points by damped iteration of the stationary map.
Setting (a): h = 0, from a fully active start (high branch); setting (b): ORN drive, from a quiet start (up-sweep) and
from above (down-sweep), to flag mean-field bistability.

**M-C. Renormalised branching operator and lifetimes (setting a).** At the QMF active state: B~ = f(g E diag(e*)), e*_j =
O_j/rho_j; Perron triple (lambda~, u~, v~); sigma~ = lambda~/alpha_eff (frozen-resource branching ratio);
lambda_rest = <u~, f(gE) v~> (undepressed gain of the same mode); X* = lambda~/lambda_rest; K = c_r K_0(u~, v~).
- c_r (inhibition shrink factor) = M1's K_needed / K_noD from `results/analysis2/phase_complete.json` (median K needed
  over M1 slow-window points divided by M1's no-coincidence K), interpolated in r. This is the **M1-calibrated** variant.
  The **parameter-free** variant uses c_r = 1 (inhibition enters only through the QMF state).
- n* = K (1 - 1/sigma~). Three lifetimes from the QS state:
  (F) frozen resources: logistic chain (lambda~, K);
  (A) adiabatic resources: b(n) = lambda_rest X_ad(n) n (1 - n/K), X_ad from the profile v~ with the exact e(rho);
  (2D) piecewise-deterministic 2-D chain in (n, y): b = lambda_rest y n (1 - n/K), d = alpha n, each birth depletes
       y by U y / N_y, dy/dt = (1 - y)/TAU between births; N_y = U TAU alpha X* n*/(1 - X*) so that its deterministic
       fixed point is (n*, X*). Mean extinction time from the QS state by a sparse linear solve.
  The 2D value is the primary prediction; F and A are the lower and upper brackets.
- Crossings tau = 200 (S) and tau = 3000 (onset) interpolated in (ln g, ln tau). Grid: 40-60 log points per r in
  [0.2, 12]. r in R_CORE = 0, 0.25, 0.5, 1, 2, 4, 6. Variants at r = 0, 1, 4: KW09, N15, tau30, allsyn.
- Also reported at each g: PR of the QMF occupancy, AL-module share, region shares, n*, sigma~, K, X*, and the
  deterministic transient from the fully active start (minimum of the mode amplitude during resource recovery), which
  predicts early extinctions.

**M-D. Linear-noise approximation.** z = (rho, O, R1, R0) per depressing neuron, rho otherwise. One-step Jacobian A of the
mean map at the fixed point; per-neuron noise covariance Q from the Bernoulli on/off events (resource given activity at
its conditional mean). Stationary covariance C = A C A^T + Q. For an observable c^T z: Var = sum_s w_s^T Q w_s with
w_s = (A^T)^s c; ACF(k) = c^T A^k (C c), with C c from a checkpointed reverse (Horner) pass. Observables: nonORN
activity, AL-module activity, nonORN onset count (for the MR estimator, with the same-step onset/state noise
cross-covariance). Derived: chi = n Var (per MODEL2 5b), chi/rho, tau_int (m2_obs.tau_ac rule on the exact ACF), the OSC
rule of m2_obs.osc_metrics applied to the exact ACF, the slowest relaxation time, m_MR = fit of r_k = b m^k over k = 1..20
steps (m2_obs.mr_branching). Setting (b) at ORN 0.04 for r in R_CORE; setting (a) QS states at selected g.

**M-E. Pulse protocol (Levina test).** Quasi-static mean resources x_j between avalanches; branching matrix
M(x) = f(g E diag(x))/alpha; expected onsets per avalanche F = (I - M)^{-1} s (s = uniform over ORNs); mean avalanche
duration from a Poisson Galton-Watson recursion with mean rho(M); per-neuron onset rate nu_j = F_j/(T_av + T_gap);
x_j = 1/(1 + U TAU nu_j); solved self-consistently -> self-organised eps = 1 - rho(M(x*)) for gap G = 0, 10, 100.
Avalanche cutoffs: branching cutoff S_eps ~ 1/eps^2; within-avalanche depletion cutoff S_dep ~ (K_s/U)^(2/3) with
K_s = sum v / <u, v^2> of the self-organised mode (derivation in stochastic.md). Predicted "critical-like" (MODEL2 rule
needs largest/xmin >= 100 with xmin <= 10) where min(S_eps, S_dep) >= 1000.

**M-F. Finite size and modules.** Configuration nulls N = 5k...167k (m2_common.write_graph, my own copies) and isolated
modules (AL module, AL left lobe, AL right lobe, and superclass blocks: central brain, VNC, optic lobe) through M-B/M-C
(c fixed at the full-brain value): onset, S, width W = ln(on/S), and for modules the isolated onset vs the full brain.
Scaling statement for W(N) from the chain (W ~ 0.74/sqrt(K) near sigma~ = 1).

## 3. Pre-registered comparison rules (for the synthesis; this lane does not grade itself before writing predictions)

- **Boundaries (a):** a predicted S or onset "agrees" if it lies inside the simulated total interval or within +-10% of
  the simulated value (the M1 r = 0 standard was 5-9%). The r-trend agrees if the predicted ratio on(r = 6)/on(r = 0) is
  within +-20% of the simulated ratio.
- **Where activity lives:** predicted PR at the simulated onset within a factor 2; AL share within a factor 2.
- **Driven:** chi_nonORN and tau_int within a factor 2 at >= 70% of g points in the data-allowed region; the presence or
  absence of an interior peak agrees.
- **OSC:** predicted period within +-30% where both flag OSC; predicted OSC-strong (coherence >= 0.7) anywhere counts as a
  failure if the simulation has none.
- **Branching ratio:** predicted m_MR within 0.01 of simulated where both exist; predicted verdict "|1 - m| < 0.01 over a
  factor 1.5 in g" (yes/no) agrees.
- **Levina:** predicted verdict (supported / not supported / inconclusive) under MODEL2's rule agrees.
- **FSS:** predicted sign of dW/dN and null onsets within +-15%.
- A prediction that agrees only in its M1-calibrated variant is reported as such.
