# Model 2 (M2): the model, its variants, the phase definitions, the flags and the grids

Written 2026-11-09 by lane m2/sim. This is the shared spec for restarting the phase-diagram theory and experiments.
- **Why.** `results/aldef/REPORT.md` (verified) found that the self-sustained antennal lobe (AL) of the old model is a
  defect of its synapse dynamics, so phase diagrams are rebuilt on a corrected model.
- **Code.** Simulator `code/ald_sim.py` (options added for M2; defaults unchanged and re-tested bit-identical with
  `code/swc.py`). Shared helpers in `code/m2_common.py` (flags, protocols, grids, regions). Reference observables in
  `code/m2_obs.py` (lifetimes, crossings, PR, oscillation flag, avalanches, branching ratio). Array runner
  `code/m2_array.sbatch`.
- **Rule.** Every lane builds its argument lines with `m2_common.args_line()` and classifies with `m2_obs`, so all
  lanes use the same criteria.
- Section 9 gives the validation results and per-run costs. All paths are relative to `/MALECNS_ROOT`.

---------------------------------------------------------------------------------------------------------------------

## 1. The model

**Units and update.** One binary unit a_i ∈ {0, 1} per neuron of the MaleCNS v1.0 connectome (N = 167,216; 24,562,483
nonzero edges in the operator at r = 1). Synchronous steps of dt = 0.2 model time units. In each step, using the old
state:
- an active unit turns off with p_off = 1 − e^(−dt) = 0.1813, so α_eff = p_off/dt = 0.906;
- a quiescent unit turns on with p_on,i = 1 − exp(−β tanh(s_i)₊ dt), β = 1;
- **s_i = g Σ_j c w_ij σ_ij a_j e_j + h_i**:
  - w_ij = synapse count (linear);
  - c = 1/742.7 = 1/(mean in-strength over neurons with input) (`--norm raw`);
  - σ_ij = σ_j = +1 for ACh, −r for GABA/Glu/His, 0 for DA/OA/5-HT/unclear (consensus NT of the release);
  - e_j = efficacy of j's current active epoch (below);
  - h_i = drive.

**(1) Short-term depression (Tsodyks–Markram per spike, `--std_mode spike`)** on the outputs of every presynaptic neuron
with consensus sign +1 (ACh, n = 103,746). Inhibitory (59,273) and modulatory (4,197) outputs are not depressed.
- Each neuron j carries a resource x_j ∈ [0, 1], with x_j = 1 at rest.
- At every 0 → 1 transition of j: e_j ← x_j, then x_j ← (1 − U) x_j. The output during that active epoch is e_j.
- In every step x_j recovers: x_j ← 1 + (x_j − 1) e^(−dt/TAU).
- **U = 0.25, TAU = 157 model units.** Source: Nagel & Wilson 2016, ORN→LN, per-spike retention f = 0.75 and τ_rec = 1566 ms, read at
  τ_u = 10 ms per model unit (`results/aldef/lit_biophysics.md`, `lit_biophysics_calc.json`).
- Flags: `--std_group "sign:1=0.25,157" --std_mode spike`.
- A fully recovered neuron transmits undepressed, so the linearisation about silence is unchanged. The linear onset stays
  **g_lin = 0.1632**, exactly as in M1.
- Poisson mean field: x* = 1/(1 + U ν TAU) with ν = α_eff ρ. For example, ORNs at ρ = 0.04 sit at x* ≈ 0.41, to compare with
  0.30–0.60 for ORN→PN at 3.44 Hz in the literature (`lit_biophysics_calc.json`).
- Neurons active at t = 0 (fully active start) count as an onset at t = 0 (e = 1, x = 1 − U).

**(2) Spontaneous ORN drive.**
- All 2,639 ORNs (class `olfactory`) get h = 0.037926, which gives an isolated unit activity 0.04, i.e. 3.6 Hz at τ_u = 10 ms.
- Every other neuron has h = 0.
- Flag: `--drive olfactory=0.037926`; `m2_common.h_of_rho(ρ)` gives h for any level.
- The ORNs also receive recurrent input, including LN inhibition onto ORN terminals, which the connectome has as
  synapses onto ORNs. So their actual activity is not fixed by h. In the smoke runs (M2s, r = 1) the ORN activity was
  0.037 at g = 0.5 and 0.030 at g = 1.5 (25% below 0.04).
- **"ORN level" therefore names the isolated-unit drive, not the measured ORN rate.** Every gain uses the measured
  ρ_ORN, and lanes report ρ_ORN next to every driven point.

**Kept from M1.** Linear synapse counts, the released transmitter labels, raw normalisation, dt, α and β.

**Time mapping.**
- 1 model unit = τ_u = 10 ms. This is Benayoun et al. 2010's choice and mapping A of `lit_biophysics.md`.
- Rate ν (Hz) = α_eff ρ / τ_u = 90.6 ρ.
- TAU = τ_rec/τ_u.
- T = 3000 units = 30 s.

**Not in the model (known omissions).** Electrical coupling (eLN→PN is electrical in the fly), neuromodulation, LN→LN
depression (unmeasured), facilitation (iLN→LN), adaptation and refractoriness (not data-constrained;
`results/aldef/REPORT.md`).

**Numerical notes.**
- `--engine push` (event-driven product) equals the default spmm product up to float32 summation order. Its trajectories are
  statistically, not bitwise, reproducible: index_add_ atomics.
- The float32 resource update `1 + (x − 1)q` carries an absolute error of about 6e-8 per step. This is negligible for x ≳ 1e-3.

## 2. Variants (flags from `m2_common.MODEL`; the drive is part of the protocol, section 4)

| name | what changes | flags (in addition to the protocol) | why |
|---|---|---|---|
| **M1** (reference) | no depression | (none) | the old model; results/analysis2, pd3 |
| **M2** (default) | per-spike depression on excitatory outputs, U 0.25, TAU 157 | `--std_group "sign:1=0.25,157" --std_mode spike` | aldef recommendation |
| **M2s** (comparison) | M2 + per-target lLN1_bc signs: lLN1_bc→ALPN = 0, lLN1_bc→ALLN = +1 | M2 + `--edge_sign "lLN1_bc->ALPN=0" --edge_sign "lLN1_bc->ALLN=1"` | real eLN→PN coupling is electrical, eLN→LN chemical excitatory. Removes 6,658 edges (79,142 synapses). With consensus labels lLN1_bc→ALLN is already +1 (4,503 edges, 141,119 synapses), so the second flag pins it (it matters under sign ensembles). The other 4,825 lLN1_bc out-edges (8,784 synapses; 3,539 onto ORNs) keep +1. |
| M1s (attribution) | signs fix alone, no depression | the two `--edge_sign` flags | separates the two fixes |
| M2_allsyn | depression on every presynaptic neuron | `--std 0.25 157 --std_mode spike` | aldef E3 main arm (NW16): it also depresses inhibition |
| M2_KW09 | U 0.28, TAU 240 (Kazama & Wilson 2009 ORN→PN, α 0.72, 2.4 s) | `--std_group "sign:1=0.28,240" --std_mode spike` | parameter sensitivity |
| M2_N15 | U 0.22, TAU 89 (Nagel et al. 2015 ORN→PN, f 0.78, 893 ms) | `--std_group "sign:1=0.22,89" --std_mode spike` | parameter sensitivity |
| M2_tau30 | TAU 30 (τ_u = 50 ms mapping: 1566/50 = 31) | `--std_group "sign:1=0.25,30" --std_mode spike` | time-mapping sensitivity. Under this mapping ORNs at 3.44 Hz would be ρ ≈ 0.19, beyond our ORN grid; report ORN 0.10 as the closest level. |
| ORN levels | isolated-unit ρ_ORN ∈ {0.01, 0.02, 0.04, 0.07, 0.10} (0.9–9.1 Hz) | `--drive olfactory=` 0.009164 / 0.018533 / 0.037926 / 0.068798 / 0.102086 | drive sensitivity |
| configuration null | targets permuted (degrees and weight multiset kept) | `--null --null_seed S` | as swc.py. With `--null_seed` the graph is fixed across `--seed`, and the sidecar `null_edges_md5` identifies it. |
| weight-shuffled null | wiring kept, synapse counts shuffled | `--null --null_kind weights --null_seed S` | as swc.py |
| FSS graphs | pd3 configuration nulls of size N and real subsamples keeping the AL (c at the full-brain value) | `--graph FILE` from `m2_common.write_graph(kind, N, seed, path)` | finite-size scaling |

Reference values already measured (aldef, E3 arm `spg_exc_r1` = exactly M2's synapse model, h = 0, fully active start,
r = 1, `results/aldef/depression/key_numbers.json`):
- S (τ = 200) 1.24 [1.21, 1.29], onset (τ = 3000) 1.57 [1.53, 1.63]. M1: 0.283 / 0.344.
- AL share at the first sustained g: 6%. PR/N there: 0.089.
- E1 protocol at ORN 0.04, r = 1 (`depression/summary.json` → driven/rm_spg_exc_r1): data-allowed g ≈ 0.32–1.24. The upper end is
  interpolated across the coarse grid cell 1.15–1.5 (verification correction 3).

## 3. Initial conditions

- `--exp decay`: every unit is active at t = 0. Depressing units start with e = 1, x = 1 − U.
- `--exp quiet`: every unit is silent at t = 0, x = 1.
- With h = 0 everywhere, silence is absorbing. The run stops early when every replica is silent; this is swc.py's convention.

## 4. Protocols (flags from `m2_common.protocol(name)`; all use `--norm raw --engine push --push_max 0.012`)

| protocol | what | key flags | outputs used |
|---|---|---|---|
| **decay_h0** (setting a) | h = 0 everywhere, fully active start, T = 3000, B = 64 | `--exp decay --T 3000 --reg_last --win 1500:3000 [--rec_counts --counts_win 1500:3000 [--counts_reg]]` + standard regions | t, surv (extinction times), occ (second half), win_mean/win_m2, reg_rho, cnt_act |
| **driven** (setting b) | quiet start, ORNs driven for 0 < t ≤ 1500, then every ORN clamped to 0 until t = 4500 | `--exp quiet --T 4500 --drive olfactory=h --switch_at 1500 --sw_clamp_off olfactory --rec_lin 10 --reg_last --win 500:1500 --win 1600:1800 --win 2500:4500 --occ_win 500:1500 [--rec_counts --counts_win 500:1500 [--counts_reg]]` | win_mean/win_m2 (driven window 500–1500; removal windows), occ (driven window), reg_last (post-removal extinction), cnt_* |
| **aval_pulse** | h = 0, quiet start; one ORN seeded whenever a replica is silent, after a gap ~ Geometric with mean ≈ G (G = 0: next step) | `--exp quiet --T 3000 --pulse olfactory --pulse_gap G --rec_counts --no_occ` | cnt_act, cnt_on → avalanches |
| **ramp_driven** (hysteresis) | quiet start, ORN drive, g up through the grid then back down, dwell D per g, state and resources carried | `--exp quiet --drive olfactory=h --ramp D --ramp_loop --ramp_avg 0.5`, ascending `--g` | seg_g, seg_rho, seg_rho2, seg_reg, seg_x, seg_alive |
| **ramp_h0_down** | h = 0, fully active start at the top g, down then up. Pulses (gap G) restart activity after extinction, because otherwise the up-branch can never ignite. | `--exp decay --ramp D --ramp_loop --pulse olfactory --pulse_gap G`, descending `--g` | as above |

**Palp-intact removal (optional, E1 'ant').** Replace `--sw_clamp_off olfactory` by
`--sw_clamp_off "class:olfactory&~type:ORN_(VA4|VC1|VC2|VA7l|VM7d|VM7v)"`.

**Standard regions** (`m2_common.REGIONS`; added to ald_sim's default AL_module / AL_rest / CNS_rest). All index files hold
global ids, so they also work on `--graph` subgraphs:
- ORN (`olfactory`, 2,639);
- nonORN (`~olfactory`, 164,577);
- uPN_ant (220 antennal uniglomerular ACh PNs, `results/aldef/drive/groups/uPN_ant.npz`);
- ALsys (AL without ORNs ∪ AL module, `groups/ALsys.npz`);
- lLN1_bc (30).

## 5. Phase definitions

All lifetimes, crossings and PR use `code/m2_obs.py`. The thresholds below are conventions on smooth crossovers, as in
M1. g_lin is the only parameter-free line.

### 5a. Setting (a): h = 0, decay from a fully active start (directly comparable with M1)

Per (g, r), from B = 64 replicas, T = 3000:
- **Lifetime.** τ = censored-exponential MLE of the per-replica extinction times: Σ min(t_i, T) / deaths, with t_i the first
  recorded time with no active unit, on the default log record grid as in M1 (`m2_obs.extinction_times`, `tau_mle`).
- **Q** (quiescent): τ < 200.
- **S** (slow window): 200 ≤ τ < 3000.
- **L** (localized sustained): τ ≥ 3000 and PR < N/4.
- **D** (delocalized sustained): τ ≥ 3000 and PR ≥ N/4.
- **PR.** PR = (Σ_i o_i)² / Σ_i o_i² of the replica-mean occupancy o_i over the second half (t > 1500; `occ`), over all N
  neurons as in M1.
- **Boundaries.** Isotonic regression of ln τ (or PR) in g, then the first crossing interpolated in ln g
  (`m2_obs.first_crossing`). Uncertainty = the envelope of the bracketing grid cell and a 95% replica bootstrap
  (2000 resamples), as analysis2.

**Oscillation flag (O).** Computed in the quasi-stationary window 1500 < t ≤ 3000, on the replicas still alive at T
(at least 4 such replicas, else "n/a"). The signal is x_b(t) = per-step active fraction (`cnt_act / N`; with
`--counts_reg` also the AL module and nonORN). Using `m2_obs.osc_metrics` on the replica-averaged autocorrelation C(τ):
- **Trough and peak.** The trough C_min is the first local minimum of C that is at least half as deep as C's global
  minimum (lags 1 … window/2). This rule ignores noise wiggles. The peak C_peak is the maximum of C over lags
  (lag_min, 3·lag_min], and its lag is the period P.
- **OSC** if all three hold:
  - C_min ≤ −0.10;
  - C_peak ≥ 0.10;
  - P ≤ window/4 = 375 units (at least 4 periods in the window).
- **OSC-strong** (weakly damped, limit-cycle-like): OSC and C(2P)/C(P) ≥ 0.7, i.e. coherence time ≥ 2.8 periods.
  White measurement noise only lowers C at lag 0, so this ratio does not depend on the noise level.
- These thresholds were revised once, on 2026-11-09 before any production run. The first version used the first local
  minimum and "C_peak ≥ 0.5" for strong; on smoke data it latched onto noise wiggles, and on a synthetic test it called an
  undamped noisy oscillation "not strong". Tests: `m2_obs_test.py` O4 (AR(2) damped: OSC, not strong; undamped noisy
  sine: strong; white noise and AR(1): no OSC).
- **macroscopic vs quasi-cycle:** an OSC point counts as a *population oscillation (bursting)* if CV_t = std_t/mean_t of
  x_b ≥ 0.3. Otherwise it counts as a *noise-sustained quasi-cycle*: Benayoun et al. 2010 showed such quasi-cycles in the
  stochastic Wilson–Cowan model; their amplitude falls as 1/√N, while their normalised ACF does not.
- Any OSC point is checked on one smaller FSS graph (section 6). A macroscopic oscillation keeps var_t(x) as N grows;
  for a quasi-cycle, N·var_t(x) stays constant.
- Also reported: P, C_peak, the periodogram peak frequency and its peak/background ratio, τ_int (integrated autocorrelation time).

**Where activity lives** (at every sustained point and at the first sustained g):
- the AL-module share Σ_module o_i / Σ o_i;
- the shares of the standard regions;
- the top 5 superclasses by share, with enrichment = share / size fraction;
- the QS active fraction per region (`win_mean` over 1500–3000 among replicas alive at T).

### 5b. Setting (b): driven by ORNs (silence is no longer absorbing)

Measured per (g, r, ORN level), driven window W_d = (500, 1500]:
- **Driven activity.** ρ_drv = replica mean of `win_mean[nonORN]` in W_d. ρ_ORN = the same for ORN.
- **Gain.** G = ρ_uPN_ant / ρ_ORN (E1 definition; unit-free).
- **Fluctuation.**
  - χ_X = n_X ⟨var_t ρ_X⟩_b for X = nonORN and AL_module, with var_t = `win_m2 − win_mean²` per replica in W_d.
  - χ_X / ⟨ρ_X⟩ (Fano-like) is reported too.
  - The near-critical-amplifier marker is the interior maximum of χ_AL/⟨ρ_AL⟩ along g (bootstrap over replicas, as
    analysis2's driven plane).
- **Autocorrelation time.** τ_int and τ_e of ρ_nonORN(t) from `cnt_act` in W_d (`m2_obs.tau_ac`; counts include the ORNs,
  so with `--counts_reg` use the nonORN region).
- **PR.** PR_nonORN = PR of the occupancy over the 164,577 nonORN neurons in W_d (`--occ_win 500:1500`). Delocalized
  (modifier **-D**) if PR_nonORN ≥ 164,577/4.
- **Oscillation.** OSC flag as in 5a, on ρ_nonORN(t) in W_d (window 1000 units, so P ≤ 250).
- **Stationarity check.** |mean(500–1000) − mean(1000–1500)| ≤ 10% of the mean. Otherwise the point is "non-stationary"
  and reported as such. This needs `--win 500:1000 --win 1000:1500` or the counts.
- **Deafferentation** (every ORN clamped to 0 at T1 = 1500, then h = 0 everywhere, so silence is absorbing):
  - τ_post = censored MLE of the per-replica extinction time of nonORN activity, t = `reg_last[nonORN]` − 1500. A replica
    is censored at 3000 if it is still active at T = 4500.
  - S1 (E1): mean uPN_ant activity over 1600–1800 (100–300 units after removal) ≤ 10% of its driven value.
  - S2 (E1): ≤ 5% of replicas (≤ 3/64) have ALsys activity at T (`reg_last[ALsys]` = 4500).
  - The **removal class** is assigned as follows:
    - **silent**: τ_post < 200, S1 and S2 all hold;
    - **self-sustained**: τ_post ≥ 3000;
    - **transient**: anything else.

**Driven phases** (assigned in this order):

| phase | definition |
|---|---|
| **SS** self-sustaining | removal class self-sustained (τ_post ≥ 3000). Sub-labels **SS-L** (PR_nonORN < N_nonORN/4) and **SS-D** (≥). |
| **DT** driven, slowly reverberating | removal class transient (activity outlives the input by more than 200 units = 2 s, or S1/S2 fail, but dies within 3000) |
| **DA** driven, amplifying | removal class silent and G ≥ 0.29 |
| **DQ** driven-quiet | removal class silent and G < 0.29 (PNs below 1 Hz at ORN 3.44 Hz) |
| modifiers | **-O** (OSC flag), **-D** (PR_nonORN ≥ N_nonORN/4) on any phase; **-NS** non-stationary |

**Data-allowed region:** G ∈ [0.29, 1.45] AND S1 AND S2. This is E1's primary criterion: PN 1–5 Hz over ORN 3.44 Hz,
Kazama & Wilson 2009, and PNs silent after removing every ORN (`results/aldef/drive/plan.md`). It is reported at ORN 0.04
(primary) and at the other levels. DA ∩ {G ≤ 1.45} is the allowed region. The broad gain window [0.077, 1.67] and the
palp-intact test are reported alongside, not used for the main region.

### 5c. Comparing (a) and (b)

Report, per r:
- g_lin;
- the (a) boundaries S, on and D;
- the (b) boundaries: DQ→DA (G = 0.29), the top of the allowed region, DA→DT (silent lost) and DT→SS (τ_post = 3000);
- the χ_AL maximum.

Expectation from aldef, to be tested, not assumed: the DT→SS line lies near the (a) onset, and the top of the allowed
region lies at or below the (a) S line. In E1 it was set by the τ = 200 boundary or by the gain ceiling.

## 6. Observables for the nature of the transitions

Pre-registered criteria. Each lane copies them into its plan.md before running and may only tighten them.

**Finite-size scaling.**
- Graphs:
  - configuration nulls of size N ∈ {5k, 10k, 20k, 40k, 80k, 167k}, 3 seeds each (`write_graph("cnull", N, seed)`);
  - real subsamples keeping the AL module, N ∈ {20k, 40k, 80k} plus the full brain, 2 seeds each (`write_graph("realsub", ...)`;
    c fixed at the full-brain value).
- Measure: the width of the setting-(a) crossover W(N) = ln(g_on/g_S).
- A sharpening transition means W falls monotonically with N (Spearman ρ ≤ −0.9) and W(167k)/W(5k) ≤ 0.5.
  A "finite-K-rounded crossover" means W stays within ±20% across the realsub sizes. The same two outcomes were
  found for M1 in pd3; compare with `results/pd3/redraw_fss.md`.
- Driven: the height of the χ_AL (or χ_nonORN) maximum against N, as a growth exponent.

**Hysteresis** (ramp_driven and ramp_h0_down; dwell D ∈ {50, 200, 800}; B = 32).
- Loop area A(D) = Σ_g |⟨seg_rho⟩_up − ⟨seg_rho⟩_down| Δ ln g over the visited g, with the same replicas going up and
  then down. Use the nonORN region (seg_reg) for the driven ramp.
- Noise floor: a label-permutation test, swapping up/down per replica and g at random (2000 permutations).
- **First-order (hysteresis)** if A(800) exceeds the 99th percentile of its permutation distribution AND A(800)/A(200) ≥ 0.7,
  i.e. the loop does not shrink with slower ramps.
- A shrinking loop (A(800)/A(200) < 0.5) is a finite-rate effect. The pd3 M1 result for comparison: loops shrank 3.6–15.5×
  between D 50 and 800.
- Also report the bimodality of the per-replica seg_rho at fixed g. Bimodal = bimodality coefficient > 0.555 with
  both modes holding ≥ 10% of the replicas.

**Avalanches.**
- (i) Pulse protocol (h = 0, separation of timescales):
  - an avalanche = a maximal run of steps with activity;
  - size S = number of onsets including the seed; duration T in steps (`m2_obs.avalanches`).
- (ii) Driven regime: thresholded avalanches of the nonORN count in W_d. Threshold = the replica's median count; size =
  area above the threshold (Poil et al. 2012). Also the multistep-regression branching ratio m of the nonORN onsets
  (`m2_obs.mr_branching`, Wilting & Priesemann 2018).
- Fits: P(S) ~ S^−τ_S and P(T) ~ T^−τ_T (discrete power-law MLE with Clauset's KS scan, `powerlaw_fit`), and ⟨S⟩(T) ~
  T^γ (`mean_size_given_T`). The crackling-noise prediction is γ = (τ_T − 1)/(τ_S − 1). Mean-field DP values:
  τ_S = 3/2, τ_T = 2, γ = 2.
- **Levina, Herrmann & Geisel 2007 test** (Nat Phys 3:857: depressing synapses self-organise avalanches to criticality
  over a wide range of coupling).
  - A g counts as **critical-like** if the size fit gives τ_S ∈ [1.3, 1.7] with KS ≤ 0.03, xmin ≤ 10, n_tail ≥ 1000 and
    largest size / xmin ≥ 100. The same rule is used for M1 and M2, pulse protocol, gap G ∈ {0, 10, 100}, R_CORE, the g grid
    0.1–3.
  - **LHG supported** if M2's critical-like g values span a factor ≥ 1.5 in g while M1's span < 1.2, or M1 has none.
  - **Not supported** if M2's span is < 1.2.
  - **Inconclusive** if neither model has any critical-like g. That is possible: pd3 found heterogeneous-network, not
    MF-DP, avalanches in M1.
  - In the driven regime, the analogous marker is |1 − m| < 0.01 over a g range of factor ≥ 1.5.

**Order parameter.** The QS active fraction (decay_h0, survivors, 1500–3000) against g. A first-order onset would show
a jump together with hysteresis or bimodality. A jump alone is not enough, because survivor-conditioning creates apparent jumps.

## 7. Grids

| grid | values | use |
|---|---|---|
| g coarse | `m2_common.G_COARSE`: 40 log points 0.05–30 (×1.178) | every protocol, stage 1 |
| g refine | `m2_common.refine(lo, hi, 8)`: 8 log points inside every bracketing cell of a boundary (×1.02) | stage 2 |
| r | `R_ALL` = M1's 23 values 0, 0.125, …, 1, 1.25, …, 3, 3.5, …, 6 | decay_h0 and driven at ORN 0.04 (the diagrams compared with M1) |
| r core | `R_CORE` = 0, 0.25, 0.5, 1, 2, 4, 6 | avalanches, ramps, FSS, variants, the other ORN levels. These 7 points span M1's range at the r values where pd3 ran its expensive programs. |
| ORN level | 0.04 main; 0.01, 0.02, 0.07, 0.10 | driven protocols |
| replicas | B = 64 (decay, driven, pulse); 32 per ramp | the τ MLE needs deaths: 64 replicas give a ±25% (1 SE) τ at 16 deaths |
| T | decay 3000 (= M1, τ thresholds 200 / 3000); driven 1500 + 3000; pulse 3000 per task (pool tasks); ramp dwell 200 (50, 800 for the D test) | |

Expected location of the M2 boundaries (aldef E3, r = 1): S 1.24, onset 1.57, data-allowed g 0.32–1.24. The 0.05–30 grid
covers them with margin, and also the r = 6 onsets, which were 1.7× the r = 1 value in M1.

## 8. New simulator flags (ald_sim.py section 8 of the docstring; all off by default)

| flag | meaning |
|---|---|
| `--edge_sign "PRE->POST=V"` | per-edge sign for edges from selector PRE to selector POST (V ≥ 0 as is, V < 0 → V·r); repeatable, later wins; applied after `--sign_override` and on every ensemble draw. Edges with coefficient 0 are dropped. The sidecar records edges and synapses changed. |
| `--null_seed S` | fixes the null-graph permutation (default: the run's RNG, swc.py behaviour); sidecar `null_edges_md5` |
| `--graph FILE.npz` | another graph (pre, post, w, sign; optional sub = global ids for labels, c). Without sub, only all/none/sign:/idx: selectors work and the default regions are off. |
| `--rec_counts`, `--counts_win T0:T1`, `--counts_reg` | per-step int32 counts: cnt_act / cnt_on [g, s, B] (+ cnt_reg_act / cnt_reg_on [g, s, R, B]), cnt_t |
| `--pulse SEL`, `--pulse_gap G` | seed one SEL neuron in each replica that is silent, with probability 1 − e^(−dt/G) per silent step (G = 0: always). The silent interval is Geometric with mean dt/(1 − e^(−dt/G)) ≥ one step. Seeds are ordinary onsets (they deplete). Refractory or clamped neurons are not seeded. Separate RNG, so the main stream is untouched. No early stop. |
| `--ramp D`, `--ramp_loop`, `--ramp_avg F` | one trajectory through the `--g` list (dwell D each; loop = and back); T = n_seg·D. Outputs seg_g, seg_rho, seg_rho2, seg_alive [n_seg, B], seg_reg, seg_reg2 [n_seg, R, B], seg_x, seg_xa [n_seg, R], averaged over the last fraction F of each dwell. |
| `--init SEL`, `--init_frac F` | (needs `--exp quiet`) start with SEL active, everything else silent; with F < 1 each SEL neuron is active with probability F per replica (separate RNG). Initial actives count as an onset (x = 1 − U). For partial starts such as one AL lobe (`module&side:L`) or random densities (de Candia-style module phases). |

Existing options used by M2 (unchanged): `--std_group`, `--std_mode spike`, `--drive`, `--switch_at`/`--sw_clamp_off`,
`--win`, `--reg_last`, `--occ_win`, `--rec_lin`, `--region`, `--engine push`.

## 9. Validation and costs

### 9.1 Simulator versions and tests (all in `results/m2/sim/validation/`)

| md5 | what | tests |
|---|---|---|
| 92c2e0d3 | pre-M2 (every aldef lane); kept as `code/ald_sim.py.bak-m2-pre` | reference |
| 52293c18 | M2 v1: edge_sign, null_seed, graph, counts, pulse, ramp (kept as `results/m2/sim/dev/ald_sim_m2v1_52293c18.py`) | work copy: jobs 186100 (`*_workcopy_job186100.json`); live file: 186273 + 186308 (`*_live_52293c18_*.json`) |
| **61580c76** | **current `code/ald_sim.py`** = v1 + `--init/--init_frac` | work copy: 186475 (`*_work2_job186475.json`); live file: 187154 (`validate_m2.json`, `unit_tests_m2.json`, `obs_tests.json`) |

Diff against the pre-M2 file: `results/m2/sim/ald_sim_m2.diff`. Both swaps were atomic (os.replace) and guarded by an md5 check
of the live file.

- **Unit tests** (`code/m2_sim_test.py`, small graphs, exact or analytic expectations): 29/29 pass. They cover:
  - edge signs: exact zeros, r scaling, later-wins, on top of node overrides;
  - the graph file's own c;
  - null_seed hashes;
  - count identities and z-tests;
  - pulse renewal formulas (gap 0, gap 2, refractory, all exact or z < 5): sizes all 1, clamps respected, main RNG
    stream untouched, seeds deplete (relative error 1e-6);
  - ramp against an exact carried-resource recursion (max |z| 0.67), schedule/loop/T, single dwell = plain run bitwise,
    early stop;
  - init;
  - regression: 6 pre-existing simulate() configurations bitwise equal to the pre-M2 module, and the real-cache operator
    bitwise equal;
  - real-cache set sizes.
- **Observables** (`code/m2_obs_test.py`): 11/11 pass.
  - censored τ MLE;
  - crossings / isotonic fit;
  - PR;
  - OSC flag: damped AR(2) = OSC not strong; undamped noisy sine = strong; white noise and AR(1) = none;
  - avalanche extraction (exact);
  - critical Galton–Watson exponents: τ_S 1.51, τ_T 2.02, γ 1.89; subcritical rejected;
  - MR branching ratio within 0.01, also under 10% subsampling.
- **Real cache** (`code/m2_validate.py`):
  - V1: swc.py vs ald_sim.py bit-identical rho/surv/occ in 4 cases (decay raw r 1, spread from module seeds,
    in-norm weight-shuffled null, h = 1e-3) and on the trivial general path;
  - V2: pre-M2 vs current, every npz key bit-identical on 5 aldef option sets (E1, E3 spike and rate + adaptation +
    refractory + clamps, E2 signs + sqrt weights, type-level sign ensemble);
  - V3: trivial new options change nothing (identity edge sign, counts, `--null_seed 0` = `--seed 0`, single-dwell ramp,
    pulse on a never-silent run);
  - V4: M2 v1 vs current, all keys bit-identical on lines using edge_sign + counts_reg + null_seed, pulse + spike
    depression, ramp loop + drive + depression.

### 9.2 Smoke tests and per-run costs (array m2-sim-smoke 186175 + 186350, 21 tasks, r = 1, production lengths)

Files:
- `results/m2/sim/smoke/` (args `code/m2_smoke_args.py`);
- checks and the cost table: `code/m2_smoke_check.py` → `costs.json`, `costs.md`.

Every task ran to completion. Task 2 was preempted by Slurm after 3 s and rerun as 186350.

Sanity checks passed:
- the removal clamp zeroes the ORNs exactly;
- every pulse avalanche starts after a silent step and has size ≥ 1;
- ramp schedules are as requested;
- M2s removes 6,658 lLN1_bc→ALPN edges and pins 4,503 lLN1_bc→ALLN edges;
- the configuration null with `--null_seed 0` has the same hash in the smoke task and in the validation run;
- the FSS graph has N = 20,000 (2,885,953 operator edges).

The ORN check ("within 10% of 0.04") **failed at g ≥ 1.5**: measured ρ_ORN 0.030 (g 1.5) and 0.024 (g 3.0). This is
recurrent inhibition onto ORNs, a model property, not a bug; see section 1.

Costs below are wall-clock ms per step on one GPU (N = 167,216; 1 step = 0.2 units; graph setup ≈ 3 s). Measured on the
cards the jobs landed on (A40, A5000, A5500, A6000, RTX PRO 6000 Blackwell), so treat them as ±30%.

| protocol (M2 unless noted) | engine | B | ms/step (g values) | seconds per g for the full run | npz per g |
|---|---|---|---|---|---|
| decay_h0, T 3000 (15,000 steps), counts total 1500–3000 | push | 64 | 5.3 (g 0.8, dies) / 5.8 (1.6, at onset) / 9.7 (3.2, sustained) | ≤ 80 / 87 / 145 | ≈ 1.1 MB |
| same, spmm engine | spmm | 64 | 9.5 (1.6 and 3.2) | 143 | ≈ 1.5 MB |
| same + `--counts_reg` (8 regions) | push | 64 | 11.8 (3.2; A40) | 177 | 10 MB |
| decay_h0 M1 | push | 64 | 3.3 (0.3) / 6.5 (0.6) | 49 / 97 | 0.8 MB |
| decay_h0 variants at g 1.6 (allsyn / KW09 / N15 / tau30) | push | 64 | 8.0 / 4.2 / 7.7 / 9.7 (A5000) | 63–145 | ≤ 1.8 MB |
| nulls at g 1.6 / 3.2 (targets; weights) | push | 64 | 8.1 / 11.3; 6.6 / 11.5 (A5500) | 99–172 | 1.8 MB |
| **driven + removal, T 4500 (22,500 steps), counts_reg 500–1500** (the depression + drive speed check) | push | 64 | 4.7 (g 0.5) / 6.8 (1.5) / 11.4 (3.0) (A40) | 106 / 152 / 256 | 6.9 MB |
| driven, no counts_reg (M2s) | push | 64 | 3.9 (0.5) / 5.5 (1.5) (A6000) | 88 / 125 | 1.3 MB |
| driven, spmm | spmm | 64 | 11.4 (1.5; A40) | 257 | 1.6 MB |
| aval_pulse, T 3000, per-step counts (gap 0) | push | 64 | 5.3 (g 1.0) / 6.7 (1.5) | 80 / 100 | 2.6 MB |
| aval_pulse gap 10 / M1 gap 0 | push | 64 | 1.8 (1.5; Blackwell) / 3.1 (0.17, 0.25) | 26 / 46 | 1–2.7 MB |
| ramp_driven / ramp_h0_down, 12 g loop = 23 dwells × 100 units (11,500 steps) | push | 32 | 3.5 | 41 per loop (dwell 200: ~80; 800: ~330) | 0.4 MB |
| FSS configuration null N = 20,000 | push | 64 | 0.8–1.7 | 13–25 | 0.9 MB |

The speed check asked for (depression + ORN drive, N = 167k, B = 64, T = 3000 = 15,000 steps, one GPU) is
4.7–11.4 ms/step, i.e. **71–171 s per g value** (driven M2, A40). It rises with g as activity spreads.
- The push engine is 1.6–1.7× faster than spmm while activity is sparse (decay at the onset, driven at g 1.5) and equal
  once activity is dense (it falls back to spmm).
- Early-stopped decay runs (Q, low S) cost much less than the full-run figure.

**Sizing guide** (≈ 100–150 s per (g, r) point at B 64; pack about 8–12 g per task for 15–30 min tasks; use `%` limits around 40):

| program | points | GPU hours |
|---|---|---|
| decay_h0 diagram | 23 r × 40 g | ≈ 25–35 |
| driven diagram at ORN 0.04 | 23 r × 40 g | ≈ 35–50 |
| driven, other 4 ORN levels | R_CORE × 40 g each | ≈ 40 |
| avalanches (M1 + M2, 3 gaps, R_CORE × 20 g) | | ≈ 25 |
| ramps (R_CORE × 3 dwells × 2 protocols × 2 models) | 12-g loops ≈ 3; 40-g loops ≈ 11 (dwell 800: ≈ 1,100 s per loop) | ≈ 3–11 |
| FSS (6 sizes × 3 seeds × R_CORE subset × 20 g) | small graphs are 5–10× cheaper | ≈ 10 |

**Smoke observations** (illustrative only: one seed, few g, r = 1; not results):
- Decay M2 at g = 1.6 is at the onset: τ 5,450 (push) and 2,915 (spmm, other seed), against E3's onset 1.57.
- Driven M2:
  - g 0.5: DA, data-allowed, G = 0.40, τ_post 11;
  - g 1.5: DT, τ_post ≈ 1,790;
  - g 3.0: SS-L, PR_nonORN/n = 0.11.
- No OSC flag at any smoke point.
- The 100-unit-dwell driven ramp ends lower on the way down at every g, while the AL-module resource lags (0.70 → 0.43 →
  0.68). This is the expected finite-rate depression memory, so the dwell-scaling test in section 6 is essential.
- Pulse avalanches in M2 are already huge at g ≥ 1 (mean size 1,400 at g 1.0; at g 1.5 the network is active 99.9% of
  the time). The Levina test therefore needs the low end of the g grid (0.1–1).

### 9.3 What lanes must not do

- Never `cp` over `code/ald_sim.py`. Changes are merged into a work copy and swapped only if the live md5 matches; then
  `code/m2_validate.sbatch <new> code/ald_sim.py.bak-m2-pre <previous>` is rerun.
- Run jobs named `m2-<lane>-*`.
- Pin `--null_seed` for nulls that must be shared across tasks.
- Report ρ_ORN next to every driven point.
