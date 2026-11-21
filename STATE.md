# MaleCNS Griffiths-phase investigation — STATE

User request (2026-11-08): investigate maleCNS; recover a phase diagram with the **Wilson–Cowan model**, specifically a
**Griffiths phase** (proposed correlate of wakeful consciousness), and give a theoretical explanation from the
**spectrum** of the connectome.

## Data
- MaleCNS v1.0 (Janelia, released 2026-06-08, CC BY 4.0), preprocessed cache in `/MALECNS_CACHE`:
  memory-mapped `.npy` arrays `pre`/`post` (uint32 indices into `body_ids`), `weight` (synapse counts), `sign` (consensus
  NT: ACh +1, GABA/Glu/His −1, modulatory/unclear 0), `superclasses`, `types`, `positions` (µm), `sides`.
- 167,216 neurons, 25,587,572 directed edges, 124M synapses (minconf 0.5). Provenance: `~/malecns-flygym/docs/data.md`.
- Python: conda base (`/MALECNS_PYBIN/python`; numpy/scipy/torch). The flygym `.venv` has no torch.

## Plan
1. **Structure + spectrum** (`code/spectrum.py`, job 161353, thickstun CPU):
   - degree and strength tails;
   - top eigenpairs of the symmetrized excitatory matrix (weighted, binary, log-weight), the directed excitatory matrix,
     the signed matrix and the in-normalized signed matrix;
   - IPR localization of the leading eigenvectors and where they sit (superclass/type);
   - a directed configuration-model null (targets permuted).
2. **Dynamics, Wilson–Cowan on the neuron graph (GPU, torch sparse):**
   - τ dx_i/dt = −x_i + (1 − r x_i) S(g Σ_j W_ij x_j − θ + h_i) + noise, with signs from NTs (E/I);
   - control parameters: global coupling g, inhibition scale, threshold θ;
   - Griffiths diagnostics:
     - (a) spreading from localized seeds: average activity A(t) ~ t^−α with α varying continuously over a g range;
     - (b) the distribution of relaxation/survival times;
     - (c) susceptibility/variance across noise realizations over an extended g interval;
     - compare with the configuration-model null and with region-shuffled graphs.
3. **Phase diagram** in (g, inhibition) or (g, θ): quiescent, Griffiths (power-law relaxation, localized activity),
   active and oscillatory regions; locate the boundaries.
4. **Theory:**
   - the lower boundary ≈ the activation of the most localized eigenmodes (g ≈ 1/λ_max of the localized states);
   - the upper boundary ≈ the delocalized bulk;
   - relate the width to the IPR spectrum and the Lifshitz tail of the top eigenvalues (quenched mean field; Odor et al.;
     Moretti & Muñoz 2013);
   - for the signed and non-normal W, use pseudospectra and leading left/right eigenvector overlap.

## Results
### Stage 1: spectrum (job 161373, results/spectrum.json; 2026-11-08)
- **Heavy tails.**
  - In-strength: median 345 synapses, max 119,936 (CV 2.2).
  - Out-strength: max 140,897.
  - Edge weight: median 2, max 2,591.
  - 59% of synapses come from excitatory (ACh) neurons and 38% from inhibitory (GABA/Glu/His) ones.
- **Weighted excitatory matrix (symmetrized): STRONG MODULE LOCALIZATION.**
  - Top eigenvalues 4647, 4385 (a near-degenerate pair, probably left/right), then 2901, 2371, ...
  - The leading eigenvector puts 84% of its weight on 100 neurons and 99% on 1,000, all central-brain intrinsic:
    antennal-lobe local neurons (lLN1/lLN2 types) and projection neurons. That is an antennal-lobe module.
  - 21 of the top 100 eigenvectors have IPR > 0.01.
- **Binary graph: delocalized** (N·IPR 57, no eigenvector with IPR > 0.01). The localization comes from the synapse
  weights, not from the degrees. The log-weight graph is intermediate (N·IPR 546, same antennal-lobe module).
- **Configuration-model null (degrees and weights kept, wiring randomized).**
  - The leading eigenvalue falls 4647 → 1280 and its eigenvector delocalizes (IPR 0.0017).
  - 49 eigenvectors are hub-star localized, with IPR 0.17–0.3, i.e. a few neurons each.
  - So the real graph's localization is a strongly connected MODULE of about 100–1000 neurons, not single hubs.
- **Directed matrices.**
  - Excitatory: top Re 4158, same antennal-lobe localization.
  - Signed: top Re 3778 (IPR 0.02, antennal lobe), then complex pairs Re 1090 ± 1317i (oscillatory modes).
- **In-normalized signed matrix** (rows divided by in-strength, the input scaling of the dynamics model).
  - The top eigenvalues form a dense ladder just below 1: 0.914, 0.873, 0.847, 0.835, 0.828, 0.822.
  - Every one is extremely localized (IPR 0.44–0.49, about 2 neurons) in the optic lobe (Dm2, Mi15, R1-R6, Tm3, ...).
  - This is the classic rare-region / Lifshitz-tail picture. CHECK whether these are biological motifs or artifacts of
    neurons with a single dominant excitatory input.

### Stage 2: stochastic Wilson–Cowan (`code/swc.py`)
- Smoke test 161407: about 2 ms per step at B=16 and 11 ms per step at B=256 on an A6000.
  - Decay from fully active, in-norm, r=1: dies by T=100 at g ≤ 1; persists at g ≥ 2 (ρ = 0.7%, 3.9%, 7.5% at g = 2, 4, 8).
- Decay sweeps submitted (T=3000, dt 0.2, B=16):
  - in-norm, g 0.8–4: r=1 (161481), r=0.5 (161484), r=2 (161486), null r=1 (161489);
  - raw-norm, g 0.05–2 (log-spaced): r=1 (161491), null (161492).
  - Next: fit ρ(t) for a power law with a g-dependent exponent (Griffiths) vs exponential decay vs saturation.
  - Then spreading experiments and a finer (g, r) grid.
- **Decay sweeps (161481–161492), first read** (ρ(t) from a fully active start, B=16, T=3000):
  - **In-norm (every neuron's maximal input = 1).**
    - Real graph: dies in finite time up to g≈1.8 (die-out time grows from about 30 to about 2800); partial survival at
      g = 1.85–1.95; sustained above about 2.
    - Null: the same picture shifted slightly lower (threshold about 1.75).
    - The decays below threshold are fast (log-log slope −2 to −10), so there is no broad power-law window in this
      normalization. Inhibition r = 0.5 / 1 / 2 shifts the threshold only slightly (about 1.75 / 1.85 / 2.0).
  - **Raw norm (synapse counts / global mean in-strength; keeps the hub and module heterogeneity).**
    - Real graph: onset at g≈0.30–0.34, a slow-decay window g≈0.20–0.34 (slopes −0.5 to −2.6, finite die-out at 80–900),
      then ρ rises smoothly and slowly over g = 0.34 → 2 (from 1e-3 to 1e-2).
    - Null: onset at g≈1.06–1.2, a narrower slow window (0.93–1.06).
    - The real/null onset ratio, about 3.2, matches the λmax ratio 4647/1280 = 3.6. Activity onset is set by the
      localized antennal-lobe module (quenched mean field: g_c ≈ 1/λ_loc). Suggestive of a smeared, Griffiths-like
      window; needs finer g, more replicas and longer T.
- **Stage 2b, submitted 2026-11-08 ~19:00.** `swc.py` now also saves `occ`: per-neuron activity averaged over replicas in
  the second half of the run, for localization of activity.
  - Fine decay raw g 0.15–0.60, 31 points, T=8000, B=32 (161646); null g 0.60–1.40 (161655).
  - Spreading from single seeds, B=512, T=4000: raw g 0.20–0.45 (161657); null g 0.85–1.25 (161660).
- **Slow-window analysis** (2026-11-08 ~19:40; `code/analyze.py` → `results/analysis.{json,txt}`, figures in `figs/`, sent
  to the user):
  - **Decay below onset.** In both the real graph and the null, the fitted power-law exponent b varies continuously:
    - real −5.5 (g .15) → −0.44 (g .33);
    - null −4.7 (g .60) → −0.6 (g 1.08).

    A power law fits better than an exponential in both, so the continuously varying decay exponent alone does NOT
    separate the real graph from the null. The slow window (lifetime > 200) is wider in the real graph: g .255 → .345,
    ratio 1.35, against the null's .97 → 1.11, ratio 1.14.
  - **The decisive difference is the LOCALIZATION of the active state.**
    - Real graph, just above onset: the participation ratio is about 950 neurons, 52% of the activity is in the
      antennal-lobe module (the top-1000 neurons of the leading eigenvector), and the top 300 active neurons are 298
      cb_intrinsic. The PR grows only slowly with g (2,800 at g = 0.6), so activity stays localized over a broad g range.
    - Null: it delocalizes immediately at onset (PR about 75,000, 1% in the AL).
    - This is the dynamical counterpart of the spectral localization: a smeared, rare-region-dominated transition in
      the real connectome vs a sharp bulk transition in the null.
  - The activity maps (fig6) show occupancy spreading from the central brain/AL region outward into the optic lobes and
    VNC as g rises from 0.33 to 0.6.
  - Spreading: real-network run 161657 was PREEMPTED, so it was resubmitted on ellis as 163130 (g 0.30–0.60). The null
    spreading run is done: P(T) about 0.04–0.05 at g 1.15–1.25.
- **Submitted:** an upper sweep, raw g 0.6–4, to find where activity delocalizes, i.e. the bulk threshold (163140); movie
  snapshots of the real graph at g .25/.36/.6/2.0 (163143) and of the null at 1.15/2.0 (163145). Code `code/snap.py`;
  rendering is still to be written.
- **Explainer artifact** (user 11-08: "explained to me in an artifact"): https://claude.ai/artifact/VWAKYgXVrQ4n38XGW3X7Bs
  - Source: `explainer/index.html` (published from the session scratchpad; copy kept here) with `figs/*.png`.
  - Covers the question, the data, the model equations (MathJax), every figure with "how to read / what it shows", the
    theory sketch, caveats, open runs and a glossary.
  - Republish from the same scratchpad path in the same session, or pass the url from a new session after a read.

## 2026-11-08 ~21:20 update
- upper_decay_raw_r1 (g 0.6-4, T 3000, B 16): all sustained; rho(T) 0.16% -> 2.6%; PR(occ) 2785 -> 24260 (flattening), AL share 0.33 -> 0.05. No sudden delocalization up to g=4 (null jumps to PR ~75k at its onset). Looks smeared.
- spread_raw_r1 (163130, B 512): g 0.30/0.34/0.38 all seeds die (<~100 t); 0.45 P(T)=0.002 (1/512), 0.60 0.006 (3/512); survivors ~150-300 active. Null 4-5% just above onset. Too few survivors for delta -> need module-seeded spreading.
- spectrum_sim.py (164455; null_in still computing at write time) + spectrum_exc.py (164540): exact simulated operator, raw scale:
  real signed 5.064 (g* .197, 1/IPR ~50); real exc-only 5.574 (g* .179, ~72); null signed 0.374 (g* 2.68!, ~970); null exc-only 1.102 (g* .908, ~2000).
  in-norm real 0.914 (g* 1.09; observed 1.85-2.0; IPR .435 = 2-neuron modes).
  Null observed onset 1.11 < signed MF 2.68 -> rectified tanh_+ makes inhibition ineffective at sparse activity; exc-only operator is the right linearization.
  observed/MF(exc): real 1.9, null 1.2 -> fluctuation penalty of small localized mode; explains onset ratio 3.2 vs eigen ratio 5.1. (Earlier "3.2 matches 3.6" used symmetrized matrices; corrected on page.)
- code/movie.py -> figs/activity_movie.mp4 (crf 30, 3 MB) + poster. figs.py fig7 now includes upper sweep. analyze.py includes upper sweep.
- Explainer v2 published (same URL); copy in explainer/.
- Next: weight-shuffled null (--null_kind weights), module-seeded spreading, long slow-window runs, (g,r) diagram, h>0.

## 2026-11-08 ~22:00 PHASE DIAGRAM — goal set by user ("build a phase diagram of the brain"; then explain it spectrally)
Stage 1 sweeps (code/pd_submit.sh, ids in results/pd/jobs_stage1.txt; constraint dropped from swc_job.sbatch, any GPU):
 (g,r) real r in {0,.25,.5,1,1.5,2,3,4,6}; config null + weight-shuffled null (new swc.py --null_kind weights) at r in {0,1,4};
 g = logspace(0.1, 30, 32), T 3000, B 16, fully active start. Driven (g,h) r=1, h in 1e-5..1e-1, T 1000 (+null h=1e-3).
 Griffiths-window: module-seeded spreading (seeds ~ |v0|^2 over top-1000, results/seeds_module_{real,null}.npz), long T=30000 runs.
Analysis: code/phase.py (phases Q/S/L/D, operational defs in docstring, fixed before seeing data) -> figs/pd_*.png, results/pd/phase.json.
Theory jobs: code/theory_qmf.py (GPU; rectified Perron root mu(r) + discrete-time QMF dynamics on identical grids, real/null/wnull,
 + driven), code/spectrum_r.py (CPU, signed lam(r)), code/spectrum_tail.py (CPU, top-400 dir / 1500 sym eigenpairs for island counting).

### Hypothesised phase diagram (written BEFORE stage-1 results; test against it)
H1 Quiescent Q for g < g_QMF(r) = (1-e^{-dt})/(dt mu(r)) = 0.906/mu(r), mu = rectified Perron root. Real r=1: mu in [5.06, 5.57]
   -> g_QMF ~0.16-0.18. Inhibition enters only through mu(r) (rectification: dispersed inhibition does nothing at sparse activity).
H2 Slow/Griffiths window S between g_QMF(r) and the sustained onset g_on(r) ~= kappa * g_QMF(r), with kappa set by the size of the
   localized leading mode (finite-size extinction): kappa_real ~1.9-2.1, kappa_null ~1.2-1.35. => on log g, the S/L boundary runs
   PARALLEL to 1/mu(r), shifted by a constant. This is the main spectral prediction for the (g, r) plane.
H3 Localized active L above g_on: QMF fixed point reproduces rho(g) and PR(g) within a small factor away from onset; PR(g) grows as
   more localized tail eigenmodes cross g * lambda_k ~ const (island counting); smeared, no jump.
H4 Delocalized D: configuration null and weight-shuffled null go Q -> narrow S -> D directly (no L phase; leading modes delocalized
   or single hubs). Real: D appears only at large g and small r (r=0 first); for r >= 1 inhibition may prevent D up to g = 30.
H5 Driven (h > 0): no absorbing phase; replica fluctuations N var(rho) are high across a BROAD g range spanning the S window
   (Griffiths: extended criticality), and the peak moves down toward g_QMF as h grows (drive re-seeds the module, so extinction
   no longer limits it). Null: a narrower peak near its onset.

### Theory (explainer/theory.html, published as theory.html in the explainer artifact) — 2026-11-08 ~22:50
- Derivation: exact Moebius linearisation of the stochastic generator -> branching operator B_ij(g) = beta tanh(g E_ij) (exc only);
  P1: lambda_PF(B(g_lin)) = alpha_eff, INDEPENDENT of r. Pair operator variant P(g). Amplitude eq with (1-a) + coincidence term
  D(g,r) (only place r enters). 1-D logistic birth-death with K_eff = 1/(<u^2,v>(<u,v^2> + D/lam)); exact extinction-time
  recursion; P2 onset tau(g)=T; window ~ sqrt(2 ln(alpha T)/K_eff); drifts with sqrt(ln T). Islands sum (Griffiths decay,
  smeared recruitment), drive -> telegraph islands, fluctuation peak K Phi(R) = ln(1/(beta h n p_est)).
- REVISION (before comparison): first draft used QMF rectified mu(r) for P1; theory_mode showed QMF null r=1 g=3.43 > observed 1.11
  (known since earlier today) -> mean-input closure overstates inhibition near silence. Only that previously measured number used.
- Numbers: g_lin real 0.1632 (weak-synapse 0.1626), null 0.829 (0.823), wnull 0.663 (0.660). QMF mu(1): real 5.079, null 0.264, wnull 1.268.
  Observed/ g_lin: real r=1 2.1, null r=1 1.34. Early P2 output: real r=1 K_eff ~17 (too small -> onset ~0.5?), r=0 K~61; null r=0 K~5000, r=1 ~96.
- Jobs: th-branch-{real,null,wnull} (theory_branch.py -> results/pd/branch_*.json), th-mode (mode_*.json), th-qmf (qmf_*.npz), tail-* (tail_*.npz).

### Flow field + theory vs sim (2026-11-08 ~23:40)
- explainer/flow.html (built by code/flow_build.py from explainer/flow_template.html; params code/flow_params.py -> results/pd/flow_params.json):
  theoretical (g,r) diagram (Q/S/L/D: tau=200, tau=3000 from theory_branch wide run; QMF PR>=N/4 for D) + sim markers;
  2-variable module/rest flow (module exact from theory; rest lumped, gain set to switch on at QMF D line); noisy runs with K0=54.9.
- Theory vs sim, real: r=0 S 0.255/0.251, onset 0.309(0.298 fine grid)/0.302, D 1.58/1.58 (all match).
  r>=0.25: theory overstates inhibition. Theory tau=200 line ~ sim tau=3000 line for r>=1 (lifetimes underestimated ~15x
  -> K_eff too small; factorised coincidence term). QMF D line also too inhibitory (r=0.25 3.30 vs 2.74; r=0.5 8.3 vs 4.8; r=1 none vs 17).
- Spectral tail: all top-400 exc modes localized (max 1/IPR ~650) down to lambda 0.24.

### Flow page v2 + GPU theory + module-alone check (2026-11-09 ~00:30)
- GPU port: code/theory_branch_gpu.py (Slurm array, torch power iteration; 8 tasks done in 15-46 s each vs >9 min CPU; K_eff
  matches CPU 54.7 vs 54.8). Dense r (23 values 0-6), FMAX 30: results/pd/branch_real_dense_part*.json. QMF dense r: qmf_real_dense.npz.
  USER: "use GPUs to parallelize computation whenever possible" -> default to GPU arrays for theory/sims from now on.
- code/module_sim.py (GPU): the 526-neuron AL module ALONE (internal synapses, rest silent, 64 replicas, T 3000) reproduces the
  whole-brain onsets: all-survive r=1 0.446 (brain 0.436), r=4 0.66 (0.63); slow from 0.246 at r=0. results/module/.
- flow.html v2 (code/flow_build.py + explainer/flow_template.html): smooth theory curves (interpolated crossings, monotone,
  light 1-2-1 smoothing in log g), sim points + dashed curves; definitions box; streamline flow (evenly spaced, marching dashes,
  labelled fixed points); x now = fraction of module neurons (kappa = m_cap*sum v/n = 0.256); live in-browser simulation of the
  real module (CSC push, exact rates) + log-time traces of real module vs reduced-model runs. Mobile canvas-growth bug fixed.

### Complete phase diagram (started 2026-11-08 23:20; user: "analyze them; I want the complete simulated vs theoretical phase diagram")
- GPU refinement: pd2-sim array 168141 (54 tasks, %14, --nice 500): onset_real_r<23 r> (40 g .2-2.6, B 32), deloc_real (r<=2),
  onset_{null,wnull} (r 0,.5,1,2,4,6), drive_h* (r=1, 40 g), drive null. Args: results/pd2/args_sim.txt; sbatch code/swc_array.sbatch.
  Null theory GPU arrays th-bgpu-null 168142 / -wnull 168143 (branch_{null,wnull}_dense_part*); QMF th-qmf2-* 168144-150 -> results/pd2/qmf_*.
- Workflow wf_a0c2f241-2a2 (malecns-phase-complete): analyze spreading / long runs / driven -> adversarial verify -> assemble
  (waits for pd2) -> critic. Outputs code/analysis2_*.py, results/analysis2/*, figs/pd2/*; flow page data results/analysis2/flow_sim_boundaries.json.

### Complete phase diagram assembled (2026-11-09 ~01:30; code/analysis2_phase.py + analysis2_phase_figs.py)
- All 54 pd2-sim outputs + 7 pd2 QMF files present (preempted 168141_5/_43 rerun as 169200_5, 169836_43; th-qmf2-wnull 168150 rerun as 169201).
- Matched criteria (tau_sim censored-exp MLE = 200 / 3000, PR = N/4 among sustained) vs theory (logtau = ln 200 / ln 3000, QMF PR N/4).
  Real r=0: S 0.245 vs 0.234, on 0.291 vs 0.266, D 1.45 vs 1.47; no-coincidence variant (D=0) gives S 0.234, on 0.270 (inside sim CI).
  r>=0.25: theory too high, on ratio 1.38 (0.25) -> 1.78 (1) -> 2.61 (4); K_needed/K_eff ~1.9-2.2 for every r>=0.25 (1.30 at r=0); nulls same factor.
  QMF D too inhibitory (real r=0.5 7.67 vs 4.44; none for r=0.75-1.5 while sim D 8.3-28.9); config null D at onset for all r; weight-shuffled null HAS an L phase (r=0 on 0.73, D 1.21).
- Outputs: results/analysis2/phase_complete.{json,md}, phase_complete_table.md, flow_sim_boundaries.json; figs/pd2/phase_{real,nulls,driven}.png, tau_sim_vs_theory.png.

### Critic review of the complete phase diagram (2026-11-09; code/analysis2_critic.py, analysis2_critic_fig.py)
- Independent recomputation from raw files (critic.json, critic_run.log): theory ln tau reproduced to 1e-11; real-graph sim S/on within ~2% of
  analysis2_phase.py for all 23 r; every sim boundary inside its model-free 95% data bracket; theory above the bracket at every real r incl. r=0.
- Corrections in phase_complete.md (marked [critic]): QMF D only known to a 20% grid cell (now in tables + bars on phase_real/phase_nulls);
  config null r<=0.75 QMF D is an upper bound; config-null r=0 "0.1-0.5% agreement" -> same 8% grid cell only; wnull K_needed/K_eff not a
  constant factor (2.93..1.36), coincidence-term attribution is a hypothesis (r=0 needs K 75.8 > no-coincidence K 70.7).
- Operational phase.py criteria are B-dependent (B=32: S_op ~ tau 52, L_op ~ tau 1.4e5); L_op sits x1.11-1.68 above the tau=T onset; theory
  still too high under them. Pair-operator theory variant moves boundaries up 3-16% (further from sim).
- phase_driven.png now 2x2 with sim vs QMF density (QMF under-predicts 2.2-2.4x at g 0.5-3, h<=1e-3) and the null curve; new critic_boundaries.png.
- Open: finer QMF grid for D, D untested at several r, driven plane only r=1 (no null/inhibition axis), no PR bootstrap, cause of the x2 K miss.

### Phase diagram v2 (started 2026-11-09 ~03:00): de Candia-style diagram + second-order question
- User: recover a phase diagram like de Candia et al. 2026 (Front Comput Neurosci 20:1744991; SL/SH/Bm broken symmetry, 2nd-order
  critical lines with MF-DP avalanches, 1st-order dashed lines) instead of a Griffiths region; "is the steep smooth crossover a
  second order transition?"; plus accepted: lifetime heat-map redraw (g_lin solid, tau/PR contours dashed) + finite-size check on nulls.
- Paper stored in refs/decandia2026_symmetry_breaking_modular.{pdf,txt}.
- Workflow wf_98fb990d-4fb (malecns-phase-diagram-v2): research (literature, theory) -> design plan (results/pd3/plan.md) ->
  4 GPU programs (redraw_fss, module_theory, criticality, multistability; jobs pd3-*, --nice 500) -> verify each -> synthesize
  (results/pd3/REPORT.md, figs/pd3/) -> critic. Code code/pd3_*.py.

### Antennal-lobe literature question (started 2026-11-09 ~03:30)
- User: the AL's special status; possible AL-sleep link in flies; do our findings align with empirical results on consciousness
  or brain phase transitions associated with this region?
- Workflow wf_eebe6fe0-5e5 (malecns-antennal-lobe-literature): 3 literature lines (AL & sleep/arousal; anesthesia &
  consciousness proxies; criticality/state transitions in olfactory systems & fly-connectome models) + data grounding of the
  AL module (composition, why it dominates: ORN convergence / synapse counts, robustness to removing ORNs / capping weights)
  -> source-verification of every citation -> synthesis results/al_lit/REPORT.md (+ REPORT_section.html for the explainer) -> critic.

## Antennal-lobe literature question — DONE (2026-11-09 ~05:30)
Workflow wf_eebe6fe0-5e5 finished. 3 literature lines + data grounding + 4 fact-checkers + synthesis + critic ("PASS WITH FIXES", 32 edits applied).
Outputs: results/al_lit/REPORT.md (saved from the synthesis summary; the agent was not allowed to write it), REPORT_section.html
(post-critic; pre-critic copy kept), line{1,2,3}_*.md with Verification sections, data_grounding.{md,json}, verify3*.json.
Published as explainer section 11 "The antennal lobe: why it dominates, and what the literature says" (artifact v10); Open
questions → 12 (list refreshed), Glossary → 13.
Key points: AL dominance = all-to-all cholinergic lLN1_bc clique (15/lobe, median 204 synapses) + PN↔LN loops; λ 5.57 vs
next 1.88; ORNs not in module (removing them: 5.55); decisive dependence on the ACh label of lLN1_bc/lLN2 and on synapse
counts (binarized wiring → MB leads). Literature: AL modulates sleep amount/timing (ORN silencing, Mz699, TPN-II), AL
synapses/glia track sleep need but not specially; no study shows AL activity switching with sleep state; no AL recordings
under anesthesia or across verified sleep found; seizures and consciousness proposals point elsewhere (MB α/β, CX).
Interpretation: AL-only active regime reads as sleep-like; recruitment beyond AL ≈ wake. Open issues: Cook 2025 and
Kirszenblat 2025 from abstracts only; Dai 2024 ex-vivo deafferented brain ≈ model h = 0 (possible sharper test).

## Is the self-sustained AL a model defect? (started 2026-11-09 ~05:45)
User: "Yes, investigate this" (after: real resting AL activity is ORN-driven and PNs go silent without antennae; our AL
self-sustains at h = 0). Workflow wf_50011f5c-30d (malecns-al-defect): Prepare = literature on lLN1_bc identity / eLN
physiology / synaptic depression / deafferentation (results/aldef/lit_biophysics.md) ∥ new simulator code/ald_sim.py
(per-population drive + clamp, sign overrides + NT-probability ensembles, weight transforms, Tsodyks-Markram depression,
adaptation, refractoriness; must reproduce swc.py with features off) → Experiments E1 ORN drive / antenna removal (allowed g
range), E2 transmitter signs (lLN1_bc inhibitory/0, ensembles), E3 depression/adaptation (hypothesis: g_lin unchanged,
lifetime/amplitude reduced), E4 weight transforms in simulation → Synthesis (report text + results/aldef/section.html as
explainer section 12) → adversarial Verify. Jobs ald-*, --nice=500, gpu/ellis. Does not touch swc.py / pd3_* / analysis2_*.

## AL-defect study: extended simulator code/ald_sim.py (2026-11-09, ald-* jobs)
- Question (user): is the AL's self-sustained ignition at h = 0 a model defect? New simulator `code/ald_sim.py` (swc.py untouched):
  per-population drive (`--drive "SEL=h"`), `--clamp_off/--clamp_on SEL`, `--sign_override "SEL=V"`, `--sign_ensemble K`
  (release NT confidence -> sign probabilities; `--ens_unit neuron|type`), `--wtransform linear|cap:k|sqrt|log1p|binary` +
  `--wnorm renorm|none`, STD (`--std U TAU`, `--std_group`), adaptation (`--adapt D TAUA`), `--refractory N`, region traces
  (AL_module / AL_rest / CNS_rest + `--region`), JSON sidecar. Selector grammar + all flags in the docstring.
- Validation (job 176834, A6000): features off -> bit-identical rho/surv/occ to swc.py in 4 cases + general code path
  (`results/aldef/validation/validate.json`); 17/17 small-graph unit tests with exact expectations (`unit_tests.json`).
  Speed B = 64, T = 3000: swc 7.91 ms/step, ald off 7.89, ald all features 10.64 (1.35x).
- Smoke (array 177020, `results/aldef/smoke/`, T = 300, B = 16, illustrative only): lLN1_bc -> -1 or 0 kills activity at
  g 0.30/0.45 (baseline survives 100% at 0.45); ORN drive h = 0.02 gives ORN rate 0.0215 = exact formula at g = 0.
- Array runner: `code/ald_array.sbatch <argsfile>` (one ald_sim.py arg line per task; -J ald-<name>).
- **ald_sim.py now carries E2 (`--engine push`) AND E1 options (2026-11-09 06:17, md5 4210b6d6...).** E1 added, default off:
  `--switch_at T1` + `--sw_drive/--sw_clamp_off/--sw_clamp_on` (two-phase protocols, e.g. antenna removal), `--rec_lin DT`,
  `--win T0:T1` (win_mean/win_m2), `--reg_last`, `--occ_win T0:T1`. Diff vs the E2 version: `results/aldef/drive/ald_sim_e1.diff`.
  WARNING to every lane: never `cp` a private copy over code/ald_sim.py -- merge, and swap only if the md5 on disk is
  the one you started from (E1 overwrote E2's edit for ~50 s at 06:14:28-06:15:18 by mistake; restored and merged).

## AL-defect study E1: ORN drive / antenna removal (DONE 2026-11-09 ~07:30; array ald-e1-main 177652, 80/80 completed)
- Protocol (`code/ald_drive_args.py`, criteria pre-registered in `results/aldef/drive/plan.md`): quiet start, all 2,639 ORNs
  driven (isolated-unit ρ 0.01–0.10 = 0.9–9.1 Hz at τ_u 10 ms), everything else h = 0, T1 = 1500; then all ORNs clamped
  (`all`) or only the 2,454 antennal ORNs (`ant`, palps kept) for 3000 units. r ∈ {0,1}, 24 g in 0–1, B = 64, `--engine push`.
  Analysis `code/ald_drive_analyze.py` → `rows.json`, `summary.json`, `table.md`; figures `code/ald_drive_figs.py` → `figs/aldef/drive_*.png`.
- Allowed g (gain G = ρ_uPN_ant/ρ_ORN in [0.29, 1.45] AND PNs silent after removing all ORNs, interpolated):
  r = 0: [0.090–0.096, 0.176–0.231]; r = 1: [0.100–0.119, 0.215–0.289] (range over ORN levels). Upper end = τ = 200
  boundary (S 0.234 / 0.283) or the gain ceiling; always below the h = 0 onset (0.266 / 0.344). Adding the palp-intact test
  (antennal PNs ≤ 10 % of driven with palps still driven) cuts the upper end to 0.165–0.176 (r = 0) / 0.177–0.192 (r = 1).
- Above the S boundary the model contradicts both observations: G 2–4.6 (PNs fire 2–4.6× the ORN rate) and PNs keep 51–80 %
  of their driven activity 1000–3000 units after removing every ORN. Post-removal extinction times match the h = 0 lifetimes.
- AL-module fluctuation peak under realistic ORN drive: χ_time at g 0.16–0.18 (r = 0), 0.20–0.24 (r = 1), inside the allowed
  range; the h = 1e-5 driven-plane peak (0.30) is above it. Recurrent amplification over feed-forward at the upper edge ≈ 2–2.6×.
- ald_sim.py: E1 options merged on top of E2's push engine (see the warning entry above); validation in results/aldef/drive/validation/.

## AL-defect study E3: depression / adaptation / refractoriness (started 2026-11-09 ~07:20)
- **ald_sim.py change (07:50, md5 92c2e0d3...; was 4210b6d6...):** new option `--std_mode rate|spike` (default `rate` =
  unchanged behaviour). `spike` = Tsodyks–Markram per epoch (U = fraction used per epoch, efficacy fixed at onset), so a
  recovered neuron is undepressed and g_lin is exactly unchanged; the old rate mode weakens even the first epoch by
  φ ≈ 0.77–0.82 at literature U/TAU (g_lin × 1.2–1.3). Diff: results/aldef/depression/ald_sim_e3.diff. Checks
  (job 179533): 14/14 unit tests (code/ald_dep_test.py → results/aldef/depression/validation/unit_tests_e3.json),
  bit-identity swc vs new (off), old vs new (rate STD + adapt + refractory + push), spike U=0 and instant-recovery vs off
  (validate_e3.json; the "new" file there was the identical work copy).
- Plan + pre-registered criteria: results/aldef/depression/plan.md. Code code/ald_dep_*.py, jobs ald-e3-*.

## AL-defect study, E2 transmitter signs (2026-11-09, jobs ald-e2-*; DONE)
- ald_sim.py change (opt-in, defaults unchanged): `--engine push` (event-driven W@X over active neurons; falls back to spmm
  above `--push_max`·nnz·B, default 0.05) + `--push_max`. Default path still bit-identical to swc.py (2 cases); push matvec
  rel. diff ≤ 2.5e-6; same-seed short runs identical; independent long runs agree (results/aldef/signs/validation/
  validate_push.json); ~2.5-3x faster near onset (B 64). Backup of the pre-change file: code/ald_sim.py.bak-e2-pre-push.
- Code: ald_sign_{spec,args,args_fine,analyze,figs,validate}.py. Results results/aldef/signs/ (summary.json, spec/, sim/ 131
  coarse runs, sim_fine/ 11 refinement runs); figs figs/aldef/signs_{phase,curves,ignition,ensemble,spectrum}.png.
- Spectrum: lLN1_bc not exc -> lambda1 5.574 -> 3.174 (still AL, g_lin 0.287); + ACh lLN2 (40) -> 1.879 VNC leg mode
  (g_lin 0.485, 1/IPR 632, AL module share 0). Sim onsets (tau=3000) r0/1/3: base 0.263/0.348/0.458; (i) lLN1_bc -r
  -/1.036/1.366; (ii) 0 0.448/1.003/1.353; (iii) 0.566/1.042/1.381. Without the clique the first sustained activity at r>=1 is
  broad (PR 13-18k neurons; VNC + CB + optic lobe; AL module share <=9%); D line ~unchanged (r0 1.47 -> 1.48/1.64, r1 14.4 ->
  14.5-17.0). Ensembles (type-level 24 / neuron-level 16 draws): AL ignites first whenever all lLN1_bc are excitatory (20/24).

## AL-defect study E3 results (DONE 2026-11-09 ~09:15; all numbers in results/aldef/depression/key_numbers.json, summary.json, table.md)
- Runs (r = 1 unless noted; decay, T 3000, B 64, push engine): stage 1 array 179789 (110 tasks: 34 arms × 27 g in 0.15–8 +
  8 weak-drive arms), stage 2 181164 (51 tasks, refinement), isolation 180835 (AL alone / CNS without AL), E1-protocol removal
  with depression 180233 (10), mesoscopic theory 180376-80, theory vectors 179585, CPU analysis 181578. Figures figs/aldef/depression_*.png.
- Literature depression (per spike, all synapses) moves the sustained onset 0.346 → 0.81 (slow) / 1.14 (N15) / 1.38 (NW16) /
  1.66 (KW09); S moves alike (window on/S 1.19 → 1.21–1.25). Rate mode: 0.86–2.26. Lifetime at g = 0.378 (base: 92 % alive at T):
  17–52 units. QS AL activity at g 1.95: 0.30 → 0.04–0.19.
- Linear onset: per-spike mode leaves the weak-drive AL gain below g_lin unchanged (ratio 1.02 ± 0.04, h 1e-5); rate mode
  lowers it (0.76 ± 0.03; first-epoch factor φ 0.80). Above g ≈ 0.19 depression caps the amplification (gain at g 0.30: 1810 → 64).
- AL no longer leads with depression: AL alone sustains only at 3.00 (NW16) / 4.53 (KW09) vs no-depression 0.337; CNS without
  the AL sustains at 1.48 / 1.74 (no depression 1.04). AL share at the depressed onsets 6–11 % (base 47 %), PR ×10.
- Population-specific (NW16): lLN1_bc outputs alone → onset 0.944; LNs 0.995; whole AL 1.01; PNs 0.487; ORNs / outside AL: no change.
- Adaptation (D·τa 0.1–3, τa 50) 0.42–1.71; refractory 5–50 steps 0.47–2.97 (strengths not data-constrained).
- E1 criteria with depression (ORN ρ 0.04): allowed g 0.110–0.272 (base) → 0.247–1.175 (NW16), 0.350–1.386 (KW09).
- Theory (code/ald_dep_theory.py; K calibrated κ = 1.36 / 2.19): frozen-resource chain predicts AL-alone onsets 2.83 vs 3.00 (NW16),
  4.37 vs 4.53 (KW09) and weak-depression whole-CNS onsets within 3 %; adiabatic limit too low (1.40); mesoscopic (n, x) model 2.47.

## AL-defect study E4: weight transforms in simulation (DONE 2026-11-09 ~09:20; jobs ald-e4-spec 180946, ald-e4-coarse 181031 (+182046 rerun of preempted task 19), ald-e4-fine 182078)
- Transforms linear / cap:50 / cap:20 / cap:10 / sqrt / log1p / binary, `--wnorm renorm` (equal mean in-strength 1), r = 0, 1, decay
  start, h = 0, T = 3000. Coarse 42 g (0.72–180 × g_lin(T), B 32), fine 20 g (0.7–1.25 × onset, B 64), causal arms 18 g
  (0.85–4 × onset, B 32): AL module clamped off (`--clamp_off module`) and AL module alone (`--clamp_off "~module"`).
  Ran a frozen byte-identical copy of ald_sim.py (results/aldef/weights/ald_sim_used.py, md5 92c2e0d3; code/ald_wt_array.sbatch).
  No change to ald_sim.py. Code code/ald_wt_{spec,regions,args,analyze,figs}.py (+ ald_wt_spec.sbatch); results
  results/aldef/weights/{spec/,sim*/,summary.json,table.md}; figs figs/aldef/weights_{spectrum,onsets,curves,ignition,causal}.png.
- Spectra (renorm units): lam1 5.57/3.60/2.52/1.91/2.06/1.75/1.36; g_lin 0.163/0.252/0.360/0.474/0.441/0.518/0.667; AL leads
  except binary (MB KC mode 1.359 > AL mode 1.209). Linear-c values match the grounding (3.339, 2.042, 1.301, 0.768, 0.478, 0.280).
- Sustained onset (τ = 3000) r0: 0.262/0.352/0.478/0.602/0.573/0.652/0.732 (on/g_lin 1.61→1.10); r1: 0.346/0.496/0.762/1.045/
  0.960/1.145/0.824 (on/g_lin 1.97–2.21, binary 1.23). Equal-λ1 units: r0 0.262 vs 0.205–0.227 (binary 0.178); r1 0.346 vs
  0.321–0.360 (binary 0.201). Linear reproduces E2 base (0.2623 vs 0.2633; 0.3459 vs 0.3476).
- Who ignites (causal ratios noAL/intact, ALonly/intact): AL alone at linear, cap50, cap20 (both r) and cap10, sqrt (r1)
  [noAL 1.31–3.04, ALonly 0.98–1.09]; cooperative at r0 cap10/sqrt/log1p and r1 log1p [both 1.09–1.26]; AL not needed with
  binary weights [noAL 0.97–0.99, ALonly 2.21/3.59] -> MB Kenyon cells (r1: 79 % of activity, KCg-m) / broad CB+MB+VNC (r0).
- Localization: PR at onset r0 1.2k → 2.4k/4.9k/9.0k/7.4k/14.2k/18.7k; L phase shrinks (D/on r0 5.6 → 1.6; r1 41.8 → 12–29).
- Verdict: weights do not create or remove self-sustained activity at h = 0 (every transform has one); linear counts make the
  AL the unique early igniter, and only extreme compression (cap ≤ 10 at r = 0, log1p, binary) displaces it — to other
  self-sustaining loops (CB, MB KC–KC). The weight rule (cause b) is not the defect.

## AL-defect study: synthesis (DONE 2026-11-09 ~09:40; no jobs)
- Answer: the self-sustained AL at h = 0 is a MODEL defect (synapse dynamics), not a weight/label defect and not a real
  prediction. Ranked causes: (1) no short-term depression (literature per-spike depression: onset x3.3-4.8 at r = 1, AL alone
  3.00/4.53 vs rest 1.48/1.74 -> AL no longer leads; weak-drive gain below g_lin unchanged); (2) lLN1_bc clique as uniform
  excitatory chemical synapses (sign 0/inh: x1.7-3.0; real eLN->PN is electrical; per-edge signs untested); (3) h = 0 with g
  above the data-allowed range (E1: unmodified model passes deafferentation tests for g 0.09-0.29); (4) weights: not the cause.
- Fly position: published model g 0.10-0.29 (r = 1) = 0.63-0.84 of AL onset, rest of brain <= 0.28 of its onset; with NW16
  depression 0.25-1.18 = 0.85 of the (brain-wide) onset, AL <= 0.39 of its own. AL-only regime ("sleep-like" in section 11)
  is outside the allowed range and absent in corrected models -> drop that mapping.
- After fixes: slow window same relative width (on/S 1.18-1.27), no AL-only localized phase (first sustained PR 9-18k neurons,
  AL module 0.3-13%), D at r = 0 with depression ~6.3 (interpolated), onset far above g_lin (x8.5-10) -> fluctuation-limited
  crossover; criticality/hysteresis on corrected models not tested.
- Recommended default: --std_mode spike on excitatory outputs (U 0.25, TAU 157), ORN drive rho_iso 0.04, linear weights,
  g inside the allowed range (0.32-1.24 at r = 1); next: per-edge signs (lLN1_bc->PN 0, ->LN +1), palps test with depression.
- Files: code/ald_synth_calc.py -> results/aldef/synth/synth_numbers.json; code/ald_synth_fig.py -> figs/aldef/synth_phase.png;
  results/aldef/section.html (explainer section 12, NOT yet inserted into explainer/index.html; figures copied to
  explainer/figs/aldef/). Report text returned to the workflow (REPORT.md not written by the synthesis agent).

## AL-defect study: adversarial verification of the synthesis (DONE 2026-11-09 ~10:45; array ald-verify-sim 183656, 7/7 completed)
- Code: code/ald_verify.py (recomputes every load-bearing number from the RAW runs with an independent implementation and
  3 onset definitions -> results/aldef/verify/verify.json); code/ald_verify_simcheck.py + ald_verify_sim.sbatch (new-seed
  runs -> results/aldef/verify/sim/). section.html corrected in place; synthesis version kept as results/aldef/section.html.pre_verify.
- Confirmed: 28 onsets within 0.4 % of the lane values; ratios stable across on_tau / on_half / on_late (<= 2 %); 5 onsets
  replicated with new seeds within 1.3-2.8 % (NW16 1.363 vs 1.384, lLN1_bc sign 0 0.991 vs 1.003, AL alone+NW16 2.92 vs 3.00,
  CNS-without-AL+NW16 1.458 vs 1.481); current ald_sim.py (md5 92c2e0d3) bit-identical to swc.py, 3 new cases; push vs spmm
  engine onset 0.3454 vs 0.3459 (B 128); E1 ranges, E4 causal ratios, D 6.29, ensemble 20/24, spectra; quotes of Olsen 2007,
  Kazama & Wilson 2009, Yaksi & Wilson 2010, Kazama 2011, Eckstein 2024, Nagel & Wilson 2016 (+ Tobin 2017, K&W 2008, Benayoun 2010).
- Corrections: (1) depression vs clique-sign ranking is confounded -- restricted to the AL both give x2.9 at r = 1 (1.011 vs
  1.003; CNS without AL 1.042); the extra factor of all-synapse depression is the rest of the brain (1.04 -> 1.48) -> tie;
  (2) depression factors depend on tau_u (TAU 30 arm: x2.5, AL 18 % at onset); (3) depressed allowed-range upper ends are
  interpolated between grid points 1.15 and 1.5 (upper/onset 0.69-1.08, not 0.79-0.85); (4) "far above any linear
  instability / fluctuation-limited crossover" wrong: 47 top-400 modes (43 non-AL) past threshold at g 1.38; (5) above the
  slow window the GAIN test passes at the highest ORN levels -- only the silence test always fails; (6) Kazama 2011
  "severalfold" = odor-evoked lateral input, palps intact; (7) 3.6 Hz not 3.4 Hz at rho 0.04; (8) lifetimes 15-70 units.
- New measurement from existing E3 removal runs: under ORN drive WITH depression the AL-module fluctuation measure is flat,
  1.2-2.3 over g 0.06-2.6 (no depression: peak 4.2 at g 0.22) -> no near-critical amplifier with depression.

## AL-defect investigation — DONE (2026-11-09 ~10:00)
Workflow wf_50011f5c-30d finished (8/8). Report: results/aldef/REPORT.md (verified; pre-verify synthesis kept as
REPORT_synthesis_preverify.md); explainer section 12 "Is the antennal-lobe result a model defect?" (artifact v11;
Open questions → 13, Glossary → 14; figs/aldef/*.png).
- Verdict: defect of the model's synapse dynamics, not of wiring, not a real prediction. Two features can each remove it
  and the simulations can't rank them: missing short-term depression; lLN1_bc as plain chemical excitation onto every
  target. At r = 1 every fix that removes the AL lead puts the onset at 0.94–1.04 (rest of brain alone 1.04).
  All-synapse depression (NW16) → 1.38 because it also depresses the rest of the brain.
- E1 antenna removal: allowed g (realistic PN/ORN gain AND PNs silent after ORN removal) = 0.090–0.231 (r=0),
  0.100–0.289 (r=1); palps-intact test caps it at 0.165–0.192, i.e. below the slow window and onset. With NW16
  depression ≈ 0.25–1.18 (upper end bracketed 1.15–1.5).
- E2 signs: lLN1_bc non-excitatory → λ 3.17, g_lin 0.287; +40 ACh lLN2 inhibitory → λ 1.879 (VNC leg mode leads).
  The release's own NT uncertainty makes a non-excitatory clique unlikely (P ACh ≈ 0.753).
- E3 depression (new --std_mode spike keeps g_lin exact): sustained onset ×2.3–4.8; first sustained state is brain-wide
  (AL 5–11%); under ORN drive the depressed AL is a quiet moderate amplifier (fluctuation flat 1.2–2.3) not near-critical.
- E4 weights: not the cause; every weight rule self-sustains at h = 0; binary hands ignition to the MB.
- Interpretation change: drop the AL-only regime as a sleep correlate (outside the data-allowed range; absent with depression).
- Open: per-target signs (lLN1_bc→PN 0, →LN +1) not supported by the simulator; LN→LN depression unmeasured;
  τ_u dependence (τ_rec 30 → ×2.5, AL 18%); finer g grid 1.15–1.45 for the depressed upper ends.

## Model 2 phase-diagram program (started 2026-11-09 ~10:10)
User: "can we restart all of our theory and experiment for phase diagrams with this new model". M2 = per-spike
depression on excitatory outputs (ald_sim --std_mode spike, U 0.25, TAU 157) + ORN drive (0.04 isolated activity, h≈0.038);
variant M2s = per-target signs (lLN1_bc→PN 0, →LN +1); M1 = old model as reference. Workflow wf_aae614ed-511:
Spec (results/m2/MODEL2.md + simulator options, jobs m2-*) → Run: experiments decay (h=0 Q/S/L/D + oscillation flag,
nulls), driven (ORN-driven phases + allowed region), nature (FSS, hysteresis, avalanches incl. Levina-style SOC test,
oscillations), modules (de Candia phases under M2) ∥ theory meanfield (QMF + resources, Hopf/saddle-node, driven
response) and stochastic (renormalized branching, resource birth-death, linear-noise, SOC) → synthesis → verify →
explainer/model2.html (new page; not published until reviewed). Phase-diagram v2 (wf_98fb990d-4fb, M1) left to finish as
the old-model baseline.

## M2 restart, lane spec/simulator (2026-11-09 ~10:40; jobs m2-validate 186100, m2-validate-live 186273, m2-unit-live 186308, m2-sim-smoke 186175)
- Spec: results/m2/MODEL2.md (model equations, variants M1/M2/M2s/M1s/M2_allsyn/KW09/N15/tau30, ORN levels, nulls, FSS graphs; protocols
  decay_h0 / driven (+ removal) / aval_pulse / ramp_driven / ramp_h0_down; phase definitions (a) Q/S/L/D + OSC flag,
  (b) DQ/DA/DT/SS-L/SS-D + -O/-D/-NS modifiers + data-allowed region; FSS / hysteresis / avalanche (Levina 2007) criteria; grids).
- code/ald_sim.py now md5 52293c18 (was 92c2e0d3; backup code/ald_sim.py.bak-m2-pre; diff results/m2/sim/ald_sim_m2.diff). Options ADDED,
  defaults unchanged: --edge_sign "PRE->POST=V", --null_seed, --graph FILE.npz, --rec_counts/--counts_win/--counts_reg, --pulse/--pulse_gap,
  --ramp D/--ramp_loop/--ramp_avg. Swapped atomically after validation of the work copy (results/m2/sim/dev/ald_sim_work.py):
  28/28 unit tests (code/m2_sim_test.py), 9/9 observable tests (code/m2_obs_test.py), real-cache V1 swc.py bit-identity (4 cases + general
  path), V2 old-vs-new ald_sim bitwise on 5 aldef option sets, V3 trivial new options = no-ops (code/m2_validate.py) -> results/m2/sim/validation/.
- New shared code: code/m2_common.py (flags/protocols/grids/regions/args_line/write_graph), code/m2_obs.py (tau MLE, crossings, PR, ACF/OSC
  flag, fluctuation, avalanches, power-law fit, <S>(T), MR branching ratio), code/m2_array.sbatch (job names m2-*; SIM env overrides the file).
- Smoke array (21 tasks, production lengths, r = 1): code/m2_smoke_args.py -> results/m2/sim/smoke/; checks + costs code/m2_smoke_check.py.
- 10:55: second swap, code/ald_sim.py md5 61580c76 (= v1 52293c18 + `--init SEL` / `--init_frac F` partial quiet starts for module-phase lanes);
  work copy validated in job 186475 (29/29 unit, 11/11 obs, V1-V4 bit-identical incl. V4 = M2 v1 vs new on M2-option lines); live re-test 187154: ALL PASS (same counts).
  m2_obs OSC rule revised before any production run (deep-trough ACF + coherence ratio C(2P)/C(P) >= 0.7 for OSC-strong).
  Smoke costs (results/m2/sim/smoke/costs.md): decay_h0 5-10 ms/step (<= 145 s per g at T 3000), driven+removal 4.7-11.4 ms/step
  (106-256 s per g at T 4500), pulse 5-7, ramp 3.5 (B 32); push 1.6-1.7x faster than spmm while activity is sparse.
  Note: measured ORN activity falls below 0.04 at high g (0.030 at g 1.5, 0.024 at g 3.0; recurrent inhibition onto ORNs).

## Phase diagram v3 (pd3, Model 1): synthesis (2026-11-09; workflow wf_98fb990d-4fb, synthesis step)
Four programs ran and were adversarially verified (each md has a Verification section; corrections applied):
results/pd3/{redraw_fss,module_theory,criticality,multistability}.md (+ .json). The synthesis is code/pd3_synthesis.py (CPU, no
simulation). It writes results/pd3/synthesis.json (76 sourced numbers) and the figures figs/pd3/phase_diagram_v3.png (THE
de Candia-style diagram: (a) real network, (b) de Candia closure vs the stochastic tests), figs/pd3/q2_second_order_evidence.png
and figs/pd3/fss_summary.png. The heat map is figs/pd3/phase_lifetime_heatmap.png (redraw_fss). REPORT.md text was returned to
the workflow; the subagent was not allowed to write report files. Save it as results/pd3/REPORT.md if it is missing.
- **Q1 (de Candia diagram).**
  - Realised: one continuous critical line, g_lin = 0.1632 (vertical, r-independent). The phases are SL; M (metastable AL,
    finite-K rounded band up to tau = 3000: 0.266 at r 0, 0.344 at r 1, 0.579 at r 6); B_AL (AL-only, both lobes on; the
    analogue of B1 with m = the whole AL); a smeared island cascade (1% lines); and D (SH-like; PR >= N/4: 1.47 at r 0,
    14.5 at r 1, none up to 30 for r >= 1.75).
  - NOT realised in the stochastic model: L/R broken symmetry (M1 0/48), first-order lines (sweep loops shrink 6.5-15.5x from
    D 50 to 800; M2b 0/14), coexistence (M3 1 attractor in 23 cells; M3b 0/40 QMF cells).
  - de Candia's closure on the 31 P0 modules does predict B1L|B1R (1,426 cells, r >= 2.5) and 46 first-order intervals:
    closure artefacts (within-module input CV 1.2-8.6). Not a Griffiths phase.
- **Q2 (is the steep crossover second order?).** Not as a phase transition of the brain. It is the finite-size-rounded image of
  a continuous onset of the AL module at g_lin: mean field continuous; no first-order signature found; x* follows the
  threshold at g_lin at r <= 1 (closure rejected), ~5% off at r 6; null onsets approach g_lin with N (1.336 -> 1.061 at r 0).
  The rounding is set by a fixed module: K_fit ~70 (r 0) / ~36 (r 1); real subsamples 5k-80k and the AL alone within 1.8% of
  the full brain. It is NOT MF-DP: <S>(T) slope peak 2.35 / 2.24 vs one-mode controls 1.57-1.92, among other deviations.
  sqrt-K rounding holds only with fitted K; pre-registered Q2-a/b/c(r 1)/d fail, e passes.
- **Q3.** Heat map done. FSS: FSS-1 PASS (nulls sharpen, Spearman -1), FSS-2 PASS (real fixed within 1.1%), FSS-3 mixed
  (K_fit ~ N^0.667 < 0.7).
- **Caveats.**
  - Model 1 only: the AL self-sustainment is an M1 defect (aldef). Model 2 is redoing this diagram (results/m2/).
  - Only 2 of the 14 M2b centres lie at the onset. The r = 0, h = 1e-5 loop is still open at D = 800, though shrinking.
  - The QMF's oscillating regime (r >= 0.5, g >~ 1-2) is untested in simulation.
  - Doc fix: swc.py uses c = 1/745.91 (mean over the 166,499 neurons with nonzero in-strength), not 1/742.7.
- **Jobs (merged from the programs' JOBS.md; all completed, none queued).**
  - redraw_fss: 175132 (smoke); 175172 (54-task FSS array); fine grid 175681, 175816, 176126, 176564, 176756; reruns
    176015, 175816_0.
  - module_theory: QMF 175683 + 176342; closure 176386 (P0), 176402 (P1); MT5 175546; MT3b 177888; CPU 175284, 176327, 176328.
  - criticality: smoke 178550, 178697, 178922; V0 178665, 178827; avalanches 179269 + 179638 + 180126; QS 179306 + 181954;
    C4 178729_20-21; field 179627. 178729_0-19 were cancelled for bugs (stale QS store; float32 field drive).
  - multistability: 179161, 179202, 179561 (M1), 179562 (M2a), 179563 (M2c), 179564 (M3), 179565 (M3b), 180689 / 180780 /
    181423 (M2b); 132 outputs, ~19.4 GPU-h.

## pd3 completeness critic (2026-11-09, workflow wf_98fb990d-4fb, critic step)
- REPORT.md was still missing when the critic ran (the subagents may not write report files). The critic's corrections and
  its "Critic review" section were returned to the workflow, to be applied when REPORT.md is saved.
- Recomputed from RAW outputs by `code/pd3_critic_recompute.py` (own PAVA/crossing/MLE/Perron; imports no project code) ->
  `results/pd3/critic_recompute.json`. All six reproduce:
  - g_lin 0.1631526 (published 0.1631527);
  - tau = 200 / 3000 crossings 0.2336 / 0.2662 (r 0) and 0.2828 / 0.3443 (r 1);
  - D (PR = N/4) at r = 0: 1.4708;
  - FSS null seed-mean x_on 1.336 -> 1.061 (r 0) and 3.713 -> 1.336 (r 1); realsub max deviation 1.10% / 0.99%;
  - M1 late lobe asymmetry -0.001 .. +0.047 over 48 points, c1 = 0;
  - island-1 1% crossings 0.382 / 0.559 / 0.949 (r 0 / 1 / 6).
- New finding (`code/pd3_critic_alshare.py` -> `results/pd3/critic_alshare.json`): in B_AL (tau = 3000 onset .. island-1 1% line;
  r = 0, 1, 6) the 526 AL neurons carry only 29-49% of all activity. Per neuron the non-AL occupancy is 2e-4 .. 1.3e-3 against
  0.05-0.19 in the AL. "Rest nearly silent" therefore holds per neuron, not in total: B_AL is a B1-like state with heavy spill-over.
- Figures relabelled with `code/pd3_critic_figs.py` (it patches copies of the pd3_synthesis.py / pd3_redraw.py sources in memory;
  the originals are unchanged; pre-critic PNGs are in `figs/pd3/verify_orig/*_precritic.png`; heat-map arrays were regenerated
  into `results/pd3/critic_regen/` and are identical to `results/pd3/redraw/`):
  - `phase_diagram_v3.png`: the suptitle no longer calls g_lin a "second-order" line; the g_lin legend reads "linear instability of
    silence"; the M1 yellow markers are labelled as the 2-module closure's B1 window; the B_AL legend carries the AL share; the
    median-island / rest 1% lines are no longer drawn through rows where they were not measured.
  - `phase_lifetime_heatmap.png` and `_nulls.png`: "g_lin: mean-field critical line" -> "linear instability of silence (exact
    linearisation; r-independent)". The QMF (mean-field) silent edge moves with r; g_lin does not.

## M2 restart, lane decay (h = 0 decay diagram; DONE 2026-11-09 ~15:00; results/m2/decay/, figs/m2/decay_*.png)
- Plan (pre-registered) results/m2/decay/plan.md (+ post-hoc sec. 8, OSC finite-size check sec. 9). Jobs: results/m2/decay/JOBS.md
  (m2-decay-s1 187595 etc.; ~72 GPU-h; preempted tasks rerun with the same lines/seeds; every output present). Code:
  code/m2_decay_{args,analyze,report,figs,keynums,osccheck,graphs,rerun}.py, code/m2_decay_cpu.sbatch. Numbers: boundaries.json,
  key_numbers.json, tables.md, where.json, osc.json, osccheck.json, lifetimes.json, m1_codecheck.json.
- M2 real, 23 r, 64 g each (40 coarse + 24 refined), B 64: S 0.98-1.57, onset 1.18-2.02 (x3.4-4.8 of M1; r 0 -> 6), on/S 1.20-1.33
  (M1 1.14-1.37). D 6.29 (r 0), ~14 (r 0.25-1.25), falling to 8.03 (r 6): the L phase is bounded at every r (M1: D not reached by g 30 at r 1.75/2/3/4/6;
  M1 untested above g 2.6 at the other r >= 2.25).
- First sustained state brain-wide: PR 13.6k-19.1k neurons (M1 ~1k), AL module 5.0-6.3 % (M1 40-50 %); CB + VNC intrinsic, DNs x5.6-8.8,
  VNC motor x9-10; D = optic-lobe recruitment. Same at S (PR 11.6k-15.6k, AL 5-7 %).
- Nulls: configuration null onset 1.37-1.61, D at the onset (as M1); real onset now ABOVE the null for r >= 1 (x1.15-1.42; M1 x0.3-0.5).
  Weight-shuffled null onset 1.41-1.59, D 2.8-5.4.
- M2s = M2 within ~6 % everywhere (AL 4.4-5.1 %). M1s (signs fix without depression): still AL-led (43-50 %), onset x1.17-1.31 of M1.
  Variants r 0/1/4 (onset x M2): KW09 1.16-1.41, N15 0.73-0.85, allsyn 0.87-1.02, tau30 0.49-0.71 (AL 12-18 %); none AL-led.
- Oscillation flag: 86 of 3,410 (point, signal) pairs, all r <= 0.5 except one 4-replica point; none in M1/M1s; damped (C(2P)/C(P)
  mostly 0.1-0.5), period 53-152 units. Finite-size check: configuration-null var_t ~ N^-1.01 with N-independent ACF -> noise-driven
  quasi-cycle; real subsamples inconclusive (flag absent at 40k/80k).
- Lifetimes (T 30000, 128 rep, r 0/1/4 + null r 1): no Griffiths signature at any g; pre-registered verdict exponential (r 4),
  inconclusive (r 0, r 1, null) because many replicas die early (t < 200) while the start's resources recover; post hoc the residual
  lifetimes after t = 200 pass an exponential KS test at 21/21 points. Verdict: no Griffiths-like window in M2.
- M1 pipeline check: S 3/3, onset 2/3 (r 4: 0.492 [0.422, 0.504] vs a2 0.508, intervals overlap); my code reproduces a2 exactly on a2's raw files.

## M2 restart, lane driven (DONE 2026-11-09 ~15:10; arrays m2-driven-A 187523, -B 187524, refinements C_1 190101 / C_2 193666, 14 rerun arrays (35 preempted lines; jobs in results/m2/driven/jobs.json), CPU analysis 200730)
- 494 GPU runs, 4,036 driven points (protocol `driven`, B 64, ald_sim md5 61580c76), 142 GPU-h. Plan + 4 amendments: results/m2/driven/plan.md;
  tables results/m2/driven/table.md; data curves.json / summary.json / points.csv / extra_numbers.json / checks.json; code code/m2_driven_{args,analyze,report,extra,figs,refine,rerun}.py;
  figs figs/m2/driven_{phase_M2,phase_compare,curves,orn_levels,variants,peaks,palp}.png.
- M2 real ORN 0.04 (23 r): DQ -> DA -> DT -> SS-L -> SS-D at every r. Allowed region 0.216-0.968 (r 0), 0.317-1.23 (r 1), 0.474-1.57 (r 6) (M1: 0.092-0.215 / 0.114-0.279 / 0.187-0.382);
  DT->SS 1.21 / 1.57 / 2.07 = decay-lane onset x0.97-1.03 at all r; allowed top = decay S x0.94-1.03. Allowed region shrinks with ORN drive (none at 0.10 for r >= 1).
- No fluctuation/susceptibility peak of real M2 tracks g_lin, DT->SS or the (a) lines (pre-registered test negative): tau_int peak g 0.69-0.96 (0.40-0.61 x DT->SS, also in the weight null),
  AL Fano a broad plateau in SS-L; M1's sharp AL peak sits at 0.55-0.66 x its SS line; the configuration null's AL Fano peak tracks its transition (7/7 r).
- Field slope vs transmitted ORN input (post hoc) falls through 0.5 at g 0.19-0.29 (1.2-1.8 g_lin): driven activity is depression-set, not input-set, over most of the allowed region.
- Negative: palp-intact test fails throughout M2's allowed region at r 0, 1 (antennal PN residual 10-34 %); M2s halves the leak, sliver at r 0 only. OSC 14/4036 points, none strong (r 0 near DT/SS; weight null r 0).

## M2 restart, lane modules (de Candia module phases under M2; DONE 2026-11-09 ~16:50; results/m2/modules/, figs/m2/modules_*.png)
- Plan (pre-registered 15:00, + run notes sec. 9): results/m2/modules/plan.md. Code: code/m2_modules_{setup,args,phase,analyze,figs,report,rerun}.py,
  code/m2_modules_cpu.sbatch. Tables/numbers: tables.md, key_numbers.json, phase_A.json, lr.json (+ lr_extremes.json), multistab.json, iso.json.
  Jobs: JOBS.md (smoke 202987; LRa 203117; DRb 203119; ICa 203121; ISOg 203431; ISOc 203412 thickstun CPU; rerun 205284); 222 runs, 19.4 GPU-h + 2.1 CPU-h.
- Regions: pd3 partitions reused (P0 34 spectral modules; P1 Leiden 50 >= 50 + small; P2 superclass x side >= 50 + small) + HEM_L/R;
  groups/regions.npz; isolated-module subgraphs graphs/*.npz (full-brain c).
- Part A (existing decay/driven occupancies): M2 (a) at the first sustained g 31 (r 0) .. 4 (r 6) of 34 P0 modules >= 1% (32-33 at 0.1%);
  M1 always 3 (the AL). 13-32 of 34 modules reach 1% within 1.25 x onset in M2 vs 3 in M1; median module crossing 1.0-1.4 x onset (M1 1.7-3.0).
  The AL lobes are still the most active modules per neuron at the M2 onset (with isl1/isl19/isl16 = VNC/CB islands; P2: motor neurons, DNs).
  SH (all 34 >= 1%) M2 4.19 (r 0) .. 21.6 (r 4), none <= 30 at r 6; M1 0.97 .. 12.0. (b) below DT->SS only the AL lobes are on (2-3 modules).
- Part B L/R: 0 broken-symmetry points of 220 testable (c1/c2/c3/c4 all 0) for AL lobes and hemispheres, M1/M2/M2s/M2_tau30, h = 0 and ORN drive;
  M2 t_sym medians 0.8-2.8 (AL), 1.8-7.6 (hemispheres); L-R correlation positive (AL 0.05-0.65, HEM 0.27-0.90); smallest L-vs-R KS p 0.012.
  ORNs are not L/R balanced in the connectome (R 1,344 / L 884 / 411 unlabelled): driven hemisphere tilt -0.03 to -0.12.
- Part C: one cluster in all 96 (a) and 72 (b) cells, every partition; largest start-mean distance 0.056 (threshold 0.2). Quiet vs all under
  ORN drive: ratio 0.996-1.012, smallest KS p 0.036 -> no coexistence candidate (stage 2 not needed).
- Part D: no module-alone state in the network; in M2 every start (even isl5, 221 neurons, or 1% random) survives like the all-active start
  (M1 near onset: isl1 3%, OL 12-16%). Isolated, M2: AL 2.19 / 11.5 / none <= 30 (r 0/1/4), CB 1.42 / 2.30 / 2.95, VNC 3.36 / 6.18 / 8.37,
  OL 5.0 / 6.8 / 7.9 vs full brain 1.18 / 1.57 / 1.94 -> nothing sustains alone at the full-brain onset (CB closest, x1.20-1.52);
  M1: AL alone = full brain (0.268 / 0.338 / 0.493), single lobes x1.24-1.71.

## M2 restart, lane nature (DONE 2026-11-09 ~16:50; results/m2/nature/, figs/m2/nature_{fss,hysteresis,avalanches,osc}.png)
- Plan + 3 amendments: results/m2/nature/plan.md (A1 uniform real subsample = primary real family, written before any FSS output;
  A2 tie-correct KS for the critical-like rule, m2_obs.powerlaw_fit's KS rejects an exact critical branching process -> pl_check.json;
  A3 post-hoc extra seeds for the one hysteresis trigger). Jobs results/m2/nature/JOBS.md (539 GPU tasks, 38.8 GPU-h).
  Code code/m2_nature_{graphs,args,fss,dfss,hyst,aval,osc,summary,figs,rerun}.py. Numbers: summary.json (+ fss/dfss/hyst/aval/
  driven_branch/osc.json).
- FSS (h = 0 onset width W = ln(on/S)): configuration nulls narrow with N (r 1: 0.45 -> 0.075, r 4: 0.47 -> 0.088, 5k -> 167k; pass);
  r 0 0.27 -> 0.136, ratio 0.502 vs <= 0.5 and flat above 40k (fail by a hair). Null onset falls toward the null's g_lin
  (x_on 4.6 -> 1.9 r 0; 7.0 -> 1.67 r 1), QS density at onset falls ~N^-0.45..-0.68: continuous, strongly rounded.
  Real uniform subsamples (20k -> full): W 0.30 -> 0.19 (r 0), 0.39 -> 0.26 (r 1), 0.42 -> 0.28 (r 4), ~N^-0.2; onset moves 4.4 -> 1.18
  (r 0): neither sharpening nor fixed -> pre-registered "inconclusive / drifting". AL-kept families confounded (AL over/under-coupled).
  Pre-registered T1 verdict: inconclusive at r 0/1/4 (no first-order sign; null sharpens; real family drifts slowly).
- Hysteresis (M1 + M2, R_CORE, D 50/200/800 (+3200), both protocols): M2 14/14 finite-rate, loops shrink 11-90x from D 50 to 800
  (normalised area 0.09-0.29 -> 0.003-0.012), at the permutation floor by D 800-3200; no driven-ramp bimodality. M1: one pre-registered
  "first-order" trigger (driven r 1, A at noise level at every D) was noise (two extra seeds p 0.74 / 0.57; D 3200 -> finite-rate).
- Avalanches (pulse, gaps 0/10/100, R_CORE, M1 + M2): 0 critical-like points in either model -> LHG test inconclusive in 21/21 cells.
  M2 tau_S passes through 1.3-1.7 at 0.16-0.32 x onset but KS 0.031-0.080; large-avalanche bump from 0.5-0.68 x onset (population
  bursts); <S>(T) slope 1.96-2.24 but crackling prediction 2.36-2.77; MR m per unit <= 0.977 (pulse), <= 0.94 (ORN-driven, never
  within 0.01 of 1). Driven chi_nonORN has no peak at DT->SS (rises to the grid edge; cnull r 0 interior peak flat in N, a = -0.01).
- Oscillations: OSC only at r 0, g 1.15-1.31 (0.97-1.11 x onset), P 73-83 units, coherence 0.11-0.37 (one 7-replica point 0.78), no 3
  consecutive OSC-strong points at any r -> no Hopf; var_t ~ N^-1.0 (cnull) / N^-1.19 (uniform real subsample) at r 0: noise quasi-cycle.

## M2 restart, lane theory/meanfield (DONE 2026-11-09 ~19:40; results/m2/theory/{plan_meanfield.md, derivation.md, meanfield_*.json}, figs/m2/meanfield_{phase,driven,modes}.png)
- No experiment-lane output was read; predictions pre-registered in plan_meanfield.md (+ amendment A1, correction C1).
- Code: m2_mf_core.py (exact per-neuron moment map of spike-mode TM resources; QMF and coincidence-corrected CQMF
  closures; JVP), m2_mf_run.py (GPU; decay / driven protocols, cutoff extinction, ring-down, sigma_x), m2_mf_glin.py,
  m2_mf_chain.py (AL-mode birth-death chain with resources, M1 rows, kappa per r), m2_mf_analyze.py, m2_mf_predict.py,
  m2_mf_figs.py, m2_mf_test.py (T1-T6 PASS; moment map = ald_sim Monte Carlo to 0.1%).
- Jobs m2-theory-*: smoke 209306/209483, main 209547 + main2 209691, var 209548 (partly) + var2 210641, glin 209704,
  reruns 211715/213316/214798, CQMF driven redo drv2 211907 (C1: first CQMF let inhibition cancel only synaptic input,
  not the ORN drive; superseded files in runs/superseded_cqmf_driven_v1/).
- Key: g_lin 0.16315 unchanged (M2s 0.1854, nulls 0.829/0.663); closure V1 -> CQMF primary (0.451 vs 0.487); A1 valid
  (M1 onset at CQMF QS size ~106 active neurons for every r, max/min 1.38); M2 onset estimates r=0: cutoff 0.32 <
  equal-size 0.63 < AL-chain 2.93 (adiabatic 1.33); r=1: 0.27 < 1.23 < 2.82 (1.39); r>=1 equal-size plateau 1.20-1.24;
  no bistability (CQMF); r=0 no mean-field oscillation (slow-fast theorem), r>0 large slow MF oscillations from g ~1.0-1.6
  (periods ~50-250); damped modes 30-200 units (M1 4-12); sigma_x within 1.1 of 1 for g 0.36-0.69 at r=0; driven ORN
  0.04: G=0.29 at 0.21 (r0) / 0.42 (r1), G max 0.6-0.9 for r>=0.5, removal-SS (equal size) 0.64 / 1.24.

## M2 restart, lane theory-stochastic (DONE 2026-11-09 ~18:50; results/m2/theory/stochastic.md + stochastic_*.json, figs/m2/stoch_*.png)
- Plan/pre-registration results/m2/theory/stochastic_plan.md (NOT blind: the STATE.md summaries of the M2 experiment lanes were
  read before planning; no experiment file opened before the prediction JSONs were written, md5s in stochastic.md sec. 9).
  Code code/m2_stoch_{core,run,modes,graphs,predict,tables,figs,test,pulselog}.py; jobs m2-stoch-* (list in stochastic.md sec. 9).
- Theory: exact per-unit resource moments (e(rho) = renewal formula, kappa = 35.3); depression shrinks every mode's K by 1+kappa = 36.3
  (AL K_dep ~2); first-order branching (noisy-OR) closure for the active profile (subtractive QMF crushes the net at r = 1);
  renormalised frozen operator; 2-D (n, resource) piecewise-deterministic chain (frozen/adiabatic brackets); LNA (validated on a
  400-unit exact simulation: var 3.88e-4 vs 3.82e-4, MR m 0.757 vs 0.742); pulse-protocol self-organised eps + cutoffs.
- Validation M1 (r 0, PF): S 0.233 / on 0.265 vs sim 0.234 / 0.266; CAL r1 0.277/0.330 (sim 0.283/0.344).
- M2 predictions (r 0 PF): single renormalised (AL) mode = pre-registered primary S 1.18 / on 1.66 (upper bound on g); pooled
  0.58 / 0.74 (lower bound); coherent LNA on ~1.22-1.32. CAL r1 / r4 / r6 single on 2.49 / 3.97 / 4.60 (islands-max 2.49/3.14/3.29).
  Profile at onset PR ~15.5k, AL 5.2%, CB 57%, VNC 25%: AL is the only self-sustaining island below g ~2.1 (bulk sigma~ 0.95-0.98).
  Depression reverses mode ranking (AL mode alone 2.77, broad CB mode k2 1.50). Unique mean-field fixed point from 6 starts
  (no coexistence / L-R breaking). Driven LNA: gain window 0.214-1.03 (ORN 0.04), Fano peak 59 at 0.91, tau_int peak 11 at 0.65,
  no chi peak at transitions, MR m max 0.989/step (0.948/unit), never |1-m|<0.01; damped quasi-cycle P 55-150, OSC marginal,
  never strong. Pulse: real M2 never critical-like (S_dep 62-95 from AL-localised self-organised mode; bursts from g ~0.55-0.6);
  configuration null critical-like g 0.91-3.0 (K_s 3e4-6e4). FSS null W 0.309 -> 0.175 (5k -> 167k, ~N^-0.16).
- Post-hoc check vs these STATE.md summaries (stochastic.md sec. 11): primary misses the r 0 onset by +41% (bracket contains it);
  r-trend too steep (M1 calibration); profile, driven, OSC, MR, Levina, null reversal, modules (AL/OL) agree; VNC-alone single wrong.

## M2 restart, lane synth (synthesis; DONE 2026-11-09 ~20:30; no jobs, CPU only)
- Plan (pre-registered before any recheck/table): results/m2/synth/plan.md. Code: code/m2_synth_{recheck,table,figs}.py.
  Outputs: results/m2/synth/{recheck.json (+recheck_v1_binaryS1.json, logs), agreement.json, agreement.csv};
  figs/m2/final_{decay,decay_nulls,driven,m1_vs_m2,agreement}.png. REPORT text returned to the workflow (not written as a file).
- Recheck from RAW npz with own estimator/PAVA/crossings: decay S/on/D 47/47/46 (+1 both-none) pass (M2 real 23 r, both nulls,
  M2s 7 r, M1 r 0/1/4); driven DQ->DA, DT->SS, tau_post=200, allowed top 51/51/51/39 pass (+12 nulls with no allowed region:
  in both nulls G reaches 0.29 only at g 1.4-3.3, after S1 is lost at 1.15-1.36 -> nulls have NO data-allowed region).
- Theory vs sim (rule: inside sim interval or +-10%): no pre-registered point estimate of the h = 0 onset agrees at any r
  (MF equal-size x0.53-0.80; stochastic single mode x1.41-2.28); brackets hold (MF [cutoff, AL-chain frozen] 23/23, x9-16 wide;
  stochastic [pooled, single] 7/7). MF AL-chain adiabatic (meant as upper bound) is within 10% at 16/23 r but below sim at 21/23.
  r-trend on(6)/on(0) sim 1.71: MF 1.93 agree, stochastic 2.77 fail. MF D agrees only at r 0 (6.08 vs 6.29); none <= 30 at r 0.25-1.5.
  Driven r = 0: LNA G = 0.29 at 0.213 (sim 0.216), G = 1.45 at 1.03 (1.05), Fano peak 59 (55), tau_int peak 11.0 (10.1), AL Fano 2.20 (2.19);
  LNA has no r-dependence -> fails at r >= 1. MF G = 0.29 agrees at 10/23 r (r <= 0.625, r >= 4.5), up to x2.5 off at r 1.75.
  MF removal-SS (equal-size) x0.53-0.81 (fail 23/23). Where activity lives at the simulated onset: both theories within x2 (PR, AL share).
- Key negatives for the program: no g satisfies gain >= 0.29 AND the palp-intact test in M2 at r 0 and 1 (palp passes only g <= 0.186,
  gain edge 0.216 / 0.317); the real-graph onset is not pinned by any module (FSS W ~ N^-0.2); no SOC (0 critical-like points, MR <= 0.977).

## M2 restart, lane verify (adversarial verification of the synthesis; DONE 2026-11-09 ~20:45; results/m2/verify/)
- Plan (pre-registered 20:10, before any number): results/m2/verify/plan.md. Code: code/m2_verify.py (own estimators, imports no
  project module), code/m2_verify_args.py, code/m2_verify_sim.py, code/m2_verify_tests.sbatch. Outputs: verify.json (recompute),
  reruns.json (new seeds), corrections.json (9 corrections), jobs.json, simtest/ (live unit 29/29, obs 11/11 PASS on ald_sim 61580c76).
- Recompute from raw files: every checked lane number reproduces (decay S/on/D at r 0/0.25/1/4/6 and nulls; driven DQ->DA, top,
  DT->SS at all 23 r; palp; FSS W ratios 0.506/0.168/0.187 and realsubu 0.61-0.67; hysteresis A50/A800 17-39; MR m_step 0.988;
  lifetimes tau_late 4194, early 13%; iso AL 2.18 / CB 1.41; g_lin 0.1631526 exact; spike-mode efficacy = renewal formula to 0.02%;
  estimator (log-grid vs exact times) <= 0.3%; finite T (3000 vs 30000) <= 0.9%).
- New-seed GPU reruns (m2-verify-sim 219854, 15 tasks, 2.9 GPU-h): 16/16 lines replicate (S/on r 0, r 1; on r 6; D r 0; cnull r 0 same
  graph and NEW graph null_seed 2 -> real/null 0.742; driven r 0 DQ->DA 0.215 / top 0.978 / DT->SS 1.208; r 1 0.318 / 1.219; iso CB 1.43,
  AL 2.16). Palp-intact at r 0 with 256 replicas: CONFIRMED no g passes gain + palp (S2' 0.114 [0.112, 0.116] at the gain edge 0.216).
- Corrections to the synthesis: weight-shuffled null DOES have a small allowed region at r 0 (0.84-1.15) and r 0.25; MF D agrees at
  r 0, 2.5, 2.75 and is 19-35% LOW at r >= 3 (not "same over-inhibition"); real-graph OSC flags only at r 0, threshold-marginal, the one
  "strong" flag is an estimator artefact; oscillation FSS verdict depends on subsample family; branching "never within 0.01" is
  time-base dependent (driven m_step 0.988); palp limit ~0.20 not 0.186; KS bug affects no other user; neither theory lane was blind
  (STATE.md exposure, undisclosed by meanfield); cnull r 0 W-ratio cut undetermined by seed noise.

## M2 restart, explainer page (DONE 2026-11-09 ~21:05; NOT published; review before publishing)
- New page explainer/model2.html ("New model"; 10 sections: what changed (links notebook sec. 12 via ./#al-defect), two settings,
  h = 0 diagram + theory, driven diagram + theory, agreement scorecard, transition nature, SOC (Levina) test, M1 vs M2, open problems,
  glossary of new phase names). Text uses the VERIFIED synthesis with all 9 corrections of results/m2/verify/corrections.json applied
  (figure captions note what final_*.png do not draw: weight-shuffled null allowed region; MF D agreeing at r 2.5/2.75).
- 11 figures copied to explainer/figs/m2/ (final_decay, final_decay_nulls, final_driven, driven_palp, final_agreement, nature_fss,
  nature_hysteresis, decay_lifetimes, nature_avalanches, final_m1_vs_m2, modules_phase_map). Inline to-scale SVG strip (M1 vs M2 at r = 1).
- Nav entry "New model" added to the .pages nav of index.html, theory.html, flow_template.html (+ toc-rail line in index/theory);
  flow.html rebuilt with code/flow_build.py (diff = nav line only). flow_build.py also copies flow.html into the scratchpad publish
  mirror (mcns_explainer/); that copy was RESTORED to the pre-nav version so the mirror is unchanged. To publish: copy index.html,
  theory.html, flow.html, model2.html and figs/m2/*.png into the mirror and add model2.html + figs/m2/*.png to the `files` map.
- Playwright (scratchpad shots/model2_check.py): 1280 and 390 px, light and dark: no horizontal page scroll, 11/11 images load,
  all in-page anchors land at the top, 0 console errors; the other three pages show the new nav entry.
