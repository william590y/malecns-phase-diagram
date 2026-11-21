# Project notes — MaleCNS phase-diagram investigation

*(The working notes kept during the project, published as they were. Cluster-specific rules describe the Cornell cluster the work ran on. Start with README.md and AGENTS.md.)*

Started 2026-11-08 at the user's request: recover a **phase diagram** of the MaleCNS connectome under a stochastic
Wilson–Cowan model and explain it from the connectome's **spectrum**. The first target was a **Griffiths phase** (the
user's proposed correlate of wakeful consciousness; Moretti & Muñoz 2013). That was not found in the original model. Since
2026-11-09 there have been two further targets:
- a **de Candia-style diagram** (module phases, second- vs first-order lines;
  `refs/decandia2026_symmetry_breaking_modular.{pdf,txt}`);
- since 2026-11-09 ~10:10, a **full restart on the corrected "Model 2"** (below). The user asked: "restart all of our
  theory and experiment for phase diagrams with this new model".

Live plan, job ids and every result: `STATE.md` (keep it current). `/MALECNS_ROOT` is a symlink to this directory.

## State as of 2026-11-16
- **Idle since 2026-11-09 ~21:05.** No project file changed after 11-09 21:01 (the last CLAUDE.md edit) until this
  update, and no project jobs (prefixes below) were queued on 11-16. Every program is DONE: M1 complete diagram (`results/analysis2/`), AL literature
  (`results/al_lit/`), AL defect (`results/aldef/`), pd3 on M1 (`results/pd3/`), and the Model 2 program (`results/m2/`).
  The newest verified result is `results/m2/REPORT.md` (11-09). Details: "Results" and "In progress / next" below.
- **Headline answers so far:**
  - M1 has a Griffiths-like slow window caused by one rare region (the AL), not a Griffiths phase.
  - M2 has no Griffiths-like window. Its onset is brain-wide and the transition is continuous and rounded.
  - At r 0 and r 1, no g passes all three data tests; the palp-intact test misses by 7% in g.
- **Explainer:** the live version includes `model2.html`. A file listing on 11-16 confirmed this. The last entry in
  STATE.md ("NOT published") was written before the v12 publish.
  - The scratchpad publish mirror (`mcns_explainer/`) no longer exists.
  - To republish: read the artifact by URL (`action: read`), build from `explainer/`, then publish with `url`.
  - The live `index.html` (129,652 B) equals `explainer/index.html` (129,106 B) except for the document skeleton
    (doctype/head/body wrapper) that publishing adds (diff checked 11-16). The live flow.html, theory.html and model2.html
    have exactly the same byte sizes as their local copies. So `explainer/` is the current source of the live pages.
- **Simulator:** `code/ald_sim.py` is md5 61580c76 (checked 11-16). Its live tests passed 29/29 unit and 11/11
  observable (job 219855).
- **Open decisions (the user picks the next direction; none is chosen).** The candidates come from the m2/aldef reports:
  - the palp leak (electrical eLN→PN coupling, or LN→LN dynamics?);
  - theory with inhibition;
  - the TAU 30 time-unit dependence;
  - per-target signs vs depression (this needs a measurement of the lLN1_bc→lLN1_bc synapse).

  Before reusing MODEL2 criteria, fix a single OSC autocorrelation estimator and the time base of the branching ratio.
  `figs/m2/final_*.png` were not regenerated after verification: they omit the weight-shuffled null's allowed region and
  the mean-field D agreeing at r 2.5 and 2.75.

## Explainer artifact (the user reads this)
https://claude.ai/artifact/VWAKYgXVrQ4n38XGW3X7Bs. It is published from the scratchpad copy
`/tmp/claude-*/.../scratchpad/mcns_explainer/` (mirror in `explainer/`). That scratchpad copy was gone by 2026-11-16, so
work from `explainer/` (see State above). New figures go in the `files` map of the publish.

**`index.html`** is the notebook, sections 1–14:
- 10: complete phase diagram, sim vs theory (Model 1);
- 11: the antennal lobe and the sleep/consciousness literature;
- 12: is the antennal-lobe result a model defect? (figures in `figs/aldef/`);
- 13: open questions;
- 14: glossary.

**`theory.html`** is the derivation: P1–P5, the revision note, and section 10 "how the predictions fared".

**`flow.html`** is the animated page. `code/flow_build.py` BUILDS it from `explainer/flow_template.html`.
- After any data change: rebuild, copy to the scratchpad dir, republish with `files`.
- Test with Playwright (`scratchpad/shots/*.py`), run from a directory without a stray `inspect.py`.
- Mobile trap (fixed): the canvas base height comes from `data-h`, never from the dpr-scaled `height`; the
  ResizeObserver reacts to width changes only.

**`model2.html`** (published 11-09 as explainer v12): the "New model" page from wf_aae614ed-511, built from the verified report `results/m2/REPORT.md` with all 9 corrections; figures in `explainer/figs/m2/`. Publish it like the other pages, with `model2.html` and `figs/m2/*.png` in the `files` map.

## Data (read-only)
- MaleCNS v1.0 (Janelia, CC BY 4.0), preprocessed in `/MALECNS_CACHE` (provenance `~/malecns-flygym/docs/data.md`).
  Never modify it.
- Arrays: `pre`/`post`, `weight` (synapse counts), `sign` (ACh +1; GABA/Glu/His −1; DA/5-HT/OA/unclear 0), `superclasses`,
  `types`, `positions` (soma µm; 27,182 NaN), `sides`.
- Size: 167,216 neurons, 25,587,572 edges, 124M synapses.

## Models
**Model 1 (M1, the original; `code/swc.py`)**
- Stochastic Wilson–Cowan (Benayoun et al. 2010).
- Binary units; 1→0 at α = 1; 0→1 at β·tanh(sᵢ)₊ (β = 1).
- sᵢ = g Σⱼ cᵢ wᵢⱼ σⱼ aⱼ + h, with σ = +1 exc, −r inh, 0 modulatory.
- `--norm raw` (c = 1/745.91 = 1/mean in-strength over the 166,499 neurons with nonzero in-strength; keeps heterogeneity)
  is the main setting. Older notes say 1/742.7 (total in-strength / all N); every simulator uses swc.py's 1/745.91.
- Synchronous Δt = 0.2 (α_eff = 0.906). h = 0 makes silence absorbing.
- Nulls: `--null` configuration (targets permuted); `--null_kind weights` (wiring kept, counts shuffled).

**Model 2 (M2, current default; `code/ald_sim.py`; spec `results/m2/MODEL2.md`)** is M1 plus two changes:
- **(1) per-spike Tsodyks–Markram depression** on outputs of excitatory (ACh) neurons:
  - `--std_group "sign:1=0.25,157" --std_mode spike`, i.e. U 0.25, TAU 157 model units = 1.57 s at τ_u = 10 ms
    (Nagel & Wilson 2016 ORN→LN);
  - a rested neuron transmits undepressed, so **g_lin = 0.1632 is unchanged**;
  - never use the default `rate` mode, which moves g_lin up 1.2–1.3×.
- **(2) spontaneous ORN drive:** `--drive olfactory=0.037926` (isolated-unit activity 0.04 ≈ 3.6 Hz); every other neuron h = 0.
  - "ORN level" names the isolated drive. Recurrent input lowers the measured ORN rate, so gains use the measured ρ_ORN.

Variants:
- **M2s:** per-target signs, `--edge_sign` lLN1_bc→PN 0, →LN +1.
- **Sensitivity:** all-synapse depression; KW09/N15 depression values; TAU 30 (τ_u = 50 ms); ORN levels 0.01–0.10.
- **Time mapping:** 1 unit = 10 ms; ν(Hz) = 90.6 ρ.

## Environment and compute rules
- Python `/MALECNS_PYBIN/python` (numpy, scipy, torch, matplotlib, networkx 3.3, playwright; no
  igraph/leidenalg).
  - Runners: `code/swc_array.sbatch` (M1), `code/ald_array.sbatch` / `code/m2_array.sbatch` (M2), `theory_branch_gpu.sbatch`,
    `module_job.sbatch`.
  - Always pass:
    - `--constraint="ampere|ada|hopper|blackwell"`, because torch 2.11+cu128 needs CC ≥ 7.5;
    - `--nice=500`, to yield to other jobs;
    - `--exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01`.
  - Submit with `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch`. GPU graph memory is under 1 GB, so any supported card works.
- **Never call squeue/sacct/scontrol from array tasks or per-item loops.** Use at most one `squeue -u $USER` per driver
  Check liveness through files instead.
- Job prefixes of this project: `swc`, `pd-`, `pd3-`, `th-`, `ald-`, `m2-`.
- CPU-only jobs (scipy eigs): `-p thickstun`. Disk: keep everything here (home is nearly full).

## Code (all under `code/`)
| file | what |
| --- | --- |
| `swc.py` | M1 GPU simulator: `--exp decay` (fully active start) / `spread` (single seed; `--seed_set`), `--norm`, `--g` list, `--r`, `--h`, `--null [--null_kind]`, `--T`, `--B`. Saves t, g, rho[g,t,B], surv, occ[g,N]. Do not modify. |
| `ald_sim.py` | M2 simulator, a superset of swc.py and bit-identical to it with options off (re-tested). `--drive`/`--clamp_off` (per-population h, deafferentation), `--sign_override`/`--sign_ensemble`, `--edge_sign` (per-target), `--wtransform`, depression `--std_group … --std_mode spike|rate`, adaptation, refractoriness, `--null_seed`, `--graph`, `--rec_counts` (avalanches), `--pulse`, `--ramp` (hysteresis); per-region traces. Add options only; never change defaults. |
| `m2_common.py`, `m2_obs.py`, `m2_array.sbatch` | M2 shared flags/protocols/grids/regions (`args_line()`, `h_of_rho()`), reference observables (τ, crossings, PR, oscillation flag, avalanches, branching ratio), GPU array runner (jobs `m2-*`). Tests: `m2_sim_test.py`, `m2_obs_test.py`, `m2_validate.py`. Every M2 lane uses these, so the criteria agree. |
| `ald_*.py` | AL-defect experiments: drive/antenna removal (`ald_drive_*`), transmitter signs (`ald_sign_*`), depression (`ald_dep_*`, incl. theory), weight transforms (`ald_wt_*`), verification (`ald_verify*.py`) |
| `al_*.py` | AL data grounding for the literature study: module composition, edges, spectra, recruitment (`al_verify_3.py` recomputes the numbers) |
| `snap.py`, `movie.py` | bit-packed full-state snapshots → `figs/activity_movie.mp4` |
| `spectrum*.py` | eigen-analyses (CPU): symmetrized/directed/signed spectra, exact simulated operator, excitatory-only, λ(r), top-400 tail |
| `theory_qmf.py` | GPU: rectified Perron root μ(r) + discrete-time quenched mean field (M1) |
| `theory_branch.py` / `theory_branch_gpu.py` | M1 onset theory: branching operator B(g)=f(gE), g_lin, pair operator, coincidence term D(g,r), K_eff, exact birth–death lifetime (use the GPU array version) |
| `theory_mode.py`, `module_sim.py` | rectified leading mode/K formula; the 526-neuron AL module alone + `results/module/module_graph.npz` for the browser |
| `phase.py`, `analyze.py`, `figs.py` | stage-1 classification (superseded criteria) |
| `analysis2_*.py` | verified M1 analyses: spreading, long runs, driven plane, complete phase diagram (τ criteria), critic |
| `pd3_*.py` | phase-diagram v2 (M1): module partition/stats, FSS, redraw, multistability, criticality |
| `flow_params.py`, `flow_build.py` | flow-page parameters and build |

## Results (details and numbers: STATE.md and the REPORT.md files named)
**Model 1** (`results/analysis2/phase_complete.md`):
- **Spectrum:** the leading excitatory modes are localized on the antennal-lobe (AL) module (λ_PF(E) = 5.574 raw); all
  top-400 modes are localized. The configuration null is delocalized; the weight-shuffled null is in between.
- **Onset theory:** g_lin = 0.1632 (r-independent).
- **Complete diagram (23 r):** quiescent → slow window (τ 200–3000) → localized active → delocalized.
  - Theory matches at r = 0 (onset .291 vs .266).
  - It overstates inhibition for r > 0 (it needs K ≈ 2× the computed K_eff).
- **Not a Griffiths phase:**
  - Slow-window lifetimes are exponential: one rare region, the AL module.
  - The "varying exponent" was a fit-window artifact.
  - Q/S/L are conventions on a smooth crossover; g_lin is the only non-arbitrary line.
- **Driven plane:** fluctuation peak at g ≈ 0.30 (h = 1e-5).

**AL literature** (`results/al_lit/REPORT.md`, section 11):
- AL dominance comes from an all-to-all cholinergic lLN1_bc clique (15 per lobe, median 204 synapses), not sensory input.
- The AL–sleep literature is modulatory only.
- No empirical AL–consciousness or AL phase-transition result was found.

**AL defect** (`results/aldef/REPORT.md`, section 12; verified):
- **Verdict:** M1's self-sustained AL is a **defect of synapse dynamics**. Two features each remove it, and they tie:
  - missing short-term depression;
  - lLN1_bc acting as plain chemical excitation onto every target.
- **Not the cause:** the weights, or the ACh label (P ≈ 0.75).
- **Antenna-removal constraints:** the real fly sits below ignition, driven by ORNs. Allowed g is 0.09–0.23 (r = 0) and
  0.10–0.29 (r = 1); the palps-intact test caps it at about 0.19.
- **With depression:** the onset rises 2.3–4.8× and is brain-wide; the driven AL is a quiet, moderate amplifier.
- **Interpretation change:** drop the "AL-only regime ≈ sleep" reading.

## In progress / next
- **Model 2 program: DONE 11-09** (wf_aae614ed-511; verified report `results/m2/REPORT.md`, pre-verify synthesis
  `results/m2/REPORT_synthesis_preverify.md`, lanes in `results/m2/<lane>/`, figures `figs/m2/final_*.png`).
  - **h = 0 diagram:** same phase order Q → S → L → D at every r.
    - Onset 1.18–2.02, i.e. 3.4–4.8× M1 and 7–12× g_lin (g_lin unchanged).
    - The first sustained state is brain-wide: 13.6k–19.1k neurons, with the AL holding only 5–6%.
    - D is finite at every r (6.3–14.2).
  - **Driven diagram (ORN level 0.04):** each line equals the corresponding h = 0 line (ratios 0.94–1.03).
    - Data-allowed g: 0.216–0.968 (r 0), 0.317–1.23 (r 1).
    - The palp-intact test passes only below g ≈ 0.20, so no g passes all three data tests at r 0 or 1. This is a
      near miss (7% in g).
  - **Nature of the transition:** continuous and rounded.
    - Hysteresis loops shrink; there is no L/R symmetry breaking, no Griffiths window and no SOC (0 critical-like
      points).
  - **Theory:** it brackets the onset but no point estimate is within 10% (mean field ×0.53–0.80, stochastic
    ×1.41–2.28). The LNA gets the driven gain window at r 0 within 2–9%.
  - **Open:** the palp leak (electrical eLN→PN coupling or LN→LN dynamics?); theory with inhibition; dependence on the
    time unit (TAU 30); the KS bug in `m2_obs.powerlaw_fit` for tied sizes. Verify correction 7 found no other user:
    pd3 never used it, so pd3 needs no recheck.
- **Phase diagram v2 on M1: DONE 11-09** (wf_98fb990d-4fb → `results/pd3/REPORT.md`, with the critic's 9 corrections
  applied by the main session; figures `figs/pd3/phase_diagram_v3.png`, `q2_second_order_evidence.png`,
  `phase_lifetime_heatmap.png`). It is the M1 baseline that Model 2 is compared against.
  - **Diagram:** one solid line, g_lin = 0.1632, independent of r.
  - **Phases:** SL, then M (metastable AL band, edge moves with T), then B_AL (both lobes; AL is 29–49% of all activity), then
    a smeared island cascade, then D (only for r ≤ 1.5).
  - **Not found:** no L/R broken symmetry (0/48), no first-order lines, no coexistence. de Candia's closure predicts them
    (1,426 B1L|B1R cells, 46 first-order intervals), but these are closure artefacts.
  - **Second order?** Not as a phase transition of the brain. The result is consistent with a finite-size-rounded
    continuous onset at g_lin of one fixed module (K_fit ≈ 70 at r = 0, 36 at r = 1). Null networks sharpen with N; the
    real brain does not (the AL alone is within 1.8%).
  - **Universality:** not a single finite-K MF-DP mode; the class is undetermined at K ≈ 10–90. Pre-registered criteria
    a–d FAIL, e (not first order) PASSES.
- Open: per-target signs vs depression (needs a measurement of the lLN1_bc→lLN1_bc synapse); M1 theory's ~2× K error;
  the τ_u dependence of the depression results.
