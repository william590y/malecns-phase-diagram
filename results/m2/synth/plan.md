# Lane m2/synth: plan and pre-registered criteria (written 2026-11-09 ~19:50, before any recheck or table was computed)

Synthesis of the M2 restart (workflow wf_aae614ed-511). Inputs: results/m2/{MODEL2.md, decay, driven, nature, modules,
theory}. M1 references: results/analysis2/phase_complete.md(.json), results/pd3/REPORT.md, results/aldef/REPORT.md.
No new simulation; CPU analysis only. Code: code/m2_synth_recheck.py, code/m2_synth_table.py, code/m2_synth_figs.py.
Outputs: results/m2/synth/{recheck.json, agreement.json, agreement.csv}, figs/m2/final_*.png.

## 1. Independent recheck of the lane numbers (own code, imports no project module)
- Decay (setting a): from the raw npz files listed in results/m2/decay/sim/index_s1.json and index_s2*.json (plus their
  reruns, matched by output path; T = 3000 only; long runs s3a, where-at-S s3b and x9 excluded), pool replicas per
  (model, graph, r, g). tau = censored-exponential MLE sum(min(t_i, 3000))/deaths, t_i = first record time with surv = 0.
  Own PAVA on ln tau, first crossing of ln 200 (S) and ln 3000 (onset), linear in ln g. D = first g with PR >= N/4 on
  the occupancy of sustained points (own PAVA on PR).
  Rows: M2 real (all 23 r), M2 configuration null and weight-shuffled null (7 core r), M2s (7 core r), M1 real (r 0, 1, 4).
- Driven (setting b, ORN 0.04): from results/m2/driven/runs/<set>/*.npz (all chunks and refinements of a row). Per g:
  G = mean_b(uPN_ant win 500-1500) / mean_b(ORN win 500-1500); tau_post = censored MLE of reg_last[nonORN] - 1500
  (censor 3000); S1 = mean uPN_ant over 1600-1800 <= 0.1 x its driven mean. Lines: first crossing of G = 0.29 (DQ->DA),
  of tau_post = 3000 (DT->SS), and the first g where S1 fails (allowed top by S1), all linear in ln g after own PAVA.
  Rows: M2 real (all 23 r), M1, both nulls, M2s (7 core r).
- **Pass rule:** a recomputed line passes if it lies within 2% of the lane value or inside the lane's total interval.
  Any failure is reported with both values; the lane value is not silently replaced.

## 2. Theory vs simulation agreement (the stochastic lane's pre-registered rules, stochastic_plan.md sec. 3, applied
to both theory lanes; written there before any comparison)
- **Boundary agrees** if the predicted value lies inside the simulated total interval or within +-10% of the simulated
  value. Else **fails**, with the ratio theory/simulation reported.
- **Bracket**: a theory that only gives bounds is graded "bracket holds" if lower <= sim <= upper, else "bracket fails".
  The bracket width (upper/lower) is reported, because a wide bracket is weak.
- **r-trend agrees** if the predicted on(r = 6)/on(r = 0) is within +-20% of the simulated ratio.
- Which value is graded as primary (fixed by each lane before seeing experiments):
  - meanfield lane: CQMF; onset/S point estimate = the equal-size rule (amendment A1, labelled heuristic and expected to
    UNDER-estimate); bounds = cutoff line (lower) and AL-mode chain, adiabatic and frozen (upper); D = PR(N/4) of CQMF.
    Driven: G = 0.29 crossing and the equal-size removal-SS line.
  - stochastic lane: single renormalised mode, 2-D chain (PF at r = 0; M1-calibrated, CAL, at r > 0); bracket =
    pooled 2-D (lower) / single (upper); FSS null onsets; driven LNA gain window at r = 0.
  - Secondary variants (QMF, frozen/adiabatic, fluctuation-based estimate) are reported but not graded.
- Qualitative predictions (bistability, OSC/Hopf, Levina verdict, branching ratio, Griffiths, L/R symmetry, where activity
  lives (PR within x2, AL share within x2)) are graded agree / disagree against the experiment lanes' pre-registered
  verdicts.
- A prediction that agrees only through an M1-calibrated factor is marked CAL.

## 3. Figures (final_*.png)
- final_decay.png: setting (a) M2 real (S, onset, D with intervals) + theory (MF equal-size, MF cutoff, AL-chain adiabatic,
  stochastic single) + g_lin; panel with M1 lines for comparison.
- final_decay_nulls.png: configuration and weight-shuffled nulls, sim vs theory.
- final_driven.png: setting (b) driven phase map of M2 real at ORN 0.04 (phase tiles from the lane's curves), allowed
  region, decay onset overlay, theory lines (MF G = 0.29, MF removal-SS equal-size, stochastic gain window at r = 0).
- final_m1_vs_m2.png: M1 and M2 diagrams side by side on the same axes, with AL share and PR at onset.
- final_agreement.png: theory/simulation ratio for every graded boundary vs r, with the +-10% band.

## 4. What this lane will not do
- No new simulation. No change to any other lane's files. No REPORT.md (returned to the workflow instead).
