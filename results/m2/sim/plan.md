# Lane m2/sim: Model-2 spec + simulator: plan and pass criteria

Written 2026-11-09. Honest timing: the test pass criteria below were written into the docstrings of
`code/m2_sim_test.py`, `code/m2_obs_test.py` and `code/m2_validate.py` before validation job 186100 ran. This file
copies them and was written while that job was running (before any result was read). The smoke-test criteria (section 3)
were written before smoke array 186175 produced any output.

## 1. Simulator changes (code/ald_sim.py, options ADDED, defaults unchanged)
`--edge_sign`, `--null_seed`, `--graph`, `--rec_counts` (+ `--counts_win`, `--counts_reg`), `--pulse` (+ `--pulse_gap`),
`--ramp` (+ `--ramp_loop`, `--ramp_avg`). They were developed in the work copy `results/m2/sim/dev/ald_sim_work.py`, starting from
`ald_sim_base_92c2e0d3.py` (a byte copy of code/ald_sim.py, md5 92c2e0d3). The work copy is swapped in only if code/ald_sim.py still has md5
92c2e0d3 at swap time; no `cp` over a changed file. The swap is atomic (os.replace of a file in code/).

## 2. Pass criteria (all must hold before the swap)
- **Unit tests** (`code/m2_sim_test.py`, U1-U7): every test passes. The z-tests use |z| < 5. The exact identities are
  bitwise or exact-integer equalities.
- **Observables** (`code/m2_obs_test.py`, O1-O7): every test passes on synthetic data with known answers.
- **Real-cache validation** (`code/m2_validate.py`):
  - V1: swc.py and the new ald_sim.py give bit-identical t/g/rho/surv/occ in 4 cases plus the trivial general path.
  - V2: the old ald_sim.py (md5 92c2e0d3) and the new one give every npz key identically on 5 option sets of the aldef lanes.
  - V3: the trivial new options (identity edge sign, counts, null_seed 0 = seed 0, single-dwell ramp, pulse on a
    never-silent run) change nothing.
- After the swap, the same validation runs again on code/ald_sim.py itself. It must pass again; this is the
  bit-identity re-test of the file the lanes actually use.

## 3. Smoke tests (array m2-sim-smoke, `code/m2_smoke_args.py`, 21 tasks, production lengths, r = 1)
Purpose: show that every protocol runs, and measure the per-run cost (wall time, ms/step, output size) so that lanes
can size their arrays. A smoke test is not a result. Its numbers are illustrative and not used for any phase claim.
Criteria:
- every task exits 0 and writes its npz + json;
- sanity checks (code/m2_smoke_check.py):
  - decay_h0 runs have extinction or survival records;
  - driven runs have ORN activity within 10% of the isolated-unit value 0.04 during the driven window. This is only
    approximate: ORNs receive recurrent input. A larger deviation is reported, not "fixed".
  - after ORN removal the ORN region is exactly 0.
  - pulse runs have avalanches with size >= 1 and a silent step between them.
  - ramp runs have seg_g equal to the requested loop.
  - M2s runs report 6,658 removed lLN1_bc -> ALPN edges.
  - null runs with --null_seed 0 report the same null hash for every task of the same kind.
  - the FSS graph run has N = 20,000.
- Cost table: wall_s, ms/step per g (sidecar) and npz size, per protocol and engine.

## 4. Outcome (2026-11-09 ~11:05)
- Every criterion of section 2 was met, on both work copies and on the live file after each swap.
  - Live code/ald_sim.py md5 61580c76, job 187154: 29/29 unit tests (one test, U8 for `--init`, was added with the second
    version), 11/11 observable tests, V1 5/5, V2 5/5, V3 5/5, V4 4/4 bit-identical.
- Deviations from this plan:
  - (a) A second simulator version added `--init`/`--init_frac` (needed by module-phase lanes); it was validated the same way.
  - (b) The OSC rule in m2_obs.py was revised once, before any production run (MODEL2.md 5a).
  - (c) The U4 depletion test was changed from an absolute to a relative tolerance after the first run. The first version
    passed only because at x ~ 1e-8 the float32 recovery update loses precision.
  - (d) The test harness could not import a `.bak` file; fixed and rerun (job 186308).
  - (e) The driven protocol got `--occ_win 500:1500` before the smoke array, so its PR measures the driven state.
- Smoke (section 3): 21/21 tasks completed (one preemption rerun). All checks passed except "ORN activity within 10% of
  0.04", which failed at g >= 1.5 (0.030, 0.024). That is recurrent inhibition onto ORNs, a model property. It is
  reported in MODEL2.md section 1 and was not changed.
