# pd3 multistability: jobs

All arrays: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch -J pd3-multistability-<part> --array=...%12 code/pd3_array.sbatch results/pd3/multistability/args_<part>.txt`
(code/pd3_array.sbatch: -p gpu,ellis,default_partition, --gres=gpu:1, --constraint="ampere|ada|hopper|blackwell", --nice=500,
--exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01, -t 02:00:00). Args files written by code/pd3_multistab_args.py;
missing outputs: `python code/pd3_multistab_args.py --missing results/pd3/multistability/args_<part>.txt`.

| date (2026-11-09) | job | part | tasks | status / notes |
|---|---|---|---|---|
| 07:06 | 179161 | smoke (all 7 modes, tiny T; outputs in results/pd3/multistab/smoke/) | 1 | done, all modes ran; one stray smoke file icq_r6_g0.5184.npz written to the production dir was deleted |
| 07:08 | 179202 | validate (fast W(r) path + split E/I path) | 2 | done: PASS at all 6 g for both paths (results/pd3/multistab/validation.json); 14.8 ms/step at B = 384 on RTX 6000 Ada (fast), 47.8 ms/step on A6000 (split) |
| 07:21 | 179561 | m1 (al_sym, 7 r) | 7 | done: 7/7 COMPLETED, no reruns |
| 07:21 | 179562 | m2a (36 sweeps + 6 h = 0 down-sweeps) | 42 | done: 42/42 COMPLETED, no reruns |
| 07:21 | 179563 | m2c (r-sweeps, g 0.6 / 2 / 8) | 3 | done: 3/3 COMPLETED |
| 07:21 | 179564 | m3 (ic, 4 r x 6 g) | 24 | done: 24/24 COMPLETED |
| 07:21 | 179565 | m3b (icq, 40 QMF multistable cells in 15 tasks) | 15 | done: 15/15 COMPLETED (40/40 cell files; longest task 48 min) |
| 08:00 | 180689 | m2b (6 loop centres of the pre-registered hysteresis candidates + 1 control: largest remaining loop, r = 0, h = 1e-5, g 0.225; args_m2b.txt from pd3_multistab_args.py --m2b; kept as args_m2b_v1.txt) | 7 | done: 7/7 COMPLETED |
| 08:02 | 180780 | m2b, indices 0,7,8,10 of the regenerated args_m2b.txt (11 lines; the first 7-line version is kept as args_m2b_v1.txt): the 4 extra loop centres of candidates found by applying the same pre-registered criterion to the PR loop area A_PR | 4 | done: 4/4 COMPLETED |
| 08:23 | 181423 | m2br: M2b at the loop centre of each M2c r-sweep (fwd run; args_m2b_rsweep.txt; added, see multistability.md deviations) | 3 | done: 3/3 COMPLETED |

All 132 outputs present (2 validation + 7 + 42 + 3 + 24 + 40 + 11 + 3) (`python code/pd3_multistab_args.py --missing <args>` returns nothing for every part); no task failed, was preempted or timed out (sacct). Total in-script GPU wall time ~19.4 h (sum of `seconds` in the output metas).
