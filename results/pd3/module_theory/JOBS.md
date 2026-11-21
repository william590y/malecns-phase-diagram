# module_theory jobs (program MT of results/pd3/plan.md section 3), 2026-11-09

All GPU jobs: `code/pd3_mt_array.sbatch <argsfile>` (one `<script> <args>` line per array index; `-p gpu,ellis,default_partition
--gres=gpu:1 --constraint="ampere|ada|hopper|blackwell" --nice=500 --exclude=...`), submitted with
`env -u SLURM_JOB_ID -u SLURM_JOBID sbatch -J pd3-module_theory-<x>`. CPU jobs: `code/pd3_mt_cpu.sbatch` (thickstun, 16 cores, 128G).
Logs: `logs/pd3-module_theory-<x>_<job>[_<task>].out`.

| job | name | what | final status |
|---|---|---|---|
| (local) | — | `pd3_partition_v0.py` -> `results/pd3/modules/partition_P0.npz` (34 modules) | done |
| 175264 | pd3-module_theory-smoke | `pd3_qmf_ic.py --r 1 --smoke`, pure float64 | done: 7x slower than needed; found that the QMF does not settle at g >= ~1.4 (r = 1) |
| 175376 | pd3-module_theory-smoke2 | smoke test, float32 sparse product + float64 state | done |
| 175422 | pd3-module_theory-qmf | first MT3 production attempt (9 tasks) | cancelled by me after ~8 min: float32 rounding cycles keep residuals ~1e-8 > 1e-9, every column would have run to the step cap |
| 175633 | pd3-module_theory-smoke3 | smoke test of the float32 + float64-polish scheme | done (polish reaches < 1e-9) |
| 175683 | pd3-module_theory-qmf | MT3, 9 tasks r in {0,.25,.5,1,1.5,2,3,4,6} (`args_qmf.txt`) | tasks 0, 2-8 done; task 1 (r = 0.25) PREEMPTED 05:18 (portal-compute-03), not requeued |
| 176342 | pd3-module_theory-qmf | rerun of task 1 only (r = 0.25; `args_qmf_rerun1.txt`, --requeue) | done |
| 175284 | pd3-module_theory-leiden | `pd3_modules.py` (CPU): Leiden P1 (gamma x seeds) + P2 | done: gamma 4, seed 1 |
| (local) | — | `pd3_module_stats.py P0` | done |
| 176327 | pd3-module_theory-statsP1 | `pd3_module_stats.py P1` (CPU) | done |
| 176328 | pd3-module_theory-statsP2 | `pd3_module_stats.py P2` (CPU, after 176327) | done |
| (local) | — | `pd3_coarse_mf.py P0 --validate` | done: 1525/1525 labels identical to theory_module2 |
| 175507 | pd3-module_theory-coarse | first MT2 P0 run | cancelled by me after ~25 min (no freeze rule: 29% of columns never settle, see module_theory.md deviation 6) |
| 176013 | pd3-module_theory-coarse | MT2 P0 with freeze rule | PREEMPTED 05:14 (bhattacharjee-compute-03), not requeued |
| 176275 | pd3-module_theory-coarse | MT2 P0, --requeue | PREEMPTED again 05:27 (bhattacharjee-compute-01); I cancelled the requeue |
| 176386 | pd3-module_theory-coarse | MT2 P0 on `-p ellis` | done (4,633 s) |
| 176402 | pd3-module_theory-coarse | MT2 P1 on `-p ellis` | done (2,658 s) |
| 175546 | pd3-module_theory-island | MT5, 12 tasks: AL, isl1-5 alone x r in {0, 1} (`args_island.txt`) | done |
| 176515 | pd3-module_theory-multi | MT3b `pd3_qmf_multi.py`, first pass (r = 0.5, 1.5, 2) | done (superseded by 177888) |
| 177888 | pd3-module_theory-multi | MT3b on all r after the distance criterion (`args_multi.txt`) | done |

Post-processing on CPU (no job): `pd3_qmf_analyze.py`, `pd3_coarse_post.py P0/P1`, `pd3_module_occ.py P0`, figure scripts,
`pd3_module_theory_summary.py`. Nothing is still running for this program.
