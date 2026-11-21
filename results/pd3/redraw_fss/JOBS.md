# redraw_fss jobs (program redraw_fss of results/pd3/plan.md)

| job | name | what | status |
|---|---|---|---|
| 175132 | pd3-redraw_fss-smoke | 4-task smoke test of pd3_fss_sim.py (2 g points, T 600; cnull 10k s1 r0, realsub 5k s1 r1, cnull 167216 s0 r0 / s1 r1) -> results/pd3/redraw_fss/smoke/ | COMPLETED (all 4 ok; validation exact) |
| 175172 | pd3-redraw_fss-sim | production FSS array, 54 tasks (%16, -t 03:00:00), args results/pd3/redraw_fss/args_fss.txt (cnull 7 N x 3 seeds, realsub 3 N x 2 seeds, r 0/1) -> results/pd3/redraw_fss/fss_*.npz | COMPLETED 53/54 (task 35 preempted, rerun 176015); longest task 1 h 11 min (cnull 167216 s1 r1) |
| 175681 | pd3-redraw_fss-fine | fine-grid follow-up batch 1 (12 tasks, %8, -t 02:00:00): code/pd3_fss_fine.py, 16 ln-spaced x across each coarse run's [x_S lower bracket, x_on upper bracket]; args results/pd3/redraw_fss/args_fine_1.txt (cnull N >= 20k finished at submission) -> fss_fine_*.npz | COMPLETED 11/12 (task 8 preempted, rerun in 175816) |
| 175681_8 | pd3-redraw_fss-fine | cnull 40000 s3 r0 fine task PREEMPTED after 36 s (not requeued); rerun as line 1 of args_fine_2.txt (resumes from its partial checkpoint) | rerun below |
| 175816 | pd3-redraw_fss-fine | fine-grid batch 2 (5 tasks): rerun of 175681_8 + cnull r = 1 runs (20k s3, 40k s2/s3, 80k s1) finished since batch 1; args results/pd3/redraw_fss/args_fine_2.txt | COMPLETED |
| 175172_35 | pd3-redraw_fss-sim | cnull 10000 s2 r0 PREEMPTED after 50 s (not requeued) | rerun as 176015 |
| 176015 | pd3-redraw_fss-sim | rerun of 175172_35 (args results/pd3/redraw_fss/args_rerun_1.txt; resumes from checkpoint) | COMPLETED |
| 176126 | pd3-redraw_fss-fine | fine-grid batch 3 (5 tasks): cnull 167216 s0/s1/s2 r0, cnull 80000 s2/s3 r1; args results/pd3/redraw_fss/args_fine_3.txt | COMPLETED |
| 176564 | pd3-redraw_fss-fine | fine-grid batch 4 (2 tasks): cnull 167216 s0/s2 r1; args results/pd3/redraw_fss/args_fine_4.txt | COMPLETED |
| 176756 | pd3-redraw_fss-fine | fine-grid batch 5 (1 task): cnull 167216 s1 r1; args results/pd3/redraw_fss/args_fine_5.txt | COMPLETED |

**Final state (2026-11-09):** 54/54 coarse outputs `fss_{cnull,realsub}_N*_s*_r*.npz` and 24/24 fine outputs `fss_fine_cnull_N*_s*_r*.npz` (all cnull with N >= 20k, both r); no partial checkpoints left. Two tasks were preempted without requeue and were rerun from their checkpoints (175172_35 -> 176015_0, 175681_8 -> 175816_0). Analysis: `code/pd3_fss_analyze.py` (CPU, local) -> `fss_analysis.json`, `fss_analysis.log`; heat map `code/pd3_redraw.py` (CPU, local) -> `results/pd3/redraw/`.
