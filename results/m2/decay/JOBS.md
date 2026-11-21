# m2/decay jobs (all on code/ald_sim.py md5 61580c76, runner code/m2_array.sbatch)
| job | name | args | tasks | note |
|---|---|---|---|---|
| 187595 | m2-decay-s1 | sim/args_s1.txt | 256 | stage 1 coarse (submitted 11:09) |
| 188257 | m2-decay-s3a | sim/args_s3aa.txt | 24 | long runs M2 real r 0, 1 (11:31) |
| 188258 | m2-decay-s2 | sim/args_s2a.txt | 24 | refinement, first 8 complete rows (11:31) |
| 188612 | m2-decay-s3a | sim/args_s3ab.txt | 12 | long runs M2 real r 4 (11:45) |
| 188613 | m2-decay-s2 | sim/args_s2b.txt | 21 | refinement, 7 more rows (11:45) |
| 188713 | m2-decay-s1r | sim/args_s1_rerun1.txt | 12 | reruns of preempted stage-1 tasks (same lines/seeds; 11:48) |
| 189137 | m2-decay-s2 | sim/args_s2c.txt | 30 | refinement, 11 more rows (12:01) |
| 190323 | m2-decay-s2 | sim/args_s2d.txt | 36 | refinement, 13 more rows (12:24) |
| 190865 | m2-decay-rr | sim/args_rerun2.txt | 6 | reruns: 5 preempted stage-1 tasks + 1 preempted long run (12:35) |
| 190949 | m2-decay-x9 | sim/args_x9.txt | 10 | OSC finite-size check, plan section 9 (12:37) |
| 191821 | m2-decay-s3a | sim/args_s3ae.txt | 12 | long runs M2 configuration null r 1 (12:55) |
| 191822 | m2-decay-s2 | sim/args_s2e.txt | 54 | refinement, 19 more rows (12:55) |
| 191825 | m2-decay-rr | sim/args_s1_rerun3.txt | 1 | rerun of s1_M2_real_r2.75_p3 on ellis only (12:55); cancelled 13:36 while still pending |
| 193475 | m2-decay-s3b | sim/args_s3b.txt | 28 | where-at-S runs (occ_win 50:400), g = S crossing of the 13:20 interim analysis (13:24) |
| 194166 | m2-decay-rr | sim/args_s2_rerun1.txt | 5 | reruns of preempted stage-2 tasks (13:34) |
| 194169 | m2-decay-rr | sim/args_s1_rerun3.txt | 1 | same rerun on gpu,ellis (13:34) |
| 194185 | m2-decay-s2 | sim/args_s2f.txt | 11 | refinement of the rows completed by reruns (13:35) |
| 194186 | m2-decay-rr | sim/args_s1_rerun4.txt | 1 | rerun of s1_M2_tau30_real_r1_p3 (preempted in 190865) (13:35) |
| 197237 | m2-decay-s2 | sim/args_s2g.txt | 6 | refinement of M2 real r 2.75 and M2_tau30 r 1 (14:17) |
| 197244 | m2-decay-rr | sim/args_s2_rerun2.txt | 11 | reruns of preempted stage-2 batch-e tasks (14:17) |
| 198203 | m2-decay-rr | sim/args_s2_rerun3.txt | 1 | rerun of s2_M2_N15_real_r1_D (preempted twice) (14:29) |
