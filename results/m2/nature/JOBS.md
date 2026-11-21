# m2/nature jobs (code/ald_sim.py md5 61580c76; runner code/m2_array.sbatch unless noted)
| job | name | args | tasks | submitted |
|---|---|---|---|---|
| 202761 | m2-nature-graphs | (wrap) m2_nature_graphs.py write | 1 | 15:00 |
| 202976 | m2-nature-glin | m2_nature_py.sbatch m2_nature_graphs.py glin | 1 | 15:02 |
| 202977 | m2-nature-hyst | args_hyst.txt | 96 | 15:02 |
| 202979 | m2-nature-fss1 | args_fss1.txt | 85 | 15:02 |
| 202980 | m2-nature-aval | args_aval.txt | 132 | 15:02 |
| 202981 | m2-nature-dfss | args_dfss.txt | 56 | 15:02 |
| 203087 | m2-nature-graphsu | (wrap) m2_nature_graphs.py write + glin realsubu | 1 | 15:06 |
| 203088 | m2-nature-fss1realsubu | args_fss1realsubu.txt (after 203087) | 18 | 15:06 |
| 203089 | m2-nature-dfssrealsubu | args_dfssrealsubu.txt (after 203087) | 24 | 15:06 |
| 203236 | m2-nature-drvbr | (wrap, CPU thickstun) m2_nature_aval.py driven --shard i 6 | 6 | 15:09 |
| 204057 | m2-nature-drvbr | (wrap, CPU thickstun) m2_nature_aval.py driven --shard i 12 (resubmitted with the tie-correct KS; 203236 cancelled, outputs deleted) | 12 | 15:22 |
| 204075 | m2-nature-rr | args_hyst_rr1.txt | 5 | 15:23 |
| 204076 | m2-nature-rr | args_fss1_rr1.txt | 1 | 15:23 |
| 204085 | m2-nature-hystx | args_hystx.txt (post hoc, amendment 3) | 8 | 15:23 |
| 204128 | m2-nature-fss2 | args_fss2a.txt (stage 2, complete stage-1 curves) | 82 | 15:26 |
| 204316 | m2-nature-avsum | (wrap, CPU thickstun) m2_nature_aval.py pulse --shard i 12 (files present at 15:37) | 12 | 15:37 |
| 205075 | m2-nature-avsum | (wrap, CPU) pulse summaries, second pass | 6 | 15:49 |
| 205081 | m2-nature-rr | args_aval_rr1.txt | 2 | 15:49 |
| 206277 | m2-nature-fss2 | args_fss2b.txt (stage 2: realsubu + cnull 80k s1 r4) | 21 | 16:08 |
| 208788 | m2-nature-fss3 | args_fss3.txt (stage 3: refine the extended realsub / realsub-cn cells) | 6 | 16:30 |
| 208788 | m2-nature-fss3 | args_fss3.txt (stage 3: refine the extended realsub / realsub-cn cells) | 6 | 16:27 |

All arrays finished; every line in every index_*.json has its output (preempted tasks rerun with the same line and seed:
hyst 5, fss1 1, aval 2). Total 539 GPU tasks, 38.8 GPU-h (sacct). Analyses ran on the login node or thickstun CPU jobs
(m2-nature-drvbr 204057, m2-nature-avsum 204316 / 205075).
