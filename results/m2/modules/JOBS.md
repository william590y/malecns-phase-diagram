# m2/modules jobs (code/ald_sim.py md5 61580c76, runner code/m2_array.sbatch; args in sim/args_<SET>.txt, index sim/index_<SET>.json)
| job | name | args | tasks | note |
|---|---|---|---|---|
| 202987 | m2-modules-smoke | sim/args_SMOKE.txt | 4 | smoke (short T), submitted 15:02 |
| 203117 | m2-modules-LRa | sim/args_LRa.txt | 62 | Part B (a) L/R starts, submitted 15:07 (%25, 1.5 h limit) |
| 203119 | m2-modules-DRb | sim/args_DRb.txt | 52 | Parts B+C (b) ORN-driven starts (%25, 2 h) |
| 203121 | m2-modules-ICa | sim/args_ICa.txt | 54 | Part C (a) module / random starts (%20, 1.5 h) |
| 203123 | m2-modules-ISO | sim/args_ISO.txt | 54 | Part D(ii) modules alone (%15, 2 h) |
| 203123 | m2-modules-ISO | sim/args_ISO.txt | 54 | cancelled before any task started (15:20): split into ISOg (GPU) + ISOc (CPU) below |
| 203412 | m2-modules-ISOc | sim/args_ISOc.txt | 36 | Part D(ii) small modules alone (AL, AL_L, AL_R, isl1, isl3, isl5) on thickstun CPU (`--device cpu`, code/m2_modules_cpu.sbatch) |
| 203431 | m2-modules-ISOg | sim/args_ISOg.txt | 18 | Part D(ii) CB, VNC, OL alone on GPU (%10, 2 h) |
| 205284 | m2-modules-rr | sim/args_LRa_rerun1.txt | 2 | reruns of preempted LRa tasks 3, 14 (same lines and seeds) |
| – | – | – | – | all outputs present (LRa 62, DRb 52, ICa 54, ISO 54); preempted LRa 3, 14 rerun in 205284; no other failure. Wall time from the sidecars: 222 runs, 21.5 h (19.4 GPU-h + 2.1 CPU-h) |
