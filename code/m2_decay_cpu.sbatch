#!/bin/bash
#SBATCH -J m2-decay-sum
#SBATCH -p thickstun
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# CPU array for lane m2/decay: summarize sim files (shard = array task id of n) or run another subcommand.
# usage: sbatch --array=0-(n-1) m2_decay_cpu.sbatch sum n     |  sbatch m2_decay_cpu.sbatch analyze
cd /MALECNS_ROOT/code
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
if [ "$1" = "sum" ]; then
  /MALECNS_PYBIN/python m2_decay_analyze.py sum --shard ${SLURM_ARRAY_TASK_ID:-0} ${2:-1}
else
  /MALECNS_PYBIN/python m2_decay_analyze.py "$@"
fi
