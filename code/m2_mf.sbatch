#!/bin/bash
#SBATCH -J m2-theory-mf
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# Mean-field (lane m2/theory) GPU array: line SLURM_ARRAY_TASK_ID+1 of <argsfile> = the arguments of one python call
# (script path first). usage: env -u SLURM_JOB_ID -u SLURM_JOBID sbatch -J m2-theory-<name> --array=0-(n-1)%K m2_mf.sbatch <argsfile>
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
echo "host $(hostname) task ${SLURM_ARRAY_TASK_ID}: ${ARGS}"
nvidia-smi --query-gpu=name --format=csv,noheader
eval /MALECNS_PYBIN/python ${ARGS}
