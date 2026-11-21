#!/bin/bash
#SBATCH -p gpu,ellis,default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# usage: sbatch --array=0-(n-1)%k swc_array.sbatch <argsfile>; line SLURM_ARRAY_TASK_ID+1 of argsfile = swc.py arguments
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
nvidia-smi --query-gpu=name --format=csv,noheader
eval /MALECNS_PYBIN/python /MALECNS_ROOT/code/swc.py $ARGS
