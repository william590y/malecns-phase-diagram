#!/bin/bash
#SBATCH -p gpu,ellis,default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# usage: sbatch --array=0-(n-1) theory_branch_gpu.sbatch <kind> <r,..> <FMAX> <NG> <ntasks> <suffix>
/MALECNS_PYBIN/python /MALECNS_ROOT/code/theory_branch_gpu.py "$1" "$2" "$3" "$4" "$SLURM_ARRAY_TASK_ID" "$5" "$6"
