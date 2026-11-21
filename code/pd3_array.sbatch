#!/bin/bash
#SBATCH -p gpu,ellis,default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# pd3 generic GPU array. usage: env -u SLURM_JOB_ID -u SLURM_JOBID sbatch -J pd3-<prog> --array=0-(n-1)%K pd3_array.sbatch <argsfile>
# line SLURM_ARRAY_TASK_ID+1 of argsfile = '<pd3 script> <args>' (script name relative to code/)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
echo "host $(hostname) task ${SLURM_ARRAY_TASK_ID}: ${LINE}"
nvidia-smi --query-gpu=name --format=csv,noheader
cd /MALECNS_ROOT/code
eval /MALECNS_PYBIN/python /MALECNS_ROOT/code/$LINE
