#!/bin/bash
#SBATCH -J ald-sim
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# E4 copy of ald_array.sbatch running the frozen snapshot results/aldef/weights/ald_sim_used.py (md5 in ald_sim_used.md5).
# AL-defect study (ald_*) GPU array: line SLURM_ARRAY_TASK_ID+1 of <argsfile> = the arguments of ONE ald_sim.py run
# (blank lines / lines starting with '#' are not allowed: the line number is the task id).
# usage: env -u SLURM_JOB_ID -u SLURM_JOBID sbatch -J ald-<name> --array=0-(n-1)%K \
#            /MALECNS_ROOT/code/ald_array.sbatch <argsfile>
# Job names must start with "ald-". Quote selectors inside the args file, e.g. --drive "olfactory=0.02".
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
echo "host $(hostname) task ${SLURM_ARRAY_TASK_ID}: ${ARGS}"
nvidia-smi --query-gpu=name --format=csv,noheader
eval /MALECNS_PYBIN/python /MALECNS_ROOT/results/aldef/weights/ald_sim_used.py $ARGS
