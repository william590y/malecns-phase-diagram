#!/bin/bash
#SBATCH -J m2-sim
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# Model-2 GPU array: line SLURM_ARRAY_TASK_ID+1 of <argsfile> = the arguments of ONE ald_sim.py run (no blank or
# comment lines: the line number is the task id). Build the lines with code/m2_common.py args_line().
# usage: env -u SLURM_JOB_ID -u SLURM_JOBID sbatch -J m2-<lane>-<name> --array=0-(n-1)%K \
#            /MALECNS_ROOT/code/m2_array.sbatch <argsfile>
# Job names must start with "m2-". SIM (env) overrides the simulator file (default code/ald_sim.py).
SIM=${SIM:-/MALECNS_ROOT/code/ald_sim.py}
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
echo "host $(hostname) task ${SLURM_ARRAY_TASK_ID} sim ${SIM} md5 $(md5sum ${SIM} | cut -c1-8): ${ARGS}"
nvidia-smi --query-gpu=name --format=csv,noheader
eval /MALECNS_PYBIN/python ${SIM} $ARGS
