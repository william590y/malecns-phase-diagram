#!/bin/bash
#SBATCH -J m2-modules-cpu
#SBATCH -p thickstun
#SBATCH -c 4
#SBATCH --mem=12G
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
echo cpu run
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
eval /MALECNS_PYBIN/python ${SIM} $ARGS
