#!/bin/bash
#SBATCH -J m2-validate
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
# Model-2 simulator checks on ONE GPU: unit tests (small graphs), observables tests, real-cache validation.
# usage: env -u SLURM_JOB_ID -u SLURM_JOBID sbatch code/m2_validate.sbatch <new ald_sim file> <old ald_sim file> [<prev M2 file>]
echo "host $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd /MALECNS_ROOT/code
PY=/MALECNS_PYBIN/python
$PY m2_sim_test.py "$1" "$2"
$PY m2_obs_test.py
$PY m2_validate.py "$1" "$2" $3
