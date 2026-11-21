#!/bin/bash
#SBATCH -J ald-e4-spec
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# E4 weight-transform spectra: one transform per array task (0-6).
TR=(linear cap:50 cap:20 cap:10 sqrt log1p binary)
echo "host $(hostname) task ${SLURM_ARRAY_TASK_ID}: ${TR[$SLURM_ARRAY_TASK_ID]}"
nvidia-smi --query-gpu=name --format=csv,noheader
/MALECNS_PYBIN/python /MALECNS_ROOT/code/ald_wt_spec.py ${TR[$SLURM_ARRAY_TASK_ID]}
