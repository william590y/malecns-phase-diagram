#!/bin/bash
#SBATCH -J ald-e3-gpu
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
# E3: run one python script on a GPU. usage: sbatch -J ald-e3-<name> ald_dep_gpu.sbatch <script.py> [args...]
echo "host $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd /MALECNS_ROOT/code
/MALECNS_PYBIN/python "$@"
