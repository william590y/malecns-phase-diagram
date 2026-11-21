#!/bin/bash
#SBATCH -p gpu,ellis,default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"   # torch 2.11+cu128 needs CC >= 7.5; older cards fail with no kernel image
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
nvidia-smi --query-gpu=name --format=csv,noheader
/MALECNS_PYBIN/python /MALECNS_ROOT/code/module_sim.py "$@"
