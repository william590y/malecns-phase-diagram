#!/bin/bash
#SBATCH -J ald-e1-test
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
echo "host $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
/MALECNS_PYBIN/python /MALECNS_ROOT/code/ald_drive_test.py --cli
