#!/bin/bash
#SBATCH -J mcns-swc-smoke
#SBATCH -p gpu,ellis,default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada|a40|l40s"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o /MALECNS_ROOT/logs/swc_smoke_%j.out
nvidia-smi --query-gpu=name --format=csv,noheader
/MALECNS_PYBIN/python /MALECNS_ROOT/code/swc.py --exp decay --norm in --g 0.5 1 2 4 8 --T 100 --dt 0.2 --B 16 --out /MALECNS_ROOT/results/smoke_decay.npz
/MALECNS_PYBIN/python /MALECNS_ROOT/code/swc.py --exp spread --norm in --g 4 8 --T 100 --dt 0.2 --B 256 --out /MALECNS_ROOT/results/smoke_spread.npz
