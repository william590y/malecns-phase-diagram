#!/bin/bash
#SBATCH -p ellis
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
/MALECNS_PYBIN/python /MALECNS_ROOT/code/snap.py "$@"
