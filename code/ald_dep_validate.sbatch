#!/bin/bash
#SBATCH -J ald-e3-validate
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
# E3: unit tests of --std_mode spike + full-CNS bit-identity checks. usage: sbatch ald_dep_validate.sbatch <new> <old>
echo "host $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd /MALECNS_ROOT/code
/MALECNS_PYBIN/python ald_dep_test.py "$1"
/MALECNS_PYBIN/python ald_dep_validate.py "$1" "$2"
