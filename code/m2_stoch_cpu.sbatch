#!/bin/bash
#SBATCH -J m2-stoch-cpu
#SBATCH -p thickstun
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
# lane m2/theory-stochastic CPU job: runs "python <script> <args...>"
echo "host $(hostname): $@"
/MALECNS_PYBIN/python -W ignore "$@"
