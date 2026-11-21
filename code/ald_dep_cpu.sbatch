#!/bin/bash
#SBATCH -J ald-e3-ana
#SBATCH -p thickstun
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
# E3: CPU analysis job. usage: sbatch ald_dep_cpu.sbatch <script.py> [args...]
cd /MALECNS_ROOT/code
/MALECNS_PYBIN/python "$@"
