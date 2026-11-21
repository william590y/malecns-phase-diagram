#!/bin/bash
#SBATCH -J th-mode
#SBATCH -p thickstun
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
/MALECNS_PYBIN/python /MALECNS_ROOT/code/theory_mode.py "$@"
