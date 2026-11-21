#!/bin/bash
#SBATCH -J mcns-specr
#SBATCH -p thickstun
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /MALECNS_ROOT/logs/specr_%j.out
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
/MALECNS_PYBIN/python /MALECNS_ROOT/code/spectrum_r.py
