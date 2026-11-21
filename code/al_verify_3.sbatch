#!/bin/bash
#SBATCH -p thickstun
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 01:30:00
#SBATCH --nice=500
#SBATCH -J al-verify3
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
# Independent fact-check of the AL data-grounding summary (al_verify_3.py).
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /MALECNS_ROOT/code
/MALECNS_PYBIN/python al_verify_3.py
