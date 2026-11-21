#!/bin/bash
#SBATCH -p thickstun
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH -t 02:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
# AL data grounding: region edge statistics, region Perron roots, exact lambda decomposition (al_edges.py).
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /MALECNS_ROOT/code
/MALECNS_PYBIN/python al_edges.py
