#!/bin/bash
#SBATCH -J m2-driven-analyze
#SBATCH -p thickstun
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
# CPU analysis of lane m2/driven: extraction (cached) + curves/lines/bootstrap + figures.
cd /MALECNS_ROOT/code
/MALECNS_PYBIN/python m2_driven_analyze.py all --jobs 30 && /MALECNS_PYBIN/python m2_driven_figs.py
