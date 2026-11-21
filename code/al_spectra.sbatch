#!/bin/bash
#SBATCH -p thickstun
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# AL data grounding: whole-CNS excitatory eigen-solves under structural perturbations (al_spectra.py), 4 tasks.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /MALECNS_ROOT/code
case $SLURM_ARRAY_TASK_ID in
 0) V="full peel no_AL ct_nt LN_nonexc" ;;
 1) V="no_ORN no_AL_sensory cap50 cap20 cap10 log1p sqrt binary" ;;
 2) V="soma_only soma_cap50 noORN_cap50 drop_ORN>LN drop_ORN>PN drop_ORN>ORN drop_PN>LN" ;;
 3) V="drop_LN>PN drop_PN<>LN drop_LN>LN drop_PN>PN drop_LN>ORN drop_PN>ORN drop_HRN/TRN>any drop_PN_presyn" ;;
esac
/MALECNS_PYBIN/python al_spectra.py $V
