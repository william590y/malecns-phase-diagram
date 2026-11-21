#!/bin/bash
#SBATCH -J m2-verify-tests
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
# Lane m2/verify: live re-run of the simulator unit tests and observable tests on the CURRENT code/ald_sim.py, from copies
# of code/m2_sim_test.py / m2_obs_test.py that write to results/m2/verify/simtest (the lane's own files are untouched).
D=/MALECNS_ROOT/results/m2/verify/simtest
C=/MALECNS_ROOT/code
echo "host $(hostname) ald_sim md5 $(md5sum $C/ald_sim.py | cut -c1-8)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd $D
/MALECNS_PYBIN/python $D/m2_sim_test_copy.py $C/ald_sim.py $C/ald_sim.py.bak-m2-pre > $D/unit.log 2>&1; echo "unit exit $?"
/MALECNS_PYBIN/python $D/m2_obs_test_copy.py > $D/obs.log 2>&1; echo "obs exit $?"
