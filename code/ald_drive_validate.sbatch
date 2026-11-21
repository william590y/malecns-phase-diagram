#!/bin/bash
#SBATCH -J ald-e1-validate
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
# E1 (2026-11-09): re-validate ald_sim.py after adding the --switch_at/--win/--reg_last/--rec_lin/--occ_win options.
# Runs the simulator's own ald_test.py --quick and ald_validate.py --skip_perf UNCHANGED except that their output folder
# is redirected to results/aldef/drive/validation (so results/aldef/validation is not overwritten), then ald_drive_test.py --cli.
echo "host $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
CD=/MALECNS_ROOT/code
NEW=/MALECNS_ROOT/results/aldef/drive/validation
PY=/MALECNS_PYBIN/python
cd $CD
$PY - --quick <<PYEOF
import sys
src = open("$CD/ald_test.py").read()
old = 'OUT = f"{S.ROOT}/results/aldef/validation"'
assert src.count(old) == 1
sys.argv = ["$CD/ald_test.py", "--quick"]
exec(compile(src.replace(old, 'OUT = "$NEW"'), "$CD/ald_test.py", "exec"), {"__name__": "__main__", "__file__": "$CD/ald_test.py"})
PYEOF
$PY - <<PYEOF
import sys
src = open("$CD/ald_validate.py").read()
old = 'OUT = f"{ROOT}/results/aldef/validation"'
assert src.count(old) == 1
sys.argv = ["$CD/ald_validate.py", "--skip_perf"]
exec(compile(src.replace(old, 'OUT = "$NEW"'), "$CD/ald_validate.py", "exec"), {"__name__": "__main__", "__file__": "$CD/ald_validate.py"})
PYEOF
$PY $CD/ald_drive_test.py --cli
