#!/bin/bash
# lane m2/driven: every 5 min resubmit preempted tasks (guarded) for the given waves; stops after $1 minutes.
END=$(( $(date +%s) + 60 * ${1:-300} )); shift
cd /MALECNS_ROOT/code
while [ $(date +%s) -lt $END ]; do
  for w in "$@"; do
    /MALECNS_PYBIN/python m2_driven_rerun.py $w --submit --limit 10 2>&1 | sed "s/^/$(date +%H:%M) /"
  done
  sleep 300
done
