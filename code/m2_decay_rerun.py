"""m2_decay_rerun.py STAGEFILE OUTFILE JOBID:ARGSFILE [JOBID:ARGSFILE ...]
Lines of STAGEFILE whose --out file is missing and whose output is not being produced by a task that is still queued or
running (task ids of each job index the lines of its own ARGSFILE) -> OUTFILE (same lines, same seeds)."""
import os, re, subprocess, sys
src, dst, jobs = sys.argv[1], sys.argv[2], sys.argv[3:]
out_of = lambda l: re.search(r"--out (\S+)", l).group(1)
live = set()
for ja in jobs:
    j, af = ja.split(":", 1)
    al = open(af).read().splitlines()
    q = subprocess.run(["squeue", "-j", j, "-h", "-r", "-o", "%i"], capture_output=True, text=True).stdout.split()
    for x in q:
        if "_" in x:
            live.add(out_of(al[int(x.split("_")[1])]))
keep = [l for l in open(src).read().splitlines() if not os.path.exists(out_of(l)) and out_of(l) not in live]
open(dst, "w").write("\n".join(keep) + ("\n" if keep else ""))
print(len(keep), "lines ->", dst)
for l in keep:
    print("  ", os.path.basename(out_of(l)))
