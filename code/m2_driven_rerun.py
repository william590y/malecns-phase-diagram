"""m2_driven_rerun.py -- resubmit lane m2/driven task lines whose output is missing and whose task is no longer queued
or running (Slurm preemption does not requeue here). Usage: python m2_driven_rerun.py WAVE [--submit] [--limit K]
Reads results/m2/driven/jobs.json (wave -> job ids, the original and earlier reruns) and args_WAVE.txt / index_WAVE.json.
Writes args_WAVE_rerunN.txt + rerun_WAVE_N.json (line map) and records the new job id in jobs.json.
"""
import json, os, subprocess, sys
RES = "/MALECNS_ROOT/results/m2/driven"
CODE = "/MALECNS_ROOT/code"
wave = sys.argv[1]
jobs = json.load(open(f"{RES}/jobs.json"))
lines = open(f"{RES}/args_{wave}.txt").read().splitlines()
idx = json.load(open(f"{RES}/index_{wave}.json"))
# active tasks of this wave: original array + reruns (map rerun task -> original line)
qp = subprocess.run(["squeue", "-u", __import__("os").environ["USER"], "-h", "-o", "%i %j %T"], capture_output=True, text=True)
if qp.returncode != 0 or "m2-" not in qp.stdout:
    sys.exit(f"squeue failed or shows no m2 job (rc {qp.returncode}); not rerunning anything")
q = qp.stdout.split("\n")
active_lines = set()
runs = jobs.get(f"{wave}_reruns", [])
for row in q:
    if not row.strip():
        continue
    jid, name, st = row.split()
    base, _, task = jid.partition("_")
    if base == str(jobs.get(wave)):
        if task.startswith("["):        # pending remainder like [120-254%40]
            rng = task.strip("[]").split("%")[0]
            for part in rng.split(","):
                a, _, b = part.partition("-")
                active_lines.update(range(int(a), int(b or a) + 1))
        else:
            active_lines.add(int(task))
    for rr in runs:
        if base == str(rr["job"]):
            if task.startswith("["):
                rng = task.strip("[]").split("%")[0]
                for part in rng.split(","):
                    a, _, b = part.partition("-")
                    active_lines.update(rr["lines"][int(a):int(b or a) + 1])
            else:
                active_lines.add(rr["lines"][int(task)])
missing = [i for i, m in enumerate(idx) if not (os.path.exists(m["out"]) and os.path.exists(m["out"][:-4] + ".json")) and i not in active_lines]   # sidecar is written after the npz: both must exist
print(f"wave {wave}: {len(missing)} lines to rerun: {missing[:40]}")
MAXN = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 20
if len(missing) > MAXN:
    sys.exit(f"{len(missing)} > --max {MAXN} lines to rerun: refusing (check squeue by hand)")
if missing and "--submit" in sys.argv:
    n = len(runs) + 1
    af = f"{RES}/args_{wave}_rerun{n}.txt"
    open(af, "w").write("\n".join(lines[i] for i in missing) + "\n")
    lim = sys.argv[sys.argv.index("--limit") + 1] if "--limit" in sys.argv else "20"
    out = subprocess.run(["sbatch", "--parsable", "-J", f"m2-driven-{wave}r{n}", f"--array=0-{len(missing) - 1}%{lim}",
                          f"{CODE}/m2_array.sbatch", af], capture_output=True, text=True,
                         env={k: v for k, v in os.environ.items() if k not in ("SLURM_JOB_ID", "SLURM_JOBID")})
    jid = out.stdout.strip()
    print("submitted", jid, out.stderr.strip())
    runs.append({"job": jid, "lines": missing, "args": os.path.basename(af)})
    jobs[f"{wave}_reruns"] = runs
    json.dump(jobs, open(f"{RES}/jobs.json", "w"), indent=1)
