"""m2_nature_rerun.py -- rerun lines of lane m2/nature whose output is missing and whose array task ended without
success (PREEMPTED / FAILED / CANCELLED / NODE_FAIL / TIMEOUT / OUT_OF_MEMORY) and is not queued any more.
Same line, same seed, same output path.

  python m2_nature_rerun.py <name> <jobid> [<jobid> ...]     (name = args_<name>.txt / index_<name>.json; the jobids
                                                             are the arrays that ran those lines, task id = line index)
Writes args_<name>_rrK.txt + index_<name>_rrK.json (K = next free number) and prints the number of lines.
"""
import json, os, subprocess, sys

OUT = "/MALECNS_ROOT/results/m2/nature"
BAD = ("PREEMPTED", "FAILED", "CANCELLED", "NODE_FAIL", "TIMEOUT", "OUT_OF_MEMORY")


def main():
    name, jobs = sys.argv[1], sys.argv[2:]
    lines = open(f"{OUT}/args_{name}.txt").read().splitlines()
    idx = json.load(open(f"{OUT}/index_{name}.json"))
    states = {}
    for j in jobs:
        out = subprocess.run(["sacct", "-j", j, "-X", "-n", "-P", "-o", "JobID,State"], capture_output=True,
                             text=True).stdout
        for row in out.splitlines():
            jid, st = row.split("|")[:2]
            if "_" in jid and "[" not in jid:
                t = int(jid.split("_")[1])
                states.setdefault(t, []).append(st.split()[0])
    q = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j"], capture_output=True,
                       text=True).stdout.split()
    rows = []
    for t, (l, e) in enumerate(zip(lines, idx)):
        if os.path.exists(e["out"]):
            continue
        st = states.get(t, [])
        if st and st[-1] in BAD:
            rows.append((l, e))
    k = 1
    while os.path.exists(f"{OUT}/args_{name}_rr{k}.txt"):
        k += 1
    if not rows:
        print(0); return
    with open(f"{OUT}/args_{name}_rr{k}.txt", "w") as f:
        for l, e in rows:
            f.write(l + "\n")
    json.dump([e for l, e in rows], open(f"{OUT}/index_{name}_rr{k}.json", "w"), indent=0)
    print(len(rows), f"{OUT}/args_{name}_rr{k}.txt", [e["name"] for l, e in rows])


if __name__ == "__main__":
    main()
