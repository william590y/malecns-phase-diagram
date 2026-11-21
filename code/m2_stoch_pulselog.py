"""Recover pulse-task rows from the job logs (the tasks were cancelled after the informative g range to save GPU time;
each log line holds every number of a row). Writes results/m2/theory/stochastic/pulse/<name>.json."""
import json, re, sys
L = "/MALECNS_ROOT/logs"
R = "/MALECNS_ROOT/results/m2/theory/stochastic/pulse"
pat = re.compile(r"gap (\S+) g (\S+) m (\S+) eps (\S+) <S> (\S+) Tav (\S+) Ks (\S+) S_eps (\S+) S_dep (\S+) AL_F (\S+) xAL (\S+) it (\S+) dx (\S+) (\w+)")
for job, name, model in (("210428_0", "M2", "M2"), ("210428_2", "M2_gap100", "M2"), ("210428_4", "M2_cnull", "M2")):
    rows = []
    for line in open(f"{L}/m2-stoch-pulse_{job}.out"):
        m = pat.search(line)
        if m:
            v = m.groups()
            f = lambda x: float(x)
            rows.append({"gap": f(v[0]), "g": f(v[1]), "m": f(v[2]), "eps": f(v[3]), "mean_size": f(v[4]), "T_av": f(v[5]),
                         "K_s": f(v[6]), "S_eps": f(v[7]), "S_dep": f(v[8]), "share_AL_F": f(v[9]), "x_AL": f(v[10]),
                         "it": int(v[11]), "dx": f(v[12]), "method": v[13], "K0": None, "x_min": None, "PR_v": None, "AL_v": None,
                         "converged": (v[13] == "direct" and f(v[12]) < 1e-5) or (v[13] == "bisect" and f(v[3]) > 0)})
    json.dump({"task": "pulse", "model": model, "source": f"log m2-stoch-pulse_{job}.out (task cancelled; rows parsed)",
               "rows": rows}, open(f"{R}/{name}.json", "w"), indent=1)
    print(name, len(rows))
