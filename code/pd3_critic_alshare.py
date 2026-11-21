#!/usr/bin/env python
"""pd3 critic: how silent is 'the rest of the brain' in the B_AL band? (REPORT section 3.1 says 'nearly silent').
From the raw occupancies of the real-graph T = 3000 decay runs (h = 0), pooled by g (weights B), for r = 0, 1, 6:
  AL share of total activity = sum(occ over the 526 AL-module neurons) / sum(occ over all neurons)
  mean occupancy of the AL neurons and of the non-AL neurons; PR/N of the occupancy.
Rows are printed between the tau = 3000 onset and the island-1 1% line (phase_complete.json, module_occ_P0.json).
Output: results/pd3/critic_alshare.json.  CPU only.
"""
import glob
import json

import numpy as np

ROOT = "/MALECNS_ROOT"
pc = json.load(open(f"{ROOT}/results/analysis2/phase_complete.json"))
occj = json.load(open(f"{ROOT}/results/pd3/modules/module_occ_P0.json"))
memb = np.load(f"{ROOT}/results/pd3/modules/partition_P0.npz", allow_pickle=True)
names = list(memb["names"]); m = memb["membership"]
AL = np.isin(m, [names.index("AL-L"), names.index("AL-R"), names.index("AL-mid")])
print("AL neurons:", int(AL.sum()))
files = sorted(glob.glob(f"{ROOT}/results/pd/decay_r*_real.npz")) + sorted(glob.glob(f"{ROOT}/results/pd2/onset_real_r*.npz"))
out = {}
for r in (0.0, 1.0, 6.0):
    on = pc["sim"]["real"][f"{r:g}"]["on"]["g"]; i1 = occj["sim"][f"{r:g}"]["cross_1pct_survivors"]["isl1"]
    pts = []
    for f in files:
        d = np.load(f, allow_pickle=True); mt = json.loads(str(d["meta"]))
        if float(mt["r"]) != r or abs(float(mt["T"]) - 3000) > 1e-6 or float(mt["h"]) != 0 or mt.get("null") or "occ" not in d.files:
            continue
        half = d["t"] >= 1500
        for i, g in enumerate(d["g"]):
            if on * 0.999 <= g <= 1.6 * i1:
                o = d["occ"][i].astype(np.float64); s2 = float(d["surv"][i][half].mean())
                pts.append((float(g), float(o[AL].sum() / max(o.sum(), 1e-300)), float(o[AL].mean()), float(o[~AL].mean()),
                            float(o.sum() ** 2 / max((o ** 2).sum(), 1e-300) / len(o)), s2, f.split("/")[-1]))
    pts.sort()
    out[f"{r:g}"] = {"g_on": on, "g_isl1_1pct": i1,
                     "rows": [dict(zip(["g", "AL_share", "occ_AL_mean", "occ_nonAL_mean", "PR_over_N", "surv2", "file"], p)) for p in pts]}
    print(f"r = {r:g}: onset {on:.4f}, island-1 1% {i1:.4f}")
    for p in pts:
        print(f"   g {p[0]:.4f}  AL share {p[1]:.3f}  occ AL {p[2]:.3f}  occ non-AL {p[3]:.2e}  PR/N {p[4]:.2e}  surv2 {p[5]:.2f}  {p[6]}")
json.dump({"generated_by": "code/pd3_critic_alshare.py", "results": out}, open(f"{ROOT}/results/pd3/critic_alshare.json", "w"), indent=1)
