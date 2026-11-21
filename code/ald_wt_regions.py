"""E4: per-region neuron index files (al_common.labels coarse regions) for ald_sim.py --region "NAME=idx:<file>"
-> results/aldef/weights/regions/<NAME>.npz (key idx) + regions.json (sizes)."""
import json, sys
import numpy as np
ROOT = "/MALECNS_ROOT"
sys.path.insert(0, f"{ROOT}/code")
from al_common import labels   # noqa: E402
L = labels()
reg, cat, ty = L["reg"], L["cat"], L["ty"]
R = {"MB": reg == "MB", "MB_KC": cat == "MB KC", "VNC": reg == "VNC",
     "OL": np.isin(reg, ["OL intrinsic", "VPN/VCN", "OL photoreceptors"]), "CX": reg == "CX", "LH": reg == "LH",
     "CB_other": reg == "CB other", "AN_DN": reg == "AN/DN", "lLN1_bc": ty == "lLN1_bc"}
out = {}
for k, m in R.items():
    np.savez(f"{ROOT}/results/aldef/weights/regions/{k}.npz", idx=np.where(m)[0])
    out[k] = int(m.sum())
json.dump(out, open(f"{ROOT}/results/aldef/weights/regions/regions.json", "w"), indent=1)
print(out)
