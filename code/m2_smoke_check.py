"""Checks + cost table for the Model-2 smoke array (results/m2/sim/plan.md section 3; 2026-11-09).
Usage: python m2_smoke_check.py -> results/m2/sim/smoke/costs.json, costs.md
Smoke numbers are ILLUSTRATIVE (one r, few g, one seed); they test that each protocol and the m2_obs pipeline work.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_obs as O
import m2_common as M

D = f"{M.ROOT}/results/m2/sim/smoke"
idx = json.load(open(f"{D}/index.json"))
N = 167216
rows, checks, illus = [], {}, {}


def load(tag):
    p = f"{D}/{tag}.npz"
    if not os.path.exists(p):
        return None, None
    return np.load(p, allow_pickle=True), json.load(open(f"{D}/{tag}.json"))


def ri(z, name):
    return list(z["reg_names"]).index(name)


for tag in idx["tasks"]:
    z, s = load(tag)
    if z is None:
        rows.append({"task": tag, "status": "MISSING"}); checks[tag] = {"exists": False}; continue
    opt = s["options"]
    row = {"task": tag, "gpu": s.get("gpu"), "engine": opt["engine"], "B": opt["B"], "g": opt["g"],
           "T": s.get("ramp", {}).get("T", opt["T"]), "steps": s["steps"], "wall_s": round(s["wall_s"], 1),
           "ms_per_step": [round(x, 2) for x in s["ms_per_step"][0]],
           "npz_MB": round(os.path.getsize(f"{D}/{tag}.npz") / 1e6, 2),
           "s_per_g_full": [round(x * s["steps"] / 1e3, 1) for x in s["ms_per_step"][0]]}
    rows.append(row)
    c = {"exists": True}
    t = z["t"]
    if tag.startswith(("decay", "null", "fss")):
        T = opt["T"]
        taus = []
        for gi in range(len(opt["g"])):
            ext = O.extinction_times(t, z["surv"][gi])
            tau, d = O.tau_mle(ext, T)
            taus.append({"g": opt["g"][gi], "tau": tau if np.isfinite(tau) else "inf", "deaths": d,
                         "alive_T": float(z["surv"][gi][-1].mean())})
        c["tau"] = taus
        c["records_ok"] = bool(z["surv"].shape[1] == len(t))
        if "occ" in z.files:
            c["PR_over_N"] = [round(O.participation_ratio(z["occ"][gi])[1], 4) for gi in range(len(opt["g"]))]
        if "cnt_act" in z.files:
            osc = []
            for gi in range(len(opt["g"])):
                alive = z["surv"][gi][-1] > 0
                if alive.sum() >= 4:
                    x = z["cnt_act"][gi][:, alive].astype(np.float64) / s["N"]
                    m = O.osc_metrics(x, 0.2)
                    osc.append({"g": opt["g"][gi], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
                else:
                    osc.append({"g": opt["g"][gi], "n/a": "fewer than 4 replicas alive at T"})
            c["osc"] = osc
        if tag.startswith("fss"):
            c["N"] = s["N"]; c["N_ok"] = s["N"] == 20000
        if tag.startswith("null"):
            c["null_edges_md5"] = s["null_edges_md5"]
        if "edge_sign" in s:
            c["edge_sign_n_edges"] = [e["n_edges"] for e in s["edge_sign"]]
    if tag.startswith("driven"):
        iORN, iU, iA, iN, iM = (ri(z, n) for n in ("ORN", "uPN_ant", "ALsys", "nonORN", "AL_module"))
        wm, w2 = z["win_mean"], z["win_m2"]
        per = []
        for gi, g in enumerate(opt["g"]):
            rORN = wm[gi, 0, iORN].mean(); rU = wm[gi, 0, iU].mean()
            G = rU / rORN
            post_orn = float(wm[gi, 1:, iORN].max())
            S1 = wm[gi, 1, iU].mean() <= 0.1 * rU
            last = z["reg_last"][gi]
            Tend = opt["T"]
            S2 = (last[iA] >= Tend - 0.1).mean() <= 0.05
            ext = np.where(last[iN] >= Tend - 0.1, np.inf, last[iN] + 0.2 - 1500)
            taup, d = O.tau_mle(ext, 3000.0)
            cls = ("self-sustained" if taup >= 3000 else ("silent" if (taup < 200 and S1 and S2) else "transient"))
            varM = (w2[gi, 0, iM] - wm[gi, 0, iM] ** 2)
            chiAL = 526 * varM.mean()
            prn = O.participation_ratio(z["occ"][gi], np.load(f"{M.ROOT}/results/aldef/drive/groups/ORN.npz")["idx"]
                                        .__array__() if False else None)
            o = z["occ"][gi].astype(np.float64)
            orn_mask = np.zeros(N, bool); orn_mask[np.load(f"{M.GRP}/ORN.npz")["idx"]] = True
            pr_non = O.participation_ratio(o, ~orn_mask)
            if cls == "self-sustained":
                ph = "SS-" + ("D" if pr_non[1] >= 0.25 else "L")
            elif cls == "transient":
                ph = "DT"
            else:
                ph = "DA" if G >= 0.29 else "DQ"
            allowed = bool(0.29 <= G <= 1.45 and S1 and S2)
            item = {"g": g, "rho_ORN": round(float(rORN), 5), "rho_nonORN": round(float(wm[gi, 0, iN].mean()), 6),
                    "G_uPN_over_ORN": round(float(G), 3), "ORN_after_removal_max": post_orn, "S1": bool(S1), "S2": bool(S2),
                    "tau_post": (round(taup, 1) if np.isfinite(taup) else "inf"), "deaths_post": d, "removal_class": cls,
                    "chi_AL": round(float(chiAL), 4), "PR_nonORN_over_n": round(pr_non[1], 4), "phase": ph,
                    "data_allowed": allowed}
            if "cnt_act" in z.files:
                x = z["cnt_act"][gi].astype(np.float64) / N
                m = O.osc_metrics(x, 0.2)
                item.update(tau_int=round(m["tau_int"], 2), osc=m["osc"], c_min=round(m["c_min"], 3),
                            c_peak=(round(m["c_peak"], 3) if np.isfinite(m["c_peak"]) else None))
                if "cnt_reg_act" in z.files:
                    a = z["cnt_reg_act"][gi][:, iN, :]; on = z["cnt_reg_on"][gi][:, iN, :]
                    item["MR_m_nonORN_onsets"] = round(O.mr_branching(on, 20)["m"], 4)
                    av = O.avalanches(a, None, thr=int(np.median(a)))
                    item["n_thr_avalanches"] = int(len(av["size"]))
            per.append(item)
        c["per_g"] = per
        c["ORN_within_10pct_of_0.04"] = [abs(p["rho_ORN"] / 0.04 - 1) <= 0.10 for p in per]
        c["ORN_zero_after_removal"] = all(p["ORN_after_removal_max"] == 0 for p in per)
        if "edge_sign" in s:
            c["edge_sign_n_edges"] = [e["n_edges"] for e in s["edge_sign"]]
    if tag.startswith("aval"):
        per = []
        for gi, g in enumerate(opt["g"]):
            a, on = z["cnt_act"][gi], z["cnt_on"][gi]
            av = O.avalanches(a, on)
            sil = (a == 0)
            starts_after_silence = True
            if len(av["start"]):
                starts_after_silence = bool(np.all(sil[av["start"] - 1, av["rep"]]))
            fit = O.powerlaw_fit(av["size"], xmin_max=100) if len(av["size"]) >= 200 else {}
            per.append({"g": g, "n_avalanches": int(len(av["size"])), "min_size": int(av["size"].min()) if len(av["size"]) else None,
                        "mean_size": float(av["size"].mean()) if len(av["size"]) else None,
                        "max_size": int(av["size"].max()) if len(av["size"]) else None,
                        "max_dur_steps": int(av["dur"].max()) if len(av["dur"]) else None,
                        "silent_step_before_each": starts_after_silence,
                        "frac_time_active": float((a > 0).mean()), "powerlaw_fit_illustrative": fit,
                        "n_pulses_seed_check": int(s["pulse"]["n"])})
        c["per_g"] = per
    if tag.startswith("ramp"):
        g = opt["g"]; want = list(g) + list(g[-2::-1])
        c["seg_g_ok"] = bool(np.allclose(z["seg_g"], want))
        iN = ri(z, "nonORN")
        c["seg_nonORN"] = [round(float(v), 6) for v in z["seg_reg"][:, iN, :].mean(1)]
        c["seg_g"] = [float(v) for v in z["seg_g"]]
        c["seg_x_AL_module"] = [round(float(v), 4) for v in z["seg_x"][:, ri(z, "AL_module")]] if "seg_x" in z.files else None
    checks[tag] = c

# cross-task checks
hashes = {k: v.get("null_edges_md5") for k, v in checks.items() if "null_edges_md5" in v}
vrun = f"{M.ROOT}/results/m2/sim/validation/runs/V3_null_ns0.json"
if os.path.exists(vrun):
    hashes["validation_V3_null_ns0(targets, null_seed 0)"] = json.load(open(vrun))["null_edges_md5"]
summary = {
    "all_tasks_present": all(v.get("exists") for v in checks.values()),
    "M2s_edges": {k: v.get("edge_sign_n_edges") for k, v in checks.items() if "edge_sign_n_edges" in v},
    "null_hashes": hashes,
}
json.dump({"rows": rows, "checks": checks, "summary": summary}, open(f"{D}/costs.json", "w"), indent=1, default=str)
lines = ["| task | GPU | engine | B | g | T | ms/step per g | wall s | s per g (full run) | npz MB |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if r.get("status") == "MISSING":
        lines.append(f"| {r['task']} | MISSING | | | | | | | | |"); continue
    lines.append(f"| {r['task']} | {r['gpu']} | {r['engine']} | {r['B']} | {r['g']} | {r['T']:g} | {r['ms_per_step']} | "
                 f"{r['wall_s']} | {r['s_per_g_full']} | {r['npz_MB']} |")
open(f"{D}/costs.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(json.dumps(summary, indent=1))
