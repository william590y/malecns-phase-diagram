"""Check the GPU smoke-test outputs of ald_sim.py (results/aldef/smoke/s*.npz) -> results/aldef/smoke/smoke_check.json.
Checks: array shapes, sidecar JSON present, drive-only rate of ORNs at g = 0 vs the exact isolated-unit formula,
zero activity outside the driven population at g = 0, clamped population silent, ensemble bookkeeping."""
import glob, json, os
import numpy as np

D = "/MALECNS_ROOT/results/aldef/smoke"
C = "/MALECNS_CACHE"
out = {}
for f in sorted(glob.glob(f"{D}/s*.npz")):
    z = np.load(f, allow_pickle=True); side = json.load(open(f[:-4] + ".json"))
    o = {"keys": sorted(z.files), "rho": list(z["rho"].shape), "reg_rho": list(z["reg_rho"].shape),
         "reg_names": z["reg_names"].tolist(), "rho_T_mean": side["rho_T_mean"], "surv_T": side["surv_T"],
         "reg_rho_T_mean": side.get("reg_rho_T_mean"), "ms_per_step": side["ms_per_step"], "gpu": side.get("gpu")}
    for k in ("res_x", "res_x_act", "res_theta", "signs_ens", "ens_draw"):
        if k in z.files:
            o[k] = list(z[k].shape)
    for k in ("drive", "clamp_off", "sign_override", "std", "adapt", "ensemble", "mean_in_strength_model_units"):
        if k in side:
            o[k] = side[k]
    out[os.path.basename(f)] = o

# drive-only rate at g = 0 (s3): ORNs h = 0.02, everything else 0
z = np.load(f"{D}/s3_orn_drive.npz", allow_pickle=True)
cl = np.load(f"{C}/classes.npy", allow_pickle=True).astype(str)
orn = cl == "olfactory"
dt, h = 0.2, 0.02
pon = 1 - np.exp(-np.tanh(h) * dt); poff = 1 - np.exp(-dt)
rho_ex = pon / (pon + poff)
occ0 = z["occ"][0].astype(np.float64)
nsteps = int(round(300 / dt)) // 2
se = np.sqrt(rho_ex * (1 - rho_ex) * ((2 - pon - poff) / (pon + poff)) / (orn.sum() * 16 * nsteps))
out["check_drive_only_g0"] = {"ORN_occ_mean": occ0[orn].mean(), "exact": rho_ex, "se": se,
                              "z": (occ0[orn].mean() - rho_ex) / se, "max_occ_nonORN": float(occ0[~orn].max()),
                              "pass": bool(abs(occ0[orn].mean() - rho_ex) < 5 * se + 1e-3 * rho_ex and occ0[~orn].max() == 0)}
# clamp (s4): antennal ORNs never active
z4 = np.load(f"{D}/s4_clamp_std_adapt_ref.npz", allow_pickle=True)
ty = np.load(f"{C}/types.npy", allow_pickle=True).astype(str)
import re
palp = np.array([re.fullmatch(r"ORN_(VA4|VC1|VC2|VA7l|VM7d|VM7v)", t) is not None for t in ty])
ant = orn & ~palp
out["check_clamp_off"] = {"n_clamped": int(ant.sum()), "max_occ_clamped": float(z4["occ"][:, ant].astype(float).max()),
                          "pass": bool(z4["occ"][:, ant].astype(float).max() == 0)}
z5 = np.load(f"{D}/s5_sign_ens_type.npz", allow_pickle=True)
out["check_ensemble"] = {"rho_shape": list(z5["rho"].shape), "ens_draw": np.unique(z5["ens_draw"]).tolist(),
                         "pass": bool(z5["rho"].shape[2] == 32 and z5["signs_ens"].shape == (2, len(cl)))}
json.dump(out, open(f"{D}/smoke_check.json", "w"), indent=1, default=float)
for k in ("check_drive_only_g0", "check_clamp_off", "check_ensemble"):
    print(k, out[k])
for k, v in out.items():
    if k.startswith("s"):
        print(k, v["rho"], v["reg_rho"], "rhoT", np.round(v["rho_T_mean"], 6).tolist(), "survT", v["surv_T"],
              "ms/step", np.round(v["ms_per_step"], 2).tolist(), v.get("ensemble", {}).get("draws", ""))
