"""E3 args generator: one ald_sim.py argument line per Slurm array task (see results/aldef/depression/plan.md).
Usage: python ald_dep_args.py stage1  -> results/aldef/depression/args_stage1.txt + index_stage1.json
       python ald_dep_args.py stage2 <json with {"arm": [g, ...]}> -> args_stage2.txt + index_stage2.json
"""
import json, sys
import numpy as np

ROOT = "/MALECNS_ROOT"
OUT = f"{ROOT}/results/aldef/depression"
COMMON = ("--exp decay --norm raw --T 3000 --B 64 --engine push --push_max 0.012 --reg_last --rec_lin 100 "
          "--win 1500:3000 --region lLN1_bc=type:lLN1_bc --region PN=class:ALPN --region LN=class:ALLN "
          "--region ORN=class:olfactory")
G1 = np.r_[np.geomspace(0.15, 0.6, 16), np.geomspace(0.6, 8.0, 12)[1:]]
NTASK = 3

LIT = {"KW09": (0.28, 0.34, 240), "N15": (0.22, 0.25, 89), "NW16": (0.25, 0.29, 157), "slow": (0.09, 0.09, 63)}
# (U_spike = 1 - f, U_rate = U_geometric_epoch, TAU) from results/aldef/lit_biophysics_calc.json (tau_u = 10 ms)
U0, TAU0 = 0.25, 157            # main point (spike mode)
GROUPS = {"exc": "sign:1", "ORN": "class:olfactory", "PN": "class:ALPN", "LN": "class:ALLN", "lLN1bc": "type:lLN1_bc",
          "AL": "al;module"}


def arms():
    A = {}
    for r in (1, 0):
        A[f"base_r{r}"] = dict(r=r, extra="", kind="base")
    for k, (us, ur, tau) in LIT.items():
        A[f"sp_{k}_r1"] = dict(r=1, extra=f"--std {us} {tau} --std_mode spike", kind="std_spike", U=us, TAU=tau, lit=k)
        A[f"rt_{k}_r1"] = dict(r=1, extra=f"--std {ur} {tau}", kind="std_rate", U=ur, TAU=tau, lit=k)
    for tau in (3, 10, 30):
        A[f"sp_U25_tau{tau}_r1"] = dict(r=1, extra=f"--std 0.25 {tau} --std_mode spike", kind="std_spike", U=0.25, TAU=tau)
    for gname, sel in GROUPS.items():
        A[f"spg_{gname}_r1"] = dict(r=1, extra=f'--std_group "{sel}={U0},{TAU0}" --std_mode spike', kind="std_group",
                                   U=U0, TAU=TAU0, group=sel)
    A["spg_notAL_r1"] = dict(r=1, extra=f'--std {U0} {TAU0} --std_group "al;module=0,1" --std_mode spike',
                             kind="std_group", U=U0, TAU=TAU0, group="~(al;module)")
    for dtau in (0.1, 0.3, 1, 3):
        A[f"ad_DT{dtau:g}_ta50_r1"] = dict(r=1, extra=f"--adapt {dtau / 50:g} 50", kind="adapt", D=dtau / 50, TAUA=50)
    A["ad_DT1_ta10_r1"] = dict(r=1, extra="--adapt 0.1 10", kind="adapt", D=0.1, TAUA=10)
    A["adg_LN_DT1_ta50_r1"] = dict(r=1, extra='--adapt_group "class:ALLN=0.02,50"', kind="adapt_group", D=0.02, TAUA=50,
                                   group="class:ALLN")
    for n in (5, 10, 25, 50):
        A[f"ref{n}_r1"] = dict(r=1, extra=f"--refractory {n}", kind="refractory", nref=n)
    for k, ex in (("base", ""), ("sp_NW16", f"--std {LIT['NW16'][0]} {LIT['NW16'][2]} --std_mode spike"),
                  ("sp_KW09", f"--std {LIT['KW09'][0]} {LIT['KW09'][2]} --std_mode spike")):
        A[f"iso_{k}_noAL_r1"] = dict(r=1, extra=f'{ex} --clamp_off "al;module"', kind="isolation", variant=k, clamp="al;module")
        A[f"iso_{k}_ALonly_r1"] = dict(r=1, extra=f'{ex} --clamp_off "cns_rest"', kind="isolation", variant=k, clamp="cns_rest")
    A["sp_NW16_r0"] = dict(r=0, extra=f"--std {LIT['NW16'][0]} {LIT['NW16'][2]} --std_mode spike", kind="std_spike",
                           U=LIT["NW16"][0], TAU=LIT["NW16"][2], lit="NW16")
    A["rt_NW16_r0"] = dict(r=0, extra=f"--std {LIT['NW16'][1]} {LIT['NW16'][2]}", kind="std_rate", U=LIT["NW16"][1],
                           TAU=LIT["NW16"][2], lit="NW16")
    A["ad_DT1_ta50_r0"] = dict(r=0, extra="--adapt 0.02 50", kind="adapt", D=0.02, TAUA=50)
    A["ref10_r0"] = dict(r=0, extra="--refractory 10", kind="refractory", nref=10)
    return A


def driven():
    D = {}
    gs = " ".join(f"{x:.4g}" for x in np.geomspace(0.08, 0.42, 16))
    var = {"base": "", "sp_NW16": f"--std {LIT['NW16'][0]} {LIT['NW16'][2]} --std_mode spike",
           "sp_KW09": f"--std {LIT['KW09'][0]} {LIT['KW09'][2]} --std_mode spike",
           "rt_NW16": f"--std {LIT['NW16'][1]} {LIT['NW16'][2]}"}
    for h in ("1e-5", "1e-4"):
        for k, ex in var.items():
            D[f"drv_{k}_h{h}_r1"] = dict(r=1, h=float(h), extra=ex, g=gs, kind="driven")
    return D


def main():
    stage = sys.argv[1]
    lines, index = [], []
    if stage == "stage1":
        sid = 3000
        for name, a in arms().items():
            for k in range(NTASK):
                gs = " ".join(f"{x:.4g}" for x in G1[k::NTASK])
                out = f"{OUT}/sim/{name}_t{k}.npz"
                lines.append(f"{COMMON} --r {a['r']} --g {gs} --seed {sid} {a['extra']} --out {out}")
                index.append({"arm": name, "task": k, "out": out, "seed": sid, **a}); sid += 1
        for name, a in driven().items():
            out = f"{OUT}/sim/{name}.npz"
            lines.append(f"--exp quiet --norm raw --T 3000 --B 64 --engine push --push_max 0.012 --h {a['h']:g} "
                         f"--win 1000:3000 --r {a['r']} --g {a['g']} --seed {sid} {a['extra']} --out {out}")
            index.append({"arm": name, "task": 0, "out": out, "seed": sid, **{k: v for k, v in a.items() if k != 'g'}}); sid += 1
    elif stage == "rm":                 # E1 protocol (ORN drive -> all ORNs clamped at 1500) with depression
        grp = f"{ROOT}/results/aldef/drive/groups"
        reg = " ".join(f"--region {n}={grp}/{n}.npz" for n in ("ORN", "uPN_ant", "ALsys", "lLN1_bc"))
        gl = [0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3, 0.36, 0.44, 0.55, 0.7, 0.9, 1.15, 1.5, 2.0, 2.6]
        var = {"base": "", "sp_NW16": f"--std {LIT['NW16'][0]} {LIT['NW16'][2]} --std_mode spike",
               "sp_KW09": f"--std {LIT['KW09'][0]} {LIT['KW09'][2]} --std_mode spike",
               "spg_exc": f'--std_group "sign:1={U0},{TAU0}" --std_mode spike',
               "rt_NW16": f"--std {LIT['NW16'][1]} {LIT['NW16'][2]}"}
        sid = 4000
        for k, ex in var.items():
            for t in range(2):
                gs = " ".join(f"{x:g}" for x in gl[t::2])
                out = f"{OUT}/rm/rm_{k}_r1_t{t}.npz"
                lines.append(f"--exp quiet --norm raw --T 4500 --B 64 --switch_at 1500 --win 500:1500 --win 1600:1800 "
                             f"--win 2500:4500 --reg_last --engine push --push_max 0.012 {reg} --r 1 --g {gs} "
                             f"--drive olfactory=0.037926 --sw_clamp_off olfactory --seed {sid} {ex} --out {out}")
                index.append({"arm": f"rm_{k}_r1", "task": t, "out": out, "seed": sid, "kind": "removal", "r": 1,
                              "variant": k, "orn_rho_iso": 0.04, "h_orn": 0.037926}); sid += 1
    elif stage == "iso":
        sid = 6000
        for name, a in arms().items():
            if a["kind"] != "isolation":
                continue
            for k in range(NTASK):
                gs = " ".join(f"{x:.4g}" for x in np.r_[G1, [10.5, 13.8]][k::NTASK])
                out = f"{OUT}/sim/{name}_t{k}.npz"
                lines.append(f"{COMMON} --r {a['r']} --g {gs} --seed {sid} {a['extra']} --out {out}")
                index.append({"arm": name, "task": k, "out": out, "seed": sid, **a}); sid += 1
    elif stage == "stage2":
        spec = json.load(open(sys.argv[2]))
        A = arms(); sid = 5000
        for name, gl in spec.items():
            a = A[name]
            for k in range(0, len(gl), 4):
                gs = " ".join(f"{x:.4g}" for x in gl[k:k + 4])
                out = f"{OUT}/sim2/{name}_s{k // 4}.npz"
                lines.append(f"{COMMON} --r {a['r']} --g {gs} --seed {sid} {a['extra']} --out {out}")
                index.append({"arm": name, "task": k // 4, "out": out, "seed": sid, **a}); sid += 1
    open(f"{OUT}/args_{stage}.txt", "w").write("\n".join(lines) + "\n")
    json.dump(index, open(f"{OUT}/index_{stage}.json", "w"), indent=1)
    print(len(lines), "tasks")


if __name__ == "__main__":
    main()
