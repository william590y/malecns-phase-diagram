"""m2_common.py -- shared constants, model/variant flags, protocols, grids and regions for Model 2 (2026-11-09).
Spec: results/m2/MODEL2.md. Every lane builds its ald_sim.py argument lines with these helpers so that the model, the
regions and the windows are identical across lanes.

  h_of_rho(rho)               drive h giving an isolated unit the stationary active fraction rho (exact 2-state chain)
  MODEL[name]                 synapse-dynamics / sign flags of a model variant (M1, M2, M2s, sensitivity variants)
  protocol(name, ...)         protocol flags (decay_h0, driven, aval_pulse, ramp_driven, ramp_h0_down)
  REGIONS / region_flags()    the standard M2 regions (global-id index files: valid on --graph subgraphs too)
  G_COARSE, R_ALL, R_CORE, ORN_LEVELS, refine()   grids
  args_line(model, proto, g, r, out, seed, ...)    one ald_sim.py argument line
  write_graph(kind, N, seed, path)                 pd3 FSS graph (pd3_fss_graph.build, read-only use) -> --graph file
"""
import json, os
import numpy as np

ROOT = "/MALECNS_ROOT"
SIM = f"{ROOT}/code/ald_sim.py"
GRP = f"{ROOT}/results/aldef/drive/groups"          # index sets written by code/ald_drive_args.py (global cache ids)
DT, BETA = 0.2, 1.0
P_OFF = 1 - np.exp(-DT)
ALPHA_EFF = P_OFF / DT                              # 0.906
G_LIN = 0.16315                                     # linear onset of the undepressed network (M1; unchanged in spike mode)


def h_of_rho(rho):
    """h with isolated-unit stationary activity rho = p_on / (p_on + p_off), p_on = 1 - exp(-beta tanh(h) dt)."""
    p_on = rho * P_OFF / (1 - rho)
    return float(np.arctanh(-np.log(1 - p_on) / (BETA * DT)))


def hz_of_rho(rho, tau_u_ms=10.0):
    """Rate in Hz of a unit with active fraction rho if one model time unit = tau_u ms (one epoch = one spike)."""
    return ALPHA_EFF * rho / (tau_u_ms * 1e-3)


ORN_LEVELS = (0.01, 0.02, 0.04, 0.07, 0.10)
ORN_MAIN = 0.04
H_ORN = {r: round(h_of_rho(r), 6) for r in ORN_LEVELS}          # 0.04 -> 0.037926 (as E1)

# depression parameters (spike mode: U = 1 - f per epoch, TAU = tau_rec / tau_u with tau_u = 10 ms)
# sources: results/aldef/lit_biophysics_calc.json (Nagel & Wilson 2016 ORN->LN f 0.75, 1566 ms; Kazama & Wilson 2009
# ORN->PN alpha 0.72, 2.4 s; Nagel et al. 2015 ORN->PN f 0.78, 893 ms)
DEP = {"NW16": (0.25, 157), "KW09": (0.28, 240), "N15": (0.22, 89), "NW16_tau50ms": (0.25, 30)}

M2S_EDGES = ('--edge_sign "lLN1_bc->ALPN=0" --edge_sign "lLN1_bc->ALLN=1"')


def _dep(U, TAU, group="sign:1"):
    if group == "all":
        return f"--std {U} {TAU} --std_mode spike"
    return f'--std_group "{group}={U},{TAU}" --std_mode spike'


MODEL = {
    "M1": "",                                                        # old model: no depression
    "M2": _dep(*DEP["NW16"]),                                        # default: exc outputs, U 0.25, TAU 157
    "M2s": _dep(*DEP["NW16"]) + " " + M2S_EDGES,                     # + per-target lLN1_bc signs
    "M2_allsyn": _dep(*DEP["NW16"], group="all"),                    # depression on every presynaptic neuron
    "M2_KW09": _dep(*DEP["KW09"]),
    "M2_N15": _dep(*DEP["N15"]),
    "M2_tau30": _dep(*DEP["NW16_tau50ms"]),                          # tau_u = 50 ms mapping
    "M1s": M2S_EDGES,                                                # signs fix alone (no depression), for attribution
}

# standard regions (besides ald_sim's default AL_module / AL_rest / CNS_rest)
REGIONS = {
    "ORN": "olfactory",                                  # 2,639 ORNs (drive target)
    "nonORN": "~olfactory",                              # everything the drive does not touch directly
    "uPN_ant": f"{GRP}/uPN_ant.npz",                     # 220 antennal uniglomerular ACh PNs (gain numerator, E1)
    "ALsys": f"{GRP}/ALsys.npz",                         # AL without ORNs, union the AL module (deafferentation test)
    "lLN1_bc": "type:lLN1_bc",
}


def region_flags(names=None):
    names = names or list(REGIONS)
    return " ".join(f'--region "{n}={REGIONS[n]}"' for n in names)


# grids
G_COARSE = np.round(np.geomspace(0.05, 30.0, 40), 5)                   # x1.178 per step
R_ALL = (0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5,
         5, 5.5, 6)                                                    # M1's 23 values
R_CORE = (0, 0.25, 0.5, 1, 2, 4, 6)                                    # subset for expensive protocols


def refine(g_lo, g_hi, n=8):
    """n log-spaced interior points of a bracketing cell (stage-2 refinement)."""
    return np.round(np.geomspace(g_lo, g_hi, n + 2)[1:-1], 5)


T_DECAY = 3000.0
T1_DRIVE, T_POST = 1500.0, 3000.0


def protocol(name, orn=ORN_MAIN, counts=True, counts_reg=False, gap=0.0, dwell=200.0, pulse_sel="olfactory"):
    """ald_sim.py flags of a protocol (MODEL2.md section 4). Engine: push (event-driven; fast while activity is sparse)."""
    eng = "--engine push --push_max 0.012"
    reg = region_flags()
    if name == "decay_h0":            # (a) h = 0, fully active start, T = 3000; QS window 1500-3000
        f = (f"--exp decay --norm raw --T {T_DECAY:g} --reg_last --win 1500:3000 {eng} {reg}")
        if counts:
            f += " --rec_counts --counts_win 1500:3000" + (" --counts_reg" if counts_reg else "")
        return f
    if name == "driven":              # (b) ORN drive from a quiet start, then every ORN removed at T1
        T = T1_DRIVE + T_POST
        f = (f"--exp quiet --norm raw --T {T:g} --drive olfactory={H_ORN.get(orn, round(h_of_rho(orn), 6))} "
             f"--switch_at {T1_DRIVE:g} --sw_clamp_off olfactory --rec_lin 10 --reg_last "
             f"--win 500:1500 --win 1600:1800 --win 2500:4500 --occ_win 500:1500 {eng} {reg}")
        if counts:
            f += " --rec_counts --counts_win 500:1500" + (" --counts_reg" if counts_reg else "")
        return f
    if name == "aval_pulse":          # separation of timescales: h = 0, one seed when silent
        return (f'--exp quiet --norm raw --T {T_DECAY:g} --pulse "{pulse_sel}" --pulse_gap {gap:g} --rec_counts '
                f"{eng} --no_occ")
    if name == "ramp_driven":         # hysteresis: quiet start, ORN drive, g up then down (pass ascending g)
        return (f"--exp quiet --norm raw --drive olfactory={H_ORN.get(orn, round(h_of_rho(orn), 6))} "
                f"--ramp {dwell:g} --ramp_loop --ramp_avg 0.5 {eng} {reg}")
    if name == "ramp_h0_down":        # h = 0: fully active start, g down then up (pass descending g); seeds via pulse
        return (f'--exp decay --norm raw --ramp {dwell:g} --ramp_loop --ramp_avg 0.5 --pulse "{pulse_sel}" '
                f"--pulse_gap {gap:g} {eng} {reg}")
    raise ValueError(name)


def args_line(model, proto, g, r, out, seed, B=64, extra="", **pk):
    gs = " ".join(f"{x:g}" for x in np.atleast_1d(g))
    return (f"{protocol(proto, **pk)} {MODEL[model]} --r {r:g} --g {gs} --B {B} --seed {seed} {extra} "
            f"--out {out}").replace("  ", " ")


def write_graph(kind, N, seed, path):
    """pd3 FSS graph -> ald_sim --graph file (pre, post, w, sign per local neuron, sub = global ids, c)."""
    import sys
    sys.path.insert(0, f"{ROOT}/code")
    import pd3_fss_graph as F
    g = F.build(kind, N, seed)
    sign = np.load("/MALECNS_CACHE/sign.npy")
    sub = g["sub"] if g["sub"] is not None else np.arange(len(sign))
    np.savez(path, pre=g["pre"].astype(np.int64), post=g["post"].astype(np.int64), w=g["w"].astype(np.float32),
             sign=sign[sub], sub=sub.astype(np.int64), c=float(g["c"]), kind=kind, seed=seed)
    return {"N": int(g["N"]), "n_edges": int(g["n_edges"]), "c": float(g["c"])}


if __name__ == "__main__":
    print(json.dumps({"H_ORN": H_ORN, "Hz_at_10ms": {r: hz_of_rho(r) for r in ORN_LEVELS}, "G_COARSE": G_COARSE.tolist(),
                      "MODEL": MODEL}, indent=1))
