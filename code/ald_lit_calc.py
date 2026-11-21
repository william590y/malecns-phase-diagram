"""ald_lit_calc.py -- arithmetic behind results/aldef/lit_biophysics.md (2026-11-09).

No simulation. Converts published Drosophila antennal-lobe (AL) numbers into the units of our stochastic
Wilson-Cowan model (code/swc.py, code/ald_sim.py conventions) and computes steady-state short-term depression
from published depression fits. Every literature input below is quoted, with its source, in lit_biophysics.md.

Model conventions used (CLAUDE.md; ald_sim.py docstring):
  active -> quiescent at alpha = 1 (the time unit); quiescent -> active at beta*tanh(s)_+, beta = 1;
  synchronous steps dt = 0.2: p_off = 1 - exp(-dt), p_on = 1 - exp(-beta tanh(h) dt);
  isolated unit with drive h: rho = p_on / (p_on + p_off);
  ald_sim --std U TAU: dx/dt = (1 - x)/TAU - U a x ; mean-field x* = 1/(1 + U rho TAU).
ASSUMPTION (stated in the notes): one model time unit = tau_u milliseconds, and one active epoch = one spike.
Then a neuron's firing rate is nu = (activations per unit time)/tau_u = alpha_eff * rho / tau_u, alpha_eff = p_off/dt.

Output: results/aldef/lit_biophysics_calc.json
"""
import json, math

OUT = "/MALECNS_ROOT/results/aldef/lit_biophysics_calc.json"
DT, ALPHA, BETA = 0.2, 1.0, 1.0
P_OFF = 1 - math.exp(-ALPHA * DT)
ALPHA_EFF = P_OFF / DT                    # activations per unit time per unit of rho (= 0.906)
MEAN_ACTIVE = DT / P_OFF                  # mean active-epoch length in model units (= 1.104)
C_RAW = 1 / 742.7                         # --norm raw constant (CLAUDE.md)
TAU_U_MS = {"A_10ms_Benayoun": 10.0, "B_50ms_uEPSP": 50.0}

def rho_from_rate(nu_hz, tau_u_ms):
    return nu_hz * (tau_u_ms / 1000.0) / ALPHA_EFF

def rate_from_rho(rho, tau_u_ms):
    return ALPHA_EFF * rho / (tau_u_ms / 1000.0)

def h_for_rho(rho):
    """isolated-unit drive h giving stationary activity rho (inverse of rho = p_on/(p_on+p_off))."""
    p_on = rho * P_OFF / (1 - rho)
    if p_on >= 1:
        return float("inf")
    x = -math.log(1 - p_on) / (BETA * DT)  # = tanh(h)
    return float("inf") if x >= 1 else math.atanh(x)

def rho_for_h(h):
    p_on = 1 - math.exp(-BETA * math.tanh(h) * DT)
    return p_on / (p_on + P_OFF)

def U_from_f(f):
    """depletion rate U (per unit ACTIVE time, ald_sim convention) such that the mean resource retained over one
    active epoch equals the published per-spike retention f. Active length k*dt, k ~ Geometric(P_OFF) on 1,2,...
    E[exp(-U dt k)] = P_OFF e /(1 - (1-P_OFF) e), e = exp(-U dt). Recovery during the epoch is ignored (TAU >> 1)."""
    lo, hi = 0.0, 20.0
    for _ in range(200):
        U = 0.5 * (lo + hi)
        e = math.exp(-U * DT)
        m = P_OFF * e / (1 - (1 - P_OFF) * e)
        lo, hi = (U, hi) if m > f else (lo, U)
    return 0.5 * (lo + hi)

def A_periodic(f, tau_s, nu):
    """steady-state amplitude just before each spike, regular train at nu Hz: A = (1-E)/(1-fE), E = exp(-1/(nu tau))."""
    E = math.exp(-1.0 / (nu * tau_s))
    return (1 - E) / (1 - f * E)

def A_poisson(f, tau_s, nu):
    """mean steady-state amplitude before each spike, Poisson train: q = E[exp(-ISI/tau)] = nu tau/(1+nu tau);
    m = (1-q)/(1-f q)."""
    q = nu * tau_s / (1 + nu * tau_s)
    return (1 - q) / (1 - f * q)

# ---- published depression fits (amplitude multiplied by f after each spike, recovery time constant tau) ----
STD = {
    "ORN->PN DM4, Kazama&Wilson 2009 (alpha=0.72, tau=2.4 s)": (0.72, 2.4),
    "ORN->PN DM6/VM2 10 Hz, Nagel et al. 2015 (f=0.78, tau=893 ms)": (0.78, 0.893),
    "ORN->PN fast IMI-resistant component, Nagel 2015 (f=0.77, tau=1006 ms)": (0.77, 1.006),
    "ORN->PN slow curare-resistant component, Nagel 2015 (f=0.91, tau=629 ms)": (0.91, 0.629),
    "ORN->LN (GABAergic LNs), Nagel&Wilson 2016 (f=0.75, tau=1566 ms)": (0.75, 1.566),
}
RATES = {"DM4 ORN spont 3.44 Hz (K&W 2009)": 3.44, "VM7 ORN spont ~10 Hz (Olsen&Wilson 2008)": 10.0,
         "LN spont 4.6 Hz (Nagel 2015/2016)": 4.6, "20 Hz": 20.0, "50 Hz (odor-evoked ORN)": 50.0}

out = {"model": {"dt": DT, "p_off": P_OFF, "alpha_eff": ALPHA_EFF, "mean_active_epoch_units": MEAN_ACTIVE,
                 "c_raw": C_RAW, "rho_max_saturated_drive": 0.5}}

# 1. time-unit mapping: rates <-> rho, h
targets = {"ORN DM4 3.44 Hz": 3.44, "ORN VM7 10 Hz": 10.0, "palp ORN 3-13 Hz (low)": 3.0, "palp ORN 3-13 Hz (high)": 13.0,
           "PN 1 Hz": 1.0, "PN 5 Hz": 5.0, "LN 2.3 Hz": 2.3, "LN 4.6 Hz": 4.6}
mp = {}
for key, tu in TAU_U_MS.items():
    d = {"tau_u_ms": tu, "max_rate_Hz_at_rho_0.5": rate_from_rho(0.5, tu),
         "h_1e-5": {"rho": rho_for_h(1e-5), "rate_Hz": rate_from_rho(rho_for_h(1e-5), tu)},
         "h_1e-3": {"rho": rho_for_h(1e-3), "rate_Hz": rate_from_rho(rho_for_h(1e-3), tu)}}
    for name, nu in targets.items():
        r = rho_from_rate(nu, tu)
        d[name] = {"rho": r, "h_isolated_unit": h_for_rho(r)}
    d["tau_rec_in_model_units"] = {k: v[1] * 1000.0 / tu for k, v in STD.items()}
    mp[key] = d
out["mapping"] = mp

# 2. published per-spike retention -> ald_sim U
out["U_from_f"] = {k: {"f": f, "tau_s": tau, "U_geometric_epoch": U_from_f(f), "U_naive_-lnf": -math.log(f)}
                   for k, (f, tau) in STD.items()}

# 3. steady-state depression at physiological rates (literature parameters only)
out["steady_state_A"] = {k: {rn: {"periodic": A_periodic(f, tau, nu), "poisson": A_poisson(f, tau, nu)}
                             for rn, nu in RATES.items()} for k, (f, tau) in STD.items()}

# 4. mean-field resource x* = 1/(1 + U rho TAU) in model units, for the two mappings, at several rho
mf = {}
for key, tu in TAU_U_MS.items():
    rows = {}
    for k, (f, tau) in STD.items():
        U, TAU = U_from_f(f), tau * 1000.0 / tu
        rows[k] = {f"rho={r}": 1.0 / (1.0 + U * r * TAU) for r in (0.01, 0.04, 0.1, 0.2, 0.3, 0.5)}
    mf[key] = rows
out["meanfield_x_star"] = mf

# 5. order-of-magnitude ORN drive onto one PN in model units (inputs: Tobin 2017 ~23 synapses per ORN->PN
#    connection; Kazama&Wilson 2009 17.4 DM4 ORNs per antenna, bilateral projection; c_raw)
n_orn, w_conn = 2 * 17.4, 23.0
pn = {}
for key, tu in TAU_U_MS.items():
    r_orn = rho_from_rate(3.44, tu)
    rows = {}
    for g in (0.163, 0.30):
        s = g * C_RAW * n_orn * w_conn * r_orn
        rho_pn = rho_for_h(s)              # treats s as a constant drive (no PN recurrence, no inhibition, no depression)
        rows[f"g={g}"] = {"s_from_ORNs": s, "PN_rho": rho_pn, "PN_rate_Hz": rate_from_rho(rho_pn, tu)}
    rows["rho_ORN"] = r_orn
    pn[key] = rows
out["PN_drive_estimate"] = {"inputs": {"n_ORN_presyn": n_orn, "synapses_per_ORN_PN_connection": w_conn,
                                       "ORN_rate_Hz": 3.44}, **pn}

with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
