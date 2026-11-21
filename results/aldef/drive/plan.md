# E1 — ORN drive and antenna removal: plan and pre-registered criteria

Written 2026-11-09 ~06:25, before any production run of this experiment (args: `code/ald_drive_args.py` → `args_main.txt`).

## Question
Does the model have a coupling range g in which (i) PNs fed by ORNs at realistic spontaneous rates fire at realistic
spontaneous rates, and (ii) PNs go silent quickly once the ORN input is removed? How does that range compare with the
h = 0 onset (r = 0: S 0.234, onset 0.266; r = 1: S 0.283, onset 0.344; `results/analysis2/phase_complete_table.md`),
g_lin = 0.1632, and the driven-plane fluctuation peak (g ≈ 0.30 at h = 1e-5, r = 1; `results/analysis2/driven.md`)?

## Runs
- Quiet start. Phase 1 (0 < t ≤ 1500): all 2,639 ORNs driven at h_ORN, every other neuron h = 0. Phase 2
  (1500 < t ≤ 4500): `all` = every ORN clamped to 0; `ant` = only the 2,454 antennal ORNs clamped, palp ORNs stay driven.
- r ∈ {0, 1}; ORN levels ρ_iso ∈ {0.01, 0.02, 0.04, 0.07, 0.10} (isolated-unit active fraction; `levels.json`);
  24 g values 0–1; B = 64 replicas; `--engine push` (E2's event-driven product, statistically equivalent to spmm).
- Unit mapping: rates in the model are ν = 0.906 ρ / τ_u. Read each ORN level either as an ORN rate at τ_u = 10 ms
  (0.9, 1.8, 3.6, 6.3, 9.1 Hz) or as τ_u with ORNs at 3.44 Hz (2.6, 5.3, 10.5, 18.4, 26.3 ms). Rate RATIOS are unit-free.

## Measures (per g, r, ORN level)
- Driven state: window 500 < t ≤ 1500, per-replica means of every region; ρ_ORN (measured, all ORNs), ρ_uPN (antennal
  uPN, n = 220), AL module, CNS rest. AL gain G = ρ_uPN_ant / ρ_ORN.
- Removal: replica-mean uPN_ant activity in windows after removal (0–5, 5–20, 20–50, 50–100, 100–300, 300–1000,
  1000–3000 units), relative to its driven value; times to fall below 1/e and 10 % of the driven value; per-replica last
  activity time of the AL system (AL without ORNs ∪ AL module) and of uPN_ant; fraction of replicas still active at the end.
- Fluctuations of the AL module in the driven window: χ_time = n·⟨var_t⟩/⟨ρ⟩ and χ_rep = n·var_rep(time mean)/⟨ρ⟩.

## Pre-registered criteria
1. **Realistic driven PN rate** (unit-free): G = ρ_uPN_ant / ρ_ORN in
   - primary window [0.29, 1.45] (PN 1–5 Hz over ORN 3.44 Hz, Kazama & Wilson 2009),
   - broad window [0.077, 1.67] (PN 1–5 Hz over ORN 3–13 Hz).
2. **Silent without ORNs** (`all` protocol):
   - S1 (fast): mean uPN_ant activity 100–300 units after removal ≤ 10 % of its driven value;
   - S2 (lasting): ≤ 5 % of replicas (≤ 3/64) have any AL-system activity at the end of the run (3000 units after removal).
3. **Palp-intact removal** (`ant` protocol, Olsen et al. 2007): S1′ = uPN_ant activity 100–300 units after removal
   ≤ 10 % of driven; S2′ = uPN_ant activity 1000–3000 units after removal ≤ 10 % of driven.
4. **Allowed range** = g values meeting 1 (primary; broad reported too) and 2 (S1 and S2); a second version also requires 3.
   Reported per r and ORN level; compared with g_lin, the τ = 200 / 3000 boundaries and the fluctuation peak.
5. Also reported (no threshold): the fall of uPN activity (immediate drop, 1/e and 10 % times), and whether the allowed
   range sits where G rises steeply (amplification just below ignition).
Sensitivity: thresholds 1 % instead of 10 % for S1/S1′/S2′; S2 with 'any uPN_ant activity' instead of AL system.
