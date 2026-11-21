# Line 4: data grounding — what the antennal-lobe module is, and why it dominates

Compiled 2026-11-09 from the MaleCNS v1.0 cache (read only). No web sources: everything below was computed here.
All numbers are in `data_grounding.json` (same directory), built by `code/al_report.py` from the per-analysis files in
`grounding/`.

- Code: `code/al_common.py` (labels, operator), `al_spectra.py` (+ `.sbatch`, jobs 174849 / 175063, thickstun CPU: whole-CNS
  eigen-solves under 28 perturbations plus a region "peel"), `al_edges.py` (+ `.sbatch`, job 174876: region edge statistics,
  region Perron roots, exact λ decomposition, AL subgraphs), `al_module.py` (Q1), `al_tail.py` (Q3, top-20 tail modes),
  `al_recruit.py` (simulated switch-on order from the existing r = 0 / r = 1 sweeps), `al_figs.py`, `al_report.py`.
- Figures (`figs/al/`): `al_module_composition.png`, `al_robustness.png`, `al_why.png`, `al_lambda_decomposition.png`,
  `al_tail_order.png`.

**Operator.** The excitatory operator is the one the simulation uses at r = 0: M[post, pre] = c·w for presynaptic ACh
neurons, with c = ~~1/742.7~~ **1/745.9** (global mean in-strength) *[verification 2026-11-09: the code (`scale_c`,
`swc.py` raw norm, `edges.json` "c") uses the mean over the 166,499 neurons WITH input, 745.91; 742.7 is the mean over all
167,216 neurons. Every λ below was computed with 1/745.9 and is correct; only the stated constant was wrong]*. In these units λ_PF = 5.574, and α_eff/λ (α_eff = 0.906) is the naive
linear switch-on coupling (0.163 for the AL, the same as g_lin = 0.1632). Unless stated otherwise, "mode" means a right
eigenvector, which shows where activity sits. The left eigenvector shows which neurons' activity drives the mode, so it
picks out the best ignition seeds.

**Labels.** The flat release has no per-synapse neuropil (ROI) table. So "region" is the region a *neuron* is annotated to:

- AL = classes olfactory/hygro/thermosensory, ALPN, ALLN, ALIN, ALON;
- MB = KC, MBON, DAN, APL, DPM;
- CX = class CX;
- LH = untyped-class central neurons whose type starts with "LH";
- OL = optic-lobe intrinsic;
- VNC, AN/DN, and the rest of the central brain ("CB other").

An edge belongs to a region when both of its endpoints do. "MB calyx" and "MB lobes" are approximated by the PN→KC and
KC→KC edge sets.

---------------------------------------------------------------------------------------------------

## Bottom line

1. **The "AL module" is the AL's local-neuron network, not an ORN→PN relay.**
   - Of its 526 neurons, 479 are AL neurons: 217 uPNs (50 of 66 glomeruli), 53 mPNs, 188 LNs, 10 IN/ON, 10 HRN_VP1d and
     1 TRN_VP1m. Another 30 are LH neurons, plus 11 central, 4 MB (APL ×2, DPM ×2) and 2 DNb05.
   - **It contains 0 of the 2,639 ORNs.**
   - **30 cholinergic lLN1_bc local neurons (15 per lobe) carry 51% of the leading mode** (|v|²), lLN2 LNs carry 29% and
     uPNs 9%. DL2v_adPN, for example, carries only 0.17%.
   - ORNs appear only on the input side: 3.3% of the left eigenvector (mostly ORN_DP1m), and 39% of the module's excitatory
     input synapses.
2. **Why the AL dominates: a fully connected, strongly weighted excitatory clique.**
   - In each lobe, every one of the 15 × 14 ordered lLN1_bc pairs is connected: 420 of 420 same-side ordered pairs, with a
     median of 204 synapses per connection and no cross-side edges.
   - This clique alone has λ = 3.92, which is about (n−1)·w̄·c = ~~14 × 207 / 742.7 = 3.90~~ 14 × 206.8 / 745.9 = 3.88
     *[verification: corrected for the c actually used; conclusion unchanged]*.
   - Adding the cholinergic lLN2 LNs and the reciprocal PN↔LN loops brings λ to 5.57, **3.0× the best mode anywhere else in
     the CNS** (1.88, a VNC leg premotor/motor-neuron mode).
   - Exact decomposition λ = Σ u_i M_ij v_j / (u·v): LN→LN 64.7% (lLN1_bc→lLN1_bc alone 42.2%), PN→LN 16.8%, LN→PN 16.5%.
     All ORN edges together contribute <1%, although they are the AL's largest synapse pools (ORN→LN 633k and ORN→PN 503k
     excitatory synapses).
   - 161 of the 200 strongest reciprocal excitatory pairs (2-cycles) in the whole CNS lie inside the AL.
3. **Robustness: a property of the weighted, signed wiring, not of sensory convergence.** λ₁ and its region under each
   perturbation (the full list is in `al_robustness.png`):

   | perturbation | λ₁ | leading region | margin over best non-AL mode |
   |---|---:|---|---:|
   | ORNs removed | 5.55 | AL | 2.95× |
   | ORNs, HRNs and TRNs removed | 5.54 | AL | 2.95× |
   | only neurons whose soma lies in the volume (drops all 27,182 without one, incl. every sensory afferent) | 5.50 | AL | 2.94× |
   | cell-type-level NT predictions as signs | 5.57 | AL | 2.98× |
   | weights capped at 50 | 3.34 | AL | 2.21× |
   | weights capped at 20 | 2.04 | AL | 1.76× |
   | weights capped at 10 | 1.30 | AL | 1.44× |
   | sqrt weights | 0.77 | AL | 1.49× |
   | log(1+w) weights | 0.48 | AL | 1.32× |

   - **Binarization** removes the AL's lead: the MB's dense, weak, reciprocal KC–KC wiring wins (0.280 vs the AL's best,
     0.249).
   - **Making the AL local neurons non-excitatory removes the AL from the top 8 modes entirely.** λ₁ drops to 1.88, the
     same VNC mode that leads when the whole AL is deleted.
   - So the special status rests on two empirical facts:
     - (a) multi-synaptic connections, tens to hundreds of synapses each (moderately important);
     - (b) the cholinergic identity of lLN1_bc and of several lLN2 types (decisive).
   - The release gives lLN1_bc as acetylcholine in the consensus, body-level and cell-type-level predictions, and in its
     `ground_truth` field (30/30), with mean prediction confidence 0.75.
4. **Which regions come next (the switch-on order).**
   - After the AL (modes 1, 2, 4 and 6), the next tier, at λ ≈ 1.0–1.9, is **motor circuitry**:
     - VNC leg premotor interneurons and motor neurons (λ 1.88);
     - abdominal VNC (1.71);
     - GNG motor, MN11D/GNG001 (1.35);
     - a VNC + CX mix with ExR6/ExR4 (1.31 ± 0.007i);
     - wing/haltere motor neurons (1.22);
     - DNg33 with ascending neurons (1.01).
   - Each region taken alone has a Perron root of: AL 5.57, VNC 1.70, central other 1.47, **CX 1.28** (ExR6, ExR4, EPG, EL,
     PEN_a), OL 0.52, LH 0.49, **MB 0.47** (APL, DPM), visual projection 0.41, JO 0.25.
   - **The simulations follow this order.**
     - At r = 0 the AL modes switch on at g ≈ 0.26.
     - All motor and CX modes switch on together at g ≈ 0.50 (spectral prediction α_eff/λ₃ = 0.48), and abdominal VNC at
       0.63.
     - At r = 1 the values are 0.34, then 1.0–1.35, then 2.28.
   - MB and LH light up early (19% and 12% of all activity at g = 0.30, r = 0), but only because AL projection neurons drive
     them. Their own loops are weak.
5. **What this means for the consciousness/sleep question (for the literature lines to check).**
   - The model's "AL phase" is the ignition of a recurrent cholinergic local-neuron assembly. Biologically, that points to
     the excitatory-LN (eLN) system of the AL.
   - It is a structural prediction that the AL can sustain or reverberate activity at lower gain than any other region. It
     does not say anything about sleep by itself.
   - The CX ring and ExR neurons, where the fly sleep literature points (see line 1), appear only in the second tier, at
     ~~about 3×~~ the AL's coupling, together with the VNC/GNG motor loops. *[verification: "3×" is the ratio for the FIRST
     motor-tier mode (VNC leg, 5.574/1.879 = 2.97). For the CX itself the spectral ratio is ≈ 4.3× (mode 9–10:
     g_on 0.693 vs 0.163; peel step 6: 5.57/1.30); in the r = 0 simulation the whole tier, CX included, switches on at
     0.50 vs 0.26 ≈ 1.9×.]*
   - Two things the connectome cannot capture:
     - **gap junctions**, through which eLNs are reported to couple to PNs;
     - **neuromodulation** (DA/OA/5-HT neurons have sign 0 here).
   - Both bear directly on any state-dependence.

---------------------------------------------------------------------------------------------------

## Q1. What exactly is the module?

**Module definition:** the union of the 99% supports of the two leading right eigenvectors of the r = 0 operator,
`results/module/module_graph.npz`, n = 526.

**Superclass:**
- cb_intrinsic 513;
- cb_sensory 11 (10 HRN_VP1d, 1 TRN_VP1m);
- descending_neuron 2 (DNb05).

**Sides:**
- R 275, L 248, other 3.
- Mode 1 is 69% R / 31% L and mode 2 is 70% L / 30% R.
- Each lobe alone has λ_R = 5.49 and λ_L = 5.41. Weak cross-lobe excitation (mostly bilateral ORN axons, plus bilateral
  PNs) mixes them into the near-degenerate pair 5.574 / 5.389.
- So read the pair as **one bilateral AL system**, not as "left mode" and "right mode".

**Neurotransmitter** (consensus, with share of mode 1 |v|²):
- ACh 366 (79.0%);
- GABA 96 (12.6%);
- unclear 32 (7.9%);
- Glu 25 (0.4%);
- OA 3, 5-HT 2, DA 2.

Model sign: +1 for 366, −1 for 121, 0 for 39. The left eigenvector lies entirely on ACh neurons, by construction: only they
send excitation. GABA/Glu LNs appear in the right vector only as receivers of the excitatory core. They matter dynamically
only at r > 0, where they raise the onset. Inside the module, 18% of synapses are inhibitory and 71% excitatory.

**By category** (neurons in module / in CNS; share of |v|² of right mode 1, right mode 2 and left mode 1):

| category | module / CNS | right 1 | right 2 | left 1 |
|---|---|---:|---:|---:|
| lLN1 (30 lLN1_bc + 2 lLN1_a + 2 others) | 34 / 59 | 0.521 | 0.554 | 0.520 |
| lLN2 (all lLN2 in the CNS) | 92 / 92 | 0.291 | 0.286 | 0.200 |
| uPN | 217 / 328 | 0.090 | 0.087 | 0.194 |
| v2LN / vLN / il3LN | 34 / 136 | 0.054 | 0.041 | 0.008 |
| mPN | 53 / 358 | 0.019 | 0.012 | 0.024 |
| other LN | 28 / 123 | 0.008 | 0.007 | 0.010 |
| LH neurons | 30 / 2,026 | 0.007 | 0.005 | 0.000 |
| MB (APL, DPM) | 4 / 441 | 0.002 | 0.002 | 0.000 |
| central other | 11 / 21,512 | 0.002 | 0.001 | 0.000 |
| ALIN / ALON | 10 / 38 | 0.002 | 0.001 | 0.006 |
| HRN / TRN | 11 / 91 | 0.001 | 0.001 | 0.004 |
| AN / DN | 2 / 3,725 | 0.001 | 0.000 | 0.000 |
| **ORN** | **0 / 2,639** | 0.001 | 0.000 | **0.033** |

**Local-neuron types** (all of each type is in the module; share of right 1 / left 1; NT consensus; `ground_truth` field):

| type | n | NT (ground truth) | right 1 | left 1 |
|---|---:|---|---:|---:|
| **lLN1_bc** | 30 | ACh (gt ACh 30/30, conf 0.75) | **0.508** | **0.506** |
| lLN2T_a | 5 | ACh (gt ACh) | 0.057 | 0.018 |
| v2LN30 | 2 | unclear (sign 0) | 0.038 | 0 |
| lLN2F_a | 4 | unclear | 0.034 | 0 |
| lLN2F_b | 4 | GABA | 0.034 | 0 |
| lLN2P_a | 7 | GABA | 0.029 | 0 |
| lLN2T_c | 4 | ACh (gt ACh) | 0.022 | 0.054 |
| lLN2P_c | 9 | GABA | 0.019 | 0 |
| lLN2X04 | 4 | ACh | 0.019 | 0.015 |
| lLN2X12 | 11 | ACh | 0.017 | 0.043 |
| lLN2X05 | 4 | ACh (gt ACh) | 0.017 | 0.016 |
| lLN2P_b | 12 | GABA (gt GABA) | 0.015 | 0 |
| il3LN6 | 2 | GABA (gt GABA) | 0.014 | 0 |

**Hemilineage.** 140 module neurons come from ALl1_dorsal, which carries 82% of mode 1. The lLN1_bc annotation fields
match hemibrain types lLN1_b/lLN1_c and FlyWire type lLN1_bc. Across the CNS, the 420 ALLN neurons are by consensus 137
ACh, 115 GABA, 99 Glu, 67 unclear; their ground_truth fields give 47 ACh, 14 GABA.

**Glomeruli.**
- The module holds 217 of 328 uPNs, from 50 of the 66 uPN glomeruli. Usually all sister PNs of a glomerulus are included
  (DA1 13, VA1v 12, VM5d 12, DL2d 8, DL2v 8, VL2a 8, ...).
- The uPN weight is small and spread out. The largest shares are DP1m (2 cells: 1.2% of right 1, 2.1% of left 1), DP1l,
  DM1, VL2p, V and DC1.
- DP1m_adPN ↔ lLN2T_c is the strongest PN–LN 2-cycle (440 / 226 synapses).

**ORNs** (54 glomeruli, 2,639 cells; the largest classes are DA1 204, VA1d 132, VA1v 130, DL3 103, VL2a 98):
- None are in the module.
- They hold 3.3% of the left eigenvector. ORN_DP1m alone holds 1.0% (31 cells), followed by DM1 0.23%, DP1l 0.20%,
  DC1 0.14%, VL2a, VA2, DL5.

**Module input** (excitatory synapses onto the module):
- 40% from within the module;
- 39% from ORNs;
- 12% from KCs (onto the APL/DPM members);
- 1.7% from HRN/TRN.

**Module output** (excitatory synapses from the module):
- 51% stays in the module;
- 18% goes to the MB;
- 13% goes to the LH;
- 9% goes to other central neurons.

**Tracing status (caveat).**
- 228 of 526 module neurons, and all 30 lLN1_bc, are "Prelim Roughly traced". These neurons carry 80% of mode 1.
- Across the CNS, 22% of neurons have this status.
- The lLN1_bc clique is stereotyped (15 vs 15 cells, every same-side pair connected) and matches the hemibrain and FlyWire
  types. That argues against a merge artifact, but its synapse counts are still preliminary.

---------------------------------------------------------------------------------------------------

## Q2. Why does the AL dominate?

### 2a. Connection statistics inside each region

The table covers all edges with both endpoints in the region. "Within exc in" is the mean excitatory in-strength each
neuron receives from inside its own region, in c-units (1 = the CNS mean total in-strength). "Perron" is the leading
eigenvalue of the region's own excitatory subgraph, shown as raw / weights capped at 50 / binary.

| region (neurons) | syn/conn mean · median · p99 | conns ≥ 50 syn (share of syn) | weighted reciprocity | within exc in (mean) | Perron raw / cap50 / binary | Perron mode's top types |
|---|---|---|---:|---:|---|---|
| **AL** (3,874) | 7.5 · 2 · 84 | 2.3% (32%) | 0.41 | 0.95 | **5.57** / 3.34 / 0.249 | lLN1_bc 52%, lLN2T_a |
| AL without ORN/HRN/TRN (1,144) | **12.0 · 3 · 149** | **4.9% (46%)** | **0.51** | **1.40** | 5.54 / – / – | lLN1_bc 53% |
| MB (4,505) | 2.7 · 1 · 30 | 0.5% (12%) | 0.53 | 0.66 | 0.47 / 0.47 / **0.279** | APL 55%, DPM 10% |
| KC→KC only (4,064 KCs, 158 conns per KC) | 1.8 · 1 · 7 | 0.0% | 0.61 | 0.38 | 0.46 | KCg-m, KCg-d |
| PN→KC (calyx input) | 17.3 · 16 · 47 | 0.7% | 0.007 (feed-forward) | – | – | – |
| CX (2,950) | 6.4 · 2 · 58 | 1.4% (18%) | 0.37 | 0.50 | 1.28 / 1.18 / 0.098 | ExR6, ExR4, EPG, EL, PEN_a |
| VNC (20,429) | 6.3 · 2 · 69 | 1.7% (26%) | 0.13 | 0.57 | 1.70 / 0.78 / 0.118 | INXXX149, INXXX137 |
| central brain, other (21,512) | 5.3 · 2 · 56 | 1.3% (21%) | 0.16 | 0.74 | 1.47 / 1.21 / 0.238 | AVLP016, AVLP079, AVLP076 |
| LH (2,026) | 4.0 · 2 · 34 | 0.5% (10%) | 0.20 | 0.16 | 0.49 / 0.28 / 0.050 | LHPV6o1, LHPV1c2 |
| OL intrinsic / columns (89,403) | 4.1 · 2 · 31 | 0.3% (6%) | 0.21 | 0.32 | 0.52 / 0.49 / 0.091 | Li39 64% |
| visual projection / centrifugal (9,766) | 3.4 · 2 · 27 | 0.3% (8%) | 0.27 | 0.24 | 0.41 / 0.41 / 0.160 | LoVC16 |
| whole CNS (167,216) | 4.9 · 2 · 46 | 0.9% (16%) | 0.17 | 0.59 (all exc in) | 5.57 | – |

**Reading.** The AL's non-sensory core combines three properties that no other region has together:

- the heaviest connections (mean 12 synapses, 4.9% of connections ≥ 50 synapses holding 46% of its synapses);
- high weighted reciprocity (0.51);
- the largest excitatory input from within the region.

Each alone is not unique:

- **Strong single connections:** the strongest individual excitatory edges in the CNS are visual (LT1b→PVLP106, 1,439
  synapses). The AL holds 12 of the top 200 edges and 10% of edges ≥ 135 synapses.
- **Reciprocity:** the MB is at least as reciprocal, but with 1–2-synapse KC–KC contacts. That is why the MB wins only the
  binary graph.

What the AL monopolizes is **strong reciprocal excitation**: 161 of the 200 strongest excitatory 2-cycles (ranked by
√(w_ij·w_ji)) are AL↔AL. The few 2-cycles stronger than the lLN1_bc pairs belong to single reciprocal pairs: GNG117
(914/973 synapses), DNg33 (773/723), GNG040, dPR1. These pairs seed the motor-tier modes 7 and 17.

### 2b. Which AL edges make λ (exact decomposition and leave-one-class-out)

Leave-one-class-out removes the edge class from the whole-CNS operator and re-solves (in the λ₁ column, ← marks the leading
region when it is not the AL):

| edge class (ACh presynaptic) | exc synapses | share of λ₁ (exact) | λ₁ without it | AL share of mode 1 |
|---|---:|---:|---:|---:|
| LN → LN | 396k | **64.7%** (lLN1_bc→lLN1_bc 42.2%) | **2.80** | 81% |
| PN → LN | 402k | 16.8% | 4.49 | 97% |
| LN → PN | 217k | 16.5% | 4.51 | 100% |
| PN ↔ LN (both) | 619k | 33.3% | 4.49 | 100% |
| all PN presynaptic | – | – | 4.49 | 100% |
| PN → PN | 130k | 0.4% | 5.55 | 99% |
| LN → ORN | 68k | 0.4% | 5.55 | 99% |
| ORN → PN | 503k | 0.2% | 5.56 | 99% |
| ORN → LN | 633k | 0.2% | 5.56 | 99% |
| ORN → ORN | 185k | <0.1% | 5.57 | 99% |
| PN → ORN | 9k | <0.1% | 5.57 | 99% |
| HRN/TRN → any | 106k | ~0.2% | 5.56 | 99% |
| every AL LN made non-excitatory | – | – | **1.88 (← VNC)** | 0% (AL absent from top 8) |
| whole AL removed | – | – | 1.88 (← VNC) | – |

**AL subgraphs** (Perron root of the AL-internal excitatory subgraph):

| subgraph | λ |
|---|---:|
| whole AL | 5.57 |
| LNs + PNs | 5.53 |
| all LNs | 4.46 |
| lLN1 + lLN2 | 4.43 |
| **lLN1_bc alone (30 cells)** | **3.92** |
| AL without lLN1_bc | 3.17 (still above every other region; carried by cholinergic lLN2 types) |
| AL without lLN1 and lLN2 | 0.65 |
| AL without any LN | 0.63 |
| ORNs alone | 0.44 |
| PNs alone | 0.39 |

The PN–ORN feed-forward relay has almost no recurrent gain. The AL's gain is the cholinergic-LN assembly plus its
reciprocal loops with PNs.

### 2c. Robustness

Every variant is a whole-CNS re-solve with the global c fixed. The margin is λ₁ divided by the best mode that does not sit
on the AL (AL share < 50%) among the top 8.

| variant | λ₁ | leading region (share) | 1/IPR | margin |
|---|---:|---|---:|---:|
| full | 5.574 | AL (99%) | 72 | 2.97× |
| ORNs removed | 5.550 | AL (99%) | 69 | 2.95× |
| ORNs + HRNs + TRNs removed | 5.540 | AL (99%) | 69 | 2.95× |
| soma in volume only | 5.504 | AL (99%) | 72 | 2.94× |
| cell-type-level NT signs (4,215 neurons change sign, none in the AL) | 5.574 | AL (99%) | 72 | 2.98× |
| cap 50 | 3.339 | AL (94%) | 182 | 2.21× |
| soma only + cap 50 | 3.230 | AL (94%) | 168 | 2.15× |
| no ORNs + cap 50 | 3.288 | AL (94%) | 179 | 2.18× |
| cap 20 | 2.042 | AL (90%) | 254 | 1.76× |
| cap 10 | 1.301 | AL (88%) | 292 | 1.44× |
| sqrt(w) | 0.768 | AL (91%) | 222 | 1.49× |
| log(1+w) | 0.478 | AL (87%) | 276 | 1.32× |
| binary | 0.280 | **MB (93%: KCg-m, PAM/PPL DANs)** | 339 | AL is mode 4 (0.249) |
| AL LNs non-excitatory | 1.879 | **VNC (53%) / AN-DN / central** | 632 | AL not in top 8 |

Under compressive transforms the lead persists, but the mode broadens: 1/IPR rises from 72 to 180–290, lLN1_bc's share
falls from 51% to about 12% (cap 10, log), and lLN2 and uPNs share more of the mode. For caps ≤ 20 and sqrt/log, the best competitor shifts from the VNC
leg mode to a central AVLP016 / AVLP076 mode.

**Answer to "robust or artifact?"**

- **Not sensory convergence.** There are no ORNs in the module, removing them changes λ by 0.4%, and all ORN edges
  together give <1% of λ. ORNs supply input, not recurrence.
- **Partly synapse counting.** The AL stays first under caps down to 10 synapses and under sqrt/log, so a few giant edges
  are not the cause. But on the binary graph its lead is gone: there it falls to second behind the MB, 0.249 vs 0.280. The
  property is "many reciprocal multi-synaptic excitatory connections".
- **Decisively, transmitter identity.** Without excitatory LNs the AL is ordinary: subgraph λ 0.63, below the CX (1.28) and
  the VNC (1.70). The AL's special status in our model therefore equals the claim that the lLN1_bc (and lLN2T / lLN2X)
  local neurons are excitatory, cholinergic LNs. The release's NT predictions and ground_truth labels say they are.
- **Model-dependent.** With per-neuron input normalization (`--norm in`, used early in the project), the leading modes
  were instead 2-neuron optic-lobe motifs. The AL result belongs to the raw-count, global-c model.

---------------------------------------------------------------------------------------------------

## Q3. Which regions come next (spectral tail and simulated switch-on order)

**Top-20 modes** of the r = 0 operator (`results/pd/tail_real_r0_dir.npz`). g_on = α_eff/λ is the naive switch-on coupling.

| k | λ | g_on | 1/IPR | leading region (share) | top types |
|---:|---:|---:|---:|---|---|
| 1 | 5.574 | 0.163 | 72 | AL 99% (R 69%) | lLN1_bc, lLN2T_a, v2LN30, lLN2F_a |
| 2 | 5.389 | 0.168 | 68 | AL 99% (L 70%) | same |
| 3 | 1.879 | 0.482 | 631 | VNC 53%, AN/DN 21%, central 21% | IN08A002, IN09A002, IN13A001, sternotrochanter MN, Tr flexor MN (leg premotor/MN) |
| 4 | 1.738 | 0.521 | 77 | AL 74%, VNC 13% | lLN2F_b, lLN2P_a, il3LN6, lLN1_bc |
| 5 | 1.713 | 0.529 | 32 | VNC 97% | INXXX149, INXXX137, MNad64 (abdominal) |
| 6 | 1.563 | 0.580 | 45 | AL 92%, MB 5% | lLN2F_b, lLN2P_a, v2LN30 |
| 7 | 1.352 | 0.670 | 16 | central 42%, CB motor 39% | MN11D, GNG001, GNG117 (GNG motor) |
| 8 | 1.319 | 0.687 | 81 | VNC 84% | INXXX328, INXXX385, MNad04/48 |
| 9–10 | 1.308 ± 0.007i | 0.693 | 195 | VNC 76%, **CX 15%** | INXXX328, **ExR6**, INXXX385, **ExR4** |
| 11–13 | 1.26 (12–13: ± 0.18i) | 0.72 | 199–324 | VNC 68–92% | IN08A002, IN13A001, IN19A004 (leg) |
| 14 | 1.223 | 0.741 | 65 | VNC 92% | hg1 MN, vMS12_a, dMS2 (wing/haltere) |
| 15–16 | 1.071 ± 0.02i | 0.846 | 147 | VNC 83% | leg premotor/MN |
| 17 | 1.007 | 0.900 | 15 | AN/DN 79% | DNg33, AN09A005, AN27X013 |
| 18–20 | 0.97–0.98 | 0.93 | 94–116 | VNC 75–95% | hg1 MN, Tr flexor MN |

- **Order of first appearance:** AL (k = 1) → VNC (k = 3) → central/GNG motor (k = 7) → AN/DN (k = 17).
- **No sensory-integration or memory region appears in the top 20:** no MB, LH, OL or JO mode. The CX enters only inside
  mode 9–10.

**Peel.** Removing the leading region and re-solving gives:

| step | λ | leading mode |
|---:|---:|---|
| 0 | 5.57 | AL |
| 1 | 1.88 | VNC leg mode |
| 2 | 1.84 | central AVLP016 / DNg16 / DNg75 mode |
| 3 | 1.71 | abdominal VNC |
| 4 | 1.37 | abdominal MN / DN |
| 5 | 1.32 | GNG motor (MN11D, GNG001) |
| 6 | **1.30** | **CX: ExR6 26%, ExR4 18%, EPG 15%, EL 8%, PEN_a 5%, ER6 5%** |
| 7 | 1.27 | VNC abdominal |

The CX's own Perron root is 1.28. The MB's is 0.47 (APL/DPM, not KCs) and the LH's 0.49.

**Simulation check** (`al_recruit.py`; the existing decay sweeps, fully active start, T = 3000, h = 0). The table gives the
g at which the mean occupancy on each mode's 90% support first reaches 0.05; plot in `al_tail_order.png`.

| support | r = 0 | r = 1 |
|---|---:|---:|
| AL modes 1, 2, 6 | 0.260 | 0.339 |
| AL mode 4 | 0.278 | 0.386 |
| VNC leg / GNG / CX modes 3, 7–16, 18–20 | 0.50 (all within one grid step; spectral g_on 0.48–0.93) | 1.03–1.35 |
| DNg33 (17) | 0.57 | 1.35 |
| abdominal VNC (5) | 0.63 | 2.28 |

- The ordering AL ≪ motor tier matches the spectrum, and the first motor mode turns on where α_eff/λ₃ predicts.
- The AL turns on at 0.26 rather than 0.163, which is the known finite-size shift (slow window).
- Once the first motor mode ignites, the rest of that tier co-ignites, because the motor modes are coupled.

Activity share by region at r = 0:

| g | AL | MB | LH | central other | VNC | AN/DN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 52% | 19% | 12% | 13% | 2% | 1.6% |
| 0.50 | 20% | 13% | 8% | 26% | 19% | 7.5% |
| 0.80 | 6% | 6.5% | 3.8% | 37% | 26% | 9% |

The early MB/LH activity is driven by AL output (PN→KC, PN→LH; 18% and 13% of the module's excitatory output). The MB's
and LH's own loops are weak (Perron 0.47 and 0.49).

---------------------------------------------------------------------------------------------------

## Caveats for interpretation

- **Sign model.**
  - ACh = +1; GABA/Glu/His = −1; DA/OA/5-HT/unclear = 0.
  - "Unclear" LNs (v2LN30, lLN2F_a, lLN2T_d; about 8% of mode 1) are silent senders. If they are excitatory, the AL only
    gets stronger.
  - Glutamate in the AL (vLN/v2LN) is treated as inhibitory.
- **Missing biology.**
  - Gap junctions (eLN–PN electrical coupling) and neuromodulation are not in the connectome.
  - Inhibition (r > 0) is where the GABAergic lLN2P / lLN2F_b / il3LN6 receivers act. They are strongly driven by the
    cholinergic core, which is why inhibition raises the AL's onset.
- **Labels and quality.**
  - Regions are neuron-annotation based; there is no synapse ROI.
  - Most of the module is "Prelim Roughly traced".
- **Normalization.** The AL's special status belongs to the raw synapse-count, global-c normalization of the simulation.

---------------------------------------------------------------------------------------------------

## Verification (independent recompute, 2026-11-09)

Fact-check of this file and of the summary built from it. I wrote an independent script, `code/al_verify_3.py`, which
does not import `al_common.py` or reuse any saved eigenvectors. It rebuilds the r = 0 excitatory operator from the raw
cache arrays (`pre`/`post`/`weight`, with the sign taken from the consensus `nt` string), re-solves it with ARPACK, and
reads the raw release feathers for `ground_truth` and `statusLabel`.

- Runs: Slurm job 175573 (thickstun CPU, 104 s) → `results/al_lit/verify3.json`, and job 175597 (`--sim`) →
  `results/al_lit/verify3_sim.json`.
- The edge list has no duplicate (pre, post) pairs.

| # | claim | recomputed | status |
|---|---|---|---|
| V1 | λ₁, λ₂ = 5.574, 5.389; top-6 (right) 5.574 / 5.389 / 1.879 / 1.738 / 1.713 / 1.563; left spectrum the same | 5.5739, 5.3891, 1.8791, 1.7380, 1.7128, 1.5634 (right and left agree) | verified |
| V1 | c = 1/742.7 | the code uses **1/745.91**, the mean in-strength over the 166,499 neurons with input; 742.71 is the mean over all 167,216 neurons. Every λ is correct for 1/745.9 | **misreported constant** (annotated above) |
| V2 | module n = 526, defined as the union of the 99% supports of right modes 1 and 2 | 526; all 526 identical to `module_graph.npz` | verified |
| V2 | 479 AL neurons; 0 of 2,639 ORNs; 11 HRN/TRN; 188 LNs incl. all 92 lLN2 and 34 lLN1; PN total 270 (= 217 uPN + 53 mPN) | 479; 0 / 2,639; 11; 188, 92/92, 34; 270 | verified (the uPN/mPN split and the 50/66 glomeruli were not recomputed) |
| V2 | NT counts ACh 366 / GABA 96 / unclear 32 / Glu 25 (+ OA 3, 5-HT 2, DA 2); sides R 275 / L 248 / other 3 | identical | verified |
| V2 | lLN1_bc share of mode 1: 51% (right), 50.6% (left); lLN2 29%; mode 1 is 69/31 R/L and mode 2 is 30/70 | 0.508, 0.506; 0.291; 0.688/0.311; 0.300/0.700 | verified |
| V2 | ORNs hold 3.3% of left mode 1 (ORN_DP1m 1.0%) and supply 39% of the module's excitatory input | 3.25% (DP1m 1.03%); 38.96% | verified |
| V3 | lLN1_bc: 15 + 15 cells, all ACh; 420/420 same-side ordered pairs connected; median 204 synapses; no cross-side edges; clique λ 3.92 | 15/15, ACh 30/30; 420/420; median 203.5 (mean 206.8); 0 cross-side; λ 3.921 | verified |
| V3 | uniform-clique value 14 × 207 / 742.7 = 3.90 | 14 × 206.8 / 745.9 = **3.88** | minor slip (annotated above) |
| V4 | exact decomposition: LN→LN 64.7% (lLN1_bc→lLN1_bc 42.2%), PN→LN 16.8%, LN→PN 16.5%, ORN edges <1%; ORN→LN 633k, ORN→PN 503k synapses | Σ = 5.5739 (exact); 64.67% (42.15%), 16.78%, 16.54%; edges touching an ORN 0.87%; 633,253 and 502,874 synapses | verified |
| V5 | 161 of the 200 strongest excitatory 2-cycles (√(w_ij·w_ji), both neurons ACh) are AL↔AL | 161 of 1,114,480 excitatory 2-cycles; the 200th has √(w·w) = 190 | verified |
| V6 | region Perron roots: AL 5.57, CX 1.28, VNC 1.70, LH 0.49, MB 0.47; binary MB 0.279, AL 0.249 | 5.573, 1.284, 1.702, 0.491, 0.467; 0.2786, 0.2489 | verified |
| V6 | AL core (1,144 cells): 12.0 synapses per connection, median 3, 4.9% of connections ≥ 50 synapses holding 46% of synapses, weighted reciprocity 0.51; MB 2.7 / 0.53; CX 6.4 / 0.37; VNC 6.3 / 0.13; LH 4.0 / 0.20 | 12.04, 3, 4.86%, 45.5%, 0.507; MB 2.74 / **0.526**; CX 6.35 / 0.372; VNC 6.26 / 0.128; LH 3.97 / 0.196 | verified (but see the note below) |
| V7 | cap at 50: λ₁ 3.339, AL 94%, margin 2.21× | 3.3389, AL 94.1%; best non-AL mode 1.5115, giving 2.21× | verified |
| V7 | binary: MB leads at 0.280; the AL is mode 4 at 0.249 | 0.2799 (MB 93.5%); mode 4 = 0.2490 (AL 71%) | verified |
| V7 | AL LNs non-excitatory: λ₁ 1.879, a VNC mode; AL absent from the top 8 | 1.8792 (VNC 53%); AL share 0.000 in all top 8 | verified |
| V8 | lLN1_bc: `ground_truth` ACh 30/30, mean confidence 0.75; body- and cell-type-level predictions are ACh | 30/30 ACh in all three fields; mean confidence 0.753 | verified |
| V8 | "Prelim Roughly traced": 228 of 526 module neurons, all 30 lLN1_bc, 80% of mode 1, 22% CNS-wide | 228; 30/30; 79.6%; 21.8% | verified |
| V9 | simulated switch-on (mean occupancy ≥ 0.05 on the 90% support) at r = 0: AL 0.26, first motor mode 0.50, abdominal VNC 0.63 | 0.260 (modes 1, 2 and 6), 0.502 (mode 3), 0.630 (mode 5) | verified |
| V9 | at r = 1: AL 0.34, motor tier 1.0–1.35, abdominal 2.28; activity share at g 0.30, r = 0: AL 52%, MB 19%, LH 12% | 0.339; mode 3 1.176; 2.28; 51.8% / 18.6% / 11.9% | verified |

V9 uses the mode supports saved in `tail_real_r0_dir.npz`. Its top-6 eigenvalues match the independent V1 solve.

**Not recomputed:**
- the lobe-alone λ_R / λ_L (5.49 / 5.41);
- the other robustness rows (no ORN, soma only, cell-type NT, cap 20/10, sqrt, log);
- tail modes 7–20 and the peel;
- the 30 LH / 11 central composition;
- the 12% KC input.

**Statements in the summary (not in this file) that the numbers do not support:**
1. "The MB, CX, OL, VNC and LH all fall well below on these measures" (mean weight, share of heavy connections,
   reciprocity) is wrong for reciprocity. The MB's weighted reciprocity is **0.526**, slightly *above* the AL core's
   0.507, and KC→KC alone reaches 0.61. This file's own wording is correct: "three properties that no other region has
   *together*".
2. "The CX ring is the strongest non-motor brain region": only if "central brain, other" counts as motor. Its Perron root
   is 1.47 > CX 1.28, and its mode is AVLP016 / AVLP079 / AVLP076, which is mostly AVLP interneurons, plus DNs in the
   peel.
3. "CX ... at about 3× the AL's coupling" is imprecise; see the annotation in Bottom line §5. 3× is the ratio for the
   first motor-tier mode. For the CX it is about 4.3× spectrally, and the simulation gives about 1.9× for the whole tier.

**Verdict.** Every recomputed number (about 40 values across V1–V9) reproduces to the stated precision. The data analysis
is sound. The only errors are:
- a mislabelled normalization constant (742.7 vs 745.9), which does not affect any λ;
- the clique arithmetic that follows from it (3.90 → 3.88);
- three pieces of summary prose: the claim about MB reciprocity, "CX = strongest non-motor region", and "≈3×" for the CX.
